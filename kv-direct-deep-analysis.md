# KV-Direct Deep Integration Analysis for llama.cpp

## Executive Summary

After tracing through llama.cpp's architecture layer by layer, I've identified that KV-Direct integration is **more complex than initially apparent** but **fundamentally feasible**. The key challenge isn't the math (which is elegant) — it's threading the residual stream through llama.cpp's highly optimized, graph-reusing compute pipeline.

---

## 1. The Mathematical Foundation (Verified)

The paper's core claim holds: for any transformer layer, given the residual stream input `x`, the keys and values are:

```
K = x @ W_k^T    (plus optional bias, head splitting, normalization)
V = x @ W_v^T    (plus optional bias, head splitting, normalization)
```

This is **exact**, not approximate. The residual stream contains all information needed to reconstruct K and V.

**Verification in llama.cpp's Qwen3.5 implementation** (`qwen35.cpp`):

```cpp
// Line 129-145: Full attention layer
ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, ...);
ggml_tensor * Kcur      = build_lora_mm(model.layers[il].wk, cur, ...);  // ← K from residual
ggml_tensor * Vcur      = build_lora_mm(model.layers[il].wv, cur, ...);  // ← V from residual

// Line 219: Linear attention (DeltaNet)  
auto qkvz = build_qkvz(cur, il);  // cur is the residual stream
```

The `cur` tensor at these call sites **is** the normalized residual stream. The K/V projections are single matrix multiplications from this vector.

---

## 2. Architectural Integration Points (Detailed)

### 2.1 Where Residual Streams Exist in the Compute Graph

Tracing through `qwen35.cpp` layer-by-layer:

```
Layer il:
┌─────────────────────────────────────────────────┐
│ inpL = residual from previous layer             │ ← CHECKPOINT THIS (pre-attn)
│                                                  │
│ cur = attn_norm(inpL)                           │
│ Kcur = cur @ W_k^T                              │
│ Vcur = cur @ W_v^T                              │
│ attn_out = attention(Q, K, V)                   │
│                                                  │
│ cur = attn_out + inpL                           │ ← CHECKPOINT THIS (post-attn residual)
│ ffn_residual = cur                              │
│                                                  │
│ cur = build_ffn(attn_post_norm(cur))            │
│ cur = cur + ffn_residual                        │ ← CHECKPOINT THIS (post-ffn residual)
│                                                  │
│ inpL = cur  // → next layer                     │
└─────────────────────────────────────────────────┘
```

**Critical insight**: There are **three** residual stream tensors per layer:
1. **Pre-attention** (`inpL`): Input to the layer, before any normalization
2. **Post-attention** (`cur = attn_out + inpL`): After attention residual connection
3. **Post-FFN** (`cur = ffn_out + ffn_residual`): Final output of the layer

For KV-Direct, we need the **pre-attention normalized residual** — that's what gets projected to K/V. In Qwen3.5, this is `cur` after `build_norm()` at line 29, before the K/V projections at lines 141-144.

### 2.2 The Graph Reuse Problem (Major Complexity)

llama.cpp aggressively reuses compute graphs across tokens for performance. From `llama-graph.cpp`:

```cpp
// llm_graph_input_attn_kv::can_reuse()
bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);
    this->mctx = mctx;
    
    bool res = true;
    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);
    return res;
}
```

**The problem**: Graph reuse assumes the KV cache structure is stable. If we switch to recomputing K/V from residuals on-demand, the graph structure changes fundamentally:

- **Current**: `get_k()` returns a view into cached tensor → fast pointer arithmetic
- **KV-Direct**: `recompute_k()` inserts a `ggml_mul_mat()` node into the graph → new op every time

This means either:
1. **Disable graph reuse** for KV-Direct mode (performance hit), or
2. **Pre-build a "recomputation graph template"** that can be reused (complex but viable)

### 2.3 The Memory Context Interface

The attention computation receives K/V through `llm_graph_input_attn_kv`:

```cpp
// llama-graph.h lines 284-322
class llm_graph_input_attn_kv : public llm_graph_input_i {
public:
    ggml_tensor * get_k_idxs() const { return self_k_idxs; }
    ggml_tensor * get_v_idxs() const { return self_v_idxs; }
    ggml_tensor * get_kq_mask() const { return self_kq_mask_cnv; }
    
    // These are set by the KV cache context
    ggml_tensor * self_k_idxs = nullptr;  // I64 [n_batch] - indices into K cache
    ggml_tensor * self_v_idxs = nullptr;  // I64 [n_batch] - indices into V cache
    ggml_tensor * self_kq_mask = nullptr; // attention mask
};
```

The KV cache provides **indices** into the cached K/V tensors, not the tensors themselves. The actual K/V retrieval happens inside `build_attn()` which calls back into the memory context.

**For KV-Direct**, we need to replace this with:
- Indices into the **residual checkpoint buffer**
- A recomputation step that produces K/V from residuals before attention

---

## 3. Implementation Strategy (Revised)

### Phase 1: Add Residual Checkpoint Capture

**File: `src/llama-graph.h`** — Add a new graph input type for residual checkpoints:

```cpp
class llm_graph_input_residual : public llm_graph_input_i {
public:
    llm_graph_input_residual(const llama_cparams & cparams) : cparams(cparams) {}
    
    void set_input(const llama_ubatch * ubatch) override;
    
    // Residual checkpoint buffer: [n_embd, n_window, n_seqs]
    ggml_tensor * residual_buf = nullptr;
    
    const llama_cparams cparams;
};
```

**File: `src/models/qwen35.cpp`** — Capture residuals at the right points:

```cpp
// After line 29 (post-normalization, pre-KV projection):
cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);

if (cparams.kv_direct) {
    // Register this tensor as the residual checkpoint for this layer
    // This needs to be accessible to the KV-Direct cache
    res->residual_for_kv[il] = cur;
}

// ... K/V projections use 'cur' as before
```

### Phase 2: KV-Direct Cache Implementation

**New file: `src/llama-kv-cache-direct.h`**

```cpp
class llama_kv_cache_direct : public llama_memory_i {
public:
    struct config {
        uint32_t window_size = 64;           // Recent tokens in fast buffer
        uint32_t checkpoint_dtype_bits = 16; // FP16 by default
        bool evict_to_disk = false;          // Optional: page old checkpoints
        std::string disk_path;               // For eviction
    };
    
    llama_kv_cache_direct(const llama_model & model, const config & cfg);
    
    // Store residual checkpoint for token 'pos' at layer 'il'
    void store_residual(uint32_t pos, uint32_t il, ggml_tensor * resid);
    
    // Recompute K from stored residual
    ggml_tensor * recompute_k(ggml_context * ctx, uint32_t pos, uint32_t il) const;
    
    // Recompute V from stored residual  
    ggml_tensor * recompute_v(ggml_context * ctx, uint32_t pos, uint32_t il) const;
    
    // Memory is bounded: window_size × n_layers × n_embd × dtype_size
    size_t memory_usage() const override;
    
private:
    const llama_model & model;
    config cfg;
    
    // Ring buffer of recent residuals: [n_layers, n_embd, n_window]
    std::vector<ggml_tensor *> residual_ring;
    
    // Evicted checkpoints (can be paged)
    std::map<uint32_t, std::vector<float>> evicted_checkpoints;
};
```

### Phase 3: Wire Into Attention Computation

**File: `src/llama-graph.cpp`** — Modify `build_attn()` to support KV-Direct:

```cpp
ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,  // May be nullptr in KV-Direct mode
        ggml_tensor * v_cur,  // May be nullptr in KV-Direct mode
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
              float   kq_scale,
                int   il) const {
    
    if (cparams.kv_direct && !k_cur && !v_cur) {
        // KV-Direct mode: recompute K/V from residual checkpoints
        auto * kv_direct = static_cast<llama_kv_cache_direct *>(mctx->get_base());
        
        // Get current token position
        const uint32_t pos = /* extract from ubatch */;
        
        // Recompute K and V
        k_cur = kv_direct->recompute_k(ctx0, pos, il);
        v_cur = kv_direct->recompute_v(ctx0, pos, il);
    }
    
    // ... rest of attention computation unchanged
}
```

---

## 4. Performance Analysis (Realistic)

### 4.1 Recomputation Cost Per Token

For Qwen3.5-27B at layer il:

| Operation | Standard Cache | KV-Direct |
|-----------|---------------|-----------|
| K retrieval | Memory read (~16KB) | `resid @ W_k^T`: ~8K MACs |
| V retrieval | Memory read (~16KB) | `resid @ W_v^T`: ~8K MACs |
| **Total per layer** | ~32KB read | ~16K MACs |
| **All 64 layers** | ~2MB read | ~1M MACs |

On RTX 4090:
- Memory bandwidth: ~1 TB/s → 2MB read ≈ **2 μs**
- FP16 Tensor Core throughput: ~100 TFLOPS → 1M MACs ≈ **10 ns**

**Theoretical**: KV-Direct recomputation is **~200x faster** than cache reads for a single token.

**Practical caveats**:
- Graph construction overhead (ggml node creation)
- Kernel launch latency for small matmuls
- Memory access pattern for W_k/W_v weights (may not be cached)

### 4.2 When KV-Direct Wins

```
Break-even analysis:

Standard cache cost:  O(context_length × n_layers × head_dim × n_heads) memory
KV-Direct cost:       O(window_size × n_layers × n_embd) memory + O(n_layers × n_embd × head_dim) compute per token

For Qwen3.5-27B (n_embd=8192, n_heads≈150, head_dim≈150):
  Standard:  context × 64 × 150 × 150 × 2 bytes ≈ context × 2.3 MB
  KV-Direct: 64 × 64 × 8192 × 2 bytes + 64 × 8192 × 150 MACs ≈ 67 MB + 78M MACs

Break-even at context_length ≈ 30 tokens for memory,
but KV-Direct always wins on memory for context > 64 (window size).
```

### 4.3 The Real Bottleneck: Graph Construction

The biggest performance risk isn't the matmul — it's **ggml graph construction overhead**. Every call to `ggml_mul_mat()` creates a new tensor node, allocates metadata, and potentially triggers graph optimization.

**Mitigation**: Use `ggml_cpy()` with pre-computed views, or batch residual-to-KV conversions into a single fused kernel.

---

## 5. Hidden Complexities (The "Gotchas")

### 5.1 Multi-Sequence Batching

llama.cpp supports multiple sequences in a single batch. The KV cache uses `seq_id` bitsets to track which sequences occupy each cell:

```cpp
// llama-kv-cells.h
using seq_set_t = std::bitset<LLAMA_MAX_SEQ>;
std::vector<seq_set_t> seq;  // Which sequences use each cell
```

For KV-Direct, residual checkpoints must be indexed by **(sequence_id, position)**, not just position. This adds complexity to the checkpoint storage and retrieval.

### 5.2 RoPE (Rotary Position Embeddings)

RoPE is applied **after** K/V projection:

```cpp
// qwen35.cpp lines 162-172
Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, ...);
Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, ...);
```

The residual checkpoint must be stored **before** RoPE is applied. When recomputing K/V, we recompute the projection AND reapply RoPE. This means we also need to store or recompute position information for each checkpoint.

### 5.3 Quantization

llama.cpp supports KV cache quantization (Q4_K, Q8_K, etc.). The residual checkpoints could also be quantized:

- FP16 residual: 8192 × 2 = 16KB per token per layer
- Q4_K residual: 8192 × 0.5 = 4KB per token per layer (4x savings)

But quantization introduces reconstruction error, which **violates the zero-error guarantee** of KV-Direct. This is a fundamental tradeoff.

### 5.4 Hybrid Models (Qwen3.5 Specific)

Qwen3.5 has 48 DeltaNet layers + 16 full attention layers. DeltaNet layers use a different state mechanism (convolutional states + SSM states):

```cpp
// qwen35.cpp line 244-245
ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);
```

For DeltaNet layers, KV-Direct doesn't apply directly — these layers don't use traditional K/V caches. The integration only affects the 16 full attention layers.

---

## 6. Recommended Integration Approach

Given the complexities, I recommend a **phased, conservative approach**:

### Step 1: Add Residual Checkpointing as an Observer (No Behavior Change)

Modify model build functions to capture and store residual streams alongside the existing KV cache. This adds memory overhead but no functional changes. Use this to verify the residuals are correct by comparing recomputed K/V against cached K/V.

### Step 2: Add KV-Direct as a Separate Cache Type

Implement `llama_kv_cache_direct` as an alternative to `llama_kv_cache`, selectable via `llama_cparams.kv_direct = true`. This keeps the existing code path untouched.

### Step 3: Hybrid Mode (Best of Both Worlds)

Implement a hybrid where:
- Recent tokens (within window): use standard KV cache for fast access
- Older tokens: store residuals only, recompute K/V on demand

This gives bounded memory growth while minimizing recomputation overhead.

### Step 4: Optimize Recomputation

Fuse the residual-to-KV projection with RoPE application into a single kernel. Pre-load W_k/W_v weights into shared memory. Use CUDA Graphs to eliminate kernel launch overhead.

---

## 7. Files That Need Modification

| File | Changes | Complexity |
|------|---------|------------|
| `llama-cparams.h` | Add `kv_direct`, `kv_direct_window` fields | Trivial |
| `llama-kv-cache-direct.h` | New file: KV-Direct cache class | Medium |
| `llama-kv-cache-direct.cpp` | New file: Implementation | High |
| `llama-graph.h` | Add residual checkpoint input type | Low |
| `llama-graph.cpp` | Modify `build_attn()` for KV-Direct path | Medium |
| `models/qwen35.cpp` | Capture residuals at layer boundaries | Low |
| `models/llama.cpp` | Same for LLaMA architecture | Low |
| `llama-context.cpp` | Wire up KV-Direct cache creation | Low |

---

## 8. Testing Plan

1. **Correctness**: Run identical prompts with standard cache and KV-Direct, compare outputs token-by-token (must be identical)
2. **Memory**: Verify bounded memory usage with increasing context length
3. **Performance**: Benchmark tokens/sec across context lengths (1k, 4k, 8k, 16k, 32k)
4. **Edge cases**: 
   - Multi-sequence batching
   - Speculative decoding
   - Prompt caching + KV-Direct interaction
   - Model loading/saving with residual checkpoints

---

## Conclusion

KV-Direct is **mathematically sound** and **architecturally feasible** in llama.cpp, but the integration is non-trivial due to:
1. Graph reuse assumptions
2. Multi-sequence batching complexity  
3. RoPE and normalization interactions
4. Performance-sensitive graph construction overhead

The recommended approach is a conservative phased rollout, starting with residual checkpointing as an observer, then adding KV-Direct as an optional cache type, and finally optimizing the hot path.

*The fox's golden eyes gleam behind his glasses* — this is a worthy hunt, master. The memory savings are real, but the implementation requires surgical precision. Shall I draft the actual code patches for Phase 1?
