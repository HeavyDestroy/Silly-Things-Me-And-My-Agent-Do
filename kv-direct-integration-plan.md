# KV-Direct Integration Plan for llama.cpp

## Core Insight

The KV cache is provably redundant. Keys and values are deterministic linear projections of the residual stream:

```
K = (x_resid + b_k) @ W_k^T
V = (x_resid + b_v) @ W_v^T
```

Store one residual vector per token (~5KB for Qwen3.5-27B) instead of full KV pairs (~136KB). Recompute K/V on demand with **zero reconstruction error**.

## Architecture Overview

### Current llama.cpp KV Cache Flow

```
Token t enters layer il:
  1. inpL = residual stream from previous layer (or token embedding for layer 0)
  2. attn_norm(inpL) → Q, K, V projections
  3. K, V stored in kv_cache.layers[il].{k,v} at cell index = t
  4. Attention computation uses cached K,V for all prior tokens
  5. Residual connection: cur = attn_out + inpL
  6. FFN block with another residual connection
  7. Final residual stream exits layer → becomes inpL for next layer
```

### KV-Direct Modified Flow

```
Token t enters layer il:
  1. inpL = residual stream from previous layer
  2. attn_norm(inpL) → Q, K, V projections
  3. Store ONLY the residual stream (inpL) in residual_cache at index t
  4. For attention, recompute K,V on-demand:
     - Recent tokens (within window W): use fast path, recompute from residual
     - Older tokens: recompute from residual checkpoint
  5. No KV cache storage — memory bounded by residual checkpoints + window W
```

## Implementation Strategy

### Phase 1: Add Residual Stream Checkpointing

**File: `src/llama-kv-cache.h`**

Add a new cache type that stores residual streams instead of KV pairs:

```cpp
// New enum value for KV-Direct mode
enum llama_kv_cache_type {
    LLAMA_KV_CACHE_TYPE_DEFAULT,
    LLAMA_KV_CACHE_TYPE_SWA,        // existing sliding window attention
    LLAMA_KV_CACHE_TYPE_DIRECT,     // KV-Direct: residual checkpoints only
};
```

**File: `src/llama-cparams.h`**

Add configuration parameters:

```cpp
struct llama_cparams {
    // ... existing fields ...
    
    // KV-Direct settings
    bool kv_direct = false;              // Enable KV-Direct mode
    uint32_t kv_direct_window = 64;      // Recent tokens with fast-path access
    uint32_t kv_direct_checkpoint_interval = 1;  // Checkpoint every N tokens
};
```

### Phase 2: Modify Graph Building to Capture Residuals

**File: `src/models/qwen35.cpp`** (and other model files)

The residual stream is already computed — we just need to expose it:

```cpp
// In build_layer_attn() and build_layer_attn_linear():
// After line 49: cur = ggml_add(ctx0, cur, inpSA);
// 
// inpSA IS the residual stream from the previous layer
// cur (after add) IS the residual stream after this layer's attention
//
// We need to register these tensors as "residual checkpoints" 
// that the KV-Direct cache can access

// Add a callback or registration point:
if (cparams.kv_direct) {
    // Register inpSA as the residual checkpoint for this token at this layer
    // This tensor is already in the compute graph — we just need to 
    // make it accessible to the cache layer
    res->residual_checkpoints[il] = inpSA;
}
```

### Phase 3: Implement On-Demand K/V Recomputation

**File: `src/llama-kv-cache.cpp`**

Replace `get_k()` and `get_v()` to recompute from residuals:

```cpp
ggml_tensor * llama_kv_cache::get_k_direct(
    ggml_context * ctx, 
    int32_t il, 
    uint32_t token_idx) const {
    
    // Retrieve residual checkpoint for this token at this layer
    ggml_tensor * resid = get_residual_checkpoint(il, token_idx);
    
    // Recompute K from residual: K = (resid + b_k) @ W_k^T
    // This is a single matmul — much cheaper than storing the full cache
    
    // Get the K projection weights for this layer
    const auto & layer = model.layers[il];
    ggml_tensor * w_k = layer.wk;  // or wqkv depending on architecture
    
    // K = resid @ W_k^T (simplified — actual implementation needs 
    // to handle bias, head splitting, etc.)
    ggml_tensor * k_recomputed = ggml_mul_mat(ctx, w_k, resid);
    
    return k_recomputed;
}
```

### Phase 4: Hybrid Cache Manager

**New file: `src/llama-kv-cache-direct.h`**

```cpp
class llama_kv_cache_direct : public llama_memory_i {
public:
    struct residual_checkpoint {
        uint32_t token_idx;
        uint32_t layer_idx;
        ggml_tensor * residual;  // [n_embd, n_tokens=1, n_seqs]
    };
    
    // Bounded memory: only store W recent tokens + checkpoints
    void set_window_size(uint32_t window);
    
    // Store a residual checkpoint for token t at layer il
    void store_checkpoint(uint32_t t, uint32_t il, ggml_tensor * resid);
    
    // Retrieve and recompute K/V from checkpoint
    ggml_tensor * recompute_k(ggml_context * ctx, uint32_t t, uint32_t il) const;
    ggml_tensor * recompute_v(ggml_context * ctx, uint32_t t, uint32_t il) const;
    
    // Memory is bounded: O(window_size * n_layers * n_embd)
    // NOT O(context_length * n_layers * n_heads * head_dim)
    size_t memory_usage() const override;
};
```

## Memory Analysis

### Standard KV Cache (Qwen3.5-27B, 128k context)

```
n_layers = 64 (48 DeltaNet + 16 attention)
n_heads_kv = varies by layer (128 for DeltaNet, 256 for attention)
head_dim = 128 (DeltaNet), 256 (attention)
context = 131,072 tokens

KV memory ≈ context × n_layers × n_heads_kv × head_dim × 2 (K+V) × dtype_size
         ≈ 131072 × 64 × ~150 × ~150 × 2 × 2 bytes (fp16)
         ≈ 96 GB (!!!)

Even with quantization (Q4_K): ~24 GB
```

### KV-Direct (same model, unbounded context)

```
n_embd = 8192 (hidden dimension)
window_size = 64 (recent tokens for fast path)
checkpoint_interval = 1 (store every token's residual)

Residual memory = window_size × n_layers × n_embd × dtype_size
                = 64 × 64 × 8192 × 2 bytes
                ≈ 67 MB

Plus checkpoint storage (can be evicted to disk):
  Full context residuals = 131072 × 64 × 8192 × 2 ≈ 136 GB
  BUT: checkpoints can be paged, compressed, or selectively evicted
  
Active memory stays bounded at ~67 MB regardless of context length!
```

## Integration Points in llama.cpp

### 1. Model Loading (`src/llama-model.cpp`)

No changes needed — the projection weights (W_k, W_v) are already loaded.

### 2. Context Creation (`src/llama-context.cpp`)

```cpp
// In llama_init_context_with_model():
if (cparams.kv_direct) {
    ctx->kv = new llama_kv_cache_direct(model, cparams);
} else {
    ctx->kv = new llama_kv_cache(model, ...);  // existing path
}
```

### 3. Graph Building (`src/llama-graph.cpp`)

The compute graph already produces residual streams — we just need to:
1. Mark them as "checkpointable" tensors
2. Wire them into the KV-Direct cache after each layer

### 4. Attention Computation (`src/models/*.cpp`)

Replace calls to `inp->get_attn()->get_k()/get_v()` with:
```cpp
if (cparams.kv_direct) {
    // Recompute from residual checkpoint
    k = kv_cache->recompute_k(ctx, token_idx, il);
    v = kv_cache->recompute_v(ctx, token_idx, il);
} else {
    // Standard cache lookup
    k = inp->get_attn()->get_k(ctx, il);
    v = inp->get_attn()->get_v(ctx, il);
}
```

## Performance Considerations

### Recomputation Cost

For each attention query against token t:
- **Standard**: Memory read of cached K,V (~136KB for Qwen3.5-27B)
- **KV-Direct**: One matmul (resid @ W_k^T) + one matmul (resid @ W_v^T)

The paper shows recomputation is **up to 5x faster** than reading cached tensors at moderate batch sizes because:
1. Matrix multiply is highly optimized (cuBLAS, etc.)
2. Memory bandwidth is the bottleneck for large caches
3. Residual vectors fit in L2/L3 cache

### When KV-Direct Wins

- **Long contexts** (>8k tokens): Memory savings dominate
- **Multi-turn conversations**: Bounded memory prevents OOM
- **Batched inference**: Recomputation parallelizes well
- **Memory-constrained environments**: Mobile, edge devices

### When Standard Cache Wins

- **Short contexts** (<1k tokens): Overhead of recomputation not justified
- **Single-query latency**: Cache lookup is faster for one-off queries
- **Very large batch sizes**: Amortized cache read cost becomes negligible

## Testing Plan

1. **Correctness**: Verify token-identical output vs standard cache (paper claims 100% match)
2. **Memory**: Confirm bounded memory usage with increasing context
3. **Performance**: Benchmark recomputation vs cache lookup across batch sizes
4. **Edge cases**: 
   - Mixed DeltaNet + attention layers (Qwen3.5 hybrid)
   - Multi-sequence batching
   - Speculative decoding integration

## References

- Paper: "The Residual Stream Is All You Need" — arXiv:2603.19664
- Repo: https://github.com/Kaleemullahqasim/KV-Direct
- PDF: ~/kenji-workspace/KV-Research/kv-direct-paper.pdf
