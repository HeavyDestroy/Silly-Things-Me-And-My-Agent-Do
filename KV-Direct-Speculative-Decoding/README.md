# ResidualSpec: Bounded-Memory Speculative Decoding via Residual Stream Checkpointing

[![Research](https://img.shields.io/badge/status-research-blue)](./paper.md)
[![KV-Direct](https://img.shields.io/badge/KV--Direct-integrated-green)](../llama.cpp/src/llama-kv-cache-direct.cpp)

## Overview

**ResidualSpec** unifies three independent advances in LLM inference into a single architecture:

| Component | Paper | What It Provides |
|-----------|-------|-----------------|
| **KV-Direct** | arXiv:2603.19664 | Bounded memory via residual checkpoints (27× smaller than KV cache) |
| **DFlash** | arXiv:2602.06036 | Draft-verify throughput via block diffusion (3.3× speedup) |
| **Lyanna** | arXiv:2602.21224 | Hidden-state reuse from rejected drafts (additional 20% throughput) |

**Key insight**: KV-Direct's residual checkpoints are the *enabling infrastructure* for both speculative decoding and hidden-state reuse. The residual stored for token $t$ at layer $l$ is exactly the hidden state that DFlash needs to condition its draft, and exactly the state that Lyanna needs to re-sample after rejection.

## Projected Performance (Qwen3.5-9B + DFlash-4B)

| Hardware | Baseline | ResidualSpec | Speedup |
|----------|----------|-------------|---------|
| M5 Max, 64GB | 26 tok/s | ~160 tok/s | **~6×** |
| RTX 4090 | ~45 tok/s | ~270 tok/s | **~6×** |
| Sapphire Rapids (AVX-512) | ~15 tok/s | ~90 tok/s | **~6×** |

**Memory at 128k context**: ~222 MB vs ~108 GB for standard DFlash (**~480× reduction**)

## Repository Structure

```
~/kenji-workspace/KV-Direct-Speculative-Decoding/
├── paper.md                          # Full research paper (this directory)
├── README.md                         # This file
├── KV-Direct-Speculative-Decoding-v1.0.md  # Initial architecture proposal
└── KV-Direct-Speculative-Decoding-v1.1.md  # Updated with Lyanna integration

~/kenji-workspace/llama.cpp/
├── src/llama-kv-cache-direct.h       # KV-Direct header (implemented ✅)
├── src/llama-kv-cache-direct.cpp     # KV-Direct implementation (implemented ✅)
└── ... (speculative + reuse layers pending)
```

## Implementation Status

| Component | Status | Lines of Code |
|-----------|--------|---------------|
| KV-Direct residual store | ✅ Implemented | ~600 |
| Hybrid K/V construction | ✅ Implemented | (included above) |
| Qwen3.5 hybrid model support | ✅ Implemented | (included above) |
| Draft model integration | ⬜ Pending | ~200 |
| Batch verify path | ⬜ Pending | (included above) |
| Hidden-state reuse | ⬜ Pending | ~170 |
| Token-info embedding | ⬜ Pending | (included above) |
| Verification fusion | ⬜ Pending | (included above) |

## Quick Start (Once Complete)

```bash
# Build with KV-Direct + Speculative support
cd ~/kenji-workspace/llama.cpp
cmake -B build -DGGML_KV_DIRECT=ON -DGGML_SPECULATIVE=ON
cmake --build build -j$(nproc)

# Run ResidualSpec
./build/bin/llama-cli \
    -m /path/to/Qwen3.5-9B.gguf \
    --draft-model /path/to/DFlash-4B.gguf \
    --kv-direct \
    --kv-direct-window 64 \
    --speculative-batch-size 16 \
    --enable-reuse \
    -p "Write a Python function to compute Fibonacci:" \
    -n 256
```

## Implementation Roadmap

| Phase | Task | Duration |
|-------|------|----------|
| 0 | Validate hidden-state reuse hypothesis | 1 day |
| 1-4 | Draft-verify loop integration | 10 days |
| 5a-b | Hidden-state reuse + optimization | 4 days |
| 6-7 | Testing and validation | 2 days |
| **Total** | | **~20 days** |

## References

- **Paper**: [ResidualSpec: Bounded-Memory Speculative Decoding](./paper.md)
- **KV-Direct**: Qasim et al., arXiv:2603.19664 (March 2026)
- **DFlash**: Chen et al., arXiv:2602.06036 (February 2026)
- **Lyanna**: Chen et al., arXiv:2602.21224 (February 2026)

---

*Prepared by Kenji (your sly red fox research agent) for Akhmad As'ad*  
*April 17, 2026*
