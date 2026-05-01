# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Ring Flash Attention is a distributed attention library implementing sequence parallelism across multiple GPUs. It wraps `flash_attn` primitives inside a ring-topology communication pattern so each GPU holds only a shard of Q/K/V while maintaining numerically correct attention across all ranks.

## Commands

**Install from source:**
```bash
pip install .
```

**Run a single test (requires 8 GPUs by default):**
```bash
torchrun --nproc_per_node 8 test/test_ring_flash_attn_func.py
torchrun --nproc_per_node 4 test/test_llama_flash_attn.py
```

**Run the full test suite (also runs with `torch.compile`):**
```bash
bash test/test.sh
```

**Benchmarks:**
```bash
torchrun --nproc_per_node 8 benchmark/benchmark_varlen_kvpacked_func.py
```

**Formatting:**
```bash
black .
```

## Architecture

### Attention variants (ring_flash_attn/)

Each variant is a self-contained module with matching `*_func`, `*_kvpacked_func`, and `*_qkvpacked_func` entry points:

| Module | API | Notes |
|---|---|---|
| `ring_flash_attn.py` | batch | Baseline ring attention |
| `ring_flash_attn_varlen.py` | varlen | Variable-length / packed sequences |
| `zigzag_ring_flash_attn.py` | batch | Causal only; 85% efficiency on H800 (vs 52% baseline) |
| `zigzag_ring_flash_attn_varlen.py` | varlen | Zigzag + packed sequences |
| `stripe_flash_attn.py` | batch | Stripe variant (arXiv 2311.09431); causal only |
| `llama3_flash_attn_varlen.py` | varlen | Llama3 context parallelism (arXiv 2407.21783); **recommended for most use cases** |
| `llama_fwd_ring_bwd_flash_attn.py` | varlen | Hybrid: ring forward, flash backward; lower memory |

### Core abstractions (ring_flash_attn/utils.py)

- **`RingComm`** — P2P send/recv over a `torch.distributed` process group arranged in a ring. `send_recv()` / `send_recv_kv()` overlap compute and communication; `commit()` + `wait()` finalize transfers.
- **`AllGatherComm`** — wraps collective all-gather in the same interface so callers can swap communication strategies.
- **`update_out_and_lse()`** — numerically-stable LogSumExp accumulation that merges a new attention block result into the running output and LSE tensors. This is the heart of the numerically-correct ring reduction.
- **`flatten_varlen_lse()` / `unflatten_varlen_lse()`** — reshape LSE tensors between flash_attn's `(batch, heads, seqlen)` layout and the packed varlen layout.

### HuggingFace adapter (ring_flash_attn/adapters/hf_adapter.py)

`substitute_hf_flash_attn()` monkey-patches HF attention modules at runtime to route through ring attention. `DATA_PARAMS` is a module-level dict that conveys `cu_seqlens` and related state across rank boundaries during forward passes. Call `update_ring_flash_attn_params()` before each forward step and toggle `use_ring_attn` to switch between ring and vanilla flash attention.

### Triton kernels (ring_flash_attn/triton_utils.py)

`flatten_kernel` / `unflatten_kernel` are Triton implementations of the LSE reshape operations, used instead of the PyTorch versions when available.

### Key design constraints

- **No dropout support** — saving RNG states across all ring ranks is not implemented.
- **No sliding-window attention** — the varlen implementation does not support `window_size`.
- **fp32 LSE buffer** — outputs accumulate in fp32 to avoid bf16 precision loss; this increases memory above the theoretical minimum.
- All tests use `torchrun` with NCCL; NVLink is required for competitive throughput.
