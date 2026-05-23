"""Single-GPU benchmark: Triton fp32 backward vs flash-attn CUDA backward.

This compares the per-step backward kernels directly (no ring/distributed
machinery). flash-attn's CUDA backward is fp16/bf16-only; the Triton backward
also supports fp32 — that mode is benchmarked separately since it has no
flash-attn counterpart.

Run:
    python benchmark/benchmark_triton_vs_flash_attn_backward.py
    # or pick a single size set / dtype:
    python benchmark/benchmark_triton_vs_flash_attn_backward.py --seqlen 4096

Expect Triton to be slower than flash-attn's CUDA kernel — the upstream
ROCm/Tri Dao README explicitly notes this. The point of fp32 mode is the
precision win for sensitive training, not throughput.
"""
import argparse
import math
import time

import torch

# Optional: flash-attn may not be importable on all environments (ABI
# mismatches against the installed torch version are common). Allow the
# script to still report Triton-only numbers in that case.
try:
    from flash_attn.flash_attn_interface import _flash_attn_backward
    _FLASH_AVAILABLE = True
except Exception as _e:
    _flash_attn_backward = None
    _FLASH_AVAILABLE = False
    _flash_import_error = _e

# Import the Triton backward directly from its submodule path so we don't
# trigger ring_flash_attn/__init__.py (which imports flash_attn).
import importlib.util as _il
import pathlib as _pl
_spec = _il.spec_from_file_location(
    "_rfa_triton_fused_attention",
    _pl.Path(__file__).resolve().parent.parent
    / "ring_flash_attn" / "triton" / "fused_attention.py",
)
_mod = _il.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
triton_flash_attn_backward = _mod.triton_flash_attn_backward


# Llama-3-8B-ish head config; vary seqlen and dtype only.
_DEFAULT_SIZES = [
    # (B, S, H, D)
    (1, 1024, 32, 128),
    (1, 2048, 32, 128),
    (1, 4096, 32, 128),
    (1, 8192, 32, 128),
]


def _setup(B, S, H, D, dtype, device):
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(B, S, H, D, device=device, dtype=dtype)
    sm_scale = 1.0 / math.sqrt(D)
    # Use PyTorch SDPA as a quick forward; we just need (out, lse) as inputs
    # to the backward kernels. The kernels will recompute attention internally.
    q32, k32, v32 = (
        q.detach().float().transpose(1, 2),
        k.detach().float().transpose(1, 2),
        v.detach().float().transpose(1, 2),
    )
    scores = (q32 @ k32.transpose(-1, -2)) * sm_scale
    lse = scores.logsumexp(dim=-1)  # (B, H, S) fp32
    p = scores.softmax(dim=-1)
    out = (p @ v32).transpose(1, 2).contiguous().to(dtype)
    return q.detach(), k.detach(), v.detach(), out, lse, dout, sm_scale


def _bench_callable(fn, num_iter=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / num_iter * 1000.0  # ms / iter


def _attention_bwd_flops(B, S, H, D, causal):
    """FLOPS for one attention backward pass.

    Follows the convention in triton's 06-fused-attention.py benchmark:
    forward is 2 batched matmuls of (B, H, S, D) @ (B, H, D, S) and
    (B, H, S, S) @ (B, H, S, D), each = 2 * B*H*S*S*D MACs.
    Backward does ~2 forward worth of matmul + 0.5 recompute = 2.5x forward.
    Causal halves the work (lower-triangular).
    """
    flops_per_matmul = 2.0 * B * H * S * S * D
    total = 2 * flops_per_matmul  # forward = 2 matmuls
    if causal:
        total *= 0.5
    total *= 2.5  # backward + recompute
    return total


def run_size(B, S, H, D, dtype, causal, device, num_iter, warmup):
    q, k, v, out, lse, dout, sm_scale = _setup(B, S, H, D, dtype, device)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    # --- Triton backward (in native dtype) ---
    def run_triton():
        triton_flash_attn_backward(
            dout, q, k, v, out, lse,
            dq, dk, dv,
            dropout_p=0.0, softmax_scale=sm_scale, causal=causal,
        )
    triton_ms = _bench_callable(run_triton, num_iter, warmup)

    # --- flash-attn CUDA backward (only valid for fp16/bf16, and only if
    # the library imported successfully) ---
    if _FLASH_AVAILABLE and dtype in (torch.float16, torch.bfloat16):
        def run_flash():
            _flash_attn_backward(
                dout, q, k, v, out, lse,
                dq, dk, dv,
                dropout_p=0.0, softmax_scale=sm_scale, causal=causal,
                window_size_left=-1, window_size_right=-1,
                softcap=0.0, alibi_slopes=None, deterministic=False,
            )
        flash_ms = _bench_callable(run_flash, num_iter, warmup)
    else:
        flash_ms = None

    return triton_ms, flash_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-iter", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--seqlen", type=int, default=None,
                   help="Restrict to a single seqlen instead of the full grid")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32", "all"), default="all")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sizes = _DEFAULT_SIZES if args.seqlen is None else [
        (1, args.seqlen, 32, 128),
    ]
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if args.dtype == "all":
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
    else:
        dtypes = [dtype_map[args.dtype]]

    print(f"# Triton-vs-flash-attn backward, causal={args.causal}, "
          f"iters={args.num_iter} (after {args.warmup} warmup)")
    print(f"# GPU: {torch.cuda.get_device_name()}")
    print("# Throughput uses the attention-bwd FLOP count (fwd + 0.5x recompute, "
          "halved for causal), matching the triton/Tri Dao convention.")
    if not _FLASH_AVAILABLE:
        print(f"# flash-attn unavailable ({type(_flash_import_error).__name__}: "
              f"{_flash_import_error}); reporting Triton numbers only.")
    print()
    header = (
        f"{'dtype':>6s}  {'(B,S,H,D)':>22s}  "
        f"{'triton ms':>10s}  {'flash ms':>10s}  {'triton TFLOPS':>14s}  "
        f"{'flash TFLOPS':>13s}  {'tok/s (triton)':>16s}  "
        f"{'tok/s (flash)':>15s}  {'speedup':>8s}"
    )
    print(header)
    for dtype in dtypes:
        for (B, S, H, D) in sizes:
            try:
                triton_ms, flash_ms = run_size(
                    B, S, H, D, dtype, args.causal, device,
                    args.num_iter, args.warmup,
                )
            except Exception as e:
                print(f"{str(dtype):>6s}  {(B,S,H,D)!s:>22s}  FAILED: {e}")
                continue
            flops = _attention_bwd_flops(B, S, H, D, args.causal)
            tokens = B * S
            triton_tflops = flops * 1e-12 / (triton_ms * 1e-3)
            triton_tok = tokens / (triton_ms * 1e-3)
            if flash_ms is None:
                flash_ms_str = "(n/a)"
                flash_tflops_str = "—"
                flash_tok_str = "—"
                speedup = "—"
            else:
                flash_tflops = flops * 1e-12 / (flash_ms * 1e-3)
                flash_tok = tokens / (flash_ms * 1e-3)
                flash_ms_str = f"{flash_ms:10.3f}"
                flash_tflops_str = f"{flash_tflops:13.2f}"
                flash_tok_str = f"{flash_tok:15.0f}"
                # >1 means Triton is faster; <1 means flash-attn is faster.
                speedup = f"{flash_ms/triton_ms:.2f}x"
            dtype_short = {torch.bfloat16: "bf16",
                           torch.float16: "fp16",
                           torch.float32: "fp32"}[dtype]
            print(
                f"{dtype_short:>6s}  {(B,S,H,D)!s:>22s}  "
                f"{triton_ms:10.3f}  {flash_ms_str:>10s}  "
                f"{triton_tflops:14.2f}  {flash_tflops_str:>13s}  "
                f"{triton_tok:16.0f}  {flash_tok_str:>15s}  "
                f"{speedup:>8s}"
            )


if __name__ == "__main__":
    main()
