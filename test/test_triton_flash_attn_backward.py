"""Unit tests for ring_flash_attn.triton.fused_attention.triton_flash_attn_backward.

Compares the Triton backward output against a high-precision PyTorch autograd
reference across dtypes (fp16, bf16, fp32), causal/non-causal, head dims, and
sequence lengths — including non-divisible seqlens and head dims that are not
powers of two. fp32 is the headline capability since flash-attn's CUDA backward
rejects it.

Run on a single GPU (no torchrun needed):

    pytest -xvs test/test_triton_flash_attn_backward.py
"""
import math

import pytest
import torch

from ring_flash_attn.triton.fused_attention import triton_flash_attn_backward


@pytest.fixture(autouse=True)
def _seed_and_disable_tf32():
    torch.manual_seed(0)
    # Keep the reference matmul in true fp32 even when GPU defaults to TF32.
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = prev_matmul
    torch.backends.cudnn.allow_tf32 = prev_cudnn


def _reference_attention_backward(q, k, v, dout, softmax_scale, causal):
    """Compute (out, lse, dq, dk, dv) in fp32 via PyTorch autograd."""
    device = q.device
    B, S, H, D = q.shape
    q32 = q.detach().float().transpose(1, 2).requires_grad_()
    k32 = k.detach().float().transpose(1, 2).requires_grad_()
    v32 = v.detach().float().transpose(1, 2).requires_grad_()
    scores = (q32 @ k32.transpose(-1, -2)) * softmax_scale
    if causal:
        mask = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    p = scores.softmax(dim=-1)
    out_full = (p @ v32).transpose(1, 2).contiguous()
    lse = scores.logsumexp(dim=-1)
    out_full.backward(dout.float())
    dq_ref = q32.grad.transpose(1, 2).contiguous()
    dk_ref = k32.grad.transpose(1, 2).contiguous()
    dv_ref = v32.grad.transpose(1, 2).contiguous()
    return out_full, lse, dq_ref, dk_ref, dv_ref


# Tolerance per dtype, max-abs error on dq/dk/dv after upcast to fp32.
TOL = {
    torch.float32: 5e-6,
    torch.float16: 5e-3,
    torch.bfloat16: 3e-2,
}


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("B,S,H,D", [
    (1, 128, 2, 64),
    (1, 127, 2, 64),     # non-divisible seq
    (2, 256, 4, 40),     # non-power-of-2 head dim
    (1, 1024, 2, 128),   # max head dim
])
def test_triton_backward_matches_reference(B, S, H, D, dtype, causal):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)
    sm_scale = 1.0 / math.sqrt(D)

    out_full, lse, dq_ref, dk_ref, dv_ref = _reference_attention_backward(
        q, k, v, torch.randn(B, S, H, D, device=device, dtype=dtype), sm_scale, causal
    )
    # Re-make dout deterministically (autograd already consumed our first one).
    torch.manual_seed(1)
    dout = torch.randn(B, S, H, D, device=device, dtype=dtype)
    # Redo the reference backward with the same dout so it matches our run.
    out_full, lse, dq_ref, dk_ref, dv_ref = _reference_attention_backward(
        q, k, v, dout, sm_scale, causal
    )

    out = out_full.to(dtype)
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    triton_flash_attn_backward(
        dout, q, k, v, out, lse,
        dq, dk, dv,
        softmax_scale=sm_scale, causal=causal,
    )
    tol = TOL[dtype]
    assert (dq.float() - dq_ref).abs().max().item() < tol, (
        f"dq mismatch for B={B} S={S} H={H} D={D} dtype={dtype} causal={causal}"
    )
    assert (dk.float() - dk_ref).abs().max().item() < tol, (
        f"dk mismatch for B={B} S={S} H={H} D={D} dtype={dtype} causal={causal}"
    )
    assert (dv.float() - dv_ref).abs().max().item() < tol, (
        f"dv mismatch for B={B} S={S} H={H} D={D} dtype={dtype} causal={causal}"
    )


def test_triton_backward_raises_on_unsupported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    B, S, H, D = 1, 64, 2, 32
    dtype = torch.float32
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)
    dout = torch.randn(B, S, H, D, device=device, dtype=dtype)
    out = torch.zeros_like(q)
    lse = torch.zeros((B, H, S), device=device, dtype=torch.float32)
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)

    with pytest.raises(NotImplementedError, match="dropout"):
        triton_flash_attn_backward(dout, q, k, v, out, lse, dq, dk, dv, dropout_p=0.1)
    with pytest.raises(NotImplementedError, match="alibi"):
        triton_flash_attn_backward(dout, q, k, v, out, lse, dq, dk, dv,
                                   alibi_slopes=torch.zeros(H, device=device))
    with pytest.raises(NotImplementedError, match="softcap"):
        triton_flash_attn_backward(dout, q, k, v, out, lse, dq, dk, dv, softcap=10.0)
    with pytest.raises(NotImplementedError, match="window"):
        triton_flash_attn_backward(dout, q, k, v, out, lse, dq, dk, dv,
                                   window_size_left=128, window_size_right=128)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("B,S,H,D", [
    (1, 512, 8, 64),
    (1, 1024, 8, 64),
])
def test_fp32_strictly_more_accurate_than_bf16(B, S, H, D, causal, capsys):
    """The headline reason to use the Triton backward is precision.

    With bf16 inputs:
      - The native-dtype bf16 backward accumulates and stores everything in bf16.
      - The "fp32" mode upcasts inputs to fp32 before the kernel runs, so the
        per-step backward is computed in IEEE fp32 and only the final cast back
        to q.dtype is lossy.

    This test confirms the fp32 path is strictly closer to a true-fp32 PyTorch
    autograd reference than the bf16 path is. Also prints the gap for the log.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)
    dout = torch.randn(B, S, H, D, device=device, dtype=dtype)
    sm_scale = 1.0 / math.sqrt(D)

    out_full, lse, dq_ref, dk_ref, dv_ref = _reference_attention_backward(
        q, k, v, dout, sm_scale, causal
    )
    out_bf16 = out_full.to(dtype)

    # --- Path A: native bf16 backward ---
    dq_bf16 = torch.zeros_like(q)
    dk_bf16 = torch.zeros_like(k)
    dv_bf16 = torch.zeros_like(v)
    triton_flash_attn_backward(
        dout, q, k, v, out_bf16, lse,
        dq_bf16, dk_bf16, dv_bf16,
        softmax_scale=sm_scale, causal=causal,
    )

    # --- Path B: upcast to fp32, run the kernel in fp32, cast back ---
    q_f, k_f, v_f, out_f, dout_f = (
        q.float(), k.float(), v.float(), out_bf16.float(), dout.float()
    )
    dq_fp32 = torch.zeros_like(q_f)
    dk_fp32 = torch.zeros_like(k_f)
    dv_fp32 = torch.zeros_like(v_f)
    triton_flash_attn_backward(
        dout_f, q_f, k_f, v_f, out_f, lse,
        dq_fp32, dk_fp32, dv_fp32,
        softmax_scale=sm_scale, causal=causal,
    )
    # Cast back to bf16 to match how the autograd Function returns gradients.
    dq_fp32_b = dq_fp32.to(dtype)
    dk_fp32_b = dk_fp32.to(dtype)
    dv_fp32_b = dv_fp32.to(dtype)

    def maxerr(a, b):
        return (a.float() - b).abs().max().item()

    err_bf16 = {
        "dq": maxerr(dq_bf16, dq_ref),
        "dk": maxerr(dk_bf16, dk_ref),
        "dv": maxerr(dv_bf16, dv_ref),
    }
    err_fp32 = {
        "dq": maxerr(dq_fp32_b, dq_ref),
        "dk": maxerr(dk_fp32_b, dk_ref),
        "dv": maxerr(dv_fp32_b, dv_ref),
    }

    with capsys.disabled():
        print(f"\n  B={B} S={S} H={H} D={D} causal={causal}")
        for g in ("dq", "dk", "dv"):
            ratio = err_bf16[g] / max(err_fp32[g], 1e-12)
            print(f"    {g}: bf16-bwd err={err_bf16[g]:.3e}  "
                  f"fp32-bwd err={err_fp32[g]:.3e}  bf16/fp32={ratio:.2f}x")

    # The fp32 path should be at least as accurate as the bf16 path on every
    # gradient. Allow a tiny slack for any single gradient (1.1x) since rounding
    # of the final bf16 cast can occasionally produce a tie on small tensors,
    # but require a clear win on the average.
    for g in ("dq", "dk", "dv"):
        assert err_fp32[g] <= err_bf16[g] * 1.1, (
            f"fp32 path is not at least as accurate as bf16 for {g}: "
            f"bf16={err_bf16[g]} fp32={err_fp32[g]}"
        )
    mean_bf16 = sum(err_bf16.values()) / 3
    mean_fp32 = sum(err_fp32.values()) / 3
    assert mean_fp32 < mean_bf16, (
        f"fp32 path is not on average more accurate than bf16: "
        f"mean bf16={mean_bf16} mean fp32={mean_fp32}"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-xvs"]))
