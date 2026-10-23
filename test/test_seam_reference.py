"""Check the backend seam against plain PyTorch attention, on one GPU.

The ring tests compare against ``seam_reference_attn`` / ``seam_reference_varlen_attn``,
which run on the same ``fa_forward``/``fa_backward`` adapter as the ring path. An adapter
bug that hits every block call (a dropped ``softmax_scale``, a swapped ``cu_seqlens``,
the wrong tuple slot for lse) shifts both sides identically and those tests still show
~0 diffs. This test closes that loop with a reference that shares no code with the seam,
so it works the same under every backend (RING_FLASH_ATTN_BACKEND).

Tolerances follow flash-attn's own tests: the seam's error against an fp32 reference
must be within 2x the error of the same math done in bf16, plus a small floor. lse is
fp32 in every backend, so it gets a fixed absolute bound instead.

Run: python test/test_seam_reference.py  (exits non-zero on any mismatch)
"""

import sys

import torch

from ring_flash_attn.flash_attn_backend import (
    BACKEND,
    fa_varlen_backward,
    fa_varlen_forward,
)
from utils import seam_reference_attn, seam_reference_varlen_attn

DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
LSE_ATOL = 1e-3
FAILURES = []


def naive_attn(q, k, v, dout, causal, softmax_scale, upcast):
    """(b, sq, h, d) x (b, sk, hk, d) -> out, lse (b, h, sq), dq, dk, dv.

    ``upcast=False`` keeps scores and softmax in bf16; that run only sets the tolerance.
    Causal masking is bottom-right aligned, as in flash-attn >= 2.1.
    """
    q, k, v = (t.detach().clone().requires_grad_() for t in (q, k, v))
    qc, kc, vc = (t.float() for t in (q, k, v)) if upcast else (q, k, v)
    groups = q.shape[2] // k.shape[2]
    kc, vc = (t.repeat_interleave(groups, dim=2) for t in (kc, vc))
    scores = torch.einsum("bqhd,bkhd->bhqk", qc, kc) * softmax_scale
    if causal:
        sq, sk = scores.shape[-2:]
        mask = torch.ones(sq, sk, dtype=torch.bool, device=q.device).triu(sk - sq + 1)
        scores = scores.masked_fill(mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(scores, dim=-1), vc)
    out.backward(dout.to(out.dtype))
    return out.detach(), lse.detach().float(), q.grad, k.grad, v.grad


def naive_varlen_attn(q, k, v, dout, cu_q, cu_k, causal, softmax_scale, upcast):
    """Packed (total, h, d) inputs; returns lse as (h, total_q)."""
    outs = []
    for i in range(len(cu_q) - 1):
        sq, sk = slice(cu_q[i], cu_q[i + 1]), slice(cu_k[i], cu_k[i + 1])
        outs.append(
            naive_attn(
                q[None, sq], k[None, sk], v[None, sk], dout[None, sq],
                causal, softmax_scale, upcast,
            )
        )
    out, lse, dq, dk, dv = zip(*outs)
    return (
        torch.cat([o[0] for o in out]),
        torch.cat([l[0] for l in lse], dim=-1),
        torch.cat([g[0] for g in dq]),
        torch.cat([g[0] for g in dk]),
        torch.cat([g[0] for g in dv]),
    )


def varlen_lse_to_packed(lse, cu_q):
    """Accept either varlen lse layout and return (h, total_q); fail on anything else."""
    if lse.dim() == 2:
        return lse
    if lse.dim() == 3:  # older FA2: (batch, h, max_seqlen) padded
        return torch.cat(
            [lse[i, :, : cu_q[i + 1] - cu_q[i]] for i in range(len(cu_q) - 1)], dim=-1
        )
    raise AssertionError(f"unexpected varlen lse shape {tuple(lse.shape)}")


def check(case, got, ref, ref_lowp):
    names = ("out", "lse", "dq", "dk", "dv")
    for name, g, r, rl in zip(names, got, ref, ref_lowp):
        if g.shape != r.shape:
            FAILURES.append(f"{case} {name}: shape {tuple(g.shape)} != {tuple(r.shape)}")
            print(f"  {name:3} SHAPE MISMATCH {tuple(g.shape)} vs {tuple(r.shape)}")
            continue
        err = (g.float() - r.float()).abs().max().item()
        if name == "lse":
            # lse is fp32 in every backend, and its bf16 run is too coarse to set a bound
            bound = LSE_ATOL
        else:
            bound = 2 * (rl.float() - r.float()).abs().max().item() + 1e-4
        ok = err <= bound
        print(f"  {name:3} err {err:.3g} <= {bound:.3g} {'ok' if ok else 'FAIL'}")
        if not ok:
            FAILURES.append(f"{case} {name}: err {err:.3g} > bound {bound:.3g}")


def randn(*shape):
    return torch.randn(*shape, device=DEVICE, dtype=DTYPE)


def dense_case(causal, softmax_scale, nheads, nheads_k, d):
    case = f"dense causal={causal} scale={softmax_scale} h={nheads}/{nheads_k} d={d}"
    print(case)
    b, s = 2, 1000
    q, k, v = randn(b, s, nheads, d), randn(b, s, nheads_k, d), randn(b, s, nheads_k, d)
    dout = randn(b, s, nheads, d)
    got = seam_reference_attn(q, k, v, dout, causal, softmax_scale=softmax_scale)
    scale = softmax_scale or d ** (-0.5)
    ref = naive_attn(q, k, v, dout, causal, scale, upcast=True)
    ref_lowp = naive_attn(q, k, v, dout, causal, scale, upcast=False)
    check(case, got, ref, ref_lowp)


def varlen_case(causal, softmax_scale, nheads, nheads_k, d):
    case = f"varlen causal={causal} scale={softmax_scale} h={nheads}/{nheads_k} d={d}"
    print(case)
    cu = [0, 120, 1248, 2232]
    cu_t = torch.tensor(cu, dtype=torch.int32, device=DEVICE)
    max_seqlen = max(b - a for a, b in zip(cu, cu[1:]))
    q, k, v = randn(cu[-1], nheads, d), randn(cu[-1], nheads_k, d), randn(cu[-1], nheads_k, d)
    dout = randn(cu[-1], nheads, d)
    out, lse, dq, dk, dv = seam_reference_varlen_attn(
        q, k, v, dout, cu_t, max_seqlen, causal, softmax_scale=softmax_scale
    )
    got = (out, varlen_lse_to_packed(lse, cu), dq, dk, dv)
    scale = softmax_scale or d ** (-0.5)
    ref = naive_varlen_attn(q, k, v, dout, cu, cu, causal, scale, upcast=True)
    ref_lowp = naive_varlen_attn(q, k, v, dout, cu, cu, causal, scale, upcast=False)
    check(case, got, ref, ref_lowp)


def varlen_cross_case(d):
    """cu_seqlens_q != cu_seqlens_k, which the self-attention helpers never exercise.

    Non-causal, so the result does not depend on the causal alignment convention.
    """
    case = f"varlen q/k lengths differ, non-causal, d={d}"
    print(case)
    nheads = 4
    cu_q, cu_k = [0, 100, 612, 700], [0, 300, 400, 1424]
    cu_q_t, cu_k_t = (
        torch.tensor(c, dtype=torch.int32, device=DEVICE) for c in (cu_q, cu_k)
    )
    max_q = max(b - a for a, b in zip(cu_q, cu_q[1:]))
    max_k = max(b - a for a, b in zip(cu_k, cu_k[1:]))
    q, dout = randn(cu_q[-1], nheads, d), randn(cu_q[-1], nheads, d)
    k, v = randn(cu_k[-1], nheads, d), randn(cu_k[-1], nheads, d)
    scale = d ** (-0.5)
    out, lse = fa_varlen_forward(
        q, k, v, cu_q_t, cu_k_t, max_q, max_k, softmax_scale=scale, causal=False
    )
    dq, dk, dv = (torch.empty_like(t) for t in (q, k, v))
    fa_varlen_backward(
        dout, q, k, v, out, lse, dq, dk, dv, cu_q_t, cu_k_t, max_q, max_k,
        softmax_scale=scale, causal=False,
    )
    got = (out, varlen_lse_to_packed(lse, cu_q), dq, dk, dv)
    ref = naive_varlen_attn(q, k, v, dout, cu_q, cu_k, False, scale, upcast=True)
    ref_lowp = naive_varlen_attn(q, k, v, dout, cu_q, cu_k, False, scale, upcast=False)
    check(case, got, ref, ref_lowp)


def main():
    torch.manual_seed(0)
    print(f"backend: {BACKEND}")
    for causal in (True, False):
        dense_case(causal, None, nheads=4, nheads_k=4, d=64)
        varlen_case(causal, None, nheads=4, nheads_k=4, d=64)
    # a non-default scale catches an adapter that drops or recomputes softmax_scale
    dense_case(True, 0.1, nheads=4, nheads_k=4, d=128)
    varlen_case(True, 0.1, nheads=4, nheads_k=4, d=128)
    # GQA, as used by the llama3 variant
    dense_case(True, None, nheads=8, nheads_k=2, d=32)
    varlen_case(True, None, nheads=8, nheads_k=2, d=32)
    varlen_cross_case(d=64)

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  {f}")
        sys.exit(1)
    print("\nall seam reference checks passed")


if __name__ == "__main__":
    main()
