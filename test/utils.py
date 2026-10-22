import random

import torch
import torch.distributed as dist


def set_seed(rank, seed=42):
    seed = rank + seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"{msg}: "
                f"max {a.abs().max().item():.3g}, "
                f"mean {a.abs().mean().item():.3g}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[{rank}] "
                f"max {a.abs().max().item():.3g}, "
                f"mean {a.abs().mean().item():.3g}",
                flush=True,
            )
        dist.barrier()


def seam_reference_attn(q, k, v, dout, causal, softmax_scale=None):
    """Plain (non-ring) attention fwd+bwd on the full sequence via the backend seam.

    Uses whichever backend ring_flash_attn selected (RING_FLASH_ATTN_BACKEND), so the
    reference runs on the same kernels as the ring path and the test works with
    flash-attn-4 alone, where `from flash_attn import flash_attn_*_func` does not exist.
    Inputs are (batch, seqlen, nheads, d); returns out, lse (batch, nheads, seqlen) and
    dq, dk, dv.
    """
    from ring_flash_attn.flash_attn_backend import fa_backward, fa_forward

    q, k, v = (t.detach().contiguous() for t in (q, k, v))
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, lse = fa_forward(q, k, v, softmax_scale=softmax_scale, causal=causal)
    dq, dk, dv = (torch.empty_like(t) for t in (q, k, v))
    fa_backward(
        dout.contiguous(), q, k, v, out, lse, dq, dk, dv,
        softmax_scale=softmax_scale, causal=causal,
    )
    return out, lse, dq, dk, dv


def seam_reference_varlen_attn(q, k, v, dout, cu_seqlens, max_seqlen, causal, softmax_scale=None):
    """Varlen counterpart of `seam_reference_attn`, for packed (total, nheads, d) inputs.

    Self-attention only: cu_seqlens and max_seqlen are used for both q and k. Returns out,
    lse and dq, dk, dv; lse keeps whatever layout the selected backend produces.
    """
    from ring_flash_attn.flash_attn_backend import (
        fa_varlen_backward,
        fa_varlen_forward,
    )

    q, k, v = (t.detach().contiguous() for t in (q, k, v))
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, lse = fa_varlen_forward(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        softmax_scale=softmax_scale, causal=causal,
    )
    dq, dk, dv = (torch.empty_like(t) for t in (q, k, v))
    fa_varlen_backward(
        dout.contiguous(), q, k, v, out, lse, dq, dk, dv,
        cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        softmax_scale=softmax_scale, causal=causal,
    )
    return out, lse, dq, dk, dv
