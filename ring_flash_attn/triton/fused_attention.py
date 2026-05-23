"""
Triton flash-attention backward kernel for ring-flash-attention.

This is a vendored and lightly adapted version of the backward pass from the
ROCm/Tri Dao Triton implementation of FlashAttention:

    https://github.com/ROCm/flash-attention/blob/main/flash_attn/flash_attn_triton.py

which is itself derived from Phil Tillet's OpenAI Triton tutorial
``06-fused-attention.py``. The forward pass is intentionally NOT vendored —
upstream flash-attn provides a faster CUDA forward. This file exposes only a
backward function with the same call signature as
``torch.ops.flash_attn._flash_attn_backward`` so it can be dropped in to
``ring_flash_attn_backward`` and ``llama_flash_attn_backward`` to optionally
compute the per-step backward in fp32 (which the upstream CUDA backward
refuses).

Adaptations to the upstream code:
  * Removed the forward kernel/wrapper and the autograd ``Function`` classes.
  * Replaced deprecated ``tl.dot(..., trans_a=True / trans_b=True)`` with
    explicit ``tl.trans(...)`` calls (Triton 3.x removed the kwargs).
  * Wrapper accepts fp16, bf16, AND fp32 inputs.
  * Optional AMD/HIP-specific kernel kwargs (``waves_per_eu``,
    ``allow_flush_denorm``) cordoned behind a single ``enable_hip_opts`` flag
    so the file is clean for NVIDIA by default.

The bias path is left in the kernels verbatim. The wrapper always invokes the
kernels with ``BIAS_TYPE="none"`` and a null bias pointer, so the bias-handling
branches compile away as dead constexpr code.
"""
import math

import torch
import triton
import triton.language as tl


def _is_hip() -> bool:
    try:
        return triton.runtime.driver.active.get_current_target().backend == "hip"
    except Exception:
        return False


@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_store_dk_dv(
    dk_ptrs,
    dv_ptrs,
    dk,
    dv,
    offs_n,
    offs_d,
    seqlen_k,
    headdim,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
):
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    Q,
    K,
    V,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_bm,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    seqlen_q,
    seqlen_k,
    headdim,
    ATOMIC_ADD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dk_dv(
            dk_ptrs,
            dv_ptrs,
            dk,
            dv,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
        )
        return
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        # recompute p = softmax(qk, dim=-1).T
        qk = tl.dot(q, tl.trans(k), input_precision=DOT_PRECISION)
        if not EVEN_N:
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        if BIAS_TYPE != "none":
            tl.debug_barrier()
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            qk = qk * softmax_scale + bias
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        if BIAS_TYPE == "none":
            p = tl.exp(qk * softmax_scale - lse_i[:, None])
        else:
            p = tl.exp(qk - lse_i[:, None])
        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
        dv += tl.dot(tl.trans(p.to(do.dtype)), do, input_precision=DOT_PRECISION)
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        dp = tl.dot(do, tl.trans(v), input_precision=DOT_PRECISION)
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        Di = tl.load(D + offs_m_curr)
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        dk += tl.dot(tl.trans(ds), q, input_precision=DOT_PRECISION)
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq += tl.dot(ds, k, input_precision=DOT_PRECISION)
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            else:
                if EVEN_HEADDIM:
                    dq = tl.load(
                        dq_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k, input_precision=DOT_PRECISION)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy="evict_last",
                    )
                else:
                    dq = tl.load(
                        dq_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k, input_precision=DOT_PRECISION)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )
        else:
            dq = tl.dot(ds, k, input_precision=DOT_PRECISION)
            if EVEN_M & EVEN_HEADDIM:
                tl.atomic_add(dq_ptrs, dq)
            else:
                if EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                else:
                    tl.atomic_add(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    )
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if BIAS_TYPE == "matrix":
            b_ptrs += BLOCK_M * stride_bm
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
    )


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    if BIAS_TYPE != "none":
        Bias += off_b * stride_bb + off_h * stride_bh
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                Bias,
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_bm,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                BIAS_TYPE=BIAS_TYPE,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                DOT_PRECISION=DOT_PRECISION,
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            start_n,
            Q,
            K,
            V,
            Bias,
            DO,
            DQ,
            DK,
            DV,
            LSE,
            D,
            softmax_scale,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_bm,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            seqlen_q,
            seqlen_k,
            headdim,
            ATOMIC_ADD=True,
            BIAS_TYPE=BIAS_TYPE,
            IS_CAUSAL=IS_CAUSAL,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def triton_flash_attn_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size_left=-1,
    window_size_right=-1,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    *,
    enable_hip_opts: bool = False,
):
    """Triton flash-attention backward with the flash-attn CUDA signature.

    Drop-in replacement for ``torch.ops.flash_attn._flash_attn_backward``.
    Writes gradients into ``dq``/``dk``/``dv`` in place.

    Supports fp16, bf16, AND fp32. Causal and non-causal both supported.
    The following CUDA-backward features are not implemented in the Triton
    kernel; this wrapper raises ``NotImplementedError`` if a caller sets a
    non-default value, because silent ignore would produce wrong gradients:
    ``dropout_p > 0``, ``alibi_slopes is not None``, ``softcap > 0``,
    or any non-default ``window_size_*``. ``deterministic`` is accepted
    (the default ``SEQUENCE_PARALLEL=False`` autotune config is already
    deterministic; the ``SEQUENCE_PARALLEL=True`` config uses ``atomic_add``
    on dq but Triton autotune may still pick it — for strict determinism
    callers should set ``deterministic=True`` and we will force the
    non-parallel config).

    When ``enable_hip_opts=True`` and a HIP/ROCm backend is detected, the
    kernel launch passes ``waves_per_eu`` and ``allow_flush_denorm``. This
    is the only AMD-specific code path; with the flag off (the default) the
    kernel is launched with backend-agnostic kwargs.
    """
    if dropout_p > 0.0:
        raise NotImplementedError(
            "triton_flash_attn_backward does not support dropout_p > 0"
        )
    if alibi_slopes is not None:
        raise NotImplementedError(
            "triton_flash_attn_backward does not support alibi_slopes"
        )
    if softcap > 0.0:
        raise NotImplementedError(
            "triton_flash_attn_backward does not support softcap > 0"
        )
    if window_size_left != -1 or window_size_right != -1:
        raise NotImplementedError(
            "triton_flash_attn_backward does not support sliding window attention"
        )

    if q.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"triton_flash_attn_backward expects q in fp16/bf16/fp32, got {q.dtype}"
        )
    if not (q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype):
        raise TypeError(
            "triton_flash_attn_backward expects q, k, v, out, dout to have the same dtype"
        )

    if dout.stride(-1) != 1:
        dout = dout.contiguous()

    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d), "k must match q shape except seqlen"
    assert v.shape == (batch, seqlen_k, nheads, d), "v must match k shape"
    assert d <= 128, "head dim > 128 not supported by Triton backward"
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == out.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128

    # flash-attn ships LSE with shape (batch, heads, seq); the Triton kernel
    # indexes LSE with seqlen_q_rounded stride. Pad if needed.
    if softmax_lse.shape[-1] != seqlen_q_rounded:
        lse_padded = torch.empty(
            (batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32
        )
        lse_padded[:, :, :seqlen_q].copy_(softmax_lse[:, :, :seqlen_q])
        # Pad with 0; the boundary mask in the kernel skips these rows.
        if seqlen_q_rounded > seqlen_q:
            lse_padded[:, :, seqlen_q:].zero_()
        lse_for_kernel = lse_padded
    else:
        lse_for_kernel = softmax_lse.to(torch.float32) if softmax_lse.dtype != torch.float32 else softmax_lse

    # dq_accum is read-modify-written by the kernel (when SEQUENCE_PARALLEL=False)
    # via load/store with eviction_policy='evict_last'; must be zero-initialized.
    dq_accum = torch.zeros_like(q, dtype=torch.float32)
    delta = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    # Block sizes tuned by element size and per-CTA shared memory budget.
    # fp32 tiles need 4x the SRAM of fp16/bf16, so we shrink for fp32.
    # On consumer Ada (sm_89, ~99 KB shared/CTA) the upstream 128x128 bf16
    # config also OOMs, so we default to 64x64 there too. A100/H100 with
    # >=160 KB could safely run 128x128 — that's a future tuning win.
    if q.dtype == torch.float32:
        BLOCK_M, BLOCK_N = 32, 32
        num_warps = 4
        # Use IEEE-754 fp32 matmul, not TF32 (default on Ampere+).
        dot_precision = "ieee"
    else:
        BLOCK_M, BLOCK_N = 64, 64
        num_warps = 4
        # input_precision is ignored for fp16/bf16; pass tf32 as a no-op default.
        dot_precision = "tf32"
    num_stages = 1
    SEQUENCE_PARALLEL = False  # deterministic (no atomic_add on dq)

    pre_grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[pre_grid](
        out,
        dout,
        delta,
        out.stride(0),
        out.stride(2),
        out.stride(1),
        dout.stride(0),
        dout.stride(2),
        dout.stride(1),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d,
        BLOCK_M=128,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    extra_kern_args = {}
    if enable_hip_opts and _is_hip():
        # Mirror the AMD-tuning idiom from experiments/evoformer.py:
        # higher occupancy for small head dims, flush denormals for speed.
        extra_kern_args["waves_per_eu"] = 3 if d <= 64 else 2
        extra_kern_args["allow_flush_denorm"] = True

    grid = (triton.cdiv(seqlen_k, BLOCK_N) if SEQUENCE_PARALLEL else 1, batch * nheads)
    _bwd_kernel[grid](
        q,
        k,
        v,
        None,                       # Bias (BIAS_TYPE="none")
        dout,
        dq_accum,
        dk,
        dv,
        lse_for_kernel,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        0, 0, 0,                    # bias strides (unused)
        dout.stride(0),
        dout.stride(2),
        dout.stride(1),
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        dk.stride(0),
        dk.stride(2),
        dk.stride(1),
        dv.stride(0),
        dv.stride(2),
        dv.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        "none",                     # BIAS_TYPE
        causal,
        BLOCK_HEADDIM,
        SEQUENCE_PARALLEL,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        DOT_PRECISION=dot_precision,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kern_args,
    )
    # Land the fp32 dq accumulator into the caller's buffer (whatever dtype).
    dq.copy_(dq_accum)
