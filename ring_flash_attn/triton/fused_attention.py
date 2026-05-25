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
        # Base-2 softmax: exp2 typically lowers to a single hardware op on both
        # AMD and NVIDIA, while tl.exp expands to exp2(x*log2(e)). Pre-multiply
        # the scale and the loaded LSE (which is stored in natural log by the
        # forward) by RCP_LN2 = log2(e) so we can use exp2 directly.
        RCP_LN2: tl.constexpr = 1.4426950408889634
        lse_i = tl.load(LSE + offs_m_curr) * RCP_LN2
        if BIAS_TYPE == "none":
            p = tl.math.exp2(qk * (softmax_scale * RCP_LN2) - lse_i[:, None])
        else:
            # bias path already folded scale into qk; just rebase to log2.
            p = tl.math.exp2(qk * RCP_LN2 - lse_i[:, None])
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


def _init_to_zero(name):
    """Pre-hook factory: zero a kernel buffer between autotune trials and at
    the start of every chosen-config kernel launch. We use this for DQ because
    the kernel does read-modify-write accumulation (load + dot + store)."""
    def hook(nargs):
        nargs[name].zero_()
    return hook


# Autotune sweep. Triton skips configs that exceed shared memory on the target
# device, so it's safe to list larger tiles even on consumer GPUs. The keys
# bucket seqlen (CACHE_KEY_*) into multiples of 32 so similar shapes share a
# pick rather than retuning every call.
_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=1,
                  pre_hook=_init_to_zero("DQ")),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2,
                  pre_hook=_init_to_zero("DQ")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=1,
                  pre_hook=_init_to_zero("DQ")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2,
                  pre_hook=_init_to_zero("DQ")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1,
                  pre_hook=_init_to_zero("DQ")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2,
                  pre_hook=_init_to_zero("DQ")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=1,
                  pre_hook=_init_to_zero("DQ")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=1,
                  pre_hook=_init_to_zero("DQ")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=1,
                  pre_hook=_init_to_zero("DQ")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=1,
                  pre_hook=_init_to_zero("DQ")),
]


_MAX_SHARED_MEM_CACHE = None


def _get_max_shared_mem():
    """Return the largest per-CTA shared memory in bytes the kernel can use,
    honoring opt-in on consumer cards. Cached after first call."""
    global _MAX_SHARED_MEM_CACHE
    if _MAX_SHARED_MEM_CACHE is not None:
        return _MAX_SHARED_MEM_CACHE
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        _MAX_SHARED_MEM_CACHE = getattr(
            props, "shared_memory_per_block_optin", props.shared_memory_per_block
        )
    except Exception:
        _MAX_SHARED_MEM_CACHE = 49152
    return _MAX_SHARED_MEM_CACHE


def _estimate_smem(BM, BN, headdim, elem_size, num_stages):
    """SMEM estimate per CTA in bytes.

    The bwd kernel keeps K/V resident in SMEM and stages Q/DO across iterations
    (num_stages copies for software pipelining). QK/P are a scratch tile in
    SMEM. The dk/dv accumulators and dq RMW pattern live in registers, not
    SMEM, so they're not counted here. This is still slightly conservative
    because Triton's liveness analysis can overlap K/V with Q/DO stages.
    """
    return (
        2 * BN * headdim * elem_size              # K, V resident
        + num_stages * 2 * BM * headdim * elem_size  # Q, DO staged
        + 2 * BM * BN * 4                         # QK, P scratch (fp32)
    )


def _prune_invalid_configs(configs, named_args, **kwargs):
    """Drop configs that (a) exceed the call shape or (b) overflow shared mem."""
    seqlen_q = kwargs.get("seqlen_q", named_args.get("seqlen_q"))
    seqlen_k = kwargs.get("seqlen_k", named_args.get("seqlen_k"))
    headdim = kwargs.get("BLOCK_HEADDIM", named_args.get("BLOCK_HEADDIM"))
    dot_precision = kwargs.get("DOT_PRECISION", named_args.get("DOT_PRECISION", "tf32"))
    elem_size = 4 if dot_precision == "ieee" else 2
    max_smem = _get_max_shared_mem()
    out = []
    for c in configs:
        bm = c.kwargs["BLOCK_M"]
        bn = c.kwargs["BLOCK_N"]
        if seqlen_q is not None and bm > max(seqlen_q, 32):
            continue
        if seqlen_k is not None and bn > max(seqlen_k, 32):
            continue
        if headdim is not None:
            est = _estimate_smem(bm, bn, headdim, elem_size, c.num_stages)
            if est > max_smem:
                continue
        out.append(c)
    if not out:
        # Always return at least the smallest config so autotune has something
        # to try. The wrapper will error on launch if even this OORs.
        smallest = min(configs, key=lambda c: c.kwargs["BLOCK_M"] * c.kwargs["BLOCK_N"])
        return [smallest]
    return out


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "IS_CAUSAL",
         "BLOCK_HEADDIM", "DOT_PRECISION"],
    prune_configs_by={"early_config_prune": _prune_invalid_configs},
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        # AMD alignment hints: vector load width depends on stride divisibility.
        # These are compile-time constexpr so unmet conditions cost nothing.
        "HEADDIM_DIV_16": lambda args: args["headdim"] % 16 == 0,
        "HEADDIM_DIV_8":  lambda args: args["headdim"] % 8 == 0,
        "SEQLEN_Q_DIV_16": lambda args: args["seqlen_q"] % 16 == 0,
        "SEQLEN_Q_DIV_8":  lambda args: args["seqlen_q"] % 8 == 0,
        "SEQLEN_K_DIV_16": lambda args: args["seqlen_k"] % 16 == 0,
        "SEQLEN_K_DIV_8":  lambda args: args["seqlen_k"] % 8 == 0,
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
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    HEADDIM_DIV_16: tl.constexpr,
    HEADDIM_DIV_8: tl.constexpr,
    SEQLEN_Q_DIV_16: tl.constexpr,
    SEQLEN_Q_DIV_8: tl.constexpr,
    SEQLEN_K_DIV_16: tl.constexpr,
    SEQLEN_K_DIV_8: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    # --- Integer-range and alignment hints ---
    # tl.assume narrows the value range the compiler reasons about, which
    # tightens address-calc codegen on AMD's integer pipe. tl.multiple_of
    # promises strides are multiples of N elements, enabling
    # global_load_dwordx{2,4} (8/16-byte vector loads). All conditions here
    # are constexpr — unmet ones cost nothing.
    tl.assume(stride_qb > 0)
    tl.assume(stride_qh > 0)
    tl.assume(stride_qm > 0)
    tl.assume(stride_kb > 0)
    tl.assume(stride_kh > 0)
    tl.assume(stride_kn > 0)
    tl.assume(stride_vb > 0)
    tl.assume(stride_vh > 0)
    tl.assume(stride_vn > 0)
    tl.assume(stride_dob > 0)
    tl.assume(stride_doh > 0)
    tl.assume(stride_dom > 0)
    tl.assume(stride_dqb > 0)
    tl.assume(stride_dqh > 0)
    tl.assume(stride_dqm > 0)
    tl.assume(stride_dkb > 0)
    tl.assume(stride_dkh > 0)
    tl.assume(stride_dkn > 0)
    tl.assume(stride_dvb > 0)
    tl.assume(stride_dvh > 0)
    tl.assume(stride_dvn > 0)
    tl.assume(seqlen_q > 0)
    tl.assume(seqlen_k > 0)
    tl.assume(nheads > 0)

    # In (B, S, H, D) layout, the head stride equals D. When D % 16 == 0,
    # the seq stride (H*D) and batch stride (S*H*D) inherit that alignment.
    if HEADDIM_DIV_16:
        stride_qh = tl.multiple_of(stride_qh, 16)
        stride_kh = tl.multiple_of(stride_kh, 16)
        stride_vh = tl.multiple_of(stride_vh, 16)
        stride_doh = tl.multiple_of(stride_doh, 16)
        stride_dqh = tl.multiple_of(stride_dqh, 16)
        stride_dkh = tl.multiple_of(stride_dkh, 16)
        stride_dvh = tl.multiple_of(stride_dvh, 16)
        stride_qm = tl.multiple_of(stride_qm, 16)
        stride_kn = tl.multiple_of(stride_kn, 16)
        stride_vn = tl.multiple_of(stride_vn, 16)
        stride_dom = tl.multiple_of(stride_dom, 16)
        stride_dqm = tl.multiple_of(stride_dqm, 16)
        stride_dkn = tl.multiple_of(stride_dkn, 16)
        stride_dvn = tl.multiple_of(stride_dvn, 16)
    elif HEADDIM_DIV_8:
        stride_qh = tl.multiple_of(stride_qh, 8)
        stride_kh = tl.multiple_of(stride_kh, 8)
        stride_vh = tl.multiple_of(stride_vh, 8)
        stride_doh = tl.multiple_of(stride_doh, 8)
        stride_dqh = tl.multiple_of(stride_dqh, 8)
        stride_dkh = tl.multiple_of(stride_dkh, 8)
        stride_dvh = tl.multiple_of(stride_dvh, 8)
        stride_qm = tl.multiple_of(stride_qm, 8)
        stride_kn = tl.multiple_of(stride_kn, 8)
        stride_vn = tl.multiple_of(stride_vn, 8)
        stride_dom = tl.multiple_of(stride_dom, 8)
        stride_dqm = tl.multiple_of(stride_dqm, 8)
        stride_dkn = tl.multiple_of(stride_dkn, 8)
        stride_dvn = tl.multiple_of(stride_dvn, 8)

    # Hint divisibility of the sequence axis itself (helps loop codegen and
    # boundary masking on the last block).
    if SEQLEN_Q_DIV_16:
        seqlen_q = tl.multiple_of(seqlen_q, 16)
    elif SEQLEN_Q_DIV_8:
        seqlen_q = tl.multiple_of(seqlen_q, 8)
    if SEQLEN_K_DIV_16:
        seqlen_k = tl.multiple_of(seqlen_k, 16)
    elif SEQLEN_K_DIV_8:
        seqlen_k = tl.multiple_of(seqlen_k, 8)

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
    (SEQUENCE_PARALLEL is hard-coded to False — no atomic-add on dq).

    Block sizes, num_warps, and num_stages are chosen by ``@triton.autotune``
    keyed on the dtype and (bucketed) sequence lengths. The first call for a
    given shape/dtype combo therefore pays an autotune cost (~seconds); after
    that the choice is cached for the process lifetime. Set
    ``TRITON_PRINT_AUTOTUNING=1`` in the environment to log which config was
    picked — useful when tuning the config list for new hardware.

    ``enable_hip_opts`` is an experimental knob that, when set on a HIP/ROCm
    backend, adds ``waves_per_eu`` and ``allow_flush_denorm`` to the kernel
    launch (the AMD-tuning idiom from experiments/evoformer.py). Early MI300A
    testing showed this REGRESSED throughput vs the backend-agnostic launch,
    so it is OFF by default. Benchmark before enabling on your hardware.
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
    # via load/store with eviction_policy='evict_last'. The autotune pre_hook
    # zeros it before every kernel launch (and between autotune trials), so
    # we can allocate uninitialized here.
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    delta = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    # BLOCK_M/BLOCK_N/num_warps/num_stages are chosen by @triton.autotune.
    # DOT_PRECISION is a constexpr we set from the input dtype: fp32 needs
    # 'ieee' to bypass Triton's default TF32 fast-path; for fp16/bf16 the
    # field is a no-op (tf32 is conventional here).
    dot_precision = "ieee" if q.dtype == torch.float32 else "tf32"
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

    # SEQUENCE_PARALLEL=False -> grid is (1, batch*nheads); the kernel walks
    # the seqlen_k dimension internally. seqlen //32 buckets the autotune key.
    grid = (1, batch * nheads)
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
        seqlen_q // 32,             # CACHE_KEY_SEQLEN_Q
        seqlen_k // 32,             # CACHE_KEY_SEQLEN_K
        "none",                     # BIAS_TYPE
        causal,
        BLOCK_HEADDIM,
        SEQUENCE_PARALLEL,
        DOT_PRECISION=dot_precision,
        **extra_kern_args,
    )
    # Land the fp32 dq accumulator into the caller's buffer (whatever dtype).
    dq.copy_(dq_accum)
