import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .ring_flash_attn import ring_flash_attn_backward
from einops import rearrange
from typing import Optional, Tuple
from .utils import get_default_args, AllGatherComm as Comm
import logging
import torch.distributed._tensor as distp_tensor
import flash_attn
import os

if torch.__version__ >= "2.4.0" and flash_attn.__version__ >= "2.7.0":
    _wrapped_flash_attn_forward = torch.ops.flash_attn._flash_attn_forward
else:
    _wrapped_flash_attn_forward = _flash_attn_forward

if torch.__version__ >= "2.4.0":
    _wrapped_flash_attn_backward = torch.ops.flash_attn._flash_attn_backward
else:
    _wrapped_flash_attn_backward = _flash_attn_backward

def llama_flash_attn_forward(
    process_group: dist.ProcessGroup,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads_k_stride: int,
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    head_first_stride: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Llama-style flash attention forward pass with ring communication.

    This function performs the forward pass of attention, distributing the key and value tensors
    across a process group using all-gather operations. It supports grouped-query attention (GQA)
    and multi-query attention (MQA) by processing heads in strides.

    Args:
        process_group (dist.ProcessGroup): The distributed process group for communication.
        q (torch.Tensor): Query tensor of shape `(batch_size, seq_len, num_heads, head_dim)`.
        k (torch.Tensor): Key tensor of shape `(batch_size, seq_len, num_kv_heads, head_dim)`.
        v (torch.Tensor): Value tensor of shape `(batch_size, seq_len, num_kv_heads, head_dim)`.
        heads_k_stride (int): The number of key/value heads to process in each communication step.
        softmax_scale (float): The scale factor for softmax.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        causal (bool, optional): Whether to apply causal masking. Defaults to True.
        window_size (Tuple[int, int], optional): Sliding window size for attention. Defaults to (-1, -1).
        softcap (float, optional): Softcap for attention scores. Defaults to 0.0.
        alibi_slopes (Optional[torch.Tensor], optional): ALiBi slopes for positional bias. Defaults to None.
        deterministic (bool, optional): Whether to use deterministic algorithms. Defaults to False.
        head_first_stride (Optional[int], optional): A different (smaller) stride for the first group of heads.
            This is an optimization to increase communication/computation overlap. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - out (torch.Tensor): The attention output tensor.
            - lse (torch.Tensor): The log-sum-exp of the attention scores for the backward pass.
    """
    out_list = []
    lse_list = []
    # logging.debug(f"bwd q {q[0,:2,0,:3]}")     

    nheads = q.shape[2]
    # total_k, nheads_k, head_dim = k.shape
    batch_k, seq_k, nheads_k, head_dim = k.shape
    assert nheads_k % heads_k_stride == 0

    world_size = dist.get_world_size(process_group)

    # Main buffers for standard stride operations
    kv_buffer = torch.empty(
        (2, world_size, batch_k, seq_k, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )
    kv_buffer_copy = torch.empty_like(kv_buffer)

    if causal:
        local_rank = dist.get_rank(process_group)

    if head_first_stride is not None:
        assert 0 < head_first_stride < heads_k_stride, (
            "head_first_stride must be between 0 and heads_k_stride"
        )
        stride_pattern = [
            head_first_stride,
            heads_k_stride - head_first_stride,
        ] + [heads_k_stride] * ((nheads_k - heads_k_stride) // heads_k_stride)
        k_0 = k[:, :, :head_first_stride].contiguous()
        v_0 = v[:, :, :head_first_stride].contiguous()

        # Allocate one smaller buffer for the first step
        kv_buffer_small1 = torch.empty(
            (2, world_size, batch_k, seq_k, head_first_stride, head_dim),
            dtype=k.dtype, device=k.device
        )
        # Allocate a second buffer for the second, non-standard step
        kv_buffer_small2 = torch.empty(
            (2, world_size, batch_k, seq_k, heads_k_stride - head_first_stride, head_dim),
            dtype=k.dtype, device=k.device
        )
        initial_kv_buffer = kv_buffer_small1
    else:
        # When head_first_stride is None, it behaves as before
        stride_pattern = [heads_k_stride] * (nheads_k // heads_k_stride)
        k_0 = k[:, :, :heads_k_stride].contiguous()
        v_0 = v[:, :, :heads_k_stride].contiguous()
        initial_kv_buffer = kv_buffer_copy

    comm = Comm(process_group)
    # Pass the main tensor slices to all_gather
    comm.all_gather(initial_kv_buffer[0], k_0)
    comm.all_gather(initial_kv_buffer[1], v_0)

    current_head = 0
    for step, stride in enumerate(stride_pattern):
        i = current_head
        current_head += stride

        comm.wait()

        if head_first_stride is not None:
            if step == 0:
                # Read from the first small buffer
                current_kv_buffer = kv_buffer_small1
            elif step == 1:
                # Read from the second small buffer
                current_kv_buffer = kv_buffer_small2
            elif step == 2:
                # This is the first "full" buffer. Data was gathered into kv_buffer.
                current_kv_buffer = kv_buffer
            else:
                # From step 3 onwards, the normal double-buffer swap applies
                kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer
                current_kv_buffer = kv_buffer
        else:
            # Original behavior: swap the main storage tensors
            kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer
            current_kv_buffer = kv_buffer

        if step < len(stride_pattern) - 1:
            # all_gather the next kv slice
            kv_slice_left = current_head
            next_stride = stride_pattern[step + 1]
            kv_slice_right = current_head + next_stride
            send_k = k[:, :, kv_slice_left:kv_slice_right].contiguous()
            send_v = v[:, :, kv_slice_left:kv_slice_right].contiguous()

            if head_first_stride is not None:
                if step == 0:
                    # Step 0: Gather chunk 1 into kv_buffer_small2
                    next_buffer = kv_buffer_small2
                elif step == 1:
                    # Step 1: Gather chunk 2 into kv_buffer
                    next_buffer = kv_buffer
                else:
                    # Step 2+: Gather into kv_buffer_copy (normal swap pattern)
                    next_buffer = kv_buffer_copy
            else:
                # Original behavior
                next_buffer = kv_buffer_copy

            # Pass the main tensor slices for the next round
            comm.all_gather(next_buffer[0], send_k)
            comm.all_gather(next_buffer[1], send_v)


        q_i = q[:, :, i * nheads // nheads_k : current_head * nheads // nheads_k]
        # kv_buffer[0] has shape (batch_k, seq_k, world_size, heads_k_stride, head_dim)
        # We want k_i to be (batch_k, seq_k * world_size, heads_k_stride, head_dim)
        if causal:
            k_i = current_kv_buffer[0][:(local_rank + 1), :, :, :, :].contiguous()
            v_i = current_kv_buffer[1][:(local_rank + 1), :, :, :, :].contiguous()
            k_i = rearrange(k_i, 'w b s hs dh -> b (w s) hs dh')
            v_i = rearrange(v_i, 'w b s hs dh -> b (w s) hs dh')
        else:
            k_i = rearrange(current_kv_buffer[0], "w b s hs dh -> b (w s) hs dh")
            v_i = rearrange(current_kv_buffer[1], "w b s hs dh -> b (w s) hs dh")

        # params = get_default_args(_flash_attn_varlen_forward).copy()
        params = {
            "q": q_i,
            "k": k_i,
            "v": v_i,
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal, # 'step' was not defined in this scope
            "window_size_left": window_size[0],
            "window_size_right": window_size[1],
            "softcap": softcap,
            "alibi_slopes": alibi_slopes,
            "return_softmax": True and dropout_p > 0,
        }
        # logging.debug(f"fwd i {i} k_ishape {k_i.shape} s{k_i[0,:3,0,:2]} e{k_i[0,-3:,0,:2]} q_i.shape {q_i.shape} params {params}")     
        # process_id = os.getpid()
        # if not os.path.exists('./logging/k_buffer_{}.pt'.format(process_id)):
        #     torch.save(k_i.detach(), './logging/k_buffer_{}.pt'.format(process_id))
        # out, _, _, _, _, lse, _, _ = _flash_attn_varlen_forward(**params)
        outputs = _wrapped_flash_attn_forward(**params)
        if len(outputs) == 8:
            out, _, _, _, _, lse, _, _ = outputs
        else:
            assert len(outputs) == 4
            out, lse, _, _ = outputs
        out_list.append(out)
        lse_list.append(lse)

    # out = torch.cat(out_list, dim=1)
    out = torch.cat(out_list, dim=2)
    # lse (B H S)
    lse = torch.cat(lse_list, dim=-2)
    return out, lse


def llama_flash_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    heads_k_stride,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
):
    """
    Llama-style flash attention backward pass.

    This function performs the backward pass of attention, distributing the key and value tensors
    across a process group using all-gather operations.

    Args:
        process_group (dist.ProcessGroup): The distributed process group for communication.
        dout (torch.Tensor): Gradient of the output tensor.
        q (torch.Tensor): Query tensor from the forward pass.
        k (torch.Tensor): Key tensor from the forward pass.
        v (torch.Tensor): Value tensor from the forward pass.
        out (torch.Tensor): Output tensor from the forward pass.
        softmax_lse (torch.Tensor): Log-sum-exp of the attention scores from the forward pass.
        heads_k_stride (int): The number of key/value heads to process in each communication step.
        softmax_scale (float): The scale factor for softmax.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        causal (bool, optional): Whether to apply causal masking. Defaults to True.
        window_size (Tuple[int, int], optional): Sliding window size for attention. Defaults to (-1, -1).
        softcap (float, optional): Softcap for attention scores. Defaults to 0.0.
        alibi_slopes (Optional[torch.Tensor], optional): ALiBi slopes for positional bias. Defaults to None.
        deterministic (bool, optional): Whether to use deterministic algorithms. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the gradients with respect to
            the query, key, and value tensors (`dq`, `dk`, `dv`).
    """
    nheads = q.shape[2]
    batch_k, seq_k, nheads_k, head_dim = k.shape
    assert nheads_k % heads_k_stride == 0

    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    
    kv_buffer = torch.empty(
        (2, world_size, batch_k, seq_k, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )
    kv_buffer_copy = torch.empty_like(kv_buffer)

    # Buffer for gradients coming OUT of Flash Attention
    # Shape: [Batch, Seq * WorldSize, Heads, HeadDim]
    dkv_buffer = torch.empty(
        (2, batch_k, seq_k * world_size, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    ) 

    # Buffer for input to reduce_scatter (Needs to be rank-contiguous)
    scatter_input_buffer = torch.empty(
        (2, world_size, batch_k, seq_k, heads_k_stride, head_dim),
        dtype=torch.float32,
        device=k.device,
    )

    # Buffer for output of reduce_scatter (Float32 for stability)
    # We use this regardless of stride since dk.float() would require full copies
    dkv_reduce_output_buffer = torch.empty(
        (2, batch_k, seq_k, heads_k_stride, head_dim),
        dtype=torch.float32,
        device=k.device,
    )
    
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    comm = Comm(process_group)

    k_0 = k[:, :, :heads_k_stride].contiguous()
    v_0 = v[:, :, :heads_k_stride].contiguous()

    # Pass the main tensor slices to all_gather
    comm.all_gather(kv_buffer_copy[0], k_0)
    comm.all_gather(kv_buffer_copy[1], v_0)

    for i in range(0, nheads_k, heads_k_stride):
        # Must zero out because for Causal, we don't write to the entire sequence length
        dkv_buffer.zero_()

        q_slice = slice(
            i * nheads // nheads_k, (i + heads_k_stride) * nheads // nheads_k
        )
        q_i = q[:, :, q_slice]
        dout_i = dout[:, :, q_slice]
        out_i = out[:, :, q_slice]
        dq_i = dq[:, :, q_slice]
        if softmax_lse.dim() == 3:
            lse_i = softmax_lse[:, q_slice].contiguous()
        else:
            lse_i = softmax_lse[q_slice]

        comm.wait()
        # Swap the main storage tensors
        kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

        if i < nheads_k - heads_k_stride:
            # all_gather the next kv slice
            kv_slice_left = i + heads_k_stride
            kv_slice_right = kv_slice_left + heads_k_stride
            send_k = k[:, :, kv_slice_left:kv_slice_right].contiguous()
            send_v = v[:, :, kv_slice_left:kv_slice_right].contiguous()
            # Pass the main tensor slices for the next round
            comm.all_gather(kv_buffer_copy[0], send_k)
            comm.all_gather(kv_buffer_copy[1], send_v)

        # kv_buffer[0] has shape (batch_k, seq_k, world_size, heads_k_stride, head_dim)
        # We want k_i to be (batch_k, seq_k * world_size, heads_k_stride, head_dim)
        if causal:
            k_i = kv_buffer[0][:(rank + 1)]
            v_i = kv_buffer[1][:(rank + 1)]
            k_i = rearrange(k_i, 'w b s hs dh -> b (w s) hs dh').contiguous()
            v_i = rearrange(v_i, 'w b s hs dh -> b (w s) hs dh').contiguous()
        else:
            k_i = rearrange(kv_buffer[0], 'w b s hs dh -> b (w s) hs dh').contiguous()
            v_i = rearrange(kv_buffer[1], 'w b s hs dh -> b (w s) hs dh').contiguous()

        if causal:
            # dk must have the same shape as k
            dk_i = dkv_buffer[0][:, :k_i.shape[1]]
            dv_i = dkv_buffer[1][:, :k_i.shape[1]]
        else:
            dk_i = dkv_buffer[0]
            dv_i = dkv_buffer[1]

        # params = get_default_args(_flash_attn_varlen_backward).copy()
        params = {
                "dout": dout_i,
                "q": q_i,
                "k": k_i,
                "v": v_i,
                "out": out_i,
                "softmax_lse": lse_i,
                "dq": dq_i,
                "dk": dk_i,
                "dv": dv_i,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "window_size_left": window_size[0],
                "window_size_right": window_size[1],
                "softcap": softcap,
                "alibi_slopes": alibi_slopes,
                "deterministic": deterministic,
        }
        _wrapped_flash_attn_backward(**params)

        # We do not need k_i, v_i, or the slices anymore.
        del k_i, v_i, q_i, dout_i, out_i

        # Target for reduce_scatter is always our FP32 buffer
        dk_i = dkv_reduce_output_buffer[0]
        dv_i = dkv_reduce_output_buffer[1]

        # Rearrange dkv_buffer so the first dimension represents the World Rank.
        # dkv_buffer is [2, B, W*S, H, D]. We need to transform to [2, W, B, S, H, D]
        # 1. View: Split W*S -> W, S.  Shape: [2, B, W, S, H, D]
        grad_view = dkv_buffer.view(2, batch_k, world_size, seq_k, heads_k_stride, head_dim)
        # 2. Permute: Swap B and W. Shape: [2, W, B, S, H, D]
        grad_permuted = grad_view.permute(0, 2, 1, 3, 4, 5)
        scatter_input_buffer.copy_(grad_permuted)

        # dist.reduce_scatter_tensor concat, gather, reduce on dim=0
        dist.reduce_scatter_tensor(dk_i, scatter_input_buffer[0], group=process_group)
        dist.reduce_scatter_tensor(dv_i, scatter_input_buffer[1], group=process_group)

        # Final copy back to main bf16/fp16 tensor
        if heads_k_stride != nheads_k:
            dk[:, :, i : i + heads_k_stride].copy_(dk_i)
            dv[:, :, i : i + heads_k_stride].copy_(dv_i)
        else:
            dk.copy_(dk_i)
            dv.copy_(dv_i)

    return dq, dk, dv


class LlamaRingFlashAttnFunc(torch.autograd.Function):
    """
    Autograd function for Llama-style ring attention.

    This function implements a custom forward and backward pass for attention
    computation that uses all-gather for keys and values in the
    forward pass, but a standard ring-based backward pass. This is a hybrid
    approach, where the forward pass (`llama_flash_attn_forward`) is optimized
    for numerical stability, and the backward pass (`ring_flash_attn_backward`)
    is a more general ring attention implementation.
    """
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads_k_stride: int,
        head_first_stride: Optional[int],
        dropout_p: float,
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        return_softmax: bool,
        group: dist.ProcessGroup,
        bwd_event_sync: bool,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, None]:
        """
        Forward pass for Llama-style ring attention.
        For loop in llama_flash_attn_forward could lead to early allocation of tensors
        by the CPU, leading to memory explosion. CUDA event syncing can control this.
        Let parent model handle forward event syncing for efficiency.

        Args:
            ctx: The context object for autograd.
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            heads_k_stride (int): Stride for key/value heads in GQA/MQA.
            head_first_stride (Optional[int]): A different stride for the first group of heads.
            dropout_p (float): Dropout probability.
            softmax_scale (Optional[float]): Scale factor for softmax. If None, calculated from head dimension.
            causal (bool): Whether to apply causal masking.
            window_size (Tuple[int, int]): Sliding window size.
            alibi_slopes (Optional[torch.Tensor]): ALiBi slopes for positional bias.
            deterministic (bool): Whether to use deterministic algorithms.
            return_softmax (bool): Whether to return the softmax log-sum-exp.
            group (dist.ProcessGroup): The distributed process group.
            bwd_event_sync (bool): If True, syncs a CUDA event in the backward pass to control memory allocation.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]: The attention output, and optionally the LSE and a None placeholder.
        """
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        # out shape (batch, seq, heads, head_dim)
        # softmax_lse shape (batch, seq, heads)
        out, softmax_lse = llama_flash_attn_forward(
            group,
            q,
            k,
            v,
            heads_k_stride=heads_k_stride,
            head_first_stride=head_first_stride,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        ctx.bwd_event_sync = bwd_event_sync
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dout: torch.Tensor,
        *args,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass for Llama-style ring attention.

        Uses `ring_flash_attn_backward`.

        Args:
            ctx: The context object for autograd.
            dout (torch.Tensor): Gradient of the output.
            *args: Other gradients.

        Returns:
            Tuple[Optional[torch.Tensor], ...]: Gradients for the inputs of the forward pass.
        """
        time_event = None
        if ctx.bwd_event_sync:
            time_event = torch.cuda.Event(enable_timing=False)
            time_event.record()
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = ring_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        if ctx.bwd_event_sync:
            time_event.synchronize()
        # forward takes 14 args excluding ctx. return 3 grad + 11 None
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None

class LlamaFlashAttnFunc(torch.autograd.Function):
    """
    Autograd function for Llama-style attention. NOT FINISHED
    """
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads_k_stride: int,
        head_first_stride: Optional[int],
        dropout_p: float,
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        return_softmax: bool,
        group: dist.ProcessGroup,
        bwd_event_sync: bool,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, None]:
        """
        Forward pass for Llama-style ring attention.
        For loop in llama_flash_attn_forward could lead to early allocation of tensors
        by the CPU, leading to memory explosion. CUDA event syncing can control this.
        Let parent model handle forward event syncing for efficiency.

        Args:
            ctx: The context object for autograd.
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            heads_k_stride (int): Stride for key/value heads in GQA/MQA.
            head_first_stride (Optional[int]): A different stride for the first group of heads.
            dropout_p (float): Dropout probability.
            softmax_scale (Optional[float]): Scale factor for softmax. If None, calculated from head dimension.
            causal (bool): Whether to apply causal masking.
            window_size (Tuple[int, int]): Sliding window size.
            alibi_slopes (Optional[torch.Tensor]): ALiBi slopes for positional bias.
            deterministic (bool): Whether to use deterministic algorithms.
            return_softmax (bool): Whether to return the softmax log-sum-exp.
            group (dist.ProcessGroup): The distributed process group.
            bwd_event_sync (bool): If True, syncs a CUDA event in the backward pass to control memory allocation.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]: The attention output, and optionally the LSE and a None placeholder.
        """
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        # out shape (batch, seq, heads, head_dim)
        # softmax_lse shape (batch, seq, heads)
        out, softmax_lse = llama_flash_attn_forward(
            group,
            q,
            k,
            v,
            heads_k_stride=heads_k_stride,
            head_first_stride=head_first_stride,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        ctx.bwd_event_sync = bwd_event_sync
        ctx.heads_k_stride = heads_k_stride
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        time_event = None
        if ctx.bwd_event_sync:
            time_event = torch.cuda.Event(enable_timing=False)
            time_event.record()
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = llama_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            heads_k_stride=ctx.heads_k_stride,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        if ctx.bwd_event_sync:
            time_event.synchronize()
        # forward takes 14 args excluding ctx. return 3 grad + 11 None
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None


def llama_fwd_ring_bwd_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads_k_stride: int = 1,
    head_first_stride: Optional[int] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: Optional[dist.ProcessGroup] = None,
    bwd_event_sync: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, None]:
    """
    Performs Llama-style ring flash attention with a custom backward pass.

    This function is a wrapper around `LlamaRingFlashAttnFunc`. It uses an
    all-gather-based forward pass (`llama_flash_attn_forward`) which is
    numerically stable, and a standard ring-based
    backward pass (`ring_flash_attn_backward`).

    Args:
        q (torch.Tensor): Query tensor of shape `(batch, seq_len, n_heads, head_dim)`.
        k (torch.Tensor): Key tensor of shape `(batch, seq_len, n_kv_heads, head_dim)`.
        v (torch.Tensor): Value tensor of shape `(batch, seq_len, n_kv_heads, head_dim)`.
        heads_k_stride (int, optional): The number of key/value heads to process in each
            communication step. Defaults to 1.
        head_first_stride (Optional[int], optional): A different stride for the first group of heads
            to improve communication/computation overlap. Defaults to None. 
            Must be smaller than heads_k_stride.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        softmax_scale (Optional[float], optional): The scale factor for softmax. If None, it is
            calculated as `1.0 / sqrt(head_dim)`. Defaults to None.
        causal (bool, optional): Whether to apply causal masking. Defaults to False.
        window_size (Tuple[int, int], optional): Sliding window size for attention. Defaults to (-1, -1).
        alibi_slopes (Optional[torch.Tensor], optional): ALiBi slopes for positional bias. Defaults to None.
        deterministic (bool, optional): Whether to use deterministic algorithms. Defaults to False.
        return_attn_probs (bool, optional): Whether to return the log-sum-exp of the attention scores.
            If True, the output is a tuple `(out, lse, None)`. Defaults to False.
        group (Optional[dist.ProcessGroup], optional): The distributed process group. If None, the default
            process group is used. Defaults to None.
        bwd_event_sync (bool, optional): If True, synchronizes a CUDA event in the backward pass to
            control GPU memory allocation and prevent spikes. Defaults to False.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
            - The attention output tensor if `return_attn_probs` is False.
            - A tuple `(output, lse, None)` if `return_attn_probs` is True.
    """
    return LlamaRingFlashAttnFunc.apply(
        q,
        k,
        v,
        heads_k_stride,
        head_first_stride,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        bwd_event_sync,
    )

def llama_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads_k_stride: int = 1,
    head_first_stride: Optional[int] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: Optional[dist.ProcessGroup] = None,
    bwd_event_sync: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, None]:
    # logging.debug(f"q {q[0,:2,3,:4]}")
    return LlamaFlashAttnFunc.apply(
        q,
        k,
        v,
        heads_k_stride,
        head_first_stride,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        bwd_event_sync,
    )

from .llama3_flash_attn_varlen import llama3_flash_attn_varlen_backward, llama3_flash_attn_varlen_forward, llama3_flash_attn_prepare_cu_seqlens, Llama3FlashAttnVarlenFunc


def llama3_flash_attn_varlen_custom_func(
    q,
    k,
    v,
    heads_k_stride,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    mesh=None,
):
    '''
    The input is sharded in total_length dim
    for world size 4:
        rank 0 takes 1/2 of seq 1
        rank 1 takes 2/2 of seq 1
        rank 2 takes 1/2 of seq 2
        rank 3 takes 2/2 of seq 2
    That way each rank Q attends to corresonding k when one acutually use varlen
    inefficient all gather just a temp fix for testing
    '''
    batch_k, seq_k, nheads_k, head_dim = k.shape
    # logging.debug(f"q {q[0,:2,3,:4]}")
    q_k_v = []
    for t in (q, k, v):
        q_k_v.append(
            distp_tensor.DTensor.from_local(
                t, mesh, [distp_tensor.Shard(1)]
            ).redistribute(mesh, [distp_tensor.Replicate()]).view(-1, nheads_k, head_dim
            ).redistribute(mesh, [distp_tensor.Shard(0)]).to_local() 
        )
    q, k, v = q_k_v
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    cu_seqlens = torch.arange(0, (batch_k + 1) * seq_k*world_size, step=seq_k*world_size,
                              dtype=torch.int32, device=k.device)
    (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, local_k_slice
    ) = llama3_flash_attn_prepare_cu_seqlens(cu_seqlens, causal, rank, world_size)
    # logging.debug(f"{cu_seqlens_q}, {cu_seqlens_k}, {max_seqlen_q}, {max_seqlen_k}, {local_k_slice}")
    # logging.debug(f"{cu_seqlens}, {causal}, {rank}, {world_size}")
    # q, k, v = [rearrange(t, 'b s h d -> (b s) h d', b=) for t in (q, k, v)]
    # k = k.contiguous().view(-1,  nheads_k, head_dim)
    # v = v.contiguous().view(-1,  nheads_k, head_dim)
    # q = q.contiguous().view(-1,  nheads_k, head_dim)
    output = Llama3FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        heads_k_stride,
        local_k_slice,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )
    if return_attn_probs:
        (out, softmax_lse, none) = output
        out = distp_tensor.DTensor.from_local(
                out, mesh, [distp_tensor.Shard(0)]
            ).redistribute(mesh, [distp_tensor.Replicate()]).view(batch_k, seq_k*world_size, nheads_k, head_dim
            ).redistribute(mesh, [distp_tensor.Shard(1)]).to_local() 
        return out, softmax_lse, none
    else:
        output = distp_tensor.DTensor.from_local(
                output, mesh, [distp_tensor.Shard(0)]
            ).redistribute(mesh, [distp_tensor.Replicate()]).view(batch_k, seq_k*world_size, nheads_k, head_dim
            ).redistribute(mesh, [distp_tensor.Shard(1)]).to_local() 
        logging.debug(f" out {output[0,:2,3,:4]}")     
        return output
