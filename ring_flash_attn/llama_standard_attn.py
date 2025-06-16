import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, List
from .utils import AllGatherComm as Comm
from einops import rearrange


class LlamaStandardAttn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        heads_k_stride=1,  # default 1 always works, but need optimize
        dropout_p=0.0,
        softmax_scale=None,
        key_padding_mask=None,
        attn_q_chunk_size=128,  # default 128 q tokens per chunk
        causal=False,  # TODO: To implement
        return_attn_probs=False,
        process_group=None,
        bwd_event_sync=False,
    ):
        time_event = torch.cuda.Event(enable_timing=False)
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        k = k.contiguous()
        v = v.contiguous()
        out, probs = llama_standard_attn_forward(
            q,
            k,
            v,
            process_group=process_group,
            key_padding_mask=key_padding_mask,
            heads_k_stride=heads_k_stride,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            attn_q_chunk_size=attn_q_chunk_size,
            time_event=time_event,
        )
        time_event.synchronize()
        # logging.debug(f"out {out[0,:2,3,:4]} out {softmax_lse[0,:2,:5]}")     
        # this should be out_padded
        ctx.save_for_backward(q, k, v, probs)  # don't need out
        ctx.key_padding_mask = key_padding_mask

        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.process_group = process_group
        ctx.bwd_event_sync = bwd_event_sync
        ctx.heads_k_stride = heads_k_stride
        ctx.attn_q_chunk_size = attn_q_chunk_size
        ctx.causal = causal
        return out if not return_attn_probs else (out, probs)

    @staticmethod
    def backward(ctx, dout, *args):
        time_event = None
        if ctx.bwd_event_sync:
            time_event = torch.cuda.Event(enable_timing=False)
        q, k, v, probs = ctx.saved_tensors  # don't need out
        dq, dk, dv = llama_standard_attn_backward(
            ctx.process_group,
            dout,
            q,
            k,
            v,
            out,
            probs=probs,
            heads_k_stride=ctx.heads_k_stride,
            attn_q_chunk_size=ctx.attn_q_chunk_size,
            key_padding_mask=ctx.key_padding_mask,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            time_event=time_event,
        )
        if ctx.bwd_event_sync:
            time_event.synchronize()
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def llama_standard_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads_k_stride: int,
    process_group: Optional[dist.ProcessGroup],
    attn_q_chunk_size: Optional[int],
    dropout_p: float = 0.0,
    key_padding_mask: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,  # TODO: To implement
    time_event=None,  # Sync GPU,CPU to lower vRAM allocation; no sync by default
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Forward pass for the Llama standard attention module.
    Intput entsor shape matches flash-attention's
    [batch_size, local_seq_len_q, num_q_heads, head_dim]
    i.e., [B S H D]
    Internally, the function uses [B H S D]

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.
        dropout_p (float): Dropout probability.
        key_padding_mask (Optional[torch.Tensor]): Boolean mask for padding in keys.
            Shape (batch_size, local_seq_len_kv). True for valid tokens, False for padding.
        softmax_scale (Optional[float]): Scale for softmax. Defaults to 1/sqrt(head_dim).
        causal (bool): Whether to apply causal masking. NOT IMPLEMENTED YET.

    Returns:
        torch.Tensor: Output of the attention mechanism, shape (batch_size, local_seq_len_q, num_q_heads, head_dim).
        Optional[torch.Tensor]: Attention probabilities if return_attn_probs is True, else None.
                                    Shape (batch_size, num_kv_heads, num_q_groups, local_seq_len_q, global_seq_len_kv).
    """
    batch_size, local_seq_len_q, num_q_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape
    assert num_q_heads == num_kv_heads
    assert num_q_heads % heads_k_stride == 0
    assert causal is False, "Causal masking is not implemented yet."

    gathered_key_padding_mask = key_padding_mask
    
    world_size = 1
    q, k, v = rearrange([q, k, v], 'qkv b s h d -> qkv b h s d')

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)
    # softmax_scale applied on attn_scores instead of q for ease of testing

    output_list: List[torch.Tensor] = []
    probs_list: List[torch.Tensor] = []
    if process_group is not None:
        comm = Comm(process_group)
        world_size = dist.get_world_size(process_group)

        if key_padding_mask is not None:
            mask_list = [torch.empty_like(key_padding_mask) for _ in range(world_size)]
            dist.all_gather(mask_list, key_padding_mask.contiguous(), group=process_group)
            gathered_key_padding_mask = torch.cat(mask_list, dim=1)

        kv_buffer = torch.empty(
            (2, world_size, batch_size, heads_k_stride, local_seq_len_q, head_dim),
            dtype=k.dtype,
            device=k.device,
        )
        kv_buffer_copy = torch.empty_like(kv_buffer)

        k_0 = k[:, :heads_k_stride, :].contiguous()
        v_0 = v[:, :heads_k_stride, :].contiguous()

        comm = Comm(process_group)
        # Pass the main tensor slices to all_gather
        comm.all_gather(kv_buffer_copy[0], k_0)
        comm.all_gather(kv_buffer_copy[1], v_0)

        for i in range(0, num_kv_heads, heads_k_stride):

            comm.wait()
            # Swap the main storage tensors
            kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

            if i < num_kv_heads - heads_k_stride:
                # all_gather the next kv slice
                kv_slice_left = i + heads_k_stride
                kv_slice_right = kv_slice_left + heads_k_stride
                send_k = k[:, kv_slice_left:kv_slice_right].contiguous()
                send_v = v[:, kv_slice_left:kv_slice_right].contiguous()
                # Pass the main tensor slices for the next round
                comm.all_gather(kv_buffer_copy[0], send_k)
                comm.all_gather(kv_buffer_copy[1], send_v)

            q_i = q[:, i : (i + heads_k_stride)]
            # kv_buffer[0] has shape [world_size b h s d ]
            # We want k_i to be [b h world_size*s d ]
            k_i = rearrange(kv_buffer[0], 'w b h s d -> b h (w s) d')
            v_i = rearrange(kv_buffer[1], 'w b h s d -> b h (w s) d')

            output, probs = chunked_query_self_attn(
                q_i,
                k_i,
                v_i,
                softmax_scale=softmax_scale,
                attn_q_chunk_size=attn_q_chunk_size,
                dropout_p=dropout_p,
                key_padding_mask=gathered_key_padding_mask,
            )
            output_list.append(output)
            probs_list.append(probs)
    else: # Single process, no all_gather needed
        # Loop over head strides for consistency, though not strictly necessary for single GPU
        # if heads_k_stride == num_kv_heads
        for i in range(0, num_kv_heads, heads_k_stride):
            q_i = q[:, i : (i + heads_k_stride)]
            k_i = k[:, i : (i + heads_k_stride)]
            v_i = v[:, i : (i + heads_k_stride)]
            output, probs = chunked_query_self_attn(
                q_i,
                k_i,
                v_i,
                softmax_scale=softmax_scale,
                attn_q_chunk_size=attn_q_chunk_size,
                dropout_p=dropout_p,
                key_padding_mask=key_padding_mask
            )
            output_list.append(output)
            probs_list.append(probs)
    # output concat heads [B H S D]; S is the local sequence length
    output = rearrange(output_list,'hstride b h s d -> b s (hstride h) d')
    # probs concat heads [B H Sq Sk]; Sq is local sequence length of q
    probs = rearrange(probs_list,'hstride b h sq sk -> b (hstride h) sq sk')
    return output, probs


def chunked_query_self_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    attn_q_chunk_size: Optional[int] = None,
    dropout_p: float = 0.0,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Self-attention chunked.

    Intput tensor shape [B H S D]
    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.

    Returns:
        torch.Tensor: Output of the self-attention mechanism.
        Optional[torch.Tensor]: Attention probabilities if required.
    """
    batch_size, num_q_heads, local_seq_len_q, head_dim = q.shape
    _, _, global_seq_len_kv, _ = k.shape
    if attn_q_chunk_size is None:
        attn_q_chunk_size = local_seq_len_q

    all_attn_output = torch.empty(
        (batch_size, num_q_heads, local_seq_len_q, head_dim),
        dtype=q.dtype,
        device=q.device,
    )
    all_attn_probs = torch.empty(
        (batch_size, num_q_heads, local_seq_len_q, global_seq_len_kv),
        dtype=q.dtype,
        device=q.device,
    )

    for q_start in range(0, local_seq_len_q, attn_q_chunk_size):
        q_chunk_end = min(q_start + attn_q_chunk_size, local_seq_len_q)
        attn_score = q[:, :, q_start:q_chunk_end] @ k.transpose(-2, -1)
        attn_score = attn_score.float() * softmax_scale
        if key_padding_mask is not None:
            mask_to_apply = key_padding_mask.view(batch_size, 1, 1, global_seq_len_kv)
            attn_score = attn_score.masked_fill(
                ~mask_to_apply, float('-inf'))
        attn_probs = torch.softmax(attn_score, dim=-1).type_as(q)
        if dropout_p > 0.0:
            # TODO: Check if this is correct dropout application
            attn_probs = nn.functional.dropout(
                attn_probs, p=dropout_p, training=dropout_p > 0.0
            )                
        all_attn_probs[:, :, q_start:q_chunk_end] = attn_probs
        all_attn_output[:, :, q_start:q_chunk_end] = attn_probs @ v

    return all_attn_output, all_attn_probs


def llama_standard_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    # out,
    probs,
    heads_k_stride,
    softmax_scale,
    attn_q_chunk_size,
    key_padding_mask,
    dropout_p=0.0,
    causal=False,
    time_event: Optional[torch.cuda.Event] = None,
):
    nheads = q.shape[1]
    batch_k, seq_k, nheads_k, head_dim = k.shape
    assert nheads_k % heads_k_stride == 0

    q, k, v = rearrange([q, k, v], 'qkv b s h d -> qkv b h s d')
    dout = rearrange(dout, 'b s h d -> b h s d')

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    gathered_key_padding_mask = key_padding_mask
    if process_group is not None:
        comm = Comm(process_group)
        world_size = dist.get_world_size(process_group)

        if key_padding_mask is not None:
            mask_list = [torch.empty_like(key_padding_mask) for _ in range(world_size)]
            dist.all_gather(mask_list, key_padding_mask.contiguous(), group=process_group)
            gathered_key_padding_mask = torch.cat(mask_list, dim=1)

        world_size = dist.get_world_size(process_group)
        kv_buffer = torch.empty(
            (2, world_size, batch_k, heads_k_stride, head_dim),
            dtype=k.dtype,
            device=k.device,
        )
        kv_buffer_copy = torch.empty_like(kv_buffer)

        dkv_buffer = torch.empty(
            (2, world_size, batch_k, heads_k_stride, head_dim),
            dtype=k.dtype,
            device=k.device,
        )

        if heads_k_stride != nheads_k:
            kv_contiguous_buffer = torch.empty(
                (2, world_size, batch_k, heads_k_stride, head_dim),
                dtype=k.dtype,
                device=k.device,
            )

        comm = Comm(process_group)

        k_0 = k[:, :heads_k_stride].contiguous()
        v_0 = v[:, :heads_k_stride].contiguous()
        comm.all_gather(kv_buffer_copy[0], k_0)
        comm.all_gather(kv_buffer_copy[1], v_0)

        for i in range(0, nheads_k, heads_k_stride):
            dkv_buffer.zero_()
            if time_event is not None and i == nheads_k - heads_k_stride:
                time_event.record()
            q_slice = slice(
                i * nheads // nheads_k, (i + heads_k_stride) * nheads // nheads_k
            )
            q_i = q[:, q_slice]
            dout_i = dout[:, q_slice]
            # out_i = out[:, q_slice]
            dq_i = dq[:, q_slice]
            comm.wait()
            kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

            if i < nheads_k - heads_k_stride:
                # all_gather the next kv slice
                kv_slice_left = i + heads_k_stride
                kv_slice_right = kv_slice_left + heads_k_stride
                send_k = k[:, kv_slice_left:kv_slice_right].contiguous()
                send_v = v[:, kv_slice_left:kv_slice_right].contiguous()
                comm.all_gather(kv_buffer_copy[0], send_k)
                comm.all_gather(kv_buffer_copy[1], send_v)

            # kv_buffer[0] has shape (batch_k, seq_k, world_size, heads_k_stride, head_dim)
            # We want k_i to be (batch_k, seq_k * world_size, heads_k_stride, head_dim)
            k_i = rearrange(kv_buffer[0], 'w b s hs dh -> b (w s) hs dh')
            v_i = rearrange(kv_buffer[1], 'w b s hs dh -> b (w s) hs dh')
            dk_i = rearrange(dkv_buffer[0], 'w b s hs dh -> b (w s) hs dh')
            dv_i = rearrange(dkv_buffer[1], 'w b s hs dh -> b (w s) hs dh')

            chunked_query_self_attn_backward(
                dout=dout_i,
                q=q_i,
                k=k_i,
                v=v_i,
                # out=out_i,
                probs=probs[:,kv_slice_left:kv_slice_right],
                dq=dq_i,
                dk=dk_i,
                dv=dv_i,
                attn_q_chunk_size=attn_q_chunk_size,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                key_padding_mask=gathered_key_padding_mask,
                causal=causal,  # TODO: To implement
            )
            if heads_k_stride != nheads_k:
                # reduce_scatter needs contiguous buffer
                dk_i = kv_contiguous_buffer[0]
                dv_i = kv_contiguous_buffer[1]
            else:
                dk_i = dk
                dv_i = dv

            dist.reduce_scatter_tensor(dk_i, dkv_buffer[0], group=process_group)
            dist.reduce_scatter_tensor(dv_i, dkv_buffer[1], group=process_group)

            if heads_k_stride != nheads_k:
                dk[:, i : i + heads_k_stride] = dk_i
                dv[:, i : i + heads_k_stride] = dv_i
    else: # Single process, no communication needed
        for i in range(0, nheads_k, heads_k_stride):
            q_head_slice = slice(i, i + heads_k_stride)

            q_i = q[:, q_head_slice, :, :]
            k_i = k[:, i : i + heads_k_stride, :, :] # k is already local
            v_i = v[:, i : i + heads_k_stride, :, :] # v is already local
            dout_i = dout[:, q_head_slice, :, :]
            probs_i = probs[:, q_head_slice, :, :]
            
            dq_view = dq[:, q_head_slice, :, :]
            dk_view = dk[:, i : i + heads_k_stride, :, :]
            dv_view = dv[:, i : i + heads_k_stride, :, :]

            chunked_query_self_attn_backward(
                dout=dout_i, q=q_i, k=k_i, v=v_i, probs=probs_i,
                dq=dq_view, dk=dk_view, dv=dv_view,
                attn_q_chunk_size=attn_q_chunk_size, dropout_p=dropout_p,
                softmax_scale=softmax_scale, key_padding_mask=key_padding_mask, causal=causal,
            )
    return dq, dk, dv


def chunked_query_self_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # out: torch.Tensor,
    dv: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    attn_q_chunk_size: Optional[int] = None,
    dropout_p: float = 0.0,
    softmax_scale: float = 1.0,
    key_padding_mask: Optional[torch.Tensor] = None,
    causal: bool = False,  # TODO: To implement
):
    """
    Backward pass for chunked self-attention.
    Assumes dq, dk, dv are pre-allocated tensors. dk and dv should be zeroed
    by the caller if this is the first accumulation step for them. dq is a slice
    that will be directly written to.
    Shapes:
        q:    (batch_size, num_heads, local_seq_len_q_total, head_dim)
        k:    (batch_size, num_heads, global_seq_len_kv_total, head_dim)
        v:    (batch_size, num_heads, global_seq_len_kv_total, head_dim)
        probs:(batch_size, num_heads, local_seq_len_q_total, global_seq_len_kv_total)
        dq:   (batch_size, num_heads, local_seq_len_q_total, head_dim) (output slice)
        dk:   (batch_size, num_heads, global_seq_len_kv_total, head_dim) (accumulated output)
        dv:   (batch_size, num_heads, global_seq_len_kv_total, head_dim) (accumulated output)
    """
    batch_size, num_heads, local_seq_len_q_total, head_dim = q.shape

    if attn_q_chunk_size is None or attn_q_chunk_size <= 0:
        attn_q_chunk_size = local_seq_len_q_total

    # Effects of key_padding_mask and causal are embedded in `probs`
    # `out` tensor is not used as `probs` are available.

    for q_start in range(0, local_seq_len_q_total, attn_q_chunk_size):
        q_chunk_end = min(q_start + attn_q_chunk_size, local_seq_len_q_total)

        # Slice inputs for the current q_chunk
        # q_c, dout_c are (B, H, S_q_chunk, D)
        q_c = q[:, :, q_start:q_chunk_end, :]
        dout_c = dout[:, :, q_start:q_chunk_end, :]
        # probs_c are P(q_chunk, k_full), shape (B, H, S_q_chunk, S_k_total)
        probs_c = probs[:, :, q_start:q_chunk_end, :]

        # Grad w.r.t. V: dv_contrib = P.T @ dO
        # dv_update shape: (B, H, S_k_total, D)
        dv_update = probs_c.transpose(-2, -1) @ dout_c
        dv += dv_update # Accumulate into the provided dv tensor

        # Grad w.r.t. P_dropped (attention probabilities after dropout)
        # dP_dropped = dO @ V.T
        # dp_c_after_dropout shape: (B, H, S_q_chunk, S_k_total)
        dp_c_after_dropout = dout_c @ v.transpose(-2, -1)

        dp_c_before_dropout = dp_c_after_dropout
        if dropout_p > 0.0:  # TODO: dropout is untested
            # Regenerate dropout mask. Uses global PyTorch RNG.
            # For bitwise reproducibility with FlashAttention, specific RNG (Philox) and seed management would be needed.
            # This approach is similar to PyTorch's internal _scaled_dot_product_attention backward.
            dropout_mask = torch.empty_like(probs_c, dtype=torch.bool).bernoulli_(1.0 - dropout_p)
            dropout_mask_float = dropout_mask.type_as(dp_c_after_dropout)
            # dL/dP = (dL/dP_dropped) * M / (1-p)
            dp_c_before_dropout = dp_c_after_dropout.mul(dropout_mask_float).mul_(1.0 / (1.0 - dropout_p))

        # Grad w.r.t. S (attention scores): dS = P * (dP_before_dropout - sum(P * dP_before_dropout))
        # ds_c shape: (B, H, S_q_chunk, S_k_total)
        ds_c = probs_c * (dp_c_before_dropout - (probs_c * dp_c_before_dropout
            ).float().sum(dim=-1, keepdim=True)).type_as(q)

        # Grad w.r.t. Q: dQ_scaled = dS @ K => dQ = (dS @ K) * softmax_scale
        # dq_update_for_chunk shape: (B, H, S_q_chunk, D)
        dq_update_for_chunk = (ds_c @ k) * softmax_scale
        # Assign to the corresponding slice of the pre-allocated dq tensor
        dq[:, :, q_start:q_chunk_end, :] = dq_update_for_chunk

        # Grad w.r.t. K: dK = dS.T @ (Q_scaled) = dS.T @ (Q * softmax_scale)
        # dk_update shape: (B, H, S_k_total, D)
        # If q_c is Q_orig_chunk * softmax_scale (i.e., pre-scaled query chunk),
        # then dK_orig = ((Q_orig_chunk * softmax_scale).T @ dS_c) * softmax_scale
        dk_increment = (ds_c.transpose(-2, -1) @ q_c) * softmax_scale
        dk += dk_increment # Accumulate into the provided dk tensor

def llama_standard_attn_func(
    q,
    k,
    v,
    heads_k_stride=1,  # default 1 always works, but need optimize
    dropout_p=0.0,
    softmax_scale=None,
    key_padding_mask=None,
    attn_q_chunk_size=None,
    causal=False,
    return_attn_probs=False,
    group=None,
    bwd_event_sync=False,
):
    '''Llama standard attention function.
    Differs from flash_attn interface by missing:
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,

    Args:
        q (torch.Tensor): Query tensor shape [B S H D].
        k (torch.Tensor): Key tensor [B S H D].
        v (torch.Tensor): Value tensor [B S H D].
        heads_k_stride (int): Number of key heads to process in one step.
        dropout_p (float): Dropout probability. TODO: Not tested.
        softmax_scale (Optional[float]): Scale for softmax. Defaults to 1/sqrt(head_dim).
        key_padding_mask (Optional[torch.Tensor]): Mask for padding in keys.
        attn_q_chunk_size (Optional[int]): Chunk size for query attention.
        causal (bool): Whether to apply causal masking. NOT IMPLEMENTED YET.
        return_attn_probs (bool): Whether to return attention probabilities.
        group (Optional[dist.ProcessGroup]): Process group for distributed operations.
        bwd_event_sync (bool): Whether to synchronize backward events.
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, local_seq_len_q, num_q_heads, head_dim).
        Optional[torch.Tensor]: Attention probabilities if return_attn_probs is True, else None.
    '''
    return LlamaStandardAttn.apply(
        q,
        k,
        v,
        heads_k_stride,
        dropout_p,
        softmax_scale,
        key_padding_mask,
        attn_q_chunk_size,
        causal,
        return_attn_probs,
        process_group,
        bwd_event_sync,
    )
