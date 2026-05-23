from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import inspect
from functools import cache


__all__ = ["update_out_and_lse", "RingComm", "get_default_args"]


@cache
def _get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if "softcap" in args:
        args["softcap"] = 0.0
    return args


def get_default_args(func):
    if inspect.isfunction(func):
        return _get_default_args(func)
    else:
        # Use the origin _init_fn in CustomOpDef
        return _get_default_args(func._init_fn)


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


@torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


@torch.jit.script
def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty(
        (num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device
    )
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, : end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(
            dist.isend, to_send, self.send_rank, group=self._process_group
        )
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v


class AllGatherComm:
    def __init__(self, group=None) -> None:
        self.group = group
        self.handles = []

    def all_gather(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor):
        handle = dist.all_gather_into_tensor(
            output_tensor, input_tensor, group=self.group, async_op=True
        )
        self.handles.append(handle)

    def wait(self):
        for handle in self.handles:
            handle.wait()
        self.handles = []


class ReduceScatterHandleManager:
    def __init__(
        self,
        group=None,
        group_name: Optional[str] = None,
        use_coalesced: Optional[bool] = None,
    ):
        if group is None:
            if not dist.is_initialized():
                raise RuntimeError(
                    "torch.distributed is not initialized; initialize it or pass `group`"
                )
            group = dist.group.WORLD

        self.group = group
        self._world_size = dist.get_world_size(self.group)

        # Allow explicit override of the c10d functional group name to avoid
        # fragile introspection of ProcessGroup internals.
        if group_name is None:
            group_name = getattr(self.group, "group_name", None)
            if callable(group_name):
                try:
                    group_name = group_name()
                except TypeError:
                    group_name = None

        if group_name is None:
            group_name = getattr(self.group, "name", None)
            if callable(group_name):
                try:
                    group_name = group_name()
                except TypeError:
                    group_name = None

        if group_name is None:
            # some implementations expose a private attribute
            group_name = getattr(self.group, "_group_name", None)

        if group_name is None:
            raise RuntimeError(
                "Could not determine a process-group name for c10d functional collectives; pass explicit `group_name`"
            )

        self._group_name = group_name

        self._supports_coalesced = hasattr(
            torch.ops._c10d_functional, "reduce_scatter_tensor_coalesced"
        )
        if use_coalesced is None:
            self._use_coalesced = self._supports_coalesced
        else:
            self._use_coalesced = use_coalesced and self._supports_coalesced

        self.pending = None

    def issue(
        self,
        scatter_in_dk: torch.Tensor,
        scatter_in_dv: torch.Tensor,
        head_offset: int,
        width: int,
    ):
        if self.pending is not None:
            raise RuntimeError("issue called before draining previous reduce_scatter")

        if self._use_coalesced:
            out_dk, out_dv = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
                [scatter_in_dk, scatter_in_dv],
                "sum",
                self._world_size,
                self._group_name,
            )
        else:
            out_dk = torch.ops._c10d_functional.reduce_scatter_tensor(
                scatter_in_dk, "sum", self._world_size, self._group_name
            )
            out_dv = torch.ops._c10d_functional.reduce_scatter_tensor(
                scatter_in_dv, "sum", self._world_size, self._group_name
            )

        self.pending = (out_dk, out_dv, head_offset, width)

    def drain_to(self, dk: torch.Tensor, dv: torch.Tensor):
        if self.pending is None:
            return
        out_dk, out_dv, head_offset, width = self.pending
        out_dk = torch.ops._c10d_functional.wait_tensor(out_dk)
        out_dv = torch.ops._c10d_functional.wait_tensor(out_dv)
        dk_slice = dk[:, :, head_offset:head_offset + width]
        dv_slice = dv[:, :, head_offset:head_offset + width]
        dk_slice.copy_(out_dk.reshape(dk_slice.shape))
        dv_slice.copy_(out_dv.reshape(dv_slice.shape))
        self.pending = None
