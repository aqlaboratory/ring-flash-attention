#!/bin/bash

set -ex

num_gpus=8

tests=(
  test_llama3_flash_attn_varlen_func.py
  test_llama3_prepare_cu_seqlens.py
  test_ring_flash_attn_func.py
  test_ring_flash_attn_varlen_func.py
  test_stripe_flash_attn_func.py
  test_zigzag_ring_flash_attn_func.py
  test_zigzag_ring_flash_attn_varlen_func.py
  test_cond_llama_fwd_ring_bwd_flash_attn.py
  test_pack_first_stride.py
  test_llama_stride_flags.py
)

cpu_tests=(
  test_reduce_scatter_handle_manager
  test_llama_bwd_stride_flags
  test_fa3_arg_binding
)

# Run as bare module names with test/ on the path. The previous `test.<name>` form
# resolved to the CPython stdlib `test` package (there is no test/__init__.py here),
# so these never actually ran.
for test in "${cpu_tests[@]}"; do
  PYTHONPATH=.:test python -m unittest "$test"
done

for test in "${tests[@]}"; do
  torchrun --nproc_per_node $num_gpus test/$test
done

torchrun --nproc_per_node $num_gpus test/test_triton_kernels.py

for test in "${tests[@]}"; do
  torchrun --nproc_per_node $num_gpus test/$test compile
done
