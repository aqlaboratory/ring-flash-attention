#!/bin/bash

set -ex

cd "$(dirname "$0")/.."
# Test this checkout. torchrun puts only test/ on the path, so without this the GPU tests
# would import whatever ring_flash_attn is installed in site-packages instead.
export PYTHONPATH="$PWD:$PWD/test${PYTHONPATH:+:$PYTHONPATH}"

num_gpus=${NUM_GPUS:-8}

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

# The ring variants each GPU test calls, as named in _VALIDATED_VARIANTS. Under fa3/fa4 a
# test runs only if all of them are in scope there; tests calling none always run.
declare -A test_variants=(
  [test_llama3_flash_attn_varlen_func.py]="llama3_flash_attn_varlen"
  [test_llama3_prepare_cu_seqlens.py]=""
  [test_ring_flash_attn_func.py]="ring_flash_attn_forward ring_flash_attn_backward"
  [test_ring_flash_attn_varlen_func.py]="ring_flash_attn_varlen"
  [test_stripe_flash_attn_func.py]="stripe_flash_attn"
  [test_zigzag_ring_flash_attn_func.py]="zigzag_ring_flash_attn"
  [test_zigzag_ring_flash_attn_varlen_func.py]="zigzag_ring_flash_attn_varlen"
  [test_cond_llama_fwd_ring_bwd_flash_attn.py]="llama_fwd_ring_bwd_flash_attn"
  [test_pack_first_stride.py]=""
  [test_llama_stride_flags.py]="llama_fwd_ring_bwd_flash_attn"
)

cpu_tests=(
  test_reduce_scatter_handle_manager
  test_llama_bwd_stride_flags
  test_fa3_arg_binding
  test_fa4_arg_binding
  test_flash_attn_backend_selector
  test_flash_attn_cold_import
)

# Run as bare module names with test/ on the path. The previous `test.<name>` form
# resolved to the CPython stdlib `test` package (there is no test/__init__.py here),
# so these never actually ran.
for test in "${cpu_tests[@]}"; do
  python -m unittest "$test"
done

# Resolved the same way the tests will resolve it (RING_FLASH_ATTN_BACKEND, default auto).
# scope is "ALL" under fa2, else the in-scope variant names.
# (A plain assignment, so set -e stops the run if the import fails.)
backend_info=$(python -c "
from ring_flash_attn.flash_attn_backend import BACKEND, _VALIDATED_VARIANTS
s = _VALIDATED_VARIANTS.get(BACKEND)
print(BACKEND, 'ALL' if s is None else ' '.join(sorted(s)))
")
read -r backend scope <<< "$backend_info"

in_scope() {
  [ "$scope" = "ALL" ] && return 0
  for variant in ${test_variants[$1]}; do
    [[ " $scope " == *" $variant "* ]] || return 1
  done
  return 0
}

gpu_tests=()
skipped=()
for test in "${tests[@]}"; do
  if [[ ! -v test_variants[$test] ]]; then
    echo "test.sh: $test has no test_variants entry" >&2
    exit 1
  fi
  if in_scope "$test"; then
    gpu_tests+=("$test")
  else
    skipped+=("$test")
  fi
done
echo "backend: $backend; skipping (not in scope for $backend): ${skipped[*]:-none}"

# Seam vs plain PyTorch first: the ring tests below compare against the seam, so their
# diffs mean nothing if the seam itself is wrong.
python test/test_seam_reference.py

for test in "${gpu_tests[@]}"; do
  torchrun --nproc_per_node $num_gpus test/$test
done

torchrun --nproc_per_node $num_gpus test/test_triton_kernels.py

for test in "${gpu_tests[@]}"; do
  torchrun --nproc_per_node $num_gpus test/$test compile
done
