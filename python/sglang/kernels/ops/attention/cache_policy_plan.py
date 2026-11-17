"""Run-length/Fenwick CPU position compiler for frozen-death cache policy."""

from functools import cache
from hashlib import sha256


@cache
def load_cache_policy_plan():
    from tvm_ffi.cpp import load_inline

    return load_inline(
        "sglang_cache_policy_" + sha256(_SOURCE.encode()).hexdigest()[:16],
        cpp_sources=_SOURCE,
        extra_cflags=["-std=c++20", "-O3"],
    )


_SOURCE = r"""
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

void frozen_death_positions(tvm::ffi::TensorView expiry,
                           tvm::ffi::TensorView output) {
  auto valid = [](tvm::ffi::TensorView t) {
    return t.device().device_type == kDLCPU && t.ndim() == 1 &&
           t.is_contiguous() && t.dtype().code == kDLInt && t.dtype().bits == 32;
  };
  if (!valid(expiry) || !valid(output) || expiry.size(0) != output.size(0))
    throw std::invalid_argument("Frozen-death positions require equal CPU int32 vectors");
  const int64_t n = expiry.size(0);
  if (n >= INT32_MAX) throw std::invalid_argument("Too many tokens");
  if (!n) return;
  const auto* death = static_cast<const int32_t*>(expiry.data_ptr());
  auto* pos = static_cast<int32_t*>(output.data_ptr());
  std::vector<int32_t> starts{0};
  for (int32_t i = 0; i < n; ++i) {
    if (death[i] <= i) throw std::invalid_argument("Token expires before birth");
    if (i && death[i] != death[i-1]) starts.push_back(i);
  }
  starts.push_back(n);
  const int32_t runs = starts.size()-1;
  std::vector<int32_t> order(runs), tree(runs+1, 0);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return death[starts[a]] < death[starts[b]];
  });
  for (int32_t a = 0; a < runs;) {
    int32_t b = a+1;
    while (b < runs && death[starts[order[b]]] == death[starts[order[a]]]) ++b;
    // All tokens dying together observe the same pre-event positions.
    for (int32_t k = a; k < b; ++k) {
      const int32_t r = order[k];
      int32_t removed = 0;
      for (int32_t j = r; j; j -= j & -j) removed += tree[j];
      for (int32_t i = starts[r]; i < starts[r+1]; ++i) pos[i] = i-removed;
    }
    for (int32_t k = a; k < b; ++k) {
      const int32_t r = order[k], length = starts[r+1]-starts[r];
      for (int32_t j = r+1; j <= runs; j += j & -j) tree[j] += length;
    }
    a = b;
  }
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(frozen_death_positions, frozen_death_positions);
"""
