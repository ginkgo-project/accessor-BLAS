#pragma once

#include <cooperative_groups.h>

#include <cinttypes>

namespace kernel {

namespace cg = cooperative_groups;
constexpr int WARP_SIZE{32};

template <unsigned int subgroup_size, typename TileParentT, typename ValueType,
          typename Callable>
__device__ __forceinline__ ValueType
reduce(const cg::thread_block_tile<subgroup_size, TileParentT> &warp,
            ValueType local_data, Callable &&reduce_op) {
    // assert: warp.size() == subgroup_size && subgroup_size power of 2
#pragma unroll
    for (std::int32_t bitmask = 1; bitmask < subgroup_size; bitmask <<= 1) {
        const auto remote_data = warp.shfl_xor(local_data, bitmask);
        local_data = reduce_op(local_data, remote_data);
    }
    return local_data;
}

// MUST be called with group.size() >= WARP_SIZE
template <typename Group, typename ValueType, typename Callable>
__device__ void reduce(Group &&group, ValueType *__restrict__ shared,
                       Callable &&reduce_op) {
    const auto local_id = group.thread_rank();
    for (auto i = group.size() / 2; i >= WARP_SIZE; i /= 2) {
        group.sync();
        if (local_id < i) {
            shared[local_id] =
                reduce_op(shared[local_id], shared[local_id + i]);
        }
    }
    auto warp = cg::tiled_partition<WARP_SIZE>(group);
    if (local_id < WARP_SIZE) {
        const auto local_data = shared[local_id];
        auto result = reduce(warp, local_data, reduce_op);
        if (warp.thread_rank() == 0) {
            shared[0] = result;
        }
    }
}

}  // namespace kernel

