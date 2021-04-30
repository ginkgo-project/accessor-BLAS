#pragma once

#include <cooperative_groups.h>
#include <cinttypes>

namespace kernel {

namespace cg = cooperative_groups;

template <typename Group, typename ValueType, typename Callable>
__device__ void reduce(Group &&group, ValueType *__restrict__ shared,
                       Callable &&reduce_op) {
    const auto local_id = group.thread_rank();
    constexpr int warp_size = 32;
    for (auto i = group.size() / 2; i >= warp_size; i /= 2) {
        group.sync();
        if (local_id < i) {
            shared[local_id] =
                reduce_op(shared[local_id], shared[local_id + i]);
        }
    }
    auto warp = cg::tiled_partition<warp_size>(group);
    if (local_id < warp_size) {
        auto local_data = shared[local_id];
#pragma unroll
        for (std::int32_t bitmask = 1; bitmask < warp.size(); bitmask <<= 1) {
            const auto remote_data = warp.shfl_xor(local_data, bitmask);
            local_data = reduce_op(local_data, remote_data);
        }
        if (warp.thread_rank() == 0) {
            shared[0] = local_data;
        }
    }
}


}  // namespace kernel

