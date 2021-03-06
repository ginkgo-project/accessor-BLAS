#pragma once

#include <cinttypes>


#include <cooperative_groups.h>
#include <cuda.h>


namespace kernel {


namespace cg = cooperative_groups;
constexpr int WARP_SIZE{32};


/**
 * Reduces values in registers in a sub-warp.
 *
 * @param warp  CUDA thread block tile, specifying the sub-warp
 * @param local_data  value of the current thread that is reduced
 * @param reduce_op  reduction operation used in this reduction
 *
 * @returns the butterfly reduction from the subwarp. All threads in the
 *          given sub-warp return the same result.
 *
 * @note cg::thread_block_tile in CUDA >= 11 has an additional type argument,
 *       which CUDA 10 and lower does not have.
 */
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000

template <unsigned int subgroup_size, typename ValueType, typename Callable,
          typename... TileParams>
__device__ __forceinline__ ValueType
reduce(const cg::thread_block_tile<subgroup_size, TileParams...> &warp,
       ValueType local_data, Callable &&reduce_op)
{
#elif defined(CUDA_VERSION) && CUDA_VERSION < 11000

template <unsigned int subgroup_size, typename ValueType, typename Callable>
__device__ __forceinline__ ValueType
reduce(const cg::thread_block_tile<subgroup_size> &warp, ValueType local_data,
       Callable &&reduce_op)
{
#endif
    // assert: warp.size() == subgroup_size && subgroup_size power of 2
#pragma unroll
    for (std::int32_t bitmask = 1; bitmask < subgroup_size; bitmask <<= 1) {
        const auto remote_data = warp.shfl_xor(local_data, bitmask);
        local_data = reduce_op(local_data, remote_data);
    }
    return local_data;
}

// MUST be called with group.size() >= WARP_SIZE
/**
 * Reduces values in shared in the thread block group `group`. Exactly
 * group.size() values in shared are reduced, and shared will be overwritten
 * during the reduction. The result of the reduction will be in shared[0].
 * For the best result, use __shared__ memory for shared.
 *
 * @note This must be called with group.size >= WARP_SIZE
 *
 * @param group  CUDA thread block, specifying the thread block used in the
 *               reduction
 * @param shared  memory which will be reduced, containing group.size() elements
 * @param reduce_op  reduction operation used in this reduction
 */
template <typename Group, typename ValueType, typename Callable>
__device__ void reduce(Group &&group, ValueType *__restrict__ shared,
                       Callable &&reduce_op)
{
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
