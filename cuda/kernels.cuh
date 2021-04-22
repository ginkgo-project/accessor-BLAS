#pragma once

#include <cooperative_groups.h>

#include <accessor/range.hpp>
#include <accessor/reduced_row_major.hpp>
#include <cinttypes>

#include "utils.cuh"

namespace kernel {

namespace cg = cooperative_groups;

template <typename Group, typename ValueType>
__device__ void reduce(Group &&group, ValueType *__restrict__ shared) {
    const auto local_id = group.thread_rank();
    for (auto i = group.size() / 2; i >= 1; i /= 2) {
        group.sync();
        if (local_id < i) {
            shared[local_id] = shared[local_id] + shared[local_id + i];
        }
    }
}

template <std::int64_t block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void spmv(
    const matrix_info minfo, const ValueType *__restrict__ mtx,
    const matrix_info vinfo, const ValueType *__restrict__ b,
    ValueType *__restrict__ res) {
    // expect vinfo.size[1] == 1
    const std::int64_t row_idx{blockIdx.x};
    if (row_idx > minfo.size[0]) {
        return;
    }
    const std::int64_t row_start = row_idx * minfo.stride;
    __shared__ ValueType shared[block_size];
    ValueType local_result{};
    const auto group = cg::this_thread_block();
    const auto local_id = group.thread_rank();

    for (std::int64_t col = local_id; col < minfo.size[1]; col += block_size) {
        const auto mtx_val = mtx[row_start + col];
        const auto b_val = b[col * vinfo.stride];
        local_result += mtx_val * b_val;
    }
    shared[local_id] = local_result;
    reduce(group, shared);
    if (local_id == 0) {
        res[row_idx * vinfo.stride] = shared[local_id];
    }
}

template <std::int64_t block_size, typename MtxRange, typename BRange, typename ResRange>
__global__ __launch_bounds__(block_size) void acc_spmv(
    MtxRange mtx, BRange b, ResRange res) {
    using ar_type = decltype(+mtx(0, 0));
    // expect vinfo.size[1] == 1
    const std::int64_t row_idx{blockIdx.x};
    if (row_idx > mtx.length(0)) {
        return;
    }

    const auto num_cols = mtx.length(1);
    __shared__ ar_type shared[block_size];
    ar_type local_result{};
    const auto group = cg::this_thread_block();
    const auto local_id = group.thread_rank();

    for (std::int64_t col = local_id; col < num_cols; col += block_size) {
        local_result += mtx(row_idx, col) * b(col, 1);
    }
    shared[local_id] = local_result;
    reduce(group, shared);
    if (local_id == 0) {
        res(row_idx, 1) = shared[local_id];
    }
}

}  // namespace kernel

template <typename ValueType>
ValueType ceildiv(ValueType a, ValueType b) {
    return (a - 1) / b + 1;
}

template <typename ValueType>
void spmv(const matrix_info minfo, const ValueType *mtx,
          const matrix_info vinfo, const ValueType *b, ValueType *res) {
    constexpr std::int32_t block_size{512};
    const dim3 block(block_size, 1, 1);
    // const dim3 grid(ceildiv<std::int32_t>(minfo.size[0], block_size), 1, 1);
    const dim3 grid(minfo.size[0], 1, 1);

    kernel::spmv<block_size, ValueType>
        <<<grid, block>>>(minfo, mtx, vinfo, b, res);
}

template <typename ArType, typename StType>
void acc_spmv(const matrix_info minfo, const StType *mtx,
              const matrix_info vinfo, const StType *b, StType *res) {
    constexpr std::int32_t block_size{512};
    const dim3 block(block_size, 1, 1);
    // const dim3 grid(ceildiv<std::int32_t>(minfo.size[0], block_size), 1, 1);
    const dim3 grid(minfo.size[0], 1, 1);

    // Accessor Setup
    constexpr std::size_t dimensionality{2};
    std::array<std::size_t, dimensionality - 1> m_stride{minfo.stride};
    std::array<std::size_t, dimensionality - 1> v_stride{vinfo.stride};

    using accessor =
        gko::acc::reduced_row_major<dimensionality, ArType, StType>;
    using range = gko::acc::range<accessor>;
    using c_range = gko::acc::range<typename accessor::const_accessor>;
    auto m_acc = c_range(minfo.size, mtx, m_stride);
    auto b_acc = c_range(vinfo.size, b, v_stride);
    auto res_acc = range(vinfo.size, res, v_stride);

    kernel::acc_spmv<block_size><<<grid, block>>>(m_acc, b_acc, res_acc);
}

/*
template<typename MtxAccessor, typename VecAccessor>
__global__ spmv()
*/
