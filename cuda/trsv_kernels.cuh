#pragma once

#include <cooperative_groups.h>
#include <cublas_v2.h>

#include <accessor/range.hpp>
#include <accessor/reduced_row_major.hpp>
#include <cinttypes>

#include "kernel_utils.cuh"
#include "utils.cuh"

enum class tmtx_t { upper, lower };
enum class dmtx_t { non_unit, unit };

namespace kernel {

// Must be called with 2-D blocks, and 1-D grid
template <std::int64_t block_size>
__global__ __launch_bounds__(block_size) void test_idx() {
    const std::int64_t diag_block_id = blockIdx.x;
    const std::int64_t row_idx =
        diag_block_id * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    const auto group = cg::this_thread_block();
    const int idx = group.thread_rank();
    const auto warp = cg::tiled_partition<WARP_SIZE>(group);
    const int warp_idx = warp.thread_rank();

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    printf("%d, %d: id = %d; warp: %d\n", tidx, tidy, idx, warp_idx);
}

}  // namespace kernel

template <typename ValueType>
void control_trsv(const matrix_info m_info, tmtx_t ttype, dmtx_t dtype,
                  const ValueType *mtx, const matrix_info x_info,
                  ValueType *x) {
    if (x_info.size[1] != 1) {
        throw "Error!";
    };
    if (ttype == tmtx_t::lower) {
        for (std::size_t row = 0; row < m_info.size[0]; ++row) {
            ValueType local_res{0};
            for (std::size_t col = 0; col < row; ++col) {
                const std::size_t midx = row * m_info.stride + col;
                local_res += mtx[midx] * x[col * x_info.stride];
            }
            const ValueType diag =
                dtype == dmtx_t::unit ? 1 : mtx[row * m_info.stride + row];
            const auto rhs_val = x[row * x_info.stride];
            x[row * x_info.stride] = (rhs_val - local_res) / diag;
        }
    } else {
        for (std::size_t inv_row = 0; inv_row < m_info.size[0]; ++inv_row) {
            const auto row = m_info.size[0] - 1 - inv_row;
            ValueType local_res{0};
            for (std::size_t inv_col = 0; inv_col < inv_row; ++inv_col) {
                const auto col = m_info.size[0] - 1 - inv_col;
                const std::size_t midx = row * m_info.stride + col;
                local_res += mtx[midx] * x[col * x_info.stride];
            }
            const ValueType diag =
                dtype == dmtx_t::unit ? 1 : mtx[row * m_info.stride + row];
            const auto rhs_val = x[row * x_info.stride];
            x[row * x_info.stride] = (rhs_val - local_res) / diag;
        }
    }
}

namespace kernel {

// Must be called with 2-D blocks, and a 1-element grid
template <std::int64_t diag_block_size, dmtx_t dmtx, typename ValueType>
__global__ __launch_bounds__(diag_block_size *diag_block_size) void lower_trsv(
    const std::int64_t diag_block_start, const matrix_info m_info,
    const ValueType *__restrict__ mtx, const matrix_info x_info,
    ValueType *__restrict__ x) {
    // assert: blockDim.x == blockDim.y == diag_block_size
    // assert: blockIdx.x == blockIdx.y == 1
    const std::int64_t row_idx = diag_block_start + threadIdx.x;
    const std::int64_t col_idx = diag_block_start + threadIdx.y;
    if (diag_block_start >= m_info.size[0]) {
        return;
    }
    const auto group = cg::this_thread_block();
    constexpr int shared_stride = diag_block_size + 1;
    __shared__ ValueType shared[diag_block_size * shared_stride];

    // Read into shared memory all necessary matrix data and store it transposed
    // (with stride = BS+1)
    if (threadIdx.x <= threadIdx.y && row_idx < m_info.size[0] &&
        col_idx < m_info.size[1]) {
        shared[threadIdx.x * shared_stride + threadIdx.y] =
            (dmtx == dmtx_t::unit && threadIdx.x == threadIdx.y)
                ? ValueType{1}
                : mtx[row_idx + col_idx * m_info.stride];
    }
    group.sync();
    auto warp = cg::tiled_partition<WARP_SIZE>(group);
    if (group.thread_rank() / WARP_SIZE == 0) {
        const auto warp_id = warp.thread_rank();
        const std::int64_t x_idx = (diag_block_start + warp_id) * x_info.stride;
        auto local_solution = x[x_idx];

        for (int i = 0; i < WARP_SIZE; ++i) {
            const auto mtx_val = shared[i * shared_stride + warp_id];
            const auto current_x = warp.shfl(local_solution / mtx_val, i);
            if (warp_id == i) {
                local_solution = current_x;
            }
            if (warp_id > i) {
                local_solution -= mtx_val * current_x;
            }
        }
        // Write solution back
        if (diag_block_start + warp_id < x_info.size[0]) {
            x[x_idx] = local_solution;
        }
    }
}

// Must be called with 1-D blocks, and 1-D grid
template <std::int64_t block_size, std::int64_t solved_size, typename ValueType>
__global__ __launch_bounds__(block_size) void lower_trsv_apply(
    const std::int64_t apply_start, const matrix_info m_info,
    const ValueType *__restrict__ mtx, const matrix_info x_info,
    ValueType *__restrict__ x) {
    // assert: blockDim.x == blockDim.y == diag_block_size
    // assert: blockDim.x % solved_size == 0
    const std::int64_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const std::int64_t mtx_row =
        apply_start + solved_size + global_id / solved_size;
    const std::int64_t mtx_col = apply_start + global_id % solved_size;
    if (mtx_row >= m_info.size[0]) {
        return;
    }
    const auto group = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(group);
    __shared__ ValueType cached_solution[solved_size];

    // Fill in the cached solution with the first solved_size threads
    if (group.thread_rank() < solved_size) {
        cached_solution[group.thread_rank()] =
            x[(apply_start + group.thread_rank()) * x_info.stride];
    }
    group.sync();
    auto local_result =
        (mtx_col < m_info.size[1])
            ? mtx[mtx_row * m_info.stride + mtx_col] *
                  cached_solution[warp.thread_rank() * x_info.stride]
            : ValueType{0};
    auto updated_x = reduce(warp, local_result,
                            [](ValueType a, ValueType b) { return a + b; });
    // each warp writes once
    if (warp.thread_rank() == 0) {
        x[mtx_row] -= updated_x;
    }
}

}  // namespace kernel

template <typename ValueType>
void trsv(const matrix_info m_info, tmtx_t ttype, dmtx_t dtype,
          const ValueType *mtx, const matrix_info x_info, ValueType *x) {
    constexpr std::int32_t diag_block_size{kernel::WARP_SIZE};
    const dim3 block_solve(diag_block_size, diag_block_size, 1);
    // const dim3 grid(ceildiv<std::int32_t>(m_info.size[0], block_size), 1, 1);
    const dim3 grid_solve(1, 1, 1);
    auto run_solve = [&](std::int64_t block_start) {
        if (dtype == dmtx_t::unit) {
            kernel::lower_trsv<diag_block_size, dmtx_t::unit>
                <<<grid_solve, block_solve>>>(block_start, m_info, mtx, x_info,
                                              x);
        } else {
            kernel::lower_trsv<diag_block_size, dmtx_t::non_unit>
                <<<grid_solve, block_solve>>>(block_start, m_info, mtx, x_info,
                                              x);
        }
    };
    const std::int32_t block_size_apply = kernel::WARP_SIZE * kernel::WARP_SIZE;
    const dim3 block_apply(block_size_apply, 1, 1);
    for (std::int64_t block_start = 0; block_start < m_info.size[0];
         block_start += kernel::WARP_SIZE) {
        run_solve(block_start);
        const dim3 grid_apply(
            ceildiv(static_cast<std::int64_t>(m_info.size[0]) - block_start,
                    static_cast<std::int64_t>(block_size_apply /
                                              kernel::WARP_SIZE)),
            1, 1);
        //*
        if (block_start + kernel::WARP_SIZE < m_info.size[0]) {
            kernel::lower_trsv_apply<block_size_apply, kernel::WARP_SIZE>
                <<<grid_apply, block_apply>>>(block_start, m_info, mtx, x_info,
                                              x);
        }
        //*/
    }
}

namespace kernel {

__global__ __launch_bounds__(1) void trsv_init(std::uint32_t *block_idxs) {
    block_idxs[0] = ~std::uint32_t{0};  // last ready block column
    block_idxs[1] = 0;                  // next block row
}

// PTX instruction nanosleep kind of yields
// See:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-nanosleep
// Note: Looks like the best number for grid is: ceildiv(n, WARP_SIZE) since:
//       n=1000, grid = 32; n=10000, grid = 313
// Paper to follow for Impl.: https://epubs.siam.org/doi/abs/10.1137/12088358X

// Must be called with 2-D blocks, and a 1-D grid
template <std::int32_t swarp_size, std::int32_t swarps_per_block, dmtx_t dmtx,
          typename ValueType>
__global__ __launch_bounds__(swarps_per_block *swarp_size) void lower_trsv_2(
    const matrix_info m_info, const ValueType *__restrict__ mtx,
    const matrix_info x_info, ValueType *__restrict__ x,
    std::uint32_t *idx_helper) {
    static_assert(swarp_size <= WARP_SIZE,
                  "Subwarp size must be smaller than the WARP_SIZE");
    static_assert((swarp_size & (swarp_size - 1)) == 0,
                  "swarp_size must be a power of 2");
    static_assert(swarp_size % (swarps_per_block) == 0,
                  "swarp_size must be a multiple of swarps_per_block");
    // assert: blockDim.x == swarp_size; blockDim.y = swarps_per_block;
    //         blockDim.z = 1
    constexpr int triang_stride = swarp_size + 1;

    //__shared__ ValueType shared[swarp_size * triang_stride];
    __shared__ ValueType triang[swarp_size * triang_stride];
    __shared__ std::uint32_t shared_row_block_idx[1];
    //= reinterpret_cast<std::uint32_t *>(
    //    triang_stride + swarp_size * triang_stride);
    // correction value for x[block_col * swarp_size + threadIdx.x]
    __shared__ ValueType x_correction[swarp_size];

    const auto group = cg::this_thread_block();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        *shared_row_block_idx = atomicInc(idx_helper + 1, ~std::uint32_t{0});
    }
    group.sync();
    const std::int64_t row_block_idx = *shared_row_block_idx;
    // printf("(x, y): (%d, %d) ");

    if (row_block_idx * swarp_size >= m_info.size[0]) {
        return;
    }

    // All threads: load triangular matrix into shared memory
    // Note: Read it coalesced and transpose it.
    //       L is stored in column major for fast updates
    for (int row = threadIdx.y; row < swarp_size; row += swarps_per_block) {
        // threadIdx.x stores the column here to read coalesced
        const auto col = threadIdx.x;
        const std::int64_t global_row = row_block_idx * swarp_size + row;
        const std::int64_t global_col = row_block_idx * swarp_size + col;
        triang[col * triang_stride + row] =
            (dmtx == dmtx_t::unit && col == row || row < col ||
             global_row >= m_info.size[0])
                ? ValueType{1}
                : mtx[global_row * m_info.stride + global_col];
    }
    group.sync();
    // TODO maybe change type of idx_helper
    volatile std::int32_t *last_finished_col =
        reinterpret_cast<std::int32_t *>(idx_helper);

    constexpr int num_local_rows = swarp_size / swarps_per_block;
    ValueType local_row_result[num_local_rows] = {};
    for (int col_block = 0; col_block < row_block_idx; ++col_block) {
        const auto global_col = col_block * swarp_size + threadIdx.x;
        // Wait until result is known for current column block
        // Maybe add __nanosleep(200) to ensure it is yielded
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            while (*last_finished_col < col_block) {
            }
        }
        group.sync();
        const auto x_cached = x[global_col * x_info.stride];
#pragma unroll
        for (int local_row_idx = 0; local_row_idx < num_local_rows;
             ++local_row_idx) {
            const auto global_row = row_block_idx * swarp_size +
                                    local_row_idx * swarps_per_block +
                                    threadIdx.y;
            const std::int64_t mtx_idx =
                global_row * m_info.stride + global_col;
            local_row_result[local_row_idx] += x_cached * mtx[mtx_idx];
        }
    }
    const auto swarp = cg::tiled_partition<swarp_size>(group);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < swarp_size; ++i) {
            x_correction[i] = 0;
        }
    }
    group.sync();
#pragma unroll
    for (int local_row_idx = 0; local_row_idx < num_local_rows;
         ++local_row_idx) {
        x_correction[local_row_idx * swarps_per_block + threadIdx.y] =
            reduce(swarp, local_row_result[local_row_idx],
                   [](ValueType a, ValueType b) { return a + b; });
    }
    group.sync();

    // Solve triangular system
    if (threadIdx.y == 0) {
        const auto row = threadIdx.x;
        const std::int64_t x_idx =
            (row_block_idx * swarp_size + row) * x_info.stride;
        auto local_solution = x[x_idx] - x_correction[row];

        for (int col = 0; col < swarp_size; ++col) {
            const auto mtx_val = triang[col * triang_stride + row];
            const auto current_x = swarp.shfl(local_solution / mtx_val, col);
            if (row == col) {
                local_solution = current_x;
            }
            if (row > col) {
                local_solution -= mtx_val * current_x;
            }
        }
        x[x_idx] = local_solution;
        __threadfence();
        if (threadIdx.x == 0) {
            *last_finished_col = static_cast<std::uint32_t>(row_block_idx);
        }
    }
}

}  // namespace kernel

template <typename ValueType>
void trsv_2(const matrix_info m_info, tmtx_t ttype, dmtx_t dtype,
            const ValueType *mtx, const matrix_info x_info, ValueType *x,
            std::uint32_t *trsv_helper) {
    constexpr std::int32_t subwarp_size{kernel::WARP_SIZE};
    constexpr std::int32_t swarps_per_block{4};
    const dim3 block_solve(subwarp_size, swarps_per_block, 1);
    const dim3 grid_solve(
        ceildiv(m_info.size[0], static_cast<std::size_t>(subwarp_size)), 1, 1);

    kernel::trsv_init<<<1, 1>>>(trsv_helper);
    if (dtype == dmtx_t::unit) {
        kernel::lower_trsv_2<subwarp_size, swarps_per_block, dmtx_t::unit>
            <<<grid_solve, block_solve>>>(m_info, mtx, x_info, x, trsv_helper);
    } else {
        kernel::lower_trsv_2<subwarp_size, swarps_per_block, dmtx_t::non_unit>
            <<<grid_solve, block_solve>>>(m_info, mtx, x_info, x, trsv_helper);
    }
}

namespace kernel {

// PTX instruction nanosleep kind of yields
// See:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-nanosleep
// Note: Looks like the best number for grid is: ceildiv(n, WARP_SIZE) since:
//       n=1000, grid = 32; n=10000, grid = 313
// Paper to follow for Impl.: https://epubs.siam.org/doi/abs/10.1137/12088358X

// Must be called with 2-D blocks, and a 1-D grid
template <std::int32_t swarp_size, std::int32_t swarps_per_block, dmtx_t dmtx,
          typename ValueType>
__global__ __launch_bounds__(swarps_per_block *swarp_size) void lower_trsv_3(
    const matrix_info m_info, const ValueType *__restrict__ mtx,
    const matrix_info x_info, ValueType *__restrict__ x,
    std::uint32_t *idx_helper) {
    static_assert(swarp_size <= WARP_SIZE,
                  "Subwarp size must be smaller than the WARP_SIZE");
    static_assert((swarp_size & (swarp_size - 1)) == 0,
                  "swarp_size must be a power of 2");
    static_assert(swarp_size % (swarps_per_block) == 0,
                  "swarp_size must be a multiple of swarps_per_block");
    // assert: blockDim.x == swarp_size; blockDim.y = swarps_per_block;
    //         blockDim.z = 1
    constexpr int triang_stride = swarp_size + 1;

    // stores the trianglular system in column major
    __shared__ ValueType triang[swarp_size * triang_stride];
    //__shared__ ValueType mtx_cache[swarp_size * triang_stride];
    __shared__ std::uint32_t shared_row_block_idx[1];
    //= reinterpret_cast<std::uint32_t *>(
    //    triang_stride + swarp_size * triang_stride);
    // correction value for x[block_col * swarp_size + threadIdx.x]
    __shared__ ValueType x_correction[swarp_size];

    const auto group = cg::this_thread_block();
    const auto swarp = cg::tiled_partition<swarp_size>(group);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        *shared_row_block_idx = atomicInc(idx_helper + 1, ~std::uint32_t{0});
    }
    group.sync();
    const std::int64_t row_block_idx = *shared_row_block_idx;

    if (row_block_idx * swarp_size >= m_info.size[0]) {
        return;
    }

    // All threads: load triangular matrix into shared memory
    // Note: Read it coalesced and transpose it.
    //       L is stored in column major for fast updates
    for (int row = threadIdx.y; row < swarp_size; row += swarps_per_block) {
        // threadIdx.x stores the column here to read coalesced
        const auto col = threadIdx.x;
        const std::int64_t global_row = row_block_idx * swarp_size + row;
        const std::int64_t global_col = row_block_idx * swarp_size + col;
        triang[col * triang_stride + row] =
            (dmtx == dmtx_t::unit && col == row)
                ? ValueType{1}
                : (col <= row && global_row < m_info.size[0] &&
                   global_col < m_info.size[1])
                      ? mtx[global_row * m_info.stride + global_col]
                      : ValueType{0};
    }
    group.sync();
    // Invert lower triangular matrix

    // First: make diagonals 1
    if (dmtx == dmtx_t::non_unit) {
        if (threadIdx.y == 0) {
            for (int row = 0; row < swarp_size; ++row) {
                const int col = threadIdx.x;
                const auto triang_idx = col * triang_stride + row;
                const auto local_val = triang[triang_idx];
                const auto diag_el = ValueType{1} / swarp.shfl(local_val, row);
                if (col <= row) {
                    triang[triang_idx] =
                        (row == col) ? diag_el : local_val * diag_el;
                }
            }
        }
    }
    // No need to sync since the same warp will continue to compute

    if (threadIdx.y == 0) {
        // Compute remaining inverse in-place (with Gauss-Jordan elimination)
        for (int diag_id = 0; diag_id < swarp_size; ++diag_id) {
            const int col = threadIdx.x;
            const auto diag_el = triang[diag_id * triang_stride + diag_id];
            for (int row = diag_id + 1; row < swarp_size; ++row) {
                auto factor =
                    -triang[diag_id * triang_stride + row];  // broadcast
                const int triang_idx = col * triang_stride + row;
                const auto triang_val = triang[triang_idx];
                if (col < row) {
                    triang[triang_idx] =
                        (col == diag_id)
                            ? factor * diag_el
                            : triang_val +
                                  factor *
                                      triang[col * triang_stride + diag_id];
                }
                swarp.sync();
            }
        }
    }
    group.sync();

    // TODO maybe change type of idx_helper
    volatile std::int32_t *last_finished_col =
        reinterpret_cast<std::int32_t *>(idx_helper);

    constexpr int num_local_rows = swarp_size / swarps_per_block;
    ValueType local_row_result[num_local_rows] = {};
    for (int col_block = 0; col_block < row_block_idx; ++col_block) {
        const auto global_col = col_block * swarp_size + threadIdx.x;
        // Wait until result is known for current column block
        // Maybe add __nanosleep(200) to ensure it is yielded
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            while (*last_finished_col < col_block) {
            }
        }
        group.sync();
        const auto x_cached = x[global_col * x_info.stride];
#pragma unroll
        for (int local_row_idx = 0; local_row_idx < num_local_rows;
             ++local_row_idx) {
            const auto global_row = row_block_idx * swarp_size +
                                    local_row_idx * swarps_per_block +
                                    threadIdx.y;
            const std::int64_t mtx_idx =
                global_row * m_info.stride + global_col;
            if (global_row < m_info.size[0]) {
                local_row_result[local_row_idx] += x_cached * mtx[mtx_idx];
            }
        }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < swarp_size; ++i) {
            x_correction[i] = 0;
        }
    }
    group.sync();
#pragma unroll
    for (int local_row_idx = 0; local_row_idx < num_local_rows;
         ++local_row_idx) {
        x_correction[local_row_idx * swarps_per_block + threadIdx.y] =
            reduce(swarp, local_row_result[local_row_idx],
                   [](ValueType a, ValueType b) { return a + b; });
    }
    group.sync();

    // Solve triangular system with GEMV (since triang. sys. is inverted)
    if (threadIdx.y == 0) {
        const int row = threadIdx.x;
        const std::int64_t x_idx =
            (row_block_idx * swarp_size + row) * x_info.stride;
        // compute the local x and distribute it via shfl
        const ValueType local_x = (x_idx < x_info.size[0])
                                      ? x[x_idx] - x_correction[row]
                                      : ValueType{0};
        ValueType local_solution{};
        for (int col = 0; col < swarp_size; ++col) {
            // GEMV
            local_solution +=
                triang[col * triang_stride + row] * swarp.shfl(local_x, col);
        }
        if (x_idx < x_info.size[0]) {
            x[x_idx] = local_solution;
        }
        __threadfence();
        if (threadIdx.x == 0) {
            *last_finished_col = static_cast<std::uint32_t>(row_block_idx);
        }
    }
}

}  // namespace kernel

template <typename ValueType>
void trsv_3(const matrix_info m_info, tmtx_t ttype, dmtx_t dtype,
            const ValueType *mtx, const matrix_info x_info, ValueType *x,
            std::uint32_t *trsv_helper) {
    constexpr std::int32_t subwarp_size{kernel::WARP_SIZE};
    constexpr std::int32_t swarps_per_block{4};
    const dim3 block_solve(subwarp_size, swarps_per_block, 1);
    const dim3 grid_solve(
        ceildiv(m_info.size[0], static_cast<std::size_t>(subwarp_size)), 1, 1);

    kernel::trsv_init<<<1, 1>>>(trsv_helper);
    if (dtype == dmtx_t::unit) {
        kernel::lower_trsv_3<subwarp_size, swarps_per_block, dmtx_t::unit>
            <<<grid_solve, block_solve>>>(m_info, mtx, x_info, x, trsv_helper);
    } else {
        kernel::lower_trsv_3<subwarp_size, swarps_per_block, dmtx_t::non_unit>
            <<<grid_solve, block_solve>>>(m_info, mtx, x_info, x, trsv_helper);
    }
}

namespace kernel {

// Paper to follow for Impl.: https://epubs.siam.org/doi/abs/10.1137/12088358X

// Must be called with 2-D blocks, and a 1-D grid
template <std::int32_t swarp_size, std::int32_t swarps_per_block, dmtx_t dmtx,
          typename MtxAccessor, typename VecAccessor>
__global__ __launch_bounds__(swarps_per_block *swarp_size) void acc_lower_trsv(
    gko::acc::range<MtxAccessor> mtx, gko::acc::range<VecAccessor> x,
    std::uint32_t *idx_helper) {
    static_assert(swarp_size <= WARP_SIZE,
                  "Subwarp size must be smaller than the WARP_SIZE");
    static_assert((swarp_size & (swarp_size - 1)) == 0,
                  "swarp_size must be a power of 2");
    static_assert(swarp_size % (swarps_per_block) == 0,
                  "swarp_size must be a multiple of swarps_per_block");
    // assert: blockDim.x == swarp_size; blockDim.y = swarps_per_block;
    //         blockDim.z = 1
    using ar_type = typename MtxAccessor::arithmetic_type;
    constexpr int triang_stride = swarp_size + 1;

    // stores the trianglular system in column major
    __shared__ ar_type triang[swarp_size * triang_stride];
    __shared__ std::uint32_t shared_row_block_idx[1];
    //= reinterpret_cast<std::uint32_t *>(
    //    triang_stride + swarp_size * triang_stride);
    // correction value for x[block_col * swarp_size + threadIdx.x]
    __shared__ ar_type x_correction[swarp_size];

    const auto group = cg::this_thread_block();
    const auto swarp = cg::tiled_partition<swarp_size>(group);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        *shared_row_block_idx = atomicInc(idx_helper + 1, ~std::uint32_t{0});
    }
    group.sync();
    const std::int64_t row_block_idx = *shared_row_block_idx;

    if (row_block_idx * swarp_size >= mtx.length(0)) {
        return;
    }

    // All threads: load triangular matrix into shared memory
    // Note: Read it coalesced and transpose it.
    //       L is stored in column major for fast updates
    for (int row = threadIdx.y; row < swarp_size; row += swarps_per_block) {
        // threadIdx.x stores the column here to read coalesced
        const auto col = threadIdx.x;
        const std::int64_t global_row = row_block_idx * swarp_size + row;
        const std::int64_t global_col = row_block_idx * swarp_size + col;
        triang[col * triang_stride + row] =
            (dmtx == dmtx_t::unit && col == row)
                ? ar_type{1}
                : (col <= row && global_row < mtx.length(0) &&
                   global_col < mtx.length(1))
                      ? mtx(global_row, global_col)
                      : ar_type{0};
    }
    group.sync();
    // Invert lower triangular matrix

    // First: make diagonals 1
    if (dmtx == dmtx_t::non_unit) {
        if (threadIdx.y == 0) {
            for (int row = 0; row < swarp_size; ++row) {
                const int col = threadIdx.x;
                const auto triang_idx = col * triang_stride + row;
                const auto local_val = triang[triang_idx];
                const auto diag_el = ar_type{1} / swarp.shfl(local_val, row);
                if (col <= row) {
                    triang[triang_idx] =
                        (row == col) ? diag_el : local_val * diag_el;
                }
            }
        }
    }
    // No need to sync since the same warp will continue to compute

    if (threadIdx.y == 0) {
        // Compute remaining inverse in-place (with Gauss-Jordan elimination)
        for (int diag_id = 0; diag_id < swarp_size; ++diag_id) {
            const int col = threadIdx.x;
            const auto diag_el = triang[diag_id * triang_stride + diag_id];
            for (int row = diag_id + 1; row < swarp_size; ++row) {
                auto factor =
                    -triang[diag_id * triang_stride + row];  // broadcast
                const int triang_idx = col * triang_stride + row;
                const auto triang_val = triang[triang_idx];
                if (col < row) {
                    triang[triang_idx] =
                        (col == diag_id)
                            ? factor * diag_el
                            : triang_val +
                                  factor *
                                      triang[col * triang_stride + diag_id];
                }
                swarp.sync();
            }
        }
    }
    group.sync();

    // TODO maybe change type of idx_helper
    volatile std::int32_t *last_finished_col =
        reinterpret_cast<std::int32_t *>(idx_helper);

    //*
    constexpr int num_local_rows = swarp_size / swarps_per_block;
    ar_type local_row_result[num_local_rows] = {};
    for (int col_block = 0; col_block < row_block_idx; ++col_block) {
        const auto global_col = col_block * swarp_size + threadIdx.x;
        // Wait until result is known for current column block
        // Maybe add __nanosleep(200) to ensure it is yielded
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            while (*last_finished_col < col_block) {
            }
        }
        group.sync();
        const auto x_cached = x(global_col, 0);
#pragma unroll
        for (int local_row_idx = 0; local_row_idx < num_local_rows;
             ++local_row_idx) {
            const auto global_row = row_block_idx * swarp_size +
                                    local_row_idx * swarps_per_block +
                                    threadIdx.y;
            if (global_row < mtx.length(0)) {
                local_row_result[local_row_idx] +=
                    x_cached * mtx(global_row, global_col);
            }
        }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < swarp_size; ++i) {
            x_correction[i] = 0;
        }
    }
    group.sync();
#pragma unroll
    for (int local_row_idx = 0; local_row_idx < num_local_rows;
         ++local_row_idx) {
        x_correction[local_row_idx * swarps_per_block + threadIdx.y] =
            reduce(swarp, local_row_result[local_row_idx],
                   [](ar_type a, ar_type b) { return a + b; });
    }
    group.sync();

    // Solve triangular system with GEMV (since triang. sys. is inverted)
    if (threadIdx.y == 0) {
        const int row = threadIdx.x;
        const std::int64_t x_idx = row_block_idx * swarp_size + row;
        // compute the local x and distribute it via shfl
        const ar_type local_x = (x_idx < x.length(0))
                                    ? x(x_idx, 0) - x_correction[row]
                                    : ar_type{0};
        ar_type local_solution{};
        for (int col = 0; col < swarp_size; ++col) {
            // GEMV
            local_solution +=
                triang[col * triang_stride + row] * swarp.shfl(local_x, col);
        }
        if (x_idx < x.length(0)) {
            x(x_idx, 0) = local_solution;
        }
        __threadfence();
        if (threadIdx.x == 0) {
            *last_finished_col = static_cast<std::uint32_t>(row_block_idx);
        }
    }
}

}  // namespace kernel

template <typename ArType, typename StType>
void acc_trsv(const matrix_info m_info, tmtx_t ttype, dmtx_t dtype,
              const StType *mtx, const matrix_info x_info, StType *x,
              std::uint32_t *trsv_helper) {
    // Accessor Setup
    constexpr std::size_t dimensionality{2};
    std::array<std::size_t, dimensionality - 1> m_stride{m_info.stride};
    std::array<std::size_t, dimensionality - 1> x_stride{x_info.stride};

    using accessor =
        gko::acc::reduced_row_major<dimensionality, ArType, StType>;
    using range = gko::acc::range<accessor>;
    using c_range = gko::acc::range<typename accessor::const_accessor>;
    auto m_acc = c_range(m_info.size, mtx, m_stride);
    auto x_acc = range(x_info.size, x, x_stride);

    constexpr std::int32_t subwarp_size{kernel::WARP_SIZE};
    constexpr std::int32_t swarps_per_block{4};
    const dim3 block_solve(subwarp_size, swarps_per_block, 1);
    const dim3 grid_solve(
        ceildiv(m_info.size[0], static_cast<std::size_t>(subwarp_size)), 1, 1);

    kernel::trsv_init<<<1, 1>>>(trsv_helper);
    if (dtype == dmtx_t::unit) {
        kernel::acc_lower_trsv<subwarp_size, swarps_per_block, dmtx_t::unit>
            <<<grid_solve, block_solve>>>(m_acc, x_acc, trsv_helper);
    } else {
        kernel::acc_lower_trsv<subwarp_size, swarps_per_block, dmtx_t::non_unit>
            <<<grid_solve, block_solve>>>(m_acc, x_acc, trsv_helper);
    }
}

#define BIND_CUBLAS_TRSV(ValueType, CublasName)                                \
    void cublas_trsv(cublasHandle_t handle, cublasFillMode_t uplo,             \
                     cublasOperation_t trans, cublasDiagType_t dig, int n,     \
                     const ValueType *A, int lda, ValueType *x, int incx) {    \
        CUBLAS_CALL(CublasName(handle, uplo, trans, dig, n, A, lda, x, incx)); \
    }
BIND_CUBLAS_TRSV(double, cublasDtrsv)
BIND_CUBLAS_TRSV(float, cublasStrsv)
#undef BIND_CUBLAS_TRSV

template <typename ValueType>
void cublas_trsv(cublasHandle_t handle, tmtx_t ttype, dmtx_t dtype,
                 const matrix_info m_info, const ValueType *mtx,
                 const matrix_info x_info, ValueType *x) {
    // Note: CUBLAS expects the matrices to be stored in column major,
    //       so it needs to be transposed & upper, lower needs to be swapped
    cublasFillMode_t uplo = ttype == tmtx_t::upper ? CUBLAS_FILL_MODE_LOWER
                                                   : CUBLAS_FILL_MODE_UPPER;
    cublasDiagType_t diag =
        dtype == dmtx_t::unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
    cublas_trsv(
        handle, uplo, CUBLAS_OP_T, diag, static_cast<int>(m_info.size[0]), mtx,
        static_cast<int>(m_info.stride), x, static_cast<int>(x_info.stride));
}

