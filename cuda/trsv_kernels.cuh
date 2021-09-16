#pragma once

#include <cinttypes>


#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <accessor/range.hpp>
#include <accessor/reduced_row_major.hpp>


#include "kernel_utils.cuh"
#include "utils.cuh"


enum class tmtx_t { upper, lower };
enum class dmtx_t { non_unit, unit };


namespace kernel {


// Must be called with 2-D blocks and 1-D grid
template <std::int64_t block_size>
__global__ __launch_bounds__(block_size) void test_idx()
{
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
                  const ValueType *mtx, const matrix_info x_info, ValueType *x)
{
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


// Implementation follows paper: A Fast Dense Triangular Solve in CUDA
//                               https://doi.org/10.1137/12088358X

__global__ __launch_bounds__(1) void trsv_init(std::uint32_t *block_idxs)
{
    block_idxs[0] = ~std::uint32_t{0};  // last ready block column
    block_idxs[1] = 0;                  // next block row
}

// Must be called with 2-D blocks and a 1-D grid
template <std::int32_t swarp_size, std::int32_t swarps_per_block, dmtx_t dmtx,
          typename ValueType>
__global__ __launch_bounds__(swarps_per_block *swarp_size) void lower_trsv(
    const matrix_info m_info, const ValueType *__restrict__ mtx,
    const matrix_info x_info, ValueType *__restrict__ x,
    std::uint32_t *col_row_global_helper)
{
    static_assert(swarp_size <= WARP_SIZE,
                  "Subwarp size must be smaller than the WARP_SIZE");
    static_assert((swarp_size & (swarp_size - 1)) == 0,
                  "swarp_size must be a power of 2");
    static_assert(swarp_size % (swarps_per_block) == 0,
                  "swarp_size must be a multiple of swarps_per_block");
    // assert: blockDim.x == swarp_size; blockDim.y = swarps_per_block;
    //         blockDim.z = 1
    using index_type = std::int64_t;
    constexpr int triang_stride = swarp_size + 1;

    // stores the trianglular system in column major
    __shared__ ValueType triang[swarp_size * triang_stride];
    __shared__ std::uint32_t shared_row_block_idx[1];
    __shared__ ValueType x_correction[swarp_size];

    const auto group = cg::this_thread_block();
    const auto swarp = cg::tiled_partition<swarp_size>(group);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        *shared_row_block_idx =
            atomicInc(col_row_global_helper + 1, ~std::uint32_t{0});
    }
    group.sync();
    const index_type row_block_idx = *shared_row_block_idx;

    if (row_block_idx * swarp_size >= m_info.size[0]) {
        return;
    }

    // All threads: load triangular matrix into shared memory
    // Note: Read it coalesced and transpose it.
    //       L is stored in column major for fast updates
    for (index_type row = threadIdx.y; row < swarp_size;
         row += swarps_per_block) {
        // threadIdx.x stores the column here to read coalesced
        const index_type col = threadIdx.x;
        const index_type global_row = row_block_idx * swarp_size + row;
        const index_type global_col = row_block_idx * swarp_size + col;
        triang[col * triang_stride + row] =
            (dmtx == dmtx_t::unit && col == row) ? ValueType{1}
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
                const auto factor =
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

    constexpr index_type num_local_rows = swarp_size / swarps_per_block;
    ValueType local_row_result[num_local_rows] = {};
    for (index_type col_block = 0; col_block < row_block_idx; ++col_block) {
        const index_type global_col = col_block * swarp_size + threadIdx.x;

        // Wait until result is known for current column block
        // Maybe add __nanosleep(200) to ensure it is yielded
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            // Note: this needs to be signed since the initial value is
            //       ~0 (all ones)
            volatile auto *last_col = reinterpret_cast<std::make_signed_t<
                std::remove_pointer_t<decltype(col_row_global_helper)>> *>(
                col_row_global_helper);
            while (*last_col < col_block) {
            }
        }
        group.sync();
        // Make sure that data is only read after synchronization
        __threadfence();

        const auto x_cached = x[global_col * x_info.stride];
        const index_type end_local_row_idx =
            (m_info.size[0] - 1 - row_block_idx * swarp_size - threadIdx.y) /
            swarps_per_block;
#pragma unroll
        for (index_type local_row_idx = 0; local_row_idx < num_local_rows;
             ++local_row_idx) {
            const index_type global_row = row_block_idx * swarp_size +
                                          threadIdx.y +
                                          local_row_idx * swarps_per_block;
            // Bound check necessary, we could be at the bottom of the matrix
            if (local_row_idx <= end_local_row_idx) {
                local_row_result[local_row_idx] +=
                    x_cached * mtx[global_row * m_info.stride + global_col];
            }
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
        const index_type row = threadIdx.x;
        const index_type x_idx =
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
            col_row_global_helper[0] =
                static_cast<std::uint32_t>(row_block_idx);
        }
    }
}

// Must be called with 2-D blocks and a 1-D grid
template <std::int32_t swarp_size, std::int32_t swarps_per_block, dmtx_t dmtx,
          typename ValueType>
__global__ __launch_bounds__(swarps_per_block *swarp_size) void upper_trsv(
    const matrix_info m_info, const ValueType *__restrict__ mtx,
    const matrix_info x_info, ValueType *__restrict__ x,
    std::uint32_t *col_row_global_helper)
{
    static_assert(swarp_size <= WARP_SIZE,
                  "Subwarp size must be smaller than the WARP_SIZE");
    static_assert((swarp_size & (swarp_size - 1)) == 0,
                  "swarp_size must be a power of 2");
    static_assert(swarp_size % (swarps_per_block) == 0,
                  "swarp_size must be a multiple of swarps_per_block");
    // assert: blockDim.x == swarp_size; blockDim.y = swarps_per_block;
    //         blockDim.z = 1
    using index_type = std::int64_t;
    constexpr int triang_stride = swarp_size + 1;

    // stores the trianglular system in column major
    __shared__ ValueType triang[swarp_size * triang_stride];
    __shared__ std::uint32_t shared_row_block_idx[1];
    __shared__ ValueType x_correction[swarp_size];

    const auto group = cg::this_thread_block();
    const auto swarp = cg::tiled_partition<swarp_size>(group);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        *shared_row_block_idx =
            atomicInc(col_row_global_helper + 1, ~std::uint32_t{0});
    }
    group.sync();
    const index_type row_block_idx = *shared_row_block_idx;

    if (row_block_idx * swarp_size >= m_info.size[0]) {
        return;
    }

    // All threads: load triangular matrix into shared memory
    // Note: Read it coalesced and transpose it.
    //       U is stored in column major for fast updates
    for (index_type row = threadIdx.y; row < swarp_size;
         row += swarps_per_block) {
        // threadIdx.x stores the column here to read coalesced
        const index_type col = threadIdx.x;
        const index_type global_row =
            m_info.size[0] - (row_block_idx + 1) * swarp_size + row;
        const index_type global_col =
            m_info.size[1] - (row_block_idx + 1) * swarp_size + col;
        triang[col * triang_stride + row] =
            (dmtx == dmtx_t::unit && col == row) ? ValueType{1}
            : (row <= col && 0 <= global_row && 0 <= global_col)
                ? mtx[global_row * m_info.stride + global_col]
                : ValueType{0};
    }
    group.sync();
    // Invert lower triangular matrix

    // First: make diagonals 1
    if (dmtx == dmtx_t::non_unit) {
        if (threadIdx.y == 0) {
            for (int row = swarp_size; row >= 0; --row) {
                const int col = threadIdx.x;
                const auto triang_idx = col * triang_stride + row;
                const auto local_val = triang[triang_idx];
                const auto diag_el = ValueType{1} / swarp.shfl(local_val, row);
                if (row <= col) {
                    triang[triang_idx] =
                        (row == col) ? diag_el : local_val * diag_el;
                }
            }
        }
    }
    // No need to sync since the same warp will continue to compute

    if (threadIdx.y == 0) {
        // Compute remaining inverse in-place (with Gauss-Jordan elimination)
        for (int diag_id = swarp_size - 1; diag_id >= 0; --diag_id) {
            const int col = threadIdx.x;
            const auto diag_el = triang[diag_id * triang_stride + diag_id];
            for (int row = diag_id - 1; row >= 0; --row) {
                // broadcast
                const auto factor = -triang[diag_id * triang_stride + row];
                const int triang_idx = col * triang_stride + row;
                const auto triang_val = triang[triang_idx];
                if (row < col) {
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

    constexpr index_type num_local_rows = swarp_size / swarps_per_block;
    ValueType local_row_result[num_local_rows] = {};
    for (index_type col_block = 1; col_block <= row_block_idx; ++col_block) {
        const index_type global_col =
            m_info.size[1] - (col_block * swarp_size) + threadIdx.x;

        // Wait until result is known for current column block
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            // Note: this needs to be signed since the initial value is
            //       ~0 (all ones)
            volatile auto *last_col = reinterpret_cast<std::make_signed_t<
                std::remove_pointer_t<decltype(col_row_global_helper)>> *>(
                col_row_global_helper);
            while (*last_col < col_block - 1) {
            }
        }
        group.sync();
        // Make sure that data is only read after synchronization
        __threadfence();

        const auto x_cached = x[global_col * x_info.stride];
        const index_type start_local_row_idx =
            ceildiv(-static_cast<index_type>(m_info.size[0]) +
                        (row_block_idx + 1) * swarp_size - threadIdx.y,
                    static_cast<index_type>(swarps_per_block));
#pragma unroll
        for (index_type local_row_idx = 0; local_row_idx < num_local_rows;
             ++local_row_idx) {
            const index_type global_row =
                m_info.size[0] - (row_block_idx + 1) * swarp_size +
                threadIdx.y + local_row_idx * swarps_per_block;
            // Bound check necessary, we could be at the top of the matrix
            if (start_local_row_idx <= local_row_idx) {
                local_row_result[local_row_idx] +=
                    x_cached * mtx[global_row * m_info.stride + global_col];
            }
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
        const index_type row = threadIdx.x;
        const index_type x_idx =
            (x_info.size[0] - (row_block_idx + 1) * swarp_size + row) *
            x_info.stride;
        // compute the local x and distribute it via shfl
        const ValueType local_x =
            (0 <= x_idx) ? x[x_idx] - x_correction[row] : ValueType{0};
        ValueType local_solution{};
        for (int col = 0; col < swarp_size; ++col) {
            // GEMV
            local_solution +=
                triang[col * triang_stride + row] * swarp.shfl(local_x, col);
        }
        if (0 <= x_idx) {
            x[x_idx] = local_solution;
        }
        __threadfence();
        if (threadIdx.x == 0) {
            col_row_global_helper[0] =
                static_cast<std::uint32_t>(row_block_idx);
        }
    }
}


}  // namespace kernel


template <typename ValueType>
void trsv(const matrix_info m_info, tmtx_t ttype, dmtx_t dtype,
            const ValueType *mtx, const matrix_info x_info, ValueType *x,
            std::uint32_t *trsv_helper)
{
    constexpr std::int32_t subwarp_size{kernel::WARP_SIZE};
    constexpr std::int32_t swarps_per_block{4};
    const dim3 block_solve(subwarp_size, swarps_per_block, 1);
    const dim3 grid_solve(
        ceildiv(m_info.size[0], static_cast<std::size_t>(subwarp_size)), 1, 1);

    kernel::trsv_init<<<1, 1>>>(trsv_helper);
    if (dtype == dmtx_t::unit) {
        if (ttype == tmtx_t::lower) {
            kernel::lower_trsv<subwarp_size, swarps_per_block, dmtx_t::unit>
                <<<grid_solve, block_solve>>>(m_info, mtx, x_info, x,
                                              trsv_helper);
        } else {
            kernel::upper_trsv<subwarp_size, swarps_per_block, dmtx_t::unit>
                <<<grid_solve, block_solve>>>(m_info, mtx, x_info, x,
                                              trsv_helper);
        }
    } else {
        if (ttype == tmtx_t::lower) {
            kernel::lower_trsv<subwarp_size, swarps_per_block,
                                 dmtx_t::non_unit><<<grid_solve, block_solve>>>(
                m_info, mtx, x_info, x, trsv_helper);
        } else {
            kernel::upper_trsv<subwarp_size, swarps_per_block,
                                 dmtx_t::non_unit><<<grid_solve, block_solve>>>(
                m_info, mtx, x_info, x, trsv_helper);
        }
    }
}


namespace kernel {


// Implementation follows paper: A Fast Dense Triangular Solve in CUDA
//                               https://doi.org/10.1137/12088358X

// Must be called with 2-D blocks and a 1-D grid
template <std::int32_t swarp_size, std::int32_t swarps_per_block, dmtx_t dmtx,
          typename MtxAccessor, typename VecAccessor>
__global__ __launch_bounds__(swarps_per_block *swarp_size) void acc_lower_trsv(
    gko::acc::range<MtxAccessor> mtx, gko::acc::range<VecAccessor> x,
    std::uint32_t *col_row_global_helper)
{
    static_assert(swarp_size <= WARP_SIZE,
                  "Subwarp size must be smaller than the WARP_SIZE");
    static_assert((swarp_size & (swarp_size - 1)) == 0,
                  "swarp_size must be a power of 2");
    static_assert(swarp_size % (swarps_per_block) == 0,
                  "swarp_size must be a multiple of swarps_per_block");
    // assert: blockDim.x == swarp_size; blockDim.y = swarps_per_block;
    //         blockDim.z = 1
    using ar_type = typename MtxAccessor::arithmetic_type;
    using index_type = std::int64_t;
    constexpr int triang_stride = swarp_size + 1;

    // stores the trianglular system in column major
    __shared__ ar_type triang[swarp_size * triang_stride];
    __shared__ std::uint32_t shared_row_block_idx[1];
    __shared__ ar_type x_correction[swarp_size];

    const auto group = cg::this_thread_block();
    const auto swarp = cg::tiled_partition<swarp_size>(group);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        *shared_row_block_idx =
            atomicInc(col_row_global_helper + 1, ~std::uint32_t{0});
    }
    group.sync();
    const index_type row_block_idx = *shared_row_block_idx;

    if (row_block_idx * swarp_size >= mtx.length(0)) {
        return;
    }

    // All threads: load triangular matrix into shared memory
    // Note: Read it coalesced and transpose it.
    //       L is stored in column major for fast updates
    for (index_type row = threadIdx.y; row < swarp_size;
         row += swarps_per_block) {
        // threadIdx.x stores the column here to read coalesced
        const index_type col = threadIdx.x;
        const index_type global_row = row_block_idx * swarp_size + row;
        const index_type global_col = row_block_idx * swarp_size + col;
        triang[col * triang_stride + row] =
            (dmtx == dmtx_t::unit && col == row) ? ar_type{1}
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

    constexpr index_type num_local_rows = swarp_size / swarps_per_block;
    ar_type local_row_result[num_local_rows] = {};
    for (index_type col_block = 0; col_block < row_block_idx; ++col_block) {
        const index_type global_col = col_block * swarp_size + threadIdx.x;

        // Wait until result is known for current column block
        // Maybe add __nanosleep(200) to ensure it is yielded
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            // Note: this needs to be signed since the initial value is
            //       ~0 (all ones)
            volatile auto *last_col = reinterpret_cast<std::make_signed_t<
                std::remove_pointer_t<decltype(col_row_global_helper)>> *>(
                col_row_global_helper);
            while (*last_col < col_block) {
            }
        }
        group.sync();
        // Make sure that data is only read after synchronization
        __threadfence();
        const auto x_cached = x(global_col, 0);
        const index_type end_local_row_idx =
            (mtx.length(0) - 1 - row_block_idx * swarp_size - threadIdx.y) /
            swarps_per_block;
#pragma unroll
        for (index_type local_row_idx = 0; local_row_idx < num_local_rows;
             ++local_row_idx) {
            // Bound check necessary, we could be at the bottom of the matrix
            if (local_row_idx <= end_local_row_idx) {
                local_row_result[local_row_idx] +=
                    x_cached * mtx(row_block_idx * swarp_size + threadIdx.y +
                                       local_row_idx * swarps_per_block,
                                   global_col);
            }
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
        const index_type row = threadIdx.x;
        const index_type x_idx = row_block_idx * swarp_size + row;
        // compute the local x and distribute it via shfl
        const ar_type local_x = (x_idx < x.length(0))
                                    ? x(x_idx, 0) - x_correction[row]
                                    : ar_type{0};
        ar_type local_solution{};
#pragma unroll
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
            col_row_global_helper[0] =
                static_cast<std::uint32_t>(row_block_idx);
        }
    }
}

// Must be called with 2-D blocks and a 1-D grid
template <std::int32_t swarp_size, std::int32_t swarps_per_block, dmtx_t dmtx,
          typename MtxAccessor, typename VecAccessor>
__global__ __launch_bounds__(swarps_per_block *swarp_size) void acc_upper_trsv(
    gko::acc::range<MtxAccessor> mtx, gko::acc::range<VecAccessor> x,
    std::uint32_t *col_row_global_helper)
{
    static_assert(swarp_size <= WARP_SIZE,
                  "Subwarp size must be smaller than the WARP_SIZE");
    static_assert((swarp_size & (swarp_size - 1)) == 0,
                  "swarp_size must be a power of 2");
    static_assert(swarp_size % (swarps_per_block) == 0,
                  "swarp_size must be a multiple of swarps_per_block");
    // assert: blockDim.x == swarp_size; blockDim.y = swarps_per_block;
    //         blockDim.z = 1
    using ar_type = typename MtxAccessor::arithmetic_type;
    using index_type = std::int64_t;
    constexpr int triang_stride = swarp_size + 1;

    // stores the trianglular system in column major
    __shared__ ar_type triang[swarp_size * triang_stride];
    __shared__ std::uint32_t shared_row_block_idx[1];
    __shared__ ar_type x_correction[swarp_size];

    const auto group = cg::this_thread_block();
    const auto swarp = cg::tiled_partition<swarp_size>(group);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        *shared_row_block_idx =
            atomicInc(col_row_global_helper + 1, ~std::uint32_t{0});
    }
    group.sync();
    const index_type row_block_idx = *shared_row_block_idx;

    if (row_block_idx * swarp_size >= mtx.length(0)) {
        return;
    }

    // All threads: load triangular matrix into shared memory
    // Note: Read it coalesced and transpose it.
    //       U is stored in column major for fast updates
    for (index_type row = threadIdx.y; row < swarp_size;
         row += swarps_per_block) {
        // threadIdx.x stores the column here to read coalesced
        const index_type col = threadIdx.x;
        const index_type global_row =
            mtx.length(0) - (row_block_idx + 1) * swarp_size + row;
        const index_type global_col =
            mtx.length(1) - (row_block_idx + 1) * swarp_size + col;
        triang[col * triang_stride + row] =
            (dmtx == dmtx_t::unit && col == row) ? ar_type{1}
            : (row <= col && 0 <= global_row && 0 <= global_col)
                ? mtx(global_row, global_col)
                : ar_type{0};
    }
    group.sync();
    // Invert lower triangular matrix

    // First: make diagonals 1
    if (dmtx == dmtx_t::non_unit) {
        if (threadIdx.y == 0) {
            for (int row = swarp_size; row >= 0; --row) {
                const int col = threadIdx.x;
                const auto triang_idx = col * triang_stride + row;
                const auto local_val = triang[triang_idx];
                const auto diag_el = ar_type{1} / swarp.shfl(local_val, row);
                if (row <= col) {
                    triang[triang_idx] =
                        (row == col) ? diag_el : local_val * diag_el;
                }
            }
        }
    }
    // No need to sync since the same warp will continue to compute

    if (threadIdx.y == 0) {
        // Compute remaining inverse in-place (with Gauss-Jordan elimination)
        for (int diag_id = swarp_size - 1; diag_id >= 0; --diag_id) {
            const int col = threadIdx.x;
            const auto diag_el = triang[diag_id * triang_stride + diag_id];
            for (int row = diag_id - 1; row >= 0; --row) {
                // broadcast
                const auto factor = -triang[diag_id * triang_stride + row];
                const int triang_idx = col * triang_stride + row;
                const auto triang_val = triang[triang_idx];
                if (row < col) {
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

    constexpr index_type num_local_rows = swarp_size / swarps_per_block;
    ar_type local_row_result[num_local_rows] = {};
    for (index_type col_block = 1; col_block <= row_block_idx; ++col_block) {
        const index_type global_col =
            mtx.length(1) - (col_block * swarp_size) + threadIdx.x;

        // Wait until result is known for current column block
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            // Note: this needs to be signed since the initial value is
            //       ~0 (all ones)
            volatile auto *last_col = reinterpret_cast<std::make_signed_t<
                std::remove_pointer_t<decltype(col_row_global_helper)>> *>(
                col_row_global_helper);
            while (*last_col < col_block - 1) {
            }
        }
        group.sync();
        // Make sure that data is only read after synchronization
        __threadfence();

        const ar_type x_cached = x(global_col, 0);
        const index_type start_local_row_idx =
            ceildiv(-static_cast<index_type>(mtx.length(0)) +
                        (row_block_idx + 1) * swarp_size - threadIdx.y,
                    static_cast<index_type>(swarps_per_block));
#pragma unroll
        for (index_type local_row_idx = 0; local_row_idx < num_local_rows;
             ++local_row_idx) {
            const index_type global_row =
                mtx.length(0) - (row_block_idx + 1) * swarp_size + threadIdx.y +
                local_row_idx * swarps_per_block;
            // Bound check necessary, we could be at the top of the matrix
            if (start_local_row_idx <= local_row_idx) {
                local_row_result[local_row_idx] +=
                    x_cached * mtx(global_row, global_col);
            }
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
        const index_type row = threadIdx.x;
        const index_type x_idx =
            (x.length(0) - (row_block_idx + 1) * swarp_size + row);
        // compute the local x and distribute it via shfl
        const ar_type local_x =
            (0 <= x_idx) ? x(x_idx, 0) - x_correction[row] : ar_type{0};
        ar_type local_solution{};
        for (int col = 0; col < swarp_size; ++col) {
            // GEMV
            local_solution +=
                triang[col * triang_stride + row] * swarp.shfl(local_x, col);
        }
        if (0 <= x_idx) {
            x(x_idx, 0) = local_solution;
        }
        __threadfence();
        if (threadIdx.x == 0) {
            col_row_global_helper[0] =
                static_cast<std::uint32_t>(row_block_idx);
        }
    }
}


}  // namespace kernel


template <typename ArType, typename StType>
void acc_trsv(const matrix_info m_info, tmtx_t ttype, dmtx_t dtype,
              const StType *mtx, const matrix_info x_info, StType *x,
              std::uint32_t *trsv_helper)
{
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
        if (ttype == tmtx_t::lower) {
            kernel::acc_lower_trsv<subwarp_size, swarps_per_block, dmtx_t::unit>
                <<<grid_solve, block_solve>>>(m_acc, x_acc, trsv_helper);
        } else {
            kernel::acc_upper_trsv<subwarp_size, swarps_per_block, dmtx_t::unit>
                <<<grid_solve, block_solve>>>(m_acc, x_acc, trsv_helper);
        }
    } else {
        if (ttype == tmtx_t::lower) {
            kernel::acc_lower_trsv<subwarp_size, swarps_per_block,
                                   dmtx_t::non_unit>
                <<<grid_solve, block_solve>>>(m_acc, x_acc, trsv_helper);
        } else {
            kernel::acc_upper_trsv<subwarp_size, swarps_per_block,
                                   dmtx_t::non_unit>
                <<<grid_solve, block_solve>>>(m_acc, x_acc, trsv_helper);
        }
    }
}

#define BIND_CUBLAS_TRSV(ValueType, CublasName)                                \
    void cublas_trsv(cublasHandle_t handle, cublasFillMode_t uplo,             \
                     cublasOperation_t trans, cublasDiagType_t dig, int n,     \
                     const ValueType *A, int lda, ValueType *x, int incx)      \
    {                                                                          \
        CUBLAS_CALL(CublasName(handle, uplo, trans, dig, n, A, lda, x, incx)); \
    }
BIND_CUBLAS_TRSV(double, cublasDtrsv)
BIND_CUBLAS_TRSV(float, cublasStrsv)
#undef BIND_CUBLAS_TRSV

template <typename ValueType>
void cublas_trsv(cublasHandle_t handle, tmtx_t ttype, dmtx_t dtype,
                 const matrix_info m_info, const ValueType *mtx,
                 const matrix_info x_info, ValueType *x)
{
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


namespace kernel {


template <typename InType, typename OutType>
__global__ __launch_bounds__(256) void copy_vector(
    std::int64_t N, const InType *__restrict__ in, std::int64_t in_stride,
    OutType *__restrict__ out, std::int64_t out_stride)
{
    const std::int64_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < N) {
        out[tidx * out_stride] = static_cast<OutType>(in[tidx * in_stride]);
    }
}

// computes element wise: inout += in;
template <typename InType, typename InOutType>
__global__ __launch_bounds__(256) void update_vector(
    std::int64_t N, const InType *__restrict__ in, std::int64_t in_stride,
    InOutType *__restrict__ inout, std::int64_t inout_stride)
{
    const std::int64_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < N) {
        inout[tidx * inout_stride] +=
            static_cast<InOutType>(in[tidx * in_stride]);
    }
}


}  // namespace kernel


template <typename InType, typename OutType>
void copy_vector(const InType *in, const matrix_info i_info, OutType *out,
            const matrix_info o_info)
{
    if (i_info.size[1] != o_info.size[1] || i_info.size[0] != o_info.size[0] ||
        i_info.size[1] != 1) {
        throw "Not supported!";
    }
    const dim3 block(256);
    const dim3 grid(ceildiv(i_info.size[0], static_cast<std::size_t>(256)), 1, 1);
    kernel::copy_vector<<<grid, block>>>(i_info.size[0], in, i_info.stride, out,
                                         o_info.stride);
}


template <typename InType, typename InOutType>
void update_vector(const InType *in, const matrix_info i_info, InOutType *inout,
              const matrix_info io_info)
{
    if (i_info.size[1] != io_info.size[1] || i_info.size[0] != io_info.size[0] ||
        i_info.size[1] != 1) {
        throw "Not supported!";
    }
    const dim3 block(256);
    const dim3 grid(ceildiv(i_info.size[0], static_cast<std::size_t>(256)), 1, 1);
    kernel::update_vector<<<grid, block>>>(i_info.size[0], in, i_info.stride,
                                           inout, io_info.stride);
}
