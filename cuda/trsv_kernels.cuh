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

template <std::int64_t block_size, typename MtxRange, typename XRange,
          typename ResRange, typename ArType>
__global__ __launch_bounds__(block_size) void acc_trsv(ArType alpha,
                                                       MtxRange mtx, XRange x,
                                                       ArType beta,
                                                       ResRange res) {}

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

// PTX instruction nanosleep kind of yields
// See: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-nanosleep
// Note: Looks like the best number for grid is: ceildiv(n, WARP_SIZE) since:
//       n=1000, grid = 32; n=10000, grid = 313
// Paper to follow for Impl.: https://epubs.siam.org/doi/abs/10.1137/12088358X

// Must be called with 2-D blocks, and a 1-element grid
template <std::int64_t diag_block_size, dmtx_t dmtx, typename ValueType>
__global__
__launch_bounds__(diag_block_size *diag_block_size) void lower_trsv_2(
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
template <std::int64_t block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void lower_trsv_apply_2(
    const std::int64_t apply_start, const matrix_info m_info,
    const ValueType *__restrict__ mtx, const matrix_info x_info,
    const ValueType *__restrict__ solution, ValueType *__restrict__ x) {
    // assert: blockDim.x == block_size
    const std::int64_t mtx_row = apply_start + block_size + blockIdx.x;
    const std::int64_t mtx_col = apply_start + threadIdx.x;

    __shared__ ValueType local_result[block_size];
    const auto group = cg::this_thread_block();

    if (mtx_row >= m_info.size[0]) {
        return;
    }

    local_result[threadIdx.x] = (mtx_col < m_info.size[1])
                                    ? mtx[mtx_row * m_info.stride + mtx_col] *
                                          solution[threadIdx.x * x_info.stride]
                                    : ValueType{0};
    group.sync();
    reduce(group, local_result, [](ValueType a, ValueType b) { return a + b; });
    if (threadIdx.x == 0) {
        x[mtx_row * x_info.stride] -= local_result[0];
    }
}

}  // namespace kernel
template <typename ValueType>
void trsv_2(const matrix_info m_info, tmtx_t ttype, dmtx_t dtype,
            const ValueType *mtx, const matrix_info x_info, ValueType *x) {
    constexpr std::int32_t diag_block_size{kernel::WARP_SIZE};
    const dim3 block_solve(diag_block_size, diag_block_size, 1);
    // const dim3 grid(ceildiv<std::int32_t>(m_info.size[0], block_size), 1, 1);
    const dim3 grid_solve(1, 1, 1);
    auto run_solve = [&](std::int64_t block_start) {
        if (dtype == dmtx_t::unit) {
            kernel::lower_trsv_2<diag_block_size, dmtx_t::unit>
                <<<grid_solve, block_solve>>>(block_start, m_info, mtx, x_info,
                                              x);
        } else {
            kernel::lower_trsv_2<diag_block_size, dmtx_t::non_unit>
                <<<grid_solve, block_solve>>>(block_start, m_info, mtx, x_info,
                                              x);
        }
    };

    const std::int32_t block_size_apply =
        kernel::WARP_SIZE;  // * kernel::WARP_SIZE;
    const dim3 block_apply(block_size_apply, 1, 1);

    for (std::int64_t block_start = 0; block_start < m_info.size[0];
         block_start += block_size_apply) {
        run_solve(block_start);
        /*
        kernel::lower_trsv_apply_2<kernel::WARP_SIZE>
            <<<kernel::WARP_SIZE, kernel::WARP_SIZE>>>(
                block_start, m_info, mtx, x_info,
                x + block_start * x_info.stride, x);
        run_solve(block_start + kernel::WARP_SIZE);
        //*/

        const dim3 grid_apply(
            ceildiv(static_cast<std::int32_t>(m_info.size[0] - block_start),
                    1),  // kernel::WARP_SIZE),
            1, 1);

        kernel::lower_trsv_apply_2<block_size_apply>
            <<<grid_apply, block_apply>>>(block_start, m_info, mtx, x_info,
                                          x + block_start * x_info.stride, x);
    }
}

template <typename ArType, typename StType>
void acc_trsv(const matrix_info m_info, tmtx_t ttype, dmtx_t dtype,
              const StType *mtx, const matrix_info x_info, StType *x) {
    /*
    constexpr std::int32_t block_size{512};
    const dim3 block(block_size, 1, 1);
    // const dim3 grid(ceildiv<std::int32_t>(m_info.size[0], block_size), 1, 1);
    const dim3 grid(m_info.size[0], 1, 1);

    // Accessor Setup
    constexpr std::size_t dimensionality{2};
    std::array<std::size_t, dimensionality - 1> m_stride{m_info.stride};
    std::array<std::size_t, dimensionality - 1> x_stride{x_info.stride};
    std::array<std::size_t, dimensionality - 1> res_stride{res_info.stride};

    using accessor =
        gko::acc::reduced_row_major<dimensionality, ArType, StType>;
    using range = gko::acc::range<accessor>;
    using c_range = gko::acc::range<typename accessor::const_accessor>;
    auto m_acc = c_range(m_info.size, mtx, m_stride);
    auto x_acc = c_range(x_info.size, x, x_stride);
    auto res_acc = range(res_info.size, res, res_stride);

    kernel::acc_trsv<block_size>
        <<<grid, block>>>(alpha, m_acc, x_acc, beta, res_acc);
    */
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

