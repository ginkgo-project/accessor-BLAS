#pragma once

#include <cooperative_groups.h>
#include <cublas_v2.h>

#include <accessor/range.hpp>
#include <accessor/reduced_row_major.hpp>
#include <cinttypes>

#include "utils.cuh"

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

template <std::int64_t block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void gemv(
    const matrix_info minfo, ValueType alpha, const ValueType *__restrict__ mtx,
    const matrix_info x_info, const ValueType *__restrict__ x,
    const matrix_info res_info, ValueType beta, ValueType *__restrict__ res) {
    // expect x_info.size[1] == 1
    const std::int64_t row_idx{blockIdx.x};
    if (row_idx > minfo.size[0]) {
        return;
    }
    const std::int64_t row_start = row_idx * minfo.stride;
    //__shared__ ValueType shared[block_size];
    __shared__ char shared_impl[sizeof(ValueType) * block_size];
    auto shared = reinterpret_cast<ValueType *>(shared_impl);
    ValueType local_result{};
    const auto group = cg::this_thread_block();
    const auto local_id = group.thread_rank();

    for (std::int64_t col = local_id; col < minfo.size[1]; col += block_size) {
        const auto mtx_val = mtx[row_start + col];
        const auto b_val = x[col * x_info.stride];
        local_result += mtx_val * b_val;
    }
    shared[local_id] = local_result;
    reduce(group, shared, [](ValueType a, ValueType b) { return a + b; });
    if (local_id == 0) {
        const auto res_idx = row_idx * res_info.stride;
        if (beta == ValueType{0}) {
            res[res_idx] = alpha * shared[local_id];
        } else {
            res[res_idx] = alpha * shared[local_id] + beta * res[res_idx];
        }
    }
}

template <std::int64_t block_size, typename MtxRange, typename XRange,
          typename ResRange, typename ArType>
__global__ __launch_bounds__(block_size) void acc_gemv(ArType alpha,
                                                       MtxRange mtx, XRange x,
                                                       ArType beta,
                                                       ResRange res) {
    using ar_type = decltype(mtx(0, 0) + mtx(0, 0));
    static_assert(std::is_same<ArType, ar_type>::value, "Types must be equal!");
    // static_assert(std::is_same<ar_type, double>::value, "Type must be
    // double!!!");
    // expect x_info.size[1] == 1
    const std::int64_t row_idx{blockIdx.x};
    if (row_idx > mtx.length(0)) {
        return;
    }

    const auto num_cols = mtx.length(1);
    __shared__ char shared_impl[sizeof(ar_type) * block_size];
    auto shared = reinterpret_cast<ar_type *>(shared_impl);
    ar_type local_result{};
    const auto group = cg::this_thread_block();
    const auto local_id = group.thread_rank();

    for (std::int64_t col = local_id; col < num_cols; col += block_size) {
        local_result += mtx(row_idx, col) * x(col, 0);
    }
    shared[local_id] = local_result;
    reduce(group, shared, [](ar_type a, ar_type b) { return a + b; });
    if (local_id == 0) {
        if (beta == ArType{0}) {
            res(row_idx, 0) = alpha * shared[local_id];
        } else {
            res(row_idx, 0) = alpha * shared[local_id] + beta * res(row_idx, 0);
        }
    }
}

}  // namespace kernel

template <typename ValueType>
ValueType ceildiv(ValueType a, ValueType b) {
    return (a - 1) / b + 1;
}

template <typename ValueType>
void gemv(const matrix_info minfo, ValueType alpha, const ValueType *mtx,
          const matrix_info x_info, const ValueType *x,
          const matrix_info res_info, ValueType beta, ValueType *res) {
    constexpr std::int32_t block_size{512};
    const dim3 block(block_size, 1, 1);
    // const dim3 grid(ceildiv<std::int32_t>(minfo.size[0], block_size), 1, 1);
    const dim3 grid(minfo.size[0], 1, 1);

    kernel::gemv<block_size, ValueType>
        <<<grid, block>>>(minfo, alpha, mtx, x_info, x, res_info, beta, res);
}

template <typename ArType, typename StType>
void acc_gemv(const matrix_info minfo, ArType alpha, const StType *mtx,
              const matrix_info x_info, const StType *x,
              const matrix_info res_info, ArType beta, StType *res) {
    constexpr std::int32_t block_size{512};
    const dim3 block(block_size, 1, 1);
    // const dim3 grid(ceildiv<std::int32_t>(minfo.size[0], block_size), 1, 1);
    const dim3 grid(minfo.size[0], 1, 1);

    // Accessor Setup
    constexpr std::size_t dimensionality{2};
    std::array<std::size_t, dimensionality - 1> m_stride{minfo.stride};
    std::array<std::size_t, dimensionality - 1> x_stride{x_info.stride};
    std::array<std::size_t, dimensionality - 1> res_stride{res_info.stride};

    using accessor =
        gko::acc::reduced_row_major<dimensionality, ArType, StType>;
    using range = gko::acc::range<accessor>;
    using c_range = gko::acc::range<typename accessor::const_accessor>;
    auto m_acc = c_range(minfo.size, mtx, m_stride);
    auto x_acc = c_range(x_info.size, x, x_stride);
    auto res_acc = range(res_info.size, res, res_stride);

    kernel::acc_gemv<block_size>
        <<<grid, block>>>(alpha, m_acc, x_acc, beta, res_acc);
}

#define BIND_CUBLAS_GEMM(ValueType, CublasName)                                \
    void cublas_gemm(cublasHandle_t handle, cublasOperation_t transa,          \
                     cublasOperation_t transb, int m, int n, int k,            \
                     const ValueType *alpha, const ValueType *a, int lda,      \
                     const ValueType *b, int ldb, const ValueType *beta,       \
                     ValueType *c, int ldc) {                                  \
        CUBLAS_CALL(CublasName(handle, transa, transb, m, n, k, alpha, a, lda, \
                               b, ldb, beta, c, ldc));                         \
    }
BIND_CUBLAS_GEMM(double, cublasDgemm)
BIND_CUBLAS_GEMM(float, cublasSgemm)
#undef BIND_CUBLAS_GEMM

#define BIND_CUBLAS_GEMV(ValueType, CublasName)                              \
    void cublas_gemv(cublasHandle_t handle, cublasOperation_t transa, int m, \
                     int n, const ValueType *alpha, const ValueType *A,      \
                     int lda, const ValueType *x, int incx,                  \
                     const ValueType *beta, ValueType *y, int incy) {        \
        CUBLAS_CALL(CublasName(handle, transa, m, n, alpha, A, lda, x, incx, \
                               beta, y, incy));                              \
    }
BIND_CUBLAS_GEMV(double, cublasDgemv)
BIND_CUBLAS_GEMV(float, cublasSgemv)
#undef BIND_CUBLAS_GEMV

template <typename ValueType>
void cublas_gemv(cublasHandle_t handle, const matrix_info minfo,
                 ValueType alpha, const ValueType *mtx,
                 const matrix_info x_info, const ValueType *x,
                 const matrix_info res_info, ValueType beta, ValueType *y) {
    // Note: CUBLAS expects the matrices to be stored in column major,
    //       so the sizes will be transposed for the cublas call
    // auto alpha = ValueType{1};
    // auto beta = ValueType{0};
    //*
    cublas_gemv(handle, CUBLAS_OP_T, static_cast<int>(minfo.size[0]),
                static_cast<int>(minfo.size[1]), &alpha, mtx,
                static_cast<int>(minfo.stride), x,
                static_cast<int>(x_info.stride), &beta, y,
                static_cast<int>(res_info.stride));
    /*/
    cublas_gemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(x_info.size[1]),
        static_cast<int>(x_info.size[0]), static_cast<int>(minfo.size[1]),
        &alpha, x, x_info.stride, mtx, static_cast<int>(minfo.stride), &beta,
        y, static_cast<int>(res_info.stride));
    //*/
}

