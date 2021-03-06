#pragma once

#include <cinttypes>


#include <cooperative_groups.h>
#include <cublas_v2.h>


// Accessor headers
#include <accessor/range.hpp>
#include <accessor/reduced_row_major.hpp>


#include "kernel_utils.cuh"
#include "utils.cuh"


namespace kernel {


namespace cg = cooperative_groups;


/**
 * Computes the GEMV: res = alpha * mtx * x + beta * res
 *
 * @note This function expects mtx to be in row-major format
 */
template <std::int64_t block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void gemv(
    const matrix_info m_info, ValueType alpha, const ValueType *__restrict__ mtx,
    const matrix_info x_info, const ValueType *__restrict__ x,
    const matrix_info res_info, ValueType beta, ValueType *__restrict__ res)
{
    // expect x_info.size[1] == 1
    const std::int64_t row_idx{blockIdx.x};
    if (row_idx >= m_info.size[0]) {
        return;
    }
    const std::int64_t row_start = row_idx * m_info.stride;
    __shared__ char shared_impl[sizeof(ValueType) * block_size];
    auto shared = reinterpret_cast<ValueType *>(shared_impl);

    ValueType local_result{};
    const auto group = cg::this_thread_block();
    const auto local_id = group.thread_rank();

    for (std::int64_t col = local_id; col < m_info.size[1]; col += block_size) {
        const auto mtx_val = mtx[row_start + col];
        const auto x_val = x[col * x_info.stride];
        local_result += mtx_val * x_val;
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


/**
 * Computes the GEMV: res = alpha * mtx * x + beta * res
 *
 * @note This function only performs well if mtx is in row-major format
 *       (otherwise, the memory operations are not coalesced)
 *
 * @internal The main difference to the non-accessor GEMV implementation is that
 *           the information how data is accessed is now stored in the accessor
 *           and not as a separate parameter. Other than that, only the read and
 *           write accesses are different, as they now go through the accessor
 *           instead of being hand-computed.
 */
template <std::int64_t block_size, typename MtxRange, typename XRange,
          typename ResRange, typename ArType>
__global__ __launch_bounds__(block_size) void acc_gemv(ArType alpha,
                                                       MtxRange mtx, XRange x,
                                                       ArType beta,
                                                       ResRange res)
{
    using ar_type = decltype(alpha * mtx(0, 0) * x(0, 0) + beta * res(0, 0));
    static_assert(std::is_same<ArType, ar_type>::value, "Types must be equal!");
    // expect x_info.size[1] == 1
    const std::int64_t row_idx{blockIdx.x};
    if (row_idx >= mtx.length(0)) {
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


/**
 * Computes the GEMV: res = alpha * mtx * x + beta * res
 * using a self-implemented kernel without the accessor.
 *
 * @note This function expects mtx to be in row-major format
 *
 * @tparam ValueType  type of the input and output parameters
 *
 * @param m_info  Information about the matrix
 * @param alpha  alpha factor for the GEMV
 * @param mtx  matrix
 * @param x_info  Information about the x vector
 * @param x  x vector
 * @param res_info  Information about the res vector
 * @param beta  beta factor for the GEMV
 * @param res  res vector
 **/
template <typename ValueType>
void gemv(const matrix_info m_info, ValueType alpha, const ValueType *mtx,
          const matrix_info x_info, const ValueType *x,
          const matrix_info res_info, ValueType beta, ValueType *res)
{
    constexpr std::int32_t block_size{512};
    const dim3 block(block_size, 1, 1);
    const dim3 grid(m_info.size[0], 1, 1);

    kernel::gemv<block_size, ValueType>
        <<<grid, block>>>(m_info, alpha, mtx, x_info, x, res_info, beta, res);
}


/**
 * Computes the GEMV: res = alpha * mtx * x + beta * res
 * using a kernel utilizing the accessor.
 *
 * @note This function expects mtx to be in row-major format
 *
 * @tparam ArType  arithmetic type that should be used in the GEMV
 * @tparam StType  storage type of the GEMV
 *
 * @param m_info  Information about the matrix
 * @param alpha  alpha factor for the GEMV
 * @param mtx  matrix
 * @param x_info  Information about the x vector
 * @param x  x vector
 * @param res_info  Information about the res vector
 * @param beta  beta factor for the GEMV
 * @param res  res vector
 **/
template <typename ArType, typename StType>
void acc_gemv(const matrix_info m_info, ArType alpha, const StType *mtx,
              const matrix_info x_info, const StType *x,
              const matrix_info res_info, ArType beta, StType *res)
{
    constexpr std::int32_t block_size{512};
    const dim3 block(block_size, 1, 1);
    const dim3 grid(m_info.size[0], 1, 1);

    // Accessor Setup
    constexpr std::size_t dimensionality{2};
    std::array<gko::acc::size_type, dimensionality - 1> m_stride{m_info.stride};
    std::array<gko::acc::size_type, dimensionality - 1> x_stride{x_info.stride};
    std::array<gko::acc::size_type, dimensionality - 1> res_stride{res_info.stride};

    using accessor =
        gko::acc::reduced_row_major<dimensionality, ArType, StType>;
    using range = gko::acc::range<accessor>;
    using c_range = gko::acc::range<typename accessor::const_accessor>;
    auto m_acc = c_range(m_info.size, mtx, m_stride);
    auto x_acc = c_range(x_info.size, x, x_stride);
    auto res_acc = range(res_info.size, res, res_stride);

    kernel::acc_gemv<block_size>
        <<<grid, block>>>(alpha, m_acc, x_acc, beta, res_acc);
}


// Use a macro to overload the CUBLAS GEMV calls instead of hand-writing them.
// Also allows for easier extension (for example for complex types).
#define BIND_CUBLAS_GEMV(ValueType, CublasName)                              \
    void cublas_gemv(cublasHandle_t handle, cublasOperation_t transa, int m, \
                     int n, const ValueType *alpha, const ValueType *A,      \
                     int lda, const ValueType *x, int incx,                  \
                     const ValueType *beta, ValueType *y, int incy)          \
    {                                                                        \
        CUBLAS_CALL(CublasName(handle, transa, m, n, alpha, A, lda, x, incx, \
                               beta, y, incy));                              \
    }
BIND_CUBLAS_GEMV(double, cublasDgemv)
BIND_CUBLAS_GEMV(float, cublasSgemv)
#undef BIND_CUBLAS_GEMV


/**
 * Computes the GEMV: res = alpha * mtx * x + beta * res
 * using the CUBLAS vendor implementation
 *
 * @note This function expects mtx to be in row-major format
 *
 * @tparam ValueType  type of the input and output parameters
 *
 * @param haldle  CUBLAS handle which is required for CUBLAS operations
 * @param m_info  Information about the matrix
 * @param alpha  alpha factor for the GEMV
 * @param mtx  matrix
 * @param x_info  Information about the x vector
 * @param x  x vector
 * @param res_info  Information about the res vector
 * @param beta  beta factor for the GEMV
 * @param res  res vector
 **/
template <typename ValueType>
void cublas_gemv(cublasHandle_t handle, const matrix_info m_info,
                 ValueType alpha, const ValueType *mtx,
                 const matrix_info x_info, const ValueType *x,
                 const matrix_info res_info, ValueType beta, ValueType *y)
{
    // Note: CUBLAS expects the matrices to be stored in column major,
    //       so the sizes will be transposed for the cublas call
    cublas_gemv(handle, CUBLAS_OP_T, static_cast<int>(m_info.size[0]),
                static_cast<int>(m_info.size[1]), &alpha, mtx,
                static_cast<int>(m_info.stride), x,
                static_cast<int>(x_info.stride), &beta, y,
                static_cast<int>(res_info.stride));
}
