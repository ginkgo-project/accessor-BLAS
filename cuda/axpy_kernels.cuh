#pragma once

#include <cinttypes>


#include <cooperative_groups.h>
#include <cublas_v2.h>


// Accessor headers
#include "accessor/range.hpp"
#include "accessor/reduced_row_major.hpp"


#include "kernel_utils.cuh"
#include "utils.cuh"


constexpr int axpy_blocks_per_sm{32};
constexpr int axpy_block_size{512};


namespace kernel {


namespace cg = cooperative_groups;


/**
 * Computes the AXPY: res = alpha * x + res
 */
template <std::int64_t block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void axpy(
    ValueType alpha, const matrix_info x_info, const ValueType *__restrict__ x,
    const matrix_info res_info, ValueType *__restrict__ res)
{
    // expect x_info.size[1] == 1
    const std::uint64_t global_tidx = blockIdx.x * block_size + threadIdx.x;
    const std::uint64_t grid_stride = block_size * gridDim.x;

    for (std::uint64_t i = global_tidx; i < x_info.size[0]; i += grid_stride) {
        const auto x_idx = i * x_info.stride;
        const auto res_idx = i * res_info.stride;

        res[res_idx] = alpha * x[x_idx] + res[res_idx];
    }
}

/**
 * Computes the AXPY: res = alpha * x + res
 *
 * @internal The main difference to the non-accessor AXPY implementation is that
 *           the information how data is accessed is now stored in the accessor
 *           and not as a separate parameter. Other than that, only the read and
 *           write accesses are different, as they now go through the accessor
 *           instead of being hand-computed.
 */
template <std::int64_t block_size, typename XRange, typename ResRange,
          typename ArType>
__global__ __launch_bounds__(block_size) void acc_axpy(ArType alpha, XRange x,
                                                       ResRange res)
{
    using ar_type = decltype(alpha * x(0, 0) + res(0, 0));
    static_assert(std::is_same<ArType, ar_type>::value, "Types must be equal!");
    // expect x_info.size[1] == 1
    const std::uint64_t global_tidx = blockIdx.x * block_size + threadIdx.x;
    const std::uint64_t grid_stride = block_size * gridDim.x;

    for (std::uint64_t i = global_tidx; i < x.length(0); i += grid_stride) {
        res(i, 0) = alpha * x(i, 0) + res(i, 0);
    }
}
}  // namespace kernel


/**
 * Computes the AXPY: res = alpha * x + res
 * using a self-implemented kernel without the accessor.
 *
 * @tparam ValueType  type of the input and output parameters
 *
 * @param alpha  alpha factor for the AXPY
 * @param x_info  Information about the x vector
 * @param x  x vector
 * @param res_info  Information about the res vector
 * @param res  res vector
 **/
template <typename ValueType>
void axpy(myBlasHandle *handle, ValueType alpha, const matrix_info x_info,
          const ValueType *x, const matrix_info res_info, ValueType *res)
{
    constexpr std::int32_t block_size{axpy_block_size};
    const dim3 block(block_size, 1, 1);
    const dim3 grid(
        handle->get_device_property().multiProcessorCount * axpy_blocks_per_sm,
        1, 1);

    kernel::axpy<block_size, ValueType>
        <<<grid, block>>>(alpha, x_info, x, res_info, res);
}

/**
 * Computes the AXPY: res = alpha * x + res
 * using a kernel utilizing the accessor.
 *
 * @tparam ArType  arithmetic type that should be used in the AXPY
 * @tparam StType  storage type of the AXPY
 *
 * @param alpha  alpha factor for the AXPY
 * @param x_info  Information about the x vector
 * @param x  x vector
 * @param res_info  Information about the res vector
 * @param res  res vector
 **/
template <typename ArType, typename StType>
void acc_axpy(myBlasHandle *handle, ArType alpha, const matrix_info x_info,
              const StType *x, const matrix_info res_info, StType *res)
{
    constexpr std::int32_t block_size{axpy_block_size};
    const dim3 block(block_size, 1, 1);
    const dim3 grid(
        handle->get_device_property().multiProcessorCount * axpy_blocks_per_sm,
        1, 1);

    // Accessor Setup
    constexpr std::size_t dimensionality{2};
    std::array<gko::acc::size_type, dimensionality - 1> x_stride{x_info.stride};
    std::array<gko::acc::size_type, dimensionality - 1> res_stride{
        res_info.stride};

    using accessor =
        gko::acc::reduced_row_major<dimensionality, ArType, StType>;
    using range = gko::acc::range<accessor>;
    using c_range = gko::acc::range<typename accessor::const_accessor>;
    auto x_acc = c_range(x_info.size, x, x_stride);
    auto res_acc = range(res_info.size, res, res_stride);

    kernel::acc_axpy<block_size><<<grid, block>>>(alpha, x_acc, res_acc);
}

// Use a macro to overload the CUBLAS AXPY calls instead of hand-writing them.
// Also allows for easier extension (for example for complex types).
#define BIND_CUBLAS_AXPY(ValueType, CublasName)                            \
    void cublas_axpy(cublasHandle_t handle, int n, const ValueType *alpha, \
                     const ValueType *x, int incx, ValueType *y, int incy) \
    {                                                                      \
        CUBLAS_CALL(CublasName(handle, n, alpha, x, incx, y, incy));       \
    }
BIND_CUBLAS_AXPY(double, cublasDaxpy)
BIND_CUBLAS_AXPY(float, cublasSaxpy)
#undef BIND_CUBLAS_AXPY


/**
 * Computes the AXPY: res = alpha * x + res
 * using the CUBLAS vendor implementation
 *
 * @tparam ValueType  type of the input and output parameters
 *
 * @param haldle  CUBLAS handle which is required for CUBLAS operations
 * @param alpha  alpha factor for the AXPY
 * @param x_info  Information about the x vector
 * @param x  x vector
 * @param res_info  Information about the res vector
 * @param res  res vector
 **/
template <typename ValueType>
void cublas_axpy(cublasHandle_t handle, ValueType alpha,
                 const matrix_info x_info, const ValueType *x,
                 const matrix_info res_info, ValueType *res)
{
    cublas_axpy(handle, static_cast<int>(x_info.size[0]), &alpha, x,
                static_cast<int>(x_info.stride), res,
                static_cast<int>(res_info.stride));
}
