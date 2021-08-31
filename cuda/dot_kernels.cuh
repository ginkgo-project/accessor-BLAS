#pragma once

#include <cinttypes>


#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <accessor/range.hpp>
#include <accessor/reduced_row_major.hpp>


#include "atomics.cuh"
#include "kernel_utils.cuh"
#include "utils.cuh"


constexpr int grids_per_sm{32};
constexpr int dot_block_size{1024};


class myBlasHandle {
public:
    myBlasHandle()
    {
        CUDA_CALL(cudaGetDeviceProperties(&device_prop_, 0));
        CUDA_CALL(cudaMalloc(&device_storage_, device_storage_size_bytes_));
    }

    const cudaDeviceProp &get_device_property() const { return device_prop_; }

    template <typename T>
    T *get_device_value_ptr()
    {
        static_assert(sizeof(T) < device_storage_size_bytes_,
                      "The expected type is too large for the device storage!");
        return reinterpret_cast<T *>(device_storage_);
    }

    ~myBlasHandle() { cudaFree(&device_storage_); }

private:
    static constexpr std::size_t device_storage_size_bytes_{16};
    cudaDeviceProp device_prop_;
    void *device_storage_;
};


namespace kernel {


namespace cg = cooperative_groups;

template <typename ValueType>
__global__ __launch_bounds__(1) void init_res(ValueType *__restrict__ res)
{
    *res = ValueType{0};
}

template <std::int64_t block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void dot(
    const std::int32_t n, const ValueType *__restrict__ x,
    const std::int32_t x_stride, const ValueType *__restrict__ y,
    const std::int32_t y_stride, ValueType *__restrict__ res)
{
    // Here, using int32 is fine since input & stride are also in int32
    using index_type = std::int32_t;

    const index_type start = blockIdx.x * blockDim.x + threadIdx.x;
    const index_type increment = blockDim.x * gridDim.x;

    ValueType local_result{};
    const auto group = cg::this_thread_block();
    const auto local_id = group.thread_rank();

    __shared__ char shared_impl[sizeof(ValueType) * block_size];
    auto shared = reinterpret_cast<ValueType *>(shared_impl);

    for (index_type idx = start; idx < n; idx += increment) {
        const auto x_val = x[idx * x_stride];
        const auto y_val = y[idx * y_stride];
        local_result += x_val * y_val;
    }
    shared[local_id] = local_result;
    reduce(group, shared, [](ValueType a, ValueType b) { return a + b; });
    if (local_id == 0) {
        atomic_add(res, shared[local_id]);
    }
}

// Currently, x and y need to be 2D to also have a proper stride parameter
// (lowest-dim must always be 0)
template <std::int64_t block_size, typename XRange, typename YRange,
          typename ResType>
__global__ __launch_bounds__(block_size) void acc_dot(XRange x, YRange y,
                                                      ResType *__restrict__ res)
{
    using ar_type = decltype(x(0, 0) + y(0, 0));
    // Here, using int64 results in better performance since the stride in the
    // accessors is stored in uint64
    using index_type = std::int64_t;
    // static_assert(std::is_same<ResType, ar_type>::value, "Types must be
    // equal!");

    const index_type start = blockIdx.x * blockDim.x + threadIdx.x;
    const index_type increment = blockDim.x * gridDim.x;

    ar_type local_result{};
    const auto group = cg::this_thread_block();
    const auto local_id = group.thread_rank();

    __shared__ char shared_impl[sizeof(ar_type) * block_size];
    auto shared = reinterpret_cast<ar_type *>(shared_impl);

    for (index_type idx = start; idx < x.length(0); idx += increment) {
        local_result += x(idx, 0) * y(idx, 0);
    }
    shared[local_id] = local_result;
    reduce(group, shared, [](ar_type a, ar_type b) { return a + b; });
    if (local_id == 0) {
        atomic_add(res, static_cast<ResType>(shared[local_id]));
    }
}


template <typename InType, typename OutType>
__global__ __launch_bounds__(1) void cast_result(const InType *__restrict__ in,
                                                 OutType *__restrict__ out)
{
    *out = static_cast<OutType>(*in);
}


}  // namespace kernel


template <typename ValueType>
void control_dot(const matrix_info x_info, const ValueType *x,
                 const matrix_info y_info, const ValueType *y, ValueType *res)
{
    if (x_info.size[0] != y_info.size[0] || x_info.size[1] != 1 ||
        y_info.size[1] != 1) {
        throw std::runtime_error("Mismatching vectors!");
    }
    ValueType result{0};
    for (std::size_t idx = 0; idx < x_info.size[0]; ++idx) {
        result += x[idx * x_info.stride] * y[idx * y_info.stride];
    }
    *res = result;
}

template <typename ValueType>
void dot(myBlasHandle *handle, const matrix_info x_info, const ValueType *x,
         const matrix_info y_info, const ValueType *y, ValueType *res)
{
    constexpr std::int32_t block_size{dot_block_size};
    const dim3 block(block_size, 1, 1);
    const dim3 grid(
        handle->get_device_property().multiProcessorCount * grids_per_sm, 1, 1);

    kernel::init_res<<<1, 1>>>(res);
    kernel::dot<block_size, ValueType>
        <<<grid, block>>>(static_cast<std::int32_t>(x_info.size[0]), x,
                          static_cast<std::int32_t>(x_info.stride), y,
                          static_cast<std::int32_t>(y_info.stride), res);
}

template <typename ArType, typename StType, typename ResType>
void acc_dot(myBlasHandle *handle, const matrix_info x_info, const StType *x,
             const matrix_info y_info, const StType *y, ResType *res)
{
    constexpr std::int32_t block_size{dot_block_size};
    const dim3 block(block_size, 1, 1);
    const dim3 grid(
        handle->get_device_property().multiProcessorCount * grids_per_sm, 1, 1);

    // Accessor Setup
    constexpr std::size_t dimensionality{2};
    std::array<std::size_t, dimensionality - 1> x_stride{x_info.stride};
    std::array<std::size_t, dimensionality - 1> y_stride{y_info.stride};

    using accessor =
        gko::acc::reduced_row_major<dimensionality, ArType, StType>;
    using range = gko::acc::range<accessor>;
    using c_range = gko::acc::range<typename accessor::const_accessor>;
    auto x_acc = c_range(x_info.size, x, x_stride);
    auto y_acc = c_range(y_info.size, y, y_stride);

    ArType *acc_dot_res{nullptr};
    constexpr bool use_temporary_storage{!std::is_same<ArType, ResType>::value};
    if (use_temporary_storage) {
        acc_dot_res = handle->get_device_value_ptr<ArType>();
    } else {
        // reinterpret_cast only necessary to make compiler happy. They should
        // be the same type.
        acc_dot_res = reinterpret_cast<ArType *>(res);
    }

    kernel::init_res<<<1, 1>>>(acc_dot_res);
    kernel::acc_dot<block_size><<<grid, block>>>(x_acc, y_acc, acc_dot_res);

    if (use_temporary_storage) {
        kernel::cast_result<<<1, 1>>>(acc_dot_res, res);
    }
}

#define BIND_CUBLAS_DOT(ValueType, CublasName)                              \
    void cublas_dot(cublasHandle_t handle, int n, const ValueType *x,       \
                    int incx, const ValueType *y, int incy, ValueType *res) \
    {                                                                       \
        CUBLAS_CALL(CublasName(handle, n, x, incx, y, incy, res));          \
    }
BIND_CUBLAS_DOT(double, cublasDdot)
BIND_CUBLAS_DOT(float, cublasSdot)
#undef BIND_CUBLAS_DOT

template <typename ValueType>
void cublas_dot(cublasHandle_t handle, const matrix_info x_info,
                const ValueType *x, const matrix_info y_info,
                const ValueType *y, ValueType *res)
{
    cublas_dot(handle, static_cast<int>(x_info.size[0]), x,
               static_cast<int>(x_info.stride), y,
               static_cast<int>(y_info.stride), res);
}
