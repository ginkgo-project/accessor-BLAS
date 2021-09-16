#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>


#include <cusolverDn.h>


#include "matrix_helper.cuh"
#include "memory.cuh"
#include "utils.cuh"


namespace detail {


std::string get_cusolver_error_string(cusolverStatus_t err)
{
    switch (err) {
    case CUSOLVER_STATUS_SUCCESS:
        return "Success";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "Not initialized";
    case CUSOLVER_STATUS_ALLOC_FAILED:
        return "Allocation failed";
    case CUSOLVER_STATUS_INVALID_VALUE:
        return "Invalid value";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "Architecture mismatch";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "Execution failed";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "Internal error";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "Matrix type not supported";
    default:
        return std::string("Unknown Error: ") + std::to_string(err);
    };
}


}  // namespace detail


#define CUSOLVER_CALL(call)                                                   \
    do {                                                                      \
        auto err = call;                                                      \
        if (err != CUSOLVER_STATUS_SUCCESS) {                                 \
            std::cerr << "CuSolver error in file " << __FILE__                \
                      << " L:" << __LINE__ << "; Error: " << err << '\n';     \
            throw std::runtime_error(std::string("Error: ") +                 \
                                     detail::get_cusolver_error_string(err)); \
        }                                                                     \
    } while (false)

using CusolverContext = std::remove_pointer_t<cusolverDnHandle_t>;

std::unique_ptr<CusolverContext, std::function<void(cusolverDnHandle_t)>>
cusolver_get_handle()
{
    cusolverDnHandle_t handle;
    CUSOLVER_CALL(cusolverDnCreate(&handle));
    return {handle, [](cusolverDnHandle_t handle) {
                CUSOLVER_CALL(cusolverDnDestroy(handle));
            }};
}

/*
namespace {
#define BIND_CUBLAS_GEAM(ValueType, CublasName)                             \
    void cublas_geam(cublasHandle_t handle, cublasOperation_t transa,       \
                     cublasOperation_t transb, int m, int n,                \
                     const ValueType *alpha, const ValueType *A, int lda,   \
                     const ValueType *beta, const ValueType *B, int ldb,    \
                     ValueType *C, int ldc)                                 \
    {                                                                       \
        CUBLAS_CALL(CublasName(handle, transa, transb, m, n, alpha, A, lda, \
                               beta, B, ldb, C, ldc));                      \
    }
BIND_CUBLAS_GEMV(double, cublasDgemv)
BIND_CUBLAS_GEMV(float, cublasSgemv)
#undef BIND_CUBLAS_GEMV

template <typename T>
void cublas_transpose(cublasHandle_t handle, const matrix_info minfo, T *mtx)
{
    T alpha{1};
    T beta{0};
    cublas_geam(handle, CUBLAS_OP_T, CUBLAS_OP_N, );
}


}  // namespace
*/


template <typename ValueType>
class IrMemory {
private:
    static constexpr auto CPU_device = Memory<ValueType>::Device::cpu;
    static constexpr auto GPU_device = Memory<ValueType>::Device::gpu;

public:
    IrMemory(std::size_t max_rows, std::size_t max_cols,
             Memory<ValueType> orig_mtx)
        : m_info_{{max_rows, max_cols}},
          x_info_{{max_cols, 1}},
          pivot_size_{std::max(m_info_.size[0], m_info_.size[1])},
          cpu_pivot_(Memory<int>::Device::cpu, pivot_size_),
          cpu_mtx_(CPU_device, m_info_.get_1d_size()),
          cpu_x_(CPU_device, x_info_.get_1d_size()),
          gpu_mtx_(GPU_device, m_info_.get_1d_size()),
          gpu_x_(GPU_device, x_info_.get_1d_size())
    {
        gpu_mtx_.copy_from(orig_mtx);
        gpu_x_.copy_from(cpu_x_);

        if (m_info_.size[0] != m_info_.size[1]) {
            throw std::runtime_error(std::string("Matrix is not quadratic: ") +
                                     std::to_string(m_info_.size[0]) + " x " +
                                     std::to_string(m_info_.size[1]));
        }
        // Factorize matrix into L and U on the CUDA device:
        auto handle = cusolver_get_handle();

        Memory<int> cpu_info(Memory<int>::Device::cpu, 1);
        *cpu_info.data() = 0;
        Memory<int> gpu_info(Memory<int>::Device::gpu, 1);
        gpu_info = cpu_info;

        for (std::size_t i = 0; i < pivot_size_; ++i) {
            cpu_pivot_.data()[i] = 0;
        }
        Memory<int> gpu_pivot(Memory<int>::Device::gpu, pivot_size_);
        gpu_pivot = cpu_pivot_;

        int workspace_size{};

        // Get workspace size requirement
        CUSOLVER_CALL(cusolverDnDgetrf_bufferSize(
            handle.get(), static_cast<int>(m_info_.size[0]),
            static_cast<int>(m_info_.size[1]), gpu_mtx_.data(),
            static_cast<int>(m_info_.stride), &workspace_size));

        Memory<ValueType> gpu_workspace(GPU_device, workspace_size);

        // Expects the matrix in column-major
        // Run matrix factorization
        CUSOLVER_CALL(cusolverDnDgetrf(
            handle.get(), static_cast<int>(m_info_.size[0]),
            static_cast<int>(m_info_.size[1]), gpu_mtx_.data(),
            static_cast<int>(m_info_.stride), gpu_workspace.data(),
            gpu_pivot.data(), gpu_info.data()));

        cpu_pivot_.copy_from(gpu_pivot);
        cpu_info = gpu_info;
        synchronize();

        cpu_mtx_ = gpu_mtx_;
    }

    template <typename OtherType>
    IrMemory(const IrMemory<OtherType> &other)
        : m_info_(other.m_info_),
          x_info_(other.x_info_),
          pivot_size_{other.get_pivot_size()},
          cpu_pivot_(Memory<int>::Device::cpu, pivot_size_),
          cpu_mtx_(CPU_device, m_info_.get_1d_size()),
          cpu_x_(CPU_device, x_info_.get_1d_size()),
          gpu_mtx_(GPU_device, m_info_.get_1d_size()),
          gpu_x_(GPU_device, x_info_.get_1d_size())
    {
        cpu_pivot_.copy_from(other.cpu_pivot());

        convert(other);

        gpu_mtx_.copy_from(cpu_mtx_);
        gpu_x_.copy_from(cpu_x_);
    }

    void sync_x() { cpu_x_.copy_from(gpu_x_); }

private:
    template <typename OtherType>
    void convert(const IrMemory<OtherType> &other)
    {
        convert_with(other,
                     [](OtherType val) { return static_cast<ValueType>(val); });
    }

    template <typename OtherType, typename Callable>
    void convert_with(const IrMemory<OtherType> &other,
                      Callable &&convert_function)
    {
        convert_mtx(m_info_, other.cpu_mtx_const(), cpu_mtx(),
                    convert_function);
        convert_mtx(x_info_, other.cpu_x_const(), cpu_x(), convert_function);
    }

protected:
    ValueType *cpu_mtx() { return cpu_mtx_.data(); }

public:
    std::size_t get_pivot_size() const { return pivot_size_; }
    ValueType *cpu_x() { return cpu_x_.data(); }
    Memory<ValueType> &cpu_x_memory() { return cpu_x_; }

    const Memory<int> &cpu_pivot() const { return cpu_pivot_; }
    const ValueType *cpu_mtx_const() const { return cpu_mtx_.const_data(); }
    const ValueType *cpu_x_const() const { return cpu_x_.const_data(); }
    const ValueType *cpu_mtx() const { return cpu_mtx_const(); }
    const ValueType *cpu_x() const { return cpu_x_const(); }

    Memory<ValueType> &cpu_mtx_memory() { return cpu_mtx_; }

protected:
    ValueType *gpu_mtx() { return gpu_mtx_.data(); }

public:
    ValueType *gpu_x() { return gpu_x_.data(); }
    Memory<ValueType> &gpu_x_memory() { return gpu_x_; }

    const ValueType *gpu_mtx_const() const { return gpu_mtx_.const_data(); }
    const ValueType *gpu_x_const() const { return gpu_x_.const_data(); }
    const ValueType *gpu_mtx() const { return gpu_mtx_const(); }
    const ValueType *gpu_x() const { return gpu_x_const(); }
    
    Memory<ValueType> &gpu_mtx_memory() { return gpu_mtx_; }

    const matrix_info m_info_;
    const matrix_info x_info_;

private:
    const std::size_t pivot_size_;
    Memory<int> cpu_pivot_;
    Memory<ValueType> cpu_mtx_;
    Memory<ValueType> cpu_x_;

    Memory<ValueType> gpu_mtx_;
    Memory<ValueType> gpu_x_;
};
