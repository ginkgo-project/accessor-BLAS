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

template <typename ValueType>
class TrsvMemory {
private:
    static constexpr auto CPU_device = Memory<ValueType>::Device::cpu;
    static constexpr auto GPU_device = Memory<ValueType>::Device::gpu;

public:
    template <typename MtxGen, typename VectGen>
    TrsvMemory(std::size_t max_rows, std::size_t max_cols, MtxGen &&cpu_mtx_gen,
               VectGen &&cpu_vect_gen)
        : m_info_{{max_rows, max_cols}},
          x_info_{{max_cols, 1}},
          cpu_mtx_(cpu_mtx_gen(m_info_)),
          cpu_x_(cpu_vect_gen(x_info_)),
          cpu_x_init_(cpu_x_),
          gpu_mtx_(GPU_device, m_info_.get_1d_size()),
          gpu_x_(GPU_device, x_info_.get_1d_size()),
          gpu_x_init_(GPU_device, x_info_.get_1d_size())
    {
        gpu_mtx_.copy_from(cpu_mtx_);
        gpu_x_.copy_from(cpu_x_);
        gpu_x_init_.copy_from(cpu_x_init_);

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

        const auto pivot_size = std::max(m_info_.size[0], m_info_.size[1]);
        Memory<int> cpu_pivot(Memory<int>::Device::cpu, pivot_size);
        Memory<int> gpu_pivot(Memory<int>::Device::gpu, pivot_size);
        for (std::size_t i = 0; i < pivot_size; ++i) {
            cpu_pivot.data()[i] = i;
        }
        gpu_pivot = cpu_pivot;

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

        cpu_info = gpu_info;
        synchronize();

        cpu_mtx_ = gpu_mtx_;
    }

    template <typename OtherType>
    TrsvMemory(const TrsvMemory<OtherType> &other)
        : m_info_(other.m_info_),
          x_info_(other.x_info_),
          cpu_mtx_(CPU_device, m_info_.get_1d_size()),
          cpu_x_(CPU_device, x_info_.get_1d_size()),
          cpu_x_init_(CPU_device, x_info_.get_1d_size()),
          gpu_mtx_(GPU_device, m_info_.get_1d_size()),
          gpu_x_(GPU_device, x_info_.get_1d_size()),
          gpu_x_init_(GPU_device, x_info_.get_1d_size())
    {
        // Note: conversion must be adopted if `error_type` is used
        convert(other);

        gpu_mtx_.copy_from(cpu_mtx_);
        gpu_x_.copy_from(cpu_x_);
        gpu_x_init_.copy_from(cpu_x_init_);
    }

    void sync_x() { cpu_x_.copy_from(gpu_x_); }

    void reset_x()
    {
        cpu_x_.copy_from(cpu_x_init_);
        gpu_x_.copy_from(gpu_x_init_);
    }

private:
    template <typename OtherType>
    std::enable_if_t<std::is_floating_point<OtherType>::value> convert(
        const TrsvMemory<OtherType> &other)
    {
        convert_with(other,
                     [](OtherType val) { return static_cast<ValueType>(val); });
    }

    template <typename OtherType>
    std::enable_if_t<!std::is_floating_point<OtherType>::value> convert(
        const TrsvMemory<OtherType> &other)
    {
        convert_with(other, [](OtherType val) {
            return ValueType{
                static_cast<typename ValueType::value_type>(val.v),
                static_cast<typename ValueType::value_type>(val.e)};
        });
    }

    template <typename OtherType, typename Callable>
    void convert_with(const TrsvMemory<OtherType> &other,
                      Callable &&convert_function)
    {
        convert_mtx(m_info_, other.cpu_mtx_const(), cpu_mtx(),
                    convert_function);
        convert_mtx(x_info_, other.cpu_x_const(), cpu_x(), convert_function);
        cpu_x_init_.copy_from(cpu_x_);
    }

protected:
    ValueType *cpu_mtx() { return cpu_mtx_.data(); }

public:
    ValueType *cpu_x() { return cpu_x_.data(); }
    Memory<ValueType> &cpu_x_memory() { return cpu_x_; }

    const ValueType *cpu_mtx_const() const { return cpu_mtx_.const_data(); }
    const ValueType *cpu_x_const() const { return cpu_x_.const_data(); }
    const ValueType *cpu_x_init_const() const
    {
        return cpu_x_init_.const_data();
    }
    const ValueType *cpu_mtx() const { return cpu_mtx_const(); }
    const ValueType *cpu_x() const { return cpu_x_const(); }

protected:
    ValueType *gpu_mtx() { return gpu_mtx_.data(); }

public:
    ValueType *gpu_x() { return gpu_x_.data(); }
    Memory<ValueType> &gpu_x_memory() { return gpu_x_; }

    const ValueType *gpu_mtx_const() const { return gpu_mtx_.const_data(); }
    const ValueType *gpu_x_const() const { return gpu_x_.const_data(); }
    const ValueType *gpu_x_init_const() const
    {
        return gpu_x_init_.const_data();
    }
    const ValueType *gpu_mtx() const { return gpu_mtx_const(); }
    const ValueType *gpu_x() const { return gpu_x_const(); }

    const matrix_info m_info_;
    const matrix_info x_info_;

private:
    Memory<ValueType> cpu_mtx_;
    Memory<ValueType> cpu_x_;
    Memory<ValueType> cpu_x_init_;

    Memory<ValueType> gpu_mtx_;
    Memory<ValueType> gpu_x_;
    Memory<ValueType> gpu_x_init_;
};
