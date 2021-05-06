#pragma once

#include <type_traits>

#include "matrix_helper.cuh"
#include "memory.cuh"
#include "utils.cuh"

template <typename ValueType>
class DotMemory {
   private:
    static constexpr auto CPU_device = Memory<ValueType>::Device::cpu;
    static constexpr auto GPU_device = Memory<ValueType>::Device::gpu;

   public:
    template <typename VectDist, typename RndEngine>
    DotMemory(std::size_t max_rows, VectDist &&vect_dist, RndEngine &&engine)
        : x_info_{{max_rows, 1}},
          y_info_{{max_rows, 1}},
          cpu_x_(gen_mtx<ValueType>(x_info_, vect_dist, engine)),
          cpu_y_(gen_mtx<ValueType>(y_info_, vect_dist, engine)),
          cpu_res_(CPU_device, 1),
          gpu_x_(GPU_device, x_info_.get_1d_size()),
          gpu_y_(GPU_device, y_info_.get_1d_size()),
          gpu_res_(GPU_device, 1) {
        *cpu_res_.data() = ValueType{-999};
        copy_cpu_to_gpu();
    }
    template <typename OtherType>
    DotMemory(const DotMemory<OtherType> &other)
        : x_info_(other.x_info_),
          y_info_(other.y_info_),
          cpu_x_(CPU_device, x_info_.get_1d_size()),
          cpu_y_(CPU_device, y_info_.get_1d_size()),
          cpu_res_(CPU_device, 1),
          gpu_x_(GPU_device, x_info_.get_1d_size()),
          gpu_y_(GPU_device, y_info_.get_1d_size()),
          gpu_res_(GPU_device, 1) {
        // Note: conversion must be adopted if `error_type` is used
        convert(other);

        copy_cpu_to_gpu();
    }

    ValueType get_result() {
        cpu_res_.copy_from(gpu_res_);
        return *cpu_res_.data();
    }
    void copy_cpu_to_gpu() {
        gpu_x_.copy_from(cpu_x_);
        gpu_y_.copy_from(cpu_y_);
        gpu_res_.copy_from(cpu_res_);
    }

    template <typename OtherType>
    void convert_from(const DotMemory<OtherType> &other) {
        convert(other);
        copy_cpu_to_gpu();
    }

   private:
    template <typename OtherType>
    std::enable_if_t<std::is_floating_point<OtherType>::value> convert(
        const DotMemory<OtherType> &other) {
        convert_with(other,
                     [](OtherType val) { return static_cast<ValueType>(val); });
    }

    template <typename OtherType>
    std::enable_if_t<!std::is_floating_point<OtherType>::value> convert(
        const DotMemory<OtherType> &other) {
        convert_with(other, [](OtherType val) {
            return ValueType{
                static_cast<typename ValueType::value_type>(val.v),
                static_cast<typename ValueType::value_type>(val.e)};
        });
    }

    template <typename OtherType, typename Callable>
    void convert_with(const DotMemory<OtherType> &other,
                      Callable &&convert_function) {
        convert_mtx(x_info_, other.cpu_x(), cpu_x_nc(), convert_function);
        convert_mtx(y_info_, other.cpu_y(), cpu_y_nc(), convert_function);
        convert_mtx(matrix_info{{1, 1}}, other.cpu_res(), cpu_res_nc(),
                    convert_function);
    }

   public:
    // Non-const data access needed for conversions and re-randomizing the
    // vectors
    ValueType *cpu_x_nc() { return cpu_x_.data(); }
    ValueType *cpu_y_nc() { return cpu_y_.data(); }
    ValueType *cpu_res_nc() { return cpu_res_.data(); }

   public:
    const ValueType *cpu_x() const { return cpu_x_.data(); }
    const ValueType *cpu_y() const { return cpu_y_.data(); }
    ValueType *cpu_res() { return cpu_res_.data(); }
    const ValueType *cpu_res() const { return cpu_res_.data(); }

    const ValueType *gpu_x() const { return gpu_x_.data(); }
    const ValueType *gpu_y() const { return gpu_y_.data(); }
    ValueType *gpu_res() { return gpu_res_.data(); }
    const ValueType *gpu_res() const { return gpu_res_.data(); }

    const matrix_info x_info_;
    const matrix_info y_info_;

   private:
    Memory<ValueType> cpu_x_;
    Memory<ValueType> cpu_y_;
    Memory<ValueType> cpu_res_;

    Memory<ValueType> gpu_x_;
    Memory<ValueType> gpu_y_;
    Memory<ValueType> gpu_res_;
};

