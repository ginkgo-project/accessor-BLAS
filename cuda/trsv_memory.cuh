#pragma once

#include <type_traits>

#include "matrix_helper.cuh"
#include "memory.cuh"
#include "utils.cuh"

template <typename ValueType>
class TrsvMemory {
   private:
    static constexpr auto CPU_device = Memory<ValueType>::Device::cpu;
    static constexpr auto GPU_device = Memory<ValueType>::Device::gpu;

   public:
    template <typename MtxDist, typename VectDist, typename RndEngine>
    TrsvMemory(std::size_t max_rows, std::size_t max_cols, MtxDist &&mtx_dist,
               VectDist &&vect_dist, RndEngine &&engine)
        : m_info_{{max_rows, max_cols}},
          x_info_{{max_cols, 1}},
          cpu_mtx_(gen_mtx<ValueType>(m_info_, mtx_dist, engine)),
          cpu_x_(gen_mtx<ValueType>(x_info_, vect_dist, engine)),
          cpu_x_init_(cpu_x_),
          gpu_mtx_(GPU_device, m_info_.get_1d_size()),
          gpu_x_(GPU_device, x_info_.get_1d_size()),
          gpu_x_init_(GPU_device, x_info_.get_1d_size()) {
        gpu_mtx_.copy_from(cpu_mtx_);
        gpu_x_.copy_from(cpu_x_);
        gpu_x_init_.copy_from(cpu_x_init_);
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
          gpu_x_init_(GPU_device, x_info_.get_1d_size()) {
        // Note: conversion must be adopted if `error_type` is used
        convert(other);

        gpu_mtx_.copy_from(cpu_mtx_);
        gpu_x_.copy_from(cpu_x_);
        gpu_x_init_.copy_from(cpu_x_init_);
    }

    void sync_x() { cpu_x_.copy_from(gpu_x_); }

    void reset_x() {
        cpu_x_.copy_from(cpu_x_init_);
        gpu_x_.copy_from(gpu_x_init_);
    }

   private:
    template <typename OtherType>
    std::enable_if_t<std::is_floating_point<OtherType>::value> convert(
        const TrsvMemory<OtherType> &other) {
        convert_with(other,
                     [](OtherType val) { return static_cast<ValueType>(val); });
    }

    template <typename OtherType>
    std::enable_if_t<!std::is_floating_point<OtherType>::value> convert(
        const TrsvMemory<OtherType> &other) {
        convert_with(other, [](OtherType val) {
            return ValueType{
                static_cast<typename ValueType::value_type>(val.v),
                static_cast<typename ValueType::value_type>(val.e)};
        });
    }

    template <typename OtherType, typename Callable>
    void convert_with(const TrsvMemory<OtherType> &other,
                      Callable &&convert_function) {
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
    const ValueType *cpu_x_init_const() const {
        return cpu_x_init_.const_data();
    }
    const ValueType *cpu_mtx() const { return cpu_mtx_const(); }
    const ValueType *cpu_x() const { return cpu_x_const(); }

   protected:
    ValueType *gpu_mtx() { return gpu_mtx_.data(); }
    // ValueType *gpu_x() { return gpu_x_.const_data(); }

   public:
    ValueType *gpu_x() { return gpu_x_.data(); }
    Memory<ValueType> &gpu_x_memory() { return gpu_x_; }

    const ValueType *gpu_mtx_const() const { return gpu_mtx_.const_data(); }
    const ValueType *gpu_x_const() const { return gpu_x_.const_data(); }
    const ValueType *gpu_x_init_const() const {
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

