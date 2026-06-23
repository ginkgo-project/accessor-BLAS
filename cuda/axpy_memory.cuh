#pragma once

#include <type_traits>


#include "matrix_helper.cuh"
#include "memory.cuh"
#include "utils.cuh"


/**
 * Manages the host and device memory for all AXPY benchmarks in the specified
 * precision.
 *
 * @tparam ValueType  The precision the managed data is in.
 */
template <typename ValueType>
class AxpyMemory {
private:
    static constexpr auto CPU_device = Memory<ValueType>::Device::cpu;
    static constexpr auto GPU_device = Memory<ValueType>::Device::gpu;

public:
    /**
     * Allocates and initializes the memory randomly (with the given
     * distribution) and mirrors that data for both CPU and GPU. Random values
     * are generated on the CPU with vect_dist(engine).
     *
     * @tparam VectDist  type of the distribution
     * @tparam RndEngine  type of the random engine
     *
     * @param size  the size of each vector that is generated
     * @param vect_dist  distribution for the randomly generated values
     * @param engine  random engine used to generate the values
     */
    template <typename VectDist, typename RndEngine>
    AxpyMemory(matrix_info::size_type size, VectDist &&vect_dist,
               RndEngine &&engine)
        : x_info_{{size, 1}},
          res_info_{{size, 1}},
          cpu_x_(gen_mtx<ValueType>(x_info_, vect_dist, engine)),
          cpu_res_(gen_mtx<ValueType>(res_info_, vect_dist, engine)),
          gpu_x_(GPU_device, x_info_.get_1d_size()),
          gpu_res_(GPU_device, res_info_.get_1d_size())
    {
        copy_cpu_to_gpu();
    }

    /**
     * Creates a copy of the data from another AxpyMemory (with potentially a
     * different memory type). To convert different memory types, static_cast is
     * used.
     *
     * @tparam OtherType  memory type of the other object (can be different from
     * ValueType)
     *
     * @param other  AxpyMemory object that is copied.
     */
    template <typename OtherType>
    AxpyMemory(const AxpyMemory<OtherType> &other)
        : x_info_(other.x_info_),
          res_info_(other.res_info_),
          cpu_x_(CPU_device, x_info_.get_1d_size()),
          cpu_res_(CPU_device, res_info_.get_1d_size()),
          gpu_x_(GPU_device, x_info_.get_1d_size()),
          gpu_res_(GPU_device, res_info_.get_1d_size())
    {
        convert(other);
        copy_cpu_to_gpu();
    }

    /**
     * Syncronizes the result from GPU to CPU.
     */
    void sync_result() { cpu_res_.copy_from(gpu_res_); }

    /**
     * Copies all memory from CPU to GPU.
     */
    void copy_cpu_to_gpu()
    {
        gpu_x_.copy_from(cpu_x_);
        gpu_res_.copy_from(cpu_res_);
    }

private:
    template <typename OtherType>
    void convert(const AxpyMemory<OtherType> &other)
    {
        convert_with(other,
                     [](OtherType val) { return static_cast<ValueType>(val); });
    }

    template <typename OtherType, typename Callable>
    void convert_with(const AxpyMemory<OtherType> &other,
                      Callable &&convert_function)
    {
        convert_mtx(x_info_, other.cpu_x(), cpu_x(), convert_function);
        convert_mtx(res_info_, other.cpu_res(), cpu_res(), convert_function);
    }

public:
    ValueType *cpu_x() { return cpu_x_.data(); }
    ValueType *cpu_res() { return cpu_res_.data(); }
    const ValueType *cpu_x() const { return cpu_x_.const_data(); }
    const ValueType *cpu_res() const { return cpu_res_.const_data(); }

    ValueType *gpu_res() { return gpu_res_.data(); }
    const ValueType *gpu_x() const { return gpu_x_.const_data(); }
    const ValueType *gpu_res() const { return gpu_res_.const_data(); }

    // Needed to snapshot/restore res between error measurements
    Memory<ValueType> &cpu_res_memory() { return cpu_res_; }
    const Memory<ValueType> &cpu_res_memory() const { return cpu_res_; }
    Memory<ValueType> &gpu_res_memory() { return gpu_res_; }
    const Memory<ValueType> &gpu_res_memory() const { return gpu_res_; }

    // Informations about the x and res vectors
    const matrix_info x_info_;
    const matrix_info res_info_;

private:
    Memory<ValueType> cpu_x_;
    Memory<ValueType> cpu_res_;

    Memory<ValueType> gpu_x_;
    Memory<ValueType> gpu_res_;
};
