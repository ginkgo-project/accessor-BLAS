#pragma once

#include <type_traits>


#include "matrix_helper.cuh"
#include "memory.cuh"
#include "utils.cuh"


/**
 * Manages the host and device memory for all DOT benchmarks in the specified
 * precision.
 *
 * @tparam ValueType  The precision the managed data is in.
 */
template <typename ValueType>
class DotMemory {
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
    DotMemory(matrix_info::size_type size, VectDist &&vect_dist, RndEngine &&engine)
        : x_info_{{size, 1}},
          y_info_{{size, 1}},
          cpu_x_(gen_mtx<ValueType>(x_info_, vect_dist, engine)),
          cpu_y_(gen_mtx<ValueType>(y_info_, vect_dist, engine)),
          cpu_res_(CPU_device, 1),
          gpu_x_(GPU_device, x_info_.get_1d_size()),
          gpu_y_(GPU_device, y_info_.get_1d_size()),
          gpu_res_(GPU_device, 1)
    {
        *cpu_res_.data() = ValueType{-999};
        copy_cpu_to_gpu();
    }

    /**
     * Creates a copy of the data from another DotMemory (with potentially a
     * different memory type). To convert different memory types, static_cast is
     * used.
     *
     * @tparam OtherType  memory type of the other object (can be different from
     * ValueType)
     *
     * @param other  DotMemory object that is copied.
     */
    template <typename OtherType>
    DotMemory(const DotMemory<OtherType> &other)
        : x_info_(other.x_info_),
          y_info_(other.y_info_),
          cpu_x_(CPU_device, x_info_.get_1d_size()),
          cpu_y_(CPU_device, y_info_.get_1d_size()),
          cpu_res_(CPU_device, 1),
          gpu_x_(GPU_device, x_info_.get_1d_size()),
          gpu_y_(GPU_device, y_info_.get_1d_size()),
          gpu_res_(GPU_device, 1)
    {
        convert(other);

        copy_cpu_to_gpu();
    }

    /**
     * Copies the data from the GPU result to CPU and returns the result value
     *
     * @returns the result value that was stored in GPU memory
     */
    ValueType get_result()
    {
        cpu_res_.copy_from(gpu_res_);
        return *cpu_res_.data();
    }

    /**
     * Copies all memory from CPU to GPU.
     */
    void copy_cpu_to_gpu()
    {
        gpu_x_.copy_from(cpu_x_);
        gpu_y_.copy_from(cpu_y_);
        gpu_res_.copy_from(cpu_res_);
    }

    /**
     * Copies the data from another DotMemory (with potentially a different
     * memory type). To convert different memory types, static_cast is used.
     *
     * @tparam OtherType  memory type of the other object (can be different from
     * ValueType)
     *
     * @param other  DotMemory object that is copied.
     */
    template <typename OtherType>
    void convert_from(const DotMemory<OtherType> &other)
    {
        convert(other);
        copy_cpu_to_gpu();
    }

private:
    template <typename OtherType>
    void convert(const DotMemory<OtherType> &other)
    {
        convert_with(other,
                     [](OtherType val) { return static_cast<ValueType>(val); });
    }

    template <typename OtherType, typename Callable>
    void convert_with(const DotMemory<OtherType> &other,
                      Callable &&convert_function)
    {
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

    // All of them return a plain pointer to the corresponding CPU memory
    const ValueType *cpu_x() const { return cpu_x_.data(); }
    const ValueType *cpu_y() const { return cpu_y_.data(); }
    ValueType *cpu_res() { return cpu_res_.data(); }
    const ValueType *cpu_res() const { return cpu_res_.data(); }

    // All of them return a plain pointer to the corresponding GPU memory
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
