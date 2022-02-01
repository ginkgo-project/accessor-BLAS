#pragma once

#include <type_traits>


#include "matrix_helper.cuh"
#include "memory.cuh"
#include "utils.cuh"


/**
 * Manages the host and device memory for all GEMV benchmarks in the specified
 * precision.
 *
 * @tparam ValueType  The precision the managed data is in.
 */
template <typename ValueType>
class GemvMemory {
private:
    static constexpr auto CPU_device = Memory<ValueType>::Device::cpu;
    static constexpr auto GPU_device = Memory<ValueType>::Device::gpu;

public:
    /**
     * Allocates and initializes the memory randomly (with the given
     * distribution) and mirrors that data for both CPU and GPU. Random values
     * are generated on the CPU with vect_dist(engine) and mtx_dist(engine).
     *
     * @tparam MtxDist  type of the matrix distribution
     * @tparam VectDist  type of the vector distribution
     * @tparam RndEngine  type of the random engine
     *
     * @param max_size  the size (both rows and cols) of the matrix and all
                        vectors that are allocated
     * @param mtx_dist  distribution for the randomly generated matrix values
     * @param vect_dist  distribution for the randomly generated vector values
     * @param engine  random engine used to generate the values
     */
    template <typename MtxDist, typename VectDist, typename RndEngine>
    GemvMemory(matrix_info::size_type max_size, MtxDist &&mtx_dist,
               VectDist &&vect_dist, RndEngine &&engine)
        : m_info_{{max_size, max_size}},
          x_info_{{max_size, 1}},
          res_info_{{max_size, 1}},
          cpu_mtx_(gen_mtx<ValueType>(m_info_, mtx_dist, engine)),
          cpu_x_(gen_mtx<ValueType>(x_info_, vect_dist, engine)),
          cpu_res_(gen_mtx<ValueType>(res_info_, vect_dist, engine)),
          gpu_mtx_(GPU_device, m_info_.get_1d_size()),
          gpu_x_(GPU_device, x_info_.get_1d_size()),
          gpu_res_(GPU_device, res_info_.get_1d_size())
    {
        gpu_mtx_.copy_from(cpu_mtx_);
        gpu_x_.copy_from(cpu_x_);
        gpu_res_.copy_from(cpu_res_);
    }

    /**
     * Creates a copy of the data from another GemvMemory (with potentially a
     * different memory type). To convert different memory types, static_cast is
     * used.
     *
     * @tparam OtherType  memory type of the other object (can be different from
     * ValueType)
     *
     * @param other  GemvMemory object that is copied.
     */
    template <typename OtherType>
    GemvMemory(const GemvMemory<OtherType> &other)
        : m_info_(other.m_info_),
          x_info_(other.x_info_),
          res_info_(other.res_info_),
          cpu_mtx_(CPU_device, m_info_.get_1d_size()),
          cpu_x_(CPU_device, x_info_.get_1d_size()),
          cpu_res_(CPU_device, res_info_.get_1d_size()),
          gpu_mtx_(GPU_device, m_info_.get_1d_size()),
          gpu_x_(GPU_device, x_info_.get_1d_size()),
          gpu_res_(GPU_device, res_info_.get_1d_size())
    {
        convert(other);

        gpu_mtx_.copy_from(cpu_mtx_);
        gpu_x_.copy_from(cpu_x_);
        gpu_res_.copy_from(cpu_res_);
    }

    /**
     * Syncronizes the result from GPU to CPU.
     */
    void sync_result() { cpu_res_.copy_from(gpu_res_); }

private:
    template <typename OtherType>
    void convert(const GemvMemory<OtherType> &other)
    {
        convert_with(other,
                     [](OtherType val) { return static_cast<ValueType>(val); });
    }

    template <typename OtherType, typename Callable>
    void convert_with(const GemvMemory<OtherType> &other,
                      Callable &&convert_function)
    {
        convert_mtx(m_info_, other.cpu_mtx_const(), cpu_mtx(),
                    convert_function);
        convert_mtx(x_info_, other.cpu_x_const(), cpu_x(), convert_function);
        convert_mtx(res_info_, other.cpu_res_const(), cpu_res(),
                    convert_function);
    }

protected:
    ValueType *cpu_mtx() { return cpu_mtx_.data(); }
    ValueType *cpu_x() { return cpu_x_.data(); }

public:
    // Returns the CPU result pointer
    ValueType *cpu_res() { return cpu_res_.data(); }

    /**
     * @returns the memory object used for the CPU result
     */
    Memory<ValueType> &cpu_res_memory() { return cpu_res_; }

    // Return plain pointer to the corresponding CPU memory
    const ValueType *cpu_mtx_const() const { return cpu_mtx_.const_data(); }
    const ValueType *cpu_x_const() const { return cpu_x_.const_data(); }
    const ValueType *cpu_res_const() const { return cpu_res_.const_data(); }
    const ValueType *cpu_mtx() const { return cpu_mtx_const(); }
    const ValueType *cpu_x() const { return cpu_x_const(); }
    const ValueType *cpu_res() const { return cpu_res_const(); }

protected:
    // protected as they are only used internally
    ValueType *gpu_mtx() { return gpu_mtx_.const_data(); }
    ValueType *gpu_x() { return gpu_x_.const_data(); }

public:
    // Returns the GPU result pointer
    ValueType *gpu_res() { return gpu_res_.data(); }
    /**
     * @returns the memory object used for the GPU result
     */
    Memory<ValueType> &gpu_res_memory() { return gpu_res_; }

    // Return plain pointer to the corresponding CPU memory
    const ValueType *gpu_mtx_const() const { return gpu_mtx_.const_data(); }
    const ValueType *gpu_x_const() const { return gpu_x_.const_data(); }
    const ValueType *gpu_res_const() const { return gpu_res_.const_data(); }
    const ValueType *gpu_mtx() const { return gpu_mtx_const(); }
    const ValueType *gpu_x() const { return gpu_x_const(); }
    const ValueType *gpu_res() const { return gpu_res_const(); }

    // Informations about the matrix and the x and res vectors
    const matrix_info m_info_;
    const matrix_info x_info_;
    const matrix_info res_info_;

private:
    Memory<ValueType> cpu_mtx_;
    Memory<ValueType> cpu_x_;
    Memory<ValueType> cpu_res_;

    Memory<ValueType> gpu_mtx_;
    Memory<ValueType> gpu_x_;
    Memory<ValueType> gpu_res_;
};
