#pragma once

#include <cstring>


#include "utils.cuh"


/**
 * Memory management object for both CPU and GPU.
 *
 * @tparam T  type of the values the memory should contain.
 */
template <typename T>
class Memory {
public:
    /**
     * Enum listing the different devices the memory can be stored.
     */
    enum class Device { cpu, gpu };

    /**
     * Allocates memory, so the given amount of elements have space.
     *
     * @param device  target device for the memory
     * @param num_elems  number of elements in the memory to allocate
     */
    Memory(Device device, std::size_t num_elems)
        : device_{device}, num_elems_{num_elems}, data_{nullptr}
    {
        switch (device_) {
        case Device::cpu:
            data_ = new T[num_elems_];
            break;
        case Device::gpu:
            CUDA_CALL(cudaMalloc(&data_, num_elems_ * sizeof(T)));
            break;
        default:
            throw std::runtime_error("Unsupported device");
        };
    }

    /**
     * Copies the memory from another management object and creates a new one.
     *
     * @param other  memory to copy
     */
    Memory(const Memory &other) : Memory(other.device_, other.num_elems_)
    {
        *this = other;
    }

    ~Memory()
    {
        switch (device_) {
        case Device::cpu:
            delete[] data_;
            data_ = nullptr;
            break;
        case Device::gpu:
            synchronize();
            cudaFree(data_);
            data_ = nullptr;
            break;
        };
    }

    /**
     * @returns the device this memory is stored on.
     */
    Device get_device() const { return device_; }

    /**
     * @returns number of elements this memory is storing.
     */
    std::size_t get_num_elems() const { return num_elems_; }

    /**
     * @returns the data pointer to the managed memory.
     */
    T *data() { return data_; }

    /**
     * @returns the const data pointer to the managed memory.
     */
    const T *data() const { return data_; }

    /**
     * @returns the const data pointer to the managed memory.
     */
    const T *const_data() const { return data_; }

    /**
     * Copies the data from the other memory to this one. The devices can
     * mismatch, but the amount of elements must be identical.
     *
     * @param other  the memory to copy
     */
    void copy_from(const Memory &other)
    {
        if (num_elems_ != other.num_elems_) {
            throw std::runtime_error("Mismatching number of elements");
        }
        std::size_t size_bytes = num_elems_ * sizeof(T);
        if (device_ == Device::cpu) {
            if (other.device_ == Device::cpu) {
                std::memcpy(data_, other.data_, size_bytes);
            } else {  //   other.device_ == Device::gpu:
                CUDA_CALL(cudaMemcpy(data_, other.data_, size_bytes,
                                     cudaMemcpyDeviceToHost));
            }
        } else {  // device_ == device::gpu
            if (other.device_ == Device::cpu) {
                CUDA_CALL(cudaMemcpy(data_, other.data_, size_bytes,
                                     cudaMemcpyHostToDevice));
            } else {  //   other.device_ == Device::gpu:
                CUDA_CALL(cudaMemcpy(data_, other.data_, size_bytes,
                                     cudaMemcpyDeviceToDevice));
            };
        }
    }

    /**
     * Copies the data from the other memory to this one. The devices can
     * mismatch, but the amount of elements must be identical.
     *
     * @param other  the memory to copy
     */
    Memory &operator=(const Memory &other)
    {
        this->copy_from(other);
        return *this;
    }

protected:
    const Device device_;
    const std::size_t num_elems_;
    T *data_;
};
