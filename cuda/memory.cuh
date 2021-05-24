#pragma once

#include <cstring>

#include "utils.cuh"

template <typename T>
class Memory {
   public:
    enum class Device { cpu, gpu };

    Memory(Device device, std::size_t num_elems)
        : device_{device}, num_elems_{num_elems}, data_{nullptr} {
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

    Memory(const Memory &other) : Memory(other.device_, other.num_elems_) {
        *this = other;
    }

    ~Memory() {
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

    Device get_device() const { return device_; }

    std::size_t get_num_elems() const { return num_elems_; }

    T *data() { return data_; }

    const T *data() const { return data_; }

    const T *const_data() const { return data_; }

    void copy_from(const Memory &other) {
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

    Memory &operator=(const Memory &other) {
        this->copy_from(other);
        return *this;
    }

   protected:
    const Device device_;
    const std::size_t num_elems_;
    T *data_;
};
