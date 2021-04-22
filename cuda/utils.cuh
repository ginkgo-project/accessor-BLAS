#pragma once

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>


#define CUDA_CALL(call)                                                  \
    do {                                                                 \
        auto err = call;                                                 \
        if (err != cudaSuccess) {                                        \
            std::cerr << "Cuda error in file " << __FILE__               \
                      << " L:" << __LINE__                               \
                      << "; Error: " << cudaGetErrorString(err) << '\n'; \
            throw std::runtime_error(cudaGetErrorString(err));           \
        }                                                                \
    } while (false)

struct matrix_info {
    const std::array<std::size_t, 2> size;
    const std::size_t stride;

    constexpr matrix_info(const std::array<std::size_t, 2> size)
        : size(size), stride{size[1]} {}
    constexpr matrix_info(const std::array<std::size_t, 2> size,
                       const std::size_t stride)
        : size(size), stride{stride} {}

    std::size_t get_1d_size() const { return size[0] * stride; }
    std::size_t get_num_elems() const { return size[0] * size[1]; }
};

template <typename ValueType, typename ValueDist, typename Engine>
std::vector<ValueType> gen_mtx(const matrix_info &info, ValueDist &&dist,
                               Engine &&engine) {
    if (info.stride < info.size[1]) {
        throw "Error!";
    }
    std::vector<ValueType> res(info.get_1d_size());

    for (std::size_t row = 0; row < info.size[0]; ++row) {
        for (std::size_t col = 0; col < info.size[1]; ++col) {
            const std::size_t idx = row * info.stride + col;
            res[idx] = dist(engine);
        }
    }

    return res;
}

template <typename ResultType, typename InputType>
std::vector<ResultType> convert_mtx(const matrix_info &info,
                                    const std::vector<InputType> &input) {
    std::vector<ResultType> output(info.get_1d_size());
    for (std::size_t row = 0; row < info.size[0]; ++row) {
        for (std::size_t col = 0; col < info.size[1]; ++col) {
            const std::size_t idx = row * info.stride + col;
            output[idx] = static_cast<ResultType>(input[idx]);
        }
    }
    return output;
}

template <typename ValueType>
void print_mtx(const matrix_info &info, const std::vector<ValueType> &vec) {
    for (std::size_t i = 0; i < info.size[0]; ++i) {
        for (std::size_t j = 0; j < info.size[1]; ++j) {
            const auto idx = i * info.stride + j;
            std::cout << vec[idx] << '\t';
        }
        std::cout << '\n';
    }
}

///////////// GPU relevant code \\\\\\\\\\\\\


void synchronize() { CUDA_CALL(cudaDeviceSynchronize()); }


template <typename ValueType>
class GpuMemory {
   public:
    GpuMemory(std::size_t num_elems)
        : num_elems_{num_elems}, size_{num_elems * sizeof(ValueType)} {
        cudaSetDevice(0);
        CUDA_CALL(cudaMalloc(&data_, size_));
    }
    ~GpuMemory() { cudaFree(data_); }

    ValueType *data() {
        return data_;
    }

    void re_allocate() {
        CUDA_CALL(cudaFree(data_));
        CUDA_CALL(cudaMalloc(&data_, size_));
    }

    void copy_from(const std::vector<ValueType> &vec) {
        if (vec.size() > num_elems_) {
            throw "Error!!";
        }
        CUDA_CALL(cudaMemcpy(data_, vec.data(), vec.size() * sizeof(ValueType),
                             cudaMemcpyHostToDevice));
    }

    std::size_t get_num_elems() const { return num_elems_; }

    std::size_t get_byte_size() const { return size_; }

    std::vector<ValueType> get_vector() const {
        std::vector<ValueType> vec(num_elems_);
        CUDA_CALL(cudaMemcpy(vec.data(), data_, num_elems_ * sizeof(ValueType),
                             cudaMemcpyDeviceToHost));
        return vec;
    }

   private:
    std::size_t size_;
    std::size_t num_elems_;
    ValueType *data_;
};


struct cuda_event {
   public:
    cuda_event() { CUDA_CALL(cudaEventCreate(&ev_)); }

    ~cuda_event() { cudaEventDestroy(ev_); }

    void reset() {
        CUDA_CALL(cudaEventDestroy(ev_));
        CUDA_CALL(cudaEventCreate(&ev_));
    }

    cudaEvent_t &get() { return ev_; }

   private:
    cudaEvent_t ev_;
};

class CudaTimer {
   public:
    void start() { CUDA_CALL(cudaEventRecord(start_.get(), 0)); }

    void stop() {
        CUDA_CALL(cudaEventRecord(end_.get(), 0));
        CUDA_CALL(cudaEventSynchronize(end_.get()));
    }

    void reset() {
        start_.reset();
        end_.reset();
    }

    // Returns the time in ms
    double get_time() {
        float time{};
        CUDA_CALL(cudaEventElapsedTime(&time, start_.get(), end_.get()));
        return time;
    }

   private:
    cuda_event start_;
    cuda_event end_;
};


