#pragma once

#include <cublas_v2.h>

#include <array>
#include <functional>
#include <memory>
#include <stdexcept>

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

#define CUBLAS_CALL(call)                                                 \
    do {                                                                  \
        auto err = call;                                                  \
        if (err != CUBLAS_STATUS_SUCCESS) {                               \
            std::cerr << "Cuda error in file " << __FILE__                \
                      << " L:" << __LINE__ << "; Error: " << err << '\n'; \
            throw std::runtime_error("Error");                            \
        }                                                                 \
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

///////////// GPU relevant code \\\\\\\\\\\\\


void synchronize() { CUDA_CALL(cudaDeviceSynchronize()); }

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

using CublasContext = std::remove_pointer_t<cublasHandle_t>;

std::unique_ptr<CublasContext, std::function<void(cublasHandle_t)>>
get_cublas_handle() {
    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));
    CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    // CUBLAS_POINTER_MODE_DEVICE
    return {handle,
            [](cublasHandle_t handle) { CUBLAS_CALL(cublasDestroy(handle)); }};
}

