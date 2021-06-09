#pragma once

#include <cublas_v2.h>

#include <array>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

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
            std::cerr << "CuBLAS error in file " << __FILE__              \
                      << " L:" << __LINE__ << "; Error: " << err << '\n'; \
            throw std::runtime_error(std::string("Error: ") +             \
                                     std::to_string(err));                \
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

template <typename ValueType>
constexpr ValueType ceildiv(ValueType a, ValueType b) {
    return (a <= 0) ? a / b : (a - 1) / b + 1;
}

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
cublas_get_handle() {
    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));
    CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    // CUBLAS_POINTER_MODE_DEVICE
    return {handle,
            [](cublasHandle_t handle) { CUBLAS_CALL(cublasDestroy(handle)); }};
}

void cublas_set_host_ptr_mode(cublasHandle_t handle) {
    CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
}

void cublas_set_device_ptr_mode(cublasHandle_t handle) {
    CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
}

template <typename Callable>
double benchmark_function(Callable func, bool skip = false) {
    constexpr int bench_iters{10};
    double time_ms[bench_iters];
    CudaTimer ctimer;
    // Warmup
    func();
    synchronize();
    if (skip) {
        return {};
    }
    for (int i = 0; i < bench_iters; ++i) {
        ctimer.start();
        func();
        ctimer.stop();
        time_ms[i] = ctimer.get_time();
        ctimer.reset();
    }

    // Reduce timings to one value
    double result_ms{std::numeric_limits<double>::max()};
    for (int i = 0; i < bench_iters; ++i) {
        result_ms = std::min(result_ms, time_ms[i]);
    }
    // result_ms /= static_cast<double>(bench_iters);
    return bench_iters == 0 ? double{} : result_ms;
}

