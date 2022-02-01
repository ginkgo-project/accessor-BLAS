#pragma once

#include <array>
#include <cassert>
#include <cinttypes>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>


#include <cublas_v2.h>


/**
 * Contains information about a row-major matrix.
 */
struct matrix_info {
    using size_type = std::int64_t;
    // 2D size of the matrix
    const std::array<size_type, 2> size;
    // stride used for the rows.
    const size_type stride;

    /**
     * Sets the given size and stride.
     *
     * @param size  2D size
     * @param stride  stride (must be larger than size[1])
     */
    constexpr matrix_info(const std::array<size_type, 2> size,
                          const size_type stride)
        : size(size), stride{stride}
    {}

    /**
     * Sets the given size and stride.
     *
     * @param size  2D size
     * @param stride  stride (must be larger than size[1])
     */
    constexpr matrix_info(const std::array<size_type, 2> size)
        : matrix_info{size, size[1]}
    {}

    /**
     * @returns the total amount of elements (including elements in the stride)
     *          this matrix occupies.
     */
    size_type get_1d_size() const { return size[0] * stride; }
    /**
     * @returns the number of elements the matrix has (does not consider the
     *          stride)
     */
    size_type get_num_elems() const { return size[0] * size[1]; }
};


/**
 * Performs a ceil division between two positive numbers.
 *
 * @note if one of the numbers are not positive, the result will be inaccurate
 *
 * @tparam ValueType  value type of both operands
 *
 * @param a  divided
 * @param b  divisor
 *
 * @returns the ceil division of a/b
 */
template <typename ValueType>
constexpr ValueType ceildiv(ValueType a, ValueType b)
{
    return (a <= 0) ? a / b : (a - 1) / b + 1;
}

///////////// GPU relevant code \\\\\\\\\\\\\


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


/**
 * Synchronizes the device to the host (it waits until the device is done with
 * its actions).
 */
void synchronize() { CUDA_CALL(cudaDeviceSynchronize()); }


/**
 * RAII wrapper for a CUDA event
 */
struct cuda_event {
public:
    /**
     * Creates a CUDA event
     */
    cuda_event() { CUDA_CALL(cudaEventCreate(&ev_)); }

    ~cuda_event() { cudaEventDestroy(ev_); }

    /**
     * Resets a CUDA event by destroying and re-creating it.
     */
    void reset()
    {
        CUDA_CALL(cudaEventDestroy(ev_));
        CUDA_CALL(cudaEventCreate(&ev_));
    }

    /**
     * @returns the wrapped CUDA event
     */
    cudaEvent_t &get() { return ev_; }

private:
    cudaEvent_t ev_;
};


/**
 * Timer, utilizing CUDA events to measure time more accurately for GPU kernels
 */
class CudaTimer {
public:
    /**
     * Starts the timing
     */
    void start() { CUDA_CALL(cudaEventRecord(start_.get(), 0)); }

    /**
     * Stops the timing and waits for all operations to finish on the GPU
     */
    void stop()
    {
        CUDA_CALL(cudaEventRecord(end_.get(), 0));
        CUDA_CALL(cudaEventSynchronize(end_.get()));
    }

    /**
     * Resets the timer
     */
    void reset()
    {
        start_.reset();
        end_.reset();
    }

    /**
     * @returns the time in [ms] between the start() and stop() calls.
     */
    double get_time()
    {
        float time{};
        CUDA_CALL(cudaEventElapsedTime(&time, start_.get(), end_.get()));
        return time;
    }

private:
    cuda_event start_;
    cuda_event end_;
};

using CublasContext = std::remove_pointer_t<cublasHandle_t>;

/**
 * @returns a unique_ptr, wrapping a cublas handle to add RAII.
 */
std::unique_ptr<CublasContext, std::function<void(cublasHandle_t)>>
cublas_get_handle()
{
    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));
    CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    return {handle,
            [](cublasHandle_t handle) { CUBLAS_CALL(cublasDestroy(handle)); }};
}

/**
 * Sets the pointer mode of the given handle to host
 *
 * @param handle  cublas handle to set the pointer mode
 */
void cublas_set_host_ptr_mode(cublasHandle_t handle)
{
    CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
}

/**
 * Sets the pointer mode of the given handle to device
 *
 * @param handle  cublas handle to set the pointer mode
 */
void cublas_set_device_ptr_mode(cublasHandle_t handle)
{
    CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
}


/**
 * Benchmarks the given function func.
 *
 * @note This function utilizes CudaTimer to meaasure the time, so only GPU
 *       kernels can be benchmarked accurately with this function.
 *
 * @tparam Callable  type of the function to benchmark
 *
 * @param func  function to benchmark
 * @param skip  determines if a benchmark run is requested (false) or if the
 *              function is only supposed to be called once without measuring
 *              the time (true)
 *
 * @returns the time in [ms] the function takes to run. If skip was set to true,
 *          0 is returned.
 */
template <typename Callable>
double benchmark_function(Callable func, bool skip = false)
{
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
    return bench_iters == 0 ? double{} : result_ms;
}


/**
 * Reduces `tmp` in a binary tree fashion with the reduce operator `op`.
 * Overwrites `tmp` in the process.
 *
 * @tparam OutputType  Type of the result
 * @tparam InputType  Type of the input
 * @tparam ReduceOp  Callable type to the reduce operation
 *
 * @param info  Matrix information; Must be a vector (only a single column)
 * @param tmp  storage the reduction is performed upon. Data will be overwritten
 *             during the reduction
 * @param op  Reduce operation that is used internally. Must be able to take two
 *            InputType arguments.
 *
 * @returns the result of the reduction
 */
template <typename OutputType, typename InputType, typename ReduceOp>
OutputType reduce(const matrix_info info, InputType *tmp, ReduceOp op)
{
    // The given matrix must only have a single column!
    assert(info.size[1] == 1);
    std::int64_t end = info.size[0];
    for (std::int64_t halfway = ceildiv(info.size[0], std::int64_t{2});
         halfway > 1; halfway = ceildiv(halfway, std::int64_t{2})) {
        for (std::int64_t row = 0; row < halfway; ++row) {
            if (row + halfway < end) {
                const auto midx = row * info.stride;
                const auto midx2 = (row + halfway) * info.stride;
                tmp[midx] = op(tmp[midx], tmp[midx2]);
            }
        }
        end = halfway;
    }
    return static_cast<OutputType>(info.size[0] == 1 ? op(tmp[0], {})
                                                     : op(tmp[0], tmp[1]));
}


/**
 * Compares `mtx1` to `mtx2` by computing: $tmp = abs(mtx1 - mtx2)$, followed by
 * returning $norm1(tmp)$. Both matrices must only have a single column.
 *
 * @note tmp is overwritten in the norm process.
 *
 * @tparam ReferenceType  Type of the values of mtx1
 * @tparam OtherType  Type of the values of mtx2
 * @tparam ValueType  Type used to compute the difference and the norm
 *
 * @param info  Matrix information (must be usable for botht mtx1 and mtx2)
 */
template <typename ReferenceType, typename OtherType, typename ValueType>
ValueType compare(const matrix_info info, const ReferenceType *mtx1,
                  const OtherType *mtx2, ValueType *tmp)
{
    // The given matrix must only have a single column!
    assert(info.size[1] == 1);

    for (typename matrix_info::size_type row = 0; row < info.size[0]; ++row) {
        const auto midx = row * info.stride;
        const ValueType v1 = mtx1[midx];
        const ValueType v2 = mtx2[midx];
        const auto delta = std::abs(v1 - v2);
        tmp[midx] = delta;
    }

    return reduce<ValueType>(
        info, tmp, [](ValueType o1, ValueType o2) { return o1 + o2; });
}
