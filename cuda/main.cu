#include <array>
#include <cmath>
#include <ios>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>

//#include "../error_tobias.hpp"
#include "gemv_memory.cuh"
#include "kernels.cuh"
#include "matrix_helper.cuh"
#include "memory.cuh"
#include "utils.cuh"

// TODO: finish error computation

template <typename ValueType>
void control_gemv(const matrix_info m_info, ValueType alpha,
                  const ValueType *mtx, const matrix_info x_info,
                  ValueType beta, const ValueType *x, ValueType *res) {
    if (x_info.size[1] != 1) {
        throw "Error!";
    }
    for (std::size_t row = 0; row < m_info.size[0]; ++row) {
        ValueType local_res{0};
        for (std::size_t col = 0; col < m_info.size[1]; ++col) {
            const std::size_t midx = row * m_info.stride + col;
            local_res += mtx[midx] * x[col * x_info.stride];
        }
        auto res_idx = row * x_info.stride;
        res[res_idx] = beta * res[res_idx] + alpha * local_res;
    }
}

template <typename OutputType, typename InputType, typename ReduceOp>
OutputType reduce(const matrix_info info, InputType *tmp, ReduceOp op) {
    std::size_t end = info.size[0];
    for (std::size_t halfway = ceildiv(info.size[0], std::size_t{2});
         halfway > 1; halfway = ceildiv(halfway, std::size_t{2})) {
        for (std::size_t row = 0; row < halfway; ++row) {
            if (row + halfway < end) {
                for (std::size_t col = 0; col < info.size[1]; ++col) {
                    const std::size_t midx = row * info.stride + col;
                    const std::size_t midx2 =
                        (row + halfway) * info.stride + col;
                    tmp[midx] = op(tmp[midx], tmp[midx2]);
                }
            }
        }
        end = halfway;
    }
    return static_cast<OutputType>(info.size[0] == 1 ? op(tmp[0], {})
                                                     : op(tmp[0], tmp[1]));
}

template <typename T>
std::enable_if_t<std::is_floating_point<T>::value, T> get_value(T val) {
    return val;
}

template <typename T>
std::enable_if_t<!std::is_floating_point<T>::value, typename T::value_type>
get_value(T val) {
    return val.e;
}

template <typename ReferenceType, typename OtherType, typename ValueType>
ValueType compare(const matrix_info info, const ReferenceType *mtx1,
                  const OtherType *mtx2, ValueType *tmp) {
    using return_type = decltype(get_value(ReferenceType{}));
    static_assert(std::is_same<return_type, ValueType>::value,
                  "Types must match!");

    for (std::size_t row = 0; row < info.size[0]; ++row) {
        const std::size_t midx = row * info.stride;
        const auto v1 = get_value(mtx1[midx]);
        const decltype(v1) v2 = get_value(mtx2[midx]);
        if (std::is_floating_point<decltype(v1)>::value) {
            const auto delta = std::abs(v1 - v2);
            tmp[midx] = delta;
        } else {
            // only compute the 1-norm of the error
            const auto error = std::abs(v2);
            tmp[midx] = error;
        }
    }

    return reduce<ValueType>(
        info, tmp, [](ValueType o1, ValueType o2) { return o1 + o2; });
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

int main(int argc, char **argv) {
    /*
    using ar_type = error_number<double>;
    using st_type = error_number<float>;
    using value_type = ar_type::value_type;
    /*/
    using ar_type = double;
    using st_type = float;
    using value_type = ar_type;
    //*/

    constexpr ar_type ar_alpha{1.0};
    constexpr ar_type ar_beta{1.0};
    constexpr st_type st_alpha{static_cast<st_type>(ar_alpha)};
    constexpr st_type st_beta{static_cast<st_type>(ar_beta)};

    bool measure_error{false};

    const std::string use_error_string("--error");
    if (argc == 2 && std::string(argv[1]) == use_error_string) {
        measure_error = true;
    } else if (argc > 1) {
        const std::string binary(argv[0]);
        std::cerr << "Unsupported parameters!\n";
        std::cerr << "Usage: " << binary << " [" << use_error_string << "]\n";
        std::cerr << "With " << use_error_string
                  << ":    compute error of GeMVs\n"
                  << "Without parameters: benchmark different GeMVs\n";
        return 1;
    }

    const bool reset_result{measure_error && ar_beta != 0};
    const bool normalize_error{true};

    constexpr std::size_t max_rows{24 * 1024};
    constexpr std::size_t max_cols{max_rows};
    constexpr char DELIM{';'};

    std::default_random_engine rengine(42);
    std::uniform_real_distribution<value_type> mtx_dist(-2.0, 2.0);
    auto vector_dist = mtx_dist;

    auto ar_data =
        GemvMemory<ar_type>(max_rows, max_cols, mtx_dist, vector_dist, rengine);
    auto st_data = GemvMemory<st_type>(ar_data);

    auto cublasHandle = get_cublas_handle();

    static_assert(max_rows == max_cols, "Matrix must be square!");

    // Additional memory and lambdas for error computation
    const auto max_res_num_elems = ar_data.cpu_res_memory().get_num_elems();
    Memory<ar_type> cpu_res_ref(Memory<ar_type>::Device::cpu,
                                max_res_num_elems);
    value_type res_ref_norm{1.0};
    auto ar_cpu_res_init = ar_data.cpu_res_memory();
    auto st_cpu_res_init = st_data.cpu_res_memory();
    Memory<value_type> reduce_memory(Memory<ar_type>::Device::cpu,
                                     max_res_num_elems);
    auto ar_compute_error = [&](const matrix_info &x_info) {
        value_type error{};
        if (measure_error) {
            ar_data.sync_result();
            error = compare(x_info, cpu_res_ref.const_data(),
                            ar_data.cpu_res_const(), reduce_memory.data());
        }
        if (reset_result) {
            ar_data.gpu_res_memory() = ar_cpu_res_init;
        }
        return error / res_ref_norm;
    };
    auto st_compute_error = [&](const matrix_info &x_info) {
        value_type error{};
        if (measure_error) {
            st_data.sync_result();
            error = compare(x_info, cpu_res_ref.const_data(),
                            st_data.cpu_res_const(), reduce_memory.data());
        }
        if (reset_result) {
            st_data.gpu_res_memory() = st_cpu_res_init;
        }
        return error / res_ref_norm;
    };

    if (!measure_error) {
        std::cout << "Num Rows" << DELIM << "GEMV double" << DELIM
                  << "GEMV float" << DELIM << "GEMV Acc<fp64, fp64>" << DELIM
                  << "GEMV Acc<fp64, fp32>" << DELIM << "CUBLAS GEMV fp64"
                  << DELIM << "CUBLAS GEMV fp32" << '\n';
    } else {
        std::cout << "Num Rows" << DELIM << "Error GEMV double" << DELIM
                  << "Error GEMV float" << DELIM << "Error GEMV Acc<fp64, fp64>"
                  << DELIM << "Error GEMV Acc<fp64, fp32>" << DELIM
                  << "Error CUBLAS GEMV fp64" << DELIM
                  << "Error CUBLAS GEMV fp32" << '\n';
    }

    std::cout.precision(16);
    std::cout << std::scientific;

    constexpr auto start = max_rows / 48;
    constexpr auto row_incr = start;
    for (auto num_rows = start; num_rows <= max_rows; num_rows += row_incr) {
        const matrix_info m_info{{num_rows, num_rows}};
        const matrix_info x_info{{num_rows, 1}};

        double ar_time{};
        auto ar_func = [&]() {
            gemv(m_info, ar_alpha, ar_data.gpu_mtx_const(), x_info,
                 ar_data.gpu_x_const(), ar_beta, ar_data.gpu_res());
        };
        double st_time{};
        auto st_func = [&]() {
            gemv(m_info, st_alpha, st_data.gpu_mtx_const(), x_info,
                 st_data.gpu_x_const(), st_beta, st_data.gpu_res());
        };
        double acc_ar_time{};
        auto acc_ar_func = [&]() {
            acc_gemv<ar_type>(m_info, ar_alpha, ar_data.gpu_mtx_const(), x_info,
                              ar_data.gpu_x_const(), ar_beta,
                              ar_data.gpu_res());
        };
        double acc_mix_time{};
        auto acc_mix_func = [&]() {
            acc_gemv<ar_type>(m_info, ar_alpha, st_data.gpu_mtx_const(), x_info,
                              st_data.gpu_x_const(), ar_beta,
                              st_data.gpu_res());
        };
        double cublas_ar_time{};
        auto cublas_ar_func = [&]() {
            cublas_gemv(cublasHandle.get(), m_info, ar_alpha,
                        ar_data.gpu_mtx_const(), x_info, ar_data.gpu_x_const(),
                        ar_beta, ar_data.gpu_res());
        };
        double cublas_st_time{};
        auto cublas_st_func = [&]() {
            cublas_gemv(cublasHandle.get(), m_info, st_alpha,
                        st_data.gpu_mtx_const(), x_info, st_data.gpu_x_const(),
                        st_beta, st_data.gpu_res());
        };
        value_type ar_error{};  // [[gnu::unused, maybe_unused]]
        value_type st_error{};
        ar_type acc_ar_error{};
        value_type acc_mix_error{};
        value_type cublas_ar_error{};
        value_type cublas_st_error{};

        // control_gemv(m_info, v_matrix, x_info, v_b, v_res_ref);

        ar_time = benchmark_function(ar_func, measure_error);
        // Use the result here as the reference
        if (measure_error) {
            cpu_res_ref = ar_data.gpu_res_memory();
            if (normalize_error) {
                res_ref_norm = reduce<value_type>(
                    x_info, cpu_res_ref.data(), [](ar_type a, ar_type b) {
                        return std::abs(a) + std::abs(b);
                    });
                // copy again since reduce overwrites
                cpu_res_ref = ar_data.gpu_res_memory();
            }
        }
        ar_error = ar_compute_error(x_info);

        st_time = benchmark_function(st_func, measure_error);
        st_error = st_compute_error(x_info);

        acc_ar_time = benchmark_function(acc_ar_func, measure_error);
        acc_ar_error = ar_compute_error(x_info);

        acc_mix_time = benchmark_function(acc_mix_func, measure_error);
        acc_mix_error = st_compute_error(x_info);

        cublas_ar_time = benchmark_function(cublas_ar_func, measure_error);
        cublas_ar_error = ar_compute_error(x_info);

        cublas_st_time = benchmark_function(cublas_st_func, measure_error);
        cublas_st_error = st_compute_error(x_info);

        if (!measure_error) {
            std::cout << num_rows << DELIM << ar_time << DELIM << st_time
                      << DELIM << acc_ar_time << DELIM << acc_mix_time << DELIM
                      << cublas_ar_time << DELIM << cublas_st_time << '\n';
        } else {
            std::cout << num_rows << DELIM << ar_error << DELIM << st_error
                      << DELIM << acc_ar_error << DELIM << acc_mix_error
                      << DELIM << cublas_ar_error << DELIM << cublas_st_error
                      << '\n';
        }
    }
}
