#include <array>
#include <cmath>
#include <ios>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "gemv_kernels.cuh"
#include "gemv_memory.cuh"
#include "memory.cuh"
#include "utils.cuh"

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
                  << ":    compute error of GEMVs\n"
                  << "Without parameters: benchmark different GEMVs\n";
        return 1;
    }

    const bool reset_result{measure_error && ar_beta != 0};
    const bool normalize_error{true};

    constexpr std::size_t max_rows{24500};
    // constexpr std::size_t max_rows{6500};
    // constexpr std::size_t max_rows{8 * 1024};
    constexpr std::size_t max_cols{max_rows};
    constexpr char DELIM{';'};

    std::default_random_engine rengine(42);
    std::uniform_real_distribution<value_type> mtx_dist(-1.0, 1.0);
    // std::uniform_real_distribution<value_type> mtx_dist(-10.0, 10.0);
    auto vector_dist = mtx_dist;

    auto ar_data =
        GemvMemory<ar_type>(max_rows, max_cols, mtx_dist, vector_dist, rengine);
    auto st_data = GemvMemory<st_type>(ar_data);

    auto cublasHandle = cublas_get_handle();

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
    auto ar_compute_error = [&](matrix_info x_info) {
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
    auto st_compute_error = [&](matrix_info x_info) {
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

    std::cout.precision(16);
    std::cout << std::scientific;

    constexpr std::size_t benchmark_num{7};
    constexpr std::size_t benchmark_reference{0};  //{benchmark_num - 2};
    using benchmark_info_t =
        std::tuple<std::string,
                   std::function<void(matrix_info, matrix_info, matrix_info)>,
                   std::function<value_type(matrix_info)>>;
    std::array<benchmark_info_t, benchmark_num> benchmark_info = {
        benchmark_info_t{
            "GEMV fp64",
            [&](matrix_info m_info, matrix_info x_info, matrix_info res_info) {
                gemv(m_info, ar_alpha, ar_data.gpu_mtx_const(), x_info,
                     ar_data.gpu_x_const(), res_info, ar_beta,
                     ar_data.gpu_res());
            },
            ar_compute_error},
        benchmark_info_t{
            "GEMV fp32",
            [&](matrix_info m_info, matrix_info x_info, matrix_info res_info) {
                gemv(m_info, st_alpha, st_data.gpu_mtx_const(), x_info,
                     st_data.gpu_x_const(), res_info, st_beta,
                     st_data.gpu_res());
            },
            st_compute_error},
        benchmark_info_t{
            "GEMV Acc<fp64, fp64>",
            [&](matrix_info m_info, matrix_info x_info, matrix_info res_info) {
                acc_gemv<ar_type>(m_info, ar_alpha, ar_data.gpu_mtx_const(),
                                  x_info, ar_data.gpu_x_const(), res_info,
                                  ar_beta, ar_data.gpu_res());
            },
            ar_compute_error},
        benchmark_info_t{
            "GEMV Acc<fp64, fp32>",
            [&](matrix_info m_info, matrix_info x_info, matrix_info res_info) {
                acc_gemv<ar_type>(m_info, st_alpha, st_data.gpu_mtx_const(),
                                  x_info, st_data.gpu_x_const(), res_info,
                                  st_beta, st_data.gpu_res());
            },
            st_compute_error},
        benchmark_info_t{
            "GEMV Acc<fp32, fp32>",
            [&](matrix_info m_info, matrix_info x_info, matrix_info res_info) {
                acc_gemv<st_type>(m_info, st_alpha, st_data.gpu_mtx_const(),
                                  x_info, st_data.gpu_x_const(), res_info,
                                  st_beta, st_data.gpu_res());
            },
            st_compute_error},
        benchmark_info_t{
            "CUBLAS GEMV fp64",
            [&](matrix_info m_info, matrix_info x_info, matrix_info res_info) {
                cublas_gemv(cublasHandle.get(), m_info, ar_alpha,
                            ar_data.gpu_mtx_const(), x_info,
                            ar_data.gpu_x_const(), res_info, ar_beta,
                            ar_data.gpu_res());
            },
            ar_compute_error},
        benchmark_info_t{
            "CUBLAS GEMV fp32",
            [&](matrix_info m_info, matrix_info x_info, matrix_info res_info) {
                cublas_gemv(cublasHandle.get(), m_info, st_alpha,
                            st_data.gpu_mtx_const(), x_info,
                            st_data.gpu_x_const(), res_info, st_beta,
                            st_data.gpu_res());
            },
            st_compute_error},
    };

    std::cout << "Num rows";
    for (const auto &info : benchmark_info) {
        if (!measure_error) {
            std::cout << DELIM << std::get<0>(info);
        } else {
            std::cout << DELIM << "Error " << std::get<0>(info);
        }
    }
    std::cout << '\n';

    std::cout.precision(16);
    // showpos: show + sign for positive numbers
    std::cout << std::scientific << std::showpos;
    constexpr auto start = 100;  // max_rows / 48;
    constexpr auto row_incr = start;  // start;
    for (std::size_t num_rows = start; num_rows <= max_rows;
         num_rows += row_incr) {
        std::array<value_type, benchmark_num> local_res{};
        const matrix_info m_info{{num_rows, num_rows}, max_rows};
        const matrix_info x_info{{num_rows, 1}};
        const matrix_info res_info{{num_rows, 1}};

        if (measure_error) {
            std::get<1>(benchmark_info[benchmark_reference])(m_info, x_info,
                                                             res_info);
            cpu_res_ref = ar_data.gpu_res_memory();
            if (normalize_error) {
                res_ref_norm = reduce<value_type>(
                    res_info, cpu_res_ref.data(), [](ar_type a, ar_type b) {
                        return std::abs(a) + std::abs(b);
                    });
                // copy again since the reduce operation overwrites
                cpu_res_ref = ar_data.gpu_res_memory();
                ar_data.gpu_res_memory() = ar_cpu_res_init;
            }
        }
        for (std::size_t i = 0; i < benchmark_num; ++i) {
            auto local_func = [&]() {
                std::get<1>(benchmark_info[i])(m_info, x_info, res_info);
            };
            if (!measure_error) {
                local_res[i] = benchmark_function(local_func, measure_error);
            } else {
                benchmark_function(local_func, measure_error);
                local_res[i] = std::get<2>(benchmark_info[i])(x_info);
            }
        }

        std::cout << num_rows;
        for (const auto &res : local_res) {
            std::cout << DELIM << res;
        }
        std::cout << '\n';
    }
}
