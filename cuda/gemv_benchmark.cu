#include <cmath>
#include <ios>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>


#include "gemv_kernels.cuh"
#include "gemv_memory.cuh"
#include "memory.cuh"
#include "utils.cuh"


int main(int argc, char **argv)
{
    using ar_type = double;
    using st_type = float;
    using size_type = matrix_info::size_type;

    constexpr ar_type ar_alpha{1.0};
    constexpr ar_type ar_beta{1.0};
    constexpr st_type st_alpha{static_cast<st_type>(ar_alpha)};
    constexpr st_type st_beta{static_cast<st_type>(ar_beta)};

    constexpr size_type default_max_size{24500};
    constexpr size_type min_size{100};
    size_type max_size{default_max_size};

    bool measure_error{false};

    const std::string use_error_string("--error");
    const std::string set_size_string("--size");

    auto print_usage = [&]() {
        const std::string binary(argv[0]);
        std::cerr << "Usage: " << binary << " [" << use_error_string << "] "
                  << '[' << set_size_string << "=SIZE"
                  << "]\n";
        std::cerr << "With:\n"
                  << use_error_string << ":    compute errors of the GEMVs\n"
                  << set_size_string
                  << ":     set the maximum size (used for both rows and cols) "
                     "of the matrix. Default value: "
                  << default_max_size << "; Min value: " << min_size << '\n'
                  << "Without parameters: benchmark different GEMVs\n";
    };

    // Process the input arguments
    for (int i = 1; i < argc; ++i) {
        const std::string current(argv[i]);
        if (current == use_error_string) {
            measure_error = true;
        } else if (current.substr(0, set_size_string.size()) ==
                   set_size_string) {
            max_size = std::stoll(current.substr(set_size_string.size() + 1));
        } else {
            std::cerr << "Unsupported parameter: " << current << '\n';
            print_usage();
            return 1;
        }
    }
    if (max_size < min_size) {
        std::cerr << "The matrix size needs to be at least " << min_size
                  << '\n';
        return 1;
    }

    // The result vector only needs to be reset if we measure the error AND the
    // beta value is non-zero. Otherwise, the initial result values don't matter
    // for the computation.
    const bool reset_result{measure_error && ar_beta != 0};
    const bool normalize_error{true};

    constexpr char DELIM{';'};

    std::default_random_engine rengine(42);
    std::uniform_real_distribution<ar_type> mtx_dist(-1.0, 1.0);
    auto vector_dist = mtx_dist;

    // Allocate host and device memory
    auto ar_data =
        GemvMemory<ar_type>(max_size, mtx_dist, vector_dist, rengine);
    auto st_data = GemvMemory<st_type>(ar_data);

    auto cublasHandle = cublas_get_handle();

    // Additional memory and lambdas for error computation
    const auto max_res_num_elems = ar_data.cpu_res_memory().get_num_elems();
    Memory<ar_type> cpu_res_ref(Memory<ar_type>::Device::cpu,
                                max_res_num_elems);
    ar_type res_ref_norm{1.0};
    auto ar_cpu_res_init = ar_data.cpu_res_memory();
    auto st_cpu_res_init = st_data.cpu_res_memory();
    Memory<ar_type> reduce_memory(Memory<ar_type>::Device::cpu,
                                     max_res_num_elems);
    auto ar_compute_error = [&](matrix_info x_info) {
        ar_type error{};
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
        ar_type error{};
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

    constexpr size_type benchmark_reference{0};
    using benchmark_info_t =
        std::tuple<std::string,
                   std::function<void(matrix_info, matrix_info, matrix_info)>,
                   std::function<ar_type(matrix_info)>>;
    // This vector contains all necessary information to perform the benchmark.
    // First, the name of the benchmark, second, a lambda taking the matrix, x
    // and result information which then runs the corresponding kernel
    std::vector<benchmark_info_t> benchmark_info = {
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
    const size_type benchmark_num{static_cast<size_type>(benchmark_info.size())};

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

    std::vector<ar_type> local_res(benchmark_num);
    constexpr auto start = min_size;
    constexpr auto row_incr = start;
    for (size_type num_rows = start; num_rows <= max_size;
         num_rows += row_incr) {
        const matrix_info m_info{{num_rows, num_rows}, max_size};
        const matrix_info x_info{{num_rows, 1}};
        const matrix_info res_info{{num_rows, 1}};

        if (measure_error) {
            std::get<1>(benchmark_info[benchmark_reference])(m_info, x_info,
                                                             res_info);
            cpu_res_ref.copy_from(ar_data.gpu_res_memory());
            if (normalize_error) {
                res_ref_norm = reduce<ar_type>(
                    res_info, cpu_res_ref.data(), [](ar_type a, ar_type b) {
                        return std::abs(a) + std::abs(b);
                    });
                // copy again since the reduce operation overwrites
                cpu_res_ref.copy_from(ar_data.gpu_res_memory());
                ar_data.gpu_res_memory().copy_from(ar_cpu_res_init);
            }
        }
        for (size_type i = 0; i < benchmark_num; ++i) {
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
