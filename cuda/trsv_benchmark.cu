#include <cmath>
#include <ios>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <vector>


#include "memory.cuh"
#include "trsv_kernels.cuh"
#include "trsv_memory.cuh"
#include "utils.cuh"


int main(int argc, char **argv)
{
    using ar_type = double;
    using st_type = float;

    constexpr tmtx_t t_matrix_type = tmtx_t::upper;
    constexpr dmtx_t d_matrix_type = dmtx_t::unit;

    constexpr std::size_t default_max_size{24 * 1000};
    constexpr std::size_t min_size{100};

    std::size_t max_size{default_max_size};
    bool measure_error{false};

    const std::string use_error_string("--error");
    const std::string set_size_string("--size");

    auto print_usage = [&]() {
        const std::string binary(argv[0]);
        std::cerr << "Usage: " << binary << " [" << use_error_string << "] "
                  << '[' << set_size_string << "=SIZE"
                  << "]\n";
        std::cerr << "With:\n"
                  << use_error_string << ":    compute errors of TRSV\n"
                  << set_size_string
                  << ":     set the maximum size (used for both rows and cols) "
                     "of the matrix. Default value: "
                  << default_max_size << "; Min value: " << min_size << '\n'
                  << "Without parameters: benchmark different TRSVs\n";
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

    constexpr char DELIM{';'};

    std::default_random_engine rengine(42);
    std::uniform_real_distribution<ar_type> mtx_dist(-1.0, 1.0);
    auto vector_dist = mtx_dist;

    auto cpu_mtx_gen = [&](matrix_info m_info) {
        return gen_mtx<ar_type>(m_info, mtx_dist, rengine);
    };
    auto cpu_vect_gen = [&](matrix_info v_info) {
        return gen_mtx<ar_type>(v_info, vector_dist, rengine);
    };

    // Allocate host and device memory
    auto ar_data =
        TrsvMemory<ar_type>(max_size, cpu_mtx_gen, cpu_vect_gen);
    auto st_data = TrsvMemory<st_type>(ar_data);

    auto cublasHandle = cublas_get_handle();

    // Additional memory and lambdas for error computation
    const auto max_res_num_elems = ar_data.cpu_x_memory().get_num_elems();
    Memory<ar_type> cpu_x_ref(Memory<ar_type>::Device::cpu, max_res_num_elems);
    ar_type res_ref_norm{1.0};
    Memory<ar_type> reduce_memory(Memory<ar_type>::Device::cpu,
                                     max_res_num_elems);
    Memory<std::uint32_t> trsv_helper(Memory<std::uint32_t>::Device::gpu, 2);

    auto ar_compute_error = [&](matrix_info x_info) {
        ar_type error{};
        ar_data.sync_x();
        error = compare(x_info, cpu_x_ref.const_data(), ar_data.cpu_x_const(),
                        reduce_memory.data());
        ar_data.reset_x();
        return error / res_ref_norm;
    };
    auto st_compute_error = [&](matrix_info x_info) {
        ar_type error{};
        st_data.sync_x();
        error = compare(x_info, cpu_x_ref.const_data(), st_data.cpu_x_const(),
                        reduce_memory.data());
        st_data.reset_x();
        return error / res_ref_norm;
    };

    // Setting up names and associated benchmark and error functions

    constexpr std::size_t benchmark_reference{0};
    using benchmark_info_t =
        std::tuple<std::string, std::function<void(matrix_info, matrix_info)>,
                   std::function<ar_type(matrix_info)>>;
    // This vector contains all necessary information to perform the benchmark.
    // First, the name of the benchmark, second, a lambda taking the matrix and
    // the vector information which then runs the corresponding kernel
    std::vector<benchmark_info_t> benchmark_info = {
        benchmark_info_t{"TRSV fp64",
                         [&](matrix_info m_info, matrix_info x_info) {
                             trsv(m_info, t_matrix_type, d_matrix_type,
                                  ar_data.gpu_mtx_const(), x_info,
                                  ar_data.gpu_x(), trsv_helper.data());
                         },
                         ar_compute_error},
        benchmark_info_t{"TRSV fp32",
                         [&](matrix_info m_info, matrix_info x_info) {
                             trsv(m_info, t_matrix_type, d_matrix_type,
                                  st_data.gpu_mtx_const(), x_info,
                                  st_data.gpu_x(), trsv_helper.data());
                         },
                         st_compute_error},
        benchmark_info_t{"TRSV Acc<fp64, fp64>",
                         [&](matrix_info m_info, matrix_info x_info) {
                             acc_trsv<ar_type>(
                                 m_info, t_matrix_type, d_matrix_type,
                                 ar_data.gpu_mtx_const(), x_info,
                                 ar_data.gpu_x(), trsv_helper.data());
                         },
                         ar_compute_error},
        benchmark_info_t{"TRSV Acc<fp64, fp32>",
                         [&](matrix_info m_info, matrix_info x_info) {
                             acc_trsv<ar_type>(
                                 m_info, t_matrix_type, d_matrix_type,
                                 st_data.gpu_mtx_const(), x_info,
                                 st_data.gpu_x(), trsv_helper.data());
                         },
                         st_compute_error},
        benchmark_info_t{"TRSV Acc<fp32, fp32>",
                         [&](matrix_info m_info, matrix_info x_info) {
                             acc_trsv<st_type>(
                                 m_info, t_matrix_type, d_matrix_type,
                                 st_data.gpu_mtx_const(), x_info,
                                 st_data.gpu_x(), trsv_helper.data());
                         },
                         st_compute_error},
        benchmark_info_t{"CUBLAS TRSV fp64",
                         [&](matrix_info m_info, matrix_info x_info) {
                             cublas_trsv(cublasHandle.get(), t_matrix_type,
                                         d_matrix_type, m_info,
                                         ar_data.gpu_mtx_const(), x_info,
                                         ar_data.gpu_x());
                         },
                         ar_compute_error},
        benchmark_info_t{"CUBLAS TRSV fp32",
                         [&](matrix_info m_info, matrix_info x_info) {
                             cublas_trsv(cublasHandle.get(), t_matrix_type,
                                         d_matrix_type, m_info,
                                         st_data.gpu_mtx_const(), x_info,
                                         st_data.gpu_x());
                         },
                         st_compute_error},
    };
    const std::size_t benchmark_num{benchmark_info.size()};

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

    const std::size_t start = std::min(max_size, min_size);
    const std::size_t row_incr = start;

    for (auto num_rows = start; num_rows <= max_size; num_rows += row_incr) {
        const matrix_info m_info{{num_rows, num_rows}, max_size};
        const matrix_info x_info{{num_rows, 1}};

        if (measure_error) {
            std::get<1>(benchmark_info[benchmark_reference])(m_info, x_info);
            ar_data.sync_x();
            cpu_x_ref = ar_data.cpu_x_memory();
            res_ref_norm = reduce<ar_type>(
                x_info, cpu_x_ref.data(),
                [](ar_type a, ar_type b) { return std::abs(a) + std::abs(b); });
            // copy again since the reduce operation overwrites
            cpu_x_ref = ar_data.cpu_x_memory();
            ar_data.reset_x();
        }
        for (std::size_t i = 0; i < benchmark_num; ++i) {
            auto local_func = [&]() {
                std::get<1>(benchmark_info[i])(m_info, x_info);
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
