#include <cmath>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>


#include "dot_kernels.cuh"
#include "dot_memory.cuh"
#include "memory.cuh"
#include "utils.cuh"


int main(int argc, char **argv)
{
    using ar_type = double;
    using st_type = float;
    using value_type = ar_type;

    constexpr std::size_t max_size{535 * 1000 * 1000};
    constexpr char DELIM{';'};

    bool detailed_error{false};

    const std::string use_error_string("--error");
    if (argc == 2 && std::string(argv[1]) == use_error_string) {
        detailed_error = true;
    } else if (argc > 1) {
        const std::string binary(argv[0]);
        std::cerr << "Unsupported parameters!\n";
        std::cerr << "Usage: " << binary << " [" << use_error_string << "]\n";
        std::cerr << "With " << use_error_string
                  << ":    compute detailed error of DOTs\n"
                  << "Without parameters: benchmark different DOTs\n";
        return 1;
    }
    std::default_random_engine rengine(42);
    std::uniform_real_distribution<value_type> vector_dist(0.0, 1.0);

    auto ar_data = DotMemory<ar_type>(max_size, vector_dist, rengine);
    auto st_data = DotMemory<st_type>(ar_data);

    auto cublas_handle = cublas_get_handle();
    cublas_set_device_ptr_mode(cublas_handle.get());

    cudaDeviceProp device_prop;
    CUDA_CALL(cudaGetDeviceProperties(&device_prop, 0));

    auto ar_get_result = [&ar_data]() { return ar_data.get_result(); };
    auto st_get_result = [&st_data]() {
        return static_cast<ar_type>(st_data.get_result());
    };

    constexpr std::size_t benchmark_reference{0};
    using benchmark_info_t =
        std::tuple<std::string, std::function<void(matrix_info, matrix_info)>,
                   std::function<value_type()>>;
    std::vector<benchmark_info_t> benchmark_info = {
        benchmark_info_t{"DOT fp64",
                         [&](matrix_info x_info, matrix_info y_info) {
                             dot(device_prop, x_info, ar_data.gpu_x(), y_info,
                                 ar_data.gpu_y(), ar_data.gpu_res());
                         },
                         ar_get_result},
        benchmark_info_t{"DOT fp32",
                         [&](matrix_info x_info, matrix_info y_info) {
                             dot(device_prop, x_info, st_data.gpu_x(), y_info,
                                 st_data.gpu_y(), st_data.gpu_res());
                         },
                         st_get_result},
        benchmark_info_t{"DOT Acc<fp64, fp64>",
                         [&](matrix_info x_info, matrix_info y_info) {
                             acc_dot(device_prop, x_info, ar_data.gpu_x(),
                                     y_info, ar_data.gpu_y(),
                                     ar_data.gpu_res());
                         },
                         ar_get_result},
        benchmark_info_t{"DOT Acc<fp64, fp32>",
                         [&](matrix_info x_info, matrix_info y_info) {
                             acc_dot(device_prop, x_info, st_data.gpu_x(),
                                     y_info, st_data.gpu_y(),
                                     ar_data.gpu_res());
                         },
                         ar_get_result},
        benchmark_info_t{"DOT Acc<fp32, fp32>",
                         [&](matrix_info x_info, matrix_info y_info) {
                             acc_dot(device_prop, x_info, st_data.gpu_x(),
                                     y_info, st_data.gpu_y(),
                                     st_data.gpu_res());
                         },
                         st_get_result},
        benchmark_info_t{"CUBLAS DOT fp64",
                         [&](matrix_info x_info, matrix_info y_info) {
                             cublas_dot(cublas_handle.get(), x_info,
                                        ar_data.gpu_x(), y_info,
                                        ar_data.gpu_y(), ar_data.gpu_res());
                         },
                         ar_get_result},
        benchmark_info_t{"CUBLAS DOT fp32",
                         [&](matrix_info x_info, matrix_info y_info) {
                             cublas_dot(cublas_handle.get(), x_info,
                                        st_data.gpu_x(), y_info,
                                        st_data.gpu_y(), st_data.gpu_res());
                         },
                         st_get_result}};
    const std::size_t benchmark_num{benchmark_info.size()};

    std::cout << "Distribution vector: [" << vector_dist.a() << ','
              << vector_dist.b() << ")\n";


    std::cout << "Vector Size";
    if (!detailed_error) {
        for (const auto &info : benchmark_info) {
            std::cout << DELIM << std::get<0>(info);
        }
        for (const auto &info : benchmark_info) {
            std::cout << DELIM << "Error " << std::get<0>(info);
        }
    } else {
        for (const auto &info : benchmark_info) {
            std::cout << DELIM << "Error " << std::get<0>(info);
        }
    }
    std::cout << '\n';

    std::cout.precision(16);
    std::cout << std::scientific;

    auto get_error = [](value_type res, value_type ref_res) -> value_type {
        return std::abs(res - ref_res) / std::abs(ref_res);
    };

    // Number of elements of a vector at the start of the benchmark
    constexpr std::size_t start = std::min(max_size, std::size_t{1'000'000});
    // Increase in number of elements between consecutive benchmark runs
    constexpr std::size_t row_incr = 2'000'000;
    // Number of benchmark runs (ignoring randomization)
    constexpr std::size_t steps = (max_size - start) / row_incr;
    // Number of benchmark restarts with a different randomization for vectors
    // Only used for a detailed error run
    constexpr std::size_t randomize_num{10};

    std::vector<std::size_t> benchmark_vec_size((steps + 1));
    std::vector<double> benchmark_time((steps + 1) * benchmark_num);
    std::vector<value_type> benchmark_error((steps + 1) * benchmark_num);
    // stores the result for all different benchmark runs to compute the error
    std::vector<value_type> raw_result(benchmark_num);

    for (std::size_t randomize = 0;
         (detailed_error && randomize < randomize_num) ||
         (!detailed_error && randomize < 1);
         ++randomize) {
        if (randomize != 0) {
            write_random({{max_size, 1}}, vector_dist, rengine,
                         ar_data.cpu_x_nc());
            write_random({{max_size, 1}}, vector_dist, rengine,
                         ar_data.cpu_y_nc());
            ar_data.copy_cpu_to_gpu();
            st_data.convert_from(ar_data);
        }
        for (std::size_t vec_size = start, i = 0; vec_size <= max_size;
             vec_size += row_incr, ++i) {
            benchmark_vec_size.at(i) = vec_size;
            const matrix_info x_info{{vec_size, 1}};
            const matrix_info y_info{{vec_size, 1}};

            for (std::size_t bi = 0; bi < benchmark_num; ++bi) {
                const std::size_t idx = i * benchmark_num + bi;
                auto curr_lambda = [&]() {
                    std::get<1>(benchmark_info[bi])(x_info, y_info);
                };
                benchmark_time.at(idx) =
                    benchmark_function(curr_lambda, detailed_error);
                raw_result[bi] = std::get<2>(benchmark_info[bi])();
            }
            const value_type result_ref = raw_result[benchmark_reference];
            for (std::size_t bi = 0; bi < benchmark_num; ++bi) {
                const std::size_t idx = i * benchmark_num + bi;
                benchmark_error.at(idx) +=
                    get_error(raw_result[bi], result_ref);
            }
        }
    }
    for (std::size_t i = 0; i <= steps; ++i) {
        if (!detailed_error) {
            std::cout << benchmark_vec_size[i];
            for (std::size_t bi = 0; bi < benchmark_num; ++bi) {
                std::cout << DELIM << benchmark_time[i * benchmark_num + bi];
            }
            for (std::size_t bi = 0; bi < benchmark_num; ++bi) {
                std::cout << DELIM << benchmark_error[i * benchmark_num + bi];
            }
            std::cout << '\n';
        } else {
            std::cout << benchmark_vec_size[i];
            for (std::size_t bi = 0; bi < benchmark_num; ++bi) {
                std::cout << DELIM
                          << benchmark_error[i * benchmark_num + bi] /
                                 static_cast<value_type>(randomize_num);
            }
            std::cout << '\n';
        }
    }
}
