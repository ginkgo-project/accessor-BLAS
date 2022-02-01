#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>


#include "dot_kernels.cuh"
#include "dot_memory.cuh"
#include "memory.cuh"
#include "utils.cuh"


int main(int argc, char **argv)
{
    using ar_type = double;
    using st_type = float;
    using size_type = matrix_info::size_type;

    constexpr size_type min_size{1'000'000};
    constexpr size_type default_max_size{535 * 1000 * 1000};
    constexpr char DELIM{';'};

    bool detailed_error{false};
    size_type max_size{default_max_size};

    const std::string use_error_string("--error");
    const std::string set_size_string("--size");

    auto print_usage = [&]() {
        const std::string binary(argv[0]);
        std::cerr << "Usage: " << binary << " [" << use_error_string << "] "
                  << '[' << set_size_string << "=SIZE"
                  << "]\n";
        std::cerr << "With:\n"
                  << use_error_string
                  << ":    compute detailed error of the DOTs\n"
                  << set_size_string
                  << ":     set the maximum size of a vector. Default value: "
                  << default_max_size << "; Min value: " << min_size << '\n'
                  << "Without parameters: benchmark different DOTs\n";
    };

    // Process the input arguments
    for (int i = 1; i < argc; ++i) {
        const std::string current(argv[i]);
        if (current == use_error_string) {
            detailed_error = true;
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
        std::cerr << "The vector size needs to be at least " << min_size
                  << '\n';
        return 1;
    }


    std::default_random_engine rengine(42);
    std::uniform_real_distribution<ar_type> vector_dist(-1.0, 1.0);

    // Allocate host and device memory
    auto ar_data = DotMemory<ar_type>(max_size, vector_dist, rengine);
    auto st_data = DotMemory<st_type>(ar_data);

    auto cublas_handle = cublas_get_handle();
    cublas_set_device_ptr_mode(cublas_handle.get());

    auto my_handle = std::make_unique<myBlasHandle>();

    auto ar_get_result = [&ar_data]() { return ar_data.get_result(); };
    auto st_get_result = [&st_data]() {
        return static_cast<ar_type>(st_data.get_result());
    };

    constexpr size_type benchmark_reference{0};
    using benchmark_info_t =
        std::tuple<std::string, std::function<void(matrix_info, matrix_info)>,
                   std::function<ar_type()>>;
    // This vector contains all necessary information to perform the benchmark.
    // First, the name of the benchmark, second, a lambda taking the x and y
    // information of the vectors which then runs the corresponding kernel
    std::vector<benchmark_info_t> benchmark_info = {
        benchmark_info_t{"DOT fp64",
                         [&](matrix_info x_info, matrix_info y_info) {
                             dot(my_handle.get(), x_info, ar_data.gpu_x(),
                                 y_info, ar_data.gpu_y(), ar_data.gpu_res());
                         },
                         ar_get_result},
        benchmark_info_t{"DOT fp32",
                         [&](matrix_info x_info, matrix_info y_info) {
                             dot(my_handle.get(), x_info, st_data.gpu_x(),
                                 y_info, st_data.gpu_y(), st_data.gpu_res());
                         },
                         st_get_result},
        benchmark_info_t{"DOT Acc<fp64, fp64>",
                         [&](matrix_info x_info, matrix_info y_info) {
                             acc_dot<ar_type>(
                                 my_handle.get(), x_info, ar_data.gpu_x(),
                                 y_info, ar_data.gpu_y(), ar_data.gpu_res());
                         },
                         ar_get_result},
        benchmark_info_t{"DOT Acc<fp64, fp32>",
                         [&](matrix_info x_info, matrix_info y_info) {
                             acc_dot<ar_type>(
                                 my_handle.get(), x_info, st_data.gpu_x(),
                                 y_info, st_data.gpu_y(), st_data.gpu_res());
                         },
                         st_get_result},
        benchmark_info_t{"DOT Acc<fp32, fp32>",
                         [&](matrix_info x_info, matrix_info y_info) {
                             acc_dot<st_type>(
                                 my_handle.get(), x_info, st_data.gpu_x(),
                                 y_info, st_data.gpu_y(), st_data.gpu_res());
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
    const size_type benchmark_num{static_cast<size_type>(benchmark_info.size())};


    std::cout << "Vector Size";
    if (!detailed_error) {
        for (const auto &info : benchmark_info) {
            std::cout << DELIM << std::get<0>(info);
        }
    }
    for (const auto &info : benchmark_info) {
        std::cout << DELIM << "Error " << std::get<0>(info);
    }
    std::cout << '\n';

    std::cout.precision(16);
    std::cout << std::scientific;

    // Helper lambda to compute the error, provided the actual result and a
    // reference result
    auto get_error = [](ar_type res, ar_type ref_res) -> ar_type {
        return std::abs(res - ref_res) / std::abs(ref_res);
    };

    // Number of elements of a vector at the start of the benchmark
    const size_type start = std::min(max_size, min_size);
    // Increase in number of elements between consecutive benchmark runs
    constexpr size_type row_incr = 2'000'000;
    // Number of benchmark runs (ignoring randomization)
    const size_type steps =
        (max_size < start) ? 0 : (max_size - start) / row_incr;
    // Number of benchmark restarts with a different randomization for vectors
    // Only used for a detailed error run
    constexpr size_type max_randomize_num{10};

    std::vector<size_type> benchmark_vec_size((steps + 1));
    std::vector<double> benchmark_time((steps + 1) * benchmark_num);
    // std::vector<ar_type> benchmark_error((steps + 1) * benchmark_num);
    // stores the result for all different benchmark runs to compute the error
    const auto actual_randomize_num = detailed_error ? max_randomize_num : 1;
    std::vector<ar_type> raw_result(actual_randomize_num * (steps + 1) *
                                       benchmark_num);
    const auto get_raw_idx = [benchmark_num, actual_randomize_num](
                                 size_type rnd, size_type step,
                                 size_type bi) {
        return step * actual_randomize_num * benchmark_num +
               bi * actual_randomize_num + rnd;
    };

    // Run all benchmarks and collect the raw data here
    for (size_type randomize = 0; randomize < actual_randomize_num;
         ++randomize) {
        if (randomize != 0) {
            write_random({{max_size, 1}}, vector_dist, rengine,
                         ar_data.cpu_x_nc());
            write_random({{max_size, 1}}, vector_dist, rengine,
                         ar_data.cpu_y_nc());
            ar_data.copy_cpu_to_gpu();
            st_data.convert_from(ar_data);
        }
        for (size_type vec_size = start, i = 0; vec_size <= max_size;
             vec_size += row_incr, ++i) {
            benchmark_vec_size.at(i) = vec_size;
            const matrix_info x_info{{vec_size, 1}};
            const matrix_info y_info{{vec_size, 1}};

            for (size_type bi = 0; bi < benchmark_num; ++bi) {
                const size_type idx = i * benchmark_num + bi;
                auto curr_lambda = [&]() {
                    std::get<1>(benchmark_info[bi])(x_info, y_info);
                };
                benchmark_time.at(idx) =
                    benchmark_function(curr_lambda, detailed_error);
                raw_result[get_raw_idx(randomize, i, bi)] =
                    std::get<2>(benchmark_info[bi])();
            }
        }
    }

    // Print the evaluated results
    for (size_type i = 0; i <= steps; ++i) {
        if (!detailed_error) {
            std::cout << benchmark_vec_size[i];
            for (size_type bi = 0; bi < benchmark_num; ++bi) {
                std::cout << DELIM << benchmark_time[i * benchmark_num + bi];
            }
            const auto result_ref =
                raw_result[get_raw_idx(0, i, benchmark_reference)];
            for (size_type bi = 0; bi < benchmark_num; ++bi) {
                std::cout << DELIM
                          << get_error(raw_result[i * benchmark_num + bi],
                                       result_ref);
            }
            std::cout << '\n';
        } else {
            std::cout << benchmark_vec_size[i];
            for (size_type bi = 0; bi < benchmark_num; ++bi) {
                // sort and compute the median
                std::array<ar_type, max_randomize_num> local_error;
                for (size_type rnd = 0; rnd < actual_randomize_num; ++rnd) {
                    const auto result_ref =
                        raw_result[get_raw_idx(rnd, i, benchmark_reference)];
                    local_error[rnd] = get_error(
                        raw_result[get_raw_idx(rnd, i, bi)], result_ref);
                }
                // Compute the median of the error
                std::sort(local_error.begin(), local_error.end());
                ar_type median{};
                if (actual_randomize_num % 2 == 1) {
                    median = local_error[actual_randomize_num / 2];
                } else {
                    const auto begin_middle = actual_randomize_num / 2 - 1;
                    median = (local_error[begin_middle] +
                              local_error[begin_middle + 1]) /
                             2.0;
                }
                std::cout << DELIM << median;
            }
            std::cout << '\n';
        }
    }
    if (!detailed_error) {
        return 0;
    }
    // Additionally, print the actual result of the DOT for each computed
    // instance.
    std::cout << "--------------------------------------------------\n";
    std::cout << "Random iter" << DELIM << "Vector Size";
    for (const auto &info : benchmark_info) {
        std::cout << DELIM << "Result " << std::get<0>(info);
    }
    std::cout << '\n';
    for (size_type i = 0; i <= steps; ++i) {
        for (size_type randomize = 0; randomize < actual_randomize_num;
             ++randomize) {
            std::cout << randomize << DELIM << benchmark_vec_size[i];
            for (size_type bi = 0; bi < benchmark_num; ++bi) {
                std::cout << DELIM << raw_result[get_raw_idx(randomize, i, bi)];
            }
            std::cout << '\n';
        }
    }
}
