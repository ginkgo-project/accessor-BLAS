#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <random>
#include <type_traits>

//#include "../error_tobias.hpp"
#include "dot_kernels.cuh"
#include "dot_memory.cuh"
#include "memory.cuh"
#include "utils.cuh"

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

    constexpr std::size_t max_size{512 * 1024 * 1024};
    // constexpr std::size_t max_size{128 * 1024 * 1024};
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
    // std::uniform_real_distribution<value_type> vector_dist(-100.0, 100.0);
    /*
    std::uniform_real_distribution<value_type> value_dist(1.0, 2.0);
    std::uniform_int_distribution<int> sign_dist(0, 1);
    auto vector_dist = [&value_dist, &sign_dist](auto &&engine) -> value_type{
        return static_cast<bool>(sign_dist(engine)) ? -value_dist(engine) :
    value_dist(engine);
    };
    /*/
    std::uniform_real_distribution<value_type> vector_dist(-1.0, 1.0);
    //*/

    auto ar_data = DotMemory<ar_type>(max_size, vector_dist, rengine);
    auto st_data = DotMemory<st_type>(ar_data);

    auto cublas_handle = cublas_get_handle();
    cublas_set_device_ptr_mode(cublas_handle.get());

    cudaDeviceProp device_prop;
    CUDA_CALL(cudaGetDeviceProperties(&device_prop, 0));
    // std::cout << "Number SMs: " << device_prop.multiProcessorCount << '\n';
    constexpr std::size_t benchmark_num{6};
    constexpr std::size_t benchmark_reference{0};
    std::array<std::string, benchmark_num> header_strings_time = {
        "DOT fp64",
        "DOT fp32",
        "DOT Acc<fp64, fp64>",
        "DOT Acc<fp64, fp32>",
        "CUBLAS DOT fp64",
        "CUBLAS DOT fp32"};
    std::array<std::string, benchmark_num> header_strings_error;
    for (std::size_t i = 0; i < header_strings_time.size(); ++i) {
        header_strings_error[i] =
            std::string("Error ") + header_strings_time[i];
    }

    std::array<std::function<void(matrix_info, matrix_info)>, benchmark_num>
        benchmark_lambdas = {
            [&](matrix_info x_info, matrix_info y_info) {
                dot(device_prop, x_info, ar_data.gpu_x(), y_info,
                    ar_data.gpu_y(), ar_data.gpu_res());
            },
            [&](matrix_info x_info, matrix_info y_info) {
                dot(device_prop, x_info, st_data.gpu_x(), y_info,
                    st_data.gpu_y(), st_data.gpu_res());
            },
            [&](matrix_info x_info, matrix_info y_info) {
                acc_dot(device_prop, x_info, ar_data.gpu_x(), y_info,
                        ar_data.gpu_y(), ar_data.gpu_res());
            },
            [&](matrix_info x_info, matrix_info y_info) {
                acc_dot(device_prop, x_info, st_data.gpu_x(), y_info,
                        st_data.gpu_y(), ar_data.gpu_res());
            },
            [&](matrix_info x_info, matrix_info y_info) {
                cublas_dot(cublas_handle.get(), x_info, ar_data.gpu_x(), y_info,
                           ar_data.gpu_y(), ar_data.gpu_res());
            },
            [&](matrix_info x_info, matrix_info y_info) {
                cublas_dot(cublas_handle.get(), x_info, st_data.gpu_x(), y_info,
                           st_data.gpu_y(), st_data.gpu_res());
            }};

    std::array<std::function<value_type()>, benchmark_num> benchmark_get_error =
        {[&]() {
            return ar_data.get_result(); },
         [&]() {
            return st_data.get_result(); },
         [&]() {
            return ar_data.get_result(); },
         [&]() {
            return ar_data.get_result(); },
         [&]() {
            return ar_data.get_result(); },
         [&]() {
            return st_data.get_result(); }};

    if (!detailed_error) {
        std::cout << "Vector Size";
        for (const auto &str : header_strings_time) {
            std::cout << DELIM << str;
        }
        for (const auto &str : header_strings_error) {
            std::cout << DELIM << str;
        }
        std::cout << '\n';
    } else {
        std::cout << "Vector Size";
        for (const auto &str : header_strings_error) {
            std::cout << DELIM << str;
        }
        std::cout << '\n';
    }

    std::cout.precision(16);
    std::cout << std::scientific;

    auto get_error = [](value_type res, value_type ref_res) -> value_type {
        return std::abs(res - ref_res) / std::abs(ref_res);
        // return std::abs(res);
    };
    // constexpr std::size_t steps = 1024 - 1;
    constexpr std::size_t start = std::min(max_size, std::size_t{1'000'000});
    constexpr std::size_t row_incr = 2'000'000;  // (max_size - start) / steps;
    constexpr std::size_t steps = (max_size - start) / row_incr;
    constexpr std::size_t randomize_num{10};

    std::vector<std::size_t> benchmark_vec_size((steps + 1));
    std::vector<double> benchmark_time((steps + 1) * benchmark_num);
    std::vector<value_type> benchmark_error((steps + 1) * benchmark_num);

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
        for (std::size_t vec_size = start, i = 0; vec_size < max_size;
             vec_size += row_incr, ++i) {
            benchmark_vec_size.at(i) = vec_size;
            const matrix_info x_info{{vec_size, 1}};
            const matrix_info y_info{{vec_size, 1}};

            std::array<value_type, benchmark_num> raw_error{};
            for (std::size_t bi = 0; bi < benchmark_num; ++bi) {
                const std::size_t idx = i * benchmark_num + bi;
                auto curr_lambda = [&]() {
                    benchmark_lambdas[bi](x_info, y_info);
                };
                benchmark_time.at(idx) =
                    benchmark_function(curr_lambda, detailed_error);
                raw_error[bi] = benchmark_get_error[bi]();
            }
            const value_type result_ref = raw_error[benchmark_reference];
            for (std::size_t bi = 0; bi < benchmark_num; ++bi) {
                const std::size_t idx = i * benchmark_num + bi;
                benchmark_error.at(idx) += get_error(raw_error[bi], result_ref);
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
