#include <cmath>
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

    std::default_random_engine rengine(42);
    std::uniform_real_distribution<value_type> vector_dist(-2.0, 2.0);

    auto ar_data = DotMemory<ar_type>(max_size, vector_dist, rengine);
    auto st_data = DotMemory<st_type>(ar_data);

    auto cublas_handle = cublas_get_handle();
    cublas_set_device_ptr_mode(cublas_handle.get());

    cudaDeviceProp device_prop;
    CUDA_CALL(cudaGetDeviceProperties(&device_prop, 0));
    // std::cout << "Number SMs: " << device_prop.multiProcessorCount << '\n';

    std::cout << "Vector Size" << DELIM << "DOT fp64" << DELIM << "DOT fp32"
              << DELIM << "DOT Acc<fp64, fp64>" << DELIM
              << "DOT Acc<fp64, fp32>" << DELIM << "CUBLAS DOT fp64" << DELIM
              << "CUBLAS DOT fp32" << DELIM;
    std::cout << "Error DOT fp64" << DELIM << "Error DOT fp32" << DELIM
              << "Error DOT Acc<fp64, fp64>" << DELIM
              << "Error DOT Acc<fp64, fp32>" << DELIM << "Error CUBLAS DOT fp64"
              << DELIM << "Error CUBLAS DOT fp32" << '\n';

    std::cout.precision(16);
    std::cout << std::scientific;

    auto get_error = [](value_type res, value_type ref_res) -> value_type {
        return std::abs(res - ref_res) / std::abs(ref_res);
        // return res;
    };
    constexpr std::size_t steps = 64;
    constexpr auto start = max_size / 1024;
    // constexpr auto start = max_size / 16;
    constexpr auto row_incr = (max_size - start) / steps;
    for (auto vec_size = start; vec_size <= max_size; vec_size += row_incr) {
        const matrix_info x_info{{vec_size, 1}};
        const matrix_info y_info{{vec_size, 1}};

        double ar_time{};
        auto ar_func = [&]() {
            dot(device_prop, x_info, ar_data.gpu_x(), y_info, ar_data.gpu_y(),
                ar_data.gpu_res());
        };
        double st_time{};
        auto st_func = [&]() {
            dot(device_prop, x_info, st_data.gpu_x(), y_info, st_data.gpu_y(),
                st_data.gpu_res());
        };
        double acc_ar_time{};
        auto acc_ar_func = [&]() {
            acc_dot(device_prop, x_info, ar_data.gpu_x(), y_info,
                    ar_data.gpu_y(), ar_data.gpu_res());
        };
        double acc_mix_time{};
        auto acc_mix_func = [&]() {
            acc_dot(device_prop, x_info, st_data.gpu_x(), y_info,
                    st_data.gpu_y(), ar_data.gpu_res());
        };
        double cublas_ar_time{};
        auto cublas_ar_func = [&]() {
            cublas_dot(cublas_handle.get(), x_info, ar_data.gpu_x(), y_info,
                       ar_data.gpu_y(), ar_data.gpu_res());
        };
        double cublas_st_time{};
        auto cublas_st_func = [&]() {
            cublas_dot(cublas_handle.get(), x_info, st_data.gpu_x(), y_info,
                       st_data.gpu_y(), st_data.gpu_res());
        };
        value_type ar_error{};  // [[gnu::unused, maybe_unused]]
        value_type st_error{};
        value_type acc_ar_error{};
        value_type acc_mix_error{};
        value_type cublas_ar_error{};
        value_type cublas_st_error{};

        ar_time = benchmark_function(ar_func);
        const value_type result_ref = ar_data.get_result();
        ar_error = get_error(ar_data.get_result(), result_ref);
        // Use the result here as the reference

        st_time = benchmark_function(st_func);
        st_error = get_error(st_data.get_result(), result_ref);

        acc_ar_time = benchmark_function(acc_ar_func);
        acc_ar_error = get_error(ar_data.get_result(), result_ref);

        acc_mix_time = benchmark_function(acc_mix_func);
        acc_mix_error = get_error(ar_data.get_result(), result_ref);

        cublas_ar_time = benchmark_function(cublas_ar_func);
        cublas_ar_error = get_error(ar_data.get_result(), result_ref);

        cublas_st_time = benchmark_function(cublas_st_func);
        cublas_st_error = get_error(st_data.get_result(), result_ref);
        std::cout << vec_size << DELIM << ar_time << DELIM << st_time << DELIM
                  << acc_ar_time << DELIM << acc_mix_time << DELIM
                  << cublas_ar_time << DELIM << cublas_st_time << DELIM;
        std::cout << ar_error << DELIM << st_error << DELIM << acc_ar_error
                  << DELIM << acc_mix_error << DELIM << cublas_ar_error << DELIM
                  << cublas_st_error << '\n';
    }
}
