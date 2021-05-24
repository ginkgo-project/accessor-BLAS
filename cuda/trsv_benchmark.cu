#include <array>
#include <cmath>
#include <ios>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

#include "memory.cuh"
#include "trsv_kernels.cuh"
#include "trsv_memory.cuh"
#include "utils.cuh"

// Reason for triangular matrices to be ill-conditioned:
// http://www.math.lsa.umich.edu/~divakar/1-ViswanathTrefethen1998.pdf

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

    constexpr tmtx_t t_matrix_type = tmtx_t::lower;
    constexpr dmtx_t d_matrix_type = dmtx_t::unit;

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

    constexpr std::size_t max_rows{24 * 1024};
    // constexpr std::size_t max_rows{37};
    // constexpr std::size_t max_rows{8 * 1024};
    constexpr std::size_t max_cols{max_rows};
    constexpr char DELIM{';'};

    constexpr std::size_t start = 50;  // max_rows - 4;
    // constexpr auto start = max_rows / 48;
    constexpr std::size_t row_incr = start;  // start;

    std::default_random_engine rengine(42);
    std::uniform_real_distribution<value_type> mtx_dist(0.0, 1.0);
    // std::uniform_real_distribution<value_type> mtx_dist(-10.0, 10.0);
    auto vector_dist = mtx_dist;
    auto cpu_mtx_gen = [&](matrix_info m_info) {
        // return gen_dd_mtx<ar_type>(m_info, mtx_dist, rengine, 1);
        return gen_mtx<ar_type>(m_info, mtx_dist, rengine);
    };
    auto cpu_vect_gen = [&](matrix_info v_info) {
        return gen_mtx<ar_type>(v_info, vector_dist, rengine);
    };

    auto ar_data =
        TrsvMemory<ar_type>(max_rows, max_cols, cpu_mtx_gen, cpu_vect_gen);
    auto st_data = TrsvMemory<st_type>(ar_data);

    auto cublasHandle = cublas_get_handle();

    static_assert(max_rows == max_cols, "Matrix must be square!");

    // Additional memory and lambdas for error computation
    const auto max_res_num_elems = ar_data.cpu_x_memory().get_num_elems();
    Memory<ar_type> cpu_x_ref(Memory<ar_type>::Device::cpu, max_res_num_elems);
    value_type res_ref_norm{1.0};
    Memory<value_type> reduce_memory(Memory<ar_type>::Device::cpu,
                                     max_res_num_elems);
    Memory<std::uint32_t> trsv_helper(Memory<std::uint32_t>::Device::gpu, 2);

    auto ar_compute_error = [&](matrix_info x_info) {
        value_type error{};
        ar_data.sync_x();
        error = compare(x_info, cpu_x_ref.const_data(), ar_data.cpu_x_const(),
                        reduce_memory.data());
        // std::cout << '\n';
        // print_mtx(x_info, ar_data.cpu_x_const());
        ar_data.reset_x();
        return error / res_ref_norm;
    };
    auto st_compute_error = [&](matrix_info x_info) {
        value_type error{};
        st_data.sync_x();
        error = compare(x_info, cpu_x_ref.const_data(), st_data.cpu_x_const(),
                        reduce_memory.data());
        st_data.reset_x();
        return error / res_ref_norm;
    };

    // Setting up names and associated benchmark and error functions

    constexpr std::size_t benchmark_num{7};
    constexpr std::size_t benchmark_reference{0};  //{benchmark_num - 2};
    using benchmark_info_t =
        std::tuple<std::string, std::function<void(matrix_info, matrix_info)>,
                   std::string, std::function<value_type(matrix_info)>>;
    std::array<benchmark_info_t, benchmark_num> benchmark_info = {
        /*
        benchmark_info_t{"TRSV multi-kernel fp64",
                         [&](matrix_info m_info, matrix_info x_info) {
                             trsv(m_info, t_matrix_type, d_matrix_type,
                                  ar_data.gpu_mtx_const(), x_info,
                                  ar_data.gpu_x());
                         },
                         "Error TRSV multi-kernel fp64", ar_compute_error},
        benchmark_info_t{"TRSV single kernel fp64",
                         [&](matrix_info m_info, matrix_info x_info) {
                             trsv_2(m_info, t_matrix_type, d_matrix_type,
                                    ar_data.gpu_mtx_const(), x_info,
                                    ar_data.gpu_x(), trsv_helper.data());
                         },
                         "Error TRSV single kernel fp64", ar_compute_error},
        */
        benchmark_info_t{"TRSV fp64",
                         [&](matrix_info m_info, matrix_info x_info) {
                             trsv_3(m_info, t_matrix_type, d_matrix_type,
                                    ar_data.gpu_mtx_const(), x_info,
                                    ar_data.gpu_x(), trsv_helper.data());
                         },
                         "Error TRSV fp64", ar_compute_error},
        benchmark_info_t{"TRSV fp32",
                         [&](matrix_info m_info, matrix_info x_info) {
                             trsv_3(m_info, t_matrix_type, d_matrix_type,
                                    st_data.gpu_mtx_const(), x_info,
                                    st_data.gpu_x(), trsv_helper.data());
                         },
                         "Error TRSV fp32", st_compute_error},
        benchmark_info_t{"TRSV Acc<fp64, fp64>",
                         [&](matrix_info m_info, matrix_info x_info) {
                             acc_trsv<ar_type>(
                                 m_info, t_matrix_type, d_matrix_type,
                                 ar_data.gpu_mtx_const(), x_info,
                                 ar_data.gpu_x(), trsv_helper.data());
                         },
                         "Error TRSV Acc<fp64, fp64>", ar_compute_error},
        benchmark_info_t{"TRSV Acc<fp64, fp32>",
                         [&](matrix_info m_info, matrix_info x_info) {
                             acc_trsv<ar_type>(
                                 m_info, t_matrix_type, d_matrix_type,
                                 st_data.gpu_mtx_const(), x_info,
                                 st_data.gpu_x(), trsv_helper.data());
                         },
                         "Error TRSV Acc<fp64, fp32>", st_compute_error},
        benchmark_info_t{"TRSV Acc<fp32, fp32>",
                         [&](matrix_info m_info, matrix_info x_info) {
                             acc_trsv<st_type>(
                                 m_info, t_matrix_type, d_matrix_type,
                                 st_data.gpu_mtx_const(), x_info,
                                 st_data.gpu_x(), trsv_helper.data());
                         },
                         "Error TRSV Acc<fp32, fp32>", st_compute_error},
        benchmark_info_t{"CUBLAS TRSV fp64",
                         [&](matrix_info m_info, matrix_info x_info) {
                             cublas_trsv(cublasHandle.get(), t_matrix_type,
                                         d_matrix_type, m_info,
                                         ar_data.gpu_mtx_const(), x_info,
                                         ar_data.gpu_x());
                         },
                         "Error CUBLAS TRSV fp64", ar_compute_error},
        benchmark_info_t{"CUBLAS TRSV fp32",
                         [&](matrix_info m_info, matrix_info x_info) {
                             cublas_trsv(cublasHandle.get(), t_matrix_type,
                                         d_matrix_type, m_info,
                                         st_data.gpu_mtx_const(), x_info,
                                         st_data.gpu_x());
                         },
                         "Error CUBLAS TRSV fp32", st_compute_error},
        /*
        benchmark_info_t{"Hand TRSV fp64",
                         [&](matrix_info m_info, matrix_info x_info) {
                             control_trsv(m_info, t_matrix_type, d_matrix_type,
                                          ar_data.cpu_mtx_const(), x_info,
                                          ar_data.cpu_x());
                         },
                         "Error Hand TRSV fp64",
                         [&](matrix_info x_info) {
                             value_type error{};
                             if (measure_error) {
                                 error = compare(x_info, cpu_x_ref.const_data(),
                                                 ar_data.cpu_x_const(),
                                                 reduce_memory.data());
                             }
                             ar_data.reset_x();
                             return error / res_ref_norm;
                         }},
        */
    };

    std::cout << "Num rows";
    for (const auto &info : benchmark_info) {
        if (!measure_error) {
            std::cout << DELIM << std::get<0>(info);
        } else {
            std::cout << DELIM << std::get<2>(info);
        }
    }
    std::cout << '\n';

    std::cout.precision(16);
    // showpos: show + sign for positive numbers
    std::cout << std::scientific << std::showpos;

    // std::vector<std::array<value_type, benchmark_num>>

    for (auto num_rows = start; num_rows <= max_rows; num_rows += row_incr) {
        std::array<value_type, benchmark_num> local_res{};
        const matrix_info m_info{{num_rows, num_rows}, num_rows};
        const matrix_info x_info{{num_rows, 1}};

        if (measure_error) {
            std::get<1>(benchmark_info[benchmark_reference])(m_info, x_info);
            ar_data.sync_x();
            cpu_x_ref = ar_data.cpu_x_memory();
            res_ref_norm = reduce<value_type>(
                x_info, cpu_x_ref.data(),
                [](ar_type a, ar_type b) { return std::abs(a) + std::abs(b); });
            // copy again since reduce overwrites
            cpu_x_ref = ar_data.cpu_x_memory();
            //*/
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
                local_res[i] = std::get<3>(benchmark_info[i])(x_info);
            }
        }

        std::cout << num_rows;
        for (const auto &res : local_res) {
            std::cout << DELIM << res;
        }
        std::cout << '\n';
    }
}
