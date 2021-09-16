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


#include "gemv_kernels.cuh"
#include "ir_memory.cuh"
#include "memory.cuh"
#include "trsv_kernels.cuh"
#include "utils.cuh"


template <typename OutputType, typename InputType, typename ReduceOp>
OutputType reduce(const matrix_info info, InputType *tmp, ReduceOp op)
{
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
std::enable_if_t<std::is_floating_point<T>::value, T> get_value(T val)
{
    return val;
}

template <typename T>
std::enable_if_t<!std::is_floating_point<T>::value, typename T::value_type>
get_value(T val)
{
    return val.e;
}

template <typename ReferenceType, typename OtherType, typename ValueType>
ValueType compare(const matrix_info info, const ReferenceType *mtx1,
                  const OtherType *mtx2, ValueType *tmp)
{
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

constexpr tmtx_t invert(tmtx_t val)
{
    return val == tmtx_t::upper ? tmtx_t::lower : tmtx_t::upper;
}

constexpr dmtx_t invert(dmtx_t val)
{
    return val == dmtx_t::unit ? dmtx_t::non_unit : dmtx_t::unit;
}


// Overwrites `data`
template <typename ValueType>
ValueType norm2(const matrix_info info, ValueType *data)
{
    for (std::size_t i = 0; i < info.size[0]; ++i) {
        const auto val = data[i * info.stride];
        data[i * info.stride] = val * val;
    }
    return std::sqrt(reduce<ValueType>(
        info, data, [](ValueType o1, ValueType o2) { return o1 + o2; }));
}


// Applies the pivoting to the given matrix (therefore, changing it).
// Pivoting views the matrix as column major
template <typename ValueType>
void apply_row_pivot(ValueType *data, const matrix_info &m_info,
                     const int *row_pivot)
{
    for (std::size_t row = 0; row < m_info.size[0]; ++row) {
        const std::size_t new_row = row_pivot[row] - 1;
        for (std::size_t col = 0; col < m_info.size[1]; ++col) {
            const std::size_t idx = col * m_info.stride + row;
            const std::size_t new_idx = col * m_info.stride + new_row;
            std::swap(data[idx], data[new_idx]);
        }
    }
}

template <typename ValueType>
void transpose_mtx(ValueType *data, const matrix_info &m_info)
{
    for (std::size_t row = 0; row < m_info.size[0]; ++row) {
        for (std::size_t col = row + 1; col < m_info.size[1]; ++col) {
            const std::size_t idx = col * m_info.stride + row;
            const std::size_t t_idx = row * m_info.stride + col;
            std::swap(data[idx], data[t_idx]);
        }
    }
}


int main(int argc, char **argv)
{
    using ar_type = double;
    using st_type = float;
    using value_type = ar_type;

    // constexpr tmtx_t t_matrix_type = tmtx_t::upper;
    // constexpr dmtx_t d_matrix_type = dmtx_t::unit;
    constexpr tmtx_t t_matrix_type = tmtx_t::lower;
    constexpr dmtx_t d_matrix_type = dmtx_t::unit;
    constexpr tmtx_t t_matrix_type2 = invert(t_matrix_type);
    constexpr dmtx_t d_matrix_type2 = invert(d_matrix_type);

    const std::string use_error_string("--error");
    if (argc == 2 && std::string(argv[1]) == use_error_string) {
    } else if (argc > 1) {
        const std::string binary(argv[0]);
        std::cerr << "Unsupported parameters!\n";
        std::cerr << "Usage: " << binary << " [" << use_error_string << "]\n";
        std::cerr << "With " << use_error_string
                  << ":    compute error of IRs\n"
                  << "Without parameters: benchmark different IRs\n";
        return 1;
    }

    //constexpr std::size_t max_rows{24 * 1000};
    constexpr std::size_t max_rows{12 * 1000};
    constexpr std::size_t max_cols{max_rows};
    // constexpr std::size_t max_rows{10};
    // constexpr std::size_t max_cols{max_rows};
    constexpr char DELIM{';'};
    constexpr int max_iter{10};
    const matrix_info max_size{{max_rows, max_cols}};

    std::default_random_engine rengine(42);
    //*
    std::uniform_real_distribution<ar_type> mtx_dist(-1.0, 1.0);
    auto vector_dist = mtx_dist;

    std::cout << "Distribution matrix: [" << mtx_dist.a() << ',' << mtx_dist.b()
              << "); vector: [" << vector_dist.a() << ',' << vector_dist.b()
              << "); Type mtx-dist: " << typeid(mtx_dist).name() << "\n";
    /*/
    std::normal_distribution<value_type> mtx_dist(0.0, 1.0);
    auto vector_dist = mtx_dist;

    std::cout << "Distribution matrix: "
              << "mean: " << mtx_dist.mean()
              << ", stddev: " << mtx_dist.stddev() << ";"
              << "vector: "
              << "mean: " << vector_dist.mean()
              << ", stddev: " << vector_dist.stddev() << ";"
              << "Type mtx-dist: " << typeid(mtx_dist).name() << "\n";
    //*/

    auto cpu_mtx_orig = gen_mtx<ar_type>(max_size, mtx_dist, rengine);

    Memory<ar_type> gpu_mtx_orig(Memory<ar_type>::Device::gpu,
                                 max_size.get_1d_size());
    auto cpu_dp_rhs =
        gen_mtx<ar_type>({{max_cols, 1}, 1}, vector_dist, rengine);
    for (int i = 0; i < max_cols; ++i) {
        cpu_dp_rhs.data()[i] = ar_type{1};
    }
    Memory<ar_type> gpu_dp_rhs(Memory<ar_type>::Device::gpu, max_cols);
    gpu_dp_rhs.copy_from(cpu_dp_rhs);

    auto ar_data = IrMemory<ar_type>(max_rows, max_cols, cpu_mtx_orig);

    // permutate the original MTX, so it matches the LU decomposition
    apply_row_pivot(cpu_mtx_orig.data(), max_size,
                    ar_data.cpu_pivot().const_data());
    transpose_mtx(cpu_mtx_orig.data(), max_size);
    gpu_mtx_orig.copy_from(cpu_mtx_orig);

    // Transpose the LU matrix, so it is row-major
    transpose_mtx(ar_data.cpu_mtx_memory().data(), max_size);
    ar_data.gpu_mtx_memory().copy_from(ar_data.cpu_mtx_memory());

    // Clone the properly set up data now to storage_type
    auto st_data = IrMemory<st_type>(ar_data);

    auto cublasHandle = cublas_get_handle();

    static_assert(max_rows == max_cols, "Matrix must be square!");

    // Additional memory and lambdas for error computation
    Memory<ar_type> cpu_dp_x(Memory<ar_type>::Device::cpu, max_cols);
    Memory<ar_type> gpu_dp_x(Memory<ar_type>::Device::gpu, max_cols);
    Memory<ar_type> gpu_dp_res(Memory<ar_type>::Device::gpu, max_cols);
    Memory<ar_type> cpu_dp_res(Memory<ar_type>::Device::cpu, max_cols);

    Memory<std::uint32_t> trsv_helper(Memory<std::uint32_t>::Device::gpu, 2);

    // Setting up names and associated benchmark and error functions

    using benchmark_info_t =
        std::tuple<std::string, std::function<void(matrix_info, matrix_info)>>;
    std::vector<benchmark_info_t> benchmark_info = {
        benchmark_info_t{"IR fp64",
                         [&](matrix_info m_info, matrix_info x_info) {
                             copy_vector(gpu_dp_res.const_data(), x_info,
                                         ar_data.gpu_x(), x_info);

                             trsv_3(m_info, t_matrix_type, d_matrix_type,
                                    ar_data.gpu_mtx_const(), x_info,
                                    ar_data.gpu_x(), trsv_helper.data());
                             trsv_3(m_info, t_matrix_type2, d_matrix_type2,
                                    ar_data.gpu_mtx_const(), x_info,
                                    ar_data.gpu_x(), trsv_helper.data());
                             update_vector(ar_data.gpu_x_const(), x_info,
                                           gpu_dp_x.data(), x_info);
                         }},
        benchmark_info_t{"IR fp32",
                         [&](matrix_info m_info, matrix_info x_info) {
                             copy_vector(gpu_dp_res.const_data(), x_info,
                                         st_data.gpu_x(), x_info);
                             trsv_3(m_info, t_matrix_type, d_matrix_type,
                                    st_data.gpu_mtx_const(), x_info,
                                    st_data.gpu_x(), trsv_helper.data());
                             trsv_3(m_info, t_matrix_type2, d_matrix_type2,
                                    st_data.gpu_mtx_const(), x_info,
                                    st_data.gpu_x(), trsv_helper.data());
                             update_vector(st_data.gpu_x(), x_info,
                                           gpu_dp_x.data(), x_info);
                         }},
        benchmark_info_t{
            "IR Acc<fp64, fp32>",
            [&](matrix_info m_info, matrix_info x_info) {
                copy_vector(gpu_dp_res.const_data(), x_info, st_data.gpu_x(),
                            x_info);
                acc_trsv<ar_type>(m_info, t_matrix_type, d_matrix_type,
                                  st_data.gpu_mtx_const(), x_info,
                                  st_data.gpu_x(), trsv_helper.data());
                acc_trsv<ar_type>(m_info, t_matrix_type2, d_matrix_type2,
                                  st_data.gpu_mtx_const(), x_info,
                                  st_data.gpu_x(), trsv_helper.data());
                update_vector(st_data.gpu_x(), x_info, gpu_dp_x.data(), x_info);
            }},
        benchmark_info_t{
            "CUBLAS IR fp64",
            [&](matrix_info m_info, matrix_info x_info) {
                copy_vector(gpu_dp_res.const_data(), x_info, ar_data.gpu_x(),
                            x_info);
                cublas_trsv(cublasHandle.get(), t_matrix_type, d_matrix_type,
                            m_info, ar_data.gpu_mtx_const(), x_info,
                            ar_data.gpu_x());

                cublas_trsv(cublasHandle.get(), t_matrix_type2, d_matrix_type2,
                            m_info, ar_data.gpu_mtx_const(), x_info,
                            ar_data.gpu_x());
                update_vector(ar_data.gpu_x(), x_info, gpu_dp_x.data(), x_info);
            }},
        benchmark_info_t{
            "CUBLAS IR fp32",
            [&](matrix_info m_info, matrix_info x_info) {
                copy_vector(gpu_dp_res.const_data(), x_info, st_data.gpu_x(),
                            x_info);
                cublas_trsv(cublasHandle.get(), t_matrix_type, d_matrix_type,
                            m_info, st_data.gpu_mtx_const(), x_info,
                            st_data.gpu_x());
                cublas_trsv(cublasHandle.get(), t_matrix_type2, d_matrix_type2,
                            m_info, st_data.gpu_mtx_const(), x_info,
                            st_data.gpu_x());
                update_vector(st_data.gpu_x(), x_info, gpu_dp_x.data(), x_info);
            }},
    };
    const std::size_t benchmark_num{benchmark_info.size()};

    std::cout << "Num rows";
    for (const auto &info : benchmark_info) {
        std::cout << DELIM << "Res norm " << std::get<0>(info);
    }
    std::cout << '\n';

    std::cout.precision(16);
    // showpos: show + sign for positive numbers
    std::cout << std::scientific << std::showpos;

    std::vector<ar_type> res_norm_results(max_iter * benchmark_num);

    for (int b_idx = 0; b_idx < benchmark_num; ++b_idx) {
        for (int i = 0; i < max_cols; ++i) {
            cpu_dp_x.data()[i] = ar_type{};
        }
        gpu_dp_x.copy_from(cpu_dp_x);
        for (int i = 0; i < max_iter; ++i) {
            const auto m_info = max_size;
            const matrix_info x_info{{max_cols, 1}, 1};
            gpu_dp_res.copy_from(cpu_dp_rhs);
            gemv(m_info, ar_type{-1}, gpu_mtx_orig.const_data(), x_info,
                 gpu_dp_x.const_data(), x_info, ar_type{1}, gpu_dp_res.data());
            cpu_dp_res.copy_from(gpu_dp_res);

            auto norm = norm2(x_info, cpu_dp_res.data());
            res_norm_results[i * benchmark_num + b_idx] = norm;
            std::get<1>(benchmark_info[b_idx])(m_info, x_info);
        }
    }
    for (int i = 0; i < max_iter; ++i) {
        std::cout << i;
        for (int b_idx = 0; b_idx < benchmark_num; ++b_idx) {
            std::cout << DELIM << res_norm_results[i * benchmark_num + b_idx];
        }
        std::cout << '\n';
    }
}
