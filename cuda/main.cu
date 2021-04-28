#include <array>
#include <cmath>
#include <ios>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../error_tobias.hpp"
#include "kernels.cuh"
#include "utils.cuh"

template <typename ValueType>
void control_gemv(const matrix_info minfo, const std::vector<ValueType> &mtx,
                  const matrix_info vinfo, const std::vector<ValueType> &b,
                  std::vector<ValueType> &res) {
    for (std::size_t row = 0; row < minfo.size[0]; ++row) {
        for (std::size_t col = 0; col < minfo.size[1]; ++col) {
            const std::size_t midx = row * minfo.stride + col;
            res[row] += mtx[midx] * b[col];
        }
    }
}

template <typename OutputType, typename VectorType, typename ReduceOp>
OutputType reduce(const matrix_info info, VectorType &tmp, ReduceOp op) {
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
ValueType compare(const matrix_info info,
                  const std::vector<ReferenceType> &mtx1,
                  const std::vector<OtherType> &mtx2,
                  std::vector<ValueType> &tmp) {
    // ReferenceType error{};
    using return_type = decltype(get_value(ReferenceType{}));
    if (info.get_1d_size() > mtx1.size() || info.get_1d_size() > mtx2.size() ||
        info.get_1d_size() > tmp.size()) {
        throw "Error";
    }
    for (std::size_t row = 0; row < info.size[0]; ++row) {
        for (std::size_t col = 0; col < info.size[1]; ++col) {
            const std::size_t midx = row * info.stride + col;
            tmp[midx] = ValueType{};
        }
    }
    for (std::size_t row = 0; row < info.size[0]; ++row) {
        for (std::size_t col = 0; col < info.size[1]; ++col) {
            const std::size_t midx = row * info.stride + col;
            const auto v1 = get_value(mtx1[midx]);
            const decltype(v1) v2 = get_value(mtx2[midx]);
            const auto delta = std::abs(v1 - v2);
            tmp[midx] = delta;
        }
    }
    /*
    std::cout << '\n';
    for (std::size_t i = 0; i < info.size[0]; ++i) {
        std::cout << tmp[i] << ' ';
    }
    std::cout << '\n';
    for (std::size_t i = 0; i < info.size[0]; ++i) {
        std::cout << i % 10 << ' ';
        for (int j = 10; j < tmp[i]; j *= 10)
            std::cout << ' ';
    }
    std::cout << '\n';
    //*/
    return reduce<ValueType>(
        info, tmp, [](ValueType o1, ValueType o2) { return o1 + o2; });
}

template <typename t1, typename t2, typename t3, typename t4, typename t5>
void test(t1, t2, t3, t4, t5) {
    static_assert(std::is_same<t1, t2>::value,
                  "GKO_DECLARE_OUTPLACE_ABSOLUTE_DENSE_KERNEL");
}

template <typename Callable>
double benchmark_function(Callable func) {
    constexpr int bench_iters{10};
    double time_ms[bench_iters];
    CudaTimer ctimer;
    // Warmup
    func();
    synchronize();
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

int main() {
    /*
    using ar_type = error_number<double>;
    using st_type = error_number<float>;
    using value_type = ar_type::value_type;
    auto convert_func = [](ar_type val) {
        return st_type{static_cast<st_type::value_type>(val.v),
                       static_cast<st_type::value_type>(val.e)};
    };
    /*/
    using ar_type = double;
    using st_type = float;
    using value_type = ar_type;
    auto convert_func = [](ar_type val) { return static_cast<st_type>(val); };

    //*/

    constexpr std::size_t max_rows{24 * 1024};
    constexpr matrix_info max_minfo{{max_rows, max_rows}};
    constexpr char DELIM{';'};

    std::default_random_engine rengine(42);
    std::uniform_real_distribution<value_type> mtx_dist(-2.0, 2.0);
    // std::normal_distribution<value_type> mtx_dist(1, 2);
    //*
    // std::uniform_real_distribution<value_type> b_dist(-2.0, 2.0);
    // std::uniform_real_distribution<value_type> b_dist(1.0, 1.0);
    auto b_dist = mtx_dist;
    /*/

    auto b_dist = [rnd = 0](auto val) mutable {
        return (rnd = (rnd + 1) % 40) == 0
                   ? 1
                   : std::numeric_limits<float>::epsilon() / 2;
        // return std::numeric_limits<float>::epsilon() / 2;
    };
    std::cout << std::numeric_limits<float>::epsilon() / 2 << '\n';
    //*/

    /*
    constexpr std::size_t tmp_size{1000};
    std::vector<float> tmp(tmp_size);
    for (int i = 0; i < tmp_size; ++i) {
        tmp[i] = b_dist(1);
    }

    double d_sum{};
    float f_sum{};
    for (int i = 0; i < tmp_size; ++i) {
        d_sum += tmp[i];
        f_sum += tmp[i];
    }
    std::cout.precision(16);
    std::cout << std::scientific;
    std::cout << "float  sum = " << f_sum
            << "\ndouble sum = " << static_cast<float>(d_sum)
            << '\n';
    return 0;
    //*/
    /*
    constexpr std::size_t red_size{1};
    std::cout << "Beginning...\n";
    std::vector<int> a(red_size, 0);
    std::vector<int> b(red_size, 1);
    std::vector<int> tmp(red_size);
    std::cout << "Running compare...\n";
    std::cout << compare({{red_size, 1}}, a, b, tmp) << '\n';
    return 0;
    //*/

    const matrix_info max_vinfo{{max_rows, 1}};

    auto v_matrix = gen_mtx<ar_type>(max_minfo, mtx_dist, rengine);
    std::vector<st_type> s_matrix(max_minfo.get_1d_size());
    convert_mtx<st_type>(max_minfo, v_matrix, s_matrix, convert_func);

    auto v_b = gen_mtx<ar_type>(max_vinfo, b_dist, rengine);
    std::vector<st_type> s_b(max_vinfo.get_1d_size());
    convert_mtx<st_type>(max_vinfo, v_b, s_b, convert_func);

    auto v_res = std::vector<ar_type>(max_vinfo.get_1d_size(), ar_type{});
    auto v_res_ref = std::vector<ar_type>(max_vinfo.get_1d_size(), ar_type{});
    auto s_res = std::vector<st_type>(max_vinfo.get_1d_size(), st_type{});
    auto v_reduce =
        std::vector<value_type>(max_vinfo.get_1d_size(), value_type{});

    auto dv_matrix = GpuMemory<ar_type>(max_minfo.get_1d_size());
    dv_matrix.copy_from(v_matrix);
    auto ds_matrix = GpuMemory<st_type>(max_minfo.get_1d_size());
    ds_matrix.copy_from(s_matrix);

    auto dv_b = GpuMemory<ar_type>(max_vinfo.get_1d_size());
    dv_b.copy_from(v_b);
    auto ds_b = GpuMemory<st_type>(max_vinfo.get_1d_size());
    ds_b.copy_from(s_b);
    auto dv_res = GpuMemory<ar_type>(max_vinfo.get_1d_size());
    auto ds_res = GpuMemory<st_type>(max_vinfo.get_1d_size());

    auto cublasHandle = get_cublas_handle();

    std::cout << "Num Rows" << DELIM << "GEMV double" << DELIM << "GEMV float"
              << DELIM << "GEMV Acc<fp64, fp64>" << DELIM
              << "GEMV Acc<fp64, fp32>" << DELIM << "CUBLAS GEMV fp64" << DELIM
              << "CUBLAS GEMV fp32" << '\n';

    std::cout.precision(16);
    std::cout << std::scientific;
    /*
    std::cout << "single_error           Acc<fp64, fp32> error    single_error "
                 "/ acc_error\n";
    */

    constexpr auto start = max_rows / 48;
    constexpr auto row_incr = start;
    for (auto num_rows = start; num_rows <= max_rows; num_rows += row_incr) {
        // for (int i = 0; i < 10; ++i) {
        const matrix_info minfo{{num_rows, num_rows}};
        const matrix_info vinfo{{num_rows, 1}};

        v_matrix = gen_mtx<ar_type>(minfo, mtx_dist, rengine);
        convert_mtx<st_type>(minfo, v_matrix, s_matrix, convert_func);
        dv_matrix.copy_from(v_matrix);
        ds_matrix.copy_from(s_matrix);

        double d_time{};
        auto d_func = [&]() {
            gemv(minfo, dv_matrix.data(), vinfo, dv_b.data(), dv_res.data());
        };
        double s_time{};
        auto s_func = [&]() {
            gemv(minfo, ds_matrix.data(), vinfo, ds_b.data(), ds_res.data());
        };
        double avv_time{};
        auto avv_func = [&]() {
            acc_gemv<ar_type>(minfo, dv_matrix.data(), vinfo, dv_b.data(),
                              dv_res.data());
        };
        double avs_time{};
        auto avs_func = [&]() {
            acc_gemv<ar_type>(minfo, ds_matrix.data(), vinfo, ds_b.data(),
                              ds_res.data());
        };
        double cd_time{};
        auto cd_func = [&]() {
            cublas_gemv(cublasHandle.get(), minfo, dv_matrix.data(), vinfo,
                        dv_b.data(), dv_res.data());
        };
        double cs_time{};
        auto cs_func = [&]() {
            cublas_gemv(cublasHandle.get(), minfo, ds_matrix.data(), vinfo,
                        ds_b.data(), ds_res.data());
        };
        // ar_type d_error{};
        [[gnu::unused, maybe_unused]] value_type s_error{};
        //[[ gnu::unused, maybe_unused ]] ar_type avv_error{};
        [[gnu::unused, maybe_unused]] value_type avs_error{};
        [[gnu::unused, maybe_unused]] value_type cv_error{};
        [[gnu::unused, maybe_unused]] value_type cs_error{};

        // control_gemv(minfo, v_matrix, vinfo, v_b, v_res_ref);

        d_time = benchmark_function(d_func);
        // d_func();
        // dv_res.get_vector(v_res);
        // d_error = compare(vinfo, v_res_ref, v_res, v_reduce);
        // v_res_ref = v_res;

        s_time = benchmark_function(s_func);
        // ds_res.get_vector(s_res);
        //*
        // std::cout << "single: x_res[0] = " << s_res[0] << '\n';
        // s_error = compare(vinfo, v_res_ref, s_res, v_reduce);
        /*/
        convert_mtx(vinfo, s_res, v_reduce, [](st_type v) { return v.e; });
        s_error = reduce<value_type>(
            vinfo, v_reduce, [](value_type a, value_type b) { return a + b; });
        //*/

        avv_time = benchmark_function(avv_func);
        avs_time = benchmark_function(avs_func);
        // ds_res.get_vector(s_res);
        //*
        // std::cout << "access: x_res[0] = " << s_res[0] << '\n';
        // avs_error = compare(vinfo, v_res_ref, s_res, v_reduce);
        /*/
        convert_mtx(vinfo, s_res, v_reduce, [](st_type v) { return v.e; });
        avs_error = reduce<value_type>(
            vinfo, v_reduce, [](value_type a, value_type b) { return a + b; });
        //*/

        cd_time = benchmark_function(cd_func);
        // dv_res.get_vector(d_res);
        // std::cout << s_error << ' ' << avs_error << '\t'
        //          << (s_error / avs_error) << '\n';
        cs_time = benchmark_function(cs_func);
        // ds_res.get_vector(s_res);
        //*
        // std::cout << "access: x_res[0] = " << s_res[0] << '\n';
        // cs_error = compare(vinfo, v_res_ref, s_res, v_reduce);

        //*
        std::cout << num_rows << DELIM << d_time << DELIM << s_time << DELIM
                  << avv_time << DELIM << avs_time << DELIM << cd_time << DELIM
                  << cs_time << '\n';
        //*/
        /*
        std::cout << "Comparison:"
                << "\nDouble: " << d_error
                << "\nSingle: " << s_error
                << "\nAcc_vv: " << avv_error
                << "\nAcc_vs: " << avs_error
                << '\n';
        //*/
    }
}
