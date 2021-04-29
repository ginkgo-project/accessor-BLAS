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

// TODO properly use alpha and beta for GEMV
template <typename ValueType>
void control_gemv(const matrix_info m_info, ValueType alpha,
                  const std::vector<ValueType> &mtx, const matrix_info x_info,
                  ValueType beta, const std::vector<ValueType> &x,
                  std::vector<ValueType> &res) {
    if (x_info.size[1] != 1) {
        throw "Error!";
    }
    for (std::size_t i = 0; i < x_info.size[0]; ++i) {
        res[i * x_info.stride] *= beta;
    }
    for (std::size_t row = 0; row < m_info.size[0]; ++row) {
        for (std::size_t col = 0; col < m_info.size[1]; ++col) {
            const std::size_t midx = row * m_info.stride + col;
            res[row] += alpha * mtx[midx] * x[col * x_info.stride];
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
        info.get_1d_size() > tmp.size() || info.size[1] != 1) {
        throw "Error";
    }
    for (std::size_t row = 0; row < info.size[0]; ++row) {
        const std::size_t midx = row * info.stride;
        tmp[midx] = ValueType{};
    }
    for (std::size_t row = 0; row < info.size[0]; ++row) {
        const std::size_t midx = row * info.stride;
        const auto v1 = get_value(mtx1[midx]);
        const decltype(v1) v2 = get_value(mtx2[midx]);
        const auto delta = std::abs(v1 - v2);
        tmp[midx] = delta;
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

template <typename ValueType>
class GemvMemory {
   public:
    template <typename MtxDist, typename VectDist, typename RndEngine>
    GemvMemory(std::size_t max_rows, std::size_t max_cols, MtxDist &&mtx_dist,
               VectDist &&vect_dist, RndEngine &&engine)
        : m_info_{{max_rows, max_cols}},
          x_info_{{max_cols, 1}},
          res_info_{{max_rows, 1}},
          cpu_mtx_(gen_mtx<ValueType>(m_info_, mtx_dist, engine)),
          cpu_x_(gen_mtx<ValueType>(x_info_, vect_dist, engine)),
          cpu_res_(gen_mtx<ValueType>(res_info_, vect_dist, engine)),
          gpu_mtx_(m_info_.get_1d_size()),
          gpu_x_(x_info_.get_1d_size()),
          gpu_res_(res_info_.get_1d_size()) {
        gpu_mtx_.copy_from(cpu_mtx_);
        gpu_x_.copy_from(cpu_x_);
        gpu_res_.copy_from(cpu_res_);
    }
    template <typename OtherType>
    GemvMemory(const GemvMemory<OtherType> &other)
        : m_info_(other.m_info_),
          x_info_(other.x_info_),
          res_info_(other.res_info_),
          cpu_mtx_(m_info_.get_1d_size()),
          cpu_x_(x_info_.get_1d_size()),
          cpu_res_(res_info_.get_1d_size()),
          gpu_mtx_(m_info_.get_1d_size()),
          gpu_x_(x_info_.get_1d_size()),
          gpu_res_(res_info_.get_1d_size()) {
        convert_mtx(m_info_, other.cpu_mtx_, cpu_mtx_,
                    [](OtherType v) { return static_cast<ValueType>(v); });
        convert_mtx(x_info_, other.cpu_x_, cpu_x_,
                    [](OtherType v) { return static_cast<ValueType>(v); });
        convert_mtx(res_info_, other.cpu_res_, cpu_res_,
                    [](OtherType v) { return static_cast<ValueType>(v); });
        gpu_mtx_.copy_from(cpu_mtx_);
        gpu_x_.copy_from(cpu_x_);
        gpu_res_.copy_from(cpu_res_);
    }

    ValueType *cpu_mtx() { return cpu_mtx_.data(); }
    ValueType *cpu_x() { return cpu_x_.data(); }
    ValueType *cpu_res() { return cpu_res_.data(); }

    ValueType *gpu_mtx() { return gpu_mtx_.data(); }
    ValueType *gpu_x() { return gpu_x_.data(); }
    ValueType *gpu_res() { return gpu_res_.data(); }

    const matrix_info m_info_;
    const matrix_info x_info_;
    const matrix_info res_info_;

   //private:
    std::vector<ValueType> cpu_mtx_;
    std::vector<ValueType> cpu_x_;
    std::vector<ValueType> cpu_res_;

    GpuMemory<ValueType> gpu_mtx_;
    GpuMemory<ValueType> gpu_x_;
    GpuMemory<ValueType> gpu_res_;
};

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

    //*/

    constexpr std::size_t max_rows{24 * 1024};
    constexpr std::size_t max_cols{max_rows};
    constexpr char DELIM{';'};

    const ar_type ar_alpha{1.0};
    const ar_type ar_beta{1.0};
    const st_type st_alpha{static_cast<st_type>(ar_alpha)};
    const st_type st_beta{static_cast<st_type>(ar_beta)};
    std::default_random_engine rengine(42);
    std::uniform_real_distribution<value_type> mtx_dist(-2.0, 2.0);
    // std::normal_distribution<value_type> mtx_dist(1, 2);
    //*
    // std::uniform_real_distribution<value_type> vector_dist(-2.0, 2.0);
    // std::uniform_real_distribution<value_type> vector_dist(1.0, 1.0);
    auto vector_dist = mtx_dist;
    /*/

    auto vector_dist = [rnd = 0](auto val) mutable {
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
        tmp[i] = vector_dist(1);
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

    // auto v_res_ref = std::vector<ar_type>(max_x_info.get_1d_size(),
    // ar_type{}); auto v_reduce =
    //    std::vector<value_type>(max_x_info.get_1d_size(), value_type{});

    auto ar_data =
        GemvMemory<ar_type>(max_rows, max_cols, mtx_dist, vector_dist, rengine);
    auto st_data = GemvMemory<st_type>(ar_data);

    auto cublasHandle = get_cublas_handle();

    static_assert(max_rows == max_cols, "Matrix must be square!");

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
        const matrix_info m_info{{num_rows, num_rows}};
        const matrix_info x_info{{num_rows, 1}};
        // const matrix_info res_info{{num_rows, 1}};

        // v_matrix = gen_mtx<ar_type>(m_info, mtx_dist, rengine);
        // convert_mtx<st_type>(m_info, v_matrix, s_matrix, convert_func);
        // dv_matrix.copy_from(v_matrix);
        // ds_matrix.copy_from(s_matrix);

        double ar_time{};
        auto ar_func = [&]() {
            gemv(m_info, ar_alpha, ar_data.gpu_mtx(), x_info, ar_data.gpu_x(),
                 ar_beta, ar_data.gpu_res());
        };
        double st_time{};
        auto st_func = [&]() {
            gemv(m_info, st_alpha, st_data.gpu_mtx(), x_info, st_data.gpu_x(),
                 st_beta, st_data.gpu_res());
        };
        double acc_ar_time{};
        auto acc_ar_func = [&]() {
            acc_gemv<ar_type>(m_info, ar_alpha, ar_data.gpu_mtx(), x_info,
                              ar_data.gpu_x(), ar_beta, ar_data.gpu_res());
        };
        double acc_mix_time{};
        auto acc_mix_func = [&]() {
            acc_gemv<ar_type>(m_info, ar_alpha, st_data.gpu_mtx(), x_info,
                              st_data.gpu_x(), ar_beta, st_data.gpu_res());
        };
        double cublas_ar_time{};
        auto cublas_ar_func = [&]() {
            cublas_gemv(cublasHandle.get(), m_info, ar_alpha, ar_data.gpu_mtx(),
                        x_info, ar_data.gpu_x(), ar_beta, ar_data.gpu_res());
        };
        double cublas_st_time{};
        auto cublas_st_func = [&]() {
            cublas_gemv(cublasHandle.get(), m_info, st_alpha, st_data.gpu_mtx(),
                        x_info, st_data.gpu_x(), st_beta, st_data.gpu_res());
        };
        // ar_type d_error{};
        [[gnu::unused, maybe_unused]] value_type st_error{};
        //[[ gnu::unused, maybe_unused ]] ar_type acc_ar_error{};
        [[gnu::unused, maybe_unused]] value_type acc_mix_error{};
        [[gnu::unused, maybe_unused]] value_type cublas_ar_error{};
        [[gnu::unused, maybe_unused]] value_type cublas_st_error{};

        // control_gemv(m_info, v_matrix, x_info, v_b, v_res_ref);

        ar_time = benchmark_function(ar_func);
        // ar_func();
        // dv_res.get_vector(v_res);
        // d_error = compare(x_info, v_res_ref, v_res, v_reduce);
        // v_res_ref = v_res;

        st_time = benchmark_function(st_func);
        // ds_res.get_vector(s_res);
        //*
        // std::cout << "single: x_res[0] = " << s_res[0] << '\n';
        // st_error = compare(x_info, v_res_ref, s_res, v_reduce);
        /*/
        convert_mtx(x_info, s_res, v_reduce, [](st_type v) { return v.e; });
        st_error = reduce<value_type>(
            x_info, v_reduce, [](value_type a, value_type b) { return a + b; });
        //*/

        acc_ar_time = benchmark_function(acc_ar_func);
        acc_mix_time = benchmark_function(acc_mix_func);
        // ds_res.get_vector(s_res);
        //*
        // std::cout << "access: x_res[0] = " << s_res[0] << '\n';
        // acc_mix_error = compare(x_info, v_res_ref, s_res, v_reduce);
        /*/
        convert_mtx(x_info, s_res, v_reduce, [](st_type v) { return v.e; });
        acc_mix_error = reduce<value_type>(
            x_info, v_reduce, [](value_type a, value_type b) { return a + b; });
        //*/

        cublas_ar_time = benchmark_function(cublas_ar_func);
        // dv_res.get_vector(v_res);
        // auto cd_error = compare(x_info, v_res_ref, v_res, v_reduce);
        // std::cout << cd_error << '\n';
        // std::cout << st_error << ' ' << acc_mix_error << '\t'
        //          << (st_error / acc_mix_error) << '\n';
        cublas_st_time = benchmark_function(cublas_st_func);
        // ds_res.get_vector(s_res);
        //*
        // std::cout << "access: x_res[0] = " << s_res[0] << '\n';
        // cublas_st_error = compare(x_info, v_res_ref, s_res, v_reduce);

        //*
        std::cout << num_rows << DELIM << ar_time << DELIM << st_time << DELIM
                  << acc_ar_time << DELIM << acc_mix_time << DELIM
                  << cublas_ar_time << DELIM << cublas_st_time << '\n';
        //*/
        /*
        std::cout << "Comparison:"
                << "\nDouble: " << d_error
                << "\nSingle: " << st_error
                << "\nAcc_vv: " << acc_ar_error
                << "\nAcc_vs: " << acc_mix_error
                << '\n';
        //*/
    }
}
