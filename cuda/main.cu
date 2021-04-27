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
void control_spmv(const matrix_info minfo, const std::vector<ValueType> &mtx,
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
    return info.size[0] == 1 ? op(tmp[0], {}) : op(tmp[0], tmp[1]);
}

template <typename ReferenceType, typename OtherType>
ReferenceType compare(const matrix_info info,
                      const std::vector<ReferenceType> &mtx1,
                      const std::vector<OtherType> &mtx2,
                      std::vector<ReferenceType> &tmp) {
    // ReferenceType error{};
    if (info.get_1d_size() > mtx1.size() || info.get_1d_size() > mtx2.size() ||
        info.get_1d_size() > tmp.size()) {
        throw "Error";
    }
    for (std::size_t row = 0; row < info.size[0]; ++row) {
        for (std::size_t col = 0; col < info.size[1]; ++col) {
            const std::size_t midx = row * info.stride + col;
            tmp[midx] = ReferenceType{};
        }
    }
    for (std::size_t row = 0; row < info.size[0]; ++row) {
        for (std::size_t col = 0; col < info.size[1]; ++col) {
            const std::size_t midx = row * info.stride + col;
            const auto v1 = mtx1[midx];
            const ReferenceType v2 = mtx2[midx];
            const auto delta = std::abs(v1 - v2);
            if (delta > std::numeric_limits<ReferenceType>::epsilon()) {
                /*
                std::cout << "Mismatch at (" << row << ", " << col
                          << "): " << v1 << " vs. " << v2 << '\n';
                //*/
                // error = std::max(delta, error);
                tmp[midx] = delta;
            }
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
    return reduce<ReferenceType>(
        info, tmp, [](ReferenceType o1, ReferenceType o2) { return o1 + o2; });
}

template <typename t1, typename t2, typename t3, typename t4, typename t5>
void test(t1, t2, t3, t4, t5) {
    static_assert(std::is_same<t1, t2>::value,
                  "GKO_DECLARE_OUTPLACE_ABSOLUTE_DENSE_KERNEL");
}

template <typename Callable>
double benchmark_function(Callable func) {
    constexpr int bench_iters{0};
    CudaTimer ctimer;
    double time_ms{};
    // Warmup
    func();
    synchronize();
    for (int i = 0; i < bench_iters; ++i) {
        ctimer.start();
        func();
        ctimer.stop();
        time_ms += ctimer.get_time();
        ctimer.reset();
    }
    return bench_iters == 0 ? double{}
                            : time_ms / static_cast<double>(bench_iters);
}

int main() {
    using vtype = double; //error_number<double>;
    using stype = float; //error_number<float>;
    // constexpr std::int32_t max_rows{24 * 1024};
    constexpr std::size_t max_rows{16 * 1024};
    constexpr matrix_info max_minfo{{max_rows, max_rows}};
    constexpr char DELIM{';'};

    std::default_random_engine rengine(42);
    std::uniform_real_distribution<double> val_dist(1.0, 2.0);
    std::uniform_real_distribution<float> one_dist(1.0, 2.0);

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

    auto v_matrix = gen_mtx<vtype>(max_minfo, val_dist, rengine);
    std::vector<stype> s_matrix(max_minfo.get_1d_size());
    convert_mtx<stype>(max_minfo, v_matrix, s_matrix);

    auto v_b = gen_mtx<vtype>(max_vinfo, one_dist, rengine);
    std::vector<stype> s_b(max_vinfo.get_1d_size());
    convert_mtx<stype>(max_vinfo, v_b, s_b);

    auto v_res = std::vector<vtype>(max_vinfo.get_1d_size(), vtype{});
    auto v_res_ref = std::vector<vtype>(max_vinfo.get_1d_size(), vtype{});
    auto s_res = std::vector<stype>(max_vinfo.get_1d_size(), stype{});
    auto v_reduce = std::vector<vtype>(max_vinfo.get_1d_size(), vtype{});

    auto dv_matrix = GpuMemory<vtype>(max_minfo.get_1d_size());
    dv_matrix.copy_from(v_matrix);
    auto ds_matrix = GpuMemory<stype>(max_minfo.get_1d_size());
    ds_matrix.copy_from(s_matrix);

    auto dv_b = GpuMemory<vtype>(max_vinfo.get_1d_size());
    dv_b.copy_from(v_b);
    auto ds_b = GpuMemory<stype>(max_vinfo.get_1d_size());
    ds_b.copy_from(s_b);
    auto dv_res = GpuMemory<vtype>(max_vinfo.get_1d_size());
    auto ds_res = GpuMemory<stype>(max_vinfo.get_1d_size());

    std::cout << "Num Rows" << DELIM << "GEMV double" << DELIM << "GEMV float"
              << DELIM << "GEMV Acc<fp64, fp64>" << DELIM
              << "GEMV Acc<fp64, fp32>" << '\n';

    std::cout.precision(16);
    std::cout << std::scientific;

    int single_better{};
    int acc_better{};
    // constexpr auto start = max_rows / 1;
    // constexpr auto row_incr = start;
    // for (auto num_rows = start; num_rows <= max_rows; num_rows += row_incr) {
    for (int i = 0; i < 10; ++i) {
        const matrix_info minfo{{max_rows, max_rows}};
        const matrix_info vinfo{{max_rows, 1}};

        v_matrix = gen_mtx<vtype>(max_minfo, val_dist, rengine);
        convert_mtx<stype>(max_minfo, v_matrix, s_matrix);
        dv_matrix.copy_from(v_matrix);
        ds_matrix.copy_from(s_matrix);

        double d_time{};
        auto d_func = [&]() {
            spmv(minfo, dv_matrix.data(), vinfo, dv_b.data(), dv_res.data());
        };
        double s_time{};
        auto s_func = [&]() {
            spmv(minfo, ds_matrix.data(), vinfo, ds_b.data(), ds_res.data());
        };
        double avs_time{};
        auto avs_func = [&]() {
            acc_spmv<double>(minfo, dv_matrix.data(), vinfo, dv_b.data(),
                             dv_res.data());
        };
        double ads_time{};
        auto ads_func = [&]() {
            acc_spmv<double>(minfo, ds_matrix.data(), vinfo, ds_b.data(),
                             ds_res.data());
        };
        vtype d_error{};
        vtype s_error{};
        vtype avv_error{};
        vtype avs_error{};

        // control_spmv(minfo, v_matrix, vinfo, v_b, v_res_ref);

        // d_time = benchmark_function(d_func);
        d_func();
        dv_res.get_vector(v_res);
        // d_error = compare(max_vinfo, v_res_ref, v_res, v_reduce);
        v_res_ref = v_res;

        s_time = benchmark_function(s_func);
        ds_res.get_vector(s_res);
        s_error = compare(max_vinfo, v_res_ref, s_res, v_reduce);

        // avs_time = benchmark_function(avs_func);
        // dv_res.get_vector(v_res);
        // avv_error = compare(max_vinfo, v_res_ref, v_res, v_reduce);

        ads_time = benchmark_function(ads_func);
        ds_res.get_vector(s_res);
        avs_error = compare(max_vinfo, v_res_ref, s_res, v_reduce);

        std::cout << s_error << ' ' << avs_error << ":\t" << (s_error / avs_error) << '\n';

        /*
        std::cout << num_rows << DELIM
                  << d_time << DELIM
                  << s_time << DELIM
                  << avs_time << DELIM
                  << ads_time << '\n';
        //*/
        /*
        std::cout << "Comparison:"
                << "\nDouble: " << d_error
                << "\nSingle: " << s_error
                << "\nAcc_vv: " << avv_error
                << "\nAcc_vs: " << avs_error
                << '\n';
        //*/
        /*
        if (s_error < avs_error) {
            ++single_better;
            std::cout << "Single more precise\n";
        } else if (avs_error < s_error) {
            ++acc_better;
            std::cout << "Acc<fp64, fp32> more precise\n";
        } else {
            std::cout << "Equally precise\n";
        }
        //*/
    }

    std::cout << "Single vs. Accessor better: " << single_better << " vs "
              << acc_better << '\n';
}
