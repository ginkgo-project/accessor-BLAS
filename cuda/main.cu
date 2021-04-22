#include <array>
#include <cmath>
#include <ios>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

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

template <typename ValueType>
void compare(const matrix_info info, const std::vector<ValueType> &mtx1,
             const std::vector<ValueType> &mtx2) {
    for (std::size_t row = 0; row < info.size[0]; ++row) {
        for (std::size_t col = 0; col < info.size[1]; ++col) {
            const std::size_t midx = row * info.stride + col;
            const auto v1 = mtx1[midx];
            const auto v2 = mtx2[midx];
            if (std::abs(v1 - v2) >
                std::numeric_limits<ValueType>::epsilon() * 8) {
                std::cout << "Mismatch at (" << row << ", " << col
                          << "): " << v1 << " vs. " << v2 << '\n';
            }
        }
    }
}

template <typename t1, typename t2, typename t3, typename t4, typename t5>
void test(t1, t2, t3, t4, t5) {
    static_assert(std::is_same<t1, t2>::value,
                  "GKO_DECLARE_OUTPLACE_ABSOLUTE_DENSE_KERNEL");
}

int main() {
    using vtype = double;
    using stype = float;
    constexpr int bench_iters{10};
    constexpr std::int32_t max_rows{24 * 1024};
    constexpr std::int32_t max_mtx_size{max_rows * max_rows};
    std::default_random_engine rengine(42);
    std::uniform_real_distribution<stype> val_dist(1.0, 2.0);
    std::uniform_real_distribution<stype> one_dist(1.0, 1.0);
    constexpr matrix_info max_minfo{{max_rows, max_rows}};
    constexpr char DELIM{';'};

    const matrix_info max_vinfo{{max_rows, 1}};

    auto v_matrix = gen_mtx<vtype>(max_minfo, val_dist, rengine);
    auto s_matrix = convert_mtx<stype>(max_minfo, v_matrix);

    auto v_b = gen_mtx<vtype>(max_vinfo, one_dist, rengine);
    auto s_b = convert_mtx<stype>(max_vinfo, v_matrix);

    auto v_res = std::vector<vtype>(max_vinfo.get_1d_size(), vtype{});
    auto s_res = std::vector<stype>(max_vinfo.get_1d_size(), stype{});

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
        << DELIM << "GEMV Acc<fp64, fp64>"<< DELIM << "GEMV Acc<fp32, fp64>" << '\n';

    constexpr int start{max_rows / 48};
    constexpr int row_incr{start};
    for (int num_rows = start; num_rows <= max_rows; num_rows += row_incr) {
        std::cout << num_rows << DELIM;

        const matrix_info minfo{{num_rows, num_rows}};
        const matrix_info vinfo{{num_rows, 1}};

        CudaTimer ctimer;

        // static_assert(minfo.size[0] == minfo.size[1], "Matrix must be
        // square!");

        // Benchmark double
        // Warmup
        spmv(minfo, dv_matrix.data(), vinfo, dv_b.data(), dv_res.data());
        synchronize();

        double double_time{};
        for (int i = 0; i < bench_iters; ++i) {
            ctimer.start();
            spmv(minfo, dv_matrix.data(), vinfo, dv_b.data(), dv_res.data());
            ctimer.stop();
            double_time += ctimer.get_time();
            ctimer.reset();
        }
        std::cout << double_time / bench_iters << DELIM;

        // Benchmark float
        // Warmup
        spmv(minfo, ds_matrix.data(), vinfo, ds_b.data(), ds_res.data());
        synchronize();

        double single_time{};
        for (int i = 0; i < bench_iters; ++i) {
            ctimer.start();
            spmv(minfo, ds_matrix.data(), vinfo, ds_b.data(), ds_res.data());
            ctimer.stop();
            single_time += ctimer.get_time();
            ctimer.reset();
        }
        std::cout << single_time / bench_iters << DELIM;

        // Benchmark Accessor<fp64, fp32>
        // Warmup
        acc_spmv<vtype>(minfo, ds_matrix.data(), vinfo, ds_b.data(),
                        ds_res.data());
        synchronize();

        double acc_dd_time{};
        for (int i = 0; i < bench_iters; ++i) {
            ctimer.start();
            acc_spmv<vtype>(minfo, dv_matrix.data(), vinfo, dv_b.data(),
                            dv_res.data());
            ctimer.stop();
            acc_dd_time += ctimer.get_time();
            ctimer.reset();
        }
        std::cout << acc_dd_time / bench_iters << DELIM;

        double acc_df_time{};
        for (int i = 0; i < bench_iters; ++i) {
            ctimer.start();
            acc_spmv<vtype>(minfo, ds_matrix.data(), vinfo, ds_b.data(),
                            ds_res.data());
            ctimer.stop();
            acc_df_time += ctimer.get_time();
            ctimer.reset();
        }
        std::cout << acc_df_time / bench_iters << '\n';

    }

    auto gpu_res = dv_res.get_vector();

    // Control impl:
    control_spmv(max_minfo, v_matrix, max_vinfo, v_b, v_res);

    std::cout.precision(16);
    std::cout << std::scientific;

    std::cout << "Comparison:\n";
    compare(max_vinfo, v_res, gpu_res);
}
