#pragma once

#include <iostream>
#include <stdexcept>

#include "memory.cuh"
#include "utils.cuh"

template <typename ValueType, typename ValueDist, typename Engine>
Memory<ValueType> gen_mtx(const matrix_info &info, ValueDist &&dist,
                          Engine &&engine) {
    if (info.stride < info.size[1]) {
        throw std::runtime_error("Wrong use of stride");
    }
    Memory<ValueType> res(Memory<ValueType>::Device::cpu, info.get_1d_size());
    auto ptr = res.data();

    for (std::size_t row = 0; row < info.size[0]; ++row) {
        for (std::size_t col = 0; col < info.size[1]; ++col) {
            const std::size_t idx = row * info.stride + col;
            ptr[idx] = dist(engine);
        }
    }

    return res;
}

template <typename ResultType, typename InputType, typename Callable>
void convert_mtx(const matrix_info &info, const InputType *input,
                 ResultType *output, Callable convert) {
    for (std::size_t row = 0; row < info.size[0]; ++row) {
        for (std::size_t col = 0; col < info.size[1]; ++col) {
            const std::size_t idx = row * info.stride + col;
            output[idx] = convert(input[idx]);
        }
    }
}

template <typename ValueType>
void print_mtx(const matrix_info &info, const ValueType *vec) {
    for (std::size_t i = 0; i < info.size[0]; ++i) {
        for (std::size_t j = 0; j < info.size[1]; ++j) {
            const auto idx = i * info.stride + j;
            std::cout << vec[idx] << '\t';
        }
        std::cout << '\n';
    }
}

