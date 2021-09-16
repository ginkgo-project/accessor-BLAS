#pragma once

#include <cmath>
#include <ios>
#include <iostream>
#include <stdexcept>


#include "memory.cuh"
#include "utils.cuh"


// Note: sub-normal values are filtered out
template <typename ValueType, typename ValueDist, typename Engine>
Memory<ValueType> gen_mtx(const matrix_info &info, ValueDist &&dist,
                          Engine &&engine)
{
    if (info.stride < info.size[1]) {
        throw std::runtime_error("Wrong use of stride");
    }
    Memory<ValueType> res(Memory<ValueType>::Device::cpu, info.get_1d_size());
    auto ptr = res.data();

    for (std::size_t row = 0; row < info.size[0]; ++row) {
        for (std::size_t col = 0; col < info.size[1]; ++col) {
            const std::size_t idx = row * info.stride + col;
            ValueType val{};
            do {
                val = dist(engine);
            } while (!std::isnormal(val));
            ptr[idx] = val;
        }
    }

    return res;
}


template <typename ValueType, typename ValueDist, typename Engine>
void write_random(const matrix_info &info, ValueDist &&dist, Engine &&engine,
                  ValueType *io)
{
    for (std::size_t row = 0; row < info.size[0]; ++row) {
        for (std::size_t col = 0; col < info.size[1]; ++col) {
            const std::size_t idx = row * info.stride + col;
            io[idx] = dist(engine);
        }
    }
}


template <typename ResultType, typename InputType, typename Callable>
void convert_mtx(const matrix_info &info, const InputType *input,
                 ResultType *output, Callable convert)
{
    for (std::size_t row = 0; row < info.size[0]; ++row) {
        for (std::size_t col = 0; col < info.size[1]; ++col) {
            const std::size_t idx = row * info.stride + col;
            output[idx] = convert(input[idx]);
        }
    }
}


template <typename ValueType>
void print_mtx(const matrix_info &info, const ValueType *vec)
{
    auto cout_flags = std::cout.flags();
    auto old_prec = std::cout.precision();
    std::cout.precision(7);
    // showpos: show + sign for positive numbers
    std::cout << std::fixed << std::showpos;
    for (std::size_t i = 0; i < info.size[0]; ++i) {
        for (std::size_t j = 0; j < info.size[1]; ++j) {
            const auto idx = i * info.stride + j;
            std::cout << vec[idx] << '\t';
        }
        std::cout << '\n';
    }
    // Does not copy some, like precision info
    std::cout.flags(cout_flags);
    std::cout.precision(old_prec);
}
