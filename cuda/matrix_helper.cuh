#pragma once

#include <cmath>
#include <ios>
#include <iostream>
#include <stdexcept>


#include "memory.cuh"
#include "utils.cuh"


/**
 * Generates a random matrix and returns its memory (on the CPU).
 *
 * @note sub-normal values are filtered out
 *
 * @tparam ValueType  value type of the matrix elements
 * @tparam ValueDist  type of the value distribution
 * @tparam Engine  type of the random generator engine
 *
 * @param info  information about the matrix to be generated
 * @param dist  value distribution
 * @param engine  random generating engine
 *
 * @returns the memory containing a randomly generated matrix on the CPU
 */
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


/**
 * Overwrites the memory with random values.
 *
 * @tparam ValueType  value type of the matrix elements
 * @tparam ValueDist  type of the value distribution
 * @tparam Engine  type of the random generator engine
 *
 * @param info  information about the matrix to be written
 * @param dist  value distribution
 * @param engine  random generating engine
 * @param out  memory where the randomly generated values will be written to
 */
template <typename ValueType, typename ValueDist, typename Engine>
void write_random(const matrix_info &info, ValueDist &&dist, Engine &&engine,
                  ValueType *out)
{
    for (std::size_t row = 0; row < info.size[0]; ++row) {
        for (std::size_t col = 0; col < info.size[1]; ++col) {
            const std::size_t idx = row * info.stride + col;
            out[idx] = dist(engine);
        }
    }
}


/**
 * Reads the input matrix, converts the values, and writes the result to the
 * output matrix.
 *
 * @tparam ResultType  result / output type
 * @tparam InputType  type of the input matrix
 * @tparam Callable  type of the conversion function
 *
 * @param info  information about the matrix (both input and output must adhere
 *              to this information)
 * @param input  input matrix
 * @param output  output matrix, where the conversion will be written to
 * @param convert  conversion function used to convert each input value to the
 *                 output value
 */
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


/**
 * Prints the matrix to standard output.
 *
 * @tparam ValueType  type of the matrix values
 *
 * @param info  information about the matrix (both input and output must adhere
 *              to this information)
 * @param mtx  input matrix
 */
template <typename ValueType>
void print_mtx(const matrix_info &info, const ValueType *mtx)
{
    auto cout_flags = std::cout.flags();
    auto old_prec = std::cout.precision();
    std::cout.precision(7);
    // showpos: show + sign for positive numbers
    std::cout << std::fixed << std::showpos;
    for (std::size_t i = 0; i < info.size[0]; ++i) {
        for (std::size_t j = 0; j < info.size[1]; ++j) {
            const auto idx = i * info.stride + j;
            std::cout << mtx[idx] << '\t';
        }
        std::cout << '\n';
    }
    // Does not copy some, like precision info
    std::cout.flags(cout_flags);
    std::cout.precision(old_prec);
}
