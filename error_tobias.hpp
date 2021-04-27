#pragma once

#include <cmath>
#include <iostream>
#include <limits>

template <typename T>
struct error_number {
    T v;
    T e;
    constexpr static T unit_roundoff() {
        return std::numeric_limits<T>::epsilon() / 2;
    }
    constexpr error_number(T v, T e) : v{v}, e{e} {}
    constexpr error_number(T v) : v{v}, e{} {}
    constexpr error_number() : v{}, e{} {}
    explicit operator T() const { return v; }
    constexpr error_number& operator=(T val) {
        v = val;
        e = T{};
    }
    constexpr error_number& operator=(error_number other) {
        v = other.v;
        e = other.e;
    }
};

template <typename T>
error_number<T> operator+(error_number<T> a, error_number<T> b) {
    error_number<T> result{};
    result.v = a.v + b.v;
    result.e = a.unit_roundoff() * std::abs(result.v) + a.e + b.e;
    return result;
}

template <typename T>
error_number<T> operator-(error_number<T> a, error_number<T> b) {
    error_number<T> result{};
    result.v = a.v - b.v;
    result.e = a.unit_roundoff() * std::abs(result.v) + a.e + b.e;
    return result;
}

template <typename T>
error_number<T> operator-(error_number<T> a) {
    return {-a.v, a.e};
}

template <typename T>
error_number<T> operator*(error_number<T> a, error_number<T> b) {
    error_number<T> result{};
    result.v = a.v * b.v;
    result.e = a.unit_roundoff() * std::abs(result.v) + a.e * std::abs(b.v) +
               b.e * std::abs(a.v);
    return result;
}

template <typename T>
error_number<T> operator/(error_number<T> a, error_number<T> b) {
    error_number<T> result{};
    result.v = a.v / b.v;
    result.e = a.unit_roundoff() * std::abs(result.v) +
               (a.e * std::abs(b.v) + b.e * std::abs(a.v)) / (b.v * b.v);
    return result;
}

template <typename T>
error_number<T>& operator+=(error_number<T>& a, error_number<T> b) {
    a = a + b;
    return a;
}

template <typename T>
error_number<T>& operator-=(error_number<T>& a, error_number<T> b) {
    a = a - b;
    return a;
}

template <typename T>
error_number<T>& operator*=(error_number<T>& a, error_number<T> b) {
    a = a * b;
    return a;
}

template <typename T>
error_number<T>& operator/=(error_number<T>& a, error_number<T> b) {
    a = a / b;
    return a;
}

namespace std {

template <typename T>
error_number<T> sqrt(error_number<T> a) {
    error_number<T> result{};
    result.v = std::sqrt(a.v);
    result.e =
        a.unit_roundoff() * std::abs(result.v) + a.e / (2 * std::abs(result.v));
    return result;
}

template <typename T>
error_number<T> log(error_number<T> a) {
    error_number<T> result{};
    result.v = std::log(a.v);
    result.e =
        a.unit_roundoff() * std::abs(result.v) + a.e / std::abs(result.v);
    return result;
}

template <typename T>
error_number<T> pow(error_number<T> a, T b) {
    error_number<T> result{};
    result.v = std::pow(a.v, b);
    result.e =
        a.unit_roundoff() * std::abs(result.v) + a.e * b * std::pow(a.v, b - 1);
    return result;
}

template <typename T>
error_number<T> abs(error_number<T> a) {
    error_number<T> result{};
    result.v = std::abs(a.v);
    result.e = a.e;
    return result;
}

template <typename T>
error_number<T> min(error_number<T> a, error_number<T> b) {
    return a.v > b.v ? b : a;
}

template <typename T>
error_number<T> max(error_number<T> a, error_number<T> b) {
    return a.v > b.v ? a : b;
}

}  // namespace std

template <typename T>
std::ostream& operator<<(std::ostream& stream, error_number<T> a) {
    return stream << a.v << " + " << a.e;
}

template <typename T>
std::istream& operator>>(std::istream& stream, error_number<T>& a) {
    a.e = 0;
    return stream >> a.v;
}
