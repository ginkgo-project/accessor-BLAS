#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

#define GKO_ATTR __host__ __device__

template <typename T>
struct error_number {
    using value_type = T;

    T v;
    T e;
    constexpr static T unit_roundoff() {
        return std::numeric_limits<T>::epsilon() / 2;
    }
    constexpr error_number(T v, T e) : v{v}, e{e} {}
    constexpr error_number(T v) : v{v}, e{} {}
    constexpr error_number() : v{}, e{} {}
    template <typename O>
    constexpr error_number(error_number<O> o)
        : v{static_cast<T>(o.v)},
          e{(unit_roundoff() <= error_number<O>::unit_roundoff())
                ? static_cast<T>(o.e)
                : static_cast<T>(o.e +
                                 error_number<T>::unit_roundoff() * o.v)} {}

    explicit operator T() const { return v; }
    constexpr error_number& operator=(T val) {
        v = val;
        e = T{};
        return *this;
    }
    constexpr error_number& operator=(error_number other) {
        v = other.v;
        e = other.e;
        return *this;
    }
};

template <typename T>
GKO_ATTR error_number<T> operator+(error_number<T> a, error_number<T> b) {
    error_number<T> result{};
    result.v = a.v + b.v;
    result.e = a.unit_roundoff() * std::abs(result.v) + a.e + b.e;
    return result;
}

template <typename T>
GKO_ATTR error_number<T> operator-(error_number<T> a, error_number<T> b) {
    error_number<T> result{};
    result.v = a.v - b.v;
    result.e = a.unit_roundoff() * std::abs(result.v) + a.e + b.e;
    return result;
}

template <typename T>
GKO_ATTR error_number<T> operator-(error_number<T> a) {
    return {-a.v, a.e};
}

template <typename T>
GKO_ATTR error_number<T> operator*(error_number<T> a, error_number<T> b) {
    error_number<T> result{};
    result.v = a.v * b.v;
    result.e = a.unit_roundoff() * std::abs(result.v) + a.e * std::abs(b.v) +
               b.e * std::abs(a.v);
    return result;
}

template <typename T>
GKO_ATTR error_number<T> operator/(error_number<T> a, error_number<T> b) {
    error_number<T> result{};
    result.v = a.v / b.v;
    result.e = a.unit_roundoff() * std::abs(result.v) +
               (a.e * std::abs(b.v) + b.e * std::abs(a.v)) / (b.v * b.v);
    return result;
}

template <typename T>
GKO_ATTR error_number<T>& operator+=(error_number<T>& a, error_number<T> b) {
    a = a + b;
    return a;
}

template <typename T>
GKO_ATTR error_number<T>& operator-=(error_number<T>& a, error_number<T> b) {
    a = a - b;
    return a;
}

template <typename T>
GKO_ATTR error_number<T>& operator*=(error_number<T>& a, error_number<T> b) {
    a = a * b;
    return a;
}

template <typename T>
GKO_ATTR error_number<T>& operator/=(error_number<T>& a, error_number<T> b) {
    a = a / b;
    return a;
}

namespace std {

template <typename T>
GKO_ATTR error_number<T> sqrt(error_number<T> a) {
    error_number<T> result{};
    result.v = std::sqrt(a.v);
    result.e =
        a.unit_roundoff() * std::abs(result.v) + a.e / (2 * std::abs(result.v));
    return result;
}

template <typename T>
GKO_ATTR error_number<T> log(error_number<T> a) {
    error_number<T> result{};
    result.v = std::log(a.v);
    result.e =
        a.unit_roundoff() * std::abs(result.v) + a.e / std::abs(result.v);
    return result;
}

template <typename T>
GKO_ATTR error_number<T> pow(error_number<T> a, T b) {
    error_number<T> result{};
    result.v = std::pow(a.v, b);
    result.e =
        a.unit_roundoff() * std::abs(result.v) + a.e * b * std::pow(a.v, b - 1);
    return result;
}

template <typename T>
GKO_ATTR error_number<T> abs(error_number<T> a) {
    error_number<T> result{};
    result.v = std::abs(a.v);
    result.e = a.e;
    return result;
}

template <typename T>
GKO_ATTR error_number<T> min(error_number<T> a, error_number<T> b) {
    return a.v > b.v ? b : a;
}

template <typename T>
GKO_ATTR error_number<T> max(error_number<T> a, error_number<T> b) {
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
