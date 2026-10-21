/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace detail {

// An independent execution oracle for source-operation estimates. Constants,
// copies, comparisons and unary signs are free; elementary results cost one.
class CountingReal {
    double value_ = 0;
    inline static uint64_t count_ = 0;

public:
    using value_type = double;

    CountingReal() = default;
    template <class Value, std::enable_if_t<std::is_arithmetic_v<Value>, int> = 0>
    CountingReal(Value value)
      : value_(value) {}

    double value() const {
        return value_;
    }
    static void reset() {
        count_ = 0;
    }
    static uint64_t count() {
        return count_;
    }
    static constexpr int size() {
        return 1;
    }
    static void add(uint64_t count) {
        count_ += count;
    }
    CountingReal& operator+=(CountingReal other) {
        ++count_;
        value_ += other.value_;
        return *this;
    }
    CountingReal& operator-=(CountingReal other) {
        ++count_;
        value_ -= other.value_;
        return *this;
    }
    CountingReal& operator*=(CountingReal other) {
        ++count_;
        value_ *= other.value_;
        return *this;
    }
    friend CountingReal operator+(CountingReal left, CountingReal right) {
        return left += right;
    }
    friend CountingReal operator-(CountingReal left, CountingReal right) {
        return left -= right;
    }
    friend CountingReal operator*(CountingReal left, CountingReal right) {
        return left *= right;
    }
    friend CountingReal operator/(CountingReal left, CountingReal right) {
        ++count_;
        return left.value_ / right.value_;
    }
    CountingReal operator-() const {
        return -value_;
    }
    friend bool operator<(CountingReal left, CountingReal right) {
        return left.value_ < right.value_;
    }
    friend bool operator>(CountingReal left, CountingReal right) {
        return left.value_ > right.value_;
    }
    friend bool operator>=(CountingReal left, CountingReal right) {
        return left.value_ >= right.value_;
    }

    friend CountingReal simdFma(CountingReal first, CountingReal second, CountingReal third) {
        CountingReal::add(2);
        return std::fma(first.value(), second.value(), third.value());
    }

    friend CountingReal simdSqrt(CountingReal value) {
        CountingReal::add(1);
        return std::sqrt(value.value());
    }

    friend CountingReal exp(CountingReal value) {
        CountingReal::add(1);
        return std::exp(value.value());
    }

    friend void sincos(CountingReal value, CountingReal* sine, CountingReal* cosine) {
        CountingReal::add(2);
        *sine = std::sin(value.value());
        *cosine = std::cos(value.value());
    }

    friend void erfcexp(CountingReal value, CountingReal* complement, CountingReal* exponential) {
        CountingReal::add(2);
        *complement = std::erfc(value.value());
        *exponential = std::exp(-value.value() * value.value());
    }
};

}    // namespace detail
