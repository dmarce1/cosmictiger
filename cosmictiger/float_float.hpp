/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifdef __CUDACC__
#define COSMICTIGER_FLOAT_FLOAT_HD __host__ __device__
#else
#define COSMICTIGER_FLOAT_FLOAT_HD
#endif

// Keep compiler fast-math assumptions out of this header, then restore the
// caller's options below. Volatile rounding barriers are retained explicitly.
// CUDA device arithmetic is protected separately by the primitives below.
#if defined(__clang__)
#pragma float_control(precise, on, push)
#elif defined(__GNUC__) && !defined(__CUDACC__)
#pragma GCC push_options
#pragma GCC optimize("no-fast-math", "fp-contract=off")
#endif

namespace detail {

// Each primitive is one rounding step. Volatile stores prevent host fast-math
// reassociation and unintended contraction across these steps. They do not
// change the hardware rounding mode or disable host FTZ/DAZ.
// On CUDA, explicit PTX omits .ftz even under --use_fast_math; the IEEE FMA
// intrinsic likewise ignores that flag. See the NVIDIA CUDA Math API,
// Single Precision Intrinsics, __fmaf_ieee_rn:
// https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html
COSMICTIGER_FLOAT_FLOAT_HD inline float floatFloatAdd(float left, float right) {
#ifdef __CUDA_ARCH__
    float result;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(left), "f"(right));
    return result;
#else
    float volatile result = left + right;
    return result;
#endif
}

COSMICTIGER_FLOAT_FLOAT_HD inline float floatFloatSubtract(float left, float right) {
#ifdef __CUDA_ARCH__
    float result;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(left), "f"(right));
    return result;
#else
    float volatile result = left - right;
    return result;
#endif
}

COSMICTIGER_FLOAT_FLOAT_HD inline float floatFloatMultiply(float left, float right) {
#ifdef __CUDA_ARCH__
    float result;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(left), "f"(right));
    return result;
#else
    float volatile result = left * right;
    return result;
#endif
}

COSMICTIGER_FLOAT_FLOAT_HD inline float floatFloatDivide(float left, float right) {
#ifdef __CUDA_ARCH__
    float result;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(left), "f"(right));
    return result;
#else
    // Volatile operands also prevent constant reciprocal substitution.
    float volatile numerator = left;
    float volatile denominator = right;
    float volatile result = numerator / denominator;
    return result;
#endif
}

COSMICTIGER_FLOAT_FLOAT_HD inline float floatFloatFma(float left, float right, float addend) {
#ifdef __CUDA_ARCH__
    return __fmaf_ieee_rn(left, right, addend);
#else
    float volatile result = ::fmaf(left, right, addend);
    return result;
#endif
}

COSMICTIGER_FLOAT_FLOAT_HD inline float floatFloatSqrt(float value) {
#ifdef __CUDA_ARCH__
    float result;
    asm("sqrt.rn.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
#else
    float volatile argument = value;
    float volatile result = ::sqrtf(argument);
    return result;
#endif
}

COSMICTIGER_FLOAT_FLOAT_HD inline uint32_t floatFloatBits(float value) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(value);
#else
    return std::bit_cast<uint32_t>(value);
#endif
}

COSMICTIGER_FLOAT_FLOAT_HD inline float floatFloatFromBits(uint32_t bits) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(bits);
#else
    return std::bit_cast<float>(bits);
#endif
}

COSMICTIGER_FLOAT_FLOAT_HD inline float floatFloatNegate(float value) {
    // Plain CUDA unary minus becomes neg.ftz.f32 under --use_fast_math.
    return floatFloatFromBits(floatFloatBits(value) ^ 0x80000000U);
}

COSMICTIGER_FLOAT_FLOAT_HD inline bool floatFloatFinite(float value) {
    return (floatFloatBits(value) & 0x7f800000U) != 0x7f800000U;
}

COSMICTIGER_FLOAT_FLOAT_HD inline uint32_t floatFloatOrder(float value) {
    uint32_t const bits = floatFloatBits(value);
    // Integer keys preserve subnormal comparisons under CUDA fast math and
    // equate the two signed zeros. NaNs are handled before comparing keys.
    if ((bits & 0x7fffffffU) == 0) {
        return 0x80000000U;
    }
    return (bits & 0x80000000U) != 0 ? ~bits : (bits ^ 0x80000000U);
}

COSMICTIGER_FLOAT_FLOAT_HD inline float floatFloatNarrow(double value) {
#ifdef __CUDA_ARCH__
    uint64_t const bits = static_cast<uint64_t>(__double_as_longlong(value));
#else
    uint64_t const bits = std::bit_cast<uint64_t>(value);
#endif
    if ((bits & UINT64_C(0x7fffffffffffffff)) == 0) {
        return floatFloatFromBits(static_cast<uint32_t>(bits >> 32) & 0x80000000U);
    }
#ifdef __CUDA_ARCH__
    float result;
    asm("cvt.rn.f32.f64 %0, %1;" : "=f"(result) : "d"(value));
    return result;
#else
    float volatile result = static_cast<float>(value);
    return result;
#endif
}

COSMICTIGER_FLOAT_FLOAT_HD inline double floatFloatWiden(float value) {
#ifdef __CUDA_ARCH__
    double result;
    asm("cvt.f64.f32 %0, %1;" : "=d"(result) : "f"(value));
    return result;
#else
    double volatile result = static_cast<double>(value);
    return result;
#endif
}

}    // namespace detail

// An unevaluated, normalized sum high_ + low_, stored in exactly two floats.
// Ordinary normal-range arithmetic has about 48 significant binary bits
// (roughly 14 decimal digits), but this is not IEEE double or correctly rounded
// 48-bit arithmetic. The exponent range remains that of float; precision falls
// near underflow, and intermediate overflow is not rescued by the low word.
// Requires binary32, round-to-nearest/ties-to-even, and, on the host, gradual
// underflow (FTZ/DAZ disabled). Local compiler options and volatile barriers
// protect host rounding steps in a -ffast-math translation unit, but that
// option may still enable FTZ/DAZ at program startup.
// Arithmetic itself uses only float operations; double is used for conversions.
//
// Algorithms and their range assumptions:
// M. Joldes, J.-M. Muller, V. Popescu, "Tight and Rigorous Error Bounds for
// Basic Building Blocks of Double-Word Arithmetic", ACM TOMS 44(2), 2017.
// https://doi.org/10.1145/3121432 (Algorithms 2, 3, 6, 7, 12, 17).
// Corrected/formalized analysis: J.-M. Muller, L. Rideau, ACM TOMS 48(1), 2022.
// https://doi.org/10.1145/3484514 . Accessible author survey with algorithms:
// S. Boldo et al., "Floating-point arithmetic", Acta Numerica 32, 2023,
// pp. 269-271. https://doi.org/10.1017/S0962492922000101 .
// Historical float-float GPU implementation: G. Da Graca and D. Defour, 2006,
// https://arxiv.org/abs/cs/0603115 .
//
// General finite-path operation counts (before compiler simplification):
//              add/sub   multiply   FMA   divide   sqrt   FLOPs
// + or -          20          0      0       0       0      20
// *                4          2      3       0       0      12
// /               13          2      1       2       0      19
// sqrt            24          1      1       1       1      29
// fma             24          2      3       0       0      32
// Count FMA as two FLOPs and each other arithmetic operation as one. Unary
// signs use zero FLOPs. Exclude conversions, branches, classification and
// integer operations. Each fastTwoSum zero shortcut saves three FLOPs.
// Division/sqrt count as one here, but have different costs from an addition.
class FloatFloat {
    float high_ = 0.0f;
    float low_ = 0.0f;

    COSMICTIGER_FLOAT_FLOAT_HD static FloatFloat twoSum(float left, float right) {
        using namespace detail;
        FloatFloat result;
        result.high_ = floatFloatAdd(left, right);
        if (!floatFloatFinite(result.high_)) {
            return result;
        }
        float const virtualRight = floatFloatSubtract(result.high_, left);
        float const virtualLeft = floatFloatSubtract(result.high_, virtualRight);
        float const rightError = floatFloatSubtract(right, virtualRight);
        float const leftError = floatFloatSubtract(left, virtualLeft);
        result.low_ = floatFloatAdd(leftError, rightError);
        return result;
    }

    // Fast2Sum requires |left| >= |right|; used only at the normalization
    // steps justified by the cited double-word algorithms.
    COSMICTIGER_FLOAT_FLOAT_HD static FloatFloat fastTwoSum(float left, float right) {
        if ((detail::floatFloatBits(right) & 0x7fffffffU) == 0) {
            return FloatFloat(left);
        }
        FloatFloat result;
        result.high_ = detail::floatFloatAdd(left, right);
        if (detail::floatFloatFinite(result.high_)) {
            result.low_ =
                detail::floatFloatSubtract(right, detail::floatFloatSubtract(result.high_, left));
        }
        return result;
    }

    // TwoProdFMA: the residual is exact if it is representable and the
    // rounded product is finite (Joldes et al., Algorithm 3).
    COSMICTIGER_FLOAT_FLOAT_HD static FloatFloat twoProduct(float left, float right) {
        FloatFloat result;
        result.high_ = detail::floatFloatMultiply(left, right);
        if (detail::floatFloatFinite(result.high_)) {
            result.low_ =
                detail::floatFloatFma(left, right, detail::floatFloatNegate(result.high_));
        }
        return result;
    }

    // DWTimesFP1 (Algorithm 7), used by DWDivDW2 below.
    COSMICTIGER_FLOAT_FLOAT_HD FloatFloat multiplyFloat(float right) const {
        FloatFloat const product = twoProduct(high_, right);
        float const cross = detail::floatFloatMultiply(low_, right);
        FloatFloat const sum = fastTwoSum(product.high_, cross);
        float const correction = detail::floatFloatAdd(sum.low_, product.low_);
        return fastTwoSum(sum.high_, correction);
    }

public:
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559 &&
        std::numeric_limits<float>::digits == 24);

    COSMICTIGER_FLOAT_FLOAT_HD constexpr FloatFloat() = default;

    COSMICTIGER_FLOAT_FLOAT_HD constexpr FloatFloat(float value)
      : high_(value) {}

    COSMICTIGER_FLOAT_FLOAT_HD FloatFloat(double value)
      : high_(detail::floatFloatNarrow(value)) {
        if (isFinite()) {
            double const high = detail::floatFloatWiden(high_);
            double volatile residual = value - high;
            float const low = detail::floatFloatNarrow(residual);
            if ((detail::floatFloatBits(low) & 0x7fffffffU) != 0) {
                *this = twoSum(high_, low);
            }
        }
    }

    // As with conversion through double, integer inputs wider than 53 bits
    // may round before splitting. FloatFloat arithmetic is itself approximate.
    template <class Integer, std::enable_if_t<std::is_integral_v<Integer>, int> = 0>
    COSMICTIGER_FLOAT_FLOAT_HD FloatFloat(Integer value)
      : FloatFloat(static_cast<double>(value)) {}

    // Accept arbitrary components; callers cannot mutate the normalized pair.
    COSMICTIGER_FLOAT_FLOAT_HD static FloatFloat fromComponents(float high, float low) {
        return twoSum(high, low);
    }

    COSMICTIGER_FLOAT_FLOAT_HD float high() const {
        return high_;
    }

    COSMICTIGER_FLOAT_FLOAT_HD float low() const {
        return low_;
    }

    COSMICTIGER_FLOAT_FLOAT_HD bool isFinite() const {
        return detail::floatFloatFinite(high_);
    }

    COSMICTIGER_FLOAT_FLOAT_HD bool isNan() const {
        return (detail::floatFloatBits(high_) & 0x7fffffffU) > 0x7f800000U;
    }

    COSMICTIGER_FLOAT_FLOAT_HD float toFloat() const {
        return high_;
    }

    COSMICTIGER_FLOAT_FLOAT_HD double toDouble() const {
        if ((detail::floatFloatBits(low_) & 0x7fffffffU) == 0) {
            return detail::floatFloatWiden(high_);
        }
        double volatile result = detail::floatFloatWiden(high_) + detail::floatFloatWiden(low_);
        return result;
    }

    COSMICTIGER_FLOAT_FLOAT_HD explicit operator float() const {
        return toFloat();
    }

    COSMICTIGER_FLOAT_FLOAT_HD explicit operator double() const {
        return toDouble();
    }

    COSMICTIGER_FLOAT_FLOAT_HD FloatFloat operator+() const {
        return *this;
    }

    COSMICTIGER_FLOAT_FLOAT_HD FloatFloat operator-() const {
        FloatFloat result;
        result.high_ = detail::floatFloatNegate(high_);
        result.low_ = detail::floatFloatNegate(low_);
        return result;
    }

    // Accurate DWPlusDW (Algorithm 6); unlike "sloppy" addition this retains
    // the low-word sum's error when the high words cancel.
    COSMICTIGER_FLOAT_FLOAT_HD friend FloatFloat operator+(FloatFloat left, FloatFloat right) {
        FloatFloat const highSum = twoSum(left.high_, right.high_);
        if (!highSum.isFinite()) {
            return highSum;
        }
        FloatFloat const lowSum = twoSum(left.low_, right.low_);
        float const correction = detail::floatFloatAdd(highSum.low_, lowSum.high_);
        FloatFloat const sum = fastTwoSum(highSum.high_, correction);
        float const remainder = detail::floatFloatAdd(lowSum.low_, sum.low_);
        return fastTwoSum(sum.high_, remainder);
    }

    COSMICTIGER_FLOAT_FLOAT_HD friend FloatFloat operator-(FloatFloat left, FloatFloat right) {
        return left + (-right);
    }

    // DWTimesDW3 (Algorithm 12), including the low * low contribution.
    COSMICTIGER_FLOAT_FLOAT_HD friend FloatFloat operator*(FloatFloat left, FloatFloat right) {
        FloatFloat const product = twoProduct(left.high_, right.high_);
        if (!product.isFinite()) {
            return product;
        }
        float const lowProduct = detail::floatFloatMultiply(left.low_, right.low_);
        float const firstCross = detail::floatFloatFma(left.high_, right.low_, lowProduct);
        float const secondCross = detail::floatFloatFma(left.low_, right.high_, firstCross);
        float const correction = detail::floatFloatAdd(product.low_, secondCross);
        return fastTwoSum(product.high_, correction);
    }

    // Convenience composition of the existing operators, with their separate
    // rounding steps. General-path cost: 12 + 20 = 32 FLOPs.
    COSMICTIGER_FLOAT_FLOAT_HD friend FloatFloat fma(
        FloatFloat left, FloatFloat right, FloatFloat addend) {
        return left * right + addend;
    }

    // DWDivDW2 (Algorithm 17), with the Algorithm 7 product above.
    COSMICTIGER_FLOAT_FLOAT_HD friend FloatFloat operator/(FloatFloat left, FloatFloat right) {
        float const quotient = detail::floatFloatDivide(left.high_, right.high_);
        if (!detail::floatFloatFinite(quotient) ||
            (detail::floatFloatBits(quotient) & 0x7fffffffU) == 0) {
            return FloatFloat(quotient);
        }
        FloatFloat const product = right.multiplyFloat(quotient);
        float const highResidual = detail::floatFloatSubtract(left.high_, product.high_);
        float const lowResidual = detail::floatFloatSubtract(left.low_, product.low_);
        float const residual = detail::floatFloatAdd(highResidual, lowResidual);
        float const correction = detail::floatFloatDivide(residual, right.high_);
        return fastTwoSum(quotient, correction);
    }

    // One Newton correction to a float square root. TwoProdFMA retains the
    // square's rounding error before computing x - root * root, then we form
    // root + (x - root * root) / (2 * root). For related Newton refinement
    // (using reciprocal roots), see Hida, Li and Bailey, "Quad-Double
    // Arithmetic: Algorithms, Implementation, and Application", Sec. 4.2:
    // https://www.davidhbailey.com/dhbpapers/quad-double.pdf .
    // As with multiplication, tiny squared-root residuals can underflow;
    // full double-word precision is not promised near float's underflow limit.
    COSMICTIGER_FLOAT_FLOAT_HD friend FloatFloat sqrt(FloatFloat value) {
        float const root = detail::floatFloatSqrt(value.high_);
        if (!detail::floatFloatFinite(root) || (detail::floatFloatBits(root) & 0x7fffffffU) == 0) {
            return FloatFloat(root);
        }
        FloatFloat const residual = value - twoProduct(root, root);
        float const correction =
            detail::floatFloatDivide(residual.high_, detail::floatFloatAdd(root, root));
        return fastTwoSum(root, correction);
    }

    // Future special functions can use argument reduction followed by a
    // Taylor expansion evaluated with FloatFloat operations and coefficients
    // split into high/low words. For f(high + low), expand in low using
    // derivatives at high, retaining enough terms for O(u^2) accuracy. The
    // base value f(high) must also be computed to FloatFloat accuracy; calling
    // a float function there and merely adding low*f'(high) loses that accuracy.
    // Check convergence, truncation error and range separately for each function.
    // See also Hida, Li and Bailey, cited above, Sec. 5 (Taylor series and
    // argument reduction for transcendental functions).

    COSMICTIGER_FLOAT_FLOAT_HD FloatFloat& operator+=(FloatFloat right) {
        return *this = *this + right;
    }

    COSMICTIGER_FLOAT_FLOAT_HD FloatFloat& operator-=(FloatFloat right) {
        return *this = *this - right;
    }

    COSMICTIGER_FLOAT_FLOAT_HD FloatFloat& operator*=(FloatFloat right) {
        return *this = *this * right;
    }

    COSMICTIGER_FLOAT_FLOAT_HD FloatFloat& operator/=(FloatFloat right) {
        return *this = *this / right;
    }

    COSMICTIGER_FLOAT_FLOAT_HD friend bool operator==(FloatFloat left, FloatFloat right) {
        return !left.isNan() && !right.isNan() &&
            detail::floatFloatOrder(left.high_) == detail::floatFloatOrder(right.high_) &&
            detail::floatFloatOrder(left.low_) == detail::floatFloatOrder(right.low_);
    }

    COSMICTIGER_FLOAT_FLOAT_HD friend bool operator!=(FloatFloat left, FloatFloat right) {
        return !(left == right);
    }

    COSMICTIGER_FLOAT_FLOAT_HD friend bool operator<(FloatFloat left, FloatFloat right) {
        if (left.isNan() || right.isNan()) {
            return false;
        }
        uint32_t const leftHigh = detail::floatFloatOrder(left.high_);
        uint32_t const rightHigh = detail::floatFloatOrder(right.high_);
        return leftHigh < rightHigh ||
            (leftHigh == rightHigh &&
                detail::floatFloatOrder(left.low_) < detail::floatFloatOrder(right.low_));
    }

    COSMICTIGER_FLOAT_FLOAT_HD friend bool operator>(FloatFloat left, FloatFloat right) {
        return right < left;
    }

    COSMICTIGER_FLOAT_FLOAT_HD friend bool operator<=(FloatFloat left, FloatFloat right) {
        return left < right || left == right;
    }

    COSMICTIGER_FLOAT_FLOAT_HD friend bool operator>=(FloatFloat left, FloatFloat right) {
        return right <= left;
    }
};

static_assert(sizeof(FloatFloat) == 2 * sizeof(float));
static_assert(std::is_trivially_copyable_v<FloatFloat>);

#if defined(__clang__)
#pragma float_control(pop)
#elif defined(__GNUC__) && !defined(__CUDACC__)
#pragma GCC pop_options
#endif

#undef COSMICTIGER_FLOAT_FLOAT_HD
