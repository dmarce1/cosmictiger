/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>

#include <boost/multiprecision/cpp_bin_float.hpp>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

#include <cosmictiger/float_float.hpp>

namespace detail {

using Reference = boost::multiprecision::cpp_bin_float_quad;

class FloatingEnvironment {
public:
    FloatingEnvironment()
      : roundingMode_(std::fegetround()) {
        std::fesetround(FE_TONEAREST);
#if defined(__SSE__)
        control_ = _mm_getcsr();
        // The fast-math executable starts with FTZ/DAZ enabled on some toolchains.
        // The arithmetic contract requires gradual underflow on the host.
        _mm_setcsr(control_ & ~unsigned(0x8040));
#endif
    }

    ~FloatingEnvironment() {
#if defined(__SSE__)
        _mm_setcsr(control_);
#endif
        std::fesetround(roundingMode_);
    }

private:
    int roundingMode_;
#if defined(__SSE__)
    unsigned control_;
#endif
};

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Reference reference(FloatFloat value) {
    return Reference(value.high()) + Reference(value.low());
}

void close(FloatFloat actual, Reference const& expected, char const* label,
    Reference const& absoluteTolerance = 0) {
    Reference const error = abs(reference(actual) - expected);
    Reference const tolerance = abs(expected) * ldexp(Reference(12), -48) + absoluteTolerance;
    if (!actual.isFinite() || error > tolerance) {
        std::fprintf(stderr, "%s: (%.9g, %.9g), expected %.21Lg; error %.6Lg, bound %.6Lg\n", label,
            actual.high(), actual.low(), expected.convert_to<long double>(),
            error.convert_to<long double>(), tolerance.convert_to<long double>());
        throw std::runtime_error(label);
    }
    if (actual.high() != 0.0f) {
        int exponent;
        std::frexp(double(actual.high()), &exponent);
        Reference const halfUlp = ldexp(Reference(1), exponent - 25);
        require(abs(Reference(actual.low())) <= halfUlp, "Result components must not overlap");
    } else {
        require(actual.low() == 0.0f, "A zero high component must have a zero low component");
    }
}

void testConstructionAndComparison() {
    static_assert(sizeof(FloatFloat) == 2 * sizeof(float));
    static_assert(std::is_trivially_copyable_v<FloatFloat>);
    FloatFloat const zero;
    require(zero.high() == 0.0f && zero.low() == 0.0f, "Default construction must produce zero");
    close(FloatFloat(17), Reference(17), "Integer construction");
    for (float const input : {0.0f, -0.0f, 1.0f, -3.25f, 0x1.fffffep100f}) {
        FloatFloat const value(input);
        require(value.high() == input && value.low() == 0.0f, "Float construction must be exact");
        require(value.toFloat() == input && float(value) == input, "Float conversion");
        require(value.toDouble() == double(input) && double(value) == double(input),
            "Double conversion of a single float");
    }
    for (double const input : {1.0 + 0x1p-40, -1.0 - 0x1p-40, 3.141592653589793,
             -0x1.123456789abcdp70, 0x1.abcdef1234567p-70}) {
        FloatFloat const value(input);
        close(value, Reference(input), "Double construction");
        require(value.toDouble() == double(value.high()) + double(value.low()),
            "Double conversion must retain the low component");
        require(double(value) == value.toDouble() && float(value) == value.toFloat(),
            "Named and explicit conversions must agree");
    }
    FloatFloat const above = FloatFloat::fromComponents(1.0f, 0x1p-40f);
    FloatFloat const below = FloatFloat::fromComponents(1.0f, -0x1p-40f);
    FloatFloat const one(1.0f);
    require(above > one && above >= one && !(above < one) && !(above <= one),
        "Ordering must distinguish a positive low component");
    require(below < one && below <= one && !(below > one) && !(below >= one),
        "Ordering must distinguish a negative low component");
    require(above != one && !(above == one), "Equality must include the low component");
    require(
        one == FloatFloat::fromComponents(0.5f, 0.5f), "Equality must compare normalized values");
    require(one <= one && one >= one && !(one != one), "Reflexive comparisons");
    require(FloatFloat(-0.0f) == zero, "Signed zeros must compare equal");
#ifndef __FAST_MATH__
    // A fast-math caller may discard zero signs before entering the class.
    FloatFloat const negativeZero(std::bit_cast<float>(uint32_t(0x80000000)));
    require(std::bit_cast<uint32_t>(negativeZero.toFloat()) == 0x80000000U,
        "Float conversion must preserve negative zero");
    require(std::bit_cast<uint64_t>(negativeZero.toDouble()) == 0x8000000000000000ULL,
        "Double conversion must preserve negative zero");
    require(std::bit_cast<uint32_t>(
                FloatFloat(std::bit_cast<double>(uint64_t(0x8000000000000000ULL))).high()) ==
            0x80000000U,
        "Construction from double must preserve negative zero");
#endif
    close(+above, reference(above), "Unary plus");
    close(-above, -reference(above), "Unary minus");
    close(FloatFloat::fromComponents(0x1p-30f, 1.0f), Reference(1) + ldexp(Reference(1), -30),
        "Normalization must allow a larger second component");
    close(
        FloatFloat::fromComponents(1.0f, -1.0f), Reference(0), "Exact normalization cancellation");
    close(FloatFloat::fromComponents(1.0f, -0x1.fffffep-1f), ldexp(Reference(1), -24),
        "Near cancellation normalization");
}

void testArithmetic() {
    FloatFloat const one(1.0f);
    FloatFloat const increment(0x1p-30f);
    FloatFloat accumulated = one;
    for (int index = 0; index < 10000; ++index) {
        accumulated += increment;
    }
    close(accumulated, Reference(1) + Reference(10000) * ldexp(Reference(1), -30),
        "Repeated increments below one float ulp");
    FloatFloat const leftProduct = FloatFloat::fromComponents(0x1.5970bep0f, -0x1.1e636p-25f);
    FloatFloat const rightProduct = FloatFloat::fromComponents(0x1.8e7a42p0f, -0x1.c8d56p-25f);
    Reference const exactProduct = reference(leftProduct) * reference(rightProduct);
    close(leftProduct * rightProduct, exactProduct, "Low-times-low product contribution");
    // For these inputs, omitting low*low moves the error past this absolute bound.
    require(abs(reference(leftProduct * rightProduct) - exactProduct) < ldexp(Reference(1), -47),
        "Multiplication must include the low-times-low contribution");
    close(one / FloatFloat(3), Reference(1) / 3, "Repeating binary quotient");
    close((one + increment) - one, ldexp(Reference(1), -30), "Cancellation must expose low bits");
    FloatFloat compound = FloatFloat::fromComponents(1.25f, 0x1p-27f);
    Reference expected = reference(compound);
    compound += leftProduct;
    expected += reference(leftProduct);
    close(compound, expected, "Compound addition");
    compound -= rightProduct;
    expected -= reference(rightProduct);
    close(compound, expected, "Compound subtraction");
    compound *= leftProduct;
    expected *= reference(leftProduct);
    close(compound, expected, "Compound multiplication");
    compound /= rightProduct;
    expected /= reference(rightProduct);
    close(compound, expected, "Compound division");
    close(2.0f + one, Reference(3), "Implicit mixed float arithmetic");
    close(one * 2, Reference(2), "Implicit mixed integer arithmetic");
    close(one + (1.0 + 0x1p-40), Reference(2) + ldexp(Reference(1), -40),
        "Implicit mixed double arithmetic");
}

FloatFloat randomValue(std::mt19937& generator) {
    std::uniform_int_distribution<int> exponentDistribution(-40, 40);
    std::uniform_int_distribution<int> mantissaDistribution(0, 0x7fffff);
    std::uniform_int_distribution<int> lowDistribution(-0x400000, 0x400000);
    int const exponent = exponentDistribution(generator);
    float high = std::ldexp(1.0f + float(mantissaDistribution(generator)) * 0x1p-23f, exponent);
    if (generator() & 1) {
        high = -high;
    }
    float const low = std::ldexp(float(lowDistribution(generator)), exponent - 48);
    return FloatFloat::fromComponents(high, low);
}

void testRandomArithmetic() {
    std::mt19937 generator(0x7b48d125);
    for (int index = 0; index < 4096; ++index) {
        FloatFloat const left = randomValue(generator);
        FloatFloat const right = randomValue(generator);
        Reference const leftReference = reference(left);
        Reference const rightReference = reference(right);
        close(left + right, leftReference + rightReference, "Random addition");
        close(left - right, leftReference - rightReference, "Random subtraction");
        close(left * right, leftReference * rightReference, "Random multiplication");
        close(left / right, leftReference / rightReference, "Random division");
        FloatFloat const nearOpposite = FloatFloat::fromComponents(-left.high(), left.low());
        close(left + nearOpposite, leftReference + reference(nearOpposite),
            "Random cancellation of high components");
        require((left < right) == (leftReference < rightReference), "Random comparison");
    }
}

void testMultiplyAdd() {
    FloatFloat const left = FloatFloat::fromComponents(1.0f, 0x1p-40f);
    FloatFloat const right = FloatFloat::fromComponents(1.0f, -0x1p-40f);
    FloatFloat const zero(0.0f);
    // The product rounds to one before the addend is applied.
    require(fma(left, right, FloatFloat(-1.0f)) == zero,
        "Multiply-add must preserve separate multiplication and addition rounding");
    close(fma(left, zero, right), reference(right), "Multiply-add with a zero multiplier");
    close(fma(left, right, zero), reference(left) * reference(right),
        "Multiply-add with a zero addend");
    std::mt19937 generator(0x893af271);
    for (int index = 0; index < 128; ++index) {
        FloatFloat const first = randomValue(generator);
        FloatFloat const second = randomValue(generator);
        FloatFloat const addend = randomValue(generator);
        Reference const product = reference(first) * reference(second);
        Reference const added = reference(addend);
        Reference const inputTolerance = (abs(product) + abs(added)) * ldexp(Reference(12), -48);
        close(fma(first, second, addend), product + added, "Random multiply-add", inputTolerance);
    }
}

void testSquareRoot() {
    for (FloatFloat const input : {FloatFloat(2), FloatFloat(4), FloatFloat(0x1p100f),
             FloatFloat(0x1p-100f), FloatFloat(std::numeric_limits<float>::min()),
             FloatFloat(std::numeric_limits<float>::max()),
             FloatFloat::fromComponents(1.0f, 0x1p-40f),
             FloatFloat::fromComponents(1.0f, -0x1p-40f)}) {
        close(sqrt(input), sqrt(reference(input)), "Square root including limits and near one");
    }
    std::mt19937 generator(0x264e913f);
    for (int index = 0; index < 2048; ++index) {
        FloatFloat input = randomValue(generator);
        if (input < FloatFloat(0)) {
            input = -input;
        }
        close(sqrt(input), sqrt(reference(input)), "Random square root");
    }
    close(sqrt(FloatFloat(0.0f)), Reference(0), "Square root of zero");
#ifndef __FAST_MATH__
    // Nonfinite/sign tests need a caller without finite-only/no-signed-zero
    // assumptions; the header cannot undo optimizations in its callers.
    FloatFloat const infinity(std::numeric_limits<float>::infinity());
    FloatFloat const nan(std::numeric_limits<float>::quiet_NaN());
    require(sqrt(infinity) == infinity, "Square root of infinity");
    require(sqrt(-infinity).isNan() && sqrt(FloatFloat(-1)).isNan() && sqrt(nan).isNan(),
        "Square root must handle negative and NaN inputs");
    require(std::bit_cast<uint32_t>(sqrt(FloatFloat(0.0f)).high()) == 0,
        "Square root must preserve positive zero");
    require(std::bit_cast<uint32_t>(
                sqrt(FloatFloat(std::bit_cast<float>(uint32_t(0x80000000)))).high()) == 0x80000000U,
        "Square root must preserve negative zero");
#endif
}

void testSpecialValues() {
    FloatFloat const infinity(std::numeric_limits<float>::infinity());
    FloatFloat const nan(std::numeric_limits<float>::quiet_NaN());
    FloatFloat const one(1.0f);
    FloatFloat const zero(0.0f);
    require(!infinity.isFinite() && !infinity.isNan(), "Infinity classification");
    require(!nan.isFinite() && nan.isNan(), "NaN classification");
    require(one.isFinite() && !one.isNan(), "Finite classification");
    require(infinity > one && -infinity < one, "Infinity ordering");
    require(!(nan == one) && nan != one && !(nan < one) && !(nan > one) && !(nan <= one) &&
            !(nan >= one) && !(one < nan) && !(one > nan),
        "NaN comparisons must be unordered");
    require((infinity - infinity).isNan(), "Infinity subtraction must produce NaN");
    require((infinity * zero).isNan(), "Infinity times zero must produce NaN");
    require((zero / zero).isNan(), "Zero divided by zero must produce NaN");
    require((nan + one).isNan() && (one * nan).isNan(), "NaNs must propagate");
    require(!(one / zero).isFinite() && !(one / zero).isNan(), "Division by zero must be infinite");
    close(one / infinity, Reference(0), "Finite divided by infinity");
    require(
        infinity + one == infinity && infinity * one == infinity, "Finite operations on infinity");
}

void testUnderflow() {
    FloatFloat const minimum(std::bit_cast<float>(uint32_t(0x00800000)));
    FloatFloat const denormal(std::bit_cast<float>(uint32_t(1)));
    close(denormal, ldexp(Reference(1), -149), "Subnormal construction");
    close(minimum + denormal, ldexp(Reference(1), -126) + ldexp(Reference(1), -149),
        "Subnormal addition");
    close(minimum * FloatFloat(0.5f), ldexp(Reference(1), -127), "Subnormal product");
    close(minimum / FloatFloat(2.0f), ldexp(Reference(1), -127), "Subnormal quotient");
    close(-denormal, -ldexp(Reference(1), -149), "Subnormal negation");
    close(minimum - FloatFloat::fromComponents(minimum.high(), -denormal.high()),
        ldexp(Reference(1), -149), "Subnormal cancellation");
    close(FloatFloat(0x1.23456789abcdp-126), Reference(0x1.23456789abcdp-126),
        "Conversion loses precision at the bottom of the float exponent range",
        ldexp(Reference(1), -149));
}

}    // namespace detail

int main() {
    try {
        detail::require(std::fegetround() == FE_TONEAREST, "Tests require round to nearest");
        detail::testConstructionAndComparison();
        detail::testArithmetic();
        detail::testRandomArithmetic();
        detail::testMultiplyAdd();
        detail::testSquareRoot();
#ifndef __FAST_MATH__
        detail::testSpecialValues();
#endif
        std::puts("FloatFloat: normal-range arithmetic and sqrt passed in startup FP environment");
        {
            detail::FloatingEnvironment const environment;
            detail::testUnderflow();
        }
        std::puts("FloatFloat: subnormal cases passed with gradual underflow enabled");
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "FloatFloat test failed: %s\n", error.what());
        return 1;
    }
}
