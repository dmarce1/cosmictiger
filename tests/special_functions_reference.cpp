#include <cmath>
#include <stdexcept>

#include "special_functions_reference.hpp"

namespace special_functions_test {

long double reference(UnaryFunction function, float input) {
    long double const value = static_cast<long double>(input);
    switch (function) {
    case UnaryFunction::Sin:
        return std::sin(value);
    case UnaryFunction::Cos:
        return std::cos(value);
    case UnaryFunction::Exp:
        return std::exp(value);
    case UnaryFunction::Exp2:
        return std::exp2(value);
    case UnaryFunction::Log:
        return std::log(value);
    case UnaryFunction::Log2:
        return std::log2(value);
    case UnaryFunction::Erf:
        return std::erf(value);
    case UnaryFunction::Erfc:
        return std::erfc(value);
    case UnaryFunction::Sqrt:
        return std::sqrt(value);
    case UnaryFunction::Rsqrt:
        return 1.0L / std::sqrt(value);
    }
    throw std::runtime_error("unknown unary function");
}

long double exponentialSquareReference(float input) {
    long double const value = static_cast<long double>(input);
    return std::exp(-(value * value));
}

long double powerReference(float base, float exponent) {
    return std::pow(static_cast<long double>(base), static_cast<long double>(exponent));
}

long double fmaReference(float first, float second, float third) {
    // std::fma(float,float,float) specifies one rounding of the exact product
    // and sum, including the nearly canceled cases in this test.
    return std::fma(first, second, third);
}

}    // namespace special_functions_test
