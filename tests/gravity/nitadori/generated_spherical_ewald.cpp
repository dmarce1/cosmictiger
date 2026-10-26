/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include <cosmictiger/gravity/nitadori/generated_spherical_ewald.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace counted {
inline size_t operations = 0;
class Real {
    double value_ = 0;

public:
    using value_type = double;
    Real() = default;
    Real(double value)
      : value_(value) {}
    template <typename Generator>
        requires requires(Generator g) { g(0); }
    Real(Generator generator)
      : value_(generator(0)) {}
    double value() const {
        return value_;
    }
    double operator[](int) const {
        return value_;
    }
    friend bool operator<(Real a, Real b) {
        return a.value_ < b.value_;
    }
    friend bool operator>(Real a, Real b) {
        return a.value_ > b.value_;
    }
    friend bool operator==(Real a, Real b) {
        return a.value_ == b.value_;
    }
    Real operator-() const {
        return Real(-value_);
    }
    friend Real operator+(Real a, Real b) {
        ++operations;
        return Real(a.value_ + b.value_);
    }
    friend Real operator-(Real a, Real b) {
        ++operations;
        return Real(a.value_ - b.value_);
    }
    friend Real operator*(Real a, Real b) {
        ++operations;
        return Real(a.value_ * b.value_);
    }
    friend Real operator/(Real a, Real b) {
        ++operations;
        return Real(a.value_ / b.value_);
    }
    Real& operator+=(Real other) {
        *this = *this + other;
        return *this;
    }
    Real& operator*=(Real other) {
        *this = *this * other;
        return *this;
    }
};
}    // namespace counted
template <>
inline counted::Real simdSqrt(counted::Real const& value) {
    counted::operations += 7;
    return counted::Real(std::sqrt(value.value()));
}
template <>
inline counted::Real simdFma(
    counted::Real const& a, counted::Real const& b, counted::Real const& c) {
    ++counted::operations;
    return counted::Real(std::fma(a.value(), b.value(), c.value()));
}

namespace {
#ifdef FMM_MIN_ORDER
constexpr int kMinCompiledDegree = FMM_MIN_ORDER - 1;
#else
constexpr int kMinCompiledDegree = 2;
#endif
#ifdef FMM_MAX_ORDER
constexpr int kMaxCompiledDegree = FMM_MAX_ORDER - 1;
#else
constexpr int kMaxCompiledDegree = 8;
#endif
static_assert(
    2 <= kMinCompiledDegree && kMinCompiledDegree <= kMaxCompiledDegree && kMaxCompiledDegree <= 8);
#ifdef GENERATED_SPHERICAL_EWALD_TEST_DEGREE
constexpr int kMinTestDegree = GENERATED_SPHERICAL_EWALD_TEST_DEGREE;
constexpr int kMaxTestDegree = GENERATED_SPHERICAL_EWALD_TEST_DEGREE;
static_assert(kMinCompiledDegree <= kMinTestDegree && kMaxTestDegree <= kMaxCompiledDegree,
    "The requested spherical Ewald test degree must be inside the compiled FMM order range");
#else
constexpr int kMinTestDegree = kMinCompiledDegree;
constexpr int kMaxTestDegree = kMaxCompiledDegree;
#endif

using Complex = std::complex<double>;
using Point = std::array<double, 3>;
constexpr double kPi = 3.1415926535897932384626433832795;
size_t checks = 0;

int index(int n, int m) {
    return n * (n + 1) + m;
}
double factorial(int n) {
    double result = 1;
    for (int k = 2; k <= n; ++k) {
        result *= k;
    }
    return result;
}

void near(double actual, double expected, double tolerance, char const* label) {
    ++checks;
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::fprintf(stderr, "%s: got %.17g, expected %.17g, tolerance %.4g\n", label, actual,
            expected, tolerance);
        throw std::runtime_error(label);
    }
}

// Complex loop recurrence, without the emitter's real component arithmetic.
std::vector<Complex> regular(int degree, Point p) {
    std::vector<Complex> result((degree + 1) * (degree + 1));
    double const radiusSquared = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    result[0] = 1;
    for (int m = 0; m <= degree; ++m) {
        if (m) {
            result[index(m, m)] = result[index(m - 1, m - 1)] * Complex(p[0], p[1]) / double(2 * m);
        }
        if (m < degree) {
            result[index(m + 1, m)] = p[2] * result[index(m, m)];
        }
        for (int n = m + 2; n <= degree; ++n) {
            result[index(n, m)] = ((2 * n - 1) * p[2] * result[index(n - 1, m)] -
                                      radiusSquared * result[index(n - 2, m)]) /
                double(n * n - m * m);
        }
    }
    return result;
}

Complex get(std::vector<Complex> const& values, int n, int m) {
    if (std::abs(m) > n) {
        return 0;
    }
    auto const value = values[index(n, std::abs(m))];
    return m >= 0 ? value : (m % 2 ? -1. : 1.) * std::conj(value);
}

std::array<std::array<double, 2>, 32> quadrature() {
    std::array<std::array<double, 2>, 32> result{};
    for (int node = 0; node < 32; ++node) {
        double x = std::cos(kPi * (node + .75) / 32.5);
        double derivative = 0;
        for (int iteration = 0; iteration < 20; ++iteration) {
            double previous = 1;
            double value = x;
            for (int n = 2; n <= 32; ++n) {
                double const next = ((2 * n - 1) * x * value - (n - 1) * previous) / n;
                previous = value;
                value = next;
            }
            derivative = 32 * (x * value - previous) / (x * x - 1);
            double const change = value / derivative;
            x -= change;
            if (std::abs(change) < 2e-16) {
                break;
            }
        }
        result[node] = {(x + 1) / 2, 1 / ((1 - x * x) * derivative * derivative)};
    }
    return result;
}

// Independent central Gaussian quadrature; explicit finite upper-gamma sums;
// larger image sphere and unpaired reciprocal lattice. No generated routines.
std::vector<Complex> reference(int degree, Point p) {
    auto const polynomial = regular(degree, p);
    std::vector<Complex> result(polynomial.size());
    std::vector<Complex> compensation(polynomial.size());
    auto const add = [&](int slot, Complex value) {
        auto const delta = value - compensation[slot];
        auto const next = result[slot] + delta;
        compensation[slot] = (next - result[slot]) - delta;
        result[slot] = next;
    };
    auto const nodes = quadrature();
    double const radiusSquared = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    for (int n = 0; n <= degree; ++n) {
        double integral = 0;
        for (auto const& node : nodes) {
            integral += node[1] * std::pow(node[0], 2 * n) *
                std::exp(-4 * radiusSquared * node[0] * node[0]);
        }
        double const radial =
            (n % 2 ? -1. : 1.) * 2 * std::pow(2., 2 * n + 1) * integral / std::tgamma(n + .5);
        for (int m = 0; m <= n; ++m) {
            add(index(n, m),
                radial * factorial(n - m) * factorial(n + m) * polynomial[index(n, m)]);
        }
    }
    for (int i = -7; i <= 7; ++i) {
        for (int j = -7; j <= 7; ++j) {
            for (int k = -7; k <= 7; ++k) {
                int const norm = i * i + j * j + k * k;
                if (!norm || norm > 49) {
                    continue;
                }
                Point const image{p[0] + i, p[1] + j, p[2] + k};
                auto const u = regular(degree, image);
                double const r2 = image[0] * image[0] + image[1] * image[1] + image[2] * image[2];
                double const r = std::sqrt(r2);
                for (int n = 0; n <= degree; ++n) {
                    double screening = std::erfc(2 * r);
                    for (int q = 0; q < n; ++q) {
                        screening +=
                            std::exp(-4 * r2) * std::pow(4 * r2, q + .5) / std::tgamma(q + 1.5);
                    }
                    double const radial = (n % 2 ? 1. : -1.) * screening / std::pow(r, 2 * n + 1);
                    for (int m = 0; m <= n; ++m) {
                        add(index(n, m),
                            radial * factorial(n - m) * factorial(n + m) * u[index(n, m)]);
                    }
                }
                auto const wave = regular(degree, {double(i), double(j), double(k)});
                double const phase = 2 * kPi * (i * p[0] + j * p[1] + k * p[2]);
                double const amplitude = -std::exp(-kPi * kPi * norm / 4) / (kPi * norm);
                double oddFactorial = 1;
                for (int n = 0; n <= degree; ++n) {
                    if (n) {
                        oddFactorial *= 2 * n - 1;
                    }
                    double const oscillation = n % 4 == 0 ? std::cos(phase) :
                        n % 4 == 1                        ? -std::sin(phase) :
                        n % 4 == 2                        ? -std::cos(phase) :
                                                            std::sin(phase);
                    double const radial =
                        amplitude * std::pow(2 * kPi, n) * oscillation / oddFactorial;
                    for (int m = 0; m <= n; ++m) {
                        add(index(n, m),
                            radial * factorial(n - m) * factorial(n + m) * wave[index(n, m)]);
                    }
                }
            }
        }
    }
    add(0, kPi / 4);
    return result;
}

template <int degree, typename Real>
void scalarCases() {
    using namespace cosmictiger::gravity::spherical;
    constexpr int kCount = (degree + 1) * (degree + 1);
    double const epsilon = std::is_same_v<Real, float> ? 3e-5 : 8e-12;
    std::array<Point, 6> const points{{{0, 0, 0}, {1e-7, -2e-7, 3e-7}, {.213, -.172, .319},
        {0, 0, .25}, {.499, -.499, .499}, {-.25, .125, 0}}};
    for (auto const& p : points) {
        std::array<Real, kCount> greens{};
        Real trace{};
        EwaldGreensKernel<degree>::template evaluate<Real>(
            greens, trace, Real(p[0]), Real(p[1]), Real(p[2]));
        auto const exact =
            reference(degree, {double(Real(p[0])), double(Real(p[1])), double(Real(p[2]))});
        near(trace, 4 * kPi / 3, epsilon, "Green trace");
        for (int n = 0; n <= degree; ++n) {
            double rankScale = 1;
            for (int m = 0; m <= n; ++m) {
                rankScale = std::max(rankScale, std::abs(exact[index(n, m)]));
            }
            for (int m = -n; m <= n; ++m) {
                double const expected =
                    m >= 0 ? exact[index(n, m)].real() : exact[index(n, -m)].imag();
                near(greens[index(n, m)], expected,
                    (std::abs(p[0]) + std::abs(p[1]) + std::abs(p[2]) < 1e-5 ?
                            std::max(epsilon, 2e-9) :
                            epsilon) *
                        rankScale,
                    "Green harmonic");
            }
        }
        Point const source{.013, -.009, .007};
        auto const sourceHarmonics = regular(degree - 1, {-source[0], -source[1], -source[2]});
        std::array<Real, degree * degree> moments{};
        for (int n = 0; n < degree; ++n) {
            for (int m = -n; m <= n; ++m) {
                moments[index(n, m)] = Real(m >= 0 ? sourceHarmonics[index(n, m)].real() :
                                                     sourceHarmonics[index(n, -m)].imag());
            }
        }
        double const sourceTrace =
            source[0] * source[0] + source[1] * source[1] + source[2] * source[2];
        std::array<Real, kCount> local{};
        local.fill(Real(.125));
        Real localTrace = Real(.25);
        EwaldM2lKernel<degree>::template add<Real>(
            local, localTrace, moments, Real(sourceTrace), greens, trace);
        for (int n = 0; n <= degree; ++n) {
            for (int m = 0; m <= n; ++m) {
                Complex expected{};
                for (int k = 0; k <= std::min(degree - n, degree - 1); ++k) {
                    for (int l = -k; l <= k; ++l) {
                        expected +=
                            std::conj(get(sourceHarmonics, k, l)) * get(exact, n + k, m + l);
                    }
                }
                if (n == 0 && degree >= 3) {
                    expected -= .5 * (4 * kPi / 3) * sourceTrace;
                }
                if (n == 1 && m == 0) {
                    expected += (4 * kPi / 3) * source[2];
                }
                if (n == 1 && m == 1) {
                    expected += (4 * kPi / 3) * Complex(source[0], source[1]);
                }
                double const tolerance =
                    std::max(epsilon * 10, 3e-9) * std::max(1., std::abs(expected));
                near(local[index(n, m)], .125 + expected.real(), tolerance, "M2L real");
                if (m) {
                    near(local[index(n, -m)], .125 + expected.imag(), tolerance, "M2L imaginary");
                }
            }
        }
        near(localTrace, .25 + 4 * kPi / 3, epsilon, "M2L trace accumulation");
    }
    // Pure background tests isolate its signs and accumulation from harmonics.
    std::array<Real, kCount> zero{}, shifted{};
    Real shiftedTrace = Real(.25);
    EwaldL2lKernel<degree>::template add<Real>(
        shifted, shiftedTrace, zero, Real(2), Real(.1), Real(-.2), Real(.3));
    near(shifted[0], -.14, epsilon, "background translated potential");
    near(shifted[3], -.2, epsilon, "background translated x");
    near(shifted[1], .4, epsilon, "background translated y");
    near(shifted[2], -.6, epsilon, "background translated z");
    near(shiftedTrace, 2.25, epsilon, "background translated trace");
    std::array<Real, 4> field{};
    EwaldL2pKernel<degree>::template add<Real>(field, zero, Real(2), Real(.1), Real(-.2), Real(.3));
    near(field[0], -.14, epsilon, "background point potential");
    near(field[1], .2, epsilon, "background point x");
    near(field[2], -.4, epsilon, "background point y");
    near(field[3], .6, epsilon, "background point z");
}

template <int degree>
void countCase() {
    using namespace cosmictiger::gravity::spherical;
    std::array<counted::Real, (degree + 1) * (degree + 1)> greens{}, local{};
    std::array<counted::Real, degree * degree> source{};
    counted::Real trace{}, localTrace{};
    counted::operations = 0;
    EwaldGreensKernel<degree>::template evaluate<counted::Real>(
        greens, trace, counted::Real(.2), counted::Real(-.1), counted::Real(.3));
    near(double(counted::operations), EwaldGreensKernel<degree>::kFlops_, 0,
        "Ewald emitted arithmetic count");
    counted::operations = 0;
    EwaldM2lKernel<degree>::template add<counted::Real>(
        local, localTrace, source, counted::Real(.01), greens, trace);
    near(double(counted::operations), EwaldM2lKernel<degree>::kFlops_, 0,
        "Ewald contraction arithmetic count");
}

template <int degree, typename Scalar>
void simdCase() {
    using namespace cosmictiger::gravity::spherical;
    using Pack = SimdVector<Scalar, 4>;
    std::array<Point, 4> const positions{
        {{0, 0, 0}, {1e-7, -2e-7, 3e-7}, {.21, -.17, .31}, {0, 0, .25}}};
    Pack const x([&](auto lane) { return Scalar(positions[lane][0]); });
    Pack const y([&](auto lane) { return Scalar(positions[lane][1]); });
    Pack const z([&](auto lane) { return Scalar(positions[lane][2]); });
    std::array<Pack, (degree + 1) * (degree + 1)> output{};
    Pack trace;
    EwaldGreensKernel<degree>::template evaluate<Pack>(output, trace, x, y, z);
    for (int lane = 0; lane < 4; ++lane) {
        std::array<Scalar, (degree + 1) * (degree + 1)> expected{};
        Scalar scalarTrace{};
        EwaldGreensKernel<degree>::template evaluate<Scalar>(
            expected, scalarTrace, Scalar(x[lane]), Scalar(y[lane]), Scalar(z[lane]));
        for (int slot = 0; slot < (degree + 1) * (degree + 1); ++slot) {
            near(output[slot][lane], expected[slot],
                3e-12 * std::max(1., std::abs(double(expected[slot]))), "SIMD mixed-lane Ewald");
        }
        near(trace[lane], scalarTrace, 0, "SIMD trace");
    }
}

template <int degree>
void run() {
    scalarCases<degree, float>();
    scalarCases<degree, double>();
    countCase<degree>();
    simdCase<degree, float>();
    simdCase<degree, double>();
    if constexpr (degree < kMaxTestDegree) {
        run<degree + 1>();
    }
}
}    // namespace

int main() {
    try {
        run<kMinTestDegree>();
        std::printf(
            "Spherical Ewald: %zu independent coefficient/trace/FLOP checks passed\n", checks);
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
