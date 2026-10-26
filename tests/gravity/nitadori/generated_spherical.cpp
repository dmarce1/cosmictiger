/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cosmictiger/gravity/fmm_kernels/spherical_kernels.hpp>

namespace detail {

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
#ifdef GENERATED_SPHERICAL_TEST_DEGREE
constexpr int kMinTestDegree = GENERATED_SPHERICAL_TEST_DEGREE;
constexpr int kMaxTestDegree = GENERATED_SPHERICAL_TEST_DEGREE;
static_assert(kMinCompiledDegree <= kMinTestDegree && kMaxTestDegree <= kMaxCompiledDegree,
    "The requested spherical test degree must be inside the compiled FMM order range");
#else
constexpr int kMinTestDegree = kMinCompiledDegree;
constexpr int kMaxTestDegree = kMaxCompiledDegree;
#endif

namespace spherical = cosmictiger::gravity::spherical;
using Complex = std::complex<double>;
using Point = std::array<double, 3>;
using Harmonics = std::vector<Complex>;
size_t checks = 0;
double worst = 0;
std::string context;

int index(int n, int m) {
    return n * (n + 1) + m;
}

Complex coefficient(Harmonics const& values, int n, int m) {
    if (n < 0 || std::abs(m) > n) {
        return 0;
    }
    Complex const value = values[index(n, std::abs(m))];
    return m < 0 ? (m % 2 ? -1.0 : 1.0) * std::conj(value) : value;
}

Harmonics regular(int degree, Point const& point) {
    Harmonics result((degree + 1) * (degree + 1));
    double const radiusSquared = point[0] * point[0] + point[1] * point[1] + point[2] * point[2];
    Complex const xy(point[0], point[1]);
    result[0] = 1;
    for (int m = 0; m <= degree; ++m) {
        if (m) {
            result[index(m, m)] = xy * result[index(m - 1, m - 1)] / double(2 * m);
        }
        if (m < degree) {
            result[index(m + 1, m)] = point[2] * result[index(m, m)];
        }
        for (int n = m + 2; n <= degree; ++n) {
            result[index(n, m)] = ((2 * n - 1) * point[2] * result[index(n - 1, m)] -
                                      radiusSquared * result[index(n - 2, m)]) /
                double(n * n - m * m);
        }
    }
    return result;
}

Harmonics singular(int degree, Point const& point) {
    Harmonics result((degree + 1) * (degree + 1));
    double const inverseSquared =
        1 / (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
    Complex const xy = -Complex(point[0], point[1]) * inverseSquared;
    double const z = -point[2] * inverseSquared;
    result[0] = -std::sqrt(inverseSquared);
    for (int m = 0; m <= degree; ++m) {
        if (m) {
            result[index(m, m)] = double(2 * m - 1) * xy * result[index(m - 1, m - 1)];
        }
        if (m < degree) {
            result[index(m + 1, m)] = double(2 * m + 1) * z * result[index(m, m)];
        }
        for (int n = m + 2; n <= degree; ++n) {
            result[index(n, m)] = double(2 * n - 1) * z * result[index(n - 1, m)] -
                double((n - 1) * (n - 1) - m * m) * inverseSquared * result[index(n - 2, m)];
        }
    }
    return result;
}

Point negate(Point value) {
    for (double& component : value) {
        component = -component;
    }
    return value;
}

Harmonics m2l(Harmonics const& source, Harmonics const& greens, int degree, int targetDegree) {
    Harmonics result((targetDegree + 1) * (targetDegree + 1));
    for (int n = 0; n <= targetDegree; ++n) {
        for (int m = 0; m <= n; ++m) {
            for (int k = 0; k <= std::min(degree - 1, degree - n); ++k) {
                for (int l = -k; l <= k; ++l) {
                    result[index(n, m)] +=
                        std::conj(coefficient(source, k, l)) * coefficient(greens, n + k, m + l);
                }
            }
        }
    }
    return result;
}

Harmonics translateMoments(Harmonics const& source, Point const& shift, int degree) {
    Harmonics const harmonics = regular(degree, negate(shift));
    Harmonics result((degree + 1) * (degree + 1));
    for (int n = 0; n <= degree; ++n) {
        for (int m = 0; m <= n; ++m) {
            for (int k = 0; k <= n; ++k) {
                for (int l = -k; l <= k; ++l) {
                    result[index(n, m)] +=
                        coefficient(source, n - k, m - l) * coefficient(harmonics, k, l);
                }
            }
        }
    }
    return result;
}

Harmonics translateLocal(
    Harmonics const& source, Point const& shift, int degree, int targetDegree) {
    Harmonics const harmonics = regular(degree, shift);
    Harmonics result((targetDegree + 1) * (targetDegree + 1));
    for (int n = 0; n <= targetDegree; ++n) {
        for (int m = 0; m <= n; ++m) {
            for (int k = 0; k <= degree - n; ++k) {
                for (int l = -k; l <= k; ++l) {
                    result[index(n, m)] +=
                        coefficient(source, n + k, m + l) * std::conj(coefficient(harmonics, k, l));
                }
            }
        }
    }
    return result;
}

template <typename Real>
constexpr int laneCount() {
    if constexpr (std::is_arithmetic_v<Real>) {
        return 1;
    } else {
        return Real::size();
    }
}

template <typename Real>
double lane(Real const& value, int i) {
    if constexpr (std::is_arithmetic_v<Real>) {
        return value;
    } else {
        return value[i];
    }
}

template <typename Real>
void setLane(Real& value, int i, double input) {
    if constexpr (std::is_arithmetic_v<Real>) {
        value = Real(input);
    } else {
        value[i] = input;
    }
}

template <typename Real>
constexpr double tolerance() {
    if constexpr (std::is_same_v<Real, float>) {
        return 4e-5;
    } else {
        return 2e-10;
    }
}

void check(double actual, double expected, double epsilon) {
    ++checks;
    double const error = std::abs(actual - expected);
    worst = std::max(worst, error);
    if (!std::isfinite(actual) || !std::isfinite(expected) || error > epsilon) {
        std::fprintf(stderr, "%s actual=%.17g expected=%.17g error=%.6g tolerance=%.6g\n",
            context.c_str(), actual, expected, error, epsilon);
        throw std::runtime_error("generated spherical kernel mismatch");
    }
}

template <typename Coefficients>
void clear(Coefficients& values, double initial = 0) {
    using Real = std::remove_reference_t<decltype(values[0])>;
    for (int i = 0; i < values.kCount_; ++i) {
        values[i] = simdBroadcast<Real>(initial);
    }
}

template <typename Coefficients>
void compare(Coefficients const& actual, Harmonics const& expected, int laneIndex,
    double initial = 0, bool relativeRank = false) {
    using Real = std::remove_cv_t<std::remove_reference_t<decltype(actual[0])>>;
    for (int n = 0; n <= Coefficients::kDegree_; ++n) {
        double scale = relativeRank ? 1e-100 : 1;
        for (int m = 0; m <= n; ++m) {
            scale = std::max(scale, std::abs(expected[index(n, m)]));
        }
        for (int m = 0; m <= n; ++m) {
            check((lane(actual[index(n, m)], laneIndex) - initial) / scale,
                expected[index(n, m)].real() / scale, tolerance<Real>());
            if (m) {
                check((lane(actual[index(n, -m)], laneIndex) - initial) / scale,
                    expected[index(n, m)].imag() / scale, tolerance<Real>());
            }
        }
    }
}

template <typename Real>
void compareField(
    std::array<Real, 4> const& field, Harmonics const& expected, int laneIndex, double initial) {
    std::array<double, 4> const oracle = {
        expected[0].real(), -expected[3].real(), -expected[3].imag(), -expected[2].real()};
    double scale = 1;
    for (double value : oracle) {
        scale = std::max(scale, std::abs(value));
    }
    for (int i = 0; i < 4; ++i) {
        check((lane(field[i], laneIndex) - initial) / scale, oracle[i] / scale, tolerance<Real>());
    }
}

template <typename Real, int degree>
void testSelected(spherical::Multipole<Real, degree + 1> const& source,
    spherical::Expansion<Real, degree + 1> const& local, Real x, Real y, Real z,
    std::array<Point, 4> const& shifts, std::array<Harmonics, 4> const& sourceOracle,
    std::array<Harmonics, 4> const& localOracle) {
    context = "degree " + std::to_string(degree) + " selected operators";
    spherical::Multipole<Real, degree + 1> moments;
    clear(moments, 0.125);
    spherical::detail::M2M<degree - 1>::template add<Real>(moments, source, x, y, z);
    for (int laneIndex = 0; laneIndex < laneCount<Real>(); ++laneIndex) {
        compare(moments, translateMoments(sourceOracle[laneIndex], shifts[laneIndex], degree - 1),
            laneIndex, 0.125);
    }
    spherical::Expansion<Real, degree + 1> shifted;
    clear(shifted, 0.125);
    spherical::detail::L2L<degree>::template add<Real>(shifted, local, x, y, z);
    std::array<Real, 4> field;
    field.fill(simdBroadcast<Real>(0.125));
    spherical::detail::L2P<degree>::template add<Real>(field, local, x, y, z);
    for (int laneIndex = 0; laneIndex < laneCount<Real>(); ++laneIndex) {
        compare(shifted, translateLocal(localOracle[laneIndex], shifts[laneIndex], degree, degree),
            laneIndex, 0.125);
        compareField(field, translateLocal(localOracle[laneIndex], shifts[laneIndex], degree, 1),
            laneIndex, 0.125);
    }
    // Keep zero translation in M2M/L2L, but use a nonsingular z displacement for M2L.
    for (int laneIndex = 0; laneIndex < laneCount<Real>(); ++laneIndex) {
        setLane(z, laneIndex, shifts[laneIndex][2] + 1.75);
    }
    clear(shifted, 0.125);
    field.fill(simdBroadcast<Real>(0.125));
    spherical::detail::M2L<degree>::template add<Real>(shifted, source, x, y, z);
    spherical::detail::M2P<degree>::template add<Real>(field, source, x, y, z);
    for (int laneIndex = 0; laneIndex < laneCount<Real>(); ++laneIndex) {
        Point point = shifts[laneIndex];
        point[2] = lane(z, laneIndex);
        Harmonics const greens = singular(degree, point);
        compare(shifted, m2l(sourceOracle[laneIndex], greens, degree, degree), laneIndex, 0.125);
        compareField(field, m2l(sourceOracle[laneIndex], greens, degree, 1), laneIndex, 0.125);
    }
}

template <typename Real, int degree>
void testOrder() {
    // Every lane has different geometry. Includes both axis signs, origin,
    // near-axis values and nonzero off-axis translations across the fixtures.
    std::array<Point, 8> const points = {{{0.31, -0.23, 0.17}, {0, 0, 0}, {0, 0, -0.41},
        {0.4, 0, 0}, {0, -0.3, 0}, {1e-12, -2e-12, 0.2}, {-0.2, 0.1, -0.3}, {0, 0, 0.6}}};
    for (int fixture = 0; fixture < 8; ++fixture) {
        std::array<Point, 4> shifts;
        std::array<Harmonics, 4> sourceOracle;
        std::array<Harmonics, 4> localOracle;
        spherical::Multipole<Real, degree + 1> source;
        spherical::Expansion<Real, degree + 1> local;
        clear(source);
        clear(local);
        Real x{}, y{}, z{}, mass{};
        for (int laneIndex = 0; laneIndex < laneCount<Real>(); ++laneIndex) {
            shifts[laneIndex] = points[(fixture + laneIndex) % points.size()];
            // Round the oracle input to float when testing float kernels.
            if constexpr (std::is_same_v<Real, float>) {
                for (double& value : shifts[laneIndex]) {
                    value = float(value);
                }
            }
            for (int axis = 0; axis < 3; ++axis) {
                setLane(axis == 0 ? x : axis == 1 ? y : z, laneIndex, shifts[laneIndex][axis]);
            }
            setLane(mass, laneIndex, 1);
            sourceOracle[laneIndex] = regular(degree - 1, negate(shifts[laneIndex]));
            localOracle[laneIndex].resize((degree + 1) * (degree + 1));
            for (int n = 0; n <= degree; ++n) {
                for (int m = 0; m <= n; ++m) {
                    double const real = std::sin(0.3 * (index(n, m) + laneIndex + 1));
                    double const imaginary = m ? std::cos(0.2 * (index(n, m) + laneIndex + 1)) : 0;
                    setLane(local[index(n, m)], laneIndex, real);
                    if (m) {
                        setLane(local[index(n, -m)], laneIndex, imaginary);
                    }
                    localOracle[laneIndex][index(n, m)] =
                        Complex(lane(local[index(n, m)], laneIndex),
                            m ? lane(local[index(n, -m)], laneIndex) : 0);
                }
            }
        }
        spherical::detail::P2M<degree - 1>::template add<Real>(source, mass, x, y, z);
        context = "P2M degree " + std::to_string(degree);
        for (int laneIndex = 0; laneIndex < laneCount<Real>(); ++laneIndex) {
            compare(source, sourceOracle[laneIndex], laneIndex, 0, true);
        }
        // Independent arbitrary coefficients exercise every rank/order, rather
        // than relying only on point moments tied to the translation direction.
        for (int laneIndex = 0; laneIndex < laneCount<Real>(); ++laneIndex) {
            for (int n = 0; n < degree; ++n) {
                for (int m = 0; m <= n; ++m) {
                    setLane(source[index(n, m)], laneIndex,
                        std::sin(0.29 * (index(n, m) + 7 * fixture + laneIndex + 1)));
                    if (m) {
                        setLane(source[index(n, -m)], laneIndex,
                            std::cos(0.17 * (index(n, m) + 5 * fixture + laneIndex + 1)));
                    }
                    sourceOracle[laneIndex][index(n, m)] =
                        Complex(lane(source[index(n, m)], laneIndex),
                            m ? lane(source[index(n, -m)], laneIndex) : 0);
                }
            }
        }
        testSelected<Real, degree>(source, local, x, y, z, shifts, sourceOracle, localOracle);
        for (int laneIndex = 0; laneIndex < laneCount<Real>(); ++laneIndex) {
            setLane(z, laneIndex, shifts[laneIndex][2] + 1.75);
        }
        spherical::Green<Real, degree + 1> green;
        spherical::Expansion<Real, degree + 1> pointLocal, derivativeLocal, selectedLocal;
        clear(pointLocal);
        clear(derivativeLocal);
        clear(selectedLocal);
        spherical::detail::Green<degree>::template evaluate<Real>(green, x, y, z);
        spherical::detail::P2L<degree>::template add<Real>(pointLocal, mass, x, y, z);
        spherical::detail::M2LDerivatives<degree>::template add<Real>(
            derivativeLocal, source, green);
        spherical::detail::M2L<degree>::template add<Real>(selectedLocal, source, x, y, z);
        context = "Green/P2L/arbitrary M2L/selected M2L degree " + std::to_string(degree);
        for (int laneIndex = 0; laneIndex < laneCount<Real>(); ++laneIndex) {
            Point point = shifts[laneIndex];
            point[2] = lane(z, laneIndex);
            Harmonics const greens = singular(degree, point);
            compare(green, greens, laneIndex);
            compare(pointLocal, greens, laneIndex);
            Harmonics const expected = m2l(sourceOracle[laneIndex], greens, degree, degree);
            compare(derivativeLocal, expected, laneIndex);
            compare(selectedLocal, expected, laneIndex);
        }
    }
}

template <int degree>
void directParticleCheck() {
    spherical::Multipole<double, degree + 1> source;
    clear(source);
    std::array<Point, 3> const particles = {
        {{0.002, -0.001, 0.003}, {-0.002, 0.003, 0.001}, {0.001, 0.001, -0.002}}};
    for (Point const& point : particles) {
        spherical::detail::P2M<degree - 1>::template add<double>(
            source, 0.7, point[0], point[1], point[2]);
    }
    Point const target = {0.9, -0.7, 1.1};
    std::array<double, 4> field{};
    spherical::detail::M2P<degree>::template add<double>(
        field, source, target[0], target[1], target[2]);
    std::array<double, 4> expected{};
    for (Point const& point : particles) {
        Point displacement;
        double radiusSquared = 0;
        for (int axis = 0; axis < 3; ++axis) {
            displacement[axis] = target[axis] - point[axis];
            radiusSquared += displacement[axis] * displacement[axis];
        }
        double const inverseRadius = 1 / std::sqrt(radiusSquared);
        expected[0] -= 0.7 * inverseRadius;
        for (int axis = 0; axis < 3; ++axis) {
            expected[axis + 1] -= 0.7 * displacement[axis] * inverseRadius / radiusSquared;
        }
    }
    context = "direct particle force degree " + std::to_string(degree);
    for (int component = 0; component < 4; ++component) {
        check(field[component], expected[component], 20 * std::pow(0.004, degree) + 2e-13);
    }
}

template <typename Real, int degree = kMinTestDegree>
void allOrders() {
    testOrder<Real, degree>();
    if constexpr (degree < kMaxTestDegree) {
        allOrders<Real, degree + 1>();
    }
}

template <int degree = kMinTestDegree>
void allDirectParticleChecks() {
    directParticleCheck<degree>();
    if constexpr (degree < kMaxTestDegree) {
        allDirectParticleChecks<degree + 1>();
    }
}

}    // namespace detail

int main() {
    try {
        detail::allOrders<float>();
        detail::allOrders<double>();
#if !defined(__CUDACC__) && !defined(GENERATED_SPHERICAL_SKIP_SIMD)
        detail::allOrders<SimdVector<double, 4>>();
#endif
        detail::allDirectParticleChecks();
        std::printf(
            "Generated Spherical: %zu independent checks passed, worst scaled discrepancy %.3e\n",
            detail::checks, detail::worst);
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
