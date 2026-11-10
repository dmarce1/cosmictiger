/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cosmictiger/gravity/fmm/tree_local.hpp>
#include <cosmictiger/gravity/kernels/basis_conversion.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>
#include <cosmictiger/gravity/kernels/spherical/p2m.hpp>
#include <cosmictiger/math/fixed.hpp>

void cosmictigerAssert(char const*, bool condition, char const*, int) {
    if (!condition) {
        std::abort();
    }
}

namespace detail {
namespace cartesian = cosmictiger::gravity::cartesian;
namespace spherical = cosmictiger::gravity::spherical;
using Point = Vector3<double>;
using Complex = std::complex<long double>;
constexpr int kOrder = BASIS_CONVERSION_TEST_ORDER;
size_t checks = 0;
double worst = 0;
std::string context;

template <class Real>
constexpr int lanes() {
    if constexpr (std::is_arithmetic_v<Real>) {
        return 1;
    } else {
        return Real::size();
    }
}
template <class Real>
double lane(Real const& value, int i) {
    if constexpr (std::is_arithmetic_v<Real>) {
        return value;
    } else {
        return value[i];
    }
}
template <class Real>
void setLane(Real& value, int i, double input) {
    if constexpr (std::is_arithmetic_v<Real>) {
        value = Real(input);
    } else {
        value[i] = static_cast<typename Real::value_type>(input);
    }
}
template <class Real>
constexpr double tolerance() {
    if constexpr (std::is_arithmetic_v<Real>) {
        return std::is_same_v<Real, float> ? 4e-5 : 3e-11;
    } else {
        return tolerance<typename Real::value_type>();
    }
}
void check(double actual, double expected, double epsilon) {
    ++checks;
    double const error = std::abs(actual - expected) / std::max(1.0, std::abs(expected));
    worst = std::max(worst, error);
    if (!std::isfinite(error) || error > epsilon) {
        std::fprintf(stderr, "%s: %.17g != %.17g (scaled error %.3e)\n", context.c_str(), actual,
            expected, error);
        throw std::runtime_error("basis conversion mismatch");
    }
}
template <class Coefficients>
size_t coefficientCount(Coefficients const& values) {
    if constexpr (requires { values.size(); }) {
        return values.size();
    } else {
        return Coefficients::kCount_;
    }
}
template <class Coefficients>
void clear(Coefficients& values, double initial = 0) {
    using Real = std::remove_reference_t<decltype(values[0])>;
    for (size_t i = 0; i < coefficientCount(values); ++i) {
        values[i] = math::broadcast<Real>(initial);
    }
}
template <class Real>
Vector3<Real> point(Point const& p) {
    return {math::broadcast<Real>(p[0]), math::broadcast<Real>(p[1]), math::broadcast<Real>(p[2])};
}

long double factorial(int n) {
    long double value = 1;
    for (int i = 2; i <= n; ++i) {
        value *= i;
    }
    return value;
}

// Independent closed polynomial, derived from Dehnen (2014), Eq. 2b,
// https://doi.org/10.1186/s40668-014-0001-7 and the finite Legendre sum.
// Uₙₘ = Σₖ (-1)ᵏ(2n-2k)! (x+iy)ᵐ zⁿ⁻ᵐ⁻²ᵏ (r²)ᵏ /
//       [2ⁿ k! (n-k)! (n-m-2k)! (n+m)!]. Differentiate each term analytically.
// Symbol directory: U is the regular harmonic; n,m are degree and azimuthal
// order; k is the sum index; x,y,z are coordinates; r²=x²+y²+z².
// Code names: r2 = r². Results store U, ∂ₓU, ∂ᵧU, ∂zU.
std::array<Complex, 4> regular(int n, int m, Point const& p) {
    std::array<Complex, 4> result{};
    Complex const w(p[0], p[1]);
    long double const z = p[2];
    long double const r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    for (int k = 0; 2 * k <= n - m; ++k) {
        int const a = n - m - 2 * k;
        long double const c = (k % 2 ? -1 : 1) * factorial(2 * n - 2 * k) /
            (std::pow(2.L, n) * factorial(k) * factorial(n - k) * factorial(a) * factorial(n + m));
        Complex const v = c * std::pow(w, m) * std::pow(z, a) * std::pow(r2, k);
        result[0] += v;
        if (m) {
            Complex const d = c * static_cast<long double>(m) * std::pow(w, m - 1) *
                std::pow(z, a) * std::pow(r2, k);
            result[1] += d;
            result[2] += Complex(0, 1) * d;
        }
        if (a) {
            result[3] += c * static_cast<long double>(a) * std::pow(w, m) * std::pow(z, a - 1) *
                std::pow(r2, k);
        }
        if (k) {
            for (int axis = 0; axis < 3; ++axis) {
                result[axis + 1] +=
                    c * (2.L * k * p[axis]) * std::pow(w, m) * std::pow(z, a) * std::pow(r2, k - 1);
            }
        }
    }
    return result;
}

template <class Real>
void basisChecks() {
    using Cl = cartesian::Expansion<Real, kOrder>;
    using Sl = spherical::Expansion<Real, kOrder>;
    using Cm = cartesian::Multipole<Real, kOrder>;
    using Sm = spherical::Multipole<Real, kOrder>;
    static_assert(std::is_constructible_v<Cl, Sl const&>);
    static_assert(std::is_constructible_v<Sm, Cm const&>);
    static_assert(!std::is_convertible_v<Sl, Cl>);
    static_assert(!std::is_convertible_v<Cm, Sm>);
    static_assert(!std::is_constructible_v<Sl, Cl const&>);
    static_assert(!std::is_constructible_v<Cm, Sm const&>);
    using Field = cosmictiger::gravity::Field<Real>;
    std::array<Point, 3> const samples{{{.37, -.29, .43}, {-.41, .19, -.31}, {0, 0, .53}}};
    context = "every local component against a closed polynomial";
    for (int n = 0; n < kOrder; ++n) {
        for (int m = -n; m <= n; ++m) {
            Sl s;
            clear(s);
            for (int j = 0; j < lanes<Real>(); ++j) {
                setLane(s[n * (n + 1) + m], j, 1 + .125 * j);
            }
            Cl c;
            clear(c, 17);
            c = static_cast<Cl>(s);
            for (auto const& p : samples) {
                Field cf;
                clear(cf);
                cartesian::L2P(cf, c, point<Real>(p));
                auto const oracle = regular(n, std::abs(m), p);
                for (int component = 0; component < 4; ++component) {
                    double const value =
                        double(m < 0 ? oracle[component].imag() : oracle[component].real());
                    for (int j = 0; j < lanes<Real>(); ++j) {
                        check(lane(cf[component], j),
                            value * (m ? 2 : 1) * (component ? -1 : 1) * (1 + .125 * j),
                            tolerance<Real>());
                    }
                }
            }
        }
    }
}

template <class Real>
void cloudCheck(std::array<Point, 4> const& positions) {
    cartesian::Multipole<Real, kOrder> c;
    clear(c);
    spherical::Multipole<Real, kOrder> s;
    clear(s, 19);
    std::array<double, 4> const masses{.7, 1.3, .2, 2.1};
    for (size_t i = 0; i < positions.size(); ++i) {
        cartesian::P2M(c, math::broadcast<Real>(masses[i]), point<Real>(positions[i]));
    }
    s = static_cast<spherical::Multipole<Real, kOrder>>(c);
    for (int n = 0; n <= kOrder - 2; ++n) {
        double rankScale = 1e-100;
        std::array<Complex, kOrder> expected{};
        for (int m = 0; m <= n; ++m) {
            for (size_t i = 0; i < positions.size(); ++i) {
                expected[m] += static_cast<long double>((n % 2 ? -1 : 1) * masses[i]) *
                    regular(n, m, positions[i])[0];
            }
            rankScale = std::max(rankScale, double(std::abs(expected[m])));
        }
        for (int m = 0; m <= n; ++m) {
            check(lane(s.real(n, m), 0) / rankScale, double(expected[m].real()) / rankScale,
                tolerance<Real>());
            if (m) {
                check(lane(s.imaginary(n, m), 0) / rankScale,
                    double(expected[m].imag()) / rankScale, tolerance<Real>());
            }
        }
    }
}

template <class Fixed>
void seamCheck() {
    context = "fixed-point periodic seam, subtract before conversion";
    std::array<Point, 4> offsets;
    Fixed const center(.997);
    std::array<double, 4> const coordinates{.003, .991, .007, .982};
    for (size_t i = 0; i < offsets.size(); ++i) {
        double const x = doubleDistance(Fixed(coordinates[i]), center);
        check(x, coordinates[i] - .997 + (coordinates[i] < .5 ? 1 : 0), 3e-10);
        offsets[i] = {x, -.7 * x, .3 * x};
    }
    cloudCheck<double>(offsets);
}

void sourceScalingCheck() {
    // The Cartesian tree uses physical coordinates; PMMM source moments use mesh-cell units.
    // Derived boundary scaling: x_tree = κ h u_mesh.
    double const κ = kScaleFactor;
    for (double const h : {.125, .03125, .0078125}) {
        context = "mesh/tree source rank scaling";
        cartesian::Multipole<double, kOrder> cm;
        spherical::Multipole<double, kOrder> sm, direct;
        clear(cm);
        clear(sm);
        clear(direct);
        for (int i = 0; i < 4; ++i) {
            Point const u{.12 + .03 * i, -.21 + .04 * i, .18 - .05 * i};
            cartesian::P2M(cm, .7 + .3 * i, Point{κ * h * u[0], κ * h * u[1], κ * h * u[2]});
            spherical::detail::P2M<kOrder - 2, double>(direct, .7 + .3 * i, u[0], u[1], u[2]);
        }
        sm = static_cast<spherical::Multipole<double, kOrder>>(cm);
        for (int n = 0; n <= kOrder - 2; ++n) {
            double scale = 1e-100;
            for (int m = -n; m <= n; ++m) {
                scale = std::max(scale, std::abs(direct[n * (n + 1) + m]));
            }
            for (int m = -n; m <= n; ++m) {
                check(sm[n * (n + 1) + m] * std::pow(κ * h, -n) / scale,
                    direct[n * (n + 1) + m] / scale, 3e-11);
            }
        }
    }
}

template <class Real, class PositionType>
void productionTreeLocal() {
    namespace local = cosmictiger::gravity::tree_local;
    context = "production mesh/tree local, native centers and potential sentinels";
    using Local = cartesian::Expansion<Real, kOrder>;
    using Field = cosmictiger::gravity::Field<Real>;
    constexpr double kTolerance = std::is_same_v<Real, float> ? 2e-6 : 3e-11;
    for (int meshSize : {4, 8, 16, 64}) {
        double const h = 1. / meshSize;
        double const βₘ = 2.3 * h * h;
        double const K = local::backgroundCurvature(βₘ, meshSize);
        for (bool tiny : {false, true}) {
            Vector3<PositionType> mesh, root, child, sink;
            for (int axis = 0; axis < 3; ++axis) {
                mesh[axis] = 1. - .5 * h;
                root[axis] = mesh[axis].toDouble() + (.12 - .09 * axis) * h;
                child[axis] = mesh[axis].toDouble() + (.21 - .15 * axis) * h;
                sink[axis] = mesh[axis].toDouble() + (.31 - .22 * axis) * h;
                if (tiny) {
                    mesh[axis].raw() = PositionType::max().raw() - 128;
                    root[axis].raw() = mesh[axis].raw() + 5 + axis;
                    child[axis].raw() = mesh[axis].raw() + 13 + axis;
                    sink[axis].raw() = mesh[axis].raw() + 21 + axis;
                }
            }
            Point u;
            for (int axis = 0; axis < 3; ++axis) {
                u[axis] = unwrappedDoubleDistance(sink[axis], mesh[axis]) * meshSize;
                if (tiny) {
                    double const unit = kScaleFactor * PositionType::kToFloatFactor_;
                    check(local::displacement<Real>(root, mesh)[axis] / unit, 5 + axis, 2e-15);
                    check(local::displacement<Real>(child, root)[axis] / unit, 8, 2e-15);
                    check(local::displacement<Real>(sink, child)[axis] / unit, 8, 2e-15);
                }
            }
            for (int n = 0; n < kOrder; ++n) {
                for (int m = -n; m <= n; ++m) {
                    spherical::Expansion<double, kOrder> source{};
                    source(n, m) = 1;
                    auto const oracle = regular(n, std::abs(m), u);
                    for (bool doPotential : {true, false}) {
                        cartesian::Expansion<double, kOrder> initial{}, rootDouble{};
                        rootDouble[0] = doPotential ? 0 : 73;
                        local::fromMesh(initial, source.data(), βₘ, meshSize);
                        local::translate(
                            rootDouble, initial, local::displacement(root, mesh), K, doPotential);
                        Local atRoot{}, atChild{};
                        for (int i = 0; i < atRoot.size(); ++i) {
                            atRoot[i] = Real(rootDouble[i]);
                        }
                        atChild[0] = doPotential ? 0 : 73;
                        local::translate(atChild, atRoot, local::displacement<Real>(child, root),
                            Real(K), doPotential);
                        Field field{};
                        field[0] = 37;
                        local::evaluate(field, atChild, local::displacement<Real>(sink, child),
                            Real(K), doPotential);
                        for (int component = 0; component < 4; ++component) {
                            double value =
                                double(m < 0 ? oracle[component].imag() : oracle[component].real());
                            value *= m == 0 ? 1 : 2;
                            if (component == 0) {
                                double const expected = 37 + value - .5 * βₘ * u.dot(u);
                                check(field[0], doPotential ? expected : 37, kTolerance);
                            } else {
                                check(field[component], (-value + βₘ * u[component - 1]) / h,
                                    kTolerance);
                            }
                        }
                        if (!doPotential) {
                            check(atRoot[0], 73, 0);
                            check(atChild[0], 73, 0);
                        }
                    }
                }
            }
        }
    }
}
}    // namespace detail

int main() {
    try {
        detail::basisChecks<float>();
        detail::basisChecks<double>();
        detail::basisChecks<SimdVector<float, 4>>();
        detail::basisChecks<SimdVector<double, 2>>();
        detail::context = "unequal-mass asymmetric source, independent P2M";
        std::array<detail::Point, 4> const cloud{
            {{.19, -.27, .31}, {-.13, .23, .17}, {.32, .11, -.21}, {-.07, -.19, .29}}};
        detail::cloudCheck<float>(cloud);
        detail::cloudCheck<double>(cloud);
        detail::seamCheck<Fixed32>();
        detail::seamCheck<Fixed64>();
        detail::sourceScalingCheck();
        detail::productionTreeLocal<double, Fixed32>();
        detail::productionTreeLocal<double, Fixed64>();
        detail::productionTreeLocal<float, Fixed32>();
        detail::productionTreeLocal<float, Fixed64>();
        std::printf("Basis conversions q=%d: %zu checks passed, worst scaled discrepancy %.3e\n",
            detail::kOrder, detail::checks, detail::worst);
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
