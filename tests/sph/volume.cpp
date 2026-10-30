/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/math/radial_kernel.hpp>
#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/sph/variational/equations.hpp>
#include <cosmictiger/sph/volume.hpp>
#include <tests/sph/test_helpers.hpp>

// Double-precision references for cosmictiger/sph/volume.hpp and
// cosmictiger/sph/variational/equations.hpp. A small isolated particle system with its own
// exact support solve supplies the discrete thermal energy whose gradient the compatible
// density-energy force must be (docs/variational_sph.md).
namespace detail {

using Vector = std::array<double, 3>;
using sph_volume::SupportConstraint;

Vector difference(Vector const& left, Vector const& right) {
    return {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}

double dot(Vector const& left, Vector const& right) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

Vector cross(Vector const& left, Vector const& right) {
    return {left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0]};
}

std::vector<radial_kernel::KernelSpec> selectableKernels() {
    std::vector<radial_kernel::KernelSpec> kernels;
    for (int degree = 2; degree <= 9; ++degree) {
        kernels.push_back({radial_kernel::Family::Spline, degree});
    }
    for (int degree : {2, 5, 8}) {
        kernels.push_back({radial_kernel::Family::Wendland, degree});
    }
    for (int degree : {2, 4, 6, 8}) {
        kernels.push_back({radial_kernel::Family::Ferrers, degree});
    }
    return kernels;
}

std::string kernelName(radial_kernel::KernelSpec spec) {
    char const* family = spec.family == radial_kernel::Family::Spline ?
        "spline" :
        (spec.family == radial_kernel::Family::Wendland ? "wendland" : "ferrers");
    return std::string(family) + std::to_string(spec.degree);
}

// A reproducible uniform generator independent of the standard library's distributions.
class Generator {
    std::uint64_t state_;

public:
    explicit Generator(std::uint64_t seed)
      : state_(seed) {}

    double next() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(state_ >> 11) * 0x1.0p-53;
    }
};

// W(r, H) = H^-3 w(r/H).
double kernel(radial_kernel::KernelTable const& table, double radius, double support) {
    double const q = radius / support;
    return q < 1.0 ? radial_kernel::value(table, q) / (support * support * support) : 0.0;
}

struct System {
    radial_kernel::KernelTable table;
    std::vector<Vector> position;
    std::vector<double> mass;
    std::vector<double> weight;
    std::vector<double> entropy;    // P_i = K_i rho*_i^gamma
    double gamma = 5.0 / 3.0;
    double targetSum = 0.0;    // sum_j w(q_ij) of the constraint H^3 n = constant
    double minimumSupport = 1.0e-3;
    double maximumSupport = 1.0e3;
    std::vector<double> support;
    std::vector<SupportConstraint> constraint;

    int size() const {
        return int(position.size());
    }

    double kernelSum(int i, double supportRadius) const {
        double sum = 0.0;
        for (int j = 0; j < size(); ++j) {
            double const q = std::sqrt(dot(difference(position[i], position[j]),
                                 difference(position[i], position[j]))) /
                supportRadius;
            sum += q < 1.0 ? radial_kernel::value(table, q) : 0.0;
        }
        return sum;
    }

    // The support solving sum_j w(r_ij/H) = target, bracketed and bisected to roundoff;
    // the sum increases monotonically with H.
    double freeSupport(int i) const {
        double lower = 1.0e-3;
        double upper = 0.05;
        detail::require(kernelSum(i, lower) < targetSum, "target exceeds the self term");
        while (kernelSum(i, upper) < targetSum) {
            upper *= 2.0;
        }
        for (int iteration = 0; iteration < 200 && upper - lower > 1.0e-15 * upper; ++iteration) {
            double const middle = 0.5 * (lower + upper);
            (kernelSum(i, middle) < targetSum ? lower : upper) = middle;
        }
        return 0.5 * (lower + upper);
    }

    void solveSupports() {
        support.resize(size());
        constraint.resize(size());
        for (int i = 0; i < size(); ++i) {
            double const free = freeSupport(i);
            if (minimumSupport == maximumSupport) {
                support[i] = minimumSupport;
                constraint[i] = SupportConstraint::Fixed;
            } else if (free < minimumSupport) {
                support[i] = minimumSupport;
                constraint[i] = SupportConstraint::LowerBound;
            } else if (free > maximumSupport) {
                support[i] = maximumSupport;
                constraint[i] = SupportConstraint::UpperBound;
            } else {
                support[i] = free;
                constraint[i] = SupportConstraint::Free;
            }
        }
    }

    sph_volume::SupportSums<double> sums(int i) const {
        auto result = sph_volume::emptySupportSums<double>();
        for (int j = 0; j < size(); ++j) {
            double const q = std::sqrt(dot(difference(position[i], position[j]),
                                 difference(position[i], position[j]))) /
                support[i];
            if (q < 1.0) {
                sph_volume::accumulateSupportSums(result, weight[j], radial_kernel::value(table, q),
                    radial_kernel::derivative(table, q), q);
            }
        }
        return result;
    }

    double weightedDensity(int i) const {
        return sph_volume::weightedDensity(sums(i), 1.0 / support[i]);
    }

    double volumeDensity(int i) const {
        return sph_volume::volumeDensity(mass[i], weight[i], weightedDensity(i));
    }

    double pressure(int i) const {
        return entropy[i] * std::pow(volumeDensity(i), gamma);
    }

    // m_i u_i at the fixed entropy K_i, so that d(m_i u_i) = -P_i dV_i.
    double thermalEnergy(int i) const {
        return entropy[i] * mass[i] * std::pow(volumeDensity(i), gamma - 1.0) / (gamma - 1.0);
    }

    double energy() const {
        double total = 0.0;
        for (int i = 0; i < size(); ++i) {
            total += thermalEnergy(i);
        }
        return total;
    }

    double gradientFactor(double radius, double supportRadius) const {
        double const q = radius / supportRadius;
        return q < 1.0 ? sph_volume::kernelGradientFactor(
                             radial_kernel::derivative(table, q), 1.0 / supportRadius, radius) :
                         0.0;
    }

    struct Rates {
        std::vector<Vector> force;    // m_i dv_i/dt
        std::vector<double> work;     // d(m_i u_i)/dt
    };

    // Hopkins (2013) equation 12 and the compatible work with the directed Q_ij.
    Rates rates(std::vector<Vector> const& velocity) const {
        std::vector<double> factor(size());
        std::vector<double> zeta(size());
        for (int i = 0; i < size(); ++i) {
            factor[i] =
                variational::pressureVolumeFactor(pressure(i), weight[i], weightedDensity(i));
            zeta[i] = sph_volume::gradHCorrection(sums(i), constraint[i]).zeta;
        }
        Rates result{std::vector<Vector>(size(), Vector{}), std::vector<double>(size(), 0.0)};
        for (int i = 0; i < size(); ++i) {
            for (int j = 0; j < size(); ++j) {
                if (i == j) {
                    continue;
                }
                Vector const separation = difference(position[i], position[j]);
                double const radius = std::sqrt(dot(separation, separation));
                double const gradientI = gradientFactor(radius, support[i]);
                double const gradientJ = gradientFactor(radius, support[j]);
                double const coefficientIJ =
                    variational::directedCoefficient(factor[i], weight[j], zeta[i]);
                double const coefficientJI =
                    variational::directedCoefficient(factor[j], weight[i], zeta[j]);
                double const pairForce = variational::pairForceFactor(
                    coefficientIJ, coefficientJI, gradientI, gradientJ);
                for (int dim = 0; dim < 3; ++dim) {
                    result.force[i][dim] -= pairForce * separation[dim];
                }
                result.work[i] += variational::pairWork(coefficientIJ, gradientI,
                    dot(difference(velocity[i], velocity[j]), separation));
            }
        }
        return result;
    }
};

enum class WeightModel { EqualMass, UnequalMass, UnequalWeight };

System makeSystem(radial_kernel::KernelSpec spec, WeightModel model, double gamma,
    std::uint64_t seed, int count = 48, double neighborCount = 32.0) {
    System system;
    system.table = radial_kernel::makeKernelTable(spec);
    system.gamma = gamma;
    system.targetSum = neighborCount * 3.0 / (4.0 * 3.141592653589793238462643383279502884);
    Generator generator(seed);
    for (int i = 0; i < count; ++i) {
        // A denser core gives a range of supports in one system.
        double const squeeze = i < count / 3 ? 0.45 : 1.0;
        system.position.push_back({0.5 + squeeze * (generator.next() - 0.5),
            0.5 + squeeze * (generator.next() - 0.5), 0.5 + squeeze * (generator.next() - 0.5)});
        double const mass =
            model == WeightModel::EqualMass ? 1.0 / count : (0.5 + 1.5 * generator.next()) / count;
        system.mass.push_back(mass);
        system.weight.push_back(
            model == WeightModel::UnequalWeight ? (0.4 + 1.6 * generator.next()) / count : mass);
        system.entropy.push_back(0.5 + generator.next());
    }
    system.solveSupports();
    return system;
}

// Bounds between well-separated free supports near the given quantiles, so that
// finite-difference perturbations cannot move a support across a bound.
void holdSupportsAtBothBounds(System& system, double lowerQuantile, double upperQuantile) {
    std::vector<double> sorted = system.support;
    std::sort(sorted.begin(), sorted.end());
    auto const gapBound = [&](double quantile) {
        int const centre = int(quantile * (sorted.size() - 1));
        int best = centre;
        for (int k = std::max(0, centre - 3); k <= std::min(int(sorted.size()) - 2, centre + 3);
            ++k) {
            if (sorted[k + 1] - sorted[k] > sorted[best + 1] - sorted[best]) {
                best = k;
            }
        }
        detail::require((sorted[best + 1] - sorted[best]) > 1.0e-4 * sorted[best],
            "free supports too close to place a bound");
        return 0.5 * (sorted[best] + sorted[best + 1]);
    };
    system.minimumSupport = gapBound(lowerQuantile);
    system.maximumSupport = gapBound(upperQuantile);
    system.solveSupports();
}

int countConstraint(System const& system, SupportConstraint constraint) {
    return int(std::count(system.constraint.begin(), system.constraint.end(), constraint));
}

}    // namespace detail

using namespace detail;

TEST(SphVolume, KernelSupportDerivativeMatchesFiniteDifferences) {
    for (auto const spec : selectableKernels()) {
        SCOPED_TRACE(kernelName(spec));
        auto const table = radial_kernel::makeKernelTable(spec);
        for (double const support : {0.6, 1.7}) {
            double scale = 0.0;
            std::vector<std::array<double, 2>> samples;
            for (int k = 0; k < 25; ++k) {
                double const radius = (k + 0.37) / 25.0 * support;
                double const q = radius / support;
                double const analytic =
                    sph_volume::kernelSupportDerivative(radial_kernel::value(table, q),
                        radial_kernel::derivative(table, q), q, 1.0 / support);
                double const step = 1.0e-6 * support;
                double const numeric = (kernel(table, radius, support + step) -
                                           kernel(table, radius, support - step)) /
                    (2.0 * step);
                scale = std::max(scale, std::abs(analytic));
                samples.push_back({analytic, numeric});
            }
            for (auto const& [analytic, numeric] : samples) {
                EXPECT_NEAR(analytic, numeric, 2.0e-7 * scale);
            }
        }
    }
}

TEST(SphVolume, KernelGradientFactorMatchesFiniteDifferences) {
    for (auto const spec : selectableKernels()) {
        SCOPED_TRACE(kernelName(spec));
        auto const table = radial_kernel::makeKernelTable(spec);
        double const support = 0.8;
        for (int k = 0; k < 25; ++k) {
            double const radius = (k + 0.41) / 25.0 * support;
            double const q = radius / support;
            double const analytic =
                sph_volume::kernelGradientFactor(
                    radial_kernel::derivative(table, q), 1.0 / support, radius) *
                radius;
            double const step = 1.0e-6 * support;
            double const numeric =
                (kernel(table, radius + step, support) - kernel(table, radius - step, support)) /
                (2.0 * step);
            EXPECT_NEAR(analytic, numeric, 1.0e-6 * std::max(1.0, std::abs(analytic)));
        }
    }
}

// n_i and y_i, their H derivatives, V_i, rho*_i, Omega_i and zeta_i with unequal weights
// and masses, compared with direct sums and finite differences in H.
TEST(SphVolume, WeightedVolumeAndConstraintDerivatives) {
    auto system =
        makeSystem({radial_kernel::Family::Wendland, 5}, WeightModel::UnequalWeight, 1.4, 17);
    for (int i = 0; i < system.size(); i += 5) {
        double const support = system.support[i];
        auto const sums = system.sums(i);
        auto const direct = [&](double supportRadius, bool weighted) {
            double sum = 0.0;
            for (int j = 0; j < system.size(); ++j) {
                Vector const separation = difference(system.position[i], system.position[j]);
                sum += (weighted ? system.weight[j] : 1.0) *
                    kernel(system.table, std::sqrt(dot(separation, separation)), supportRadius);
            }
            return sum;
        };
        double const n = sph_volume::numberDensity(sums, 1.0 / support);
        double const y = sph_volume::weightedDensity(sums, 1.0 / support);
        EXPECT_NEAR(n, direct(support, false), 1.0e-13 * n);
        EXPECT_NEAR(y, direct(support, true), 1.0e-13 * y);
        double const step = 1.0e-6 * support;
        double const dndH = sph_volume::numberDensityDerivative(sums, 1.0 / support);
        double const dydH = sph_volume::weightedDensityDerivative(sums, 1.0 / support);
        EXPECT_NEAR(dndH,
            (direct(support + step, false) - direct(support - step, false)) / (2.0 * step),
            1.0e-6 * std::abs(dndH));
        EXPECT_NEAR(dydH,
            (direct(support + step, true) - direct(support - step, true)) / (2.0 * step),
            1.0e-6 * std::abs(dydH));
        double const omega = sph_volume::omega(sums);
        double const zeta = sph_volume::zeta(sums);
        EXPECT_NEAR(omega, 1.0 + support / (3.0 * n) * dndH, 1.0e-12);
        EXPECT_NEAR(zeta, support / (3.0 * n * omega) * dydH, 1.0e-12 * system.weight[i]);
        double const volume = sph_volume::volume(system.weight[i], y);
        double const volumeDensity = sph_volume::volumeDensity(system.mass[i], system.weight[i], y);
        EXPECT_NEAR(volume * volumeDensity, system.mass[i], 1.0e-15 * system.mass[i]);
        EXPECT_GT(omega, 0.0);
    }
}

// x_i = m_i with equal masses: y = m n, zeta = m (1 - 1/Omega), f_ij = 1/Omega, and the
// directed coefficients give Price (2012) equation 30.
TEST(SphVolume, EqualMassReducesToStandardGradH) {
    auto system =
        makeSystem({radial_kernel::Family::Spline, 3}, WeightModel::EqualMass, 5.0 / 3.0, 5);
    double const mass = system.mass[0];
    for (int i = 0; i < system.size(); ++i) {
        auto const sums = system.sums(i);
        auto const equal =
            sph_volume::equalWeightSupportSums(sums.numberSum, sums.numberMoment, mass);
        EXPECT_NEAR(equal.weightedSum, sums.weightedSum, 1.0e-15 * sums.weightedSum);
        EXPECT_NEAR(
            equal.weightedMoment, sums.weightedMoment, 1.0e-14 * std::abs(sums.weightedMoment));
        double const omega = sph_volume::omega(sums);
        double const zeta = sph_volume::zeta(sums);
        EXPECT_NEAR(zeta, mass * (1.0 - 1.0 / omega), 1.0e-13 * mass);
        EXPECT_NEAR(sph_volume::correctionFactor(zeta, mass), 1.0 / omega, 1.0e-13);
        EXPECT_NEAR(mass - zeta, mass / omega, 1.0e-13 * mass);
    }
    std::vector<Vector> const still(system.size(), Vector{});
    auto const rates = system.rates(still);
    for (int i = 0; i < system.size(); ++i) {
        Vector price{};
        double const omegaI = sph_volume::omega(system.sums(i));
        double const densityI = system.weightedDensity(i);
        for (int j = 0; j < system.size(); ++j) {
            if (i == j) {
                continue;
            }
            double const omegaJ = sph_volume::omega(system.sums(j));
            double const densityJ = system.weightedDensity(j);
            Vector const separation = difference(system.position[i], system.position[j]);
            double const radius = std::sqrt(dot(separation, separation));
            double const pair = mass * mass *
                (system.pressure(i) / (omegaI * densityI * densityI) *
                        system.gradientFactor(radius, system.support[i]) +
                    system.pressure(j) / (omegaJ * densityJ * densityJ) *
                        system.gradientFactor(radius, system.support[j]));
            for (int dim = 0; dim < 3; ++dim) {
                price[dim] -= pair * separation[dim];
            }
        }
        for (int dim = 0; dim < 3; ++dim) {
            EXPECT_NEAR(
                rates.force[i][dim], price[dim], 1.0e-12 * std::max(1.0, std::abs(price[dim])));
        }
    }
}

// dy_i/dr_k and dH_i/dr_k with the support re-solved after each perturbation:
// dy_i = sum_j (x_j - zeta_i) grad_i W_ij . dr_ij and dH_i = -H_i/(3 n_i Omega_i) dn_i.
TEST(SphVolume, ConstrainedDensityDerivatives) {
    auto system =
        makeSystem({radial_kernel::Family::Spline, 4}, WeightModel::UnequalWeight, 5.0 / 3.0, 23);
    for (int i : {0, 7, 30}) {
        auto const sums = system.sums(i);
        double const zeta = sph_volume::zeta(sums);
        double const omega = sph_volume::omega(sums);
        double const n = sph_volume::numberDensity(sums, 1.0 / system.support[i]);
        for (int k : {i, (i + 3) % system.size(), (i + 11) % system.size()}) {
            Vector expectedDensity{};
            Vector expectedSupport{};
            for (int j = 0; j < system.size(); ++j) {
                if (j == i || (k != i && j != k)) {
                    continue;
                }
                Vector const separation = difference(system.position[i], system.position[j]);
                double const gradient = system.gradientFactor(
                    std::sqrt(dot(separation, separation)), system.support[i]);
                double const sign = k == i ? 1.0 : -1.0;
                for (int dim = 0; dim < 3; ++dim) {
                    expectedDensity[dim] +=
                        sign * (system.weight[j] - zeta) * gradient * separation[dim];
                    expectedSupport[dim] -=
                        sign * system.support[i] / (3.0 * n * omega) * gradient * separation[dim];
                }
            }
            for (int dim = 0; dim < 3; ++dim) {
                double const step = 1.0e-6;
                auto perturbed = system;
                perturbed.position[k][dim] += step;
                perturbed.solveSupports();
                double const densityPlus = perturbed.weightedDensity(i);
                double const supportPlus = perturbed.support[i];
                perturbed.position[k][dim] -= 2.0 * step;
                perturbed.solveSupports();
                double const densityScale = system.weightedDensity(i) / system.support[i];
                EXPECT_NEAR(expectedDensity[dim],
                    (densityPlus - perturbed.weightedDensity(i)) / (2.0 * step),
                    2.0e-6 * densityScale);
                EXPECT_NEAR(expectedSupport[dim],
                    (supportPlus - perturbed.support[i]) / (2.0 * step), 2.0e-6);
            }
        }
    }
}

// The force is minus the gradient of sum_i m_i u_i(rho*_i) at fixed entropy, with every
// support re-solved, for free supports, fixed supports and both active bounds.
TEST(SphVolume, ForceIsTheGradientOfTheDiscreteThermalEnergy) {
    struct Case {
        char const* label;
        radial_kernel::KernelSpec spec;
        WeightModel weights;
        double gamma;
        int supports;    // 0 free, 1 fixed, 2 both bounds active
    };
    std::array<Case, 6> const cases{{
        {"free cubic, gamma 5/3", {radial_kernel::Family::Spline, 3}, WeightModel::EqualMass,
            5.0 / 3.0, 0},
        {"free Wendland C4, gamma 7/5", {radial_kernel::Family::Wendland, 8},
            WeightModel::EqualMass, 1.4, 0},
        {"free unequal masses", {radial_kernel::Family::Ferrers, 6}, WeightModel::UnequalMass, 1.4,
            0},
        {"free unequal weights", {radial_kernel::Family::Spline, 5}, WeightModel::UnequalWeight,
            5.0 / 3.0, 0},
        {"fixed supports", {radial_kernel::Family::Spline, 3}, WeightModel::UnequalWeight, 1.4, 1},
        {"both bounds active", {radial_kernel::Family::Wendland, 5}, WeightModel::UnequalMass,
            5.0 / 3.0, 2},
    }};
    for (auto const& testCase : cases) {
        SCOPED_TRACE(testCase.label);
        auto system = makeSystem(testCase.spec, testCase.weights, testCase.gamma, 41);
        if (testCase.supports == 1) {
            system.minimumSupport = system.maximumSupport = 0.42;
            system.solveSupports();
            EXPECT_EQ(countConstraint(system, SupportConstraint::Fixed), system.size());
        } else if (testCase.supports == 2) {
            holdSupportsAtBothBounds(system, 0.25, 0.75);
            EXPECT_GT(countConstraint(system, SupportConstraint::LowerBound), 5);
            EXPECT_GT(countConstraint(system, SupportConstraint::UpperBound), 5);
            EXPECT_GT(countConstraint(system, SupportConstraint::Free), 5);
        } else {
            EXPECT_EQ(countConstraint(system, SupportConstraint::Free), system.size());
        }
        std::vector<Vector> const still(system.size(), Vector{});
        auto const rates = system.rates(still);
        double scale = 0.0;
        for (auto const& force : rates.force) {
            scale = std::max(scale, std::sqrt(dot(force, force)));
        }
        double const step = 1.0e-6;
        for (int k = 0; k < system.size(); k += 3) {
            for (int dim = 0; dim < 3; ++dim) {
                auto perturbed = system;
                perturbed.position[k][dim] += step;
                perturbed.solveSupports();
                ASSERT_EQ(perturbed.constraint, system.constraint);
                double const energyPlus = perturbed.energy();
                perturbed.position[k][dim] -= 2.0 * step;
                perturbed.solveSupports();
                ASSERT_EQ(perturbed.constraint, system.constraint);
                double const numeric = -(energyPlus - perturbed.energy()) / (2.0 * step);
                EXPECT_NEAR(rates.force[k][dim], numeric, 1.0e-6 * scale)
                    << "particle " << k << " dimension " << dim;
            }
        }
    }
}

// Pair antisymmetry: linear and angular momentum, and kinetic plus internal energy.
TEST(SphVolume, PairRatesConserveMomentumAngularMomentumAndEnergy) {
    for (int supports = 0; supports < 2; ++supports) {
        auto system =
            makeSystem({radial_kernel::Family::Spline, 3}, WeightModel::UnequalWeight, 1.4, 3);
        if (supports == 1) {
            holdSupportsAtBothBounds(system, 0.3, 0.7);
        }
        Generator generator(99);
        std::vector<Vector> velocity(system.size());
        for (auto& v : velocity) {
            v = {generator.next() - 0.5, generator.next() - 0.5, generator.next() - 0.5};
        }
        auto const rates = system.rates(velocity);
        Vector momentum{};
        Vector angularMomentum{};
        double power = 0.0;
        double scale = 0.0;
        for (int i = 0; i < system.size(); ++i) {
            Vector const torque = cross(system.position[i], rates.force[i]);
            for (int dim = 0; dim < 3; ++dim) {
                momentum[dim] += rates.force[i][dim];
                angularMomentum[dim] += torque[dim];
            }
            power += dot(velocity[i], rates.force[i]) + rates.work[i];
            scale += std::sqrt(dot(rates.force[i], rates.force[i])) + std::abs(rates.work[i]);
        }
        for (int dim = 0; dim < 3; ++dim) {
            EXPECT_NEAR(momentum[dim], 0.0, 1.0e-13 * scale);
            EXPECT_NEAR(angularMomentum[dim], 0.0, 1.0e-13 * scale);
        }
        EXPECT_NEAR(power, 0.0, 1.0e-13 * scale);
    }
}

// Each particle's work is its own adiabatic rate d(m_i u_i)/dt = P_i x_i/y_i^2 dy_i/dt
// along r + v t with supports re-solved: the discrete entropy K_i does not change.
TEST(SphVolume, WorkMatchesDensityContinuityAtFixedEntropy) {
    for (int supports = 0; supports < 2; ++supports) {
        auto system =
            makeSystem({radial_kernel::Family::Ferrers, 4}, WeightModel::UnequalWeight, 1.4, 61);
        if (supports == 1) {
            holdSupportsAtBothBounds(system, 0.3, 0.7);
        }
        Generator generator(7);
        std::vector<Vector> velocity(system.size());
        for (auto& v : velocity) {
            v = {generator.next() - 0.5, generator.next() - 0.5, generator.next() - 0.5};
        }
        auto const rates = system.rates(velocity);
        double const step = 1.0e-6;
        auto advanced = [&](double time) {
            auto moved = system;
            for (int i = 0; i < moved.size(); ++i) {
                for (int dim = 0; dim < 3; ++dim) {
                    moved.position[i][dim] += velocity[i][dim] * time;
                }
            }
            moved.solveSupports();
            EXPECT_EQ(moved.constraint, system.constraint);
            return moved;
        };
        auto const forward = advanced(step);
        auto const backward = advanced(-step);
        for (int i = 0; i < system.size(); ++i) {
            double const numericWork =
                (forward.thermalEnergy(i) - backward.thermalEnergy(i)) / (2.0 * step);
            double const densityRate =
                (forward.weightedDensity(i) - backward.weightedDensity(i)) / (2.0 * step);
            double const y = system.weightedDensity(i);
            double const continuityWork =
                system.pressure(i) * system.weight[i] / (y * y) * densityRate;
            double const scale = std::max(std::abs(rates.work[i]), system.thermalEnergy(i));
            EXPECT_NEAR(rates.work[i], numericWork, 1.0e-6 * scale) << "particle " << i;
            EXPECT_NEAR(rates.work[i], continuityWork, 1.0e-6 * scale) << "particle " << i;
        }
    }
}

// For a free support, sum_j (x_j - zeta_i) r_ij . grad_i W_ij = -3 y_i exactly, so the
// Hubble flow's pair work is the analytic homogeneous term -3 (da/dt / a) P_i V_i.
// A held support has dH/dr = 0 and no such identity; with comoving bounds its support
// dilates with the box, which the analytic term supplies (docs/variational_sph.md).
TEST(SphVolume, HubbleFlowWorkIsTheHomogeneousExpansionWork) {
    auto system =
        makeSystem({radial_kernel::Family::Spline, 3}, WeightModel::UnequalWeight, 5.0 / 3.0, 77);
    holdSupportsAtBothBounds(system, 0.3, 0.7);
    double const hubble = 0.37;
    std::vector<Vector> velocity(system.size());
    for (int i = 0; i < system.size(); ++i) {
        for (int dim = 0; dim < 3; ++dim) {
            velocity[i][dim] = hubble * system.position[i][dim];
        }
    }
    auto const rates = system.rates(velocity);
    double heldMismatch = 0.0;
    for (int i = 0; i < system.size(); ++i) {
        double const y = system.weightedDensity(i);
        double const expansionWork =
            -3.0 * hubble * system.pressure(i) * sph_volume::volume(system.weight[i], y);
        if (system.constraint[i] == SupportConstraint::Free) {
            EXPECT_NEAR(rates.work[i], expansionWork, 1.0e-12 * std::abs(expansionWork));
        } else {
            heldMismatch = std::max(heldMismatch, std::abs(rates.work[i] / expansionWork - 1.0));
        }
    }
    EXPECT_GT(heldMismatch, 1.0e-3);
    // A uniform dilation with dilating (comoving) bounds scales every y_i by (1 + e)^-3.
    double const dilation = 1.0e-6;
    auto scaled = [&](double factor) {
        auto moved = system;
        for (auto& r : moved.position) {
            for (auto& component : r) {
                component *= factor;
            }
        }
        moved.minimumSupport *= factor;
        moved.maximumSupport *= factor;
        moved.solveSupports();
        return moved;
    };
    auto const expanded = scaled(1.0 + dilation);
    auto const contracted = scaled(1.0 - dilation);
    for (int i = 0; i < system.size(); ++i) {
        double const numericWork =
            hubble * (expanded.thermalEnergy(i) - contracted.thermalEnergy(i)) / (2.0 * dilation);
        double const expansionWork = -3.0 * hubble * system.pressure(i) *
            sph_volume::volume(system.weight[i], system.weightedDensity(i));
        EXPECT_NEAR(numericWork, expansionWork, 1.0e-7 * std::abs(expansionWork));
    }
}

TEST(SphVolume, SupportClassificationAndHeldCorrections) {
    float const tolerance = 1.0e-4f;
    EXPECT_EQ(
        sph_volume::classifySupport(0.3f, 0.3f, 0.3f, 0.0f, tolerance), SupportConstraint::Fixed);
    EXPECT_EQ(sph_volume::classifySupport(0.1f, 0.1f, 0.5f, 2.0e-4f, tolerance),
        SupportConstraint::LowerBound);
    EXPECT_EQ(
        sph_volume::classifySupport(0.1f, 0.1f, 0.5f, 5.0e-5f, tolerance), SupportConstraint::Free);
    EXPECT_EQ(
        sph_volume::classifySupport(0.1f, 0.1f, 0.5f, -0.3f, tolerance), SupportConstraint::Free);
    EXPECT_EQ(sph_volume::classifySupport(0.5f, 0.1f, 0.5f, -2.0e-4f, tolerance),
        SupportConstraint::UpperBound);
    EXPECT_EQ(
        sph_volume::classifySupport(0.5f, 0.1f, 0.5f, 0.2f, tolerance), SupportConstraint::Free);
    EXPECT_EQ(
        sph_volume::classifySupport(0.3f, 0.1f, 0.5f, 0.2f, tolerance), SupportConstraint::Free);
    // A held support uses zeta = 0 even where its sums alone would be singular.
    auto const selfOnly = sph_volume::equalWeightSupportSums(2.5f, 0.0f, 0.01f);
    for (auto const held :
        {SupportConstraint::LowerBound, SupportConstraint::UpperBound, SupportConstraint::Fixed}) {
        auto const correction = sph_volume::gradHCorrection(selfOnly, held);
        EXPECT_EQ(correction.zeta, 0.0f);
        EXPECT_EQ(correction.omega, 1.0f);
        EXPECT_TRUE(sph_volume::isWellConditioned(correction));
    }
}

TEST(SphVolume, UnusableAndIllConditionedCorrectionsAreDetected) {
    float const infinity = std::numeric_limits<float>::infinity();
    float const notANumber = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(sph_volume::isUsable({notANumber, 0.0f}));
    EXPECT_FALSE(sph_volume::isUsable({1.0f, notANumber}));
    EXPECT_FALSE(sph_volume::isUsable({1.0f, infinity}));
    EXPECT_FALSE(sph_volume::isUsable({0.0f, 0.0f}));
    EXPECT_FALSE(sph_volume::isUsable({-0.2f, 0.1f}));
    EXPECT_TRUE(sph_volume::isUsable({0.05f, -0.2f}));
    EXPECT_FALSE(sph_volume::isWellConditioned({0.05f, -0.2f}));
    EXPECT_TRUE(sph_volume::isWellConditioned({0.9f, -0.01f}));
    EXPECT_TRUE(sph_volume::isFiniteValue(std::numeric_limits<float>::max()));
    EXPECT_FALSE(sph_volume::isFiniteValue(-infinity));
}

// The production loops use these helpers with SIMD packs: each lane matches the scalar.
TEST(SphVolume, SimdPacksMatchScalarAlgebra) {
    using Pack = SimdVector<double, 4>;
    auto system =
        makeSystem({radial_kernel::Family::Spline, 3}, WeightModel::UnequalWeight, 1.4, 13);
    auto packSums = sph_volume::emptySupportSums<Pack>();
    int const i = 4;
    auto const scalarSums = system.sums(i);
    for (int first = 0; first < system.size(); first += 4) {
        Pack weight(0.0);
        Pack value(0.0);
        Pack derivative(0.0);
        Pack q(0.0);
        for (int lane = 0; lane < 4; ++lane) {
            int const j = first + lane;
            Vector const separation = difference(system.position[i], system.position[j]);
            double const laneQ = std::sqrt(dot(separation, separation)) / system.support[i];
            if (laneQ < 1.0) {
                simdSetLane(weight, lane, system.weight[j]);
                simdSetLane(value, lane, radial_kernel::value(system.table, laneQ));
                simdSetLane(derivative, lane, radial_kernel::derivative(system.table, laneQ));
                simdSetLane(q, lane, laneQ);
            }
        }
        sph_volume::accumulateSupportSums(packSums, weight, value, derivative, q);
    }
    EXPECT_NEAR(simdSum(packSums.numberSum), scalarSums.numberSum, 1.0e-13 * scalarSums.numberSum);
    EXPECT_NEAR(simdSum(packSums.weightedMoment), scalarSums.weightedMoment,
        1.0e-13 * std::abs(scalarSums.weightedMoment));
    Pack const pressure([](auto lane) { return 1.0 + 0.25 * double(lane); });
    Pack const density([](auto lane) { return 2.0 - 0.125 * double(lane); });
    Pack const coefficient = variational::directedCoefficient(
        variational::pressureVolumeFactor(pressure, Pack(0.3), density), Pack(0.2), Pack(0.05));
    for (int lane = 0; lane < 4; ++lane) {
        double const scalar = variational::directedCoefficient(
            variational::pressureVolumeFactor(1.0 + 0.25 * lane, 0.3, 2.0 - 0.125 * lane), 0.2,
            0.05);
        EXPECT_NEAR(coefficient[lane], scalar, 1.0e-15);
    }
}
