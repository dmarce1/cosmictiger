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
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/variable_smoothing_lengths.hpp>
#include <cosmictiger/sph/volume.hpp>
#include <tests/sph/test_helpers.hpp>

// Double-precision references for cosmictiger/sph/volume.hpp and
// cosmictiger/sph/variable_smoothing_lengths.hpp. A small isolated particle system with its own
// exact support solve supplies the discrete thermal energy whose gradient the compatible
// density-energy force must be (docs/variational_sph.md).
namespace detail {

using Vector = Vector3<double>;

// Where a solved support sits relative to its bound layers.
enum class SupportClass { Interior, LowerLayer, UpperLayer, Fixed };

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

// W(r, h) = h^-3 w(r/h).
double kernel(radial_kernel::KernelTable<double> const& table, double radius, double support) {
    double const q = radius / support;
    return q < 1.0 ? radial_kernel::value(table, q) / (support * support * support) : 0.0;
}

struct System {
    radial_kernel::KernelTable<double> table;
    std::vector<Vector> position;
    std::vector<double> mass;
    std::vector<double> weight;
    std::vector<double> entropy;    // Pᵢ = Kᵢ rho*_i^gamma
    double gamma = 5.0 / 3.0;
    double targetSum = 0.0;    // interior target sum_j w(qᵢⱼ) = 3 N₀/(4π)
    double minimumSupport = 1.0e-3;
    double maximumSupport = 1.0e3;
    std::vector<double> support;
    std::vector<double> stiffness;    // psi of each support: one in the interior
    std::vector<SupportClass> supportClass;

    int size() const {
        return int(position.size());
    }

    double kernelSum(int i, double smoothingLength) const {
        double sum = 0.0;
        for (int j = 0; j < size(); ++j) {
            double const q = std::sqrt((position[i] - position[j]).squaredNorm()) / smoothingLength;
            sum += q < 1.0 ? radial_kernel::value(table, q) : 0.0;
        }
        return sum;
    }

    sph_volume::SupportLayers<double> layers() const {
        return sph_volume::supportLayers(minimumSupport, maximumSupport);
    }

    // The bounded target of the solve, sum_j w = targetSum f(h), and its stiffness.
    sph_volume::BoundedTarget<double> target(double smoothingLength) const {
        return sph_volume::boundedTarget(smoothingLength, minimumSupport, maximumSupport, layers());
    }

    // The support solving sum_j w(rᵢⱼ/h) = targetSum f(h) on the production solve's open
    // interval, bracketed and bisected to roundoff; the residual increases monotonically
    // with h. An iterate pinned at a limit with the bound's residual sign stays there.
    double boundedSupport(int i) const {
        auto const junctions = layers();
        double lower = minimumSupport +
            sph_volume::kSupportMargin * (junctions.lowerJunction - minimumSupport);
        double upper = maximumSupport -
            sph_volume::kSupportMargin * (maximumSupport - junctions.upperJunction);
        auto const residual = [&](double smoothingLength) {
            return kernelSum(i, smoothingLength) - targetSum * target(smoothingLength).factor;
        };
        if (residual(lower) >= 0.0) {
            return lower;
        }
        if (residual(upper) <= 0.0) {
            return upper;
        }
        for (int iteration = 0; iteration < 200 && upper - lower > 1.0e-15 * upper; ++iteration) {
            double const middle = 0.5 * (lower + upper);
            (residual(middle) < 0.0 ? lower : upper) = middle;
        }
        return 0.5 * (lower + upper);
    }

    void solveSupports() {
        support.resize(size());
        stiffness.resize(size());
        supportClass.resize(size());
        auto const junctions = layers();
        for (int i = 0; i < size(); ++i) {
            if (minimumSupport == maximumSupport) {
                support[i] = minimumSupport;
                stiffness[i] = 0.0;
                supportClass[i] = SupportClass::Fixed;
                continue;
            }
            support[i] = boundedSupport(i);
            stiffness[i] = target(support[i]).stiffness;
            supportClass[i] = support[i] < junctions.lowerJunction ?
                SupportClass::LowerLayer :
                (support[i] > junctions.upperJunction ? SupportClass::UpperLayer :
                                                        SupportClass::Interior);
        }
    }

    sph_volume::SupportSums<double> sums(int i) const {
        auto result = sph_volume::emptySupportSums<double>();
        for (int j = 0; j < size(); ++j) {
            double const q = std::sqrt((position[i] - position[j]).squaredNorm()) / support[i];
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

    // mᵢ uᵢ at the fixed entropy Kᵢ, so that d(mᵢ uᵢ) = -Pᵢ dVᵢ.
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

    double gradientFactor(double radius, double smoothingLength) const {
        double const q = radius / smoothingLength;
        return q < 1.0 ? sph_volume::kernelGradientFactor(
                             radial_kernel::derivative(table, q), 1.0 / smoothingLength, radius) :
                         0.0;
    }

    struct Rates {
        std::vector<Vector> force;    // mᵢ dvᵢ/dt
        std::vector<double> work;     // d(mᵢ uᵢ)/dt
    };

    // Hopkins (2013) equation 12 and the compatible work with the directed Qᵢⱼ.
    Rates rates(std::vector<Vector> const& velocity) const {
        std::vector<double> factor(size());
        std::vector<double> weightedSupportCorrection(size());
        for (int i = 0; i < size(); ++i) {
            factor[i] =
                variational::pressureVolumeFactor(pressure(i), weight[i], weightedDensity(i));
            weightedSupportCorrection[i] =
                sph_volume::gradHCorrection(sums(i), stiffness[i]).weightedSupportCorrection;
        }
        Rates result{std::vector<Vector>(size(), Vector{}), std::vector<double>(size(), 0.0)};
        for (int i = 0; i < size(); ++i) {
            for (int j = 0; j < size(); ++j) {
                if (i == j) {
                    continue;
                }
                Vector const separation = position[i] - position[j];
                double const radius = std::sqrt(separation.squaredNorm());
                double const gradientI = gradientFactor(radius, support[i]);
                double const gradientJ = gradientFactor(radius, support[j]);
                double const coefficientIJ = variational::directedCoefficient(
                    factor[i], weight[j], weightedSupportCorrection[i]);
                double const coefficientJI = variational::directedCoefficient(
                    factor[j], weight[i], weightedSupportCorrection[j]);
                double const pairForce = variational::pairForceFactor(
                    coefficientIJ, coefficientJI, gradientI, gradientJ);
                for (int dim = 0; dim < 3; ++dim) {
                    result.force[i][dim] -= pairForce * separation[dim];
                }
                result.work[i] += variational::pairWork(
                    coefficientIJ, gradientI, (velocity[i] - velocity[j]).dot(separation));
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

// Bounds whose layer junctions sit at the given quantiles of the free supports, so that
// the supports below the lower quantile fall into the lower layer and those above the
// upper quantile into the upper layer, while the rest stay interior. The layers are
// smooth, so finite-difference perturbations may move a support across a junction.
void boundSupportsInLayers(System& system, double lowerQuantile, double upperQuantile) {
    std::vector<double> sorted = system.support;
    std::sort(sorted.begin(), sorted.end());
    system.minimumSupport =
        sorted[std::size_t(lowerQuantile * (sorted.size() - 1))] / sph_volume::kLowerLayerRatio;
    system.maximumSupport =
        sorted[std::size_t(upperQuantile * (sorted.size() - 1))] / sph_volume::kUpperLayerRatio;
    auto const junctions = system.layers();
    detail::require(junctions.lowerJunction < junctions.upperJunction,
        "the layers of the quantile bounds overlap");
    system.solveSupports();
}

int countClass(System const& system, SupportClass supportClass) {
    return int(std::count(system.supportClass.begin(), system.supportClass.end(), supportClass));
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

// nᵢ and yᵢ, their h derivatives, Vᵢ, rho*_i, Omega_i and zeta_i with unequal weights
// and masses, compared with direct sums and finite differences in h.
TEST(SphVolume, WeightedVolumeAndConstraintDerivatives) {
    auto system =
        makeSystem({radial_kernel::Family::Wendland, 5}, WeightModel::UnequalWeight, 1.4, 17);
    for (int i = 0; i < system.size(); i += 5) {
        double const support = system.support[i];
        auto const sums = system.sums(i);
        auto const direct = [&](double smoothingLength, bool weighted) {
            double sum = 0.0;
            for (int j = 0; j < system.size(); ++j) {
                Vector const separation = system.position[i] - system.position[j];
                sum += (weighted ? system.weight[j] : 1.0) *
                    kernel(system.table, std::sqrt(separation.squaredNorm()), smoothingLength);
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
        double const supportJacobian = sph_volume::supportJacobian(sums);
        double const weightedSupportCorrection = sph_volume::weightedSupportCorrection(sums);
        EXPECT_NEAR(supportJacobian, 1.0 + support / (3.0 * n) * dndH, 1.0e-12);
        EXPECT_NEAR(weightedSupportCorrection, support / (3.0 * n * supportJacobian) * dydH,
            1.0e-12 * system.weight[i]);
        double const volume = sph_volume::volume(system.weight[i], y);
        double const volumeDensity = sph_volume::volumeDensity(system.mass[i], system.weight[i], y);
        EXPECT_NEAR(volume * volumeDensity, system.mass[i], 1.0e-15 * system.mass[i]);
        EXPECT_GT(supportJacobian, 0.0);
    }
}

// xᵢ = mᵢ with equal masses: y = m n, zeta = m (1 - 1/Omega), fᵢⱼ = 1/Omega, and the
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
        double const supportJacobian = sph_volume::supportJacobian(sums);
        double const weightedSupportCorrection = sph_volume::weightedSupportCorrection(sums);
        EXPECT_NEAR(
            weightedSupportCorrection, mass * (1.0 - 1.0 / supportJacobian), 1.0e-13 * mass);
        EXPECT_NEAR(sph_volume::correctionFactor(weightedSupportCorrection, mass),
            1.0 / supportJacobian, 1.0e-13);
        EXPECT_NEAR(mass - weightedSupportCorrection, mass / supportJacobian, 1.0e-13 * mass);
    }
    std::vector<Vector> const still(system.size(), Vector{});
    auto const rates = system.rates(still);
    for (int i = 0; i < system.size(); ++i) {
        Vector price{};
        double const omegaI = sph_volume::supportJacobian(system.sums(i));
        double const densityI = system.weightedDensity(i);
        for (int j = 0; j < system.size(); ++j) {
            if (i == j) {
                continue;
            }
            double const omegaJ = sph_volume::supportJacobian(system.sums(j));
            double const densityJ = system.weightedDensity(j);
            Vector const separation = system.position[i] - system.position[j];
            double const radius = std::sqrt(separation.squaredNorm());
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

// dyᵢ/drₖ and dhᵢ/drₖ with the support re-solved after each perturbation:
// dyᵢ = sum_j (xⱼ - zeta_i) grad_i Wᵢⱼ . drᵢⱼ and dhᵢ = -hᵢ/(3 nᵢ Omega_i) dnᵢ.
TEST(SphVolume, ConstrainedDensityDerivatives) {
    auto system =
        makeSystem({radial_kernel::Family::Spline, 4}, WeightModel::UnequalWeight, 5.0 / 3.0, 23);
    for (int i : {0, 7, 30}) {
        auto const sums = system.sums(i);
        double const weightedSupportCorrection = sph_volume::weightedSupportCorrection(sums);
        double const supportJacobian = sph_volume::supportJacobian(sums);
        double const n = sph_volume::numberDensity(sums, 1.0 / system.support[i]);
        for (int k : {i, (i + 3) % system.size(), (i + 11) % system.size()}) {
            Vector expectedDensity{};
            Vector expectedSupport{};
            for (int j = 0; j < system.size(); ++j) {
                if (j == i || (k != i && j != k)) {
                    continue;
                }
                Vector const separation = system.position[i] - system.position[j];
                double const gradient =
                    system.gradientFactor(std::sqrt(separation.squaredNorm()), system.support[i]);
                double const sign = k == i ? 1.0 : -1.0;
                for (int dim = 0; dim < 3; ++dim) {
                    expectedDensity[dim] += sign * (system.weight[j] - weightedSupportCorrection) *
                        gradient * separation[dim];
                    expectedSupport[dim] -= sign * system.support[i] / (3.0 * n * supportJacobian) *
                        gradient * separation[dim];
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

// The force is minus the gradient of sum_i mᵢ uᵢ(rho*_i) at fixed entropy, with every
// support re-solved, for interior supports, fixed supports and supports in both bound
// layers, where the stiffness-blended Omega and zeta must match the h-dependent target.
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
            EXPECT_EQ(countClass(system, SupportClass::Fixed), system.size());
        } else if (testCase.supports == 2) {
            boundSupportsInLayers(system, 0.25, 0.75);
            EXPECT_GT(countClass(system, SupportClass::LowerLayer), 5);
            EXPECT_GT(countClass(system, SupportClass::UpperLayer), 5);
            EXPECT_GT(countClass(system, SupportClass::Interior), 5);
        } else {
            EXPECT_EQ(countClass(system, SupportClass::Interior), system.size());
        }
        std::vector<Vector> const still(system.size(), Vector{});
        auto const rates = system.rates(still);
        double scale = 0.0;
        for (auto const& force : rates.force) {
            scale = std::max(scale, std::sqrt(force.squaredNorm()));
        }
        double const step = 1.0e-6;
        for (int k = 0; k < system.size(); k += 3) {
            for (int dim = 0; dim < 3; ++dim) {
                auto perturbed = system;
                perturbed.position[k][dim] += step;
                perturbed.solveSupports();
                double const energyPlus = perturbed.energy();
                perturbed.position[k][dim] -= 2.0 * step;
                perturbed.solveSupports();
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
            boundSupportsInLayers(system, 0.3, 0.7);
        }
        Generator generator(99);
        std::vector<Vector> velocity(system.size());
        for (auto& particleVelocity : velocity) {
            particleVelocity = {
                generator.next() - 0.5, generator.next() - 0.5, generator.next() - 0.5};
        }
        auto const rates = system.rates(velocity);
        Vector momentum{};
        Vector angularMomentum{};
        double power = 0.0;
        double scale = 0.0;
        for (int i = 0; i < system.size(); ++i) {
            Vector const torque = system.position[i].cross(rates.force[i]);
            for (int dim = 0; dim < 3; ++dim) {
                momentum[dim] += rates.force[i][dim];
                angularMomentum[dim] += torque[dim];
            }
            power += velocity[i].dot(rates.force[i]) + rates.work[i];
            scale += std::sqrt(rates.force[i].squaredNorm()) + std::abs(rates.work[i]);
        }
        for (int dim = 0; dim < 3; ++dim) {
            EXPECT_NEAR(momentum[dim], 0.0, 1.0e-13 * scale);
            EXPECT_NEAR(angularMomentum[dim], 0.0, 1.0e-13 * scale);
        }
        EXPECT_NEAR(power, 0.0, 1.0e-13 * scale);
    }
}

// Each particle's work is its own adiabatic rate d(mᵢ uᵢ)/dt = Pᵢ xᵢ/yᵢ^2 dyᵢ/dt
// along r + v t with supports re-solved: the discrete entropy Kᵢ does not change.
TEST(SphVolume, WorkMatchesDensityContinuityAtFixedEntropy) {
    for (int supports = 0; supports < 2; ++supports) {
        auto system =
            makeSystem({radial_kernel::Family::Ferrers, 4}, WeightModel::UnequalWeight, 1.4, 61);
        if (supports == 1) {
            boundSupportsInLayers(system, 0.3, 0.7);
        }
        Generator generator(7);
        std::vector<Vector> velocity(system.size());
        for (auto& particleVelocity : velocity) {
            particleVelocity = {
                generator.next() - 0.5, generator.next() - 0.5, generator.next() - 0.5};
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

// For an interior support, sum_j (xⱼ - zeta_i) rᵢⱼ . grad_i Wᵢⱼ = -3 yᵢ exactly, so the
// Hubble flow's pair work is the analytic homogeneous term -3 (da/dt / a) Pᵢ Vᵢ.
// A support inside a bound layer does not dilate with the flow, because its target depends
// on h against fixed bounds, so the identity fails there; with comoving bounds the layers
// dilate with the box and the analytic term is exact for every support
// (docs/variational_sph.md).
TEST(SphVolume, HubbleFlowWorkIsTheHomogeneousExpansionWork) {
    auto system =
        makeSystem({radial_kernel::Family::Spline, 3}, WeightModel::UnequalWeight, 5.0 / 3.0, 77);
    boundSupportsInLayers(system, 0.3, 0.7);
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
        if (system.supportClass[i] == SupportClass::Interior) {
            EXPECT_NEAR(rates.work[i], expansionWork, 1.0e-12 * std::abs(expansionWork));
        } else {
            heldMismatch = std::max(heldMismatch, std::abs(rates.work[i] / expansionWork - 1.0));
        }
    }
    EXPECT_GT(heldMismatch, 1.0e-3);
    // A uniform dilation with dilating (comoving) bounds scales every yᵢ by (1 + e)^-3.
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

// The bound layers: junctions, the merged case, and the piecewise target with its slope,
// its C² joins, its limits and the stiffness psi = D/(D + chi).
TEST(SphVolume, BoundedTargetLayersAndStiffness) {
    auto const wide = sph_volume::supportLayers(1.0, 1000.0);
    EXPECT_DOUBLE_EQ(wide.lowerJunction, 1.5);
    EXPECT_DOUBLE_EQ(wide.upperJunction, 800.0);
    auto const narrow = sph_volume::supportLayers(0.1, 0.2);
    EXPECT_NEAR(narrow.lowerJunction, 0.15, 1.0e-15);
    EXPECT_NEAR(narrow.upperJunction, 0.16, 1.0e-15);
    auto const merged = sph_volume::supportLayers(0.43, 0.49);
    EXPECT_NEAR(merged.lowerJunction, 0.46, 1.0e-15);
    EXPECT_NEAR(merged.upperJunction, 0.46, 1.0e-15);
    auto const fixed = sph_volume::supportLayers(0.3, 0.3);
    EXPECT_DOUBLE_EQ(fixed.lowerJunction, 0.3);
    EXPECT_DOUBLE_EQ(fixed.upperJunction, 0.3);
    auto const fixedTarget = sph_volume::boundedTarget(0.3, 0.3, 0.3, fixed);
    EXPECT_DOUBLE_EQ(fixedTarget.factor, 1.0);
    EXPECT_DOUBLE_EQ(fixedTarget.stiffness, 1.0);

    double const minimum = 1.0;
    double const maximum = 1000.0;
    auto const target = [&](double h) {
        return sph_volume::boundedTarget(h, minimum, maximum, wide);
    };
    for (double const h : {1.5, 2.0, 10.0, 100.0, 800.0}) {
        auto const interior = target(h);
        EXPECT_DOUBLE_EQ(interior.factor, 1.0);
        EXPECT_DOUBLE_EQ(interior.slope, 0.0);
        EXPECT_DOUBLE_EQ(interior.stiffness, 1.0);
    }
    // Strictly decreasing, with the analytic slope and stiffness psi = D/(D + chi),
    // chi = -h f'/f, across both layers.
    double previous = std::numeric_limits<double>::infinity();
    for (double const h : {1.0001, 1.001, 1.01, 1.1, 1.25, 1.4, 1.49, 805.0, 850.0, 900.0, 950.0,
             990.0, 999.0, 999.9}) {
        auto const value = target(h);
        EXPECT_LT(value.factor, previous) << h;
        previous = value.factor;
        EXPECT_LT(value.slope, 0.0) << h;
        double const step = 1.0e-4 *
            std::min({h - minimum, maximum - h, std::abs(h - wide.lowerJunction),
                std::abs(h - wide.upperJunction)});
        double const numeric = (target(h + step).factor - target(h - step).factor) / (2.0 * step);
        EXPECT_NEAR(value.slope, numeric, 1.0e-6 * std::abs(value.slope)) << h;
        double const chi = -h * value.slope / value.factor;
        EXPECT_NEAR(value.stiffness, 3.0 / (3.0 + chi), 1.0e-12) << h;
        EXPECT_GT(value.stiffness, 0.0);
        EXPECT_LT(value.stiffness, 1.0);
    }
    // Limits: infinite at the minimum, zero at the maximum, stiffness zero at both.
    auto const nearMinimum = target(1.0 + 1.0e-6 * 0.5);
    EXPECT_GT(nearMinimum.factor, 1.0e5);
    EXPECT_LT(nearMinimum.stiffness, 1.0e-4);
    auto const nearMaximum = target(1000.0 - 1.0e-6 * 200.0);
    EXPECT_LT(nearMaximum.factor, 1.0e-5);
    EXPECT_LT(nearMaximum.stiffness, 1.0e-4);
    // C² joins: each layer departs from one as the cube of its coordinate, so at a
    // coordinate of 0.01 the value is off by 1e-6, the slope by 3e-4/width and the second
    // derivative by 0.06/width².
    for (auto const [junction, width, sign] :
        {std::array<double, 3>{1.5, 0.5, -1.0}, std::array<double, 3>{800.0, 200.0, 1.0}}) {
        double const h = junction + sign * 0.01 * width;
        auto const value = target(h);
        EXPECT_NEAR(value.factor, 1.0, 2.0e-6);
        EXPECT_NEAR(value.slope * width, 0.0, 4.0e-4);
        double const step = 1.0e-3 * width;
        double const second =
            (target(h + step).factor - 2.0 * value.factor + target(h - step).factor) /
            (step * step);
        EXPECT_NEAR(second * width * width, 0.0, 0.1);
        EXPECT_NEAR(value.stiffness, 1.0, 1.0e-3);
    }

    // The corrections: psi = 1 is the free formula, psi = 0 the fixed-support limit, and
    // the equal-weight identity zeta = x (1 - 1/Omega) holds for every psi.
    auto const selfOnly = sph_volume::equalWeightSupportSums(2.5f, 0.0f, 0.01f);
    auto const held = sph_volume::gradHCorrection(selfOnly, 0.0f);
    EXPECT_EQ(held.weightedSupportCorrection, 0.0f);
    EXPECT_EQ(held.supportJacobian, 1.0f);
    EXPECT_TRUE(sph_volume::isWellConditioned(held));
    auto const sums = sph_volume::equalWeightSupportSums(4.0, -9.0, 0.02);
    auto const free = sph_volume::gradHCorrection(sums, 1.0);
    EXPECT_NEAR(free.supportJacobian, sph_volume::supportJacobian(sums), 1.0e-15);
    EXPECT_NEAR(
        free.weightedSupportCorrection, sph_volume::weightedSupportCorrection(sums), 1.0e-15);
    for (double const psi : {0.0, 0.2, 0.37, 0.9, 1.0}) {
        auto const blended = sph_volume::gradHCorrection(sums, psi);
        EXPECT_NEAR(
            blended.supportJacobian, psi * sph_volume::supportJacobian(sums) + 1.0 - psi, 1.0e-15);
        EXPECT_NEAR(blended.weightedSupportCorrection, 0.02 * (1.0 - 1.0 / blended.supportJacobian),
            1.0e-15);
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
            Vector const separation = system.position[i] - system.position[j];
            double const laneQ = std::sqrt(separation.squaredNorm()) / system.support[i];
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
