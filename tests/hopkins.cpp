/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/hydro/hopkins/hopkins.hpp>
#include <cosmictiger/hydro/hopkins/hopkins_simd.hpp>

namespace detail {

hopkins::Vector scaled(hopkins::Vector value, double factor) {
    for (int dim = 0; dim < 3; dim++) {
        value[dim] *= factor;
    }
    return value;
}

hopkins::Primitive boosted(hopkins::Primitive value, hopkins::Vector const& boost) {
    for (int dim = 0; dim < 3; dim++) {
        value.velocity[dim] += boost[dim];
    }
    return value;
}

struct RiemannInput {
    hopkins::Primitive left;
    hopkins::Primitive right;
    hopkins::Vector area{1.0, 0.0, 0.0};
    hopkins::Vector frame{};
};

void setPrimitiveLane(
    hopkins::SimdPrimitive& packed, size_t lane, hopkins::Primitive const& scalar) {
    packed.density[lane] = scalar.density;
    packed.pressure[lane] = scalar.pressure;
    for (int dim = 0; dim < 3; dim++) {
        packed.velocity[dim][lane] = scalar.velocity[dim];
    }
}

std::vector<hopkins::Flux> packedFluxes(std::vector<RiemannInput> const& inputs, bool finiteMass) {
    using Real = hopkins::SimdReal;
    constexpr size_t kWidth = Real::size();
    double const invalid = std::numeric_limits<double>::quiet_NaN();
    std::vector<hopkins::Flux> results(inputs.size());
    for (size_t first = 0; first < inputs.size(); first += kWidth) {
        hopkins::SimdPrimitive left{Real(invalid), {}, Real(invalid)};
        hopkins::SimdPrimitive right = left;
        hopkins::SimdVector3 area{};
        hopkins::SimdVector3 frame{};
        for (int dim = 0; dim < 3; dim++) {
            left.velocity[dim] = right.velocity[dim] = area[dim] = frame[dim] = Real(invalid);
        }
        size_t const count = std::min(kWidth, inputs.size() - first);
        for (size_t lane = 0; lane < count; lane++) {
            auto const& input = inputs[first + lane];
            setPrimitiveLane(left, lane, input.left);
            setPrimitiveLane(right, lane, input.right);
            for (int dim = 0; dim < 3; dim++) {
                area[dim][lane] = input.area[dim];
                frame[dim][lane] = input.frame[dim];
            }
        }
        auto const active = Real([](auto lane) { return double(lane); }) < Real(double(count));
        auto const packed =
            hopkins::riemannFluxValues(left, right, area, frame, 1.4, finiteMass, active);
        for (size_t lane = 0; lane < kWidth; lane++) {
            if (lane < count) {
                auto& result = results[first + lane];
                result.mass = packed.mass[lane];
                result.energy = packed.energy[lane];
                result.signalSpeed = packed.signalSpeed[lane];
                result.contactSpeed = packed.contactSpeed[lane];
                for (int dim = 0; dim < 3; dim++) {
                    result.momentum[dim] = packed.momentum[dim][lane];
                }
            } else {
                EXPECT_DOUBLE_EQ(packed.mass[lane], 0.0);
                EXPECT_DOUBLE_EQ(packed.energy[lane], 0.0);
                EXPECT_DOUBLE_EQ(packed.signalSpeed[lane], 0.0);
                EXPECT_DOUBLE_EQ(packed.contactSpeed[lane], 0.0);
                for (int dim = 0; dim < 3; dim++) {
                    EXPECT_DOUBLE_EQ(packed.momentum[dim][lane], 0.0);
                }
            }
        }
    }
    return results;
}

void expectFluxNear(
    hopkins::Flux const& actual, hopkins::Flux const& expected, double tolerance = 4.0e-13) {
    auto compare = [&](double value, double reference) {
        EXPECT_NEAR(value, reference, tolerance * std::max(1.0, std::abs(reference)));
    };
    compare(actual.mass, expected.mass);
    compare(actual.energy, expected.energy);
    compare(actual.signalSpeed, expected.signalSpeed);
    compare(actual.contactSpeed, expected.contactSpeed);
    for (int dim = 0; dim < 3; dim++) {
        compare(actual.momentum[dim], expected.momentum[dim]);
    }
}

hopkins::Flux fourRegionHllc(
    hopkins::Primitive const& left, hopkins::Primitive const& right, int& region) {
    // Independent textbook four-region construction in a stationary unit x-face.
    // In particular, do not use the production clipped-speed correction formula.
    constexpr double kGamma = 1.4;
    double const leftSound = std::sqrt(kGamma * left.pressure / left.density);
    double const rightSound = std::sqrt(kGamma * right.pressure / right.density);
    double const leftSpeed = std::min(left.velocity[0] - leftSound, right.velocity[0] - rightSound);
    double const rightSpeed =
        std::max(left.velocity[0] + leftSound, right.velocity[0] + rightSound);
    double const leftImpedance = left.density * (leftSpeed - left.velocity[0]);
    double const rightImpedance = right.density * (rightSpeed - right.velocity[0]);
    double const contact = (right.pressure - left.pressure + leftImpedance * left.velocity[0] -
                               rightImpedance * right.velocity[0]) /
        (leftImpedance - rightImpedance);
    auto conserved = [&](hopkins::Primitive const& state) {
        std::array<double, 5> result{state.density, 0.0, 0.0, 0.0, state.pressure / (kGamma - 1.0)};
        for (int dim = 0; dim < 3; dim++) {
            result[dim + 1] = state.density * state.velocity[dim];
            result[4] += 0.5 * state.density * state.velocity[dim] * state.velocity[dim];
        }
        return result;
    };
    auto physicalFlux = [&](hopkins::Primitive const& state) {
        auto result = conserved(state);
        for (double& field : result) {
            field *= state.velocity[0];
        }
        result[1] += state.pressure;
        result[4] += state.pressure * state.velocity[0];
        return result;
    };
    auto starFlux = [&](hopkins::Primitive const& state, double wave) {
        auto const original = conserved(state);
        double const density = state.density * (wave - state.velocity[0]) / (wave - contact);
        std::array<double, 5> star{density, density * contact, density * state.velocity[1],
            density * state.velocity[2],
            density *
                (original[4] / state.density +
                    (contact - state.velocity[0]) *
                        (contact + state.pressure / (state.density * (wave - state.velocity[0]))))};
        auto result = physicalFlux(state);
        for (size_t field = 0; field < result.size(); field++) {
            result[field] += wave * (star[field] - original[field]);
        }
        return result;
    };
    std::array<double, 5> flux{};
    if (leftSpeed >= 0.0) {
        region = 0;
        flux = physicalFlux(left);
    } else if (contact >= 0.0) {
        region = 1;
        flux = starFlux(left, leftSpeed);
    } else if (rightSpeed >= 0.0) {
        region = 2;
        flux = starFlux(right, rightSpeed);
    } else {
        region = 3;
        flux = physicalFlux(right);
    }
    return {flux[0], {flux[1], flux[2], flux[3]}, flux[4],
        std::max(std::abs(leftSpeed), std::abs(rightSpeed)), contact};
}

hopkins::Geometry geometryFromNormalizedMoment(double volume, hopkins::Matrix moment) {
    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
            moment[row][column] /= volume;
        }
    }
    return hopkins::makeGeometry(1.0 / volume, moment);
}

double sodDensity(double position, double time) {
    // Independent exact gamma=1.4 Sod solution; star values are the roots of
    // Toro's shock/rarefaction pressure equation, not our approximate solver.
    constexpr double kGamma = 1.4;
    constexpr double kPressure = 0.30313017805064685;
    constexpr double kVelocity = 0.92745262004894991;
    double const soundLeft = std::sqrt(kGamma);
    double const soundStar = soundLeft * std::pow(kPressure, (kGamma - 1.0) / (2.0 * kGamma));
    double const speed = (position - 0.5) / time;
    if (speed <= -soundLeft) {
        return 1.0;
    }
    if (speed < kVelocity - soundStar) {
        double const localSound = 2.0 / (kGamma + 1.0) * (soundLeft - 0.5 * (kGamma - 1.0) * speed);
        return std::pow(localSound / soundLeft, 2.0 / (kGamma - 1.0));
    }
    if (speed < kVelocity) {
        return std::pow(kPressure, 1.0 / kGamma);
    }
    double const shockSpeed = std::sqrt(kGamma * 0.1 / 0.125) *
        std::sqrt(
            (kGamma + 1.0) / (2.0 * kGamma) * kPressure / 0.1 + (kGamma - 1.0) / (2.0 * kGamma));
    if (speed < shockSpeed) {
        double const ratio = kPressure / 0.1;
        double const gammaRatio = (kGamma - 1.0) / (kGamma + 1.0);
        return 0.125 * (ratio + gammaRatio) / (gammaRatio * ratio + 1.0);
    }
    return 0.125;
}

struct SodResult {
    double densityError = 0.0;
    double totalMass = 0.0;
    double totalMomentum = 0.0;
    double totalEnergy = 0.0;
    double minimumPressure = 1.0;
};

SodResult evolveSod(int count) {
    constexpr double kGamma = 1.4;
    constexpr double kEndTime = 0.15;
    double const width = 1.0 / count;
    std::vector<hopkins::Primitive> values(count);
    for (int i = 0; i < count; i++) {
        values[i] =
            i < count / 2 ? hopkins::Primitive{1.0, {}, 1.0} : hopkins::Primitive{0.125, {}, 0.1};
    }
    double time = 0.0;
    while (time < kEndTime) {
        double speed = 0.0;
        for (auto const& value : values) {
            speed = std::max(speed,
                std::abs(value.velocity[0]) + std::sqrt(kGamma * value.pressure / value.density));
        }
        double const timestep = std::min(0.2 * width / speed, kEndTime - time);
        std::vector<hopkins::Gradient> gradients(count);
        auto limited = [](double left, double center, double right, double spacing) {
            double const slope = (right - left) / (2.0 * spacing);
            double const limiter = hopkins::gradientLimiter(center, std::min({left, center, right}),
                std::max({left, center, right}), center - 0.5 * spacing * std::abs(slope),
                center + 0.5 * spacing * std::abs(slope), 1.0);
            return slope * limiter;
        };
        for (int i = 1; i < count - 1; i++) {
            gradients[i].density[0] =
                limited(values[i - 1].density, values[i].density, values[i + 1].density, width);
            gradients[i].pressure[0] =
                limited(values[i - 1].pressure, values[i].pressure, values[i + 1].pressure, width);
            gradients[i].velocity[0][0] = limited(
                values[i - 1].velocity[0], values[i].velocity[0], values[i + 1].velocity[0], width);
        }
        std::vector<hopkins::Flux> fluxes(count + 1);
        for (int face = 0; face <= count; face++) {
            int const i = std::max(0, face - 1);
            int const j = std::min(count - 1, face);
            auto const left = hopkins::reconstruct(values[i], gradients[i], {0.5 * width, 0, 0}, {},
                0.5 * timestep, kGamma, &values[j]);
            auto const right = hopkins::reconstruct(values[j], gradients[j], {-0.5 * width, 0, 0},
                {}, 0.5 * timestep, kGamma, &values[i]);
            fluxes[face] = hopkins::riemannFlux(left, right, {1, 0, 0}, {}, kGamma, false);
        }
        for (int i = 0; i < count; i++) {
            double const mass =
                values[i].density - timestep / width * (fluxes[i + 1].mass - fluxes[i].mass);
            double const momentum = values[i].density * values[i].velocity[0] -
                timestep / width * (fluxes[i + 1].momentum[0] - fluxes[i].momentum[0]);
            double const energy = values[i].pressure / (kGamma - 1.0) +
                0.5 * values[i].density * values[i].velocity[0] * values[i].velocity[0] -
                timestep / width * (fluxes[i + 1].energy - fluxes[i].energy);
            values[i] = {mass, {momentum / mass, 0.0, 0.0},
                (kGamma - 1.0) * (energy - 0.5 * momentum * momentum / mass)};
            if (!hopkins::physical(values[i])) {
                throw std::runtime_error("Nonphysical state in independent Sod integration");
            }
        }
        time += timestep;
    }
    SodResult result;
    for (int i = 0; i < count; i++) {
        auto const& value = values[i];
        result.densityError +=
            width * std::abs(value.density - sodDensity((i + 0.5) * width, kEndTime));
        result.totalMass += width * value.density;
        result.totalMomentum += width * value.density * value.velocity[0];
        result.totalEnergy += width *
            (value.pressure / (kGamma - 1.0) +
                0.5 * value.density * value.velocity[0] * value.velocity[0]);
        result.minimumPressure = std::min(result.minimumPressure, value.pressure);
    }
    return result;
}

}    // namespace detail

TEST(Hopkins, SupportRadiusStepConvergesToMonotoneRoot) {
    constexpr double kMinimum = 0.1;
    constexpr double kMaximum = 4.0;
    double lower = kMinimum, upper = kMaximum, radius = 0.35;
    for (int iteration = 0; iteration < 64; iteration++) {
        double const residual = radius * radius * radius - 8.0;
        if (std::abs(residual) < 1.0e-13) {
            break;
        }
        radius = hopkins::supportRadiusStep(
            radius, residual, 3.0 * radius * radius, kMinimum, kMaximum, lower, upper);
        EXPECT_LE(lower, 2.0);
        EXPECT_GE(upper, 2.0);
        EXPECT_GE(radius, lower);
        EXPECT_LE(radius, upper);
    }
    EXPECT_NEAR(radius, 2.0, 1.0e-13);
}

TEST(Hopkins, SupportRadiusStepShrinksWithPositiveResidualAndFlatDerivative) {
    double lower = 0.1, upper = 4.0;
    double const radius = hopkins::supportRadiusStep(1.0, 3.0, 0.0, 0.1, 4.0, lower, upper);
    EXPECT_LT(radius, 1.0);
    EXPECT_GE(radius, lower);
    EXPECT_LE(radius, upper);
    EXPECT_DOUBLE_EQ(upper, 1.0);
    lower = 0.1;
    upper = 4.0;
    EXPECT_GT(hopkins::supportRadiusStep(1.0, -3.0, 0.0, 0.1, 4.0, lower, upper), 1.0);
}

TEST(Hopkins, SupportRadiusStepRetainsBracketAndEscapesStagnation) {
    double lower = 0.9, upper = 1.1;
    double const overshoot = hopkins::supportRadiusStep(1.0, -0.5, 1.0e-12, 0.1, 4.0, lower, upper);
    EXPECT_GT(overshoot, 1.0);
    EXPECT_LT(overshoot, 1.1);
    EXPECT_DOUBLE_EQ(lower, 1.0);
    lower = 0.9;
    upper = 1.1;
    double const stalled = hopkins::supportRadiusStep(1.0, 0.5, 1.0e30, 0.1, 4.0, lower, upper);
    EXPECT_LT(stalled, 1.0);
    EXPECT_GT(stalled, 0.9);
    EXPECT_DOUBLE_EQ(upper, 1.0);
    EXPECT_DOUBLE_EQ(
        hopkins::supportRadiusStep(stalled, 0.0, 1.0, 0.1, 4.0, lower, upper), stalled);
}

TEST(Hopkins, FloatSupportRadiusTerminatesForAnUnrepresentableRoot) {
    // A monotone function whose Newton iterates cross a root lying halfway
    // between adjacent floats. Rounded Newton steps alone alternate forever.
    double const left = 0.015625;
    double const right = std::nextafter(float(left), std::numeric_limits<float>::infinity());
    double const middle = 0.5 * (left + right);
    double const halfWidth = 0.5 * (right - left);
    for (double start : {left, right}) {
        double lower = left, upper = right, radius = start;
        for (int iteration = 0; iteration < 4; iteration++) {
            double const x = (radius - middle) / halfWidth;
            double const residual = x - x * x * x / 8.0;
            double const derivative = (1.0 - 3.0 * x * x / 8.0) / halfWidth;
            double const next = hopkins::supportRadiusStep(
                radius, residual, derivative, left, right, lower, upper, true);
            EXPECT_DOUBLE_EQ(next, start);
            EXPECT_LE(std::abs(next - middle), halfWidth);
            radius = next;
        }
    }
}

TEST(Hopkins, OneDimensionalCubicLatticeHasUnitFacesAndExactLinearGradient) {
    // Normalized cubic spline at h=2*dx: W(0)=2/(3*dx), W(dx)=1/(6*dx).
    double const spacing = 0.125;
    hopkins::Matrix moment{};
    moment[0][0] = spacing / 3.0;
    auto const geometry = hopkins::makeGeometry(1.0 / spacing, moment, 100.0, 1);
    ASSERT_TRUE(geometry.valid);
    EXPECT_DOUBLE_EQ(geometry.volume, spacing);
    EXPECT_NEAR(geometry.conditionNumber, 1.0, 1.0e-14);
    EXPECT_DOUBLE_EQ(geometry.inverseMoment[1][1], 0.0);
    EXPECT_DOUBLE_EQ(geometry.inverseMoment[2][2], 0.0);
    auto const weight =
        hopkins::gradientWeight(geometry, {spacing, 0.0, 0.0}, 1.0 / (6.0 * spacing));
    EXPECT_NEAR(2.0 * weight[0] * spacing * 3.5, 3.5, 1.0e-14);
    auto const area = hopkins::faceArea(
        geometry, geometry, {spacing, 0.0, 0.0}, 1.0 / (6.0 * spacing), 1.0 / (6.0 * spacing));
    EXPECT_NEAR(area[0], 1.0, 1.0e-14);
    EXPECT_DOUBLE_EQ(area[1], 0.0);
    EXPECT_DOUBLE_EQ(area[2], 0.0);
    EXPECT_FALSE(hopkins::makeGeometry(1.0 / spacing, moment).valid);
}

TEST(Hopkins, MatrixGradientIsLinearExactOnIrregularCloud) {
    std::array<hopkins::Vector, 7> const positions{{{0.3, -0.2, 0.4}, {-0.5, 0.1, 0.2},
        {0.1, 0.7, -0.3}, {-0.2, -0.3, -0.4}, {0.6, 0.3, 0.1}, {0.2, 0.1, -0.7}, {-0.1, 0.4, 0.5}}};
    hopkins::Matrix moment{};
    double kernelSum = 2.3;
    std::array<double, 7> weights{};
    for (size_t i = 0; i < positions.size(); i++) {
        weights[i] = std::exp(-hopkins::dot(positions[i], positions[i]));
        kernelSum += weights[i];
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 3; column++) {
                moment[row][column] += weights[i] * positions[i][row] * positions[i][column];
            }
        }
    }
    auto const geometry = hopkins::makeGeometry(kernelSum, moment);
    ASSERT_TRUE(geometry.valid);
    EXPECT_DOUBLE_EQ(geometry.volume, 1.0 / kernelSum);
    hopkins::Vector const expected{1.3, -2.1, 0.71};
    hopkins::Vector measured{};
    for (size_t i = 0; i < positions.size(); i++) {
        double const difference = hopkins::dot(expected, positions[i]);
        auto const weight = hopkins::gradientWeight(geometry, positions[i], weights[i]);
        for (int dim = 0; dim < 3; dim++) {
            measured[dim] += difference * weight[dim];
        }
    }
    for (int dim = 0; dim < 3; dim++) {
        EXPECT_NEAR(measured[dim], expected[dim], 2.0e-14);
    }
}

TEST(Hopkins, SingularAndPoorlyConditionedGeometryRequiresMoreNeighbors) {
    hopkins::Matrix moment{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    auto geometry = hopkins::makeGeometry(2.0, moment);
    ASSERT_TRUE(geometry.valid);
    EXPECT_NEAR(geometry.conditionNumber, 1.0, 1.0e-15);
    moment[2][2] = 1.0e-8;
    geometry = hopkins::makeGeometry(2.0, moment);
    EXPECT_FALSE(geometry.valid);
    EXPECT_GT(geometry.conditionNumber, 100.0);
    moment[2][2] = 0.0;
    EXPECT_FALSE(hopkins::makeGeometry(2.0, moment).valid);
    EXPECT_FALSE(hopkins::makeGeometry(0.0, moment).valid);
}

TEST(Hopkins, FaceAreaIsAntisymmetricForUnequalVolumes) {
    hopkins::Matrix const momentLeft{{{2, 0.1, -0.2}, {0.1, 3, 0.4}, {-0.2, 0.4, 1.0}}};
    hopkins::Matrix const momentRight{{{3, 0.2, 0.1}, {0.2, 1, 0.0}, {0.1, 0.0, 2.0}}};
    auto const left = hopkins::makeGeometry(7.0, momentLeft);
    auto const right = hopkins::makeGeometry(11.0, momentRight);
    hopkins::Vector const separation{0.2, -0.4, 0.1};
    auto const forward = hopkins::faceArea(left, right, separation, 0.8, 0.4);
    auto const reverse = hopkins::faceArea(right, left, detail::scaled(separation, -1.0), 0.4, 0.8);
    for (int dim = 0; dim < 3; dim++) {
        EXPECT_DOUBLE_EQ(forward[dim], -reverse[dim]);
    }
    EXPECT_GT(hopkins::dot(forward, separation), 0.0);
}

TEST(Hopkins, GeometryRateVanishesUnderUniformTranslation) {
    hopkins::Matrix const moment{{{2.0, 0.1, -0.2}, {0.1, 3.0, 0.4}, {-0.2, 0.4, 1.0}}};
    auto const geometry = detail::geometryFromNormalizedMoment(0.7, moment);
    ASSERT_TRUE(geometry.valid);
    auto const rate = hopkins::geometryRate(geometry, 0.0, {});
    hopkins::Vector const separation{0.2, -0.4, 0.1};
    hopkins::Vector const translation{1.5, -0.3, 2.7};
    hopkins::Vector separationRate{};
    for (int dim = 0; dim < 3; dim++) {
        separationRate[dim] = translation[dim] - translation[dim];
        for (int column = 0; column < 3; column++) {
            EXPECT_DOUBLE_EQ(rate.inverseMoment[dim][column], 0.0);
        }
    }
    auto const areaRate = hopkins::faceAreaRate(
        geometry, geometry, rate, rate, separation, separationRate, 0.8, 0.8, 0.0, 0.0);
    EXPECT_DOUBLE_EQ(rate.volume, 0.0);
    for (double component : areaRate) {
        EXPECT_DOUBLE_EQ(component, 0.0);
    }
}

TEST(Hopkins, FaceAreaRateTracksHomologousDilation) {
    constexpr double kExpansion = 0.37;
    hopkins::Matrix const moment{{{2.0, 0.1, -0.2}, {0.1, 3.0, 0.4}, {-0.2, 0.4, 1.0}}};
    auto const left = detail::geometryFromNormalizedMoment(0.7, moment);
    auto const right = detail::geometryFromNormalizedMoment(1.3, moment);
    ASSERT_TRUE(left.valid);
    ASSERT_TRUE(right.valid);
    hopkins::Matrix momentRate{};
    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
            momentRate[row][column] = 2.0 * kExpansion * moment[row][column];
        }
    }
    auto const leftRate = hopkins::geometryRate(left, 3.0 * kExpansion * left.volume, momentRate);
    auto const rightRate =
        hopkins::geometryRate(right, 3.0 * kExpansion * right.volume, momentRate);
    hopkins::Vector const separation{0.2, -0.4, 0.1};
    auto const area = hopkins::faceArea(left, right, separation, 0.8, 0.4);
    auto const areaRate = hopkins::faceAreaRate(left, right, leftRate, rightRate, separation,
        detail::scaled(separation, kExpansion), 0.8, 0.4, -3.0 * kExpansion * 0.8,
        -3.0 * kExpansion * 0.4);
    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
            EXPECT_NEAR(leftRate.inverseMoment[row][column],
                -2.0 * kExpansion * left.inverseMoment[row][column], 1.0e-14);
        }
        EXPECT_NEAR(areaRate[row], 2.0 * kExpansion * area[row], 1.0e-14);
    }
}

TEST(Hopkins, GeometryAndFaceAreaRatesMatchIndependentFiniteDifferences) {
    constexpr double kStep = 1.0e-5;
    hopkins::Matrix const leftMoment{{{2.0, 0.1, -0.2}, {0.1, 3.0, 0.4}, {-0.2, 0.4, 1.0}}};
    hopkins::Matrix const rightMoment{{{3.0, 0.2, 0.1}, {0.2, 1.0, 0.0}, {0.1, 0.0, 2.0}}};
    hopkins::Matrix const leftMomentRate{{{0.2, -0.1, 0.3}, {-0.1, -0.4, 0.2}, {0.3, 0.2, 0.1}}};
    hopkins::Matrix const rightMomentRate{{{-0.3, 0.1, 0.2}, {0.1, 0.2, -0.3}, {0.2, -0.3, 0.4}}};
    auto const shiftedGeometry = [](double volume, double volumeRate, hopkins::Matrix moment,
                                     hopkins::Matrix const& momentRate, double time) {
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 3; column++) {
                moment[row][column] += time * momentRate[row][column];
            }
        }
        return detail::geometryFromNormalizedMoment(volume + time * volumeRate, moment);
    };
    auto const left = shiftedGeometry(0.7, 0.13, leftMoment, leftMomentRate, 0.0);
    auto const right = shiftedGeometry(1.3, -0.21, rightMoment, rightMomentRate, 0.0);
    auto const leftRate = hopkins::geometryRate(left, 0.13, leftMomentRate);
    auto const rightRate = hopkins::geometryRate(right, -0.21, rightMomentRate);
    auto const leftBefore = shiftedGeometry(0.7, 0.13, leftMoment, leftMomentRate, -kStep);
    auto const leftAfter = shiftedGeometry(0.7, 0.13, leftMoment, leftMomentRate, kStep);
    auto const rightBefore = shiftedGeometry(1.3, -0.21, rightMoment, rightMomentRate, -kStep);
    auto const rightAfter = shiftedGeometry(1.3, -0.21, rightMoment, rightMomentRate, kStep);
    ASSERT_TRUE(leftBefore.valid && leftAfter.valid && rightBefore.valid && rightAfter.valid);
    hopkins::Vector const separation{0.2, -0.4, 0.1};
    hopkins::Vector const separationRate{-0.1, 0.2, 0.3};
    hopkins::Vector separationBefore{}, separationAfter{};
    for (int dim = 0; dim < 3; dim++) {
        separationBefore[dim] = separation[dim] - kStep * separationRate[dim];
        separationAfter[dim] = separation[dim] + kStep * separationRate[dim];
    }
    auto const areaBefore = hopkins::faceArea(
        leftBefore, rightBefore, separationBefore, 0.8 - kStep * 0.17, 0.4 + kStep * 0.11);
    auto const areaAfter = hopkins::faceArea(
        leftAfter, rightAfter, separationAfter, 0.8 + kStep * 0.17, 0.4 - kStep * 0.11);
    auto const areaRate = hopkins::faceAreaRate(
        left, right, leftRate, rightRate, separation, separationRate, 0.8, 0.4, 0.17, -0.11);
    auto const reversedRate =
        hopkins::faceAreaRate(right, left, rightRate, leftRate, detail::scaled(separation, -1.0),
            detail::scaled(separationRate, -1.0), 0.4, 0.8, -0.11, 0.17);
    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
            EXPECT_NEAR(leftRate.inverseMoment[row][column],
                (leftAfter.inverseMoment[row][column] - leftBefore.inverseMoment[row][column]) /
                    (2.0 * kStep),
                1.0e-10);
            EXPECT_NEAR(rightRate.inverseMoment[row][column],
                (rightAfter.inverseMoment[row][column] - rightBefore.inverseMoment[row][column]) /
                    (2.0 * kStep),
                1.0e-10);
        }
        EXPECT_NEAR(areaRate[row], (areaAfter[row] - areaBefore[row]) / (2.0 * kStep), 1.0e-10);
        EXPECT_DOUBLE_EQ(reversedRate[row], -areaRate[row]);
    }
}

TEST(Hopkins, PairLimiterPreservesLinearFieldsAndPositiveStates) {
    EXPECT_DOUBLE_EQ(hopkins::pairLimiter(1.0, 3.0, 1.8, 0.4), 1.8);
    EXPECT_DOUBLE_EQ(hopkins::pairLimiter(3.0, 1.0, 1.8, 0.6), 1.8);
    EXPECT_GT(hopkins::pairLimiter(0.01, 10.0, -5.0, 0.5), 0.0);
    EXPECT_LT(hopkins::pairLimiter(-10.0, -0.01, 5.0, 0.5), 0.0);
    EXPECT_DOUBLE_EQ(hopkins::pairLimiter(2.0, 2.0, 5.0, 0.5), 2.0);
    EXPECT_DOUBLE_EQ(hopkins::gradientLimiter(1.0, 0.0, 2.0, 0.5, 1.5, 1.0), 1.0);
}

TEST(Hopkins, HancockPredictorUsesMovingFrameEulerDerivatives) {
    hopkins::Primitive const center{2.0, {3.0, 2.0, -1.0}, 4.0};
    hopkins::Gradient gradient;
    gradient.density = {1.0, -0.4, 0.2};
    gradient.pressure = {0.3, 0.5, -0.1};
    gradient.velocity = {{{0.2, 0.1, 0.3}, {-0.1, 0.4, 0.2}, {0.2, -0.3, -0.1}}};
    hopkins::Vector const offset{0.1, -0.2, 0.3};
    hopkins::Vector const frame{2.0, 1.0, -1.0};
    auto const result = hopkins::reconstruct(center, gradient, offset, frame, 0.02, 1.4);
    EXPECT_NEAR(result.density, 2.24 - 0.02 * (0.6 + 1.0), 1.0e-14);
    EXPECT_NEAR(result.pressure, 3.9 - 0.02 * (0.8 + 2.8), 1.0e-14);
    auto const boost = hopkins::Vector{21.0, -10.0, 7.0};
    hopkins::Vector boostedFrame = frame;
    for (int dim = 0; dim < 3; dim++) {
        boostedFrame[dim] += boost[dim];
    }
    auto const moving = hopkins::reconstruct(
        detail::boosted(center, boost), gradient, offset, boostedFrame, 0.02, 1.4);
    EXPECT_NEAR(moving.density, result.density, 1.0e-14);
    EXPECT_NEAR(moving.pressure, result.pressure, 1.0e-14);
    for (int dim = 0; dim < 3; dim++) {
        EXPECT_NEAR(moving.velocity[dim], result.velocity[dim] + boost[dim], 1.0e-14);
    }
}

TEST(Hopkins, MovingContactHasNoMassOrThermalDiffusion) {
    hopkins::Primitive const left{1.0, {2.0, -1.0, 0.5}, 3.0};
    hopkins::Primitive const right{8.0, left.velocity, left.pressure};
    hopkins::Vector const area{0.2, -0.3, 0.4};
    for (bool const finiteMass : {false, true}) {
        auto const flux =
            hopkins::riemannFlux(left, right, area, left.velocity, 5.0 / 3.0, finiteMass);
        EXPECT_DOUBLE_EQ(flux.mass, 0.0);
        for (int dim = 0; dim < 3; dim++) {
            EXPECT_NEAR(flux.momentum[dim], left.pressure * area[dim], 1.0e-14);
        }
        EXPECT_NEAR(flux.energy, left.pressure * hopkins::dot(left.velocity, area), 1.0e-14);
    }
}

TEST(Hopkins, MfmEliminatesMassFluxWhileMfvTransfersMass) {
    hopkins::Primitive const left{1.0, {}, 1.0};
    hopkins::Primitive const right{0.125, {}, 0.1};
    auto const mfm = hopkins::riemannFlux(left, right, {1, 0, 0}, {}, 1.4, true);
    auto const mfv = hopkins::riemannFlux(left, right, {1, 0, 0}, {}, 1.4, false);
    EXPECT_DOUBLE_EQ(mfm.mass, 0.0);
    EXPECT_GT(mfv.mass, 0.0);
    EXPECT_GT(mfm.momentum[0], 0.0);
    EXPECT_GT(mfm.energy, 0.0);
    EXPECT_GT(mfm.contactSpeed, 0.0);
}

TEST(Hopkins, MfvMassFluxWorkCompensatesUniformPotentialGradient) {
    hopkins::Vector const leftPosition{0.2, 0.3, 0.5};
    hopkins::Vector const rightPosition{0.4, 0.1, 0.6};
    hopkins::Vector const acceleration{0.7, -1.2, 0.4};
    hopkins::Vector separation{};
    for (int dim = 0; dim < 3; dim++) {
        separation[dim] = rightPosition[dim] - leftPosition[dim];
    }
    double const potentialLeft = -hopkins::dot(acceleration, leftPosition);
    double const potentialRight = -hopkins::dot(acceleration, rightPosition);
    constexpr double kTimestep = 0.03;
    for (double const massFlux : {-0.4, 0.0, 0.4}) {
        auto const work =
            hopkins::pairGravityWork(massFlux, separation, acceleration, acceleration);
        double const potentialBefore = 1.3 * potentialLeft + 0.7 * potentialRight;
        double const potentialAfter = (1.3 - kTimestep * massFlux) * potentialLeft +
            (0.7 + kTimestep * massFlux) * potentialRight;
        double const energyChange = kTimestep * (work.leftEnergyRate + work.rightEnergyRate);
        EXPECT_NEAR(potentialAfter - potentialBefore + energyChange, 0.0, 1.0e-15);
        auto const reverse = hopkins::pairGravityWork(
            -massFlux, detail::scaled(separation, -1.0), acceleration, acceleration);
        EXPECT_DOUBLE_EQ(reverse.leftEnergyRate, work.rightEnergyRate);
        EXPECT_DOUBLE_EQ(reverse.rightEnergyRate, work.leftEnergyRate);
        if (massFlux == 0.0) {
            EXPECT_DOUBLE_EQ(work.leftEnergyRate, 0.0);
            EXPECT_DOUBLE_EQ(work.rightEnergyRate, 0.0);
        }
    }
}

TEST(Hopkins, MfvMassFluxWorkIsExactForLinearAcceleration) {
    // phi = (2*x^2 + 3*y^2 + 4*z^2)/2, with a=-grad(phi).
    hopkins::Vector const leftPosition{0.1, 0.2, 0.3};
    hopkins::Vector const rightPosition{0.4, 0.15, 0.5};
    hopkins::Vector separation{};
    hopkins::Vector leftAcceleration{};
    hopkins::Vector rightAcceleration{};
    double potentialDifference = 0.0;
    for (int dim = 0; dim < 3; dim++) {
        double const coefficient = dim + 2.0;
        separation[dim] = rightPosition[dim] - leftPosition[dim];
        leftAcceleration[dim] = -coefficient * leftPosition[dim];
        rightAcceleration[dim] = -coefficient * rightPosition[dim];
        potentialDifference += 0.5 * coefficient *
            (rightPosition[dim] * rightPosition[dim] - leftPosition[dim] * leftPosition[dim]);
    }
    constexpr double kMassFlux = 0.17;
    auto const work =
        hopkins::pairGravityWork(kMassFlux, separation, leftAcceleration, rightAcceleration);
    EXPECT_NEAR(
        work.leftEnergyRate + work.rightEnergyRate + kMassFlux * potentialDifference, 0.0, 1.0e-15);
    auto const reverse = hopkins::pairGravityWork(
        -kMassFlux, detail::scaled(separation, -1.0), rightAcceleration, leftAcceleration);
    EXPECT_DOUBLE_EQ(reverse.leftEnergyRate, work.rightEnergyRate);
    EXPECT_DOUBLE_EQ(reverse.rightEnergyRate, work.leftEnergyRate);
}

TEST(Hopkins, CachedMassFluxWorkRecoversTimeDependentPotentialTransport) {
    // phi(x,t)=sum_d (c_d+s_d*t)*x_d^2/2. The independently integrated
    // transport energy is -integral F(t)*(phi(right,t)-phi(left,t)) dt.
    // This excludes partial(phi)/partial(t), which is not mass-transport work.
    std::array<hopkins::Vector, 2> const positions{
        hopkins::Vector{0.1, 0.2, 0.3}, hopkins::Vector{0.4, 0.15, 0.5}};
    hopkins::Vector const curvature{2.0, 3.0, 4.0};
    hopkins::Vector const curvatureRate{0.7, -0.4, 0.2};
    hopkins::Vector separation{};
    double potentialDifference = 0.0;
    double potentialDifferenceRate = 0.0;
    for (int dim = 0; dim < 3; dim++) {
        separation[dim] = positions[1][dim] - positions[0][dim];
        double const squareDifference =
            positions[1][dim] * positions[1][dim] - positions[0][dim] * positions[0][dim];
        potentialDifference += 0.5 * curvature[dim] * squareDifference;
        potentialDifferenceRate += 0.5 * curvatureRate[dim] * squareDifference;
    }
    constexpr double kInterval = 0.173;
    constexpr double kMassFlux = 0.17;
    constexpr double kMassFluxRate = -0.23;
    double const expected = -(kMassFlux * potentialDifference * kInterval +
        0.5 * (kMassFlux * potentialDifferenceRate + kMassFluxRate * potentialDifference) *
            kInterval * kInterval +
        kMassFluxRate * potentialDifferenceRate * kInterval * kInterval * kInterval / 3.0);
    auto const initial = hopkins::pairGravityWorkVector(kMassFlux, separation);
    auto const midpoint =
        hopkins::pairGravityWorkVector(kMassFlux + 0.5 * kInterval * kMassFluxRate, separation);
    // Test a complete raw global interval and a cached traditional interval
    // consumed in unequal pieces, with different source anchors at its ends.
    for (auto const& boundaries :
        {std::vector<double>{0.0, kInterval}, std::vector<double>{0.0, 0.029, 0.101, kInterval}}) {
        for (auto const& sourceTimes :
            {std::array<double, 2>{0.0, 0.0}, std::array<double, 2>{-0.11, -0.04}}) {
            double work = 0.0;
            std::array<hopkins::Vector, 2> pending{};
            for (size_t step = 1; step < boundaries.size(); step++) {
                double const start = boundaries[step - 1];
                auto const integrated = hopkins::integratePairGravityWork(
                    initial, midpoint, kInterval, start, boundaries[step] - start);
                for (size_t endpoint = 0; endpoint < positions.size(); endpoint++) {
                    for (int dim = 0; dim < 3; dim++) {
                        double const acceleration =
                            -(curvature[dim] + curvatureRate[dim] * sourceTimes[endpoint]) *
                            positions[endpoint][dim];
                        work += integrated.integral[dim] * acceleration;
                        pending[endpoint][dim] +=
                            (start - sourceTimes[endpoint]) * integrated.integral[dim] +
                            integrated.firstMoment[dim];
                    }
                }
            }
            for (size_t endpoint = 0; endpoint < positions.size(); endpoint++) {
                for (int dim = 0; dim < 3; dim++) {
                    double const accelerationChange = -curvatureRate[dim] *
                        (kInterval - sourceTimes[endpoint]) * positions[endpoint][dim];
                    work += pending[endpoint][dim] * accelerationChange /
                        (kInterval - sourceTimes[endpoint]);
                }
            }
            EXPECT_NEAR(work, expected, 2.0e-17);
        }
    }
}

TEST(Hopkins, FluxIsPairAntisymmetricAndGalileanCovariant) {
    hopkins::Primitive const left{0.8, {0.7, -0.4, 0.3}, 0.9};
    hopkins::Primitive const right{1.3, {-0.2, 0.8, -0.1}, 0.4};
    hopkins::Vector const frame{0.3, 0.2, 0.1};
    hopkins::Vector const area{0.3, -0.1, 0.2};
    hopkins::Vector const boost{40.0, -20.0, 10.0};
    hopkins::Vector boostedFrame = frame;
    for (int dim = 0; dim < 3; dim++) {
        boostedFrame[dim] += boost[dim];
    }
    for (bool const finiteMass : {false, true}) {
        auto const forward = hopkins::riemannFlux(left, right, area, frame, 1.4, finiteMass);
        auto const reverse =
            hopkins::riemannFlux(right, left, detail::scaled(area, -1.0), frame, 1.4, finiteMass);
        EXPECT_NEAR(forward.mass, -reverse.mass, 2.0e-15);
        EXPECT_NEAR(forward.energy, -reverse.energy, 2.0e-15);
        auto const shifted = hopkins::riemannFlux(detail::boosted(left, boost),
            detail::boosted(right, boost), area, boostedFrame, 1.4, finiteMass);
        EXPECT_NEAR(shifted.mass, forward.mass, 1.0e-14);
        for (int dim = 0; dim < 3; dim++) {
            EXPECT_NEAR(forward.momentum[dim], -reverse.momentum[dim], 2.0e-15);
            EXPECT_NEAR(
                shifted.momentum[dim], forward.momentum[dim] + boost[dim] * forward.mass, 2.0e-12);
        }
        EXPECT_NEAR(shifted.energy,
            forward.energy + hopkins::dot(boost, forward.momentum) +
                0.5 * hopkins::dot(boost, boost) * forward.mass,
            2.0e-11);
    }
}

TEST(Hopkins, ExtremeRarefactionFailsWithoutRiemannFallback) {
    hopkins::Primitive const left{1.0, {-8.0, 0.0, 0.0}, 1.0e-5};
    hopkins::Primitive const right{0.2, {8.0, 0.0, 0.0}, 1.0e-6};
    for (bool const finiteMass : {false, true}) {
        EXPECT_THROW(
            hopkins::riemannFlux(left, right, {1, 0, 0}, {}, 1.4, finiteMass), std::runtime_error);
    }
    auto invalid = left;
    invalid.pressure = -1.0;
    EXPECT_THROW(
        hopkins::riemannFlux(invalid, right, {1, 0, 0}, {}, 1.4, true), std::invalid_argument);
}

TEST(HopkinsSimd, ClippedSpeedsMatchIndependentFourRegionHllc) {
    hopkins::Primitive const sodLeft{1.0, {0.0, 0.2, -0.1}, 1.0};
    hopkins::Primitive const sodRight{0.125, {0.0, -0.3, 0.4}, 0.1};
    std::vector<detail::RiemannInput> const cases{
        {detail::boosted(sodLeft, {4.0, 0.0, 0.0}), detail::boosted(sodRight, {4.0, 0.0, 0.0})},
        {sodLeft, sodRight}, {sodRight, sodLeft},
        {detail::boosted(sodLeft, {-4.0, 0.0, 0.0}), detail::boosted(sodRight, {-4.0, 0.0, 0.0})},
        {{1.0, {}, 1.0}, {8.0, {}, 1.0}},
        {{1.0, {-0.2, 0.1, 0.0}, 1.0}, {1.0, {0.2, -0.1, 0.0}, 1.0}},
        {{0.9, {0.5, 0.2, 0.1}, 0.8}, {1.1, {-0.3, -0.2, 0.3}, 1.2}}};
    auto const packed = detail::packedFluxes(cases, false);
    std::array<bool, 4> seen{};
    for (size_t i = 0; i < cases.size(); i++) {
        SCOPED_TRACE(i);
        int region = -1;
        auto const reference = detail::fourRegionHllc(cases[i].left, cases[i].right, region);
        seen[region] = true;
        auto const scalar = hopkins::riemannFlux(
            cases[i].left, cases[i].right, cases[i].area, cases[i].frame, 1.4, false);
        detail::expectFluxNear(scalar, reference);
        detail::expectFluxNear(packed[i], reference);
    }
    EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }));
}

TEST(HopkinsSimd, UniformAndMixedContactMasksMatchScalar) {
    constexpr size_t kWidth = hopkins::SimdReal::size();
    detail::RiemannInput const leftward{{0.125, {}, 0.1}, {1.0, {}, 1.0}};
    detail::RiemannInput const rightward{leftward.right, leftward.left};
    for (int pattern = 0; pattern < 3; pattern++) {
        std::vector<detail::RiemannInput> cases(kWidth * 2 + 1);
        for (size_t i = 0; i < cases.size(); i++) {
            cases[i] = (pattern == 0 || (pattern == 2 && i % 2 == 0)) ? leftward : rightward;
        }
        for (bool const finiteMass : {false, true}) {
            auto const packed = detail::packedFluxes(cases, finiteMass);
            for (size_t i = 0; i < cases.size(); i++) {
                SCOPED_TRACE(i);
                auto const& input = cases[i];
                auto const scalar = hopkins::riemannFlux(
                    input.left, input.right, input.area, input.frame, 1.4, finiteMass);
                detail::expectFluxNear(packed[i], scalar);
                EXPECT_EQ(
                    packed[i].contactSpeed < 0.0, pattern == 0 || (pattern == 2 && i % 2 == 0));
                if (finiteMass) {
                    EXPECT_DOUBLE_EQ(packed[i].mass, 0.0);
                }
            }
        }
    }
}

TEST(HopkinsSimd, MovingFacesPreserveReversalAndGalileanCovariance) {
    std::vector<detail::RiemannInput> cases;
    hopkins::Vector const boost{40.0, -20.0, 10.0};
    constexpr size_t kCount = hopkins::SimdReal::size() + 1;
    for (size_t i = 0; i < kCount; i++) {
        double const index = double(i);
        detail::RiemannInput input{{0.8 + 0.1 * index, {0.7, -0.4, 0.3}, 0.9},
            {1.3, {-0.2, 0.8, -0.1}, 0.4 + 0.05 * index}, {0.3, -0.1, 0.2}, {0.3, 0.2, 0.1}};
        cases.push_back(input);
    }
    auto reversed = cases;
    auto shifted = cases;
    for (size_t i = 0; i < cases.size(); i++) {
        std::swap(reversed[i].left, reversed[i].right);
        reversed[i].area = detail::scaled(reversed[i].area, -1.0);
        shifted[i].left = detail::boosted(shifted[i].left, boost);
        shifted[i].right = detail::boosted(shifted[i].right, boost);
        for (int dim = 0; dim < 3; dim++) {
            shifted[i].frame[dim] += boost[dim];
        }
    }
    for (bool const finiteMass : {false, true}) {
        auto const forward = detail::packedFluxes(cases, finiteMass);
        auto const reverse = detail::packedFluxes(reversed, finiteMass);
        auto const moving = detail::packedFluxes(shifted, finiteMass);
        for (size_t i = 0; i < cases.size(); i++) {
            SCOPED_TRACE(i);
            auto const& input = cases[i];
            detail::expectFluxNear(forward[i],
                hopkins::riemannFlux(
                    input.left, input.right, input.area, input.frame, 1.4, finiteMass));
            EXPECT_NEAR(forward[i].mass, -reverse[i].mass, 2.0e-14);
            EXPECT_NEAR(forward[i].energy, -reverse[i].energy, 2.0e-14);
            EXPECT_NEAR(moving[i].mass, forward[i].mass, 2.0e-14);
            for (int dim = 0; dim < 3; dim++) {
                EXPECT_NEAR(forward[i].momentum[dim], -reverse[i].momentum[dim], 2.0e-14);
                EXPECT_NEAR(moving[i].momentum[dim],
                    forward[i].momentum[dim] + boost[dim] * forward[i].mass, 3.0e-12);
            }
            EXPECT_NEAR(moving[i].energy,
                forward[i].energy + hopkins::dot(boost, forward[i].momentum) +
                    0.5 * hopkins::dot(boost, boost) * forward[i].mass,
                3.0e-11);
            if (finiteMass) {
                EXPECT_DOUBLE_EQ(forward[i].mass, 0.0);
            }
        }
    }
}

TEST(HopkinsSimd, ZeroAreaAndInvalidTailLanesAreSafeAndProduceZeroFlux) {
    constexpr size_t kWidth = hopkins::SimdReal::size();
    detail::RiemannInput const regular{{1.0, {}, 1.0}, {0.125, {}, 0.1}};
    auto zeroArea = regular;
    zeroArea.area = {};
    for (bool const finiteMass : {false, true}) {
        for (size_t count = 1; count <= kWidth; count++) {
            std::vector<detail::RiemannInput> cases(count, regular);
            cases.back() = zeroArea;
            std::feclearexcept(FE_ALL_EXCEPT);
            auto const fluxes = detail::packedFluxes(cases, finiteMass);
            EXPECT_EQ(std::fetestexcept(FE_INVALID | FE_DIVBYZERO), 0);
            detail::expectFluxNear(fluxes.back(), {});
            for (size_t lane = 0; lane + 1 < count; lane++) {
                detail::expectFluxNear(fluxes[lane],
                    hopkins::riemannFlux(
                        regular.left, regular.right, regular.area, regular.frame, 1.4, finiteMass));
            }
        }
    }
}

TEST(HopkinsSimd, FailureReportsTheActiveLaneWithoutRetrying) {
    constexpr size_t kWidth = hopkins::SimdReal::size();
    detail::RiemannInput const regular{{1.0, {}, 1.0}, {0.125, {}, 0.1}};
    size_t const badLane = kWidth - 1;
    for (bool const finiteMass : {false, true}) {
        std::vector<detail::RiemannInput> cases(kWidth, regular);
        cases[badLane] = {{1.0, {-8.0, 0.0, 0.0}, 1.0e-5}, {0.2, {8.0, 0.0, 0.0}, 1.0e-6}};
        try {
            detail::packedFluxes(cases, finiteMass);
            FAIL() << "Nonpositive HLLC star pressure must fail";
        } catch (std::runtime_error const& error) {
            EXPECT_NE(std::string(error.what()).find("HLLC"), std::string::npos);
            EXPECT_NE(std::string(error.what()).find("lane " + std::to_string(badLane)),
                std::string::npos);
        }
        cases[badLane] = regular;
        cases[badLane].left.pressure = -1.0;
        EXPECT_THROW(detail::packedFluxes(cases, finiteMass), std::invalid_argument);
    }
}

TEST(HopkinsSimd, PackedReconstructionMatchesScalarLimitersAndPhysicalRecovery) {
    using Real = hopkins::SimdReal;
    constexpr size_t kWidth = Real::size();
    bool sawRecovery = false;
    bool sawReconstructed = false;
    for (size_t batch = 0; batch < 3; batch++) {
        hopkins::SimdPrimitive centers;
        hopkins::SimdPrimitive neighbors;
        hopkins::SimdGradient gradients;
        hopkins::SimdVector3 offsets{};
        hopkins::SimdVector3 frames{};
        Real times{};
        Real fractions{};
        std::array<hopkins::Primitive, kWidth> expected{};
        for (size_t lane = 0; lane < kWidth; lane++) {
            size_t const index = batch * kWidth + lane;
            double const sign = index % 2 == 0 ? 1.0 : -1.0;
            hopkins::Primitive const center{
                1.0 + 0.1 * double(index), {sign * 0.5, 0.0, -sign * 0.3}, 0.8};
            hopkins::Primitive neighbor{center.density + sign * 0.3,
                {-sign * 0.8, 0.2, -sign * 0.3}, index % 3 == 0 ? 0.8 : 0.8 + sign * 0.2};
            hopkins::Gradient gradient;
            gradient.density = {0.7, -0.3, 0.2};
            gradient.pressure = {0.3, 0.5, -0.1};
            gradient.velocity = {{{0.2, 0.1, 0.3}, {-0.1, 0.4, 0.2}, {0.2, -0.3, 0.1}}};
            hopkins::Vector const offset{0.1 * sign, -0.2, 0.3};
            hopkins::Vector const frame{0.2, 0.1, -0.1};
            double const halfTimestep = index % 3 == 0 ? 10.0 : 0.02;
            double const fraction = index % 2 == 0 ? 0.25 : 0.75;
            expected[lane] = hopkins::reconstruct(
                center, gradient, offset, frame, halfTimestep, 1.4, &neighbor, fraction);
            if (halfTimestep == 10.0) {
                EXPECT_DOUBLE_EQ(expected[lane].density, center.density);
                EXPECT_DOUBLE_EQ(expected[lane].pressure, center.pressure);
                sawRecovery = true;
            } else {
                EXPECT_NE(expected[lane].density, center.density);
                sawReconstructed = true;
            }
            detail::setPrimitiveLane(centers, lane, center);
            detail::setPrimitiveLane(neighbors, lane, neighbor);
            times[lane] = halfTimestep;
            fractions[lane] = fraction;
            for (int dim = 0; dim < 3; dim++) {
                gradients.density[dim][lane] = gradient.density[dim];
                gradients.pressure[dim][lane] = gradient.pressure[dim];
                offsets[dim][lane] = offset[dim];
                frames[dim][lane] = frame[dim];
                for (int component = 0; component < 3; component++) {
                    gradients.velocity[component][dim][lane] = gradient.velocity[component][dim];
                }
            }
        }
        FlopCount operations = 0;
        auto const actual = hopkins::reconstructValues(
            centers, gradients, offsets, frames, times, 1.4, neighbors, fractions, &operations);
        EXPECT_EQ(operations, FlopCount(kWidth) * 157);
        for (size_t lane = 0; lane < kWidth; lane++) {
            SCOPED_TRACE(batch * kWidth + lane);
            EXPECT_NEAR(actual.density[lane], expected[lane].density, 2.0e-14);
            EXPECT_NEAR(actual.pressure[lane], expected[lane].pressure, 2.0e-14);
            for (int dim = 0; dim < 3; dim++) {
                EXPECT_NEAR(actual.velocity[dim][lane], expected[lane].velocity[dim], 2.0e-14);
            }
        }
    }
    EXPECT_TRUE(sawRecovery);
    EXPECT_TRUE(sawReconstructed);
}

TEST(Hopkins, IndependentSodIntegrationIsAccuratePositiveAndConservative) {
    auto const coarse = detail::evolveSod(100);
    auto const fine = detail::evolveSod(200);
    EXPECT_LT(fine.densityError, 0.012);
    EXPECT_LT(fine.densityError, 0.8 * coarse.densityError);
    EXPECT_GT(fine.minimumPressure, 0.0);
    EXPECT_NEAR(fine.totalMass, 0.5625, 2.0e-13);
    EXPECT_NEAR(fine.totalMomentum, (1.0 - 0.1) * 0.15, 2.0e-13);
    EXPECT_NEAR(fine.totalEnergy, 0.55 / 0.4, 3.0e-13);
}

TEST(Hopkins, ConservedPrimitiveConversionUsesTheAnchorVolume) {
    hopkins::Conserved const anchor{2.0, {6.0, -2.0, 4.0}, 19.0};
    auto const primitive = hopkins::primitiveFromConserved(anchor, 4.0, 1.4);
    EXPECT_DOUBLE_EQ(primitive.density, 0.5);
    EXPECT_EQ(primitive.velocity, (hopkins::Vector{3.0, -1.0, 2.0}));
    EXPECT_NEAR(primitive.pressure, 0.5, 1.0e-15);
    // Predicting an inactive anchor must not replace its old geometric volume
    // with the volume obtained from the globally drifted positions.
    auto const differentTime = hopkins::primitiveFromConserved(anchor, 2.0, 1.4);
    EXPECT_DOUBLE_EQ(differentTime.density, 2.0 * primitive.density);
    EXPECT_DOUBLE_EQ(differentTime.pressure, 2.0 * primitive.pressure);
}

TEST(Hopkins, OwnStepSourceClosureUsesEndpointMassesAndForces) {
    hopkins::Conserved const anchor{2.0, {0.6, -0.4, 0.2}, 20.0};
    hopkins::Conserved const transfer{0.3, {0.2, -0.1, 0.05}, 0.7};
    hopkins::Vector const oldForce{0.5, -0.25, 0.1};
    hopkins::Vector const newForce{0.8, -0.1, -0.2};
    double const duration = 0.4;
    auto const closed = hopkins::closeConservedInterval(
        anchor, transfer, oldForce, newForce, duration, 0.0, 0.0, 0.0);
    EXPECT_DOUBLE_EQ(closed.mass, anchor.mass + transfer.mass);
    for (int dim = 0; dim < 3; ++dim) {
        double const exact = anchor.momentum[dim] + transfer.momentum[dim] +
            0.5 * duration * (anchor.mass * oldForce[dim] + closed.mass * newForce[dim]);
        EXPECT_NEAR(closed.momentum[dim], exact, 2.0e-15);
    }
}

TEST(Hopkins, PendingFineTransfersDoNotChangeTheCoarseAnchor) {
    hopkins::Conserved coarse{2.0, {0.3, -0.2, 0.1}, 12.0};
    hopkins::Conserved fine{1.0, {-0.1, 0.4, -0.2}, 7.0};
    auto const totalMass = coarse.mass + fine.mass;
    auto const totalEnergy = coarse.energy + fine.energy;
    hopkins::Vector totalMomentum{};
    for (int dim = 0; dim < 3; ++dim) {
        totalMomentum[dim] = coarse.momentum[dim] + fine.momentum[dim];
    }
    // Two coarse intervals; a fine partner supplies two independently completed
    // transfers per interval. Only the fine particle commits at the midpoint.
    for (int interval = 0; interval < 2; ++interval) {
        auto const anchorPrimitive = hopkins::primitiveFromConserved(coarse, 2.0, 1.4);
        hopkins::Conserved pending;
        for (int step = 0; step < 2; ++step) {
            hopkins::Conserved const delta{0.01 * (step + 1), {0.02, -0.01, 0.03}, 0.04};
            pending.mass += delta.mass;
            pending.energy += delta.energy;
            hopkins::Conserved opposite{-delta.mass, {}, -delta.energy};
            for (int dim = 0; dim < 3; ++dim) {
                pending.momentum[dim] += delta.momentum[dim];
                opposite.momentum[dim] = -delta.momentum[dim];
            }
            fine = hopkins::closeConservedInterval(fine, opposite, {}, {}, 0.5, 0.0, 0.0, 0.0);
            auto const stillAnchored = hopkins::primitiveFromConserved(coarse, 2.0, 1.4);
            EXPECT_DOUBLE_EQ(stillAnchored.density, anchorPrimitive.density);
            EXPECT_DOUBLE_EQ(stillAnchored.pressure, anchorPrimitive.pressure);
            EXPECT_EQ(stillAnchored.velocity, anchorPrimitive.velocity);
            EXPECT_NEAR(coarse.mass + pending.mass + fine.mass, totalMass, 1.0e-15);
        }
        coarse = hopkins::closeConservedInterval(coarse, pending, {}, {}, 1.0, 0.0, 0.0, 0.0);
        EXPECT_NEAR(coarse.energy + fine.energy, totalEnergy, 8.0e-15);
        for (int dim = 0; dim < 3; ++dim) {
            EXPECT_NEAR(coarse.momentum[dim] + fine.momentum[dim], totalMomentum[dim], 1.0e-15);
        }
    }
}

TEST(Hopkins, OwnStepClosureRejectsInvalidPublishedTransfers) {
    hopkins::Conserved const anchor{1.0, {}, 1.0};
    EXPECT_THROW(
        hopkins::closeConservedInterval(anchor, {-2.0, {}, 0.0}, {}, {}, 1.0, 0.0, 0.0, 0.0),
        std::runtime_error);
    EXPECT_THROW(
        hopkins::closeConservedInterval(anchor, {0.0, {}, -2.0}, {}, {}, 1.0, 0.0, 0.0, 0.0),
        std::runtime_error);
}

TEST(Hopkins, CachedFluxCompleteIntervalIsMidpointUpdate) {
    hopkins::Conserved const start{1.0, {2.0, -3.0, 4.0}, 5.0};
    hopkins::Conserved const midpoint{4.0, {6.0, -4.0, 9.0}, 12.0};
    auto const integral = hopkins::integrateFlux(start, midpoint, 2.0, 0.0, 2.0);
    EXPECT_DOUBLE_EQ(integral.mass, 8.0);
    EXPECT_DOUBLE_EQ(integral.energy, 24.0);
    EXPECT_EQ(integral.momentum, (hopkins::Vector{12.0, -8.0, 18.0}));
}

TEST(Hopkins, CachedFluxArbitrarySubdivisionsAreAdditiveAndAntisymmetric) {
    hopkins::Conserved const start{1.3, {2.1, -3.2, 4.3}, 5.4};
    hopkins::Conserved const midpoint{4.1, {6.2, -4.3, 9.4}, 12.5};
    std::array<double, 6> const boundaries{0.0, 0.031, 0.27, 0.68, 1.12, 1.3};
    hopkins::Conserved left;
    hopkins::Conserved right;
    for (size_t i = 1; i < boundaries.size(); i++) {
        auto const increment = hopkins::integrateFlux(start, midpoint, boundaries.back(),
            boundaries[i - 1], boundaries[i] - boundaries[i - 1]);
        left.mass -= increment.mass;
        right.mass += increment.mass;
        left.energy -= increment.energy;
        right.energy += increment.energy;
        for (int dim = 0; dim < 3; dim++) {
            left.momentum[dim] -= increment.momentum[dim];
            right.momentum[dim] += increment.momentum[dim];
            EXPECT_DOUBLE_EQ(left.momentum[dim] + right.momentum[dim], 0.0);
        }
        EXPECT_DOUBLE_EQ(left.mass + right.mass, 0.0);
        EXPECT_DOUBLE_EQ(left.energy + right.energy, 0.0);
    }
    EXPECT_NEAR(right.mass, boundaries.back() * midpoint.mass, 2.0e-15);
    EXPECT_NEAR(right.energy, boundaries.back() * midpoint.energy, 4.0e-15);
    for (int dim = 0; dim < 3; dim++) {
        EXPECT_NEAR(right.momentum[dim], boundaries.back() * midpoint.momentum[dim], 4.0e-15);
    }
}

TEST(Hopkins, CachedFluxWakeupTruncatesBeforeRefreshing) {
    // Independent time-dependent flux F(t)=a+b*t. Wake up a slow interface at
    // t=0.3 before its original interval ends, then use two shorter intervals.
    // Its running conserved state must never include the unused future flux.
    hopkins::Conserved const coefficients{1.7, {0.3, -0.9, 2.5}, 4.2};
    hopkins::Conserved const slopes{-0.2, {1.3, 0.8, -0.4}, 0.7};
    auto flux = [&](double time) {
        hopkins::Conserved value;
        value.mass = coefficients.mass + time * slopes.mass;
        value.energy = coefficients.energy + time * slopes.energy;
        for (int dim = 0; dim < 3; dim++) {
            value.momentum[dim] = coefficients.momentum[dim] + time * slopes.momentum[dim];
        }
        return value;
    };
    hopkins::Conserved sum;
    std::array<double, 4> const boundaries{0.0, 0.3, 0.65, 1.0};
    for (size_t i = 1; i < boundaries.size(); i++) {
        double const begin = boundaries[i - 1];
        double const end = boundaries[i];
        double const cachedInterval = i == 1 ? 1.0 : end - begin;
        auto const increment = hopkins::integrateFlux(
            flux(begin), flux(begin + 0.5 * cachedInterval), cachedInterval, 0.0, end - begin);
        sum.mass += increment.mass;
        sum.energy += increment.energy;
        EXPECT_NEAR(sum.mass, coefficients.mass * end + 0.5 * slopes.mass * end * end, 1.0e-15);
        EXPECT_NEAR(
            sum.energy, coefficients.energy * end + 0.5 * slopes.energy * end * end, 2.0e-15);
        for (int dim = 0; dim < 3; dim++) {
            sum.momentum[dim] += increment.momentum[dim];
            EXPECT_NEAR(sum.momentum[dim],
                coefficients.momentum[dim] * end + 0.5 * slopes.momentum[dim] * end * end, 1.0e-15);
        }
    }
}

TEST(Hopkins, CachedFluxRejectsInvalidIntervalsAndStates) {
    hopkins::Conserved flux{1.0, {2.0, 3.0, 4.0}, 5.0};
    double const infinity = std::numeric_limits<double>::infinity();
    double const nan = std::numeric_limits<double>::quiet_NaN();
    for (auto const& times : {std::array<double, 3>{0.0, 0.0, 0.0}, {-1.0, 0.0, 0.0},
             {infinity, 0.0, 0.0}, {1.0, -0.1, 0.1}, {1.0, 0.0, -0.1}, {1.0, 0.9, 0.2},
             {1.0, infinity, 0.0}, {1.0, 0.0, nan}}) {
        EXPECT_THROW(hopkins::integrateFlux(flux, flux, times[0], times[1], times[2]),
            std::invalid_argument);
    }
    auto const empty = hopkins::integrateFlux(flux, flux, 1.0, 1.0, 0.0);
    EXPECT_DOUBLE_EQ(empty.mass, 0.0);
    EXPECT_DOUBLE_EQ(empty.energy, 0.0);
    EXPECT_EQ(empty.momentum, hopkins::Vector{});
    flux.momentum[2] = nan;
    EXPECT_THROW(hopkins::integrateFlux(flux, flux, 1.0, 0.0, 1.0), std::invalid_argument);
}

TEST(Hopkins, PairTimestepUsesTheFasterEndpoint) {
    for (int left = 0; left <= 6; left++) {
        for (int right = 0; right <= 6; right++) {
            double const timestep = hopkins::pairTimestep(8.0, left, right);
            EXPECT_LE(timestep, std::ldexp(8.0, -left));
            EXPECT_LE(timestep, std::ldexp(8.0, -right));
            EXPECT_TRUE(timestep == std::ldexp(8.0, -left) || timestep == std::ldexp(8.0, -right));
            EXPECT_DOUBLE_EQ(timestep, hopkins::pairTimestep(8.0, right, left));
        }
    }
    EXPECT_THROW(hopkins::pairTimestep(0.0, 0, 0), std::invalid_argument);
    EXPECT_THROW(hopkins::pairTimestep(1.0, -1, 0), std::invalid_argument);
    EXPECT_THROW(hopkins::pairTimestep(1.0, 0, 2000), std::invalid_argument);
}

TEST(Hopkins, PredictorClockRecoversElapsedTimeAcrossWraps) {
    for (uint64_t origin : {uint64_t(0), uint64_t(65530), uint64_t(1234567890123)}) {
        uint16_t const anchor = uint16_t(origin);
        for (uint64_t elapsed :
            {uint64_t(0), uint64_t(1), uint64_t(7), uint64_t(1024), uint64_t(32768)}) {
            EXPECT_EQ(hopkins::elapsedTicks(origin + elapsed, anchor), elapsed);
        }
    }
    // A future or excessively stale anchor must remain visibly outside the
    // maximum permitted age, rather than being clamped into a valid interval.
    EXPECT_GT(hopkins::elapsedTicks(100, 101), 32768);
    EXPECT_GT(hopkins::elapsedTicks(40000, 0), 32768);
}

TEST(Hopkins, PredictorClockHandlesWakeupAndSynchronizedCoarsening) {
    uint64_t tick = 65520;
    uint16_t anchor = uint16_t(tick);
    tick += 8;
    EXPECT_EQ(hopkins::elapsedTicks(tick, anchor), 8);
    // Wake the coarse cell before its old interval ends. Its new fine anchor
    // replaces the old phase; the next event crosses the 16-bit clock boundary.
    anchor = uint16_t(tick);
    tick += 8;
    EXPECT_EQ(hopkins::elapsedTicks(tick, anchor), 8);
    EXPECT_EQ(tick % 32768, 0);
    // At this synchronized boundary a coarsened base-step predictor may remain
    // fixed for the full longest interval, with no writes during fine events.
    anchor = uint16_t(tick);
    for (uint64_t elapsed = 1; elapsed <= 32768; elapsed *= 2) {
        EXPECT_EQ(hopkins::elapsedTicks(tick + elapsed, anchor), elapsed);
        EXPECT_EQ(anchor, uint16_t(tick));
    }
}

TEST(Hopkins, SourceCorrectionMatchesIndependentForceIntegral) {
    constexpr double kDuration = 0.7;
    constexpr int kIntervals = 1000;
    double const width = kDuration / kIntervals;
    for (double damping : {0.0, 1.0e-12, 1.0e-4, 0.4, 3.0}) {
        // Direct Simpson quadrature of the damped response to g(t)=t/T.
        double integral = 0.0;
        for (int i = 0; i <= kIntervals; i++) {
            double const time = i * width;
            double const coefficient = i == 0 || i == kIntervals ? 1.0 : (i % 2 == 0 ? 2.0 : 4.0);
            integral += coefficient * std::exp(-damping * (kDuration - time)) * time / kDuration;
        }
        integral *= width / 3.0;
        EXPECT_NEAR(hopkins::sourceCorrectionWeight(kDuration, damping), integral, 2.0e-13);
    }
    EXPECT_DOUBLE_EQ(hopkins::sourceCorrectionWeight(0.0, 0.4), 0.0);
    EXPECT_DOUBLE_EQ(hopkins::sourceCorrectionWeight(kDuration, 0.0), 0.5 * kDuration);
    EXPECT_THROW(hopkins::sourceCorrectionWeight(-1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(hopkins::sourceCorrectionWeight(1.0, -1.0), std::invalid_argument);
}

TEST(Hopkins, UnequalRawSourceIntervalsRecoverLinearForceIntegral) {
    constexpr double kInitialVelocity = -0.4;
    constexpr double kInitialAcceleration = 1.25;
    constexpr double kAccelerationSlope = -0.7;
    // Non-dyadic, unequal intervals exercise endpoint correction with the
    // completed interval, independent of the following timestep's length.
    std::array<double, 5> const durations{0.037, 0.081, 0.013, 0.096, 0.023};
    for (double damping : {0.0, 0.4, 3.0}) {
        double time = 0.0;
        double velocity = kInitialVelocity;
        for (double duration : durations) {
            double const acceleration = kInitialAcceleration + kAccelerationSlope * time;
            double const decay = std::exp(-damping * duration);
            double const impulse =
                damping == 0.0 ? duration : -std::expm1(-damping * duration) / damping;
            velocity = decay * velocity + impulse * acceleration;
            velocity +=
                hopkins::sourceCorrectionWeight(duration, damping) * kAccelerationSlope * duration;
            time += duration;
            // Independent exact solution of v' + damping*v = g0 + slope*t.
            double exact;
            if (damping == 0.0) {
                exact = kInitialVelocity + kInitialAcceleration * time +
                    0.5 * kAccelerationSlope * time * time;
            } else {
                double const response = -std::expm1(-damping * time) / damping;
                exact = kInitialVelocity * std::exp(-damping * time) +
                    kInitialAcceleration * response +
                    kAccelerationSlope * (time - response) / damping;
            }
            EXPECT_NEAR(velocity, exact, 2.0e-14);
        }
    }
}

TEST(Hopkins, ConstantSourceKeepsItsAnalyticImpulseAcrossSubsteps) {
    constexpr double kAcceleration = 1.25;
    constexpr double kEndTime = 1.6;
    constexpr int kForceIntervals = 16;
    double const forceTimestep = kEndTime / kForceIntervals;
    for (double damping : {0.0, 0.4}) {
        double const response =
            damping == 0.0 ? kEndTime : -std::expm1(-damping * kEndTime) / damping;
        double const exactVelocity =
            -0.4 * std::exp(-damping * kEndTime) + kAcceleration * response;
        double const exactPosition = damping == 0.0 ?
            0.2 - 0.4 * kEndTime + 0.5 * kAcceleration * kEndTime * kEndTime :
            0.2 - 0.4 * response + kAcceleration * (kEndTime - response) / damping;
        double previousPositionError = 1.0;
        for (int substeps : {1, 2, 4}) {
            double const timestep = forceTimestep / substeps;
            double const decay = std::exp(-damping * timestep);
            double const impulse =
                damping == 0.0 ? timestep : -std::expm1(-damping * timestep) / damping;
            double position = 0.2;
            double velocity = -0.4;
            double previousAcceleration = kAcceleration;
            for (int interval = 0; interval < kForceIntervals; interval++) {
                for (int substep = 0; substep < substeps; substep++) {
                    double const nextVelocity = decay * velocity + impulse * previousAcceleration;
                    position += 0.5 * timestep * (velocity + nextVelocity);
                    velocity = nextVelocity;
                }
                velocity += hopkins::sourceCorrectionWeight(forceTimestep, damping) *
                    (kAcceleration - previousAcceleration);
                previousAcceleration = kAcceleration;
            }
            EXPECT_NEAR(velocity, exactVelocity, 2.0e-14);
            double const error = std::abs(position - exactPosition);
            if (damping == 0.0) {
                EXPECT_LT(error, 2.0e-14);
            } else {
                EXPECT_LT(error, 0.26 * previousPositionError);
            }
            previousPositionError = error;
        }
    }
}

TEST(Hopkins, MultirateSourceCorrectionConvergesAtSecondOrder) {
    constexpr double kEndTime = 2.0;
    auto evolve = [&](int forceIntervals, int substeps, double damping) {
        double const forceTimestep = kEndTime / forceIntervals;
        double const timestep = forceTimestep / substeps;
        double const decay = std::exp(-damping * timestep);
        double const impulse =
            damping == 0.0 ? timestep : -std::expm1(-damping * timestep) / damping;
        std::array<double, 2> state{1.0, 0.0};
        double acceleration = -state[0];
        for (int interval = 0; interval < forceIntervals; interval++) {
            for (int substep = 0; substep < substeps; substep++) {
                double const nextVelocity = decay * state[1] + impulse * acceleration;
                state[0] += 0.5 * timestep * (state[1] + nextVelocity);
                state[1] = nextVelocity;
            }
            double const nextAcceleration = -state[0];
            state[1] += hopkins::sourceCorrectionWeight(forceTimestep, damping) *
                (nextAcceleration - acceleration);
            acceleration = nextAcceleration;
        }
        return state;
    };
    for (double damping : {0.0, 0.4}) {
        double const frequency = std::sqrt(1.0 - 0.25 * damping * damping);
        double const decay = std::exp(-0.5 * damping * kEndTime);
        // Exact damped oscillator x''+damping*x'+x=0, x(0)=1, v(0)=0.
        std::array<double, 2> const exact{decay *
                (std::cos(frequency * kEndTime) +
                    0.5 * damping / frequency * std::sin(frequency * kEndTime)),
            -decay * std::sin(frequency * kEndTime) / frequency};
        for (int substeps : {1, 2, 4}) {
            auto const coarse = evolve(32, substeps, damping);
            auto const fine = evolve(64, substeps, damping);
            for (int component = 0; component < 2; component++) {
                double const coarseError = std::abs(coarse[component] - exact[component]);
                double const fineError = std::abs(fine[component] - exact[component]);
                EXPECT_GT(coarseError, 1.0e-8);
                EXPECT_LT(fineError, 0.27 * coarseError);
                EXPECT_LT(fineError, 3.0e-4);
            }
        }
    }
}

TEST(Hopkins, InactivePrimitivePredictionPreservesItsAnchor) {
    hopkins::Primitive const anchor{2.0, {0.4, -0.3, 0.2}, 3.0};
    hopkins::Gradient gradient;
    gradient.density = {9.0, -8.0, 7.0};
    // A stationary material contact has no density-advection term even with
    // a density gradient and a nonzero bulk velocity.
    auto const prediction = hopkins::predictPrimitive(anchor, gradient, 0.2, 1.4);
    EXPECT_DOUBLE_EQ(prediction.density, anchor.density);
    EXPECT_EQ(prediction.velocity, anchor.velocity);
    EXPECT_DOUBLE_EQ(prediction.pressure, anchor.pressure);
    gradient.pressure = {1.0, 2.0, -3.0};
    auto const changed = hopkins::predictPrimitive(anchor, gradient, 0.2, 1.4);
    EXPECT_DOUBLE_EQ(changed.density, 2.0);
    EXPECT_DOUBLE_EQ(changed.pressure, 3.0);
    EXPECT_EQ(anchor.velocity, (hopkins::Vector{0.4, -0.3, 0.2}));
    EXPECT_DOUBLE_EQ(anchor.density, 2.0);
    EXPECT_DOUBLE_EQ(anchor.pressure, 3.0);
    EXPECT_NEAR(changed.velocity[0], 0.3, 1.0e-15);
    EXPECT_NEAR(changed.velocity[1], -0.5, 1.0e-15);
    EXPECT_NEAR(changed.velocity[2], 0.5, 1.0e-15);
}

TEST(Hopkins, InactivePrimitivePredictionUsesMaterialDivergence) {
    hopkins::Primitive const anchor{2.0, {0.4, -0.3, 0.2}, 3.0};
    hopkins::Gradient gradient;
    gradient.velocity = {{{0.1, 20.0, -10.0}, {-15.0, 0.2, 25.0}, {30.0, -35.0, -0.05}}};
    auto const prediction = hopkins::predictPrimitive(anchor, gradient, 0.2, 1.4);
    EXPECT_NEAR(prediction.density, 1.9, 1.0e-15);
    EXPECT_NEAR(prediction.pressure, 2.79, 1.0e-15);
    EXPECT_EQ(prediction.velocity, anchor.velocity);
    EXPECT_TRUE(hopkins::physical(prediction));
}

TEST(Hopkins, InactivePrimitivePredictionExposesNonphysicalStates) {
    hopkins::Primitive const anchor{1.0, {}, 1.0};
    hopkins::Gradient gradient;
    gradient.velocity[0][0] = 2.0;
    auto const prediction = hopkins::predictPrimitive(anchor, gradient, 1.0, 1.4);
    EXPECT_FALSE(hopkins::physical(prediction));
    EXPECT_LT(prediction.density, 0.0);
    EXPECT_LT(prediction.pressure, 0.0);
    EXPECT_DOUBLE_EQ(anchor.density, 1.0);
    EXPECT_DOUBLE_EQ(anchor.pressure, 1.0);
}

TEST(Hopkins, InactivePrimitivePredictionIsGalileanInvariant) {
    hopkins::Primitive const anchor{2.0, {0.4, -0.3, 0.2}, 3.0};
    hopkins::Gradient gradient;
    gradient.pressure = {0.7, -0.4, 0.2};
    gradient.velocity[0][0] = -0.3;
    hopkins::Vector const boost{10.0, -20.0, 30.0};
    auto const original = hopkins::predictPrimitive(anchor, gradient, 0.125, 1.4);
    auto const shifted =
        hopkins::predictPrimitive(detail::boosted(anchor, boost), gradient, 0.125, 1.4);
    EXPECT_DOUBLE_EQ(shifted.density, original.density);
    EXPECT_DOUBLE_EQ(shifted.pressure, original.pressure);
    for (int dim = 0; dim < 3; dim++) {
        EXPECT_NEAR(shifted.velocity[dim], original.velocity[dim] + boost[dim], 5.0e-15);
    }
}

TEST(HopkinsFlops, CachedFluxAndSourceBranchesKeepNumericalResults) {
    hopkins::Conserved const start{1.0, {2.0, 3.0, 4.0}, 5.0};
    hopkins::Conserved const midpoint{2.0, {3.0, 4.0, 5.0}, 6.0};
    FlopCount operations = 0;
    auto const expected = hopkins::integrateFlux(start, midpoint, 1.0, 0.25, 0.5);
    auto const actual = hopkins::integrateFlux(start, midpoint, 1.0, 0.25, 0.5, &operations);
    EXPECT_EQ(operations, 24);
    EXPECT_EQ(actual.mass, expected.mass);
    EXPECT_EQ(actual.momentum, expected.momentum);
    EXPECT_EQ(actual.energy, expected.energy);
    operations = 0;
    EXPECT_EQ(hopkins::integrateFlux(start, midpoint, 1.0, 0.25, 0.0, &operations).mass, 0.0);
    EXPECT_EQ(operations, 4);
    operations = 0;
    EXPECT_EQ(hopkins::pairTimestep(1.0, 2, 3, &operations), 0.125);
    EXPECT_EQ(operations, 1);
    operations = 0;
    EXPECT_EQ(hopkins::sourceCorrectionWeight(0.2, 0.0, &operations), 0.1);
    EXPECT_EQ(operations, 10);
    operations = 0;
    EXPECT_EQ(hopkins::sourceCorrectionWeight(0.2, 3.0, &operations),
        hopkins::sourceCorrectionWeight(0.2, 3.0));
    EXPECT_EQ(operations, 5);
}

TEST(HopkinsFlops, GeometryCountsExecutedDimensionsAndSingularExit) {
    hopkins::Matrix moment{};
    for (int dim = 0; dim < 3; dim++) {
        moment[dim][dim] = 2.0;
    }
    // Analytic Cholesky/triangular-solve costs, including three-component row norms.
    std::array<FlopCount, 3> const expected{22, 60, 132};
    for (int dimensions = 1; dimensions <= 3; dimensions++) {
        FlopCount operations = 0;
        auto const geometry = hopkins::makeGeometry(2.0, moment, 100.0, dimensions, &operations);
        EXPECT_TRUE(geometry.valid);
        EXPECT_EQ(operations, expected[dimensions - 1]);
        EXPECT_EQ(geometry.inverseMoment,
            hopkins::makeGeometry(2.0, moment, 100.0, dimensions).inverseMoment);
    }
    FlopCount operations = 0;
    EXPECT_FALSE(hopkins::makeGeometry(0.0, moment, 100.0, 3, &operations).valid);
    EXPECT_EQ(operations, 0);
    EXPECT_FALSE(hopkins::makeGeometry(1.0, {}, 100.0, 3, &operations).valid);
    EXPECT_EQ(operations, 10);
}

TEST(HopkinsFlops, GeometryAndGravityWorkPropagateNestedCounts) {
    hopkins::Geometry geometry;
    geometry.volume = 2.0;
    for (int dim = 0; dim < 3; dim++) {
        geometry.inverseMoment[dim][dim] = 1.0;
    }
    hopkins::Vector const displacement{1.0, 2.0, 3.0};
    FlopCount operations = 0;
    auto const rate = hopkins::geometryRate(geometry, 0.2, {}, &operations);
    EXPECT_EQ(operations, 108);
    operations = 0;
    hopkins::faceArea(geometry, geometry, displacement, 0.5, 0.5, &operations);
    EXPECT_EQ(operations, 51);
    operations = 0;
    hopkins::faceAreaRate(
        geometry, geometry, rate, rate, displacement, {}, 0.5, 0.5, 0.1, 0.1, &operations);
    EXPECT_EQ(operations, 135);
    operations = 0;
    hopkins::pairGravityWork(2.0, displacement, displacement, displacement, &operations);
    EXPECT_EQ(operations, 16);
    operations = 0;
    hopkins::integratePairGravityWork(displacement, displacement, 1.0, 0.25, 0.5, &operations);
    EXPECT_EQ(operations, 51);
}

TEST(HopkinsFlops, LimitersCountOnlyExecutedBranches) {
    FlopCount operations = 0;
    EXPECT_EQ(hopkins::pairLimiter(1.0, 1.0, 2.0, 0.5, &operations), 1.0);
    EXPECT_EQ(operations, 0);
    hopkins::pairLimiter(1.0, 2.0, 2.0, 0.5, &operations);
    EXPECT_EQ(operations, 9);
    operations = 0;
    hopkins::pairLimiter(0.1, 1.0, 0.2, 0.5, &operations);
    EXPECT_EQ(operations, 12);
    operations = 0;
    hopkins::pairLimiter(-1.0, -0.1, -0.2, 0.5, &operations);
    EXPECT_EQ(operations, 12);
    operations = 0;
    hopkins::gradientLimiter(1.0, 0.0, 2.0, 1.0, 1.0, 1.0, &operations);
    EXPECT_EQ(operations, 2);
    operations = 0;
    hopkins::gradientLimiter(1.0, 0.0, 2.0, 0.5, 1.5, 1.0, &operations);
    EXPECT_EQ(operations, 10);
    double lower = 0.5;
    double upper = 2.0;
    operations = 0;
    EXPECT_EQ(
        hopkins::supportRadiusStep(1.0, -0.1, 1.0, 0.1, 3.0, lower, upper, false, &operations),
        1.1);
    EXPECT_EQ(operations, 4);
    lower = 0.95;
    upper = 2.0;
    operations = 0;
    EXPECT_EQ(
        hopkins::supportRadiusStep(1.0, 0.1, 0.01, 0.1, 3.0, lower, upper, false, &operations),
        0.975);
    EXPECT_EQ(operations, 7);
}

TEST(HopkinsFlops, ReconstructionAndPredictionIncludeNestedDotProducts) {
    hopkins::Primitive const state{1.0, {0.1, 0.2, 0.3}, 1.0};
    FlopCount operations = 0;
    auto const predicted = hopkins::predictPrimitive(state, {}, 0.2, 1.4, &operations);
    EXPECT_EQ(predicted.velocity, state.velocity);
    EXPECT_EQ(operations, 19);
    operations = 0;
    auto const reconstructed =
        hopkins::reconstruct(state, {}, {}, {}, 0.1, 1.4, &state, 0.5, &operations);
    EXPECT_EQ(reconstructed.velocity, state.velocity);
    EXPECT_EQ(operations, 82);
}

TEST(HopkinsFlops, FloatStorageConversionsCountAtDoublePrecision) {
    double lower = 0.5;
    double upper = 2.0;
    FlopCount doubleOperations = 0;
    FlopCount singleOperations = 0;
    auto const radius = hopkins::supportRadiusStep(
        1.0, -0.1, 1.0, 0.1, 3.0, lower, upper, true, &doubleOperations, &singleOperations);
    EXPECT_EQ(radius, double(float(1.1)));
    // Four Newton/clamp operations plus the two narrowing/promotion pairs.
    EXPECT_EQ(doubleOperations, 8);
    EXPECT_EQ(singleOperations, 1);
    lower = 1.0;
    upper = std::nextafter(1.0f, std::numeric_limits<float>::infinity());
    doubleOperations = 0;
    singleOperations = 0;
    EXPECT_EQ(hopkins::supportRadiusStep(lower, -0.1, 1.0, 0.1, 3.0, lower, upper, true,
                  &doubleOperations, &singleOperations),
        1.0);
    // Adjacent float endpoints return before the Newton step or output round trip.
    EXPECT_EQ(doubleOperations, 2);
    EXPECT_EQ(singleOperations, 1);
}

TEST(HopkinsFlops, RiemannCountsSharedScalarAndPackedArithmetic) {
    hopkins::Primitive const state{1.0, {}, 1.0};
    hopkins::Vector const area{1.0, 0.0, 0.0};
    for (bool const finiteMass : {false, true}) {
        FlopCount operations = 0;
        auto const actual =
            hopkins::riemannFlux(state, state, area, {}, 1.4, finiteMass, &operations);
        auto const expected = hopkins::riemannFlux(state, state, area, {}, 1.4, finiteMass);
        EXPECT_EQ(actual.mass, expected.mass);
        EXPECT_EQ(actual.momentum, expected.momentum);
        EXPECT_EQ(actual.energy, expected.energy);
        EXPECT_EQ(operations, finiteMass ? 146 : 169);
        using Real = hopkins::SimdReal;
        hopkins::SimdPrimitive const packed{Real(1.0), {}, Real(1.0)};
        hopkins::SimdVector3 const packedArea{Real(1.0), Real(0.0), Real(0.0)};
        // A partial pack still executes full-width arithmetic. Mask work is not
        // counted, and inactive lanes must neither fail nor reach the output.
        auto const active = Real([](auto lane) { return double(lane); }) < Real(1.0);
        operations = 0;
        auto const flux = hopkins::riemannFluxValues(packed, packed, packedArea,
            hopkins::SimdVector3{}, 1.4, finiteMass, active, &operations);
        EXPECT_EQ(operations, FlopCount(Real::size()) * (finiteMass ? 146 : 169));
        EXPECT_DOUBLE_EQ(flux.momentum[0][0], 1.0);
        operations = 0;
        hopkins::riemannFluxValues(packed, packed, hopkins::SimdVector3{}, hopkins::SimdVector3{},
            1.4, finiteMass, active, &operations);
        EXPECT_EQ(operations, FlopCount(Real::size()) * 6);
    }
    FlopCount operations = 0;
    hopkins::riemannFlux(state, state, {}, {}, 1.4, false, &operations);
    EXPECT_EQ(operations, 6);
}
