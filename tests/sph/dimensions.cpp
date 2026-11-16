// Verify runtime SPH dimensions, density kernels, inactive-axis padding, and variational forces.
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/position.hpp>
#include <cosmictiger/math/radial_density_kernel.hpp>
#include <cosmictiger/particles/particle_types.hpp>
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/variable_smoothing_lengths.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>
#include <cosmictiger/sph/volume.hpp>

static_assert(sizeof(ParticleVector<float>) == kDimensions * sizeof(float));
static_assert(sizeof(ParticleVector<double>) == kDimensions * sizeof(double));
static_assert(std::is_trivially_copyable_v<ParticleVector<float>>);
static_assert(sizeof(ParticleVector<Fixed32>) == kDimensions * sizeof(Fixed32));
static_assert(sizeof(ParticleVector<Fixed64>) == kDimensions * sizeof(Fixed64));

namespace detail {

void near(double actual, double expected, double tolerance, char const* label) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::cerr << label << ": " << actual << " expected " << expected << '\n';
        throw std::runtime_error(label);
    }
}

using SpatialVector = sph_geometry::SpatialVector<double>;
inline constexpr radial_kernel::KernelSpec kCubicKernel{radial_kernel::Family::Spline, 3};

void geometry() {
    std::array<double, kDimensions> const volumes{
        2.0, std::numbers::pi, 4.0 * std::numbers::pi / 3.0};
    near(sph_geometry::unitBallVolume(), volumes[spatialDimensions() - 1], 0.0, "unit ball volume");
    double const support = sph_geometry::supportRadius(0.3, 1.7, 12.0);
    near(volumes[spatialDimensions() - 1] * std::pow(support, spatialDimensions()) * 1.7 / 0.3,
        12.0, 1e-13, "support radius");

    // Deliberately nonzero inactive inputs must not enter SPH vector calculations.
    std::array<double, kDimensions> const components{0.125, 0.25, 0.375};
    auto const vector =
        sph_geometry::loadSpatialVector(&components[kXi], &components[kYi], &components[kZi], 0);
    auto const padded = sph_geometry::paddedVector(vector);
    SpatialVector left(0.375);
    SpatialVector right(0.125);
    auto const separation = distance(left, right);
    double expectedNormSquared = 0.0;
    for (int axis = 0; axis < kDimensions; ++axis) {
        bool const active = axis < spatialDimensions();
        near(vector[axis], active ? components[axis] : 0.0, 0.0, "gather padding");
        near(padded[axis], active ? components[axis] : 0.0, 0.0, "padded vector");
        near(separation[axis], active ? 0.25 : 0.0, 0.0, "separation padding");
        if (active) {
            expectedNormSquared += components[axis] * components[axis];
        }
    }
    near(vector.dot(vector), expectedNormSquared, 0.0, "active vector dot product");
}

void kernels() {
    std::vector<radial_kernel::KernelSpec> specs;
    constexpr int kMaximumSplineDegree = 9;
    constexpr int kMaximumFerrersDegree = 8;
    for (int degree = 2; degree <= kMaximumSplineDegree; ++degree) {
        specs.push_back({radial_kernel::Family::Spline, degree});
    }
    for (int degree = spatialDimensions() == 1 ? 1 : 2; degree <= kMaximumSplineDegree;
        degree += 3) {
        specs.push_back({radial_kernel::Family::Wendland, degree});
    }
    for (int degree = 2; degree <= kMaximumFerrersDegree; degree += 2) {
        specs.push_back({radial_kernel::Family::Ferrers, degree});
    }
    for (auto const spec : specs) {
        auto const table = makeHydroKernelTable<double>(spec);
        auto const floatTable = makeHydroKernelTable<float>(spec);
        if (table.pieceCount == 0 || floatTable.pieceCount == 0) {
            throw std::runtime_error("Missing dimensional kernel");
        }
        constexpr int kSamples = 32768;
        double integral = 0;
        for (int sample = 0; sample < kSamples; ++sample) {
            double const radius = (sample + 0.5) / kSamples;
            integral +=
                radial_kernel::value(table, radius) * std::pow(radius, spatialDimensions() - 1);
        }
        near(integral * spatialDimensions() * sph_geometry::unitBallVolume() / kSamples, 1.0, 1e-7,
            "kernel integral");
        double const radius = 0.231;
        double const support = 0.73;
        auto const density = [&](double length) {
            return radial_kernel::value(table, radius / length) /
                std::pow(length, spatialDimensions());
        };
        double const derivative = sph_volume::kernelSupportDerivative(
            radial_kernel::value(table, radius / support),
            radial_kernel::derivative(table, radius / support), radius / support, 1.0 / support);
        near(derivative, (density(support + 1e-6) - density(support - 1e-6)) / 2e-6, 2e-6,
            "support derivative");
        near(radial_kernel::value(floatTable, float(radius)), radial_kernel::value(table, radius),
            2e-5, "float kernel");
    }
    auto const kernel = makeHydroKernelTable<double>(kCubicKernel);
    near(monaghan::lengthPerSupport(double(kernel.smoothingLengthPerEffectiveRadius)), 0.5, 1e-14,
        "cubic viscosity length");
}

struct State {
    double support;       // Compact kernel radius in test length units.
    double density;       // Mass density in the selected spatial dimension.
    double correction;    // Weighted support correction for equal unit particle masses.
    double pressure;      // Ideal-gas pressure with unit entropy and gamma = 5/3.
};

std::vector<State> states(std::vector<SpatialVector> const& positions) {
    auto const kernel = makeHydroKernelTable<double>(kCubicKernel);
    std::vector<State> result;
    double const target = 2.5 * radial_kernel::value(kernel, 0.0);
    for (auto const& position : positions) {
        double lower = 1e-6;
        double upper = 100;
        constexpr int kSupportSolveIterations = 70;
        for (int iteration = 0; iteration < kSupportSolveIterations; ++iteration) {
            double const support = (lower + upper) / 2;
            double sum = 0;
            for (auto const& neighbor : positions) {
                sum += radial_kernel::value(kernel, (position - neighbor).norm() / support);
            }
            (sum < target ? lower : upper) = support;
        }
        double const support = (lower + upper) / 2;
        sph_volume::SupportSums<double> sums{};
        for (auto const& neighbor : positions) {
            double const radius = (position - neighbor).norm() / support;
            sph_volume::accumulateSupportSums(sums, 1.0, radial_kernel::value(kernel, radius),
                radial_kernel::derivative(kernel, radius), radius);
        }
        double const density = sph_volume::weightedDensity(sums, 1.0 / support);
        auto const correction =
            sph_volume::gradHCorrection(sums, sph_volume::SupportConstraint::Free);
        result.push_back(
            {support, density, correction.weightedSupportCorrection, std::pow(density, 5.0 / 3.0)});
    }
    return result;
}

void variationalForce() {
    auto const kernel = makeHydroKernelTable<double>(kCubicKernel);
    constexpr size_t kLatticeSide = 4;
    size_t count = 1;
    for (int dim = 0; dim < spatialDimensions(); ++dim) {
        count *= kLatticeSide;
    }
    std::vector<SpatialVector> positions(count, SpatialVector(0.0));
    for (size_t particle = 0; particle < count; ++particle) {
        size_t index = particle;
        for (int dim = 0; dim < spatialDimensions(); ++dim) {
            positions[particle][dim] =
                0.2 * (index % kLatticeSide) + 0.009 * std::sin(particle + 3 * dim);
            index /= kLatticeSide;
        }
    }
    auto const baseline = states(positions);
    SpatialVector force(0.0);
    for (size_t neighbor = 1; neighbor < count; ++neighbor) {
        auto const separation = positions[0] - positions[neighbor];
        double const radius = separation.norm();
        auto const& first = baseline[0];
        auto const& second = baseline[neighbor];
        double const firstGradient = sph_volume::kernelGradientFactor(
            radial_kernel::derivative(kernel, radius / first.support), 1 / first.support, radius);
        double const secondGradient = sph_volume::kernelGradientFactor(
            radial_kernel::derivative(kernel, radius / second.support), 1 / second.support, radius);
        double const firstCoefficient =
            first.pressure / (first.density * first.density) * (1 - first.correction);
        double const secondCoefficient =
            second.pressure / (second.density * second.density) * (1 - second.correction);
        force -= separation *
            variational::pairForceFactor(
                firstCoefficient, secondCoefficient, firstGradient, secondGradient);
    }
    auto const energy = [&](std::vector<SpatialVector> const& coordinates) {
        double total = 0;
        for (auto const& state : states(coordinates)) {
            total += std::pow(state.density, 2.0 / 3.0) / (2.0 / 3.0);
        }
        return total;
    };
    for (int dim = 0; dim < spatialDimensions(); ++dim) {
        auto plus = positions;
        auto minus = positions;
        plus[0][dim] += 1e-6;
        minus[0][dim] -= 1e-6;
        double const reference = -(energy(plus) - energy(minus)) / 2e-6;
        near(force[dim], reference, 2e-5 * std::max(1.0, std::abs(reference)), "variational force");
    }
}

}    // namespace detail

int main() {
    // Revisit previous dimensions to expose cached normalization or calibration values.
    for (int dimension : {1, 2, 3, 1, 3}) {
        setSpatialDimensions(dimension);
        detail::geometry();
        detail::kernels();
        detail::variationalForce();
        std::cout << "D=" << spatialDimensions()
                  << " runtime geometry, kernel, grad-H and variational force checks passed\n";
    }
}
