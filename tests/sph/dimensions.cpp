#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cosmictiger/math/radial_density_kernel.hpp>
#include <cosmictiger/sph/geometry.hpp>
#include <cosmictiger/sph/variable_smoothing_lengths.hpp>
#include <cosmictiger/sph/viscosity/cullen_and_dehnen.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>
#include <cosmictiger/sph/volume.hpp>

namespace detail {

void near(double actual, double expected, double tolerance, char const* label) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::cerr << label << ": " << actual << " expected " << expected << '\n';
        throw std::runtime_error(label);
    }
}

using SpatialVector = sph_geometry::SpatialVector<double>;
auto const kernel =
    radial_kernel::makeDensityKernelTable<kSpatialDimensions>({radial_kernel::Family::Spline, 3});

void kernels() {
    std::vector<radial_kernel::KernelSpec> specs;
    for (int degree = 2; degree <= 9; ++degree) {
        specs.push_back({radial_kernel::Family::Spline, degree});
    }
    for (int degree = kSpatialDimensions == 1 ? 1 : 2; degree <= 9; degree += 3) {
        specs.push_back({radial_kernel::Family::Wendland, degree});
    }
    for (int degree = 2; degree <= 8; degree += 2) {
        specs.push_back({radial_kernel::Family::Ferrers, degree});
    }
    for (auto const spec : specs) {
        auto const table = radial_kernel::makeDensityKernelTable<kSpatialDimensions>(spec);
        auto const floatTable =
            radial_kernel::makeDensityKernelTable<kSpatialDimensions, float>(spec);
        if (table.pieceCount == 0 || floatTable.pieceCount == 0) {
            throw std::runtime_error("Missing dimensional kernel");
        }
        constexpr int kSamples = 32768;
        double integral = 0;
        for (int sample = 0; sample < kSamples; ++sample) {
            double const radius = (sample + 0.5) / kSamples;
            integral +=
                radial_kernel::value(table, radius) * std::pow(radius, kSpatialDimensions - 1);
        }
        near(integral * kSpatialDimensions * sph_geometry::kUnitBallVolume / kSamples, 1.0, 1e-7,
            "kernel integral");
        double const radius = 0.231;
        double const support = 0.73;
        auto const density = [&](double length) {
            return radial_kernel::value(table, radius / length) /
                std::pow(length, kSpatialDimensions);
        };
        double const derivative = sph_volume::kernelSupportDerivative(
            radial_kernel::value(table, radius / support),
            radial_kernel::derivative(table, radius / support), radius / support, 1.0 / support);
        near(derivative, (density(support + 1e-6) - density(support - 1e-6)) / 2e-6, 2e-6,
            "support derivative");
        near(radial_kernel::value(floatTable, float(radius)), radial_kernel::value(table, radius),
            2e-5, "float kernel");
    }
    near(monaghan::kViscosityWidthScale / kernel.smoothingLengthPerEffectiveRadius, 0.5, 1e-14,
        "cubic viscosity length");
}

struct State {
    double support;
    double density;
    double correction;
    double pressure;
};

std::vector<State> states(std::vector<SpatialVector> const& positions) {
    std::vector<State> result;
    double const target = 2.5 * radial_kernel::value(kernel, 0.0);
    for (auto const& position : positions) {
        double lower = 1e-6;
        double upper = 100;
        for (int iteration = 0; iteration < 70; ++iteration) {
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
    size_t count = 1;
    for (int dim = 0; dim < kSpatialDimensions; ++dim) {
        count *= 4;
    }
    std::vector<SpatialVector> positions(count);
    for (size_t particle = 0; particle < count; ++particle) {
        size_t index = particle;
        for (int dim = 0; dim < kSpatialDimensions; ++dim) {
            positions[particle][dim] = 0.2 * (index % 4) + 0.009 * std::sin(particle + 3 * dim);
            index /= 4;
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
    for (int dim = 0; dim < kSpatialDimensions; ++dim) {
        auto plus = positions;
        auto minus = positions;
        plus[0][dim] += 1e-6;
        minus[0][dim] -= 1e-6;
        double const reference = -(energy(plus) - energy(minus)) / 2e-6;
        near(force[dim], reference, 2e-5 * std::max(1.0, std::abs(reference)), "variational force");
    }
}

void gradients() {
    cullen_and_dehnen::GradientSums<double> sums;
    for (int row = 0; row < kSpatialDimensions; ++row) {
        for (int column = 0; column < kSpatialDimensions; ++column) {
            sums.moment[row][column] = row == column ? row + 1.0 : 0.0;
            sums.velocity[row][column] = (column + 1.0) * (2 * row - column + 0.25);
            sums.acceleration[row][column] = 0.0;
        }
    }
    double velocity[3][3];
    double acceleration[3][3];
    size_t operations = 0;
    int const rank =
        cullen_and_dehnen::reconstructGradients(sums, velocity, acceleration, operations);
    near(rank, kSpatialDimensions, 0, "gradient rank");
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            near(velocity[row][column],
                row < kSpatialDimensions && column < kSpatialDimensions ? 2 * row - column + 0.25 :
                                                                          0.0,
                1e-12, "affine gradient");
        }
    }
}

}    // namespace detail

int main() {
    detail::kernels();
    detail::variationalForce();
    detail::gradients();
    std::cout << "D=" << kSpatialDimensions
              << " kernel, grad-H, variational force and matrix checks passed\n";
}
