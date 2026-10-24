#include <array>
#include <cmath>
#include <numbers>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/particle_mesh_cloud.hpp>
#include <cosmictiger/math/radial_density_kernel.hpp>

TEST(ParticleMeshCloud, PartitionAndCentroid) {
    for (int order = 1; order <= 4; ++order) {
        for (int sample = -100; sample <= 100; ++sample) {
            double const position = sample / 37.0;
            auto const cloud = particle_mesh::cloudWeights(position, order);
            double sum = 0;
            double moment = 0;
            for (int index = 0; index < cloud.count; ++index) {
                EXPECT_GE(cloud.weights[index], 0);
                sum += cloud.weights[index];
                moment += cloud.weights[index] * (cloud.first + index);
            }
            EXPECT_NEAR(sum, 1, 4e-15);
            if (order > 1) {
                EXPECT_NEAR(moment, position, 4e-15);
            } else {
                EXPECT_LE(std::abs(moment - position), 0.5);
            }
        }
    }
}

TEST(ParticleMeshCloud, SameCardinalShapesAsSph) {
    for (int order = 3; order <= 4; ++order) {
        auto const table =
            radial_kernel::makeDensityKernelTable<1>({radial_kernel::Family::Spline, order - 1});
        double const radius = order / 2.0;
        for (int sample = 0; sample < 101; ++sample) {
            double const position = sample / 100.0;
            auto const cloud = particle_mesh::cloudWeights(position, order);
            for (int index = 0; index < cloud.count; ++index) {
                double const separation = std::abs(position - cloud.first - index);
                double const reference = radial_kernel::value(table, separation / radius) / radius;
                EXPECT_NEAR(cloud.weights[index], reference, 4e-15);
            }
        }
    }
}

TEST(ParticleMeshCloud, PeriodicWrapAndGridTranslation) {
    for (int order = 1; order <= 4; ++order) {
        for (double position : {-0.125, 0.0, 0.5, 7.875, 8.0}) {
            auto const original = particle_mesh::cloudWeights(position, order);
            auto const translated = particle_mesh::cloudWeights(position + 8, order);
            for (int index = 0; index < order; ++index) {
                EXPECT_EQ(particle_mesh::periodicIndex(original.first + index, 8),
                    particle_mesh::periodicIndex(translated.first + index, 8));
                EXPECT_DOUBLE_EQ(original.weights[index], translated.weights[index]);
            }
        }
    }
}

TEST(ParticleMeshCloud, Fixed64CellAndHalfCellBoundaries) {
    unsigned __int128 const period = static_cast<unsigned __int128>(1) << 64;
    uint64_t const belowOneThird = static_cast<uint64_t>(period / 3);
    // Conversion to double followed by multiplication rounds this to 8. The
    // exact coordinate is in cell 7, whose PCS stencil ends at node 9.
    auto const pcs = particle_mesh::cloudWeightsFromFixed(belowOneThird, 24, 4);
    EXPECT_EQ(pcs.first, 6);
    EXPECT_EQ(pcs.first + pcs.count - 1, 9);
    uint64_t const belowHalfCell = static_cast<uint64_t>(period / 48);
    EXPECT_EQ(particle_mesh::cloudWeightsFromFixed(belowHalfCell, 24, 1).first, 0);
    EXPECT_EQ(particle_mesh::cloudWeightsFromFixed(belowHalfCell + 1, 24, 1).first, 1);
    EXPECT_EQ(particle_mesh::cloudWeightsFromFixed(belowHalfCell, 24, 3).first, -1);
    EXPECT_EQ(particle_mesh::cloudWeightsFromFixed(belowHalfCell + 1, 24, 3).first, 0);
    for (int order = 1; order <= 4; ++order) {
        auto const cloud =
            particle_mesh::cloudWeightsFromFixed(std::numeric_limits<uint64_t>::max(), 24, order);
        double sum = 0;
        for (int index = 0; index < cloud.count; ++index) {
            sum += cloud.weights[index];
        }
        EXPECT_NEAR(sum, 1, 4e-15);
    }
}

TEST(ParticleMeshCloud, CompensationIncludesSourceAndSink) {
    constexpr int kMeshSize = 32;
    for (int order = 1; order <= 4; ++order) {
        for (std::array<int, 3> mode : {std::array<int, 3>{1, 0, 0}, {2, 3, 1}, {16, 0, 0}}) {
            double const argumentX = std::numbers::pi * mode[0] / kMeshSize;
            double const argumentY = std::numbers::pi * mode[1] / kMeshSize;
            double const argumentZ = std::numbers::pi * mode[2] / kMeshSize;
            auto const sinc = [](double value) { return value == 0 ? 1 : std::sin(value) / value; };
            double const window =
                std::pow(sinc(argumentX) * sinc(argumentY) * sinc(argumentZ), order);
            double const waveSquared = 4 * std::numbers::pi * std::numbers::pi *
                (mode[0] * mode[0] + mode[1] * mode[1] + mode[2] * mode[2]);
            double const compensated =
                particle_mesh::spectralPotentialFactor(kMeshSize, order, mode[0], mode[1], mode[2]);
            EXPECT_NEAR(compensated * window * window, -4 * std::numbers::pi / waveSquared, 1e-15);
        }
        EXPECT_DOUBLE_EQ(particle_mesh::spectralPotentialFactor(kMeshSize, order, 0, 0, 0), 0);
    }
}

TEST(ParticleMeshCloud, OddDerivativeAndNyquist) {
    for (int meshSize : {7, 8}) {
        for (int index = 0; index < meshSize; ++index) {
            EXPECT_DOUBLE_EQ(particle_mesh::derivativeWavenumber(index, meshSize),
                -particle_mesh::derivativeWavenumber((meshSize - index) % meshSize, meshSize));
        }
    }
    EXPECT_DOUBLE_EQ(particle_mesh::derivativeWavenumber(4, 8), 0);
    EXPECT_NE(particle_mesh::derivativeWavenumber(3, 7), 0);
}

TEST(ParticleMeshCloud, NamedCloudsAndWidths) {
    std::array<char const*, 4> const names = {"ngp", "cic", "tsc", "pcs"};
    for (int order = 1; order <= 4; ++order) {
        EXPECT_EQ(particle_mesh::cloudOrder(names[order - 1]), order);
        EXPECT_DOUBLE_EQ(particle_mesh::effectiveRadius(16, order), std::sqrt(order / 3.0) / 16);
    }
    EXPECT_THROW(particle_mesh::cloudOrder("spherical"), std::invalid_argument);
    EXPECT_THROW(particle_mesh::cloudWeights(0, 5), std::invalid_argument);
}
