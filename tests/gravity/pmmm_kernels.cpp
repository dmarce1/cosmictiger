/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp>
#include <cosmictiger/gravity/mesh_size.hpp>
#include <cosmictiger/gravity/periodic_fmm.hpp>
#include <cosmictiger/gravity/pmmm_kernels.hpp>
#include <cosmictiger/math/vector3.hpp>

// This numerical test has no HPX runtime; kernel assertions still
// needs an assertion implementation even though the public API checks orders.
void cosmictigerAssert(char const* condition, bool satisfied, char const* file, int line) {
    if (!satisfied) {
        std::fprintf(stderr, "%s:%d: assertion \"%s\" failed\n", file, line, condition);
        throw std::runtime_error("PMMM kernel assertion failed");
    }
}

#include "periodic_reference.hpp"
#include "pmmm_reference.hpp"

namespace pmmm_test {
namespace detail {

    using Complex = std::complex<double>;
    using Point = Vector3<double>;

    int index(int degree, int mode) {
        return degree * (degree + 1) + mode;
    }

    Complex harmonic(std::vector<double> const& coefficients, int degree, int mode) {
        int const magnitude = std::abs(mode);
        Complex const value(coefficients[index(degree, magnitude)],
            magnitude == 0 ? 0.0 : coefficients[index(degree, -magnitude)]);
        return mode < 0 ? (magnitude % 2 == 0 ? 1.0 : -1.0) * std::conj(value) : value;
    }

    std::vector<Complex> regular(int degree, Point const& point) {
        std::vector<Complex> result((degree + 1) * (degree + 1));
        double const radiusSquared =
            point[0] * point[0] + point[1] * point[1] + point[2] * point[2];
        result[0] = 1.0;
        for (int mode = 0; mode <= degree; ++mode) {
            if (mode != 0) {
                result[index(mode, mode)] = result[index(mode - 1, mode - 1)] *
                    Complex(point[0], point[1]) / double(2 * mode);
            }
            if (mode < degree) {
                result[index(mode + 1, mode)] = point[2] * result[index(mode, mode)];
            }
            for (int rank = mode + 2; rank <= degree; ++rank) {
                result[index(rank, mode)] =
                    (double(2 * rank - 1) * point[2] * result[index(rank - 1, mode)] -
                        radiusSquared * result[index(rank - 2, mode)]) /
                    double(rank * rank - mode * mode);
            }
        }
        return result;
    }

    double factorial(int degree) {
        double result = 1.0;
        for (int factor = 2; factor <= degree; ++factor) {
            result *= factor;
        }
        return result;
    }

    // Independent complex-harmonic sum; no real contraction table or generated
    // kernels are used. Only actual real spatial fields enter this routine.
    std::vector<double> contract(
        int order, std::vector<double> const& moments, std::vector<double> const& greens) {
        int const degree = order - 1;
        std::vector<double> result(order * order);
        for (int rank = 0; rank <= degree; ++rank) {
            for (int mode = 0; mode <= rank; ++mode) {
                Complex sum = 0.0;
                for (int sourceRank = 0; sourceRank <= order - 2; ++sourceRank) {
                    if (rank + sourceRank > degree) {
                        continue;
                    }
                    for (int sourceMode = -sourceRank; sourceMode <= sourceRank; ++sourceMode) {
                        sum += std::conj(harmonic(moments, sourceRank, sourceMode)) *
                            harmonic(greens, rank + sourceRank, mode + sourceMode);
                    }
                }
                result[index(rank, mode)] = sum.real();
                if (mode != 0) {
                    result[index(rank, -mode)] = sum.imag();
                }
            }
        }
        return result;
    }

    double potential(
        std::vector<double> const& local, int degree, Point const& point, double trace) {
        auto const values = regular(degree, point);
        double sum =
            -0.5 * trace * (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
        for (int rank = 0; rank <= degree; ++rank) {
            for (int mode = 0; mode <= rank; ++mode) {
                sum += (mode == 0 ? 1.0 : 2.0) *
                    (std::conj(harmonic(local, rank, mode)) * values[index(rank, mode)]).real();
            }
        }
        return sum;
    }

}    // namespace detail

TEST(PmmmMeshSize, RecognizesAndRoundsToPowersOfTwo) {
    using pmmm::detail::isPowerOfTwo;
    using pmmm::detail::nextPowerOfTwo;
    EXPECT_FALSE(isPowerOfTwo(0));
    EXPECT_TRUE(isPowerOfTwo(1));
    EXPECT_TRUE(isPowerOfTwo(2));
    EXPECT_FALSE(isPowerOfTwo(3));
    EXPECT_FALSE(isPowerOfTwo(12));
    EXPECT_TRUE(isPowerOfTwo(uint64_t(1) << 40));
    EXPECT_EQ(nextPowerOfTwo(0), 1);
    EXPECT_EQ(nextPowerOfTwo(1), 1);
    EXPECT_EQ(nextPowerOfTwo(3), 4);
    EXPECT_EQ(nextPowerOfTwo(5), 8);
    EXPECT_EQ(nextPowerOfTwo(8), 8);
    EXPECT_EQ(nextPowerOfTwo(9), 16);
    EXPECT_EQ(nextPowerOfTwo(1 << 30), 1 << 30);
    EXPECT_THROW(nextPowerOfTwo((1 << 30) + 1), std::overflow_error);
}

TEST(PmmmMeshSize, LocalityCountsNeedEnoughCellsPerAxis) {
    using pmmm::detail::minimumMeshSizeForLocalities;
    EXPECT_EQ(minimumMeshSizeForLocalities(1), 1);
    EXPECT_EQ(minimumMeshSizeForLocalities(2), 2);
    EXPECT_EQ(minimumMeshSizeForLocalities(4), 2);
    EXPECT_EQ(minimumMeshSizeForLocalities(8), 2);
    EXPECT_EQ(minimumMeshSizeForLocalities(16), 4);
    EXPECT_EQ(minimumMeshSizeForLocalities(64), 4);
    EXPECT_EQ(minimumMeshSizeForLocalities(128), 8);
    EXPECT_EQ(minimumMeshSizeForLocalities(512), 8);
    EXPECT_EQ(minimumMeshSizeForLocalities(4096), 16);
    EXPECT_THROW(minimumMeshSizeForLocalities(0), std::invalid_argument);
    EXPECT_THROW(minimumMeshSizeForLocalities(3), std::invalid_argument);
    EXPECT_THROW(minimumMeshSizeForLocalities(6), std::invalid_argument);
}

TEST(PmmmMeshSize, OpeningStencilMatchesFilledCellSpheres) {
    using pmmm::detail::nearCellOffset;
    // Exact boundary: theta=1/2 gives radius squared 12.
    EXPECT_TRUE(nearCellOffset(2, 2, 2, 0.5));
    EXPECT_TRUE(nearCellOffset(2, 2, 2, std::nextafter(0.5, 0.0)));
    EXPECT_FALSE(nearCellOffset(2, 2, 2, std::nextafter(0.5, 1.0)));
    EXPECT_FALSE(nearCellOffset(3, 2, 0, 0.5));
    EXPECT_TRUE(nearCellOffset(3, 0, 0, 0.5));
    EXPECT_FALSE(nearCellOffset(3, 3, 3, 0.5));
    EXPECT_TRUE(nearCellOffset(1, 1, 1, 1.0));
    EXPECT_FALSE(nearCellOffset(2, 0, 0, 1.0));
    int defaultCount = 0;
    for (int x = -3; x <= 3; ++x) {
        for (int y = -3; y <= 3; ++y) {
            for (int z = -3; z <= 3; ++z) {
                defaultCount += int(nearCellOffset(x, y, z, 0.5773502691896257));
            }
        }
    }
    EXPECT_EQ(defaultCount, 123);
    for (int side : {1, 2, 3, 4, 5, 8, 12}) {
        for (double theta : {1.0, 0.7, 0.5, 0.3, 1.e-30}) {
            int const width = pmmm::detail::nearStencilWidth(side, theta);
            std::vector<int> visits(side * side * side, 0);
            for (int x = pmmm::detail::nearOffsetBegin(side, width);
                x <= pmmm::detail::nearOffsetEnd(side, width); ++x) {
                for (int y = pmmm::detail::nearOffsetBegin(side, width);
                    y <= pmmm::detail::nearOffsetEnd(side, width); ++y) {
                    for (int z = pmmm::detail::nearOffsetBegin(side, width);
                        z <= pmmm::detail::nearOffsetEnd(side, width); ++z) {
                        if (nearCellOffset(x, y, z, theta)) {
                            ++visits[(((x + side) % side) * side + (y + side) % side) * side +
                                (z + side) % side];
                        }
                    }
                }
            }
            double minimumSquared = std::numeric_limits<double>::infinity();
            for (int x = 0; x < side; ++x) {
                for (int y = 0; y < side; ++y) {
                    for (int z = 0; z < side; ++z) {
                        double const dx = std::min(x, side - x);
                        double const dy = std::min(y, side - y);
                        double const dz = std::min(z, side - z);
                        // Independent enclosing-sphere separation oracle.
                        bool const near =
                            std::sqrt(dx * dx + dy * dy + dz * dz) * theta <= std::sqrt(3.0);
                        EXPECT_EQ(visits[(x * side + y) * side + z], int(near));
                        if (!near) {
                            double const bx = std::max(dx - 1, 0.0);
                            double const by = std::max(dy - 1, 0.0);
                            double const bz = std::max(dz - 1, 0.0);
                            minimumSquared = std::min(minimumSquared, bx * bx + by * by + bz * bz);
                        }
                    }
                }
            }
            EXPECT_DOUBLE_EQ(
                pmmm::detail::nearStencilSupport(side, theta), std::sqrt(minimumSquared) / side);
        }
    }
}

TEST(PmmmMeshSize, NearOffsetsVisitEachPeriodicCellOnce) {
    for (int side = 1; side <= 12; ++side) {
        for (int width = 1; width <= 4; ++width) {
            std::vector<int> visits(side, 0);
            for (int offset = pmmm::detail::nearOffsetBegin(side, width);
                offset <= pmmm::detail::nearOffsetEnd(side, width); ++offset) {
                ++visits[(offset + side) % side];
            }
            int count = 0;
            for (int cell = 0; cell < side; ++cell) {
                EXPECT_LE(visits[cell], 1);
                EXPECT_EQ(visits[cell] != 0, std::min(cell, side - cell) <= width);
                count += visits[cell];
            }
            EXPECT_EQ(count, std::min(side, 2 * width + 1));
        }
    }
}

TEST(PmmmKernels, MomentsUseNegativeCellOffsetAndAccumulate) {
    std::array<detail::Point, 3> const offsets{
        {{0.25, -0.375, 0.125}, {-0.2, 0.1, -0.3}, {0, 0, 0}}};
    std::array<double, 3> const masses{1.5, 0.75, 2.0};
    for (int order = kFmmOrder; order <= kFmmOrder; ++order) {
        int const degree = order - 2;
        std::vector<double> moments((degree + 1) * (degree + 1), 0.125);
        std::vector<double> expected = moments;
        for (size_t particle = 0; particle < offsets.size(); ++particle) {
            auto point = offsets[particle];
            pmmm::addMoments(order, moments.data(), masses[particle], point);
            for (auto& coordinate : point) {
                coordinate = -coordinate;
            }
            auto const values = detail::regular(degree, point);
            for (int rank = 0; rank <= degree; ++rank) {
                for (int mode = 0; mode <= rank; ++mode) {
                    expected[detail::index(rank, mode)] +=
                        masses[particle] * values[detail::index(rank, mode)].real();
                    if (mode != 0) {
                        expected[detail::index(rank, -mode)] +=
                            masses[particle] * values[detail::index(rank, mode)].imag();
                    }
                }
            }
        }
        for (size_t slot = 0; slot < moments.size(); ++slot) {
            EXPECT_NEAR(moments[slot], expected[slot], 2e-15);
        }
    }
}

TEST(PmmmKernels, RealContractionMatchesComplexSumAndTotalDegreeTruncation) {
    for (int order = kFmmOrder; order <= kFmmOrder; ++order) {
        std::vector<double> moments((order - 1) * (order - 1));
        std::vector<double> greens(order * order);
        for (size_t slot = 0; slot < moments.size(); ++slot) {
            moments[slot] = std::sin(0.37 * double(slot + 1));
        }
        for (size_t slot = 0; slot < greens.size(); ++slot) {
            greens[slot] = std::cos(0.29 * double(slot + 2));
        }
        auto const expected = detail::contract(order, moments, greens);
        std::vector<double> actual(order * order);
        auto const terms = pmmm::contractionTerms(order);
        for (auto const& term : terms) {
            int const sourceDegree = int(std::sqrt(double(term.source)));
            int const localDegree = int(std::sqrt(double(term.local)));
            EXPECT_LE(sourceDegree + localDegree, order - 1);
            EXPECT_EQ(int(std::sqrt(double(term.green))), sourceDegree + localDegree);
            actual[term.local] += term.weight * moments[term.source] * greens[term.green];
        }
        for (size_t slot = 0; slot < actual.size(); ++slot) {
            EXPECT_NEAR(actual[slot], expected[slot], 5e-13);
        }
        // Highest local rank receives only source monopoles at this total order.
        for (int slot = (order - 1) * (order - 1); slot < order * order; ++slot) {
            EXPECT_DOUBLE_EQ(actual[slot], moments[0] * greens[slot]);
        }
    }
}

TEST(PmmmKernels, FourierProductsMatchDirectPeriodicCellConvolution) {
    constexpr int kCells = 5;
    for (int order = kFmmOrder; order <= kFmmOrder; ++order) {
        int const sourceCount = (order - 1) * (order - 1);
        int const localCount = order * order;
        std::vector<std::vector<double>> moments(kCells, std::vector<double>(sourceCount));
        std::vector<std::vector<double>> greens(kCells, std::vector<double>(localCount));
        for (int cell = 0; cell < kCells; ++cell) {
            for (int slot = 0; slot < sourceCount; ++slot) {
                moments[cell][slot] = std::sin(0.17 * (cell + 1) * (slot + 2));
            }
            for (int slot = 0; slot < localCount; ++slot) {
                greens[cell][slot] = std::cos(0.13 * (cell + 2) * (slot + 1));
            }
        }
        std::vector<std::vector<detail::Complex>> spectrum(
            kCells, std::vector<detail::Complex>(localCount));
        auto const terms = pmmm::contractionTerms(order);
        for (int frequency = 0; frequency < kCells; ++frequency) {
            std::vector<detail::Complex> sourceSpectrum(sourceCount);
            std::vector<detail::Complex> greenSpectrum(localCount);
            for (int cell = 0; cell < kCells; ++cell) {
                auto const phase =
                    std::polar(1.0, -2.0 * std::numbers::pi * frequency * cell / kCells);
                for (int slot = 0; slot < sourceCount; ++slot) {
                    sourceSpectrum[slot] += moments[cell][slot] * phase;
                }
                for (int slot = 0; slot < localCount; ++slot) {
                    greenSpectrum[slot] += greens[cell][slot] * phase;
                }
            }
            for (auto const& term : terms) {
                spectrum[frequency][term.local] +=
                    term.weight * sourceSpectrum[term.source] * greenSpectrum[term.green];
            }
        }
        for (int target = 0; target < kCells; ++target) {
            std::vector<double> expected(localCount);
            for (int source = 0; source < kCells; ++source) {
                auto const contribution = detail::contract(
                    order, moments[source], greens[(target - source + kCells) % kCells]);
                for (int slot = 0; slot < localCount; ++slot) {
                    expected[slot] += contribution[slot];
                }
            }
            for (int slot = 0; slot < localCount; ++slot) {
                detail::Complex actual = 0.0;
                for (int frequency = 0; frequency < kCells; ++frequency) {
                    actual += spectrum[frequency][slot] *
                        std::polar(1.0, 2.0 * std::numbers::pi * frequency * target / kCells) /
                        double(kCells);
                }
                EXPECT_NEAR(actual.real(), expected[slot], 2e-12);
                EXPECT_NEAR(actual.imag(), 0.0, 2e-12);
            }
        }
    }
}

TEST(PmmmKernels, GreenNearExcludesEntirePeriodicKernel) {
    constexpr double kCellWidth = 0.125;
    for (detail::Point const point : {detail::Point{0, 0, 0}, detail::Point{.25, -.125, .375}}) {
        std::vector<double> masked(kFmmOrder * kFmmOrder, 99);
        pmmm::green(kFmmOrder, masked.data(), point, kCellWidth, true);
        for (double coefficient : masked) {
            EXPECT_DOUBLE_EQ(coefficient, 0.0);
        }
    }
    detail::Point const point{.25, -.125, .375};
    detail::Point const negative{-.25, .125, -.375};
    std::vector<double> full(kFmmOrder * kFmmOrder), opposite(full.size()), scaled(full.size());
    pmmm::green(kFmmOrder, full.data(), point, kCellWidth, false);
    pmmm::green(kFmmOrder, opposite.data(), negative, kCellWidth, false);
    pmmm::green(kFmmOrder, scaled.data(), point, 2 * kCellWidth, false);
    for (int rank = 0; rank < kFmmOrder; ++rank) {
        for (int slot = rank * rank; slot < (rank + 1) * (rank + 1); ++slot) {
            EXPECT_NEAR(opposite[slot], (rank % 2 ? -1.0 : 1.0) * full[slot], 2e-12);
            EXPECT_DOUBLE_EQ(scaled[slot], std::ldexp(full[slot], rank));
        }
    }
}

TEST(CartesianEwald, AcceptanceUsesSecondImageAndProtectsImagePlanes) {
    EXPECT_TRUE(::detail::periodicFmmAccepted({0, 0, 0}, .1f, .1f, 2, .01f));
    EXPECT_FALSE(::detail::periodicFmmAccepted({0, 0, 0}, .3f, .3f, 2, .01f));
    EXPECT_TRUE(::detail::periodicFmmAccepted({.49f, 0, 0}, .01f, .01f, 2, .01f));
    EXPECT_FALSE(::detail::periodicFmmAccepted({.49f, 0, 0}, .15f, .15f, 2, .01f));
    EXPECT_FALSE(::detail::periodicFmmAccepted({.4f, 0, 0}, .1f, .1f, 1, .3f));
}

TEST(CartesianEwald, FloatPointMatchesIndependentLattice) {
    for (detail::Point const point : {detail::Point{0, 0, 0}, detail::Point{1e-6, 0, 0},
             detail::Point{.1, -.2, .3}, detail::Point{.499, -.499, .499}}) {
        Vector3<float> const position{float(point[0]), float(point[1]), float(point[2])};
        auto const reference = periodicCorrectionReference(point);
        std::array<float, 4> actual{};
        ::detail::periodicFmmPoint(position, 1, true, actual[0], actual[1], actual[2], actual[3]);
        for (int component = 0; component < 4; ++component) {
            EXPECT_NEAR(actual[component], reference[component], 6e-6) << component;
        }
    }
}

TEST(CartesianEwald, FloatMultipoleAndCurvatureMatchDisplacedSources) {
    if constexpr (kFmmOrder < 5) {
        GTEST_SKIP() << "This accuracy fixture requires potential degree at least four";
    }
    namespace cartesian = cosmictiger::gravity::cartesian;
    Vector3<float> const sources[] = {{.01f, -.015f, .02f}, {-.02f, .005f, -.01f}};
    float const masses[] = {1.25f, .75f};
    cartesian::Multipole<float> moments{};
    float trace = 0;
    for (int i = 0; i < 2; ++i) {
        Vector3<float> const scaled = kScaleFactor * sources[i];
        cartesian::P2M(masses[i], scaled, moments);
        trace += masses[i] * scaled.dot(scaled);
    }
    for (Vector3<float> const center : {Vector3<float>{0, 0, 0}, Vector3<float>{.3f, -.2f, .1f}}) {
        cartesian::Expansion<float> local{};
        float curvature = 0;
        ::detail::periodicFmmM2L(moments, trace, center, true, local, curvature);
        Vector3<float> const offset{.007f, -.011f, .013f};
        Vector3<float> const scaled = kScaleFactor * offset;
        cosmictiger::gravity::Field<float> field{};
        cartesian::L2P(local, scaled, field);
        field[0] += .5f * curvature * scaled[2] * scaled[2];
        field[3] -= curvature * scaled[2];
        std::array<double, 4> reference{};
        for (int i = 0; i < 2; ++i) {
            detail::Point const separation{double(center[0]) + offset[0] - sources[i][0],
                double(center[1]) + offset[1] - sources[i][1],
                double(center[2]) + offset[2] - sources[i][2]};
            auto const correction = periodicCorrectionReference(separation);
            for (int component = 0; component < 4; ++component) {
                reference[component] += masses[i] * correction[component];
            }
        }
        for (int component = 0; component < 4; ++component) {
            float const scale = component == 0 ? kScaleFactor : kScaleFactor2;
            EXPECT_NEAR(scale * field[component], reference[component], 2e-5) << component;
        }
    }
}

TEST(PmmmKernels, EvaluationMatchesIndependentPolynomialAndPhysicalGradient) {
    detail::Point const point{0.19, -0.23, 0.31};
    constexpr double kCellWidth = 0.125;
    constexpr double kTrace = 0.7;
    constexpr double kStep = 2e-5;
    for (int order = kFmmOrder; order <= kFmmOrder; ++order) {
        std::vector<double> local(order * order);
        for (size_t slot = 0; slot < local.size(); ++slot) {
            local[slot] = std::sin(0.27 * double(slot + 1));
        }
        std::array<double, 4> field{99, 99, 99, 99};
        pmmm::evaluate(order, field, local.data(), kTrace, point, kCellWidth);
        EXPECT_NEAR(field[0], detail::potential(local, order - 1, point, kTrace), 2e-14);
        for (int dimension = 0; dimension < 3; ++dimension) {
            auto plus = point;
            auto minus = point;
            plus[dimension] += kStep;
            minus[dimension] -= kStep;
            double const gradient = (detail::potential(local, order - 1, plus, kTrace) -
                                        detail::potential(local, order - 1, minus, kTrace)) /
                (2 * kStep * kCellWidth);
            EXPECT_NEAR(field[dimension + 1], -gradient, 2e-8);
        }
    }
}

TEST(PmmmKernels, BackgroundUsesDipolesAndSourceTraceAndCancelsSelfGradient) {
    constexpr double kMass = 2.5;
    constexpr double kCellWidth = 0.125;
    constexpr double kBackground = 4.0 * std::numbers::pi / 3.0 * kCellWidth * kCellWidth;
    detail::Point const offset{0.25, -0.125, 0.375};
    for (int order = std::max(4, kFmmOrder); order <= kFmmOrder; ++order) {
        std::vector<double> moments((order - 1) * (order - 1));
        std::vector<double> local(order * order);
        pmmm::addMoments(order, moments.data(), kMass, offset);
        double const sourceTrace =
            kMass * (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]);
        local[0] = -0.5 * kBackground * sourceTrace;
        local[3] = -2 * kBackground * moments[3];
        local[1] = -2 * kBackground * moments[1];
        local[2] = -kBackground * moments[2];
        std::array<double, 4> field;
        pmmm::evaluate(order, field, local.data(), kBackground * moments[0], offset, kCellWidth);
        for (double const value : field) {
            EXPECT_NEAR(value, 0.0, 2e-14);
        }
    }
}

TEST(PmmmKernels, RejectsInvalidOrderWidthAndPrimaryOrigin) {
    EXPECT_THROW(pmmm::contractionTerms(kFmmOrder - 1), std::invalid_argument);
    EXPECT_THROW(pmmm::contractionTerms(kFmmOrder + 1), std::invalid_argument);
    std::vector<double> coefficients(kFmmOrder * kFmmOrder);
    EXPECT_THROW(pmmm::green(kFmmOrder, coefficients.data(), {0, 0, 0}, 0.125, false),
        std::invalid_argument);
    EXPECT_THROW(
        pmmm::green(kFmmOrder, coefficients.data(), {0, 0, 0}, 0.0, true), std::invalid_argument);
    EXPECT_THROW(pmmm::green(kFmmOrder, coefficients.data(), {0.6, 0, 0}, 0.125, true),
        std::invalid_argument);
}

}    // namespace pmmm_test
