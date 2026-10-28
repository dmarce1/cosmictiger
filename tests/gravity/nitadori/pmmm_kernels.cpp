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
#include <cosmictiger/gravity/nitadori/kernels.hpp>
#include <cosmictiger/gravity/nitadori/mesh_size.hpp>

// This numerical test has no HPX runtime; fmmDispatch's failure branch still
// needs an assertion implementation even though the public API checks orders.
void cosmictigerAssert(char const* condition, bool satisfied, char const* file, int line) {
    if (!satisfied) {
        std::fprintf(stderr, "%s:%d: assertion \"%s\" failed\n", file, line, condition);
        throw std::runtime_error("PMMM kernel assertion failed");
    }
}

namespace pmmm_test {
namespace detail {

    using Complex = std::complex<double>;
    using Point = std::array<double, 3>;

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

TEST(PmmmMeshSize, ScalesCellVolumeAndRoundsTheSideUp) {
    auto const meshSize = pmmm::detail::hierarchicalMeshSize;
    EXPECT_EQ(meshSize(32, 1, 1, 1), 32);
    EXPECT_EQ(meshSize(32, 1, 1, 8), 16);
    EXPECT_EQ(meshSize(32, 1, 1, 64), 8);
    EXPECT_EQ(meshSize(32, 1, 1, 512), 4);
    EXPECT_EQ(meshSize(8, 2, 3, 7), 7);
    EXPECT_EQ(meshSize(8, 1, 64, 512), 4);
    EXPECT_EQ(meshSize(8, 1, 65, 512), 5);
    EXPECT_EQ(meshSize(9, 1, 64, 729), 4);
    EXPECT_EQ(meshSize(9, 1, 65, 729), 5);
}

TEST(PmmmMeshSize, PreservesNearGeometryAndHandlesEmptyPopulations) {
    auto const meshSize = pmmm::detail::hierarchicalMeshSize;
    for (int nearWidth = 1; nearWidth <= 4; ++nearWidth) {
        int const minimumSize = 2 * nearWidth + 1;
        EXPECT_EQ(meshSize(32, nearWidth, 1, 32768), minimumSize);
        EXPECT_EQ(meshSize(32, nearWidth, 0, 32768), minimumSize);
        EXPECT_EQ(meshSize(32, nearWidth, 0, 0), minimumSize);
        EXPECT_EQ(meshSize(minimumSize, nearWidth, 1, 8), minimumSize);
        EXPECT_EQ(meshSize(minimumSize, nearWidth, 8, 8), minimumSize);
    }
}

TEST(PmmmMeshSize, ResolvesCubeBoundariesWithLargeParticleCounts) {
    constexpr uint64_t kTotal = uint64_t(1) << 63;
    constexpr uint64_t kEighth = kTotal / 8;
    constexpr uint64_t kMaximum = std::numeric_limits<uint64_t>::max();
    auto const meshSize = pmmm::detail::hierarchicalMeshSize;
    EXPECT_EQ(meshSize(32, 1, kEighth - 1, kTotal), 16);
    EXPECT_EQ(meshSize(32, 1, kEighth, kTotal), 16);
    EXPECT_EQ(meshSize(32, 1, kEighth + 1, kTotal), 17);
    EXPECT_EQ(meshSize(32, 1, kMaximum, kMaximum), 32);
    EXPECT_EQ(meshSize(32, 1, kMaximum / 8, kMaximum), 16);
}

TEST(PmmmMeshSize, RejectsInvalidGeometryAndCounts) {
    auto const meshSize = pmmm::detail::hierarchicalMeshSize;
    EXPECT_THROW(meshSize(4, 2, 1, 1), std::invalid_argument);
    EXPECT_THROW(meshSize(8, 0, 1, 1), std::invalid_argument);
    EXPECT_THROW(meshSize(8, 1, 2, 1), std::invalid_argument);
    EXPECT_THROW(meshSize(8, 1, 1, 0), std::invalid_argument);
}

TEST(PmmmKernels, MomentsUseNegativeCellOffsetAndAccumulate) {
    std::array<detail::Point, 3> const offsets{
        {{0.25, -0.375, 0.125}, {-0.2, 0.1, -0.3}, {0, 0, 0}}};
    std::array<double, 3> const masses{1.5, 0.75, 2.0};
    for (int order = kFmmMinOrder; order <= kFmmMaxOrder; ++order) {
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
    for (int order = kFmmMinOrder; order <= kFmmMaxOrder; ++order) {
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
    for (int order = kFmmMinOrder; order <= kFmmMaxOrder; ++order) {
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

TEST(PmmmKernels, GreenNearExclusionParityOriginAndCellScaling) {
    detail::Point const point{0.25, -0.125, 0.375};
    detail::Point const negative{-point[0], -point[1], -point[2]};
    constexpr double kCellWidth = 0.125;
    double const radius = std::hypot(point[0], point[1], point[2]);
    for (int order = kFmmMinOrder; order <= kFmmMaxOrder; ++order) {
        int const degree = order - 1;
        std::vector<double> full(order * order), near(order * order), opposite(order * order),
            scaled(order * order), origin(order * order);
        pmmm::green(order, full.data(), point, kCellWidth, false);
        pmmm::green(order, near.data(), point, kCellWidth, true);
        pmmm::green(order, opposite.data(), negative, kCellWidth, true);
        pmmm::green(order, scaled.data(), point, 2 * kCellWidth, true);
        pmmm::green(order, origin.data(), {0, 0, 0}, kCellWidth, true);
        auto const regular = detail::regular(degree, point);
        EXPECT_NEAR(origin[0], 2.8372974794806195, 2e-13);
        EXPECT_GT(std::abs(near[0]), 0.01);
        for (int rank = 0; rank <= degree; ++rank) {
            for (int mode = 0; mode <= rank; ++mode) {
                double const factor = (rank % 2 == 0 ? -1.0 : 1.0) *
                    detail::factorial(rank - mode) * detail::factorial(rank + mode) *
                    std::pow(kCellWidth, rank) / std::pow(radius, 2 * rank + 1);
                auto const primary = factor * regular[detail::index(rank, mode)];
                int const slot = detail::index(rank, mode);
                EXPECT_NEAR(full[slot] - near[slot], primary.real(),
                    2e-12 * std::max(1.0, std::abs(primary.real())));
                if (mode != 0) {
                    int const imaginarySlot = detail::index(rank, -mode);
                    EXPECT_NEAR(full[imaginarySlot] - near[imaginarySlot], primary.imag(),
                        2e-12 * std::max(1.0, std::abs(primary.imag())));
                }
            }
            for (int slot = rank * rank; slot < (rank + 1) * (rank + 1); ++slot) {
                EXPECT_NEAR(opposite[slot], (rank % 2 == 0 ? 1.0 : -1.0) * near[slot], 2e-12);
                EXPECT_DOUBLE_EQ(scaled[slot], std::ldexp(near[slot], rank));
                EXPECT_TRUE(std::isfinite(origin[slot]));
                if (rank % 2 != 0) {
                    EXPECT_NEAR(origin[slot], 0.0, 2e-13);
                }
            }
        }
    }
}

TEST(PmmmKernels, EvaluationMatchesIndependentPolynomialAndPhysicalGradient) {
    detail::Point const point{0.19, -0.23, 0.31};
    constexpr double kCellWidth = 0.125;
    constexpr double kTrace = 0.7;
    constexpr double kStep = 2e-5;
    for (int order = kFmmMinOrder; order <= kFmmMaxOrder; ++order) {
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
    for (int order = std::max(4, kFmmMinOrder); order <= kFmmMaxOrder; ++order) {
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
    EXPECT_THROW(pmmm::contractionTerms(kFmmMinOrder - 1), std::invalid_argument);
    EXPECT_THROW(pmmm::contractionTerms(kFmmMaxOrder + 1), std::invalid_argument);
    std::vector<double> coefficients(kFmmMaxOrder * kFmmMaxOrder);
    EXPECT_THROW(pmmm::green(kFmmMinOrder, coefficients.data(), {0, 0, 0}, 0.125, false),
        std::invalid_argument);
    EXPECT_THROW(pmmm::green(kFmmMinOrder, coefficients.data(), {0, 0, 0}, 0.0, true),
        std::invalid_argument);
    EXPECT_THROW(pmmm::green(kFmmMinOrder, coefficients.data(), {0.6, 0, 0}, 0.125, true),
        std::invalid_argument);
}

}    // namespace pmmm_test
