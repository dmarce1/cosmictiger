/* CosmicTiger - PMMM near-FMM image geometry regression tests; BSL-1.0. */

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/fmm_geometry.hpp>

void cosmictigerAssert(char const*, bool condition, char const*, int) {
    if (!condition) {
        throw std::runtime_error("FMM geometry assertion failed");
    }
}

namespace detail {

using WideInteger = __int128;
using WideUnsigned = unsigned __int128;
constexpr WideInteger kPositionUnit = WideInteger(1) << kPositionBits;

Position rawPosition(PositionInteger raw) {
    Position result;
    result.setInteger(raw);
    return result;
}

int exactCell(Position source, int meshSize) {
    return int(WideUnsigned(source.raw()) * unsigned(meshSize) / WideUnsigned(kPositionUnit));
}

// Choose among the three source images by integer cell-center separation.
// The even-mesh tie follows the convolution's positive half-mesh displacement.
int referenceImage(Position source, int dim, FmmGeometry const& geometry) {
    int const sourceCell = exactCell(source, geometry.meshSize);
    int bestImage = 0;
    int bestDistance = std::numeric_limits<int>::max();
    int bestDisplacement = 0;
    for (int image = -1; image <= 1; ++image) {
        int const displacement = geometry.sinkCell[dim] - sourceCell - image * geometry.meshSize;
        if (std::abs(displacement) < bestDistance ||
            (std::abs(displacement) == bestDistance && displacement > bestDisplacement)) {
            bestDistance = std::abs(displacement);
            bestDisplacement = displacement;
            bestImage = image;
        }
    }
    return bestImage;
}

float referenceDifference(Position sink, Position source, int dim, FmmGeometry const& geometry) {
    WideInteger difference = WideInteger(sink.raw()) - WideInteger(source.raw());
    if (geometry.meshSize == 0) {
        if (difference >= kPositionUnit / 2) {
            difference -= kPositionUnit;
        } else if (difference < -kPositionUnit / 2) {
            difference += kPositionUnit;
        }
    } else {
        difference -= WideInteger(referenceImage(source, dim, geometry)) * kPositionUnit;
    }
    return float(std::ldexp(static_cast<long double>(difference), -kPositionBits));
}

float referenceBox(Fixed32Range const& box, Array<Position, kDimensions> const& point,
    Array<int, kDimensions> const& image) {
    long double squared = 0;
    constexpr WideInteger kBoundaryUnit = WideInteger(1) << (kPositionBits - 29);
    for (int dim = 0; dim < kDimensions; ++dim) {
        WideInteger const coordinate =
            WideInteger(point[dim].raw()) + WideInteger(image[dim]) * kPositionUnit;
        WideInteger const lower = WideInteger(box.begin[dim].raw()) * kBoundaryUnit;
        WideInteger const upper = WideInteger(box.end[dim].raw()) * kBoundaryUnit;
        WideInteger const outside =
            std::max(WideInteger(0), std::max(lower - coordinate, coordinate - upper));
        long double const distance = std::ldexp(static_cast<long double>(outside), -kPositionBits);
        squared += distance * distance;
    }
    return float(squared);
}

void checkDifferences(std::vector<Position> const& sinks, std::vector<Position> const& sources,
    FmmGeometry const& geometry, int dim = kXDimension) {
    ASSERT_EQ(sinks.size(), sources.size());
    for (size_t begin = 0; begin < sinks.size(); begin += kSimdFloatSize) {
        SimdPosition sinkPack{}, sourcePack{};
        for (int lane = 0; lane < kSimdFloatSize; ++lane) {
            size_t const index = std::min(begin + lane, sinks.size() - 1);
            simdSetLane(sinkPack, lane, std::bit_cast<SignedPositionInteger>(sinks[index].raw()));
            simdSetLane(
                sourcePack, lane, std::bit_cast<SignedPositionInteger>(sources[index].raw()));
        }
        auto const actualPack = fmmPositionDifference(sinkPack, sourcePack, dim, geometry);
        for (int lane = 0; lane < kSimdFloatSize && begin + lane < sinks.size(); ++lane) {
            size_t const index = begin + lane;
            float const expected = referenceDifference(sinks[index], sources[index], dim, geometry);
            SCOPED_TRACE(::testing::Message() << "bits=" << kPositionBits << " mesh="
                                              << geometry.meshSize << " sample=" << index);
            EXPECT_FLOAT_EQ(
                fmmPositionDifference(sinks[index], sources[index], dim, geometry), expected);
            EXPECT_FLOAT_EQ(actualPack[lane], expected);
        }
    }
}

void checkBoxes(FmmGeometry const& geometry, std::array<Fixed32Range, kSimdFloatSize> const& boxes,
    std::array<Array<Position, kDimensions>, kSimdFloatSize> const& sinks,
    std::array<Array<Position, kDimensions>, kSimdFloatSize> const& sources) {
    Range<SimdDouble8> boxPack{};
    Array<SimdPosition, kDimensions> sinkPack{}, sourcePack{};
    for (int lane = 0; lane < kSimdFloatSize; ++lane) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            simdSetLane(boxPack.begin[dim], lane, boxes[lane].begin[dim].toDouble());
            simdSetLane(boxPack.end[dim], lane, boxes[lane].end[dim].toDouble());
            simdSetLane(
                sinkPack[dim], lane, std::bit_cast<SignedPositionInteger>(sinks[lane][dim].raw()));
            simdSetLane(sourcePack[dim], lane,
                std::bit_cast<SignedPositionInteger>(sources[lane][dim].raw()));
        }
    }
    auto const sinkDistances = fmmSinkBoxDistanceSquared(boxPack, sourcePack, geometry);
    auto const sourceDistances =
        fmmSourceBoxDistanceSquared(boxPack, sinkPack, sourcePack, geometry);
    for (int lane = 0; lane < kSimdFloatSize; ++lane) {
        Array<int, kDimensions> image;
        for (int dim = 0; dim < kDimensions; ++dim) {
            image[dim] = referenceImage(sources[lane][dim], dim, geometry);
        }
        float const expectedSink = referenceBox(boxes[lane], sources[lane], image);
        for (int dim = 0; dim < kDimensions; ++dim) {
            image[dim] = -image[dim];
        }
        float const expectedSource = referenceBox(boxes[lane], sinks[lane], image);
        SCOPED_TRACE(::testing::Message()
            << "bits=" << kPositionBits << " mesh=" << geometry.meshSize << " lane=" << lane);
        EXPECT_FLOAT_EQ(
            fmmSinkBoxDistanceSquared(boxes[lane], sources[lane], geometry), expectedSink);
        EXPECT_FLOAT_EQ(sinkDistances[lane], expectedSink);
        EXPECT_FLOAT_EQ(
            fmmSourceBoxDistanceSquared(boxes[lane], sinks[lane], sources[lane], geometry),
            expectedSource);
        EXPECT_FLOAT_EQ(sourceDistances[lane], expectedSource);
    }
}

}    // namespace detail

TEST(FmmGeometry, DefaultKeepsNearestParticleImages) {
    PositionInteger const last = std::numeric_limits<PositionInteger>::max();
    PositionInteger const half = PositionInteger(1) << (kPositionBits - 1);
    std::vector<Position> sinks, sources;
    for (PositionInteger raw : {PositionInteger(0), PositionInteger(1), half - 3, half + 3, last}) {
        for (PositionInteger other : {PositionInteger(0), PositionInteger(1), half, last}) {
            sinks.push_back(detail::rawPosition(raw));
            sources.push_back(detail::rawPosition(other));
        }
    }
    detail::checkDifferences(sinks, sources, {});
}

TEST(FmmGeometry, NearCellImageCanExceedHalfBox) {
    for (int meshSize : {3, 5}) {
        FmmGeometry geometry;
        geometry.meshSize = meshSize;
        std::vector<Position> const sinks{Position(0.01), Position(0.02), Position(0.04)};
        std::vector<Position> const sources{Position(0.59), Position(0.58), Position(0.57)};
        detail::checkDifferences(sinks, sources, geometry);
        EXPECT_LT(fmmPositionDifference(sinks[0], sources[0], kXDimension, geometry), -0.5f);
        EXPECT_GT(fmmPositionDifference(sinks[0], sources[0], kXDimension, {}), 0.0f);
        geometry.sinkCell[kXDimension] = meshSize - 1;
        detail::checkDifferences({Position(0.99)}, {Position(0.41)}, geometry);
        EXPECT_GT(
            fmmPositionDifference(Position(0.99), Position(0.41), kXDimension, geometry), 0.5f);
    }
}

TEST(FmmGeometry, ExactCellBoundarySelectsOneImageForTheWholeCell) {
    for (int meshSize : {3, 5, 6, 7, 16}) {
        for (int sinkCell = 0; sinkCell < meshSize; ++sinkCell) {
            FmmGeometry geometry;
            geometry.meshSize = meshSize;
            geometry.sinkCell[kXDimension] = sinkCell;
            Position const sink((sinkCell + 0.5) / meshSize);
            std::vector<Position> sinks, sources;
            for (int cell = 1; cell < meshSize; ++cell) {
                auto const threshold = PositionInteger(
                    (detail::WideUnsigned(cell) * detail::WideUnsigned(detail::kPositionUnit) +
                        unsigned(meshSize - 1)) /
                    unsigned(meshSize));
                for (int offset : {-1, 0, 1}) {
                    Position const source = detail::rawPosition(threshold + offset);
                    EXPECT_EQ(detail::fmmSourceImage(source, kXDimension, geometry),
                        detail::referenceImage(source, kXDimension, geometry));
                    sinks.push_back(sink);
                    sources.push_back(source);
                }
            }
            detail::checkDifferences(sinks, sources, geometry);
        }
    }
}

TEST(FmmGeometry, LowBitsSurviveWithinCellsAndAcrossTheSeam) {
    PositionInteger const last = std::numeric_limits<PositionInteger>::max();
    PositionInteger const half = PositionInteger(1) << (kPositionBits - 1);
    for (int meshSize : {3, 5, 7}) {
        FmmGeometry geometry;
        geometry.meshSize = meshSize;
        std::vector<Position> sinks, sources;
        for (int offset = 0; offset < 2 * kSimdFloatSize; ++offset) {
            sinks.push_back(detail::rawPosition(PositionInteger(offset + 1)));
            sources.push_back(detail::rawPosition(last - offset));
        }
        detail::checkDifferences(sinks, sources, geometry);
        geometry.sinkCell[kXDimension] = meshSize - 1;
        detail::checkDifferences(sources, sinks, geometry);
        geometry.sinkCell[kXDimension] = meshSize / 2;
        for (size_t i = 0; i < sinks.size(); ++i) {
            sinks[i] = detail::rawPosition(half + i + 1);
            sources[i] = detail::rawPosition(half);
        }
        detail::checkDifferences(sinks, sources, geometry);
        EXPECT_EQ(
            fmmPositionDifference(sinks[0], sources[0], kXDimension, geometry), kPositionToFloat);
    }
}

TEST(FmmGeometry, BoxImagesAgreeWithExactIntegerReference) {
    for (int meshSize : {3, 5, 6, 7}) {
        FmmGeometry geometry;
        geometry.meshSize = meshSize;
        geometry.sinkCell[kXDimension] = 0;
        geometry.sinkCell[kYDimension] = meshSize / 2;
        geometry.sinkCell[kZDimension] = meshSize - 1;
        std::array<Fixed32Range, kSimdFloatSize> boxes;
        std::array<Array<Position, kDimensions>, kSimdFloatSize> sinks, sources;
        for (int lane = 0; lane < kSimdFloatSize; ++lane) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                int const sourceCell = (lane + dim) % meshSize;
                int const boxCell = (lane & 1) ? sourceCell : geometry.sinkCell[dim];
                boxes[lane].begin[dim] = (boxCell + 0.1) / meshSize;
                boxes[lane].end[dim] = (boxCell + 0.9) / meshSize;
                sinks[lane][dim] = (geometry.sinkCell[dim] + 0.05) / meshSize;
                sources[lane][dim] = (sourceCell + 0.95) / meshSize;
            }
        }
        detail::checkBoxes(geometry, boxes, sinks, sources);
    }
}

TEST(FmmGeometry, BoxEndpointsKeepSubDoubleSeamDistances) {
    PositionInteger const last = std::numeric_limits<PositionInteger>::max();
    for (bool upper : {false, true}) {
        FmmGeometry geometry;
        geometry.meshSize = 3;
        geometry.sinkCell[kXDimension] = upper ? 2 : 0;
        std::array<Fixed32Range, kSimdFloatSize> boxes;
        std::array<Array<Position, kDimensions>, kSimdFloatSize> sinks, sources;
        for (int lane = 0; lane < kSimdFloatSize; ++lane) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                boxes[lane].begin[dim] = boxes[lane].end[dim] = 0.0;
                sinks[lane][dim] = sources[lane][dim] = 0.0;
            }
            boxes[lane].begin[kXDimension] = boxes[lane].end[kXDimension] = upper ? 1.0 : 0.0;
            sinks[lane][kXDimension] =
                detail::rawPosition(upper ? last - lane : PositionInteger(lane + 1));
            sources[lane][kXDimension] =
                detail::rawPosition(upper ? PositionInteger(lane + 1) : last - lane);
        }
        detail::checkBoxes(geometry, boxes, sinks, sources);
        for (int lane = 0; lane < kSimdFloatSize; ++lane) {
            boxes[lane].begin[kXDimension] = boxes[lane].end[kXDimension] = upper ? 0.0 : 1.0;
        }
        detail::checkBoxes(geometry, boxes, sinks, sources);
    }
}
