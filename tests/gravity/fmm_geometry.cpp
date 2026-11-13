/* CosmicTiger - periodic FMM geometry regression tests; BSL-1.0. */

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/fmm_geometry.hpp>
#include <cosmictiger/math/vector3.hpp>

TEST(FmmGeometryFlops, DisplacementUsesSinglePrecision) {
    auto check = [](Position sink, Position source) {
        FlopCounts counts{};
        float const counted = fmmPositionDifference(sink, source, &counts);
        EXPECT_EQ(counted, fmmPositionDifference(sink, source));
        EXPECT_EQ(counts.singlePrecision, kCountFlops ? 1 : 0);
        EXPECT_EQ(counts.doublePrecision, 0);
        EXPECT_EQ(counts.unclassified, 0);
    };
    check(Position(0.3), Position(0.2));
    check(Position(0.99), Position(0.01));
}

TEST(FmmGeometryFlops, BoxGeometrySeparatesDoubleCoordinatesFromFloatNorms) {
    FmmBox box;
    Vector3<Position> point;
    for (int dim = 0; dim < kDimensions; ++dim) {
        box.begin[dim] = 0.125;
        box.end[dim] = 0.25;
        point[dim] = 0.5;
    }
    FlopCounts periodic{};
    float const periodicValue = fmmBoxDistanceSquared(box, point, &periodic);
    EXPECT_EQ(periodicValue, fmmBoxDistanceSquared(box, point));
    EXPECT_EQ(periodic.singlePrecision, kCountFlops ? 6 : 0);
    EXPECT_EQ(periodic.doublePrecision, kCountFlops ? 6 : 0);
    EXPECT_EQ(periodic.unclassified, 0);
}

void cosmictigerAssert(char const*, bool condition, char const*, int) {
    if (!condition) {
        throw std::runtime_error("FMM geometry assertion failed");
    }
}

namespace detail {

using WideInteger = __int128;
constexpr WideInteger kPositionUnit = WideInteger(1) << kPositionBits;

Position rawPosition(PositionInteger raw) {
    Position result;
    result.setInteger(raw);
    return result;
}

float referenceDifference(Position sink, Position source) {
    WideInteger difference = WideInteger(sink.raw()) - WideInteger(source.raw());
    if (difference >= kPositionUnit / 2) {
        difference -= kPositionUnit;
    } else if (difference < -kPositionUnit / 2) {
        difference += kPositionUnit;
    }
    return float(std::ldexp(static_cast<long double>(difference), -kPositionBits));
}

void checkDifferences(std::vector<Position> const& sinks, std::vector<Position> const& sources) {
    ASSERT_EQ(sinks.size(), sources.size());
    for (size_t begin = 0; begin < sinks.size(); begin += kSimdFloatSize) {
        SimdPosition sinkPack{}, sourcePack{};
        for (int lane = 0; lane < kSimdFloatSize; ++lane) {
            size_t const index = std::min(begin + lane, sinks.size() - 1);
            simdSetLane(sinkPack, lane, std::bit_cast<SignedPositionInteger>(sinks[index].raw()));
            simdSetLane(
                sourcePack, lane, std::bit_cast<SignedPositionInteger>(sources[index].raw()));
        }
        auto const actualPack = fmmPositionDifference(sinkPack, sourcePack);
        for (int lane = 0; lane < kSimdFloatSize && begin + lane < sinks.size(); ++lane) {
            size_t const index = begin + lane;
            float const expected = referenceDifference(sinks[index], sources[index]);
            SCOPED_TRACE(::testing::Message() << "bits=" << kPositionBits << " sample=" << index);
            EXPECT_FLOAT_EQ(fmmPositionDifference(sinks[index], sources[index]), expected);
            EXPECT_FLOAT_EQ(actualPack[lane], expected);
        }
    }
}

}    // namespace detail

TEST(FmmGeometry, KeepsNearestParticleImages) {
    PositionInteger const last = std::numeric_limits<PositionInteger>::max();
    PositionInteger const half = PositionInteger(1) << (kPositionBits - 1);
    std::vector<Position> sinks, sources;
    for (PositionInteger raw : {PositionInteger(0), PositionInteger(1), half - 3, half + 3, last}) {
        for (PositionInteger other : {PositionInteger(0), PositionInteger(1), half, last}) {
            sinks.push_back(detail::rawPosition(raw));
            sources.push_back(detail::rawPosition(other));
        }
    }
    detail::checkDifferences(sinks, sources);
}

TEST(FmmGeometry, PeriodicBoxContainmentUsesTheWrappedInterval) {
    for (double shift : {-1.0, 0.0, 1.0}) {
        FmmBox box;
        Vector3<Position> point;
        for (int dim = 0; dim < kDimensions; ++dim) {
            box.begin[dim] = -0.125 + shift;
            box.end[dim] = 0.125 + shift;
            point[dim] = 0.9375;
        }
        EXPECT_EQ(fmmBoxDistanceSquared(box, point), 0.0f);
        point[0] = 0.375;
        EXPECT_EQ(fmmBoxDistanceSquared(box, point), 0.0625f);
        Range<SimdDouble8> packedBox;
        Vector3<SimdPosition> packedPoint;
        for (int dim = 0; dim < kDimensions; ++dim) {
            packedBox.begin[dim] = math::broadcast<SimdDouble8>(box.begin[dim].toDouble());
            packedBox.end[dim] = math::broadcast<SimdDouble8>(box.begin[dim].toDouble() +
                double(box.extentTicks(dim)) * Fixed32::kToFloatFactor_);
            packedPoint[dim] = math::broadcast<SimdPosition>(point[dim].raw());
        }
        auto const distances = fmmBoxDistanceSquared(packedBox, packedPoint);
        for (int lane = 0; lane < kSimdFloatSize; ++lane) {
            EXPECT_EQ(distances[lane], 0.0625f);
        }
    }
}

TEST(FmmGeometry, DegenerateAndWideRangesChooseValidDimensions) {
    Range<int> box;
    for (int dim = 0; dim < kDimensions; ++dim) {
        box.begin[dim] = 0;
        box.end[dim] = dim == 0 ? 4 : dim + 1;
    }
    EXPECT_EQ(box.shortestDim(), 1);
    EXPECT_EQ(box.longestDim(), 0);
    box.end = box.begin;
    EXPECT_GE(box.longestDim(), 0);
    EXPECT_LT(box.longestDim(), kDimensions);
    auto const halves = box.split();
    EXPECT_EQ(halves.first.end, box.end);
    EXPECT_EQ(halves.second.begin, box.begin);
}
