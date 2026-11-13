/* CosmicTiger - owned-source arithmetic and transport checks; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>

// Instantiate only the source construction definitions, independently of the
// application's walk, particle arena, configured order and position width.
#define COSMICTIGER_FMM_INSTANCE_ORDER ORDER
#include <cosmictiger/gravity/tree_multipole.hpp>

namespace detail {
size_t checks = 0;
double worst = 0;
void check(double actual, double expected, double scale) {
    ++checks;
    double const error = std::abs(actual - expected) / std::max(scale, 1.e-250);
    worst = std::max(worst, error);
    if (!std::isfinite(error) || error > 3.e-12) {
        std::fprintf(stderr, "source q=%d fixed%d: %.17g != %.17g scale %.3e error %.3e\n",
            kFmmOrder, kPositionBits, actual, expected, scale, error);
        throw std::runtime_error("owned-source arithmetic mismatch");
    }
}
}    // namespace detail

void cosmictigerAssert(char const*, bool value, char const*, int) {
    if (!value) {
        std::abort();
    }
}
// The fixture supplies explicit borrowed masses and positions; any fallback to
// the application's particle store is an error, rather than a hidden oracle.
Options const& getOptions() {
    std::abort();
}
char* particlesTy = nullptr;

namespace detail {
/// Compare a two-level source reduction with direct Cartesian moments at the root.
template <int order>
void cloud(double extent, bool subDoubleCenters) {
    namespace cartesian = cosmictiger::gravity::cartesian;
    std::array<std::array<Position, 9>, kDimensions> positions;
    std::array<float, 9> masses{.125f, .75f, 1.25f, .375f, 2.5f, .25f, .625f, 1.75f, .875f};
    Vector3<Position> center, leftCenter, rightCenter;
    for (int axis = 0; axis < kDimensions; ++axis) {
        double const coordinate = 1. - .5 * extent;
        center[axis] = coordinate + .11 * extent;
        leftCenter[axis] = center[axis].toDouble() - .07 * extent;
        rightCenter[axis] = center[axis].toDouble() + .05 * extent;
        if (subDoubleCenters) {
            center[axis].raw() = Position::max().raw() - 64;
            leftCenter[axis].raw() = center[axis].raw() - 4;
            rightCenter[axis].raw() = center[axis].raw() + 6;
        }
        for (size_t i = 0; i < masses.size(); ++i) {
            if (subDoubleCenters) {
                positions[axis][i].raw() = center[axis].raw() - 13 + (i * (axis + 3)) % 29;
            } else {
                positions[axis][i] = coordinate + (.027 * ((i * (axis + 3)) % 17) - .21) * extent;
            }
        }
    }
    FmmSourceView sources;
    sources.count = masses.size();
    sources.masses = masses.data();
    for (int axis = 0; axis < kDimensions; ++axis) {
        sources.positions[axis] = positions[axis].data();
    }
    SourceMultipoleBuffer left, right;
    treeComputeMultipole<order>(left, {0, 4}, leftCenter, sources);
    treeComputeMultipole<order>(right, {4, 9}, rightCenter, sources);
    SourceMultipoleBuffer root;
    treeComputeMultipole<order>(root, left, right, leftCenter, rightCenter, center);
    cartesian::Multipole<double> expected{};
    double maximumOffset = 0;
    for (size_t i = 0; i < masses.size(); ++i) {
        Vector3<double> offset;
        for (int axis = 0; axis < kDimensions; ++axis) {
            offset[axis] = unwrappedDoubleDistance(positions[axis][i], center[axis]);
        }
        auto const scaledOffset = fmmScaleDisplacement(offset);
        for (int axis = 0; axis < kDimensions; ++axis) {
            maximumOffset = std::max(maximumOffset, std::abs(scaledOffset[axis]));
        }
        cartesian::P2M(double(masses[i]), scaledOffset, expected);
    }
    double rankScale = expected[0];
    for (int n = 0; n <= order - 2; ++n) {
        if (n) {
            rankScale *= maximumOffset;
        }
        for (int i = n * n; i < (n + 1) * (n + 1); ++i) {
            check(root[i], expected[i], rankScale);
        }
    }
}

void transport() {
    SourceMultipoleBuffer value;
    int const count = fmmMultipoleCoefficientCount(kFmmOrder);
    for (int i = 0; i < count; ++i) {
        value[i] = i + .125 + std::ldexp(1., -36);
    }
    auto roundTrip = [&](SourceMultipoleBuffer const& input) {
        std::vector<char> data;
        {
            hpx::serialization::output_archive archive(data);
            archive << input;
        }
        SourceMultipoleBuffer output;
        hpx::serialization::input_archive archive(data);
        archive >> output;
        for (int i = 0; i < count; ++i) {
            check(output[i], input[i], 1);
        }
    };
    roundTrip(value);
    auto const narrowed = value.narrow();
    for (int i = 0; i < count; ++i) {
        if (narrowed[i] != float(value[i])) {
            throw std::runtime_error("source moment narrowing mismatch");
        }
    }
}
}    // namespace detail

int main() {
    try {
        detail::transport();
        for (int inverseExtent : {1, 2, 3, 4, 5, 7, 8, 12, 30, 32, 1024}) {
            detail::cloud<kFmmOrder>(1. / inverseExtent, false);
        }
        detail::cloud<kFmmOrder>(1, true);
        std::printf(
            "Owned source fixed%d build order: %zu checks passed, worst rank-scaled error %.3e\n",
            kPositionBits, detail::checks, detail::worst);
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
