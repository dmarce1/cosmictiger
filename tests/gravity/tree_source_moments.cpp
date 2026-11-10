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
#define COSMICTIGER_FMM_INSTANCE_ORDER 7
#include <cosmictiger/gravity/kernels/spherical/p2m.hpp>
#include <cosmictiger/gravity/tree_multipole.hpp>

namespace detail {
int selectedOrder = 3;
size_t checks = 0;
double worst = 0;
void check(double actual, double expected, double scale) {
    ++checks;
    double const error = std::abs(actual - expected) / std::max(scale, 1.e-250);
    worst = std::max(worst, error);
    if (!std::isfinite(error) || error > 3.e-12) {
        std::fprintf(stderr, "source q=%d fixed%d: %.17g != %.17g scale %.3e error %.3e\n",
            selectedOrder, kPositionBits, actual, expected, scale, error);
        throw std::runtime_error("owned-source arithmetic mismatch");
    }
}
}    // namespace detail

int fmmOrder() {
    return detail::selectedOrder;
}
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
/// Check rebuilt leaf and parent moments when a frozen cell spans the periodic seam.
template <int order>
void frozenCloud() {
    namespace cartesian = cosmictiger::gravity::cartesian;
    std::array<std::array<Position, 8>, kDimensions> positions;
    std::array<float, 8> masses{.125f, .75f, 1.25f, .375f, 2.5f, .25f, .625f, 1.75f};
    Vector3<Position> center, leftCenter, rightCenter;
    FmmSourceView sources;
    sources.store = FmmParticleStore::Ghost;
    sources.count = masses.size();
    sources.masses = masses.data();
    sources.geometry.meshSize = 8;
    sources.geometry.frozenMembership = true;
    for (int axis = 0; axis < kDimensions; ++axis) {
        center[axis] = .99;
        leftCenter[axis] = .96;
        rightCenter[axis] = .04;
        sources.positions[axis] = positions[axis].data();
        for (size_t i = 0; i < masses.size(); ++i) {
            positions[axis][i] = -.09 + .025 * ((i + axis) % masses.size());
        }
    }
    TreeBuildMoments left, right;
    left.useDouble();
    right.useDouble();
    treeComputeMultipole<order>(left.precise(), {0, 4}, leftCenter, sources);
    treeComputeMultipole<order>(right.precise(), {4, 8}, rightCenter, sources);
    SourceMultipoleBuffer root;
    treeComputeMultipole<order>(
        root, left, right, leftCenter, rightCenter, center, sources.geometry);
    cartesian::Multipole<double, order> expected{};
    for (size_t i = 0; i < masses.size(); ++i) {
        Vector3<double> offset;
        for (int axis = 0; axis < kDimensions; ++axis) {
            // Independent unfolding for this fixture's cell-zero frame.
            double const point = positions[axis][i].toDouble();
            offset[axis] = (point > .5 ? point - 1 : point) - (center[axis].toDouble() - 1);
        }
        cartesian::P2M(expected, double(masses[i]), fmmScaleDisplacement(offset));
    }
    for (int i = 0; i < expected.size(); ++i) {
        check(root[i], expected[i], std::max(1., std::abs(expected[i])));
    }
}

template <int order>
void cloud(int meshSize, bool subDoubleCenters) {
    std::array<std::array<Position, 9>, 3> positions;
    std::array<float, 9> masses{.125f, .75f, 1.25f, .375f, 2.5f, .25f, .625f, 1.75f, .875f};
    Vector3<Position> center, leftCenter, rightCenter, meshCenter;
    Vector3<double> centerCorrection;
    for (int axis = 0; axis < 3; ++axis) {
        double const coordinate = 1. - .5 / meshSize;
        meshCenter[axis] = coordinate;
        centerCorrection[axis] = meshCenter[axis].toDouble() - coordinate;
        center[axis] = meshCenter[axis].toDouble() + .11 / meshSize;
        leftCenter[axis] = center[axis].toDouble() - .07 / meshSize;
        rightCenter[axis] = center[axis].toDouble() + .05 / meshSize;
        if (subDoubleCenters) {
            center[axis].raw() = Position::max().raw() - 64;
            meshCenter[axis] = center[axis];
            centerCorrection[axis] = 0;
            leftCenter[axis].raw() = center[axis].raw() - 4;
            rightCenter[axis].raw() = center[axis].raw() + 6;
        }
        for (size_t i = 0; i < masses.size(); ++i) {
            if (subDoubleCenters) {
                positions[axis][i].raw() = center[axis].raw() - 13 + (i * (axis + 3)) % 29;
            } else {
                positions[axis][i] =
                    meshCenter[axis].toDouble() + (.027 * ((i * (axis + 3)) % 17) - .21) / meshSize;
            }
        }
    }
    FmmSourceView sources;
    sources.store = FmmParticleStore::Ghost;
    sources.count = masses.size();
    sources.masses = masses.data();
    for (int axis = 0; axis < 3; ++axis) {
        sources.positions[axis] = positions[axis].data();
    }
    TreeBuildMoments left, right;
    left.useDouble();
    right.useDouble();
    treeComputeMultipole<order>(left.precise(), {0, 4}, leftCenter, sources);
    treeComputeMultipole<order>(right.precise(), {4, 9}, rightCenter, sources);
    SourceMultipoleBuffer root;
    treeComputeMultipole<order>(root, left, right, leftCenter, rightCenter, center);
    std::array<double, FmmOrder<order>::kMultipoleCoefficients_> mapped{}, expected{};
    treeSourceToMesh<order>(mapped.data(), root, center, meshCenter, meshSize, centerCorrection);
    double maximumOffset = 0;
    for (size_t i = 0; i < masses.size(); ++i) {
        Vector3<double> offset;
        for (int axis = 0; axis < 3; ++axis) {
            offset[axis] = subDoubleCenters ?
                unwrappedDoubleDistance(positions[axis][i], meshCenter[axis]) * meshSize :
                positions[axis][i].toDouble() * meshSize - (meshSize - .5);
            maximumOffset = std::max(maximumOffset, std::abs(offset[axis]));
        }
        cosmictiger::gravity::spherical::detail::P2M<order - 2, double>(
            expected, masses[i], offset[0], offset[1], offset[2]);
    }
    double rankScale = expected[0];
    for (int n = 0; n <= order - 2; ++n) {
        if (n) {
            rankScale *= maximumOffset / n;
        }
        for (int m = -n; m <= n; ++m) {
            int const i = n * (n + 1) + m;
            check(mapped[i], expected[i], rankScale);
        }
    }
}

void transport() {
    TreeBuildMoments value;
    value.useDouble();
    int const count = fmmMultipoleCoefficientCount(fmmOrder());
    for (int i = 0; i < count; ++i) {
        value.precise()[i] = i + .125 + std::ldexp(1., -36);
    }
    auto roundTrip = [&](TreeBuildMoments const& input) {
        std::vector<char> data;
        {
            hpx::serialization::output_archive archive(data);
            archive << input;
        }
        TreeBuildMoments output;
        hpx::serialization::input_archive archive(data);
        archive >> output;
        if (output.isDouble() != input.isDouble()) {
            throw std::runtime_error("source precision lost");
        }
        for (int i = 0; i < count; ++i) {
            check(output[i], input[i], 1);
        }
        return data.size();
    };
    size_t const wide = roundTrip(value);
    value.discardDouble();
    size_t const narrow = roundTrip(value);
    if (wide != narrow + count * sizeof(float)) {
        throw std::runtime_error("inactive source coefficients or double ghost payload serialized");
    }
}
}    // namespace detail

int main() {
    try {
        for (int order = 3; order <= 9; ++order) {
            detail::selectedOrder = order;
            detail::transport();
            fmmDispatch(order, [](auto tag) {
                detail::frozenCloud<decltype(tag)::value>();
                for (int meshSize : {1, 2, 3, 4, 5, 7, 8, 12, 30, 32, 1024}) {
                    detail::cloud<decltype(tag)::value>(meshSize, false);
                }
                detail::cloud<decltype(tag)::value>(1, true);
            });
        }
        std::printf(
            "Owned source fixed%d q=3..9: %zu checks passed, worst rank-scaled error %.3e\n",
            kPositionBits, detail::checks, detail::worst);
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
