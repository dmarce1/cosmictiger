/* CosmicTiger - P3M screened short-range FMM; BSL-1.0. */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cosmictiger/gravity/p3m/short_pair.hpp>
#include <cosmictiger/math/vector3.hpp>

namespace cosmictiger::gravity::p3m::fmm {

struct ShortNode {
    Vector3<double> center{};
    double radius = 0;
    // Particle range in the tree's sorted particle array.
    int begin = 0;
    int end = 0;
    int left = -1;
    int right = -1;
    bool leaf = true;
};

// Tree over the particles of one search-mesh cell, with full symmetric Cartesian
// moments of the runtime FMM order about each node's center of mass.
class ShortTree {
public:
    std::vector<ShortNode> nodes;
    std::vector<ShortParticle> particles;
    // One record of coefficientCount(order - 2) doubles per node.
    std::vector<double> moments;
    int order = 0;
    int momentCount = 0;

    double const* nodeMoments(int node) const {
        return moments.data() + size_t(node) * momentCount;
    }
};

struct ShortWalkStatistics {
    uint64_t directPairs = 0;
    uint64_t acceptedMultipoles = 0;
    uint64_t cutoffRejections = 0;
    uint64_t sourceOpenings = 0;
};

struct ShortWalkParameters {
    ShortPairParameters pair;
    double splitRadius = 0;
    double cutoffRadius = 0;
    double openingAngle = 0.5;
    int bucketSize = 32;
    bool doPotential = true;
};

// Build a tree over the particles (copied, then kd-sorted) for the runtime order.
std::shared_ptr<ShortTree> buildShortTree(
    std::vector<ShortParticle> particles, ShortWalkParameters const& parameters);

// Accumulate the short field of every source tree (each with its own whole-box
// image shift) on the particles of the sink tree. fields is indexed like the
// sink tree's particle array and is overwritten as {potential, ax, ay, az}.
// pairCurvature, when supplied, is overwritten with each sink's direct-pair
// coefficient over the leaf pairs evaluated directly (short_pair.hpp).
ShortWalkStatistics walkShortTree(ShortTree const& sinks,
    std::vector<std::pair<ShortTree const*, Vector3<double>>> const& sources,
    ShortWalkParameters const& parameters, std::vector<std::array<double, 4>>& fields,
    std::vector<double>* pairCurvature = nullptr);

// The screened multipole acceptance: a cell pair is accepted when the summed
// radii stay below openingAngle times the kernel's local length, the smaller
// of the separation d and 2 r_s^2 / d where the Gaussian factor steepens.
inline double screenedAcceptanceLength(double distance, double splitRadius) {
    double const gaussian = 2.0 * splitRadius * splitRadius / distance;
    return distance < gaussian ? distance : gaussian;
}

}    // namespace cosmictiger::gravity::p3m::fmm
