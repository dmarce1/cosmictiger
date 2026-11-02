/* CosmicTiger - particle-mesh and P3M geometry derived from the options; BSL-1.0. */
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/pm/cloud.hpp>
#include <cosmictiger/gravity/pmmm/mesh_size.hpp>

namespace pm {

inline int cloudOrder(Gravity const& gravity) {
    return cloudOrder(gravity.cloud);
}

// Gaussian split radius r_s in box units; the shared kernel uses alpha = 1/(2 r_s).
inline double splitRadius(Gravity const& gravity) {
    if (gravity.meshSize <= 0) {
        throw std::invalid_argument("P3M split radius requires a positive gravity.meshSize");
    }
    return gravity.p3mSplitCells / gravity.meshSize;
}

// Short-range pair cutoff in box units. Pairs beyond it contribute only through the mesh.
inline double cutoffRadius(Gravity const& gravity) {
    return gravity.p3mCutoff * splitRadius(gravity);
}

// Largest power-of-two search mesh whose neighbor cube of pmmmNearWidth cells
// covers the cutoff radius, so every pair inside the cutoff lies in the cube.
inline int shortMeshSize(Gravity const& gravity) {
    double const cutoff = cutoffRadius(gravity);
    if (!(cutoff > 0.0) || !std::isfinite(cutoff)) {
        throw std::invalid_argument("P3M cutoff radius must be finite and positive");
    }
    int side = 1;
    while (2 * side * cutoff <= gravity.pmmmNearWidth && 2 * side <= gravity.meshSize) {
        side *= 2;
    }
    return side;
}

// Granularity of the particle-domain cuts for the cell-owned short solvers.
inline int coarseMeshSize(Gravity const& gravity) {
    if (usesMeshGravity(gravity)) {
        return pmmm::detail::resolvedCoarseMeshSize(
            gravity.meshSize, gravity.pmmmCoarseMeshSize, gravity.pmmmNearWidth);
    }
    if (usesSplitParticleMesh(gravity)) {
        int const shortMesh =
            gravity.p3mMeshSize > 0 ? gravity.p3mMeshSize : shortMeshSize(gravity);
        return pmmm::detail::resolvedCoarseMeshSize(
            shortMesh, gravity.pmmmCoarseMeshSize, gravity.pmmmNearWidth);
    }
    return 0;
}

// Effective smoothing length of the mesh force for the PM timestep criterion:
// the cloud's radius for pure PM, combined in quadrature with the Gaussian
// force width sqrt(2) r_s for the split long force.
inline double longForceEffectiveRadius(Gravity const& gravity) {
    double const cloud = effectiveRadius(gravity.meshSize, cloudOrder(gravity));
    if (!usesSplitParticleMesh(gravity)) {
        return cloud;
    }
    double const rs = splitRadius(gravity);
    return std::sqrt(cloud * cloud + 2.0 * rs * rs);
}

}    // namespace pm
