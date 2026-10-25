/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

struct PmmmNearParticle {
    std::array<double, 3> offset{};
    double mass = 0;
    int rank = 0;
    int64_t index = 0;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & offset & mass & rank & index;
    }
};

// Coordinates are in cell units about the same sink-cell center, with the
// source-cell periodic image already selected. No nearest-image wrapping is
// performed here. Fields are overwritten as {potential, ax, ay, az}, in unit-box
// coordinates without G. Only identical primary particle IDs are excluded.
void pmmmNearField(std::vector<PmmmNearParticle> const& sinks,
    std::vector<PmmmNearParticle> const& sources, double cellWidth, bool doPotential,
    std::vector<std::array<double, 4>>& fields);

namespace pmmm::detail {

#ifdef __CUDACC__
#define COSMICTIGER_PMMM_NEAR_HD __host__ __device__
#else
#define COSMICTIGER_PMMM_NEAR_HD
#endif

// Keep the shared arithmetic independent of HPX and kernel-table headers. ADL
// resolves forceFactor/potential to the selected radial_kernel::KernelTable.
template <class Kernel>
COSMICTIGER_PMMM_NEAR_HD inline void accumulateNearPair(PmmmNearParticle const& sink,
    PmmmNearParticle const& source, double cellWidth, double inverseSupportRadius,
    double inverseSupportRadiusCubed, bool doPotential, Kernel const& kernel,
    std::array<double, 4>& field) {
    if (source.rank == sink.rank && source.index == sink.index) {
        return;
    }
    double const dx = (sink.offset[0] - source.offset[0]) * cellWidth;
    double const dy = (sink.offset[1] - source.offset[1]) * cellWidth;
    double const dz = (sink.offset[2] - source.offset[2]) * cellWidth;
    double const q = ::sqrt(dx * dx + dy * dy + dz * dz) * inverseSupportRadius;
    double const factor = source.mass * inverseSupportRadiusCubed * forceFactor(kernel, q);
    field[1] -= factor * dx;
    field[2] -= factor * dy;
    field[3] -= factor * dz;
    if (doPotential) {
        field[0] -= source.mass * inverseSupportRadius * potential(kernel, q);
    }
}

#undef COSMICTIGER_PMMM_NEAR_HD

#ifdef USE_CUDA
void pmmmNearFieldCuda(std::vector<PmmmNearParticle> const& sinks,
    std::vector<PmmmNearParticle> const& sources, double cellWidth, double supportRadius,
    bool doPotential, std::vector<std::array<double, 4>>& fields);
#endif

}    // namespace pmmm::detail
