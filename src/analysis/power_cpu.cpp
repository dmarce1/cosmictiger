/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 SPDX-License-Identifier: BSL-1.0
 */

#include <cmath>

#include <cosmictiger/analysis/power_cpu.hpp>
#include <cosmictiger/cosmology/fft.hpp>

void accumulateDensityCpuKernel(Fixed32 const* positionX, Fixed32 const* positionY,
    Fixed32 const* positionZ, float const* masses, size_t particleCount, int fold, int gridSize,
    Range<int64_t> const& box, float* density) {
    Fixed32 const* const positions[NDIM] = {positionX, positionY, positionZ};
    for (size_t index = 0; index < particleCount; index++) {
        int cell[NDIM];
        float offsets[NDIM];
        bool owned = true;
        for (int dimension = 0; dimension < NDIM; dimension++) {
            float const coordinate = std::fmod(fold * positions[dimension][index].toFloat(), 1.0f);
            float const gridCoordinate = coordinate * gridSize;
            cell[dimension] = static_cast<int>(gridCoordinate);
            // Each particle belongs to exactly one interior tile. The full
            // six-point stencil then fits inside that tile's padded allocation.
            if (cell[dimension] < box.begin[dimension] - CLOUD_MIN ||
                cell[dimension] >= box.end[dimension] - CLOUD_MAX) {
                owned = false;
                break;
            }
            offsets[dimension] = gridCoordinate - cell[dimension];
        }
        if (!owned) {
            continue;
        }
        float weights[NDIM][CLOUD_MAX - CLOUD_MIN + 1];
        for (int dimension = 0; dimension < NDIM; dimension++) {
            for (int offset = CLOUD_MIN; offset <= CLOUD_MAX; offset++) {
                weights[dimension][offset - CLOUD_MIN] = cloudWeight(offsets[dimension] - offset);
            }
        }
        float const mass = masses == nullptr ? 1.0f : masses[index];
        for (int x = CLOUD_MIN; x <= CLOUD_MAX; x++) {
            for (int y = CLOUD_MIN; y <= CLOUD_MAX; y++) {
                for (int z = CLOUD_MIN; z <= CLOUD_MAX; z++) {
                    float const weight = weights[XDIM][x - CLOUD_MIN] *
                        weights[YDIM][y - CLOUD_MIN] * weights[ZDIM][z - CLOUD_MIN] * mass;
                    density[box.index(cell[XDIM] + x, cell[YDIM] + y, cell[ZDIM] + z)] += weight;
                }
            }
        }
    }
}
