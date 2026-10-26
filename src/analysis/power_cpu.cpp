/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 SPDX-License-Identifier: BSL-1.0
 */

#include <cmath>

#include <cosmictiger/analysis/power_cpu.hpp>
#include <cosmictiger/cosmology/fft.hpp>

void accumulateDensityCpuKernel(Position const* positionX, Position const* positionY,
    Position const* positionZ, float const* masses, size_t particleCount, int fold, int gridSize,
    Range<int64_t> const& box, float* density) {
    Position const* const positions[kDimensions] = {positionX, positionY, positionZ};
    for (size_t index = 0; index < particleCount; index++) {
        int cell[kDimensions];
        float offsets[kDimensions];
        bool owned = true;
        for (int dimension = 0; dimension < kDimensions; dimension++) {
            float const coordinate = std::fmod(fold * positions[dimension][index].toFloat(), 1.0f);
            float const gridCoordinate = coordinate * gridSize;
            cell[dimension] = static_cast<int>(gridCoordinate);
            // Each particle belongs to exactly one interior tile. The full
            // six-point stencil then fits inside that tile's padded allocation.
            if (cell[dimension] < box.begin[dimension] - kCloudMin ||
                cell[dimension] >= box.end[dimension] - kCloudMax) {
                owned = false;
                break;
            }
            offsets[dimension] = gridCoordinate - cell[dimension];
        }
        if (!owned) {
            continue;
        }
        float weights[kDimensions][kCloudMax - kCloudMin + 1];
        for (int dimension = 0; dimension < kDimensions; dimension++) {
            for (int offset = kCloudMin; offset <= kCloudMax; offset++) {
                weights[dimension][offset - kCloudMin] = cloudWeight(offsets[dimension] - offset);
            }
        }
        float const mass = masses == nullptr ? 1.0f : masses[index];
        for (int x = kCloudMin; x <= kCloudMax; x++) {
            for (int y = kCloudMin; y <= kCloudMax; y++) {
                for (int z = kCloudMin; z <= kCloudMax; z++) {
                    float const weight = weights[kXDimension][x - kCloudMin] *
                        weights[kYDimension][y - kCloudMin] * weights[kZDimension][z - kCloudMin] *
                        mass;
                    density[box.index(cell[kXDimension] + x, cell[kYDimension] + y,
                        cell[kZDimension] + z)] += weight;
                }
            }
        }
    }
}
