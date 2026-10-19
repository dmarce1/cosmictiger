/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/gravity.hpp>

// The CUDA operators are order templates in gravity_cuda.hpp; this unit owns
// the interaction counters they share.

__managed__ double iclose;
__managed__ double idirect;
__managed__ double numberOfProcessedSinkParticles;
__managed__ bool useGravityCounters = false;

void resetGravityCounters() {
    if (!cudaEnabled()) {
        return;
    }
    iclose = idirect = numberOfProcessedSinkParticles = 0;
}

void setGravityCounterUse(bool code) {
    if (!cudaEnabled()) {
        return;
    }
    useGravityCounters = code;
}

void getGravityCounters(double& close, double& direct) {
    if (!cudaEnabled()) {
        close = direct = 0.0;
        return;
    }
    if (numberOfProcessedSinkParticles == 0.0) {
        close = direct = 0.0;
    } else {
        close = iclose / numberOfProcessedSinkParticles;
        direct = idirect / numberOfProcessedSinkParticles;
    }
}
