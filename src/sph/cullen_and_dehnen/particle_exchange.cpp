/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/sph/cullen_and_dehnen/particle_exchange.hpp>

#include "../particle_cache.hpp"

static Vector<Array<float, kDimensions + 1>> sphParticlesFetchViscosityDiagnosticsCacheLine(
    ParticleInteger index);

HPX_PLAIN_ACTION(sphParticlesFetchViscosityDiagnosticsCacheLine,
    SphParticlesFetchViscosityDiagnosticsCacheLineAction);

static Array<std::unordered_map<detail::SphCacheLineId,
                 std::shared_ptr<hpx::shared_future<Vector<Array<float, kDimensions + 1>>>>,
                 detail::SphCacheLineHashHi>,
    kParticleCacheSize>
    viscosityDiagnosticsParticleCache;
static Array<SpinlockType, kParticleCacheSize> viscosityDiagnosticsMutexes;

static Vector<Array<float, kDimensions + 1>> sphParticlesFetchViscosityDiagnosticsCacheLine(
    ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<Array<float, kDimensions + 1>> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; ++i) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            line[i - begin][dim] = sphParticlesAcceleration(dim, i);
        }
        line[i - begin][kDimensions] = sphParticlesCullenDehnenStates[i].divergence;
    }
    return line;
}

void sphParticlesGlobalReadViscosityDiagnostics(ParticleGlobalRange range, float* accelerationX,
    float* accelerationY, float* accelerationZ, float* divergence, ParticleInteger offset) {
    float* const acceleration[kDimensions] = {accelerationX, accelerationY, accelerationZ};
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    if (range.range.first == range.range.second) {
        return;
    }
    if (range.proc == hpxRank()) {
        for (ParticleInteger i = range.range.first; i < range.range.second; ++i) {
            ParticleInteger const destination = offset + i - range.range.first;
            for (int dim = 0; dim < kDimensions; ++dim) {
                if (acceleration[dim]) {
                    acceleration[dim][destination] = sphParticlesAcceleration(dim, i);
                }
            }
            if (divergence) {
                divergence[destination] = sphParticlesCullenDehnenStates[i].divergence;
            }
        }
    } else {
        ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
        ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
        ParticleInteger destination = offset;
        for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
            detail::SphCacheLineId const lineId{range.proc, line};
            auto const* values =
                detail::readSphCacheLine<SphParticlesFetchViscosityDiagnosticsCacheLineAction>(
                    lineId, viscosityDiagnosticsParticleCache, viscosityDiagnosticsMutexes);
            ParticleInteger const begin = std::max(line, range.range.first);
            ParticleInteger const end = std::min(line + lineSize, range.range.second);
            for (ParticleInteger i = begin; i < end; ++i) {
                for (int dim = 0; dim < kDimensions; ++dim) {
                    if (acceleration[dim]) {
                        acceleration[dim][destination] = values[i - line][dim];
                    }
                }
                if (divergence) {
                    divergence[destination] = values[i - line][kDimensions];
                }
                ++destination;
            }
        }
    }
}

namespace cullen_and_dehnen {

void clearDiagnosticCache() {
    viscosityDiagnosticsParticleCache = decltype(viscosityDiagnosticsParticleCache)();
}

}    // namespace cullen_and_dehnen
