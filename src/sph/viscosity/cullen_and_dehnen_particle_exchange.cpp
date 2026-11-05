/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/viscosity/cullen_and_dehnen_particle_exchange.hpp>

#include "../particle_cache.hpp"

// CD equation (17) needs only the sign. A positive timestep normalization
// preserves that sign without a per-source clock or cosmological scale.
static Vector<float> sphParticlesFetchViscosityDiagnosticsCacheLine(ParticleInteger index);
static Vector<float> sphParticlesFetchCurrentViscosityDiagnosticsCacheLine(ParticleInteger index);

HPX_PLAIN_ACTION(sphParticlesFetchViscosityDiagnosticsCacheLine,
    SphParticlesFetchViscosityDiagnosticsCacheLineAction);
HPX_PLAIN_ACTION(sphParticlesFetchCurrentViscosityDiagnosticsCacheLine,
    SphParticlesFetchCurrentViscosityDiagnosticsCacheLineAction);

using ViscosityDiagnosticCache =
    Array<std::unordered_map<detail::SphCacheLineId,
              std::shared_ptr<hpx::shared_future<Vector<float>>>, detail::SphCacheLineHashHi>,
        kParticleCacheSize>;
static ViscosityDiagnosticCache viscosityDiagnosticsParticleCache;
static ViscosityDiagnosticCache currentViscosityDiagnosticsParticleCache;
static Array<SpinlockType, kParticleCacheSize> viscosityDiagnosticsMutexes;
static Array<SpinlockType, kParticleCacheSize> currentViscosityDiagnosticsMutexes;

static float sphCurrentDivergenceSign(ParticleInteger index) {
    ALWAYS_ASSERT(sphParticlesViscosityDiagnostics != nullptr);
    float const divergence = sphParticlesViscosityDiagnostics[index].pendingDivergence;
    ALWAYS_ASSERT(sph_viscosity::detail::isFinite(divergence));
    return divergence > 0.0f ? 1.0f : (divergence < 0.0f ? -1.0f : 0.0f);
}

static Vector<float> sphParticlesFetchViscosityDiagnosticsCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<float> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; ++i) {
        line[i - begin] = sphDivergenceSign(sphScaledDivergence[i]);
    }
    return line;
}

static Vector<float> sphParticlesFetchCurrentViscosityDiagnosticsCacheLine(ParticleInteger index) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    Vector<float> line(lineSize);
    ParticleInteger const begin = (index / lineSize) * lineSize;
    ParticleInteger const end = std::min(sphParticlesSize(), begin + lineSize);
    for (ParticleInteger i = begin; i < end; ++i) {
        line[i - begin] = sphCurrentDivergenceSign(i);
    }
    return line;
}

void sphParticlesGlobalReadViscosityDiagnostics(
    ParticleGlobalRange range, float* divergence, ParticleInteger offset, bool currentGradient) {
    ParticleInteger const lineSize = getOptions().compute.particleCacheLineSize;
    if (range.range.first == range.range.second) {
        return;
    }
    if (range.proc == hpxRank()) {
        for (ParticleInteger i = range.range.first; i < range.range.second; ++i) {
            ParticleInteger const destination = offset + i - range.range.first;
            if (divergence) {
                divergence[destination] = currentGradient ?
                    sphCurrentDivergenceSign(i) :
                    sphDivergenceSign(sphScaledDivergence[i]);
            }
        }
    } else {
        ParticleInteger const startLine = (range.range.first / lineSize) * lineSize;
        ParticleInteger const stopLine = ((range.range.second - 1) / lineSize) * lineSize;
        ParticleInteger destination = offset;
        for (ParticleInteger line = startLine; line <= stopLine; line += lineSize) {
            detail::SphCacheLineId const lineId{range.proc, line};
            auto const* values = currentGradient ?
                detail::readSphCacheLine<
                    SphParticlesFetchCurrentViscosityDiagnosticsCacheLineAction>(lineId,
                    currentViscosityDiagnosticsParticleCache, currentViscosityDiagnosticsMutexes) :
                detail::readSphCacheLine<SphParticlesFetchViscosityDiagnosticsCacheLineAction>(
                    lineId, viscosityDiagnosticsParticleCache, viscosityDiagnosticsMutexes);
            ParticleInteger const begin = std::max(line, range.range.first);
            ParticleInteger const end = std::min(line + lineSize, range.range.second);
            for (ParticleInteger i = begin; i < end; ++i) {
                if (divergence) {
                    divergence[destination] = values[i - line];
                }
                ++destination;
            }
        }
    }
}

HPX_PLAIN_ACTION(sphParticlesPublishDivergence, SphParticlesPublishDivergenceAction);
void sphParticlesPublishDivergence(SphRunParameters const& parameters) {
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<SphParticlesPublishDivergenceAction>(child, parameters));
    }
    for (ParticleInteger index = 0; index < sphParticlesSize(); ++index) {
        if (parameters.isActiveRung(sphParticlesRung(index)) && sphParticlesConverged(index)) {
            ALWAYS_ASSERT(sphParticlesViscosityDiagnostics != nullptr);
            double const sampleTimestep = sphViscosityStoredSampleTimestep(sphParticlesRung(index),
                parameters.viscosityBaseTimestep, parameters.viscosityPreviousTimestep,
                parameters.globalTimestep);
            float const divergence = sphParticlesViscosityDiagnostics[index].pendingDivergence;
            if (!sphEncodeDivergence(sphScaledDivergence[index], divergence, sampleTimestep)) {
                THROW_ERROR("SPH output divergence exceeds finite float16 storage\n");
            }
        }
    }
    for (auto& future : pending) {
        future.get();
    }
}

namespace cullen_and_dehnen {

void clearDiagnosticCache() {
    viscosityDiagnosticsParticleCache = ViscosityDiagnosticCache();
    currentViscosityDiagnosticsParticleCache = ViscosityDiagnosticCache();
}

}    // namespace cullen_and_dehnen
