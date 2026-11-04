/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/sph/flops.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/time_integration.hpp>

namespace detail {

static bool midpointEventActive = false;

}    // namespace detail

void sphMidpointSetEventActive(bool active);
HPX_PLAIN_ACTION(sphMidpointSetEventActive, SphMidpointSetEventActiveAction);
HPX_PLAIN_ACTION(sphMidpointTargetCount, SphMidpointTargetCountAction);
HPX_PLAIN_ACTION(sphParticlesApplyMidpointUpdates, SphParticlesApplyMidpointUpdatesAction);
HPX_PLAIN_ACTION(sphParticlesSelectMidpointRungs, SphParticlesSelectMidpointRungsAction);

bool sphMidpointEnabled() {
    auto const& options = getOptions();
    return options.hydro.enabled && options.hydro.method == "sph" && !options.chemistry.enabled &&
        !options.starFormation.enabled &&
        (options.hydro.viscosityEvolution == "none" ||
            options.hydro.viscosityEvolution == "morrisAndMonaghan") &&
        (!options.gravity.enabled || options.gravity.method == "fmm");
}

bool sphMidpointEventActive() {
    return detail::midpointEventActive;
}

void sphMidpointSetEventActive(bool active) {
    Vector<hpx::future<void>> children;
    for (auto const& child : hpxChildren()) {
        children.push_back(hpx::async<SphMidpointSetEventActiveAction>(child, active));
    }
    detail::midpointEventActive = active;
    hpx::wait_all(children.begin(), children.end());
    for (auto& child : children) {
        child.get();
    }
}

void sphMidpointBeginEvent() {
    ALWAYS_ASSERT(sphMidpointEnabled() && !detail::midpointEventActive);
    sphParticlesReleaseGravityStorage();
    sphMidpointSetEventActive(true);
}

void sphMidpointEndEvent() {
    sphMidpointSetEventActive(false);
}

size_t sphMidpointTargetCount(int exactRung) {
    Vector<hpx::future<size_t>> children;
    for (auto const& child : hpxChildren()) {
        children.push_back(hpx::async<SphMidpointTargetCountAction>(child, exactRung));
    }
    size_t count = 0;
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        count += exactRung < 0 || sphParticlesRung(gas) == exactRung;
    }
    hpx::wait_all(children.begin(), children.end());
    for (auto& child : children) {
        count += child.get();
    }
    return count;
}

void sphParticlesApplyMidpointUpdates(int exactRung, double timestep) {
    ALWAYS_ASSERT(detail::midpointEventActive && std::isfinite(timestep) && timestep >= 0.0);
    if (timestep == 0.0) {
        return;
    }
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(
            hpx::async<SphParticlesApplyMidpointUpdatesAction>(child, exactRung, timestep));
    }
    int const workers = std::max(1, hpxHardwareConcurrency());
    for (int worker = 0; worker < workers; ++worker) {
        pending.push_back(hpx::async(hpx::annotated_function(
            [exactRung, timestep, workers, worker]() {
                ParticleInteger const begin = size_t(worker) * sphParticlesSize() / workers;
                ParticleInteger const end = size_t(worker + 1) * sphParticlesSize() / workers;
                sph_flops::Counter flops;
                for (ParticleInteger gas = begin; gas < end; ++gas) {
                    if (exactRung >= 0 && sphParticlesRung(gas) != exactRung) {
                        continue;
                    }
                    // Locally derived midpoint operator at fixed positions:
                    //   v ← v + Δt a_hydro; A ← A + Δt Ȧ.
                    // v → sphParticlesVelocity; Δt → timestep;
                    // a_hydro → sphParticlesAcceleration; A → sphParticlesEntropy;
                    // Ȧ → sphParticlesEntropyRate. The driver supplies the symmetric
                    // gravity/drift composition; viscosity uses a first-order kick.
                    for (int dimension = 0; dimension < kDimensions; ++dimension) {
                        sphParticlesVelocity(dimension, gas) +=
                            sphParticlesAcceleration(dimension, gas) * timestep;
                    }
                    ALWAYS_ASSERT(sphParticlesEntropy(gas) > 0.0f);
                    sphParticlesEntropy(gas) += sphParticlesEntropyRate(gas) * timestep;
                    ALWAYS_ASSERT(
                        std::isfinite(sphParticlesEntropy(gas)) && sphParticlesEntropy(gas) > 0.0f);
                    flops.doublePrecision += 5 * (kDimensions + 1);
                }
                addCpuFlops(flops.counts());
            },
            "sph.particles.midpointKick")));
    }
    hpx::wait_all(pending.begin(), pending.end());
    for (auto& future : pending) {
        future.get();
    }
}

int sphParticlesSelectMidpointRungs(int minimumRung) {
    ALWAYS_ASSERT(sphMidpointEnabled() && minimumRung >= 0);
    Vector<hpx::future<int>> children;
    for (auto const& child : hpxChildren()) {
        children.push_back(hpx::async<SphParticlesSelectMidpointRungsAction>(child, minimumRung));
    }
    int maximumRung = 0;
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        auto& rung = sphParticlesRung(gas);
        if (rung >= minimumRung) {
            rung = std::max(
                {minimumRung, int(rung) - 1, sphParticlesHydroRung(gas), getOptions().run.minRung});
            ALWAYS_ASSERT(rung >= 0 && rung < kMaxRung);
        }
        maximumRung = std::max(maximumRung, int(rung));
    }
    hpx::wait_all(children.begin(), children.end());
    for (auto& child : children) {
        maximumRung = std::max(maximumRung, child.get());
    }
    return maximumRung;
}
