/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/particles/drift.hpp>
#include <cosmictiger/particles/drift_cpu.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/tree.hpp>
#endif

HPX_PLAIN_ACTION(drift, DriftAction);

namespace detail {

/// Enable coordinator-level drift timing only on rank zero when profiling is compiled in.
bool profileDriftCoordinator() {
    // Both drift entry paths execute locally on the driver and recurse only to
    // child localities. CPU workers and CUDA kernels never call drift().
#if COSMICTIGER_PROFILE
    return hpxRank() == 0;
#else
    return false;
#endif
}

}    // namespace detail

/// Partition the selected particle range among CPU workers and collect optional lightcone
/// crossings.
void cpuDrift(int rung, double a, double dt, double tau0, double tau1, double tauMax, double drag) {
    auto const& opts = getOptions();
    auto const range =
        rung < 0 ? Pair<ParticleInteger>(0, particlesSize()) : particlesCurrentRange();
    size_t const count = range.second - range.first;
    if (count == 0) {
        return;
    }
    DriftParameters parameters;
    parameters.rung = rung;
    parameters.scale = a;
    parameters.timestep = dt;
    parameters.tau0 = tau0;
    parameters.tau1 = tau1;
    parameters.tauMax = tauMax;
    // Match the glass damping policy in cudaDrift, including its override of drag.
    drag = opts.run.scenario == "glass" ? std::numbers::pi : 0.0;
    parameters.drag = drag;
#if !defined(COSMICTIGER_ENABLE_ROCKSTAR) || COSMICTIGER_ENABLE_ROCKSTAR
    parameters.lightcone = opts.lightcone.enabled && tauMax > 0.0f && tauMax - tau1 < 1.0f;
#endif
    parameters.mapSize = opts.lightcone.mapSize;
    constexpr size_t kMinimumChunkSize = 4096;
    size_t const workers = std::min(
        size_t(std::max(1, hpxHardwareConcurrency())), (count - 1) / kMinimumChunkSize + 1);
    std::vector<std::vector<LightconeEntry>> entries(parameters.lightcone ? workers : 0);
    auto const run = [&](size_t worker) {
        size_t const begin = size_t(range.first) + worker * count / workers;
        size_t const end = size_t(range.first) + (worker + 1) * count / workers;
        // particlesType() returns a shared static when hydro is off; pass the array only then.
        char const* const types = opts.hydro.enabled ? particlesTy + begin : nullptr;
        cpuDriftKernel(particlesPositionData(kXi, begin), particlesPositionData(kYi, begin),
            particlesPositionData(kZi, begin), particlesVelData() + begin, &particlesRung(begin),
            types, end - begin, parameters, parameters.lightcone ? &entries[worker] : nullptr);
    };
    if (workers == 1) {
        run(0);
    } else {
        std::vector<hpx::future<void>> futures;
        futures.reserve(workers);
        try {
            for (size_t worker = 0; worker < workers; ++worker) {
                futures.push_back(
                    hpx::async(hpx::annotated_function(run, "particles.drift"), worker));
            }
        } catch (...) {
            hpx::wait_all(futures);
            throw;
        }
        // Keep captured arrays alive until every worker is done, including on exceptions.
        hpx::wait_all(futures);
        for (auto& future : futures) {
            future.get();
        }
    }
#if !defined(COSMICTIGER_ENABLE_ROCKSTAR) || COSMICTIGER_ENABLE_ROCKSTAR
    if (parameters.lightcone) {
        profile::ScopedPhase profilePhase("drift.lightcone", detail::profileDriftCoordinator());
        lcAddParticles(entries, a, tau0);
    }
#endif
}

/// Expand SPH swept bounds and drift selected particles on the chosen backend, joining
/// distributed work.
void drift(
    double scale, double dt, double tau0, double tau1, double tauMax, int rung, double drag) {
#if COSMICTIGER_WITH_SPH || defined(USE_CUDA)
    auto const& opts = getOptions();
#endif
    Vector<hpx::future<void>> futures;
    if (rung < 0) {
        for (auto const& child : hpxChildren()) {
            futures.push_back(
                hpx::async<DriftAction>(child, scale, dt, tau0, tau1, tauMax, rung, drag));
        }
    }
#if COSMICTIGER_WITH_SPH
    if (opts.hydro.enabled) {
        // Same displacement factor as the CPU/CUDA drift; expand before wrapping.
        sphTreeDriftBounds(dt / scale, rung);
    }
#endif
#ifdef USE_CUDA
    if (opts.drift.cuda) {
        {
            profile::ScopedPhase profilePhase("drift.cuda", detail::profileDriftCoordinator());
            cudaDrift(rung, scale, dt, tau0, tau1, tauMax, drag);
        }
        profile::ScopedPhase profilePhase("drift.memoryAdvice", detail::profileDriftCoordinator());
        particlesMemadviseCpu();
    } else
#endif
    {
        profile::ScopedPhase profilePhase("drift.cpu", detail::profileDriftCoordinator());
        cpuDrift(rung, scale, dt, tau0, tau1, tauMax, drag);
    }
    if (!futures.empty()) {
        profile::ScopedPhase profilePhase("drift.wait", detail::profileDriftCoordinator());
        for (auto& future : futures) {
            future.get();
        }
    }
}
