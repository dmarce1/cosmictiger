/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/particles/drift.hpp>
#include <cosmictiger/particles/drift_cpu.hpp>

HPX_PLAIN_ACTION(drift, DriftAction);

void cpuDrift(int rung, float a, float dt, float tau0, float tau1, float tauMax, double drag) {
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
    drag = getOptions().initial.createGlass ? M_PI : 0.0;
    parameters.drag = drag;
    parameters.lightcone = getOptions().lightcone.enabled && tauMax > 0.0f && tauMax - tau1 < 1.0f;
    parameters.mapSize = getOptions().lightcone.mapSize;
    constexpr size_t kMinimumChunkSize = 4096;
    size_t const workers = std::min(
        size_t(std::max(1, hpxHardwareConcurrency())), (count - 1) / kMinimumChunkSize + 1);
    std::vector<std::vector<LightconeEntry>> entries(parameters.lightcone ? workers : 0);
    auto const run = [&](size_t worker) {
        size_t const begin = size_t(range.first) + worker * count / workers;
        size_t const end = size_t(range.first) + (worker + 1) * count / workers;
        cpuDriftKernel(&particlesPos(XDIM, begin), &particlesPos(YDIM, begin),
            &particlesPos(ZDIM, begin), particlesVelData() + begin, &particlesRung(begin),
            end - begin, parameters, parameters.lightcone ? &entries[worker] : nullptr);
    };
    if (workers == 1) {
        run(0);
    } else {
        std::vector<hpx::future<void>> futures;
        futures.reserve(workers);
        try {
            for (size_t worker = 0; worker < workers; ++worker) {
                futures.push_back(hpx::async(run, worker));
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
    if (parameters.lightcone) {
        lcAddParticles(entries, a, tau0);
    }
}

void drift(
    double scale, double dt, double tau0, double tau1, double tauMax, int rung, double drag) {
    profilerEnter(__FUNCTION__);
    Vector<hpx::future<void>> futures;
    if (rung < 0) {
        for (auto const& child : hpxChildren()) {
            futures.push_back(
                hpx::async<DriftAction>(child, scale, dt, tau0, tau1, tauMax, rung, drag));
        }
    }
#ifdef USE_CUDA
    if (getOptions().compute.cuda) {
        cudaDrift(rung, scale, dt, tau0, tau1, tauMax, drag);
        particlesMemadviseCpu();
    } else
#endif
    {
        cpuDrift(rung, scale, dt, tau0, tau1, tauMax, drag);
    }
    for (auto& future : futures) {
        future.get();
    }
    profilerExit();
}
