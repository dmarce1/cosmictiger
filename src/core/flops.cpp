/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/timer.hpp>

static double cpuFlops = 0.0;
static SpinlockType mutex;
static Timer tm;

HPX_PLAIN_ACTION(resetFlops, ResetFlopsAction);
HPX_PLAIN_ACTION(flopsPerSecond, FlopsPerSecondAction);

void resetFlops() {
    Vector<hpx::future<void>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<ResetFlopsAction>(c));
    }
    resetGpuFlops();
    cpuFlops = 0.0;
    tm.stop();
    tm.reset();
    tm.start();
    hpx::wait_all(futs.begin(), futs.end());
}

double flopsPerSecond() {
    Vector<hpx::future<double>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<FlopsPerSecondAction>(c));
    }
    tm.stop();
    double result = (getGpuFlops() + cpuFlops);
    tm.start();
    for (auto& f : futs) {
        result += f.get();
    }
    if (hpxRank() == 0) {
        result /= tm.read();
    }
    return result;
}

void addCpuFlops(int count) {
#ifdef COUNT_FLOPS
    std::lock_guard<SpinlockType> lock(mutex);
    cpuFlops += count;
#endif
}

#ifndef USE_CUDA
void resetGpuFlops() {}
double getGpuFlops() {
    return 0.0;
}
#endif
