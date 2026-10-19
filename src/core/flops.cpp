/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

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
