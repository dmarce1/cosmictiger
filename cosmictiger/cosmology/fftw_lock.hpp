/* CosmicTiger - shared FFTW planner serialization; BSL-1.0. */

#pragma once

#include <mutex>
#include <utility>

#include <hpx/runtime_local/run_as_os_thread.hpp>

// FFTW setup/planning/destruction share process-global state even when the
// transforms belong to unrelated serial and MPI plans. Execution stays outside
// this lock. Hold it only on native threads: threaded planning may need HPX
// workers, and an HPX task must not suspend while owning a std::mutex.
inline std::mutex& fftwPlannerMutex() {
    static std::mutex mutex;
    return mutex;
}

// HPX callers suspend while a native I/O thread acquires the planner lock.
template <class Function>
auto withFftwPlannerLock(Function function) {
    return hpx::run_as_os_thread([function = std::move(function)]() mutable {
        std::lock_guard<std::mutex> lock(fftwPlannerMutex());
        return function();
    }).get();
}
