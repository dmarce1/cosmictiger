/* CosmicTiger - shared FFTW planner serialization; BSL-1.0. */

#pragma once

#include <mutex>

// FFTW setup/planning/destruction share process-global state even when the
// transforms belong to unrelated serial and MPI plans. Execution stays outside
// this lock. MPI callers acquire it only from hpx::run_as_os_thread.
inline std::mutex& fftwPlannerMutex() {
    static std::mutex mutex;
    return mutex;
}
