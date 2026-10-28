/* CosmicTiger - FFTW parallel work on HPX workers; BSL-1.0. */

#include <cstddef>
#include <exception>
#include <vector>

#include <fftw3.h>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/runtime_local/thread_pool_helpers.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/cosmology/fftw_lock.hpp>
#include <cosmictiger/cosmology/fftw_threads.hpp>

namespace fftw_threads {
namespace detail {

    void parallelLoop(
        void* (*work)(char*), char* jobData, size_t elementSize, int jobs, void* context) noexcept {
        try {
            auto& executor = *static_cast<hpx::execution::parallel_executor*>(context);
            std::vector<hpx::future<void>> futures;
            futures.reserve(jobs);
            for (int job = 0; job < jobs; ++job) {
                char* const data = jobData + elementSize * size_t(job);
                futures.push_back(hpx::parallel::execution::async_execute(executor,
                    hpx::annotated_function([work, data]() { work(data); }, "fftw.executeJob")));
            }
            // FFTW owns jobData, often on its caller's stack. All jobs must finish
            // before returning. HPX callers suspend; I/O-thread callers wait natively.
            for (auto& future : futures) {
                future.get();
            }
        } catch (...) {
            // The C callback has no error channel. Never return incomplete work or
            // let an exception unwind through FFTW with jobs still using its data.
            std::terminate();
        }
    }

}    // namespace detail

void initialize() {
    static auto executor = []() {
        hpx::launch policy = hpx::launch::async;
        hpx::threads::thread_schedule_hint hint;
        // A waiting I/O thread must not execute a queued job itself.
        hint.runs_as_child_mode(hpx::threads::thread_execution_hint::none);
        policy.set_hint(hint);
        return hpx::execution::parallel_executor(
            &hpx::resource::get_thread_pool(size_t(0)), policy);
    }();
    // Bind the pool while the runtime is available. FFTW later invokes the
    // callback from an I/O thread that need not have HPX worker-local state.
    withFftwPlannerLock([]() { fftwf_threads_set_callback(detail::parallelLoop, &executor); });
}

}    // namespace fftw_threads
