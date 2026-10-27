/* CosmicTiger - HPX APEX and CPU sampler coexistence regression; BSL-1.0. */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hpx/futures/promise.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/post.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/runtime_distributed/find_here.hpp>
#include <hpx/runtime_local/get_worker_thread_num.hpp>
#include <hpx/runtime_local/runtime_local_fwd.hpp>
#include <hpx/runtime_local/startup_function.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/threading_base/set_thread_state.hpp>
#include <time.h>
#include <unistd.h>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/interrupt_profile.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/profiling_runtime.hpp>

#if !defined(HPX_HAVE_APEX) || !defined(HPX_HAVE_THREAD_DESCRIPTION)
#error "The APEX regression requires HPX APEX support and thread descriptions"
#endif

namespace detail {

bool interruptExit = false;

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::uint64_t threadCpuNanoseconds() {
    timespec value{};
    require(
        ::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0, "cannot read the worker CPU clock");
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000 +
        static_cast<std::uint64_t>(value.tv_nsec);
}

// No PROFILE() or APEX annotation: a suspended/resumed HPX context must still
// produce ordinary ELF function attribution in the independent CPU sampler.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
std::uint64_t ordinaryHotFunction(std::uint64_t value) {
    auto const start = threadCpuNanoseconds();
    do {
        for (size_t step = 0; step < 65536; step++) {
            value ^= value << 13;
            value ^= value >> 7;
            value ^= value << 17;
        }
    } while (threadCpuNanoseconds() - start < 30000000);
    return value;
}

// Distinct sampled bodies let the TAU regression detect task-label leakage.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
std::uint64_t ordinaryParentFunction(std::uint64_t value) {
    auto const start = threadCpuNanoseconds();
    do {
        for (size_t step = 0; step < 65536; step++) {
            value ^= value << 11;
            value ^= value >> 7;
            value ^= value << 17;
        }
    } while (threadCpuNanoseconds() - start < 30000000);
    return value;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
std::uint64_t ordinaryChildFunction(std::uint64_t value) {
    auto const start = threadCpuNanoseconds();
    do {
        for (size_t step = 0; step < 65536; step++) {
            value ^= value << 13;
            value ^= value >> 9;
            value ^= value << 17;
        }
    } while (threadCpuNanoseconds() - start < 30000000);
    return value;
}

std::uint64_t childTask(std::uint64_t seed) {
    return ordinaryChildFunction(seed);
}

std::uint64_t parentTask(std::uint64_t seed) {
    hpx::this_thread::yield();
    auto child = hpx::async(
        hpx::launch::async, hpx::annotated_function(&childTask, "cosmictiger.apex.child"), seed);
    auto const value = child.get();
    hpx::this_thread::yield();
    return ordinaryParentFunction(value);
}

std::uint64_t continuationTask(hpx::future<std::uint64_t> predecessor) {
    auto const value = predecessor.get();
    require(value != 0, "post fixture lost its nonzero state");
    hpx::this_thread::yield();
    return ordinaryHotFunction(value);
}

// FULL thread descriptions should resolve this plain function pointer without
// requiring an annotated_function wrapper or a PROFILE() registration.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
std::uint64_t unannotatedTask(std::uint64_t seed) {
    hpx::this_thread::yield();
    return ordinaryHotFunction(seed);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
std::uint64_t unannotatedForkTask(std::uint64_t seed) {
    hpx::this_thread::yield();
    return ordinaryHotFunction(seed);
}

// Keep this ordinary function distinct from ordinaryHotFunction so samples
// identify the work performed on each side of an actual task migration.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
std::uint64_t ordinaryMigratedFunction(std::uint64_t value) {
    auto const start = threadCpuNanoseconds();
    do {
        for (size_t step = 0; step < 65536; step++) {
            value ^= value << 7;
            value ^= value >> 9;
            value ^= value << 8;
        }
    } while (threadCpuNanoseconds() - start < 30000000);
    return value;
}

std::uint64_t migratedTask(std::uint64_t seed) {
    require(hpx::get_num_worker_threads() == 2, "migration fixture requires exactly two workers");
    size_t const workerBefore = hpx::get_worker_thread_num();
    require(workerBefore < 2, "migration fixture started outside its two workers");
    size_t const workerTarget = 1 - workerBefore;
    auto const value = ordinaryMigratedFunction(seed);
    auto const self = hpx::threads::get_self_id();
    auto resumer = hpx::async(hpx::launch::async,
        hpx::annotated_function(
            [self, workerTarget]() {
                // HPX 1.11's public state setter discards scheduling hints. This
                // exported detail API is confined to this migration fixture:
                // retry_on_active handles a wake request before suspension, and
                // bound priority prevents the original worker stealing it back.
                auto const previous = hpx::threads::detail::set_thread_state(self,
                    hpx::threads::thread_schedule_state::pending,
                    hpx::threads::thread_restart_state::signaled,
                    hpx::threads::thread_priority::bound,
                    hpx::threads::thread_schedule_hint(static_cast<std::int16_t>(workerTarget)),
                    true);
                require(previous.state() == hpx::threads::thread_schedule_state::active ||
                        previous.state() == hpx::threads::thread_schedule_state::suspended,
                    "migration resumer did not find a running or suspended task");
            },
            "cosmictiger.apex.migration_resume"));
    hpx::this_thread::suspend(hpx::threads::thread_schedule_state::suspended);
    // Acknowledge completion of the resumer and propagate any helper failure.
    resumer.get();
    size_t const workerAfter = hpx::get_worker_thread_num();
    require(workerAfter == workerTarget && workerAfter != workerBefore,
        "HPX task did not resume on the requested different worker");
    auto const result = ordinaryMigratedFunction(value);
    std::printf("APEX_MIGRATION_WORKERS=%zu,%zu\n", workerBefore, workerAfter);
    return result;
}

}    // namespace detail

std::uint64_t apexRegressionAction(std::uint64_t seed) {
    hpx::this_thread::yield();
    return detail::ordinaryHotFunction(seed);
}
HPX_PLAIN_ACTION(apexRegressionAction, ApexRegressionAction);

int hpx_main(int, char*[]) {
    // Each move-only closure has a different object address. All instances must
    // aggregate under one stable label, including work launched by an executor.
    std::vector<hpx::future<std::uint64_t>> closures;
    for (std::uint64_t task = 1; task <= 12; task++) {
        closures.push_back(hpx::async(HPX_PRIORITY_HI,
            hpx::annotated_function(
                [value = std::make_unique<std::uint64_t>(task)]() {
                    hpx::this_thread::yield();
                    return *value;
                },
                "cosmictiger.apex.lambda")));
    }
    for (size_t task = 0; task < closures.size(); task++) {
        detail::require(closures[task].get() == task + 1, "annotated closure lost its capture");
    }
    // Parallel algorithms create their own task closures; annotate their
    // executor so those internal tasks inherit the application phase name.
    std::vector<int> values(131072);
    std::iota(values.rbegin(), values.rend(), 1);
    hpx::sort(annotatedParallelPolicy("cosmictiger.apex.sort"), values.begin(), values.end()).get();
    detail::require(std::is_sorted(values.begin(), values.end()), "annotated sort failed");
    std::vector<int> copy(values.size());
    hpx::copy(annotatedParallelPolicy("cosmictiger.apex.copy"), values.begin(), values.end(),
        copy.begin())
        .get();
    detail::require(copy == values, "annotated copy failed");
    hpx::fill(annotatedParallelPolicy("cosmictiger.apex.fill"), values.begin(), values.end(), 42)
        .get();
    detail::require(
        std::all_of(values.begin(), values.end(), [](int value) { return value == 42; }),
        "annotated fill failed");

    std::vector<hpx::future<std::uint64_t>> work;
    for (std::uint64_t task = 1; task <= 4; task++) {
        work.push_back(hpx::async(hpx::launch::async,
            hpx::annotated_function(&detail::parentTask, "cosmictiger.apex.parent"), task));
    }
    hpx::promise<std::uint64_t> completion;
    auto posted = completion.get_future();
    auto continued = posted.then(hpx::launch::async,
        hpx::annotated_function(&detail::continuationTask, "cosmictiger.apex.continuation"));
    hpx::post(hpx::annotated_function(
        [&completion]() {
            hpx::this_thread::yield();
            completion.set_value(detail::ordinaryHotFunction(5));
        },
        "cosmictiger.apex.post"));
    auto action = hpx::async<ApexRegressionAction>(hpx::launch::async, hpx::find_here(), 6);
    auto unannotated = hpx::async(hpx::launch::async, &detail::unannotatedTask, 7);
    auto forked = hpx::async(hpx::launch::fork, &detail::unannotatedForkTask, 8);
    auto migrated = hpx::async(hpx::launch::async,
        hpx::annotated_function(&detail::migratedTask, "cosmictiger.apex.migrated"), 9);
    for (auto& result : work) {
        detail::require(result.get() != 0, "async fixture lost its nonzero state");
    }
    detail::require(continued.get() != 0, "continuation fixture lost its nonzero state");
    detail::require(action.get() != 0, "action fixture lost its nonzero state");
    detail::require(unannotated.get() != 0, "unannotated fixture lost its nonzero state");
    detail::require(forked.get() != 0, "fork fixture lost its nonzero state");
    detail::require(migrated.get() != 0, "migration fixture lost its nonzero state");
    if (detail::interruptExit) {
        // The test signals this PID, not tau_exec's shell or an MPI launcher.
        std::ofstream ready("interrupt_ready.pid");
        ready << ::getpid() << '\n';
        ready.close();
        detail::require(static_cast<bool>(ready), "cannot publish interrupt readiness");
        try {
            while (true) {
                profile::checkForInterrupt();
                hpx::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        } catch (profile::Interrupted const&) {
            detail::require(profile::interruptRequested(), "interrupt request was lost");
        }
    }
    return hpx::finalize();
}

int main(int argc, char* argv[]) {
    try {
        profile::configureRuntimeProfiling();
#if defined(PROFILE_DISABLE)
        bool samplerDisabled = true;
#else
        bool samplerDisabled = false;
#endif
        int remaining = 1;
        for (int argument = 1; argument < argc; argument++) {
            std::string const option(argv[argument]);
            if (option == "--sampler-disabled") {
                samplerDisabled = true;
            } else if (option == "--interrupt-exit") {
                detail::interruptExit = true;
            } else {
                argv[remaining++] = argv[argument];
            }
        }
        argv[argc = remaining] = nullptr;
        if (!samplerDisabled) {
            detail::require(profile::profileStart(), "cannot start the CPU sampler");
        }
        profile::InterruptReportHandler interruptHandler;
        hpx::register_startup_function([&interruptHandler]() { interruptHandler.install(); });
        // HPX owns APEX initialization, task suspend/resume, and finalization.
        // The fixture intentionally never calls apex::init/start/stop/finalize.
        std::vector<std::string> configuration = {
            "hpx.os_threads=2", "hpx.stacks.small_size=524288"};
#ifdef HPX_EARLY
        int const status = hpx::init(argc, argv, configuration);
#else
        hpx::init_params parameters;
        parameters.cfg = std::move(configuration);
        int const status = hpx::init(argc, argv, parameters);
#endif
        detail::require(status == 0, "HPX APEX fixture failed");
        if (detail::interruptExit) {
            std::puts("APEX_INTERRUPT_FINALIZED=1");
        }
        if (!samplerDisabled) {
            profile::profileStop();
            std::ofstream file("cpu_samples.txt");
            file << profile::profileReport();
            file.close();
            detail::require(static_cast<bool>(file), "cannot save the CPU sample report");
        }
#if defined(HPX_HAVE_THREAD_DESCRIPTION_FULL)
        std::puts("APEX_FULL_THREAD_DESCRIPTION=1");
#else
        std::puts("APEX_FULL_THREAD_DESCRIPTION=0");
#endif
#if defined(PROFILE_DISABLE)
        std::puts("APEX_CPU_SAMPLER_COMPILED=0");
#else
        std::puts("APEX_CPU_SAMPLER_COMPILED=1");
#endif
        std::puts("APEX profiler regression passed");
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "APEX profiler regression: %s\n", error.what());
        return 1;
    }
}
