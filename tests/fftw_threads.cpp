/* CosmicTiger - FFTW HPX callback regression; BSL-1.0. */

#include <array>
#include <atomic>
#include <cstddef>

#include <fftw3.h>
#include <gtest/gtest.h>
#include <hpx/hpx_finalize.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime_local/run_as_os_thread.hpp>
#include <hpx/runtime_local/startup_function.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

#include <cosmictiger/cosmology/fftw_threads.hpp>

namespace detail {

using ParallelLoop = void (*)(void* (*) (char*), char*, size_t, int, void*);

ParallelLoop installedParallelLoop = nullptr;
void* installedCallbackData = nullptr;
int callbackInstallations = 0;

struct Job {
    std::atomic<int> calls{0};
    std::atomic<int> hpxCalls{0};
};

void* runJob(char* jobData) {
    auto& job = *reinterpret_cast<Job*>(jobData);
    job.calls.fetch_add(1, std::memory_order_relaxed);
    if (hpx::threads::get_self_ptr() != nullptr) {
        job.hpxCalls.fetch_add(1, std::memory_order_relaxed);
    }
    return nullptr;
}

void checkJobs(bool nativeCaller) {
    ASSERT_NE(installedParallelLoop, nullptr);
    for (int const count : std::array<int, 3>{0, 1, 11}) {
        // The extra element detects a callback that includes the end index.
        std::array<Job, 12> jobs{};
        auto const dispatch = [&jobs, count]() {
            bool const onHpxThread = hpx::threads::get_self_ptr() != nullptr;
            installedParallelLoop(runJob, reinterpret_cast<char*>(jobs.data()), sizeof(Job), count,
                installedCallbackData);
            return onHpxThread;
        };
        bool const onHpxThread =
            nativeCaller ? hpx::run_as_os_thread(dispatch).get() : hpx::async(dispatch).get();
        EXPECT_EQ(onHpxThread, !nativeCaller);
        for (size_t index = 0; index < jobs.size(); ++index) {
            int const expected = index < static_cast<size_t>(count) ? 1 : 0;
            EXPECT_EQ(jobs[index].calls.load(), expected) << "count=" << count << ", job=" << index;
            EXPECT_EQ(jobs[index].hpxCalls.load(), expected)
                << "count=" << count << ", job=" << index;
        }
    }
}

struct NestedJob {
    Job parent;
    std::array<Job, 7> children{};
    std::atomic<int> completedChildren{0};
};

void* runNestedJob(char* jobData) {
    auto& job = *reinterpret_cast<NestedJob*>(jobData);
    runJob(reinterpret_cast<char*>(&job.parent));
    installedParallelLoop(runJob, reinterpret_cast<char*>(job.children.data()), sizeof(Job),
        static_cast<int>(job.children.size()), installedCallbackData);
    // A nested callback must join its children before returning to FFTW.
    for (auto const& child : job.children) {
        job.completedChildren.fetch_add(child.calls.load(), std::memory_order_relaxed);
    }
    return nullptr;
}

void checkNestedJobs(bool nativeCaller) {
    ASSERT_NE(installedParallelLoop, nullptr);
    std::array<NestedJob, 5> jobs{};
    auto const dispatch = [&jobs]() {
        installedParallelLoop(runNestedJob, reinterpret_cast<char*>(jobs.data()), sizeof(NestedJob),
            static_cast<int>(jobs.size()), installedCallbackData);
    };
    if (nativeCaller) {
        hpx::run_as_os_thread(dispatch).get();
    } else {
        hpx::async(dispatch).get();
    }
    for (auto const& job : jobs) {
        EXPECT_EQ(job.parent.calls.load(), 1);
        EXPECT_EQ(job.parent.hpxCalls.load(), 1);
        EXPECT_EQ(job.completedChildren.load(), static_cast<int>(job.children.size()));
        for (auto const& child : job.children) {
            EXPECT_EQ(child.calls.load(), 1);
            EXPECT_EQ(child.hpxCalls.load(), 1);
        }
    }
}

}    // namespace detail

// Capture only the installation API. The callback itself is the production
// implementation, allowing arbitrary job counts and nested work to be checked.
extern "C" void FFTW_CDECL fftwf_threads_set_callback(
    detail::ParallelLoop parallelLoop, void* data) {
    detail::installedParallelLoop = parallelLoop;
    detail::installedCallbackData = data;
    ++detail::callbackInstallations;
}

TEST(FftwThreads, StartupInstallsCallback) {
    EXPECT_EQ(detail::callbackInstallations, 1);
    EXPECT_NE(detail::installedParallelLoop, nullptr);
}

TEST(FftwThreads, HpxCallerCompletesEveryJobOnHpx) {
    detail::checkJobs(false);
}

TEST(FftwThreads, NativeCallerCompletesEveryJobOnHpx) {
    detail::checkJobs(true);
}

TEST(FftwThreads, HpxCallerJoinsNestedJobs) {
    detail::checkNestedJobs(false);
}

TEST(FftwThreads, NativeCallerJoinsNestedJobs) {
    detail::checkNestedJobs(true);
}

int hpx_main(int, char*[]) {
    int const status = RUN_ALL_TESTS();
    hpx::finalize();
    return status;
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    if (GTEST_FLAG_GET(list_tests)) {
        return RUN_ALL_TESTS();
    }
    hpx::register_startup_function(fftw_threads::initialize);
    return hpx::init(argc, argv);
}
