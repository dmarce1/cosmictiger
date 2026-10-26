/* CosmicTiger - statistical CPU profiler regression; BSL-1.0. */

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/threads.hpp>
#include <time.h>

#include <cosmictiger/core/profiler.hpp>

namespace detail {

constexpr int profiledConstant(int value) {
    PROFILE();
    return value + 1;
}

consteval int profiledImmediate() {
    PROFILE();
    return 7;
}

template <typename Value>
constexpr Value profiledTemplate(Value value) {
    PROFILE();
    return value + 2;
}

static_assert(profiledConstant(4) == 5);
static_assert(profiledImmediate() == 7);
static_assert(profiledTemplate(4) == 6);

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

// Deliberately has no PROFILE(): attribution must come from the interrupted
// instruction's ELF symbol, even when its registered caller waits or resumes.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
std::uint64_t unregisteredHotLoop(std::uint64_t value) {
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

std::uint64_t suspendedTask(std::uint64_t seed) {
    PROFILE();
    hpx::this_thread::yield();
    auto work = hpx::async(hpx::launch::async, &unregisteredHotLoop, seed);
    auto const value = work.get();
    hpx::this_thread::yield();
    return unregisteredHotLoop(value);
}

void saveReport(std::string const& filename, std::string const& report) {
    std::ofstream file(filename);
    file << report;
    file.close();
    require(static_cast<bool>(file), "cannot save the expected report");
}

int invalidInterval() {
    require(!profile::profileStart(0), "zero sampling interval was accepted");
    profile::profileStop();
    require(!profile::profileStart(), "a stopped sampler was restarted");
    saveReport("invalid_interval.txt", profile::profileReport());
    return 0;
}

}    // namespace detail

int hpx_main(int, char*[]) {
    // A deliberately misleading label must never replace the actual PC symbol.
    detail::require(profile::profileRegisterFunction("regressionMetadataIsNotAttribution"),
        "metadata registration failed");
    int volatile runtimeValue = 4;
    detail::require(detail::profiledConstant(runtimeValue) == 5,
        "PROFILE changed a constexpr function's runtime value");
    detail::require(detail::profiledTemplate(runtimeValue) == 6 &&
            detail::profiledTemplate(static_cast<long>(runtimeValue)) == 6,
        "PROFILE changed a template specialization's runtime value");
    std::vector<hpx::future<std::uint64_t>> work;
    for (std::uint64_t task = 1; task <= 8; task++) {
        work.push_back(hpx::async(hpx::launch::async, &detail::suspendedTask, task));
    }
    for (auto& result : work) {
        detail::require(result.get() != 0, "CPU fixture lost its nonzero state");
    }
    return hpx::finalize();
}

int main(int argc, char* argv[]) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--invalid-interval") {
            return detail::invalidInterval();
        }
        bool const automaticExit = argc > 1 && std::string(argv[1]) == "--automatic-exit";
        if (automaticExit) {
            for (int argument = 1; argument + 1 < argc; argument++) {
                argv[argument] = argv[argument + 1];
            }
            argv[--argc] = nullptr;
        }
        detail::require(profile::profileStart(), "cannot start the sampler");
        std::vector<std::string> configuration = {
            "hpx.os_threads=2", "hpx.stacks.small_size=524288"};
#ifdef HPX_EARLY
        int const status = hpx::init(argc, argv, configuration);
#else
        hpx::init_params parameters;
        parameters.cfg = std::move(configuration);
        int const status = hpx::init(argc, argv, parameters);
#endif
        detail::require(status == 0, "HPX fixture failed");
        if (automaticExit) {
            detail::require(
                profile::profileReport().find("Sampling is still active") != std::string::npos,
                "automatic-exit fixture stopped sampling before process exit");
            // Leave the calling thread's sampler attached. Its TLS destructor
            // runs before the atexit callback, which must safely stop/report the
            // process-lifetime buffers without accessing destroyed TLS objects.
            std::puts("profiler automatic-exit regression passed");
            return 0;
        }
        profile::profileStop();
        auto const report = profile::profileReport();
        detail::require(!profile::profileStart(), "a stopped sampler was restarted");
        detail::require(!profile::profileRegisterFunction("afterStop"),
            "a stopped sampler accepted new metadata");
        detail::saveReport("expected.txt", report);
        detail::require(
            profile::writeProfileReport("explicit.txt"), "explicit report output failed");
        detail::require(profile::profileReport() == report,
            "a final snapshot changed after the sampler stopped");
        std::puts("profiler regression passed");
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "profiler regression: %s\n", error.what());
        return 1;
    }
}
