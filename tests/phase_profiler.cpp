// CosmicTiger - coordinator and concurrent work accounting regression; BSL-1.0.

#include <array>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#include <cosmictiger/core/profiler.hpp>

namespace detail {

struct UnwindChild {};

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Row {
    unsigned long long calls;
    double total;
    double meanUs;
};

double seconds(std::string const& duration) {
    int hours, minutes;
    double value;
    require(std::sscanf(duration.c_str(), "%d:%d:%lf", &hours, &minutes, &value) == 3,
        "Invalid duration in profile");
    return hours * 3600.0 + minutes * 60.0 + value;
}

std::unordered_map<std::string, Row> rows(std::string const& report) {
    std::istringstream input(report);
    std::string line;
    std::unordered_map<std::string, Row> result;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string name, duration;
        Row row{};
        if (!(fields >> name)) {
            continue;
        }
        if (name == "Total") {
            break;
        }
        if (fields >> row.calls >> duration) {
            row.total = seconds(duration);
            result.emplace(name, row);
        }
    }
    return result;
}

std::unordered_map<std::string, Row> workRows(std::string const& report) {
    auto const start = report.find("Profile detail:");
    if (start == std::string::npos) {
        return {};
    }
    std::istringstream input(report.substr(start));
    std::string line;
    std::unordered_map<std::string, Row> result;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string name, duration;
        Row row{};
        if (fields >> name >> row.calls >> duration >> row.meanUs) {
            row.total = seconds(duration);
            require(result.emplace(name, row).second, "Work row repeated across detail tables");
        }
    }
    return result;
}

struct WorkExpected {
    double prepare;
    double traversal;
    double periodic;
    double parallel;
};

WorkExpected exerciseWork() {
    using Clock = std::chrono::steady_clock;
    WorkExpected expected{};
    {
        profile::ScopedWork disabled(profile::WorkPhase::KickCpuM2P, false);
        disabled.next(profile::WorkPhase::KickCpuP2L);
        disabled.stop();
        auto start = Clock::now();
        profile::ScopedWork work(profile::WorkPhase::KickCpuPrepare);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        work.next(profile::WorkPhase::KickCpuTraversal);
        auto const transition = Clock::now();
        expected.prepare = std::chrono::duration<double>(transition - start).count();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        work.stop();
        expected.traversal = std::chrono::duration<double>(Clock::now() - transition).count();
        // Stopping repeatedly and advancing after stop must not create samples.
        work.stop();
        work.next(profile::WorkPhase::KickCpuM2P);
    }
    auto const start = Clock::now();
    try {
        profile::ScopedWork work(profile::WorkPhase::KickCpuPeriodic);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        throw UnwindChild{};
    } catch (UnwindChild const&) {
    }
    expected.periodic = std::chrono::duration<double>(Clock::now() - start).count();

    constexpr size_t kWorkers = 4;
    std::barrier rendezvous(static_cast<std::ptrdiff_t>(kWorkers));
    std::array<std::thread, kWorkers> workers;
    std::array<double, kWorkers> elapsed{};
    for (size_t index = 0; index < kWorkers; ++index) {
        workers[index] = std::thread([&, index] {
            rendezvous.arrive_and_wait();
            auto const begin = Clock::now();
            {
                profile::ScopedWork work(profile::WorkPhase::KickCpuP2P);
                // Every scope must be live before any worker may finish it.
                // These intervals intentionally overlap rather than nest.
                rendezvous.arrive_and_wait();
                for (size_t sample = 0; sample < 5000; ++sample) {
                    profile::recordWork(profile::WorkPhase::KickCpuM2L, 1.0e-6);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
            elapsed[index] = std::chrono::duration<double>(Clock::now() - begin).count();
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    for (double value : elapsed) {
        expected.parallel += value;
    }
    profile::recordWork(profile::WorkPhase::KickGpuHostToDevice, 0.003);
    profile::recordWork(profile::WorkPhase::KickGpuHostToDevice, 0.003);
    profile::recordWork(profile::WorkPhase::KickGpuKernel, 0.012);
    profile::recordWork(profile::WorkPhase::KickGpuKernel, 0.004);
    profile::recordWork(profile::WorkPhase::KickGpuDeviceToHost, 0.0);
    profile::recordWork(profile::WorkPhase::KickGpuCleanup, -1.0);
    profile::recordWork(
        profile::WorkPhase::KickGpuCleanup, std::numeric_limits<double>::infinity());
    profile::recordWork(
        profile::WorkPhase::KickGpuCleanup, std::numeric_limits<double>::quiet_NaN());
    profile::recordWork(profile::WorkPhase::Count, 1.0);
    return expected;
}

void checkWork(std::string const& report, WorkExpected const& expected) {
    auto const values = workRows(report);
    require(values.size() == 8, "Unexpected detail phase or disabled/stopped scope recorded");
    for (auto const* name : {"kick.cpu.prepare", "kick.cpu.traversal", "kick.cpu.periodic"}) {
        require(values.at(name).calls == 1, "Transition or scope unwind changed work calls");
    }
    require(std::abs(values.at("kick.cpu.prepare").total - expected.prepare) < 0.02,
        "Preparation work interval was lost or duplicated");
    require(std::abs(values.at("kick.cpu.traversal").total - expected.traversal) < 0.02,
        "Transition did not separate work intervals");
    require(std::abs(values.at("kick.cpu.periodic").total - expected.periodic) < 0.02,
        "Exception unwind lost work interval");
    require(values.at("kick.cpu.P2P").calls == 4, "Parallel scopes lost calls");
    require(std::abs(values.at("kick.cpu.P2P").total - expected.parallel) < 0.02,
        "Overlapping worker intervals were not summed independently");
    require(values.at("kick.cpu.M2L").calls == 20000, "Concurrent atomic samples lost calls");
    require(std::abs(values.at("kick.cpu.M2L").meanUs - 1.0) < 0.002,
        "Concurrent atomic samples lost time");
    require(values.at("kick.gpu.hostToDevice").calls == 2 &&
            std::abs(values.at("kick.gpu.hostToDevice").meanUs - 3000.0) < 0.002,
        "CUDA transfer samples were not aggregated");
    require(values.at("kick.gpu.kernel").calls == 2 &&
            std::abs(values.at("kick.gpu.kernel").meanUs - 8000.0) < 0.002,
        "CUDA kernel mean has incorrect units");
    require(values.at("kick.gpu.deviceToHost").calls == 1 &&
            values.at("kick.gpu.deviceToHost").meanUs == 0.0,
        "Zero-duration sample was lost");
    require(report.find("rank-0 local host elapsed intervals (overlap; not CPU time)") !=
            std::string::npos,
        "Host work accounting semantics missing from report");
    require(
        report.find("rank-0 CUDA device elapsed intervals (overlap possible)") != std::string::npos,
        "Device work accounting semantics missing from report");
    auto const detailStart = report.find("Profile detail:");
    require(report.find("Share", detailStart) == std::string::npos,
        "Concurrent work report incorrectly includes additive shares");
}

}    // namespace detail

int main(int argc, char** argv) {
    try {
        detail::require(argc == 2, "Expected output filename");
        detail::require(profile::profileReport().empty(), "New registry is not empty");
        auto const workExpected = detail::exerciseWork();
        auto const workOnlyReport = profile::profileReport();
#if COSMICTIGER_PROFILE
        detail::checkWork(workOnlyReport, workExpected);
        detail::require(workOnlyReport.find("exclusive coordinator") == std::string::npos,
            "Work-only report invented coordinator phases");
#else
        detail::require(workOnlyReport.empty(), "Disabled work profiler produced a report");
#endif
        double childElapsed = 0.0;
        double parentElapsed = 0.0;
        {
            profile::ScopedPhase parent("parent");
            {
                profile::ScopedPhase disabled("disabled", false);
                auto const start = std::chrono::steady_clock::now();
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                parentElapsed =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            }
            try {
                profile::ScopedPhase child("parent.child");
                auto const start = std::chrono::steady_clock::now();
                std::this_thread::sleep_for(std::chrono::milliseconds(60));
                childElapsed =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                // Reporting while a nested scope is live must neither pop it nor
                // restart the parent's clock.
                auto const snapshot = profile::profileReport();
#if COSMICTIGER_PROFILE
                detail::require(detail::rows(snapshot).at("parent.child").calls == 1,
                    "Live child missing from snapshot");
#else
                detail::require(snapshot.empty(), "Disabled profiler produced a snapshot");
#endif
                throw detail::UnwindChild{};
            } catch (detail::UnwindChild const&) {
            }
            // Re-entering the same name should aggregate two calls, and a manual
            // child can be mixed with scoped parents.
            profile::profileBegin("parent.child");
            profile::profileEnd();
        }
        auto const report = profile::profileReport();
        profile::profileOutput(argv[1]);
#if COSMICTIGER_PROFILE
        auto const values = detail::rows(report);
        detail::require(values.size() == 2, "Unexpected phase or unbalanced scope");
        detail::require(values.at("parent").calls == 1, "Parent call count changed");
        detail::require(values.at("parent.child").calls == 2, "Child calls were not aggregated");
        // Hundredths-of-a-second report rounding plus scheduler allowance.
        detail::require(std::abs(values.at("parent").total - parentElapsed) < 0.03,
            "Parent was charged its child's time");
        detail::require(std::abs(values.at("parent.child").total - childElapsed) < 0.03,
            "Child time was lost or counted twice");
        detail::checkWork(report, workExpected);
        detail::require(report.substr(report.find("Profile detail:")) == workOnlyReport,
            "Coordinator phase profiling changed independent work intervals");
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        detail::require(profile::profileReport() == report, "A phase leaked past scope exit");
        std::ifstream file(argv[1]);
        std::string const saved(
            (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        detail::require(saved == report, "Saved report differs from terminal report");
#else
        detail::require(report.empty(), "Disabled profiler produced a report");
        detail::require(!std::ifstream(argv[1]).good(), "Disabled profiler wrote an output file");
#endif
        std::puts("Phase profiler accounting passed");
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
