// CosmicTiger - coordinator phase accounting regression; BSL-1.0.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
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
        if ((fields >> name) && name != "Total" && (fields >> row.calls >> duration)) {
            row.total = seconds(duration);
            result.emplace(name, row);
        }
    }
    return result;
}

}    // namespace detail

int main(int argc, char** argv) {
    try {
        detail::require(argc == 2, "Expected output filename");
        detail::require(profile::profileReport().empty(), "New registry is not empty");
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
