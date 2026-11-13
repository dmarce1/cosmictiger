/* CosmicTiger - static particle FMM benchmark; BSL-1.0. */
#pragma once

#include <string>
#include <vector>

struct BenchmarkScenario;

namespace scenarios {
struct BenchmarkCase {
    std::string filename;
    double openingAngle;
};

std::vector<BenchmarkCase> parseBenchmarkCases(BenchmarkScenario const& options);
void runBenchmark();
}    // namespace scenarios
