/* CosmicTiger - host batching boundary for primordial chemistry; GPL-2.0-or-later. */
#pragma once

#include <cstddef>
#include <string>

#include <cosmictiger/chemistry/primordial_solver.hpp>

namespace primordial {

struct BatchInput {
    State state{};
    Environment environment{};
    float interval{};    // Physical seconds; typed conversions belong to the host caller.
};

enum class Backend { Cpu, Cuda };

// True means dispatch completed; each result carries its own integration status.
// False preserves input states and reports a backend error; no CPU retry occurs.
bool integrateBatchCpu(BatchInput const* inputs, StepResult* results, size_t count,
    SolverOptions const& options = SolverOptions{}, unsigned workers = 1);
bool integrateBatchCuda(BatchInput const* inputs, StepResult* results, size_t count,
    SolverOptions const& options = SolverOptions{}, std::string* error = nullptr);
bool integrateBatch(BatchInput const* inputs, StepResult* results, size_t count,
    SolverOptions const& options, Backend backend, unsigned workers = 1,
    std::string* error = nullptr);

}    // namespace primordial
