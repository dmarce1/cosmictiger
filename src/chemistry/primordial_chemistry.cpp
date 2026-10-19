/* CosmicTiger - primordial CPU batching; GPL-2.0-or-later. */
#include <algorithm>
#include <atomic>
#include <exception>
#include <thread>
#include <vector>

#include <cosmictiger/chemistry/primordial_chemistry.hpp>
#include <cosmictiger/chemistry/primordial_float_environment.hpp>

namespace primordial {
namespace detail {
    void initializeFailedBatch(
        BatchInput const* inputs, StepResult* results, size_t count, Status status) {
        for (size_t index = 0; index < count; ++index) {
            results[index] = StepResult{};
            results[index].state = inputs[index].state;
            results[index].status = status;
        }
    }
}    // namespace detail

bool integrateBatchCpu(BatchInput const* inputs, StepResult* results, size_t count,
    SolverOptions const& options, unsigned workers) {
    if (count == 0) {
        return true;
    }
    if (inputs == nullptr || results == nullptr) {
        return false;
    }
    workers = static_cast<unsigned>(std::min<size_t>(std::max(1U, workers), count));
    std::atomic<bool> validEnvironment{true};
    auto run = [&](unsigned worker) {
        StrictFloatEnvironment floatingPointEnvironment;
        if (!floatingPointEnvironment.valid()) {
            validEnvironment.store(false, std::memory_order_relaxed);
            return;
        }
        // Strided ownership avoids overflow in count*worker partition arithmetic.
        for (size_t index = worker; index < count;) {
            results[index] = advance(
                inputs[index].state, inputs[index].environment, inputs[index].interval, options);
            if (count - index <= workers) {
                break;
            }
            index += workers;
        }
    };
    std::vector<std::thread> threads;
    try {
        threads.reserve(workers - 1);
        for (unsigned worker = 1; worker < workers; ++worker) {
            threads.emplace_back(run, worker);
        }
        run(0);
    } catch (std::exception const&) {
        for (auto& thread : threads) {
            thread.join();
        }
        detail::initializeFailedBatch(inputs, results, count, Status::BackendFailure);
        return false;
    }
    for (auto& thread : threads) {
        thread.join();
    }
    if (!validEnvironment.load(std::memory_order_relaxed)) {
        detail::initializeFailedBatch(inputs, results, count, Status::BackendFailure);
        return false;
    }
    return true;
}

#ifndef USE_CUDA
bool integrateBatchCuda(BatchInput const* inputs, StepResult* results, size_t count,
    SolverOptions const&, std::string* error) {
    if (count == 0) {
        if (error != nullptr) {
            error->clear();
        }
        return true;
    }
    if (inputs != nullptr && results != nullptr) {
        detail::initializeFailedBatch(inputs, results, count, Status::BackendUnavailable);
    }
    if (error != nullptr) {
        *error = "Primordial chemistry was built without CUDA support";
    }
    return false;
}
#endif

bool integrateBatch(BatchInput const* inputs, StepResult* results, size_t count,
    SolverOptions const& options, Backend backend, unsigned workers, std::string* error) {
    if (error != nullptr) {
        error->clear();
    }
    if (backend == Backend::Cuda) {
        return integrateBatchCuda(inputs, results, count, options, error);
    }
    if (backend != Backend::Cpu) {
        if (inputs != nullptr && results != nullptr) {
            detail::initializeFailedBatch(inputs, results, count, Status::BackendUnavailable);
        }
        if (error != nullptr) {
            *error = "Unknown primordial chemistry backend";
        }
        return false;
    }
    bool const success = integrateBatchCpu(inputs, results, count, options, workers);
    if (!success && error != nullptr) {
        *error = "Primordial chemistry CPU batch dispatch failed";
    }
    return success;
}
}    // namespace primordial
