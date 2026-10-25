/* CosmicTiger - failure-safe collective dispatch; BSL-1.0. */
#pragma once

#include <cstddef>
#include <vector>

#include <hpx/async_base/launch_policy.hpp>
#include <hpx/futures/future.hpp>

namespace particle_mesh::detail {

// Allocate the native abort worker before any locality can enter a collective.
void prepareCollectiveDispatch();
[[noreturn]] void abortCollectiveDispatch() noexcept;

template <class Launch>
void dispatchCollectively(size_t count, Launch launch) {
    std::vector<hpx::future<void>> futures;
    futures.reserve(count);
    prepareCollectiveDispatch();
    try {
        for (size_t index = 0; index < count; ++index) {
            // Observe every failure immediately: waiting on an earlier future
            // first could hang while its rank waits for the failed participant.
            futures.push_back(launch(index).then(hpx::launch::sync, [](hpx::future<void> future) {
                try {
                    future.get();
                } catch (...) {
                    abortCollectiveDispatch();
                }
            }));
        }
        for (auto& future : futures) {
            future.get();
        }
    } catch (...) {
        // Launch or continuation attachment may fail after a subset of ranks
        // entered MPI. Do not wait for them or attempt collective cleanup.
        abortCollectiveDispatch();
    }
}

}    // namespace particle_mesh::detail
