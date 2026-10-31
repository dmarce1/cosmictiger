/* CosmicTiger - failure-safe collective dispatch; BSL-1.0. */
#pragma once

#include <cstddef>
#include <cstdio>
#include <exception>
#include <vector>

#include <hpx/async_base/launch_policy.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/threading_base/annotated_function.hpp>

namespace particle_mesh::detail {

// Allocate the native abort worker before any locality can enter a collective.
void prepareCollectiveDispatch();
[[noreturn]] void abortCollectiveDispatch() noexcept;

template <class Launch>
void dispatchCollectively(size_t count, Launch launch) {
    if (count == 1) {
        // No other participant can be stranded inside MPI; retain the ordinary
        // exception and the caller's cleanup/reporting path.
        launch(0).get();
        return;
    }
    auto const reportFailure = [](std::exception const& error, size_t participant) noexcept {
        std::fprintf(
            stderr, "Particle-mesh collective dispatch failure on participant %zu: ", participant);
        std::fputs(error.what(), stderr);
        std::fputc('\n', stderr);
        std::fflush(stderr);
    };
    std::vector<hpx::future<void>> futures;
    futures.reserve(count);
    prepareCollectiveDispatch();
    size_t index = 0;
    try {
        for (; index < count; ++index) {
            // Observe every failure immediately: waiting on an earlier future
            // first could hang while its rank waits for the failed participant.
            futures.push_back(launch(index).then(hpx::launch::sync,
                hpx::annotated_function(
                    [reportFailure, index](hpx::future<void> future) {
                        try {
                            future.get();
                        } catch (std::exception const& error) {
                            reportFailure(error, index);
                            abortCollectiveDispatch();
                        } catch (...) {
                            std::fputs("Unknown particle-mesh collective failure\n", stderr);
                            abortCollectiveDispatch();
                        }
                    },
                    "gravity.pmmm.observeCollective")));
        }
        for (auto& future : futures) {
            future.get();
        }
    } catch (std::exception const& error) {
        reportFailure(error, index);
        abortCollectiveDispatch();
    } catch (...) {
        // Launch or continuation attachment may fail after a subset of ranks
        // entered MPI. Do not wait for them or attempt collective cleanup.
        abortCollectiveDispatch();
    }
}

}    // namespace particle_mesh::detail
