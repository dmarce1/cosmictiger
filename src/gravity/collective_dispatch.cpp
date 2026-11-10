/* CosmicTiger - failure-safe collective dispatch; BSL-1.0. */

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include <mpi.h>

#include <cosmictiger/gravity/collective_dispatch.hpp>

namespace pm::detail {

class CollectiveAbortWorker {
    std::atomic<int> request_{0};
    std::atomic<bool> parked_{false};
    std::thread worker_;

public:
    /// Start the native worker that can abort a failed collective independently of HPX
    /// worker availability.
    CollectiveAbortWorker()
      : worker_([this]() {
          request_.wait(0);
          if (request_.load() == 2) {
              std::fputs("Particle-mesh collective dispatch failed; aborting MPI job\n", stderr);
              std::fflush(stderr);
              MPI_Abort(MPI_COMM_WORLD, 86);
              std::_Exit(86);
          }
      }) {}

    /// Request orderly worker termination and join the native thread.
    ~CollectiveAbortWorker() {
        request_.store(1);
        request_.notify_one();
        worker_.join();
    }

    /// Request collective abort from the prestarted native worker and park the failing
    /// caller.
    [[noreturn]] void abort() noexcept {
        request_.store(2);
        request_.notify_one();
        // The native worker is already running, so failure handling needs no
        // allocation or HPX I/O worker (which may be blocked inside MPI).
        for (;;) {
            parked_.wait(false);
        }
    }
};

/// Return the lazily constructed process-wide collective-abort worker.
CollectiveAbortWorker& collectiveAbortWorker() {
    static CollectiveAbortWorker worker;
    return worker;
}

/// Create the abort worker before entering operations whose failure could block runtime
/// workers.
void prepareCollectiveDispatch() {
    (void) collectiveAbortWorker();
}

/// Request process-wide collective termination through the prestarted abort worker.
[[noreturn]] void abortCollectiveDispatch() noexcept {
    collectiveAbortWorker().abort();
}

}    // namespace pm::detail
