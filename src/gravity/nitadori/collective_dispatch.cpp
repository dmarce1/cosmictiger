/* CosmicTiger - failure-safe collective dispatch; BSL-1.0. */

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include <mpi.h>

#include <cosmictiger/gravity/nitadori/collective_dispatch.hpp>

namespace particle_mesh::detail {

class CollectiveAbortWorker {
    std::atomic<int> request_{0};
    std::atomic<bool> parked_{false};
    std::thread worker_;

public:
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

    ~CollectiveAbortWorker() {
        request_.store(1);
        request_.notify_one();
        worker_.join();
    }

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

CollectiveAbortWorker& collectiveAbortWorker() {
    static CollectiveAbortWorker worker;
    return worker;
}

void prepareCollectiveDispatch() {
    (void) collectiveAbortWorker();
}

[[noreturn]] void abortCollectiveDispatch() noexcept {
    collectiveAbortWorker().abort();
}

}    // namespace particle_mesh::detail
