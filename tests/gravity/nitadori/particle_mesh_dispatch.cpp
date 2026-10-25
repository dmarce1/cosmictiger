/* CosmicTiger - collective dispatch failure regression; BSL-1.0. */

#include <atomic>
#include <cstdio>
#include <stdexcept>

#include <hpx/runtime_local/run_as_os_thread.hpp>
#include <mpi.h>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/gravity/nitadori/collective_dispatch.hpp>

namespace detail {

std::atomic<bool> meshCollectiveEntered{false};

void meshDispatchBlock() {
    hpx::run_as_os_thread([]() {
        MPI_Request request = MPI_REQUEST_NULL;
        if (MPI_Ibarrier(MPI_COMM_WORLD, &request) != MPI_SUCCESS) {
            throw std::runtime_error("Unable to initiate dispatch-test barrier");
        }
        // The other rank never enters this collective. Keep this native pool
        // worker occupied until the dispatch failure terminates the MPI job.
        meshCollectiveEntered.store(true, std::memory_order_release);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        throw std::runtime_error("Dispatch-test barrier unexpectedly completed");
    }).get();
}

void meshDispatchAwaitCollective() {
    while (!meshCollectiveEntered.load(std::memory_order_acquire)) {
        hpx::this_thread::yield();
    }
    std::fprintf(stderr, "PMMM dispatch test entered collective on locality %d\n", hpxRank());
    std::fflush(stderr);
}

void meshDispatchAsyncException() {
    std::fputs("PMMM dispatch test injected asynchronous failure\n", stderr);
    std::fflush(stderr);
    throw std::runtime_error("Injected PMMM collective action failure");
}

}    // namespace detail

HPX_PLAIN_ACTION(detail::meshDispatchBlock, MeshDispatchBlockAction);
HPX_PLAIN_ACTION(detail::meshDispatchAwaitCollective, MeshDispatchAwaitCollectiveAction);
HPX_PLAIN_ACTION(detail::meshDispatchAsyncException, MeshDispatchAsyncExceptionAction);

namespace detail {

void meshDispatchFailure(bool asynchronous) {
    if (hpxSize() != 2) {
        throw std::runtime_error("PMMM dispatch failure test requires exactly two localities");
    }
    // The launch-failure case occupies the coordinator's only I/O worker:
    // scheduling MPI_Abort on that pool would deadlock too. The asynchronous
    // case leaves the first future pending on the remote locality instead.
    auto const blocker = hpxLocalities()[asynchronous ? 1 : 0];
    particle_mesh::detail::dispatchCollectively(2, [&](size_t index) -> hpx::future<void> {
        if (index == 0) {
            return hpx::async<MeshDispatchBlockAction>(blocker);
        }
        hpx::async<MeshDispatchAwaitCollectiveAction>(blocker).get();
        if (asynchronous) {
            return hpx::async<MeshDispatchAsyncExceptionAction>(hpxLocalities()[1]);
        }
        std::fputs("PMMM dispatch test injected launch failure\n", stderr);
        std::fflush(stderr);
        throw std::runtime_error("Injected PMMM collective launch failure");
    });
    std::fputs("PMMM dispatch test unexpectedly returned\n", stderr);
    std::fflush(stderr);
    throw std::runtime_error("PMMM collective dispatch failed to terminate");
}

}    // namespace detail

void particleMeshDispatchLaunchFailureTest() {
    detail::meshDispatchFailure(false);
}

void particleMeshDispatchAsyncFailureTest() {
    detail::meshDispatchFailure(true);
}
