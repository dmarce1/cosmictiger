/* CosmicTiger - particle-mesh field exchange; BSL-1.0. */

#pragma once

#include <exception>
#include <type_traits>

namespace particle_mesh::detail {

// Requests and their future slots must exist before launching any reads. No
// continuation borrows the caller's output: consumption happens synchronously,
// after every launched read has completed, including on a launch failure.
template <class Requests, class Launch, class Consume>
void exchangeFields(Requests& requests, Launch launch, Consume consume) {
    auto waitForReads = [&]() {
        for (auto& [owner, request] : requests) {
            if (request.future.valid()) {
                // wait_all allocates a completion frame. Individual waits avoid
                // that allocation while unwinding an allocation/launch failure.
                request.future.wait();
            }
        }
    };
    std::exception_ptr failure;
    try {
        for (auto& [owner, request] : requests) {
            static_assert(std::is_nothrow_move_assignable_v<decltype(request.future)>);
            request.future = launch(owner, request);
        }
    } catch (...) {
        failure = std::current_exception();
    }
    // HPX waits can resume this task on a different OS thread. Leave the catch
    // first: the native C++ active-exception state does not migrate with it.
    waitForReads();
    if (failure) {
        std::rethrow_exception(failure);
    }
    for (auto& [owner, request] : requests) {
        consume(request, request.future.get());
    }
}

}    // namespace particle_mesh::detail
