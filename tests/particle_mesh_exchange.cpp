/* CosmicTiger - particle-mesh exchange failure regression; BSL-1.0. */

#include <atomic>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <hpx/futures/future.hpp>
#include <hpx/futures/promise.hpp>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/gravity/particle_mesh_exchange.hpp>

namespace detail {

struct MeshExchangeState {
    hpx::promise<int> firstEvent;
    std::atomic<bool> eventPublished{false};
    std::atomic<int> completedReads{0};
    std::atomic<int> completedAtDestruction{-1};
    std::atomic<int> writes{0};
    std::atomic<int> lateWrites{0};
    std::atomic<bool> outputAlive{false};

    void publish(int event) {
        bool expected = false;
        if (eventPublished.compare_exchange_strong(expected, true)) {
            firstEvent.set_value(event);
        }
    }
};

// Observe entry into a pending wait, while retaining real HPX future/promise
// behavior. Racing this event against exchange completion avoids sleeps and
// makes a missing drain fail deterministically rather than relying on timing.
struct ObservedMeshFuture {
    hpx::future<int> value;
    std::shared_ptr<MeshExchangeState> state;

    bool valid() const {
        return value.valid();
    }

    void wait() const {
        if (!value.is_ready()) {
            state->publish(1);
        }
        value.wait();
    }

    int get() {
        return value.get();
    }
};

struct MeshExchangeRequest {
    ObservedMeshFuture future;
};

struct MeshExchangeOutput {
    std::shared_ptr<MeshExchangeState> state;
    int total = 0;

    explicit MeshExchangeOutput(std::shared_ptr<MeshExchangeState> value)
      : state(std::move(value)) {
        state->outputAlive.store(true);
    }

    ~MeshExchangeOutput() {
        state->completedAtDestruction.store(state->completedReads.load());
        state->outputAlive.store(false);
    }

    void add(int value) {
        if (!state->outputAlive.load()) {
            ++state->lateWrites;
        }
        ++state->writes;
        total += value;
    }
};

void requireMeshExchange(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error("Particle mesh exchange regression: " + message);
    }
}

void checkMeshExchange(int failure) {
    auto state = std::make_shared<MeshExchangeState>();
    auto event = state->firstEvent.get_future();
    hpx::promise<void> release;
    auto gate = release.get_future().share();
    auto exchange = hpx::async([state, gate, failure]() {
        std::string error;
        bool allocationFailure = false;
        int total = 0;
        {
            MeshExchangeOutput output(state);
            std::map<int, MeshExchangeRequest> requests;
            requests.try_emplace(0);
            requests.try_emplace(1);
            try {
                particle_mesh::detail::exchangeFields(
                    requests,
                    [&](int owner, MeshExchangeRequest&) -> ObservedMeshFuture {
                        if (failure == 3 && owner == 1) {
                            throw std::bad_alloc();
                        }
                        if (failure == 1 && owner == 1) {
                            throw std::runtime_error("injected field request launch failure");
                        }
                        if (failure == 2 && owner == 0) {
                            return {hpx::make_exceptional_future<int>(
                                        std::runtime_error("injected field response failure")),
                                state};
                        }
                        return {hpx::async([state, gate, owner]() {
                                    gate.get();
                                    ++state->completedReads;
                                    return owner + 3;
                                }),
                            state};
                    },
                    [&](MeshExchangeRequest const&, int value) { output.add(value); });
                total = output.total;
            } catch (std::bad_alloc const&) {
                allocationFailure = true;
            } catch (std::runtime_error const& exception) {
                error = exception.what();
            }
        }
        state->publish(2);
        return std::make_tuple(error, total, allocationFailure);
    });

    // A pending read must be joined before the output can be destroyed or an
    // exception returned. Always release the gate before reporting a failure,
    // so even a deliberately broken implementation cannot strand a test task.
    int const firstEvent = event.get();
    bool const aliveAtWait = state->outputAlive.load();
    bool const finishedAtWait = exchange.is_ready();
    release.set_value();
    auto const [error, total, allocationFailure] = exchange.get();
    requireMeshExchange(firstEvent == 1 && aliveAtWait && !finishedAtWait,
        "exchange returned before its delayed reads completed");
    int const expectedReads = failure == 0 ? 2 : 1;
    requireMeshExchange(
        state->completedReads == expectedReads && state->completedAtDestruction == expectedReads,
        "a launched read survived output destruction");
    requireMeshExchange(!state->outputAlive && state->lateWrites == 0,
        "field interpolation wrote into released output");
    if (failure == 0) {
        requireMeshExchange(!allocationFailure && error.empty() && total == 7 && state->writes == 2,
            "successful exchange lost field responses");
    } else if (failure == 3) {
        requireMeshExchange(allocationFailure && error.empty() && state->writes == 0,
            "exchange changed the allocation exception type or consumed incomplete responses");
    } else {
        std::string const expected = failure == 1 ? "injected field request launch failure" :
                                                    "injected field response failure";
        requireMeshExchange(!allocationFailure && error == expected && state->writes == 0,
            "exchange lost the original exception or consumed incomplete responses");
    }
}

}    // namespace detail

void particleMeshExchangeTest() {
    detail::checkMeshExchange(1);
    detail::checkMeshExchange(3);
    detail::checkMeshExchange(2);
    detail::checkMeshExchange(0);
}
