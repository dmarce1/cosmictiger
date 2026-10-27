/* CosmicTiger - cooperative Ctrl-C shutdown; BSL-1.0. */

#pragma once

#include <exception>
#include <memory>

namespace profile {

// Install from an HPX startup callback, after HPX installs its signal handlers.
// Keep this owner alive until HPX and MPI have stopped. The signal handler only
// requests shutdown; normal runtime teardown finalizes the profiling backends.
// SIGUSR1 makes the same request when forwarded by the MPI launch wrapper.
class InterruptReportHandler {
    struct State;
    std::unique_ptr<State> state_;

public:
    InterruptReportHandler();
    ~InterruptReportHandler();
    InterruptReportHandler(InterruptReportHandler const&) = delete;
    InterruptReportHandler& operator=(InterruptReportHandler const&) = delete;

    void install();
};

class Interrupted final : public std::exception {
public:
    char const* what() const noexcept override;
};

// Call from the application's root HPX thread at safe phase boundaries, after
// joining work that must finish before cleanup. Also checks worker localities.
bool interruptRequested();
void checkForInterrupt();

}    // namespace profile
