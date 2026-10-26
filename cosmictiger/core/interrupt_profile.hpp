/* CosmicTiger - Ctrl-C profile reporting; BSL-1.0. */

#pragma once

#include <memory>

namespace profile {

// Install from an HPX startup callback, after HPX installs its signal handlers.
// Keep this owner alive until HPX has stopped. Destruction joins the reporter
// before the profiler's ordinary process-exit callback can run.
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

}    // namespace profile
