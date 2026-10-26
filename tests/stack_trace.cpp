/* CosmicTiger - fatal-signal diagnostics regression; BSL-1.0. */
#include <array>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>

#include <sys/resource.h>

#include <cosmictiger/core/diagnostic_output.hpp>
#include <cosmictiger/core/stack_trace.hpp>

namespace detail {

constexpr std::array<int, 5> kFatalSignals = {SIGABRT, SIGSEGV, SIGBUS, SIGFPE, SIGILL};

void require(bool condition, char const* message) {
    if (!condition) {
        std::fprintf(stderr, "Stack trace regression failed: %s\n", message);
        std::exit(1);
    }
}

void previousHandler(int) {}

void replacementHandler(int) {}

struct sigaction actionFor(int signalNumber) {
    struct sigaction action{};
    require(sigaction(signalNumber, nullptr, &action) == 0, "cannot inspect signal handler");
    return action;
}

void setHandler(int signalNumber, void (*handler)(int)) {
    struct sigaction action{};
    action.sa_handler = handler;
    action.sa_flags = SA_RESTART;
    sigemptyset(&action.sa_mask);
    sigaddset(&action.sa_mask, SIGUSR1);
    require(sigaction(signalNumber, &action, nullptr) == 0, "cannot set prior signal handler");
}

void checkRestored(int signalNumber, void (*handler)(int)) {
    auto const action = actionFor(signalNumber);
    require(action.sa_handler == handler, "previous signal handler was not preserved");
    require((action.sa_flags & SA_RESTART) != 0, "previous signal flags were not restored");
    require(sigismember(&action.sa_mask, SIGUSR1) == 1, "previous signal mask was not restored");
}

void checkLifecycle() {
    for (auto const signalNumber : kFatalSignals) {
        setHandler(signalNumber, previousHandler);
    }
    setHandler(SIGINT, previousHandler);
    {
        StackTraceHandler handler;
        for (auto const signalNumber : kFatalSignals) {
            checkRestored(signalNumber, previousHandler);
        }
    }
    for (auto const signalNumber : kFatalSignals) {
        checkRestored(signalNumber, previousHandler);
    }
    {
        StackTraceHandler handler;
        handler.install(17);
        auto const installed = actionFor(SIGABRT);
        require(installed.sa_handler != previousHandler, "install did not replace fatal handler");
        handler.install(99);
        require(actionFor(SIGABRT).sa_handler == installed.sa_handler,
            "repeat install changed the fatal handler");
        checkRestored(SIGINT, previousHandler);
        {
            StackTraceHandler competingHandler;
            bool rejected = false;
            try {
                competingHandler.install(42);
            } catch (std::exception const&) {
                rejected = true;
            }
            require(rejected, "a competing owner was not rejected");
        }
        require(actionFor(SIGABRT).sa_handler == installed.sa_handler,
            "competing owner destruction removed the active handler");
        setHandler(SIGILL, replacementHandler);
    }
    for (auto const signalNumber : kFatalSignals) {
        checkRestored(signalNumber, signalNumber == SIGILL ? replacementHandler : previousHandler);
    }
    checkRestored(SIGINT, previousHandler);
    {
        StackTraceHandler nextHandler;
        nextHandler.install(23);
    }
    for (auto const signalNumber : kFatalSignals) {
        checkRestored(signalNumber, signalNumber == SIGILL ? replacementHandler : previousHandler);
    }
}

int signalFromName(std::string const& name) {
    std::array<char const*, 5> const names = {"SIGABRT", "SIGSEGV", "SIGBUS", "SIGFPE", "SIGILL"};
    for (size_t index = 0; index < names.size(); ++index) {
        if (name == names[index]) {
            return kFatalSignals[index];
        }
    }
    require(false, "unknown fatal signal");
    return 0;
}

}    // namespace detail

// Keep a named frame and source location in the trace, including worker crashes.
__attribute__((noinline)) void stackTraceTestCrash(int signalNumber) {
    std::raise(signalNumber);
    std::_Exit(98);
}

int main(int argc, char* argv[]) {
    struct rlimit const coreLimit{0, 0};
    detail::require(setrlimit(RLIMIT_CORE, &coreLimit) == 0, "cannot disable test core dumps");
    detail::require(argc >= 2, "missing test mode");
    std::string const mode = argv[1];
    if (mode == "lifecycle") {
        detail::checkLifecycle();
        return 0;
    }
    if (mode == "interrupt") {
        StackTraceHandler handler;
        handler.install(17);
        std::raise(SIGINT);
        return 98;
    }
    detail::require(mode == "fatal" || mode == "worker", "unknown test mode");
    detail::require(argc == 3 || argc == 4, "fatal mode requires signal and optional output path");
    StackTraceHandler handler;
    handler.install(17);
    handler.install(99);
    // Options select the output directory after the HPX startup callback.
    if (argc == 4) {
        setStackTraceOutputPath(argv[3]);
    }
    auto const signalNumber = detail::signalFromName(argv[2]);
    if (mode == "worker") {
        std::thread worker(stackTraceTestCrash, signalNumber);
        worker.join();
    } else {
        stackTraceTestCrash(signalNumber);
    }
    return 98;
}
