// Copyright (c) 2026 Octo-Tiger AUTHORS
// Copyright (c) 2026 Dominic C. Marcello
// Adapted from Octo-Tiger stacktrace_fix, a0720ee02feebf776b797b2fdc74edf6bcc5bd83.
// Distributed under the Boost Software License, Version 1.0.

#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <boost/stacktrace.hpp>
#if !defined(_MSC_VER)
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#endif

#include <cosmictiger/core/diagnostic_output.hpp>
#include <cosmictiger/core/stack_trace.hpp>

namespace detail {

// Keep published paths immutable and alive until process shutdown. A signal
// handler can then read one without locking or touching shared_ptr reference
// counts, even if another thread is changing the output directory.
std::atomic<char const*> stackTracePath{"stack_trace.txt"};
static_assert(decltype(stackTracePath)::is_always_lock_free);
std::mutex stackTracePathMutex;
std::vector<std::unique_ptr<std::string const>> stackTracePaths;

#if !defined(_MSC_VER)
constexpr std::array<int, 5> kFatalSignals{SIGABRT, SIGSEGV, SIGBUS, SIGFPE, SIGILL};
std::atomic<bool> ownerActive{false};
std::atomic_flag reporting = ATOMIC_FLAG_INIT;
std::array<char, 384> processHeader{};
size_t processHeaderSize = 0;

void writeDescriptor(int descriptor, char const* text, size_t length) noexcept {
    while (length != 0) {
        auto const written = ::write(descriptor, text, length);
        if (written > 0) {
            text += written;
            length -= static_cast<size_t>(written);
        } else if (written == -1 && errno == EINTR) {
            continue;
        } else {
            break;
        }
    }
}

void writeDiagnostic(int descriptor, char const* text, size_t length) noexcept {
    writeDescriptor(STDERR_FILENO, text, length);
    if (descriptor >= 0) {
        writeDescriptor(descriptor, text, length);
    }
}

template <size_t length>
void writeDiagnostic(int descriptor, char const (&text)[length]) noexcept {
    writeDiagnostic(descriptor, text, length - 1);
}

[[noreturn]] void terminateWithSignal(int signal) noexcept {
    struct sigaction action{};
    action.sa_handler = SIG_DFL;
    sigemptyset(&action.sa_mask);
    sigaction(signal, &action, nullptr);
    sigset_t unblocked;
    sigemptyset(&unblocked);
    sigaddset(&unblocked, signal);
    sigprocmask(SIG_UNBLOCK, &unblocked, nullptr);
    raise(signal);
    std::_Exit(128 + signal);
}

void reportSignal(int signal) noexcept {
    if (reporting.test_and_set()) {
        terminateWithSignal(signal);
    }
    // Print the cached process identity even if stack unwinding/symbolization fails.
    // No HPX calls are made from the signal handler.
    // POSIX open/write/close are signal-safe. Opening only on a fault also
    // preserves the existing behavior of creating no file on successful runs.
    int const descriptor =
        ::open(stackTracePath.load(), O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0666);
    writeDiagnostic(descriptor, processHeader.data(), processHeaderSize);
    switch (signal) {
    case SIGABRT:
        writeDiagnostic(descriptor, "Signal: SIGABRT\n");
        break;
    case SIGSEGV:
        writeDiagnostic(descriptor, "Signal: SIGSEGV\n");
        break;
    case SIGBUS:
        writeDiagnostic(descriptor, "Signal: SIGBUS\n");
        break;
    case SIGFPE:
        writeDiagnostic(descriptor, "Signal: SIGFPE\n");
        break;
    case SIGILL:
        writeDiagnostic(descriptor, "Signal: SIGILL\n");
        break;
    }
    // Source-line formatting is best effort: Boost/libbacktrace can allocate and
    // is not async-signal-safe. This is a diagnostic, not crash recovery.
    try {
        auto const trace = stackTrace();
        writeDiagnostic(descriptor, trace.data(), trace.size());
    } catch (...) {
        writeDiagnostic(descriptor, "Unable to symbolize the stacktrace.\n");
    }
    if (descriptor >= 0) {
        ::close(descriptor);
    }
    // Preserve signal termination (and core dumps), rather than exit(-1).
    terminateWithSignal(signal);
}
#endif

}    // namespace detail

#if !defined(_MSC_VER)
struct StackTraceHandler::State {
    using SignalActions = std::array<struct sigaction, detail::kFatalSignals.size()>;
    SignalActions previous{};
    std::array<bool, detail::kFatalSignals.size()> installed{};

    explicit State(unsigned int localityId) {
        // Resolve the backend before handling a fault, outside signal context.
        (void) boost::stacktrace::to_string(boost::stacktrace::stacktrace(0, 1));
        bool expected = false;
        if (!detail::ownerActive.compare_exchange_strong(expected, true)) {
            throw std::logic_error("Only one stacktrace handler may own a process");
        }
        std::array<char, 256> hostname{};
        if (gethostname(hostname.data(), hostname.size() - 1) != 0) {
            std::snprintf(hostname.data(), hostname.size(), "unknown");
        }
        auto const count = std::snprintf(detail::processHeader.data(), detail::processHeader.size(),
            "CosmicTiger stacktrace: host=%s pid=%ld locality=%u\n", hostname.data(),
            static_cast<long>(getpid()), localityId);
        detail::processHeaderSize = count > 0 ? static_cast<size_t>(count) : 0;
        if (detail::processHeaderSize >= detail::processHeader.size()) {
            detail::processHeaderSize = detail::processHeader.size() - 1;
        }
        detail::reporting.clear();
    }

    void install() {
        struct sigaction action{};
        action.sa_handler = detail::reportSignal;
        sigemptyset(&action.sa_mask);
        // Use an existing alternate signal stack, e.g. HPX's, when available.
        action.sa_flags = SA_ONSTACK;
        for (size_t i = 0; i < detail::kFatalSignals.size(); ++i) {
            if (sigaction(detail::kFatalSignals[i], &action, &previous[i]) != 0) {
                throw std::system_error(errno, std::generic_category(),
                    "Unable to install the CosmicTiger stacktrace handler");
            }
            installed[i] = true;
        }
    }

    ~State() {
        for (size_t i = detail::kFatalSignals.size(); i > 0; --i) {
            auto const index = i - 1;
            struct sigaction current{};
            if (installed[index] &&
                sigaction(detail::kFatalSignals[index], nullptr, &current) == 0 &&
                !(current.sa_flags & SA_SIGINFO) && current.sa_handler == detail::reportSignal) {
                // Do not overwrite a handler subsequently installed by another subsystem.
                sigaction(detail::kFatalSignals[index], &previous[index], nullptr);
            }
        }
        detail::ownerActive.store(false);
    }
};
#else
struct StackTraceHandler::State {};
#endif

StackTraceHandler::StackTraceHandler() = default;
StackTraceHandler::~StackTraceHandler() = default;

void setStackTraceOutputPath(std::string const& filename) {
    std::lock_guard<std::mutex> lock(detail::stackTracePathMutex);
    if (filename != detail::stackTracePath.load()) {
        detail::stackTracePaths.push_back(std::make_unique<std::string const>(filename));
        detail::stackTracePath.store(detail::stackTracePaths.back()->c_str());
    }
}

std::string stackTraceOutputPath() {
    return detail::stackTracePath.load();
}

std::string stackTrace() {
    return boost::stacktrace::to_string(boost::stacktrace::stacktrace(0, 64));
}

void StackTraceHandler::install(unsigned int localityId) {
#if !defined(_MSC_VER)
    if (!state_) {
        auto candidate = std::make_unique<State>(localityId);
        candidate->install();
        state_ = std::move(candidate);
    }
#else
    (void) localityId;
#endif
}
