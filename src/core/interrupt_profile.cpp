/* CosmicTiger - Ctrl-C profile reporting; BSL-1.0. */

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <system_error>
#include <thread>

#if defined(__linux__)
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#endif

#include <cosmictiger/core/interrupt_profile.hpp>
#include <cosmictiger/core/profiler.hpp>

namespace profile {

#if defined(__linux__)
namespace detail {

    constexpr std::uint64_t kClosed = std::uint64_t{1} << 63;
    std::atomic<std::uint64_t> signalGate{kClosed};
    std::atomic<int> notificationDescriptor{-1};
    std::atomic<bool> interruptOwner{false};
    std::atomic_flag interruptRequested = ATOMIC_FLAG_INIT;
    static_assert(decltype(signalGate)::is_always_lock_free);
    static_assert(decltype(notificationDescriptor)::is_always_lock_free);

    [[noreturn]] void terminateWithInterrupt() noexcept {
        struct sigaction action{};
        action.sa_handler = SIG_DFL;
        ::sigemptyset(&action.sa_mask);
        ::sigaction(SIGINT, &action, nullptr);
        sigset_t mask{};
        ::sigemptyset(&mask);
        ::sigaddset(&mask, SIGINT);
        ::sigprocmask(SIG_UNBLOCK, &mask, nullptr);
        ::raise(SIGINT);
        std::_Exit(128 + SIGINT);
    }

    // No allocation, stdio, locks, HPX, or profiler teardown in signal context.
    // Returning lets an interrupted sampling handler finish before we drain it.
    void notifyInterrupt(int) noexcept {
        int const savedErrno = errno;
        auto gate = signalGate.load(std::memory_order_acquire);
        do {
            if ((gate & kClosed) != 0) {
                if (interruptRequested.test(std::memory_order_relaxed)) {
                    terminateWithInterrupt();
                }
                errno = savedErrno;
                return;
            }
        } while (!signalGate.compare_exchange_weak(
            gate, gate + 1, std::memory_order_acq_rel, std::memory_order_acquire));
        if (interruptRequested.test_and_set(std::memory_order_relaxed)) {
            terminateWithInterrupt();
        }
        char const notification = 1;
        ssize_t written;
        do {
            written = ::write(notificationDescriptor.load(std::memory_order_relaxed), &notification,
                sizeof(notification));
        } while (written < 0 && errno == EINTR);
        signalGate.fetch_sub(1, std::memory_order_release);
        if (written != sizeof(notification)) {
            terminateWithInterrupt();
        }
        errno = savedErrno;
    }

}    // namespace detail

struct InterruptReportHandler::State {
    int descriptors[2]{-1, -1};
    std::thread reporter;
    struct sigaction previous{};
    bool installed = false;

    State() {
        bool expected = false;
        if (!detail::interruptOwner.compare_exchange_strong(expected, true)) {
            throw std::logic_error("Only one interrupt profile handler may own a process");
        }
        try {
            if (::pipe2(descriptors, O_CLOEXEC) != 0 ||
                ::fcntl(descriptors[1], F_SETFL, O_NONBLOCK) != 0) {
                throw std::system_error(
                    errno, std::generic_category(), "Unable to create profile interrupt pipe");
            }
            reporter = std::thread([this]() { report(); });
            detail::interruptRequested.clear();
            detail::notificationDescriptor.store(descriptors[1], std::memory_order_relaxed);
            detail::signalGate.store(0, std::memory_order_release);
            struct sigaction action{};
            action.sa_handler = detail::notifyInterrupt;
            ::sigemptyset(&action.sa_mask);
            action.sa_flags = SA_RESTART | SA_ONSTACK;
            if (::sigaction(SIGINT, &action, &previous) != 0) {
                throw std::system_error(
                    errno, std::generic_category(), "Unable to install profile interrupt handler");
            }
            installed = true;
        } catch (...) {
            stop();
            throw;
        }
    }

    void report() noexcept {
        char notification = 0;
        ssize_t count;
        do {
            count = ::read(descriptors[0], &notification, sizeof(notification));
        } while (count < 0 && errno == EINTR);
        if (count == sizeof(notification) && notification != 0) {
            std::fputs(
                "\nCtrl-C: writing CPU profile; press Ctrl-C again to terminate immediately.\n",
                stderr);
            profileStop();
            writeProfileReport();
            // Preserve interrupt termination without invoking HPX's abort handler
            // or running global destructors concurrently with live workers.
            detail::terminateWithInterrupt();
        }
    }

    void stop() noexcept {
        // Close admission, then drain every handler that could have retained the
        // descriptor before joining/closing it. A late handler sees a closed gate.
        detail::signalGate.fetch_or(detail::kClosed, std::memory_order_acq_rel);
        while ((detail::signalGate.load(std::memory_order_acquire) & ~detail::kClosed) != 0) {
            std::this_thread::yield();
        }
        detail::notificationDescriptor.store(-1, std::memory_order_relaxed);
        if (reporter.joinable()) {
            char const notification = 0;
            ssize_t written;
            do {
                written = ::write(descriptors[1], &notification, sizeof(notification));
            } while (written < 0 && errno == EINTR);
            // At most one interrupt byte precedes this shutdown byte, so the
            // nonblocking pipe cannot fill. Join also serializes an interrupt
            // report racing normal shutdown with the later atexit report.
            reporter.join();
        }
        if (installed) {
            struct sigaction current{};
            if (::sigaction(SIGINT, nullptr, &current) == 0 && !(current.sa_flags & SA_SIGINFO) &&
                current.sa_handler == detail::notifyInterrupt) {
                ::sigaction(SIGINT, &previous, nullptr);
            }
        }
        for (int descriptor : descriptors) {
            if (descriptor >= 0) {
                ::close(descriptor);
            }
        }
        detail::interruptOwner.store(false);
    }

    ~State() {
        stop();
    }
};
#else
struct InterruptReportHandler::State {};
#endif

InterruptReportHandler::InterruptReportHandler() = default;
InterruptReportHandler::~InterruptReportHandler() = default;

void InterruptReportHandler::install() {
    if (!state_) {
        state_ = std::make_unique<State>();
    }
}

}    // namespace profile
