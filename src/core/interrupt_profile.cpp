/* CosmicTiger - cooperative Ctrl-C shutdown; BSL-1.0. */

#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/runtime.hpp>

#if defined(__linux__)
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cosmictiger/core/interrupt_profile.hpp>

namespace profile {

namespace detail {

    std::atomic<bool> shutdownRequested{false};
    static_assert(decltype(shutdownRequested)::is_always_lock_free);

    bool localInterruptRequested() noexcept {
        return shutdownRequested.load(std::memory_order_relaxed);
    }
}    // namespace detail
}    // namespace profile

HPX_PLAIN_ACTION(profile::detail::localInterruptRequested, CosmictigerInterruptQueryAction);

namespace profile {
namespace detail {

#if defined(__linux__)
    constexpr std::uint64_t kClosed = std::uint64_t{1} << 63;
    std::atomic<std::uint64_t> signalGate{kClosed};
    std::atomic<int> notificationDescriptor{-1};
    std::atomic<bool> interruptOwner{false};
    std::atomic_flag interruptRequested = ATOMIC_FLAG_INIT;
    static_assert(decltype(signalGate)::is_always_lock_free);
    static_assert(decltype(notificationDescriptor)::is_always_lock_free);

    void notifyLauncher() {
        char const* setting = std::getenv("COSMICTIGER_INTERRUPT_READY_FD");
        if (setting == nullptr) {
            return;
        }
        char* end = nullptr;
        errno = 0;
        long const value = std::strtol(setting, &end, 10);
        if (errno != 0 || end == setting || *end != '\0' || value <= STDERR_FILENO ||
            value > INT_MAX) {
            throw std::runtime_error("Invalid COSMICTIGER_INTERRUPT_READY_FD");
        }
        int const descriptor = static_cast<int>(value);
        struct stat information{};
        if (::fstat(descriptor, &information) != 0 || !S_ISFIFO(information.st_mode)) {
            throw std::runtime_error("COSMICTIGER_INTERRUPT_READY_FD must name a pipe");
        }
        char message[32];
        int const length = std::snprintf(message, sizeof(message), "%ld\n", long(::getpid()));
        ssize_t count;
        do {
            count = ::write(descriptor, message, static_cast<size_t>(length));
        } while (count < 0 && errno == EINTR);
        int const error = count < 0 ? errno : EIO;
        ::close(descriptor);
        if (count != length) {
            throw std::system_error(error, std::generic_category(),
                "Unable to notify the MPI launcher that interrupt handling is ready");
        }
    }

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
        shutdownRequested.store(true, std::memory_order_relaxed);
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
#endif

}    // namespace detail

#if defined(__linux__)
struct InterruptReportHandler::State {
    int descriptors[2]{-1, -1};
    std::thread reporter;
    struct sigaction previous{};
    struct sigaction previousUser{};
    bool installed = false;
    bool installedUser = false;

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
            detail::shutdownRequested.store(false, std::memory_order_relaxed);
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
            if (::sigaction(SIGUSR1, &action, &previousUser) != 0) {
                throw std::system_error(
                    errno, std::generic_category(), "Unable to install MPI shutdown handler");
            }
            installedUser = true;
            detail::notifyLauncher();
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
            std::fputs("\nCtrl-C: clean shutdown requested; finishing active work. "
                       "Press Ctrl-C again to terminate immediately.\n",
                stderr);
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
            // nonblocking pipe cannot fill. Join keeps the notification thread
            // within the lifetime of this handler owner.
            reporter.join();
        }
        if (installed) {
            struct sigaction current{};
            if (::sigaction(SIGINT, nullptr, &current) == 0 && !(current.sa_flags & SA_SIGINFO) &&
                current.sa_handler == detail::notifyInterrupt) {
                ::sigaction(SIGINT, &previous, nullptr);
            }
        }
        if (installedUser) {
            struct sigaction current{};
            if (::sigaction(SIGUSR1, nullptr, &current) == 0 && !(current.sa_flags & SA_SIGINFO) &&
                current.sa_handler == detail::notifyInterrupt) {
                ::sigaction(SIGUSR1, &previousUser, nullptr);
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

char const* Interrupted::what() const noexcept {
    return "Clean shutdown requested by interrupt";
}

bool interruptRequested() {
    if (detail::localInterruptRequested()) {
        return true;
    }
    // Only the root drives the application. Query worker flags here instead of
    // invoking HPX or MPI from the signal/notification thread during teardown.
    std::vector<hpx::future<bool>> requests;
    auto const here = hpx::find_here();
    for (auto const& locality : hpx::find_all_localities()) {
        if (locality != here) {
            requests.push_back(hpx::async<CosmictigerInterruptQueryAction>(locality));
        }
    }
    bool requested = false;
    for (auto& request : requests) {
        request.wait();
    }
    for (auto& request : requests) {
        requested = request.get() || requested;
    }
    if (requested) {
        detail::shutdownRequested.store(true, std::memory_order_relaxed);
    }
    return requested;
}

void checkForInterrupt() {
    if (interruptRequested()) {
        throw Interrupted{};
    }
}

}    // namespace profile
