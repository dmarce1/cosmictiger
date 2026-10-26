/* CosmicTiger - HPX locality stack-trace regression; BSL-1.0. */

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime_distributed/find_all_localities.hpp>
#include <hpx/runtime_distributed/find_here.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/runtime_local/startup_function.hpp>
#include <sys/resource.h>

#include <cosmictiger/core/stack_trace.hpp>

namespace detail {

using SignalHandler = void (*)(int);

std::atomic<unsigned int> mainCalls{0};
std::atomic<SignalHandler> installedHandler{nullptr};

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}    // namespace detail

unsigned int inspectStackTraceLocality(bool expectMain) {
    struct sigaction current{};
    auto const expected = detail::installedHandler.load();
    bool const installed = expected != nullptr && sigaction(SIGABRT, nullptr, &current) == 0 &&
        !(current.sa_flags & SA_SIGINFO) && current.sa_handler == expected;
    auto const locality = hpx::get_locality_id();
    auto const calls = detail::mainCalls.load();
    std::printf("STACK_TRACE_LOCALITY locality=%u hpx_main_calls=%u installed=%u\n", locality,
        calls, installed ? 1U : 0U);
    std::fflush(stdout);
    if (installed && calls == (expectMain ? 1U : 0U)) {
        return locality;
    }
    return std::numeric_limits<unsigned int>::max();
}
HPX_PLAIN_ACTION(inspectStackTraceLocality, InspectStackTraceLocalityAction);

// A returning raise keeps this named source frame visible even with optimization.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void stackTraceHpxCrash() {
    std::raise(SIGABRT);
    std::_Exit(98);
}
HPX_PLAIN_ACTION(stackTraceHpxCrash, StackTraceHpxCrashAction);

int hpx_main(hpx::program_options::variables_map& arguments) {
    ++detail::mainCalls;
    auto const localities = hpx::find_all_localities();
    auto const expectedCount = arguments["expected-localities"].as<unsigned int>();
    detail::require(localities.size() == expectedCount, "unexpected HPX locality count");
    auto const crashLocality = arguments["crash-locality"].as<int>();
    hpx::id_type crashTarget;
    for (auto const& locality : localities) {
        auto const id =
            hpx::async<InspectStackTraceLocalityAction>(locality, locality == hpx::find_here())
                .get();
        detail::require(id != std::numeric_limits<unsigned int>::max(),
            "stack trace startup did not cover a locality, or worker hpx_main ran");
        if (crashLocality >= 0 && id == static_cast<unsigned int>(crashLocality)) {
            crashTarget = locality;
        }
    }
    if (crashLocality >= 0) {
        detail::require(static_cast<bool>(crashTarget), "requested crash locality is absent");
        hpx::async<StackTraceHpxCrashAction>(crashTarget).get();
        return 98;
    }
    return hpx::finalize();
}

int main(int argc, char* argv[]) {
    StackTraceHandler stackTraceHandler;
    try {
        struct rlimit const coreLimit{0, 0};
        detail::require(setrlimit(RLIMIT_CORE, &coreLimit) == 0, "cannot disable core dumps");
        hpx::register_startup_function([&stackTraceHandler]() {
            struct sigaction before{};
            struct sigaction after{};
            detail::require(sigaction(SIGABRT, nullptr, &before) == 0,
                "cannot inspect the previous SIGABRT handler");
            stackTraceHandler.install(hpx::get_locality_id());
            detail::require(sigaction(SIGABRT, nullptr, &after) == 0,
                "cannot inspect the installed SIGABRT handler");
            detail::require(!(after.sa_flags & SA_SIGINFO) && after.sa_handler != SIG_DFL &&
                    after.sa_handler != SIG_IGN &&
                    ((before.sa_flags & SA_SIGINFO) || after.sa_handler != before.sa_handler),
                "startup did not install the stack trace handler");
            detail::installedHandler.store(after.sa_handler);
        });
        hpx::program_options::options_description description("Stack trace fixture");
        description.add_options()("expected-localities",
            hpx::program_options::value<unsigned int>()->default_value(1),
            "Expected locality count")("crash-locality",
            hpx::program_options::value<int>()->default_value(-1),
            "Locality to terminate with SIGABRT");
        hpx::init_params parameters;
        parameters.desc_cmdline = description;
        parameters.cfg = {"hpx.stacks.small_size=524288"};
        return hpx::init(argc, argv, parameters);
    } catch (std::exception const& error) {
        std::fprintf(stderr, "HPX stack trace fixture failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
