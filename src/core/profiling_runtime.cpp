/* CosmicTiger - runtime profiling defaults; BSL-1.0. */

#include <cerrno>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <system_error>

#include <hpx/config.hpp>
#if defined(__linux__)
#include <dlfcn.h>
#endif

#include <cosmictiger/core/profiling_runtime.hpp>

namespace profile {

void configureRuntimeProfiling() {
    char const* tau = std::getenv("APEX_TAU");
    if (tau != nullptr && std::atoi(tau) != 0) {
#if defined(HPX_HAVE_APEX) && defined(__linux__)
        for (char const* name :
            {"APEX_DISABLE", "APEX_SUSPEND", "APEX_TIME_TOP_LEVEL_OS_THREADS"}) {
            char const* setting = std::getenv(name);
            if (setting != nullptr && std::atoi(setting) != 0) {
                throw std::runtime_error(std::string("TAU task profiling requires ") + name +
                    "=0; HPX supplies the task execution intervals");
            }
            // APEX reads apex.conf later and only fills missing environment
            // entries. Keep that file from disabling task capture after this check.
            if (::setenv(name, "0", 0) != 0) {
                throw std::system_error(errno, std::generic_category(),
                    std::string("Unable to configure profiling option ") + name);
            }
        }
        if (::dlsym(RTLD_DEFAULT, "Tau_init") == nullptr) {
            throw std::runtime_error("APEX_TAU requires preloaded TAU libraries; "
                                     "launch with scripts/profile_tau.sh");
        }
#else
        throw std::runtime_error("TAU task profiling requires Linux and an APEX-enabled HPX");
#endif
    }
#if defined(HPX_HAVE_APEX)
    // APEX reads its options lazily during initialization. Set defaults before
    // MPI/HPX can initialize it; HPX remains responsible for its lifetime.
    for (char const* name : {"APEX_SCREEN_OUTPUT", "APEX_CSV_OUTPUT", "APEX_TASKTREE_OUTPUT"}) {
        if (::setenv(name, "1", 0) != 0) {
            throw std::system_error(errno, std::generic_category(),
                std::string("Unable to configure profiling option ") + name);
        }
    }
#endif
}

}    // namespace profile
