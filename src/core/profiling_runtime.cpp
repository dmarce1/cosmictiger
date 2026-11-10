/* CosmicTiger - runtime profiling defaults; BSL-1.0. */

#include <cerrno>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <system_error>

#include <hpx/config.hpp>
#if defined(HPX_HAVE_APEX)
#include <hpx/runtime_local/run_as_os_thread.hpp>
#include <hpx/runtime_local/shutdown_function.hpp>
#include <hpx/threading_base/external_timer.hpp>
#endif

#include <cosmictiger/core/profiling_runtime.hpp>

namespace profile {

/// Configure file-based APEX reports before startup and register HPX shutdown finalization.
void configureRuntimeProfiling() {
#if defined(HPX_HAVE_APEX)
    // APEX reads its options lazily during initialization. Set defaults before
    // MPI/HPX can initialize it; HPX remains responsible for its lifetime.
    // Keep detailed task reports in files; the application prints phase timings.
    if (::setenv("APEX_SCREEN_OUTPUT", "0", 0) != 0) {
        throw std::system_error(errno, std::generic_category(),
            "Unable to configure profiling option APEX_SCREEN_OUTPUT");
    }
    for (char const* name : {"APEX_CSV_OUTPUT", "APEX_TASKTREE_OUTPUT"}) {
        if (::setenv(name, "1", 0) != 0) {
            throw std::system_error(errno, std::generic_category(),
                std::string("Unable to configure profiling option ") + name);
        }
    }
    // Register before hpx::init installs APEX's own pre-shutdown callback.
    // Its MPI report reductions must not block the only HPX worker before
    // the other ranks receive their shutdown actions. Wait cooperatively for
    // native-thread finalization; APEX's later finalize call is idempotent.
    hpx::register_pre_shutdown_function(
        []() { hpx::run_as_os_thread([]() { hpx::util::external_timer::finalize(); }).get(); });
#endif
}

}    // namespace profile
