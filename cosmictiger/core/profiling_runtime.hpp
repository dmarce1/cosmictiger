/* CosmicTiger - runtime profiling defaults; BSL-1.0. */

#pragma once

namespace profile {

// Call before MPI or HPX initialization, including when CPU sampling is disabled.
// Existing APEX environment settings take precedence over these defaults.
// Registers cooperative profiling teardown before HPX's own shutdown callbacks.
void configureRuntimeProfiling();

}    // namespace profile
