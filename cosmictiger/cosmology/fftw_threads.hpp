/* CosmicTiger - FFTW parallel work on HPX workers; BSL-1.0. */

#pragma once

namespace fftw_threads {

// Register as an HPX startup function on every locality, after FFTW thread
// initialization and before any FFTW plans are created.
void initialize();

}    // namespace fftw_threads
