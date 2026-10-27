#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/interrupt_profile.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cosmology/background.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/cosmology/evolution.hpp>
#include <cosmictiger/cosmology/initialize.hpp>
#include <cosmictiger/scenarios/cosmology.hpp>

namespace scenarios {

void initializeCosmology(DriverParameters& state) {
    state.a = 1.0 / (1.0 + getOptions().cosmology.startRedshift);
    state.adot = cosmosDadt(state.a);
    state.tauMax = cosmology::configuredDuration(state.a);
    // Validate the whole requested trajectory before generating particles.
    cosmology::simulationBackground(state.tauMax);
    profile::checkForInterrupt();
    initialize(getOptions().cosmology.startRedshift);
}

}    // namespace scenarios
