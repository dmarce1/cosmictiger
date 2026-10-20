#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/cosmology/initialize.hpp>
#include <cosmictiger/scenarios/cosmology.hpp>

namespace scenarios {

void initializeCosmology(DriverParameters& state) {
    state.a = 1.0 / (1.0 + getOptions().cosmology.startRedshift);
    state.adot = cosmosDadt(state.a);
    state.tauMax = cosmosConformalTime(state.a, 1.0);
    initialize(getOptions().cosmology.startRedshift);
}

}    // namespace scenarios
