#include <stdexcept>
#include <vector>

#include <cosmictiger/scenarios/scenario.hpp>

#if COSMICTIGER_WITH_SPH
#include <cosmictiger/scenarios/blast.hpp>
#include <cosmictiger/scenarios/disc.hpp>
#include <cosmictiger/scenarios/hydro_wave.hpp>
#include <cosmictiger/scenarios/kelvin_helmholtz.hpp>
#include <cosmictiger/scenarios/polytropic_star.hpp>
#include <cosmictiger/scenarios/rayleigh_taylor.hpp>
#include <cosmictiger/scenarios/sod.hpp>
#endif

namespace scenarios {

ScenarioDefinition const& definition(std::string const& name) {
    static std::vector<ScenarioDefinition> const definitions{
#if COSMICTIGER_WITH_GRAVITY
        {"plummer", true, false, false, 64, nullptr},
#endif
#if COSMICTIGER_WITH_SPH
        {"sod", false, true, true, 20, initializeSod},
        {"blast", false, true, true, 16, initializeBlast},
        {"rayleighTaylor", true, true, true, 256, initializeRayleighTaylor},
        {"kelvinHelmholtz", false, true, true, 1024, initializeKelvinHelmholtz},
        {"disc", true, true, true, 1024, initializeDisc},
        {"polytropicStar", true, true, true, 275, initializePolytropicStar},
        {"hydroWave", false, true, true, 64, initializeHydroWave},
#endif
        {"cosmology", true, COSMICTIGER_WITH_SPH != 0, false, 64, nullptr},
#if COSMICTIGER_WITH_GRAVITY
        {"glass", true, false, false, 64, nullptr},
#endif
    };
    for (auto const& scenario : definitions) {
        if (name == scenario.name) {
            return scenario;
        }
    }
#if !COSMICTIGER_WITH_SPH
    if (name == "sod" || name == "blast" || name == "rayleighTaylor" || name == "kelvinHelmholtz" ||
        name == "disc" || name == "polytropicStar" || name == "hydroWave") {
        throw std::invalid_argument("run.scenario=" + name + " requires COSMICTIGER_ENABLE_SPH=ON");
    }
#endif
    std::string available;
    for (auto const& scenario : definitions) {
        if (!available.empty()) {
            available += ", ";
        }
        available += scenario.name;
    }
    throw std::invalid_argument("Unavailable run.scenario '" + name + "'; choose " + available);
}

}    // namespace scenarios
