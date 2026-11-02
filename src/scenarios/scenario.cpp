#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

#include <cosmictiger/scenarios/scenario.hpp>

#if COSMICTIGER_WITH_SPH
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/scenarios/blast.hpp>
#include <cosmictiger/scenarios/diagnostics.hpp>
#include <cosmictiger/scenarios/disc.hpp>
#include <cosmictiger/scenarios/gresho.hpp>
#include <cosmictiger/scenarios/hydro_wave.hpp>
#include <cosmictiger/scenarios/kelvin_helmholtz.hpp>
#include <cosmictiger/scenarios/polytropic_star.hpp>
#include <cosmictiger/scenarios/rayleigh_taylor.hpp>
#include <cosmictiger/scenarios/sod.hpp>
#if COSMICTIGER_ENABLE_HOPKINS
#include <cosmictiger/scenarios/sound_wave.hpp>
#endif
#include <cosmictiger/hydro/sph/sph_particles.hpp>
#endif

namespace scenarios {

#if COSMICTIGER_WITH_SPH
namespace detail {

    std::vector<DiagnosticSample> localDiagnosticSamples(size_t maximum) {
        size_t const count = sphParticlesSize();
        size_t const samples = std::min(count, maximum);
        std::vector<DiagnosticSample> result;
        result.reserve(samples);
        std::mt19937 generator(42 + hpxRank());
        if (count != 0) {
            std::uniform_int_distribution<size_t> distribution(0, count - 1);
            for (size_t sample = 0; sample < samples; ++sample) {
                size_t const particle = count <= maximum ? sample : distribution(generator);
                DiagnosticSample value;
                for (int dim = 0; dim < kDimensions; ++dim) {
                    value.position[dim] = sphParticlesPosition(dim, particle).toDouble();
                }
                value.density = sphParticlesDensity(particle);
                value.pressure = (getOptions().hydro.gamma - 1.0) * value.density *
                    sphParticlesInternalEnergy(particle);
                value.weight = double(count) / samples;
                result.push_back(value);
            }
        }
        return result;
    }
}    // namespace detail
}    // namespace scenarios

HPX_PLAIN_ACTION(scenarios::detail::localDiagnosticSamples, LocalDiagnosticSamplesAction);

namespace scenarios {
std::vector<DiagnosticSample> diagnosticSamples(size_t maximumPerLocality) {
    std::vector<hpx::future<std::vector<DiagnosticSample>>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<LocalDiagnosticSamplesAction>(locality, maximumPerLocality));
    }
    std::vector<DiagnosticSample> result;
    for (auto& future : futures) {
        auto samples = future.get();
        result.insert(result.end(), samples.begin(), samples.end());
    }
    return result;
}
#endif

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
#if COSMICTIGER_ENABLE_HOPKINS
        {"soundWave", false, true, true, 1, initializeSoundWave, 1},
#endif
        {"gresho", false, true, true, 16, initializeGresho},
#endif
#if COSMICTIGER_WITH_GRAVITY
        {"cosmology", true, COSMICTIGER_WITH_SPH != 0, false, 64, nullptr},
        {"glass", true, false, false, 64, nullptr},
#endif
    };
    for (auto const& scenario : definitions) {
        if (name == scenario.name) {
            return scenario;
        }
    }
#if !COSMICTIGER_ENABLE_HOPKINS
    if (name == "soundWave") {
        throw std::invalid_argument("run.scenario=soundWave requires "
                                    "COSMICTIGER_ENABLE_HOPKINS=ON "
                                    "(and COSMICTIGER_ENABLE_SPH=ON)");
    }
#endif
#if !COSMICTIGER_WITH_SPH
    if (name == "sod" || name == "blast" || name == "rayleighTaylor" || name == "kelvinHelmholtz" ||
        name == "disc" || name == "polytropicStar" || name == "hydroWave" || name == "gresho" ||
        name == "soundWave") {
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
