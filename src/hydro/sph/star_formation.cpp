/* CosmicTiger - GADGET-4 cooling and star-formation SPH coupling; BSL-1.0. */
#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/hydro/sph/sph_chemistry.hpp>
#include <cosmictiger/hydro/sph/sph_particles.hpp>
#include <cosmictiger/hydro/sph/star_formation.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/simulation_units.hpp>
#include <cosmictiger/math/vector3.hpp>
#ifdef USE_CUDA
#include <cosmictiger/hydro/sph/star_formation_cuda.hpp>
#endif

namespace detail {
using Model = springel_and_hernquist::Model;
using Input = star_formation_backend::Input;
using Output = star_formation_backend::Output;
using Parameters = star_formation_backend::Parameters;
using Operation = star_formation_backend::Operation;

Model model() {
    auto const& options = getOptions();
    auto const& settings = options.starFormation;
    auto parameters = springel_and_hernquist::defaultParameters();
    parameters.beta = settings.beta;
    parameters.evaporationFactor = settings.evaporationFactor;
    parameters.starFormationTime =
        settings.starFormationTimeGyr * 1.e9 * springel_and_hernquist::kSecondsPerYear;
    parameters.coldEnergy *= settings.coldTemperature / 1000.0;
    parameters.supernovaEnergy *= settings.supernovaTemperature / 1.e8;
    auto const cooling = sphCoolingParameters(1.0);
    // GADGET-4 constants.h normalization; this does not alter the gravity solver.
    double const hubble = 3.2407789e-18 * options.cosmology.hubble;
    double const criticalDensity =
        3.0 * hubble * hubble / (8.0 * 3.14159265358979323846 * 6.6738e-8);
    Model result;
    if (settings.criticalPhysicalDensity > 0.0) {
        result = {parameters,
            settings.criticalPhysicalDensity * gadget4::kProtonMass /
                gadget4::kHydrogenMassFraction,
            0.0};
        if (!springel_and_hernquist::validParameters(parameters)) {
            throw std::runtime_error("Invalid GADGET-4 star-formation parameters");
        }
    } else if (!springel_and_hernquist::makeModel(parameters, cooling,
                   cooling.comoving ? 1.e6 * criticalDensity : 1.e-23, result)) {
        throw std::runtime_error("Cannot calibrate the GADGET-4 physical density threshold");
    }
    result.overdensityThreshold =
        settings.criticalOverDensity * options.cosmology.omegaB * criticalDensity;
    return result;
}
Parameters parameters(double scaleFactor) {
    auto const& options = getOptions();
    auto const scales = units::scalesFromOptions(options);
    Parameters result;
    result.model = model();
    result.cooling = sphCoolingParameters(scaleFactor);
    result.densityScale = scales.densityScale().value();
    result.energyScale = scales.specificEnergyScale().value();
    result.scaleFactor = scaleFactor;
    result.timeSeconds = options.units.timeSeconds;
    result.gasMass = options.initial.gasMass;
    result.seed = options.starFormation.seed;
    return result;
}
Input input(ParticleInteger gas) {
    auto const particle = sphParticlesParticleIndex(gas);
    Input result;
    result.state = particlesStarFormationStates[particle];
    result.chemistry = sphParticlesChemistryStates[gas];
    result.energy = sphParticlesInternalEnergy(gas);
    result.density = sphParticlesDensity(gas);
    result.pressureRatio = float(gadget4::kGammaMinusOne);
    result.rung = sphParticlesRung(gas);
    for (int dim = 0; dim < kDimensions; ++dim) {
        result.velocity[dim] = particlesV[particle][dim];
    }
    return result;
}
std::vector<Input> inputs() {
    std::vector<Input> result(sphParticlesSize());
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        result[gas] = input(gas);
    }
    return result;
}
std::vector<Output> run(
    Operation operation, std::vector<Input> const& inputs, Parameters const& parameters) {
    std::vector<Output> outputs(inputs.size());
#ifdef USE_CUDA
    if (getOptions().starFormation.cuda) {
        cudaSetDevice();
        try {
            star_formation_backend::runCuda(
                operation, inputs.data(), outputs.data(), inputs.size(), parameters);
        } catch (std::exception const& error) {
            throw std::runtime_error(
                "GADGET-4 SF rank=" + std::to_string(hpxRank()) + ": " + error.what());
        }
        return outputs;
    }
#endif
    for (size_t index = 0; index < inputs.size(); ++index) {
        outputs[index] = star_formation_backend::evaluate(operation, inputs[index], {}, parameters);
        if (outputs[index].status != 0) {
            auto const& input = inputs[index];
            double const step = parameters.explicitTimestep >= 0.0 ?
                parameters.explicitTimestep :
                std::ldexp(parameters.baseTimestep, -input.rung);
            std::ostringstream message;
            message << std::scientific << std::setprecision(17) << "GADGET-4 SF rank=" << hpxRank()
                    << " particle_id=" << input.state.id << " batch_index=" << index
                    << " status=" << star_formation_backend::statusName(outputs[index].status)
                    << " rho_g_cm3=" << star_formation_backend::detail::density(input, parameters)
                    << " u_erg_g="
                    << star_formation_backend::detail::energy(input.energy, parameters)
                    << " dt_seconds=" << parameters.scaleFactor * step * parameters.timeSeconds
                    << "; no source state committed";
            throw std::runtime_error(message.str());
        }
    }
    return outputs;
}
struct Pending {
    ParticleInteger particle;
    Output output;
};
struct NewStar {
    ParticleInteger particle;
    Vector3<float> gravity;
};
std::vector<Pending> pending;
std::vector<NewStar> newStars;
}    // namespace detail

uint64_t starFormationRandomBits(uint64_t identity, uint64_t event, uint64_t stream) {
    return star_formation_backend::detail::mix(
        star_formation_backend::detail::mix(identity ^ getOptions().starFormation.seed) ^
        star_formation_backend::detail::mix(event + UINT64_C(0x9e3779b97f4a7c15)) ^
        star_formation_backend::detail::mix(stream + UINT64_C(0x632be59bd9b4e019)));
}
double starFormationUniform(uint64_t identity, uint64_t event, uint64_t stream) {
    return double(starFormationRandomBits(identity, event, stream) >> 11) * 0x1.0p-53;
}
star_formation_backend::Parameters starFormationParameters(double scaleFactor) {
    return detail::parameters(scaleFactor);
}
springel_and_hernquist::Model starFormationModel() {
    return detail::model();
}
star_formation_backend::Output starFormationDiagnostics(ParticleInteger gas, double scaleFactor) {
    auto const input = detail::input(gas);
    auto result = star_formation_backend::diagnostics(input, detail::parameters(scaleFactor));
    if (result.status != 0) {
        throw std::runtime_error("Cannot evaluate GADGET-4 SF output diagnostics");
    }
    // GADGET-4 saves SFR after cooling for subsequent stochastic conversion.
    result.specificStarFormationRate = input.state.specificStarFormationRate;
    return result;
}
void starFormationInitialize() {
    auto const model = detail::model();
    particlesInitializeStarFormationIds();
    PRINT("GADGET-4 effective star formation: rho_th=%.12e g/cm^3, nH_th=%.12e cm^-3; "
          "whole-particle conversion, equilibrium H/He cooling\n",
        model.thresholdDensity,
        model.thresholdDensity * gadget4::kHydrogenMassFraction / gadget4::kProtonMass);
}

StarFormationUpdate starFormationPrepare(
    int minimumRung, double scaleFactor, double baseTimestep, double explicitTimestep);
void starFormationCommit();
HPX_PLAIN_ACTION(starFormationPrepare, StarFormationPrepareAction);
HPX_PLAIN_ACTION(starFormationCommit, StarFormationCommitAction);
StarFormationUpdate starFormationPrepare(
    int minimumRung, double scaleFactor, double baseTimestep, double explicitTimestep) {
    std::vector<hpx::future<StarFormationUpdate>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<StarFormationPrepareAction>(
            child, minimumRung, scaleFactor, baseTimestep, explicitTimestep));
    }
    StarFormationUpdate result;
    std::exception_ptr failure;
    detail::pending.clear();
    try {
        auto parameters = detail::parameters(scaleFactor);
        parameters.baseTimestep = baseTimestep;
        parameters.explicitTimestep = explicitTimestep;
        std::vector<ParticleInteger> gasIndices;
        std::vector<detail::Input> inputs;
        for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
            if (sphParticlesRung(gas) >= minimumRung) {
                gasIndices.push_back(gas);
                inputs.push_back(detail::input(gas));
            }
        }
        auto const outputs = detail::run(detail::Operation::advance, inputs, parameters);
        detail::pending.reserve(outputs.size());
        for (size_t index = 0; index < outputs.size(); ++index) {
            auto const gas = gasIndices[index];
            auto const& output = outputs[index];
            detail::pending.push_back({sphParticlesParticleIndex(gas), output});
            result.heating += output.value;
            if (output.convert) {
                ++result.converted;
                result.maximumNewStarRung =
                    std::max(result.maximumNewStarRung, int(sphParticlesRung(gas)));
            }
        }
    } catch (...) {
        failure = std::current_exception();
    }
    for (auto& future : remote) {
        try {
            auto const child = future.get();
            result.heating += child.heating;
            result.converted += child.converted;
            result.maximumNewStarRung =
                std::max(result.maximumNewStarRung, child.maximumNewStarRung);
        } catch (...) {
            if (!failure) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        detail::pending.clear();
        std::rethrow_exception(failure);
    }
    return result;
}
void starFormationConvertGas(ParticleInteger gas, StarFormationState const& state) {
    ParticleInteger const particle = sphParticlesParticleIndex(gas);
    particlesStarFormationStates[particle] = state;
    detail::NewStar star{particle, {}};
    for (int dim = 0; dim < kDimensions; ++dim) {
        star.gravity[dim] =
            getOptions().gravity.enabled ? sphParticlesGravityAcceleration(dim, gas) : 0.0f;
    }
    detail::newStars.push_back(star);
    particlesConvertToStar(particle, state.formationTime);
}
std::vector<star_formation_backend::Output> starFormationAdvanceBatch(
    std::vector<star_formation_backend::Input> const& inputs, std::vector<double> const& intervals,
    double scaleFactor) {
    if (inputs.size() != intervals.size()) {
        throw std::invalid_argument("Star-formation batch inputs and intervals differ in size");
    }
    std::vector<star_formation_backend::Output> outputs(inputs.size());
    // The backend takes one explicit interval per batch; group equal intervals so a
    // CUDA dispatch stays batched while early closures keep their exact durations.
    std::map<double, std::vector<size_t>> groups;
    for (size_t index = 0; index < inputs.size(); ++index) {
        if (!(intervals[index] > 0.0) || !std::isfinite(intervals[index])) {
            throw std::invalid_argument("Star formation requires positive finite intervals");
        }
        groups[intervals[index]].push_back(index);
    }
    auto parameters = detail::parameters(scaleFactor);
    for (auto const& [interval, members] : groups) {
        parameters.baseTimestep = interval;
        parameters.explicitTimestep = interval;
        std::vector<detail::Input> groupInputs(members.size());
        for (size_t k = 0; k < members.size(); ++k) {
            groupInputs[k] = inputs[members[k]];
        }
        auto const groupOutputs = detail::run(detail::Operation::advance, groupInputs, parameters);
        for (size_t k = 0; k < members.size(); ++k) {
            outputs[members[k]] = groupOutputs[k];
        }
    }
    return outputs;
}
void starFormationCommit() {
    std::vector<hpx::future<void>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<StarFormationCommitAction>(child));
    }
    starFormationCommitLocal();
    for (auto& future : remote) {
        future.get();
    }
}
StarFormationUpdate starFormationPrepareLocal(std::vector<ParticleInteger> const& gasIndices,
    std::vector<double> const& intervals, double scaleFactor) {
    if (gasIndices.size() != intervals.size()) {
        throw std::invalid_argument("Star-formation slots and intervals differ in count");
    }
    detail::pending.clear();
    std::vector<detail::Input> inputs;
    inputs.reserve(gasIndices.size());
    for (auto const gas : gasIndices) {
        inputs.push_back(detail::input(gas));
    }
    auto const outputs = starFormationAdvanceBatch(inputs, intervals, scaleFactor);
    StarFormationUpdate result;
    std::vector<detail::Pending> staged;
    staged.reserve(outputs.size());
    for (size_t index = 0; index < outputs.size(); ++index) {
        auto const gas = gasIndices[index];
        auto const& output = outputs[index];
        // Commit order follows these main-particle indices, which conversions keep.
        staged.push_back({sphParticlesParticleIndex(gas), output});
        result.heating += output.value;
        if (output.convert) {
            ++result.converted;
            result.maximumNewStarRung =
                std::max(result.maximumNewStarRung, int(sphParticlesRung(gas)));
        }
    }
    detail::pending = std::move(staged);
    return result;
}
void starFormationCommitLocal() {
    for (auto const& update : detail::pending) {
        ParticleInteger const gas = particlesCatIndex(update.particle);
        auto const& output = update.output;
        if (output.convert) {
            starFormationConvertGas(gas, output.state);
        } else {
            particlesStarFormationStates[update.particle] = output.state;
            sphParticlesSetInternalEnergy(gas, output.energy);
            sphParticlesChemistryStates[gas] = output.chemistry;
        }
    }
    detail::pending.clear();
}
StarFormationUpdate starFormationClose(
    int minimumRung, double scaleFactor, double, double baseTimestep, double explicitTimestep) {
    auto const result =
        starFormationPrepare(minimumRung, scaleFactor, baseTimestep, explicitTimestep);
    starFormationCommit();
    if (result.converted != 0) {
        particlesCacheFree();
    }
    sphParticlesClearHydroCaches();
    return result;
}
HPX_PLAIN_ACTION(starFormationOpenNewStars, StarFormationOpenNewStarsAction);
int starFormationOpenNewStars(double baseTimestep, double scaleFactor, double maximumTimestep,
    int minimumRung, bool kickGravity) {
    std::vector<hpx::future<int>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<StarFormationOpenNewStarsAction>(
            child, baseTimestep, scaleFactor, maximumTimestep, minimumRung, kickGravity));
    }
    int maximumRung = 0;
    try {
        if (kickGravity) {
            auto const& options = getOptions();
            float const effectiveRadius = float(options.gravity.supportRadius) /
                selectedGravityKernel().supportRadiusPerEffectiveRadius;
            float const timestepBase = float(baseTimestep);
            float const timestepCap = float(maximumTimestep);
            float const factor = float(options.gravity.timestepEta) * std::sqrt(float(scaleFactor));
            float const sign = options.run.scenario == "glass" ? -1.0f : 1.0f;
            star_formation_backend::Parameters parameters;
            parameters.baseTimestep = baseTimestep;
            parameters.scaleFactor = scaleFactor;
            parameters.timeSeconds = options.units.timeSeconds;
            parameters.densityScale = parameters.energyScale = 1.0;
            parameters.effectiveRadius = effectiveRadius;
            parameters.timestepBase = timestepBase;
            parameters.timestepCap = timestepCap;
            parameters.timestepFactor = factor;
            parameters.kickSign = sign;
            parameters.rungLimit = kMaxRung;
            parameters.minimumRung = minimumRung;
            std::vector<star_formation_backend::Input> inputs(detail::newStars.size());
            for (size_t index = 0; index < detail::newStars.size(); ++index) {
                auto const& star = detail::newStars[index];
                auto& input = inputs[index];
                input.state = particlesStarFormationStates[star.particle];
                input.rung = particlesRung(star.particle);
                for (int dim = 0; dim < kDimensions; ++dim) {
                    input.gravity[dim] = star.gravity[dim];
                    input.velocity[dim] = particlesVel(dim, star.particle);
                }
            }
            auto const outputs =
                detail::run(star_formation_backend::Operation::openNewStar, inputs, parameters);
            // A failed batch leaves all local newborn rungs and velocities unchanged.
            for (size_t index = 0; index < detail::newStars.size(); ++index) {
                auto const particle = detail::newStars[index].particle;
                auto const& output = outputs[index];
                particlesRung(particle) = output.rung;
                maximumRung = std::max(maximumRung, output.rung);
                for (int dim = 0; dim < kDimensions; ++dim) {
                    particlesVel(dim, particle) = output.velocity[dim];
                }
            }
        }
    } catch (...) {
        hpx::wait_all(remote);
        throw;
    }
    detail::newStars.clear();
    for (auto& future : remote) {
        maximumRung = std::max(maximumRung, future.get());
    }
    return maximumRung;
}
HPX_PLAIN_ACTION(starFormationPressureIntegral, StarFormationPressureIntegralAction);
double starFormationPressureIntegral(double scaleFactor) {
    std::vector<hpx::future<double>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<StarFormationPressureIntegralAction>(child, scaleFactor));
    }
    double sum = 0.0;
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        sum += getOptions().initial.gasMass * gadget4::kGammaMinusOne *
            sphParticlesInternalEnergy(gas) / (scaleFactor * scaleFactor);
    }
    for (auto& future : remote) {
        sum += future.get();
    }
    return sum;
}
