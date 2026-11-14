/* CosmicTiger - GADGET-4 cooling and star-formation SPH coupling; BSL-1.0. */
#include <algorithm>
#include <cmath>
#include <exception>
#include <format>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <cosmictiger/chemistry/sph.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/math/simulation_units.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/thermodynamics.hpp>
#include <cosmictiger/sph/tree.hpp>
#include <cosmictiger/stars/star_formation.hpp>
#ifdef USE_CUDA
#include <cosmictiger/stars/star_formation_cuda.hpp>
#endif

namespace detail {
using Model = springel_and_hernquist::Model;
using Input = star_formation_backend::Input;
using Output = star_formation_backend::Output;
using Parameters = star_formation_backend::Parameters;
using Operation = star_formation_backend::Operation;

/// Construct the effective star-formation model and physical/overdensity thresholds from
/// resolved options.
Model model() {
    auto const& options = getOptions();
    auto const& settings = options.starFormation;
    auto parameters = springel_and_hernquist::defaultParameters();
    parameters.supernovaMassFraction = settings.beta;
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
        result.parameters = parameters;
        result.thresholdDensity = settings.criticalPhysicalDensity * gadget4::kProtonMass /
            gadget4::kHydrogenMassFraction;
        result.overdensityThreshold = 0.0;
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
/// Combine the star-formation model, cooling tables, unit scales, and random seed for a
/// backend call.
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
/// Pack one gas particle's identity, thermodynamics, rung, and velocity into a backend
/// input record.
Input input(ParticleInteger gas) {
    auto const particle = sphParticlesParticleIndex(gas);
    Input result;
    result.state = particlesStarFormationStates[particle];
    result.electronFraction = sphParticlesElectronFractions[gas];
    result.energy = sphParticlesInternalEnergy(gas);
    result.density = sphParticlesDensity(gas);
    result.pressureRatio = float(gadget4::kGammaMinusOne);
    result.rung = sphParticlesRung(gas);
    for (int dim = 0; dim < kDimensions; ++dim) {
        result.velocity[dim] = particlesV[particle][dim];
    }
    return result;
}
/// Pack backend input records for every local gas particle.
std::vector<Input> inputs() {
    std::vector<Input> result(sphParticlesSize());
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        result[gas] = input(gas);
    }
    return result;
}
/// Execute the selected backend operation and require all returned particle statuses to
/// succeed.
std::vector<Output> run(
    Operation operation, std::vector<Input> const& inputs, Parameters const& parameters) {
    std::vector<Output> outputs(inputs.size());
#ifdef USE_CUDA
    auto const& opts = getOptions();
    if (opts.starFormation.cuda) {
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
    float entropy;
};
std::vector<Pending> pending;

/// Validate backend outputs and prepare entropy, identity, and conversion updates without
/// committing live state.
StarFormationUpdate stage(std::vector<ParticleInteger> const& gasIndices,
    std::vector<Input> const& inputs, std::vector<Output> const& outputs, double scaleFactor) {
    auto const& opts = getOptions();
    if (!sphParticlesEntropyInitialized || sphParticlesThermalScaleFactor != scaleFactor) {
        throw std::logic_error("Star formation requires endpoint density and canonical entropy");
    }
    StarFormationUpdate result;
    std::vector<Pending> staged;
    staged.reserve(outputs.size());
    for (size_t index = 0; index < outputs.size(); ++index) {
        auto output = outputs[index];
        auto const& input = inputs[index];
        float entropy = 0.0f;
        if (output.state.id != input.state.id ||
            (output.convert &&
                (!(output.state.formationTime >= 0.0) ||
                    !star_formation_backend::detail::finite(output.state.formationTime) ||
                    output.state.specificStarFormationRate != 0.0))) {
            throw std::invalid_argument("Invalid prepared star-formation identity or birth state");
        }
        // Validate even a converting particle before any rank may commit.
        auto const original = sph_entropy::prepareStoredEnergy(
            input.energy, input.density, opts.hydro.heatCapacityRatio, scaleFactor);
        (void) original;
        if (!output.convert) {
            auto const prepared = sph_entropy::prepareStoredEnergy(
                output.energy, input.density, opts.hydro.heatCapacityRatio, scaleFactor);
            entropy = prepared.entropy;
            output.energy = prepared.storedEnergy;
        }
        // Derived source ledger in the existing a(K+W+U) normalization:
        // ΔQ = −m (Eₒᵤₜ − Eᵢₙ)/a, E = a²u.
        // m -> gas mass; Eₒᵤₜ -> output energy; Eᵢₙ -> input energy;
        // a -> scale factor. Use the actual representable committed entropy.
        output.value =
            -opts.initial.gasMass * (double(output.energy) - double(input.energy)) / scaleFactor;
        result.heating += output.value;
        result.converted += output.convert;
        staged.push_back({sphParticlesParticleIndex(gasIndices[index]), output, entropy});
    }
    pending = std::move(staged);
    return result;
}
}    // namespace detail

/// Derive deterministic random bits from particle identity, event, stream, and the
/// configured seed.
uint64_t starFormationRandomBits(uint64_t identity, uint64_t event, uint64_t stream) {
    auto const& opts = getOptions();
    return star_formation_backend::detail::mix(
        star_formation_backend::detail::mix(identity ^ opts.starFormation.seed) ^
        star_formation_backend::detail::mix(event + UINT64_C(0x9e3779b97f4a7c15)) ^
        star_formation_backend::detail::mix(stream + UINT64_C(0x632be59bd9b4e019)));
}
/// Convert deterministic random bits to a double uniformly spaced on [0,1).
double starFormationUniform(uint64_t identity, uint64_t event, uint64_t stream) {
    return double(starFormationRandomBits(identity, event, stream) >> 11) * 0x1.0p-53;
}
/// Return backend parameters at the requested scale factor.
star_formation_backend::Parameters starFormationParameters(double scaleFactor) {
    return detail::parameters(scaleFactor);
}
/// Return the effective star-formation model resolved from current options.
springel_and_hernquist::Model starFormationModel() {
    return detail::model();
}
/// Evaluate gas output diagnostics while retaining the saved post-cooling star-formation
/// rate.
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
/// Initialize stable particle identities and report the resolved physical formation
/// threshold.
void starFormationInitialize() {
    auto const model = detail::model();
    particlesInitializeStarFormationIds();
    std::cout << std::format(
                     "GADGET-4 effective star formation: rho_th={:.12e} g/cm^3, nH_th={:.12e} "
                     "cm^-3; whole-particle conversion, equilibrium H/He cooling\n",
                     model.thresholdDensity,
                     model.thresholdDensity * gadget4::kHydrogenMassFraction / gadget4::kProtonMass)
              << std::flush;
}

StarFormationUpdate starFormationPrepare(
    int minimumRung, double scaleFactor, double baseTimestep, double explicitTimestep);
void starFormationCommit();
void starFormationDiscardPrepared();
HPX_PLAIN_ACTION(starFormationDiscardPrepared, StarFormationDiscardPreparedAction);
/// Discard staged source and conversion updates throughout the locality tree.
void starFormationDiscardPrepared() {
    std::vector<hpx::future<void>> children;
    for (auto const& child : hpxChildren()) {
        children.push_back(hpx::async<StarFormationDiscardPreparedAction>(child));
    }
    detail::pending.clear();
    for (auto& child : children) {
        child.get();
    }
}
HPX_PLAIN_ACTION(starFormationPrepare, StarFormationPrepareAction);
HPX_PLAIN_ACTION(starFormationCommit, StarFormationCommitAction);
/// Prepare selected local and remote star-formation updates and collect their diagnostics.
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
        result = detail::stage(gasIndices, inputs, outputs, scaleFactor);
    } catch (...) {
        failure = std::current_exception();
    }
    for (auto& future : remote) {
        try {
            auto const child = future.get();
            result.heating += child.heating;
            result.converted += child.converted;
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
/// Validate a conversion's identity and replace the gas record with persistent star
/// metadata.
void starFormationConvertGas(ParticleInteger gas, StarFormationState const& state) {
    ParticleInteger const particle = sphParticlesParticleIndex(gas);
    if (state.id != particlesStarFormationStates[particle].id ||
        state.specificStarFormationRate != 0.0) {
        throw std::invalid_argument(
            "Conversion changed the particle identity or retained a gas source");
    }
    // Gas compaction preserves the main particle slot, velocity, mass and old rung.
    // Its next ordinary gravity boundary closes and opens the collisionless step.
    particlesConvertToStar(particle, state.formationTime);
    particlesStarFormationStates[particle] = state;
}
/// Group inputs by equal completed intervals and execute batched source updates for each
/// group.
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
/// Commit prepared source and conversion updates throughout the locality tree.
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
/// Validate distinct gas slots, advance their explicit intervals, and stage the resulting
/// updates.
StarFormationUpdate starFormationPrepareLocal(std::vector<ParticleInteger> const& gasIndices,
    std::vector<double> const& intervals, double scaleFactor) {
    if (gasIndices.size() != intervals.size()) {
        throw std::invalid_argument("Star-formation slots and intervals differ in count");
    }
    detail::pending.clear();
    std::vector<detail::Input> inputs;
    inputs.reserve(gasIndices.size());
    std::unordered_set<ParticleInteger> selected;
    for (auto const gas : gasIndices) {
        if (!selected.insert(gas).second) {
            throw std::invalid_argument("Star-formation batch repeats a gas particle");
        }
        inputs.push_back(detail::input(gas));
    }
    auto const outputs = starFormationAdvanceBatch(inputs, intervals, scaleFactor);
    return detail::stage(gasIndices, inputs, outputs, scaleFactor);
}
/// Apply prepared thermodynamic updates or gas-to-star conversions, then clear pending
/// state.
void starFormationCommitLocal() {
    for (auto const& update : detail::pending) {
        ParticleInteger const gas = particlesCatIndex(update.particle);
        auto const& output = update.output;
        if (output.convert) {
            starFormationConvertGas(gas, output.state);
        } else {
            particlesStarFormationStates[update.particle] = output.state;
            sphParticlesEntropies[gas] = update.entropy;
            sphParticlesElectronFractions[gas] = output.electronFraction;
        }
    }
    detail::pending.clear();
}
/// Prepare and commit a completed source interval, discarding failures and invalidating
/// caches after conversions.
StarFormationUpdate starFormationClose(
    int minimumRung, double scaleFactor, double, double baseTimestep, double explicitTimestep) {
    StarFormationUpdate result;
    try {
        result = starFormationPrepare(minimumRung, scaleFactor, baseTimestep, explicitTimestep);
    } catch (...) {
        // A successful sibling must not retain a commit-able partial transaction.
        starFormationDiscardPrepared();
        throw;
    }
    starFormationCommit();
    if (result.converted != 0) {
        // Gas slots have compacted: every retained leaf range is stale. The next neighbor
        // walk rebuilds the tree and resyncs the ghost leaves; the gravity cache is freed.
        sphTreeDestroy(false);
        particlesCacheFree();
    }
    return result;
}
HPX_PLAIN_ACTION(starFormationPressureIntegral, StarFormationPressureIntegralAction);
/// Reduce the gas thermal pressure-energy contribution across localities at the supplied
/// scale.
double starFormationPressureIntegral(double scaleFactor) {
    auto const& opts = getOptions();
    std::vector<hpx::future<double>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<StarFormationPressureIntegralAction>(child, scaleFactor));
    }
    double sum = 0.0;
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        sum += opts.initial.gasMass * gadget4::kGammaMinusOne * sphParticlesInternalEnergy(gas) /
            (scaleFactor * scaleFactor);
    }
    for (auto& future : remote) {
        sum += future.get();
    }
    return sum;
}
