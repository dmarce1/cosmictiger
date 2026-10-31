/* CosmicTiger - Springel and Hernquist (2003) SPH coupling; BSL-1.0. */
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cosmictiger/chemistry/primordial_units.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/simulation_units.hpp>
#include <cosmictiger/sph/sph_chemistry.hpp>
#include <cosmictiger/sph/sph_particles.hpp>
#include <cosmictiger/sph/star_formation.hpp>
#ifdef USE_CUDA
#include <cosmictiger/sph/star_formation_cuda.hpp>
#endif

namespace detail {
using Model = springel_and_hernquist::Model;
using Equilibrium = springel_and_hernquist::Equilibrium;
using Cooling = springel_and_hernquist::AtomicCooling;
constexpr double kSecondsPerYear = 31557600.0;
Cooling cooling() {
    return {1.0 - getOptions().cosmology.heliumFraction};
}
Model model() {
    auto const& options = getOptions().starFormation;
    auto const atomic = cooling();
    auto parameters = springel_and_hernquist::defaultParameters(atomic);
    parameters.beta = options.beta;
    parameters.evaporationFactor = options.evaporationFactor;
    parameters.starFormationTime = options.starFormationTimeGyr * 1.e9 * kSecondsPerYear;
    parameters.coldEnergy *= options.coldTemperature / 1000.0;
    parameters.supernovaEnergy *= options.supernovaTemperature / 1.e8;
    Model result;
    if (!springel_and_hernquist::makeModel(parameters, atomic, result)) {
        throw std::runtime_error(
            "Cannot calibrate SH03 threshold with the selected atomic cooling");
    }
    return result;
}
double properDensity(ParticleInteger gas, double scaleFactor) {
    auto const& options = getOptions();
    return units::toProperDensity(units::StoredDensity(sphParticlesDensity(gas)),
        units::scalesFromOptions(options),
        units::CosmologicalContext(scaleFactor, options.cosmology.hubble))
        .quantity()
        .value();
}
double physicalEnergy(ParticleInteger gas, double scaleFactor) {
    auto const& options = getOptions();
    return units::toPhysicalSpecificEnergy(
        units::StoredSpecificEnergy(sphParticlesInternalEnergy(gas)),
        units::scalesFromOptions(options),
        units::CosmologicalContext(scaleFactor, options.cosmology.hubble))
        .quantity()
        .value();
}
float storedEnergy(double energy, double scaleFactor) {
    auto const& options = getOptions();
    double const value = units::toStoredSpecificEnergy(
        units::PhysicalSpecificEnergy(units::SpecificEnergy::from_value(energy)),
        units::scalesFromOptions(options),
        units::CosmologicalContext(scaleFactor, options.cosmology.hubble))
                             .value();
    if (!(value > 0.0) || !std::isfinite(value) || value > std::numeric_limits<float>::max()) {
        throw std::runtime_error("SH03 energy cannot be represented in SPH storage");
    }
    return static_cast<float>(value);
}
Equilibrium equilibrium(Model const& currentModel, Cooling const& atomic, double density) {
    Equilibrium result;
    if (!springel_and_hernquist::evaluateEquilibrium(currentModel, atomic, density, result)) {
        throw std::runtime_error("SH03 hot phase is outside the atomic cooling domain");
    }
    return result;
}
bool starForming(ParticleInteger gas, Model const& currentModel, double density) {
    return density >= currentModel.thresholdDensity &&
        particlesStarFormationStates[sphParticlesParticleIndex(gas)].windTimeRemaining <= 0.0;
}
primordial::State phaseComposition(
    Equilibrium const& phase, Cooling const& atomic, float energy, double scaleFactor) {
    springel_and_hernquist::AtomicEquilibrium hot;
    if (!atomic.evaluateAtEnergy(phase.hotEnergy, hot)) {
        throw std::runtime_error("SH03 hot-phase composition is outside the atomic domain");
    }
    // Section 5.2 closes the unresolved phases in equilibrium. Use the same
    // primordial H/He mixture in each phase: neutral cold clouds plus CIE hot gas.
    // This projection is an interface convention, not a time-dependent cloud
    // chemistry solution. Ordinary gas resumes from this composition on exit.
    primordial::State result;
    double const heliumRatio =
        (1.0 - atomic.hydrogenMassFraction) / (4.0 * atomic.hydrogenMassFraction);
    result.abundances[0] = float(phase.coldFraction + phase.hotFraction * hot.hydrogenNeutral);
    result.abundances[1] = float(phase.hotFraction * hot.hydrogenIonized);
    result.abundances[2] =
        float(heliumRatio * (phase.coldFraction + phase.hotFraction * hot.heliumNeutral));
    result.abundances[3] = float(heliumRatio * phase.hotFraction * hot.heliumIonized);
    result.abundances[4] = float(heliumRatio * phase.hotFraction * hot.heliumDoubleIonized);
    result.abundances[5] = float(phase.hotFraction * hot.electronsPerHydrogen);
    return sphChemistrySynchronizeEnergy(result, energy, scaleFactor);
}
struct Pending {
    ParticleInteger particle;
    StarFormationState state;
    Array<float, kDimensions> velocity;
    float energy;
    primordial::State chemistry;
    bool convert;
};
struct NewStar {
    ParticleInteger particle;
    Array<float, kDimensions> gravity;
};
std::vector<Pending> pending;
std::vector<NewStar> newStars;
#ifdef USE_CUDA
star_formation_backend::Parameters cudaParameters(double scaleFactor) {
    auto const& options = getOptions();
    auto const& settings = options.starFormation;
    auto const scales = units::scalesFromOptions(options);
    units::CosmologicalContext const context(scaleFactor, options.cosmology.hubble);
    star_formation_backend::Parameters result;
    result.model = model();
    result.cooling = cooling();
    result.environment.network = options.chemistry.network == "atomic6" ?
        primordial::Network::Atomic6 :
        primordial::Network::Molecular9;
    result.environment.heliumToHydrogenNumberRatio = primordial::checkedScalar(
        options.cosmology.heliumFraction / (4.0 * result.cooling.hydrogenMassFraction),
        "He/H number ratio");
    result.environment.cmbTemperature = primordial::checkedScalar(
        2.73 * options.cosmology.thetaCmb / scaleFactor, "CMB temperature");
    result.densityScale = scales.densityScale().value();
    result.energyScale = scales.specificEnergyScale().value();
    result.hydrogenMass = units::kHydrogenMass.value();
    result.scaleFactor = context.scaleFactor();
    result.timeSeconds = options.units.timeSeconds;
    result.lengthCm = options.units.lengthCm;
    result.gasMass = options.initial.gasMass;
    result.metalYield = settings.metalYield;
    result.windMassLoading = settings.windMassLoading;
    result.windEnergyFraction = settings.windEnergyFraction;
    result.windDecouplingSeconds = settings.windDecouplingMyr * 1.e6 * kSecondsPerYear;
    result.windRecoupleDensityFraction = settings.windRecoupleDensityFraction;
    result.seed = settings.seed;
    result.winds = settings.winds;
    return result;
}
// Only gather/scatter host-owned storage here. Every per-particle physical
// calculation is performed by the CUDA batch; topology mutations remain on host.
star_formation_backend::Input cudaInput(ParticleInteger gas) {
    auto const particle = sphParticlesParticleIndex(gas);
    star_formation_backend::Input result;
    result.state = particlesStarFormationStates[particle];
    result.chemistry = sphParticlesChemistryStates[gas];
    result.chemistryHalfInterval = sphParticlesChemistrySteps[gas].halfInterval;
    result.chemistryPendingHeating = sphParticlesChemistrySteps[gas].pendingHeating;
    result.energy = sphParticlesInternalEnergy(gas);
    result.density = sphParticlesDensity(gas);
    result.pressureRatio = sphParticlesChemistryEos[gas].first;
    result.rung = sphParticlesRung(gas);
    for (int dim = 0; dim < kDimensions; ++dim) {
        result.velocity[dim] = particlesV[particle][dim];
    }
    return result;
}
std::vector<star_formation_backend::Input> cudaInputs() {
    std::vector<star_formation_backend::Input> result(sphParticlesSize());
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        result[gas] = cudaInput(gas);
    }
    return result;
}
#endif
uint64_t mix(uint64_t value) {
    // SplitMix64 finalizer: counter-based draws do not depend on rank or thread scheduling.
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}
}    // namespace detail

uint64_t starFormationRandomBits(uint64_t identity, uint64_t event, uint64_t stream) {
    return detail::mix(detail::mix(identity ^ getOptions().starFormation.seed) ^
        detail::mix(event + UINT64_C(0x9e3779b97f4a7c15)) ^
        detail::mix(stream + UINT64_C(0x632be59bd9b4e019)));
}
double starFormationUniform(uint64_t identity, uint64_t event, uint64_t stream) {
    return double(starFormationRandomBits(identity, event, stream) >> 11) * 0x1.0p-53;
}
springel_and_hernquist::Model starFormationModel() {
    return detail::model();
}
void starFormationInitialize() {
    auto const model = detail::model();
    particlesInitializeStarFormationIds();
    PRINT("SH03 equilibrium star formation: rho_th=%.12e g/cm^3, nH_th=%.12e cm^-3; N_g=1, "
          "dense atomic CIE, ordinary gas %s chemistry\n",
        model.thresholdDensity,
        model.thresholdDensity * (1.0 - getOptions().cosmology.heliumFraction) /
            springel_and_hernquist::kProtonMass,
        getOptions().chemistry.network.c_str());
}

HPX_PLAIN_ACTION(starFormationRefreshEos, StarFormationRefreshEosAction);
void starFormationRefreshEos(double scaleFactor) {
    std::vector<hpx::future<void>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<StarFormationRefreshEosAction>(child, scaleFactor));
    }
    try {
#ifdef USE_CUDA
        if (getOptions().compute.cuda) {
            cudaSetDevice();
            auto const inputs = detail::cudaInputs();
            std::vector<star_formation_backend::Output> outputs(inputs.size());
            star_formation_backend::runCuda(star_formation_backend::Operation::eos, inputs.data(),
                outputs.data(), inputs.size(), detail::cudaParameters(scaleFactor));
            for (size_t gas = 0; gas < outputs.size(); ++gas) {
                sphParticlesChemistryEos[gas] = {
                    outputs[gas].pressureRatio, outputs[gas].signalIndex};
            }
        } else
#endif
        {
            auto const currentModel = detail::model();
            auto const atomic = detail::cooling();
            for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
                double const density = detail::properDensity(gas, scaleFactor);
                double pressureRatio;
                double signalIndex;
                if (detail::starForming(gas, currentModel, density)) {
                    auto const state = detail::equilibrium(currentModel, atomic, density);
                    pressureRatio = (currentModel.parameters.gamma - 1.0) * state.effectiveEnergy /
                        detail::physicalEnergy(gas, scaleFactor);
                    // SH03 equation (19): the strict equilibrium branch is barotropic.
                    // Its acoustic derivative is dP_eff/drho, not the atomic gamma P/rho.
                    double const lowerDensity =
                        std::max(currentModel.thresholdDensity, density * 0.999);
                    double const upperDensity = density * 1.001;
                    auto const lower = detail::equilibrium(currentModel, atomic, lowerDensity);
                    auto const upper = detail::equilibrium(currentModel, atomic, upperDensity);
                    signalIndex = std::log(upper.pressure / lower.pressure) /
                        std::log(upperDensity / lowerDensity);
                } else {
                    auto const eos = sphChemistryEos(sphParticlesChemistryStates[gas],
                        sphParticlesInternalEnergy(gas), scaleFactor);
                    pressureRatio = eos.pressureEnergyRatio;
                    signalIndex = eos.adiabaticIndex;
                }
                if (!(pressureRatio > 0.0) || !(signalIndex > 0.0) ||
                    !std::isfinite(pressureRatio) || !std::isfinite(signalIndex)) {
                    throw std::runtime_error("SH03 pressure or compressibility is invalid");
                }
                sphParticlesChemistryEos[gas] = {float(pressureRatio), float(signalIndex)};
            }
        }
    } catch (...) {
        hpx::wait_all(remote);
        throw;
    }
    for (auto& future : remote) {
        future.get();
    }
    sphParticlesClearHydroCaches();
}

HPX_PLAIN_ACTION(starFormationMaximumTimestep, StarFormationMaximumTimestepAction);
double starFormationMaximumTimestep(int minimumRung, double scaleFactor) {
    std::vector<hpx::future<double>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(
            hpx::async<StarFormationMaximumTimestepAction>(child, minimumRung, scaleFactor));
    }
    double result = std::numeric_limits<double>::max();
    try {
#ifdef USE_CUDA
        if (getOptions().compute.cuda) {
            cudaSetDevice();
            auto const inputs = detail::cudaInputs();
            std::vector<star_formation_backend::Output> outputs(inputs.size());
            auto parameters = detail::cudaParameters(scaleFactor);
            parameters.minimumRung = minimumRung;
            star_formation_backend::runCuda(star_formation_backend::Operation::timestep,
                inputs.data(), outputs.data(), inputs.size(), parameters);
            for (auto const& output : outputs) {
                result = std::min(result, output.value);
            }
        } else
#endif
        {
            auto const currentModel = detail::model();
            auto const atomic = detail::cooling();
            auto const& options = getOptions();
            for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
                if (sphParticlesRung(gas) < minimumRung) {
                    continue;
                }
                double const density = detail::properDensity(gas, scaleFactor);
                double physicalLimit = std::numeric_limits<double>::max();
                auto const& metadata =
                    particlesStarFormationStates[sphParticlesParticleIndex(gas)];
                if (detail::starForming(gas, currentModel, density)) {
                    auto const state = detail::equilibrium(currentModel, atomic, density);
                    // Bound competing star/wind event splitting error; not a paper parameter.
                    physicalLimit = 0.05 /
                        (state.specificStarFormationRate *
                            (1.0 +
                                (options.starFormation.winds ?
                                        options.starFormation.windMassLoading :
                                        0.0)));
                } else if (metadata.windTimeRemaining > 0.0) {
                    physicalLimit = metadata.windTimeRemaining;
                }
                result = std::min(result, physicalLimit / scaleFactor / options.units.timeSeconds);
            }
        }
    } catch (...) {
        hpx::wait_all(remote);
        throw;
    }
    for (auto& future : remote) {
        result = std::min(result, future.get());
    }
    return result;
}

StarFormationUpdate starFormationPrepare(int minimumRung, double scaleFactor,
    double scaleDerivative, double baseTimestep, bool opening, double explicitTimestep,
    int uniformRung);
void starFormationCommit();
HPX_PLAIN_ACTION(starFormationPrepare, StarFormationPrepareAction);
HPX_PLAIN_ACTION(starFormationCommit, StarFormationCommitAction);
StarFormationUpdate starFormationPrepareImpl(int minimumRung, double scaleFactor,
    double scaleDerivative, double baseTimestep, bool opening, double explicitTimestep,
    int uniformRung, section_timing::ScopedSection* sourceTimer) {
    std::vector<hpx::future<StarFormationUpdate>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<StarFormationPrepareAction>(child, minimumRung, scaleFactor,
            scaleDerivative, baseTimestep, opening, explicitTimestep, uniformRung));
    }
    StarFormationUpdate result;
    std::exception_ptr failure;
    detail::pending.clear();
    try {
        if (!std::isfinite(explicitTimestep)) {
            throw std::runtime_error("SH03 timestep is not finite");
        }
#ifdef USE_CUDA
        if (getOptions().compute.cuda) {
            cudaSetDevice();
            std::vector<ParticleInteger> gasIndices;
            std::vector<star_formation_backend::Input> inputs;
            gasIndices.reserve(sphParticlesSize());
            inputs.reserve(sphParticlesSize());
            for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
                if (sphParticlesRung(gas) >= minimumRung) {
                    gasIndices.push_back(gas);
                    inputs.push_back(detail::cudaInput(gas));
                }
            }
            auto parameters = detail::cudaParameters(scaleFactor);
            parameters.scaleDerivative = scaleDerivative;
            parameters.baseTimestep = baseTimestep;
            parameters.explicitTimestep = explicitTimestep;
            parameters.minimumRung = minimumRung;
            parameters.uniformRung = uniformRung;
            parameters.opening = opening;
            std::vector<star_formation_backend::Output> outputs(inputs.size());
            star_formation_backend::runCuda(star_formation_backend::Operation::prepare,
                inputs.data(), outputs.data(), inputs.size(), parameters);
            std::vector<SphChemistryBatchInput> chemistryInputs;
            std::vector<size_t> chemistryIndices;
            chemistryInputs.reserve(inputs.size());
            chemistryIndices.reserve(inputs.size());
            for (size_t index = 0; index < outputs.size(); ++index) {
                if (outputs[index].chemistryInterval > 0.0) {
                    chemistryInputs.push_back({inputs[index].chemistry, inputs[index].energy,
                        inputs[index].density, outputs[index].chemistryInterval});
                    chemistryIndices.push_back(index);
                }
            }
            auto const chemistryResults =
                sphChemistryAdvanceBatch(chemistryInputs, scaleFactor, sourceTimer);
            for (size_t index = 0; index < chemistryResults.size(); ++index) {
                auto& output = outputs[chemistryIndices[index]];
                if (!output.project) {
                    output.chemistry = chemistryResults[index].state;
                    output.energy = chemistryResults[index].storedEnergy;
                }
            }
            star_formation_backend::runCuda(star_formation_backend::Operation::finish,
                inputs.data(), outputs.data(), inputs.size(), parameters);
            detail::pending.reserve(outputs.size());
            for (size_t index = 0; index < outputs.size(); ++index) {
                auto const& output = outputs[index];
                auto const gas = gasIndices[index];
                auto const particle = sphParticlesParticleIndex(gas);
                detail::Pending update{
                    particle, output.state, {}, output.energy, output.chemistry, output.convert};
                for (int dim = 0; dim < kDimensions; ++dim) {
                    update.velocity[dim] = output.velocity[dim];
                }
                detail::pending.push_back(update);
                result.heating += output.value;
                if (output.convert) {
                    ++result.converted;
                    result.maximumNewStarRung =
                        std::max(result.maximumNewStarRung, int(sphParticlesRung(gas)));
                }
            }
        } else
#endif
        {
            auto const currentModel = detail::model();
            auto const atomic = detail::cooling();
            auto const& options = getOptions();
            auto const& settings = options.starFormation;
            std::vector<SphChemistryBatchInput> chemistryInputs;
            std::vector<size_t> chemistryIndices;
            std::vector<bool> projectAfterChemistry;
            for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
                int const currentRung = sphParticlesRung(gas);
                if (currentRung < minimumRung) {
                    continue;
                }
                ParticleInteger const particle = sphParticlesParticleIndex(gas);
                auto const& previous = particlesStarFormationStates[particle];
                detail::Pending update{particle, previous, particlesV[particle],
                    sphParticlesInternalEnergy(gas), sphParticlesChemistryStates[gas], false};
                auto& state = update.state;
                int const rung = opening && uniformRung >= 0 ? uniformRung : currentRung;
                double const halfInterval = explicitTimestep >= 0.0 ?
                    0.5 * explicitTimestep :
                    std::ldexp(0.5 * baseTimestep, -rung);
                if (state.id == 0 || !(halfInterval > 0.0) || !std::isfinite(halfInterval) ||
                    (opening && (state.sourceHalfInterval != 0.0 || state.pendingHeating != 0.0)) ||
                    (!opening && state.sourceHalfInterval != halfInterval) ||
                    sphParticlesChemistrySteps[gas].halfInterval != 0.0 ||
                    sphParticlesChemistrySteps[gas].pendingHeating != 0.0) {
                    throw std::runtime_error(
                        "SH03 identity or pending SPH source interval is inconsistent");
                }
                double const midpointScale =
                    scaleFactor + (opening ? 0.5 : -0.5) * scaleDerivative * halfInterval;
                double const physicalInterval =
                    midpointScale * halfInterval * options.units.timeSeconds;
                if (!(physicalInterval > 0.0) || !std::isfinite(physicalInterval)) {
                    throw std::runtime_error("SH03 physical source interval is invalid");
                }
                double const density = detail::properDensity(gas, scaleFactor);
                if (density <
                    settings.windRecoupleDensityFraction * currentModel.thresholdDensity) {
                    state.windTimeRemaining = 0.0;
                }
                double const windInterval = std::min(state.windTimeRemaining, physicalInterval);
                state.windTimeRemaining -= windInterval;
                double const eligibleInterval = physicalInterval - windInterval;
                // A wind remains outside the multiphase source until its decoupling
                // clock expires. Split this half-step at that instant (SH03 section 4).
                bool const project =
                    eligibleInterval > 0.0 && density >= currentModel.thresholdDensity;
                double const chemistryInterval = project ? windInterval : physicalInterval;
                if (chemistryInterval > 0.0) {
                    chemistryInputs.push_back({update.chemistry, update.energy,
                        sphParticlesDensity(gas), chemistryInterval});
                    chemistryIndices.push_back(detail::pending.size());
                    projectAfterChemistry.push_back(project);
                }
                double opticalDepth = 0.0;
                if (project) {
                    auto const phase = detail::equilibrium(currentModel, atomic, density);
                    update.energy = detail::storedEnergy(phase.effectiveEnergy, scaleFactor);
                    update.chemistry =
                        detail::phaseComposition(phase, atomic, update.energy, scaleFactor);
                    opticalDepth = phase.specificStarFormationRate * eligibleInterval;
                    state.metallicity += float(springel_and_hernquist::enrichmentIncrement(
                        phase, eligibleInterval, settings.metalYield));
                }
                if (!(state.metallicity >= 0.0f && state.metallicity <= 1.0f)) {
                    throw std::runtime_error(
                        "SH03 tracer metallicity exceeded its mass-fraction domain");
                }
                state.formationOpticalDepth += opticalDepth;
                state.elapsedPhysicalTime += physicalInterval;
                state.sourceHalfInterval = halfInterval;
                detail::pending.push_back(update);
            }
            // One nonmutating batch per rank: ordinary gas and decoupled winds use
            // the existing non-equilibrium solver. Never cool u_eff separately from
            // the SH03 cloud-growth/cooling balance (equations 16-19).
            auto const chemistryResults =
                sphChemistryAdvanceBatch(chemistryInputs, scaleFactor, sourceTimer);
            for (size_t i = 0; i < chemistryResults.size(); ++i) {
                if (!projectAfterChemistry[i]) {
                    auto& update = detail::pending[chemistryIndices[i]];
                    update.chemistry = chemistryResults[i].state;
                    update.energy = chemistryResults[i].storedEnergy;
                }
            }
            for (auto& update : detail::pending) {
                ParticleInteger const gas = particlesCatIndex(update.particle);
                auto const& previous = particlesStarFormationStates[update.particle];
                auto& state = update.state;
                double kineticChange = 0.0;
                if (!opening) {
                    // SH03 equation (39), N_g=1; integrate the rate over both source halves.
                    double const probability = -std::expm1(-state.formationOpticalDepth);
                    update.convert =
                        starFormationUniform(state.id, state.eventCounter, 0) < probability;
                    if (update.convert) {
                        update.energy = 0.0f;
                        ++result.converted;
                        result.maximumNewStarRung =
                            std::max(result.maximumNewStarRung, int(sphParticlesRung(gas)));
                    } else if (settings.winds && state.windTimeRemaining == 0.0 &&
                        starFormationUniform(state.id, state.eventCounter, 1) <
                            -std::expm1(-settings.windMassLoading * state.formationOpticalDepth)) {
                        // SH03 equations (28), (41), (42): isotropic wind; zero mean momentum.
                        double const speed =
                            springel_and_hernquist::windSpeed(currentModel.parameters,
                                settings.windMassLoading, settings.windEnergyFraction);
                        double const cosine =
                            2.0 * starFormationUniform(state.id, state.eventCounter, 2) - 1.0;
                        double const azimuth = 6.2831853071795864769 *
                            starFormationUniform(state.id, state.eventCounter, 3);
                        double const sine = std::sqrt(std::max(0.0, 1.0 - cosine * cosine));
                        double const direction[kDimensions] = {
                            sine * std::cos(azimuth), sine * std::sin(azimuth), cosine};
                        double const storedSpeed = speed * scaleFactor * options.units.timeSeconds /
                            options.units.lengthCm;
                        for (int dim = 0; dim < kDimensions; ++dim) {
                            double const before = update.velocity[dim];
                            update.velocity[dim] = float(before + storedSpeed * direction[dim]);
                            kineticChange += 0.5 *
                                (double(update.velocity[dim]) * update.velocity[dim] -
                                    before * before);
                        }
                        state.windTimeRemaining =
                            settings.windDecouplingMyr * 1.e6 * detail::kSecondsPerYear;
                    }
                    ++state.eventCounter;
                    state.formationOpticalDepth = 0.0;
                }
                double const change = -options.initial.gasMass *
                    (double(update.energy) - sphParticlesInternalEnergy(gas) + kineticChange) /
                    scaleFactor;
                if (opening) {
                    state.pendingHeating = change;
                } else {
                    result.heating += previous.pendingHeating + change;
                    state.sourceHalfInterval = 0.0;
                    state.pendingHeating = 0.0;
                }
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
        std::rethrow_exception(failure);
    }
    return result;
}
StarFormationUpdate starFormationPrepare(int minimumRung, double scaleFactor,
    double scaleDerivative, double baseTimestep, bool opening, double explicitTimestep,
    int uniformRung) {
    return starFormationPrepareImpl(minimumRung, scaleFactor, scaleDerivative, baseTimestep,
        opening, explicitTimestep, uniformRung, nullptr);
}
void starFormationCommit() {
    std::vector<hpx::future<void>> remote;
    for (auto const& child : hpxChildren()) {
        remote.push_back(hpx::async<StarFormationCommitAction>(child));
    }
    for (auto const& update : detail::pending) {
        ParticleInteger const gas = particlesCatIndex(update.particle);
        particlesStarFormationStates[update.particle] = update.state;
        particlesV[update.particle] = update.velocity;
        if (update.convert) {
            detail::NewStar star{update.particle, {}};
            for (int dim = 0; dim < kDimensions; ++dim) {
                star.gravity[dim] =
                    getOptions().gravity.enabled ? sphParticlesGravityAcceleration(dim, gas) : 0.0f;
            }
            detail::newStars.push_back(star);
            particlesConvertToStar(update.particle, update.state.elapsedPhysicalTime);
        } else {
            sphParticlesInternalEnergy(gas) = update.energy;
            sphParticlesChemistryStates[gas] = update.chemistry;
        }
    }
    detail::pending.clear();
    for (auto& future : remote) {
        future.get();
    }
}
StarFormationUpdate starFormationClose(int minimumRung, double scaleFactor, double scaleDerivative,
    double baseTimestep, double explicitTimestep, section_timing::ScopedSection* sourceTimer) {
    auto const update = starFormationPrepareImpl(minimumRung, scaleFactor, scaleDerivative,
        baseTimestep, false, explicitTimestep, -1, sourceTimer);
    starFormationCommit();
    if (update.converted != 0) {
        particlesCacheFree();
    }
    sphParticlesClearHydroCaches();
    return update;
}
void starFormationOpen(int minimumRung, double scaleFactor, double scaleDerivative,
    double baseTimestep, double explicitTimestep, int uniformRung,
    section_timing::ScopedSection* sourceTimer) {
    starFormationPrepareImpl(minimumRung, scaleFactor, scaleDerivative, baseTimestep, true,
        explicitTimestep, uniformRung, sourceTimer);
    starFormationCommit();
    sphParticlesClearHydroCaches();
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
#ifdef USE_CUDA
            if (options.compute.cuda) {
                star_formation_backend::Parameters parameters;
                parameters.baseTimestep = baseTimestep;
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
                    input.rung = particlesRung(star.particle);
                    for (int dim = 0; dim < kDimensions; ++dim) {
                        input.gravity[dim] = star.gravity[dim];
                        input.velocity[dim] = particlesVel(dim, star.particle);
                    }
                }
                std::vector<star_formation_backend::Output> outputs(inputs.size());
                star_formation_backend::runCuda(star_formation_backend::Operation::openNewStar,
                    inputs.data(), outputs.data(), inputs.size(), parameters);
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
            } else
#endif
            {
                for (auto const& star : detail::newStars) {
                    // The gravity traversal still saw gas and deferred its rung proposal.
                    // Apply the ordinary FMM collisionless criterion before its first drift.
                    float accelerationSquared = 1.0e-35f;
                    for (int dim = 0; dim < kDimensions; ++dim) {
                        accelerationSquared += star.gravity[dim] * star.gravity[dim];
                    }
                    float const timestep = std::min(timestepCap,
                        std::min(
                            factor * std::sqrt(effectiveRadius / std::sqrt(accelerationSquared)),
                            timestepBase));
                    float const requestedRung = std::ceil(std::log2(timestepBase / timestep));
                    if (!(timestep > 0.0f) || !std::isfinite(timestep) ||
                        !std::isfinite(requestedRung) || requestedRung >= kMaxRung) {
                        throw std::runtime_error(
                            "New stellar gravity timestep exceeds the rung range");
                    }
                    auto& rung = particlesRung(star.particle);
                    rung = std::max({int(requestedRung), int(rung) - 1, minimumRung, 1});
                    maximumRung = std::max(maximumRung, int(rung));
                    double const half = sign * std::ldexp(0.5 * baseTimestep, -int(rung));
                    for (int dim = 0; dim < kDimensions; ++dim) {
                        particlesVel(dim, star.particle) = std::fma(double(star.gravity[dim]), half,
                            double(particlesVel(dim, star.particle)));
                    }
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
#ifdef USE_CUDA
    if (getOptions().compute.cuda) {
        auto const inputs = detail::cudaInputs();
        std::vector<star_formation_backend::Output> outputs(inputs.size());
        star_formation_backend::runCuda(star_formation_backend::Operation::pressure, inputs.data(),
            outputs.data(), inputs.size(), detail::cudaParameters(scaleFactor));
        // Preserve the existing deterministic rank-local summation order.
        for (auto const& output : outputs) {
            sum += output.value;
        }
    } else
#endif
    {
        for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
            sum += getOptions().initial.gasMass * double(sphParticlesChemistryEos[gas].first) *
                sphParticlesInternalEnergy(gas) / (scaleFactor * scaleFactor);
        }
    }
    for (auto& future : remote) {
        sum += future.get();
    }
    return sum;
}
