/* CosmicTiger - SH03 source integration regression; BSL-1.0. */

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <silo.h>

#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/io/view.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/simulation_units.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/sph/sph_chemistry.hpp>
#include <cosmictiger/sph/sph_particles.hpp>
#include <cosmictiger/sph/star_formation.hpp>

namespace detail {

constexpr double kSfScale = 0.25;
constexpr size_t kSfRecordWords = 25;
using SfSnapshot = std::vector<std::vector<uint64_t>>;
FILE* sfCheckpoint = nullptr;

void requireSf(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(std::string("Star-formation source regression: ") + message);
    }
}

void requireSfClose(double actual, double expected, double tolerance, char const* message) {
    requireSf(std::isfinite(actual) && std::isfinite(expected) &&
            std::abs(actual - expected) <=
                tolerance * std::max({std::abs(actual), std::abs(expected), 1.e-100}),
        message);
}

double sfPhysicalHalf() {
    return starFormationModel().parameters.starFormationTime * 1.e-6;
}

double sfTimestep() {
    return 16.0 * sfPhysicalHalf() / (kSfScale * getOptions().units.timeSeconds);
}

float sfStoredDensity(double density) {
    auto const& options = getOptions();
    return float(units::toStoredDensity(units::ProperDensity(units::Density::from_value(density)),
        units::scalesFromOptions(options),
        units::CosmologicalContext(kSfScale, options.cosmology.hubble))
            .value());
}

float sfStoredEnergy(double energy) {
    auto const& options = getOptions();
    return float(units::toStoredSpecificEnergy(
        units::PhysicalSpecificEnergy(units::SpecificEnergy::from_value(energy)),
        units::scalesFromOptions(options),
        units::CosmologicalContext(kSfScale, options.cosmology.hubble))
            .value());
}

double sfSpecificRate(float storedDensity) {
    auto const& options = getOptions();
    double const density = units::toProperDensity(units::StoredDensity(storedDensity),
        units::scalesFromOptions(options),
        units::CosmologicalContext(kSfScale, options.cosmology.hubble))
                               .quantity()
                               .value();
    springel_and_hernquist::Equilibrium state;
    requireSf(springel_and_hernquist::evaluateEquilibrium(
                  starFormationModel(), {1.0 - options.cosmology.heliumFraction}, density, state),
        "cannot evaluate fixture equilibrium");
    return state.specificStarFormationRate;
}

void initializeSfFixture() {
    particlesResize(3);
    sphParticlesResize(3, false);
    auto const model = starFormationModel();
    springel_and_hernquist::Equilibrium phase;
    requireSf(
        springel_and_hernquist::evaluateEquilibrium(model,
            {1.0 - getOptions().cosmology.heliumFraction}, 2.0 * model.thresholdDensity, phase),
        "cannot initialize fixture equilibrium");
    double velocityScale = 1.0;
    if (getOptions().starFormation.winds) {
        auto const& options = getOptions();
        velocityScale =
            springel_and_hernquist::windSpeed(model.parameters,
                options.starFormation.windMassLoading, options.starFormation.windEnergyFraction) *
            kSfScale * options.units.timeSeconds / options.units.lengthCm;
    }
    for (ParticleInteger index = 0; index < 3; ++index) {
        particlesType(index) = kSphType;
        particlesCatIndex(index) = index;
        sphParticlesParticleIndex(index) = index;
        particlesRung(index) = index == 0 ? 1 : (index == 1 ? 3 : 4);
        particlesStarFormationStates[index] = {};
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, index) = 0.25 + index * 0.125;
            particlesVel(dim, index) = float(velocityScale * (dim - 1) / 4.0);
            sphParticlesAcceleration(dim, index) = 0.0f;
            sphParticlesGravityAcceleration(dim, index) = 0.0f;
            sphParticlesPredictedAcceleration(dim, index) = 0.0f;
        }
        sphParticlesInternalEnergy(index) = sfStoredEnergy(
            index == 2 ? 2.0 * model.parameters.warmEnergy : 1.25 * phase.effectiveEnergy);
        sphParticlesDensity(index) =
            sfStoredDensity((index == 0 ? 8.0 : (index == 2 ? 0.5 : 2.0)) * model.thresholdDensity);
        sphParticlesSupportRadius(index) = 0.125f;
        sphParticlesInternalEnergyRate(index) = 0.0f;
        sphParticlesPredictedInternalEnergyRate(index) = 0.0f;
        sphParticlesChemistryEos[index] = {2.0f / 3.0f, 5.0f / 3.0f};
        float const energy = sphParticlesInternalEnergy(index);
        sphChemistryInitializeParticle(index, kSfScale);
        sphParticlesInternalEnergy(index) = energy;
        sphParticlesChemistryStates[index] =
            sphChemistrySynchronizeEnergy(sphParticlesChemistryStates[index], energy, kSfScale);
    }
}

void configureSfDraws(bool winds) {
    for (ParticleInteger index = 0; index < 3; ++index) {
        auto& state = particlesStarFormationStates[index];
        uint64_t event = 0;
        while (!(starFormationUniform(state.id, event, 0) > (winds ? 0.1 : 0.5) &&
            (!winds || starFormationUniform(state.id, event, 1) < 0.9))) {
            requireSf(++event < 10000, "cannot select deterministic event draw");
        }
        state.eventCounter = event;
    }
    // The coarse particle owns a different interval and must remain untouched.
    auto& inactive = particlesStarFormationStates[0];
    inactive.sourceHalfInterval = 0.75;
    inactive.pendingHeating = -0.125;
    inactive.formationOpticalDepth = 0.25;
    inactive.elapsedPhysicalTime = 128.0;
}

std::vector<uint64_t> sfLocalSnapshot() {
    std::vector<uint64_t> result;
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        auto const& state = particlesStarFormationStates[index];
        result.push_back(state.id);
        result.push_back(state.eventCounter);
        result.push_back(std::bit_cast<uint64_t>(state.formationTime));
        result.push_back(std::bit_cast<uint64_t>(state.windTimeRemaining));
        result.push_back(std::bit_cast<uint32_t>(state.metallicity));
        result.push_back(std::bit_cast<uint64_t>(state.sourceHalfInterval));
        result.push_back(std::bit_cast<uint64_t>(state.pendingHeating));
        result.push_back(std::bit_cast<uint64_t>(state.formationOpticalDepth));
        result.push_back(std::bit_cast<uint64_t>(state.elapsedPhysicalTime));
        bool const gas = particlesType(index) == kSphType;
        result.push_back(std::bit_cast<uint32_t>(
            gas ? sphParticlesInternalEnergy(particlesCatIndex(index)) : 0.0f));
        for (int dim = 0; dim < kDimensions; ++dim) {
            result.push_back(std::bit_cast<uint32_t>(particlesVel(dim, index)));
        }
        result.push_back(uint64_t(particlesRung(index)));
        result.push_back(uint64_t(particlesType(index)));
        result.push_back(
            std::bit_cast<uint32_t>(gas ? sphParticlesDensity(particlesCatIndex(index)) : 0.0f));
        primordial::State const chemistry =
            gas ? sphParticlesChemistryStates[particlesCatIndex(index)] : primordial::State{};
        for (auto const& abundance : chemistry.abundances) {
            result.push_back(std::bit_cast<uint32_t>(abundance.raw()));
        }
        result.push_back(std::bit_cast<uint32_t>(chemistry.thermalEnergy));
        SphChemistryStep const step =
            gas ? sphParticlesChemistrySteps[particlesCatIndex(index)] : SphChemistryStep{};
        result.push_back(std::bit_cast<uint64_t>(step.halfInterval));
        result.push_back(std::bit_cast<uint64_t>(step.pendingHeating));
    }
    return result;
}

void sfCheckpointOperation(int operation) {
    if (operation == 0) {
        requireSf(sfCheckpoint == nullptr, "checkpoint fixture already open");
        sfCheckpoint = std::tmpfile();
        requireSf(sfCheckpoint != nullptr, "cannot create source checkpoint fixture");
        particlesSave(sfCheckpoint);
    } else if (operation == 1) {
        requireSf(sfCheckpoint != nullptr, "no source checkpoint to reload");
        std::rewind(sfCheckpoint);
        particlesLoad(sfCheckpoint);
    } else {
        requireSf(sfCheckpoint != nullptr, "no source checkpoint to close");
        std::fclose(sfCheckpoint);
        sfCheckpoint = nullptr;
    }
}

void alterSfFixture(int operation) {
    if (operation == 0 && hpxRank() == hpxSize() - 1) {
        particlesStarFormationStates[2].id = 0;
    } else if (operation == 1) {
        particlesStarFormationStates[2].id = uint64_t(hpxRank()) * 3 + 3;
    } else if (operation == 2) {
        particlesStarFormationStates[1].formationOpticalDepth = 1000.0;
    } else if (operation == 3) {
        particlesStarFormationStates[1].formationOpticalDepth = 0.05;
    } else if (operation == 4) {
        auto& state = particlesStarFormationStates[1];
        state.windTimeRemaining = 0.5 * sfPhysicalHalf();
        uint64_t event = state.eventCounter;
        while (!(starFormationUniform(state.id, event, 0) > 0.9 &&
            starFormationUniform(state.id, event, 1) > 0.9)) {
            requireSf(++event < 100000, "cannot select no-event recoupling draw");
        }
        state.eventCounter = event;
    } else if (operation == 5) {
        particlesStarFormationStates[1].windTimeRemaining = 10.0 * sfPhysicalHalf();
        sphParticlesDensity(particlesCatIndex(1)) =
            sfStoredDensity(0.05 * starFormationModel().thresholdDensity);
    } else if (operation == 6 && hpxRank() == hpxSize() - 1) {
        sphParticlesChemistryStates[particlesCatIndex(2)].abundances[0] =
            primordial::SignedAbundance::fromRaw(2.0f);
    } else if (operation == 7 || operation == 8) {
        sphParticlesDensity(particlesCatIndex(1)) =
            sfStoredDensity((operation == 7 ? 0.5 : 2.0) * starFormationModel().thresholdDensity);
        auto& state = particlesStarFormationStates[1];
        while (starFormationUniform(state.id, state.eventCounter, 0) <= 0.9) {
            ++state.eventCounter;
        }
    }
}

std::vector<uint64_t> sfOrdinaryReference(int particle, double physicalInterval, int intervals) {
    auto const gas = particlesCatIndex(particle);
    SphChemistryBatchInput input{sphParticlesChemistryStates[gas], sphParticlesInternalEnergy(gas),
        sphParticlesDensity(gas), physicalInterval};
    for (int interval = 0; interval != intervals; ++interval) {
        auto const result = sphChemistryAdvanceBatch({input}, kSfScale).at(0);
        input.state = result.state;
        input.storedEnergy = result.storedEnergy;
    }
    std::vector<uint64_t> result{std::bit_cast<uint32_t>(input.storedEnergy)};
    for (auto const& abundance : input.state.abundances) {
        result.push_back(std::bit_cast<uint32_t>(abundance.raw()));
    }
    result.push_back(std::bit_cast<uint32_t>(input.state.thermalEnergy));
    return result;
}

void checkSfCompositionLocal(bool checkDense) {
    auto const& options = getOptions();
    auto const model = starFormationModel();
    springel_and_hernquist::AtomicCooling const cooling{1.0 - options.cosmology.heliumFraction};
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        auto const& state = sphParticlesChemistryStates[gas];
        auto const output = sphChemistryOutput(state, sphParticlesInternalEnergy(gas), kSfScale);
        requireSfClose(output[0] + output[1] + output[5] + 2.0 * (output[6] + output[7]), 1.0,
            2.e-6, "combined chemistry lost hydrogen nuclei");
        requireSfClose(output[2] + output[3] + output[4],
            options.cosmology.heliumFraction / (4.0 * (1.0 - options.cosmology.heliumFraction)),
            2.e-6, "combined chemistry lost helium nuclei");
        requireSf(sphParticlesChemistrySteps[gas].halfInterval == 0.0 &&
                sphParticlesChemistrySteps[gas].pendingHeating == 0.0,
            "combined source advanced the standalone chemistry clock or ledger");
        if (checkDense && sphParticlesParticleIndex(gas) == 1) {
            double const density = units::toProperDensity(
                units::StoredDensity(sphParticlesDensity(gas)), units::scalesFromOptions(options),
                units::CosmologicalContext(kSfScale, options.cosmology.hubble))
                                       .quantity()
                                       .value();
            springel_and_hernquist::Equilibrium phase;
            springel_and_hernquist::AtomicEquilibrium hot;
            requireSf(springel_and_hernquist::evaluateEquilibrium(model, cooling, density, phase) &&
                    cooling.evaluateAtEnergy(phase.hotEnergy, hot),
                "cannot evaluate dense projection reference");
            requireSfClose(state.abundances[0].fraction(), phase.hotFraction * hot.hydrogenIonized,
                2.e-6, "dense H ionization is not neutral-cold plus atomic-CIE hot gas");
            requireSfClose(state.abundances[1].fraction(), phase.hotFraction * hot.heliumIonized,
                2.e-6, "dense He ionization is not mass weighted");
            requireSfClose(state.abundances[2].fraction(),
                phase.hotFraction * hot.heliumDoubleIonized, 2.e-6,
                "dense He double ionization is not mass weighted");
            for (int species = 3; species != primordial::kAbundanceCount; ++species) {
                requireSf(state.abundances[species].fraction() == 0.0f,
                    "dense atomic projection retained a molecular species");
            }
            requireSfClose(sphParticlesInternalEnergy(gas), sfStoredEnergy(phase.effectiveEnergy),
                2.e-6, "dense branch applied ordinary cooling to effective SH03 energy");
        }
    }
}

void checkSfBatchBoundary() {
    auto const gas = particlesCatIndex(2);
    SphChemistryBatchInput input{sphParticlesChemistryStates[gas], sphParticlesInternalEnergy(gas),
        sphParticlesDensity(gas), 0.0};
    input.state.thermalEnergy *= 2.0f;
    auto const synchronized =
        sphChemistrySynchronizeEnergy(input.state, input.storedEnergy, kSfScale);
    auto const zero = sphChemistryAdvanceBatch({input}, kSfScale).at(0);
    requireSf(zero.storedEnergy == input.storedEnergy &&
            zero.state.thermalEnergy == synchronized.thermalEnergy,
        "zero-interval chemistry changed stored energy or retained stale source energy");
    for (int species = 0; species != primordial::kAbundanceCount; ++species) {
        requireSf(std::bit_cast<uint32_t>(zero.state.abundances[species].raw()) ==
                std::bit_cast<uint32_t>(input.state.abundances[species].raw()),
            "zero-interval chemistry changed an abundance");
    }
    for (double interval : {-1.0, std::numeric_limits<double>::quiet_NaN()}) {
        input.physicalInterval = interval;
        bool rejected = false;
        try {
            sphChemistryAdvanceBatch({input}, kSfScale);
        } catch (std::exception const&) {
            rejected = true;
        }
        requireSf(rejected, "chemistry batch accepted an invalid physical interval");
    }
    input.physicalInterval = sfPhysicalHalf();
    auto invalid = input;
    invalid.state.abundances[0] = primordial::SignedAbundance::fromRaw(2.0f);
    bool rejected = false;
    try {
        sphChemistryAdvanceBatch({input, invalid}, kSfScale);
    } catch (std::exception const&) {
        rejected = true;
    }
    requireSf(rejected, "chemistry batch accepted a final invalid composition");
}

void convertAllSfGas() {
    for (ParticleInteger particle = 0; particle < particlesSize(); ++particle) {
        particlesConvertToStar(particle, 0.0);
    }
    requireSf(sphParticlesSize() == 0, "converting every gas particle left a gas slot");
}

std::vector<double> checkSfEosLocal() {
    auto const& options = getOptions();
    auto const model = starFormationModel();
    springel_and_hernquist::AtomicCooling const cooling{1.0 - options.cosmology.heliumFraction};
    std::vector<double> expected{0.0, INFINITY, INFINITY};
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        double const density = units::toProperDensity(
            units::StoredDensity(sphParticlesDensity(gas)), units::scalesFromOptions(options),
            units::CosmologicalContext(kSfScale, options.cosmology.hubble))
                                   .quantity()
                                   .value();
        double const energy = units::toPhysicalSpecificEnergy(
            units::StoredSpecificEnergy(sphParticlesInternalEnergy(gas)),
            units::scalesFromOptions(options),
            units::CosmologicalContext(kSfScale, options.cosmology.hubble))
                                  .quantity()
                                  .value();
        auto const cache = sphParticlesChemistryEos[gas];
        double effectiveEnergy = energy;
        if (density >= model.thresholdDensity) {
            springel_and_hernquist::Equilibrium phase, lower, upper;
            requireSf(springel_and_hernquist::evaluateEquilibrium(model, cooling, density, phase) &&
                    springel_and_hernquist::evaluateEquilibrium(
                        model, cooling, 0.99 * density, lower) &&
                    springel_and_hernquist::evaluateEquilibrium(
                        model, cooling, 1.01 * density, upper),
                "cannot evaluate independent pressure derivative");
            effectiveEnergy = phase.effectiveEnergy;
            requireSfClose(double(cache.second) * double(cache.first) * energy,
                (upper.pressure - lower.pressure) / (0.02 * density), 3.e-3,
                "SF sound speed does not use the effective pressure derivative");
            double const limit = 0.05 /
                (phase.specificStarFormationRate *
                    (1.0 +
                        (options.starFormation.winds ? options.starFormation.windMassLoading :
                                                       0.0))) /
                (kSfScale * options.units.timeSeconds);
            expected[2] = std::min(expected[2], limit);
            if (sphParticlesRung(gas) >= 3) {
                expected[1] = std::min(expected[1], limit);
            }
        } else {
            requireSfClose(cache.second, model.parameters.gamma, 2.e-7,
                "ordinary gas has an incorrect adiabatic index");
        }
        requireSfClose(double(cache.first) * energy,
            (model.parameters.gamma - 1.0) * effectiveEnergy, 2.e-7,
            "EOS pressure ratio does not reproduce effective or ordinary gas pressure");
        expected[0] += options.initial.gasMass * (model.parameters.gamma - 1.0) *
            sfStoredEnergy(effectiveEnergy) / (kSfScale * kSfScale);
    }
    return expected;
}

#if COSMICTIGER_WITH_GRAVITY
Array<float, kDimensions> sfTestGravity(int mode, int rank) {
    auto const& options = getOptions();
    double const radius =
        options.gravity.supportRadius / selectedGravityKernel().supportRadiusPerEffectiveRadius;
    double const targetTimestep = mode == 0 ? std::exp2(-4.25 - rank) : 0.75;
    double const magnitude = options.gravity.timestepEta * options.gravity.timestepEta * kSfScale *
        radius / (targetTimestep * targetTimestep);
    Array<float, kDimensions> result{};
    result[0] = float(0.6 * magnitude);
    result[1] = float(-0.8 * magnitude);
    result[2] = 0.0f;
    return result;
}

void configureSfGravity(int mode) {
    auto const acceleration = sfTestGravity(mode, hpxRank());
    for (int dim = 0; dim < kDimensions; ++dim) {
        sphParticlesGravityAcceleration(dim, particlesCatIndex(1)) = acceleration[dim];
    }
}
#endif

}    // namespace detail

HPX_PLAIN_ACTION(detail::initializeSfFixture, InitializeSfFixtureAction);
HPX_PLAIN_ACTION(detail::configureSfDraws, ConfigureSfDrawsAction);
HPX_PLAIN_ACTION(detail::sfLocalSnapshot, SfLocalSnapshotAction);
HPX_PLAIN_ACTION(detail::sfCheckpointOperation, SfCheckpointOperationAction);
HPX_PLAIN_ACTION(detail::alterSfFixture, AlterSfFixtureAction);
HPX_PLAIN_ACTION(detail::sfOrdinaryReference, SfOrdinaryReferenceAction);
HPX_PLAIN_ACTION(detail::checkSfCompositionLocal, CheckSfCompositionLocalAction);
HPX_PLAIN_ACTION(detail::checkSfBatchBoundary, CheckSfBatchBoundaryAction);
HPX_PLAIN_ACTION(detail::convertAllSfGas, ConvertAllSfGasAction);
HPX_PLAIN_ACTION(detail::checkSfEosLocal, CheckSfEosLocalAction);
#if COSMICTIGER_WITH_GRAVITY
HPX_PLAIN_ACTION(detail::configureSfGravity, ConfigureSfGravityAction);
#endif

namespace detail {

template <class Action, class... Arguments>
void onSfLocalities(Arguments const&... arguments) {
    std::vector<hpx::future<void>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<Action>(locality, arguments...));
    }
    for (auto& future : pending) {
        future.get();
    }
}

SfSnapshot sfSnapshot() {
    std::vector<hpx::future<std::vector<uint64_t>>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<SfLocalSnapshotAction>(locality));
    }
    SfSnapshot result;
    for (auto& future : pending) {
        result.push_back(future.get());
    }
    return result;
}

SfSnapshot sfReference(int particle, double physicalInterval, int intervals) {
    std::vector<hpx::future<std::vector<uint64_t>>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(
            hpx::async<SfOrdinaryReferenceAction>(locality, particle, physicalInterval, intervals));
    }
    SfSnapshot result;
    for (auto& future : pending) {
        result.push_back(future.get());
    }
    return result;
}

void requireSfReference(SfSnapshot const& actual, SfSnapshot const& expected, size_t particle) {
    for (size_t rank = 0; rank < actual.size(); ++rank) {
        requireSf(actual[rank][particle * kSfRecordWords + 9] == expected[rank][0],
            "ordinary combined source energy differs from standalone chemistry batch");
        for (size_t field = 0; field != 7; ++field) {
            requireSf(
                actual[rank][particle * kSfRecordWords + 16 + field] == expected[rank][1 + field],
                "ordinary combined source composition differs from standalone chemistry batch");
        }
    }
}

double sfDouble(std::vector<uint64_t> const& words, size_t particle, size_t field) {
    return std::bit_cast<double>(words.at(particle * kSfRecordWords + field));
}

double sfFloat(std::vector<uint64_t> const& words, size_t particle, size_t field) {
    return std::bit_cast<float>(uint32_t(words.at(particle * kSfRecordWords + field)));
}

void requireSfInactive(SfSnapshot const& initial, SfSnapshot const& after) {
    for (size_t rank = 0; rank < initial.size(); ++rank) {
        requireSf(std::equal(initial[rank].begin(), initial[rank].begin() + kSfRecordWords,
                      after[rank].begin()),
            "source operator changed an inactive particle");
    }
}

void requireSfLedger(SfSnapshot const& before, SfSnapshot const& after, double heating) {
    double change = 0.0;
    for (size_t rank = 0; rank < before.size(); ++rank) {
        for (size_t particle = 0; particle < 3; ++particle) {
            change += sfFloat(after[rank], particle, 9) - sfFloat(before[rank], particle, 9);
            for (size_t dim = 0; dim < kDimensions; ++dim) {
                double const first = sfFloat(before[rank], particle, 10 + dim);
                double const last = sfFloat(after[rank], particle, 10 + dim);
                change += 0.5 * (last * last - first * first);
            }
        }
    }
    requireSfClose(heating, -getOptions().initial.gasMass * change / kSfScale, 2.e-10,
        "source ledger omitted or double-counted thermal/kinetic energy");
}

void resetSfFixture(bool winds) {
    starFormationOpenNewStars(sfTimestep(), kSfScale, sfTimestep(), 0, false);
    onSfLocalities<InitializeSfFixtureAction>();
    particlesInitializeStarFormationIds();
    onSfLocalities<ConfigureSfDrawsAction>(winds);
}

void checkSfClosure() {
    auto const before = sfSnapshot();
    starFormationRefreshEos(kSfScale);
    std::vector<hpx::future<std::vector<double>>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<CheckSfEosLocalAction>(locality));
    }
    std::vector<double> expected{0.0, INFINITY, INFINITY};
    for (auto& future : pending) {
        auto const local = future.get();
        expected[0] += local[0];
        expected[1] = std::min(expected[1], local[1]);
        expected[2] = std::min(expected[2], local[2]);
    }
    requireSfClose(starFormationPressureIntegral(kSfScale), expected[0], 3.e-7,
        "global effective-pressure integral is inconsistent with the EOS");
    requireSfClose(starFormationMaximumTimestep(3, kSfScale), expected[1], 2.e-12,
        "SF timestep includes inactive particles or has the wrong physical units");
    requireSfClose(starFormationMaximumTimestep(0, kSfScale), expected[2], 2.e-12,
        "SF timestep missed the denser coarse-rung particle");
    requireSf(expected[2] < expected[1], "fixture does not distinguish active rung selection");
    requireSf(sfSnapshot() == before, "EOS diagnostics modified persistent source/particle state");
}

void checkSfThresholdCrossing() {
    resetSfFixture(false);
    double const timestep = sfTimestep();
    starFormationOpen(3, kSfScale, 0.0, timestep);
    starFormationClose(3, kSfScale, 0.0, timestep);
    onSfLocalities<CheckSfCompositionLocalAction>(true);
    onSfLocalities<AlterSfFixtureAction>(7);
    auto const beforeOrdinary = sfSnapshot();
    auto const ordinaryReference = sfReference(1, sfPhysicalHalf(), 2);
    starFormationOpen(3, kSfScale, 0.0, timestep);
    auto const ordinary = starFormationClose(3, kSfScale, 0.0, timestep);
    auto const afterOrdinary = sfSnapshot();
    requireSfReference(afterOrdinary, ordinaryReference, 1);
    requireSfLedger(beforeOrdinary, afterOrdinary, ordinary.heating);
    for (size_t rank = 0; rank < afterOrdinary.size(); ++rank) {
        requireSf(
            afterOrdinary[rank][kSfRecordWords + 4] == beforeOrdinary[rank][kSfRecordWords + 4],
            "ordinary interval continued dense-gas enrichment");
    }
    onSfLocalities<CheckSfCompositionLocalAction>(false);
    onSfLocalities<AlterSfFixtureAction>(8);
    auto const beforeDense = sfSnapshot();
    starFormationOpen(3, kSfScale, 0.0, timestep);
    onSfLocalities<CheckSfCompositionLocalAction>(true);
    auto const dense = starFormationClose(3, kSfScale, 0.0, timestep);
    requireSf(dense.converted == 0, "threshold crossing unexpectedly converted its source fixture");
    requireSfLedger(beforeDense, sfSnapshot(), dense.heating);
    onSfLocalities<CheckSfCompositionLocalAction>(true);
}

void checkSfEmptyGas() {
    auto const original = getOptions();
    auto options = original;
    options.gravity.enabled = false;
    options.run.scenario = "sedov";
    options.starFormation.winds = false;
    setOptions(options);
    resetSfFixture(false);
    onSfLocalities<ConvertAllSfGasAction>();
    domainsRebound();
    auto const stars = sfSnapshot();
    for (auto const& words : stars) {
        for (size_t particle = 0; particle != 3; ++particle) {
            requireSf(words[particle * kSfRecordWords + 14] == kStarType,
                "empty-gas fixture lost a stellar population");
        }
    }
    for (bool global : {false, true}) {
        EnergiesT energies;
        auto const tree = sphStep1(0, false);
        requireSf(tree.numberOfActiveParticles == 0, "empty SPH tree contains active gas");
        auto const result = sphStep2(
            0, 1.0, 0.0, 1.0, 1, 0.0, &energies, false, -1, false, 0.0, global ? 0.125 : 0.0);
        requireSf(std::isfinite(result.minimumTimestep) && result.minimumTimestep > 0.0 &&
                std::isfinite(energies.kin) && energies.therm == 0.0 && energies.heating == 0.0,
            "empty SPH step returned invalid energy or timestep diagnostics");
        requireSf(!global || result.minimumTimestep == 0.125,
            "empty SPH step changed the external global timestep cap");
        requireSf(sfSnapshot() == stars, "empty SPH step modified collisionless stellar state");
    }
    setOptions(original);
}

#if COSMICTIGER_WITH_GRAVITY
void checkSfNewStarKicks() {
    auto const original = getOptions();
    auto options = original;
    options.gravity.enabled = true;
    options.gravity.supportRadius = 1.0;
    options.gravity.timestepEta = 0.25;
    options.starFormation.winds = false;
    for (int mode = 0; mode != 4; ++mode) {
        options.initial.createGlass = mode == 2;
        setOptions(options);
        resetSfFixture(false);
        onSfLocalities<ConfigureSfGravityAction>(mode);
        double const timestep = sfTimestep();
        auto const before = sfSnapshot();
        starFormationOpen(3, kSfScale, 0.0, timestep);
        onSfLocalities<AlterSfFixtureAction>(2);
        auto const conversion = starFormationClose(3, kSfScale, 0.0, timestep);
        auto const synchronized = sfSnapshot();
        requireSf(
            conversion.converted == uint64_t(hpxSize()), "gravity fixture failed to form stars");
        for (size_t rank = 0; rank < before.size(); ++rank) {
            for (size_t particle = 0; particle != 3; ++particle) {
                for (size_t dim = 0; dim < kDimensions; ++dim) {
                    requireSf(sfFloat(before[rank], particle, 10 + dim) ==
                            sfFloat(synchronized[rank], particle, 10 + dim),
                        "gas-to-star conversion changed synchronized momentum");
                }
            }
        }
        int const minimumRung = mode == 2 ? 7 : 0;
        double const maximumTimestep = mode == 1 ? std::exp2(-5.25) : 1.0;
        int const maximumRung =
            starFormationOpenNewStars(1.0, kSfScale, maximumTimestep, minimumRung, true);
        auto const kicked = sfSnapshot();
        int expectedMaximum = 0;
        for (size_t rank = 0; rank < kicked.size(); ++rank) {
            int const expectedRung =
                mode == 0 ? 5 + int(rank) : (mode == 1 ? 6 : (mode == 2 ? 7 : 2));
            expectedMaximum = std::max(expectedMaximum, expectedRung);
            requireSf(kicked[rank][kSfRecordWords + 13] == uint64_t(expectedRung),
                "new star ignored gravity, cap, minimum rung, or old-rung reduction bound");
            auto const acceleration = sfTestGravity(mode, int(rank));
            double const half =
                (options.initial.createGlass ? -0.5 : 0.5) * std::exp2(-expectedRung);
            for (size_t dim = 0; dim < kDimensions; ++dim) {
                double const expectedVelocity = float(std::fma(
                    double(acceleration[dim]), half, sfFloat(synchronized[rank], 1, 10 + dim)));
                requireSfClose(sfFloat(kicked[rank], 1, 10 + dim), expectedVelocity, 2.e-7,
                    "new star did not receive exactly its signed opening gravity half kick");
            }
        }
        requireSf(maximumRung == expectedMaximum, "new-star rung reduction missed a locality");
        requireSf(
            starFormationOpenNewStars(1.0, kSfScale, maximumTimestep, minimumRung, true) == 0 &&
                sfSnapshot() == kicked,
            "new-star gravity opening list was not consumed exactly once");
    }
    setOptions(original);
}
#endif

void requireSfCoupling(float expected) {
    for (int rank = 0; rank < hpxSize(); ++rank) {
        float coupling[5] = {-9.0f, -9.0f, -9.0f, -9.0f, -9.0f};
        starFormationReadHydroCoupling({rank, {0, 3}}, coupling, 1);
        requireSf(coupling[0] == -9.0f && coupling[4] == -9.0f && coupling[1] == 1.0f &&
                coupling[2] == expected && coupling[3] == 1.0f,
            "local/remote wind coupling flags or gather offsets are incorrect");
    }
}

using SfOutputFile = std::unique_ptr<DBfile, decltype(&DBClose)>;
using SfOutputField = std::unique_ptr<DBpointvar, decltype(&DBFreePointvar)>;

std::vector<double> sfOutputField(
    DBfile* file, std::string const& path, int count, char const* mesh, char const* unit) {
    SfOutputField field(DBGetPointvar(file, path.c_str()), DBFreePointvar);
    requireSf(bool(field), "Silo field is missing");
    requireSf(field->datatype == DB_DOUBLE && field->nels == count && field->nvals == 1,
        "Silo field has incorrect datatype or dimensions");
    requireSf(field->meshname && std::filesystem::path(field->meshname).filename() == mesh,
        "Silo field is attached to the wrong species");
    requireSf(field->units && std::string(field->units) == unit, "Silo field has wrong units");
    auto const* values = static_cast<double const*>(field->vals[0]);
    return {values, values + count};
}

void checkSfOutput(bool wind) {
    auto const originalOptions = getOptions();
    auto options = originalOptions;
    options.output.silo = true;
    options.output.directory = "star_formation_output";
    options.output.xMin = options.output.yMin = options.output.zMin = 0.0;
    options.output.xMax = options.output.yMax = options.output.zMax = 1.0;
    auto const snapshot = sfSnapshot();
    auto const model = starFormationModel();
    springel_and_hernquist::AtomicCooling const cooling{1.0 - options.cosmology.heliumFraction};
    double const density = units::toProperDensity(
        units::StoredDensity(sfFloat(snapshot.front(), 0, 15)), units::scalesFromOptions(options),
        units::CosmologicalContext(kSfScale, options.cosmology.hubble))
                               .quantity()
                               .value();
    springel_and_hernquist::Equilibrium phase;
    springel_and_hernquist::AtomicEquilibrium hot;
    requireSf(springel_and_hernquist::evaluateEquilibrium(model, cooling, density, phase) &&
            cooling.evaluateAtEnergy(phase.hotEnergy, hot),
        "cannot evaluate expected output phases");
    for (bool convertedUnits : {false, true}) {
        options.output.units = convertedUnits ? "both" : "code";
        options.output.frame = convertedUnits ? "both" : "comoving";
        setOptions(options);
        viewConfigure();
        int const cycle = 100 + 2 * int(wind) + int(convertedUnits);
        viewOutputViews(cycle, kSfScale, true);
        auto const masterName = std::filesystem::path(options.output.directory) /
            ("view.0." + std::to_string(cycle) + ".silo");
        SfOutputFile master(DBOpen(masterName.c_str(), DB_UNKNOWN, DB_READ), DBClose);
        requireSf(bool(master), "cannot open Silo master");
        int const length = DBGetVarLength(master.get(), "leaf_generation");
        requireSf(length > 1, "missing Silo generation");
        std::vector<char> generation(length);
        requireSf(DBReadVar(master.get(), "leaf_generation", generation.data()) >= 0 &&
                generation.back() == '\0',
            "cannot read Silo generation");
        std::vector<std::string> const frames = convertedUnits ?
            std::vector<std::string>{
                "cgs/comoving/", "cgs/lab/", "cosmological/comoving/", "cosmological/lab/"} :
            std::vector<std::string>{""};
        for (int rank = 0; rank < hpxSize(); ++rank) {
            auto const leafName = std::filesystem::path(masterName.string() + ".data") /
                generation.data() / ("locality_" + std::to_string(rank) + ".silo");
            SfOutputFile leaf(DBOpen(leafName.c_str(), DB_UNKNOWN, DB_READ), DBClose);
            requireSf(bool(leaf), "cannot open Silo particle leaf");
            std::string const locality = "/locality_" + std::to_string(rank) + "/";
            for (auto const& frame : frames) {
                std::string const path = locality + frame;
                requireSf(DBInqVarExists(
                              leaf.get(), (path + "star_formation_phase_convention").c_str()) &&
                        DBInqVarExists(
                            master.get(), (frame + "star_formation_phase_convention").c_str()),
                    "missing Silo phase convention metadata");
                int const count = wind ? 3 : 2;
                auto const cold =
                    sfOutputField(leaf.get(), path + "cold_fraction", count, "gas", "1");
                auto const temperature =
                    sfOutputField(leaf.get(), path + "hot_temperature", count, "gas", "K");
                auto const rate =
                    sfOutputField(leaf.get(), path + "specific_sfr", count, "gas", "1/s");
                requireSfClose(
                    cold[0], phase.coldFraction, 2.e-12, "incorrect cold-fraction output");
                requireSfClose(
                    temperature[0], hot.temperature, 2.e-12, "incorrect hot-temperature output");
                requireSfClose(rate[0], phase.specificStarFormationRate, 2.e-12,
                    "incorrect specific-SFR output");
                for (int index = 1; index < count; ++index) {
                    requireSf(cold[index] == 0.0 && temperature[index] == 0.0 && rate[index] == 0.0,
                        "ordinary gas or a decoupled wind exposed unresolved phases");
                }
                auto const low =
                    sfOutputField(leaf.get(), path + "hydro_id_low", count, "gas", "1");
                auto const high =
                    sfOutputField(leaf.get(), path + "hydro_id_high", count, "gas", "1");
                for (int index = 0; index < count; ++index) {
                    size_t const particle = !wind && index == 1 ? 2 : index;
                    uint64_t const identity = snapshot[rank][particle * kSfRecordWords];
                    requireSf(low[index] == double(uint32_t(identity)) &&
                            high[index] == double(identity >> 32),
                        "gas Silo identity lost precision or changed particle ordering");
                }
                if (!wind) {
                    auto const formed =
                        sfOutputField(leaf.get(), path + "star_formation_time", 1, "stars", "s");
                    auto const starLow =
                        sfOutputField(leaf.get(), path + "star_id_low", 1, "stars", "1");
                    auto const starHigh =
                        sfOutputField(leaf.get(), path + "star_id_high", 1, "stars", "1");
                    uint64_t const identity = snapshot[rank][kSfRecordWords];
                    requireSf(formed[0] == sfDouble(snapshot[rank], 1, 2) &&
                            starLow[0] == double(uint32_t(identity)) &&
                            starHigh[0] == double(identity >> 32),
                        "stellar Silo output lost formation time or persistent identity");
                }
            }
        }
    }
    requireSf(sfSnapshot() == snapshot, "writing phase diagnostics modified particle state");
    setOptions(originalOptions);
    viewConfigure();
}

}    // namespace detail

void starFormationTest() {
    using namespace detail;
    auto const originalOptions = getOptions();
    // Either network: the dense projection is atomic and the ordinary-gas
    // references below come from the configured chemistry batch solver.
    requireSf(originalOptions.starFormation.enabled && originalOptions.hydro.enabled &&
            originalOptions.chemistry.enabled && !originalOptions.compute.cuda,
        "fixture requires CPU SPH star formation with chemistry");
    auto options = originalOptions;
    options.starFormation.winds = false;
    setOptions(options);
    resetSfFixture(false);
    double const timestep = sfTimestep();
    auto const initial = sfSnapshot();
    checkSfClosure();
    onSfLocalities<CheckSfBatchBoundaryAction>();
    requireSf(sfSnapshot() == initial, "standalone batch validation modified particle state");
    auto const ordinaryOpening = sfReference(2, 0.5 * sfPhysicalHalf(), 1);
    auto const ordinaryCompleted = sfReference(2, 0.5 * sfPhysicalHalf(), 2);
    starFormationOpen(3, kSfScale, 0.0, timestep);
    auto const opened = sfSnapshot();
    requireSfReference(opened, ordinaryOpening, 2);
    onSfLocalities<CheckSfCompositionLocalAction>(true);
    requireSfInactive(initial, opened);
    for (size_t rank = 0; rank < opened.size(); ++rank) {
        for (size_t particle = 1; particle < 3; ++particle) {
            int const rung = int(opened[rank][particle * kSfRecordWords + 13]);
            double const half = std::ldexp(0.5 * timestep, -rung);
            requireSf(sfDouble(opened[rank], particle, 5) == half,
                "opening changed the selected rung interval");
            requireSfClose(sfDouble(opened[rank], particle, 8),
                kSfScale * half * options.units.timeSeconds, 4.e-15,
                "source interval did not convert conformal time to physical seconds");
            requireSf(opened[rank][particle * kSfRecordWords + 13] ==
                    initial[rank][particle * kSfRecordWords + 13],
                "opening changed particle rungs");
        }
    }
    bool rejected = false;
    try {
        starFormationClose(3, kSfScale, 0.0, 2.0 * timestep);
    } catch (std::exception const&) {
        rejected = true;
    }
    requireSf(rejected && sfSnapshot() == opened,
        "closing accepted a mismatched interval or changed a failed batch");
    onSfLocalities<SfCheckpointOperationAction>(0);
    auto const completed = starFormationClose(3, kSfScale, 0.0, timestep);
    auto const uninterrupted = sfSnapshot();
    requireSfReference(uninterrupted, ordinaryCompleted, 2);
    onSfLocalities<CheckSfCompositionLocalAction>(true);
    requireSf(completed.converted == 0, "small deterministic interval unexpectedly formed stars");
    requireSfInactive(initial, uninterrupted);
    requireSfLedger(initial, uninterrupted, completed.heating);
    onSfLocalities<SfCheckpointOperationAction>(1);
    requireSf(sfSnapshot() == opened, "restart lost the pending opening source state");
    auto const resumed = starFormationClose(3, kSfScale, 0.0, timestep);
    requireSf(sfSnapshot() == uninterrupted && resumed.heating == completed.heating &&
            resumed.converted == completed.converted,
        "restart between source halves changed final metadata, energy, or event draws");
    onSfLocalities<SfCheckpointOperationAction>(2);

    checkSfThresholdCrossing();
    resetSfFixture(false);
    onSfLocalities<AlterSfFixtureAction>(6);
    auto const invalidChemistry = sfSnapshot();
    rejected = false;
    try {
        starFormationOpen(3, kSfScale, 0.0, timestep);
    } catch (std::exception const&) {
        rejected = true;
    }
    requireSf(rejected && sfSnapshot() == invalidChemistry,
        "failed chemistry batch committed partial chemical, phase, or source state");

    resetSfFixture(false);
    onSfLocalities<AlterSfFixtureAction>(0);
    auto const invalid = sfSnapshot();
    rejected = false;
    try {
        starFormationOpen(3, kSfScale, 0.0, timestep);
    } catch (std::exception const&) {
        rejected = true;
    }
    requireSf(rejected && sfSnapshot() == invalid,
        "failed prepare on the last rank committed part of a source batch");
    onSfLocalities<AlterSfFixtureAction>(1);
    auto const beforeConversion = sfSnapshot();
    starFormationOpen(3, kSfScale, 0.0, timestep);
    onSfLocalities<AlterSfFixtureAction>(2);
    auto const converted = starFormationClose(3, kSfScale, 0.0, timestep);
    auto const afterConversion = sfSnapshot();
    requireSf(converted.converted == uint64_t(hpxSize()) && converted.maximumNewStarRung == 3,
        "forced event did not convert exactly the selected above-threshold gas");
    requireSfInactive(beforeConversion, afterConversion);
    requireSfLedger(beforeConversion, afterConversion, converted.heating);
    for (size_t rank = 0; rank < afterConversion.size(); ++rank) {
        auto const& words = afterConversion[rank];
        requireSf(words[kSfRecordWords + 14] == kStarType &&
                words[2 * kSfRecordWords + 14] == kSphType &&
                words[kSfRecordWords] == beforeConversion[rank][kSfRecordWords],
            "conversion changed identity or converted below-threshold gas");
        requireSfClose(sfDouble(words, 1, 2), 2.0 * sfPhysicalHalf(), 4.e-15,
            "stellar formation time does not include both physical source halves");
        requireSf(sfDouble(words, 1, 5) == 0.0 && sfDouble(words, 1, 6) == 0.0 &&
                sfDouble(words, 1, 7) == 0.0,
            "converted star retained pending gas source state");
    }
    checkSfOutput(false);
#if COSMICTIGER_WITH_GRAVITY
    checkSfNewStarKicks();
#endif

    options.starFormation.winds = true;
    options.starFormation.windMassLoading = 1000.0;
    setOptions(options);
    resetSfFixture(true);
    auto const beforeWind = sfSnapshot();
    starFormationOpen(3, kSfScale, 0.0, timestep);
    onSfLocalities<AlterSfFixtureAction>(3);
    auto const wind = starFormationClose(3, kSfScale, 0.0, timestep);
    auto const afterWind = sfSnapshot();
    onSfLocalities<CheckSfCompositionLocalAction>(false);
    requireSf(wind.converted == 0, "wind draw incorrectly formed a collisionless star");
    requireSfInactive(beforeWind, afterWind);
    requireSfLedger(beforeWind, afterWind, wind.heating);
    double const storedSpeed =
        springel_and_hernquist::windSpeed(starFormationModel().parameters,
            options.starFormation.windMassLoading, options.starFormation.windEnergyFraction) *
        kSfScale * options.units.timeSeconds / options.units.lengthCm;
    for (size_t rank = 0; rank < afterWind.size(); ++rank) {
        double speedSquared = 0.0;
        for (size_t dim = 0; dim < kDimensions; ++dim) {
            double const delta =
                sfFloat(afterWind[rank], 1, 10 + dim) - sfFloat(beforeWind[rank], 1, 10 + dim);
            speedSquared += delta * delta;
        }
        requireSfClose(speedSquared, storedSpeed * storedSpeed, 2.e-6,
            "wind velocity kick has the wrong kinetic injection budget");
        requireSf(sfDouble(afterWind[rank], 1, 3) > 0.0 &&
                afterWind[rank][kSfRecordWords + 14] == kSphType,
            "wind particle did not retain gas type and decoupling time");
    }
    requireSfCoupling(0.0f);
    checkSfOutput(true);

    auto const beforeDecoupled = sfSnapshot();
    auto const decoupledReference = sfReference(1, sfPhysicalHalf(), 2);
    starFormationOpen(3, kSfScale, 0.0, timestep);
    auto const decoupled = starFormationClose(3, kSfScale, 0.0, timestep);
    auto const afterDecoupled = sfSnapshot();
    requireSfReference(afterDecoupled, decoupledReference, 1);
    requireSfLedger(beforeDecoupled, afterDecoupled, decoupled.heating);
    requireSfCoupling(0.0f);
    onSfLocalities<CheckSfCompositionLocalAction>(false);

    onSfLocalities<AlterSfFixtureAction>(4);
    auto const beforeExpiry = sfSnapshot();
    starFormationOpen(3, kSfScale, 0.0, timestep);
    auto const expired = sfSnapshot();
    onSfLocalities<CheckSfCompositionLocalAction>(true);
    for (size_t rank = 0; rank < expired.size(); ++rank) {
        requireSf(sfDouble(expired[rank], 1, 3) == 0.0, "wind failed to recouple at timer expiry");
        double const starFormingDuration = sfPhysicalHalf() - sfDouble(beforeExpiry[rank], 1, 3);
        double const expectedHazard =
            sfSpecificRate(float(sfFloat(expired[rank], 1, 15))) * starFormingDuration;
        requireSfClose(sfDouble(expired[rank], 1, 7), expectedHazard, 2.e-6,
            "timer expiry counted decoupled time as star-forming time");
    }
    requireSfCoupling(1.0f);
    auto const expiryClosed = starFormationClose(3, kSfScale, 0.0, timestep);
    requireSf(expiryClosed.converted == 0, "timer-expiry regression unexpectedly formed stars");
    requireSfLedger(beforeExpiry, sfSnapshot(), expiryClosed.heating);
    onSfLocalities<AlterSfFixtureAction>(5);
    auto const recoupledReference = sfReference(1, sfPhysicalHalf(), 1);
    starFormationOpen(3, kSfScale, 0.0, timestep);
    auto const densityRecoupled = sfSnapshot();
    requireSfReference(densityRecoupled, recoupledReference, 1);
    for (auto const& words : densityRecoupled) {
        requireSf(sfDouble(words, 1, 3) == 0.0 && sfDouble(words, 1, 7) == 0.0,
            "low-density wind failed to recouple without star formation");
    }
    requireSfCoupling(1.0f);
    starFormationClose(3, kSfScale, 0.0, timestep);
    starFormationOpenNewStars(timestep, kSfScale, timestep, 0, false);
    checkSfEmptyGas();
    setOptions(originalOptions);
    PRINT("Star formation passed: mixed rungs, physical intervals, source energy ledger, "
          "failed-batch atomicity, restart metadata, converted=%llu, wind=%i, "
          "EOS/pressure/timestep closure, new-star gravity kicks, remote coupling, "
          "timer/density recoupling, combined chemistry/transitions/rollback, empty-gas stepping, "
          "and Silo phase/population diagnostics\n",
        static_cast<unsigned long long>(converted.converted), hpxSize());
}
