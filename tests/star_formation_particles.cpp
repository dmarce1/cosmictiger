/* CosmicTiger - star-formation population persistence regression; BSL-1.0. */

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/thermodynamics.hpp>
#include <cosmictiger/sph/tree.hpp>
#include <cosmictiger/stars/star_formation.hpp>

namespace detail {

void requireStarParticles(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(std::string("Star-formation particles: ") + message);
    }
}

float fixtureElectronFraction(int rank, int originalGasIndex) {
    return {float(0.25 + 0.125 * originalGasIndex + 0.01 * rank)};
}

void requireFixtureChemistry(ParticleInteger gasIndex, int rank, int originalGasIndex) {
    auto const expected = fixtureElectronFraction(rank, originalGasIndex);
    requireStarParticles(sphParticlesElectronFractions[gasIndex] == expected,
        "combined persistence changed the GADGET-4 electron guess");
    auto const thermal = sph_entropy::prepareStoredEnergy(
        float(2 + originalGasIndex), float(3 + originalGasIndex), getOptions().hydro.gamma, 1.0);
    requireStarParticles(sphParticlesEntropyInitialized &&
            sphParticlesEntropies[gasIndex] == thermal.entropy &&
            sphParticlesProposedRungs[gasIndex] == 3 + originalGasIndex,
        "compaction, checkpoint or migration changed canonical entropy or hydro proposal");
    if (sphParticlesBulkViscosityCoefficients != nullptr) {
        requireStarParticles(
            sphParticlesBulkViscosityCoefficients[gasIndex] == 0.25f + 0.125f * originalGasIndex,
            "compaction, checkpoint or migration changed viscosity coefficient");
    }
    if (sphParticlesCullenDehnenStates != nullptr) {
        auto const& history = sphParticlesCullenDehnenStates[gasIndex];
        requireStarParticles(history.physicalDivergence() == float(-2 - originalGasIndex) &&
                history.sampleRung == 1 + originalGasIndex &&
                history.sampledDivergence() == float(-4 - originalGasIndex),
            "compaction, checkpoint or migration changed committed viscosity history");
    }
}

void initializeStarParticleFixture() {
    particlesResize(3);
    sphParticlesResize(2, false);
    sphParticlesEntropyInitialized = true;
    sphParticlesThermalScaleFactor = 1.0;
    for (ParticleInteger index = 0; index < 3; ++index) {
        particlesStarFormationStates[index] = {};
        particlesStarFormationStates[index].id = uint64_t(hpxRank()) * 3 + index + 1;
        particlesRung(index) = 2;
        particlesType(index) = index == 0 ? kDarkMatterType : kSphType;
        particlesCatIndex(index) = index == 0 ? kNoParticleIndex : index - 1;
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, index) = 0.125 * (index + 1);
            particlesVel(dim, index) = float(index + dim);
        }
        if (index > 0) {
            ParticleInteger const gasIndex = index - 1;
            sphParticlesParticleIndex(gasIndex) = index;
            sphParticlesSupportRadius(gasIndex) = 0.125f;
            sphParticlesDensity(gasIndex) = 2.0f + index;
            sphParticlesSetInternalEnergy(gasIndex, 1.0f + index);
            sphParticlesProposedRungs[gasIndex] = char(3 + gasIndex);
            if (sphParticlesBulkViscosityCoefficients != nullptr) {
                sphParticlesBulkViscosityCoefficients[gasIndex] = 0.25f + 0.125f * gasIndex;
            }
            if (sphParticlesCullenDehnenStates != nullptr) {
                auto& history = sphParticlesCullenDehnenStates[gasIndex];
                history = {};
                history.setDivergences(float(-2 - gasIndex), float(-4 - gasIndex));
                history.sampleRung = int8_t(1 + gasIndex);
            }
            sphParticlesElectronFractions[gasIndex] = fixtureElectronFraction(hpxRank(), gasIndex);
        }
    }
}

std::vector<Particle> exerciseStarParticleFixture() {
    uint64_t const offset = uint64_t(hpxRank()) * 3;
    for (ParticleInteger index = 0; index < 3; ++index) {
        auto& state = particlesStarFormationStates[index];
        requireStarParticles(state.id == offset + uint64_t(index) + 1,
            "initialization changed the supplied source identities");
        state.id += uint64_t{1} << 60;
        state.eventCounter = uint64_t{1} << (index + 40);
        state.elapsedPhysicalTime = 16.0;
        state.specificStarFormationRate = index > 0 ? 0.125 * (index + 1) : 0.0;
    }
    auto const oldGasId = particlesStarFormationStates[2].id;
    particlesSwap(0, 2);
    particlesResolveWithSphParticles();
    requireStarParticles(
        particlesStarFormationStates[0].id == oldGasId, "permutation detached persistent metadata");
    auto const stellarId = particlesStarFormationStates[1].id;
    auto const stellarCounter = particlesStarFormationStates[1].eventCounter;
    float const mass = particlesMass(1);
    auto const velocity = particlesV[1];
    particlesConvertToStar(1, 16.0);
    requireStarParticles(particlesSize() == 3 && sphParticlesSize() == 1,
        "whole-particle conversion changed gravity count or wrong gas count");
    requireStarParticles(particlesType(1) == kStarType &&
            particlesCatIndex(1) == kNoParticleIndex && particlesMass(1) == mass,
        "conversion lost species, category, or mass");
    requireStarParticles(particlesCatIndex(0) == 0 && sphParticlesParticleIndex(0) == 0,
        "gas compaction did not repair the inverse indices");
    requireFixtureChemistry(0, hpxRank(), 1);
    for (int dim = 0; dim < kDimensions; ++dim) {
        requireStarParticles(particlesVel(dim, 1) == velocity[dim], "conversion changed momentum");
    }
    auto const& star = particlesStarFormationStates[1];
    requireStarParticles(star.id == stellarId && star.eventCounter == stellarCounter &&
            star.formationTime == 16.0 && star.specificStarFormationRate == 0.0,
        "conversion lost identity or retained pending sources");
    auto const gas = particlesStarFormationStates[0];
    FILE* file = std::tmpfile();
    requireStarParticles(file != nullptr, "cannot create checkpoint fixture");
    particlesSave(file);
    for (ParticleInteger index = 0; index < 3; ++index) {
        particlesStarFormationStates[index] = {};
    }
    sphParticlesElectronFractions[0] = 1.0f;
    std::rewind(file);
    particlesLoad(file);
    FILE* const mixedCheckpoint = file;
    requireFixtureChemistry(0, hpxRank(), 1);
    auto const& restoredGas = particlesStarFormationStates[0];
    requireStarParticles(restoredGas == gas, "checkpoint changed persistent gas source state");
    requireStarParticles(particlesStarFormationStates[1].id == stellarId &&
            particlesStarFormationStates[1].eventCounter == stellarCounter &&
            particlesStarFormationStates[1].formationTime == 16.0,
        "checkpoint changed stellar metadata");
    std::vector<Particle> result;
    for (ParticleInteger index = 0; index < 3; ++index) {
        result.push_back(particlesGetParticle(index));
    }
    particlesConvertToStar(0, 16.0);
    requireStarParticles(sphParticlesSize() == 0 && particlesSize() == 3,
        "last gas particle could not become a star");
    file = std::tmpfile();
    requireStarParticles(file != nullptr, "cannot create empty-gas checkpoint fixture");
    particlesSave(file);
    std::rewind(file);
    particlesLoad(file);
    std::fclose(file);
    requireStarParticles(sphParticlesSize() == 0 && particlesType(0) == kStarType,
        "checkpoint failed for an exhausted gas population");
    std::rewind(mixedCheckpoint);
    particlesLoad(mixedCheckpoint);
    std::fclose(mixedCheckpoint);
    requireFixtureChemistry(0, hpxRank(), 1);
    return result;
}

std::vector<Particle> starParticleMigrationSnapshot() {
    std::vector<Particle> result;
    for (ParticleInteger index = 0; index < particlesSize(); ++index) {
        result.push_back(particlesGetParticle(index));
        if (particlesType(index) == kSphType) {
            ParticleInteger const gasIndex = particlesCatIndex(index);
            requireStarParticles(gasIndex >= 0 && gasIndex < sphParticlesSize() &&
                    sphParticlesParticleIndex(gasIndex) == index,
                "migration broke gas inverse indices");
            uint64_t const originalId =
                particlesStarFormationStates[index].id - (uint64_t{1} << 60);
            requireFixtureChemistry(gasIndex, int((originalId - 1) / 3), 1);
        }
    }
    return result;
}

#if COSMICTIGER_ENABLE_FMM
void initializeNewbornGravityFixture(bool convert) {
    ParticleInteger const count = hpxRank() == 0 ? 2 : 0;
    particlesResize(count);
    sphParticlesResize(convert && count != 0 ? 1 : 0, false);
    sphParticlesEntropyInitialized = true;
    sphParticlesThermalScaleFactor = 1.0;
    for (ParticleInteger particle = 0; particle < count; ++particle) {
        particlesType(particle) =
            particle == 0 ? kDarkMatterType : (convert ? kSphType : kStarType);
        particlesCatIndex(particle) = convert && particle == 1 ? 0 : kNoParticleIndex;
        particlesRung(particle) = 2;
        auto& state = particlesStarFormationStates[particle];
        state = {};
        state.id = uint64_t(particle) + 1;
        if (particle == 1) {
            state.specificStarFormationRate = convert ? 1.0e100 : 0.0;
            if (!convert) {
                state.eventCounter = 1;
                state.elapsedPhysicalTime = 0.00025 * getOptions().units.timeSeconds;
                state.formationTime = 1.0;
            }
        }
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            particlesPos(dimension, particle) =
                dimension == kXDimension ? 0.375 - 0.125 * particle : 0.375;
            particlesVel(dimension, particle) = particle == 1 ? 0.125f * (dimension + 1) : 0.0f;
        }
    }
    if (convert && count != 0) {
        sphParticlesParticleIndex(0) = 1;
        sphParticlesDensity(0) = 1.0f;
        sphParticlesSetInternalEnergy(0, 2.0f);
        sphParticlesSupportRadius(0) = 0.05f;
        sphParticlesElectronFractions[0] = 1.0f;
    }
}
#endif

}    // namespace detail

#if COSMICTIGER_ENABLE_FMM
HPX_PLAIN_ACTION(detail::initializeNewbornGravityFixture, InitializeNewbornGravityFixtureAction);
#endif
HPX_PLAIN_ACTION(detail::initializeStarParticleFixture, InitializeStarParticleFixtureAction);
HPX_PLAIN_ACTION(detail::exerciseStarParticleFixture, ExerciseStarParticleFixtureAction);
HPX_PLAIN_ACTION(detail::starParticleMigrationSnapshot, StarParticleMigrationSnapshotAction);

namespace detail {
#if COSMICTIGER_ENABLE_FMM
std::vector<Particle> newbornGravitySnapshot() {
    std::vector<hpx::future<std::vector<Particle>>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<StarParticleMigrationSnapshotAction>(locality));
    }
    std::vector<Particle> particles;
    for (auto& future : pending) {
        auto local = future.get();
        particles.insert(particles.end(), local.begin(), local.end());
    }
    std::sort(particles.begin(), particles.end(), [](auto const& first, auto const& second) {
        return first.starFormation.id < second.starFormation.id;
    });
    return particles;
}

void verifyNewbornGravity() {
    auto const originalOptions = getOptions();
    auto options = originalOptions;
    options.gravity.enabled = true;
    options.gravity.method = "fmm";
    options.gravity.cuda = false;
    options.gravity.gm = 1.0;
    options.gravity.supportRadius = 0.05;
    options.run.scenario = "cosmology";
    options.run.minRung = 2;
    options.initial.numberOfParticles = 2;
    options.initial.darkMatterMass = 2.0;
    options.initial.gasMass = 1.0;
    std::vector<std::string> integrators;
    if (integration::kBuildHasTraditional) {
        integrators.push_back("traditional");
    }
    if (integration::kBuildHasGlobal) {
        integrators.push_back("global");
    }
    if (integration::kBuildHasHierarchical) {
        integrators.push_back("hierarchical");
    }
    for (auto const& integrator : integrators) {
        options.run.timeIntegrator = integrator;
        options.run.hierarchicalGravity = integrator == "hierarchical";
        setOptions(options);
        std::vector<Particle> reference;
        for (bool const convert : {false, true}) {
            sphTreeDestroy(false);
            std::vector<hpx::future<void>> pending;
            for (auto const& locality : hpxLocalities()) {
                pending.push_back(
                    hpx::async<InitializeNewbornGravityFixtureAction>(locality, convert));
            }
            for (auto& future : pending) {
                future.get();
            }
            particlesSortByRung(0);
            if (convert) {
                auto const source = starFormationClose(2, 1.0, 0.0, 0.001, 0.00025);
                requireStarParticles(source.converted == 1,
                    "nonzero-gravity fixture failed to create its newborn star");
            }
            auto const before = newbornGravitySnapshot();
            requireStarParticles(before.size() == 2 && before[1].type == kStarType &&
                    before[1].rung == 2 && before[1].velocity[kXDimension] == 0.125f,
                "conversion applied an extra kick or changed the old gravity rung");
            domainsRebound();
            EnergiesT energies{};
            if (options.run.hierarchicalGravity) {
#if COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL
                int minimumRung = 2;
                kickStepHierarchical(minimumRung, 2, 1.0, 0.0, 0.00025, 0.001, 0.5, &energies, 2,
                    true, nullptr, true, 0.0);
#endif
            } else {
                bool const global = integrator == "global";
                advanceForces(global ? 0 : 2, 1.0, 0.0, 0.00025, 0.001, 0.5, false, true, energies,
                    -1, false, 0.00025, global ? 0.00025 : 0.0);
            }
            auto const after = newbornGravitySnapshot();
            requireStarParticles(after.size() == 2 &&
                    std::abs(after[1].velocity[kXDimension] - before[1].velocity[kXDimension]) >
                        1.0e-6f,
                "newborn gravity equivalence fixture did not exercise a nonzero kick");
            if (!convert) {
                reference = after;
            } else {
                for (size_t index = 0; index < after.size(); ++index) {
                    requireStarParticles(
                        after[index].starFormation == reference[index].starFormation &&
                            after[index].rung == reference[index].rung &&
                            after[index].type == reference[index].type,
                        "newborn gravity boundary changed source state or selected a different "
                        "rung");
                    for (int dimension = 0; dimension < kDimensions; ++dimension) {
                        requireStarParticles(after[index].position[dimension] ==
                                    reference[index].position[dimension] &&
                                after[index].velocity[dimension] ==
                                    reference[index].velocity[dimension],
                            "newborn missed or repeated an ordinary gravity boundary kick");
                    }
                }
            }
        }
        std::cout << "Newborn ordinary gravity equivalence passed: " << integrator << '\n';
    }
    setOptions(originalOptions);
}
#endif
}    // namespace detail

void starFormationParticlesTest() {
    detail::requireStarParticles(getOptions().starFormation.enabled && getOptions().hydro.enabled &&
            getOptions().chemistry.enabled && !getOptions().starFormation.cuda,
        "fixture requires CPU SPH star formation with particle chemistry");
    for (bool const gas : {false, true}) {
        size_t const stride = domainsMigrationRecordBytes(gas);
        size_t const capacity = domainsMigrationParcelCapacity(gas);
        size_t const budget = kMaxMigrationParcelBytes - kMigrationParcelOverheadBytes;
        detail::requireStarParticles(
            capacity > 0 && capacity * stride <= budget && (capacity + 1) * stride > budget,
            "migration send boundary does not enforce the serialized byte budget");
        size_t const empty = domainsMigrationPayloadBytes(gas, 0);
        size_t const payload = domainsMigrationPayloadBytes(gas, capacity);
        detail::requireStarParticles(
            payload <= kMaxMigrationParcelBytes && payload - empty == capacity * stride,
            "a full migration vector exceeds its measured serialized byte budget");
    }
    std::vector<hpx::future<void>> initialization;
    for (auto const& locality : hpxLocalities()) {
        initialization.push_back(hpx::async<InitializeStarParticleFixtureAction>(locality));
    }
    for (auto& future : initialization) {
        future.get();
    }
    particlesInitializeStarFormationIds();
    std::vector<hpx::future<std::vector<Particle>>> results;
    for (auto const& locality : hpxLocalities()) {
        results.push_back(hpx::async<ExerciseStarParticleFixtureAction>(locality));
    }
    std::vector<Particle> beforeMigration;
    for (size_t rank = 0; rank < results.size(); ++rank) {
        auto const particles = results[rank].get();
        beforeMigration.insert(beforeMigration.end(), particles.begin(), particles.end());
        auto const& star = particles.at(1);
        detail::requireStarParticles(star.type == kStarType &&
                star.starFormation.id == (uint64_t{1} << 60) + uint64_t(rank) * 3 + 2 &&
                star.starFormation.eventCounter == uint64_t{1} << 41 &&
                star.starFormation.formationTime == 16.0,
            "HPX particle serialization lost persistent star identity");
    }
    domainsRebound();
    domainsBegin(0);
    domainsEnd();
    results.clear();
    for (auto const& locality : hpxLocalities()) {
        results.push_back(hpx::async<StarParticleMigrationSnapshotAction>(locality));
    }
    std::vector<Particle> afterMigration;
    for (auto& future : results) {
        auto const particles = future.get();
        afterMigration.insert(afterMigration.end(), particles.begin(), particles.end());
    }
    auto byId = [](Particle const& first, Particle const& second) {
        return first.starFormation.id < second.starFormation.id;
    };
    std::sort(beforeMigration.begin(), beforeMigration.end(), byId);
    std::sort(afterMigration.begin(), afterMigration.end(), byId);
    detail::requireStarParticles(
        beforeMigration.size() == afterMigration.size(), "migration changed the global population");
    for (size_t index = 0; index < beforeMigration.size(); ++index) {
        detail::requireStarParticles(beforeMigration[index].type == afterMigration[index].type &&
                beforeMigration[index].starFormation == afterMigration[index].starFormation,
            "migration lost stellar identity or saved gas source state");
    }
#if COSMICTIGER_ENABLE_FMM
    detail::verifyNewbornGravity();
#endif
    std::cout << "Star-formation particles passed: IDs, permutation, conversion, compaction, "
                 "momentum, checkpoint, combined chemistry/source metadata, empty gas, "
                 "HPX serialization and migration\n"
              << std::flush;
}
