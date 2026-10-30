/* CosmicTiger - star-formation population persistence regression; BSL-1.0. */

#include <algorithm>
#include <bit>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace detail {

void requireStarParticles(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(std::string("Star-formation particles: ") + message);
    }
}

primordial::State fixtureChemistry(int rank, int originalGasIndex) {
    primordial::State state;
    state.abundances[0].setRaw(originalGasIndex == 0 ? 0.125f : -0.125f);
    state.abundances[1].setRaw(0.125f);
    state.abundances[2].setRaw(0.0625f);
    state.thermalEnergy = 4.0f + 2.0f * rank + originalGasIndex;
    return state;
}

Pair<float> fixtureChemistryEos(int originalGasIndex) {
    return {0.5f + 0.0625f * originalGasIndex, 1.5f + 0.0625f * originalGasIndex};
}

SphChemistryStep fixtureChemistryStep(int rank, int originalGasIndex) {
    return {0.25 * (originalGasIndex + 1) + rank, -0.5 * (originalGasIndex + 1) - rank};
}

void requireFixtureChemistry(ParticleInteger gasIndex, int rank, int originalGasIndex) {
    auto const expected = fixtureChemistry(rank, originalGasIndex);
    auto const& state = sphParticlesChemistryStates[gasIndex];
    for (int species = 0; species < primordial::kAbundanceCount; ++species) {
        requireStarParticles(std::bit_cast<uint32_t>(state.abundances[species].raw()) ==
                std::bit_cast<uint32_t>(expected.abundances[species].raw()),
            "combined persistence changed a signed chemistry abundance");
    }
    requireStarParticles(state.thermalEnergy == expected.thermalEnergy,
        "combined persistence changed chemistry thermal energy");
    auto const expectedEos = fixtureChemistryEos(originalGasIndex);
    auto const eos = sphParticlesChemistryEos[gasIndex];
    requireStarParticles(eos.first == expectedEos.first && eos.second == expectedEos.second,
        "combined persistence changed chemistry EOS");
    auto const expectedStep = fixtureChemistryStep(rank, originalGasIndex);
    auto const step = sphParticlesChemistrySteps[gasIndex];
    requireStarParticles(step.halfInterval == expectedStep.halfInterval &&
            step.pendingHeating == expectedStep.pendingHeating,
        "combined persistence changed the pending chemistry source");
}

void initializeStarParticleFixture() {
    particlesResize(3);
    sphParticlesResize(2, false);
    for (ParticleInteger index = 0; index < 3; ++index) {
        particlesStarFormationStates[index] = {};
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
            sphParticlesInternalEnergy(gasIndex) = 1.0f + index;
            sphParticlesSupportRadius(gasIndex) = 0.125f;
            sphParticlesDensity(gasIndex) = 2.0f + index;
            sphParticlesChemistryStates[gasIndex] = fixtureChemistry(hpxRank(), gasIndex);
            sphParticlesChemistryEos[gasIndex] = fixtureChemistryEos(gasIndex);
            sphParticlesChemistrySteps[gasIndex] = fixtureChemistryStep(hpxRank(), gasIndex);
        }
    }
}

std::vector<Particle> exerciseStarParticleFixture() {
    uint64_t const offset = uint64_t(hpxRank()) * 3;
    for (ParticleInteger index = 0; index < 3; ++index) {
        auto& state = particlesStarFormationStates[index];
        requireStarParticles(state.id == offset + uint64_t(index) + 1,
            "distributed IDs are not unique rank-prefix identities");
        state.id += uint64_t{1} << 60;
        state.eventCounter = uint64_t{1} << (index + 40);
        state.metallicity = float(index) / 64.0f;
        state.elapsedPhysicalTime = 16.0;
        if (index > 0) {
            state.sourceHalfInterval = 0.125;
            state.pendingHeating = -0.25;
            state.formationOpticalDepth = 0.5;
            state.windTimeRemaining = 0.75;
        }
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
    requireStarParticles(
        sphParticlesInternalEnergy(0) == 3.0f, "gas compaction lost the remaining thermal state");
    requireFixtureChemistry(0, hpxRank(), 1);
    for (int dim = 0; dim < kDimensions; ++dim) {
        requireStarParticles(particlesVel(dim, 1) == velocity[dim], "conversion changed momentum");
    }
    auto const& star = particlesStarFormationStates[1];
    requireStarParticles(star.id == stellarId && star.eventCounter == stellarCounter &&
            star.formationTime == 16.0 && star.windTimeRemaining == 0.0 &&
            star.sourceHalfInterval == 0.0 && star.pendingHeating == 0.0 &&
            star.formationOpticalDepth == 0.0,
        "conversion lost identity or retained pending sources");
    auto const gas = particlesStarFormationStates[0];
    FILE* file = std::tmpfile();
    requireStarParticles(file != nullptr, "cannot create checkpoint fixture");
    particlesSave(file);
    for (ParticleInteger index = 0; index < 3; ++index) {
        particlesStarFormationStates[index] = {};
    }
    sphParticlesChemistryStates[0] = {};
    sphParticlesChemistryEos[0] = {};
    sphParticlesChemistrySteps[0] = {};
    std::rewind(file);
    particlesLoad(file);
    FILE* const mixedCheckpoint = file;
    requireFixtureChemistry(0, hpxRank(), 1);
    auto const& restoredGas = particlesStarFormationStates[0];
    requireStarParticles(restoredGas.id == gas.id && restoredGas.eventCounter == gas.eventCounter &&
            restoredGas.metallicity == gas.metallicity &&
            restoredGas.sourceHalfInterval == gas.sourceHalfInterval &&
            restoredGas.pendingHeating == gas.pendingHeating &&
            restoredGas.formationOpticalDepth == gas.formationOpticalDepth &&
            restoredGas.elapsedPhysicalTime == gas.elapsedPhysicalTime &&
            restoredGas.windTimeRemaining == gas.windTimeRemaining,
        "checkpoint changed persistent gas source state");
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
            requireStarParticles(sphParticlesInternalEnergy(gasIndex) == 3.0f,
                "migration changed gas thermal state");
            uint64_t const originalId =
                particlesStarFormationStates[index].id - (uint64_t{1} << 60);
            requireFixtureChemistry(gasIndex, int((originalId - 1) / 3), 1);
        }
    }
    return result;
}

}    // namespace detail

HPX_PLAIN_ACTION(detail::initializeStarParticleFixture, InitializeStarParticleFixtureAction);
HPX_PLAIN_ACTION(detail::exerciseStarParticleFixture, ExerciseStarParticleFixtureAction);
HPX_PLAIN_ACTION(detail::starParticleMigrationSnapshot, StarParticleMigrationSnapshotAction);

void starFormationParticlesTest() {
    detail::requireStarParticles(getOptions().starFormation.enabled && getOptions().hydro.enabled &&
            getOptions().chemistry.enabled && !getOptions().compute.cuda,
        "fixture requires CPU SPH star formation with particle chemistry");
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
            "migration lost stellar identity or pending gas source state");
    }
    PRINT("Star-formation particles passed: IDs, permutation, conversion, compaction, "
          "momentum, checkpoint, combined chemistry/source metadata, empty gas, "
          "HPX serialization and migration\n");
}
