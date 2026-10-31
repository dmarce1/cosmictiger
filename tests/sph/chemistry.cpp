/* CosmicTiger - individual SPH chemistry timestep regression; BSL-1.0. */

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/math/simulation_units.hpp>
#include <cosmictiger/sph/sph_chemistry.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace detail {

constexpr double kChemistryTestScale = 0.02;
constexpr size_t kChemistryRecordWords = primordial::kAbundanceCount + 7;
constexpr size_t kChemistryEnergyWord = primordial::kAbundanceCount + 1;
constexpr size_t kChemistryIntervalWord = primordial::kAbundanceCount + 4;
constexpr size_t kChemistryHeatingWord = primordial::kAbundanceCount + 5;
constexpr size_t kChemistryRungWord = primordial::kAbundanceCount + 6;
using ChemistrySnapshot = std::vector<std::vector<uint64_t>>;

double chemistryTestTimestep() {
    // At rung three, each source half evolves for 1e11 physical seconds.
    return 1.6e12 / (kChemistryTestScale * getOptions().units.timeSeconds);
}

void requireChemistry(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(std::string("SPH chemistry timestep regression: ") + message);
    }
}

void initializeChemistryFixture(int activeRung, bool invalid) {
    particlesResize(3);
    sphParticlesResize(3, false);
    auto const& options = getOptions();
    auto const density = units::ProperDensity(units::Density::from_value(
        units::kHydrogenMass.value() / (1.0 - options.cosmology.heliumFraction)));
    float const storedDensity = units::toStoredDensity(density, units::scalesFromOptions(options),
        units::CosmologicalContext(kChemistryTestScale, options.cosmology.hubble))
                                    .value();
    for (ParticleInteger i = 0; i < 3; ++i) {
        particlesType(i) = kSphType;
        particlesCatIndex(i) = i;
        sphParticlesParticleIndex(i) = i;
        particlesRung(i) = i == 0 ? 1 : (i == 1 ? activeRung : 4);
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, i) = 0.25 + 0.125 * i;
            particlesVel(dim, i) = 0.0f;
        }
        sphParticlesDensity(i) = storedDensity;
        sphParticlesSupportRadius(i) = 0.1f;
        sphChemistryInitializeParticle(i, kChemistryTestScale);
        auto const eos = sphChemistryEos(
            sphParticlesChemistryStates[i], sphParticlesInternalEnergy(i), kChemistryTestScale);
        sphParticlesChemistryEos[i] = {eos.pressureEnergyRatio, eos.adiabaticIndex};
    }
    // An inactive coarse particle already has an opening source half pending.
    sphParticlesChemistrySteps[0] = {std::ldexp(chemistryTestTimestep(), -2), 0.125};
    if (invalid && hpxRank() == hpxSize() - 1) {
        // Reject the last sink on the last rank after other sinks have prepared.
        sphParticlesChemistryStates[2].abundances[1] = 2.0f;
    }
}

std::vector<uint64_t> chemistryFixtureSnapshot() {
    std::vector<uint64_t> words;
    for (ParticleInteger i = 0; i < 3; ++i) {
        auto const& state = sphParticlesChemistryStates[i];
        for (int species = 0; species < primordial::kAbundanceCount; ++species) {
            words.push_back(std::bit_cast<uint32_t>(state.abundances[species]));
        }
        words.push_back(std::bit_cast<uint32_t>(state.thermalEnergy));
        words.push_back(std::bit_cast<uint32_t>(sphParticlesInternalEnergy(i)));
        words.push_back(std::bit_cast<uint32_t>(sphParticlesChemistryEos[i].first));
        words.push_back(std::bit_cast<uint32_t>(sphParticlesChemistryEos[i].second));
        words.push_back(std::bit_cast<uint64_t>(sphParticlesChemistrySteps[i].halfInterval));
        words.push_back(std::bit_cast<uint64_t>(sphParticlesChemistrySteps[i].pendingHeating));
        words.push_back(uint64_t(sphParticlesRung(i)));
    }
    return words;
}

void changeChemistryFixtureRung(int rung) {
    sphParticlesRung(1) = rung;
}

void clearChemistryFixtureSteps() {
    for (ParticleInteger i = 0; i < 3; ++i) {
        sphParticlesChemistrySteps[i] = {};
    }
}

void roundtripChemistryFixture() {
    FILE* file = std::tmpfile();
    requireChemistry(file != nullptr, "cannot open checkpoint fixture");
    sphParticlesSave(file);
    for (ParticleInteger i = 0; i < 3; ++i) {
        sphParticlesChemistrySteps[i] = {};
    }
    std::rewind(file);
    sphParticlesLoad(file);
    std::fclose(file);
}

}    // namespace detail

HPX_PLAIN_ACTION(detail::initializeChemistryFixture, InitializeChemistryFixtureAction);
HPX_PLAIN_ACTION(detail::chemistryFixtureSnapshot, ChemistryFixtureSnapshotAction);
HPX_PLAIN_ACTION(detail::changeChemistryFixtureRung, ChangeChemistryFixtureRungAction);
HPX_PLAIN_ACTION(detail::clearChemistryFixtureSteps, ClearChemistryFixtureStepsAction);
HPX_PLAIN_ACTION(detail::roundtripChemistryFixture, RoundtripChemistryFixtureAction);

namespace detail {

template <typename Action, typename... Arguments>
void onChemistryLocalities(Arguments const&... arguments) {
    std::vector<hpx::future<void>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<Action>(locality, arguments...));
    }
    for (auto& future : pending) {
        future.get();
    }
}

ChemistrySnapshot chemistrySnapshot() {
    std::vector<hpx::future<std::vector<uint64_t>>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<ChemistryFixtureSnapshotAction>(locality));
    }
    ChemistrySnapshot result;
    for (auto& future : pending) {
        result.push_back(future.get());
    }
    return result;
}

void requireInactiveUnchanged(ChemistrySnapshot const& before, ChemistrySnapshot const& after) {
    for (size_t rank = 0; rank < before.size(); ++rank) {
        requireChemistry(std::equal(before[rank].begin(),
                             before[rank].begin() + kChemistryRecordWords, after[rank].begin()),
            "inactive particle changed");
    }
}

void requireChemistryAdvanced(
    ChemistrySnapshot const& before, ChemistrySnapshot const& after, size_t firstParticle = 1) {
    for (size_t rank = 0; rank < before.size(); ++rank) {
        for (size_t particle = firstParticle; particle < 3; ++particle) {
            size_t const offset = particle * kChemistryRecordWords;
            requireChemistry(before[rank][offset + 1] != after[rank][offset + 1],
                "opening chemistry did not change the hydrogen abundance");
            requireChemistry(before[rank][offset + kChemistryEnergyWord] !=
                    after[rank][offset + kChemistryEnergyWord],
                "opening chemistry did not change the thermal energy");
        }
    }
}

void requireClosingHeating(ChemistrySnapshot const& opened, ChemistrySnapshot const& closed,
    double heating, size_t firstParticle = 1) {
    double expectedHeating = 0.0;
    for (size_t rank = 0; rank < closed.size(); ++rank) {
        for (size_t particle = firstParticle; particle < 3; ++particle) {
            size_t const offset = particle * kChemistryRecordWords;
            float const before =
                std::bit_cast<float>(uint32_t(opened[rank][offset + kChemistryEnergyWord]));
            float const after =
                std::bit_cast<float>(uint32_t(closed[rank][offset + kChemistryEnergyWord]));
            expectedHeating += std::bit_cast<double>(opened[rank][offset + kChemistryHeatingWord]) -
                getOptions().initial.gasMass * double(after - before) / kChemistryTestScale;
            requireChemistry(closed[rank][offset + kChemistryIntervalWord] == 0 &&
                    closed[rank][offset + kChemistryHeatingWord] == 0,
                "closing did not consume both pending interval and heating");
        }
    }
    requireChemistry(
        std::abs(heating - expectedHeating) <= 1.0e-12 * std::max(1.0, std::abs(expectedHeating)),
        "closing did not report opening plus closing heating exactly once");
}

}    // namespace detail

void sphChemistryTimestepsTest() {
    using namespace detail;
    auto const& options = getOptions();
    requireChemistry(options.hydro.enabled && options.chemistry.enabled && !options.compute.cuda,
        "fixture requires CPU SPH chemistry");
    requireChemistry(options.chemistry.network == "atomic6" &&
            options.chemistry.initialization == "manual" &&
            options.chemistry.initialTemperature == 1000.0,
        "fixture requires manual atomic chemistry at 1000 K");
    double const baseTimestep = chemistryTestTimestep();
    onChemistryLocalities<InitializeChemistryFixtureAction>(3, false);
    auto const initial = chemistrySnapshot();
    sphChemistryOpen(3, kChemistryTestScale, 0.0, baseTimestep);
    auto const mixed = chemistrySnapshot();
    requireInactiveUnchanged(initial, mixed);
    requireChemistryAdvanced(initial, mixed);
    for (size_t rank = 0; rank < mixed.size(); ++rank) {
        for (size_t particle = 1; particle < 3; ++particle) {
            size_t const offset = particle * kChemistryRecordWords;
            int const selectedRung = int(initial[rank][offset + kChemistryRungWord]);
            requireChemistry(mixed[rank][offset + kChemistryRungWord] == uint64_t(selectedRung),
                "opening chemistry changed a preselected SPH rung");
            requireChemistry(std::bit_cast<double>(mixed[rank][offset + kChemistryIntervalWord]) ==
                    std::ldexp(baseTimestep, -1 - selectedRung),
                "opening chemistry changed a preselected SPH half interval");
        }
        requireChemistry(
            mixed[rank][kChemistryRecordWords + 1] != mixed[rank][2 * kChemistryRecordWords + 1],
            "different source intervals did not advance distinct abundances");
    }
    onChemistryLocalities<RoundtripChemistryFixtureAction>();
    requireChemistry(mixed == chemistrySnapshot(), "checkpoint lost pending intervals or heating");

    onChemistryLocalities<ChangeChemistryFixtureRungAction>(6);
    auto const mismatched = chemistrySnapshot();
    bool rejected = false;
    try {
        sphChemistryClose(3, kChemistryTestScale, 0.0, baseTimestep);
    } catch (std::exception const&) {
        rejected = true;
    }
    requireChemistry(rejected, "closing accepted a rung changed before the old source half");
    requireChemistry(mismatched == chemistrySnapshot(), "rejected close partially committed");
    onChemistryLocalities<ChangeChemistryFixtureRungAction>(3);
    double const heating = sphChemistryClose(3, kChemistryTestScale, 0.0, baseTimestep);
    auto const closed = chemistrySnapshot();
    requireInactiveUnchanged(mixed, closed);
    requireClosingHeating(mixed, closed, heating);

    onChemistryLocalities<InitializeChemistryFixtureAction>(3, false);
    onChemistryLocalities<ClearChemistryFixtureStepsAction>();
    auto const uniformInitial = chemistrySnapshot();
    // PMMM has already selected the common rung before the opening source half.
    sphChemistryOpen(0, kChemistryTestScale, 0.0, baseTimestep, -1.0, 6);
    auto const uniform = chemistrySnapshot();
    requireChemistryAdvanced(uniformInitial, uniform, 0);
    for (auto const& rank : uniform) {
        for (size_t particle = 0; particle < 3; ++particle) {
            size_t const offset = particle * kChemistryRecordWords;
            requireChemistry(rank[offset + kChemistryRungWord] == 6 &&
                    std::bit_cast<double>(rank[offset + kChemistryIntervalWord]) ==
                        std::ldexp(baseTimestep, -7),
                "uniform opening changed the selected common rung or interval");
        }
    }

    // A global step can use an interval that is not represented by a particle rung.
    onChemistryLocalities<InitializeChemistryFixtureAction>(3, false);
    onChemistryLocalities<ClearChemistryFixtureStepsAction>();
    auto const explicitInitial = chemistrySnapshot();
    double const explicitTimestep = baseTimestep / 37.0;
    sphChemistryOpen(0, kChemistryTestScale, 0.0, baseTimestep, explicitTimestep);
    auto const explicitOpened = chemistrySnapshot();
    requireChemistryAdvanced(explicitInitial, explicitOpened, 0);
    for (size_t rank = 0; rank < explicitOpened.size(); ++rank) {
        for (size_t particle = 0; particle < 3; ++particle) {
            size_t const offset = particle * kChemistryRecordWords;
            requireChemistry(explicitOpened[rank][offset + kChemistryRungWord] ==
                        explicitInitial[rank][offset + kChemistryRungWord] &&
                    std::bit_cast<double>(explicitOpened[rank][offset + kChemistryIntervalWord]) ==
                        0.5 * explicitTimestep,
                "explicit opening changed the preselected interval or particle rungs");
        }
    }
    onChemistryLocalities<RoundtripChemistryFixtureAction>();
    requireChemistry(explicitOpened == chemistrySnapshot(),
        "checkpoint lost the explicit interval or pending heating");
    rejected = false;
    try {
        sphChemistryClose(0, kChemistryTestScale, 0.0, baseTimestep);
    } catch (std::exception const&) {
        rejected = true;
    }
    requireChemistry(rejected, "closing an explicit interval accepted a rung-based timestep");
    requireChemistry(explicitOpened == chemistrySnapshot(),
        "rejected explicit-interval close partially committed");
    double const explicitHeating =
        sphChemistryClose(0, kChemistryTestScale, 0.0, baseTimestep, explicitTimestep);
    requireClosingHeating(explicitOpened, chemistrySnapshot(), explicitHeating, 0);

    onChemistryLocalities<InitializeChemistryFixtureAction>(3, true);
    auto const invalid = chemistrySnapshot();
    rejected = false;
    try {
        sphChemistryOpen(3, kChemistryTestScale, 0.0, baseTimestep);
    } catch (std::exception const&) {
        rejected = true;
    }
    requireChemistry(rejected, "invalid source state did not reject the batch");
    requireChemistry(invalid == chemistrySnapshot(), "failed source batch partially committed");

    onChemistryLocalities<ClearChemistryFixtureStepsAction>();
    auto const invalidUniform = chemistrySnapshot();
    rejected = false;
    try {
        sphChemistryOpen(0, kChemistryTestScale, 0.0, baseTimestep, -1.0, 6);
    } catch (std::exception const&) {
        rejected = true;
    }
    requireChemistry(rejected, "invalid source state did not reject the uniform batch");
    requireChemistry(invalidUniform == chemistrySnapshot(),
        "failed uniform source batch changed chemistry, intervals or rungs");
    onChemistryLocalities<InitializeChemistryFixtureAction>(3, false);
    onChemistryLocalities<ClearChemistryFixtureStepsAction>();
    sphChemistryOpen(0, kChemistryTestScale, 0.0, baseTimestep, -1.0, 6);
    requireChemistry(uniform == chemistrySnapshot(),
        "opening after a failed batch retained state from the rejected source solve");
    PRINT("SPH chemistry timesteps passed: selected mixed and uniform rungs, explicit intervals, "
          "inactive state, checkpoint, old-step closure, heating and failed-batch atomicity "
          "(%i localities)\n",
        hpxSize());
}
