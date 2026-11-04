/* CosmicTiger - completed-step equilibrium cooling regression; BSL-1.0. */
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cosmictiger/chemistry/sph.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/math/simulation_units.hpp>
#include <cosmictiger/sph/particles.hpp>

namespace detail {
constexpr double kChemistryTestScale = 0.02;
struct ChemistryRecord {
    uint32_t electronFractionBits;
    uint32_t energyBits;
    uint32_t entropyBits;
    uint32_t densityBits;
    uint64_t rung;

    bool operator==(ChemistryRecord const&) const = default;

    template <typename Archive>
    void serialize(Archive& archive, unsigned) {
        archive & electronFractionBits & energyBits & entropyBits & densityBits & rung;
    }
};
using ChemistryLocalSnapshot = std::vector<ChemistryRecord>;
using ChemistrySnapshot = std::vector<ChemistryLocalSnapshot>;

double chemistryTestTimestep() {
    // Rung three completes 1e11 physical seconds.
    return 8.0e11 / (kChemistryTestScale * getOptions().units.timeSeconds);
}
void requireChemistry(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(std::string("SPH cooling timestep regression: ") + message);
    }
}
void initializeChemistryFixture(bool invalid) {
    particlesResize(3);
    sphParticlesResize(3, false);
    sphParticlesEntropyInitialized = false;
    sphParticlesThermalScaleFactor = kChemistryTestScale;
    auto const& options = getOptions();
    auto const scales = units::scalesFromOptions(options);
    units::CosmologicalContext const context(kChemistryTestScale, options.cosmology.hubble);
    float const density = units::toStoredDensity(
        units::ProperDensity(
            units::Density::from_value(gadget4::kProtonMass / gadget4::kHydrogenMassFraction)),
        scales, context)
                              .value();
    float const energy = units::toStoredSpecificEnergy(
        units::PhysicalSpecificEnergy(units::SpecificEnergy::from_value(1.0e14)), scales, context)
                             .value();
    for (ParticleInteger i = 0; i < 3; ++i) {
        particlesType(i) = kSphType;
        particlesCatIndex(i) = i;
        sphParticlesParticleIndex(i) = i;
        particlesRung(i) = i == 0 ? 1 : (i == 1 ? 3 : 4);
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesPos(dim, i) = 0.25 + 0.125 * i;
            particlesVel(dim, i) = 0.0f;
        }
        sphParticlesDensity(i) = density;
        sphParticlesSetInternalEnergy(i, energy);
        sphParticlesSupportRadius(i) = 0.1f;
        sphChemistryInitializeParticle(i, kChemistryTestScale);
    }
    if (invalid && hpxRank() == hpxSize() - 1) {
        // Last sink on the last rank fails after other ranks can prepare.
        sphParticlesElectronFractions[2] = std::numeric_limits<float>::quiet_NaN();
    }
}
ChemistryLocalSnapshot chemistryFixtureSnapshot() {
    requireChemistry(sphParticlesViscosityDiagnostics == nullptr,
        "temporary viscosity diagnostics survived the cooling source");
    ChemistryLocalSnapshot records;
    for (ParticleInteger i = 0; i < 3; ++i) {
        records.push_back({std::bit_cast<uint32_t>(sphParticlesElectronFractions[i]),
            std::bit_cast<uint32_t>(sphParticlesInternalEnergy(i)),
            std::bit_cast<uint32_t>(sphParticlesEntropy(i)),
            std::bit_cast<uint32_t>(sphParticlesDensity(i)), uint64_t(sphParticlesRung(i))});
    }
    return records;
}

void roundtripChemistryFixture() {
    FILE* file = std::tmpfile();
    requireChemistry(file != nullptr, "cannot open checkpoint fixture");
    sphParticlesSave(file);
    for (ParticleInteger i = 0; i < 3; ++i) {
        sphParticlesElectronFractions[i] = 0.0f;
    }
    std::rewind(file);
    sphParticlesLoad(file);
    std::fclose(file);
}
void commitChemistryAgain() {
    sphCoolingCommitLocal();
}
void rejectDuplicateCoolingTargets() {
    auto const before = chemistryFixtureSnapshot();
    bool rejected = false;
    try {
        double const interval = chemistryTestTimestep() / 8.0;
        sphCoolingPrepareLocal({1, 1}, {interval, interval}, kChemistryTestScale);
    } catch (std::invalid_argument const&) {
        rejected = true;
    }
    requireChemistry(rejected, "duplicate gas slots did not reject the source batch");
    sphCoolingCommitLocal();
    requireChemistry(before == chemistryFixtureSnapshot(), "duplicate source changed gas state");
}
void checkThermalProjection() {
    double const gamma = getOptions().hydro.gamma;
    auto const prepared = sph_entropy::prepareStoredEnergy(3.0f, 7.0f, gamma, kChemistryTestScale);
    double const expectedEntropy = (gamma - 1.0) * 3.0 * std::pow(7.0, 1.0 - gamma);
    requireChemistry(std::abs(double(prepared.entropy) - expectedEntropy) <
            2.0 * std::numeric_limits<float>::epsilon() * expectedEntropy,
        "source projection does not produce canonical entropy");
    for (auto const& input :
        std::vector<std::pair<float, float>>{{std::numeric_limits<float>::quiet_NaN(), 1.0f},
            {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()},
            {std::numeric_limits<float>::min(), std::numeric_limits<float>::max()}}) {
        bool rejected = false;
        try {
            sph_entropy::prepareStoredEnergy(input.first, input.second, gamma, kChemistryTestScale);
        } catch (std::exception const&) {
            rejected = true;
        }
        requireChemistry(rejected, "invalid or unrepresentable source entropy was accepted");
    }
}
}    // namespace detail

HPX_PLAIN_ACTION(detail::initializeChemistryFixture, InitializeChemistryFixtureAction);
HPX_PLAIN_ACTION(detail::chemistryFixtureSnapshot, ChemistryFixtureSnapshotAction);
HPX_PLAIN_ACTION(detail::roundtripChemistryFixture, RoundtripChemistryFixtureAction);
HPX_PLAIN_ACTION(detail::commitChemistryAgain, CommitChemistryAgainAction);
HPX_PLAIN_ACTION(detail::rejectDuplicateCoolingTargets, RejectDuplicateCoolingTargetsAction);

namespace detail {
template <typename Action, typename... Arguments>
void onChemistryLocalities(Arguments const&... arguments) {
    std::vector<hpx::future<void>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<Action>(locality, arguments...));
    }
    hpx::wait_all(pending);
    for (auto& future : pending) {
        future.get();
    }
}
ChemistrySnapshot chemistrySnapshot() {
    std::vector<hpx::future<ChemistryLocalSnapshot>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<ChemistryFixtureSnapshotAction>(locality));
    }
    ChemistrySnapshot result;
    for (auto& future : pending) {
        result.push_back(future.get());
    }
    return result;
}
void requireCompletedSource(ChemistrySnapshot const& before, ChemistrySnapshot const& after,
    double heating, size_t firstParticle) {
    double expectedHeating = 0.0;
    for (size_t rank = 0; rank < before.size(); ++rank) {
        requireChemistry(std::equal(before[rank].begin(), before[rank].begin() + firstParticle,
                             after[rank].begin()),
            "inactive particle changed");
        for (size_t particle = firstParticle; particle < 3; ++particle) {
            auto const& oldRecord = before[rank][particle];
            auto const& newRecord = after[rank][particle];
            requireChemistry(oldRecord.energyBits != newRecord.energyBits,
                "completed cooling did not change energy");
            requireChemistry(oldRecord.entropyBits != newRecord.entropyBits,
                "completed cooling did not change entropy");
            requireChemistry(oldRecord.densityBits == newRecord.densityBits,
                "cooling changed the endpoint density");
            requireChemistry(oldRecord.rung == newRecord.rung, "cooling reselected a hydro rung");
            float const oldEnergy = std::bit_cast<float>(oldRecord.energyBits);
            float const newEnergy = std::bit_cast<float>(newRecord.energyBits);
            double const density = std::bit_cast<float>(newRecord.densityBits);
            double const entropy = std::bit_cast<float>(newRecord.entropyBits);
            double const gamma = getOptions().hydro.gamma;
            double const expectedEntropy =
                (gamma - 1.0) * double(newEnergy) * std::pow(density, 1.0 - gamma);
            requireChemistry(std::abs(entropy - expectedEntropy) <=
                    4.0 * std::numeric_limits<float>::epsilon() * expectedEntropy,
                "committed source energy and entropy are inconsistent");
            expectedHeating -= getOptions().initial.gasMass * (double(newEnergy) - oldEnergy) /
                kChemistryTestScale;
        }
    }
    requireChemistry(
        std::abs(heating - expectedHeating) <= 1.0e-12 * std::max(1.0, std::abs(expectedHeating)),
        "completed source energy ledger differs from the committed change");
}
}    // namespace detail

void sphChemistryTimestepsTest() {
    using namespace detail;
    auto const& options = getOptions();
    requireChemistry(options.hydro.enabled && options.chemistry.enabled && !options.chemistry.cuda,
        "fixture requires CPU SPH cooling");
    double const baseTimestep = chemistryTestTimestep();
    checkThermalProjection();
    onChemistryLocalities<InitializeChemistryFixtureAction>(false);
    sphParticlesPrepareThermodynamics(kChemistryTestScale, true);
    auto const initial = chemistrySnapshot();
    double const heating = sphChemistryClose(3, kChemistryTestScale, 0.0, baseTimestep);
    auto const mixed = chemistrySnapshot();
    requireCompletedSource(initial, mixed, heating, 1);
    onChemistryLocalities<CommitChemistryAgainAction>();
    requireChemistry(mixed == chemistrySnapshot(), "consumed cooling results were applied twice");
    onChemistryLocalities<RejectDuplicateCoolingTargetsAction>();
    for (auto const& rank : mixed) {
        requireChemistry(rank[1].energyBits != rank[2].energyBits,
            "different completed intervals produced identical cooling");
    }
    onChemistryLocalities<RoundtripChemistryFixtureAction>();
    requireChemistry(
        mixed == chemistrySnapshot(), "checkpoint lost the saved electron guess or energy");

    onChemistryLocalities<InitializeChemistryFixtureAction>(false);
    sphParticlesPrepareThermodynamics(kChemistryTestScale, true);
    auto const globalInitial = chemistrySnapshot();
    double const explicitTimestep = baseTimestep / 37.0;
    double const globalHeating =
        sphChemistryClose(0, kChemistryTestScale, 0.0, baseTimestep, explicitTimestep);
    auto const global = chemistrySnapshot();
    requireCompletedSource(globalInitial, global, globalHeating, 0);
    for (auto const& rank : global) {
        for (size_t particle = 1; particle < 3; ++particle) {
            requireChemistry(rank[0].energyBits == rank[particle].energyBits,
                "explicit global interval incorrectly depended on particle rung");
        }
    }
    onChemistryLocalities<InitializeChemistryFixtureAction>(true);
    sphParticlesPrepareThermodynamics(kChemistryTestScale, true);
    auto const invalid = chemistrySnapshot();
    bool rejected = false;
    try {
        sphChemistryClose(3, kChemistryTestScale, 0.0, baseTimestep);
    } catch (std::exception const&) {
        rejected = true;
    }
    requireChemistry(rejected, "invalid source state did not reject the batch");
    requireChemistry(
        invalid == chemistrySnapshot(), "failed distributed source partially committed");
    onChemistryLocalities<CommitChemistryAgainAction>();
    requireChemistry(
        invalid == chemistrySnapshot(), "failed distributed source left committable results");
    onChemistryLocalities<InitializeChemistryFixtureAction>(false);
    sphParticlesPrepareThermodynamics(kChemistryTestScale, true);
    sphChemistryClose(3, kChemistryTestScale, 0.0, baseTimestep);
    requireChemistry(
        mixed == chemistrySnapshot(), "failed source left stale results in a subsequent batch");
    std::cout
        << std::format(
               "SPH chemistry timesteps passed: completed mixed/global intervals, inactive state, "
               "canonical entropy, checkpoint, energy ledger, consumed commits and failed-batch "
               "atomicity ({:d} localities)\n",
               hpxSize())
        << std::flush;
}
