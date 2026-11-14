/* CosmicTiger - completed midpoint source interval regression; BSL-1.0. */

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/tree.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>
#include <cosmictiger/stars/star_formation.hpp>

namespace detail {

constexpr int kSourceGridSide = 4;
constexpr int kSourceParticleCount = kSourceGridSide * kSourceGridSide * kSourceGridSide;
// Deliberately not exactly representable by the float hydro kernel parameters.
constexpr double kSourceScale = 0.3;

struct SourceTimingRecord {
    uint64_t identity{};
    uint64_t eventCounter{};
    double elapsedPhysicalTime{};
    double formationTime{};
    int rung{};
    int type{};
    uint32_t mass{};
    uint32_t entropy{};
    uint32_t bulkViscosity{};
    std::array<uint64_t, kDimensions> position{};
    std::array<uint32_t, kDimensions> velocity{};
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & identity & eventCounter & elapsedPhysicalTime & formationTime & rung & type &
            mass & entropy & bulkViscosity & position & velocity;
    }
};

void requireSourceTiming(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(std::string("SPH source timing regression: ") + message);
    }
}

void initializeSourceTiming() {
    auto const& opts = getOptions();
    sphParticlesFree();
    ParticleInteger const begin = kSourceParticleCount * hpxRank() / hpxSize();
    ParticleInteger const end = kSourceParticleCount * (hpxRank() + 1) / hpxSize();
    particlesResize(end - begin);
    sphParticlesResize(end - begin, false);
    sphParticlesEntropyInitialized = true;
    sphParticlesThermalScaleFactor = kSourceScale;
    auto const parameters = starFormationParameters(kSourceScale);
    float const storedEnergy = float(1.0e12 * kSourceScale * kSourceScale / parameters.energyScale);
    for (ParticleInteger particle = 0; particle < end - begin; ++particle) {
        int const ordinal = begin + particle;
        particlesType(particle) = kSphType;
        particlesCatIndex(particle) = particle;
        particlesRung(particle) = ordinal % 2 == 0 ? 1 : 2;
        auto& source = particlesStarFormationStates[particle];
        source = {};
        source.id = uint64_t(ordinal) + 1;
        sphParticlesParticleIndex(particle) = particle;
        sphParticlesDensity(particle) = opts.initial.gasMass * kSourceParticleCount;
        sphParticlesSetInternalEnergy(particle, storedEnergy);
        sphParticlesSmoothingLength(particle) = opts.hydro.minSmoothingLength;
        sphParticlesElectronFractions[particle] = 1.0f;
        sphParticlesSetProposedRung(particle, particlesRung(particle));
        sphParticlesBulkViscosityCoefficients[particle] = 0.25f + 0.0625f * (ordinal % 4);
        int coordinate = ordinal;
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            particlesPos(dimension, particle) =
                (double(coordinate % kSourceGridSide) + 0.5) / kSourceGridSide;
            coordinate /= kSourceGridSide;
            // Exactly balanced equal-mass momentum, with nontrivial velocity
            // gradients so a mistaken diagnostic viscosity commit is visible.
            particlesVel(dimension, particle) =
                float((ordinal % 2 == 0 ? -1 : 1) * (dimension + 1)) * 0.03125f;
        }
    }
}

std::vector<SourceTimingRecord> sourceTimingSnapshot() {
    requireSourceTiming(sphParticlesEntropyInitialized, "entropy was not canonical");
    requireSourceTiming(sphParticlesThermalScaleFactor == kSourceScale,
        "source estimate lost the exact endpoint scale");
    std::vector<SourceTimingRecord> records;
    for (ParticleInteger particle = 0; particle < particlesSize(); ++particle) {
        auto const& source = particlesStarFormationStates[particle];
        SourceTimingRecord record;
        record.identity = source.id;
        record.eventCounter = source.eventCounter;
        record.elapsedPhysicalTime = source.elapsedPhysicalTime;
        record.formationTime = source.formationTime;
        record.rung = particlesRung(particle);
        record.type = particlesType(particle);
        record.mass = std::bit_cast<uint32_t>(particlesMass(particle));
        if (record.type == kSphType) {
            auto const gas = particlesCatIndex(particle);
            record.entropy = std::bit_cast<uint32_t>(sphParticlesEntropies[gas]);
            record.bulkViscosity = uint32_t(sphParticlesBulkViscosityCoefficients[gas].bits);
        }
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            record.position[dimension] = uint64_t(particlesPos(dimension, particle).raw());
            record.velocity[dimension] = std::bit_cast<uint32_t>(particlesVel(dimension, particle));
        }
        records.push_back(record);
    }
    return records;
}

void setSourceTimingRates(uint64_t forcedIdentity) {
    for (ParticleInteger particle = 0; particle < particlesSize(); ++particle) {
        auto& source = particlesStarFormationStates[particle];
        source.specificStarFormationRate = source.id == forcedIdentity ? 1.0 : 0.0;
    }
}

}    // namespace detail

HPX_PLAIN_ACTION(detail::initializeSourceTiming, InitializeSourceTimingAction);
HPX_PLAIN_ACTION(detail::sourceTimingSnapshot, SourceTimingSnapshotAction);
HPX_PLAIN_ACTION(detail::setSourceTimingRates, SetSourceTimingRatesAction);

namespace detail {

template <class Action, class... Arguments>
void sourceTimingAll(Arguments const&... arguments) {
    std::vector<hpx::future<void>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<Action>(locality, arguments...));
    }
    for (auto& future : pending) {
        future.get();
    }
}

std::vector<SourceTimingRecord> collectSourceTiming() {
    std::vector<hpx::future<std::vector<SourceTimingRecord>>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<SourceTimingSnapshotAction>(locality));
    }
    std::vector<SourceTimingRecord> result;
    for (auto& future : pending) {
        auto const records = future.get();
        result.insert(result.end(), records.begin(), records.end());
    }
    std::sort(result.begin(), result.end(),
        [](auto const& left, auto const& right) { return left.identity < right.identity; });
    requireSourceTiming(result.size() == kSourceParticleCount, "source fixture lost particles");
    return result;
}

void verifySourceTiming(std::vector<SourceTimingRecord> const& before,
    std::vector<SourceTimingRecord> const& after, int minimumRung, double baseTimestep,
    double explicitTimestep, uint64_t convertedIdentity) {
    auto const& opts = getOptions();
    std::array<double, kDimensions> momentumBefore{}, momentumAfter{};
    bool const comoving = starFormationParameters(kSourceScale).cooling.comoving;
    for (size_t index = 0; index < before.size(); ++index) {
        auto const& previous = before[index];
        auto const& current = after[index];
        requireSourceTiming(previous.identity == current.identity &&
                previous.position == current.position && previous.velocity == current.velocity &&
                previous.mass == current.mass && previous.rung == current.rung,
            "source or zero-kick estimate changed mechanical state or a live rung");
        bool const selected = previous.type == kSphType && previous.rung >= minimumRung;
        double const duration =
            explicitTimestep >= 0.0 ? explicitTimestep : std::ldexp(baseTimestep, -previous.rung);
        double const expectedElapsed = selected ?
            previous.elapsedPhysicalTime + kSourceScale * duration * opts.units.timeSeconds :
            previous.elapsedPhysicalTime;
        requireSourceTiming(current.eventCounter == previous.eventCounter + uint64_t(selected) &&
                current.elapsedPhysicalTime == expectedElapsed,
            "source did not consume exactly one completed old-rung interval");
        if (selected && previous.identity == convertedIdentity) {
            requireSourceTiming(current.type == kStarType &&
                    current.formationTime == (comoving ? kSourceScale : expectedElapsed),
                "forced conversion did not wait for its own close or record its birth time");
        } else {
            requireSourceTiming(current.type == previous.type,
                "source converted an inactive particle or one with zero saved rate");
        }
        if (current.type == kSphType) {
            if (!selected) {
                requireSourceTiming(current.bulkViscosity == previous.bulkViscosity,
                    "endpoint estimate changed inactive viscosity");
            } else {
                requireSourceTiming((current.bulkViscosity & 0x7c00U) != 0x7c00U &&
                        (current.bulkViscosity & 0x8000U) == 0U,
                    "endpoint viscosity coefficient is not finite");
                if (previous.eventCounter == 0) {
                    requireSourceTiming(current.bulkViscosity != previous.bulkViscosity,
                        "M&M endpoint failed to evolve the coefficient over its completed "
                        "interval");
                }
            }
            if (!selected) {
                requireSourceTiming(current.entropy == previous.entropy,
                    "inactive gas received source heat or a hydro kick");
            }
        }
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            momentumBefore[dimension] += double(std::bit_cast<float>(previous.mass)) *
                std::bit_cast<float>(previous.velocity[dimension]);
            momentumAfter[dimension] += double(std::bit_cast<float>(current.mass)) *
                std::bit_cast<float>(current.velocity[dimension]);
        }
    }
    requireSourceTiming(momentumBefore == momentumAfter,
        "source estimate disturbed the reciprocal-kick momentum ledger");
}

}    // namespace detail

void sphSourceTimingTest() {
    auto const savedOptions = getOptions();
    detail::requireSourceTiming(savedOptions.hydro.enabled && savedOptions.starFormation.enabled &&
            savedOptions.chemistry.enabled,
        "fixture requires SPH, cooling, and star formation");
    auto options = savedOptions;
    options.initial.numberOfParticles = detail::kSourceParticleCount;
    options.compute.sphBucketSize = 8;
    options.hydro.minSmoothingLength = options.hydro.maxSmoothingLength = 0.45f;
    options.hydro.damping = 0.0;
    options.hydro.yAcceleration = 0.0;
    options.scenario.rayleighTaylor.zPlaneAcceleration = 0.0;
    options.scenario.disc.centralAcceleration = 0.0;
    double const baseTimestep = 1.0e8 / (detail::kSourceScale * options.units.timeSeconds);
    sphTreeDestroy(true);
    setOptions(options);
    // Give the fine interval one decay time, so an endpoint update is
    // visible in half storage even with cosmological code units.
    auto const sourceParameters = starFormationParameters(detail::kSourceScale);
    double const soundSpeed =
        std::sqrt(options.hydro.heatCapacityRatio * (options.hydro.heatCapacityRatio - 1.0) *
            1.0e12 * detail::kSourceScale * detail::kSourceScale / sourceParameters.energyScale);
    double const smoothingLength = options.hydro.minSmoothingLength /
        selectedHydroKernel().smoothingLengthPerEffectiveRadius * monaghan::kViscosityWidthScale;
    options.hydro.viscosityDecay =
        detail::kSourceScale * smoothingLength / (soundSpeed * (baseTimestep / 4.0));
    setOptions(options);
    detail::sourceTimingAll<InitializeSourceTimingAction>();
    domainsRebound();
    EnergiesT energies;
    sphMidpointStep(-1, 0.0, -1.0, baseTimestep, detail::kSourceScale, 0.0, energies);
    sphParticlesPrepareThermodynamics(detail::kSourceScale, false);
    auto const beforeMidpoint = detail::collectSourceTiming();
    sphMidpointStep(2, baseTimestep / 8.0, baseTimestep / 4.0, baseTimestep, detail::kSourceScale,
        0.0, energies);
    // Hydro kernels use a float scale. Restore the fixture's double context
    // before snapshots; this preparation does not update viscosity.
    sphParticlesPrepareThermodynamics(detail::kSourceScale, false);
    auto const afterMidpoint = detail::collectSourceTiming();
    for (size_t index = 0; index < beforeMidpoint.size(); ++index) {
        detail::requireSourceTiming(
            afterMidpoint[index].bulkViscosity == beforeMidpoint[index].bulkViscosity,
            "hydro midpoint advanced endpoint-only or inactive viscosity");
    }
    for (int event = 1; event <= 4; ++event) {
        // Arm a coarse particle after its first close. The intervening fine
        // close must leave it gas; only its next own close may convert it.
        uint64_t const forcedIdentity = event >= 3 ? 1 : 0;
        detail::sourceTimingAll<SetSourceTimingRatesAction>(forcedIdentity);
        auto const before = detail::collectSourceTiming();
        int const minimumRung = event % 2 == 0 ? 1 : 2;
        auto const estimate = sphCompleteSources(minimumRung, event * baseTimestep / 4.0,
            baseTimestep, detail::kSourceScale, 0.0, energies);
        detail::requireSourceTiming(
            std::isfinite(estimate.minimumTimestep) && estimate.minimumTimestep > 0.0,
            "source estimate did not return a usable hydro bound");
        auto const after = detail::collectSourceTiming();
        detail::verifySourceTiming(
            before, after, minimumRung, baseTimestep, -1.0, event == 4 ? forcedIdentity : 0);
        sphMidpointStep(-1, event * baseTimestep / 4.0, -1.0, baseTimestep, detail::kSourceScale,
            0.0, energies);
        sphParticlesPrepareThermodynamics(detail::kSourceScale, false);
        auto const diagnostic = detail::collectSourceTiming();
        for (size_t index = 0; index < after.size(); ++index) {
            detail::requireSourceTiming(
                diagnostic[index].bulkViscosity == after[index].bulkViscosity,
                "diagnostic density refresh replayed an endpoint viscosity update");
        }
    }
    detail::sourceTimingAll<SetSourceTimingRatesAction>(uint64_t{0});
    auto const before = detail::collectSourceTiming();
    double const globalTimestep = baseTimestep / 16.0;
    auto const estimate = sphCompleteSources(0, baseTimestep + globalTimestep, baseTimestep,
        detail::kSourceScale, 0.0, energies, globalTimestep, baseTimestep);
    detail::requireSourceTiming(
        estimate.minimumTimestep > 0.0 && estimate.minimumTimestep <= baseTimestep,
        "post-source global cap ignored the raw-step bound");
    detail::verifySourceTiming(
        before, detail::collectSourceTiming(), 0, baseTimestep, globalTimestep, 0);
    sphTreeDestroy(true);
    setOptions(savedOptions);
    std::cout << "SPH source timing passed: old-rung intervals, inactive gas, own-close "
                 "conversion, raw global interval, exact endpoint scale, unchanged mechanical "
                 "state and momentum, and endpoint-only alpha\n"
              << std::flush;
}
