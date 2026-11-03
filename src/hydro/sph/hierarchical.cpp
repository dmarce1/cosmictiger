/* CosmicTiger - hierarchical SPH source scheduling; BSL-1.0. */

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <hpx/serialization/array.hpp>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/cosmology/evolution.hpp>
#if COSMICTIGER_ENABLE_PMMM_PP || COSMICTIGER_ENABLE_PMMM_FMM
#include <cosmictiger/gravity/nitadori/solver.hpp>
#endif
#include <cosmictiger/hydro/sph/cullen_and_dehnen/viscosity.hpp>
#include <cosmictiger/hydro/sph/hierarchical.hpp>
#include <cosmictiger/hydro/sph/hierarchical_sources.hpp>
#include <cosmictiger/hydro/sph/morris_and_monaghan/viscosity.hpp>
#include <cosmictiger/hydro/sph/price_etal/viscosity.hpp>
#include <cosmictiger/hydro/sph/sph_particles.hpp>
#include <cosmictiger/hydro/sph/staged_run.hpp>
#if COSMICTIGER_ENABLE_CHEMISTRY
#include <cosmictiger/hydro/sph/sph_chemistry.hpp>
#endif
#if COSMICTIGER_ENABLE_STAR_FORMATION
#include <cosmictiger/hydro/sph/star_formation.hpp>
#endif

// Runtime hierarchy is unavailable in hydro-only builds. Keep the lifetime hook
// linkable for the shared particle storage without introducing cosmology symbols.
#if COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL && COSMICTIGER_WITH_GRAVITY
namespace detail {

using SourceState = sph_hierarchical::SourceState;
using SourceBox = sph_hierarchical::SourceBox;
static sph_hierarchical::SourceForest sourceForest;
static std::mutex sourceMutex;
static bool sourcesInitialized = false;
static uint64_t activeCount = 0;
static uint64_t densityCount = 0;
static uint64_t forceCount = 0;
static std::vector<ParticleInteger> stepGasIndices;

uint64_t spreadSpatialBits(uint64_t value) {
    value = (value | (value << 32)) & 0x1f00000000ffffULL;
    value = (value | (value << 16)) & 0x1f0000ff0000ffULL;
    value = (value | (value << 8)) & 0x100f00f00f00f00fULL;
    value = (value | (value << 4)) & 0x10c30c30c30c30c3ULL;
    return (value | (value << 2)) & 0x1249249249249249ULL;
}

std::vector<ParticleInteger> activeGas(int minimumRung) {
    std::vector<ParticleInteger> result;
    auto const range = particlesCurrentRange();
    for (auto i = range.first; i < range.second; ++i) {
        if (particlesType(i) == kSphType && particlesRung(i) >= minimumRung) {
            auto const gas = particlesCatIndex(i);
            ALWAYS_ASSERT(sphParticlesParticleIndex(gas) == i);
            result.push_back(gas);
        }
    }
    return result;
}

SourceState liveSource(ParticleInteger gas, double time) {
    auto const& options = getOptions();
    SourceState source = sphParticlesHierarchicalStates[gas];
    source.id = source.id ? source.id : (uint64_t(hpxRank()) << 48) + uint64_t(gas) + 1;
    source.epoch = source.validUntil = time;
    source.rung = sphParticlesRung(gas);
    source.minimumSupportRadius = options.hydro.minSupportRadius;
    source.maximumSupportRadius = options.hydro.maxSupportRadius;
    source.supportRadius = std::clamp(double(sphParticlesSupportRadius(gas)),
        source.minimumSupportRadius, source.maximumSupportRadius);
    source.internalEnergy = sphParticlesInternalEnergy(gas);
    source.internalEnergyRate = sphParticlesInternalEnergyRate(gas);
    // At initialization only, density and grad-H have not yet been computed.
    source.density =
        std::max(double(sphParticlesDensity(gas)), double(std::numeric_limits<float>::min()));
    source.gradHCorrection = sph_volume::isFiniteValue(sphParticlesGradHCorrections[gas]) ?
        sphParticlesGradHCorrections[gas] :
        source.gradHCorrection;
    source.balsaraFactor = options.hydro.balsaraSwitch ? sphParticlesBalsaraFactors[gas] : 1.0;
    source.viscosityAlpha = options.hydro.viscosityEvolution != "none" ?
        sphParticlesViscosityStates[gas].alpha :
        options.hydro.viscosityAlpha;
    source.physicalDivergence =
        sphParticlesCullenDehnenStates ? sphParticlesCullenDehnenStates[gas].divergence : 0.0;
    source.pressureEnergyRatio = options.hydro.gamma - 1.0;
    source.adiabaticIndex = options.hydro.gamma;
    for (int dimension = 0; dimension < kDimensions; ++dimension) {
        source.position[dimension] = sphParticlesPosition(dimension, gas).toDouble();
        source.synchronizedVelocity[dimension] = sphParticlesVelocity(dimension, gas);
        source.driftVelocity[dimension] = sphParticlesVelocity(dimension, gas);
        source.hydroAcceleration[dimension] = sphParticlesAcceleration(dimension, gas);
    }
    return source;
}

void initializeSources(double time) {
    if (sourcesInitialized) {
        return;
    }
    std::vector<SourceState> sources;
    sources.reserve(sphParticlesSize());
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        auto source = sphParticlesHierarchicalStates[gas];
        if (time == 0.0) {
            source = liveSource(gas, time);
        } else if (source.validUntil < time) {
            throw std::runtime_error("Hierarchical SPH restart has an expired source anchor");
        }
        sources.push_back(source);
    }
    sourceForest.replaceActive(0, std::move(sources));
    sourcesInitialized = true;
}

void refreshSources(int minimumRung, double time, bool storedAnchors) {
    std::lock_guard lock(sourceMutex);
    initializeSources(time);
    std::vector<SourceState> sources;
    for (auto gas : activeGas(minimumRung)) {
        sources.push_back(
            storedAnchors ? sphParticlesHierarchicalStates[gas] : liveSource(gas, time));
    }
    sourceForest.replaceActive(minimumRung, std::move(sources));
}
}    // namespace detail
HPX_PLAIN_ACTION(detail::refreshSources, RefreshHierarchicalSourcesAction);
namespace detail {

// One pass queries every sink batch of a locality at the same time. Preparing
// the forest evaluates one envelope per tree node and one position per record,
// while a batch visits a few dozen nodes, so preparation amortizes once the
// pass covers more than a few percent of the retained population.
constexpr size_t kPreparedBatchWeight = 1024;

std::vector<std::vector<SourceState>> querySourceBatches(
    std::vector<SourceBox> boxes, std::vector<double> supports, double time) {
    if (boxes.size() != supports.size()) {
        throw std::invalid_argument("Hierarchical SPH query batches and supports differ in count");
    }
    std::lock_guard lock(sourceMutex);
    if (!boxes.empty() && boxes.size() * kPreparedBatchWeight >= sourceForest.size()) {
        sourceForest.prepare(time);
    }
    std::vector<std::vector<SourceState>> result;
    result.reserve(boxes.size());
    for (size_t batch = 0; batch < boxes.size(); ++batch) {
        result.push_back(sourceForest.query(boxes[batch], supports[batch], time));
    }
    return result;
}
}    // namespace detail
HPX_PLAIN_ACTION(detail::querySourceBatches, QueryHierarchicalSourceBatchesAction);
namespace detail {

// Active gas slots of this step in Morton order. Star formation removes gas slots by
// swapping, so the list is rebuilt after conversions; the work counters stay.
void orderStepGas(int minimumRung) {
    stepGasIndices = activeGas(minimumRung);
    std::vector<std::pair<uint64_t, ParticleInteger>> ordered;
    ordered.reserve(stepGasIndices.size());
    for (auto gas : stepGasIndices) {
        uint64_t key = 0;
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            uint64_t const coordinate =
                uint64_t(sphParticlesPosition(dimension, gas).raw()) >> (kPositionBits - 21);
            key |= spreadSpatialBits(coordinate) << dimension;
        }
        ordered.emplace_back(key, gas);
    }
    std::sort(ordered.begin(), ordered.end());
    for (size_t index = 0; index < ordered.size(); ++index) {
        stepGasIndices[index] = ordered[index].second;
    }
}

void resetWork(int minimumRung, double time, double signalSoundFloor) {
    std::lock_guard lock(sourceMutex);
    initializeSources(time);
    sourceForest.resetCounters();
    sourceForest.setSignalSoundFloor(signalSoundFloor);
    orderStepGas(minimumRung);
    activeCount = stepGasIndices.size();
    densityCount = forceCount = 0;
}
}    // namespace detail
HPX_PLAIN_ACTION(detail::resetWork, ResetHierarchicalSphWorkAction);
namespace detail {

std::array<uint64_t, 6> localWork() {
    std::lock_guard lock(sourceMutex);
    auto const counters = sourceForest.counters();
    return {activeCount, counters.sourceRecordsSorted + activeCount, densityCount, forceCount,
        counters.sourceRecordsQueried, counters.sourceRecordsPredicted};
}
}    // namespace detail
HPX_PLAIN_ACTION(detail::localWork, HierarchicalSphWorkAction);
namespace detail {

std::vector<double> signalBounds(
    std::vector<SourceState> sinks, double time, double courant, double scale) {
    std::lock_guard lock(sourceMutex);
    std::vector<double> bounds;
    bounds.reserve(sinks.size());
    for (auto const& sink : sinks) {
        bounds.push_back(sourceForest.signalTimestep(sink, time, courant, scale));
    }
    return bounds;
}
}    // namespace detail
HPX_PLAIN_ACTION(detail::signalBounds, HierarchicalSignalBoundsAction);
namespace detail {

SphRunReturn limitSignalsLocal(SphRunParameters parameters, double time) {
    auto const gasIndices = activeGas(parameters.minimumRung);
    std::vector<SourceState> sinks;
    for (auto gas : gasIndices) {
        sinks.push_back(liveSource(gas, time));
    }
    std::vector<hpx::future<std::vector<double>>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<HierarchicalSignalBoundsAction>(
            locality, sinks, time, parameters.courantFactor, parameters.scaleFactor));
    }
    hpx::wait_all(pending);
    std::vector<double> bounds(sinks.size(), parameters.maximumTimestep);
    for (auto& future : pending) {
        auto const remote = future.get();
        for (size_t index = 0; index < bounds.size(); ++index) {
            bounds[index] = std::min(bounds[index], remote[index]);
        }
    }
    SphRunReturn result;
    for (size_t index = 0; index < bounds.size(); ++index) {
        int const rung = std::max(int(sphParticlesRung(gasIndices[index])),
            int(std::ceil(std::log2(parameters.baseTimestep / bounds[index]))));
        if (rung >= kMaxRung) {
            throw std::runtime_error("SPH signal timestep exceeds the available rung range");
        }
        sphParticlesRung(gasIndices[index]) = rung;
        result.maximumRung = std::max(result.maximumRung, rung);
        result.minimumTimestep = std::min(result.minimumTimestep, bounds[index]);
    }
    return result;
}
}    // namespace detail
HPX_PLAIN_ACTION(detail::limitSignalsLocal, LimitHierarchicalSignalsAction);
namespace detail {

struct GasDiagnostics {
    double kinetic = 0.0;
    double xMomentum = 0.0;
    double yMomentum = 0.0;
    double zMomentum = 0.0;
    double momentumNorm = 0.0;
    double thermal = 0.0;
    int maximumCollisionlessRung = 0;

    template <typename Archive>
    void serialize(Archive&& archive, unsigned) {
        archive & kinetic;
        archive & xMomentum;
        archive & yMomentum;
        archive & zMomentum;
        archive & momentumNorm;
        archive & thermal;
        archive & maximumCollisionlessRung;
    }
};

GasDiagnostics gasDiagnosticsLocal(int minimumRung) {
    GasDiagnostics result;
    auto const range = particlesCurrentRange();
    for (auto particle = range.first; particle < range.second; ++particle) {
        if (particlesRung(particle) < minimumRung) {
            continue;
        }
        double const mass = particlesMass(particle);
        double const vx = particlesVel(kXDimension, particle);
        double const vy = particlesVel(kYDimension, particle);
        double const vz = particlesVel(kZDimension, particle);
        double const velocitySquared = vx * vx + vy * vy + vz * vz;
        result.kinetic += 0.5 * mass * velocitySquared;
        result.xMomentum += mass * vx;
        result.yMomentum += mass * vy;
        result.zMomentum += mass * vz;
        result.momentumNorm += mass * std::sqrt(velocitySquared);
        if (particlesType(particle) == kSphType) {
            result.thermal += mass * sphParticlesInternalEnergy(particlesCatIndex(particle));
        } else {
            result.maximumCollisionlessRung =
                std::max(result.maximumCollisionlessRung, int(particlesRung(particle)));
        }
    }
    return result;
}

double gasThermalLocal() {
    double thermal = 0.0;
    for (ParticleInteger gas = 0; gas < sphParticlesSize(); ++gas) {
        thermal +=
            double(particlesMass(sphParticlesParticleIndex(gas))) * sphParticlesInternalEnergy(gas);
    }
    return thermal;
}
}    // namespace detail
HPX_PLAIN_ACTION(detail::gasDiagnosticsLocal, HierarchicalGasDiagnosticsAction);
HPX_PLAIN_ACTION(detail::gasThermalLocal, HierarchicalGasThermalAction);
namespace detail {

GasDiagnostics gasDiagnosticsEverywhere(int minimumRung) {
    std::vector<hpx::future<GasDiagnostics>> pending;
    try {
        for (auto const& locality : hpxLocalities()) {
            pending.push_back(hpx::async<HierarchicalGasDiagnosticsAction>(locality, minimumRung));
        }
    } catch (...) {
        hpx::wait_all(pending);
        throw;
    }
    hpx::wait_all(pending);
    GasDiagnostics result;
    for (auto& future : pending) {
        auto const local = future.get();
        result.kinetic += local.kinetic;
        result.xMomentum += local.xMomentum;
        result.yMomentum += local.yMomentum;
        result.zMomentum += local.zMomentum;
        result.momentumNorm += local.momentumNorm;
        result.thermal += local.thermal;
        result.maximumCollisionlessRung =
            std::max(result.maximumCollisionlessRung, local.maximumCollisionlessRung);
    }
    return result;
}

template <typename Action, typename... Args>
void everywhere(Args const&... arguments) {
    std::vector<hpx::future<void>> pending;
    try {
        for (auto const& locality : hpxLocalities()) {
            pending.push_back(hpx::async<Action>(locality, arguments...));
        }
    } catch (...) {
        hpx::wait_all(pending);
        throw;
    }
    hpx::wait_all(pending);
    for (auto& future : pending) {
        future.get();
    }
}

void updateActive(int minimumRung, int phase, double time) {
    auto const& options = getOptions();
    for (auto gas : activeGas(minimumRung)) {
        auto& anchor = sphParticlesHierarchicalStates[gas];
        double const halfStep = 0.5 * (anchor.validUntil - anchor.epoch);
        if (phase == 0) {
            sphParticlesConverged(gas) = false;
            if (time == 0.0) {
                continue;
            }
            double const tolerance = 16.0 * std::numeric_limits<double>::epsilon() *
                std::max({std::abs(time), std::abs(anchor.validUntil), 2.0 * halfStep});
            if (std::abs(time - anchor.validUntil) > tolerance) {
                std::ostringstream message;
                message << std::setprecision(17)
                        << "Hierarchical SPH attempted to close an incomplete interval: id="
                        << anchor.id << " gas=" << gas
                        << " particle=" << sphParticlesParticleIndex(gas)
                        << " rung=" << int(sphParticlesRung(gas)) << " minimum=" << minimumRung
                        << " epoch=" << anchor.epoch << " endpoint=" << anchor.validUntil
                        << " time=" << time;
                throw std::runtime_error(message.str());
            }
            auto const predicted = sph_hierarchical::predictSource(anchor, time);
            sphParticlesSupportRadius(gas) = predicted.supportRadius;
            sphParticlesDensity(gas) = predicted.density;
            if (options.hydro.viscosityEvolution == "morrisAndMonaghan") {
                morris_and_monaghan::applyUpdate(
                    sphParticlesViscosityStates[gas], options.hydro.viscosityAlphaMin, halfStep, 0);
            }
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                sphParticlesPredictedAcceleration(dimension, gas) =
                    sphParticlesAcceleration(dimension, gas);
                sphParticlesVelocity(dimension, gas) +=
                    halfStep * sphParticlesAcceleration(dimension, gas);
            }
            sphParticlesPredictedInternalEnergyRate(gas) = sphParticlesInternalEnergyRate(gas);
            sphParticlesInternalEnergy(gas) += halfStep * sphParticlesInternalEnergyRate(gas);
        } else if (phase == 1 && time > 0.0) {
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                sphParticlesVelocity(dimension, gas) += halfStep *
                    (sphParticlesAcceleration(dimension, gas) -
                        sphParticlesPredictedAcceleration(dimension, gas));
            }
            sphParticlesInternalEnergy(gas) += halfStep *
                (sphParticlesInternalEnergyRate(gas) -
                    sphParticlesPredictedInternalEnergyRate(gas));
            if (options.hydro.viscosityEvolution == "morrisAndMonaghan") {
                morris_and_monaghan::applyUpdate(
                    sphParticlesViscosityStates[gas], options.hydro.viscosityAlphaMin, halfStep, 1);
            }
        } else if (phase == 3) {
            anchor = liveSource(gas, time);
        } else if (phase == 5) {
            // Star formation removed gas after these supports converged.
            sphParticlesConverged(gas) = false;
        } else if ((phase == 2 || phase == 4) && options.hydro.viscosityGradientSwitch()) {
            auto& state = sphParticlesCullenDehnenStates[gas];
            auto& viscosity = sphParticlesViscosityStates[gas];
            // Conversion changes neighbors at this same event. Commit the new
            // shock target without evolving viscosity over the interval twice.
            double const interval = phase == 4 ? 0.0 : 2.0 * halfStep;
            viscosity.alpha = options.hydro.viscosityEvolution == "priceEtAl" ?
                price_etal::evolvedAlpha(
                    viscosity.alpha, state.targetAlpha, state.inverseDecayTime, interval) :
                cullen_and_dehnen::evolvedAlpha(
                    viscosity.alpha, state.targetAlpha, state.inverseDecayTime, interval);
            state.divergence = state.pendingDivergence;
        }
        if (!(sphParticlesInternalEnergy(gas) > 0.0f)) {
            throw std::runtime_error("Hierarchical SPH kick produced nonpositive internal energy");
        }
    }
}
}    // namespace detail
HPX_PLAIN_ACTION(detail::updateActive, UpdateHierarchicalGasAction);
namespace detail {

struct SourceResult {
    double heating = 0.0;
    uint64_t converted = 0;

    template <typename Archive>
    void serialize(Archive&& archive, unsigned) {
        archive & heating;
        archive & converted;
    }
};

// Operator-split GADGET-4 sources for the gas closing at this event, as sphStep2
// applies them after the corrector. Each particle integrates the source over the
// anchor interval its hydro closure just integrated. Every locality stages its
// results before any commits them, so a failed source commits nothing.
SourceResult prepareSourcesLocal(int minimumRung, double scaleFactor) {
    std::vector<ParticleInteger> gasIndices;
    std::vector<double> intervals;
    for (auto gas : activeGas(minimumRung)) {
        auto const& anchor = sphParticlesHierarchicalStates[gas];
        gasIndices.push_back(gas);
        intervals.push_back(anchor.validUntil - anchor.epoch);
    }
    SourceResult result;
#if COSMICTIGER_ENABLE_STAR_FORMATION
    if (getOptions().starFormation.enabled) {
        // The combined SH03 source owns cooling, as in sphStep2.
        auto const update = starFormationPrepareLocal(gasIndices, intervals, scaleFactor);
        result.heating = update.heating;
        result.converted = update.converted;
        return result;
    }
#endif
#if COSMICTIGER_ENABLE_CHEMISTRY
    if (getOptions().chemistry.enabled) {
        result.heating = sphCoolingPrepareLocal(gasIndices, intervals, scaleFactor);
        return result;
    }
#endif
    throw std::logic_error("Hierarchical SPH sources require chemistry or star formation");
}
}    // namespace detail
HPX_PLAIN_ACTION(detail::prepareSourcesLocal, PrepareHierarchicalSourcesAction);
namespace detail {

// Conversions swap gas slots, so the step's work list is rebuilt after them. Newborn
// stars select their collisionless top rungs when the gravity opening follows and
// otherwise leave no pending opening behind.
void commitSourcesLocal(int minimumRung, double baseTimestep, bool converted, bool openStars) {
#if COSMICTIGER_ENABLE_STAR_FORMATION
    if (getOptions().starFormation.enabled) {
        starFormationCommitLocal();
        if (converted) {
            std::lock_guard lock(sourceMutex);
            orderStepGas(minimumRung);
        }
        if (openStars) {
            starFormationSelectHierarchicalRungs(minimumRung, baseTimestep);
        } else {
            starFormationClearNewStarsLocal();
        }
        return;
    }
#endif
#if COSMICTIGER_ENABLE_CHEMISTRY
    if (getOptions().chemistry.enabled) {
        sphCoolingCommitLocal();
        return;
    }
#endif
    throw std::logic_error("Hierarchical SPH sources require chemistry or star formation");
}
}    // namespace detail
HPX_PLAIN_ACTION(detail::commitSourcesLocal, CommitHierarchicalSourcesAction);
namespace detail {

// Each batch has compact sink arrays and independent source rows. Inactive
// particles enter only through geometric queries against retained source trees.
struct PackedRun {
    std::array<std::vector<Position>, 3> position;
    std::array<std::vector<float>, 3> velocity, acceleration, gravity;
    std::vector<float> energy, density, support, correction, alpha, balsara, divergence;
    std::vector<float> gravityTimestep;
    std::vector<char> sourceRung, sinkRung, converged, semiactive;
    std::vector<float> sinkDensity, sinkCorrection, sinkBalsara, energyRate, logDensityRate;
    std::vector<SphThermodynamicState> thermodynamics;
    std::vector<SphAcceleration> sinkAcceleration;
    std::vector<SphViscosityState> viscosity;
    std::vector<SphCullenDehnenState> gradient;
    std::vector<ParticleInteger> indices;
    std::vector<SphTreeNode> nodes;
    std::vector<int> targets, neighbors;

    void append(SourceState const& source) {
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            position[dimension].emplace_back(source.position[dimension]);
            velocity[dimension].push_back(source.synchronizedVelocity[dimension]);
            acceleration[dimension].push_back(source.hydroAcceleration[dimension]);
        }
        energy.push_back(source.internalEnergy);
        density.push_back(source.density);
        support.push_back(source.supportRadius);
        correction.push_back(source.gradHCorrection);
        alpha.push_back(source.viscosityAlpha);
        balsara.push_back(source.balsaraFactor);
        divergence.push_back(source.physicalDivergence);
        sourceRung.push_back(source.rung);
    }

    SphRunData data() {
        SphRunData data{};
        auto const& options = getOptions();
        data.sourcePositionX = position[0].data();
        data.sourcePositionY = position[1].data();
        data.sourcePositionZ = position[2].data();
        data.sourceVelocityX = velocity[0].data();
        data.sourceVelocityY = velocity[1].data();
        data.sourceVelocityZ = velocity[2].data();
        data.sourceAccelerationX = acceleration[0].data();
        data.sourceAccelerationY = acceleration[1].data();
        data.sourceAccelerationZ = acceleration[2].data();
        data.sourceInternalEnergy = energy.data();
        data.sourceDensity = density.data();
        data.sourceSupportRadius = support.data();
        data.sourceGradHCorrection = correction.data();
        data.sourceViscosityAlpha = alpha.data();
        data.sourceBalsaraFactor = balsara.data();
        data.sourceVelocityDivergence = divergence.data();
        data.sourceRungs = sourceRung.data();
        data.sinkParticleRungs = sinkRung.data();
        data.sinkParticleIndices = indices.data();
        data.sinkDensity = sinkDensity.data();
        data.sinkGradHCorrection = sinkCorrection.data();
        data.sinkBalsaraFactor = options.hydro.balsaraSwitch ? sinkBalsara.data() : nullptr;
        data.sinkInternalEnergyRate = energyRate.data();
        data.sinkLogDensityRate = logDensityRate.data();
        data.sinkThermodynamicStates = thermodynamics.data();
        data.sinkAccelerations = sinkAcceleration.data();
        data.sinkSmoothingConverged = converged.data();
        data.sinkSemiactive = semiactive.data();
        data.sinkViscosityStates =
            options.hydro.viscosityEvolution != "none" ? viscosity.data() : nullptr;
        data.sinkCullenDehnenStates =
            options.hydro.viscosityGradientSwitch() ? gradient.data() : nullptr;
        data.sinkGravityAccelerationX = gravity[0].data();
        data.sinkGravityAccelerationY = gravity[1].data();
        data.sinkGravityAccelerationZ = gravity[2].data();
        data.sinkGravityTimestep = gravityTimestep.data();
        data.numberOfTargetNodes = targets.size();
        data.treeNodes = nodes.data();
        data.targetNodeIndices = targets.data();
        data.neighborNodeIndices = neighbors.data();
        data.targetNeighborCount = options.hydro.neighborNumber;
        data.gasMass = options.initial.gasMass;
        data.adiabaticIndex = options.hydro.gamma;
        data.selfGravityEnabled = options.gravity.enabled;
        data.gravityEffectiveRadius =
            options.gravity.supportRadius / selectedGravityKernel().supportRadiusPerEffectiveRadius;
#if COSMICTIGER_ENABLE_PMMM_PP || COSMICTIGER_ENABLE_PMMM_FMM
        if (usesMeshGravity(options.gravity)) {
            data.gravityEffectiveRadius = pmmmEffectiveRadius();
        }
#endif
        data.accelerationTimestepFactor = options.gravity.timestepEta;
        data.centralAcceleration = options.hydro.centralAcceleration;
        data.centralSupportRadius = options.hydro.centralSupportRadius;
        return data;
    }
};

SphRunReturn runLocal(SphRunParameters parameters, double time) {
    auto const& gasIndices = stepGasIndices;
    if (gasIndices.empty()) {
        return {};
    }
    if (parameters.runType == kSphRunPrehydro1) {
        for (auto gas : gasIndices) {
            densityCount += !sphParticlesConverged(gas);
        }
    } else if (parameters.runType == kSphRunPrehydro2) {
        densityCount += gasIndices.size();
    } else if (parameters.runType == kSphRunHydro) {
        forceCount += gasIndices.size();
    }
    PackedRun packed;
    for (auto gas : gasIndices) {
        packed.append(liveSource(gas, time));
        packed.indices.push_back(packed.indices.size());
        packed.sinkRung.push_back(sphParticlesRung(gas));
        packed.converged.push_back(sphParticlesConverged(gas));
        packed.semiactive.push_back(true);
        packed.sinkDensity.push_back(sphParticlesDensity(gas));
        packed.sinkCorrection.push_back(sphParticlesGradHCorrections[gas]);
        packed.sinkBalsara.push_back(
            sphParticlesBalsaraFactors ? sphParticlesBalsaraFactors[gas] : 1.0f);
        packed.energyRate.push_back(sphParticlesInternalEnergyRate(gas));
        packed.logDensityRate.push_back(sphParticlesHierarchicalStates[gas].logDensityRate);
        packed.gravityTimestep.push_back(sphParticlesGravityTimesteps[gas]);
        packed.thermodynamics.push_back(sphParticlesThermodynamicState(gas));
        packed.sinkAcceleration.push_back(sphParticlesAccelerationState(gas));
        packed.viscosity.push_back(
            sphParticlesViscosityStates ? sphParticlesViscosityStates[gas] : SphViscosityState{});
        packed.gradient.push_back(sphParticlesCullenDehnenStates ?
                sphParticlesCullenDehnenStates[gas] :
                SphCullenDehnenState{});
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            packed.gravity[dimension].push_back(sphParticlesGravityAcceleration(dimension, gas));
        }
    }
    constexpr size_t kBatchSize = 32;
    size_t const batches = (gasIndices.size() + kBatchSize - 1) / kBatchSize;
    packed.nodes.resize(2 * batches);
    std::vector<SourceBox> queryBoxes;
    std::vector<double> querySupports;
    queryBoxes.reserve(batches);
    querySupports.reserve(batches);
    for (size_t batch = 0; batch < batches; ++batch) {
        size_t const begin = batch * kBatchSize;
        size_t const end = std::min(begin + kBatchSize, gasIndices.size());
        SourceBox box;
        box.lower.fill(1.0);
        box.upper.fill(0.0);
        double maximumSupport = 0.0;
        for (size_t sink = begin; sink < end; ++sink) {
            maximumSupport = std::max(maximumSupport, double(packed.support[sink]));
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                double const coordinate = packed.position[dimension][sink].toDouble();
                box.lower[dimension] = std::min(box.lower[dimension], coordinate);
                box.upper[dimension] = std::max(box.upper[dimension], coordinate);
            }
        }
        // The support solver requires a fixed-point margin at both faces,
        // including when a sink reaches the permitted half-box support.
        maximumSupport = std::min(0.5, 1.25 * maximumSupport);
        double const padding = 4.0 * RangeFixed::min().toDouble();
        SourceBox queryBox = box;
        auto& target = packed.nodes[batch];
        target.particleRange =
            target.sinkParticleRange = {ParticleInteger(begin), ParticleInteger(end)};
        target.numberOfActiveParticles = end - begin;
        target.neighborRange = {int(batch), int(batch + 1)};
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            target.innerBox.begin[dimension] = box.lower[dimension];
            target.innerBox.end[dimension] = box.upper[dimension];
            target.outerBox.begin[dimension] = box.lower[dimension] - maximumSupport - padding;
            target.outerBox.end[dimension] = box.upper[dimension] + maximumSupport + padding;
            queryBox.lower[dimension] -= padding;
            queryBox.upper[dimension] += padding;
        }
        packed.targets.push_back(batch);
        packed.neighbors.push_back(batches + batch);
        queryBoxes.push_back(queryBox);
        querySupports.push_back(maximumSupport);
    }
    // One batched query per locality per pass, indexed [locality][batch].
    std::vector<std::vector<std::vector<SourceState>>> sources;
    if (hpxSize() == 1) {
        // The forest is local and the refresh has completed. Avoid scheduling
        // an HPX action/future for the pass.
        sources.push_back(querySourceBatches(queryBoxes, querySupports, time));
    } else {
        std::vector<hpx::future<std::vector<std::vector<SourceState>>>> queries;
        for (auto const& locality : hpxLocalities()) {
            queries.push_back(hpx::async<QueryHierarchicalSourceBatchesAction>(
                locality, queryBoxes, querySupports, time));
        }
        hpx::wait_all(queries);
        for (auto& query : queries) {
            sources.push_back(query.get());
        }
    }
    // Append each batch's rows in locality order, as the per-batch round trips
    // did, so the packed layout and the summation order are unchanged.
    for (size_t batch = 0; batch < batches; ++batch) {
        auto& neighbor = packed.nodes[batches + batch];
        neighbor.particleRange.first = packed.energy.size();
        for (auto const& locality : sources) {
            if (locality.size() != batches) {
                throw std::runtime_error(
                    "Hierarchical SPH source query returned wrong batch count");
            }
            for (auto const& source : locality[batch]) {
                packed.append(source);
            }
        }
        neighbor.particleRange.second = packed.energy.size();
    }
    auto result = sphRunStaged(parameters, packed.data(), packed.energy.size(), gasIndices.size(),
        packed.nodes.size(), packed.neighbors.size(), getOptions().hydro.cuda);
    for (size_t sink = 0; sink < gasIndices.size(); ++sink) {
        auto const gas = gasIndices[sink];
        sphParticlesRung(gas) = packed.sinkRung[sink];
        result.maximumRung = std::max(result.maximumRung, int(packed.sinkRung[sink]));
        sphParticlesConverged(gas) = packed.converged[sink];
        sphParticlesDensity(gas) = packed.sinkDensity[sink];
        if (parameters.runType == kSphRunPrehydro2) {
            auto& anchor = sphParticlesHierarchicalStates[gas];
            double const elapsed = time - anchor.epoch;
            bool const held =
                packed.thermodynamics[sink].supportRadius <= parameters.minimumSupportRadius ||
                packed.thermodynamics[sink].supportRadius >= parameters.maximumSupportRadius;
            anchor.gradHCorrectionRate = elapsed > 0.0 && !held ?
                (packed.sinkCorrection[sink] - anchor.gradHCorrection) / elapsed :
                0.0;
        }
        sphParticlesGradHCorrections[gas] = packed.sinkCorrection[sink];
        sphParticlesInternalEnergyRate(gas) = packed.energyRate[sink];
        sphParticlesHierarchicalStates[gas].logDensityRate = packed.logDensityRate[sink];
        sphParticlesThermodynamicState(gas) = packed.thermodynamics[sink];
        sphParticlesAccelerationState(gas) = packed.sinkAcceleration[sink];
        if (sphParticlesBalsaraFactors) {
            sphParticlesBalsaraFactors[gas] = packed.sinkBalsara[sink];
        }
        if (sphParticlesViscosityStates) {
            sphParticlesViscosityStates[gas] = packed.viscosity[sink];
        }
        if (sphParticlesCullenDehnenStates) {
            sphParticlesCullenDehnenStates[gas] = packed.gradient[sink];
        }
    }
    return result;
}
}    // namespace detail
HPX_PLAIN_ACTION(detail::runLocal, RunHierarchicalSphAction);
namespace detail {

SphRunReturn runEverywhere(SphRunParameters const& parameters, double time) {
    everywhere<RefreshHierarchicalSourcesAction>(parameters.minimumRung, time, false);
    std::vector<hpx::future<SphRunReturn>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<RunHierarchicalSphAction>(locality, parameters, time));
    }
    hpx::wait_all(pending);
    SphRunReturn result;
    for (auto& future : pending) {
        result += future.get();
    }
    return result;
}

double openLocal(int rung, double time, double timestep, double driftScale) {
    double kineticChange = 0.0;
    for (auto gas : activeGas(rung)) {
        if (sphParticlesRung(gas) != rung) {
            continue;
        }
        SourceState source = liveSource(gas, time);
        // HOLD's gravity opening has already committed its momentum. Retain
        // that held value here; only hydro is predicted within this drift.
        double const baseTimestep = std::ldexp(timestep, rung);
        double const endpointTicks = std::round(std::ldexp(time / baseTimestep, kMaxRung - 1)) +
            std::ldexp(1.0, kMaxRung - 1 - rung);
        source.validUntil = baseTimestep * std::ldexp(endpointTicks, 1 - kMaxRung);
        source.driftScale = driftScale;
        double const mass = particlesMass(sphParticlesParticleIndex(gas));
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            float& velocity = sphParticlesVelocity(dimension, gas);
            double const previous = velocity;
            velocity += 0.5 * timestep * sphParticlesAcceleration(dimension, gas);
            kineticChange += 0.5 * mass * (double(velocity) * velocity - previous * previous);
            source.driftVelocity[dimension] = velocity;
        }
        sphParticlesInternalEnergy(gas) += 0.5 * timestep * sphParticlesInternalEnergyRate(gas);
        if (!(sphParticlesInternalEnergy(gas) > 0.0)) {
            throw std::runtime_error(
                "Hierarchical SPH opening produced nonpositive internal energy");
        }
        if (getOptions().hydro.viscosityEvolution == "morrisAndMonaghan") {
            morris_and_monaghan::applyUpdate(sphParticlesViscosityStates[gas],
                getOptions().hydro.viscosityAlphaMin, 0.5 * timestep, 2);
        }
        sphParticlesHierarchicalStates[gas] = source;
    }
    // Faster pending cohorts are replaced later as the ladder descends.
    std::vector<SourceState> sources;
    for (auto gas : activeGas(rung)) {
        sources.push_back(sphParticlesRung(gas) == rung ? sphParticlesHierarchicalStates[gas] :
                                                          liveSource(gas, time));
    }
    std::lock_guard lock(sourceMutex);
    sourceForest.replaceActive(rung, std::move(sources));
    return kineticChange;
}
}    // namespace detail
HPX_PLAIN_ACTION(detail::openLocal, OpenHierarchicalGasAction);
namespace detail {

double openGravityLocal(int minimumRung, double halfTimestep) {
    double kineticChange = 0.0;
    for (auto gas : activeGas(minimumRung)) {
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            float& velocity = sphParticlesVelocity(dimension, gas);
            double const previous = velocity;
            velocity += halfTimestep * sphParticlesGravityAcceleration(dimension, gas);
            kineticChange += 0.5 * getOptions().initial.gasMass *
                (double(velocity) * velocity - previous * previous);
        }
    }
#if COSMICTIGER_ENABLE_STAR_FORMATION
    if (getOptions().starFormation.enabled) {
        // Stars born at this event were gas when the top kick deferred their opening.
        kineticChange += starFormationKickHierarchicalNewStars(halfTimestep);
    }
#endif
    return kineticChange;
}

SourceResult sourcesEverywhere(int minimumRung, double scaleFactor) {
    std::vector<hpx::future<SourceResult>> pending;
    try {
        for (auto const& locality : hpxLocalities()) {
            pending.push_back(
                hpx::async<PrepareHierarchicalSourcesAction>(locality, minimumRung, scaleFactor));
        }
    } catch (...) {
        hpx::wait_all(pending);
        throw;
    }
    hpx::wait_all(pending);
    SourceResult result;
    for (auto& future : pending) {
        auto const local = future.get();
        result.heating += local.heating;
        result.converted += local.converted;
    }
    return result;
}
}    // namespace detail
HPX_PLAIN_ACTION(detail::openGravityLocal, OpenHierarchicalGasGravityAction);

HierarchicalSphResult sphHierarchicalStep(int minimumRung, double scaleFactor, double expansionRate,
    double time, double baseTimestep, bool closingOnly) {
    SphRunParameters parameters;
    parameters.minimumRung = minimumRung;
    parameters.scaleFactor = scaleFactor;
    parameters.expansionRate = expansionRate;
    parameters.baseTimestep = baseTimestep;
    parameters.courantFactor = getOptions().hydro.cfl;
    parameters.maximumTimestep = std::ldexp(baseTimestep, -getOptions().run.minRung);
    if (getOptions().run.scenario == "cosmology") {
        parameters.maximumTimestep = std::min(double(parameters.maximumTimestep),
            cosmology::expansionTimestep(
                scaleFactor, expansionRate / scaleFactor, cosmosD2adt2(scaleFactor)));
        if (parameters.gradientSwitch()) {
            parameters.expansionAcceleration =
                scaleFactor * scaleFactor * scaleFactor * cosmosD2adt2(scaleFactor);
        }
    }
    parameters.excludeSelfGravity = true;
    parameters.phase = 0;
    detail::everywhere<ResetHierarchicalSphWorkAction>(minimumRung, time, 0.0);
    auto const before = detail::gasDiagnosticsEverywhere(minimumRung);
    detail::everywhere<UpdateHierarchicalGasAction>(minimumRung, 0, time);
    HierarchicalSphResult output;
    auto& result = output.run;
    auto const solveDensities = [&]() {
        parameters.runType = kSphRunPrehydro1;
        for (int pass = 0;; ++pass) {
            if (pass == 256) {
                throw std::runtime_error("Hierarchical SPH supports failed to converge");
            }
            result = detail::runEverywhere(parameters, time);
            if (!result.needsAnotherPass) {
                break;
            }
        }
        parameters.runType = kSphRunPrehydro2;
        result = detail::runEverywhere(parameters, time);
    };
    solveDensities();
    detail::everywhere<UpdateHierarchicalGasAction>(minimumRung, 2, time);
    parameters.runType = kSphRunHydro;
    result = detail::runEverywhere(parameters, time);
    detail::everywhere<UpdateHierarchicalGasAction>(minimumRung, 1, time);
    if (time > 0.0 && (getOptions().chemistry.enabled || getOptions().starFormation.enabled)) {
        auto const sources = detail::sourcesEverywhere(minimumRung, scaleFactor);
        output.heating = sources.heating;
        output.converted = sources.converted;
        bool const openStars = !closingOnly && getOptions().gravity.enabled;
        detail::everywhere<CommitHierarchicalSourcesAction>(
            minimumRung, baseTimestep, sources.converted != 0, openStars);
        if (sources.converted != 0 && !closingOnly) {
            // Converted stars leave every gas kernel before the opening force. The
            // refresh preceding each pass drops them from the retained sources.
            detail::everywhere<UpdateHierarchicalGasAction>(minimumRung, 5, time);
            solveDensities();
            detail::everywhere<UpdateHierarchicalGasAction>(minimumRung, 4, time);
        }
        terminal::detailPrint("SPH hierarchy sources: time=%.17e minimum_rung=%i heating=%.17e "
                              "stars=%llu terminal=%d\n",
            time, minimumRung, sources.heating, (unsigned long long) sources.converted,
            int(closingOnly));
    }
    auto const after = detail::gasDiagnosticsEverywhere(minimumRung);
    output.kineticDelta = after.kinetic - before.kinetic;
    output.xMomentumDelta = after.xMomentum - before.xMomentum;
    output.yMomentumDelta = after.yMomentum - before.yMomentum;
    output.zMomentumDelta = after.zMomentum - before.zMomentum;
    output.momentumNormDelta = after.momentumNorm - before.momentumNorm;
    output.kineticEnergy = after.kinetic;
    output.xMomentum = after.xMomentum;
    output.yMomentum = after.yMomentum;
    output.zMomentum = after.zMomentum;
    output.momentumNorm = after.momentumNorm;
    output.thermalEnergy = after.thermal;
    int const collisionlessRung = after.maximumCollisionlessRung;
    if (closingOnly) {
        return output;
    }
    parameters.runType = kSphRunHydro;
    parameters.phase = 1;
    result = detail::runEverywhere(parameters, time);
    detail::everywhere<RefreshHierarchicalSourcesAction>(minimumRung, time, false);
    std::vector<hpx::future<SphRunReturn>> signalQueries;
    for (auto const& locality : hpxLocalities()) {
        signalQueries.push_back(
            hpx::async<LimitHierarchicalSignalsAction>(locality, parameters, time));
    }
    hpx::wait_all(signalQueries);
    for (auto& future : signalQueries) {
        result += future.get();
    }
    parameters.runType = kSphRunRungs;
    for (int pass = 0;; ++pass) {
        if (pass == kMaxRung) {
            throw std::runtime_error("Hierarchical SPH active rung limiter failed to converge");
        }
        auto const limiter = detail::runEverywhere(parameters, time);
        result.maximumRung = std::max(result.maximumRung, limiter.maximumRung);
        if (!limiter.needsAnotherPass) {
            break;
        }
    }
    detail::everywhere<UpdateHierarchicalGasAction>(minimumRung, 3, time);
    // Report the actual active suffix after gas proposals, including existing
    // stars. The top gravity return still contains gas's OLD deferred rungs.
    // Keeping that stale maximum would descend into an empty rung, leave an
    // extra saved suffix boundary, and skip a later coarser gas closing event.
    // Ballistic stars also need these exact-rung drifts when gravity is off.
    result.maximumRung = std::max(result.maximumRung, collisionlessRung);
    return output;
}

double sphHierarchicalOpen(int rung, double time, double timestep, double driftScale) {
    std::vector<hpx::future<double>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(
            hpx::async<OpenHierarchicalGasAction>(locality, rung, time, timestep, driftScale));
    }
    hpx::wait_all(pending);
    double kineticChange = 0.0;
    for (auto& future : pending) {
        kineticChange += future.get();
    }
    return kineticChange;
}

double sphHierarchicalThermalEnergy() {
    std::vector<hpx::future<double>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<HierarchicalGasThermalAction>(locality));
    }
    hpx::wait_all(pending);
    double thermal = 0.0;
    for (auto& future : pending) {
        thermal += future.get();
    }
    return thermal;
}

void sphHierarchicalResetSources() {
    std::lock_guard lock(detail::sourceMutex);
    detail::sourceForest = sph_hierarchical::SourceForest();
    detail::sourcesInitialized = false;
    detail::stepGasIndices.clear();
}

double sphHierarchicalOpenGravity(int minimumRung, double halfTimestep) {
    std::vector<hpx::future<double>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(
            hpx::async<OpenHierarchicalGasGravityAction>(locality, minimumRung, halfTimestep));
    }
    hpx::wait_all(pending);
    double kineticChange = 0.0;
    for (auto& future : pending) {
        kineticChange += future.get();
    }
    return kineticChange;
}

void sphHierarchicalReportWork(int minimumRung, double time) {
    std::vector<hpx::future<std::array<uint64_t, 6>>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<HierarchicalSphWorkAction>(locality));
    }
    hpx::wait_all(pending);
    std::array<uint64_t, 6> totals{};
    for (auto& future : pending) {
        auto const local = future.get();
        for (size_t index = 0; index < totals.size(); ++index) {
            totals[index] += local[index];
        }
    }
    terminal::detailPrint(
        "SPH hierarchy: time=%.17e minimum_rung=%i active=%llu sorted=%llu "
        "density=%llu force=%llu source_candidates=%llu source_predictions=%llu\n",
        time, minimumRung, (unsigned long long) totals[0], (unsigned long long) totals[1],
        (unsigned long long) totals[2], (unsigned long long) totals[3],
        (unsigned long long) totals[4], (unsigned long long) totals[5]);
}

#else
void sphHierarchicalResetSources() {}
#endif
