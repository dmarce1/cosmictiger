/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/math/vector3.hpp>
#if COSMICTIGER_WITH_GRAVITY
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/cosmology/evolution.hpp>
#endif
#include <cosmictiger/hydro/hopkins/hopkins.hpp>
#include <cosmictiger/hydro/hopkins/hopkins_hydro.hpp>
#include <cosmictiger/hydro/hopkins/hopkins_state.hpp>
#include <cosmictiger/hydro/source_tree.hpp>
#include <cosmictiger/hydro/sph/sph.hpp>
#include <cosmictiger/hydro/sph/sph_flops.hpp>
#include <cosmictiger/hydro/sph/sph_kernel.hpp>
#include <cosmictiger/hydro/sph/sph_particles.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/particles/star_formation_state.hpp>
#include <cosmictiger/scenarios/scenario.hpp>
#if COSMICTIGER_ENABLE_CHEMISTRY
#include <cosmictiger/hydro/sph/sph_chemistry.hpp>
#endif
#if COSMICTIGER_ENABLE_STAR_FORMATION
#include <cosmictiger/hydro/sph/star_formation.hpp>
#endif

namespace detail {

// Stack-owned batches follow HPX tasks safely across yields. Every producer has
// its own accumulator; only completed batches enter the process-wide counters.
class HopkinsFlopBatch {
    FlopCount count_ = 0;
    bool enabled_;

public:
    explicit HopkinsFlopBatch(bool enabled = true)
      : enabled_(enabled && kCountFlops) {}

    HopkinsFlopBatch(HopkinsFlopBatch const&) = delete;
    HopkinsFlopBatch& operator=(HopkinsFlopBatch const&) = delete;
    HopkinsFlopBatch(HopkinsFlopBatch&&) = delete;
    HopkinsFlopBatch& operator=(HopkinsFlopBatch&&) = delete;

    operator FlopCount*() {
        return enabled_ ? &count_ : nullptr;
    }

    ~HopkinsFlopBatch() {
        if (enabled_) {
            addCpuFlops(count_, FlopPrecision::Double);
        }
    }
};

struct HopkinsParticle {
    Vector3<Position> position{};
    hopkins::Vector velocity{};
    hopkins::Vector acceleration{};
    HopkinsParticleState state{};
    bool active = false;
    double mass = 0.0;
    double energy = 0.0;
    double support = 0.0;
    double density = 0.0;
    int rung = 0;
    int gravityRung = 0;
    // GADGET-4 source state travels with the record; the cooling guess and the
    // SH03 metadata are read at collection and written back at the install.
    gadget4::State chemistry{};
    StarFormationState starFormation{};
    // A whole-cell conversion decided at this closure: the cell keeps its closing
    // state, opens no pair interval, and leaves the gas set when it is installed.
    bool convert = false;
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & position & velocity & acceleration & state & active & mass & energy & support &
            density & rung & gravityRung & chemistry & starFormation & convert;
    }
};

// Owner of a resident record: the locality and local gas index that store it.
struct HopkinsOwner {
    int rank = 0;
    ParticleInteger index = 0;
};

// Per-event accounting for the retained-record refresh. Every count covers the
// current event only (Begin and Finish together); the "Hopkins work" row
// reports it beside the total gas count so fine/full ratios can be measured.
struct HopkinsWork {
    uint64_t recordsReceived = 0;
    uint64_t bytesReceived = 0;
    uint64_t recordsInstalled = 0;
    uint64_t bytesInstalled = 0;
    uint64_t sourcesUpdated = 0;
    uint64_t supportCells = 0;
    uint64_t supportSums = 0;
    uint64_t geometryCells = 0;
    uint64_t rateCells = 0;
    uint64_t gradientCells = 0;
    uint64_t listQueries = 0;
    uint64_t pairsSolved = 0;
};

// The coordinator's resident workspace. Records persist between events in
// increasing stable-identifier order; owners report only changed records, and
// each event solves supports and geometry for the active cells plus the
// first-ring halo their pair fluxes need. Inactive cells outside that halo keep
// their anchors, supports and cached pair intervals untouched.
struct HopkinsWorkspace {
    bool countOperations = true;
    std::vector<HopkinsParticle> particles;
    std::vector<HopkinsOwner> owners;
    std::vector<std::vector<uint64_t>> rankIdentifiers;
    std::vector<std::vector<size_t>> rankSlots;
    std::unordered_map<uint64_t, size_t> indices;
    std::vector<hopkins::Geometry> geometry;
    std::vector<hopkins::GeometryRate> geometryRates;
    std::vector<double> supportRates;
    std::vector<hopkins::Primitive> primitives;
    std::vector<hopkins::Gradient> gradients;
    std::vector<unsigned char> dirty;
    std::vector<unsigned char> solved;
    std::vector<unsigned char> built;
    std::vector<unsigned char> touched;
    hydro_source::SourceTree tree;
    HopkinsWork work;
    uint64_t nextIdentifier = 0;
    // Set once this event's first stencil query has run: from then on no
    // resident position may change until the event closes. Neighbor sets are
    // never stored; every consumer queries the tree when it needs them.
    bool stencilsFrozen = false;
    // The tree topology decision (rebuild or keep) is made once per event, at
    // the first marking of the active set.
    bool rebuildDecided = false;
    bool eventOpen = false;
    double eventTime = 0.0;
    double expansionRate = 0.0;
    double scale = 1.0;
    double tickDuration = 0.0;
    double globalTimestep = 0.0;
    double time = 0.0;
    uint64_t tick = 0;
    // Dual-energy pressure switches taken at this event's closures, and the total
    // energy they added; the energy budget books it like a thermal source.
    size_t switched = 0;
    double switchedEnergy = 0.0;
    int darkMatterRung = 0;
    int dimensions = 3;
};

struct HopkinsPairRecord {
    uint64_t leftId = 0;
    uint64_t rightId = 0;
    uint64_t beginTick = 0;
    uint64_t endTick = 0;
    double beginTime = 0.0;
    double endTime = 0.0;
    hopkins::Conserved initial{};
    hopkins::Conserved midpoint{};
    hopkins::Vector workInitial{};
    hopkins::Vector workMidpoint{};
};
using HopkinsPairKey = std::pair<uint64_t, uint64_t>;
std::map<HopkinsPairKey, HopkinsPairRecord> hopkinsPairs;
std::unique_ptr<HopkinsWorkspace> hopkinsResident;
bool hopkinsTreePrepared = false;
bool hopkinsEventPrepared = false;
// A domain rebound happened since the last event: rebuild the tree topology.
bool hopkinsRebuildOnNextEvent = false;
bool hopkinsHierarchyPending = false;
size_t hopkinsHierarchyWoken = 0;
double hopkinsHierarchyDriftThermal = 0.0;
static_assert(kMaxRung <= std::numeric_limits<uint16_t>::digits);
constexpr int kHopkinsTickRung = kMaxRung - 1;
constexpr uint64_t kHopkinsTicksPerBase = uint64_t(1) << kHopkinsTickRung;
constexpr uint64_t kHopkinsCacheMagic = 0x484f504341434845ULL;
// The midpoint geometry differentiates the constant kernel-count constraint.
// Solve in double precision: a float support and a loose residual can obscure
// the 1e-6 sound-wave perturbation. The output support mirror alone is rounded.
constexpr double kHopkinsSupportTolerance = 1.0e-13;
// Support solves search 1.2 times the current radius; final lists use 1.0.
constexpr double kHopkinsSolveSearchFactor = 1.2;

uint64_t hopkinsTick(double time, double baseTimestep, FlopCount* operations);
void hopkinsNeighborQuery(HopkinsWorkspace const& workspace, size_t i, double factor,
    bool symmetric, FlopCount* operations, std::vector<size_t>& neighbors);

HopkinsWorkspace& hopkinsWorkspace() {
    if (!hopkinsResident) {
        hopkinsResident = std::make_unique<HopkinsWorkspace>();
    }
    return *hopkinsResident;
}

// Order every neighbor reduction by stable identifier: resident slots are kept
// in increasing identifier order, so slot order is identifier order.
std::vector<size_t> hopkinsSlotsWhere(
    HopkinsWorkspace const& workspace, std::vector<unsigned char> const& flags) {
    std::vector<size_t> slots;
    for (size_t slot = 0; slot < workspace.particles.size(); ++slot) {
        if (flags[slot]) {
            slots.push_back(slot);
        }
    }
    return slots;
}

std::vector<size_t> hopkinsActiveSlots(HopkinsWorkspace const& workspace) {
    std::vector<size_t> slots;
    for (size_t slot = 0; slot < workspace.particles.size(); ++slot) {
        if (workspace.particles[slot].active) {
            slots.push_back(slot);
        }
    }
    return slots;
}

size_t hopkinsWorkerCount(size_t count) {
    return std::min(count, size_t(std::max(1, hpxHardwareConcurrency())));
}

// Run function(worker, i) over [0, count) on a fixed set of workers, so each
// worker can own scratch storage that is merged in worker order afterwards.
// No profile::ScopedPhase may be opened inside a worker: the profiler keeps a
// single phase stack.
template <class Function>
void hopkinsParallelWorkers(size_t count, size_t workers, Function const& function) {
    std::vector<hpx::future<void>> pending;
    for (size_t worker = 0; worker < workers; worker++) {
        pending.push_back(hpx::async([&, worker]() {
            for (size_t i = count * worker / workers; i < count * (worker + 1) / workers; i++) {
                function(worker, i);
            }
        }));
    }
    hpx::wait_all(pending);
    for (auto& future : pending) {
        future.get();
    }
}

template <class Function>
void hopkinsParallel(size_t count, Function const& function) {
    hopkinsParallelWorkers(
        count, hopkinsWorkerCount(count), [&](size_t, size_t i) { function(i); });
}

hopkins::Vector hopkinsSeparation(
    HopkinsParticle const& left, HopkinsParticle const& right, FlopCount* operations) {
    hopkins::detail::countFlops(operations, 3);
    hopkins::Vector separation{};
    for (int dim = 0; dim < 3; dim++) {
        // Subtract periodic fixed-point integers before converting to double;
        // converting absolute coordinates first would discard their low bits.
        separation[dim] = doubleDistance(right.position[dim], left.position[dim]);
    }
    return separation;
}

using HopkinsSimd = hopkins::SimdReal;
using HopkinsSimdVector = hopkins::SimdVector3;
using HopkinsSimdMatrix = std::array<HopkinsSimdVector, 3>;

struct HopkinsNeighborBatch {
    std::array<size_t, HopkinsSimd::size()> indices{};
    HopkinsSimdVector separation{};
    HopkinsSimd radius{};
    size_t count = 0;
};

HopkinsSimd hopkinsSimdDot(
    HopkinsSimdVector const& left, HopkinsSimdVector const& right, FlopCount* operations) {
    hopkins::detail::countFlops(operations, 5 * HopkinsSimd::size());
    return left.dot(right);
}

template <class Function>
HopkinsSimd hopkinsNeighborValues(HopkinsNeighborBatch const& batch, Function const& function) {
    return HopkinsSimd([&](auto lane) { return function(batch.indices[size_t(lane)]); });
}

// SIMD lanes evaluate independent neighbors. Reduce their contributions in the
// existing particle-ID order: support convergence, tiny sound waves and restart
// reproducibility must not depend on the compilation target's vector width.
template <class Function>
void hopkinsNeighborBatches(HopkinsWorkspace const& workspace, size_t i,
    std::vector<size_t> const& neighbors, FlopCount* operations, Function const& function,
    bool includeRadius = true) {
    for (size_t begin = 0; begin < neighbors.size(); begin += HopkinsSimd::size()) {
        HopkinsNeighborBatch batch;
        batch.count = std::min(size_t(HopkinsSimd::size()), neighbors.size() - begin);
        for (size_t lane = 0; lane < HopkinsSimd::size(); lane++) {
            // Padding refers to the physical center, with zero separation.
            // No gather can read past the neighbor list or particle array.
            batch.indices[lane] = lane < batch.count ? neighbors[begin + lane] : i;
        }
        for (int dim = 0; dim < 3; dim++) {
            batch.separation[dim] = hopkinsNeighborValues(batch, [&](size_t j) {
                // Preserve fixed-point subtraction before double conversion.
                return doubleDistance(
                    workspace.particles[j].position[dim], workspace.particles[i].position[dim]);
            });
        }
        hopkins::detail::countFlops(operations, 3 * HopkinsSimd::size());
        if (includeRadius) {
            hopkins::detail::countFlops(operations, HopkinsSimd::size());
            batch.radius = simdSqrt(hopkinsSimdDot(batch.separation, batch.separation, operations));
        }
        function(batch);
    }
}

// Transient symmetric stencil of cell i: queried from the tree for this call
// only. Consumers that make several passes over the same cell query once and
// use the explicit-list overload.
template <class Function>
void hopkinsNeighborBatches(HopkinsWorkspace const& workspace, size_t i, FlopCount* operations,
    Function const& function, bool includeRadius = true) {
    std::vector<size_t> neighbors;
    hopkinsNeighborQuery(workspace, i, 1.0, true, operations, neighbors);
    hopkinsNeighborBatches(workspace, i, neighbors, operations, function, includeRadius);
}

HopkinsSimdVector hopkinsRelativeVelocities(HopkinsWorkspace const& workspace, size_t i,
    HopkinsNeighborBatch const& batch, FlopCount* operations) {
    HopkinsSimdVector relative;
    hopkins::detail::countFlops(operations, 6 * HopkinsSimd::size());
    for (int dim = 0; dim < 3; dim++) {
        relative[dim] = (hopkinsNeighborValues(batch,
                             [&](size_t j) { return workspace.particles[j].velocity[dim]; }) -
                            HopkinsSimd(workspace.particles[i].velocity[dim])) /
            HopkinsSimd(workspace.scale);
    }
    return relative;
}

// The sound-wave validation scenario uses the normalized 1D cubic spline.
// All other scenarios retain the original selected 3D kernel and arithmetic.
double hopkinsKernelValue(double q, int dimensions, FlopCount* operations) {
    hopkins::detail::countFlops(
        operations, sph_flops::densityKernelFlops(selectedHydroKernel(), q) + (dimensions == 1));
    return dimensions == 1 ? (M_PI / 6.0) * kernelW(q) : kernelW(q);
}

double hopkinsKernelDerivative(double q, int dimensions, FlopCount* operations) {
    hopkins::detail::countFlops(operations,
        sph_flops::densityKernelFlops(selectedHydroKernel(), q, true) + (dimensions == 1));
    return dimensions == 1 ? (M_PI / 6.0) * dkernelWDq(q) : dkernelWDq(q);
}

HopkinsSimd hopkinsKernelValue(HopkinsSimd const& q, int dimensions, FlopCount* operations) {
    hopkins::detail::countFlops(operations,
        sph_flops::densityKernelFlops(selectedHydroKernel(), q) +
            (dimensions == 1 ? HopkinsSimd::size() : 0));
    auto const value = kernelW(q);
    return dimensions == 1 ? HopkinsSimd(M_PI / 6.0) * value : value;
}

HopkinsSimd hopkinsKernelDerivative(HopkinsSimd const& q, int dimensions, FlopCount* operations) {
    hopkins::detail::countFlops(operations,
        sph_flops::densityKernelFlops(selectedHydroKernel(), q, true) +
            (dimensions == 1 ? HopkinsSimd::size() : 0));
    auto const derivative = dkernelWDq(q);
    return dimensions == 1 ? HopkinsSimd(M_PI / 6.0) * derivative : derivative;
}

double hopkinsSupportVolume(double support, int dimensions, FlopCount* operations) {
    hopkins::detail::countFlops(operations, dimensions == 1 ? 0 : 2);
    return dimensions == 1 ? support : support * support * support;
}

double hopkinsNeighborTarget(int dimensions, FlopCount* operations) {
    hopkins::detail::countFlops(operations, dimensions == 1 ? 1 : 2);
    return dimensions == 1 ? getOptions().hydro.neighborNumber / 2.0 :
                             getOptions().hydro.neighborNumber * 3.0 / (4.0 * M_PI);
}

double hopkinsKernel(double radius, double support, int dimensions, FlopCount* operations) {
    hopkins::detail::countFlops(operations, radius < support ? 2 : 0);
    return radius < support ? hopkinsKernelValue(radius / support, dimensions, operations) /
            hopkinsSupportVolume(support, dimensions, operations) :
                              0.0;
}

HopkinsSimd hopkinsKernel(
    HopkinsSimd const& radius, double support, int dimensions, FlopCount* operations) {
    hopkins::detail::countFlops(operations, 2 * HopkinsSimd::size());
    return hopkinsKernelValue(radius / HopkinsSimd(support), dimensions, operations) /
        HopkinsSimd(hopkinsSupportVolume(support, dimensions, operations));
}

hopkins::Vector hopkinsAcceleration(
    Vector3<float> const& gravity, Vector3<Position> const& position, FlopCount* operations) {
    auto const& options = getOptions();
    // Fixed-point conversion multiplies and imposed-force additions, plus
    // float acceleration promotions when gravity supplies the stored force.
    hopkins::detail::countFlops(operations, 6 + (options.gravity.enabled ? 3 : 0));
    hopkins::Vector acceleration{};
    hopkins::Vector displacement{};
    for (int dim = 0; dim < 3; dim++) {
        acceleration[dim] =
            options.gravity.enabled && !options.run.hierarchicalGravity ? gravity[dim] : 0.0;
        displacement[dim] = doubleDistance(position[dim], Position(0.5));
    }
    acceleration[1] += options.hydro.yAcceleration;
    double const z = doubleDistance(position[2], Position(0.5));
    acceleration[2] += z < 0.0 ? options.hydro.zPlaneAcceleration :
                                 (z > 0.0 ? -options.hydro.zPlaneAcceleration : 0.0);
    if (options.hydro.centralAcceleration != 0.0) {
        double const support = options.hydro.centralSupportRadius;
        double const q = hopkins::norm(displacement, operations) / support;
        hopkins::detail::countFlops(
            operations, 5 + sph_flops::forceKernelFlops(selectedGravityKernel(), q));
        double const force = options.hydro.centralAcceleration *
            radial_kernel::forceFactor(selectedGravityKernel(), q) / (support * support * support);
        for (int dim = 0; dim < 3; dim++) {
            hopkins::detail::countFlops(operations, 2);
            acceleration[dim] -= force * displacement[dim];
        }
    }
    return acceleration;
}

// Every input the owner record is derived from. Padding is zeroed so the key
// compares bytewise; a record is reported only when its key changed.
struct HopkinsOwnerKey {
    std::array<unsigned char, sizeof(HopkinsParticleState)> state{};
    Vector3<Position> position{};
    Vector3<float> velocity{};
    float mass = 0.0f;
    float energy = 0.0f;
    float support = 0.0f;
    int rung = 0;
    Vector3<float> gravity{};
    gadget4::State chemistry{};
    StarFormationState starFormation{};
};

// Owner-side mirror of what the coordinator last saw for every local gas slot.
std::vector<uint64_t> hopkinsOwnerIdentifiers;
std::vector<HopkinsOwnerKey> hopkinsOwnerKeys;

struct HopkinsDelta {
    std::vector<ParticleInteger> indices;
    std::vector<HopkinsParticle> particles;
    std::vector<uint64_t> identifiers;
    bool membershipChanged = false;
    ParticleInteger count = 0;
    int darkMatterRung = 0;
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & indices & particles & identifiers & membershipChanged & count & darkMatterRung;
    }
};

bool hopkinsSameKey(HopkinsOwnerKey const& left, HopkinsOwnerKey const& right) {
    return std::memcmp(&left, &right, sizeof(HopkinsOwnerKey)) == 0;
}

HopkinsOwnerKey hopkinsOwnerKey(ParticleInteger i, Vector3<float> const& gravity) {
    HopkinsOwnerKey key;
    std::memset(&key, 0, sizeof(key));
    auto state = sphParticlesHopkinsState(i);
    hopkinsCanonicalizePadding(state);
    std::memcpy(key.state.data(), &state, sizeof(state));
    bool const hierarchical = getOptions().run.hierarchicalGravity;
    for (int dim = 0; dim < 3; ++dim) {
        if (!(hierarchical && state.predictorInitialized)) {
            key.position[dim] = sphParticlesPosition(dim, i);
        }
        if (!state.initialized) {
            key.velocity[dim] = sphParticlesVelocity(dim, i);
        }
        key.gravity[dim] = gravity[dim];
    }
    if (!state.initialized) {
        key.mass = sphParticlesMass(i);
        key.energy = sphParticlesInternalEnergy(i);
    }
    if (!(state.supportRadius > 0.0)) {
        key.support = sphParticlesSupportRadius(i);
    }
    key.rung = sphParticlesRung(i);
    if (getOptions().chemistry.enabled) {
        key.chemistry = sphParticlesChemistryStates[i];
    }
    if (getOptions().starFormation.enabled) {
        key.starFormation = particlesStarFormationStates[sphParticlesParticleIndex(i)];
    }
    return key;
}

HopkinsParticle hopkinsOwnerRecord(
    ParticleInteger i, double time, Vector3<float> const& gravity, FlopCount* operations) {
    HopkinsParticle particle;
    particle.state = sphParticlesHopkinsState(i);
    hopkins::detail::countFlops(
        operations, particle.state.initialized ? 0 : (getOptions().hydro.method == "mfv" ? 1 : 2));
    particle.mass = particle.state.initialized ? particle.state.mass : sphParticlesMass(i);
    hopkins::detail::countFlops(operations, particle.state.initialized ? 5 : 1);
    particle.energy = particle.state.initialized ? particle.state.totalEnergy / particle.mass -
            0.5 * hopkins::dot(particle.state.momentum, particle.state.momentum, operations) /
                (particle.mass * particle.mass) :
                                                   sphParticlesInternalEnergy(i);
    hopkins::detail::countFlops(operations, particle.state.supportRadius > 0.0 ? 0 : 1);
    particle.support = particle.state.supportRadius > 0.0 ? particle.state.supportRadius :
                                                            sphParticlesSupportRadius(i);
    particle.gravityRung = sphParticlesRung(i);
    bool const hierarchical = getOptions().run.hierarchicalGravity;
    particle.rung = hierarchical && particle.state.initialized ? particle.state.hydroRung :
                                                                 particle.gravityRung;
    if (getOptions().chemistry.enabled) {
        particle.chemistry = sphParticlesChemistryStates[i];
    }
    if (getOptions().starFormation.enabled) {
        particle.starFormation = particlesStarFormationStates[sphParticlesParticleIndex(i)];
    }
    for (int dim = 0; dim < 3; dim++) {
        particle.position[dim] = hierarchical && particle.state.predictorInitialized ?
            particle.state.positionAt(dim, time) :
            sphParticlesPosition(dim, i);
        hopkins::detail::countFlops(operations, 1);
        particle.velocity[dim] = particle.state.initialized ?
            particle.state.momentum[dim] / particle.mass :
            sphParticlesVelocity(dim, i);
    }
    particle.acceleration = hopkinsAcceleration(gravity, particle.position, operations);
    return particle;
}

// Report the records whose inputs changed since the coordinator last saw them.
// A live gravity force may not be read while the kick runs; without readForces
// the previously mirrored force is reused, so a changed force is reported by the
// next collection that may read it. Membership changes carry the complete local
// identifier list; the keys are realigned by identifier across migrations.
HopkinsDelta hopkinsCollectDelta(bool countOperations, bool readForces, double time) {
    HopkinsFlopBatch operations(countOperations);
    HopkinsDelta delta;
    delta.count = sphParticlesSize();
    if (readForces) {
        for (ParticleInteger i = 0; i < particlesSize(); i++) {
            if (particlesType(i) == kDarkMatterType) {
                delta.darkMatterRung = std::max(delta.darkMatterRung, int(particlesRung(i)));
            }
        }
    }
    size_t const count = size_t(delta.count);
    bool membershipChanged = hopkinsOwnerIdentifiers.size() != count;
    for (size_t i = 0; !membershipChanged && i < count; ++i) {
        membershipChanged = hopkinsOwnerIdentifiers[i] != sphParticlesHopkinsState(i).identifier;
    }
    std::unordered_map<uint64_t, size_t> previous;
    if (membershipChanged) {
        previous.reserve(hopkinsOwnerIdentifiers.size());
        for (size_t i = 0; i < hopkinsOwnerIdentifiers.size(); ++i) {
            if (hopkinsOwnerIdentifiers[i] != std::numeric_limits<uint64_t>::max()) {
                previous.emplace(hopkinsOwnerIdentifiers[i], i);
            }
        }
    }
    std::vector<HopkinsOwnerKey> keys(count);
    std::vector<uint64_t> identifiers(count);
    for (size_t i = 0; i < count; ++i) {
        identifiers[i] = sphParticlesHopkinsState(i).identifier;
        HopkinsOwnerKey const* mirrored = nullptr;
        if (!membershipChanged) {
            mirrored = &hopkinsOwnerKeys[i];
        } else if (auto const found = previous.find(identifiers[i]); found != previous.end()) {
            mirrored = &hopkinsOwnerKeys[found->second];
        }
        Vector3<float> gravity{};
        for (int dim = 0; dim < 3; ++dim) {
            gravity[dim] = readForces ? sphParticlesGravityAcceleration(dim, i) :
                                        (mirrored != nullptr ? mirrored->gravity[dim] : 0.0f);
        }
        keys[i] = hopkinsOwnerKey(i, gravity);
        if (mirrored == nullptr || !hopkinsSameKey(*mirrored, keys[i]) ||
            identifiers[i] == std::numeric_limits<uint64_t>::max()) {
            delta.indices.push_back(ParticleInteger(i));
            delta.particles.push_back(hopkinsOwnerRecord(i, time, gravity, operations));
        }
    }
    delta.membershipChanged = membershipChanged;
    if (membershipChanged) {
        delta.identifiers = identifiers;
    }
    hopkinsOwnerIdentifiers = std::move(identifiers);
    hopkinsOwnerKeys = std::move(keys);
    return delta;
}

void hopkinsResetOwnerMirror() {
    hopkinsOwnerIdentifiers.clear();
    hopkinsOwnerKeys.clear();
}

// Publish stable IDs before any persistent neighbor query. This operation changes
// no conserved state, reconstruction anchor, support, velocity, or particle order.
void hopkinsInstallIdentifiers(std::vector<uint64_t> const& identifiers) {
    if (identifiers.size() != size_t(sphParticlesSize())) {
        throw std::runtime_error("Hopkins identity publication changed particle membership");
    }
    for (size_t index = 0; index < identifiers.size(); ++index) {
        auto const identifier = sphParticlesHopkinsState(index).identifier;
        if (identifier != std::numeric_limits<uint64_t>::max() &&
            identifier != identifiers[index]) {
            throw std::runtime_error("Hopkins identity changed during source preparation");
        }
    }
    for (size_t index = 0; index < identifiers.size(); ++index) {
        sphParticlesHopkinsState(index).identifier = identifiers[index];
        if (index < hopkinsOwnerIdentifiers.size()) {
            hopkinsOwnerIdentifiers[index] = identifiers[index];
            std::memcpy(
                hopkinsOwnerKeys[index].state.data() + offsetof(HopkinsParticleState, identifier),
                &identifiers[index], sizeof(uint64_t));
        }
    }
}

// Predicted public views of a sleeping cell: the same formula the coordinator
// applies in hopkinsCurrentViews, evaluated by the owner without communication.
hopkins::Primitive hopkinsPredictedPrimitive(
    HopkinsParticleState const& state, double time, double scale, FlopCount* operations);

// Install the coordinator's refreshed records, then refresh the predicted public
// density, velocity and internal-energy mirrors of every other initialized cell
// locally. The ordinary drifter consumes only the separately stored mesh velocity.
// Physical mass, momentum and energy are never temporarily overwritten by it.
void hopkinsInstallSparse(std::vector<ParticleInteger> const& indices,
    std::vector<HopkinsParticle> const& particles, int phase, double time, double scale,
    bool countOperations) {
    HopkinsFlopBatch operations(countOperations);
    ALWAYS_ASSERT(indices.size() == particles.size());
    bool const hierarchical = getOptions().run.hierarchicalGravity;
    std::vector<unsigned char> installed(size_t(sphParticlesSize()), 0);
    std::vector<std::pair<ParticleInteger, size_t>> converted;
    for (size_t k = 0; k < indices.size(); k++) {
        ParticleInteger const i = indices[k];
        if (i < 0 || i >= sphParticlesSize() || installed[size_t(i)]) {
            throw std::runtime_error("Hopkins install addressed an invalid gas slot");
        }
        installed[size_t(i)] = 1;
        auto const& particle = particles[k];
        if (sphParticlesHopkinsState(i).identifier != particle.state.identifier) {
            throw std::runtime_error("Hopkins install addressed a different particle");
        }
        hopkins::detail::countFlops(operations, 2);
        sphParticlesSupportRadius(i) = particle.support;
        sphParticlesDensity(i) = particle.density;
        if (phase != 0) {
            if (!hierarchical) {
                sphParticlesRung(i) = particle.rung;
            }
            sphParticlesHopkinsState(i) = particle.state;
            if (hierarchical) {
                auto& state = sphParticlesHopkinsState(i);
                state.hydroRung = particle.rung;
                if (phase == 1) {
                    state.hierarchicalActive = false;
                    for (int dim = 0; dim < 3; ++dim) {
                        // This future gravity coordinate is only a projection.
                        // Every intervening hydro wake refreshes it before the
                        // inactive prefix can enter a gravity tree again.
                        sphParticlesPosition(dim, i) = state.positionAt(dim, state.gravityEndTime);
                    }
                }
            }
            if (getOptions().hydro.method == "mfv") {
                hopkins::detail::countFlops(operations, 1);
                sphParticlesMfvMass(i) = particle.state.accountedMass();
            }
            hopkins::detail::countFlops(operations, 4);
            sphParticlesInternalEnergy(i) = particle.energy;
            for (int dim = 0; dim < 3; dim++) {
                sphParticlesVelocity(dim, i) = particle.velocity[dim];
            }
            if (getOptions().chemistry.enabled) {
                sphParticlesChemistryStates[i] = particle.chemistry;
            }
            if (getOptions().starFormation.enabled) {
                particlesStarFormationStates[sphParticlesParticleIndex(i)] = particle.starFormation;
                if (particle.convert) {
                    converted.emplace_back(i, k);
                }
            }
        }
        sphParticlesHopkinsState(i).supportRadius = particle.support;
        if (size_t(i) < hopkinsOwnerKeys.size()) {
            Vector3<float> gravity{};
            for (int dim = 0; dim < 3; ++dim) {
                gravity[dim] = sphParticlesGravityAcceleration(dim, i);
            }
            hopkinsOwnerKeys[i] = hopkinsOwnerKey(i, gravity);
        }
    }
    if (phase == 0) {
        return;
    }
    double const gamma = getOptions().hydro.gamma;
    for (ParticleInteger i = 0; i < sphParticlesSize(); ++i) {
        auto const& state = sphParticlesHopkinsState(i);
        if (installed[size_t(i)] || !state.initialized || !state.predictorInitialized) {
            continue;
        }
        auto const primitive = hopkinsPredictedPrimitive(state, time, scale, nullptr);
        if (!hopkins::physical(primitive)) {
            throw std::runtime_error("Nonphysical Hopkins own-step primitive");
        }
        hopkins::detail::countFlops(operations, 6);
        sphParticlesDensity(i) = primitive.density;
        sphParticlesInternalEnergy(i) = primitive.pressure / ((gamma - 1.0) * primitive.density);
        for (int dim = 0; dim < 3; dim++) {
            sphParticlesVelocity(dim, i) = primitive.velocity[dim];
        }
    }
    if (converted.empty()) {
        return;
    }
#if COSMICTIGER_ENABLE_STAR_FORMATION
    // Remove converted gas last and highest index first: each removal swaps the
    // current last particle, whose installed state is already complete, into the
    // slot. The owner mirror realigns itself by identifier at the next collection,
    // and the coordinator drops the vanished identifiers and rebuilds its topology.
    std::sort(converted.begin(), converted.end());
    for (auto entry = converted.rbegin(); entry != converted.rend(); ++entry) {
        starFormationConvertGas(entry->first, particles[entry->second].starFormation);
    }
#else
    throw std::logic_error("Star formation requires COSMICTIGER_ENABLE_STARS=ON");
#endif
}

void hopkinsInstallGravityMasses(std::vector<std::pair<uint64_t, double>> const& partialMasses) {
    for (ParticleInteger i = 0; i < sphParticlesSize(); ++i) {
        auto const& state = sphParticlesHopkinsState(i);
        if (!state.initialized || state.identifier == std::numeric_limits<uint64_t>::max()) {
            continue;
        }
        auto const found =
            std::lower_bound(partialMasses.begin(), partialMasses.end(), state.identifier,
                [](auto const& entry, uint64_t identifier) { return entry.first < identifier; });
        double const partial =
            found != partialMasses.end() && found->first == state.identifier ? found->second : 0.0;
        double const mass = state.accountedMass() + partial;
        if (!(mass > 0.0) || !(float(mass) > 0.0f) || !std::isfinite(float(mass))) {
            throw std::runtime_error("Invalid Hopkins MFV gravity source mass projection");
        }
        sphParticlesMfvMass(i) = mass;
    }
}

}    // namespace detail

HPX_PLAIN_ACTION(detail::hopkinsCollectDelta, HopkinsCollectDeltaAction);
HPX_PLAIN_ACTION(detail::hopkinsResetOwnerMirror, HopkinsResetOwnerMirrorAction);
HPX_PLAIN_ACTION(detail::hopkinsInstallIdentifiers, HopkinsInstallIdentifiersAction);
HPX_PLAIN_ACTION(detail::hopkinsInstallSparse, HopkinsInstallSparseAction);
HPX_PLAIN_ACTION(detail::hopkinsInstallGravityMasses, HopkinsInstallGravityMassesAction);

namespace detail {

void hopkinsResizeSlots(HopkinsWorkspace& workspace) {
    size_t const count = workspace.particles.size();
    workspace.owners.resize(count);
    workspace.geometry.assign(count, {});
    workspace.geometryRates.assign(count, {});
    workspace.supportRates.assign(count, 0.0);
    workspace.primitives.assign(count, {});
    workspace.gradients.assign(count, {});
    workspace.dirty.assign(count, 0);
    workspace.solved.assign(count, 0);
    workspace.built.assign(count, 0);
    workspace.touched.assign(count, 0);
    workspace.stencilsFrozen = false;
}

void hopkinsClearResident(HopkinsWorkspace& workspace) {
    workspace.particles.clear();
    workspace.rankIdentifiers.assign(size_t(hpxSize()), {});
    workspace.rankSlots.assign(size_t(hpxSize()), {});
    workspace.indices.clear();
    workspace.tree.clear();
    hopkinsResizeSlots(workspace);
}

enum class HopkinsMembership { Unchanged, Rebuilt, ResetRequired };

// Assign identifiers to new gas, publish them, and rebuild the slot order after
// any locality reported a membership change. Slots stay in identifier order;
// an owner change alone keeps every resident record and the tree topology.
// ResetRequired is reserved for an owner mirror that outlived the resident set.
HopkinsMembership hopkinsRebuildMembership(
    HopkinsWorkspace& workspace, std::vector<HopkinsDelta>& deltas) {
    size_t const ranks = workspace.rankIdentifiers.size();
    for (auto const& identifiers : workspace.rankIdentifiers) {
        for (uint64_t identifier : identifiers) {
            if (identifier != std::numeric_limits<uint64_t>::max()) {
                workspace.nextIdentifier = std::max(workspace.nextIdentifier, identifier + 1);
            }
        }
    }
    std::vector<hpx::future<void>> publications;
    try {
        for (size_t rank = 0; rank < ranks; ++rank) {
            auto& identifiers = workspace.rankIdentifiers[rank];
            std::unordered_map<ParticleInteger, size_t> reported;
            for (size_t k = 0; k < deltas[rank].indices.size(); ++k) {
                reported.emplace(deltas[rank].indices[k], k);
            }
            bool assigned = false;
            for (size_t index = 0; index < identifiers.size(); ++index) {
                if (identifiers[index] != std::numeric_limits<uint64_t>::max()) {
                    continue;
                }
                if (workspace.nextIdentifier == std::numeric_limits<uint64_t>::max()) {
                    throw std::overflow_error("Hopkins stable particle identifiers are exhausted");
                }
                identifiers[index] = workspace.nextIdentifier++;
                auto const found = reported.find(ParticleInteger(index));
                if (found == reported.end()) {
                    throw std::runtime_error("Hopkins new particle was not reported");
                }
                deltas[rank].particles[found->second].state.identifier = identifiers[index];
                assigned = true;
            }
            if (assigned) {
                publications.push_back(hpx::async<HopkinsInstallIdentifiersAction>(
                    hpxLocalities()[rank], identifiers));
            }
        }
    } catch (...) {
        hpx::wait_all(publications);
        throw;
    }
    hpx::wait_all(publications);
    for (auto& publication : publications) {
        publication.get();
    }
    struct Entry {
        uint64_t identifier;
        HopkinsOwner owner;
    };
    std::vector<Entry> entries;
    for (size_t rank = 0; rank < ranks; ++rank) {
        auto const& identifiers = workspace.rankIdentifiers[rank];
        for (size_t index = 0; index < identifiers.size(); ++index) {
            entries.push_back({identifiers[index], {int(rank), ParticleInteger(index)}});
        }
    }
    std::sort(entries.begin(), entries.end(),
        [](Entry const& left, Entry const& right) { return left.identifier < right.identifier; });
    for (size_t k = 1; k < entries.size(); ++k) {
        if (entries[k - 1].identifier == entries[k].identifier) {
            throw std::runtime_error("Duplicate Hopkins stable particle identifier");
        }
    }
    bool sameSet = entries.size() == workspace.particles.size();
    for (size_t slot = 0; sameSet && slot < entries.size(); ++slot) {
        sameSet = workspace.particles[slot].state.identifier == entries[slot].identifier;
    }
    if (!sameSet) {
        // Identifiers that no owner reports any more (gas converted to stars)
        // are dropped; a live pair interval referencing one is detected by the
        // settle step. New identifiers get default records filled by the deltas.
        std::vector<HopkinsParticle> particles;
        particles.reserve(entries.size());
        for (auto const& entry : entries) {
            auto const found = workspace.indices.find(entry.identifier);
            if (found != workspace.indices.end()) {
                particles.push_back(workspace.particles[found->second]);
            } else {
                particles.emplace_back();
                particles.back().state.identifier = entry.identifier;
            }
        }
        workspace.particles = std::move(particles);
        workspace.indices.clear();
        workspace.indices.reserve(entries.size());
        for (size_t slot = 0; slot < entries.size(); ++slot) {
            workspace.indices.emplace(entries[slot].identifier, slot);
        }
        hopkinsResizeSlots(workspace);
    }
    workspace.rankSlots.assign(ranks, {});
    for (size_t rank = 0; rank < ranks; ++rank) {
        workspace.rankSlots[rank].resize(workspace.rankIdentifiers[rank].size());
    }
    for (size_t slot = 0; slot < entries.size(); ++slot) {
        workspace.owners[slot] = entries[slot].owner;
        workspace.rankSlots[size_t(entries[slot].owner.rank)][size_t(entries[slot].owner.index)] =
            slot;
    }
    return sameSet ? HopkinsMembership::Unchanged : HopkinsMembership::Rebuilt;
}

// Import the records whose inputs changed since the previous event. Returns
// true when the resident identifier set changed and the tree must be rebuilt.
bool hopkinsRefreshRecords(HopkinsWorkspace& workspace, bool readForces) {
    profile::ScopedPhase profilePhase("hydro.hopkins.gather");
    bool rebuilt = false;
    for (int attempt = 0; attempt < 2; ++attempt) {
        std::vector<hpx::future<HopkinsDelta>> pending;
        for (auto const& locality : hpxLocalities()) {
            pending.push_back(hpx::async<HopkinsCollectDeltaAction>(
                locality, workspace.countOperations, readForces, workspace.time));
        }
        std::vector<HopkinsDelta> deltas;
        for (auto& future : pending) {
            deltas.push_back(future.get());
        }
        if (workspace.rankIdentifiers.size() != size_t(hpxSize())) {
            hopkinsClearResident(workspace);
        }
        if (readForces) {
            workspace.darkMatterRung = 0;
            for (auto const& delta : deltas) {
                workspace.darkMatterRung = std::max(workspace.darkMatterRung, delta.darkMatterRung);
            }
        }
        bool membershipChanged = workspace.particles.empty();
        bool resetRequired = false;
        for (size_t rank = 0; rank < deltas.size(); ++rank) {
            auto const& delta = deltas[rank];
            if (delta.membershipChanged) {
                workspace.rankIdentifiers[rank] = delta.identifiers;
                membershipChanged = true;
                workspace.work.bytesReceived += delta.identifiers.size() * sizeof(uint64_t);
            } else if (workspace.rankIdentifiers[rank].size() != size_t(delta.count)) {
                // The owner mirror outlived the resident set (a coordinator
                // reset): every owner must report its complete membership.
                resetRequired = true;
            }
        }
        if (membershipChanged && !resetRequired) {
            auto const outcome = hopkinsRebuildMembership(workspace, deltas);
            resetRequired = outcome == HopkinsMembership::ResetRequired;
            rebuilt = outcome == HopkinsMembership::Rebuilt;
        }
        if (resetRequired) {
            if (attempt != 0) {
                throw std::runtime_error("Hopkins resident membership could not be rebuilt");
            }
            hopkinsClearResident(workspace);
            std::vector<hpx::future<void>> resets;
            for (auto const& locality : hpxLocalities()) {
                resets.push_back(hpx::async<HopkinsResetOwnerMirrorAction>(locality));
            }
            hpx::wait_all(resets);
            for (auto& reset : resets) {
                reset.get();
            }
            continue;
        }
        for (size_t rank = 0; rank < deltas.size(); ++rank) {
            auto const& delta = deltas[rank];
            for (size_t k = 0; k < delta.indices.size(); ++k) {
                size_t const slot = workspace.rankSlots[rank].at(size_t(delta.indices[k]));
                auto record = delta.particles[k];
                auto& resident = workspace.particles[slot];
                if (resident.state.identifier != record.state.identifier) {
                    throw std::runtime_error("Hopkins owner record identity mismatch");
                }
                if (workspace.solved[slot]) {
                    // The coordinator owns a support it solved at this event;
                    // the owner's copy is only installed when the event closes.
                    record.support = resident.support;
                    record.state.supportRadius = resident.state.supportRadius;
                }
                if (workspace.stencilsFrozen || workspace.built[slot] || workspace.solved[slot]) {
                    bool moved = false;
                    for (int dim = 0; dim < 3; ++dim) {
                        moved = moved || resident.position[dim] != record.position[dim];
                    }
                    if (moved) {
                        throw std::runtime_error(
                            "Hopkins resident record changed position within an event");
                    }
                }
                resident = record;
                workspace.dirty[slot] = 1;
            }
            workspace.work.recordsReceived += delta.indices.size();
            workspace.work.bytesReceived +=
                delta.indices.size() * (sizeof(HopkinsParticle) + sizeof(ParticleInteger));
        }
        return rebuilt;
    }
    throw std::logic_error("Hopkins record refresh did not complete");
}

// Clean records carry their previous inputs; only their predicted hierarchical
// position and the position-dependent imposed acceleration move with time.
void hopkinsDeriveRecords(HopkinsWorkspace& workspace) {
    if (!getOptions().run.hierarchicalGravity) {
        return;
    }
    HopkinsFlopBatch operations(workspace.countOperations);
    Vector3<float> const noGravity{};
    for (size_t slot = 0; slot < workspace.particles.size(); ++slot) {
        auto& particle = workspace.particles[slot];
        if (workspace.dirty[slot] || !particle.state.predictorInitialized) {
            continue;
        }
        for (int dim = 0; dim < 3; ++dim) {
            particle.position[dim] = particle.state.positionAt(dim, workspace.time);
        }
        particle.acceleration = hopkinsAcceleration(noGravity, particle.position, operations);
    }
}

// A sleeping hierarchical cell follows its retained mesh trajectory until its
// own endpoint; every other record is anchored statically at the current event
// time and is refitted when its owner reports a changed record.
hydro_source::Source hopkinsSourceOf(HopkinsWorkspace const& workspace, size_t slot) {
    auto const& particle = workspace.particles[slot];
    auto const& state = particle.state;
    hydro_source::Source source;
    source.id = state.identifier;
    source.rung = particle.rung;
    // The broad phase must enclose the exact Fixed64 pair test even at
    // periodic faces and when support solving expands the reverse stencil.
    source.supportUpper =
        std::nextafter(particle.support, std::numeric_limits<double>::infinity()) +
        64.0 * std::numeric_limits<double>::epsilon();
    bool const trajectory = getOptions().run.hierarchicalGravity && state.predictorInitialized &&
        !particle.active && state.positionDriftScale > 0.0;
    if (trajectory) {
        source.epoch = state.positionEpoch;
        source.validUntil = std::max(workspace.time,
            state.endTime +
                64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, state.endTime));
        source.validUntil = std::max(source.validUntil, source.epoch);
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            source.position[dimension] = state.positionAnchor[dimension].toDouble();
            source.velocity[dimension] = state.meshVelocity[dimension] / state.positionDriftScale;
            source.acceleration[dimension] =
                state.meshAcceleration[dimension] / state.positionDriftScale;
        }
    } else {
        // A static anchor stays valid until its record changes: a clean record
        // whose position did not move is never refitted at a later event.
        source.epoch = workspace.time;
        source.validUntil = std::numeric_limits<double>::max();
        for (size_t dimension = 0; dimension < 3; ++dimension) {
            source.position[dimension] = particle.position[dimension].toDouble();
        }
    }
    return source;
}

void hopkinsUpdateSources(HopkinsWorkspace& workspace, std::vector<size_t> const& slots) {
    if (slots.empty()) {
        return;
    }
    profile::ScopedPhase treePhase("hydro.hopkins.tree.refresh");
    std::vector<hydro_source::Source> sources;
    sources.reserve(slots.size());
    for (size_t slot : slots) {
        sources.push_back(hopkinsSourceOf(workspace, slot));
    }
    workspace.tree.update(sources);
    workspace.work.sourcesUpdated += sources.size();
}

void hopkinsRebuildSources(HopkinsWorkspace& workspace) {
    profile::ScopedPhase treePhase("hydro.hopkins.tree.refresh");
    std::vector<hydro_source::Source> sources;
    sources.reserve(workspace.particles.size());
    for (size_t slot = 0; slot < workspace.particles.size(); ++slot) {
        sources.push_back(hopkinsSourceOf(workspace, slot));
    }
    workspace.tree.reset(std::move(sources));
    workspace.work.sourcesUpdated += workspace.particles.size();
}

bool hopkinsNeighborChecksEnabled() {
    static bool const enabled = std::getenv("COSMICTIGER_HOPKINS_CHECK_NEIGHBORS") != nullptr;
    return enabled;
}

bool hopkinsStencilMember(HopkinsWorkspace const& workspace, size_t i, size_t j, double factor,
    bool symmetric, FlopCount* operations) {
    auto const& particle = workspace.particles[i];
    auto const& other = workspace.particles[j];
    hopkins::detail::countFlops(operations, symmetric ? 2 : 1);
    double const separation =
        hopkins::norm(hopkinsSeparation(particle, other, operations), operations);
    return separation < factor * particle.support || (symmetric && separation < other.support);
}

// Diagnostic cross-check against every resident record, enabled by the
// COSMICTIGER_HOPKINS_CHECK_NEIGHBORS environment variable: a missing member
// means the tree's supportUpper of some cell lagged behind its solved support.
void hopkinsCheckNeighborQuery(HopkinsWorkspace const& workspace, size_t i, double factor,
    bool symmetric, std::vector<size_t> const& neighbors) {
    std::vector<size_t> expected;
    for (size_t j = 0; j < workspace.particles.size(); ++j) {
        if (j != i && hopkinsStencilMember(workspace, i, j, factor, symmetric, nullptr)) {
            expected.push_back(j);
        }
    }
    if (expected != neighbors) {
        throw std::runtime_error("Hopkins stencil query of cell " +
            std::to_string(workspace.particles[i].state.identifier) + " returned " +
            std::to_string(neighbors.size()) + " members; brute force finds " +
            std::to_string(expected.size()));
    }
}

// Cells within factor times this cell's own support (symmetric == false), or
// within the larger of the two supports (symmetric == true). The broad phase
// returns every source with r <= max(radius, supportUpper_j), so one query
// also yields the reverse members, provided the tree's supportUpper is current
// for every cell whose support moved; the exact Fixed64 test decides membership,
// and r < max(h_i, h_j) holds exactly when either own test holds. Nothing is
// stored between calls; slots come back in identifier order.
void hopkinsNeighborQuery(HopkinsWorkspace const& workspace, size_t i, double factor,
    bool symmetric, FlopCount* operations, std::vector<size_t>& neighbors) {
    auto const& particle = workspace.particles[i];
    hydro_source::Vector point{};
    for (size_t dimension = 0; dimension < 3; ++dimension) {
        point[dimension] = particle.position[dimension].toDouble();
    }
    hydro_source::Box const sink{point, point};
    double const radius =
        std::nextafter(factor * particle.support, std::numeric_limits<double>::infinity()) +
        64.0 * std::numeric_limits<double>::epsilon();
    neighbors.clear();
    for (uint64_t identifier : workspace.tree.query(sink, radius, workspace.time)) {
        size_t const j = workspace.indices.at(identifier);
        if (j != i && hopkinsStencilMember(workspace, i, j, factor, symmetric, operations)) {
            neighbors.push_back(j);
        }
    }
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    if (hopkinsNeighborChecksEnabled()) {
        hopkinsCheckNeighborQuery(workspace, i, factor, symmetric, neighbors);
    }
}

// First-ring halo of the active cells: every inactive member of an active
// cell's symmetric stencil. Workers collect candidates and the merge keeps slot
// (identifier) order, so the result does not depend on the thread count.
std::vector<size_t> hopkinsHaloOf(
    HopkinsWorkspace const& workspace, std::vector<size_t> const& active) {
    size_t const workers = hopkinsWorkerCount(active.size());
    std::vector<std::vector<size_t>> found(workers);
    hopkinsParallelWorkers(active.size(), workers, [&](size_t worker, size_t k) {
        HopkinsFlopBatch operations(workspace.countOperations);
        std::vector<size_t> neighbors;
        hopkinsNeighborQuery(workspace, active[k], 1.0, true, operations, neighbors);
        for (size_t j : neighbors) {
            if (!workspace.particles[j].active) {
                found[worker].push_back(j);
            }
        }
    });
    std::vector<unsigned char> inHalo(workspace.particles.size(), 0);
    for (auto const& list : found) {
        for (size_t j : list) {
            inHalo[j] = 1;
        }
    }
    return hopkinsSlotsWhere(workspace, inHalo);
}

// Solve the constant kernel-number support of the given cells only. Returns
// the cells whose support changed; their symmetric lists must be repaired.
std::vector<size_t> hopkinsSolveSupports(
    HopkinsWorkspace& workspace, std::vector<size_t> const& cells) {
    profile::ScopedPhase profilePhase("hydro.hopkins.support");
    HopkinsFlopBatch operations(workspace.countOperations);
    auto const& options = getOptions();
    double const minimum = options.hydro.minSupportRadius;
    double const maximum = options.hydro.maxSupportRadius;
    double const target = hopkinsNeighborTarget(workspace.dimensions, operations);
    std::vector<double> entry(cells.size()), lower(cells.size(), minimum),
        upper(cells.size(), maximum);
    for (size_t k = 0; k < cells.size(); ++k) {
        entry[k] = workspace.particles[cells[k]].support;
        workspace.solved[cells[k]] = 1;
        workspace.touched[cells[k]] = 1;
    }
    workspace.work.supportCells += cells.size();
    std::vector<size_t> pending(cells.size());
    std::iota(pending.begin(), pending.end(), size_t(0));
    for (int iteration = 0; iteration < 256 && !pending.empty(); iteration++) {
        for (size_t k : pending) {
            auto const& particle = workspace.particles[cells[k]];
            // Only cells inside the bounds query; a clamped cell just moves.
            workspace.work.listQueries +=
                std::clamp(particle.support, minimum, maximum) == particle.support;
        }
        std::vector<unsigned char> changed(pending.size(), 0);
        {
            profile::ScopedPhase profilePhase("hydro.hopkins.support.solve");
            hopkinsParallel(pending.size(), [&](size_t p) {
                HopkinsFlopBatch operations(workspace.countOperations);
                size_t const k = pending[p];
                auto& particle = workspace.particles[cells[k]];
                double const support = std::clamp(particle.support, minimum, maximum);
                double next = support;
                if (support == particle.support) {
                    // Own candidates at the search factor. The test is one-sided,
                    // so no tree refit is needed between solve iterations.
                    std::vector<size_t> candidates;
                    hopkinsNeighborQuery(workspace, cells[k], kHopkinsSolveSearchFactor, false,
                        operations, candidates);
                    double sum = hopkinsKernelValue(0.0, workspace.dimensions, operations),
                           derivative = 0.0;
                    hopkinsNeighborBatches(workspace, cells[k], candidates, operations,
                        [&](HopkinsNeighborBatch const& batch) {
                            hopkins::detail::countFlops(operations, 3 * HopkinsSimd::size());
                            auto const q = batch.radius / HopkinsSimd(support);
                            auto const values =
                                hopkinsKernelValue(q, workspace.dimensions, operations);
                            auto const derivatives = q *
                                hopkinsKernelDerivative(q, workspace.dimensions, operations) /
                                HopkinsSimd(support);
                            for (size_t lane = 0; lane < batch.count; lane++) {
                                if (q[lane] < 1.0) {
                                    hopkins::detail::countFlops(operations, 2);
                                    sum += values[lane];
                                    derivative -= derivatives[lane];
                                }
                            }
                        });
                    hopkins::detail::countFlops(operations, 2);
                    double const residual = sum - target;
                    if (std::abs(residual) <= kHopkinsSupportTolerance * target ||
                        (support <= minimum && residual >= 0.0) ||
                        (support >= maximum && residual <= 0.0)) {
                        return;
                    }
                    next = hopkins::supportRadiusStep(support, residual, derivative, minimum,
                        maximum, lower[k], upper[k], false, operations);
                }
                // Retain the solved double radius; the own list is requeried at
                // the new radius before the kernel sum is repeated.
                changed[p] = next != particle.support;
                particle.support = next;
            });
        }
        workspace.work.supportSums += pending.size();
        std::vector<size_t> next;
        for (size_t p = 0; p < pending.size(); ++p) {
            if (changed[p]) {
                next.push_back(pending[p]);
            }
        }
        pending = std::move(next);
    }
    if (!pending.empty()) {
        throw std::runtime_error("Hopkins kernel-support solve did not converge");
    }
    std::vector<size_t> moved;
    for (size_t k = 0; k < cells.size(); ++k) {
        if (workspace.particles[cells[k]].support != entry[k]) {
            moved.push_back(cells[k]);
        }
    }
    return moved;
}

// Build the volume partition and moment matrix of the given cells from their
// symmetric stencils. A singular matrix expands that cell's own search and the
// tree is refit so later stencil queries see the new support; the expanded
// cells are returned so the caller can re-derive the dependency halo.
std::vector<size_t> hopkinsBuildGeometry(
    HopkinsWorkspace& workspace, std::vector<size_t> const& cells) {
    profile::ScopedPhase profilePhase("hydro.hopkins.geometry");
    HopkinsFlopBatch operations(workspace.countOperations);
    std::vector<size_t> expanded;
    std::vector<size_t> todo = cells;
    for (int attempt = 0; attempt < 16 && !todo.empty(); attempt++) {
        workspace.work.listQueries += todo.size();
        {
            profile::ScopedPhase profilePhase("hydro.hopkins.geometry.solve");
            hopkinsParallel(todo.size(), [&](size_t k) {
                HopkinsFlopBatch operations(workspace.countOperations);
                size_t const i = todo[k];
                auto const& particle = workspace.particles[i];
                double kernelSum =
                    hopkinsKernel(0.0, particle.support, workspace.dimensions, operations);
                hopkins::Matrix moment{};
                hopkinsNeighborBatches(
                    workspace, i, operations, [&](HopkinsNeighborBatch const& batch) {
                        auto const weight = hopkinsKernel(
                            batch.radius, particle.support, workspace.dimensions, operations);
                        HopkinsSimdMatrix contributions;
                        hopkins::detail::countFlops(operations, 18 * HopkinsSimd::size());
                        for (int row = 0; row < 3; row++) {
                            for (int column = 0; column < 3; column++) {
                                contributions[row][column] =
                                    weight * batch.separation[row] * batch.separation[column];
                            }
                        }
                        hopkins::detail::countFlops(operations, 10 * batch.count);
                        for (size_t lane = 0; lane < batch.count; lane++) {
                            kernelSum += weight[lane];
                            for (int row = 0; row < 3; row++) {
                                for (int column = 0; column < 3; column++) {
                                    moment[row][column] += contributions[row][column][lane];
                                }
                            }
                        }
                    });
                workspace.geometry[i] = hopkins::makeGeometry(
                    kernelSum, moment, 100.0, workspace.dimensions, operations);
            });
        }
        workspace.work.geometryCells += todo.size();
        std::vector<size_t> invalid;
        for (size_t i : todo) {
            if (!workspace.geometry[i].valid) {
                auto& support = workspace.particles[i].support;
                if (support >= getOptions().hydro.maxSupportRadius) {
                    throw std::runtime_error(
                        "Hopkins gradient remains singular at hydro.maxSupportRadius");
                }
                hopkins::detail::countFlops(operations, 1);
                support = std::min(1.5 * support, getOptions().hydro.maxSupportRadius);
                invalid.push_back(i);
            } else {
                hopkins::detail::countFlops(operations, 1);
                workspace.particles[i].density =
                    workspace.particles[i].mass / workspace.geometry[i].volume;
                workspace.built[i] = 1;
                workspace.touched[i] = 1;
            }
        }
        // Appendix C: expand the actual tree search, never invert a regularized
        // singular moment or silently substitute the SPH pressure operator.
        hopkinsUpdateSources(workspace, invalid);
        expanded.insert(expanded.end(), invalid.begin(), invalid.end());
        todo = std::move(invalid);
    }
    if (!todo.empty()) {
        throw std::runtime_error(
            "Hopkins gradient matrix did not converge after 16 support expansions");
    }
    return expanded;
}

std::vector<size_t> hopkinsUnflagged(HopkinsWorkspace const& workspace,
    std::vector<size_t> const& slots, std::vector<unsigned char> const& flags) {
    std::vector<size_t> result;
    for (size_t slot : slots) {
        if (!flags[slot]) {
            result.push_back(slot);
        }
    }
    return result;
}

// Refresh supports and geometry for the active cells and the first-ring halo
// their incident pair fluxes, gradients and face rates depend on. Supports and
// geometry are solved at most once per cell per event, and the tree's
// supportUpper is refit after every solve so the symmetric stencil queries stay
// exact. Returns the refreshed set A ∪ N(A) in slot order.
std::vector<size_t> hopkinsEnsureGeometry(HopkinsWorkspace& workspace) {
    size_t const limit = 2 * workspace.particles.size() + 16;
    for (size_t pass = 0; pass < limit; ++pass) {
        bool progressed = false;
        auto const active = hopkinsActiveSlots(workspace);
        auto const unsolved = hopkinsUnflagged(workspace, active, workspace.solved);
        if (!unsolved.empty()) {
            hopkinsUpdateSources(workspace, hopkinsSolveSupports(workspace, unsolved));
            progressed = true;
        }
        // From the first stencil query on, no resident position may change
        // until the event closes (hopkinsRefreshRecords enforces this).
        workspace.stencilsFrozen = true;
        workspace.work.listQueries += active.size();
        auto const halo = hopkinsHaloOf(workspace, active);
        auto const unsolvedHalo = hopkinsUnflagged(workspace, halo, workspace.solved);
        if (!unsolvedHalo.empty()) {
            hopkinsUpdateSources(workspace, hopkinsSolveSupports(workspace, unsolvedHalo));
            progressed = true;
            continue;
        }
        std::vector<size_t> needed;
        std::merge(
            active.begin(), active.end(), halo.begin(), halo.end(), std::back_inserter(needed));
        auto const unbuilt = hopkinsUnflagged(workspace, needed, workspace.built);
        if (!unbuilt.empty()) {
            hopkinsBuildGeometry(workspace, unbuilt);
            progressed = true;
        }
        if (!progressed) {
            return needed;
        }
    }
    throw std::runtime_error("Hopkins retained geometry refresh did not converge");
}

// Start or continue the event at this time: import changed records, derive the
// predicted positions of clean records and refit their sources. Flags survive
// between a hierarchical Begin and its Finish at the same time.
void hopkinsOpenEvent(HopkinsWorkspace& workspace, double scale, double expansionRate, double time,
    double baseTimestep, bool global, bool readForces) {
    workspace.dimensions = scenarios::definition(getOptions().run.scenario).hydroDimensions;
    workspace.scale = scale;
    workspace.expansionRate = expansionRate;
    workspace.time = time;
    workspace.tickDuration = baseTimestep > 0.0 ? baseTimestep / double(kHopkinsTicksPerBase) : 0.0;
    workspace.tick = global || baseTimestep <= 0.0 ? 0 : hopkinsTick(time, baseTimestep, nullptr);
    if (!workspace.eventOpen || workspace.eventTime != time) {
        hopkinsResizeSlots(workspace);
        for (auto& particle : workspace.particles) {
            particle.active = false;
        }
        workspace.switched = 0;
        workspace.switchedEnergy = 0.0;
        workspace.eventOpen = true;
        workspace.eventTime = time;
        workspace.rebuildDecided = false;
    }
    bool const rebuilt = hopkinsRefreshRecords(workspace, readForces);
    hopkinsDeriveRecords(workspace);
    if (rebuilt || workspace.tree.size() != workspace.particles.size()) {
        hopkinsRebuildSources(workspace);
        workspace.rebuildDecided = true;
        hopkinsRebuildOnNextEvent = false;
    } else {
        hopkinsUpdateSources(workspace, hopkinsSlotsWhere(workspace, workspace.dirty));
    }
    std::fill(workspace.dirty.begin(), workspace.dirty.end(), 0);
    // Every stencil query of this event is made at workspace.time: evaluate the
    // node envelopes and source positions once; in-event refits keep them current.
    workspace.tree.prepare(workspace.time);
}

// GADGET-4 rebuilds its neighbor tree with every domain decomposition, which
// its active-particle fraction triggers. Do the same once per event: rebuild the
// topology when a rebound happened or when at least hydro.treeRebuildActiveFraction
// of the cells is active; otherwise keep refitting it in place. The stencil sets
// come from the exact pair test, so the topology never changes a result.
void hopkinsRebuildForActivity(HopkinsWorkspace& workspace) {
    if (workspace.rebuildDecided) {
        return;
    }
    workspace.rebuildDecided = true;
    size_t active = 0;
    for (auto const& particle : workspace.particles) {
        active += particle.active;
    }
    double const fraction = getOptions().hydro.treeRebuildActiveFraction;
    bool const crowded = double(active) >= fraction * double(workspace.particles.size());
    if (!hopkinsRebuildOnNextEvent && !crowded) {
        return;
    }
    hopkinsRebuildOnNextEvent = false;
    hopkinsRebuildSources(workspace);
    workspace.tree.prepare(workspace.time);
}

// Return refreshed records to their owners; each owner then refreshes the
// predicted public mirrors of its untouched cells without communication.
void hopkinsInstallTouched(HopkinsWorkspace& workspace, int phase) {
    profile::ScopedPhase profilePhase("hydro.hopkins.scatter");
    size_t const ranks = workspace.rankSlots.size();
    std::vector<std::vector<ParticleInteger>> indices(ranks);
    std::vector<std::vector<HopkinsParticle>> records(ranks);
    for (size_t slot = 0; slot < workspace.particles.size(); ++slot) {
        if (workspace.touched[slot]) {
            auto& particle = workspace.particles[slot];
            // The resident sidecar mirrors the solved double support exactly
            // as the owner stores it after this install.
            particle.state.supportRadius = particle.support;
            auto const& owner = workspace.owners[slot];
            indices[size_t(owner.rank)].push_back(owner.index);
            records[size_t(owner.rank)].push_back(particle);
        }
    }
    std::vector<hpx::future<void>> pending;
    for (size_t rank = 0; rank < ranks; ++rank) {
        workspace.work.recordsInstalled += records[rank].size();
        workspace.work.bytesInstalled +=
            records[rank].size() * (sizeof(HopkinsParticle) + sizeof(ParticleInteger));
        pending.push_back(hpx::async<HopkinsInstallSparseAction>(hpxLocalities()[rank],
            std::move(indices[rank]), std::move(records[rank]), phase, workspace.time,
            workspace.scale, workspace.countOperations));
    }
    hpx::wait_all(pending);
    for (auto& future : pending) {
        future.get();
    }
}

double hopkinsKernelRate(double radius, double radiusRate, double support, double supportRate,
    int dimensions, FlopCount* operations) {
    if (radius >= support) {
        return 0.0;
    }
    hopkins::detail::countFlops(operations, dimensions == 1 ? 9 : 11);
    double const q = radius / support;
    double const denominator =
        dimensions == 1 ? support * support : support * support * support * support;
    return (hopkinsKernelDerivative(q, dimensions, operations) * (radiusRate - q * supportRate) -
               double(dimensions) * supportRate * hopkinsKernelValue(q, dimensions, operations)) /
        denominator;
}

HopkinsSimd hopkinsKernelRate(HopkinsSimd const& radius, HopkinsSimd const& radiusRate,
    double support, double supportRate, int dimensions, FlopCount* operations) {
    hopkins::detail::countFlops(operations, 7 * HopkinsSimd::size() + (dimensions == 1 ? 2 : 4));
    auto const q = radius / HopkinsSimd(support);
    double const denominator =
        dimensions == 1 ? support * support : support * support * support * support;
    auto const rate = (hopkinsKernelDerivative(q, dimensions, operations) *
                              (radiusRate - q * HopkinsSimd(supportRate)) -
                          HopkinsSimd(double(dimensions) * supportRate) *
                              hopkinsKernelValue(q, dimensions, operations)) /
        HopkinsSimd(denominator);
    return simdSelect(radius < HopkinsSimd(support), rate, HopkinsSimd(0.0));
}

// Face rates of the active cells and their halo; both endpoints of every
// opened pair need them, and each depends only on that cell's own list.
void hopkinsGeometryRates(HopkinsWorkspace& workspace, std::vector<size_t> const& cells) {
    profile::ScopedPhase profilePhase("hydro.hopkins.geometryRates");
    workspace.work.rateCells += cells.size();
    workspace.work.listQueries += cells.size();
    hopkinsParallel(cells.size(), [&](size_t k) {
        HopkinsFlopBatch operations(workspace.countOperations);
        size_t const i = cells[k];
        if (!workspace.built[i]) {
            throw std::logic_error("Hopkins face rates require current geometry");
        }
        auto const& particle = workspace.particles[i];
        // One stencil query serves both passes over this cell.
        std::vector<size_t> neighbors;
        hopkinsNeighborQuery(workspace, i, 1.0, true, operations, neighbors);
        double radialSum = 0.0, supportSum = 0.0;
        hopkinsNeighborBatches(
            workspace, i, neighbors, operations, [&](HopkinsNeighborBatch const& batch) {
                auto const relative = hopkinsRelativeVelocities(workspace, i, batch, operations);
                hopkins::detail::countFlops(operations, 4 * HopkinsSimd::size());
                auto const q = batch.radius / HopkinsSimd(particle.support);
                auto const derivative =
                    hopkinsKernelDerivative(q, workspace.dimensions, operations);
                auto const safeRadius =
                    simdSelect(batch.radius > HopkinsSimd(0.0), batch.radius, HopkinsSimd(1.0));
                auto const radialTerms = derivative *
                    hopkinsSimdDot(batch.separation, relative, operations) / safeRadius;
                auto const supportTerms = derivative * q;
                for (size_t lane = 0; lane < batch.count; lane++) {
                    if (batch.radius[lane] > 0.0 && batch.radius[lane] < particle.support) {
                        hopkins::detail::countFlops(operations, 2);
                        radialSum += radialTerms[lane];
                        supportSum += supportTerms[lane];
                    }
                }
            });
        // Differentiate the same constant kernel-number condition used by
        // the support solve: sum W(r/h) is constant along the moving points.
        hopkins::detail::countFlops(operations, supportSum != 0.0 ? 1 : 0);
        double supportRate = supportSum != 0.0 ? radialSum / supportSum : 0.0;
        double const target = hopkinsNeighborTarget(workspace.dimensions, operations);
        hopkins::detail::countFlops(operations, 2);
        double const count =
            hopkinsSupportVolume(particle.support, workspace.dimensions, operations) /
            workspace.geometry[i].volume;
        double const tolerance = kHopkinsSupportTolerance * target;
        bool const lowerHeld = particle.support <= getOptions().hydro.minSupportRadius;
        bool const upperHeld = particle.support >= getOptions().hydro.maxSupportRadius;
        hopkins::detail::countFlops(operations, lowerHeld && supportRate >= 0.0 ? 1 : 0);
        bool const lowerInconsistent =
            lowerHeld && (supportRate < 0.0 || count > target + tolerance);
        hopkins::detail::countFlops(
            operations, !lowerInconsistent && upperHeld && supportRate <= 0.0 ? 1 : 0);
        if (lowerInconsistent || (upperHeld && (supportRate > 0.0 || count < target - tolerance))) {
            supportRate = 0.0;
        }
        workspace.supportRates[i] = supportRate;
        auto const& geometry = workspace.geometry[i];
        double kernelSumRate = hopkinsKernelRate(
            0.0, 0.0, particle.support, supportRate, workspace.dimensions, operations);
        hopkins::Matrix rawMoment{}, rawMomentRate{};
        hopkinsNeighborBatches(
            workspace, i, neighbors, operations, [&](HopkinsNeighborBatch const& batch) {
                auto const& separation = batch.separation;
                auto const relative = hopkinsRelativeVelocities(workspace, i, batch, operations);
                auto const safeRadius =
                    simdSelect(batch.radius > HopkinsSimd(0.0), batch.radius, HopkinsSimd(1.0));
                hopkins::detail::countFlops(operations, HopkinsSimd::size());
                auto const radiusRate =
                    hopkinsSimdDot(separation, relative, operations) / safeRadius;
                auto const kernel =
                    hopkinsKernel(batch.radius, particle.support, workspace.dimensions, operations);
                auto const kernelRate = hopkinsKernelRate(batch.radius, radiusRate,
                    particle.support, supportRate, workspace.dimensions, operations);
                HopkinsSimdMatrix moments, momentRates;
                hopkins::detail::countFlops(operations, 81 * HopkinsSimd::size());
                for (int row = 0; row < 3; row++) {
                    for (int column = 0; column < 3; column++) {
                        moments[row][column] = kernel * separation[row] * separation[column];
                        momentRates[row][column] =
                            kernelRate * separation[row] * separation[column] +
                            kernel *
                                (relative[row] * separation[column] +
                                    separation[row] * relative[column]);
                    }
                }
                for (size_t lane = 0; lane < batch.count; lane++) {
                    if (batch.radius[lane] < particle.support) {
                        hopkins::detail::countFlops(operations, 19);
                        kernelSumRate += kernelRate[lane];
                        for (int row = 0; row < 3; row++) {
                            for (int column = 0; column < 3; column++) {
                                rawMoment[row][column] += moments[row][column][lane];
                                rawMomentRate[row][column] += momentRates[row][column][lane];
                            }
                        }
                    }
                }
            });
        hopkins::detail::countFlops(operations, 29);
        double const volumeRate = -geometry.volume * geometry.volume * kernelSumRate;
        hopkins::Matrix momentRate{};
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 3; column++) {
                momentRate[row][column] = volumeRate * rawMoment[row][column] +
                    geometry.volume * rawMomentRate[row][column];
            }
        }
        workspace.geometryRates[i] =
            hopkins::geometryRate(geometry, volumeRate, momentRate, operations);
    });
}

hopkins::Gradient hopkinsStoredGradient(HopkinsParticleState const& state) {
    hopkins::Gradient gradient;
    gradient.density = state.gradients[0];
    gradient.pressure = state.gradients[1];
    for (int dim = 0; dim < 3; dim++) {
        gradient.velocity[dim] = math::Vector<double, 3>(state.gradients[dim + 2]);
    }
    return gradient;
}

hopkins::Primitive hopkinsPredictedPrimitive(
    HopkinsParticleState const& state, double time, double scale, FlopCount* operations) {
    auto const anchor =
        hopkins::primitiveFromConserved({state.mass, state.momentum, state.totalEnergy},
            state.anchorVolume, getOptions().hydro.gamma);
    hopkins::detail::countFlops(operations, 16 + (getOptions().hydro.damping == 0.0 ? 0 : 3));
    double const elapsed = time - state.sourceTime;
    if (elapsed < 0.0 ||
        time > state.endTime +
                32.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, state.endTime)) {
        throw std::runtime_error("Hopkins primitive prediction lies outside its own interval");
    }
    auto predicted = hopkins::predictPrimitive(anchor, hopkinsStoredGradient(state),
        elapsed / scale, getOptions().hydro.gamma, operations);
    double const dampingRate = getOptions().hydro.damping;
    double const damping = std::exp(-dampingRate * elapsed);
    double const impulse =
        dampingRate == 0.0 ? elapsed : -std::expm1(-dampingRate * elapsed) / dampingRate;
    for (int dim = 0; dim < 3; dim++) {
        predicted.velocity[dim] =
            damping * predicted.velocity[dim] + impulse * state.sourceAcceleration[dim];
    }
    predicted.pressure *= std::exp(elapsed * state.expansionCoefficient);
    return predicted;
}

hopkins::Primitive hopkinsPredictedPrimitive(
    HopkinsParticle const& particle, HopkinsWorkspace const& workspace, FlopCount* operations) {
    return hopkinsPredictedPrimitive(particle.state, workspace.time, workspace.scale, operations);
}

// Reconstruct the active cells from the primitives of every refreshed cell:
// active cells use their closed views, halo cells their retained predictors.
void hopkinsGradients(HopkinsWorkspace& workspace, std::vector<size_t> const& active,
    std::vector<size_t> const& refreshed) {
    profile::ScopedPhase profilePhase("hydro.hopkins.gradients");
    HopkinsFlopBatch operations(workspace.countOperations);
    workspace.work.gradientCells += active.size();
    workspace.work.listQueries += active.size();
    double const gamma = getOptions().hydro.gamma;
    for (size_t i : refreshed) {
        auto const& particle = workspace.particles[i];
        workspace.gradients[i] = {};
        if (particle.active || !particle.state.predictorInitialized) {
            hopkins::detail::countFlops(operations, 3);
            workspace.primitives[i] = {particle.density, particle.velocity,
                (gamma - 1.0) * particle.density * particle.energy};
        } else {
            workspace.primitives[i] = hopkinsPredictedPrimitive(particle, workspace, operations);
            workspace.gradients[i] = hopkinsStoredGradient(particle.state);
        }
        if (!hopkins::physical(workspace.primitives[i])) {
            throw std::runtime_error("Nonphysical Hopkins cell before reconstruction");
        }
    }
    hopkinsParallel(active.size(), [&](size_t k) {
        HopkinsFlopBatch operations(workspace.countOperations);
        size_t const i = active[k];
        auto& particle = workspace.particles[i];
        if (!particle.active || !workspace.built[i]) {
            throw std::logic_error("Hopkins gradients require an active cell with geometry");
        }
        auto const& center = workspace.primitives[i];
        std::array<double, 5> values{center.density, center.pressure, center.velocity[0],
            center.velocity[1], center.velocity[2]};
        auto minimum = values;
        auto maximum = values;
        std::array<hopkins::Vector, 5> gradients{};
        // One stencil query serves the gradient sum and the limiter pass.
        std::vector<size_t> neighbors;
        hopkinsNeighborQuery(workspace, i, 1.0, true, operations, neighbors);
        hopkinsNeighborBatches(
            workspace, i, neighbors, operations, [&](HopkinsNeighborBatch const& batch) {
                auto const& geometry = workspace.geometry[i];
                auto const weightScale = hopkinsKernel(batch.radius, particle.support,
                                             workspace.dimensions, operations) *
                    HopkinsSimd(geometry.volume);
                HopkinsSimdVector weight;
                hopkins::detail::countFlops(operations, 19 * HopkinsSimd::size());
                auto const projected = geometry.inverseMoment * batch.separation;
                for (int dim = 0; dim < 3; dim++) {
                    weight[dim] = projected[dim] * weightScale;
                }
                std::array<HopkinsSimd, 5> neighborValues;
                neighborValues[0] = hopkinsNeighborValues(
                    batch, [&](size_t j) { return workspace.primitives[j].density; });
                neighborValues[1] = hopkinsNeighborValues(
                    batch, [&](size_t j) { return workspace.primitives[j].pressure; });
                for (int dim = 0; dim < 3; dim++) {
                    neighborValues[dim + 2] = hopkinsNeighborValues(
                        batch, [&](size_t j) { return workspace.primitives[j].velocity[dim]; });
                }
                std::array<HopkinsSimdVector, 5> contributions;
                hopkins::detail::countFlops(operations, 20 * HopkinsSimd::size());
                for (int field = 0; field < 5; field++) {
                    auto const difference = neighborValues[field] - HopkinsSimd(values[field]);
                    for (int dim = 0; dim < 3; dim++) {
                        contributions[field][dim] = weight[dim] * difference;
                    }
                }
                hopkins::detail::countFlops(operations, 15 * batch.count);
                for (size_t lane = 0; lane < batch.count; lane++) {
                    for (int field = 0; field < 5; field++) {
                        minimum[field] =
                            std::min(minimum[field], double(neighborValues[field][lane]));
                        maximum[field] =
                            std::max(maximum[field], double(neighborValues[field][lane]));
                        for (int dim = 0; dim < 3; dim++) {
                            gradients[field][dim] += contributions[field][dim][lane];
                        }
                    }
                }
            });
        auto reconstructedMinimum = values;
        auto reconstructedMaximum = values;
        hopkinsNeighborBatches(
            workspace, i, neighbors, operations,
            [&](HopkinsNeighborBatch const& batch) {
                auto const supports = hopkinsNeighborValues(
                    batch, [&](size_t j) { return workspace.particles[j].support; });
                auto const fraction =
                    HopkinsSimd(particle.support) / (HopkinsSimd(particle.support) + supports);
                auto offset = batch.separation;
                hopkins::detail::countFlops(operations, 35 * HopkinsSimd::size());
                for (int dim = 0; dim < 3; dim++) {
                    offset[dim] *= fraction;
                }
                std::array<HopkinsSimd, 5> reconstructed;
                for (int field = 0; field < 5; field++) {
                    reconstructed[field] = HopkinsSimd(values[field]) +
                        (HopkinsSimd(gradients[field][0]) * offset[0] +
                            HopkinsSimd(gradients[field][1]) * offset[1] +
                            HopkinsSimd(gradients[field][2]) * offset[2]);
                }
                for (size_t lane = 0; lane < batch.count; lane++) {
                    for (int field = 0; field < 5; field++) {
                        reconstructedMinimum[field] = std::min(
                            reconstructedMinimum[field], double(reconstructed[field][lane]));
                        reconstructedMaximum[field] = std::max(
                            reconstructedMaximum[field], double(reconstructed[field][lane]));
                    }
                }
            },
            false);
        for (int field = 0; field < 5; field++) {
            hopkins::detail::countFlops(operations, 3);
            double const limiter = hopkins::gradientLimiter(values[field], minimum[field],
                maximum[field], reconstructedMinimum[field], reconstructedMaximum[field],
                workspace.geometry[i].conditionNumber, operations);
            for (int dim = 0; dim < 3; dim++) {
                gradients[field][dim] *= limiter;
            }
        }
        auto& gradient = workspace.gradients[i];
        gradient.density = gradients[0];
        gradient.pressure = gradients[1];
        for (int dim = 0; dim < 3; dim++) {
            gradient.velocity[dim] = math::Vector<double, 3>(gradients[dim + 2]);
        }
        // Appendix G: only a completed or explicitly woken cell replaces its
        // primitive/gradient anchor. Conserved exchanges continue accumulating
        // for sleeping cells, independently of this predictor.
        particle.state.anchorVolume = workspace.geometry[i].volume;
        particle.state.gradients = gradients;
        particle.state.predictorInitialized = true;
    });
}

}    // namespace detail

namespace detail {

uint64_t hopkinsTick(double time, double baseTimestep, FlopCount* operations) {
    hopkins::detail::countFlops(operations, 3);
    double const ticks = time / baseTimestep * double(kHopkinsTicksPerBase);
    if (!(ticks >= 0.0) || ticks >= double(uint64_t(1) << 63)) {
        throw std::runtime_error("Hopkins absolute timestep clock overflow");
    }
    return uint64_t(std::llround(ticks));
}

HopkinsStepResult hopkinsSelectTimesteps(HopkinsWorkspace& workspace, int minimumRung,
    double baseTimestep, double maximumGlobalTimestep, size_t& woken, bool allowCoarsen = true) {
    profile::ScopedPhase profilePhase("hydro.hopkins.timesteps");
    HopkinsFlopBatch operations(workspace.countOperations);
    auto const& options = getOptions();
    double const gamma = options.hydro.gamma;
    int maximumRung = std::max(
        options.run.hierarchicalGravity ? 0 : workspace.darkMatterRung, options.run.minRung);
    bool const global = maximumGlobalTimestep > 0.0;
    double minimumTimestep = maximumGlobalTimestep;
    hopkins::detail::countFlops(operations, global ? 0 : 1);
    double backgroundTimestep =
        global ? maximumGlobalTimestep : std::ldexp(baseTimestep, -options.run.minRung);
#if COSMICTIGER_WITH_GRAVITY
    if (options.run.scenario == "cosmology") {
        hopkins::detail::countFlops(operations, 1);
        backgroundTimestep = std::min(backgroundTimestep,
            cosmology::expansionTimestep(workspace.scale, workspace.expansionRate / workspace.scale,
                cosmosD2adt2(workspace.scale), 0.01, operations));
    }
#endif
    // Pass one, in parallel and without side effects: every cell's own timestep
    // bound from its symmetric stencil, and whether a sleeping cell's retained
    // prediction is still physical. Pass two applies them in slot order, so the
    // outcome does not depend on the thread count.
    size_t const count = workspace.particles.size();
    std::vector<double> timesteps(count, backgroundTimestep);
    std::vector<unsigned char> invalidPrediction(count, 0);
    workspace.work.listQueries += count;
    hopkinsParallel(count, [&](size_t i) {
        HopkinsFlopBatch operations(workspace.countOperations);
        auto const& particle = workspace.particles[i];
        double timestep = backgroundTimestep;
        hopkins::detail::countFlops(operations, workspace.dimensions == 1 ? 6 : 8);
        // Active cells were just closed, so their anchor volume is the fresh
        // geometry; sleeping cells use their own-event anchor volume, which
        // needs no geometry solve and survives a restart unchanged.
        double const radius = workspace.dimensions == 1 ?
            0.5 * particle.state.anchorVolume :
            std::cbrt(3.0 * particle.state.anchorVolume / (4.0 * M_PI));
        double const sound = std::sqrt(gamma * (gamma - 1.0) * particle.energy);
        double signal = 2.0 * sound;
        hopkinsNeighborBatches(workspace, i, operations, [&](HopkinsNeighborBatch const& batch) {
            HopkinsSimdVector relative;
            for (int dim = 0; dim < 3; dim++) {
                relative[dim] = hopkinsNeighborValues(batch, [&](size_t j) {
                    return workspace.particles[j].velocity[dim];
                }) - HopkinsSimd(particle.velocity[dim]);
            }
            auto const energy = hopkinsNeighborValues(
                batch, [&](size_t j) { return workspace.particles[j].energy; });
            auto const safeRadius =
                simdSelect(batch.radius > HopkinsSimd(0.0), batch.radius, HopkinsSimd(1.0));
            hopkins::detail::countFlops(operations, 8 * HopkinsSimd::size() + 2);
            auto const signals = HopkinsSimd(sound) +
                simdSqrt(HopkinsSimd(gamma * (gamma - 1.0)) * energy) -
                simdMin(HopkinsSimd(0.0),
                    hopkinsSimdDot(relative, batch.separation, operations) / safeRadius);
            for (size_t lane = 0; lane < batch.count; lane++) {
                if (batch.radius[lane] > 0.0) {
                    signal = std::max(signal, double(signals[lane]));
                }
            }
        });
        hopkins::detail::countFlops(operations, 3 + (workspace.dimensions == 1 ? 1 : 0));
        double const courantLength = workspace.dimensions == 1 ? 2.0 * particle.support : radius;
        timestep = std::min(timestep, options.hydro.cfl * workspace.scale * courantLength / signal);
        double const acceleration = hopkins::norm(particle.acceleration, operations);
        if (acceleration > 0.0) {
            hopkins::detail::countFlops(operations, 4);
            timestep = std::min(timestep,
                options.gravity.timestepEta * std::sqrt(workspace.scale * radius / acceleration));
        }
        if (!(timestep > 0.0) || !std::isfinite(timestep)) {
            throw std::runtime_error("Invalid Hopkins particle timestep");
        }
        timesteps[i] = timestep;
        if (!global && !particle.active && particle.state.predictorInitialized) {
            invalidPrediction[i] =
                !hopkins::physical(hopkinsPredictedPrimitive(particle, workspace, operations));
        }
    });
    for (size_t i = 0; i < count; i++) {
        auto& particle = workspace.particles[i];
        double const timestep = timesteps[i];
        if (global) {
            minimumTimestep = std::min(minimumTimestep, timestep);
            particle.active = true;
            particle.rung = 0;
            continue;
        }
        hopkins::detail::countFlops(operations, 3);
        int const required =
            std::max(options.run.minRung, int(std::ceil(std::log2(baseTimestep / timestep))));
        if (particle.active) {
            particle.rung = std::max({minimumRung, particle.rung - int(allowCoarsen), required});
        } else if (required > particle.rung || invalidPrediction[i]) {
            // The ledger contains only elapsed exchanges, so an early wake-up
            // never needs to remove a future exchange from either endpoint.
            particle.active = true;
            particle.rung = std::max(minimumRung, required);
            woken++;
        }
    }
    if (global) {
        // The workspace contains every locality's gas; this is the collective
        // raw minimum, already bounded by gravity and the driver's endpoint.
        return {0, minimumTimestep};
    }
    // Propagate wake-ups in a stable order, including after an MPI restart
    // changes the local particle storage order.
    std::vector<size_t> order;
    for (size_t i = 0; i < count; i++) {
        order.push_back(i);
    }
    std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
        return workspace.particles[left].state.identifier <
            workspace.particles[right].state.identifier;
    });
    // Appendix G: prevent a timestep contrast larger than four across an
    // interacting pair. Wake sleeping neighbors onto the shorter active bin.
    // The sweep is sequential by construction: a raise earlier in the sweep
    // changes what a later cell raises, so cells are visited in identifier
    // order with immediate effect, exactly as a full sweep over every cell.
    // Two exact shortcuts remove no-op visits: a cell whose rung is below the
    // lowest rung plus three can raise nobody, and a cell revisited with an
    // unchanged rung can raise nobody new, because its neighbours' rungs only
    // grew since. The stencils of the cells due in a chunk are queried in
    // parallel; a cell that becomes due inside the chunk queries on the spot.
    int lowestRung = kMaxRung;
    for (auto const& particle : workspace.particles) {
        lowestRung = std::min(lowestRung, particle.rung);
    }
    auto const canRaise = [&](size_t i) { return workspace.particles[i].rung >= lowestRung + 3; };
    std::vector<int> visitedRung(count, -1);
    constexpr size_t kWakeChunk = 4096;
    std::vector<size_t> due;
    std::vector<std::vector<size_t>> stencils;
    bool changed;
    do {
        changed = false;
        for (size_t begin = 0; begin < count; begin += kWakeChunk) {
            size_t const end = std::min(count, begin + kWakeChunk);
            due.clear();
            for (size_t k = begin; k < end; ++k) {
                size_t const i = order[k];
                if (canRaise(i) && workspace.particles[i].rung != visitedRung[i]) {
                    due.push_back(i);
                }
            }
            stencils.assign(due.size(), {});
            workspace.work.listQueries += due.size();
            hopkinsParallel(due.size(), [&](size_t k) {
                HopkinsFlopBatch operations(workspace.countOperations);
                hopkinsNeighborQuery(workspace, due[k], 1.0, true, operations, stencils[k]);
            });
            size_t next = 0;
            std::vector<size_t> fresh;
            for (size_t k = begin; k < end; ++k) {
                size_t const i = order[k];
                std::vector<size_t> const* stencil = nullptr;
                if (next < due.size() && due[next] == i) {
                    stencil = &stencils[next++];
                } else if (canRaise(i) && workspace.particles[i].rung != visitedRung[i]) {
                    ++workspace.work.listQueries;
                    hopkinsNeighborQuery(workspace, i, 1.0, true, operations, fresh);
                    stencil = &fresh;
                } else {
                    continue;
                }
                auto const rung = workspace.particles[i].rung;
                visitedRung[i] = rung;
                for (size_t j : *stencil) {
                    auto& neighbor = workspace.particles[j];
                    if (neighbor.rung + 2 < rung) {
                        if (!neighbor.active) {
                            woken++;
                        }
                        neighbor.active = true;
                        neighbor.rung = std::max(minimumRung, rung);
                        changed = true;
                    }
                }
            }
        }
    } while (changed);
    for (auto const& particle : workspace.particles) {
        maximumRung = std::max(maximumRung, particle.rung);
    }
    if (maximumRung >= kMaxRung) {
        throw std::runtime_error("Hopkins timestep exceeds the available rung range");
    }
    hopkins::detail::countFlops(operations, 1);
    return {maximumRung, std::ldexp(baseTimestep, -maximumRung)};
}
// Keep interval ownership and fixed-point geometry preparation scalar. Independent
// reconstructed face states and Riemann solves are evaluated together in SIMD.
struct HopkinsPairInput {
    HopkinsPairRecord record{};
    size_t i = 0;
    size_t j = 0;
    double interval = 0.0;
    double fraction = 0.0;
    bool firstOrder = false;
    hopkins::Vector separation{}, area{}, areaRate{}, frame{}, offsetLeft{}, offsetRight{};
};

HopkinsPairInput hopkinsPreparePair(
    HopkinsWorkspace const& workspace, size_t i, size_t j, uint64_t tick, bool firstOrder) {
    HopkinsFlopBatch operations(workspace.countOperations);
    auto const& leftParticle = workspace.particles[i];
    auto const& rightParticle = workspace.particles[j];
    HopkinsPairInput input;
    auto& record = input.record;
    input.i = i;
    input.j = j;
    record.leftId = leftParticle.state.identifier;
    record.rightId = rightParticle.state.identifier;
    record.beginTick = tick;
    record.endTick = std::min(leftParticle.state.endTick, rightParticle.state.endTick);
    record.beginTime = workspace.time;
    record.endTime = std::min(leftParticle.state.endTime, rightParticle.state.endTime);
    double const interval = record.endTime - record.beginTime;
    if (!(interval > 0.0)) {
        throw std::runtime_error("Hopkins pair has no positive scheduled interval");
    }
    auto const separation = hopkinsSeparation(leftParticle, rightParticle, operations);
    double const radius = hopkins::norm(separation, operations);
    auto const area = hopkins::faceArea(workspace.geometry[i], workspace.geometry[j], separation,
        hopkinsKernel(radius, leftParticle.support, workspace.dimensions, operations),
        hopkinsKernel(radius, rightParticle.support, workspace.dimensions, operations), operations);
    hopkins::detail::countFlops(operations, 6 + (radius > 0.0 ? 1 : 0));
    hopkins::Vector relative{};
    for (int dim = 0; dim < 3; dim++) {
        relative[dim] =
            (rightParticle.velocity[dim] - leftParticle.velocity[dim]) / workspace.scale;
    }
    double const radiusRate =
        radius > 0.0 ? hopkins::dot(separation, relative, operations) / radius : 0.0;
    auto const areaRate = hopkins::faceAreaRate(workspace.geometry[i], workspace.geometry[j],
        workspace.geometryRates[i], workspace.geometryRates[j], separation, relative,
        hopkinsKernel(radius, leftParticle.support, workspace.dimensions, operations),
        hopkinsKernel(radius, rightParticle.support, workspace.dimensions, operations),
        hopkinsKernelRate(radius, radiusRate, leftParticle.support, workspace.supportRates[i],
            workspace.dimensions, operations),
        hopkinsKernelRate(radius, radiusRate, rightParticle.support, workspace.supportRates[j],
            workspace.dimensions, operations),
        operations);
    hopkins::detail::countFlops(operations, 9);
    hopkins::Vector midpointArea = area;
    for (int dim = 0; dim < 3; dim++) {
        midpointArea[dim] += 0.5 * interval * areaRate[dim];
    }
    // A Taylor prediction can change sign as a tiny face leaves compact
    // support. Retry this pair with its conservative first-order flux instead
    // of reversing the face normal or rejecting an otherwise stable step.
    firstOrder = firstOrder || !std::isfinite(hopkins::norm(midpointArea, operations)) ||
        hopkins::dot(midpointArea, area, operations) < 0.0;
    hopkins::detail::countFlops(operations, 23);
    double const fraction = leftParticle.support / (leftParticle.support + rightParticle.support);
    hopkins::Vector frame{}, offsetLeft{}, offsetRight{};
    for (int dim = 0; dim < 3; dim++) {
        frame[dim] =
            (1.0 - fraction) * leftParticle.velocity[dim] + fraction * rightParticle.velocity[dim];
        offsetLeft[dim] = fraction * separation[dim];
        offsetRight[dim] = -(1.0 - fraction) * separation[dim];
    }
    input.interval = interval;
    input.firstOrder = firstOrder;
    input.separation = separation;
    input.area = area;
    input.areaRate = areaRate;
    input.frame = frame;
    input.offsetLeft = offsetLeft;
    input.offsetRight = offsetRight;
    input.fraction = fraction;
    return input;
}

HopkinsSimd hopkinsSimdExp(HopkinsSimd const& value) {
#if COSMICTIGER_SIMD_STANDARD
    return HopkinsSimd([&](auto lane) { return std::exp(value[lane]); });
#else
    return std::experimental::exp(value);
#endif
}

HopkinsSimd hopkinsSimdExpm1(HopkinsSimd const& value) {
#if COSMICTIGER_SIMD_STANDARD
    return HopkinsSimd([&](auto lane) { return std::expm1(value[lane]); });
#else
    return std::experimental::expm1(value);
#endif
}

void hopkinsPairBatch(HopkinsWorkspace const& workspace,
    std::vector<std::pair<size_t, size_t>> const& pairs, std::vector<size_t> const& solvePairs,
    size_t begin, uint64_t tick, bool firstOrder, std::vector<HopkinsPairRecord>& records) {
    constexpr size_t kWidth = HopkinsSimd::size();
    size_t const count = std::min(kWidth, solvePairs.size() - begin);
    std::array<HopkinsPairInput, kWidth> inputs;
    for (size_t lane = 0; lane < count; lane++) {
        auto const [i, j] = pairs[solvePairs[begin + lane]];
        inputs[lane] = hopkinsPreparePair(workspace, i, j, tick, firstOrder);
    }
    // Duplicate a physical input in unused lanes. Only active lanes are solved/stored.
    for (size_t lane = count; lane < kWidth; lane++) {
        inputs[lane] = inputs[0];
    }
    auto gather = [&](auto const& function) {
        return HopkinsSimd([&](auto lane) { return function(inputs[size_t(lane)]); });
    };
    auto const active =
        HopkinsSimd([&](auto lane) { return double(lane); }) < HopkinsSimd(double(count));
    auto const secondOrder = active &
        (gather([](auto const& input) { return input.firstOrder ? 0.0 : 1.0; }) > HopkinsSimd(0.0));
    bool const anySecondOrder = simdAnyOf(secondOrder);
    bool const allSecondOrder = simdAllOf(!active | secondOrder);
    HopkinsFlopBatch operations(workspace.countOperations);
    double const gamma = getOptions().hydro.gamma;
    HopkinsSimd const scale(workspace.scale);
    HopkinsSimd const one(1.0);
    HopkinsSimd const zero(0.0);
    auto const interval = gather([](auto const& input) { return input.interval; });
    auto const fraction = gather([](auto const& input) { return input.fraction; });
    hopkins::PrimitiveValues<HopkinsSimd> centers[2];
    hopkins::GradientValues<HopkinsSimd> gradients[2];
    HopkinsSimdVector offsets[2], acceleration[2];
    HopkinsSimd expansion[2];
    for (int side = 0; side < 2; side++) {
        auto endpoint = [&](HopkinsPairInput const& input) {
            return side == 0 ? input.i : input.j;
        };
        centers[side].density = gather(
            [&](auto const& input) { return workspace.primitives[endpoint(input)].density; });
        centers[side].pressure = gather(
            [&](auto const& input) { return workspace.primitives[endpoint(input)].pressure; });
        expansion[side] = gather([&](auto const& input) {
            return workspace.particles[endpoint(input)].state.expansionCoefficient;
        });
        for (int dim = 0; dim < 3; dim++) {
            centers[side].velocity[dim] = gather([&](auto const& input) {
                return workspace.primitives[endpoint(input)].velocity[dim];
            });
            gradients[side].density[dim] = gather([&](auto const& input) {
                return workspace.gradients[endpoint(input)].density[dim];
            });
            gradients[side].pressure[dim] = gather([&](auto const& input) {
                return workspace.gradients[endpoint(input)].pressure[dim];
            });
            offsets[side][dim] = gather([&](auto const& input) {
                return side == 0 ? input.offsetLeft[dim] : input.offsetRight[dim];
            });
            acceleration[side][dim] = gather([&](auto const& input) {
                return workspace.particles[endpoint(input)].state.sourceAcceleration[dim];
            });
            for (int component = 0; component < 3; component++) {
                gradients[side].velocity[dim][component] = gather([&](auto const& input) {
                    return workspace.gradients[endpoint(input)].velocity[dim][component];
                });
            }
        }
    }
    HopkinsSimdVector area, areaRate, frame;
    for (int dim = 0; dim < 3; dim++) {
        area[dim] = gather([&](auto const& input) { return input.area[dim]; });
        areaRate[dim] = gather([&](auto const& input) { return input.areaRate[dim]; });
        frame[dim] = gather([&](auto const& input) { return input.frame[dim]; });
    }
    // A first-order face can have a nonfinite rejected area predictor. Mask
    // its operands before arithmetic: zero duration alone cannot suppress 0*NaN.
    if (!allSecondOrder) {
        for (int side = 0; side < 2; side++) {
            expansion[side] = simdSelect(secondOrder, expansion[side], zero);
            for (int dim = 0; dim < 3; dim++) {
                offsets[side][dim] = simdSelect(secondOrder, offsets[side][dim], zero);
                acceleration[side][dim] = simdSelect(secondOrder, acceleration[side][dim], zero);
                gradients[side].density[dim] =
                    simdSelect(secondOrder, gradients[side].density[dim], zero);
                gradients[side].pressure[dim] =
                    simdSelect(secondOrder, gradients[side].pressure[dim], zero);
                for (int component = 0; component < 3; component++) {
                    gradients[side].velocity[dim][component] =
                        simdSelect(secondOrder, gradients[side].velocity[dim][component], zero);
                }
            }
        }
        for (int dim = 0; dim < 3; dim++) {
            areaRate[dim] = simdSelect(secondOrder, areaRate[dim], zero);
        }
    }
    auto solve = [&](HopkinsSimd const& duration, auto const& solveMask) {
        auto left = centers[0];
        auto right = centers[1];
        auto predictedFrame = frame;
        auto predictedArea = area;
        if (anySecondOrder) {
            auto const reconstructedLeft = hopkins::reconstructValues(centers[0], gradients[0],
                offsets[0], frame, duration / scale, gamma, centers[1], fraction, operations);
            auto const reconstructedRight = hopkins::reconstructValues(centers[1], gradients[1],
                offsets[1], frame, duration / scale, gamma, centers[0], one - fraction, operations);
            left = reconstructedLeft;
            right = reconstructedRight;
            if (!allSecondOrder) {
                left.density = simdSelect(secondOrder, left.density, centers[0].density);
                left.pressure = simdSelect(secondOrder, left.pressure, centers[0].pressure);
                right.density = simdSelect(secondOrder, right.density, centers[1].density);
                right.pressure = simdSelect(secondOrder, right.pressure, centers[1].pressure);
                for (int dim = 0; dim < 3; dim++) {
                    left.velocity[dim] =
                        simdSelect(secondOrder, left.velocity[dim], centers[0].velocity[dim]);
                    right.velocity[dim] =
                        simdSelect(secondOrder, right.velocity[dim], centers[1].velocity[dim]);
                }
            }
            // First-order lanes have zero source-prediction duration.
            auto const sourceDuration = simdSelect(secondOrder, duration, zero);
            auto const damping =
                hopkinsSimdExp(-HopkinsSimd(getOptions().hydro.damping) * sourceDuration);
            auto const impulse = getOptions().hydro.damping == 0.0 ?
                sourceDuration :
                -hopkinsSimdExpm1(-HopkinsSimd(getOptions().hydro.damping) * sourceDuration) /
                    HopkinsSimd(getOptions().hydro.damping);
            hopkins::detail::countFlops(
                operations, kWidth * (83 + (getOptions().hydro.damping == 0.0 ? 0 : 3)));
            for (int dim = 0; dim < 3; dim++) {
                left.velocity[dim] = damping * left.velocity[dim] + impulse * acceleration[0][dim];
                right.velocity[dim] =
                    damping * right.velocity[dim] + impulse * acceleration[1][dim];
                auto const hydroAcceleration =
                    -((one - fraction) * gradients[0].pressure[dim] / centers[0].density +
                        fraction * gradients[1].pressure[dim] / centers[1].density) /
                    scale;
                predictedFrame[dim] = damping * (frame[dim] + sourceDuration * hydroAcceleration) +
                    impulse *
                        ((one - fraction) * acceleration[0][dim] + fraction * acceleration[1][dim]);
                predictedArea[dim] += sourceDuration * areaRate[dim];
            }
            left.pressure *= hopkinsSimdExp(sourceDuration * expansion[0]);
            right.pressure *= hopkinsSimdExp(sourceDuration * expansion[1]);
        }
        return hopkins::riemannFluxValues(left, right, predictedArea, predictedFrame, gamma,
            getOptions().hydro.method == "mfm", solveMask, operations);
    };
    hopkins::FluxValues<HopkinsSimd> initial, midpoint;
    char const* solveStage = "initial spatial reconstruction";
    try {
        initial = solve(zero, active);
        midpoint = initial;
        if (anySecondOrder) {
            hopkins::detail::countFlops(operations, kWidth);
            solveStage = "midpoint prediction";
            midpoint = solve(HopkinsSimd(0.5) * interval, secondOrder);
        }
    } catch (std::exception const& error) {
        std::string message =
            std::string(error.what()) + " [Hopkins " + solveStage + "; pair IDs by SIMD lane:";
        for (size_t lane = 0; lane < count; lane++) {
            message += " " + std::to_string(lane) + "=" +
                std::to_string(inputs[lane].record.leftId) + "," +
                std::to_string(inputs[lane].record.rightId);
        }
        throw std::runtime_error(message + "]");
    }
    for (size_t lane = 0; lane < count; lane++) {
        auto const& input = inputs[lane];
        auto& record = records[solvePairs[begin + lane]];
        record = input.record;
        double const midScale = workspace.scale + 0.5 * input.interval * workspace.expansionRate;
        if (!(midScale > 0.0)) {
            throw std::runtime_error("Nonpositive scale factor in Hopkins pair prediction");
        }
        hopkins::detail::countFlops(operations, 7);
        auto const& middle = input.firstOrder ? initial : midpoint;
        double const middleScale = input.firstOrder ? workspace.scale : midScale;
        record.initial.mass = initial.mass[lane] / workspace.scale;
        record.initial.energy = initial.energy[lane] / workspace.scale;
        record.midpoint.mass = middle.mass[lane] / middleScale;
        record.midpoint.energy = middle.energy[lane] / middleScale;
        hopkins::Vector midpointSeparation{};
        for (int dim = 0; dim < 3; dim++) {
            hopkins::detail::countFlops(operations, input.firstOrder ? 3 : 7);
            record.initial.momentum[dim] = initial.momentum[dim][lane] / workspace.scale;
            record.midpoint.momentum[dim] = middle.momentum[dim][lane] / middleScale;
            midpointSeparation[dim] = input.separation[dim] +
                (input.firstOrder ? 0.0 :
                                    0.5 * input.interval / workspace.scale *
                            (workspace.particles[input.j].velocity[dim] -
                                workspace.particles[input.i].velocity[dim]));
        }
        record.workInitial =
            hopkins::pairGravityWorkVector(initial.mass[lane], input.separation, operations);
        record.workMidpoint =
            hopkins::pairGravityWorkVector(middle.mass[lane], midpointSeparation, operations);
    }
}

// Settle an interval only at its scheduled endpoint, or once when a wake/terminal
// event cuts it short. Both deposits are made from the same integral.
size_t hopkinsSettlePairs(HopkinsWorkspace& workspace, bool closingOnly) {
    profile::ScopedPhase profilePhase("hydro.hopkins.flux.settle");
    auto const& indices = workspace.indices;
    size_t settled = 0;
    for (auto iterator = hopkinsPairs.begin(); iterator != hopkinsPairs.end();) {
        auto const& record = iterator->second;
        auto const left = indices.find(record.leftId);
        auto const right = indices.find(record.rightId);
        if (left == indices.end() || right == indices.end()) {
            throw std::runtime_error("Hopkins live pair lost an endpoint");
        }
        size_t const i = left->second;
        size_t const j = right->second;
        if (!closingOnly && record.endTime > workspace.time && !workspace.particles[i].active &&
            !workspace.particles[j].active) {
            ++iterator;
            continue;
        }
        double const end = std::min(workspace.time, record.endTime);
        double const interval = record.endTime - record.beginTime;
        double const duration = end - record.beginTime;
        auto const delta =
            hopkins::integrateFlux(record.initial, record.midpoint, interval, 0.0, duration);
        auto const work = hopkins::integratePairGravityWork(
            record.workInitial, record.workMidpoint, interval, 0.0, duration);
        for (size_t endpoint : {i, j}) {
            auto& state = workspace.particles[endpoint].state;
            workspace.touched[endpoint] = 1;
            double const sign = endpoint == i ? -1.0 : 1.0;
            state.pendingMass += sign * delta.mass;
            state.pendingEnergy += sign * delta.energy;
            double const age = record.beginTime - state.sourceTime;
            if (age < -32.0 * std::numeric_limits<double>::epsilon() *
                    std::max(1.0, std::abs(state.sourceTime))) {
                throw std::runtime_error("Hopkins pair predates an endpoint's source anchor");
            }
            for (int dim = 0; dim < 3; ++dim) {
                state.pendingMomentum[dim] += sign * delta.momentum[dim];
                state.pendingGravityIntegral[dim] += work.integral[dim];
                state.pendingGravityWork[dim] += age * work.integral[dim] + work.firstMoment[dim];
            }
        }
        iterator = hopkinsPairs.erase(iterator);
        ++settled;
    }
    return settled;
}

void hopkinsInitialize(HopkinsWorkspace& workspace) {
    double const coefficient =
        (5.0 - 3.0 * getOptions().hydro.gamma) * workspace.expansionRate / workspace.scale;
    for (size_t i = 0; i < workspace.particles.size(); ++i) {
        auto& particle = workspace.particles[i];
        auto& state = particle.state;
        if (!state.initialized) {
            state.mass = particle.mass;
            state.totalEnergy = particle.mass *
                (particle.energy + 0.5 * hopkins::dot(particle.velocity, particle.velocity));
            for (int dim = 0; dim < 3; ++dim) {
                state.momentum[dim] = particle.mass * particle.velocity[dim];
            }
            state.initialized = true;
        }
        if (!state.predictorInitialized) {
            if (getOptions().run.hierarchicalGravity && !state.hierarchicalActive) {
                state.positionAnchor = particle.position;
                state.positionEpoch = workspace.time;
                state.positionDriftScale = workspace.scale;
                state.gravityEndTime = workspace.time;
                state.gravityDriftScale = workspace.scale;
                state.hydroRung = particle.rung;
            }
            state.anchorVolume = workspace.geometry[i].volume;
            state.sourceTime = workspace.time;
            state.endTime = workspace.time;
            state.beginTick = state.endTick = workspace.tick;
            state.sourceAcceleration = particle.acceleration;
            state.expansionCoefficient = coefficient;
        }
        if (!state.predictorInitialized || !(state.entropy > 0.0)) {
            double const thermal =
                hopkins::thermalEnergy({state.mass, state.momentum, state.totalEnergy}) /
                state.mass;
            if (!(thermal > 0.0) || !std::isfinite(thermal)) {
                throw std::runtime_error("Nonpositive Hopkins thermal energy at initialization");
            }
            state.entropy = hopkins::entropyFunction(
                thermal, state.mass / state.anchorVolume, getOptions().hydro.gamma);
        }
    }
}

void hopkinsCloseParticle(HopkinsWorkspace& workspace, size_t i) {
    if (!workspace.built[i]) {
        throw std::logic_error("Hopkins closure requires current geometry");
    }
    workspace.touched[i] = 1;
    auto& particle = workspace.particles[i];
    auto& state = particle.state;
    double const duration = workspace.time - state.sourceTime;
    double const coefficient =
        (5.0 - 3.0 * getOptions().hydro.gamma) * workspace.expansionRate / workspace.scale;
    double work = hopkins::dot(state.pendingGravityIntegral, state.sourceAcceleration);
    if (duration > 0.0) {
        for (int dim = 0; dim < 3; ++dim) {
            work += state.pendingGravityWork[dim] *
                (particle.acceleration[dim] - state.sourceAcceleration[dim]) / duration;
        }
    }
    auto next = hopkins::closeConservedInterval({state.mass, state.momentum, state.totalEnergy},
        {state.pendingMass, state.pendingMomentum, state.pendingEnergy}, state.sourceAcceleration,
        particle.acceleration, duration, getOptions().hydro.damping, state.expansionCoefficient,
        coefficient, work);
    auto const& options = getOptions();
    // The expansion source scales the entropy exactly as the closure scaled the thermal
    // energy over its two half intervals; a fixed-mass cell exchanges no entropy otherwise.
    double const advectedEntropy =
        state.entropy * std::exp(0.5 * duration * (state.expansionCoefficient + coefficient));
    if (options.hydro.method == "mfm" && options.hydro.dualEnergyPressureSwitch > 0.0) {
        auto const closure =
            hopkins::dualEnergyClosure(next, advectedEntropy, workspace.geometry[i].volume,
                options.hydro.gamma, options.hydro.dualEnergyPressureSwitch);
        next = closure.state;
        if (closure.switched) {
            ++workspace.switched;
            workspace.switchedEnergy += closure.correction;
        }
    } else if (!(hopkins::thermalEnergy(next) > 0.0)) {
        throw std::runtime_error(
            "Nonpositive Hopkins thermal energy at own-step closure; "
            "reduce hydro.cfl (published pair transfers cannot be retried one-sided)");
    }
    state.entropy = advectedEntropy;
    state.mass = next.mass;
    state.momentum = next.momentum;
    state.totalEnergy = next.energy;
    state.pendingMass = state.pendingEnergy = 0.0;
    state.pendingMomentum = {};
    state.pendingGravityIntegral = {};
    state.pendingGravityWork = {};
    state.sourceAcceleration = particle.acceleration;
    state.sourceTime = state.endTime = workspace.time;
    state.beginTick = state.endTick = workspace.tick;
    state.expansionCoefficient = coefficient;
    state.anchorVolume = workspace.geometry[i].volume;
    if (getOptions().run.hierarchicalGravity) {
        state.positionAnchor = particle.position;
        state.positionEpoch = workspace.time;
    }
}

// Thermal sources change the entropy by the specific energy they add to the cell. If an
// independently evolved entropy cannot absorb that increment, it restarts from the
// cell's own post-source thermal energy.
void hopkinsAddSourceEntropy(HopkinsParticle& particle, double delta) {
    auto& state = particle.state;
    double const gamma = getOptions().hydro.gamma;
    double const density = state.mass / state.anchorVolume;
    double const entropy = state.entropy + hopkins::entropyFunction(delta, density, gamma);
    if (entropy > 0.0 && std::isfinite(entropy)) {
        state.entropy = entropy;
        return;
    }
    double const thermal =
        hopkins::thermalEnergy({state.mass, state.momentum, state.totalEnergy}) / state.mass;
    if (!(thermal > 0.0) || !std::isfinite(thermal)) {
        throw std::runtime_error("Nonpositive Hopkins thermal energy after sources");
    }
    state.entropy = hopkins::entropyFunction(thermal, density, gamma);
}

// Operator-split GADGET-4 sources for cells whose own-step intervals have just closed.
// Each cell integrates equilibrium cooling, or the combined SH03 multiphase source,
// over exactly the duration it closed, including early wake-ups and the terminal
// closure. The thermal change enters the conserved anchor before the new primitives,
// timestep selection and reconstruction, mirroring the SPH step order. MFM keeps every
// cell's mass fixed, so the specific energy maps one-to-one onto the total energy; the
// float stored energy enters the source exactly as it does for SPH and only its change
// is applied, so no rounding of the double anchor accumulates. A converted cell keeps
// its closing state until the scatter removes it and opens no new pair interval.
void hopkinsApplySources(HopkinsWorkspace& workspace, std::vector<size_t> const& closing,
    std::vector<double> const& durations, EnergiesT& energies, size_t& converted) {
    auto const& options = getOptions();
    if (!options.chemistry.enabled) {
        return;
    }
#if COSMICTIGER_ENABLE_CHEMISTRY
    if (options.hydro.method != "mfm") {
        throw std::logic_error("Hopkins thermal sources require the fixed-mass MFM scheme");
    }
    profile::ScopedPhase profilePhase("hydro.hopkins.sources");
    std::vector<size_t> indices;
    std::vector<double> intervals;
    for (size_t k = 0; k < closing.size(); ++k) {
        if (durations[k] > 0.0) {
            indices.push_back(closing[k]);
            intervals.push_back(durations[k]);
        }
    }
    if (indices.empty()) {
        return;
    }
    auto const storedEnergy = [](HopkinsParticle const& particle) {
        auto const& state = particle.state;
        double const energy = state.totalEnergy / state.mass -
            0.5 * hopkins::dot(state.momentum, state.momentum) / (state.mass * state.mass);
        if (!(energy > 0.0) || !std::isfinite(energy) ||
            energy > double(std::numeric_limits<float>::max())) {
            throw std::runtime_error("Nonphysical Hopkins cell thermal energy before sources");
        }
        return float(energy);
    };
    // The gathered cell carries no density until the primitives are rebuilt; the
    // closure has just anchored the current kernel-partition volume.
    auto const storedDensity = [](HopkinsParticle const& particle) {
        auto const& state = particle.state;
        double const density = state.mass / state.anchorVolume;
        if (!(density > 0.0) || !std::isfinite(density)) {
            throw std::runtime_error("Nonphysical Hopkins cell density before sources");
        }
        return float(density);
    };
    double heating = 0.0;
    if (options.starFormation.enabled) {
#if COSMICTIGER_ENABLE_STAR_FORMATION
        std::vector<star_formation_backend::Input> inputs(indices.size());
        for (size_t k = 0; k < indices.size(); ++k) {
            auto const& particle = workspace.particles[indices[k]];
            auto& input = inputs[k];
            input.state = particle.starFormation;
            input.chemistry = particle.chemistry;
            input.energy = storedEnergy(particle);
            input.density = storedDensity(particle);
            input.pressureRatio = float(gadget4::kGammaMinusOne);
            input.rung = particle.rung;
            for (int dim = 0; dim < 3; ++dim) {
                input.velocity[dim] = float(particle.velocity[dim]);
            }
        }
        auto const outputs = starFormationAdvanceBatch(inputs, intervals, workspace.scale);
        for (size_t k = 0; k < indices.size(); ++k) {
            auto& particle = workspace.particles[indices[k]];
            auto const& output = outputs[k];
            heating += output.value;
            particle.starFormation = output.state;
            if (output.convert) {
                particle.convert = true;
                ++converted;
                continue;
            }
            double const delta = double(output.energy) - double(inputs[k].energy);
            particle.state.totalEnergy += particle.state.mass * delta;
            hopkinsAddSourceEntropy(particle, delta);
            particle.chemistry = output.chemistry;
        }
#else
        throw std::logic_error("Star formation requires COSMICTIGER_ENABLE_STARS=ON");
#endif
    } else {
        std::vector<SphChemistryBatchInput> inputs(indices.size());
        for (size_t k = 0; k < indices.size(); ++k) {
            auto const& particle = workspace.particles[indices[k]];
            inputs[k] = {particle.chemistry, storedEnergy(particle), storedDensity(particle),
                sphChemistryPhysicalInterval(intervals[k], workspace.scale),
                static_cast<long long>(particle.state.identifier)};
        }
        auto const results = sphChemistryAdvanceBatch(inputs, workspace.scale);
        for (size_t k = 0; k < indices.size(); ++k) {
            auto& particle = workspace.particles[indices[k]];
            double const delta = double(results[k].storedEnergy) - double(inputs[k].storedEnergy);
            heating -= particle.state.mass * delta / workspace.scale;
            particle.state.totalEnergy += particle.state.mass * delta;
            hopkinsAddSourceEntropy(particle, delta);
            particle.chemistry = results[k].state;
        }
    }
    energies.heating += heating;
#else
    throw std::logic_error("Equilibrium cooling requires COSMICTIGER_ENABLE_CHEMISTRY=ON");
#endif
}

// Bryan et al. (1995) update switch over the cells that closed at this instant. Every
// closure and source of the instant has already been applied, so the neighborhood
// maximum does not depend on the closure order; inactive neighbors contribute their
// anchors. A converted cell keeps its closing entropy.
void hopkinsSynchronizeEntropy(HopkinsWorkspace& workspace, std::vector<size_t> const& closing) {
    auto const& options = getOptions();
    auto const energyDensity = [&](size_t j) {
        auto const& state = workspace.particles[j].state;
        return state.totalEnergy / state.anchorVolume;
    };
    // Each closing cell reads its stencil and writes only its own entropy.
    for (size_t i : closing) {
        workspace.work.listQueries += !workspace.particles[i].convert;
    }
    hopkinsParallel(closing.size(), [&](size_t k) {
        size_t const i = closing[k];
        auto& particle = workspace.particles[i];
        if (particle.convert) {
            return;
        }
        HopkinsFlopBatch operations(workspace.countOperations);
        std::vector<size_t> neighbors;
        hopkinsNeighborQuery(workspace, i, 1.0, true, operations, neighbors);
        auto& state = particle.state;
        double maximum = energyDensity(i);
        for (size_t j : neighbors) {
            maximum = std::max(maximum, energyDensity(j));
        }
        double const thermal =
            hopkins::thermalEnergy({state.mass, state.momentum, state.totalEnergy});
        if (hopkins::entropyUpdateDue(
                thermal / state.anchorVolume, maximum, options.hydro.dualEnergyUpdateSwitch)) {
            state.entropy = hopkins::entropyFunction(
                thermal / state.mass, state.mass / state.anchorVolume, options.hydro.gamma);
        }
    });
}

void hopkinsCurrentViews(HopkinsWorkspace& workspace) {
    profile::ScopedPhase profilePhase("hydro.hopkins.primitives");
    for (size_t i = 0; i < workspace.particles.size(); ++i) {
        auto& particle = workspace.particles[i];
        auto const& state = particle.state;
        auto const primitive = particle.active ?
            hopkins::primitiveFromConserved({state.mass, state.momentum, state.totalEnergy},
                state.anchorVolume, getOptions().hydro.gamma) :
            hopkinsPredictedPrimitive(particle, workspace, nullptr);
        if (!hopkins::physical(primitive)) {
            throw std::runtime_error("Nonphysical Hopkins own-step primitive");
        }
        particle.mass = state.mass + state.pendingMass;
        particle.density = primitive.density;
        particle.velocity = primitive.velocity;
        particle.energy =
            primitive.pressure / ((getOptions().hydro.gamma - 1.0) * primitive.density);
    }
}

// Open every pair incident on an active cell that has no live interval. Slots
// are in identifier order, so the lower slot is the lower identifier.
void hopkinsOpenPairs(HopkinsWorkspace& workspace, std::vector<size_t> const& active,
    size_t& retained, size_t& opened) {
    profile::ScopedPhase profilePhase("hydro.hopkins.flux.open");
    // Workers enumerate the pairs of their active cells from transient stencil
    // queries; the merged list is sorted and deduplicated, so active-active
    // pairs seen from both sides open once and the batch order is fixed.
    size_t const workers = hopkinsWorkerCount(active.size());
    std::vector<std::vector<std::pair<size_t, size_t>>> found(workers);
    workspace.work.listQueries += active.size();
    hopkinsParallelWorkers(active.size(), workers, [&](size_t worker, size_t k) {
        HopkinsFlopBatch operations(workspace.countOperations);
        size_t const i = active[k];
        std::vector<size_t> neighbors;
        hopkinsNeighborQuery(workspace, i, 1.0, true, operations, neighbors);
        for (size_t j : neighbors) {
            size_t const lower = std::min(i, j);
            size_t const upper = std::max(i, j);
            auto const& left = workspace.particles[lower];
            auto const& right = workspace.particles[upper];
            if (!workspace.built[lower] || !workspace.built[upper]) {
                throw std::logic_error("Hopkins pair opening requires both geometries");
            }
            if (!left.convert && !right.convert &&
                !hopkinsPairs.contains({left.state.identifier, right.state.identifier})) {
                found[worker].emplace_back(lower, upper);
            }
        }
    });
    std::vector<std::pair<size_t, size_t>> pairs;
    for (auto const& list : found) {
        pairs.insert(pairs.end(), list.begin(), list.end());
    }
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    workspace.work.pairsSolved += pairs.size();
    std::vector<HopkinsPairRecord> records(pairs.size());
    std::vector<size_t> solvePairs(pairs.size());
    for (size_t i = 0; i < pairs.size(); ++i) {
        solvePairs[i] = i;
    }
    size_t const batches = (pairs.size() + HopkinsSimd::size() - 1) / HopkinsSimd::size();
    hopkinsParallel(batches, [&](size_t batch) {
        hopkinsPairBatch(workspace, pairs, solvePairs, batch * HopkinsSimd::size(), workspace.tick,
            false, records);
    });
    retained = hopkinsPairs.size();
    opened = records.size();
    for (auto const& record : records) {
        if (!hopkinsPairs.emplace(HopkinsPairKey{record.leftId, record.rightId}, record).second) {
            throw std::runtime_error("Hopkins attempted to overwrite a live pair interval");
        }
    }
}

}    // namespace detail

namespace detail {

void hopkinsWriteSoundWave(HopkinsWorkspace const& workspace, double time) {
    profile::ScopedPhase profilePhase("hydro.hopkins.diagnostics.soundWave");
    if (workspace.dimensions != 1 || getOptions().run.scenario != "soundWave") {
        throw std::logic_error("The Hopkins sound-wave diagnostic requires its 1D scenario");
    }
    char const* name = time == 0.0 ? "sound_wave_initial.csv" : "sound_wave_final.csv";
    FILE* file = std::fopen(outputPath(name).c_str(), "w");
    if (file == nullptr) {
        throw std::runtime_error("Unable to create sound-wave diagnostics");
    }
    std::fprintf(file,
        "id,x,density,exact_density,velocity,exact_velocity,pressure,exact_pressure,"
        "support,mass,energy,rung,time\n");
    double error = 0.0;
    for (auto const& particle : workspace.particles) {
        double const x = particle.position[0].toDouble();
        double const perturbation = 1.0e-6 * std::sin(2.0 * M_PI * (x - time));
        double const pressure =
            (getOptions().hydro.gamma - 1.0) * particle.density * particle.energy;
        double const exactDensity = 1.0 + perturbation;
        error += std::abs(particle.density - exactDensity);
        std::fprintf(file,
            "%llu,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g\n",
            static_cast<unsigned long long>(particle.state.identifier), x, particle.density,
            exactDensity, particle.velocity[0], perturbation, pressure, 0.6 + perturbation,
            particle.support, particle.mass, particle.state.totalEnergy, particle.rung, time);
    }
    if (std::fclose(file) != 0) {
        throw std::runtime_error("Unable to finish sound-wave diagnostics");
    }
    PRINT("Sound-wave density L1: time=%.17g N=%zu error=%.17g\n", time, workspace.particles.size(),
        error / workspace.particles.size());
}

}    // namespace detail

namespace detail {

// Active cells: a terminal closure, a cell the gravity ladder touched (set by
// Begin), an uninitialized cell, or a cell at its own endpoint. An inactive
// cell whose stored prediction turned nonphysical is woken; the count returned
// is the number of such wakes.
size_t hopkinsMarkActive(
    HopkinsWorkspace& workspace, bool global, bool closingOnly, int minimumRung) {
    size_t woken = 0;
    for (auto& particle : workspace.particles) {
        auto const& state = particle.state;
        particle.active = closingOnly || state.hierarchicalActive || !state.predictorInitialized ||
            (global ? state.endTime <= workspace.time : state.endTick <= workspace.tick);
        if (!particle.active &&
            !hopkins::physical(hopkinsPredictedPrimitive(particle, workspace, nullptr))) {
            particle.active = true;
            particle.rung = std::max(particle.rung, minimumRung);
            ++woken;
        }
    }
    return woken;
}

void hopkinsCloseEvent(HopkinsWorkspace& workspace) {
    // The event is complete: refreshed cells resume their retained trajectories
    // in the source tree, and only touched records return to their owners.
    for (auto& particle : workspace.particles) {
        particle.active = false;
    }
    hopkinsUpdateSources(workspace, hopkinsSlotsWhere(workspace, workspace.touched));
    hopkinsInstallTouched(workspace, 1);
    workspace.eventOpen = false;
}

void hopkinsReportWork(
    HopkinsWorkspace const& workspace, double time, size_t active, bool closingOnly) {
    auto const& work = workspace.work;
    auto const treeWork = workspace.tree.counters();
    size_t halo = 0;
    for (size_t slot = 0; slot < workspace.particles.size(); ++slot) {
        halo += workspace.built[slot] && !workspace.particles[slot].active;
    }
    terminal::detailPrint("Hopkins tree: time=%.17g sorted=%llu updated=%llu refitted=%llu "
                          "rebuilds=%llu candidates=%llu\n",
        time, static_cast<unsigned long long>(treeWork.sourcesSorted),
        static_cast<unsigned long long>(treeWork.sourcesUpdated),
        static_cast<unsigned long long>(treeWork.nodesRefitted),
        static_cast<unsigned long long>(treeWork.topologyRebuilds),
        static_cast<unsigned long long>(treeWork.sourcesVisited));
    terminal::detailPrint(
        "Hopkins work: time=%.17g total=%zu active=%zu halo=%zu received=%llu "
        "receivedBytes=%llu installed=%llu installedBytes=%llu sources=%llu supports=%llu "
        "supportSums=%llu geometry=%llu rates=%llu gradients=%llu queries=%llu "
        "candidates=%llu pairs=%llu terminal=%d\n",
        time, workspace.particles.size(), active, halo,
        static_cast<unsigned long long>(work.recordsReceived),
        static_cast<unsigned long long>(work.bytesReceived),
        static_cast<unsigned long long>(work.recordsInstalled),
        static_cast<unsigned long long>(work.bytesInstalled),
        static_cast<unsigned long long>(work.sourcesUpdated),
        static_cast<unsigned long long>(work.supportCells),
        static_cast<unsigned long long>(work.supportSums),
        static_cast<unsigned long long>(work.geometryCells),
        static_cast<unsigned long long>(work.rateCells),
        static_cast<unsigned long long>(work.gradientCells),
        static_cast<unsigned long long>(work.listQueries),
        static_cast<unsigned long long>(treeWork.sourcesVisited),
        static_cast<unsigned long long>(work.pairsSolved), int(closingOnly));
}

}    // namespace detail

void hopkinsSoundWaveDiagnostics(double time) {
    profile::ScopedPhase profilePhase("hydro.hopkins.diagnostics");
    auto& resident = detail::hopkinsWorkspace();
    resident.countOperations = false;
    // Refresh the resident records at the diagnostic time, then solve every
    // support and volume in a scratch copy: the diagnostic changes no retained
    // support, source trajectory, pair interval or public mirror.
    detail::hopkinsOpenEvent(resident, 1.0, 0.0, time, 0.0, true, false);
    resident.eventOpen = false;
    resident.countOperations = true;
    detail::HopkinsWorkspace scratch = resident;
    scratch.countOperations = false;
    for (auto& particle : scratch.particles) {
        particle.active = true;
    }
    detail::hopkinsRebuildSources(scratch);
    scratch.tree.prepare(scratch.time);
    detail::hopkinsEnsureGeometry(scratch);
    detail::hopkinsWriteSoundWave(scratch, time);
    {
        profile::ScopedPhase profilePhase("hydro.hopkins.caches.clear");
        sphParticlesClearCaches();
        particlesCacheFree();
    }
}

void hopkinsPrepareGravityMasses(double time) {
    if (getOptions().hydro.method != "mfv") {
        return;
    }
    if (!(time >= 0.0) || !std::isfinite(time)) {
        throw std::invalid_argument("Invalid Hopkins gravity projection time");
    }
    std::map<uint64_t, double> partial;
    for (auto const& [key, record] : detail::hopkinsPairs) {
        double const interval = record.endTime - record.beginTime;
        double const duration = std::max(0.0, std::min(time, record.endTime) - record.beginTime);
        auto const transfer =
            hopkins::integrateFlux(record.initial, record.midpoint, interval, 0.0, duration);
        partial[record.leftId] -= transfer.mass;
        partial[record.rightId] += transfer.mass;
    }
    std::vector<std::pair<uint64_t, double>> values(partial.begin(), partial.end());
    std::vector<hpx::future<void>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<HopkinsInstallGravityMassesAction>(locality, values));
    }
    for (auto& future : pending) {
        future.get();
    }
    // No force workers are live yet; invalidate source-mass projections before
    // migration, remote source imports, or multipole/mesh construction begins.
    sphParticlesClearCaches();
    particlesCacheFree();
}

void hopkinsPrepareTree() {
    if (detail::hopkinsTreePrepared) {
        throw std::runtime_error("Hopkins tree preparation was not consumed");
    }
    auto& workspace = detail::hopkinsWorkspace();
    workspace.tree.resetCounters();
    workspace.work = {};
    detail::hopkinsTreePrepared = true;
}

void hopkinsPrepareGeometry(
    double scale, double expansionRate, double time, double baseTimestep, bool global) {
    if (!detail::hopkinsTreePrepared || detail::hopkinsEventPrepared) {
        throw std::runtime_error("Invalid Hopkins geometry preparation order");
    }
    auto& workspace = detail::hopkinsWorkspace();
    detail::hopkinsOpenEvent(workspace, scale, expansionRate, time, baseTimestep, global, false);
    // Wake rungs are assigned by hopkinsKick; here the set only selects geometry.
    detail::hopkinsMarkActive(workspace, global, false, 0);
    detail::hopkinsRebuildForActivity(workspace);
    detail::hopkinsEnsureGeometry(workspace);
    detail::hopkinsEventPrepared = true;
}

HopkinsStepResult hopkinsHierarchicalBegin(int minimumGravityRung, double scale,
    double expansionRate, double time, double baseTimestep, EnergiesT& energies, bool closingOnly) {
    profile::ScopedPhase profilePhase("hydro.hopkins.hierarchy.close");
    if (!getOptions().run.hierarchicalGravity || getOptions().hydro.method != "mfm" ||
        detail::hopkinsHierarchyPending || detail::hopkinsEventPrepared) {
        throw std::runtime_error("Invalid MFM hierarchical begin order");
    }
    detail::hopkinsHierarchyWoken = 0;
    hopkinsPrepareTree();
    auto& workspace = detail::hopkinsWorkspace();
    detail::hopkinsOpenEvent(workspace, scale, expansionRate, time, baseTimestep, false, true);
    size_t gravityActive = 0, hydroDue = 0, closed = 0;
    int maximumHydroRung = getOptions().run.minRung;
    for (auto& particle : workspace.particles) {
        bool const gravityTouched = particle.gravityRung >= minimumGravityRung;
        bool const due =
            !particle.state.predictorInitialized || particle.state.endTick <= workspace.tick;
        bool const invalidPrediction = particle.state.predictorInitialized &&
            !hopkins::physical(detail::hopkinsPredictedPrimitive(particle, workspace, nullptr));
        particle.active = closingOnly || gravityTouched || due || invalidPrediction;
        detail::hopkinsHierarchyWoken += invalidPrediction && !gravityTouched && !due;
        gravityActive += gravityTouched;
        hydroDue += due;
        maximumHydroRung = std::max(maximumHydroRung, particle.rung);
    }
    detail::hopkinsRebuildForActivity(workspace);
    detail::hopkinsEnsureGeometry(workspace);
    detail::hopkinsInitialize(workspace);
    for (auto& particle : workspace.particles) {
        particle.state.hierarchicalActive = particle.active;
    }
    size_t const settled = detail::hopkinsSettlePairs(workspace, closingOnly);
    for (size_t i = 0; i < workspace.particles.size(); ++i) {
        if (workspace.particles[i].active) {
            detail::hopkinsCloseParticle(workspace, i);
            ++closed;
        }
    }
    detail::hopkinsCurrentViews(workspace);
    // Phase two publishes closed anchors before any signed gravity impulse but
    // leaves raw coordinates at the current gravity endpoint throughout the ladder.
    detail::hopkinsInstallTouched(workspace, 2);
    auto const synchronized = particlesSumEnergies();
    // Source closure precedes every gravity impulse. The closed thermal energy
    // is invariant under those kicks. Gravity replaces kinetic/momentum totals
    // with its synchronized top-ladder reduction when self-gravity is enabled.
    energies.kin = synchronized.kin / (scale * scale);
    energies.therm = synchronized.therm / (scale * scale);
    energies.xmom = synchronized.xmom / scale;
    energies.ymom = synchronized.ymom / scale;
    energies.zmom = synchronized.zmom / scale;
    energies.nmom = synchronized.nmom / scale;
    detail::hopkinsHierarchyPending = true;
    detail::hopkinsEventPrepared = true;
    terminal::detailPrint("Hopkins hierarchy begin: time=%.17g gravityMinRung=%d "
                          "gravityActive=%zu hydroDue=%zu closed=%zu total=%zu settled=%zu "
                          "hydroMaxRung=%d terminal=%d\n",
        time, minimumGravityRung, gravityActive, hydroDue, closed, workspace.particles.size(),
        settled, maximumHydroRung, int(closingOnly));
    return {maximumHydroRung, std::ldexp(baseTimestep, -maximumHydroRung)};
}

double hopkinsHierarchicalDriftThermalEnergy() {
    if (detail::hopkinsHierarchyPending) {
        throw std::runtime_error("MFM drift thermal energy requires a completed gravity ladder");
    }
    return detail::hopkinsHierarchyDriftThermal;
}

HopkinsStepResult hopkinsHierarchicalFinish(int minimumHydroRung, double scale,
    double expansionRate, double time, double baseTimestep, EnergiesT& energies, bool closingOnly) {
    if (!detail::hopkinsHierarchyPending) {
        throw std::runtime_error("MFM hierarchical finish has no matching begin");
    }
    auto const result = hopkinsKick(
        minimumHydroRung, scale, expansionRate, time, baseTimestep, energies, closingOnly);
    detail::hopkinsHierarchyPending = false;
    return result;
}

HopkinsStepResult hopkinsKick(int minimumRung, double scale, double expansionRate, double time,
    double baseTimestep, EnergiesT& energies, bool closingOnly, double previousTimestep,
    double maximumGlobalTimestep) {
    profile::ScopedPhase profilePhase("hydro.hopkins");
    bool const global = maximumGlobalTimestep > 0.0;
    bool const hierarchical = getOptions().run.hierarchicalGravity;
    if (!(maximumGlobalTimestep >= 0.0) || !std::isfinite(maximumGlobalTimestep) ||
        (global && (!(previousTimestep >= 0.0) || !std::isfinite(previousTimestep)))) {
        throw std::invalid_argument("Hopkins global timesteps must be finite and nonnegative");
    }
    if (!detail::hopkinsTreePrepared) {
        hopkinsPrepareTree();
    }
    detail::hopkinsEventPrepared = false;
    auto& workspace = detail::hopkinsWorkspace();
    // Gravity has joined: now it is safe to import fresh accelerations and the
    // dark-matter rung reduction. Geometry solved before the kick stays valid.
    detail::hopkinsOpenEvent(workspace, scale, expansionRate, time, baseTimestep, global, true);
    size_t woken = hierarchical ? detail::hopkinsHierarchyWoken : 0;
    woken += detail::hopkinsMarkActive(workspace, global, closingOnly, global ? 0 : minimumRung);
    detail::hopkinsRebuildForActivity(workspace);
    detail::hopkinsEnsureGeometry(workspace);
    detail::hopkinsInitialize(workspace);
    std::vector<unsigned char> closed(workspace.particles.size(), 0);
    HopkinsStepResult result;
    result.maximumRung = hierarchical ? 0 : workspace.darkMatterRung;
    result.minimumTimestep = maximumGlobalTimestep;
    size_t settled = 0;
    size_t converted = 0;
    bool firstSelection = true;
    // The refreshed set A ∪ N(A) of the last geometry pass: the loop below only
    // exits once a timestep selection woke nobody, so it is still current then.
    std::vector<size_t> refreshed;
    while (true) {
        settled += detail::hopkinsSettlePairs(workspace, closingOnly);
        refreshed = detail::hopkinsEnsureGeometry(workspace);
        {
            profile::ScopedPhase closePhase("hydro.hopkins.close");
            std::vector<size_t> closing;
            std::vector<double> durations;
            for (size_t i = 0; i < workspace.particles.size(); ++i) {
                if (workspace.particles[i].active && !closed[i]) {
                    closing.push_back(i);
                    durations.push_back(time - workspace.particles[i].state.sourceTime);
                    detail::hopkinsCloseParticle(workspace, i);
                    closed[i] = 1;
                }
            }
            detail::hopkinsApplySources(workspace, closing, durations, energies, converted);
            detail::hopkinsSynchronizeEntropy(workspace, closing);
        }
        detail::hopkinsCurrentViews(workspace);
        if (closingOnly) {
            break;
        }
        result = detail::hopkinsSelectTimesteps(workspace, global ? 0 : minimumRung, baseTimestep,
            maximumGlobalTimestep, woken, firstSelection);
        firstSelection = false;
        bool newlyWoken = false;
        for (size_t i = 0; i < workspace.particles.size(); ++i) {
            newlyWoken = newlyWoken || (workspace.particles[i].active && !closed[i]);
        }
        if (!newlyWoken) {
            break;
        }
    }
    // The pressure switch added the entropy estimate's excess over the total-energy
    // closure; book it like the thermal sources so the energy budget stays exact.
    energies.heating -= workspace.switchedEnergy / scale;
    size_t active = 0;
    int minimumGasRung = kMaxRung, maximumGasRung = 0;
    for (auto& particle : workspace.particles) {
        active += particle.active;
        minimumGasRung = std::min(minimumGasRung, particle.rung);
        maximumGasRung = std::max(maximumGasRung, particle.rung);
        if (particle.active && !closingOnly) {
            double const duration =
                global ? result.minimumTimestep : std::ldexp(baseTimestep, -particle.rung);
            particle.state.endTime = time + duration;
            particle.state.endTick = global ?
                0 :
                workspace.tick + (uint64_t(1) << (detail::kHopkinsTickRung - particle.rung));
        }
    }
    workspace.globalTimestep = global ? result.minimumTimestep : 0.0;
    size_t retained = detail::hopkinsPairs.size(), opened = 0;
    if (!closingOnly) {
        auto const activeSlots = detail::hopkinsActiveSlots(workspace);
        detail::hopkinsGradients(workspace, activeSlots, refreshed);
        detail::hopkinsGeometryRates(workspace, refreshed);
        for (size_t i : activeSlots) {
            auto& particle = workspace.particles[i];
            auto& state = particle.state;
            state.meshVelocity = particle.velocity;
            if (getOptions().run.hierarchicalGravity) {
                state.positionDriftScale = scale;
#if COSMICTIGER_WITH_GRAVITY
                if (getOptions().run.scenario == "cosmology") {
                    double scaleWork = scale;
                    double derivativeWork = expansionRate / scale;
                    state.positionDriftScale =
                        1.0 / cosmosAinv(derivativeWork, scaleWork, state.endTime - time);
                }
#endif
            }
            for (int dim = 0; dim < 3; ++dim) {
                state.meshAcceleration[dim] = state.sourceAcceleration[dim] -
                    workspace.gradients[i].pressure[dim] / (particle.density * scale) -
                    getOptions().hydro.damping * particle.velocity[dim];
            }
        }
        if (time == 0.0 && getOptions().run.scenario == "soundWave") {
            detail::hopkinsWriteSoundWave(workspace, time);
        }
        detail::hopkinsOpenPairs(workspace, activeSlots, retained, opened);
    } else {
        result.maximumRung = std::max(result.maximumRung, maximumGasRung);
        if (!detail::hopkinsPairs.empty()) {
            throw std::runtime_error("Hopkins terminal closure left a live pair interval");
        }
    }
    detail::hopkinsCloseEvent(workspace);
    {
        profile::ScopedPhase energyPhase("hydro.hopkins.diagnostics.energies");
        auto const sum = particlesSumEnergies();
        if (hierarchical) {
            // This sum includes all descending/opening HOLD impulses. It is
            // the drift kinetic accounting, not the synchronized diagnostic
            // paired with the top-ladder potential evaluation.
            energies.tckin = sum.kin;
            detail::hopkinsHierarchyDriftThermal = sum.therm / (scale * scale);
        } else {
            energies.kin = sum.kin / (scale * scale);
            energies.therm = sum.therm / (scale * scale);
            energies.xmom = sum.xmom / scale;
            energies.ymom = sum.ymom / scale;
            energies.zmom = sum.zmom / scale;
            energies.nmom = sum.nmom / scale;
        }
    }
    terminal::detailPrint("Hopkins step: time=%.17g active=%zu total=%zu minRung=%d maxRung=%d "
                          "pairs=%zu cached=%zu refreshed=%zu woken=%zu settled=%zu stars=%zu "
                          "switched=%zu terminal=%d\n",
        time, active, workspace.particles.size(), minimumGasRung, maximumGasRung, retained + opened,
        retained, opened, woken, settled, converted, workspace.switched, int(closingOnly));
    if (hierarchical) {
        int minimumGravityRung = kMaxRung, maximumGravityRung = 0;
        size_t futureProjections = 0;
        for (auto const& particle : workspace.particles) {
            minimumGravityRung = std::min(minimumGravityRung, particle.gravityRung);
            maximumGravityRung = std::max(maximumGravityRung, particle.gravityRung);
            futureProjections += particle.state.gravityEndTime > time;
        }
        terminal::detailPrint("Hopkins hierarchy finish: time=%.17g hydroMinRung=%d "
                              "hydroMaxRung=%d gravityMinRung=%d gravityMaxRung=%d "
                              "projected=%zu total=%zu terminal=%d\n",
            time, minimumGasRung, maximumGasRung, minimumGravityRung, maximumGravityRung,
            futureProjections, workspace.particles.size(), int(closingOnly));
    }
    detail::hopkinsReportWork(workspace, time, active, closingOnly);
    detail::hopkinsTreePrepared = false;
    {
        profile::ScopedPhase cachePhase("hydro.hopkins.caches.clear");
        sphParticlesClearCaches();
        particlesCacheFree();
    }
#if COSMICTIGER_ENABLE_STAR_FORMATION
    if (getOptions().starFormation.enabled) {
        // Gas gravity belongs to the meshless momentum update, so a newborn star
        // missed the collisionless proposal and opening half kick; the global
        // integrator's finish kick owns both, as it does for SPH.
        profile::ScopedPhase starPhase("hydro.hopkins.openNewStars");
        double maximumTimestep = std::ldexp(baseTimestep, -getOptions().run.minRung);
#if COSMICTIGER_WITH_GRAVITY
        if (getOptions().run.scenario == "cosmology") {
            maximumTimestep = std::min(maximumTimestep,
                cosmology::expansionTimestep(
                    scale, expansionRate / scale, cosmosD2adt2(scale), 0.01, nullptr));
        }
#endif
        int const newStarRung = starFormationOpenNewStars(baseTimestep, scale, maximumTimestep,
            minimumRung, getOptions().gravity.enabled && !global && !closingOnly);
        result.maximumRung = std::max(result.maximumRung, newStarRung);
    }
#endif
    return result;
}

void hopkinsDomainsRebound() {
    detail::hopkinsRebuildOnNextEvent = true;
}

void hopkinsResetSources() {
    detail::hopkinsResident.reset();
    detail::hopkinsResetOwnerMirror();
    detail::hopkinsRebuildOnNextEvent = false;
    detail::hopkinsTreePrepared = false;
    detail::hopkinsEventPrepared = false;
    detail::hopkinsHierarchyPending = false;
    detail::hopkinsHierarchyWoken = 0;
    detail::hopkinsHierarchyDriftThermal = 0.0;
}

void hopkinsWriteCheckpoint(FILE* file) {
    if (detail::hopkinsHierarchyPending) {
        throw std::runtime_error("Cannot checkpoint an unfinished MFM gravity ladder");
    }
    static_assert(sizeof(detail::HopkinsPairRecord) == 176);
    uint64_t header[4] = {detail::kHopkinsCacheMagic, 2, sizeof(detail::HopkinsPairRecord),
        hpxRank() == 0 ? uint64_t(detail::hopkinsPairs.size()) : 0};
    FWRITE(header, sizeof(uint64_t), 4, file);
    if (hpxRank() == 0) {
        for (auto& [key, record] : detail::hopkinsPairs) {
            FWRITE(&record, sizeof(record), 1, file);
        }
    }
}

void hopkinsReadCheckpoint(FILE* file) {
    hopkinsResetSources();
    uint64_t header[4];
    FREAD(header, sizeof(uint64_t), 4, file);
    if (header[0] != detail::kHopkinsCacheMagic || header[1] != 2 ||
        header[2] != sizeof(detail::HopkinsPairRecord) || (hpxRank() != 0 && header[3] != 0)) {
        throw std::runtime_error("Incompatible Hopkins pair-cache checkpoint");
    }
    detail::hopkinsPairs.clear();
    for (uint64_t i = 0; i < header[3]; i++) {
        detail::HopkinsPairRecord record;
        FREAD(&record, sizeof(record), 1, file);
        if (record.leftId >= record.rightId || !(record.beginTime < record.endTime) ||
            !std::isfinite(record.beginTime) || !std::isfinite(record.endTime) ||
            !detail::hopkinsPairs
                .emplace(detail::HopkinsPairKey{record.leftId, record.rightId}, record)
                .second) {
            throw std::runtime_error("Invalid Hopkins pair-cache checkpoint record");
        }
    }
}
