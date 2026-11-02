/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
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
#include <cosmictiger/hydro/sph/sph.hpp>
#include <cosmictiger/hydro/sph/sph_flops.hpp>
#include <cosmictiger/hydro/sph/sph_kernel.hpp>
#include <cosmictiger/hydro/sph/sph_particles.hpp>
#include <cosmictiger/hydro/sph/sph_tree.hpp>
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
    // GADGET-4 source state travels with the gathered cell; the cooling guess and
    // the SH03 metadata are read at collection and written back at the scatter.
    gadget4::State chemistry{};
    StarFormationState starFormation{};
    // A whole-cell conversion decided at this closure: the cell keeps its closing
    // state, opens no pair interval, and leaves the gas set when it is installed.
    bool convert = false;
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & position & velocity & acceleration & state & active & mass & energy & support &
            density & rung & chemistry & starFormation & convert;
    }
};

struct HopkinsLeaf {
    ParticleInteger begin = 0;
    ParticleInteger end = 0;
    std::vector<ParticleGlobalRange> neighbors;
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & begin & end & neighbors;
    }
};

struct HopkinsBlock {
    std::vector<HopkinsParticle> particles;
    std::vector<HopkinsLeaf> leaves;
    int darkMatterRung = 0;
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & particles & leaves & darkMatterRung;
    }
};

struct HopkinsWorkspace {
    bool countOperations = true;
    std::vector<HopkinsParticle> particles;
    std::vector<size_t> offsets;
    std::vector<std::vector<size_t>> neighbors;
    std::vector<hopkins::Geometry> geometry;
    std::vector<hopkins::GeometryRate> geometryRates;
    std::vector<double> supportRates;
    std::vector<hopkins::Primitive> primitives;
    std::vector<hopkins::Gradient> gradients;
    double expansionRate = 0.0;
    double scale = 1.0;
    double tickDuration = 0.0;
    double globalTimestep = 0.0;
    double time = 0.0;
    bool readForces = false;
    uint64_t tick = 0;
    double searchSupportFactor = 1.0;
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
std::unique_ptr<HopkinsWorkspace> hopkinsPrepared;
bool hopkinsTreePrepared = false;
static_assert(kMaxRung <= std::numeric_limits<uint16_t>::digits);
constexpr int kHopkinsTickRung = kMaxRung - 1;
constexpr uint64_t kHopkinsTicksPerBase = uint64_t(1) << kHopkinsTickRung;
constexpr uint64_t kHopkinsCacheMagic = 0x484f504341434845ULL;
// The midpoint geometry differentiates the constant kernel-count constraint.
// Solve in double precision: a float support and a loose residual can obscure
// the 1e-6 sound-wave perturbation. Only the tree's support mirror is rounded.
constexpr double kHopkinsSupportTolerance = 1.0e-13;

template <class Function>
void hopkinsParallel(size_t count, Function const& function) {
    size_t const workers = std::min(count, size_t(std::max(1, hpxHardwareConcurrency())));
    std::vector<hpx::future<void>> pending;
    for (size_t worker = 0; worker < workers; worker++) {
        pending.push_back(hpx::async([&, worker]() {
            for (size_t i = count * worker / workers; i < count * (worker + 1) / workers; i++) {
                function(i);
            }
        }));
    }
    hpx::wait_all(pending);
    for (auto& future : pending) {
        future.get();
    }
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
void hopkinsNeighborBatches(HopkinsWorkspace const& workspace, size_t i, FlopCount* operations,
    Function const& function, bool includeRadius = true) {
    auto const& neighbors = workspace.neighbors[i];
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

hopkins::Vector hopkinsAcceleration(ParticleInteger index, FlopCount* operations) {
    auto const& options = getOptions();
    // Fixed-point conversion multiplies and imposed-force additions, plus
    // float acceleration promotions when gravity supplies the stored force.
    hopkins::detail::countFlops(operations, 6 + (options.gravity.enabled ? 3 : 0));
    hopkins::Vector acceleration{};
    hopkins::Vector displacement{};
    for (int dim = 0; dim < 3; dim++) {
        acceleration[dim] =
            options.gravity.enabled ? sphParticlesGravityAcceleration(dim, index) : 0.0;
        displacement[dim] = doubleDistance(sphParticlesPosition(dim, index), Position(0.5));
    }
    acceleration[1] += options.hydro.yAcceleration;
    double const z = doubleDistance(sphParticlesPosition(2, index), Position(0.5));
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

HopkinsBlock hopkinsCollect(bool countOperations, bool readForces) {
    HopkinsFlopBatch operations(countOperations);
    HopkinsBlock block;
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        if (readForces && particlesType(i) == kDarkMatterType) {
            block.darkMatterRung = std::max(block.darkMatterRung, int(particlesRung(i)));
        }
    }
    block.particles.resize(sphParticlesSize());
    for (ParticleInteger i = 0; i < sphParticlesSize(); i++) {
        auto& particle = block.particles[i];
        particle.state = sphParticlesHopkinsState(i);
        hopkins::detail::countFlops(operations,
            particle.state.initialized ? 0 : (getOptions().hydro.method == "mfv" ? 1 : 2));
        particle.mass = particle.state.initialized ? particle.state.mass : sphParticlesMass(i);
        hopkins::detail::countFlops(operations, particle.state.initialized ? 5 : 1);
        particle.energy = particle.state.initialized ? particle.state.totalEnergy / particle.mass -
                0.5 * hopkins::dot(particle.state.momentum, particle.state.momentum, operations) /
                    (particle.mass * particle.mass) :
                                                       sphParticlesInternalEnergy(i);
        hopkins::detail::countFlops(operations, particle.state.supportRadius > 0.0 ? 0 : 1);
        particle.support = particle.state.supportRadius > 0.0 ? particle.state.supportRadius :
                                                                sphParticlesSupportRadius(i);
        particle.rung = sphParticlesRung(i);
        if (getOptions().chemistry.enabled) {
            particle.chemistry = sphParticlesChemistryStates[i];
        }
        if (getOptions().starFormation.enabled) {
            particle.starFormation = particlesStarFormationStates[sphParticlesParticleIndex(i)];
        }
        particle.acceleration = readForces ? hopkinsAcceleration(i, operations) : hopkins::Vector{};
        for (int dim = 0; dim < 3; dim++) {
            particle.position[dim] = sphParticlesPosition(dim, i);
            hopkins::detail::countFlops(operations, 1);
            particle.velocity[dim] = particle.state.initialized ?
                particle.state.momentum[dim] / particle.mass :
                sphParticlesVelocity(dim, i);
        }
    }
    for (int leafIndex = 0; leafIndex < sphTreeLeaflistSize(); leafIndex++) {
        auto const* node = sphTreeGetNode(sphTreeGetLeaf(leafIndex));
        if (!node->isLocalHere()) {
            continue;
        }
        HopkinsLeaf leaf;
        leaf.begin = node->particleRange.first;
        leaf.end = node->particleRange.second;
        for (int j = node->neighborRange.first; j < node->neighborRange.second; j++) {
            leaf.neighbors.push_back(sphTreeGetNode(sphTreeGetNeighbor(j))->globalParticleRange());
        }
        block.leaves.push_back(std::move(leaf));
    }
    return block;
}

// The ordinary drifter consumes only the separately stored mesh velocity.
// Physical mass, momentum and energy are never temporarily overwritten by it.
void hopkinsInstall(
    std::vector<HopkinsParticle> const& particles, int phase, bool countOperations) {
    HopkinsFlopBatch operations(countOperations);
    ALWAYS_ASSERT(particles.size() == size_t(sphParticlesSize()));
    std::vector<ParticleInteger> converted;
    for (size_t i = 0; i < particles.size(); i++) {
        auto const& particle = particles[i];
        hopkins::detail::countFlops(operations, 2);
        sphParticlesSupportRadius(i) = particle.support;
        sphParticlesDensity(i) = particle.density;
        if (phase == 1) {
            sphParticlesRung(i) = particle.rung;
            sphParticlesHopkinsState(i) = particle.state;
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
                    converted.push_back(ParticleInteger(i));
                }
            }
        }
        sphParticlesHopkinsState(i).supportRadius = particle.support;
    }
    if (converted.empty()) {
        return;
    }
#if COSMICTIGER_ENABLE_STAR_FORMATION
    // Remove converted gas last and highest index first: each removal swaps the
    // current last particle, whose installed state is already complete, into the slot.
    for (auto index = converted.rbegin(); index != converted.rend(); ++index) {
        starFormationConvertGas(*index, particles[*index].starFormation);
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

HPX_PLAIN_ACTION(detail::hopkinsCollect, HopkinsCollectAction);
HPX_PLAIN_ACTION(detail::hopkinsInstall, HopkinsInstallAction);
HPX_PLAIN_ACTION(detail::hopkinsInstallGravityMasses, HopkinsInstallGravityMassesAction);

namespace detail {

void hopkinsScatter(HopkinsWorkspace const& workspace, int phase) {
    profile::ScopedPhase profilePhase("hydro.hopkins.scatter");
    std::vector<hpx::future<void>> pending;
    for (int rank = 0; rank < hpxSize(); rank++) {
        size_t const begin = workspace.offsets[rank];
        size_t const end = workspace.offsets[rank + 1];
        std::vector<HopkinsParticle> block(
            workspace.particles.begin() + begin, workspace.particles.begin() + end);
        pending.push_back(hpx::async<HopkinsInstallAction>(
            hpxLocalities()[rank], block, phase, workspace.countOperations));
    }
    hpx::wait_all(pending);
    for (auto& future : pending) {
        future.get();
    }
}

void hopkinsGather(HopkinsWorkspace& workspace) {
    profile::ScopedPhase profilePhase("hydro.hopkins.gather");
    HopkinsFlopBatch operations(workspace.countOperations);
    std::vector<hpx::future<HopkinsBlock>> pending;
    for (auto const& locality : hpxLocalities()) {
        pending.push_back(hpx::async<HopkinsCollectAction>(
            locality, workspace.countOperations, workspace.readForces));
    }
    std::vector<HopkinsBlock> blocks;
    workspace.particles.clear();
    workspace.darkMatterRung = 0;
    workspace.offsets = {0};
    for (auto& future : pending) {
        blocks.push_back(future.get());
        workspace.darkMatterRung = std::max(workspace.darkMatterRung, blocks.back().darkMatterRung);
        auto const& particles = blocks.back().particles;
        workspace.particles.insert(workspace.particles.end(), particles.begin(), particles.end());
        workspace.offsets.push_back(workspace.particles.size());
    }
    workspace.neighbors.clear();
    workspace.neighbors.resize(workspace.particles.size());
    for (size_t rank = 0; rank < blocks.size(); rank++) {
        for (auto const& leaf : blocks[rank].leaves) {
            for (ParticleInteger local = leaf.begin; local < leaf.end; local++) {
                size_t const i = workspace.offsets[rank] + local;
                auto& neighbors = workspace.neighbors[i];
                for (auto const& range : leaf.neighbors) {
                    for (ParticleInteger index = range.range.first; index < range.range.second;
                        index++) {
                        size_t const j = workspace.offsets[range.proc] + index;
                        hopkins::detail::countFlops(operations, i != j ? 1 : 0);
                        if (i != j &&
                            hopkins::norm(hopkinsSeparation(workspace.particles[i],
                                              workspace.particles[j], operations),
                                operations) < workspace.searchSupportFactor *
                                    std::max(workspace.particles[i].support,
                                        workspace.particles[j].support)) {
                            neighbors.push_back(j);
                        }
                    }
                }
                std::sort(neighbors.begin(), neighbors.end());
                neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
            }
        }
    }
    // Form the union of directed searches. This makes reciprocity explicit
    // even when rounding at a tree-box boundary includes a pair on one side.
    for (size_t i = 0; i < workspace.neighbors.size(); i++) {
        for (size_t j : workspace.neighbors[i]) {
            auto& reciprocal = workspace.neighbors[j];
            auto const location = std::lower_bound(reciprocal.begin(), reciprocal.end(), i);
            if (location == reciprocal.end() || *location != i) {
                reciprocal.insert(location, i);
            }
        }
    }
    // Local tree order can change after redistribution or a restart. Keep
    // every floating-point neighbor reduction in stable particle-ID order.
    for (auto& neighbors : workspace.neighbors) {
        std::sort(neighbors.begin(), neighbors.end(), [&](size_t left, size_t right) {
            auto const leftId = workspace.particles[left].state.identifier;
            auto const rightId = workspace.particles[right].state.identifier;
            return leftId == rightId ? left < right : leftId < rightId;
        });
    }
}

void hopkinsNeighborRefresh(bool boxes) {
    profile::ScopedPhase profilePhase("hydro.hopkins.neighbors");
    SphTreeNeighborParameters parameters;
    parameters.minimumRung = 0;
    parameters.supportRadiusMultiplier = 1.2;
    TreeId root;
    root.proc = 0;
    root.index = 0;
    if (boxes) {
        parameters.innerSelection = kSphSetAll;
        parameters.outerSelection = kSphSetAll;
        parameters.runType = kSphTreeNeighborBoxes;
        {
            profile::ScopedPhase profilePhase("hydro.hopkins.neighbors.boxes");
            sphTreeNeighbor(parameters, root, Vector<TreeId>()).get();
        }
    }
    parameters.runType = kSphTreeNeighborNeighbors;
    parameters.innerSelection = kSphInteractionsIJ;
    Vector<TreeId> checklist;
    checklist.push_back(root);
    {
        profile::ScopedPhase profilePhase("hydro.hopkins.neighbors.refresh");
        sphTreeNeighbor(parameters, root, checklist).get();
    }
}

void hopkinsSolveSupport(HopkinsWorkspace& workspace) {
    profile::ScopedPhase profilePhase("hydro.hopkins.support");
    HopkinsFlopBatch operations(workspace.countOperations);
    auto const& options = getOptions();
    double const minimum = options.hydro.minSupportRadius;
    double const maximum = options.hydro.maxSupportRadius;
    double const target = hopkinsNeighborTarget(workspace.dimensions, operations);
    workspace.searchSupportFactor = 1.2;
    std::vector<double> lower, upper;
    for (int iteration = 0; iteration < 256; iteration++) {
        hopkinsGather(workspace);
        if (iteration == 0) {
            lower.assign(workspace.particles.size(), minimum);
            upper.assign(workspace.particles.size(), maximum);
        }
        std::vector<unsigned char> changed(workspace.particles.size(), 0);
        {
            profile::ScopedPhase profilePhase("hydro.hopkins.support.solve");
            hopkinsParallel(workspace.particles.size(), [&](size_t i) {
                HopkinsFlopBatch operations(workspace.countOperations);
                auto& particle = workspace.particles[i];
                double const support = std::clamp(particle.support, minimum, maximum);
                double next = support;
                if (support == particle.support) {
                    double sum = hopkinsKernelValue(0.0, workspace.dimensions, operations),
                           derivative = 0.0;
                    hopkinsNeighborBatches(
                        workspace, i, operations, [&](HopkinsNeighborBatch const& batch) {
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
                        maximum, lower[i], upper[i], false, operations);
                }
                // Retain the solved radius in the meshless sidecar; the SPH tree
                // consumes a float mirror with its existing padded search bounds.
                changed[i] = next != particle.support;
                particle.support = next;
            });
        }
        if (std::none_of(changed.begin(), changed.end(), [](auto value) { return value != 0; })) {
            workspace.searchSupportFactor = 1.0;
            hopkinsGather(workspace);
            return;
        }
        hopkinsScatter(workspace, 0);
        {
            profile::ScopedPhase profilePhase("hydro.hopkins.caches.clear");
            sphParticlesClearCaches();
        }
        hopkinsNeighborRefresh(true);
    }
    throw std::runtime_error("Hopkins kernel-support solve did not converge");
}

void hopkinsGeometry(HopkinsWorkspace& workspace) {
    profile::ScopedPhase profilePhase("hydro.hopkins.geometry");
    HopkinsFlopBatch operations(workspace.countOperations);
    workspace.geometry.resize(workspace.particles.size());
    for (int attempt = 0; attempt < 16; attempt++) {
        {
            profile::ScopedPhase profilePhase("hydro.hopkins.geometry.solve");
            hopkinsParallel(workspace.particles.size(), [&](size_t i) {
                HopkinsFlopBatch operations(workspace.countOperations);
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
        bool valid = true;
        for (size_t i = 0; i < workspace.particles.size(); i++) {
            if (!workspace.geometry[i].valid) {
                valid = false;
                auto& support = workspace.particles[i].support;
                if (support >= getOptions().hydro.maxSupportRadius) {
                    throw std::runtime_error(
                        "Hopkins gradient remains singular at hydro.maxSupportRadius");
                }
                hopkins::detail::countFlops(operations, 1);
                support = std::min(1.5 * support, getOptions().hydro.maxSupportRadius);
            }
        }
        if (valid) {
            for (size_t i = 0; i < workspace.particles.size(); i++) {
                hopkins::detail::countFlops(operations, 1);
                workspace.particles[i].density =
                    workspace.particles[i].mass / workspace.geometry[i].volume;
            }
            return;
        }
        // Appendix C: expand the actual tree search, never invert a regularized
        // singular moment or silently substitute the SPH pressure operator.
        hopkinsScatter(workspace, 0);
        {
            profile::ScopedPhase profilePhase("hydro.hopkins.caches.clear");
            sphParticlesClearCaches();
        }
        hopkinsNeighborRefresh(true);
        hopkinsGather(workspace);
    }
    throw std::runtime_error(
        "Hopkins gradient matrix did not converge after 16 support expansions");
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

void hopkinsGeometryRates(HopkinsWorkspace& workspace) {
    profile::ScopedPhase profilePhase("hydro.hopkins.geometryRates");
    workspace.geometryRates.resize(workspace.particles.size());
    workspace.supportRates.resize(workspace.particles.size());
    hopkinsParallel(workspace.particles.size(), [&](size_t i) {
        HopkinsFlopBatch operations(workspace.countOperations);
        auto const& particle = workspace.particles[i];
        double radialSum = 0.0, supportSum = 0.0;
        hopkinsNeighborBatches(workspace, i, operations, [&](HopkinsNeighborBatch const& batch) {
            auto const relative = hopkinsRelativeVelocities(workspace, i, batch, operations);
            hopkins::detail::countFlops(operations, 4 * HopkinsSimd::size());
            auto const q = batch.radius / HopkinsSimd(particle.support);
            auto const derivative = hopkinsKernelDerivative(q, workspace.dimensions, operations);
            auto const safeRadius =
                simdSelect(batch.radius > HopkinsSimd(0.0), batch.radius, HopkinsSimd(1.0));
            auto const radialTerms =
                derivative * hopkinsSimdDot(batch.separation, relative, operations) / safeRadius;
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
        hopkinsNeighborBatches(workspace, i, operations, [&](HopkinsNeighborBatch const& batch) {
            auto const& separation = batch.separation;
            auto const relative = hopkinsRelativeVelocities(workspace, i, batch, operations);
            auto const safeRadius =
                simdSelect(batch.radius > HopkinsSimd(0.0), batch.radius, HopkinsSimd(1.0));
            hopkins::detail::countFlops(operations, HopkinsSimd::size());
            auto const radiusRate = hopkinsSimdDot(separation, relative, operations) / safeRadius;
            auto const kernel =
                hopkinsKernel(batch.radius, particle.support, workspace.dimensions, operations);
            auto const kernelRate = hopkinsKernelRate(batch.radius, radiusRate, particle.support,
                supportRate, workspace.dimensions, operations);
            HopkinsSimdMatrix moments, momentRates;
            hopkins::detail::countFlops(operations, 81 * HopkinsSimd::size());
            for (int row = 0; row < 3; row++) {
                for (int column = 0; column < 3; column++) {
                    moments[row][column] = kernel * separation[row] * separation[column];
                    momentRates[row][column] = kernelRate * separation[row] * separation[column] +
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
        gradient.velocity[dim] = state.gradients[dim + 2];
    }
    return gradient;
}

hopkins::Primitive hopkinsPredictedPrimitive(
    HopkinsParticle const& particle, HopkinsWorkspace const& workspace, FlopCount* operations) {
    auto const& state = particle.state;
    auto const anchor =
        hopkins::primitiveFromConserved({state.mass, state.momentum, state.totalEnergy},
            state.anchorVolume, getOptions().hydro.gamma);
    hopkins::detail::countFlops(operations, 16 + (getOptions().hydro.damping == 0.0 ? 0 : 3));
    double const elapsed = workspace.time - state.sourceTime;
    if (elapsed < 0.0 ||
        workspace.time > state.endTime +
                32.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, state.endTime)) {
        throw std::runtime_error("Hopkins primitive prediction lies outside its own interval");
    }
    auto predicted = hopkins::predictPrimitive(anchor, hopkinsStoredGradient(state),
        elapsed / workspace.scale, getOptions().hydro.gamma, operations);
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

void hopkinsGradients(HopkinsWorkspace& workspace) {
    profile::ScopedPhase profilePhase("hydro.hopkins.gradients");
    HopkinsFlopBatch operations(workspace.countOperations);
    size_t const count = workspace.particles.size();
    workspace.primitives.resize(count);
    workspace.gradients.assign(count, {});
    double const gamma = getOptions().hydro.gamma;
    for (size_t i = 0; i < count; i++) {
        auto const& particle = workspace.particles[i];
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
    hopkinsParallel(count, [&](size_t i) {
        HopkinsFlopBatch operations(workspace.countOperations);
        auto& particle = workspace.particles[i];
        if (!particle.active && particle.state.predictorInitialized) {
            return;
        }
        auto const& center = workspace.primitives[i];
        std::array<double, 5> values{center.density, center.pressure, center.velocity[0],
            center.velocity[1], center.velocity[2]};
        auto minimum = values;
        auto maximum = values;
        std::array<hopkins::Vector, 5> gradients{};
        hopkinsNeighborBatches(workspace, i, operations, [&](HopkinsNeighborBatch const& batch) {
            auto const& geometry = workspace.geometry[i];
            auto const weightScale =
                hopkinsKernel(batch.radius, particle.support, workspace.dimensions, operations) *
                HopkinsSimd(geometry.volume);
            HopkinsSimdVector weight;
            hopkins::detail::countFlops(operations, 19 * HopkinsSimd::size());
            for (int dim = 0; dim < 3; dim++) {
                weight[dim] =
                    (HopkinsSimd(geometry.inverseMoment[dim][0]) * batch.separation[0] +
                        HopkinsSimd(geometry.inverseMoment[dim][1]) * batch.separation[1] +
                        HopkinsSimd(geometry.inverseMoment[dim][2]) * batch.separation[2]) *
                    weightScale;
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
                    minimum[field] = std::min(minimum[field], double(neighborValues[field][lane]));
                    maximum[field] = std::max(maximum[field], double(neighborValues[field][lane]));
                    for (int dim = 0; dim < 3; dim++) {
                        gradients[field][dim] += contributions[field][dim][lane];
                    }
                }
            }
        });
        auto reconstructedMinimum = values;
        auto reconstructedMaximum = values;
        hopkinsNeighborBatches(
            workspace, i, operations,
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
            gradient.velocity[dim] = gradients[dim + 2];
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
    int maximumRung = std::max(workspace.darkMatterRung, options.run.minRung);
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
    for (size_t i = 0; i < workspace.particles.size(); i++) {
        auto& particle = workspace.particles[i];
        double timestep = backgroundTimestep;
        hopkins::detail::countFlops(operations, workspace.dimensions == 1 ? 6 : 8);
        double const radius = workspace.dimensions == 1 ?
            0.5 * workspace.geometry[i].volume :
            std::cbrt(3.0 * workspace.geometry[i].volume / (4.0 * M_PI));
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
        } else if (required > particle.rung ||
            (particle.state.predictorInitialized &&
                !hopkins::physical(hopkinsPredictedPrimitive(particle, workspace, operations)))) {
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
    for (size_t i = 0; i < workspace.particles.size(); i++) {
        order.push_back(i);
    }
    std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
        return workspace.particles[left].state.identifier <
            workspace.particles[right].state.identifier;
    });
    // Appendix G: prevent a timestep contrast larger than four across an
    // interacting pair. Wake sleeping neighbors onto the shorter active bin.
    bool changed;
    do {
        changed = false;
        for (size_t i : order) {
            auto const rung = workspace.particles[i].rung;
            for (size_t j : workspace.neighbors[i]) {
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
    try {
        initial = solve(zero, active);
        midpoint = initial;
        if (anySecondOrder) {
            hopkins::detail::countFlops(operations, kWidth);
            midpoint = solve(HopkinsSimd(0.5) * interval, secondOrder);
        }
    } catch (std::exception const& error) {
        std::string message = std::string(error.what()) + " [Hopkins pair IDs by SIMD lane:";
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
    std::map<uint64_t, size_t> indices;
    for (size_t i = 0; i < workspace.particles.size(); ++i) {
        indices.emplace(workspace.particles[i].state.identifier, i);
    }
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
            state.anchorVolume = workspace.geometry[i].volume;
            state.sourceTime = workspace.time;
            state.endTime = workspace.time;
            state.beginTick = state.endTick = workspace.tick;
            state.sourceAcceleration = particle.acceleration;
            state.expansionCoefficient = coefficient;
        }
    }
}

void hopkinsCloseParticle(HopkinsWorkspace& workspace, size_t i) {
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
    auto const next =
        hopkins::closeConservedInterval({state.mass, state.momentum, state.totalEnergy},
            {state.pendingMass, state.pendingMomentum, state.pendingEnergy},
            state.sourceAcceleration, particle.acceleration, duration, getOptions().hydro.damping,
            state.expansionCoefficient, coefficient, work);
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
            particle.state.totalEnergy +=
                particle.state.mass * (double(output.energy) - double(inputs[k].energy));
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
            particle.chemistry = results[k].state;
        }
    }
    energies.heating += heating;
#else
    throw std::logic_error("Equilibrium cooling requires COSMICTIGER_ENABLE_CHEMISTRY=ON");
#endif
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

void hopkinsOpenPairs(HopkinsWorkspace& workspace, size_t& retained, size_t& opened) {
    profile::ScopedPhase profilePhase("hydro.hopkins.flux.open");
    std::vector<std::pair<size_t, size_t>> pairs;
    for (size_t i = 0; i < workspace.particles.size(); ++i) {
        for (size_t j : workspace.neighbors[i]) {
            auto const& left = workspace.particles[i];
            auto const& right = workspace.particles[j];
            if (left.state.identifier < right.state.identifier && (left.active || right.active) &&
                !left.convert && !right.convert &&
                !hopkinsPairs.contains({left.state.identifier, right.state.identifier})) {
                pairs.emplace_back(i, j);
            }
        }
    }
    std::sort(pairs.begin(), pairs.end(), [&](auto const& left, auto const& right) {
        auto const key = [&](auto const& pair) {
            return HopkinsPairKey{workspace.particles[pair.first].state.identifier,
                workspace.particles[pair.second].state.identifier};
        };
        return key(left) < key(right);
    });
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

void hopkinsSoundWaveDiagnostics(double time) {
    profile::ScopedPhase profilePhase("hydro.hopkins.diagnostics");
    SphTreeCreateParameters treeParameters;
    treeParameters.minimumRung = 0;
    treeParameters.supportRadiusMultiplier = 1.2;
    {
        profile::ScopedPhase profilePhase("hydro.hopkins.tree.create");
        sphTreeCreate(treeParameters);
    }
    detail::hopkinsNeighborRefresh(false);
    detail::HopkinsWorkspace workspace;
    workspace.countOperations = false;
    workspace.dimensions = scenarios::definition(getOptions().run.scenario).hydroDimensions;
    detail::hopkinsSolveSupport(workspace);
    detail::hopkinsGeometry(workspace);
    detail::hopkinsWriteSoundWave(workspace, time);
    {
        profile::ScopedPhase profilePhase("hydro.hopkins.tree.destroy");
        sphTreeDestroy();
    }
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
    SphTreeCreateParameters parameters;
    parameters.minimumRung = 0;
    parameters.supportRadiusMultiplier = 1.2;
    {
        profile::ScopedPhase profilePhase("hydro.hopkins.tree.create");
        sphTreeCreate(parameters);
    }
    detail::hopkinsNeighborRefresh(false);
    detail::hopkinsTreePrepared = true;
}

void hopkinsPrepareGeometry(double scale, double expansionRate) {
    if (!detail::hopkinsTreePrepared || detail::hopkinsPrepared) {
        throw std::runtime_error("Invalid Hopkins geometry preparation order");
    }
    auto workspace = std::make_unique<detail::HopkinsWorkspace>();
    workspace->dimensions = scenarios::definition(getOptions().run.scenario).hydroDimensions;
    workspace->scale = scale;
    workspace->expansionRate = expansionRate;
    workspace->readForces = false;
    detail::hopkinsSolveSupport(*workspace);
    detail::hopkinsGeometry(*workspace);
    detail::hopkinsPrepared = std::move(workspace);
}

HopkinsStepResult hopkinsKick(int minimumRung, double scale, double expansionRate, double time,
    double baseTimestep, EnergiesT& energies, bool closingOnly, double previousTimestep,
    double maximumGlobalTimestep) {
    profile::ScopedPhase profilePhase("hydro.hopkins");
    bool const global = maximumGlobalTimestep > 0.0;
    if (!(maximumGlobalTimestep >= 0.0) || !std::isfinite(maximumGlobalTimestep) ||
        (global && (!(previousTimestep >= 0.0) || !std::isfinite(previousTimestep)))) {
        throw std::invalid_argument("Hopkins global timesteps must be finite and nonnegative");
    }
    if (!detail::hopkinsTreePrepared) {
        hopkinsPrepareTree();
    }
    if (!detail::hopkinsPrepared) {
        hopkinsPrepareGeometry(scale, expansionRate);
    }
    auto workspaceOwner = std::move(detail::hopkinsPrepared);
    auto& workspace = *workspaceOwner;
    workspace.scale = scale;
    workspace.expansionRate = expansionRate;
    workspace.time = time;
    workspace.tickDuration = baseTimestep / double(detail::kHopkinsTicksPerBase);
    workspace.tick = global ? 0 : detail::hopkinsTick(time, baseTimestep, nullptr);
    // Gravity has joined: now it is safe to import fresh accelerations and the
    // dark-matter rung reduction. Support geometry still has identical ordering.
    workspace.readForces = true;
    detail::hopkinsGather(workspace);
    uint64_t nextId = 0;
    for (auto const& particle : workspace.particles) {
        if (particle.state.identifier != std::numeric_limits<uint64_t>::max()) {
            nextId = std::max(nextId, particle.state.identifier + 1);
        }
    }
    for (auto& particle : workspace.particles) {
        if (particle.state.identifier == std::numeric_limits<uint64_t>::max()) {
            particle.state.identifier = nextId++;
        }
    }
    detail::hopkinsInitialize(workspace);
    size_t woken = 0;
    for (auto& particle : workspace.particles) {
        particle.active = closingOnly || !particle.state.predictorInitialized ||
            (global ? particle.state.endTime <= time : particle.state.endTick <= workspace.tick);
        if (!particle.active &&
            !hopkins::physical(detail::hopkinsPredictedPrimitive(particle, workspace, nullptr))) {
            particle.active = true;
            particle.rung = std::max(particle.rung, minimumRung);
            ++woken;
        }
    }
    std::vector<unsigned char> closed(workspace.particles.size(), 0);
    HopkinsStepResult result;
    result.maximumRung = workspace.darkMatterRung;
    result.minimumTimestep = maximumGlobalTimestep;
    size_t settled = 0;
    size_t converted = 0;
    bool firstSelection = true;
    while (true) {
        settled += detail::hopkinsSettlePairs(workspace, closingOnly);
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
        detail::hopkinsGradients(workspace);
        detail::hopkinsGeometryRates(workspace);
        for (size_t i = 0; i < workspace.particles.size(); ++i) {
            auto& particle = workspace.particles[i];
            if (!particle.active) {
                continue;
            }
            auto& state = particle.state;
            state.meshVelocity = particle.velocity;
            for (int dim = 0; dim < 3; ++dim) {
                state.meshAcceleration[dim] = state.sourceAcceleration[dim] -
                    workspace.gradients[i].pressure[dim] / (particle.density * scale) -
                    getOptions().hydro.damping * particle.velocity[dim];
            }
        }
        if (time == 0.0 && getOptions().run.scenario == "soundWave") {
            detail::hopkinsWriteSoundWave(workspace, time);
        }
        detail::hopkinsOpenPairs(workspace, retained, opened);
    } else {
        result.maximumRung = std::max(result.maximumRung, maximumGasRung);
        if (!detail::hopkinsPairs.empty()) {
            throw std::runtime_error("Hopkins terminal closure left a live pair interval");
        }
    }
    detail::hopkinsScatter(workspace, 1);
    {
        profile::ScopedPhase energyPhase("hydro.hopkins.diagnostics.energies");
        auto const sum = particlesSumEnergies();
        energies.kin = sum.kin / (scale * scale);
        energies.therm = sum.therm / (scale * scale);
        energies.xmom = sum.xmom / scale;
        energies.ymom = sum.ymom / scale;
        energies.zmom = sum.zmom / scale;
        energies.nmom = sum.nmom / scale;
    }
    terminal::detailPrint("Hopkins step: time=%.17g active=%zu total=%zu minRung=%d maxRung=%d "
                          "pairs=%zu cached=%zu refreshed=%zu woken=%zu settled=%zu stars=%zu "
                          "terminal=%d\n",
        time, active, workspace.particles.size(), minimumGasRung, maximumGasRung, retained + opened,
        retained, opened, woken, settled, converted, int(closingOnly));
    {
        profile::ScopedPhase treePhase("hydro.hopkins.tree.destroy");
        sphTreeDestroy();
        detail::hopkinsTreePrepared = false;
    }
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

void hopkinsWriteCheckpoint(FILE* file) {
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
