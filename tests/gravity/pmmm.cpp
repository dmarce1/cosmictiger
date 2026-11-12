/* CosmicTiger - periodic PMMM force, distribution, and cache regression; BSL-1.0. */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <format>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>

#include <cosmictiger/core/build_config.hpp>
#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/fmm/cell_tree.hpp>
#include <cosmictiger/gravity/fmm/ghost_storage.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/kernels/spherical_ewald_kernels.hpp>
#include <cosmictiger/gravity/mesh_size.hpp>
#include <cosmictiger/gravity/solver.hpp>
#include <cosmictiger/gravity/sources.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/gravity/tree_multipole.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particle_mesh_sort.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/particles.hpp>
#endif
#include <src/gravity/state.hpp>

#include "pmmm_reference.hpp"

namespace detail {

size_t expectedPmmmParticleCount = 0;

int pmmmFixtureParticleCount(int fixture) {
    constexpr std::array<int, 9> kFixtureSizes{7, 1, 0, 4, 8, 320, 32, 7, 7};
    return kFixtureSizes.at(fixture);
}

struct PmmmSample {
    Vector3<double> position{};
    Vector3<PositionInteger> rawPosition{};
    Vector3<float> acceleration{};
    Vector3<float> velocity{};
    int rung{};
    float potential{};
    float mass{};
    int id{};
    int rank{};
    uint64_t cellKey{};

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & position & rawPosition & acceleration & potential & mass & id & rank & cellKey &
            velocity & rung;
    }
};

void requirePmmm(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error("PMMM regression: " + message);
    }
}

void checkSourceMomentSerialization() {
    TreeBuildMoments input;
    input.useDouble();
    int const count = fmmMultipoleCoefficientCount(kFmmOrder);
    for (int i = 0; i < count; ++i) {
        input.precise()[i] = .125 + i + std::ldexp(1., -36);
    }
    auto roundTrip = [&](TreeBuildMoments const& value) {
        std::vector<char> bytes;
        {
            hpx::serialization::output_archive output(bytes);
            output << value;
        }
        TreeBuildMoments copy;
        hpx::serialization::input_archive archive(bytes);
        archive >> copy;
        requirePmmm(
            copy.isDouble() == value.isDouble(), "build-return transport changed source precision");
        for (int i = 0; i < count; ++i) {
            requirePmmm(copy[i] == value[i], "build-return transport changed a coefficient");
        }
        return bytes.size();
    };
    auto const wide = roundTrip(input);
    input.discardDouble();
    auto const narrow = roundTrip(input);
    requirePmmm(wide == narrow + count * sizeof(float),
        "ghost/root-cache transport widened or serialized inactive source coefficients: float=" +
            std::to_string(narrow) + " double=" + std::to_string(wide) +
            " count=" + std::to_string(count));
    std::cout << "PMMM source transport active coefficients=" << count
              << " float/double bytes=" << narrow << "/" << wide
              << " build frame=" << sizeof(TreeCreateReturn)
              << " persistent moment=" << sizeof(MultipoleBuffer) << '\n';
}

void checkGhostMassPermutation() {
    std::array<Position, 4> x{0.875, 0.125, 0.625, 0.375};
    std::array<Position, 4> y{}, z{};
    std::array<char, 4> types{}, rungs{7, 1, 5, 3};
    std::array<float, 4> masses{0.875f, 0.125f, 0.625f, 0.375f};
    FmmSourceView view;
    view.positions[kXi] = x.data();
    view.positions[kYi] = y.data();
    view.positions[kZi] = z.data();
    view.types = types.data();
    view.rungs = rungs.data();
    view.masses = masses.data();
    view.count = 4;
    view.store = FmmParticleStore::Ghost;
    auto const middle = fmmSourcePartition(FmmParticleStore::Ghost, view, {0, 4}, 0.5, 0);
    requirePmmm(middle == 2, "ghost source split selected the wrong count");
    for (ParticleInteger i = 0; i < 4; ++i) {
        requirePmmm(fmmSourceMass(view, i) == x[i].toDouble() &&
                rungs[i] == int(8 * x[i].toDouble()) && (x[i].toDouble() < 0.5) == (i < middle),
            "ghost source permutation detached a variable mass or rung from its position");
    }
    // Exercise the production double leaf/M2M/root-conversion arithmetic with
    // explicit unequal masses, independently of the coupled-hydro solver gate.
    {
        constexpr int order = kFmmOrder;
        Vector3<Position> const center{.5, .5, .5};
        TreeBuildMoments left, right;
        left.useDouble();
        right.useDouble();
        Vector3<Position> const lc{.25, .375, .5}, rc{.75, .625, .5};
        treeComputeMultipole<order>(left.precise(), {0, 2}, lc, view);
        treeComputeMultipole<order>(right.precise(), {2, 4}, rc, view);
        SourceMultipoleBuffer root;
        treeComputeMultipole<order>(root, left, right, lc, rc, center);
        std::array<double, FmmOrder<order>::kMultipoleCoefficients_> mapped{}, direct{};
        treeSourceToMesh<order>(mapped.data(), root, center, center, 1);
        for (int i = 0; i < 4; ++i) {
            Vector3<double> offset;
            for (int axis = 0; axis < kDimensions; ++axis) {
                offset[axis] = unwrappedDoubleDistance(view.positions[axis][i], center[axis]);
            }
            pmmm::addMoments(order, direct.data(), masses[i], offset);
        }
        for (size_t i = 0; i < mapped.size(); ++i) {
            requirePmmm(std::abs(mapped[i] - direct[i]) < 2.e-13,
                "unequal-mass double root differs from direct spherical P2M");
        }
    };
}

void reservePmmmCudaFixture() {
    // Reserve the largest fixture on every rank before temporary CPU execution.
    // The application's retained particle/tree arenas assume a fixed backend.
    particlesResize(320);
    fmmGhostReserve(320);
    fmmGhostClear();
    particlesResize(0);
}

void resetPmmmSavedForces() {
    // Domain migration transfers physical particle state, not saved diagnostic
    // forces. Initialize these sentinels after migration, before testing solves.
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        int const id = int(particlesVel(kXi, i)) - 1;
        particlesPot(i) = -0.25f * float(id + 1);
        for (int dim = 0; dim < kDimensions; ++dim) {
            particlesGravityAcceleration(dim, i) = float(4 * (id + 1) + dim);
        }
    }
}

void initializePmmmFixture(int fixture, int ownership) {
    // The first two and the next pair share cells on the n=8 mesh. Their
    // different offsets generate dipoles and the independent radial moment.
    std::array<Vector3<double>, 7> const points{
        {{0.069, 0.161, 0.326}, {0.104, 0.202, 0.295}, {0.454, 0.166, 0.329}, {0.487, 0.206, 0.287},
            {0.989, 0.171, 0.317}, {0.671, 0.839, 0.582}, {0.011, 0.986, 0.971}}};
    int const total = pmmmFixtureParticleCount(fixture);
    auto owner = [ownership](int id) { return ownership == 0 ? id % hpxSize() : 0; };
    int count = 0;
    for (int id = 0; id < total; ++id) {
        if (owner(id) == hpxRank()) {
            ++count;
        }
    }
    particlesResize(count);
    int local = 0;
    for (int id = total - 1; id >= 0; --id) {
        if (owner(id) != hpxRank()) {
            continue;
        }
        for (int dim = 0; dim < kDimensions; ++dim) {
            double coordinate = 0.0;
            if (fixture == 8) {
                // Centered, distinct sources on the side-16 stage mesh eliminate
                // multipole truncation from the stage test while retaining near
                // and far pairs at width 2; every nonempty stage of this
                // seven-particle fixture selects the full side 16.
                constexpr std::array<std::array<int, kDimensions>, 7> kStageCells{{{0, 2, 4},
                    {1, 2, 4}, {6, 2, 4}, {6, 3, 4}, {15, 2, 4}, {10, 12, 8}, {0, 12, 12}}};
                coordinate = (kStageCells[id][dim] + 0.5) / 16.0;
            } else if (fixture == 7) {
                // Every particle shares one mesh cell, which coarse-aligned
                // domains must keep on a single rank. Distinct positions avoid
                // softened pairs in the independent Ewald sum.
                int const multiplier = dim == kXi ? 1 : dim == kYi ? 3 : 5;
                double const offset = 0.2 + 0.1 * ((id * multiplier) % 7);
                coordinate = (1.0 + offset) / getOptions().gravity.meshSize;
            } else if (fixture == 6) {
                int const cell = dim == kXi ? id % 8 :
                    dim == kYi              ? 4 * ((id / 8) % 2) :
                                              4 * (id / 16);
                coordinate = (cell + 0.5) / 8.0;
            } else if (fixture == 5) {
                int const side = getOptions().gravity.meshSize;
                int const group = id / 64;
                int const cell = dim == kXi ? (group == 1 ? side - 1 : group == 3 ? 1 : 0) : 1;
                std::array<Vector3<double>, 5> const centers{{{-0.22, -0.20, -0.18},
                    {0.14, 0.03, -0.06}, {0.22, 0.20, 0.18}, {0.32, 0.05, 0.09}, {0.0, 0.0, 0.0}}};
                int const multiplier = dim == kXi ? 17 : dim == kYi ? 29 : 43;
                double const jitter = (double((id * multiplier) % 64) + 0.5) / 64.0 - 0.5;
                coordinate = (cell + 0.5 + centers[group][dim] + 0.11 * jitter) / side;
            } else {
                coordinate = fixture == 4 ? 0.0625 + 0.5 * ((id >> dim) & 1) : points[id][dim];
            }
            if (fixture == 3 && id >= 2) {
                coordinate = (dim == kXi        ? 0.58 :
                                     dim == kYi ? 0.57 :
                                                  0.81) +
                    (id == 2 ? 0.013 : -0.019);
            }
            particlesPos(dim, local) = coordinate;
            if (fixture == 5 && id >= total - 16) {
                // Distinct coincident sources at the last representable point
                // outnumber the eight-particle leaf bucket, exercising robust
                // subdivision and a center conversion that rounds up to 1.
                particlesPos(dim, local).setInteger(std::numeric_limits<PositionInteger>::max());
            }
            particlesVel(dim, local) = dim == kXi ? float(id + 1) :
                dim == kYi                        ? 0.25f * float(id) :
                                                    -0.5f * float(id);
            particlesGravityAcceleration(dim, local) = float(4 * (id + 1) + dim);
        }
        particlesPot(local) = -0.25f * float(id + 1);
        particlesRung(local) = 1 + id % 4;
        ++local;
    }
}

Vector<PmmmSample> readPmmmFixture() {
    requirePmmm(particlesSize() == particlesOwnedSize(),
        "force solve retained ghost particles in the owned population");
    Vector<PmmmSample> samples(particlesSize());
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        auto& sample = samples[i];
        for (int dim = 0; dim < kDimensions; ++dim) {
            sample.position[dim] = particlesPos(dim, i).toDouble();
            sample.rawPosition[dim] = particlesPos(dim, i).raw();
        }
        sample.acceleration = pmmmAcceleration(i);
        sample.potential = pmmmPotential(i);
        sample.mass = particlesMass(i);
        sample.id = int(particlesVel(kXi, i)) - 1;
        requirePmmm(particlesVel(kXi, i) == float(sample.id + 1) &&
                particlesVel(kYi, i) == 0.25f * float(sample.id) &&
                particlesVel(kZi, i) == -0.5f * float(sample.id) &&
                particlesRung(i) == 1 + sample.id % 4,
            "force solve changed velocity or rung");
        if (saveParticlePotential(getOptions().analysis)) {
            requirePmmm(particlesPot(i) == -0.25f * float(sample.id + 1),
                "force solve changed saved potential");
        }
        sample.rank = hpxRank();
        sample.cellKey = 0;
        for (int dim = 0; dim < kDimensions; ++dim) {
            if (saveParticleAcceleration(getOptions().analysis)) {
                requirePmmm(
                    particlesGravityAcceleration(dim, i) == float(4 * (sample.id + 1) + dim),
                    "force solve changed saved acceleration");
            }
            sample.cellKey = sample.cellKey * getOptions().gravity.meshSize +
                particleMeshCoordinate(particlesPos(dim, i).raw(), getOptions().gravity.meshSize);
        }
    }
    return samples;
}

/// Move particles outside their frozen cells, including both directions through the seam.
void moveFrozenPmmmFixture() {
    for (ParticleInteger i = 0; i < particlesOwnedSize(); ++i) {
        int const id = int(particlesVel(kXi, i)) - 1;
        double const shift = id / 64 == 1 ? .09 : -.09;
        particlesPos(kXi, i) = particlesPos(kXi, i).toDouble() + shift;
        particlesPos(kYi, i) = particlesPos(kYi, i).toDouble() + .015;
    }
}

Vector<PmmmSample> readPmmmKickFixture() {
    Vector<PmmmSample> samples(particlesSize());
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        auto& sample = samples[i];
        for (int dim = 0; dim < kDimensions; ++dim) {
            sample.position[dim] = particlesPos(dim, i).toDouble();
            sample.rawPosition[dim] = particlesPos(dim, i).raw();
            sample.velocity[dim] = particlesVel(dim, i);
            if (saveParticleAcceleration(getOptions().analysis)) {
                sample.acceleration[dim] = particlesGravityAcceleration(dim, i);
            }
        }
        sample.mass = particlesMass(i);
        sample.rung = particlesRung(i);
        if (saveParticlePotential(getOptions().analysis)) {
            sample.potential = particlesPot(i);
        }
    }
    return samples;
}

struct PmmmOverlapGate {
    std::mutex mutex;
    hpx::promise<void> promise;
    hpx::shared_future<void> release = promise.get_future().share();
    hpx::promise<void> requestPromise;
    hpx::shared_future<void> requestStarted = requestPromise.get_future().share();
    uint64_t heldKey = 0;
    bool chosen = false;
    bool completedIndependent = false;
    bool completedDependentEarly = false;
    bool receivedHeld = false;
};

std::shared_ptr<PmmmOverlapGate> pmmmOverlapGate;

bool pmmmCellsNeighbor(uint64_t first, uint64_t second) {
    int const side = getOptions().gravity.meshSize;
    double distanceSquared = 0;
    for (int dim = 0; dim < kDimensions; ++dim) {
        int const difference = std::abs(int(first % side) - int(second % side));
        double const separation = std::min(difference, side - difference);
        distanceSquared += separation * separation;
        first /= side;
        second /= side;
    }
    double const theta = getOptions().gravity.openingAngle;
    return distanceSquared * theta * theta <= 3.0;
}

void installPmmmOverlapGate(bool enable, uint64_t heldKey) {
    pmmmSetCellObserver({});
    pmmmOverlapGate.reset();
    if (!enable || hpxRank() != 0) {
        return;
    }
    auto gate = std::make_shared<PmmmOverlapGate>();
    gate->heldKey = heldKey;
    pmmmOverlapGate = gate;
    pmmmSetCellObserver([gate](PmmmCellEvent event, uint64_t key) {
        // Ensure a very fast interior cell cannot release the gate before the
        // chosen remote request exists. HPX suspension leaves workers free.
        if (event == PmmmCellEvent::NearCompleted) {
            gate->requestStarted.get();
        }
        bool wait = false;
        bool release = false;
        {
            std::lock_guard<std::mutex> lock(gate->mutex);
            if (event == PmmmCellEvent::GhostRequested && key == gate->heldKey && !gate->chosen) {
                gate->chosen = true;
                wait = true;
            } else if (event == PmmmCellEvent::NearCompleted && !gate->completedIndependent) {
                if (pmmmCellsNeighbor(key, gate->heldKey)) {
                    gate->completedDependentEarly = true;
                } else {
                    gate->completedIndependent = true;
                    release = true;
                }
            } else if (event == PmmmCellEvent::GhostReady && key == gate->heldKey) {
                gate->receivedHeld = true;
            }
        }
        if (release) {
            gate->promise.set_value();
        }
        if (wait) {
            gate->requestPromise.set_value();
            gate->release.get();
        }
    });
}

Vector<int> readPmmmOverlapGate() {
    auto const gate = pmmmOverlapGate;
    if (!gate) {
        return {};
    }
    std::lock_guard<std::mutex> lock(gate->mutex);
    return {int(gate->chosen), int(gate->completedIndependent), int(gate->completedDependentEarly),
        int(gate->receivedHeld)};
}

size_t verifyPmmmTreeCell(TreeId id, uint64_t key, int meshSize) {
    auto const* node = treeGetNode(id);
    uint64_t positionKey = 0;
    uint64_t multipoleKey = 0;
    for (int dim = 0; dim < kDimensions; ++dim) {
        positionKey =
            positionKey * meshSize + particleMeshCoordinate(node->pos[dim].raw(), meshSize);
        multipoleKey =
            multipoleKey * meshSize + particleMeshCoordinate(node->mpos->pos[dim].raw(), meshSize);
    }
    requirePmmm(positionKey == key && multipoleKey == key,
        "forest node or multipole center escaped its exact mesh cell: expected=" +
            std::to_string(key) + " position=" + std::to_string(positionKey) +
            " multipole=" + std::to_string(multipoleKey) + " node=" + std::to_string(id.proc) +
            ":" + std::to_string(id.index));
    if (node->leaf) {
        // Every local leaf is checked by its owning locality before the roots
        // are combined. Compare exact integers so a Q0.64 coordinate just below
        // one cannot round onto an incorrectly inward bound during this test.
        if (node->isLocalHere()) {
            using WideInteger = __int128;
            WideInteger constexpr kBoundaryScale = WideInteger(1) << (kPositionBits - 29);
            for (ParticleInteger i = node->particleRange.first; i < node->particleRange.second;
                ++i) {
                long double radiusSquared = 0.0;
                for (int dim = 0; dim < kDimensions; ++dim) {
                    WideInteger const raw = particlesPos(dim, i).raw();
                    WideInteger const lower =
                        WideInteger(node->box.begin[dim].raw()) * kBoundaryScale;
                    WideInteger const upper =
                        WideInteger(node->box.end[dim].raw()) * kBoundaryScale;
                    requirePmmm(lower <= raw && raw <= upper,
                        "forest leaf box excludes a represented fixed-point particle");
                    long double const distance = std::ldexp(
                        static_cast<long double>(raw - WideInteger(node->pos[dim].raw())),
                        -kPositionBits);
                    radiusSquared += distance * distance;
                }
                requirePmmm(radiusSquared <= static_cast<long double>(node->radius) * node->radius,
                    "forest leaf radius excludes a represented fixed-point particle");
            }
        }
        return 0;
    }
    auto const children = node->children;
    size_t internalNodes = 1;
    for (auto const& child : children) {
        internalNodes += verifyPmmmTreeCell(child, key, meshSize);
    }
    return internalNodes;
}

struct SourceAudit {
    std::mutex mutex;
    std::map<uint64_t, int> roots, deposits, locals;
    bool failOnce = false;
    bool failFar = false;
};
std::shared_ptr<SourceAudit> sourceAudit;
void installSourceAudit(bool enable, int failingRank) {
    pmmmSetCellObserver({});
    sourceAudit.reset();
    if (!enable) {
        return;
    }
    auto audit = std::make_shared<SourceAudit>();
    audit->failFar = failingRank < -1;
    audit->failOnce = hpxRank() == (audit->failFar ? -2 - failingRank : failingRank);
    sourceAudit = audit;
    pmmmSetCellObserver([audit](PmmmCellEvent event, uint64_t key) {
        std::lock_guard<std::mutex> lock(audit->mutex);
        if (event == PmmmCellEvent::OwnedRootReady) {
            ++audit->roots[key];
            if (audit->failOnce && !audit->failFar) {
                audit->failOnce = false;
                throw std::runtime_error("injected owned-root publication failure");
            }
        }
        if (event == PmmmCellEvent::MomentsDeposited) {
            ++audit->deposits[key];
        }
        if (event == PmmmCellEvent::FarLocalReady) {
            ++audit->locals[key];
            if (audit->failOnce && audit->failFar) {
                audit->failOnce = false;
                throw std::runtime_error("injected Cartesian far-local failure");
            }
        }
    });
}
Vector<uint64_t> readSourceAudit() {
    auto const audit = sourceAudit;
    std::lock_guard<std::mutex> lock(audit->mutex);
    uint64_t duplicates = 0;
    for (auto const& [key, count] : audit->roots) {
        duplicates += count != 1;
    }
    for (auto const& [key, count] : audit->deposits) {
        duplicates += count != 1;
    }
    for (auto const& [key, count] : audit->locals) {
        duplicates += count != 1;
    }
    return {audit->roots.size(), audit->deposits.size(), duplicates, audit->locals.size()};
}

struct PrecisionMoments {
    TreeBuildMoments legacy;
    TreeBuildMoments mixed;
    TreeBuildMoments widened;
    TreeBuildMoments full;
};

template <int order>
PrecisionMoments sourcePrecision(TreeId id) {
    auto const* node = treeGetNode(id);
    PrecisionMoments result;
    result.mixed.useDouble();
    result.widened.useDouble();
    result.full.useDouble();
    Vector3<double> center;
    for (int axis = 0; axis < kDimensions; ++axis) {
        center[axis] = node->pos[axis].toDouble();
    }
    if (node->leaf) {
        auto const sources = treeSourceView(*node);
        treeComputeMultipole<order>(result.legacy.single(), node->particleRange, center, sources);
        treeComputeMixedMultipole<order>(
            result.mixed.precise(), node->particleRange, center, sources);
        treeComputeWideReductionMultipole<order>(
            result.widened.precise(), node->particleRange, center, sources);
        treeComputeMultipole<order>(result.full.precise(), node->particleRange, node->pos, sources);
    } else {
        auto const left = sourcePrecision<order>(node->children[0]);
        auto const right = sourcePrecision<order>(node->children[1]);
        auto const* l = treeGetNode(node->children[0]);
        auto const* r = treeGetNode(node->children[1]);
        Vector3<double> lc, rc;
        for (int axis = 0; axis < kDimensions; ++axis) {
            lc[axis] = l->pos[axis].toDouble();
            rc[axis] = r->pos[axis].toDouble();
        }
        treeComputeMultipole<order>(
            result.legacy.single(), left.legacy.single(), right.legacy.single(), lc, rc, center);
        treeComputeMultipole<order>(
            result.mixed.precise(), left.mixed, right.mixed, l->pos, r->pos, node->pos);
        treeComputeMultipole<order>(
            result.widened.precise(), left.widened, right.widened, l->pos, r->pos, node->pos);
        treeComputeMultipole<order>(
            result.full.precise(), left.full, right.full, l->pos, r->pos, node->pos);
    }
    MultipoleBuffer stored;
    treeReadMultipole(node->mpos, stored);
    for (int i = 0; i < FmmOrder<order>::kMultipoleCoefficients_; ++i) {
        requirePmmm(float(result.full[i]) == stored[i],
            "owned tree arena differs from narrowed full-double upward result");
    }
    return result;
}

void checkSourcePrecision(TreeCellRoot const& root) {
    [&]() {
        constexpr int order = kFmmOrder;
        int const side = getOptions().gravity.meshSize;
        auto const* node = treeGetNode(root.id);
        auto const variants = sourcePrecision<order>(root.id);
        Vector3<Position> center;
        uint64_t key = root.key;
        for (int axis = kDimensions - 1; axis >= 0; --axis) {
            center[axis] = (double(key % side) + .5) / side;
            key /= side;
        }
        std::array<double, FmmOrder<order>::kMultipoleCoefficients_> direct{};
        auto const sources = treeSourceView(*node);
        for (ParticleInteger i = node->particleRange.first; i < node->particleRange.second; ++i) {
            Vector3<double> offset;
            for (int axis = 0; axis < kDimensions; ++axis) {
                offset[axis] = (positionDomainCoordinate(sources.positions[axis][i]) * side -
                    center[axis].toDouble() * side);
            }
            pmmm::addMoments(order, direct.data(), fmmSourceMass(sources, i), offset);
        }
        std::array<double, FmmOrder<order>::kLocalCoefficients_> green{};
        pmmm::green(order, green.data(), {.43, -.37, .29}, 1. / side, false);
        auto const terms = pmmm::contractionTerms(order);
        auto field = [&](auto const& moments) {
            std::array<double, FmmOrder<order>::kLocalCoefficients_> local{};
            for (auto const& term : terms) {
                local[term.local] += term.weight * moments[term.source] * green[term.green];
            }
            std::array<double, 4> result{};
            pmmm::evaluate(order, result, local.data(), 0, {.17, -.11, .23}, 1. / side);
            return result;
        };
        auto const expectedField = field(direct);
        std::array<double, 4> coefficientError{}, fieldError{};
        int variantIndex = 0;
        for (auto const* variant :
            {&variants.legacy, &variants.mixed, &variants.widened, &variants.full}) {
            SourceMultipoleBuffer input;
            for (int i = 0; i < int(direct.size()); ++i) {
                input[i] = (*variant)[i];
            }
            std::array<double, FmmOrder<order>::kMultipoleCoefficients_> mapped{};
            treeSourceToMesh<order>(mapped.data(), input, node->pos, center, side);
            double rankScale = direct[0];
            for (int n = 0; n <= order - 2; ++n) {
                if (n) {
                    rankScale *= .5 / n;
                }
                for (int m = -n; m <= n; ++m) {
                    int const slot = n * (n + 1) + m;
                    coefficientError[variantIndex] = std::max(coefficientError[variantIndex],
                        std::abs(mapped[slot] - direct[slot]) / std::max(rankScale, 1.e-100));
                }
            }
            auto const actualField = field(mapped);
            for (int i = 0; i < 4; ++i) {
                fieldError[variantIndex] = std::max(fieldError[variantIndex],
                    std::abs(actualField[i] - expectedField[i]) /
                        std::max(1., std::abs(expectedField[i])));
            }
            ++variantIndex;
        }
        requirePmmm(coefficientError[3] < 2.e-11 && fieldError[3] < 2.e-11,
            "full-double root deposition exceeds direct double error envelope");
        std::cout << "PMMM source precision rank=" << hpxRank() << " mesh=" << side
                  << " cell=" << root.key
                  << " coefficient legacy/mixed/widened/full=" << coefficientError[0] << "/"
                  << coefficientError[1] << "/" << coefficientError[2] << "/" << coefficientError[3]
                  << " field=" << fieldError[0] << "/" << fieldError[1] << "/" << fieldError[2]
                  << "/" << fieldError[3] << '\n';
    }();
}

// Inspect the actual slab moments after a solve, independently accumulating the
// pre-refactor direct double P2M from every global source record.
void verifyDepositedMoments(Vector<PmmmSample> const& samples) {
    auto const& mesh = pmmm::detail::cache;
    auto const& state = pmmm::detail::state;
    if (pmmm::detail::frozenNearRequest(state.request) || state.request.applyKick) {
        return;
    }
    std::vector<double> expected(state.moments.size());
    double radial = 0;
    for (auto const& sample : samples) {
        uint64_t const x = sample.cellKey / (uint64_t(mesh.meshSize) * mesh.meshSize);
        if (x < uint64_t(mesh.localBegin) || x >= uint64_t(mesh.localBegin + mesh.localSize)) {
            continue;
        }
        auto const cell = pmmm::detail::coordinates(sample.cellKey);
        Vector3<double> offset;
        for (int axis = 0; axis < kDimensions; ++axis) {
            Position position;
            position.raw() = sample.rawPosition[axis];
            offset[axis] = positionDomainCoordinate(position) * mesh.meshSize - (cell[axis] + .5);
            radial += sample.mass * offset[axis] * offset[axis];
        }
        pmmm::addMoments(kFmmOrder,
            expected.data() + mesh.cellIndex(sample.cellKey) * mesh.sourceCount, sample.mass,
            offset);
    }
    for (size_t cell = 0; cell < mesh.cellCount(); ++cell) {
        double const mass = expected[cell * mesh.sourceCount];
        double scale = mass;
        for (int n = 0; n <= kFmmOrder - 2; ++n) {
            if (n) {
                scale *= .5 / n;
            }
            for (int m = -n; m <= n; ++m) {
                size_t const i = cell * mesh.sourceCount + n * (n + 1) + m;
                requirePmmm(
                    std::abs(state.moments[i] - expected[i]) <= 2.e-11 * std::max(scale, 1.e-100),
                    "deposited root coefficient differs from direct double mesh P2M");
            }
        }
    }
    requirePmmm(std::abs(state.radialTrace - radial) <= 2.e-12 * std::max(1., radial),
        "root radial summary differs from direct cell scan");
}

Vector<TreeCellRoot> buildPmmmTestForest() {
    int const meshSize = getOptions().gravity.meshSize;
    auto const cells = particleMeshSort(meshSize);
    auto roots = treeCreateCellForest(cells, meshSize);
    for (auto const& root : roots) {
        verifyPmmmTreeCell(root.id, root.key, meshSize);
        checkSourcePrecision(root);
    }
    return roots;
}

// Independent direct Ewald summation, with no generated kernels, mesh, FFT,
// harmonic coefficients, or production image tables. The primary self term is
// excluded; its finite periodic images remain, matching the FMM convention.
std::array<double, 4> directPmmmReference(size_t sink, Vector<PmmmSample> const& samples) {
    double constexpr kPi = std::numbers::pi;
    double constexpr kAlpha = 2.0;
    int constexpr kCutoff = 4;
    double const inverseSqrtPi = 1.0 / std::sqrt(kPi);
    std::array<double, 4> result{};
    for (size_t source = 0; source < samples.size(); ++source) {
        double const mass = samples[source].mass * getOptions().gravity.gm;
        Vector3<double> separation{};
        for (int dim = 0; dim < kDimensions; ++dim) {
            separation[dim] = samples[sink].position[dim] - samples[source].position[dim];
            separation[dim] -= std::nearbyint(separation[dim]);
        }
        result[0] += mass * kPi / (kAlpha * kAlpha);
        if (source == sink) {
            result[0] += mass * 2.0 * kAlpha * inverseSqrtPi;
        }
        for (int x = -kCutoff; x <= kCutoff; ++x) {
            for (int y = -kCutoff; y <= kCutoff; ++y) {
                for (int z = -kCutoff; z <= kCutoff; ++z) {
                    bool const origin = x == 0 && y == 0 && z == 0;
                    Vector3<double> const displacement{
                        separation[0] + x, separation[1] + y, separation[2] + z};
                    double const radiusSquared = displacement[0] * displacement[0] +
                        displacement[1] * displacement[1] + displacement[2] * displacement[2];
                    if (!(source == sink && origin)) {
                        double const radius = std::sqrt(radiusSquared);
                        requirePmmm(!origin || radius > getOptions().gravity.smoothingLength,
                            "direct Ewald fixture contains a softened pair");
                        double const screened = std::erfc(kAlpha * radius) / radius;
                        double const radial = (screened +
                                                  2.0 * kAlpha * inverseSqrtPi *
                                                      std::exp(-kAlpha * kAlpha * radiusSquared)) /
                            radiusSquared;
                        result[0] -= mass * screened;
                        for (int dim = 0; dim < kDimensions; ++dim) {
                            result[dim + 1] -= mass * radial * displacement[dim];
                        }
                    }
                    if (!origin) {
                        Vector3<double> const wave{2.0 * kPi * x, 2.0 * kPi * y, 2.0 * kPi * z};
                        double const waveSquared = x * x + y * y + z * z;
                        double const coefficient =
                            std::exp(-kPi * kPi * waveSquared / (kAlpha * kAlpha)) /
                            (kPi * waveSquared);
                        double const phase = wave[0] * separation[0] + wave[1] * separation[1] +
                            wave[2] * separation[2];
                        result[0] -= mass * coefficient * std::cos(phase);
                        for (int dim = 0; dim < kDimensions; ++dim) {
                            result[dim + 1] -= mass * coefficient * std::sin(phase) * wave[dim];
                        }
                    }
                }
            }
        }
    }
    return result;
}

template <size_t count>
std::complex<double> sphericalCoefficient(
    std::array<double, count> const& coefficients, int degree, int order) {
    if (std::abs(order) > degree) {
        return {};
    }
    int const magnitude = std::abs(order);
    std::complex<double> const positive{coefficients[degree * (degree + 1) + magnitude],
        magnitude ? coefficients[degree * (degree + 1) - magnitude] : 0.0};
    return order >= 0 ? positive : (magnitude % 2 ? -1.0 : 1.0) * std::conj(positive);
}

// Contract packed real harmonics through their complex storage convention,
// independently of the generated kernel and the Fourier contraction table.
template <int degree>
void addPeriodicExpansionReference(std::array<double, (degree + 1) * (degree + 1)>& local,
    double& localTrace, std::array<double, degree * degree> const& moments, double sourceTrace,
    std::array<double, (degree + 1) * (degree + 1)> const& green, double greenTrace) {
    for (int n = 0; n <= degree; ++n) {
        for (int m = 0; m <= n; ++m) {
            std::complex<double> value{};
            for (int k = 0; k <= std::min(degree - n, degree - 1); ++k) {
                for (int l = -k; l <= k; ++l) {
                    value += std::conj(sphericalCoefficient(moments, k, l)) *
                        sphericalCoefficient(green, n + k, m + l);
                }
            }
            local[n * (n + 1) + m] += value.real();
            if (m) {
                local[n * (n + 1) - m] += value.imag();
            }
        }
    }
    if constexpr (degree >= 3) {
        local[0] -= 0.5 * greenTrace * sourceTrace;
    }
    local[3] -= 2.0 * greenTrace * moments[3];
    local[1] -= 2.0 * greenTrace * moments[1];
    local[2] -= greenTrace * moments[2];
    localTrace += greenTrace * moments[0];
}

// Evaluate the selected finite expansion directly in double precision. Physical
// offsets and the independent harmonic contraction bypass mesh deposition, the
// Fourier contraction table, FFTs, and cell-width rescaling. This checks
// implementation error separately from truncation; directPmmmReference remains
// the independent full Ewald oracle.
template <int order>
std::array<double, 4> directPmmmExpansionReferenceOrder(
    size_t sink, Vector<PmmmSample> const& samples) {
    namespace spherical = cosmictiger::gravity::spherical;
    constexpr int kSourceDegree = order - 2;
    constexpr int kLocalDegree = order - 1;
    auto const& options = getOptions();
    int const n = options.gravity.meshSize;
    auto const coordinates = [n](PmmmSample const& sample) {
        return std::array<int, kDimensions>{
            int(sample.cellKey / (n * n)), int(sample.cellKey / n % n), int(sample.cellKey % n)};
    };
    auto const sinkCell = coordinates(samples[sink]);
    Vector3<double> sinkOffset{};
    for (int dim = 0; dim < kDimensions; ++dim) {
        sinkOffset[dim] = samples[sink].position[dim] - (sinkCell[dim] + 0.5) / n;
    }
    std::array<double, (kLocalDegree + 1) * (kLocalDegree + 1)> local{};
    std::array<double, 4> field{};
    double localTrace = 0.0;
    for (size_t source = 0; source < samples.size(); ++source) {
        auto const sourceCell = coordinates(samples[source]);
        Vector3<double> sourceOffset{}, displacement{}, pair{};
        double cellDistanceSquared = 0;
        double sourceTrace = 0.0;
        for (int dim = 0; dim < kDimensions; ++dim) {
            sourceOffset[dim] = samples[source].position[dim] - (sourceCell[dim] + 0.5) / n;
            int delta = (sinkCell[dim] - sourceCell[dim] + n) % n;
            if (delta > n / 2) {
                delta -= n;
            }
            cellDistanceSquared += double(delta) * delta;
            displacement[dim] = double(delta) / n;
            pair[dim] = displacement[dim] + sinkOffset[dim] - sourceOffset[dim];
            sourceTrace += samples[source].mass * sourceOffset[dim] * sourceOffset[dim];
        }
        bool const near =
            cellDistanceSquared * (options.gravity.openingAngle * options.gravity.openingAngle) <=
            3.0;
        std::array<double, (kSourceDegree + 1) * (kSourceDegree + 1)> moments{};
        spherical::detail::P2M(double(samples[source].mass), sourceOffset[0], sourceOffset[1],
            sourceOffset[2], moments);
        std::array<double, (kLocalDegree + 1) * (kLocalDegree + 1)> green{};
        double greenTrace = 0.0;
        spherical::evaluateEwaldGreen(
            greenTrace, displacement[0], displacement[1], displacement[2], green);
        if (!near) {
            std::array<double, (kLocalDegree + 1) * (kLocalDegree + 1)> primary{};
            spherical::detail::evaluateGreen(
                displacement[0], displacement[1], displacement[2], primary);
            for (size_t slot = 0; slot < green.size(); ++slot) {
                green[slot] += primary[slot];
            }
        }
        addPeriodicExpansionReference<kLocalDegree>(
            local, localTrace, moments, sourceTrace, green, greenTrace);
        if (near && source != sink) {
            double const radiusSquared = pair[0] * pair[0] + pair[1] * pair[1] + pair[2] * pair[2];
            double const radius = std::sqrt(radiusSquared);
            requirePmmm(radius > options.gravity.smoothingLength,
                "direct expansion fixture contains a softened pair");
            field[0] -= samples[source].mass / radius;
            for (int dim = 0; dim < kDimensions; ++dim) {
                field[dim + 1] -= samples[source].mass * pair[dim] / (radius * radiusSquared);
            }
        }
    }
    pmmm::detail::addSphericalLocalReference<kLocalDegree>(
        field, local, localTrace, sinkOffset[0], sinkOffset[1], sinkOffset[2]);
    for (double& component : field) {
        component *= options.gravity.gm;
    }
    return field;
}

std::array<double, 4> directPmmmExpansionReference(size_t sink, Vector<PmmmSample> const& samples) {
    return directPmmmExpansionReferenceOrder<kFmmOrder>(sink, samples);
}

double verifyPmmmReference(Vector<PmmmSample> const& samples) {
    auto const& options = getOptions();
    std::string const context = " mesh=" + std::to_string(options.gravity.meshSize) +
        " order=" + std::to_string(gravity::kBuildFmmOrder) +
        " theta=" + std::to_string(options.gravity.openingAngle);
    // Potential/force truncation budgets for these off-center fixtures, measured
    // against the independent Ewald sum with a direct double expansion for both
    // unit and mixed masses. They are not universal PMMM accuracy guarantees.
    // In particular order 6 reaches 1.615e-3 force error, even without FFTs;
    // it cannot share the order-7 5e-4 budget. Numerical error is bounded below
    // independently and never absorbed into the truncation allowance.
    constexpr std::array<std::array<double, 2>, 7> kTruncationBudgets{
        {{3.0e-2, 6.0e-2}, {1.5e-3, 2.5e-2}, {3.5e-4, 7.5e-3}, {6.0e-5, 2.0e-3}, {3.0e-5, 5.0e-4},
            {8.0e-6, 1.25e-4}, {3.0e-6, 5.0e-5}}};
    auto const budget = kTruncationBudgets[kFmmOrder - 3];
    constexpr double kNumericalTolerance = 2.0e-5;
    double maximumError = 0.0;
    Vector3<double> momentum{};
    double forceScale = 1.0;
    for (size_t sink = 0; sink < samples.size(); ++sink) {
        auto const expected = directPmmmReference(sink, samples);
        auto const expansion = directPmmmExpansionReference(sink, samples);
        double norm = 0.0;
        for (int dim = 0; dim < kDimensions; ++dim) {
            norm += expected[dim + 1] * expected[dim + 1];
        }
        norm = std::max(1.0, std::sqrt(norm));
        forceScale = std::max(forceScale, norm);
        double const potentialScale = std::max(1.0, std::abs(expected[0]));
        // Near FMM subtracts softened self potential in float; keep that
        // cancellation allowance separate from both expansion and FFT errors.
        double const selfCancellationAllowance = 4.0 * std::numeric_limits<float>::epsilon() *
            std::abs(options.gravity.gm * samples[sink].mass *
                selectedGravityKernel().centralPotential / options.gravity.smoothingLength);
        for (int component = 0; component < 4; ++component) {
            double const scale = component == 0 ? potentialScale : norm;
            double const actual = component == 0 ? samples[sink].potential :
                                                   samples[sink].acceleration[component - 1];
            double const numericalAllowance =
                kNumericalTolerance * scale + (component == 0 ? selfCancellationAllowance : 0.0);
            requirePmmm(std::abs(actual - expansion[component]) < numericalAllowance,
                "field differs from direct finite expansion for particle " + std::to_string(sink) +
                    " component=" + std::to_string(component) +
                    " actual=" + std::to_string(actual) +
                    " expected=" + std::to_string(expansion[component]) + context);
            requirePmmm(std::abs(expansion[component] - expected[component]) <
                    budget[component == 0 ? 0 : 1] * scale,
                "finite expansion exceeds the Ewald truncation budget for particle " +
                    std::to_string(sink) + " component=" + std::to_string(component) + context);
        }
        double const potentialError =
            std::abs(samples[sink].potential - expected[0]) / potentialScale;
        maximumError = std::max(maximumError, potentialError);
        requirePmmm(std::isfinite(samples[sink].potential) &&
                potentialError <
                    budget[0] + kNumericalTolerance + selfCancellationAllowance / potentialScale,
            "potential differs from direct periodic Ewald for particle " + std::to_string(sink) +
                ": scaled error " + std::to_string(potentialError) +
                " actual=" + std::to_string(samples[sink].potential) +
                " expected=" + std::to_string(expected[0]) + context);
        for (int dim = 0; dim < kDimensions; ++dim) {
            double const error =
                std::abs(samples[sink].acceleration[dim] - expected[dim + 1]) / norm;
            maximumError = std::max(maximumError, error);
            requirePmmm(std::isfinite(samples[sink].acceleration[dim]) &&
                    error < budget[1] + kNumericalTolerance,
                "force differs from direct periodic Ewald for particle " + std::to_string(sink) +
                    ": scaled error " + std::to_string(error) + " dim=" + std::to_string(dim) +
                    " actual=" + std::to_string(samples[sink].acceleration[dim]) +
                    " expected=" + std::to_string(expected[dim + 1]) + context);
            momentum[dim] += samples[sink].mass * samples[sink].acceleration[dim];
        }
    }
    for (double component : momentum) {
        requirePmmm(std::abs(component) < 3.0e-5 * forceScale,
            "unequal-mass periodic particles have nonzero total force");
    }
    return maximumError;
}

void comparePmmmSamples(
    Vector<PmmmSample> const& first, Vector<PmmmSample> const& second, bool potential = true) {
    auto const& options = getOptions();
    std::string const context = " near=fmm mesh=" + std::to_string(options.gravity.meshSize) +
        " fmmOrder=" + std::to_string(gravity::kBuildFmmOrder) +
        " theta=" + std::to_string(options.gravity.openingAngle) +
        " cuda=" + std::to_string(options.gravity.cuda);
    requirePmmm(first.size() == second.size(), "solve changed the particle count");
    for (size_t i = 0; i < first.size(); ++i) {
        requirePmmm(first[i].id == second[i].id && first[i].mass == second[i].mass,
            "solve changed a particle's identity or mass");
        for (int dim = 0; dim < kDimensions; ++dim) {
            requirePmmm(first[i].rawPosition[dim] == second[i].rawPosition[dim],
                "solve changed a particle position");
            requirePmmm(std::abs(first[i].acceleration[dim] - second[i].acceleration[dim]) <
                    2.0e-5 * std::max(1.0f, std::abs(first[i].acceleration[dim])),
                "force changed for id=" + std::to_string(first[i].id) +
                    " dim=" + std::to_string(dim) +
                    " reference=" + std::to_string(first[i].acceleration[dim]) +
                    " actual=" + std::to_string(second[i].acceleration[dim]) + context);
        }
        if (potential) {
            // The existing FMM accumulates softened self potential and subtracts
            // it in float. Bound that cancellation's rounding scale separately
            // from the force and approximation limits.
            double const selfCancellationAllowance = 4.0 * std::numeric_limits<float>::epsilon() *
                std::abs(options.gravity.gm * first[i].mass *
                    selectedGravityKernel().centralPotential / options.gravity.smoothingLength);
            requirePmmm(std::abs(first[i].potential - second[i].potential) <
                    2.0e-5 * std::max(1.0f, std::abs(first[i].potential)) +
                        selfCancellationAllowance,
                "potential changed for id=" + std::to_string(first[i].id) +
                    " reference=" + std::to_string(first[i].potential) +
                    " actual=" + std::to_string(second[i].potential) + context);
        }
    }
}

std::atomic<size_t> endpointRootBuilds{0};

void installEndpointAudit() {
    endpointRootBuilds = 0;
    pmmmSetCellObserver([](PmmmCellEvent event, uint64_t) {
        if (event == PmmmCellEvent::OwnedRootReady) {
            ++endpointRootBuilds;
        }
    });
}

std::array<double, 4> readEndpointAudit(bool checkField) {
    std::array<double, 4> result{0, 0, 1.e30, double(endpointRootBuilds.load())};
    if (!checkField) {
        pmmmSetCellObserver({});
        return result;
    }
    auto const& state = pmmm::detail::state;
    int const meshSize = pmmm::detail::cache.meshSize;
    double const h = 1. / meshSize;
    double const trace = (4 * std::numbers::pi / 3) * h * h * state.background[0];
    double const gm = getOptions().gravity.gm;
    requirePmmm(state.boundaryTreeReady, "boundary did not retain its endpoint forest");
    for (auto const& cell : state.cells) {
        auto const coordinates = pmmm::detail::coordinates(cell.key);
        auto const coefficients = pmmm::detail::localExpansion(cell.key);
        for (ParticleInteger i = cell.begin; i < cell.end; ++i) {
            Vector3<double> offset;
            for (int axis = 0; axis < kDimensions; ++axis) {
                double const center = (coordinates[axis] + .5) * h;
                Position const fixedCenter(center);
                offset[axis] = (unwrappedDoubleDistance(particlesPos(axis, i), fixedCenter) +
                                   (fixedCenter.toDouble() - center)) *
                    meshSize;
            }
            std::array<double, 4> expected{};
            pmmm::evaluate(kFmmOrder, expected, coefficients.data(), trace, offset, h);
            double norm = 0;
            for (int axis = 0; axis < kDimensions; ++axis) {
                double const acceleration = gm * expected[axis + 1];
                result[0] = std::max(result[0],
                    std::abs(double(particlesLongAcceleration(axis, i)) - acceleration) /
                        std::max(1., std::abs(acceleration)));
                norm += acceleration * acceleration;
            }
            result[1] += .5 * double(particlesMass(i)) * gm * expected[0];
            result[2] = std::min(result[2],
                state.request.eta *
                    std::sqrt(state.request.scale * h / std::max(std::sqrt(norm), 1.e-30)));
        }
    }
    return result;
}

}    // namespace detail

HPX_PLAIN_ACTION(detail::installEndpointAudit, InstallEndpointAuditAction);
HPX_PLAIN_ACTION(detail::readEndpointAudit, ReadEndpointAuditAction);
HPX_PLAIN_ACTION(detail::initializePmmmFixture, InitializePmmmFixtureAction);
HPX_PLAIN_ACTION(detail::moveFrozenPmmmFixture, MoveFrozenPmmmFixtureAction);
HPX_PLAIN_ACTION(detail::readPmmmFixture, ReadPmmmFixtureAction);
HPX_PLAIN_ACTION(detail::readPmmmKickFixture, ReadPmmmKickFixtureAction);
HPX_PLAIN_ACTION(detail::installPmmmOverlapGate, InstallPmmmOverlapGateAction);
HPX_PLAIN_ACTION(detail::readPmmmOverlapGate, ReadPmmmOverlapGateAction);
HPX_PLAIN_ACTION(detail::buildPmmmTestForest, BuildPmmmTestForestAction);
HPX_PLAIN_ACTION(detail::verifyDepositedMoments, VerifyDepositedMomentsAction);
HPX_PLAIN_ACTION(detail::installSourceAudit, InstallSourceAuditAction);
HPX_PLAIN_ACTION(detail::readSourceAudit, ReadSourceAuditAction);
HPX_PLAIN_ACTION(detail::reservePmmmCudaFixture, ReservePmmmCudaFixtureAction);
HPX_PLAIN_ACTION(detail::resetPmmmSavedForces, ResetPmmmSavedForcesAction);

namespace detail {

void preparePmmmFixture(int fixture, int ownership = 0) {
    expectedPmmmParticleCount = size_t(pmmmFixtureParticleCount(fixture));
    std::vector<hpx::future<void>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<InitializePmmmFixtureAction>(locality, fixture, ownership));
    }
    for (auto& future : futures) {
        future.get();
    }
}

void distributePmmmFixture(int fixture) {
    auto const saved = getOptions();
    auto options = saved;
    options.initial.numberOfParticles = pmmmFixtureParticleCount(fixture);
    // Fresh fixtures place every rung on arbitrary localities. Preserve the
    // PMMM-aligned domain cuts and use the explicit all-rung migration mode.
    // Disabling gravity here would let domains split an occupied mesh cell.
    setOptions(options);
    particlesSortByRung(0);
    domainsRebound();
    domainsBegin(-1);
    domainsEnd();
    setOptions(saved);
    std::vector<hpx::future<void>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<ResetPmmmSavedForcesAction>(locality));
    }
    for (auto& future : futures) {
        future.get();
    }
}

void checkPmmmForestGeometry() {
    treePrepareCellForest();
    std::vector<hpx::future<Vector<TreeCellRoot>>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<BuildPmmmTestForestAction>(locality));
    }
    hpx::wait_all(futures.begin(), futures.end());
    std::map<uint64_t, Vector<TreeId>> contributors;
    for (auto& future : futures) {
        for (auto const& root : future.get()) {
            contributors[root.key].push_back(root.id);
        }
    }
    Vector<TreeCellRoot> combinedRoots;
    for (auto const& [key, roots] : contributors) {
        auto const combined = treeCombineCellRoots(key, getOptions().gravity.meshSize, roots);
        combinedRoots.push_back({key, combined});
    }
    // Complete the forest before any remote node-cache reads: adding another
    // root can populate slots in a cache line fetched by an earlier traversal.
    size_t internalNodes = 0;
    for (auto const& root : combinedRoots) {
        internalNodes += verifyPmmmTreeCell(root.id, root.key, getOptions().gravity.meshSize);
    }
    requirePmmm(internalNodes > 0, "dense forest contained no internal tree nodes");
    treeDestroy();
    particlesCacheFree();
}

Vector<PmmmSample> gatherPmmmSamples() {
    std::vector<hpx::future<Vector<PmmmSample>>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<ReadPmmmFixtureAction>(locality));
    }
    Vector<PmmmSample> samples;
    for (auto& future : futures) {
        auto local = future.get();
        samples.insert(samples.end(), local.begin(), local.end());
    }
    requirePmmm(samples.size() == expectedPmmmParticleCount,
        "force solve changed the fixture's global owned particle count");
    std::sort(samples.begin(), samples.end(),
        [](auto const& first, auto const& second) { return first.id < second.id; });
    double mass = 0.0;
    for (size_t i = 0; i < samples.size(); ++i) {
        auto const& sample = samples[i];
        requirePmmm(sample.id == int(i), "force solve lost or duplicated an owned particle");
        mass += sample.mass;
    }
    auto const diagnostics = pmmmDiagnostics();
    requirePmmm(
        diagnostics.nearRemoteTreeFetches == 0 && diagnostics.nearRemoteParticleFetches == 0,
        "near FMM fetched remote tree nodes or particles during its local walk");
    if (hpxSize() == 1 || samples.empty()) {
        requirePmmm(diagnostics.ghostParticles == 0,
            "single-locality or empty solve retained remote ghost sources");
    }
    requirePmmm(std::abs(diagnostics.sourceMass - mass) < 1.0e-6 * std::max(1.0, mass),
        "source mass differs from the particle population");
    requirePmmm(std::abs(diagnostics.depositedMass - mass) < 2.0e-6 * std::max(1.0, mass),
        "cell moments lost mass");
    std::vector<hpx::future<void>> momentChecks;
    for (auto const& locality : hpxLocalities()) {
        momentChecks.push_back(hpx::async<VerifyDepositedMomentsAction>(locality, samples));
    }
    for (auto& check : momentChecks) {
        check.get();
    }
    return samples;
}

double pmmmFieldDifference(Vector<PmmmSample> const& reference, Vector<PmmmSample> const& actual) {
    requirePmmm(reference.size() == actual.size(), "near solver changed the particle count");
    double maximumError = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        requirePmmm(reference[i].id == actual[i].id && reference[i].mass == actual[i].mass,
            "near solver changed particle identities or masses");
        double norm = 0.0;
        double error = 0.0;
        for (int axis = 0; axis < kDimensions; ++axis) {
            requirePmmm(reference[i].rawPosition[axis] == actual[i].rawPosition[axis],
                "near solver changed a particle position");
            requirePmmm(std::isfinite(actual[i].acceleration[axis]),
                "near solver produced a nonfinite acceleration");
            norm += double(reference[i].acceleration[axis]) * reference[i].acceleration[axis];
            double const delta =
                double(reference[i].acceleration[axis]) - actual[i].acceleration[axis];
            error += delta * delta;
        }
        maximumError = std::max(maximumError, std::sqrt(error) / std::max(1.0, std::sqrt(norm)));
        requirePmmm(
            std::isfinite(actual[i].potential), "near solver produced a nonfinite potential");
        maximumError = std::max(maximumError,
            std::abs(double(reference[i].potential) - actual[i].potential) /
                std::max(1.0, std::abs(double(reference[i].potential))));
    }
    return maximumError;
}

Vector<PmmmSample> gatherPmmmKickSamples() {
    Vector<PmmmSample> samples;
    std::vector<hpx::future<Vector<PmmmSample>>> futures;
    for (auto const& locality : hpxLocalities()) {
        futures.push_back(hpx::async<ReadPmmmKickFixtureAction>(locality));
    }
    for (auto& future : futures) {
        auto local = future.get();
        samples.insert(samples.end(), local.begin(), local.end());
    }
    std::sort(samples.begin(), samples.end(), [](auto const& first, auto const& second) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            if (first.rawPosition[dim] != second.rawPosition[dim]) {
                return first.rawPosition[dim] < second.rawPosition[dim];
            }
        }
        return false;
    });
    return samples;
}

/// Compare rebuilt frozen-cell FMM trees with direct forces after cell and seam crossings.

void checkPmmmEndpointReuse() {
    auto const saved = getOptions();
    auto options = saved;
    options.hydro.enabled = false;
    options.gravity.cuda = false;
    options.gravity.pmmmOuterInterval = true;
    options.gravity.meshSize = 8;
    options.gravity.pmmmCoarseMeshSize = 0;
    options.gravity.openingAngle = 0.8660254037844386;
    options.compute.bucketSize = 8;
    pmmmDestroy();
    setOptions(options);
    preparePmmmFixture(5);
    distributePmmmFixture(5);
    particlesSortByRung(0);
    for (auto const& locality : hpxLocalities()) {
        hpx::async<InstallEndpointAuditAction>(locality).get();
    }
    auto before = gatherPmmmKickSamples();
    auto const boundary = pmmmOuterBoundary(1, 1, 0, 1, 1);
    auto after = gatherPmmmKickSamples();
    // Coincident sources can change their relative order during kd partitioning.
    // The fixture's x velocity is its unique identity and remains unchanged here.
    auto identity = [](auto const& left, auto const& right) {
        return left.velocity[0] < right.velocity[0];
    };
    std::sort(before.begin(), before.end(), identity);
    std::sort(after.begin(), after.end(), identity);
    requirePmmm(before.size() == after.size(), "boundary changed the particle population");
    for (size_t i = 0; i < before.size(); ++i) {
        requirePmmm(before[i].rung == after[i].rung && before[i].velocity == after[i].velocity,
            "long-field endpoint preparation changed velocities or rungs");
    }
    double potential = 0;
    double timestep = 1.e30;
    double builds = 0;
    for (auto const& locality : hpxLocalities()) {
        auto const audit = hpx::async<ReadEndpointAuditAction>(locality, true).get();
        requirePmmm(audit[0] < 2.e-6, "Cartesian endpoint descent differs from spherical oracle");
        potential += audit[1];
        timestep = std::min(timestep, audit[2]);
        builds += audit[3];
    }
    requirePmmm(builds > 0, "endpoint did not build any source roots");
    requirePmmm(
        std::abs(potential - boundary.potentialEnergy) < 2.e-11 * std::max(1., std::abs(potential)),
        "Cartesian endpoint potential differs from spherical oracle");
    requirePmmm(std::abs(timestep - boundary.accelerationTimestep) <
            2.e-11 * std::max(1., std::abs(timestep)),
        "Cartesian endpoint timestep differs from spherical oracle");
    auto const transforms = pmmmDiagnostics().transformCount;
    KickParameters parameters{};
    parameters.htime = true;
    parameters.top = parameters.ascending = parameters.descending = true;
    parameters.firstCall = true;
    parameters.minRung = 0;
    parameters.a = 1;
    parameters.t0 = parameters.maxDt = boundary.nextInterval;
    parameters.eta = options.gravity.timestepEta;
    parameters.gm = options.gravity.gm;
    parameters.gravitySmoothingLength = options.gravity.smoothingLength;
    parameters.frozenNear = parameters.refreshMembership = true;
    parameters.doPhi = true;
    pmmmKick(parameters, true);
    requirePmmm(pmmmDiagnostics().transformCount == transforms,
        "opening short pass repeated the long transform");
    double afterBuilds = 0;
    for (auto const& locality : hpxLocalities()) {
        afterBuilds += hpx::async<ReadEndpointAuditAction>(locality, false).get()[3];
    }
    requirePmmm(afterBuilds == builds, "opening short pass rebuilt the endpoint roots");
    pmmmDestroy();
    setOptions(saved);
    std::cout << "PMMM endpoint roots reused; Cartesian long field matches spherical oracle\n";
}

void checkPmmmFrozenFmm() {
    auto const saved = getOptions();
    auto options = saved;
    options.hydro.enabled = false;
    options.gravity.pmmmOuterInterval = true;

    options.gravity.meshSize = 8;
    options.gravity.pmmmCoarseMeshSize = 0;
    options.gravity.openingAngle = .4;
    options.compute.bucketSize = 8;
    options.analysis.saveForce = true;
    pmmmDestroy();
    setOptions(options);
    preparePmmmFixture(5);
    distributePmmmFixture(5);
    KickParameters parameters{};
    parameters.htime = true;
    parameters.ascending = parameters.descending = parameters.top = false;
    parameters.firstCall = true;
    parameters.minRung = 0;
    parameters.a = parameters.t0 = parameters.maxDt = 1;
    parameters.eta = 1;
    parameters.gm = options.gravity.gm;
    parameters.gravitySmoothingLength = options.gravity.smoothingLength;
    parameters.saveForce = true;
    parameters.doPhi = true;
    parameters.frozenNear = parameters.refreshMembership = true;
    particlesSortByRung(0);
    pmmmKick(parameters);
    auto const transforms = pmmmDiagnostics().transformCount;
    std::map<int, std::array<int, 3>> frozenCells;
    for (auto const& sample : gatherPmmmKickSamples()) {
        auto& cell = frozenCells[int(sample.velocity[0])];
        for (int dim = 0; dim < kDimensions; ++dim) {
            cell[dim] = int(sample.position[dim] * options.gravity.meshSize);
        }
    }
    parameters.refreshMembership = false;
    for (int step = 0; step < 3; ++step) {
        std::vector<hpx::future<void>> moves;
        for (auto const& locality : hpxLocalities()) {
            moves.push_back(hpx::async<MoveFrozenPmmmFixtureAction>(locality));
        }
        hpx::wait_all(moves);
        for (auto& move : moves) {
            move.get();
        }
        parameters.minRung = step == 1 ? 2 : 0;
        particlesSortByRung(parameters.minRung);
        pmmmKick(parameters);
        auto const actual = gatherPmmmKickSamples();
        auto const diagnostics = pmmmDiagnostics();
        requirePmmm(diagnostics.transformCount == transforms,
            "frozen FMM performed an inner mesh transform");
        requirePmmm(saved.gravity.cuda || diagnostics.nearCpuAcceptedMultipoles > 0,
            "frozen FMM never accepted a multipole interaction");
        requirePmmm(diagnostics.activeParticles == (step == 1 ? 240U : 320U) &&
                diagnostics.evaluatedParticles == diagnostics.activeParticles,
            "frozen FMM did not select the rung's sources and sinks");
        // Direct softened primary pairs with the original cell-selected images.
        // This oracle keeps theta and frozen membership fixed throughout the interval.
        auto direct = actual;
        double const support = options.gravity.smoothingLength;
        int const side = options.gravity.meshSize;
        for (size_t sink = 0; sink < direct.size(); ++sink) {
            if (direct[sink].rung < parameters.minRung) {
                continue;
            }
            auto const& sinkCell = frozenCells.at(int(actual[sink].velocity[0]));
            std::array<double, 4> field{};
            for (size_t source = 0; source < actual.size(); ++source) {
                if (source == sink || actual[source].rung < parameters.minRung) {
                    continue;
                }
                auto const& sourceCell = frozenCells.at(int(actual[source].velocity[0]));
                std::array<double, 3> displacement{};
                double cellDistanceSquared = 0;
                double distanceSquared = 0;
                for (int dim = 0; dim < kDimensions; ++dim) {
                    int delta = (sinkCell[dim] - sourceCell[dim] + side) % side;
                    if (delta > side / 2) {
                        delta -= side;
                    }
                    cellDistanceSquared += double(delta) * delta;
                    double sinkOffset = actual[sink].position[dim] - (sinkCell[dim] + .5) / side;
                    double sourceOffset =
                        actual[source].position[dim] - (sourceCell[dim] + .5) / side;
                    sinkOffset -= std::nearbyint(sinkOffset);
                    sourceOffset -= std::nearbyint(sourceOffset);
                    displacement[dim] = double(delta) / side + sinkOffset - sourceOffset;
                    distanceSquared += displacement[dim] * displacement[dim];
                }
                if (cellDistanceSquared *
                        (options.gravity.openingAngle * options.gravity.openingAngle) >
                    3) {
                    continue;
                }
                double const radius = std::sqrt(distanceSquared);
                double const mass = options.gravity.gm * actual[source].mass;
                double const radial = radius >= support ?
                    mass / (distanceSquared * radius) :
                    mass * radial_kernel::forceFactor(selectedGravityKernel(), radius / support) /
                        (support * support * support);
                field[0] -= radius >= support ?
                    mass / radius :
                    mass * radial_kernel::potential(selectedGravityKernel(), radius / support) /
                        support;
                for (int dim = 0; dim < kDimensions; ++dim) {
                    field[dim + 1] -= radial * displacement[dim];
                }
            }
            direct[sink].potential = field[0];
            for (int dim = 0; dim < kDimensions; ++dim) {
                direct[sink].acceleration[dim] = field[dim + 1];
            }
        }
        double const error = pmmmFieldDifference(direct, actual);
        requirePmmm(error < 2.e-3,
            "frozen FMM differs from direct active near pairs at step " + std::to_string(step) +
                ": " + std::to_string(error));
    }
    particlesSortByRung(0);
    pmmmDestroy();
    setOptions(saved);
    std::cout << "PMMM rebuilt frozen-cell FMM, seam crossings and rung selection passed\n";
}

void checkPmmmTraditionalKicks() {
    auto const saved = getOptions();
    auto options = saved;
    // Fixture 8 lies at side-16 cell centers. Keep that mesh so the
    // independent Ewald kick check is free of finite-order truncation at q=3..9.
    options.gravity.meshSize = 16;
    options.gravity.pmmmCoarseMeshSize = 0;
    options.gravity.openingAngle = 0.02;
    options.analysis.saveForce = true;
    pmmmDestroy();
    setOptions(options);
    preparePmmmFixture(8);
    distributePmmmFixture(8);
    particlesSortByRung(0);
    auto const before = gatherPmmmKickSamples();
    Vector<PmmmSample> activeSources;
    double totalMass = 0.0;
    for (auto const& sample : before) {
        totalMass += sample.mass;
        if (sample.rung >= 2) {
            activeSources.push_back(sample);
        }
    }
    requirePmmm(
        activeSources.size() < before.size(), "traditional fixture needs inactive force sources");
    KickParameters parameters{};
    parameters.htime = false;
    parameters.ascending = parameters.descending = parameters.top = true;
    parameters.firstCall = false;
    parameters.minRung = 2;
    parameters.a = parameters.t0 = parameters.maxDt = 1.0f;
    parameters.eta = 1.0e6f;
    parameters.gravitySmoothingLength = options.gravity.smoothingLength;
    parameters.gm = options.gravity.gm;
    parameters.theta = options.gravity.openingAngle;
    parameters.gpu = options.gravity.cuda;
    parameters.saveForce = true;
    parameters.doPhi = true;
    // The traditional solver must still use every source and the base mesh
    // even if a caller left a rung-selected range from a preceding stage.
    particlesSortByRung(parameters.minRung);
    auto const result = pmmmKick(parameters);
    auto const after = gatherPmmmKickSamples();
    auto const diagnostics = pmmmDiagnostics();
    requirePmmm(diagnostics.activeParticles == before.size() &&
            diagnostics.totalParticles == before.size() &&
            diagnostics.evaluatedParticles == activeSources.size() &&
            std::abs(diagnostics.sourceMass - totalMass) < 1.0e-6 &&
            std::abs(diagnostics.depositedMass - totalMass) < 1.0e-6,
        "traditional kick omitted inactive particles from its sources");
    requirePmmm(after.size() == before.size(), "traditional kick changed the particle count");
    bool inactiveSourceMatters = false;
    int minimumActiveRung = kMaxRung, maximumActiveRung = 0;
    size_t activeIndex = 0;
    for (size_t i = 0; i < before.size(); ++i) {
        auto const& initial = before[i];
        bool const active = initial.rung >= parameters.minRung;
        auto const reference = directPmmmReference(i, before);
        auto const activeReference =
            active ? directPmmmReference(activeIndex++, activeSources) : std::array<double, 4>{};
        int const expectedRung =
            active ? std::max(initial.rung - 1, parameters.minRung) : initial.rung;
        requirePmmm(after[i].rung == expectedRung,
            "traditional kick changed an inactive rung or selected a common active rung");
        if (active) {
            minimumActiveRung = std::min(minimumActiveRung, after[i].rung);
            maximumActiveRung = std::max(maximumActiveRung, after[i].rung);
        }
        for (int dim = 0; dim < kDimensions; ++dim) {
            requirePmmm(after[i].rawPosition[dim] == initial.rawPosition[dim],
                "traditional kick changed a particle position");
            if (active) {
                double const field = reference[dim + 1];
                inactiveSourceMatters |= std::abs(field - activeReference[dim + 1]) > 1.0e-3;
                requirePmmm(std::abs(after[i].acceleration[dim] - field) <
                        1.0e-3 * std::max(1.0, std::abs(field)),
                    "traditional active force differs from the all-source Ewald field");
            } else {
                requirePmmm(after[i].acceleration[dim] == initial.acceleration[dim],
                    "traditional kick changed an inactive acceleration");
            }
            if (!active) {
                requirePmmm(after[i].velocity[dim] == initial.velocity[dim],
                    "traditional gravity kicked an inactive velocity");
            } else {
                // The kick applies the acceleration it computed; its accuracy against
                // the Ewald field is checked above, so the half-step check uses it.
                double const interval =
                    std::ldexp(0.5, -initial.rung) + std::ldexp(0.5, -expectedRung);
                double const change = interval * after[i].acceleration[dim];
                double const expected = initial.velocity[dim] + change;
                requirePmmm(std::abs(after[i].velocity[dim] - expected) <
                        2.0e-6 * std::max(1.0, std::abs(expected)) + 1.0e-3 * std::abs(change),
                    "traditional DM kick did not use its old and new individual half-steps");
            }
        }
    }
    requirePmmm(inactiveSourceMatters,
        "traditional fixture cannot distinguish all-particle from active-only sources");
    requirePmmm(minimumActiveRung < maximumActiveRung && result.maxRung == maximumActiveRung,
        "traditional kick flattened distinct active rungs or reported their maximum incorrectly");
    particlesSortByRung(0);
    // Unlike the cell-center oracle above, fixture zero puts active and
    // inactive particles in the same source leaf. Its full solve supplies a
    // same-order reference for the selected sink views on CPU and CUDA.
    preparePmmmFixture(0);
    distributePmmmFixture(0);
    pmmmSolve(true);
    auto const fullFields = gatherPmmmSamples();
    particlesSortByRung(parameters.minRung);
    pmmmKick(parameters);
    auto const mixedSinks = gatherPmmmKickSamples();
    size_t expectedEvaluations = 0;
    for (auto const& reference : fullFields) {
        if (1 + reference.id % 4 < parameters.minRung) {
            continue;
        }
        ++expectedEvaluations;
        auto const found =
            std::find_if(mixedSinks.begin(), mixedSinks.end(), [&](auto const& sink) {
                for (int dim = 0; dim < kDimensions; ++dim) {
                    if (sink.rawPosition[dim] != reference.rawPosition[dim]) {
                        return false;
                    }
                }
                return true;
            });
        requirePmmm(found != mixedSinks.end(), "selected mixed-leaf sink lost its identity");
        for (int dim = 0; dim < kDimensions; ++dim) {
            requirePmmm(std::abs(found->acceleration[dim] - reference.acceleration[dim]) <
                    2.0e-5 * std::max(1.0, std::abs(double(reference.acceleration[dim]))),
                "selected mixed-leaf sink differs from the full PMMM force");
        }
    }
    requirePmmm(pmmmDiagnostics().evaluatedParticles == expectedEvaluations,
        "mixed-leaf PMMM evaluated inactive sinks");
    particlesSortByRung(0);
    pmmmDestroy();
    setOptions(saved);
    std::cout << "PMMM traditional all-source forces and active-only individual kicks passed\n"
              << std::flush;
}

void checkPmmmStreamedStages() {
    auto const saved = getOptions();
    auto options = saved;
    // Exercise the hierarchical stage operator directly in either integrator
    // build, without changing the application's compiled evolution policy.
    // Three of seven active particles exceed one eighth of the population, so
    // the stage selects the full side 16 rather than the coarse side 8.
    options.gravity.meshSize = 16;
    options.gravity.pmmmCoarseMeshSize = 8;
    options.gravity.openingAngle = 0.02;
    options.analysis.saveForce = false;
    options.analysis.saveAcceleration = false;
    options.analysis.savePotential = false;
    pmmmDestroy();
    {
        for (int batchParticles : {1, 4096}) {
            options.gravity.pmmmBatchParticles = batchParticles;
            setOptions(options);
            // At cell centers PMMM matches the independent Ewald field at every
            // supported order. Off-center force accuracy is checked separately;
            // its finite-order truncation must not set the kick tolerance.
            preparePmmmFixture(8);
            distributePmmmFixture(8);
            auto const before = gatherPmmmKickSamples();
            Vector<PmmmSample> active;
            double activeMass = 0.0;
            for (auto const& sample : before) {
                if (sample.rung >= 3) {
                    active.push_back(sample);
                    activeMass += sample.mass;
                }
            }
            requirePmmm(active.size() == 3 && before.size() == 7,
                "streaming fixture did not contain both active and inactive particles");
            particlesSortByRung(3);
            KickParameters parameters{};
            parameters.htime = true;
            parameters.ascending = true;
            parameters.descending = false;
            parameters.top = batchParticles == 1;
            parameters.sign = -1.0f;
            parameters.firstCall = false;
            parameters.saveForce = false;
            parameters.gpu = options.gravity.cuda;
            parameters.minRung = 3;
            parameters.minLevel = 0;
            parameters.a = 1.0f;
            parameters.t0 = 1.0f;
            parameters.theta = options.gravity.openingAngle;
            parameters.gravitySmoothingLength = options.gravity.smoothingLength;
            parameters.eta = options.gravity.timestepEta;
            parameters.gm = options.gravity.gm;
            parameters.nodeLoad = 1.0f;
            parameters.doPhi = true;
            pmmmKick(parameters);
            auto const after = gatherPmmmKickSamples();
            auto const diagnostics = pmmmDiagnostics();
            requirePmmm(diagnostics.meshSize == 16 && diagnostics.activeParticles == 3 &&
                    diagnostics.totalParticles == 7,
                "streamed stage did not select the full mesh from global active and owned counts");
            requirePmmm(diagnostics.storedAccelerationParticles == 0 &&
                    diagnostics.storedPotentialParticles == 0,
                "streamed stage allocated retained full-particle result arrays");
            requirePmmm(std::abs(diagnostics.sourceMass - activeMass) < 1.0e-6 &&
                    std::abs(diagnostics.depositedMass - activeMass) < 1.0e-6,
                "hierarchical stage included inactive sources in the mesh moments");
            requirePmmm(diagnostics.nearRemoteTreeFetches == 0 &&
                    diagnostics.nearRemoteParticleFetches == 0,
                "streamed local walk performed a remote fetch");
            requirePmmm(before.size() == after.size(), "streamed stage changed owned count");
            size_t selected = 0;
            for (size_t i = 0; i < before.size(); ++i) {
                bool const isActive = before[i].rung >= 3;
                auto const reference =
                    isActive ? directPmmmReference(selected++, active) : std::array<double, 4>{};
                double const factor = parameters.sign * (parameters.top ? 1.0 : -1.0) / 16.0;
                requirePmmm(
                    before[i].rung == after[i].rung, "ascending-only kick changed a particle rung");
                for (int dim = 0; dim < kDimensions; ++dim) {
                    requirePmmm(before[i].rawPosition[dim] == after[i].rawPosition[dim],
                        "streamed kick changed an owned position");
                    double const delta = factor * reference[dim + 1];
                    double const expected = before[i].velocity[dim] + delta;
                    requirePmmm(isActive ? std::abs(after[i].velocity[dim] - expected) <
                                2.0e-6 * std::max(1.0, std::abs(expected)) +
                                    1.0e-3 * std::abs(delta) :
                                           after[i].velocity[dim] == before[i].velocity[dim],
                        "stage kick differs from its active-source Ewald field or changed an "
                        "inactive sink");
                }
            }
            // A descending stage must select the next rung from the combined
            // field while independently saving just the requested diagnostic.
            options.analysis.saveAcceleration = batchParticles == 1;
            options.analysis.savePotential = batchParticles != 1;
            setOptions(options);
            parameters.ascending = false;
            parameters.descending = true;
            parameters.maxDt = parameters.t0 / 16.0f;
            parameters.eta = 1.0e6f;
            pmmmKick(parameters);
            auto const descending = gatherPmmmKickSamples();
            auto const descendingDiagnostics = pmmmDiagnostics();
            requirePmmm(descendingDiagnostics.meshSize == 16 &&
                    descendingDiagnostics.activeParticles == 3 &&
                    descendingDiagnostics.totalParticles == 7 &&
                    descendingDiagnostics.planBuilds == diagnostics.planBuilds &&
                    descendingDiagnostics.greenBuilds == diagnostics.greenBuilds,
                "descending stage changed or rebuilt the same active-particle mesh");
            requirePmmm(descending.size() == before.size(), "descending stage changed owned count");
            selected = 0;
            for (size_t i = 0; i < before.size(); ++i) {
                bool const isActive = before[i].rung >= 3;
                auto const reference =
                    isActive ? directPmmmReference(selected++, active) : std::array<double, 4>{};
                double const factor = parameters.sign * (parameters.top ? 1.0 : -1.0) / 16.0;
                requirePmmm(descending[i].rung == (isActive ? 4 : before[i].rung),
                    "descending stage selected the wrong next rung");
                for (int dim = 0; dim < kDimensions; ++dim) {
                    requirePmmm(descending[i].rawPosition[dim] == before[i].rawPosition[dim],
                        "descending kick changed an owned position");
                    double const delta = factor * reference[dim + 1];
                    double const expected = after[i].velocity[dim] + delta;
                    requirePmmm(isActive ? std::abs(descending[i].velocity[dim] - expected) <
                                2.0e-6 * std::max(1.0, std::abs(expected)) +
                                    1.0e-3 * std::abs(delta) :
                                           descending[i].velocity[dim] == before[i].velocity[dim],
                        "descending kick differs from its active-source Ewald field or changed an "
                        "inactive sink");
                    if (isActive && options.analysis.saveAcceleration) {
                        requirePmmm(std::abs(descending[i].acceleration[dim] - reference[dim + 1]) <
                                1.0e-3 * std::max(1.0, std::abs(reference[dim + 1])),
                            "acceleration-only diagnostics differ from the combined active-source "
                            "field");
                    }
                }
                if (isActive && options.analysis.savePotential) {
                    requirePmmm(std::abs(descending[i].potential - reference[0]) <
                            1.0e-3 * std::max(1.0, std::abs(reference[0])),
                        "potential-only diagnostics differ from the combined active-source field");
                }
            }
            options.analysis.saveAcceleration = false;
            options.analysis.savePotential = false;
            setOptions(options);
            particlesSortByRung(0);
        }
    }
    pmmmDestroy();
    setOptions(saved);
    std::cout
        << "PMMM streamed active-source stages, signed kicks, optional fields, and batch-size "
           "checks passed\n"
        << std::flush;
}

void checkPmmmStageMeshes() {
    auto const saved = getOptions();
    auto options = saved;
    // Stages select the smallest power of two between the coarse side 8 and
    // the base side 16 whose volume covers the active fraction: side 8 needs
    // at most one eighth of the population active, which this seven-particle
    // fixture reaches only with no active particles.
    options.gravity.meshSize = 16;
    options.gravity.pmmmCoarseMeshSize = 8;
    options.gravity.openingAngle = 0.02;
    options.analysis.saveForce = false;
    options.analysis.saveAcceleration = false;
    options.analysis.savePotential = false;
    // The configured initial population is deliberately unrelated to the
    // current owned fixture. Mesh selection must reduce actual owned counts.
    options.initial.numberOfParticles = 32768;
    pmmmDestroy();
    setOptions(options);
    preparePmmmFixture(8);
    distributePmmmFixture(8);
    auto const before = gatherPmmmKickSamples();
    pmmmSolve(true);
    auto const fullReference = gatherPmmmSamples();
    auto const initial = pmmmDiagnostics();
    requirePmmm(
        initial.meshSize == 16 && initial.activeParticles == 7 && initial.totalParticles == 7,
        "all-particle solve did not retain the configured base mesh and actual owned count");

    KickParameters parameters{};
    parameters.htime = true;
    parameters.ascending = true;
    parameters.descending = false;
    parameters.top = true;
    parameters.sign = 0.0f;
    parameters.firstCall = false;
    parameters.saveForce = false;
    parameters.gpu = options.gravity.cuda;
    parameters.minLevel = 0;
    parameters.a = 1.0f;
    parameters.t0 = 1.0f;
    parameters.theta = options.gravity.openingAngle;
    parameters.gravitySmoothingLength = options.gravity.smoothingLength;
    parameters.eta = options.gravity.timestepEta;
    parameters.gm = options.gravity.gm;
    parameters.nodeLoad = 1.0f;
    parameters.doPhi = true;
    auto stage = [&](int minRung, uint64_t expectedActive, int expectedMesh) {
        particlesSortByRung(0);
        particlesSortByRung(minRung);
        parameters.minRung = minRung;
        pmmmKick(parameters);
        auto const diagnostics = pmmmDiagnostics();
        requirePmmm(diagnostics.meshSize == expectedMesh &&
                diagnostics.activeParticles == expectedActive && diagnostics.totalParticles == 7,
            "hierarchical mesh used a local or configured initial particle count");
        requirePmmm(
            getOptions().gravity.meshSize == 16 && getOptions().gravity.pmmmCoarseMeshSize == 8,
            "hierarchical mesh selection changed the configured base or coarse mesh");
        double selectedMass = 0.0;
        for (auto const& sample : before) {
            if (sample.rung >= minRung) {
                selectedMass += sample.mass;
            }
        }
        requirePmmm(std::abs(diagnostics.sourceMass - selectedMass) < 1.0e-6 &&
                std::abs(diagnostics.depositedMass - selectedMass) < 1.0e-6,
            "changing the hierarchical mesh changed the selected source mass");
        auto const after = gatherPmmmKickSamples();
        requirePmmm(before.size() == after.size(), "mesh switching changed the owned count");
        for (size_t i = 0; i < before.size(); ++i) {
            requirePmmm(before[i].rung == after[i].rung,
                "mesh switching changed a particle rung without a descending kick");
            for (int dim = 0; dim < kDimensions; ++dim) {
                requirePmmm(before[i].rawPosition[dim] == after[i].rawPosition[dim] &&
                        before[i].velocity[dim] == after[i].velocity[dim],
                    "mesh switching changed physical particle state with a zero signed kick");
            }
        }
        return diagnostics;
    };
    auto const full = stage(0, 7, 16);
    requirePmmm(full.planBuilds == initial.planBuilds && full.greenBuilds == initial.greenBuilds,
        "full hierarchical stage rebuilt the existing all-particle mesh");
    // Three and one of seven active particles both exceed one eighth of the
    // population, so these partial stages reuse the full side-16 mesh.
    auto const partial = stage(3, 3, 16);
    requirePmmm(partial.planBuilds == full.planBuilds && partial.greenBuilds == full.greenBuilds,
        "partial stage above the coarse fraction rebuilt the retained full mesh");
    auto const sparse = stage(4, 1, 16);
    requirePmmm(
        sparse.planBuilds == partial.planBuilds && sparse.greenBuilds == partial.greenBuilds,
        "single-particle stage above the coarse fraction rebuilt the retained full mesh");
    auto const empty = stage(5, 0, 8);
    requirePmmm(
        empty.planBuilds == sparse.planBuilds + 1 && empty.greenBuilds == sparse.greenBuilds + 1,
        "empty stage did not build exactly one coarse FFT and Green cache");
    auto const emptyAgain = stage(5, 0, 8);
    requirePmmm(
        emptyAgain.planBuilds == empty.planBuilds && emptyAgain.greenBuilds == empty.greenBuilds,
        "repeated empty stage rebuilt its retained coarse mesh");
    auto const grown = stage(0, 7, 16);
    requirePmmm(
        grown.planBuilds == emptyAgain.planBuilds && grown.greenBuilds == emptyAgain.greenBuilds,
        "return to the base mesh rebuilt a retained FFT or Green cache");
    auto const coarseAgain = stage(5, 0, 8);
    requirePmmm(
        coarseAgain.planBuilds == grown.planBuilds && coarseAgain.greenBuilds == grown.greenBuilds,
        "return to the retained coarse mesh rebuilt the FFT or Green cache");

    // The retained rung-selected range must not affect an ordinary force solve.
    pmmmSolve(true);
    comparePmmmSamples(fullReference, gatherPmmmSamples());
    auto const restored = pmmmDiagnostics();
    requirePmmm(restored.meshSize == 16 && restored.activeParticles == 7 &&
            restored.totalParticles == 7 && restored.planBuilds == coarseAgain.planBuilds &&
            restored.greenBuilds == coarseAgain.greenBuilds,
        "ordinary force solve did not restore the cached full-particle mesh");
    particlesSortByRung(0);
    preparePmmmFixture(2);
    distributePmmmFixture(2);
    parameters.minRung = 0;
    pmmmKick(parameters);
    auto const noParticles = pmmmDiagnostics();
    requirePmmm(noParticles.meshSize == 8 && noParticles.activeParticles == 0 &&
            noParticles.totalParticles == 0 && noParticles.sourceMass == 0.0 &&
            noParticles.depositedMass == 0.0 && noParticles.planBuilds == restored.planBuilds &&
            noParticles.greenBuilds == restored.greenBuilds,
        "empty population did not select the retained coarse mesh");
    // The base mesh was inactive at destruction; neither it nor the active
    // coarse mesh may survive in the pool.
    pmmmDestroy();
    pmmmSolve(true);
    auto const afterDestroy = pmmmDiagnostics();
    requirePmmm(afterDestroy.meshSize == 16 && afterDestroy.activeParticles == 0 &&
            afterDestroy.totalParticles == 0 &&
            afterDestroy.planBuilds == noParticles.planBuilds + 1 &&
            afterDestroy.greenBuilds == noParticles.greenBuilds + 1,
        "explicit destruction retained an inactive FFT or Green cache");
    pmmmKick(parameters);
    auto const coarseRebuilt = pmmmDiagnostics();
    requirePmmm(coarseRebuilt.meshSize == 8 &&
            coarseRebuilt.planBuilds == afterDestroy.planBuilds + 1 &&
            coarseRebuilt.greenBuilds == afterDestroy.greenBuilds + 1,
        "explicit destruction retained the active coarse FFT or Green cache");
    particlesSortByRung(0);
    pmmmDestroy();
    setOptions(saved);
    std::cout << "PMMM hierarchical mesh selection, global counts, coarse/empty stages, and cache "
                 "reuse/destruction checks passed\n"
              << std::flush;
}

void checkPmmmCoarseMeshes() {
    auto const saved = getOptions();
    auto options = saved;
    // The coarse mesh bounds every hierarchical stage from below. With the
    // explicit coarse side 8 under the base side 16, an empty stage selects 8
    // and every nonempty stage of this seven-particle fixture selects 16; with
    // the coarse side equal to the base, every stage uses the one base mesh.
    options.gravity.meshSize = 16;
    options.gravity.pmmmCoarseMeshSize = 8;
    options.gravity.openingAngle = 0.02;
    options.analysis.saveForce = false;
    options.analysis.saveAcceleration = false;
    options.analysis.savePotential = false;
    pmmmDestroy();
    setOptions(options);
    preparePmmmFixture(8);
    distributePmmmFixture(8);
    pmmmSolve(true);
    auto const fullReference = gatherPmmmSamples();
    auto const initial = pmmmDiagnostics();
    requirePmmm(
        initial.meshSize == 16 && initial.activeParticles == 7 && initial.totalParticles == 7,
        "full solve under an explicit coarse mesh did not use the configured base mesh");

    KickParameters parameters{};
    parameters.htime = true;
    parameters.ascending = true;
    parameters.descending = false;
    parameters.top = true;
    parameters.sign = 0.0f;
    parameters.firstCall = false;
    parameters.saveForce = false;
    parameters.gpu = options.gravity.cuda;
    parameters.minLevel = 0;
    parameters.a = 1.0f;
    parameters.t0 = 1.0f;
    parameters.theta = options.gravity.openingAngle;
    parameters.gravitySmoothingLength = options.gravity.smoothingLength;
    parameters.eta = options.gravity.timestepEta;
    parameters.gm = options.gravity.gm;
    parameters.nodeLoad = 1.0f;
    parameters.doPhi = true;
    auto stage = [&](int minRung, uint64_t expectedActive, int expectedMesh) {
        particlesSortByRung(0);
        particlesSortByRung(minRung);
        parameters.minRung = minRung;
        pmmmKick(parameters);
        auto const diagnostics = pmmmDiagnostics();
        requirePmmm(diagnostics.meshSize == expectedMesh &&
                diagnostics.activeParticles == expectedActive && diagnostics.totalParticles == 7,
            "hierarchical stage did not select the expected coarse-bounded FFT mesh");
        requirePmmm(getOptions().gravity.meshSize == options.gravity.meshSize &&
                getOptions().gravity.pmmmCoarseMeshSize == options.gravity.pmmmCoarseMeshSize,
            "coarse-bounded mesh selection changed the configured base or coarse mesh");
        return diagnostics;
    };
    auto const full = stage(0, 7, 16);
    requirePmmm(full.planBuilds == initial.planBuilds && full.greenBuilds == initial.greenBuilds,
        "full stage under an explicit coarse mesh rebuilt the all-particle mesh");
    // One of seven active particles exceeds one eighth of the population, so
    // the sparse stage still selects the base side 16 and reuses its cache.
    auto const sparse = stage(4, 1, 16);
    requirePmmm(sparse.planBuilds == full.planBuilds && sparse.greenBuilds == full.greenBuilds,
        "sparse stage above the coarse fraction rebuilt the retained base mesh");
    auto const empty = stage(5, 0, 8);
    requirePmmm(
        empty.planBuilds == sparse.planBuilds + 1 && empty.greenBuilds == sparse.greenBuilds + 1,
        "first coarse stage did not build exactly one FFT and Green cache");
    auto const repeatedEmpty = stage(5, 0, 8);
    auto const restored = stage(0, 7, 16);
    requirePmmm(repeatedEmpty.planBuilds == empty.planBuilds &&
            repeatedEmpty.greenBuilds == empty.greenBuilds &&
            restored.planBuilds == empty.planBuilds && restored.greenBuilds == empty.greenBuilds,
        "coarse/full stages rebuilt retained meshes");
    pmmmSolve(true);
    comparePmmmSamples(fullReference, gatherPmmmSamples());
    requirePmmm(pmmmDiagnostics().meshSize == 16 &&
            pmmmDiagnostics().planBuilds == restored.planBuilds &&
            pmmmDiagnostics().greenBuilds == restored.greenBuilds,
        "ordinary force solve after coarse stages did not reuse the base mesh");
    // The coarse mesh is part of the cache identity; destroy before changing it.
    pmmmDestroy();

    options.gravity.pmmmCoarseMeshSize = 16;
    setOptions(options);
    auto const singleFull = stage(0, 7, 16);
    requirePmmm(singleFull.planBuilds == restored.planBuilds + 1 &&
            singleFull.greenBuilds == restored.greenBuilds + 1,
        "full stage after destruction did not rebuild exactly one base mesh cache");
    auto const singleSparse = stage(4, 1, 16);
    auto const singleEmpty = stage(5, 0, 16);
    requirePmmm(singleSparse.planBuilds == singleFull.planBuilds &&
            singleSparse.greenBuilds == singleFull.greenBuilds &&
            singleEmpty.planBuilds == singleFull.planBuilds &&
            singleEmpty.greenBuilds == singleFull.greenBuilds,
        "coarse mesh equal to the base built a second FFT or Green cache");
    pmmmSolve(true);
    comparePmmmSamples(fullReference, gatherPmmmSamples());
    requirePmmm(pmmmDiagnostics().planBuilds == singleFull.planBuilds &&
            pmmmDiagnostics().greenBuilds == singleFull.greenBuilds,
        "ordinary force solve with a base-sized coarse mesh rebuilt its cache");
    pmmmDestroy();

    // This tight angle covers every periodic cell, so the default coarse
    // mesh is capped at the full side even for an empty stage.
    options.gravity.pmmmCoarseMeshSize = 0;
    setOptions(options);
    auto const defaultEmpty = stage(5, 0, 16);
    auto const defaultFull = stage(0, 7, 16);
    requirePmmm(defaultFull.planBuilds == defaultEmpty.planBuilds &&
            defaultFull.greenBuilds == defaultEmpty.greenBuilds,
        "default coarse mesh did not respect the opening-angle stencil");
    pmmmSolve(true);
    comparePmmmSamples(fullReference, gatherPmmmSamples());
    particlesSortByRung(0);
    pmmmDestroy();
    setOptions(saved);
    std::cout
        << "PMMM explicit/default coarse mesh bounds, base-sized coarse stages, and cache reuse "
           "checks passed\n"
        << std::flush;
}

void checkPmmmCellOverlap() {
    if (hpxSize() < 2) {
        return;
    }
    auto const saved = getOptions();
    auto options = saved;
    options.gravity.meshSize = 8;
    options.gravity.openingAngle = 0.8660254037844386;
    options.gravity.pmmmBatchParticles = 1;
    pmmmDestroy();
    setOptions(options);
    preparePmmmFixture(6);
    distributePmmmFixture(6);
    pmmmSolve(true);
    auto const reference = gatherPmmmSamples();
    bool foundGate = false;
    uint64_t heldKey = 0;
    for (auto const& source : reference) {
        if (source.rank == 0) {
            continue;
        }
        bool dependent = false;
        bool independent = false;
        for (auto const& sink : reference) {
            if (sink.rank == 0) {
                dependent |= pmmmCellsNeighbor(source.cellKey, sink.cellKey);
                independent |= !pmmmCellsNeighbor(source.cellKey, sink.cellKey);
            }
        }
        if (dependent && independent) {
            heldKey = source.cellKey;
            foundGate = true;
            break;
        }
    }
    requirePmmm(
        foundGate, "overlap fixture lacks independent and remote-dependent rank-zero sinks");
    auto setGate = [heldKey](bool enabled) {
        std::vector<hpx::future<void>> futures;
        for (auto const& locality : hpxLocalities()) {
            futures.push_back(hpx::async<InstallPmmmOverlapGateAction>(locality, enabled, heldKey));
        }
        for (auto& future : futures) {
            future.get();
        }
    };
    setGate(true);
    try {
        pmmmSolve(true);
        auto const state = hpx::async<ReadPmmmOverlapGateAction>(hpxLocalities()[0]).get();
        requirePmmm(state.size() == 4 && state[0] && state[1] && !state[2] && state[3],
            "whole-cell gate did not prove independent completion before remote source arrival");
        comparePmmmSamples(reference, gatherPmmmSamples());
        requirePmmm(pmmmDiagnostics().completedWhileGhostsPending > 0,
            "cell pipeline did not report completion while the deterministic ghost gate was held");
    } catch (...) {
        setGate(false);
        throw;
    }
    setGate(false);
    pmmmDestroy();
    setOptions(saved);
    std::cout << "PMMM deterministic interior/boundary whole-cell overlap checks passed\n"
              << std::flush;
}

void checkRootDepositionRecovery() {
    auto const saved = getOptions();
    auto options = saved;
    options.gravity.meshSize = 8;
    options.gravity.openingAngle = 0.8660254037844386;
    options.gravity.pmmmBatchParticles = 1;
    pmmmDestroy();
    setOptions(options);
    preparePmmmFixture(6);
    distributePmmmFixture(6);
    auto install = [](bool enabled, int rank) {
        std::vector<hpx::future<void>> futures;
        for (auto const& locality : hpxLocalities()) {
            futures.push_back(hpx::async<InstallSourceAuditAction>(locality, enabled, rank));
        }
        for (auto& future : futures) {
            future.get();
        }
    };
    install(true, -1);
    pmmmSolve(true);
    auto const reference = gatherPmmmSamples();
    std::map<uint64_t, bool> occupied;
    for (auto const& sample : reference) {
        occupied[sample.cellKey] = true;
    }
    uint64_t built = 0, deposited = 0, converted = 0;
    for (auto const& locality : hpxLocalities()) {
        auto const counts = hpx::async<ReadSourceAuditAction>(locality).get();
        built += counts[0];
        deposited += counts[1];
        converted += counts[3];
        requirePmmm(counts[2] == 0, "owned source was built/deposited more than once");
    }
    requirePmmm(
        built == occupied.size() && deposited == occupied.size() && converted == occupied.size(),
        "owned-root build/deposition/local-conversion count differs from occupied cells");
    for (int failure : {reference.front().rank, -2 - reference.front().rank}) {
        install(true, failure);
        bool failed = false;
        try {
            pmmmSolve(true);
        } catch (std::exception const&) {
            failed = true;
        }
        requirePmmm(failed, "owned-root/far-local failure injection did not propagate");
        install(false, -1);
        pmmmSolve(true);
        comparePmmmSamples(reference, gatherPmmmSamples());
    }
    pmmmDestroy();
    setOptions(saved);
    std::cout << "PMMM owned-root exactly-once deposition and failure/recovery checks passed\n"
              << std::flush;
}

void checkPmmmFarOverlap() {
    if (hpxSize() != 1 || hpx::get_os_thread_count() < 2) {
        return;
    }
    auto const saved = getOptions();
    auto options = saved;
    options.gravity.meshSize = 8;
    options.gravity.openingAngle = 0.8660254037844386;
    options.gravity.pmmmBatchParticles = 1;
    pmmmDestroy();
    setOptions(options);
    // Eight mutually non-neighboring occupied cells. Hold the first producer
    // before publication/deposition; another producer can release a near batch.
    preparePmmmFixture(4);
    pmmmSolve(true);
    auto const reference = gatherPmmmSamples();
    hpx::promise<void> release;
    auto ready = release.get_future().share();
    std::atomic<bool> completed{false};
    pmmmSetCellObserver([&](PmmmCellEvent event, uint64_t key) {
        if (event == PmmmCellEvent::OwnedRootReady && key == 0) {
            requirePmmm(ready.wait_for(std::chrono::seconds(10)) == hpx::future_status::ready,
                "near/far overlap gate timed out before releasing the owned producer");
            ready.get();
        }
        if (event == PmmmCellEvent::NearCompleted && !completed.exchange(true)) {
            bool const early = !pmmm::detail::state.farReady.is_ready();
            release.set_value();
            requirePmmm(early, "near batch did not complete ahead of the blocked FFT");
        }
    });
    try {
        pmmmSolve(true);
    } catch (...) {
        pmmmSetCellObserver({});
        throw;
    }
    pmmmSetCellObserver({});
    requirePmmm(completed, "no near batch overlapped the held far-field producer");
    comparePmmmSamples(reference, gatherPmmmSamples());
    pmmmDestroy();
    setOptions(saved);
    std::cout << "PMMM deterministic near completion before far readiness passed\n" << std::flush;
}

void checkPmmmFmmNear() {
    auto const saved = getOptions();
    auto options = saved;
    // Coupled mesh/SPH evolution is deferred. Explicit-mass source permutation
    // and near-field algebra are covered independently above.
    options.hydro.enabled = false;
    options.compute.bucketSize = 8;
    options.gravity.openingAngle = 0.8660254037844386;
    options.gravity.smoothingLength = 0.001;
    if (saved.gravity.cuda) {
        setOptions(options);
        std::vector<hpx::future<void>> futures;
        for (auto const& locality : hpxLocalities()) {
            futures.push_back(hpx::async<ReservePmmmCudaFixtureAction>(locality));
        }
        for (auto& future : futures) {
            future.get();
        }
    }
    for (int side : {4, 8, 16}) {
        pmmmDestroy();
        options.gravity.cuda = saved.gravity.cuda;
        options.gravity.meshSize = side;
        options.gravity.openingAngle = 1.0e-6;
        setOptions(options);
        preparePmmmFixture(5);
        distributePmmmFixture(5);
        // The tree arena must also be allocated under the requested
        // backend before temporarily running the CPU reference on the same data.
        checkPmmmForestGeometry();
        options.gravity.cuda = false;
        setOptions(options);
        pmmmSolve(true);
        // A very tight angle supplies a near-direct reference; point-sized
        // cells can still use exact monopoles. The small fixtures separately
        // check the independent periodic Ewald sum.
        auto const direct = gatherPmmmSamples();
        auto const cache = pmmmDiagnostics();
        requirePmmm(direct.size() == 320, "dense near fixture lost particles");
        if (hpxSize() > 1) {
            // Coarse-aligned domains never split a mesh cell across localities,
            // yet the dense cells must still land on more than one rank so the
            // near field exchanges ghost sources below.
            std::map<uint64_t, int> cellRanks;
            std::map<int, size_t> rankPopulations;
            for (auto const& sample : direct) {
                auto const inserted = cellRanks.emplace(sample.cellKey, sample.rank);
                requirePmmm(inserted.first->second == sample.rank,
                    "coarse-aligned domains split a dense mesh cell across localities");
                ++rankPopulations[sample.rank];
            }
            requirePmmm(rankPopulations.size() > 1,
                "dense fixture did not distribute its occupied cells across localities");
        }
        // Clustered forces can strongly cancel. These angles test
        // a useful multipole approximation and convergence toward direct PP.
        // Dipole-only sources need a tighter acceptance angle to meet the
        // same approximation budget; the q>=4 fixture retains its original angle.
        pmmmDestroy();
        options.gravity.openingAngle = kFmmOrder == 3 ? 0.2 : 0.4;
        setOptions(options);
        pmmmSolve(true);
        auto const loose = gatherPmmmSamples();
        double const looseError = pmmmFieldDifference(direct, loose);
        auto const looseCounts = pmmmDiagnostics();
        requirePmmm(looseCounts.nearCpuAcceptedMultipoles > 0,
            "dense FMM near fixture never accepted a multipole interaction");
        if (hpxSize() > 1) {
            requirePmmm(looseCounts.ghostParticles > 0,
                "distributed dense fixture did not exchange ghost source particles");
        }
        pmmmDestroy();
        options.gravity.openingAngle = 0.02;
        setOptions(options);
        pmmmSolve(true);
        auto const strict = gatherPmmmSamples();
        double const strictError = pmmmFieldDifference(direct, strict);
        auto const strictCounts = pmmmDiagnostics();
        requirePmmm(
            looseError < 2.0e-2, "loose FMM near field differs excessively from direct pairs");
        requirePmmm(strictError < 5.0e-5 && strictError <= std::max(2.0e-5, 0.5 * looseError),
            "tightening the FMM near angle did not converge to direct pairs");
        requirePmmm(strictCounts.nearCpuDirectPairs > 0,
            "dense FMM near fixture did not reach direct softened/self interactions");
        pmmmSolve(false);
        auto const forceOnly = gatherPmmmSamples();
        comparePmmmSamples(strict, forceOnly, false);
        for (auto const& sample : forceOnly) {
            requirePmmm(sample.potential == 0.0f, "force-only FMM near solve computed a potential");
        }
        requirePmmm(strictCounts.planBuilds == cache.planBuilds + 2 &&
                strictCounts.greenBuilds == cache.greenBuilds + 2,
            "each reinitialized stencil should build one set of plans/Green spectra");
        pmmmSolve(true);
        comparePmmmSamples(strict, gatherPmmmSamples());
        requirePmmm(pmmmDiagnostics().ghostParticles == strictCounts.ghostParticles,
            "repeated FMM solve accumulated or lost ghost sources");
        preparePmmmFixture(5, 1);
        distributePmmmFixture(5);
        pmmmSolve(true);
        auto const repartitioned = gatherPmmmSamples();
        double const repartitionError = pmmmFieldDifference(direct, repartitioned);
        // Changing the kd decomposition changes the accepted multipoles. Test
        // both approximations against PP using the same force-vector criterion;
        // a nearly cancelling component need not match to relative float error.
        requirePmmm(repartitionError < 5.0e-5,
            "repartitioned FMM near field differs excessively from direct pairs");
        if (saved.gravity.cuda) {
            options.gravity.cuda = true;
            setOptions(options);
            pmmmSolve(true);
            auto const cuda = gatherPmmmSamples();
            requirePmmm(pmmmFieldDifference(direct, cuda) < 5.0e-5 &&
                    pmmmFieldDifference(repartitioned, cuda) < 5.0e-5,
                "CPU/CUDA FMM near fields disagree on dense periodic cells");
            options.gravity.cuda = false;
            setOptions(options);
        }
        std::cout << std::format("PMMM FMM-near n={:d} order={:d} loose={:e} strict={:e} "
                                 "repartitioned={:e} CPU accepted={:d} direct={:d}\n",
                         side, gravity::kBuildFmmOrder, looseError, strictError, repartitionError,
                         static_cast<unsigned long long>(looseCounts.nearCpuAcceptedMultipoles),
                         static_cast<unsigned long long>(strictCounts.nearCpuDirectPairs))
                  << std::flush;
    }
    pmmmDestroy();
    setOptions(saved);
}

}    // namespace detail

namespace detail {

void checkPmmmArbitraryMeshes() {
    auto const saved = getOptions();
    auto options = saved;
    options.gravity.pmmmCoarseMeshSize = 0;
    options.gravity.openingAngle = 0.5773502691896257;
    for (int side : {1, 2, 3, 5, 6, 7, 11, 12}) {
        if (side < pmmm::detail::minimumCoarseMeshSizeForLocalities(hpxSize())) {
            continue;
        }
        pmmmDestroy();
        options.gravity.meshSize = side;
        setOptions(options);
        // Mixed masses and positions span cells and the periodic seam. Compare
        // against the independently contracted finite expansion: very coarse
        // meshes need not satisfy the old fixtures' Ewald truncation budgets.
        for (int fixture : {0, 1, 4}) {
            detail::preparePmmmFixture(fixture);
            detail::distributePmmmFixture(fixture);
            pmmmSolve(true);
            auto const samples = detail::gatherPmmmSamples();
            for (size_t sink = 0; sink < samples.size(); ++sink) {
                auto const expected = detail::directPmmmExpansionReference(sink, samples);
                for (int component = 0; component < 4; ++component) {
                    double const actual = component == 0 ?
                        samples[sink].potential :
                        samples[sink].acceleration[component - 1];
                    double const allowance = component == 0 ?
                        4 * std::numeric_limits<float>::epsilon() *
                            std::abs(options.gravity.gm * samples[sink].mass *
                                selectedGravityKernel().centralPotential /
                                options.gravity.smoothingLength) :
                        0;
                    detail::requirePmmm(std::isfinite(actual) &&
                            std::abs(actual - expected[component]) <
                                2.0e-5 * std::max(1.0, std::abs(expected[component])) + allowance,
                        "arbitrary mesh differs from finite expansion: side=" +
                            std::to_string(side) + " component=" + std::to_string(component));
                }
            }
            pmmmSolve(false);
            detail::comparePmmmSamples(samples, detail::gatherPmmmSamples(), false);
        }
    }
    pmmmDestroy();
    setOptions(saved);
}

}    // namespace detail

void pmmmTest() {
    detail::checkSourceMomentSerialization();
    detail::checkGhostMassPermutation();
    auto options = getOptions();
    // Keep the legacy power-of-two Ewald force oracles as well as the
    // arbitrary-side numerical matrix and default coarse mesh (zero).
    options.gravity.pmmmCoarseMeshSize = 0;
    // Keep the live kick fixtures collisionless while mesh/SPH coupling is deferred.
    options.hydro.enabled = false;
    options.gravity.gm = 0.125;
    options.gravity.smoothingLength = 0.001;
    options.gravity.openingAngle = 0.05;
    options.analysis.saveForce = true;
    options.analysis.tracers = false;
    options.analysis.groups = false;
    options.initial.darkMatterMass = 0.75;
    options.initial.gasMass = 0.25;
    options.run.scenario = "cosmology";
    options.run.hierarchicalGravity =
        usesHierarchicalIntegration(options.run) && options.gravity.enabled;
    setOptions(options);
    detail::checkPmmmEndpointReuse();
    detail::checkPmmmFrozenFmm();
    if (options.run.test == "pmmmSources") {
        // Focus the changed source pipeline independently of the full-field
        // Ewald/finite-expansion suite, which remains in run.test=pmmm.
        for (int side : {4, 8, 16}) {
            pmmmDestroy();
            options.gravity.meshSize = side;
            options.gravity.openingAngle = 0.8660254037844386;
            setOptions(options);
            for (int fixture : {0, 1, 2, 3, 4, 7}) {
                detail::preparePmmmFixture(fixture);
                detail::distributePmmmFixture(fixture);
                pmmmSolve(true);
                auto const reference = detail::gatherPmmmSamples();
                pmmmSolve(false);
                detail::comparePmmmSamples(reference, detail::gatherPmmmSamples(), false);
            }
        }
        pmmmDestroy();
        detail::checkPmmmFmmNear();
        detail::checkRootDepositionRecovery();
        detail::checkPmmmFarOverlap();
        detail::checkPmmmCellOverlap();
        pmmmDestroy();
        std::cout << "PMMM owned-root source pipeline matrix passed\n" << std::flush;
        return;
    }
    detail::checkPmmmArbitraryMeshes();
    // Exercise several mesh sizes and angles, including lattice boundaries.
    std::array<std::array<int, 2>, 5> const configurations{
        {{8, 2}, {8, 1}, {16, 2}, {8, 3}, {16, 4}}};
    for (auto const configuration : configurations) {
        pmmmDestroy();
        options.gravity.meshSize = configuration[0];
        options.gravity.openingAngle = std::sqrt(3.0) / (configuration[1] + 1);
        setOptions(options);
        for (int fixture : {0, 1, 2, 3, 4, 7}) {
            detail::preparePmmmFixture(fixture);
            detail::distributePmmmFixture(fixture);
            resetFlops();
            pmmmSolve(true);
            FlopCount const firstFlops = readFlops();
            detail::requirePmmm(firstFlops > 0, "PMMM solve did not submit arithmetic counts");
            auto const baseline = detail::gatherPmmmSamples();
            if (fixture == 7 && hpxSize() > 1) {
                // Domains cut only on coarse-mesh planes, so the one occupied
                // cell of the Ewald fixture must stay whole on a single rank.
                for (auto const& sample : baseline) {
                    detail::requirePmmm(sample.cellKey == baseline.front().cellKey,
                        "single-cell Ewald fixture spans more than one mesh cell");
                    detail::requirePmmm(sample.rank == baseline.front().rank,
                        "coarse-aligned PMMM domains split the Ewald fixture's mesh cell across "
                        "localities");
                }
            }
            double const error = detail::verifyPmmmReference(baseline);
            detail::distributePmmmFixture(fixture);
            resetFlops();
            pmmmSolve(true);
            FlopCount const cachedFlops = readFlops();
            detail::requirePmmm(cachedFlops > 0, "cached PMMM solve did not count its work");
            if (fixture == 0) {
                detail::requirePmmm(firstFlops > cachedFlops,
                    "cached PMMM solve still charged Green initialization");
            }
            detail::comparePmmmSamples(baseline, detail::gatherPmmmSamples());
            auto const cached = pmmmDiagnostics();
            resetFlops();
            pmmmSolve(false);
            FlopCount const forceFlops = readFlops();
            detail::requirePmmm(forceFlops > 0, "force-only PMMM solve did not count its work");
            if (fixture == 2) {
                detail::requirePmmm(
                    forceFlops == cachedFlops, "empty PMMM solve invented potential work");
                // With no particles, every remaining operation is mesh work.
                // This also detects charging FFTW's global MPI count per rank.
                auto const& mesh = pmmm::detail::cache;
                FlopCount const n = options.gravity.meshSize;
                detail::requirePmmm(mesh.forwardFlops > 0 && mesh.inverseFlops > 0,
                    "FFTW plan arithmetic was not cached on the root locality");
                // Source packing narrows every real coefficient; each Fourier term promotes
                // four inputs and each output coefficient narrows both complex components.
                FlopCount const meshConversions = n * n * n * mesh.sourceCount +
                    n * n * (n / 2 + 1) * (4 * mesh.terms.size() + 2 * mesh.localCount);
                FlopCount const expected = mesh.forwardFlops + mesh.inverseFlops + 4 * n * n * n +
                    10 * n * n * (n / 2 + 1) * mesh.terms.size() + meshConversions +
                    5 * FlopCount(hpxSize() - 1);
                detail::requirePmmm(forceFlops == expected,
                    "empty PMMM mesh operation total is incomplete or counted more than once: " +
                        std::to_string(forceFlops) + " expected " + std::to_string(expected));
            }
            detail::comparePmmmSamples(baseline, detail::gatherPmmmSamples(), false);
            pmmmSolve(true);
            detail::comparePmmmSamples(baseline, detail::gatherPmmmSamples());
            auto const repeated = pmmmDiagnostics();
            detail::requirePmmm(repeated.planBuilds == cached.planBuilds &&
                    repeated.greenBuilds == cached.greenBuilds &&
                    repeated.solveCount == cached.solveCount + 2,
                "repeated force/potential solves rebuilt cached plans or Green spectra");
            if (fixture == 0) {
                detail::preparePmmmFixture(fixture, 1);
                detail::distributePmmmFixture(fixture);
                pmmmSolve(true);
                detail::comparePmmmSamples(baseline, detail::gatherPmmmSamples());
                auto const beforeDestroy = pmmmDiagnostics();
                pmmmDestroy();
                pmmmSolve(true);
                detail::comparePmmmSamples(baseline, detail::gatherPmmmSamples());
                auto const recreated = pmmmDiagnostics();
                detail::requirePmmm(recreated.planBuilds == beforeDestroy.planBuilds + 1 &&
                        recreated.greenBuilds == beforeDestroy.greenBuilds + 1,
                    "destroy/recreate did not rebuild exactly one cached plan/Green set");
            }
            std::cout
                << std::format(
                       "PMMM Ewald oracle n={:d} order={:d} theta={:e} fixture={:d} error={:e}\n",
                       configuration[0], kFmmOrder, options.gravity.openingAngle, fixture, error)
                << std::flush;
        }
    }
    pmmmDestroy();
    detail::checkPmmmFmmNear();
    detail::checkPmmmTraditionalKicks();
    detail::checkPmmmStreamedStages();
    detail::checkPmmmStageMeshes();
    detail::checkPmmmCoarseMeshes();
    detail::checkRootDepositionRecovery();
    detail::checkPmmmFarOverlap();
    detail::checkPmmmCellOverlap();
    std::cout << std::format("PMMM periodic force, mass, self, ownership, and cache checks passed "
                             "at order {:d} with {:s} near interactions\n",
                     kFmmOrder, "fmm")
              << std::flush;
}
