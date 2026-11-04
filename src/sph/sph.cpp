/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <format>
#include <iostream>
#include <memory>
#include <mutex>
#include <stack>
#include <type_traits>
#include <vector>

#include <hpx/threading_base/annotated_function.hpp>
#include <unistd.h>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/math/vector3.hpp>
#if COSMICTIGER_ENABLE_PMMM_PP || COSMICTIGER_ENABLE_PMMM_FMM
#include <cosmictiger/gravity/pmmm/solver.hpp>
#endif
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/sph/cpu.hpp>
#include <cosmictiger/sph/flops.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/time_integration.hpp>
#ifdef USE_CUDA
#include <cosmictiger/sph/cuda.hpp>
#endif
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/viscosity/cullen_and_dehnen_particle_exchange.hpp>
#if COSMICTIGER_ENABLE_STAR_FORMATION
#include <cosmictiger/stars/star_formation.hpp>
#endif

HPX_PLAIN_ACTION(sphTreeNeighbor, SphTreeNeighborAction);

struct SphTreeIdHash {
    inline size_t operator()(TreeId id) const {
        int const i = id.index;
        return i * hpxSize() + id.proc;
    }
};

inline bool rangeIntersect(
    Fixed32Range const& a, Fixed32Range const& b, FlopCounter<>* flops = nullptr) {
    return a.periodicIntersects(b, flops);
}

struct SphRunWorkspace {
    SphRunParameters params;
    Vector<SphPairSourceId, PinnedAllocator<SphPairSourceId>> hostSourceIds;
    Vector<SphPairUpdate, PinnedAllocator<SphPairUpdate>> hostPairUpdates;
    Vector<float, PinnedAllocator<float>> hostVelocityX;
    Vector<float, PinnedAllocator<float>> hostVelocityY;
    Vector<float, PinnedAllocator<float>> hostVelocityZ;
    Vector<float, PinnedAllocator<float>> hostInternalEnergy;
    Vector<float, PinnedAllocator<float>> hostDensity;
    Vector<float, PinnedAllocator<float>> hostGradHCorrection;
    Vector<float, PinnedAllocator<float>> hostBalsaraFactor;
    Vector<float, PinnedAllocator<float>> hostViscosityAlpha;
    Vector<float, PinnedAllocator<float>> hostAccelerationX;
    Vector<float, PinnedAllocator<float>> hostAccelerationY;
    Vector<float, PinnedAllocator<float>> hostAccelerationZ;
    Vector<float, PinnedAllocator<float>> hostVelocityDivergence;
    Vector<float, PinnedAllocator<float>> hostSupportRadius;
    Vector<char, PinnedAllocator<char>> hostRungs;
    Vector<SphTreeNode, PinnedAllocator<SphTreeNode>> hostTrees;
    Vector<Position, PinnedAllocator<Position>> hostPositionX;
    Vector<Position, PinnedAllocator<Position>> hostPositionY;
    Vector<Position, PinnedAllocator<Position>> hostPositionZ;
    Vector<int, PinnedAllocator<int>> hostNeighborNodeIndices;
    Vector<int> hostTargetNodeIndices;
    std::unordered_map<TreeId, int, SphTreeIdHash> treeMap;
    std::unordered_map<int, Pair<int>> neighborRanges;
    MutexType mutex;
    template <class Copy>
    void copySources(SphRunData& runData, Copy&& copy) {
        copy(runData.sourceIds, hostSourceIds);
        copy(runData.sourcePairUpdates, hostPairUpdates);
        copy(runData.sourcePositionX, hostPositionX);
        copy(runData.sourcePositionY, hostPositionY);
        copy(runData.sourcePositionZ, hostPositionZ);
        copy(runData.sourceSupportRadius, hostSupportRadius);
        copy(runData.sourceRungs, hostRungs);
        copy(runData.sourceInternalEnergy, hostInternalEnergy);
        copy(runData.sourceDensity, hostDensity);
        copy(runData.sourceGradHCorrection, hostGradHCorrection);
        copy(runData.sourceBalsaraFactor, hostBalsaraFactor);
        copy(runData.sourceBulkViscosityCoefficient, hostViscosityAlpha);
        copy(runData.sourceAccelerationX, hostAccelerationX);
        copy(runData.sourceAccelerationY, hostAccelerationY);
        copy(runData.sourceAccelerationZ, hostAccelerationZ);
        copy(runData.sourceVelocityDivergence, hostVelocityDivergence);
        copy(runData.sourceVelocityX, hostVelocityX);
        copy(runData.sourceVelocityY, hostVelocityY);
        copy(runData.sourceVelocityZ, hostVelocityZ);
        copy(runData.treeNodes, hostTrees);
        copy(runData.targetNodeIndices, hostTargetNodeIndices);
        copy(runData.neighborNodeIndices, hostNeighborNodeIndices);
    }
    SphRunData data();
    void addWork(TreeId selfid);
    // stage() gathers every source, including remote cache lines; compute() runs the
    // kernels, which update sink arrays that remote localities may still be staging.
    void stage();
    void deliverPairUpdates();
    SphRunReturn compute(bool useCuda);
    SphRunReturn run(bool useCuda) {
        stage();
        return compute(useCuda);
    }
    SphRunWorkspace(SphRunParameters p) {
        params = p;
    }
};

hpx::future<SphTreeNeighborReturn> sphTreeNeighborFork(SphTreeNeighborParameters params,
    TreeId self, Vector<TreeId> checklist, int level, bool threadme) {
    static std::atomic<int> numberOfThreads(0);
    hpx::future<SphTreeNeighborReturn> rc;
    SphTreeNode const* selfPtr = sphTreeGetNode(self);
    bool remote = false;
    bool allLocal = true;
    for (auto const& i : checklist) {
        if (i.proc != hpxRank()) {
            allLocal = false;
            break;
        }
    }
    if (self.proc != hpxRank()) {
        threadme = true;
        remote = true;
    } else if (threadme) {
        threadme =
            selfPtr->particleRange.second - selfPtr->particleRange.first > kMinKickThreadParts;
        if (threadme) {
            if (numberOfThreads++ < kKickOversubscription * hpx::thread::hardware_concurrency() ||
                !selfPtr->isLocal()) {
                threadme = true;
            } else {
                threadme = false;
                numberOfThreads--;
            }
        }
    }
    if (!threadme) {
        if (allLocal) {
            hpxYield();
        }
        rc = sphTreeNeighbor(params, self, std::move(checklist), level + 1);
    } else if (remote) {
        rc = hpx::async<SphTreeNeighborAction>(hpxLocalities()[selfPtr->processRange.first], params,
            self, std::move(checklist), level + 1);
    } else {
        rc = hpx::async(hpx::annotated_function(
                            [self, level, params](Vector<TreeId> checklist) {
                                auto rc =
                                    sphTreeNeighbor(params, self, std::move(checklist), level + 1);
                                numberOfThreads--;
                                return rc;
                            },
                            "sph.neighbors.walk"),
            std::move(checklist));
    }
    return rc;
}
HPX_PLAIN_ACTION(sphRun, SphRunAction);

bool isConverged(SphTreeNode const* self) {
    bool converged = true;
    for (int i = self->particleRange.first; i < self->particleRange.second; i++) {
        if (!sphParticlesConverged(i)) {
            converged = false;
            break;
        }
    }
    return converged;
}

static bool hasActiveNeighbors(SphTreeNode const* self) {
    bool rc = false;
    for (int i = self->neighborRange.first; i < self->neighborRange.second; i++) {
        auto const id = sphTreeGetNeighbor(i);
        if (sphTreeGetNode(id)->numberOfActiveParticles > 0) {
            rc = true;
            break;
        }
    }
    return rc;
}

void sphClearNeighborState();
HPX_PLAIN_ACTION(sphClearNeighborState, SphClearNeighborStateAction);
void sphClearNeighborState() {
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<SphClearNeighborStateAction>(child));
    }
    sphTreeFreeNeighborList();
    sphTreeClearNeighborRanges();
    hpx::wait_all(futures.begin(), futures.end());
}

hpx::future<SphTreeNeighborReturn> sphTreeNeighbor(
    SphTreeNeighborParameters params, TreeId self, Vector<TreeId> checklist, int level) {
    if (level == 0 && hpxRank() == 0 && hpxSize() > 1) {
        sphTreeCacheClear();
    }
    ALWAYS_ASSERT(params.innerSelection);
    SphTreeNeighborReturn kr;
    sph_flops::Counter flops;
    flops.doublePrecision +=
        4 * kDimensions;    // Sentinel bound conversions in the return constructor.
    SphTreeNode const* selfPtr = sphTreeGetNode(self);
    if (params.runType == kSphTreeNeighborNeighbors && level == 0) {
        sphClearNeighborState();
    }
    if (params.runType == kSphTreeNeighborNeighbors && checklist.size() == 0) {
        addCpuFlops(flops.counts());
        return hpx::make_ready_future(kr);
    }
    ASSERT(self.proc == hpxRank());
    Vector<TreeId> nextlist;
    Vector<TreeId> leaflist;
    if ((params.innerSelection & kSphInteractionsI) && selfPtr->numberOfActiveParticles) {
        if (!rangeIntersect(selfPtr->outerBox, selfPtr->innerBox)) {
            for (int dim = 0; dim < kDimensions; dim++) {
                std::cout << std::format("{:e} {:e}\n", selfPtr->outerBox.begin[dim].toDouble(),
                                 selfPtr->outerBox.end[dim].toDouble())
                          << std::flush;
                std::cout << std::format("{:e} {:e}\n", selfPtr->innerBox.begin[dim].toDouble(),
                                 selfPtr->innerBox.end[dim].toDouble())
                          << std::flush;
            }
        }
    }

    if ((params.innerSelection & kSphInteractionsJ) && selfPtr->numberOfActiveParticles) {
        ALWAYS_ASSERT(rangeIntersect(selfPtr->innerBox, selfPtr->outerBox));
    }

    if (params.runType == kSphTreeNeighborNeighbors) {
        do {
            nextlist.resize(0);
            for (int ci = 0; ci < checklist.size(); ci++) {
                auto const* other = sphTreeGetNode(checklist[ci]);
                bool test2 = false;
                bool test1 = false;
                if (params.innerSelection & kSphInteractionsI) {
                    test1 =
                        rangeIntersect(selfPtr->outerBox, other->innerBox, &flops.doublePrecision);
                }
                if (params.innerSelection & kSphInteractionsJ) {
                    test2 =
                        rangeIntersect(selfPtr->innerBox, other->outerBox, &flops.doublePrecision);
                }
                if (test1 || test2) {
                    if (other->leaf) {
                        leaflist.push_back(checklist[ci]);
                    } else {
                        for (auto const& child : other->children) {
                            if (child.index >= 0) {
                                nextlist.push_back(child);
                            }
                        }
                    }
                }
            }
            checklist = std::move(nextlist);
        } while (selfPtr->leaf && checklist.size());
    }
    if (selfPtr->leaf) {
        switch (params.runType) {
        case kSphTreeNeighborNeighbors: {
            Pair<int> rng;
            rng.first = sphTreeAllocateNeighborList(leaflist);
            rng.second = leaflist.size() + rng.first;
            sphTreeSetNeighborRange(self, rng);
        } break;
        case kSphTreeNeighborBoxes: {
            auto ibox = emptySphBounds();
            auto obox = emptySphBounds();
            flops.doublePrecision += 4 * kDimensions;    // Empty-bound conversions.
            bool factive = false;
            for (ParticleInteger i = selfPtr->particleRange.first;
                i < selfPtr->particleRange.second; i++) {
                bool const active = sphParticlesRung(i) >= params.minimumRung;
                float const h = params.supportRadiusMultiplier * sphParticlesSupportRadius(i);
                auto const myx = sphParticlesPosition(kXDimension, i);
                auto const myy = sphParticlesPosition(kYDimension, i);
                auto const myz = sphParticlesPosition(kZDimension, i);
                Vector3<Position> xValue;
                xValue[kXDimension] = myx;
                xValue[kYDimension] = myy;
                xValue[kZDimension] = myz;
                auto const tiny = 10.0 * RangeFixed::min().toDouble();
                flops.singlePrecision += 1;    // Support multiplier.
                flops.doublePrecision += 2;    // Tiny-offset construction.
                if (active) {
                    factive = true;
                }
                if ((params.outerSelection & kSphSetAll) ||
                    (active && (params.outerSelection & kSphSetActive))) {
                    for (int dim = 0; dim < kDimensions; dim++) {
                        double const position = xValue[dim].toDouble();
                        obox.begin[dim] = std::min(obox.begin[dim].toDouble(), position - h - tiny);
                        obox.end[dim] = std::max(obox.end[dim].toDouble(), position + h + tiny);
                        flops.doublePrecision += 11;    // Includes both support promotions.
                    }
                }
                if ((params.innerSelection & kSphSetAll) ||
                    (active && (params.innerSelection & kSphSetActive))) {
                    for (int dim = 0; dim < kDimensions; dim++) {
                        double const position = xValue[dim].toDouble();
                        ibox.begin[dim] = std::min(ibox.begin[dim].toDouble(), position - tiny);
                        ibox.end[dim] = std::max(ibox.end[dim].toDouble(), position + tiny);
                        flops.doublePrecision += 7;
                    }
                }
            }
            if (selfPtr->numberOfActiveParticles) {
                ALWAYS_ASSERT(factive);
            }
            kr.innerBox = ibox;
            kr.outerBox = obox;
            sphTreeSetBoxes(self, kr.innerBox, kr.outerBox);
        } break;
        }
        addCpuFlops(flops.counts());
        return hpx::make_ready_future(kr);
    } else {
        addCpuFlops(flops.counts());
        checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
        std::vector<hpx::future<SphTreeNeighborReturn>> futures;
        for (int childIndex = kSphChildCount - 1; childIndex >= 0; --childIndex) {
            TreeId const child = selfPtr->children[childIndex];
            if (child.index >= 0) {
                futures.push_back(sphTreeNeighborFork(params, child, checklist, level, true));
            }
        }
        auto const finish = [self, params](
                                std::vector<hpx::future<SphTreeNeighborReturn>>& children) {
            SphTreeNeighborReturn result;
            addCpuFlops(4 * kDimensions, FlopPrecision::Double);
            for (auto& child : children) {
                result += child.get();
            }
            if (params.runType == kSphTreeNeighborBoxes) {
                sphTreeSetBoxes(self, result.innerBox, result.outerBox);
            }
            return result;
        };
        bool ready = true;
        for (auto const& child : futures) {
            ready = ready && child.is_ready();
        }
        if (ready) {
            return hpx::make_ready_future(finish(futures));
        }
        return hpx::when_all(std::move(futures))
            .then(hpx::annotated_function(
                [finish](hpx::future<std::vector<hpx::future<SphTreeNeighborReturn>>> completed) {
                    auto children = completed.get();
                    return finish(children);
                },
                "sph.neighbors.combineChildren"));
    }
}

namespace detail {

// Keep event storage and captured stack variables alive through every launched
// operation, including when launching later work or retrieving a result throws.
// Reserve the future vector before submitting work so insertion cannot orphan
// an already launched operation through an allocation failure.
template <class Future>
class SphRunFutureDrain {
    Vector<Future>& futures_;

public:
    explicit SphRunFutureDrain(Vector<Future>& futures)
      : futures_(futures) {}

    ~SphRunFutureDrain() {
        for (auto& future : futures_) {
            if (future.valid()) {
                future.wait();
            }
        }
    }
};

// Collect this locality's target leaves for the pass into a fresh workspace.
static std::shared_ptr<SphRunWorkspace> collectSphRunWork(SphRunParameters const& params) {
    auto workspacePtr = std::make_shared<SphRunWorkspace>(params);
    int numberOfThreads = hpxHardwareConcurrency();
    if (hpxSize() > 1) {
        numberOfThreads *= 4;
    }
    static std::atomic<int> next;
    next = 0;
    Vector<hpx::future<void>> futures;
    futures.reserve(numberOfThreads);
    SphRunFutureDrain drain(futures);
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futures.push_back(hpx::async(hpx::annotated_function(
            [params, workspacePtr]() {
                int index = next++;
                while (index < sphTreeLeaflistSize()) {
                    auto const selfid = sphTreeGetLeaf(index);
                    auto const* self = sphTreeGetNode(selfid);
                    bool test = false;
                    switch (params.runType) {
                    case kSphRunRungs:
                        test = self->numberOfParticles() > 0;
                        break;

                    case kSphRunPrehydro1:
                        test = self->numberOfActiveParticles > 0 && !isConverged(self);
                        break;

                    case kSphRunPrehydro2:
                        test = hasActiveNeighbors(self);
                        test = test && (params.refreshConvergedDensity || !isConverged(self));
                        break;

                    case kSphRunHydro:
                        test = self->numberOfActiveParticles > 0;
                        break;
                    }
                    if (test) {
                        workspacePtr->addWork(selfid);
                    }

                    index = next++;
                }
            },
            "sph.run.collectLeaves")));
    }
    for (auto& future : futures) {
        future.get();
    }
    return workspacePtr;
}

static SphRunParameters prepareSphRunParameters(SphRunParameters params, bool& cuda) {
    ALWAYS_ASSERT(params.minimumRung >= 0 && params.maximumRung < kMaxRung &&
        params.minimumRung <= params.maximumRung);
    params.courantFactor = getOptions().hydro.cfl;
    if (getOptions().hydro.cuda == false) {
        cuda = false;
    }
    return params;
}

// Every locality finishes reading a pass's source state before any locality writes
// its sinks. Density/support solves and rung limiting also read fields they update.
static std::shared_ptr<SphRunWorkspace> stagedSphRun;

}    // namespace detail

// Only event-local increments cross ranks. Receive all messages before committing
// any float particle state; every pair used the same immutable source snapshot.
struct SphOwnedPairUpdate {
    ParticleInteger index;
    SphPairUpdate update;
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & index & update;
    }
};
namespace detail {
static MutexType pairUpdateMutex;
static Vector<SphOwnedPairUpdate> pendingPairUpdates;
}    // namespace detail
void sphReceivePairUpdates(Vector<SphOwnedPairUpdate> updates) {
    std::lock_guard<MutexType> lock(detail::pairUpdateMutex);
    detail::pendingPairUpdates.insert(
        detail::pendingPairUpdates.end(), updates.begin(), updates.end());
}
HPX_PLAIN_ACTION(sphReceivePairUpdates, SphReceivePairUpdatesAction);
void sphCommitPairUpdates();
HPX_PLAIN_ACTION(sphCommitPairUpdates, SphCommitPairUpdatesAction);
void sphCommitPairUpdates() {
    Vector<hpx::future<void>> children;
    for (auto const& child : hpxChildren()) {
        children.push_back(hpx::async<SphCommitPairUpdatesAction>(child));
    }
    auto& updates = detail::pendingPairUpdates;
    std::sort(updates.begin(), updates.end(),
        [](auto const& a, auto const& b) { return a.index < b.index; });
    for (size_t first = 0; first < updates.size();) {
        ParticleInteger const gas = updates[first].index;
        ALWAYS_ASSERT(gas >= 0 && gas < sphParticlesSize());
        SphPairUpdate total;
        size_t next = first;
        do {
            for (int dimension = 0; dimension < kDimensions; ++dimension) {
                total.velocity[dimension] += updates[next].update.velocity[dimension];
            }
            total.entropy += updates[next].update.entropy;
            ++next;
        } while (next < updates.size() && updates[next].index == gas);
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            sphParticlesVelocity(dimension, gas) += total.velocity[dimension];
        }
        sphParticlesEntropy(gas) += total.entropy;
        ALWAYS_ASSERT(std::isfinite(sphParticlesEntropy(gas)) && sphParticlesEntropy(gas) > 0.0f);
        first = next;
    }
    Vector<SphOwnedPairUpdate>().swap(updates);
    for (auto& child : children) {
        child.get();
    }
}

void SphRunWorkspace::deliverPairUpdates() {
    if (hostPairUpdates.empty()) {
        return;
    }
    Vector<Vector<SphOwnedPairUpdate>> destinations(hpxSize());
    for (size_t i = 0; i < hostPairUpdates.size(); ++i) {
        auto const& value = hostPairUpdates[i];
        bool nonzero = value.entropy != 0.0;
        for (int dimension = 0; dimension < kDimensions; ++dimension) {
            nonzero = nonzero || value.velocity[dimension] != 0.0;
        }
        if (nonzero) {
            auto const& owner = hostSourceIds[i];
            destinations[owner.rank].push_back({owner.index, value});
        }
    }
    Vector<hpx::future<void>> deliveries;
    for (int rank = 0; rank < hpxSize(); ++rank) {
        if (destinations[rank].empty()) {
            continue;
        }
        if (rank == hpxRank()) {
            sphReceivePairUpdates(std::move(destinations[rank]));
        } else {
            deliveries.push_back(hpx::async<SphReceivePairUpdatesAction>(
                hpxLocalities()[rank], std::move(destinations[rank])));
        }
    }
    for (auto& delivery : deliveries) {
        delivery.get();
    }
}

void sphRunStage(SphRunParameters params, bool cuda);
SphRunReturn sphRunCompute(bool cuda);
HPX_PLAIN_ACTION(sphRunStage, SphRunStageAction);
HPX_PLAIN_ACTION(sphRunCompute, SphRunComputeAction);

void sphRunStage(SphRunParameters params, bool cuda) {
    params = detail::prepareSphRunParameters(params, cuda);
    Vector<hpx::future<void>> children;
    children.reserve(hpxChildren().size());
    detail::SphRunFutureDrain drain(children);
    for (auto& c : hpxChildren()) {
        children.push_back(hpx::async<SphRunStageAction>(c, params, cuda));
    }
    auto workspacePtr = detail::collectSphRunWork(params);
    if (!workspacePtr->hostTargetNodeIndices.empty()) {
        workspacePtr->stage();
    }
    detail::stagedSphRun = std::move(workspacePtr);
    for (auto& f : children) {
        f.get();
    }
}

SphRunReturn sphRunCompute(bool cuda) {
    if (getOptions().hydro.cuda == false) {
        cuda = false;
    }
    Vector<hpx::future<SphRunReturn>> children;
    children.reserve(hpxChildren().size());
    detail::SphRunFutureDrain drain(children);
    for (auto& c : hpxChildren()) {
        children.push_back(hpx::async<SphRunComputeAction>(c, cuda));
    }
    SphRunReturn rc;
    auto workspacePtr = std::move(detail::stagedSphRun);
    ALWAYS_ASSERT(workspacePtr != nullptr);
    if (!workspacePtr->hostTargetNodeIndices.empty()) {
        rc += workspacePtr->compute(cuda);
        workspacePtr->deliverPairUpdates();
    }
    for (auto& f : children) {
        rc += f.get();
    }
    return rc;
}

SphRunReturn sphRun(SphRunParameters params, bool cuda) {
    if (hpxRank() == 0) {
        // Complete this collective before remote sources can be staged.
        sphParticlesPrepareThermodynamics(params.scaleFactor, params.runType == kSphRunHydro);
    }
    // Each phase consumes gas fields updated by the preceding phase on every locality.
    if (hpxRank() == 0 && (hpxSize() > 1 || params.midpointHydro)) {
        sphParticlesClearCaches();
        sphRunStage(params, cuda);
        auto result = sphRunCompute(cuda);
        if (params.midpointHydro && params.runType == kSphRunHydro) {
            sphCommitPairUpdates();
        }
        return result;
    }
    params = detail::prepareSphRunParameters(params, cuda);
    Vector<hpx::future<SphRunReturn>> futs;
    futs.reserve(hpxChildren().size());
    detail::SphRunFutureDrain drain(futs);
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<SphRunAction>(c, params, cuda));
    }
    SphRunReturn rc;
    auto workspacePtr = detail::collectSphRunWork(params);
    if (!workspacePtr->hostTargetNodeIndices.empty()) {
        rc += workspacePtr->run(cuda);
    }
    for (auto& f : futs) {
        rc += f.get();
    }
    return rc;
}

SphRunReturn sphMidpointForces(SphRunParameters params, bool cuda) {
    params.runType = kSphRunHydro;
    params.phase = 1;
    params.midpointHydro = true;
    params.deferAccelerationSources = false;
    // The surrounding drifts already placed sources at this split-operator
    // stage. This frozen RHS evaluation commits reciprocal impulses and viscous
    // heating once over the full positive hydro interval.
    return sphRun(params, cuda);
}

void SphRunWorkspace::addWork(TreeId selfid) {
    auto const* self = sphTreeGetNode(selfid);
    std::unique_lock<MutexType> lock(mutex);
    std::unordered_map<TreeId, int, SphTreeIdHash>::iterator iter;
    iter = treeMap.find(selfid);
    if (iter == treeMap.end()) {
        int index = hostTrees.size();
        hostTrees.resize(index + 1);
        treeMap[selfid] = index;
        hostTrees[index] = *self;
    }
    for (int i = self->neighborRange.first; i < self->neighborRange.second; i++) {
        auto const nid = sphTreeGetNeighbor(i);
        iter = treeMap.find(nid);
        if (iter == treeMap.end()) {
            int index = hostTrees.size();
            hostTrees.resize(index + 1);
            treeMap[nid] = index;
            lock.unlock();
            auto const* node = sphTreeGetNode(nid);
            lock.lock();
            hostTrees[index] = *node;
        }
    }
    int neighborBegin = hostNeighborNodeIndices.size();
    for (int i = self->neighborRange.first; i < self->neighborRange.second; i++) {
        auto const nid = sphTreeGetNeighbor(i);
        hostNeighborNodeIndices.push_back(treeMap[nid]);
    }
    int neighborEnd = hostNeighborNodeIndices.size();
    int const myindex = treeMap[selfid];
    auto& r = neighborRanges[hostTargetNodeIndices.size()];
    hostTargetNodeIndices.push_back(myindex);
    r.first = neighborBegin;
    r.second = neighborEnd;
}

void SphRunWorkspace::stage() {
    size_t particleStorageSize = 0;
    for (auto const& node : hostTrees) {
        particleStorageSize += node.particleRange.second - node.particleRange.first;
    }
    if (params.runType == kSphRunHydro && params.midpointHydro) {
        hostSourceIds.resize(particleStorageSize);
        hostPairUpdates.resize(particleStorageSize);
    }
    hostPositionX.resize(particleStorageSize);
    hostPositionY.resize(particleStorageSize);
    hostPositionZ.resize(particleStorageSize);
    if (params.runType != kSphRunPrehydro1) {
        hostSupportRadius.resize(particleStorageSize);
    }
    if (params.runType == kSphRunPrehydro2 || params.runType == kSphRunRungs ||
        (params.runType == kSphRunHydro && params.midpointHydro)) {
        hostRungs.resize(particleStorageSize);
    }
    if (params.runType == kSphRunHydro && params.balsaraSwitch) {
        hostBalsaraFactor.resize(particleStorageSize);
    }
    if (params.runType == kSphRunPrehydro2 && params.gradientSwitch()) {
        if (params.cullenDehnen) {
            // Only Cullen-Dehnen equation (17) reads the neighbors' previous divergences.
            hostVelocityDivergence.resize(particleStorageSize);
        }
    }
    if (params.runType == kSphRunHydro && params.variableViscosity) {
        hostViscosityAlpha.resize(particleStorageSize);
    }
    if (params.runType == kSphRunHydro ||
        (params.runType == kSphRunPrehydro2 &&
            (params.balsaraSwitch || params.variableViscosity))) {
        hostVelocityX.resize(particleStorageSize);
        hostVelocityY.resize(particleStorageSize);
        hostVelocityZ.resize(particleStorageSize);
    }
    if (params.runType == kSphRunHydro) {
        hostDensity.resize(particleStorageSize);
        hostGradHCorrection.resize(particleStorageSize);
    }
    if (params.runType == kSphRunHydro ||
        (params.runType == kSphRunPrehydro2 && params.variableViscosity)) {
        hostInternalEnergy.resize(particleStorageSize);
    }
    Vector<hpx::future<void>> futs;
    int const numberOfThreads = 8 * hpxHardwareConcurrency();
    std::atomic<int> index(0);
    std::atomic<ParticleInteger> particleIndex(0);
    futs.reserve(numberOfThreads);
    detail::SphRunFutureDrain drain(futs);
    for (int i = 0; i < hostTargetNodeIndices.size(); i++) {
        hostTrees[hostTargetNodeIndices[i]].neighborRange = neighborRanges[i];
    }
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async(hpx::annotated_function(
            [&index, this, &particleIndex]() {
                int thisIndex = index++;
                while (thisIndex < hostTrees.size()) {
                    auto& node = hostTrees[thisIndex];
                    ParticleInteger const size =
                        node.particleRange.second - node.particleRange.first;
                    ParticleInteger const offset = (particleIndex += size) - size;
                    if (!hostSourceIds.empty()) {
                        auto const range = node.globalParticleRange();
                        for (ParticleInteger gas = range.range.first; gas < range.range.second;
                            ++gas) {
                            hostSourceIds[offset + gas - range.range.first] = {range.proc, gas};
                        }
                    }
                    sphParticlesGlobalReadPositions(node.globalParticleRange(),
                        hostPositionX.data(), hostPositionY.data(), hostPositionZ.data(), offset);
                    if (params.runType == kSphRunPrehydro2 || params.runType == kSphRunRungs) {
                        sphParticlesGlobalReadInternalEnergyAndSupportRadius(
                            node.globalParticleRange(),
                            hostInternalEnergy.empty() ? nullptr : hostInternalEnergy.data(),
                            hostSupportRadius.data(), offset);
                        if (params.runType == kSphRunRungs && sphRungBalanceActive()) {
                            sphReadRungCandidates(
                                node.globalParticleRange(), hostRungs.data(), offset);
                        } else {
                            sphParticlesGlobalReadRungs(
                                node.globalParticleRange(), hostRungs.data(), offset);
                        }
                    } else if (params.runType == kSphRunHydro) {
                        sphParticlesGlobalReadInternalEnergyAndSupportRadius(
                            node.globalParticleRange(), hostInternalEnergy.data(),
                            hostSupportRadius.data(), offset);
                        // The correction shares the density's cache line, so a remote
                        // source's ζ is from the same density pass as its ρ.
                        sphParticlesGlobalReadDensity(node.globalParticleRange(),
                            hostDensity.data(), offset,
                            params.balsaraSwitch ? hostBalsaraFactor.data() : nullptr,
                            params.variableViscosity ? hostViscosityAlpha.data() : nullptr,
                            hostGradHCorrection.data());
                        if (params.midpointHydro) {
                            sphParticlesGlobalReadRungs(
                                node.globalParticleRange(), hostRungs.data(), offset);
                        }
                    }
                    if (params.runType == kSphRunPrehydro2 && params.gradientSwitch()) {
                        sphParticlesGlobalReadViscosityDiagnostics(node.globalParticleRange(),
                            params.cullenDehnen ? hostVelocityDivergence.data() : nullptr, offset);
                    }
                    if (!hostVelocityX.empty()) {
                        sphParticlesGlobalReadVelocities(node.globalParticleRange(),
                            hostVelocityX.data(), hostVelocityY.data(), hostVelocityZ.data(),
                            offset);
                    }
                    node.particleRange.first = offset;
                    node.particleRange.second = offset + size;
                    thisIndex = index++;
                }
            },
            "sph.run.stageParticles")));
    }
    hpx::wait_all(futs.begin(), futs.end());
    for (auto& future : futs) {
        future.get();
    }
}

SphRunData SphRunWorkspace::data() {
    SphRunData runData{};

    copySources(runData, [](auto*& destination, auto& source) {
        destination = source.empty() ? nullptr : source.data();
    });
    runData.sinkParticleIndices = &sphParticlesParticleIndex(0);
    runData.sinkSemiactive = &sphParticlesSemiactive(0);
    runData.sinkParticleRungs = &particlesRung(0);
    if (params.runType == kSphRunRungs && sphRungBalanceActive()) {
        runData.sinkRungCandidates = sphRungCandidatesData();
    }
    if (params.runType == kSphRunHydro && params.phase == 1) {
        runData.sinkProposedRungs = sphParticlesProposedRungs;
    }
    runData.sinkDensity = &sphParticlesDensity(0);
    runData.sinkGradHCorrection = sphParticlesGradHCorrections;
    runData.sinkBalsaraFactor = sphParticlesBalsaraFactors;
    runData.sinkBulkViscosityCoefficient = sphParticlesBulkViscosityCoefficients;
    runData.sinkCullenDehnenStates = sphParticlesCullenDehnenStates;
    runData.sinkViscosityDiagnostics = sphParticlesViscosityDiagnostics;
    runData.sinkSupportRadius = sphParticlesSupportRadii;
    runData.sinkEntropyRate = nullptr;
#if COSMICTIGER_WITH_GRAVITY
    runData.sinkGravityAccelerationX = nullptr;
    runData.sinkGravityAccelerationY = nullptr;
    runData.sinkGravityAccelerationZ = nullptr;
    runData.gravityEffectiveRadius = getOptions().gravity.supportRadius /
        selectedGravityKernel().supportRadiusPerEffectiveRadius;
#if COSMICTIGER_ENABLE_PMMM_PP || COSMICTIGER_ENABLE_PMMM_FMM
    if (usesMeshGravity(getOptions().gravity)) {
        runData.gravityEffectiveRadius = pmmmEffectiveRadius();
    }
#endif
    addCpuFlops(2, FlopPrecision::Double);
    runData.selfGravityEnabled = false;
#endif
    runData.adiabaticIndex = getOptions().hydro.gamma;
    runData.numberOfTargetNodes = hostTargetNodeIndices.size();
    runData.sinkSmoothingConverged = &sphParticlesConverged(0);
    runData.sinkAccelerations = nullptr;
    runData.centralAcceleration = getOptions().hydro.centralAcceleration;
    runData.centralSupportRadius = getOptions().hydro.centralSupportRadius;
    runData.targetNeighborCount = getOptions().hydro.neighborNumber;
    runData.gasMass = getOptions().initial.gasMass;
    runData.accelerationTimestepFactor = getOptions().gravity.timestepEta;

    return runData;
}

SphRunReturn SphRunWorkspace::compute(bool useCuda) {
    SphRunData runData = data();
    for (int i = 0; i < hostTargetNodeIndices.size(); i++) {
        auto const& node = hostTrees[hostTargetNodeIndices[i]];
        bool found = false;
        for (int j = node.neighborRange.first; j < node.neighborRange.second; j++) {
            if (hostNeighborNodeIndices[j] == hostTargetNodeIndices[i]) {
                found = true;
                break;
            }
        }
        ALWAYS_ASSERT(found);
    }

    if (!useCuda) {
        // A complete source snapshot is staged before any worker updates sinks.
        std::atomic<int> nextNode(0);
        Vector<hpx::future<SphRunReturn>> workers;
        int const workerCount =
            std::min(hpxHardwareConcurrency(), int(hostTargetNodeIndices.size()));
        workers.reserve(workerCount);
        try {
            for (int worker = 0; worker < workerCount; worker++) {
                workers.push_back(hpx::async(hpx::annotated_function(
                    [&]() {
                        SphRunReturn result;
                        for (int nodeIndex = nextNode++; nodeIndex < runData.numberOfTargetNodes;
                            nodeIndex = nextNode++) {
                            result += sphRunCpuNode(params, runData, nodeIndex);
                        }
                        return result;
                    },
                    params.runType == kSphRunRungs         ? "sph.cpu.rungs" :
                        params.runType == kSphRunPrehydro1 ? "sph.cpu.prehydro1" :
                        params.runType == kSphRunPrehydro2 ? "sph.cpu.prehydro2" :
                        params.runType == kSphRunHydro     ? "sph.cpu.hydro" :
                                                             "sph.cpu.unknownPhase")));
            }
        } catch (...) {
            hpx::wait_all(workers.begin(), workers.end());
            throw;
        }
        // Keep the staged arrays and referenced stack state alive even if a
        // worker reports an allocation or numerical failure through its future.
        hpx::wait_all(workers.begin(), workers.end());
        SphRunReturn result;
        for (auto& worker : workers) {
            result += worker.get();
        }
        return result;
    }
#ifdef USE_CUDA
    std::vector<std::unique_ptr<void, decltype(&cudaFree)>> sourceBuffers;
    sourceBuffers.reserve(15);
    // Declared after the source owners so failure drains submitted copies or
    // kernels before their device allocations are released.
    CudaStream stream(cudaGetDevice());
    copySources(runData, [&](auto*& device, auto const& host) {
        device = nullptr;
        if (!host.empty()) {
            size_t const bytes = sizeof(*device) * host.size();
            void* allocation = nullptr;
            CUDA_CHECK(cudaMalloc(&allocation, bytes));
            device = static_cast<std::remove_reference_t<decltype(device)>>(allocation);
            sourceBuffers.emplace_back(allocation, cudaFree);
            stream.copyAsync(allocation, host.data(), bytes, cudaMemcpyHostToDevice);
        }
    });
    auto rc = sphRunCuda(params, runData, stream);
    if (!hostPairUpdates.empty()) {
        stream.copyAsync(hostPairUpdates.data(), runData.sourcePairUpdates,
            hostPairUpdates.size() * sizeof(SphPairUpdate), cudaMemcpyDeviceToHost);
    }
    stream.wait();

    return rc;
#else
    ALWAYS_ASSERT(false);
    return {};
#endif
}
