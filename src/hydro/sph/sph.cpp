/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <memory>
#include <stack>
#include <vector>

#include <hpx/threading_base/annotated_function.hpp>
#include <unistd.h>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/math/vector3.hpp>
#if COSMICTIGER_ENABLE_PMMM_PP || COSMICTIGER_ENABLE_PMMM_FMM
#include <cosmictiger/gravity/pmmm/solver.hpp>
#endif
#include <cosmictiger/hydro/sph/sph.hpp>
#include <cosmictiger/hydro/sph/sph_cpu.hpp>
#include <cosmictiger/hydro/sph/sph_flops.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/math.hpp>
#ifdef USE_CUDA
#include <cosmictiger/hydro/sph/sph_cuda.hpp>
#endif
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/hydro/sph/cullen_and_dehnen/particle_exchange.hpp>
#include <cosmictiger/hydro/sph/sph_particles.hpp>
#if COSMICTIGER_ENABLE_STAR_FORMATION
#include <cosmictiger/hydro/sph/star_formation.hpp>
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
    void addWork(TreeId selfid);
    // stage() gathers every source, including remote cache lines; compute() runs the
    // kernels, which update sink arrays that remote localities may still be staging.
    void stage();
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
                PRINT("%e %e\n", selfPtr->outerBox.begin[dim].toDouble(),
                    selfPtr->outerBox.end[dim].toDouble());
                PRINT("%e %e\n", selfPtr->innerBox.begin[dim].toDouble(),
                    selfPtr->innerBox.end[dim].toDouble());
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
                        nextlist.push_back(other->children[kLeftChild]);
                        nextlist.push_back(other->children[kRightChild]);
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
            Fixed32Range ibox, obox;
            for (int dim = 0; dim < kDimensions; dim++) {
                ibox.begin[dim] = obox.begin[dim] = 1.9;
                ibox.end[dim] = obox.end[dim] = -.9;
            }
            flops.doublePrecision +=
                2 * kDimensions;    // Chained assignment converts each sentinel once.
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
        std::array<hpx::future<SphTreeNeighborReturn>, kChildCount> futs;
        futs[kRightChild] =
            sphTreeNeighborFork(params, selfPtr->children[kRightChild], checklist, level, true);
        futs[kLeftChild] = sphTreeNeighborFork(
            params, selfPtr->children[kLeftChild], std::move(checklist), level, false);

        auto const finish = [self, params](hpx::future<SphTreeNeighborReturn>& fl,
                                hpx::future<SphTreeNeighborReturn>& fr) {
            SphTreeNeighborReturn kr;
            addCpuFlops(4 * kDimensions, FlopPrecision::Double);
            auto const rcl = fl.get();
            auto const rcr = fr.get();
            kr += rcl;
            kr += rcr;
            if (params.runType == kSphTreeNeighborBoxes) {
                sphTreeSetBoxes(self, kr.innerBox, kr.outerBox);
            }
            return kr;
        };
        if (futs[kLeftChild].is_ready() && futs[kRightChild].is_ready()) {
            return hpx::make_ready_future(finish(futs[kLeftChild], futs[kRightChild]));
        } else {
            return hpx::when_all(futs.begin(), futs.end())
                .then(hpx::annotated_function(
                    [finish, selfPtr](
                        hpx::future<std::vector<hpx::future<SphTreeNeighborReturn>>> futsfut) {
                        auto futs = futsfut.get();
                        return finish(futs[kLeftChild], futs[kRightChild]);
                    },
                    "sph.neighbors.combineChildren"));
        }
    }
}

namespace detail {

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
                        test = self->numberOfActiveParticles > 0;
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
    params.courantFactor = getOptions().hydro.cfl;
    if (getOptions().hydro.cuda == false) {
        cuda = false;
    }
    return params;
}

// Between the predictor and the closing kick, staging predicts active gas from the stored
// rates, which the force kernels overwrite. Across localities the force pass therefore
// stages everywhere before any locality runs its kernels.
static std::shared_ptr<SphRunWorkspace> stagedSphRun;

}    // namespace detail

void sphRunStage(SphRunParameters params, bool cuda);
SphRunReturn sphRunCompute(bool cuda);
HPX_PLAIN_ACTION(sphRunStage, SphRunStageAction);
HPX_PLAIN_ACTION(sphRunCompute, SphRunComputeAction);

void sphRunStage(SphRunParameters params, bool cuda) {
    params = detail::prepareSphRunParameters(params, cuda);
    Vector<hpx::future<void>> children;
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
    for (auto& c : hpxChildren()) {
        children.push_back(hpx::async<SphRunComputeAction>(c, cuda));
    }
    SphRunReturn rc;
    auto workspacePtr = std::move(detail::stagedSphRun);
    ALWAYS_ASSERT(workspacePtr != nullptr);
    if (!workspacePtr->hostTargetNodeIndices.empty()) {
        rc += workspacePtr->compute(cuda);
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
    if (hpxRank() == 0 && hpxSize() > 1) {
        sphParticlesClearCaches();
        if (params.runType == kSphRunHydro) {
            sphRunStage(params, cuda);
            return sphRunCompute(cuda);
        }
    }
    params = detail::prepareSphRunParameters(params, cuda);
    Vector<hpx::future<SphRunReturn>> futs;
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
    hostPositionX.resize(particleStorageSize);
    hostPositionY.resize(particleStorageSize);
    hostPositionZ.resize(particleStorageSize);
    if (params.runType != kSphRunPrehydro1) {
        hostSupportRadius.resize(particleStorageSize);
    }
    if (params.runType == kSphRunPrehydro2 || params.runType == kSphRunRungs) {
        hostRungs.resize(particleStorageSize);
    }
    if (params.runType == kSphRunHydro && params.balsaraSwitch) {
        hostBalsaraFactor.resize(particleStorageSize);
    }
    if (params.runType == kSphRunPrehydro2 && params.gradientSwitch()) {
        hostAccelerationX.resize(particleStorageSize);
        hostAccelerationY.resize(particleStorageSize);
        hostAccelerationZ.resize(particleStorageSize);
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
                    sphParticlesGlobalReadPositions(node.globalParticleRange(),
                        hostPositionX.data(), hostPositionY.data(), hostPositionZ.data(), offset);
                    if (params.runType == kSphRunPrehydro2 || params.runType == kSphRunRungs) {
                        sphParticlesGlobalReadInternalEnergyAndSupportRadius(
                            node.globalParticleRange(),
                            hostInternalEnergy.empty() ? nullptr : hostInternalEnergy.data(),
                            hostSupportRadius.data(), offset);
                        sphParticlesGlobalReadRungs(
                            node.globalParticleRange(), hostRungs.data(), offset);
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
                    }
                    if (params.runType == kSphRunPrehydro2 && params.gradientSwitch()) {
                        sphParticlesGlobalReadViscosityDiagnostics(node.globalParticleRange(),
                            hostAccelerationX.data(), hostAccelerationY.data(),
                            hostAccelerationZ.data(),
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

SphRunReturn SphRunWorkspace::compute(bool useCuda) {
    SphRunData runData{};
    auto const copySources = [&](auto const& copy) {
        copy(runData.sourcePositionX, hostPositionX);
        copy(runData.sourcePositionY, hostPositionY);
        copy(runData.sourcePositionZ, hostPositionZ);
        copy(runData.sourceSupportRadius, hostSupportRadius);
        copy(runData.sourceRungs, hostRungs);
        copy(runData.sourceInternalEnergy, hostInternalEnergy);
        copy(runData.sourceDensity, hostDensity);
        copy(runData.sourceGradHCorrection, hostGradHCorrection);
        copy(runData.sourceBalsaraFactor, hostBalsaraFactor);
        copy(runData.sourceViscosityAlpha, hostViscosityAlpha);
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
    };
    copySources([](auto*& destination, auto& source) {
        destination = source.empty() ? nullptr : source.data();
    });
    runData.sinkParticleIndices = &sphParticlesParticleIndex(0);
    runData.sinkSemiactive = &sphParticlesSemiactive(0);
    runData.sinkParticleRungs = &particlesRung(0);
    if (params.runType == kSphRunHydro && params.phase == 1) {
        runData.sinkProposedRungs = sphParticlesProposedRungs;
    }
    runData.sinkDensity = &sphParticlesDensity(0);
    runData.sinkGradHCorrection = sphParticlesGradHCorrections;
    runData.sinkBalsaraFactor = sphParticlesBalsaraFactors;
    runData.sinkViscosityStates = sphParticlesViscosityStates;
    runData.sinkCullenDehnenStates = sphParticlesCullenDehnenStates;
    runData.sinkSupportRadius = sphParticlesSupportRadii;
    runData.sinkEntropyRate = sphParticlesEntropyRates;
#if COSMICTIGER_WITH_GRAVITY
    runData.sinkGravityAccelerationX = &sphParticlesGravityAcceleration(kXDimension, 0);
    runData.sinkGravityAccelerationY = &sphParticlesGravityAcceleration(kYDimension, 0);
    runData.sinkGravityAccelerationZ = &sphParticlesGravityAcceleration(kZDimension, 0);
    runData.gravityEffectiveRadius = getOptions().gravity.supportRadius /
        selectedGravityKernel().supportRadiusPerEffectiveRadius;
#if COSMICTIGER_ENABLE_PMMM_PP || COSMICTIGER_ENABLE_PMMM_FMM
    if (usesMeshGravity(getOptions().gravity)) {
        runData.gravityEffectiveRadius = pmmmEffectiveRadius();
    }
#endif
    addCpuFlops(2, FlopPrecision::Double);
    runData.selfGravityEnabled = getOptions().gravity.enabled;
#endif
    runData.adiabaticIndex = getOptions().hydro.gamma;
    runData.numberOfTargetNodes = hostTargetNodeIndices.size();
    runData.sinkSmoothingConverged = &sphParticlesConverged(0);
    runData.sinkAccelerations = &sphParticlesAccelerationState(0);
    runData.centralAcceleration = getOptions().hydro.centralAcceleration;
    runData.centralSupportRadius = getOptions().hydro.centralSupportRadius;
    runData.targetNeighborCount = getOptions().hydro.neighborNumber;
    runData.gasMass = getOptions().initial.gasMass;
    runData.accelerationTimestepFactor = getOptions().gravity.timestepEta;

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
    copySources([&](auto*& device, auto const& host) {
        device = nullptr;
        if (!host.empty()) {
            size_t const bytes = sizeof(*device) * host.size();
            CUDA_CHECK(cudaMalloc(&device, bytes));
            sourceBuffers.emplace_back(device, cudaFree);
            stream.copyAsync(device, host.data(), bytes, cudaMemcpyHostToDevice);
        }
    });
    auto rc = sphRunCuda(params, runData, stream);
    stream.wait();

    return rc;
#else
    ALWAYS_ASSERT(false);
    return {};
#endif
}
