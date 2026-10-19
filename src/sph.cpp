/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <fenv.h>
#include <memory>
#include <stack>
#include <vector>

#include <unistd.h>

#include <cosmictiger/constants.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/kernel.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_cpu.hpp>
#ifdef USE_CUDA
#include <cosmictiger/sph_cuda.hpp>
#endif
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/stack_trace.hpp>
#include <cosmictiger/timer.hpp>

HPX_PLAIN_ACTION(sphTreeNeighbor, SphTreeNeighborAction);

struct SphTreeIdHash {
    inline size_t operator()(TreeId id) const {
        int const i = id.index;
        return i * hpxSize() + id.proc;
    }
};

inline bool rangeIntersect(Fixed32Range const& a, Fixed32Range const& b) {
    return a.periodicIntersects(b);
}

struct SphRunWorkspace {
    SphRunParameters params;
    Vector<float, PinnedAllocator<float>> hostVelocityX;
    Vector<float, PinnedAllocator<float>> hostVelocityY;
    Vector<float, PinnedAllocator<float>> hostVelocityZ;
    Vector<float, PinnedAllocator<float>> hostInternalEnergy;
    Vector<float, PinnedAllocator<float>> hostPressureEnergyRatio;
    Vector<float, PinnedAllocator<float>> hostAdiabaticIndex;
    Vector<float, PinnedAllocator<float>> hostDensity;
    Vector<float, PinnedAllocator<float>> hostSupportRadius;
    Vector<char, PinnedAllocator<char>> hostRungs;
    Vector<SphTreeNode, PinnedAllocator<SphTreeNode>> hostTrees;
    Vector<Fixed32, PinnedAllocator<Fixed32>> hostPositionX;
    Vector<Fixed32, PinnedAllocator<Fixed32>> hostPositionY;
    Vector<Fixed32, PinnedAllocator<Fixed32>> hostPositionZ;
    Vector<int, PinnedAllocator<int>> hostNeighborNodeIndices;
    Vector<int> hostTargetNodeIndices;
    std::unordered_map<TreeId, int, SphTreeIdHash> treeMap;
    std::unordered_map<int, Pair<int>> neighborRanges;
    MutexType mutex;
    void addWork(TreeId selfid);
    SphRunReturn run(bool useCuda);
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
            selfPtr->particleRange.second - selfPtr->particleRange.first > MIN_KICK_THREAD_PARTS;
        if (threadme) {
            if (numberOfThreads++ < KICK_OVERSUBSCRIPTION * hpx::thread::hardware_concurrency() ||
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
        rc = hpx::async(
            [self, level, params](Vector<TreeId> checklist) {
                auto rc = sphTreeNeighbor(params, self, std::move(checklist), level + 1);
                numberOfThreads--;
                return rc;
            },
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

hpx::future<SphTreeNeighborReturn> sphTreeNeighbor(
    SphTreeNeighborParameters params, TreeId self, Vector<TreeId> checklist, int level) {
    if (level == 0 && hpxRank() == 0 && hpxSize() > 1) {
        sphTreeCacheClear();
    }
    ALWAYS_ASSERT(params.innerSelection);
    SphTreeNeighborReturn kr;
    stackTraceActivate();
    SphTreeNode const* selfPtr = sphTreeGetNode(self);
    if (params.runType == SPH_TREE_NEIGHBOR_NEIGHBORS && selfPtr->localRoot) {
        sphTreeFreeNeighborList();
        sphTreeClearNeighborRanges();
    }
    if (params.runType == SPH_TREE_NEIGHBOR_NEIGHBORS && checklist.size() == 0) {
        return hpx::make_ready_future(kr);
    }
    ASSERT(self.proc == hpxRank());
    Vector<TreeId> nextlist;
    Vector<TreeId> leaflist;
    if ((params.innerSelection | SPH_INTERACTIONS_I) && selfPtr->numberOfActiveParticles) {
        if (!rangeIntersect(selfPtr->outerBox, selfPtr->innerBox)) {
            for (int dim = 0; dim < NDIM; dim++) {
                PRINT("%e %e\n", selfPtr->outerBox.begin[dim].toDouble(),
                    selfPtr->outerBox.end[dim].toDouble());
                PRINT("%e %e\n", selfPtr->innerBox.begin[dim].toDouble(),
                    selfPtr->innerBox.end[dim].toDouble());
            }
        }
    }

    if ((params.innerSelection | SPH_INTERACTIONS_J) && selfPtr->numberOfActiveParticles) {
        ALWAYS_ASSERT(rangeIntersect(selfPtr->innerBox, selfPtr->outerBox));
    }

    if (params.runType == SPH_TREE_NEIGHBOR_NEIGHBORS) {
        do {
            nextlist.resize(0);
            for (int ci = 0; ci < checklist.size(); ci++) {
                auto const* other = sphTreeGetNode(checklist[ci]);
                bool test2 = false;
                bool test1 = false;
                if (params.innerSelection | SPH_INTERACTIONS_I) {
                    test1 = rangeIntersect(selfPtr->outerBox, other->innerBox);
                }
                if (params.innerSelection | SPH_INTERACTIONS_J) {
                    test2 = rangeIntersect(selfPtr->innerBox, other->outerBox);
                }
                if (test1 || test2) {
                    if (other->leaf) {
                        leaflist.push_back(checklist[ci]);
                    } else {
                        nextlist.push_back(other->children[LEFT]);
                        nextlist.push_back(other->children[RIGHT]);
                    }
                }
            }
            checklist = std::move(nextlist);
        } while (selfPtr->leaf && checklist.size());
    }
    if (selfPtr->leaf) {
        switch (params.runType) {
        case SPH_TREE_NEIGHBOR_NEIGHBORS: {
            Pair<int> rng;
            rng.first = sphTreeAllocateNeighborList(leaflist);
            rng.second = leaflist.size() + rng.first;
            sphTreeSetNeighborRange(self, rng);
        } break;
        case SPH_TREE_NEIGHBOR_BOXES: {
            Fixed32Range ibox, obox;
            for (int dim = 0; dim < NDIM; dim++) {
                ibox.begin[dim] = obox.begin[dim] = 1.9;
                ibox.end[dim] = obox.end[dim] = -.9;
            }
            bool factive = false;
            for (ParticleInteger i = selfPtr->particleRange.first;
                i < selfPtr->particleRange.second; i++) {
                bool const active = sphParticlesRung(i) >= params.minimumRung;
                float const h = params.supportRadiusMultiplier * sphParticlesSupportRadius(i);
                auto const myx = sphParticlesPosition(XDIM, i);
                auto const myy = sphParticlesPosition(YDIM, i);
                auto const myz = sphParticlesPosition(ZDIM, i);
                Array<Fixed32, NDIM> xValue;
                xValue[XDIM] = myx;
                xValue[YDIM] = myy;
                xValue[ZDIM] = myz;
                auto const tiny = 10.0 * RangeFixed::min().toDouble();
                if (active) {
                    factive = true;
                }
                if ((params.outerSelection & SPH_SET_ALL) ||
                    (active && (params.outerSelection & SPH_SET_ACTIVE))) {
                    for (int dim = 0; dim < NDIM; dim++) {
                        double const x = xValue[dim].toDouble();
                        obox.begin[dim] = std::min(obox.begin[dim].toDouble(), x - h - tiny);
                        obox.end[dim] = std::max(obox.end[dim].toDouble(), x + h + tiny);
                    }
                }
                if ((params.innerSelection & SPH_SET_ALL) ||
                    (active && (params.innerSelection & SPH_SET_ACTIVE))) {
                    for (int dim = 0; dim < NDIM; dim++) {
                        double const x = xValue[dim].toDouble();
                        ibox.begin[dim] = std::min(ibox.begin[dim].toDouble(), x - tiny);
                        ibox.end[dim] = std::max(ibox.end[dim].toDouble(), x + tiny);
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
        return hpx::make_ready_future(kr);
    } else {
        checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
        std::array<hpx::future<SphTreeNeighborReturn>, NCHILD> futs;
        futs[RIGHT] = sphTreeNeighborFork(params, selfPtr->children[RIGHT], checklist, level, true);
        futs[LEFT] = sphTreeNeighborFork(
            params, selfPtr->children[LEFT], std::move(checklist), level, false);

        auto const finish = [self, params](hpx::future<SphTreeNeighborReturn>& fl,
                                hpx::future<SphTreeNeighborReturn>& fr) {
            SphTreeNeighborReturn kr;
            auto const rcl = fl.get();
            auto const rcr = fr.get();
            kr += rcl;
            kr += rcr;
            if (params.runType == SPH_TREE_NEIGHBOR_BOXES) {
                sphTreeSetBoxes(self, kr.innerBox, kr.outerBox);
            }
            return kr;
        };
        if (futs[LEFT].is_ready() && futs[RIGHT].is_ready()) {
            return hpx::make_ready_future(finish(futs[LEFT], futs[RIGHT]));
        } else {
            return hpx::when_all(futs.begin(), futs.end())
                .then([finish, selfPtr](
                          hpx::future<std::vector<hpx::future<SphTreeNeighborReturn>>> futsfut) {
                    auto futs = futsfut.get();
                    return finish(futs[LEFT], futs[RIGHT]);
                });
        }
    }
}

SphRunReturn sphRun(SphRunParameters params, bool cuda) {
    // Each phase consumes gas fields updated by the preceding phase on every locality.
    if (hpxRank() == 0 && hpxSize() > 1) {
        sphParticlesClearCaches();
    }
    std::string profileName = "sph_run:" + std::to_string(params.runType);
    profilerEnter(profileName.c_str());

    feenableexcept(FE_DIVBYZERO);
    feenableexcept(FE_INVALID);
    feenableexcept(FE_OVERFLOW);
    params.courantFactor = getOptions().hydro.cfl;
    if (getOptions().compute.cuda == false) {
        cuda = false;
    }
    SphRunReturn rc;
    Vector<hpx::future<SphRunReturn>> futs;
    Vector<hpx::future<SphRunReturn>> futs2;
    std::shared_ptr<SphRunWorkspace> workspacePtr = std::make_shared<SphRunWorkspace>(params);
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<SphRunAction>(c, params, cuda));
    }
    int numberOfThreads = hpxHardwareConcurrency();
    if (hpxSize() > 1) {
        numberOfThreads *= 4;
    }
    static std::atomic<int> next;
    next = 0;
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs2.push_back(hpx::async([proc, numberOfThreads, params, cuda, workspacePtr]() {
            SphRunReturn rc;
            int index = next++;
            while (index < sphTreeLeaflistSize()) {
                auto const selfid = sphTreeGetLeaf(index);
                auto const* self = sphTreeGetNode(selfid);
                bool test = false;
                switch (params.runType) {
                case SPH_RUN_RUNGS:
                    test = self->numberOfActiveParticles > 0;
                    break;

                case SPH_RUN_PREHYDRO1:
                    test = self->numberOfActiveParticles > 0 && !isConverged(self);
                    break;

                case SPH_RUN_PREHYDRO2:
                    test = hasActiveNeighbors(self);
                    test = test && !isConverged(self);
                    break;

                case SPH_RUN_HYDRO:
                    test = self->numberOfActiveParticles > 0;
                    break;
                }
                if (test) {
                    workspacePtr->addWork(selfid);
                }

                index = next++;
            }
            return rc;
        }));
    }
    for (auto& f : futs2) {
        rc += f.get();
    }
    if (!workspacePtr->hostTargetNodeIndices.empty()) {
        rc += workspacePtr->run(cuda);
    }
    for (auto& f : futs) {
        rc += f.get();
    }
    profilerExit();
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

SphRunReturn SphRunWorkspace::run(bool useCuda) {
    size_t particleStorageSize = 0;
    for (auto const& node : hostTrees) {
        particleStorageSize += node.particleRange.second - node.particleRange.first;
    }
    hostPositionX.resize(particleStorageSize);
    hostPositionY.resize(particleStorageSize);
    hostPositionZ.resize(particleStorageSize);
    if (params.runType != SPH_RUN_PREHYDRO1) {
        hostSupportRadius.resize(particleStorageSize);
    }
    if (params.runType == SPH_RUN_PREHYDRO2 || params.runType == SPH_RUN_RUNGS) {
        hostRungs.resize(particleStorageSize);
    } else if (params.runType == SPH_RUN_HYDRO) {
        hostDensity.resize(particleStorageSize);
        hostInternalEnergy.resize(particleStorageSize);
        if (getOptions().chemistry.enabled) {
            hostPressureEnergyRatio.resize(particleStorageSize);
            hostAdiabaticIndex.resize(particleStorageSize);
        }
        hostVelocityX.resize(particleStorageSize);
        hostVelocityY.resize(particleStorageSize);
        hostVelocityZ.resize(particleStorageSize);
    }
    Vector<hpx::future<void>> futs;
    int const numberOfThreads = 8 * hpxHardwareConcurrency();
    std::atomic<int> index(0);
    std::atomic<ParticleInteger> particleIndex(0);
    for (int i = 0; i < hostTargetNodeIndices.size(); i++) {
        hostTrees[hostTargetNodeIndices[i]].neighborRange = neighborRanges[i];
    }
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([&index, this, &particleIndex]() {
            int thisIndex = index++;
            while (thisIndex < hostTrees.size()) {
                auto& node = hostTrees[thisIndex];
                ParticleInteger const size = node.particleRange.second - node.particleRange.first;
                ParticleInteger const offset = (particleIndex += size) - size;
                sphParticlesGlobalReadPositions(node.globalParticleRange(), hostPositionX.data(),
                    hostPositionY.data(), hostPositionZ.data(), offset);
                if (params.runType == SPH_RUN_PREHYDRO2 || params.runType == SPH_RUN_RUNGS) {
                    sphParticlesGlobalReadInternalEnergyAndSupportRadius(
                        node.globalParticleRange(), nullptr, hostSupportRadius.data(), offset);
                    sphParticlesGlobalReadRungs(
                        node.globalParticleRange(), hostRungs.data(), offset);
                } else if (params.runType == SPH_RUN_HYDRO) {
                    sphParticlesGlobalReadInternalEnergyAndSupportRadius(node.globalParticleRange(),
                        hostInternalEnergy.data(), hostSupportRadius.data(), offset);
                    if (getOptions().chemistry.enabled) {
                        sphParticlesGlobalReadChemistryEos(node.globalParticleRange(),
                            hostPressureEnergyRatio.data(), hostAdiabaticIndex.data(), offset);
                    }
                    sphParticlesGlobalReadVelocities(node.globalParticleRange(),
                        hostVelocityX.data(), hostVelocityY.data(), hostVelocityZ.data(), offset);
                    sphParticlesGlobalReadDensity(
                        node.globalParticleRange(), hostDensity.data(), offset);
                }
                node.particleRange.first = offset;
                node.particleRange.second = offset + size;
                thisIndex = index++;
            }
        }));
    }
    for (auto& future : futs) {
        future.get();
    }
    SphRunData runData{};
    auto const copySources = [&](auto const& copy) {
        copy(runData.sourcePositionX, hostPositionX);
        copy(runData.sourcePositionY, hostPositionY);
        copy(runData.sourcePositionZ, hostPositionZ);
        copy(runData.sourceSupportRadius, hostSupportRadius);
        copy(runData.sourceRungs, hostRungs);
        copy(runData.sourceInternalEnergy, hostInternalEnergy);
        copy(runData.sourcePressureEnergyRatio, hostPressureEnergyRatio);
        copy(runData.sourceAdiabaticIndex, hostAdiabaticIndex);
        copy(runData.sourceDensity, hostDensity);
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
    runData.sinkDensity = &sphParticlesDensity(0);
    runData.sinkThermodynamicStates = &sphParticlesThermodynamicState(0);
    runData.sinkInternalEnergyRate = &sphParticlesInternalEnergyRate(0);
    runData.sinkGravityAccelerationX = &sphParticlesGravityAcceleration(XDIM, 0);
    runData.sinkGravityAccelerationY = &sphParticlesGravityAcceleration(YDIM, 0);
    runData.sinkGravityAccelerationZ = &sphParticlesGravityAcceleration(ZDIM, 0);
    runData.adiabaticIndex = getOptions().hydro.gamma;
    runData.gravityEffectiveRadius = getOptions().gravity.supportRadius /
        selectedGravityKernel().supportRadiusPerEffectiveRadius;
    runData.numberOfTargetNodes = hostTargetNodeIndices.size();
    runData.sinkSmoothingConverged = &sphParticlesConverged(0);
    runData.selfGravityEnabled = getOptions().gravity.enabled;
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
                workers.push_back(hpx::async([&]() {
                    SphRunReturn result;
                    for (int nodeIndex = nextNode++; nodeIndex < runData.numberOfTargetNodes;
                        nodeIndex = nextNode++) {
                        result += sphRunCpuNode(params, runData, nodeIndex);
                    }
                    return result;
                }));
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
