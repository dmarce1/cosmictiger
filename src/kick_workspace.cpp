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

#include <set>

#include <cosmictiger/kick_workspace.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/timer.hpp>

HPX_PLAIN_ACTION(KickWorkspace::clearBuffers, ClearBuffersAction);

KickWorkspace::KickWorkspace(KickParameters p, ParticleInteger totalParticlesArgument) {
    totalParticles_ = totalParticlesArgument;
    params_ = p;
    numberProcessedParticles_ = 0;
}

KickWorkspace::~KickWorkspace() {}

static void addTreeNode(std::vector<TreeId>& trees, TreeId id) {
    TreeNode const* node = treeGetNode(id);
    trees.push_back(id);
    if (node->children[LEFT].index != -1) {
        addTreeNode(trees, node->children[LEFT]);
        addTreeNode(trees, node->children[RIGHT]);
    }
}

bool mortonCompare(Array<Fixed32, NDIM> a, Array<Fixed32, NDIM> b) {
    Array<unsigned, NDIM> abits;
    Array<unsigned, NDIM> bbits;
    for (int dim = 0; dim < NDIM; dim++) {
        abits[dim] = a[dim].raw();
        bbits[dim] = b[dim].raw();
    }
    for (int i = 0; i < 64; i++) {
        for (int dim = 0; dim < NDIM; dim++) {
            int const abit = abits[dim] & (unsigned) 0x80000000;
            int const bbit = bbits[dim] & (unsigned) 0x80000000;
            if (bbit && !abit) {
                return true;
            } else if (!bbit && abit) {
                return false;
            }
            abits[dim] <<= 1;
            bbits[dim] <<= 1;
        }
    }
    return false;
}

void KickWorkspace::toGpu() {
    cudaSetDevice();
    //	PRINT("Preparing gpu send on %i\n", hpx_rank());

    Timer tm;
    tm.start();
    auto const treeNodesRemoteBegin = treeNodesNextIndex();
    if (workitems_.size() == 0) {
        return;
    }

    struct GlobalPartRange {
        int rank;
        Pair<ParticleInteger> range;
        bool operator==(GlobalPartRange other) const {
            return rank == other.rank && range == other.range;
        }
    };

    struct GlobalPartRangeHash {
        size_t operator()(GlobalPartRange g) const {
            return (size_t) hpxSize() * g.range.first + g.rank;
        }
    };
    Timer tm2;
    Timer tm3;
    Timer tm4;
    tm2.start();
    MutexType mutex;
    Vector<hpx::future<void>> futs1;
    Vector<TreeId> treeIdsVector(treeIds_.begin(), treeIds_.end());
    Vector<Vector<TreeId>> idsByDepth(MAX_DEPTH);
    int numberThreads = hpxSize() == 1 ? 1 : std::max(1, (int) hpxHardwareConcurrency() / 2);
    for (int proc = 0; proc < numberThreads; proc++) {
        futs1.push_back(hpx::async([proc, numberThreads, &treeIdsVector, &idsByDepth, &mutex]() {
            std::unique_lock<MutexType> lock(mutex);
            int const b = (size_t) proc * treeIdsVector.size() / numberThreads;
            int const e = (size_t) (proc + 1) * treeIdsVector.size() / numberThreads;
            for (int i = b; i < e; i++) {
                lock.unlock();
                TreeNode const* ptr = treeGetNode(treeIdsVector[i]);
                lock.lock();
                idsByDepth[ptr->depth].push_back(treeIdsVector[i]);
            }
        }));
    }
    hpx::wait_all(futs1.begin(), futs1.end());
    tm2.stop();
    tm3.start();
    struct PartRequest {
        Vector<Vector<Pair<ParticleInteger>>> data;
        size_t count;
        PartRequest() {
            count = 0;
            data.resize(1);
        }
    };
    std::unordered_map<TreeId, int, KickWorkspaceTreeIdHash> treeMap;
    std::unordered_map<int, PartRequest> partRequests;
    std::unordered_map<GlobalPartRange, Pair<ParticleInteger>, GlobalPartRangeHash> partMap;
    std::set<TreeId> remoteRoots;
    ParticleInteger partIndex = particlesSize();
    Vector<hpx::future<void>> futs2;
    Vector<hpx::future<void>> futs3;
    size_t const opartsize = particlesSize();
    size_t const maxParts = 64 * 1024 * 1024;
    Vector<hpx::future<Vector<TreeId>>> futs0;
    Vector<TreeId> trees;
    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        auto const& ids = idsByDepth[depth];
        for (int proc = 0; proc < numberThreads; proc++) {
            futs0.push_back(hpx::async([proc, numberThreads, &ids, &treeMap, &mutex, &partRequests,
                                           &partMap, &partIndex, &remoteRoots]() {
                Vector<TreeId> trees;
                int const b = (size_t) proc * ids.size() / numberThreads;
                int const e = (size_t) (proc + 1) * ids.size() / numberThreads;
                for (int i = b; i < e; i++) {
                    if (treeMap.find(ids[i]) == treeMap.end()) {
                        TreeNode const* node = treeGetNode(ids[i]);
                        addTreeNode(trees, ids[i]);
                        int const rank = node->procRange.first;
                        if (rank != hpxRank()) {
                            auto const range = node->partRange;
                            auto const thisCount = range.second - range.first;
                            GlobalPartRange gpr;
                            gpr.rank = rank;
                            gpr.range = range;
                            size_t const numberParticles = node->numberParticles();
                            std::unique_lock<SharedMutexType> lock(particlesSharedMutex());
                            auto& entry = partRequests[rank];
                            entry.data.back().push_back(range);
                            entry.count += thisCount;
                            partMap[gpr].first = partIndex;
                            partMap[gpr].second = partIndex + numberParticles;
                            partIndex += numberParticles;
                            particlesResize(partIndex, false);
                            remoteRoots.insert(ids[i]);
                            if (entry.count > maxParts) {
                                size_t count = entry.count;
                                ParticleInteger index = 0;
                                entry.count = 0;
                                entry.data.resize(entry.data.size() + 1);
                            }
                        }
                    }
                }
                return trees;
            }));
        }
        hpx::wait_all(futs0.begin(), futs0.end());
        auto const start = trees.size();
        for (auto& f : futs0) {
            auto tmp = f.get();
            trees.reserve(trees.size() + tmp.size());
            for (auto const& entry : tmp) {
                trees.push_back(entry);
            }
        }
        for (int i = start; i < trees.size(); i++) {
            if (trees[i].proc == hpxRank()) {
                treeMap[trees[i]] = trees[i].index;
            } else {
                treeMap[trees[i]] = -1;
            }
        }
        futs0.resize(0);
    }
    tm3.stop();
    tm4.start();
    for (int i = 0; i < trees.size(); i++) {
        if (trees[i].proc != hpxRank()) {
            treeMap[trees[i]] = treeAddRemote(*treeGetNode(trees[i]));
        }
    }
    tm4.stop();

    //	PRINT("%e / %e %e %e to load tree nodes\n", tm2.read() + tm3.read() + tm4.read(),
    // tm2.read(), tm3.read(), tm4.read());
    tm2.reset();
    futs1.resize(0);
    numberThreads = std::max(1, (int) hpxHardwareConcurrency());
    for (auto i = partRequests.begin(); i != partRequests.end(); i++) {
        int const rank = i->first;
        hpx::future<void> fut = hpx::make_ready_future();
        for (int j = 0; j < i->second.data.size(); j++) {
            size_t const count = i->second.data[j].size();
            if (count) {
                fut = fut.then([rank, count, &partMap, i, j](hpx::future<void> fut) {
                    fut.get();
                    auto const& ranges = i->second.data[j];
                    auto const data = particlesGet(rank, ranges).get();
                    ParticleInteger index = 0;
                    for (int k = 0; k < ranges.size(); k++) {
                        index += ranges[k].second - ranges[k].first;
                    }
                    size_t sz = index;
                    index = 0;
                    for (int k = 0; k < ranges.size(); k++) {
                        GlobalPartRange gpr;
                        gpr.rank = rank;
                        gpr.range = ranges[k];
                        auto localRange = partMap[gpr];
                        auto const numberParticles = localRange.second - localRange.first;
                        for (int dim = 0; dim < NDIM; dim++) {
                            memcpy(&particlesPos(dim, localRange.first), &data[sz * dim + index],
                                numberParticles * sizeof(Fixed32));
                        }
                        if (getOptions().sph) {
                            Vector<Fixed32> x(numberParticles), y(numberParticles),
                                z(numberParticles);
                            Vector<char> types(numberParticles);
                            ParticleGlobalRange source;
                            source.proc = rank;
                            source.range = ranges[k];
                            particlesGlobalReadPos(
                                source, x.data(), y.data(), z.data(), types.data(), nullptr, 0);
                            std::memcpy(&particlesType(localRange.first), types.data(),
                                numberParticles * sizeof(char));
                        }
                        index += numberParticles;
                    }
                });
            }
        }
        futs1.push_back(std::move(fut));
    }
    auto* treeNodes = treeData();
    for (int proc = 0; proc < numberThreads; proc++) {
        futs2.push_back(hpx::async(
            HPX_PRIORITY_HI, [proc, numberThreads, &treeMap, &treeNodes, treeNodesRemoteBegin]() {
                for (int i = treeNodesRemoteBegin + proc; i < treeNodesNextIndex();
                    i += numberThreads) {
                    if (treeNodes[i].children[LEFT].index != -1) {
                        ASSERT(treeMap.find(treeNodes[i].children[LEFT]) != treeMap.end());
                        ASSERT(treeMap.find(treeNodes[i].children[RIGHT]) != treeMap.end());
                        auto& left = treeNodes[i].children[LEFT];
                        auto& right = treeNodes[i].children[RIGHT];
                        left.index = treeMap[left];
                        right.index = treeMap[right];
                    }
                }
            }));
    }
    hpx::wait_all(futs2.begin(), futs2.end());
    futs2.resize(0);
    std::function<void(int, ParticleInteger)> const adjustPartRefs =
        [&adjustPartRefs, &treeNodes](int index, ParticleInteger offset) -> void {
        auto& node = treeNodes[index];
        node.partRange.first += offset;
        node.partRange.second += offset;
        ASSERT(node.partRange.first >= 0);
        ASSERT(node.partRange.second <= particlesSize());
        if (node.children[LEFT].index != -1) {
            adjustPartRefs(node.children[LEFT].index, offset);
            adjustPartRefs(node.children[RIGHT].index, offset);
        }
    };
    for (auto i = remoteRoots.begin(); i != remoteRoots.end(); i++) {
        futs3.push_back(hpx::async(HPX_PRIORITY_HI,
            [i, &treeMap, &partMap, adjustPartRefs, &treeNodes, treeNodesRemoteBegin]() {
                GlobalPartRange gpr;
                ASSERT(treeMap.find(*i) != treeMap.end());
                int index = treeMap[*i];
                ASSERT(index >= treeNodesRemoteBegin);
                auto& node = treeNodes[index];
                gpr.rank = node.procRange.first;
                gpr.range = node.partRange;
                ASSERT(partMap.find(gpr) != partMap.end());
                auto localRange = partMap[gpr];
                ParticleInteger offset = localRange.first - gpr.range.first;
                ASSERT(treeNodes[index].partRange.first == gpr.range.first);
                adjustPartRefs(index, offset);
            }));
    }
    for (int proc = 0; proc < numberThreads; proc++) {
        futs2.push_back(hpx::async(HPX_PRIORITY_HI, [this, proc, numberThreads, &treeMap]() {
            for (int i = proc; i < workitems_.size(); i += numberThreads) {
                for (int j = 0; j < workitems_[i].dchecklist.size(); j++) {
                    if (workitems_[i].dchecklist[j].proc != hpxRank()) {
                        auto const iter = treeMap.find(workitems_[i].dchecklist[j]);
                        ASSERT(iter != treeMap.end());
                        workitems_[i].dchecklist[j].index = iter->second;
                    }
                }
                for (int j = 0; j < workitems_[i].echecklist.size(); j++) {
                    if (workitems_[i].echecklist[j].proc != hpxRank()) {
                        auto const iter = treeMap.find(workitems_[i].echecklist[j]);
                        ASSERT(iter != treeMap.end());
                        workitems_[i].echecklist[j].index = iter->second;
                    }
                }
            }
        }));
    }
    hpx::wait_all(futs2.begin(), futs2.end());
    futs2.resize(0);
    auto sfut = hpx::sort(PAR_EXECUTION_POLICY, workitems_.begin(), workitems_.end(),
        [](KickWorkitem const& a, KickWorkitem const& b) {
            return treeGetNode(a.self)->partRange.first < treeGetNode(b.self)->partRange.first;
        });
    hpx::wait_all(futs3.begin(), futs3.end());
    hpx::wait_all(futs1.begin(), futs1.end());
    sfut.get();
    tm.stop();
    //	PRINT("Took %e seconds to prepare gpu send\n", tm.read());
    tm.reset();
    tm.start();
    tree2Gpu();
    particlesMemadviseGpu();
    auto const kr = cudaExecuteKicks(params_, &particlesPos(XDIM, 0), &particlesPos(YDIM, 0),
        &particlesPos(ZDIM, 0), treeNodes, std::move(workitems_));
    tree2Cpu();
    tm.stop();
    particlesResize(opartsize);
    kickSetRc(kr);
    //	PRINT("GPU took %e seconds\n", tm.read());
}

void KickWorkspace::addParts(std::shared_ptr<KickWorkspace> ptr, ParticleInteger n) {
    bool doWork = false;
    std::unique_lock<MutexType> lock(mutex_);
    numberProcessedParticles_ += n;
    if (numberProcessedParticles_ == totalParticles_) {
        doWork = true;
    }
    lock.unlock();
    if (doWork) {
        ptr->toGpu();
    }
}

void KickWorkspace::clearBuffers() {
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<ClearBuffersAction>(c));
    }
    hpx::wait_all(futs.begin(), futs.end());
}

void KickWorkspace::addWork(std::shared_ptr<KickWorkspace> ptr, Expansion<float> l,
    Array<Fixed32, NDIM> pos, TreeId self, Vector<TreeId>&& dchecks, Vector<TreeId>&& echecks) {
    KickWorkitem item;
    item.l = l;
    item.pos = pos;
    item.self = self;
    bool doWork = false;
    {
        ParticleInteger const numberNodeParticles = treeGetNode(self)->numberParticles();
        std::lock_guard<MutexType> lock(mutex_);
        numberProcessedParticles_ += numberNodeParticles;
        if (numberProcessedParticles_ == totalParticles_) {
            doWork = true;
        }
        for (int i = 0; i < dchecks.size(); i++) {
            treeIds_.insert(dchecks[i]);
        }
        for (int i = 0; i < echecks.size(); i++) {
            treeIds_.insert(echecks[i]);
        }
    }
    item.dchecklist = std::move(dchecks);
    item.echecklist = std::move(echecks);
    std::unique_lock<MutexType> lock(mutex_);
    workitems_.push_back(std::move(item));
    lock.unlock();
    if (doWork) {
        ptr->toGpu();
    }
}
