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
#include <stack>

#include <cuda_runtime.h>
#include <unistd.h>

#include <cosmictiger/constants.hpp>
#include <cosmictiger/fast_future.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/kernel.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_cuda.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/stack_trace.hpp>
#include <cosmictiger/timer.hpp>

constexpr bool kVerbose = true;

HPX_PLAIN_ACTION(sphTreeNeighbor, SphTreeNeighborAction);

#define MAX_ACTIVE_WORKSPACES 1

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
    Vector<Array<float, NCHEMFRACS>, PinnedAllocator<Array<float, NCHEMFRACS>>> hostChem;
    Vector<float, PinnedAllocator<float>> hostColdFrac;
    Vector<float, PinnedAllocator<float>> hostDivv;
    Vector<float, PinnedAllocator<float>> hostVx;
    Vector<float, PinnedAllocator<float>> hostVy;
    Vector<float, PinnedAllocator<float>> hostVz;
#ifdef ENTROPY
    Vector<float, PinnedAllocator<float>> hostEntr;
#else
    Vector<float, PinnedAllocator<float>> hostEint;
#endif
    Vector<float, PinnedAllocator<float>> hostKappa;
    Vector<float, PinnedAllocator<float>> hostAlpha;
    Vector<float, PinnedAllocator<float>> hostShearv;
    Vector<float, PinnedAllocator<float>> hostRho;
    Vector<float, PinnedAllocator<float>> hostOmega;
    Vector<float, PinnedAllocator<float>> hostOmegaP;
    Vector<float, PinnedAllocator<float>> hostPre;
    Vector<float, PinnedAllocator<float>> hostH;
    Vector<char, PinnedAllocator<char>> hostRungs;
    Vector<char, PinnedAllocator<char>> hostStar;
    Vector<SphTreeNode, PinnedAllocator<SphTreeNode>> hostTrees;
    Vector<Fixed32, PinnedAllocator<Fixed32>> hostX;
    Vector<Fixed32, PinnedAllocator<Fixed32>> hostY;
    Vector<Fixed32, PinnedAllocator<Fixed32>> hostZ;
    Vector<int, PinnedAllocator<int>> hostNeighbors;
    Vector<int> hostSelflist;
    std::unordered_map<TreeId, int, SphTreeIdHash> treeMap;
    std::unordered_map<int, Pair<int>> neighborRanges;
    MutexType mutex;
    void addWork(TreeId selfid);
    SphRunReturn toGpu();
    SphRunWorkspace(SphRunParameters p) {
        params = p;
    }
};

inline bool rangeContains(Fixed32Range const& a, Array<Fixed32, NDIM> const x) {
    bool contains = true;
    for (int dim = 0; dim < NDIM; dim++) {
        if (distance(x[dim], a.begin[dim]) >= 0.0 && distance(a.end[dim], x[dim]) >= 0.0) {
        } else {
            contains = false;
            break;
        }
    }
    return contains;
}

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

bool isConverged(SphTreeNode const* self, int minrung) {
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

struct SphDataVecs {
    Vector<Fixed32> xs;
    Vector<Fixed32> ys;
    Vector<Fixed32> zs;
    Vector<char> rungs;
    Vector<float> fvels;
    Vector<float> hs;
    Vector<float> ents;
    Vector<float> vxs;
    Vector<float> vys;
    Vector<float> vzs;
    Vector<float> f0s;
    void clear() {
        xs.clear();
        ys.clear();
        zs.clear();
        rungs.clear();
        fvels.clear();
        hs.clear();
        ents.clear();
        vxs.clear();
        vys.clear();
        vzs.clear();
        f0s.clear();
    }
};

hpx::future<SphTreeNeighborReturn> sphTreeNeighbor(
    SphTreeNeighborParameters params, TreeId self, Vector<TreeId> checklist, int level) {
    if (level == 0 && hpxRank() == 0 && hpxSize() > 1) {
        sphTreeCacheClear();
    }
    ALWAYS_ASSERT(params.seti);
    //	ALWAYS_ASSERT(params.seto);
    ///	PRINT( "%i %i\n", level, checklist.size());
    Timer tm;
    SphTreeNeighborReturn kr;
    if (params.runType == SPH_TREE_NEIGHBOR_BOXES) {
        //		return hpx::make_ready_future(kr);
    }
    stackTraceActivate();
    SphTreeNode const* selfPtr = sphTreeGetNode(self);
    if (params.runType == SPH_TREE_NEIGHBOR_VALUE_AT) {
        Array<Fixed32, NDIM> x;
        x[XDIM] = params.x;
        x[YDIM] = params.y;
        x[ZDIM] = params.z;
        bool const test = selfPtr->box.contains(x);
        // PRINT("%i\n", test);
        // PRINT("%e %e %e\n", x[XDIM].to_double(), self_ptr->box.begin[0].to_double(),
        // self_ptr->box.end[0].to_double()); PRINT("%e %e %e\n", x[YDIM].to_double(),
        // self_ptr->box.begin[1].to_double(), self_ptr->box.end[1].to_double()); PRINT("%e %e
        // %e\n", x[ZDIM].to_double(), self_ptr->box.begin[2].to_double(),
        // self_ptr->box.end[2].to_double());
        if (!test && level > 3) {
            return hpx::make_ready_future(kr);
        }
    }
    if (params.runType == SPH_TREE_NEIGHBOR_NEIGHBORS && selfPtr->localRoot) {
        sphTreeFreeNeighborList();
        sphTreeClearNeighborRanges();
    }
    if (params.runType == SPH_TREE_NEIGHBOR_NEIGHBORS && checklist.size() == 0) {
        return hpx::make_ready_future(kr);
    }
    ASSERT(self.proc == hpxRank());
    bool threadLeft = params.runType != SPH_TREE_NEIGHBOR_VALUE_AT;
    Vector<TreeId> nextlist;
    Vector<TreeId> leaflist;
    Fixed32Range box;
    if ((params.seti | SPH_INTERACTIONS_I) && selfPtr->numberOfActiveParticles) {
        if (!rangeIntersect(selfPtr->outerBox, selfPtr->innerBox)) {
            for (int dim = 0; dim < NDIM; dim++) {
                PRINT("%e %e\n", selfPtr->outerBox.begin[dim].toDouble(),
                    selfPtr->outerBox.end[dim].toDouble());
                PRINT("%e %e\n", selfPtr->innerBox.begin[dim].toDouble(),
                    selfPtr->innerBox.end[dim].toDouble());
            }
        }
    }

    if ((params.seti | SPH_INTERACTIONS_J) && selfPtr->numberOfActiveParticles) {
        ALWAYS_ASSERT(rangeIntersect(selfPtr->innerBox, selfPtr->outerBox));
    }

    if (params.runType == SPH_TREE_NEIGHBOR_NEIGHBORS ||
        params.runType == SPH_TREE_NEIGHBOR_VALUE_AT) {
        do {
            nextlist.resize(0);
            for (int ci = 0; ci < checklist.size(); ci++) {
                auto const* other = sphTreeGetNode(checklist[ci]);
                bool test2 = false;
                bool test1 = false;
                if (params.seti | SPH_INTERACTIONS_I) {
                    test1 = rangeIntersect(selfPtr->outerBox, other->innerBox);
                }
                if (params.seti | SPH_INTERACTIONS_J) {
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
            Fixed32Range ibox, obox, pbox;
            float maxh = 0.f;
            for (int dim = 0; dim < NDIM; dim++) {
                pbox.begin[dim] = ibox.begin[dim] = obox.begin[dim] = 1.9;
                pbox.end[dim] = ibox.end[dim] = obox.end[dim] = -.9;
            }
            bool show = false;
            bool factive = false;
            for (ParticleInteger i = selfPtr->particleRange.first;
                i < selfPtr->particleRange.second; i++) {
                bool const active = sphParticlesRung(i) >= params.minRung;
                float const h = params.hWt * sphParticlesSmoothingLength(i);
                //				PRINT( "%e\n", params.h_wt);
                auto const myx = sphParticlesPos(XDIM, i);
                auto const myy = sphParticlesPos(YDIM, i);
                auto const myz = sphParticlesPos(ZDIM, i);
                int const k = i - selfPtr->particleRange.first;
                maxh = std::max(maxh, h);
                Array<Fixed32, NDIM> xValue;
                xValue[XDIM] = myx;
                xValue[YDIM] = myy;
                xValue[ZDIM] = myz;
                auto const tiny = 10.0 * RangeFixed::min().toDouble();
                /*				if (sph_particles_id(i) == 591) {
                 show = true;
                 PRINT("??????????? %e\n", h / params.h_wt);
                 for (int dim = 0; dim < NDIM; dim++) {
                 const double x = X[dim].to_double();
                 pbox.begin[dim] = std::min(pbox.begin[dim].to_double(), x - h - tiny);
                 pbox.end[dim] = std::max(pbox.end[dim].to_double(), x + h + tiny);
                 PRINT("%e %e %e\n", pbox.begin[dim].to_double(), x, pbox.end[dim].to_double());
                 }
                 }*/
                if (active) {
                    factive = true;
                }
                if ((params.seto & SPH_SET_ALL) || (active && (params.seto & SPH_SET_ACTIVE))) {
                    for (int dim = 0; dim < NDIM; dim++) {
                        double const x = xValue[dim].toDouble();
                        obox.begin[dim] = std::min(obox.begin[dim].toDouble(), x - h - tiny);
                        obox.end[dim] = std::max(obox.end[dim].toDouble(), x + h + tiny);
                    }
                }
                if ((params.seti & SPH_SET_ALL) || (active && (params.seti & SPH_SET_ACTIVE))) {
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
            if (show) {
                for (int dim = 0; dim < NDIM; dim++) {
                    PRINT("%e %e | %e %e\n", ibox.begin[dim].toDouble(), ibox.end[dim].toDouble(),
                        obox.begin[dim].toDouble(), obox.end[dim].toDouble());
                }
            }
            kr.innerBox = ibox;
            kr.outerBox = obox;
            sphTreeSetBoxes(self, kr.innerBox, kr.outerBox);
        } break;
        }
        return hpx::make_ready_future(kr);
    } else {
        checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
        SphTreeNode const* cl = sphTreeGetNode(selfPtr->children[LEFT]);
        SphTreeNode const* cr = sphTreeGetNode(selfPtr->children[RIGHT]);
        std::array<hpx::future<SphTreeNeighborReturn>, NCHILD> futs;
        futs[RIGHT] =
            sphTreeNeighborFork(params, selfPtr->children[RIGHT], checklist, level, threadLeft);
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
        sphParticlesCacheFree2();
    }
    //	PRINT("SPHRUN = %i\n", params.run_type);
    std::string profileName = "sph_run:" + std::to_string(params.runType);
    profilerEnter(profileName.c_str());

    feenableexcept(FE_DIVBYZERO);
    feenableexcept(FE_INVALID);
    feenableexcept(FE_OVERFLOW);
    params.cfl = getOptions().cfl;
    if (getOptions().cuda == false) {
        cuda = false;
    }
    //	cuda = false;
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
    static std::atomic<int> gpuWork;
    gpuWork = 0;
    //	PRINT( "sph_tree_list_size = %i\n", sph_tree_size());
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs2.push_back(hpx::async([proc, numberOfThreads, params, cuda, workspacePtr]() {
            SphRunReturn rc;
            int index = next++;
            SphDataVecs data;
            while (index < sphTreeLeaflistSize()) {
                data.clear();
                auto const selfid = sphTreeGetLeaf(index);
                auto const* self = sphTreeGetNode(selfid);
                bool test;
                switch (params.runType) {
                case SPH_RUN_AUX:
                    test = self->numberOfActiveParticles > 0;
                    break;

                case SPH_RUN_RUNGS:
                    test = self->numberOfActiveParticles > 0;
                    break;

                case SPH_RUN_PREHYDRO1:
                    test = self->numberOfActiveParticles > 0 && !isConverged(self, params.minRung);
                    break;

                case SPH_RUN_PREHYDRO2:
                    test = hasActiveNeighbors(self);
                    test = test && !isConverged(self, params.minRung);
                    break;

                case SPH_RUN_COND_INIT:
                    test = hasActiveNeighbors(self);
                    break;

                case SPH_RUN_CONDUCTION:
                    test = hasActiveNeighbors(self) && !isConverged(self, params.minRung);
                    break;

                case SPH_RUN_HYDRO:
                    test = self->numberOfActiveParticles > 0;
                    break;
                }
                if (test) {
                    if (cuda) {
                        gpuWork++;
                        workspacePtr->addWork(selfid);
                    } else {
                        ALWAYS_ASSERT(false);
                    }
                }

                index = next++;
            }
            return rc;
        }));
    }
    for (auto& f : futs2) {
        rc += f.get();
    }
    if (cuda && gpuWork) {
        rc += workspacePtr->toGpu();
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
    int neighborBegin = hostNeighbors.size();
    for (int i = self->neighborRange.first; i < self->neighborRange.second; i++) {
        auto const nid = sphTreeGetNeighbor(i);
        hostNeighbors.push_back(treeMap[nid]);
    }
    int neighborEnd = hostNeighbors.size();
    int const myindex = treeMap[selfid];
    auto& r = neighborRanges[hostSelflist.size()];
    hostSelflist.push_back(myindex);
    r.first = neighborBegin;
    r.second = neighborEnd;
}

SphRunReturn SphRunWorkspace::toGpu() {
    size_t particleStorageSize = 0;
    bool const chem = getOptions().chem;
    bool const stars = getOptions().stars;
    bool const gravity = getOptions().gravity;
    bool const conduction = getOptions().conduction;
    bool const diffusion = getOptions().diffusion;
#ifdef IMPLICIT_CONDUCTION
    bool const explicitConduction = false;
#else
    bool const explicitConduction = conduction;
#endif
    for (auto& node : hostTrees) {
        particleStorageSize += node.particleRange.second - node.particleRange.first;
    }
    hostX.resize(particleStorageSize);
    hostY.resize(particleStorageSize);
    hostZ.resize(particleStorageSize);
    if (params.runType == SPH_RUN_RUNGS) {
        hostRungs.resize(particleStorageSize);
        hostH.resize(particleStorageSize);
    } else if (params.runType == SPH_RUN_COND_INIT) {
        hostRungs.resize(particleStorageSize);
        hostH.resize(particleStorageSize);
#ifdef ENTROPY
        hostEntr.resize(particleStorageSize);
#else
        hostEint.resize(particleStorageSize);
#endif
        hostRho.resize(particleStorageSize);
        if (stars) {
            hostColdFrac.resize(particleStorageSize);
            hostStar.resize(particleStorageSize);
        }
    } else if (params.runType == SPH_RUN_CONDUCTION) {
        hostRho.resize(particleStorageSize);
        hostRungs.resize(particleStorageSize);
        hostH.resize(particleStorageSize);
#ifdef ENTROPY
        hostEntr.resize(particleStorageSize);
#else
        hostEint.resize(particleStorageSize);
#endif
        hostKappa.resize(particleStorageSize);
        hostOmega.resize(particleStorageSize);
        if (stars) {
            hostColdFrac.resize(particleStorageSize);
            hostStar.resize(particleStorageSize);
        }
    } else if (params.runType == SPH_RUN_PREHYDRO2) {
        hostH.resize(particleStorageSize);
#ifdef ENTROPY
        hostEntr.resize(particleStorageSize);
#else
        hostEint.resize(particleStorageSize);
#endif
        hostVx.resize(particleStorageSize);
        hostVy.resize(particleStorageSize);
        hostVz.resize(particleStorageSize);
        hostRungs.resize(particleStorageSize);
        if (stars) {
            hostStar.resize(particleStorageSize);
        }
    } else if (params.runType == SPH_RUN_HYDRO) {
        if (explicitConduction) {
            hostKappa.resize(particleStorageSize);
        }
        hostRho.resize(particleStorageSize);
        hostH.resize(particleStorageSize);
#ifdef ENTROPY
        hostEntr.resize(particleStorageSize);
#else
        hostEint.resize(particleStorageSize);
#endif
        hostVx.resize(particleStorageSize);
        hostVy.resize(particleStorageSize);
        hostVz.resize(particleStorageSize);
        hostAlpha.resize(particleStorageSize);
        hostOmegaP.resize(particleStorageSize);
        hostOmega.resize(particleStorageSize);
        hostPre.resize(particleStorageSize);
        if (stars) {
            hostColdFrac.resize(particleStorageSize);
        }
        if (diffusion) {
            hostShearv.resize(particleStorageSize);
            if (chem) {
                hostChem.resize(particleStorageSize);
            }
        }
        if (stars) {
            hostStar.resize(particleStorageSize);
        }
    }
    Vector<hpx::future<void>> futs;
    int const numberOfThreads = 8 * hpxHardwareConcurrency();
    std::atomic<int> index(0);
    std::atomic<ParticleInteger> particleIndex(0);
    for (int i = 0; i < hostSelflist.size(); i++) {
        hostTrees[hostSelflist[i]].neighborRange = neighborRanges[i];
    }
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([&index, proc, numberOfThreads, this, &particleIndex, chem, stars,
                                      conduction, explicitConduction, diffusion]() {
            int thisIndex = index++;
            while (thisIndex < hostTrees.size()) {
                auto& node = hostTrees[thisIndex];
                ParticleInteger const size = node.particleRange.second - node.particleRange.first;
                ParticleInteger const offset = (particleIndex += size) - size;
                sphParticlesGlobalReadPos(
                    node.globalParticleRange(), hostX.data(), hostY.data(), hostZ.data(), offset);
                if (params.runType == SPH_RUN_PREHYDRO2) {
                    sphParticlesGlobalReadRungs(
                        node.globalParticleRange(), hostRungs.data(), offset);
                    sphParticlesGlobalReadVels(node.globalParticleRange(), hostVx.data(),
                        hostVy.data(), hostVz.data(), offset);
#ifdef ENTROPY
                    sphParticlesGlobalReadEnergyAndSmoothlen(
                        node.globalParticleRange(), hostEntr.data(), hostH.data(), offset);
#else
                    sphParticlesGlobalReadEnergyAndSmoothlen(
                        node.globalParticleRange(), hostEint.data(), hostH.data(), offset);
#endif
                    if (stars) {
                        sphParticlesGlobalReadFcold(
                            node.globalParticleRange(), nullptr, hostStar.data(), offset);
                    }
                } else if (params.runType == SPH_RUN_HYDRO) {
                    if (stars) {
                        sphParticlesGlobalReadFcold(node.globalParticleRange(), hostColdFrac.data(),
                            hostStar.data(), offset);
                    }
#ifdef ENTROPY
                    sphParticlesGlobalReadEnergyAndSmoothlen(
                        node.globalParticleRange(), hostEntr.data(), hostH.data(), offset);
#else
                    sphParticlesGlobalReadEnergyAndSmoothlen(
                        node.globalParticleRange(), hostEint.data(), hostH.data(), offset);
#endif
                    sphParticlesGlobalReadVels(node.globalParticleRange(), hostVx.data(),
                        hostVy.data(), hostVz.data(), offset);
                    sphParticlesGlobalReadAux(node.globalParticleRange(), hostAlpha.data(),
                        hostOmega.data(), hostOmegaP.data(), hostPre.data(),
                        diffusion ? hostShearv.data() : nullptr, chem ? hostChem.data() : nullptr,
                        offset);
                    sphParticlesGlobalReadRho(node.globalParticleRange(), hostRho.data(), offset);
                } else if (params.runType == SPH_RUN_RUNGS) {
                    sphParticlesGlobalReadEnergyAndSmoothlen(
                        node.globalParticleRange(), nullptr, hostH.data(), offset);
                    sphParticlesGlobalReadRungs(
                        node.globalParticleRange(), hostRungs.data(), offset);
                } else if (params.runType == SPH_RUN_COND_INIT) {
                    sphParticlesGlobalReadRho(node.globalParticleRange(), hostRho.data(), offset);
                    if (stars) {
                        sphParticlesGlobalReadFcold(node.globalParticleRange(), hostColdFrac.data(),
                            hostStar.data(), offset);
                    }
                    sphParticlesGlobalReadRungs(
                        node.globalParticleRange(), hostRungs.data(), offset);
#ifdef ENTROPY
                    sphParticlesGlobalReadEnergyAndSmoothlen(
                        node.globalParticleRange(), hostEntr.data(), hostH.data(), offset);
#else
                    sphParticlesGlobalReadEnergyAndSmoothlen(
                        node.globalParticleRange(), hostEint.data(), hostH.data(), offset);
#endif
                } else if (params.runType == SPH_RUN_CONDUCTION) {
                    if (stars) {
                        sphParticlesGlobalReadFcold(node.globalParticleRange(), hostColdFrac.data(),
                            hostStar.data(), offset);
                    }
                    sphParticlesGlobalReadKappas(
                        node.globalParticleRange(), hostKappa.data(), offset);
                    sphParticlesGlobalReadRho(node.globalParticleRange(), hostRho.data(), offset);
                    sphParticlesGlobalReadRungs(
                        node.globalParticleRange(), hostRungs.data(), offset);
#ifdef ENTROPY
                    sphParticlesGlobalReadEnergyAndSmoothlen(
                        node.globalParticleRange(), hostEntr.data(), hostH.data(), offset);
#else
                    sphParticlesGlobalReadEnergyAndSmoothlen(
                        node.globalParticleRange(), hostEint.data(), hostH.data(), offset);
#endif
                    sphParticlesGlobalReadAux(node.globalParticleRange(), nullptr, hostOmega.data(),
                        nullptr, nullptr, nullptr, nullptr, offset);
                }
                node.particleRange.first = offset;
                node.particleRange.second = offset + size;
                thisIndex = index++;
            }
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());
    SphRunCudaData cudaData;
    CUDA_CHECK(cudaMalloc(&cudaData.selfs, sizeof(int) * hostSelflist.size()));
    CUDA_CHECK(cudaMalloc(&cudaData.x, sizeof(Fixed32) * hostX.size()));
    CUDA_CHECK(cudaMalloc(&cudaData.y, sizeof(Fixed32) * hostY.size()));
    CUDA_CHECK(cudaMalloc(&cudaData.z, sizeof(Fixed32) * hostZ.size()));
    if (params.runType == SPH_RUN_RUNGS) {
        CUDA_CHECK(cudaMalloc(&cudaData.rungs, sizeof(char) * hostRungs.size()));
        CUDA_CHECK(cudaMalloc(&cudaData.h, sizeof(float) * hostH.size()));
    } else if (params.runType == SPH_RUN_COND_INIT) {
        CUDA_CHECK(cudaMalloc(&cudaData.rungs, sizeof(char) * hostRungs.size()));
        CUDA_CHECK(cudaMalloc(&cudaData.h, sizeof(float) * hostH.size()));
#ifdef ENTROPY
        CUDA_CHECK(cudaMalloc(&cudaData.entr, sizeof(float) * hostEntr.size()));
#else
        CUDA_CHECK(cudaMalloc(&cudaData.eint, sizeof(float) * hostEint.size()));
#endif
        CUDA_CHECK(cudaMalloc(&cudaData.rho, sizeof(float) * hostRho.size()));
        if (stars) {
            CUDA_CHECK(cudaMalloc(&cudaData.coldFrac, sizeof(float) * hostColdFrac.size()));
            CUDA_CHECK(cudaMalloc(&cudaData.stars, sizeof(char) * hostStar.size()));
        }
    } else if (params.runType == SPH_RUN_CONDUCTION) {
        CUDA_CHECK(cudaMalloc(&cudaData.rungs, sizeof(char) * hostRungs.size()));
        CUDA_CHECK(cudaMalloc(&cudaData.h, sizeof(float) * hostH.size()));
#ifdef ENTROPY
        CUDA_CHECK(cudaMalloc(&cudaData.entr, sizeof(float) * hostEntr.size()));
#else
        CUDA_CHECK(cudaMalloc(&cudaData.eint, sizeof(float) * hostEint.size()));
#endif
        CUDA_CHECK(cudaMalloc(&cudaData.kappa, sizeof(float) * hostKappa.size()));
        CUDA_CHECK(cudaMalloc(&cudaData.omega, sizeof(float) * hostOmega.size()));
        CUDA_CHECK(cudaMalloc(&cudaData.rho, sizeof(float) * hostRho.size()));
        if (stars) {
            CUDA_CHECK(cudaMalloc(&cudaData.coldFrac, sizeof(float) * hostColdFrac.size()));
            CUDA_CHECK(cudaMalloc(&cudaData.stars, sizeof(char) * hostStar.size()));
        }
    } else if (params.runType == SPH_RUN_PREHYDRO2) {
        CUDA_CHECK(cudaMalloc(&cudaData.rungs, sizeof(char) * hostRungs.size()));
#ifdef ENTROPY
        CUDA_CHECK(cudaMalloc(&cudaData.entr, sizeof(float) * hostEntr.size()));
#else
        CUDA_CHECK(cudaMalloc(&cudaData.eint, sizeof(float) * hostEint.size()));
#endif
        CUDA_CHECK(cudaMalloc(&cudaData.h, sizeof(float) * hostH.size()));
        CUDA_CHECK(cudaMalloc(&cudaData.vx, sizeof(float) * hostVx.size()));
        CUDA_CHECK(cudaMalloc(&cudaData.vy, sizeof(float) * hostVy.size()));
        CUDA_CHECK(cudaMalloc(&cudaData.vz, sizeof(float) * hostVz.size()));
        if (stars) {
            CUDA_CHECK(cudaMalloc(&cudaData.stars, sizeof(char) * hostStar.size()));
        }
    } else if (params.runType == SPH_RUN_HYDRO) {
        if (explicitConduction) {
            CUDA_CHECK(cudaMalloc(&cudaData.kappa, sizeof(float) * hostKappa.size()));
        }
        CUDA_CHECK(cudaMalloc(&cudaData.h, sizeof(float) * hostH.size()));
        CUDA_CHECK(cudaMalloc(&cudaData.rho, sizeof(float) * hostRho.size()));
#ifdef ENTROPY
        CUDA_CHECK(cudaMalloc(&cudaData.entr, sizeof(float) * hostEntr.size()));
#else
        CUDA_CHECK(cudaMalloc(&cudaData.eint, sizeof(float) * hostEint.size()));
#endif
        CUDA_CHECK(cudaMalloc(&cudaData.vx, sizeof(float) * hostVx.size()));
        CUDA_CHECK(cudaMalloc(&cudaData.vy, sizeof(float) * hostVy.size()));
        CUDA_CHECK(cudaMalloc(&cudaData.vz, sizeof(float) * hostVz.size()));
        CUDA_CHECK(cudaMalloc(&cudaData.omega, sizeof(float) * hostOmega.size()));
        CUDA_CHECK(cudaMalloc(&cudaData.omegaP, sizeof(float) * hostOmegaP.size()));
        CUDA_CHECK(cudaMalloc(&cudaData.pre, sizeof(float) * hostPre.size()));
        CUDA_CHECK(cudaMalloc(&cudaData.alpha, sizeof(float) * hostAlpha.size()));
        if (stars) {
            CUDA_CHECK(cudaMalloc(&cudaData.coldFrac, sizeof(float) * hostColdFrac.size()));
            CUDA_CHECK(cudaMalloc(&cudaData.stars, sizeof(char) * hostStar.size()));
        }
        if (diffusion) {
            CUDA_CHECK(cudaMalloc(&cudaData.shearv, sizeof(float) * hostShearv.size()));
            if (chem) {
                CUDA_CHECK(
                    cudaMalloc(&cudaData.chem, NCHEMFRACS * sizeof(float) * hostChem.size()));
            }
        }
    }
    CUDA_CHECK(cudaMalloc(&cudaData.trees, sizeof(SphTreeNode) * hostTrees.size()));
    CUDA_CHECK(cudaMalloc(&cudaData.neighbors, sizeof(int) * hostNeighbors.size()));
    auto stream = cudaGetStream();
    if (params.runType == SPH_RUN_PREHYDRO2) {
#ifdef ENTROPY
        CUDA_CHECK(cudaMemcpyAsync(cudaData.entr, hostEntr.data(), sizeof(float) * hostEntr.size(),
            cudaMemcpyHostToDevice, stream));
#else
        CUDA_CHECK(cudaMemcpyAsync(cudaData.eint, hostEint.data(), sizeof(float) * hostEint.size(),
            cudaMemcpyHostToDevice, stream));
#endif
        CUDA_CHECK(cudaMemcpyAsync(cudaData.h, hostH.data(), sizeof(float) * hostH.size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cudaData.vx, hostVx.data(), sizeof(float) * hostVx.size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cudaData.vy, hostVy.data(), sizeof(float) * hostVy.size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cudaData.vz, hostVz.data(), sizeof(float) * hostVz.size(),
            cudaMemcpyHostToDevice, stream));
        if (stars) {
            CUDA_CHECK(cudaMemcpyAsync(cudaData.stars, hostStar.data(),
                sizeof(char) * hostStar.size(), cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(cudaData.rungs, hostRungs.data(),
            sizeof(char) * hostRungs.size(), cudaMemcpyHostToDevice, stream));
    } else if (params.runType == SPH_RUN_HYDRO) {
        if (explicitConduction) {
            CUDA_CHECK(cudaMemcpyAsync(cudaData.kappa, hostKappa.data(),
                sizeof(float) * hostKappa.size(), cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(cudaData.rho, hostRho.data(), sizeof(float) * hostRho.size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cudaData.h, hostH.data(), sizeof(float) * hostH.size(),
            cudaMemcpyHostToDevice, stream));
#ifdef ENTROPY
        CUDA_CHECK(cudaMemcpyAsync(cudaData.entr, hostEntr.data(), sizeof(float) * hostEntr.size(),
            cudaMemcpyHostToDevice, stream));
#else
        CUDA_CHECK(cudaMemcpyAsync(cudaData.eint, hostEint.data(), sizeof(float) * hostEint.size(),
            cudaMemcpyHostToDevice, stream));
#endif
        CUDA_CHECK(cudaMemcpyAsync(cudaData.vx, hostVx.data(), sizeof(float) * hostVx.size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cudaData.vy, hostVy.data(), sizeof(float) * hostVy.size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cudaData.vz, hostVz.data(), sizeof(float) * hostVz.size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cudaData.omega, hostOmega.data(),
            sizeof(float) * hostOmega.size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cudaData.omegaP, hostOmegaP.data(),
            sizeof(float) * hostOmegaP.size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cudaData.pre, hostPre.data(), sizeof(float) * hostPre.size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cudaData.alpha, hostAlpha.data(),
            sizeof(float) * hostAlpha.size(), cudaMemcpyHostToDevice, stream));
        if (diffusion) {
            CUDA_CHECK(cudaMemcpyAsync(cudaData.shearv, hostShearv.data(),
                sizeof(float) * hostShearv.size(), cudaMemcpyHostToDevice, stream));
            if (chem) {
                CUDA_CHECK(cudaMemcpyAsync(cudaData.chem, hostChem.data(),
                    NCHEMFRACS * sizeof(float) * hostChem.size(), cudaMemcpyHostToDevice, stream));
            }
        }
        if (stars) {
            CUDA_CHECK(cudaMemcpyAsync(cudaData.coldFrac, hostColdFrac.data(),
                sizeof(float) * hostColdFrac.size(), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(cudaData.stars, hostStar.data(),
                sizeof(char) * hostStar.size(), cudaMemcpyHostToDevice, stream));
        }
    } else if (params.runType == SPH_RUN_RUNGS) {
        CUDA_CHECK(cudaMemcpyAsync(cudaData.h, hostH.data(), sizeof(float) * hostH.size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cudaData.rungs, hostRungs.data(),
            sizeof(char) * hostRungs.size(), cudaMemcpyHostToDevice, stream));
    } else if (params.runType == SPH_RUN_COND_INIT) {
        CUDA_CHECK(cudaMemcpyAsync(cudaData.h, hostH.data(), sizeof(float) * hostH.size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cudaData.rho, hostRho.data(), sizeof(float) * hostRho.size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cudaData.rungs, hostRungs.data(),
            sizeof(char) * hostRungs.size(), cudaMemcpyHostToDevice, stream));
#ifdef ENTROPY
        CUDA_CHECK(cudaMemcpyAsync(cudaData.entr, hostEntr.data(), sizeof(float) * hostEntr.size(),
            cudaMemcpyHostToDevice, stream));
#else
        CUDA_CHECK(cudaMemcpyAsync(cudaData.eint, hostEint.data(), sizeof(float) * hostEint.size(),
            cudaMemcpyHostToDevice, stream));
#endif
        if (stars) {
            CUDA_CHECK(cudaMemcpyAsync(cudaData.coldFrac, hostColdFrac.data(),
                sizeof(float) * hostColdFrac.size(), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(cudaData.stars, hostStar.data(),
                sizeof(char) * hostStar.size(), cudaMemcpyHostToDevice, stream));
        }
    } else if (params.runType == SPH_RUN_CONDUCTION) {
        CUDA_CHECK(cudaMemcpyAsync(cudaData.omega, hostOmega.data(),
            sizeof(float) * hostOmega.size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cudaData.h, hostH.data(), sizeof(float) * hostH.size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cudaData.rungs, hostRungs.data(),
            sizeof(char) * hostRungs.size(), cudaMemcpyHostToDevice, stream));
#ifdef ENTROPY
        CUDA_CHECK(cudaMemcpyAsync(cudaData.entr, hostEntr.data(), sizeof(float) * hostEntr.size(),
            cudaMemcpyHostToDevice, stream));
#else
        CUDA_CHECK(cudaMemcpyAsync(cudaData.eint, hostEint.data(), sizeof(float) * hostEint.size(),
            cudaMemcpyHostToDevice, stream));
#endif
        CUDA_CHECK(cudaMemcpyAsync(cudaData.rho, hostRho.data(), sizeof(float) * hostRho.size(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(cudaData.kappa, hostKappa.data(),
            sizeof(float) * hostKappa.size(), cudaMemcpyHostToDevice, stream));
        if (stars) {
            CUDA_CHECK(cudaMemcpyAsync(cudaData.coldFrac, hostColdFrac.data(),
                sizeof(float) * hostColdFrac.size(), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(cudaData.stars, hostStar.data(),
                sizeof(char) * hostStar.size(), cudaMemcpyHostToDevice, stream));
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(
        cudaData.x, hostX.data(), sizeof(Fixed32) * hostX.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        cudaData.y, hostY.data(), sizeof(Fixed32) * hostY.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        cudaData.z, hostZ.data(), sizeof(Fixed32) * hostZ.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(cudaData.trees, hostTrees.data(),
        sizeof(SphTreeNode) * hostTrees.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(cudaData.selfs, hostSelflist.data(),
        sizeof(int) * hostSelflist.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(cudaData.neighbors, hostNeighbors.data(),
        sizeof(int) * hostNeighbors.size(), cudaMemcpyHostToDevice, stream));
    cudaData.sinkParticleIndices = &sphParticlesParticleIndex(0);
    cudaData.saSnk = &sphParticlesSemiactive(0);
    cudaData.rungsSnk = &particlesRung(0);
    cudaData.omegaSnk = &sphParticlesOmega(0);
    cudaData.omegaPSnk = &sphParticlesOmegaP(0);
    cudaData.preSnk = &sphParticlesPressure(0);
    cudaData.shearSnk = &sphParticlesShear(0);
    cudaData.divvSnk = &sphParticlesDivv(0);
#ifdef HOPKINS
    cudaData.rhoSnk = &sphParticlesRhoRho(0);
#else
    cudaData.rhoSnk = &sphParticlesRho(0);
#endif
    cudaData.coldMassSnk = &sphParticlesColdMass(0);
    cudaData.rec1Snk = &sphParticlesRec1(0);
    cudaData.rec2Snk = &sphParticlesRec2(0);
    cudaData.kapSnk = &sphParticlesKappa(0);
#ifdef ENTROPY
    cudaData.entr0Snk = &sphParticlesEntr0(0);
    cudaData.dentr1Snk = &sphParticlesDentr1(0);
    cudaData.dentr2Snk = &sphParticlesDentr2(0);
#else
    cudaData.eint0Snk = &sphParticlesEint0(0);
    cudaData.deint1Snk = &sphParticlesDeint1(0);
    cudaData.deint2Snk = &sphParticlesDeint2(0);
#endif
    cudaData.dcoldMass = &sphParticlesDcoldMass(0);
    cudaData.gxSnk = &sphParticlesGravityAcceleration(XDIM, 0);
    cudaData.gySnk = &sphParticlesGravityAcceleration(YDIM, 0);
    cudaData.gzSnk = &sphParticlesGravityAcceleration(ZDIM, 0);
    cudaData.defGamma = getOptions().gamma;
    cudaData.gravitySofteningLength =
        getOptions().hsoft / selectedGravityKernel().supportPerSmoothing;
    cudaData.numberOfTargetNodes = hostSelflist.size();
    cudaData.chemistry = getOptions().chem;
    cudaData.convergedSnk = &sphParticlesConverged(0);
    cudaData.gravity = getOptions().gravity;
    cudaData.conduction = getOptions().conduction;
    cudaData.rec5Snk = &sphParticlesRec5(0);
    cudaData.rec6Snk = &sphParticlesRec6(0);
    cudaData.codeDifToCgs = sqr(getOptions().codeToCm) / getOptions().codeToS;
    cudaData.centralAcceleration = getOptions().centralAcceleration;
    cudaData.hcentral = getOptions().hcentral;
    cudaData.g = getOptions().gm;
    cudaData.y0 = getOptions().y0;
    cudaData.rho0C = getOptions().rho0C;
    cudaData.rho0B = getOptions().rho0B;
    cudaData.t0 = params.t0;
    //	PRINT("HSTAR = %e %e  %e  %e  \n", cuda_data.hstar0, rho_star_phys_cgs, rho_star_phys_code,
    // rho_star_co_code);
    cudaData.m = getOptions().sphMass;
    cudaData.n = getOptions().sphNeighborNumber;
    cudaData.kappa0 = 1.31 * pow(3.0, 1.5) * pow(constants::kB, 3.5) / 4.0 / sqrt(M_PI) /
        pow(constants::kE, 4) / sqrt(constants::kMe);
    //	cuda_data.dchem_snk = &sph_particles_dchem(0);
    cudaData.eta = getOptions().eta;
    //	PRINT("Running with %i nodes\n", host_trees.size());
    PRINT("Sending %i %i\n", params.runType, hostSelflist.size());

    for (int i = 0; i < hostSelflist.size(); i++) {
        auto node = hostTrees[hostSelflist[i]];
        bool found = false;
        for (int j = node.neighborRange.first; j < node.neighborRange.second; j++) {
            if (hostNeighbors[j] == hostSelflist[i]) {
                found = true;
                break;
            }
        }
        ALWAYS_ASSERT(found);
    }

    auto rc = sphRunCuda(params, cudaData, stream);
    synchronizeCudaStream(stream);
    cudaEndStream(stream);
    if (params.runType == SPH_RUN_RUNGS) {
        CUDA_CHECK(cudaFree(cudaData.rungs));
        CUDA_CHECK(cudaFree(cudaData.h));
    } else if (params.runType == SPH_RUN_COND_INIT) {
        CUDA_CHECK(cudaFree(cudaData.rungs));
        CUDA_CHECK(cudaFree(cudaData.rho));
        CUDA_CHECK(cudaFree(cudaData.h));
#ifdef ENTROPY
        CUDA_CHECK(cudaFree(cudaData.entr));
#else
        CUDA_CHECK(cudaFree(cudaData.eint));
#endif
        if (stars) {
            CUDA_CHECK(cudaFree(cudaData.coldFrac));
            CUDA_CHECK(cudaFree(cudaData.stars));
        }
    } else if (params.runType == SPH_RUN_CONDUCTION) {
        CUDA_CHECK(cudaFree(cudaData.rungs));
        CUDA_CHECK(cudaFree(cudaData.h));
#ifdef ENTROPY
        CUDA_CHECK(cudaFree(cudaData.entr));
#else
        CUDA_CHECK(cudaFree(cudaData.eint));
#endif
        CUDA_CHECK(cudaFree(cudaData.rho));
        CUDA_CHECK(cudaFree(cudaData.omega));
        CUDA_CHECK(cudaFree(cudaData.kappa));
        if (stars) {
            CUDA_CHECK(cudaFree(cudaData.coldFrac));
            CUDA_CHECK(cudaFree(cudaData.stars));
        }
    } else if (params.runType == SPH_RUN_PREHYDRO2) {
#ifdef ENTROPY
        CUDA_CHECK(cudaFree(cudaData.entr));
#else
        CUDA_CHECK(cudaFree(cudaData.eint));
#endif
        CUDA_CHECK(cudaFree(cudaData.h));
        CUDA_CHECK(cudaFree(cudaData.vx));
        CUDA_CHECK(cudaFree(cudaData.vy));
        CUDA_CHECK(cudaFree(cudaData.vz));
        CUDA_CHECK(cudaFree(cudaData.rungs));
        if (stars) {
            CUDA_CHECK(cudaFree(cudaData.stars));
        }
    } else if (params.runType == SPH_RUN_HYDRO) {
        if (explicitConduction) {
            CUDA_CHECK(cudaFree(cudaData.kappa));
        }
        CUDA_CHECK(cudaFree(cudaData.h));
#ifdef ENTROPY
        CUDA_CHECK(cudaFree(cudaData.entr));
#else
        CUDA_CHECK(cudaFree(cudaData.eint));
#endif
        CUDA_CHECK(cudaFree(cudaData.rho));
        CUDA_CHECK(cudaFree(cudaData.vx));
        CUDA_CHECK(cudaFree(cudaData.vy));
        CUDA_CHECK(cudaFree(cudaData.vz));
        CUDA_CHECK(cudaFree(cudaData.omega));
        CUDA_CHECK(cudaFree(cudaData.pre));
        CUDA_CHECK(cudaFree(cudaData.omegaP));
        CUDA_CHECK(cudaFree(cudaData.alpha));
        if (stars) {
            CUDA_CHECK(cudaFree(cudaData.coldFrac));
            CUDA_CHECK(cudaFree(cudaData.stars));
        }
        if (diffusion) {
            CUDA_CHECK(cudaFree(cudaData.shearv));
            if (chem) {
                CUDA_CHECK(cudaFree(cudaData.chem));
            }
        }
    }
    CUDA_CHECK(cudaFree(cudaData.x));
    CUDA_CHECK(cudaFree(cudaData.y));
    CUDA_CHECK(cudaFree(cudaData.z));
    CUDA_CHECK(cudaFree(cudaData.trees));
    CUDA_CHECK(cudaFree(cudaData.selfs));
    CUDA_CHECK(cudaFree(cudaData.neighbors));

    return rc;
}

HPX_PLAIN_ACTION(sphApplyConductionUpdate, SphApplyConductionUpdateAction);

CondUpdateReturn sphApplyConductionUpdate(int minrung) {
    int const numberOfThreads = hpxHardwareConcurrency();
    float errMax = 0.0;
    std::vector<hpx::future<CondUpdateReturn>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<SphApplyConductionUpdateAction>(c, minrung));
    }
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([proc, numberOfThreads, minrung] {
            float errMax = 0.0;
            float errRms = 0.0;
            size_t n = 0;
            ParticleInteger const b = (size_t) proc * sphParticlesSize() / numberOfThreads;
            ParticleInteger const e = (size_t) (proc + 1) * sphParticlesSize() / numberOfThreads;
            for (ParticleInteger i = b; i < e; i++) {
                if (!sphParticlesConverged(i) &&
                    (sphParticlesRung(i) >= minrung || sphParticlesSemiactive(i)) &&
                    !sphParticlesIsstar(i)) {
#ifdef ENTROPY
                    float& aValue = sphParticlesEntr(i);
                    float const dA = sphParticlesDentr1(i);
                    ALWAYS_ASSERT(std::max(aValue, aValue + dA) > 0.0f);
                    float const thisErr = fabs(dA) / std::max(sphParticlesEntr0(i), aValue + dA);
#else
                    float& aValue = sphParticlesEint(i);
                    float const dA = sphParticlesDeint1(i);
                    ALWAYS_ASSERT(std::max(aValue, aValue + dA) > 0.0f);
                    float const thisErr = fabs(dA) / std::max(sphParticlesEint0(i), aValue + dA);
#endif
                    aValue += dA;
                    if (aValue < 0.0f) {
                        PRINT("%e %e\n", aValue, dA);
                    }
#ifdef ENTROPY
                    ALWAYS_ASSERT(sphParticlesEntr(i) > 0.0);
                    errMax = std::max(thisErr, errMax);
                    if (thisErr > 1.0) {
                        PRINT("%e %e %e\n", aValue - dA, dA, sphParticlesEntr0(i));
                    }
#else
                    ALWAYS_ASSERT(sphParticlesEint(i) > 0.0);
                    errMax = std::max(thisErr, errMax);
                    if (thisErr > 1.0) {
                        PRINT("%e %e %e\n", aValue - dA, dA, sphParticlesEint0(i));
                    }
#endif
                    errRms += sqr(thisErr);
                    if (thisErr < SPH_DIFFUSION_TOLER3) {
                        sphParticlesConverged(i) = true;
                    }
                    n++;
                }
            }
            CondUpdateReturn rc;
            rc.errMax = errMax;
            rc.errRms = errRms;
            rc.n = n;
            return rc;
        }));
    }
    CondUpdateReturn rc;
    rc.errMax = 0.0;
    rc.errRms = 0.0;
    rc.n = 0;
    for (auto& f : futs) {
        auto tmp = f.get();
        rc.errMax = std::max(rc.errMax, tmp.errMax);
        rc.errRms += tmp.errRms;
        rc.n += tmp.n;
    }
    if (hpxRank() == 0) {
        if (rc.n > 0) {
            rc.errRms /= rc.n;
        }
        rc.errRms = sqrtf(rc.errRms);
    }
    return rc;
}
