/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

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

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

#include <chealpix.h>
#include <healpix/healpix_base.h>
#include <silo.h>

#include <cosmictiger/bh.hpp>
#include <cosmictiger/compress.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/cosmology.hpp>
#include <cosmictiger/device_vector.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/group_entry.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/lightcone.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/timer.hpp>

using HealpixType = T_Healpix_Base<int>;

struct LcGroupData {
    Vector<LcParticle> parts;
    GroupEntry arc;
};

static std::shared_ptr<HealpixType> healpix;
static Pair<int> myPixRange;
static std::unordered_set<int> bndPix;
static LcPartMapType* partMapPtr = nullptr;
static LcTreeMapType* treeMapPtr = nullptr;
static std::unordered_map<int, std::shared_ptr<SpinlockType>> mutexMap;
static LcGroup groupIdCounter = 0;
// static vector<group_entry> saved_groups;
static Vector<LcParticle> partBuffer;
static SharedMutexType sharedMutex;
static SpinlockType uniqueMutex;
static SpinlockType leafMutex;
static DeviceVector<LcTreeId> leafNodes;
static double tauMax;
static int nsideValue;
static int npixValue;

static LcParticles lcGetParticles1(int pix);
static Pair<Vector<long long>, Vector<char>> lcGetParticles2(int pix);
static void lcSendParticles(Vector<LcParticle>);
static void lcSendBufferParticles(Vector<LcParticle>);
static int vec2pix(double x, double y, double z);
static int pix2rank(int pix);
static int computeNside(double tau);
static Vector<int> pixNeighbors(int pix);
static int epoch = 0;
void lcSiloOut(char const* filename, Vector<Subgroup> const& subgroups);

HPX_PLAIN_ACTION(lcTimeToFlush, LcTimeToFlushAction);
HPX_PLAIN_ACTION(lcParts2groups, LcParts2groupsAction);
HPX_PLAIN_ACTION(lcInit, LcInitAction);
HPX_PLAIN_ACTION(lcGetParticles1, LcGetParticles1Action);
HPX_PLAIN_ACTION(lcGetParticles2, LcGetParticles2Action);
HPX_PLAIN_ACTION(lcFormTrees, LcFormTreesAction);
HPX_PLAIN_ACTION(lcBuffer2homes, LcBuffer2homesAction);
HPX_PLAIN_ACTION(lcGroups2homes, LcGroups2homesAction);
HPX_PLAIN_ACTION(lcSendBufferParticles, LcSendBufferParticlesAction);
HPX_PLAIN_ACTION(lcSendParticles, LcSendParticlesAction);
HPX_PLAIN_ACTION(lcFindGroups, LcFindGroupsAction);
HPX_PLAIN_ACTION(lcFindNeighbors, LcFindNeighborsAction);
HPX_PLAIN_ACTION(lcParticleBoundaries1, LcParticleBoundaries1Action);
HPX_PLAIN_ACTION(lcParticleBoundaries2, LcParticleBoundaries2Action);

static std::atomic<int> nextFilenum = 0;

static int lcNextFilenum();

HPX_PLAIN_ACTION(lcNextFilenum, LcNextFilenumAction);

static int lcNextFilenum() {
    if (hpxRank() == 0) {
        return nextFilenum++;
    } else {
        LcNextFilenumAction act;
        static auto const root = hpxLocalities()[0];
        return act(root);
    }
}

size_t lcAddParts(DeviceVector<DeviceVector<LcEntry>> const& entries, double scale, double tau) {
    int const start = partBuffer.size();
    Vector<size_t> counts(entries.size() + 1);
    size_t count = 0;
    for (int i = 0; i < entries.size(); i++) {
        counts[i] = count;
        count += entries[i].size();
    }
    counts.back() = count;
    partBuffer.resize(start + count);
    int const nthreads = hpxHardwareConcurrency();
    Vector<hpx::future<void>> futs;
    std::atomic<int> index(0);
    size_t const npix = sqr((size_t) getOptions().lcMapSize) * 12;
    Vector<float> pix(npix, 0.0);
    int minPix = npix;
    int maxPix = 0;
    MutexType mutex;
    if (counts.back()) {
        for (int proc = 0; proc < nthreads; proc++) {
            futs.push_back(hpx::async([&pix, npix, &mutex, &minPix, &maxPix, scale, tau, &index,
                                          start, nthreads, proc, &entries, &counts]() {
                int k = index++;
                int thisMinPix = npix;
                int thisMaxPix = 0;
                while (k < entries.size()) {
                    auto& entry = entries[k];
                    int const b = start + counts[k];
                    int const e = start + counts[k + 1];
                    for (int i = b; i < e; i++) {
                        int const j = i - b;
                        auto& pi = partBuffer[i];
                        auto const& pj = entry[j];
                        pi.pos[XDIM] = pj.x;
                        pi.pos[YDIM] = pj.y;
                        pi.pos[ZDIM] = pj.z;
                        double const tau1 =
                            sqrtf(sqr(pj.x.toDouble(), pj.y.toDouble(), pj.z.toDouble()));
                        pix[pj.pix] += 1.0 / sqr(tau1);
                        auto const a = scale + cosmosDadtau(scale) * (tau1 - tau);
                        pi.vel[XDIM] = pj.vx / a;
                        pi.vel[YDIM] = pj.vy / a;
                        pi.vel[ZDIM] = pj.vz / a;
                        thisMaxPix = std::max(thisMaxPix, pj.pix);
                        thisMinPix = std::min(thisMinPix, pj.pix);
                    }
                    k = index++;
                }
                std::lock_guard<MutexType> lock(mutex);
                maxPix = std::max(maxPix, thisMaxPix);
                minPix = std::min(minPix, thisMinPix);
            }));
        }
        hpx::wait_all(futs.begin(), futs.end());
        std::string filename = getOptions().lcDir + "/lc.healpix." + std::to_string(epoch) + "." +
            std::to_string(hpxRank());
        FILE* fp = fopen(filename.c_str(), "wb");
        if (fp == nullptr) {
            THROW_ERROR("Unable to open %s for writing\n", filename.c_str());
        }
        int nside = getOptions().lcMapSize;
        FWRITE(&nside, sizeof(int), 1, fp);
        FWRITE(&minPix, sizeof(int), 1, fp);
        FWRITE(&maxPix, sizeof(int), 1, fp);
        FWRITE(pix.data() + minPix, sizeof(float), maxPix - minPix, fp);
        fclose(fp);
    }
    return 0;
}

int lcNside() {
    return nsideValue;
}

void lcSave(FILE* fp) {
    int dummy;
    size_t sz = partBuffer.size();
    fwrite(&sz, sizeof(size_t), 1, fp);
    fwrite(partBuffer.data(), sizeof(LcParticle), partBuffer.size(), fp);
    fwrite(&epoch, sizeof(int), 1, fp);
    fwrite(&sz, sizeof(size_t), 1, fp);
    fwrite(&groupIdCounter, sizeof(LcGroup), 1, fp);
}

void lcLoad(FILE* fp) {
    size_t sz;
    int dummy;
    FREAD(&sz, sizeof(size_t), 1, fp);
    partBuffer.resize(sz);
    FREAD(partBuffer.data(), sizeof(LcParticle), partBuffer.size(), fp);
    FREAD(&epoch, sizeof(int), 1, fp);
    FREAD(&sz, sizeof(size_t), 1, fp);
    FREAD(&groupIdCounter, sizeof(LcGroup), 1, fp);
}

size_t lcTimeToFlush(double tau, double requestedTauMax) {
    Vector<hpx::future<size_t>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<LcTimeToFlushAction>(c, tau, requestedTauMax));
    }
    size_t nparts = partBuffer.size();
    for (auto& f : futs) {
        nparts = std::max(nparts, f.get());
    }
    if (hpxRank() == 0) {
        tauMax = requestedTauMax;
        int const nside = computeNside(tau);
        int const npix = nside == 0 ? 1 : 12 * sqr(nside);
        size_t const ranks = std::min(npix, hpxSize());
        size_t const partsPerRank = getOptions().nparts / hpxSize();
        /*		double factor = 1.0 / 8.0;
         double tm = tau / tau_max * 64;
         if (tm > 63.0) {
         factor = (64.0 - tm) / 8.0;
         }*/
        if (nparts >= (M_PI / 10.0) * partsPerRank) {
            return 1;
        } else {
            return 0;
        }
    } else {
        return nparts;
    }
}

void lcParts2groups(double a, double linkLen, int ti) {
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<LcParts2groupsAction>(c, a, linkLen, ti));
    }
    std::unordered_map<LcGroup, Vector<LcEntry>> groupsMap;
    int i = 0;
    while (i < partBuffer.size()) {
        auto const grp = partBuffer[i].group;
        if (grp != LC_EDGE_GROUP) {
            LcEntry entry;
            auto const& p = partBuffer[i];
            entry.x = p.pos[XDIM];
            entry.y = p.pos[YDIM];
            entry.z = p.pos[ZDIM];
            entry.vx = p.vel[XDIM];
            entry.vy = p.vel[YDIM];
            entry.vz = p.vel[ZDIM];
            groupsMap[grp].push_back(entry);
            partBuffer[i] = partBuffer.back();
            partBuffer.pop_back();
        } else {
            i++;
        }
    }
    static int total = 0;
    auto const minSz = ROCKSTAR_MIN_GROUP;
    auto j = groupsMap.begin();
    while (j != groupsMap.end()) {
        if (j->second.size() < minSz && j->first != LC_NO_GROUP) {
            groupsMap[LC_NO_GROUP].insert(
                groupsMap[LC_NO_GROUP].end(), j->second.begin(), j->second.end());
            j = groupsMap.erase(j);
        } else {
            j++;
        }
    }
    Vector<Pair<LcGroup, Vector<LcEntry>>> groups;
    Vector<Pair<LcGroup, CompressedParticles>> groupArcs;
    for (auto i = groupsMap.begin(); i != groupsMap.end(); i++) {
        Pair<LcGroup, Vector<LcEntry>> entry;
        entry.first = i->first;
        total += i->second.size();
        entry.second = std::move(i->second);
        groups.push_back(std::move(entry));
    }
    std::sort(groups.begin(), groups.end(),
        [](Pair<LcGroup, Vector<LcEntry>> const& a, Pair<LcGroup, Vector<LcEntry>> const& b) {
            return a.second.size() > b.second.size();
        });
    std::atomic<int> nextIndex(0);
    int const nthreads = 2 * hpxHardwareConcurrency();
    MutexType mutex;
    Vector<RockstarRecord> recs;
    for (int proc = 0; proc < nthreads; proc++) {
        futs.push_back(hpx::async([&recs, &mutex, &nextIndex, &groups]() {
            int index = nextIndex++;
            while (index < groups.size()) {
                if (groups[index].first != LC_NO_GROUP) {
                    auto theseRecs = rockstarFindSubgroups(groups[index].second, tauMax);
                    for (auto& r : theseRecs) {
                        r.id.fofId = groups[index].first;
                    }
                    std::lock_guard<MutexType> lock(mutex);
                    recs.insert(recs.end(), theseRecs.begin(), theseRecs.end());
                }
                index = nextIndex++;
            }
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());
    std::string filename = getOptions().lcDir + "/lc.groups." + std::to_string(epoch) + "." +
        std::to_string(hpxRank());
    FILE* fp = fopen(filename.c_str(), "wb");
    if (fp == nullptr) {
        THROW_ERROR("Unable to open %s for writing\n", filename.c_str());
    }
    if (fwrite(recs.data(), sizeof(RockstarRecord), recs.size(), fp) != recs.size()) {
        THROW_ERROR("Unable to write data for %s\n", filename.c_str());
    }
    fclose(fp);
    epoch++;
}

static int rankFromGroupId(long long id) {
    if (id != LC_NO_GROUP && id != LC_EDGE_GROUP) {
        return (id - 1) % hpxSize();
    } else {
        return -1;
    }
}

static void lcSendParticles(Vector<LcParticle> parts) {
    Timer tm;
    auto& partMap = *partMapPtr;
    for (auto const& part : parts) {
        int const pix = vec2pix(
            part.pos[XDIM].toDouble(), part.pos[YDIM].toDouble(), part.pos[ZDIM].toDouble());
        std::unique_lock<SpinlockType> lock(*mutexMap[pix]);
        for (int dim = 0; dim < NDIM; dim++) {
            partMap[pix].pos[dim].pushBack(part.pos[dim]);
            partMap[pix].vel[dim].pushBack(part.vel[dim]);
        }
        partMap[pix].group.pushBack(part.group);
    }
}

static void lcSendBufferParticles(Vector<LcParticle> parts) {
    int start, stop;
    std::unique_lock<SharedMutexType> ulock(sharedMutex);
    start = partBuffer.size();
    partBuffer.resize(start + parts.size());
    stop = partBuffer.size();
    ulock.unlock();
    constexpr int kChunkSize = 1024;
    for (int i = 0; i < parts.size(); i += kChunkSize) {
        int const end = std::min((int) parts.size(), i + kChunkSize);
        std::shared_lock<SharedMutexType> lock(sharedMutex);
        for (int j = i; j < end; j++) {
            partBuffer[j + start] = parts[j];
        }
    }
}

void lcBuffer2homes() {
    Vector<hpx::future<void>> futs1;
    Vector<hpx::future<void>> futs2;
    for (auto const& c : hpxChildren()) {
        futs1.push_back(hpx::async<LcBuffer2homesAction>(c));
    }
    int const nthreads = hpxHardwareConcurrency();
    static MutexType mapMutex;
    for (int proc = 0; proc < nthreads; proc++) {
        futs2.push_back(hpx::async([proc, nthreads]() {
            int const begin = (size_t) proc * partBuffer.size() / nthreads;
            int const end = (size_t) (proc + 1) * partBuffer.size() / nthreads;
            std::unordered_map<int, Vector<LcParticle>> sends;
            for (int i = begin; i < end; i++) {
                auto const& part = partBuffer[i];
                int const pix = vec2pix(part.pos[XDIM].toDouble(), part.pos[YDIM].toDouble(),
                    part.pos[ZDIM].toDouble());
                int const rank = pix2rank(pix);
                sends[rank].push_back(part);
            }
            Vector<hpx::future<void>> futs;
            for (auto i = sends.begin(); i != sends.end(); i++) {
                futs.push_back(hpx::async<LcSendParticlesAction>(
                    hpxLocalities()[i->first], std::move(i->second)));
            }
            hpx::wait_all(futs.begin(), futs.end());
        }));
    }
    hpx::wait_all(futs2.begin(), futs2.end());
    partBuffer = decltype(partBuffer)();
    hpx::wait_all(futs1.begin(), futs1.end());
}

void lcGroups2homes() {
    auto& partMap = *partMapPtr;
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<LcGroups2homesAction>(c));
    }
    for (int pix = myPixRange.first; pix < myPixRange.second; pix++) {
        futs.push_back(hpx::async([pix, &partMap]() {
            Vector<hpx::future<void>> futs;
            auto& parts = partMap[pix];
            std::unordered_map<int, Vector<LcParticle>> sends;
            for (int i = 0; i < parts.group.size(); i++) {
                int rank = rankFromGroupId(parts.group[i]);
                if (rank == -1) {
                    if (parts.group[i] == LC_EDGE_GROUP) {
                        rank = hpxRank();
                    }
                }
                if (rank != -1) {
                    LcParticle part;
                    for (int dim = 0; dim < NDIM; dim++) {
                        part.pos[dim] = parts.pos[dim][i];
                        part.vel[dim] = parts.vel[dim][i];
                    }
                    part.group = parts.group[i];
                    sends[rank].push_back(part);
                }
            }
            parts = LcParticles();
            for (auto i = sends.begin(); i != sends.end(); i++) {
                futs.push_back(hpx::async<LcSendBufferParticlesAction>(
                    hpxLocalities()[i->first], std::move(i->second)));
            }
            hpx::wait_all(futs.begin(), futs.end());
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());
}

static int lcParticlesSort(int pix, Pair<int> rng, double xm, int xdim) {
    int begin = rng.first;
    int end = rng.second;
    int lo = begin;
    int hi = end;
    auto& partMap = *partMapPtr;
    auto& parts = partMap[pix];
    LcReal xmid = xm;
    while (lo < hi) {
        if (parts.pos[xdim][lo] >= xmid) {
            while (lo != hi) {
                hi--;
                if (parts.pos[xdim][hi] < xmid) {
                    for (int dim = 0; dim < NDIM; dim++) {
                        std::swap(parts.pos[dim][hi], parts.pos[dim][lo]);
                        std::swap(parts.vel[dim][hi], parts.vel[dim][lo]);
                    }
                    std::swap(parts.group[hi], parts.group[lo]);
                    break;
                }
            }
        }
        lo++;
    }
    return hi;
}

std::pair<int, Range<double>> lcTreeCreate(int pix, Range<double> box, Pair<int> partRange) {
    constexpr int kBucketSize = GROUP_BUCKET_SIZE;
    LcTreeNode thisNode;
    auto& partMap = *partMapPtr;
    auto& treeMap = *treeMapPtr;
    auto& nodes = treeMap[pix];
    auto& parts = partMap[pix];
    Range<double> partBox;
    bool leaf;
    if (partRange.second - partRange.first > kBucketSize) {
        leaf = false;
        int const xdim = box.longestDim();
        double const xmid = 0.5 * (box.begin[xdim] + box.end[xdim]);
        int const mid = lcParticlesSort(pix, partRange, xmid, xdim);
        auto boxLeft = box;
        auto boxRight = box;
        auto partsLeft = partRange;
        auto partsRight = partRange;
        boxLeft.end[xdim] = boxRight.begin[xdim] = xmid;
        partsLeft.second = partsRight.first = mid;
        auto const rcl = lcTreeCreate(pix, boxLeft, partsLeft);
        auto const rcr = lcTreeCreate(pix, boxRight, partsRight);
        for (int dim = 0; dim < NDIM; dim++) {
            partBox.begin[dim] = std::min(rcl.second.begin[dim], rcr.second.begin[dim]);
            partBox.end[dim] = std::max(rcl.second.end[dim], rcr.second.end[dim]);
        }
        thisNode.children[LEFT].pix = thisNode.children[RIGHT].pix = pix;
        thisNode.children[LEFT].index = rcl.first;
        thisNode.children[RIGHT].index = rcr.first;
    } else {
        leaf = true;
        for (int dim = 0; dim < NDIM; dim++) {
            partBox.begin[dim] = 1.0;
            partBox.end[dim] = -1.0;
        }
        for (int i = partRange.first; i < partRange.second; i++) {
            parts.group[i] = LC_NO_GROUP;
            for (int dim = 0; dim < NDIM; dim++) {
                double const x = parts.pos[dim][i].toDouble();
                partBox.begin[dim] = std::min(partBox.begin[dim], x);
                partBox.end[dim] = std::max(partBox.end[dim], x);
            }
        }
        thisNode.children[LEFT].index = thisNode.children[RIGHT].index = -1;
        thisNode.children[LEFT].pix = thisNode.children[RIGHT].pix = -1;
    }
    thisNode.partRange = partRange;
    thisNode.box = partBox;
    thisNode.pix = pix;
    thisNode.active = 1;
    thisNode.lastActive = 0;
    int index = nodes.size();
    nodes.pushBack(thisNode);
    if (leaf) {
        LcTreeId selfid;
        selfid.index = index;
        selfid.pix = pix;
        std::lock_guard<SpinlockType> lock(leafMutex);
        leafNodes.pushBack(selfid);
    }
    std::pair<int, Range<double>> rc;
    rc.first = index;
    rc.second = partBox;
    return rc;
}

void lcFindNeighborsLocal(LcTreeId selfId, Vector<LcTreeId> checklist, double linkLen) {
    static thread_local Vector<LcTreeId> nextlist;
    static thread_local Vector<LcTreeId> leaflist;
    nextlist.resize(0);
    leaflist.resize(0);
    auto& treeMap = *treeMapPtr;
    auto& self = treeMap[selfId.pix][selfId.index];
    bool const iamleaf = self.children[LEFT].index == -1;
    auto mybox = self.box.pad(linkLen * 1.01);
    do {
        for (int ci = 0; ci < checklist.size(); ci++) {
            auto const check = checklist[ci];
            auto const& other = treeMap[check.pix][check.index];
            if (mybox.intersection(other.box).volume() > 0) {
                if (other.children[LEFT].index == -1) {
                    leaflist.push_back(check);
                } else {
                    nextlist.push_back(other.children[LEFT]);
                    nextlist.push_back(other.children[RIGHT]);
                }
            }
        }
        std::swap(nextlist, checklist);
        nextlist.resize(0);
    } while (iamleaf && checklist.size());
    if (iamleaf) {
        self.neighbors.resize(leaflist.size());
        memcpy(self.neighbors.data(), leaflist.data(), sizeof(LcTreeId) * leaflist.size());
    } else {
        checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
        if (checklist.size()) {
            lcFindNeighborsLocal(self.children[LEFT], checklist, linkLen);
            lcFindNeighborsLocal(self.children[RIGHT], std::move(checklist), linkLen);
        }
    }
}

size_t lcFindNeighbors() {
    auto& treeMap = *treeMapPtr;
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<LcFindNeighborsAction>(c));
    }
    size_t rc = 0;
    double const linkLen = getOptions().lcB * pow(getOptions().nparts, -1.0 / NDIM);
    for (int pix = myPixRange.first; pix < myPixRange.second; pix++) {
        futs.push_back(hpx::async([pix, linkLen, &treeMap]() {
            Vector<int> checkPix;
            checkPix = pixNeighbors(pix);
            checkPix.push_back(pix);
            Vector<LcTreeId> checklist(checkPix.size());
            for (int i = 0; i < checkPix.size(); i++) {
                checklist[i].pix = checkPix[i];
                checklist[i].index = treeMap[checkPix[i]].size() - 1;
            }
            lcFindNeighborsLocal(checklist.back(), checklist, linkLen);
        }));
    }
    for (auto& f : futs) {
        f.get();
    }
    return rc;
}

size_t lcFindGroups() {
    auto& treeMap = *treeMapPtr;
    Vector<hpx::future<size_t>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<LcFindGroupsAction>(c));
    }
    size_t rc = 0;
    double const linkLen = getOptions().lcB * pow(getOptions().nparts, -1.0 / NDIM);
    Vector<hpx::future<void>> futs2;
    for (int pix = myPixRange.first; pix < myPixRange.second; pix++) {
        futs2.push_back(hpx::async([pix, &treeMap]() {
            auto& nodes = treeMap[pix];
            for (int i = 0; i < nodes.size(); i++) {
                nodes[i].lastActive = nodes[i].active;
                nodes[i].active = 0;
            }
        }));
    }
    hpx::wait_all(futs2.begin(), futs2.end());

    return cudaLightcone(leafNodes, partMapPtr, treeMapPtr, &groupIdCounter);
}

static LcParticles lcGetParticles1(int pix) {
    auto& partMap = *partMapPtr;
    return partMap[pix];
}

static Pair<Vector<long long>, Vector<char>> lcGetParticles2(int pix) {
    auto& partMap = *partMapPtr;
    auto& treeMap = *treeMapPtr;
    Pair<Vector<long long>, Vector<char>> rc;
    auto const& parts = partMap[pix];
    for (int i = 0; i < parts.group.size(); i++) {
        rc.first.push_back((long long) parts.group[i]);
    }
    auto const& nodes = treeMap[pix];
    for (int i = 0; i < nodes.size(); i++) {
        rc.second.push_back(nodes[i].active);
    }
    return rc;
}

static int vec2pix(double x, double y, double z) {
    if (nsideValue == 0) {
        return 0;
    } else {
        vec3 vec;
        vec.x = x;
        vec.y = y;
        vec.z = z;
        return healpix->vec2pix(vec);
    }
}

static int pix2rank(int pix) {
    int n = (size_t) pix * hpxSize() / npixValue;
    while (pix < (size_t) n * npixValue / hpxSize()) {
        n--;
    }
    while (pix >= (size_t) (n + 1) * npixValue / hpxSize()) {
        n++;
    }
    return n;
}

void lcParticleBoundaries1() {
    auto& partMap = *partMapPtr;
    auto& treeMap = *treeMapPtr;
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<LcParticleBoundaries1Action>(c));
    }
    Vector<hpx::future<LcParticles>> pfuts;
    pfuts.reserve(bndPix.size());
    for (auto pix : bndPix) {
        int const rank = pix2rank(pix);
        pfuts.push_back(hpx::async<LcGetParticles1Action>(hpxLocalities()[rank], pix));
    }
    int i = 0;
    for (auto pix : bndPix) {
        partMap[pix] = pfuts[i].get();
        i++;
    }
}

void lcParticleBoundaries2() {
    auto& partMap = *partMapPtr;
    auto& treeMap = *treeMapPtr;
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<LcParticleBoundaries2Action>(c));
    }
    Vector<hpx::future<Pair<Vector<long long>, Vector<char>>>> pfuts;
    pfuts.reserve(bndPix.size());
    for (auto pix : bndPix) {
        int const rank = pix2rank(pix);
        pfuts.push_back(hpx::async<LcGetParticles2Action>(hpxLocalities()[rank], pix));
    }
    int i = 0;
    for (auto pix : bndPix) {
        auto const p = pfuts[i].get();
        int const nthreads = hpxHardwareConcurrency();
        Vector<hpx::future<void>> futs;
        auto& parts = partMap[pix];
        for (int thread = 0; thread < nthreads; thread++) {
            futs.push_back(hpx::async([thread, nthreads, &p, &parts]() {
                int const jmin = thread * parts.group.size() / nthreads;
                int const jmax = (thread + 1) * parts.group.size() / nthreads;
                for (int j = jmin; j < jmax; j++) {
                    parts.group[j] = p.first[j];
                }
            }));
        }
        hpx::wait_all(futs.begin(), futs.end());
        auto& nodes = treeMap[pix];
        for (int thread = 0; thread < nthreads; thread++) {
            futs.push_back(hpx::async([thread, nthreads, &p, &nodes]() {
                int const jmin = thread * nodes.size() / nthreads;
                int const jmax = (thread + 1) * nodes.size() / nthreads;
                for (int j = jmin; j < jmax; j++) {
                    nodes[j].lastActive = p.second[j];
                }
            }));
        }
        hpx::wait_all(futs.begin(), futs.end());
        i++;
    }

    hpx::wait_all(futs.begin(), futs.end());
}

void lcFormTrees(double tau, double linkLen) {
    auto& partMap = *partMapPtr;
    auto& treeMap = *treeMapPtr;
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<LcFormTreesAction>(c, tau, linkLen));
    }
    for (auto iter = partMap.begin(); iter != partMap.end(); iter++) {
        int const pix = iter->first;
        futs.push_back(hpx::async([pix, linkLen, tau, &partMap]() {
            auto& parts = partMap[pix];
            Range<double> box;
            Pair<int> partRange;
            partRange.first = 0;
            partRange.second = parts.group.size();
            for (int dim = 0; dim < NDIM; dim++) {
                box.begin[dim] = 1.0;
                box.end[dim] = -1.0;
            }
            for (int i = 0; i < parts.group.size(); i++) {
                for (int dim = 0; dim < NDIM; dim++) {
                    auto const x = parts.pos[dim][i].toDouble();
                    box.begin[dim] = std::min(box.begin[dim], x);
                    box.end[dim] = std::max(box.end[dim], x);
                }
            }
            lcTreeCreate(pix, box, partRange);
            for (int i = 0; i < parts.group.size(); i++) {
                double const x = parts.pos[XDIM][i].toDouble();
                double const y = parts.pos[YDIM][i].toDouble();
                double const z = parts.pos[ZDIM][i].toDouble();
                double const r2 = sqr(x, y, z);
                double const rValue = sqrt(r2);
                if (sqr((float) (rValue - (tauMax - tau))) < (float) sqr(linkLen) && tau < tauMax) {
                    parts.group[i] = LC_EDGE_GROUP;
                } else {
                    parts.group[i] = LC_NO_GROUP;
                }
                if (rValue + 1e-4 < tauMax - tau) {
                    PRINT("-----> low R %e %e\n", rValue, tauMax - tau);
                    //		ASSERT(R >= tau_max - tau);
                }
            }
        }));
    }
    std::sort(leafNodes.begin(), leafNodes.end(), [&treeMap](LcTreeId a, LcTreeId b) {
        if (a.pix < b.pix) {
            return true;
        } else if (a.pix > b.pix) {
            return false;
        } else {
            auto const aValue = treeMap[a.pix][a.index];
            auto const bValue = treeMap[b.pix][b.index];
            return aValue.partRange.first < bValue.partRange.first;
        }
    });
    hpx::wait_all(futs.begin(), futs.end());
}

static Vector<int> pixNeighbors(int pix) {
    Vector<int> neighbors;
    if (nsideValue != 0) {
        neighbors.reserve(8);
        fix_arr<int, 8> result;
        healpix->neighbors(pix, result);
        for (int i = 0; i < 8; i++) {
            if (result[i] != -1) {
                neighbors.push_back(result[i]);
            }
        }
    }
    return neighbors;
}

static int computeNside(double tau) {
    double nside = std::sqrt(8.0 * hpxHardwareConcurrency() * hpxSize() / 12.0);
    double const w0 = std::max((tauMax - tau) / tauMax, 0.0);
    nside = std::min(
        nside, (pow(getOptions().nparts, 1.0 / NDIM) * std::sqrt(3) / getOptions().lcB / 2.0 * w0));
    nsideValue = 1;
    while (nsideValue < nside) {
        nsideValue *= 2;
    }
    nsideValue /= 2;
    //	PRINT("compute_nside = %i\n", Nside);
    return nsideValue;
}

void lcInit(double tau, double requestedTauMax) {
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<LcInitAction>(c, tau, requestedTauMax));
    }
    if (partMapPtr == nullptr) {
        CUDA_CHECK(cudaMallocManaged(&partMapPtr, sizeof(LcPartMapType)));
        CUDA_CHECK(cudaMallocManaged(&treeMapPtr, sizeof(LcTreeMapType)));
        new (partMapPtr) LcPartMapType;
        new (treeMapPtr) LcTreeMapType;
    }
    auto& partMap = *partMapPtr;
    auto& treeMap = *treeMapPtr;
    mutexMap = decltype(mutexMap)();
    bndPix = decltype(bndPix)();
    leafNodes = decltype(leafNodes)();
    tauMax = requestedTauMax;
    nsideValue = computeNside(tau);
    npixValue = nsideValue == 0 ? 1 : 12 * sqr(nsideValue);
    if (nsideValue > 0) {
        healpix = std::make_shared<HealpixType>(nsideValue, NEST, SET_NSIDE);
    } else {
        healpix = nullptr;
    }
    myPixRange.first = (size_t) hpxRank() * npixValue / hpxSize();
    myPixRange.second = (size_t) (hpxRank() + 1) * npixValue / hpxSize();
    for (int pix = myPixRange.first; pix < myPixRange.second; pix++) {
        partMap[pix].group.resize(0);
        for (int dim = 0; dim < NDIM; dim++) {
            partMap[pix].pos[dim].resize(0);
            partMap[pix].vel[dim].resize(0);
        }
        treeMap[pix].resize(0);
        mutexMap[pix] = std::make_shared<SpinlockType>();
        auto const neighbors = pixNeighbors(pix);
        for (auto const& n : neighbors) {
            if (n < myPixRange.first || n >= myPixRange.second) {
                bndPix.insert(n);
                partMap[n].group.resize(0);
                for (int dim = 0; dim < NDIM; dim++) {
                    partMap[n].pos[dim].resize(0);
                    partMap[n].vel[dim].resize(0);
                }
                treeMap[n].resize(0);
            }
        }
    }
    hpx::wait_all(futs.begin(), futs.end());
}

int lcAddParticle(double x0, double y0, double z0, double x1, double y1, double z1, float vx,
    float vy, float vz, float t0, float t1, Vector<LcParticle>& thisPartBuffer) {
    int rc = 0;
    /*static simd_float8 images[NDIM] =
     { simd_float8(0, -1, 0, -1, 0, -1, 0, -1), simd_float8(0, 0, -1, -1, 0, 0, -1, -1),
     simd_float8(0, 0, 0, 0, -1, -1, -1, -1) }; const float tau_max_inv = 1.0 / tau_max; const
     simd_float8 simd_c0 = simd_float8(tau_max_inv); array<simd_float8, NDIM> X0; array<simd_float8,
     NDIM> X1; const simd_float8 simd_tau0 = simd_float8(t0); const simd_float8 simd_tau1 =
     simd_float8(t1); simd_float8 dist0; simd_float8 dist1; X0[XDIM] = simd_float8(x0) +
     images[XDIM]; X0[YDIM] = simd_float8(y0) + images[YDIM]; X0[ZDIM] = simd_float8(z0) +
     images[ZDIM]; X1[XDIM] = simd_float8(x1) + images[XDIM]; X1[YDIM] = simd_float8(y1) +
     images[YDIM]; X1[ZDIM] = simd_float8(z1) + images[ZDIM]; dist0 = sqrt(sqr(X0[0], X0[1],
     X0[2])); dist1 = sqrt(sqr(X1[0], X1[1], X1[2])); simd_float8 tau0 = simd_tau0 + dist0;
     simd_float8 tau1 = simd_tau1 + dist1;
     simd_int8 I0 = tau0 * simd_c0;
     simd_int8 I1 = tau1 * simd_c0;
     for (int ci = 0; ci < SIMD_FLOAT8_SIZE; ci++) {
     if (dist1[ci] <= 1.0 || dist0[ci] <= 1.0) {
     static const int map_nside = get_options().lc_map_size;
     const int i0 = I0[ci];
     const int i1 = I1[ci];
     if (i0 != i1) {
     x0 = X0[XDIM][ci];
     y0 = X0[YDIM][ci];
     z0 = X0[ZDIM][ci];
     const double ti = (i0 + 1) * tau_max;
     const double sqrtauimtau0 = sqr(ti - t0);
     const double tau0mtaui = t0 - ti;
     const double u2 = sqr(vx, vy, vz);                                    // 5
     const double x2 = sqr(x0, y0, z0);                                       // 5
     const double udotx = vx * x0 + vy * y0 + vz * z0;               // 5
     const double A = 1.f - u2;                                                     // 1
     const double B = 2.0 * (tau0mtaui - udotx);                                    // 2
     const double C = sqrtauimtau0 - x2;                                            // 1
     const double t = -(B + sqrt(B * B - 4.f * A * C)) / (2.f * A);                // 15
     const double x1 = x0 + vx * t;                                            // 2
     const double y1 = y0 + vy * t;                                            // 2
     const double z1 = z0 + vz * t;                                            // 2
     long int ipix;
     if (sqr(x1, y1, z1) <= 1.f) {                                                 // 6
     rc++;
     const auto pix = vec2pix(x1, y1, z1);
     lc_particle part;
     part.pos[XDIM] = x1;
     part.pos[YDIM] = y1;
     part.pos[ZDIM] = z1;
     part.vel[XDIM] = vx;
     part.vel[YDIM] = vy;
     part.vel[ZDIM] = vz;
     this_part_buffer.push_back(part);
     }
     }
     }
     }*/
    return rc;
}

void lcAddParts(Vector<LcParticle>&& theseParts) {
    std::unique_lock<SharedMutexType> ulock(sharedMutex);
    int const start = partBuffer.size();
    partBuffer.resize(start + theseParts.size());
    int const stop = partBuffer.size();
    ulock.unlock();
    int const chunkSize = 64;
    for (int i = 0; i < theseParts.size(); i += chunkSize) {
        std::shared_lock<SharedMutexType> slock(sharedMutex);
        int const jend = std::min(i + chunkSize, (int) theseParts.size());
        for (int j = i; j < jend; j++) {
            partBuffer[j + start] = theseParts[j];
        }
    }
}

void lcSiloOut(char const* filename, Vector<Subgroup> const& subgroups) {
    Vector<float> x, y, z, vx, vy, vz;
    Vector<int> id;
    Vector<int> hostid;
    for (int i = 0; i < subgroups.size(); i++) {
        for (int j = 0; j < subgroups[i].parts.size(); j++) {
            x.push_back(subgroups[i].parts[j].x);
            y.push_back(subgroups[i].parts[j].y);
            z.push_back(subgroups[i].parts[j].z);
            vx.push_back(subgroups[i].parts[j].vx);
            vy.push_back(subgroups[i].parts[j].vy);
            vz.push_back(subgroups[i].parts[j].vz);
            id.push_back(i);
            int k = i;
            int l;
            while (k != -1) {
                l = k;
                k = subgroups[k].parent;
            }

            hostid.push_back(l);
        }
    }
    if (x.size()) {
        DBfile* db = DBCreateReal(filename, DB_CLOBBER, DB_LOCAL, "Meshless", DB_PDB);
        float* coords[] = {x.data(), y.data(), z.data()};
        DBPutPointmesh(db, "points", NDIM, coords, x.size(), DB_FLOAT, NULL);
        DBPutPointvar1(db, "vx", "points", x.data(), vx.size(), DB_FLOAT, NULL);
        DBPutPointvar1(db, "vy", "points", y.data(), vy.size(), DB_FLOAT, NULL);
        DBPutPointvar1(db, "vz", "points", z.data(), vz.size(), DB_FLOAT, NULL);
        DBPutPointvar1(db, "id", "points", id.data(), id.size(), DB_INT, NULL);
        DBPutPointvar1(db, "hostid", "points", hostid.data(), hostid.size(), DB_INT, NULL);
        DBClose(db);
    }
}
