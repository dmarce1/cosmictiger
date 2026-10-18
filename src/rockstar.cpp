#include <cosmictiger/bh.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/cosmology.hpp>
#include <cosmictiger/device_vector.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/lightcone.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/rockstar_core.hpp>
#include <cosmictiger/simulation_units.hpp>
#include <cosmictiger/timer.hpp>

void rockstarAssignLinkLen(DeviceVector<RockstarTree> const& trees,
    DeviceVector<RockstarParticle>& parts, int selfId, Vector<int> checklist, float linkingLength,
    bool phase) {
    static thread_local Vector<int> nextlist;
    static thread_local Vector<int> leaflist;
    int const dimmax = phase ? 2 * NDIM : NDIM;
    nextlist.resize(0);
    leaflist.resize(0);
    auto& self = trees[selfId];
    bool const iamleaf = self.children[LEFT] == -1;
    auto mybox = self.box.pad(linkingLength * 1.0001);
    do {
        for (int ci = 0; ci < checklist.size(); ci++) {
            auto const check = checklist[ci];
            auto const& other = trees[check];
            if (rockstar::detail::boxesOverlap(mybox, other.box, dimmax)) {
                if (other.children[LEFT] == -1) {
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
        float const linkingLengthSquared = sqr(linkingLength);
        for (int li = 0; li < leaflist.size(); li++) {
            auto const leafi = leaflist[li];
            auto const other = trees[leafi];
            for (int pi = other.particleBegin; pi < other.particleEnd; pi++) {
                auto const& part = parts[pi];
                if (rockstar::detail::boxContains(mybox, part.xValue, dimmax)) {
                    for (int pj = self.particleBegin; pj < self.particleEnd; pj++) {
                        auto& mypart = parts[pj];
                        float r2 = 0.f;
                        for (int dim = 0; dim < dimmax; dim++) {
                            r2 += sqr(mypart.xValue[dim] - part.xValue[dim]);
                        }
                        if (pi != pj && r2 <= linkingLengthSquared) {
                            mypart.minDist2 = std::min(mypart.minDist2, r2);
                        }
                    }
                }
            }
        }
    } else {
        checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
        if (checklist.size()) {
            rockstarAssignLinkLen(
                trees, parts, self.children[LEFT], checklist, linkingLength, phase);
            rockstarAssignLinkLen(
                trees, parts, self.children[RIGHT], std::move(checklist), linkingLength, phase);
        }
    }
}

void rockstarFindNeighbors(DeviceVector<int>& leaves, DeviceVector<RockstarTree>& trees, int selfId,
    Vector<int> checklist, float linkingLength, bool phase = true) {
    static thread_local Vector<int> nextlist;
    static thread_local Vector<int> leaflist;
    nextlist.resize(0);
    leaflist.resize(0);
    auto& self = trees[selfId];
    bool const iamleaf = self.children[LEFT] == -1;
    auto mybox = self.box.pad(linkingLength * 1.0001);
    do {
        for (int ci = 0; ci < checklist.size(); ci++) {
            auto const check = checklist[ci];
            auto const& other = trees[check];
            if (rockstar::detail::boxesOverlap(mybox, other.box, phase ? 2 * NDIM : NDIM)) {
                if (other.children[LEFT] == -1) {
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
        for (int i = 0; i < leaflist.size(); i++) {
            self.neighbors[i] = leaflist[i];
        }
        leaves.pushBack(selfId);
    } else {
        checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
        if (checklist.size()) {
            rockstarFindNeighbors(
                leaves, trees, self.children[LEFT], checklist, linkingLength, phase);
            rockstarFindNeighbors(
                leaves, trees, self.children[RIGHT], std::move(checklist), linkingLength, phase);
        }
    }
}

int rockstarParticlesSort(
    DeviceVector<RockstarParticle>& parts, int begin, int end, float xmid, int xdim) {
    int lo = begin;
    int hi = end;
    while (lo < hi) {
        if (parts[lo].xValue[xdim] >= xmid) {
            while (lo != hi) {
                hi--;
                if (parts[hi].xValue[xdim] < xmid) {
                    std::swap(parts[hi], parts[lo]);
                    break;
                }
            }
        }
        lo++;
    }
    return hi;
}

int rockstarFormTree(DeviceVector<RockstarParticle>& parts, DeviceVector<RockstarTree>& trees,
    Range<float, 2 * NDIM>& rng, int particleBegin, int particleEnd) {
    Array<int, NCHILD> children;
    RockstarTree node;
    int activeCount = 0;
    int const bucketSize = ROCKSTAR_BUCKET_SIZE;
    if (particleEnd - particleBegin <= bucketSize) {
        children[LEFT] = children[RIGHT] = -1;
        for (int dim = 0; dim < 2 * NDIM; dim++) {
            rng.begin[dim] = std::numeric_limits<float>::max() / 10.0;
            rng.end[dim] = -std::numeric_limits<float>::max() / 10.0;
        }
        for (int i = particleBegin; i != particleEnd; i++) {
            for (int dim = 0; dim < 2 * NDIM; dim++) {
                float const x = parts[i].xValue[dim];
                rng.begin[dim] = std::min(rng.begin[dim], x);
                rng.end[dim] = std::max(rng.end[dim], x);
            }
        }
        activeCount++;
    } else {
        float midx = rng.begin[0];
        int maxDim = 0;
        float totalMax = 0.0;
        for (int dim = 0; dim < 2 * NDIM; dim++) {
            float const xMax = rng.end[dim];
            float const xMin = rng.begin[dim];
            if (xMax - xMin > totalMax) {
                totalMax = xMax - xMin;
                maxDim = dim;
                midx = (xMax + xMin) * 0.5f;
            }
        }
        int particleMid = rockstarParticlesSort(parts, particleBegin, particleEnd, midx, maxDim);
        if (particleMid == particleBegin || particleMid == particleEnd) {
            // Coincident coordinates and rounded midpoints must still produce
            // two smaller nodes. Median partitioning also handles tied values.
            particleMid = particleBegin + (particleEnd - particleBegin) / 2;
            std::nth_element(parts.begin() + particleBegin, parts.begin() + particleMid,
                parts.begin() + particleEnd, [maxDim](auto const& a, auto const& b) {
                    return a.xValue[maxDim] < b.xValue[maxDim];
                });
            midx = parts[particleMid].xValue[maxDim];
        }
        Range<float, 2 * NDIM> rngLeft = rng;
        Range<float, 2 * NDIM> rngRight = rng;
        rngRight.begin[maxDim] = midx;
        rngLeft.end[maxDim] = midx;
        children[LEFT] = rockstarFormTree(parts, trees, rngLeft, particleBegin, particleMid);
        children[RIGHT] = rockstarFormTree(parts, trees, rngRight, particleMid, particleEnd);
        for (int dim = 0; dim < 2 * NDIM; dim++) {
            rng.begin[dim] = std::min(rngLeft.begin[dim], rngRight.begin[dim]);
            rng.end[dim] = std::max(rngLeft.end[dim], rngRight.end[dim]);
        }
        activeCount += trees[children[LEFT]].activeCount;
        activeCount += trees[children[RIGHT]].activeCount;
    }
    node.particleBegin = particleBegin;
    node.particleEnd = particleEnd;
    node.children = children;
    node.box = rng;
    node.activeCount = activeCount;
    trees.pushBack(node);
    return trees.size() - 1;
}

int rockstarFormTree(DeviceVector<RockstarTree>& trees, DeviceVector<RockstarParticle>& parts) {
    Range<float, 2 * NDIM> rng;
    for (int dim = 0; dim < 2 * NDIM; dim++) {
        rng.begin[dim] = std::numeric_limits<float>::max() / 10.0;
        rng.end[dim] = -std::numeric_limits<float>::max() / 10.0;
        for (int i = 0; i < parts.size(); i++) {
            float const x = parts[i].xValue[dim];
            rng.begin[dim] = std::min(rng.begin[dim], x);
            rng.end[dim] = std::max(rng.end[dim], x);
        }
    }
    return rockstarFormTree(parts, trees, rng, 0, parts.size());
}

float rockstarFindLinkLen(DeviceVector<RockstarTree>& trees, DeviceVector<RockstarParticle>& parts,
    int treeRoot, float ff, bool gpu) {
    if (parts.size() < 2) {
        return 0.0f;
    }
    Range<float, 2 * NDIM> rng = trees[treeRoot].box;
    //	PRINT("V %e\n", rng.volume());
    Vector<Pair<float, int>> seps;
    for (int i = 0; i < trees.size(); i++) {
        if (trees[i].children[LEFT] == -1) {
            int count = trees[i].particleEnd - trees[i].particleBegin;
            if (count) {
                float meanSep =
                    rockstar::detail::initialLinkingLength(trees[i].box, count, 2 * NDIM);
                seps.push_back(Pair<float, int>(meanSep, count));
            }
        }
    }
    std::sort(seps.begin(), seps.end(),
        [](Pair<float, int> a, Pair<float, int> b) { return a.first < b.first; });
    int ffCnt = parts.size() * ff;
    int cnt = 0;
    float maximumLinkingLength;
    for (int i = 0; i < seps.size(); i++) {
        cnt += seps[i].second;
        if (cnt >= ffCnt) {
            maximumLinkingLength = seps[i].first;
            break;
        }
    }
    //	PRINT("%e ", max_link_len);
    bool done = false;
    do {
        for (auto& p : parts) {
            p.minDist2 = std::numeric_limits<float>::max();
        }
        gpu = getOptions().compute.cuda && parts.size() > 4 * 1024;
        if (gpu) {
            DeviceVector<int> leaves;
            rockstarFindNeighbors(leaves, trees, trees.size() - 1, Vector<int>(1, trees.size() - 1),
                maximumLinkingLength);
            rockstarAssignLinklenCuda(trees, leaves, parts, maximumLinkingLength, true);
        } else {
            rockstarAssignLinkLen(
                trees, parts, treeRoot, Vector<int>(1, treeRoot), maximumLinkingLength, true);
        }
        Vector<float> dist2s(parts.size());
        for (int i = 0; i < parts.size(); i++) {
            dist2s[i] = parts[i].minDist2;
        }
        std::sort(dist2s.begin(), dist2s.end());
        int const i0 = std::min((int) (ff * parts.size()), (int) (parts.size() - 2));
        int const i1 = i0 + 1;
        if (dist2s[i1] == std::numeric_limits<float>::max()) {
            //		PRINT( "doubling\n");
            if (maximumLinkingLength == 0.0f) {
                maximumLinkingLength = rockstar::detail::initialLinkingLength(
                    trees[treeRoot].box, parts.size(), 2 * NDIM);
            } else {
                maximumLinkingLength *= 2.0;
            }
        } else {
            maximumLinkingLength = (sqrt(dist2s[i0]) + sqrt(dist2s[i1])) * 0.5f;
            done = true;
        }
    } while (!done);
    //	PRINT("%e\n", max_link_len);
    return maximumLinkingLength;
}

void rockstarFindAllLinkLens(
    DeviceVector<RockstarTree>& trees, DeviceVector<RockstarParticle>& parts, int treeRoot) {
    for (auto& p : parts) {
        p.minDist2 = std::numeric_limits<float>::max();
    }
    double maximumLinkingLength = getOptions().lightcone.linkingLength *
        pow(getOptions().initial.numberOfParticles, -1.0 / NDIM);
    bool gpu = getOptions().compute.cuda && parts.size() > 4 * 1024;
    if (gpu) {
        DeviceVector<int> leaves;
        rockstarFindNeighbors(leaves, trees, trees.size() - 1, Vector<int>(1, trees.size() - 1),
            maximumLinkingLength, false);
        rockstarAssignLinklenCuda(trees, leaves, parts, maximumLinkingLength, false);
    } else {
        rockstarAssignLinkLen(
            trees, parts, treeRoot, Vector<int>(1, treeRoot), maximumLinkingLength, false);
    }
}

size_t rockstarFindSubgroups(DeviceVector<RockstarTree>& trees,
    DeviceVector<RockstarParticle>& parts, int selfId, Vector<int> checklist, float linkingLength,
    int& nextId) {
    static thread_local Vector<int> nextlist;
    static thread_local Vector<int> leaflist;
    nextlist.resize(0);
    leaflist.resize(0);
    auto& self = trees[selfId];
    bool const iamleaf = self.children[LEFT] == -1;
    auto mybox = self.box.pad(linkingLength * 1.0001);
    do {
        for (int ci = 0; ci < checklist.size(); ci++) {
            auto const check = checklist[ci];
            auto const& other = trees[check];
            if (other.lastActive) {
                if (rockstar::detail::boxesOverlap(mybox, other.box, 2 * NDIM)) {
                    if (other.children[LEFT] == -1) {
                        leaflist.push_back(check);
                    } else {
                        nextlist.push_back(other.children[LEFT]);
                        nextlist.push_back(other.children[RIGHT]);
                    }
                }
            }
        }
        std::swap(nextlist, checklist);
        nextlist.resize(0);
    } while (iamleaf && checklist.size());
    if (iamleaf) {
        bool foundAnyLink = false;
        float const linkingLengthSquared = sqr(linkingLength);
        for (int li = 0; li < leaflist.size(); li++) {
            if (selfId != leaflist[li]) {
                auto const leafi = leaflist[li];
                auto const other = trees[leafi];
                for (int pi = other.particleBegin; pi < other.particleEnd; pi++) {
                    auto& part = parts[pi];
                    if (rockstar::detail::boxContains(mybox, part.xValue, 2 * NDIM)) {
                        for (int pj = self.particleBegin; pj < self.particleEnd; pj++) {
                            auto& mypart = parts[pj];
                            float const dx = mypart.x - part.x;
                            float const dy = mypart.y - part.y;
                            float const dz = mypart.z - part.z;
                            float const dvx = mypart.vx - part.vx;
                            float const dvy = mypart.vy - part.vy;
                            float const dvz = mypart.vz - part.vz;
                            float const r2 = sqr(dx, dy, dz) + sqr(dvx, dvy, dvz);
                            if (r2 <= linkingLengthSquared) {
                                if (mypart.subgroup == ROCKSTAR_NO_GROUP) {
                                    mypart.subgroup = nextId++;
                                    foundAnyLink = true;
                                }
                                if (part.subgroup == ROCKSTAR_NO_GROUP) {
                                    part.subgroup = nextId++;
                                    foundAnyLink = true;
                                }
                                if (mypart.subgroup > part.subgroup) {
                                    mypart.subgroup = part.subgroup;
                                    foundAnyLink = true;
                                } else if (mypart.subgroup < part.subgroup) {
                                    part.subgroup = mypart.subgroup;
                                    foundAnyLink = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        bool foundLink;
        do {
            foundLink = false;
            for (int pi = self.particleBegin; pi < self.particleEnd; pi++) {
                auto& aValue = parts[pi];
                for (int pj = pi + 1; pj < self.particleEnd; pj++) {
                    auto& bValue = parts[pj];
                    float const dx = aValue.x - bValue.x;
                    float const dy = aValue.y - bValue.y;
                    float const dz = aValue.z - bValue.z;
                    float const dvx = aValue.vx - bValue.vx;
                    float const dvy = aValue.vy - bValue.vy;
                    float const dvz = aValue.vz - bValue.vz;
                    float const r2 = sqr(dx, dy, dz) + sqr(dvx, dvy, dvz);
                    if (r2 <= linkingLengthSquared) {
                        if (aValue.subgroup == ROCKSTAR_NO_GROUP) {
                            aValue.subgroup = nextId++;
                            foundAnyLink = true;
                            foundLink = true;
                        }
                        if (bValue.subgroup == ROCKSTAR_NO_GROUP) {
                            bValue.subgroup = nextId++;
                            foundAnyLink = true;
                            foundLink = true;
                        }
                        if (aValue.subgroup != bValue.subgroup) {
                            if (aValue.subgroup < bValue.subgroup) {
                                bValue.subgroup = aValue.subgroup;
                            } else {
                                aValue.subgroup = bValue.subgroup;
                            }
                            foundAnyLink = true;
                            foundLink = true;
                        }
                    }
                }
            }
        } while (foundLink);
        self.active = foundAnyLink;
        return int(foundAnyLink);
    } else {
        checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
        int numberOfActiveLeaves = 0;
        if (checklist.size()) {
            numberOfActiveLeaves += rockstarFindSubgroups(
                trees, parts, self.children[LEFT], checklist, linkingLength, nextId);
            numberOfActiveLeaves += rockstarFindSubgroups(
                trees, parts, self.children[RIGHT], std::move(checklist), linkingLength, nextId);
        }
        self.active = numberOfActiveLeaves;
        return numberOfActiveLeaves;
    }
}

void rockstarFindSubgroups(DeviceVector<RockstarTree>& trees, DeviceVector<RockstarParticle>& parts,
    float linkingLength, int& nextId, bool gpu) {
    Timer tm;
    tm.start();
    for (int i = 0; i < parts.size(); i++) {
        parts[i].subgroup = ROCKSTAR_NO_GROUP;
    }
    for (int i = 0; i < trees.size(); i++) {
        trees[i].active = true;
    }
    int cnt;
    gpu = getOptions().compute.cuda && parts.size() > 4 * 1024;
    if (gpu) {
        DeviceVector<int> leaves;
        rockstarFindNeighbors(
            leaves, trees, trees.size() - 1, Vector<int>(1, trees.size() - 1), linkingLength);
        rockstarFindSubgroupsCuda(trees, leaves, parts, linkingLength, nextId);
    } else {
        do {
            double pctActive = 0.0;
            for (int i = 0; i < trees.size(); i++) {
                trees[i].lastActive = trees[i].active;
                pctActive += int(trees[i].active) / (double) trees.size();
            }
            int rootId = trees.size() - 1;
            cnt = rockstarFindSubgroups(
                trees, parts, rootId, Vector<int>(1, rootId), linkingLength, nextId);
            //		PRINT("%i %i %e\n", root_id + 1, parts.size(), pct_active);
        } while (cnt != 0);
        tm.stop();
    }
    //	PRINT( "find_subgroups = %e\n", tm.read());
}

struct Number {
    int n;
    Number() {
        n = 0;
    }
};

std::unordered_map<int, Number> rockstarSubgroupCounts(DeviceVector<RockstarParticle>& parts) {
    std::unordered_map<int, Number> table;
    for (int i = 0; i < parts.size(); i++) {
        if (parts[i].subgroup != ROCKSTAR_NO_GROUP) {
            table[parts[i].subgroup].n++;
        }
    }
    for (int i = 0; i < parts.size(); i++) {
        if (parts[i].subgroup != ROCKSTAR_NO_GROUP) {
            if (table[parts[i].subgroup].n < ROCKSTAR_MIN_GROUP) {
                parts[i].subgroup = ROCKSTAR_NO_GROUP;
            }
        }
    }
    for (auto i = table.begin(); i != table.end();) {
        if (i->second.n < ROCKSTAR_MIN_GROUP) {
            i = table.erase(i);
        } else {
            i++;
        }
    }
    return table;
}

struct RockstarSeed {
    int id;
    Array<float, 2 * NDIM> x;
};

Vector<Subgroup> rockstarSeeds(DeviceVector<RockstarParticle> parts, int& nextId, float rfac,
    float vfac, float scale, bool gpu, int depth = 0) {
    Vector<Subgroup> subgroups;

    float avgX = 0.0;
    float avgY = 0.0;
    float avgZ = 0.0;
    float avgVx = 0.0;
    float avgVy = 0.0;
    float avgVz = 0.0;

    for (int i = 0; i < parts.size(); i++) {
        avgX += parts[i].x;
        avgY += parts[i].y;
        avgZ += parts[i].z;
        avgVx += parts[i].vx;
        avgVy += parts[i].vy;
        avgVz += parts[i].vz;
    }
    avgX /= parts.size();
    avgY /= parts.size();
    avgZ /= parts.size();
    avgVx /= parts.size();
    avgVy /= parts.size();
    avgVz /= parts.size();

    float sigma2X = 0.0;
    float sigma2V = 0.0;
    for (int i = 0; i < parts.size(); i++) {
        parts[i].x -= avgX;
        parts[i].y -= avgY;
        parts[i].z -= avgZ;
        parts[i].vx -= avgVx;
        parts[i].vy -= avgVy;
        parts[i].vz -= avgVz;
        sigma2X += sqr(parts[i].x);
        sigma2X += sqr(parts[i].y);
        sigma2X += sqr(parts[i].z);
        sigma2V += sqr(parts[i].vx);
        sigma2V += sqr(parts[i].vy);
        sigma2V += sqr(parts[i].vz);
    }
    sigma2X /= parts.size();
    sigma2V /= parts.size();
    float sigmaX = rockstar::detail::normalizationScale(sigma2X);
    float sigmaV = rockstar::detail::normalizationScale(sigma2V);
    float sigmainvX = 1.0 / sigmaX;
    float sigmainvV = 1.0 / sigmaV;
    for (int i = 0; i < parts.size(); i++) {
        parts[i].x *= sigmainvX;
        parts[i].y *= sigmainvX;
        parts[i].z *= sigmainvX;
        parts[i].vx *= sigmainvV;
        parts[i].vy *= sigmainvV;
        parts[i].vz *= sigmainvV;
    }
    rfac *= sigmainvX;
    vfac *= sigmainvV;
    DeviceVector<RockstarTree> trees;
    std::unordered_map<int, Number> groupCounts;
    if (parts.size() > ROCKSTAR_MIN_GROUP) {
        int const rootId = rockstarFormTree(trees, parts);
        float const linkingLength = rockstarFindLinkLen(trees, parts, rootId, ROCKSTAR_FF, gpu);
        rockstarFindSubgroups(trees, parts, linkingLength, nextId, gpu);
        groupCounts = rockstarSubgroupCounts(parts);
        if (groupCounts.size() == 1 && groupCounts.begin()->second.n == parts.size()) {
            // Tied nearest-neighbor distances can retain the entire parent.
            // Such a level contains one seed and cannot be refined further.
            groupCounts.clear();
        }
    }

    for (auto i = groupCounts.begin(); i != groupCounts.end(); i++) {
        DeviceVector<RockstarParticle> theseParticles;
        int j = 0;
        while (j < parts.size()) {
            if (parts[j].subgroup == i->first) {
                theseParticles.pushBack(parts[j]);
                parts[j] = parts.back();
                parts.popBack();
            } else {
                j++;
            }
        }
        auto theseGroups = rockstarSeeds(theseParticles, nextId, rfac, vfac, scale, gpu, depth + 1);
        subgroups.insert(subgroups.end(), theseGroups.begin(), theseGroups.end());
    }

    auto const findSigmas = [rfac, vfac, scale](Subgroup& group) {
        rockstarUpdateStatistics(group, rfac, vfac, scale);
    };
    if (subgroups.size() == 0) {
        int const subgrp = nextId++;
        for (int i = 0; i < parts.size(); i++) {
            parts[i].subgroup = subgrp;
        }
        Subgroup sg;
        sg.parts = std::move(parts);
        sg.id = subgrp;
        sg.sigmaXMin = std::numeric_limits<float>::max();
        findSigmas(sg);
        subgroups.push_back(sg);
    } else if (subgroups.size() == 1) {
        int const subgrp = subgroups[0].parts[0].subgroup;
        for (int i = 0; i < parts.size(); i++) {
            parts[i].subgroup = subgrp;
            subgroups[0].parts.pushBack(parts[i]);
        }
        findSigmas(subgroups[0]);
    } else {
        bool foundMerge;
        // PRINT("Finding merges for %i subgroups\n", subgroups.size());
        for (int k = 0; k < subgroups.size();) {
            foundMerge = false;
            std::sort(subgroups.begin(), subgroups.end(), [](Subgroup const& a, Subgroup const& b) {
                return a.parts.size() < b.parts.size();
            });
            for (int l = k + 1; l < subgroups.size(); l++) {
                float const sigmaXInv = sqrt(subgroups[k].parts.size() / subgroups[k].sigma2X);
                float const sigmaVInv = sqrt(subgroups[k].parts.size() / subgroups[k].sigma2V);
                float const dx = (subgroups[k].x - subgroups[l].x) * sigmaXInv;
                float const dy = (subgroups[k].y - subgroups[l].y) * sigmaXInv;
                float const dz = (subgroups[k].z - subgroups[l].z) * sigmaXInv;
                float const dvx = (subgroups[k].vxc - subgroups[l].vxc) * sigmaVInv;
                float const dvy = (subgroups[k].vyc - subgroups[l].vyc) * sigmaVInv;
                float const dvz = (subgroups[k].vzc - subgroups[l].vzc) * sigmaVInv;
                if (sqr(dx, dy, dz) + sqr(dvx, dvy, dvz) < 200.0) {
                    for (int i = 0; i < subgroups[k].parts.size(); i++) {
                        subgroups[l].parts.pushBack(subgroups[k].parts[i]);
                        subgroups[l].parts.back().subgroup = subgroups[l].parts[0].subgroup;
                    }
                    findSigmas(subgroups[l]);
                    subgroups[k] = subgroups.back();
                    subgroups.pop_back();
                    foundMerge = true;
                    break;
                }
            }
            if (foundMerge) {
                // A merged seed has a new center and dispersion; earlier
                // candidates must be checked against that updated seed too.
                k = 0;
            } else {
                k++;
            }
        }
        for (int j = 0; j < parts.size(); j++) {
            float minDist = std::numeric_limits<float>::max();
            int minIndex = -1;
            for (int i = 0; i < subgroups.size(); i++) {
                assert(subgroups[i].parts.size());
                float const rdynInv = 1.0 / subgroups[i].rDyn;
                float const sigmaVInv = 1.0 / sqrt(subgroups[i].sigma2V);
                float const dx = (parts[j].x - subgroups[i].x) * rdynInv;
                float const dy = (parts[j].y - subgroups[i].y) * rdynInv;
                float const dz = (parts[j].z - subgroups[i].z) * rdynInv;
                float const dvx = (parts[j].vx - subgroups[i].vxc) * sigmaVInv;
                float const dvy = (parts[j].vy - subgroups[i].vyc) * sigmaVInv;
                float const dvz = (parts[j].vz - subgroups[i].vzc) * sigmaVInv;
                float const dist = sqrt(sqr(dx, dy, dz) + sqr(dvx, dvy, dvz));
                // PRINT("dist = %e mindist = %e %e %e %e %e %e %e \n", dist, min_dist, dx, dy, dz,
                // dvz, dvy, subgroups[i].sigma2_v);
                if (dist < minDist) {
                    minDist = dist;
                    minIndex = i;
                }
            }
            parts[j].subgroup = subgroups[minIndex].parts[0].subgroup;
            subgroups[minIndex].parts.pushBack(parts[j]);
        }
        for (int k = 0; k < subgroups.size(); k++) {
            findSigmas(subgroups[k]);
        }
    }
    for (int k = 0; k < subgroups.size(); k++) {
        auto& parts = subgroups[k].parts;
        auto& sg = subgroups[k];
        sg.x *= sigmaX;
        sg.y *= sigmaX;
        sg.z *= sigmaX;
        sg.vxc *= sigmaV;
        sg.vyc *= sigmaV;
        sg.vzc *= sigmaV;
        sg.vxb *= sigmaV;
        sg.vyb *= sigmaV;
        sg.vzb *= sigmaV;
        sg.x += avgX;
        sg.y += avgY;
        sg.z += avgZ;
        sg.vxb += avgVx;
        sg.vyb += avgVy;
        sg.vzb += avgVz;
        sg.vxc += avgVx;
        sg.vyc += avgVy;
        sg.vzc += avgVz;
        sg.rDyn *= sigmaX;
        sg.rVir *= sigmaX;
        sg.sigma2V *= sigmaV * sigmaV;
        sg.sigma2X *= sigmaX * sigmaX;
        sg.sigmaXMin *= sigmaX;
        sg.vcircMax *= sigmaV;
        for (int i = 0; i < parts.size(); i++) {
            parts[i].x *= sigmaX;
            parts[i].y *= sigmaX;
            parts[i].z *= sigmaX;
            parts[i].vx *= sigmaV;
            parts[i].vy *= sigmaV;
            parts[i].vz *= sigmaV;
        }
        for (int i = 0; i < parts.size(); i++) {
            parts[i].x += avgX;
            parts[i].y += avgY;
            parts[i].z += avgZ;
            parts[i].vx += avgVx;
            parts[i].vy += avgVy;
            parts[i].vz += avgVz;
        }
    }
    if (depth == 0) {
        rockstarFinalizeSubgroups(subgroups, scale);
    }
    return subgroups;
}

Vector<Subgroup> rockstarFindSubgroups(Vector<ParticleData> const& parts, float scale) {
    if (parts.size() < ROCKSTAR_MIN_GROUP) {
        return {};
    }
    DeviceVector<RockstarParticle> particles;
    for (auto const& input : parts) {
        RockstarParticle particle{};
        for (int dim = 0; dim < NDIM; ++dim) {
            particle.xValue[dim] = distance(input.x[dim], parts[0].x[dim]);
            particle.xValue[dim + NDIM] = input.v[dim];
        }
        particle.index = input.index;
        particle.rank = input.rank;
        particle.identity = input.identity;
        particle.previousHalo = input.lastGroup;
        particle.previousHost = input.lastHost;
        particles.pushBack(particle);
    }
    int nextId = 1;
    return rockstarSeeds(
        std::move(particles), nextId, 1.0f, 1.0f, scale, getOptions().compute.cuda);
}

Vector<RockstarRecord> rockstarFindSubgroups(Vector<LightconeEntry> const& input, double tauMax) {
    Vector<RockstarRecord> records;
    if (input.size() < ROCKSTAR_MIN_GROUP) {
        return records;
    }
    Array<double, NDIM> origin{};
    for (auto const& particle : input) {
        origin[0] += particle.x.toDouble() / input.size();
        origin[1] += particle.y.toDouble() / input.size();
        origin[2] += particle.z.toDouble() / input.size();
    }
    double const distanceToGroup = sqrt(sqr(origin[0], origin[1], origin[2]));
    double const initialScale = 1.0 / (1.0 + getOptions().cosmology.startRedshift);
    double const scale = cosmosTauToScale(initialScale, std::max(0.0, tauMax - distanceToGroup));
    DeviceVector<RockstarParticle> particles;
    for (size_t i = 0; i < input.size(); ++i) {
        RockstarParticle particle{};
        particle.x = input[i].x.toDouble() - origin[0];
        particle.y = input[i].y.toDouble() - origin[1];
        particle.z = input[i].z.toDouble() - origin[2];
        particle.vx = input[i].vx;
        particle.vy = input[i].vy;
        particle.vz = input[i].vz;
        particle.identity = i;
        particle.index = i;
        particles.pushBack(particle);
    }
    int nextId = 1;
    auto groups =
        rockstarSeeds(std::move(particles), nextId, 1.0f, 1.0f, scale, getOptions().compute.cuda);
    for (size_t i = 0; i < groups.size(); ++i) {
        if (groups[i].massVir >= ROCKSTAR_MIN_GROUP) {
            auto record = rockstarMakeRecord(groups, i, scale, origin);
            double const radius = sqrt(
                sqr(origin[0] + groups[i].x, origin[1] + groups[i].y, origin[2] + groups[i].z));
            double const haloScale = cosmosTauToScale(initialScale, std::max(0.0, tauMax - radius));
            record.zValue = double2float16(1.0 / haloScale - 1.0);
            records.push_back(record);
        }
    }
    return records;
}
