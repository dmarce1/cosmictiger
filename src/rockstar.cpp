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
#include <cosmictiger/timer.hpp>

void rockstarFindSubgroups(DeviceVector<RockstarParticle>& parts, float scale = 1.0);

void rockstarAssignLinkLen(DeviceVector<RockstarTree> const& trees,
    DeviceVector<RockstarParticle>& parts, int selfId, Vector<int> checklist, float linkLen,
    bool phase) {
    static thread_local Vector<int> nextlist;
    static thread_local Vector<int> leaflist;
    int const dimmax = phase ? 2 * NDIM : NDIM;
    nextlist.resize(0);
    leaflist.resize(0);
    auto& self = trees[selfId];
    bool const iamleaf = self.children[LEFT] == -1;
    auto mybox = self.box.pad(linkLen * 1.0001);
    do {
        for (int ci = 0; ci < checklist.size(); ci++) {
            auto const check = checklist[ci];
            auto const& other = trees[check];
            if (mybox.intersection(other.box).volume() > 0) {
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
        float const linkLen2 = sqr(linkLen);
        for (int li = 0; li < leaflist.size(); li++) {
            auto const leafi = leaflist[li];
            auto const other = trees[leafi];
            for (int pi = other.partBegin; pi < other.partEnd; pi++) {
                auto const& part = parts[pi];
                if (mybox.contains(part.xValue)) {
                    for (int pj = self.partBegin; pj < self.partEnd; pj++) {
                        auto& mypart = parts[pj];
                        float r2 = 0.f;
                        for (int dim = 0; dim < dimmax; dim++) {
                            r2 += sqr(mypart.xValue[dim] - part.xValue[dim]);
                        }
                        if (r2 <= linkLen2 && r2 > 0.0) {
                            mypart.minDist2 = std::min(mypart.minDist2, r2);
                        }
                    }
                }
            }
        }
    } else {
        checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
        if (checklist.size()) {
            rockstarAssignLinkLen(trees, parts, self.children[LEFT], checklist, linkLen, phase);
            rockstarAssignLinkLen(
                trees, parts, self.children[RIGHT], std::move(checklist), linkLen, phase);
        }
    }
}

void rockstarFindNeighbors(DeviceVector<int>& leaves, DeviceVector<RockstarTree>& trees, int selfId,
    Vector<int> checklist, float linkLen) {
    static thread_local Vector<int> nextlist;
    static thread_local Vector<int> leaflist;
    nextlist.resize(0);
    leaflist.resize(0);
    auto& self = trees[selfId];
    bool const iamleaf = self.children[LEFT] == -1;
    auto mybox = self.box.pad(linkLen * 1.0001);
    do {
        for (int ci = 0; ci < checklist.size(); ci++) {
            auto const check = checklist[ci];
            auto const& other = trees[check];
            if (mybox.intersection(other.box).volume() > 0) {
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
            rockstarFindNeighbors(leaves, trees, self.children[LEFT], checklist, linkLen);
            rockstarFindNeighbors(
                leaves, trees, self.children[RIGHT], std::move(checklist), linkLen);
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
    Range<float, 2 * NDIM>& rng, int partBegin, int partEnd) {
    Array<int, NCHILD> children;
    RockstarTree node;
    int activeCount = 0;
    int const bucketSize = ROCKSTAR_BUCKET_SIZE;
    if (partEnd - partBegin <= bucketSize) {
        children[LEFT] = children[RIGHT] = -1;
        for (int dim = 0; dim < 2 * NDIM; dim++) {
            rng.begin[dim] = std::numeric_limits<float>::max() / 10.0;
            rng.end[dim] = -std::numeric_limits<float>::max() / 10.0;
        }
        for (int i = partBegin; i != partEnd; i++) {
            for (int dim = 0; dim < 2 * NDIM; dim++) {
                float const x = parts[i].xValue[dim];
                rng.begin[dim] = std::min(rng.begin[dim], x);
                rng.end[dim] = std::max(rng.end[dim], x);
            }
        }
        activeCount++;
    } else {
        float midx;
        int maxDim;
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
        int const partMid = rockstarParticlesSort(parts, partBegin, partEnd, midx, maxDim);
        Range<float, 2 * NDIM> rngLeft = rng;
        Range<float, 2 * NDIM> rngRight = rng;
        rngRight.begin[maxDim] = midx;
        rngLeft.end[maxDim] = midx;
        children[LEFT] = rockstarFormTree(parts, trees, rngLeft, partBegin, partMid);
        children[RIGHT] = rockstarFormTree(parts, trees, rngRight, partMid, partEnd);
        for (int dim = 0; dim < 2 * NDIM; dim++) {
            rng.begin[dim] = std::min(rngLeft.begin[dim], rngRight.begin[dim]);
            rng.end[dim] = std::max(rngLeft.end[dim], rngRight.end[dim]);
        }
        activeCount += trees[children[LEFT]].activeCount;
        activeCount += trees[children[RIGHT]].activeCount;
    }
    node.partBegin = partBegin;
    node.partEnd = partEnd;
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
    Range<float, 2 * NDIM> rng = trees[treeRoot].box;
    //	PRINT("V %e\n", rng.volume());
    Vector<Pair<float, int>> seps;
    for (int i = 0; i < trees.size(); i++) {
        if (trees[i].children[LEFT] == -1) {
            int count = trees[i].partEnd - trees[i].partBegin;
            if (count) {
                float meanSep = pow(trees[i].box.volume() / count, 1.0 / 6.0);
                seps.push_back(Pair<float, int>(meanSep, count));
            }
        }
    }
    std::sort(seps.begin(), seps.end(),
        [](Pair<float, int> a, Pair<float, int> b) { return a.first < b.first; });
    int ffCnt = parts.size() * ff;
    int cnt = 0;
    float maxLinkLen;
    for (int i = 0; i < seps.size(); i++) {
        cnt += seps[i].second;
        if (cnt >= ffCnt) {
            maxLinkLen = seps[i].first;
            break;
        }
    }
    //	PRINT("%e ", max_link_len);
    bool done = false;
    do {
        for (auto& p : parts) {
            p.minDist2 = std::numeric_limits<float>::max();
        }
        gpu = parts.size() > 4 * 1024;
        if (gpu) {
            DeviceVector<int> leaves;
            rockstarFindNeighbors(
                leaves, trees, trees.size() - 1, Vector<int>(1, trees.size() - 1), maxLinkLen);
            rockstarAssignLinklenCuda(trees, leaves, parts, maxLinkLen, true);
        } else {
            rockstarAssignLinkLen(
                trees, parts, treeRoot, Vector<int>(1, treeRoot), maxLinkLen, true);
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
            maxLinkLen *= 2.0;
        } else {
            maxLinkLen = (sqrt(dist2s[i0]) + sqrt(dist2s[i1])) * 0.5f;
            done = true;
        }
    } while (!done);
    //	PRINT("%e\n", max_link_len);
    return maxLinkLen;
}

void rockstarFindAllLinkLens(
    DeviceVector<RockstarTree>& trees, DeviceVector<RockstarParticle>& parts, int treeRoot) {
    for (auto& p : parts) {
        p.minDist2 = std::numeric_limits<float>::max();
    }
    double maxLinkLen = getOptions().lcB * pow(getOptions().numberParticles, -1.0 / NDIM);
    bool gpu = parts.size() > 4 * 1024;
    if (gpu) {
        DeviceVector<int> leaves;
        rockstarFindNeighbors(
            leaves, trees, trees.size() - 1, Vector<int>(1, trees.size() - 1), maxLinkLen);
        rockstarAssignLinklenCuda(trees, leaves, parts, maxLinkLen, false);
    } else {
        rockstarAssignLinkLen(trees, parts, treeRoot, Vector<int>(1, treeRoot), maxLinkLen, false);
    }
}

size_t rockstarFindSubgroups(DeviceVector<RockstarTree>& trees,
    DeviceVector<RockstarParticle>& parts, int selfId, Vector<int> checklist, float linkLen,
    int& nextId) {
    static thread_local Vector<int> nextlist;
    static thread_local Vector<int> leaflist;
    nextlist.resize(0);
    leaflist.resize(0);
    auto& self = trees[selfId];
    bool const iamleaf = self.children[LEFT] == -1;
    auto mybox = self.box.pad(linkLen * 1.0001);
    do {
        for (int ci = 0; ci < checklist.size(); ci++) {
            auto const check = checklist[ci];
            auto const& other = trees[check];
            if (other.lastActive) {
                if (mybox.intersection(other.box).volume() > 0) {
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
        float const linkLen2 = sqr(linkLen);
        for (int li = 0; li < leaflist.size(); li++) {
            if (selfId != leaflist[li]) {
                auto const leafi = leaflist[li];
                auto const other = trees[leafi];
                for (int pi = other.partBegin; pi < other.partEnd; pi++) {
                    auto& part = parts[pi];
                    if (mybox.contains(part.xValue)) {
                        for (int pj = self.partBegin; pj < self.partEnd; pj++) {
                            auto& mypart = parts[pj];
                            float const dx = mypart.x - part.x;
                            float const dy = mypart.y - part.y;
                            float const dz = mypart.z - part.z;
                            float const dvx = mypart.vx - part.vx;
                            float const dvy = mypart.vy - part.vy;
                            float const dvz = mypart.vz - part.vz;
                            float const r2 = sqr(dx, dy, dz) + sqr(dvx, dvy, dvz);
                            if (r2 <= linkLen2) {
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
            for (int pi = self.partBegin; pi < self.partEnd; pi++) {
                auto& aValue = parts[pi];
                for (int pj = pi + 1; pj < self.partEnd; pj++) {
                    auto& bValue = parts[pj];
                    float const dx = aValue.x - bValue.x;
                    float const dy = aValue.y - bValue.y;
                    float const dz = aValue.z - bValue.z;
                    float const dvx = aValue.vx - bValue.vx;
                    float const dvy = aValue.vy - bValue.vy;
                    float const dvz = aValue.vz - bValue.vz;
                    float const r2 = sqr(dx, dy, dz) + sqr(dvx, dvy, dvz);
                    if (r2 <= linkLen2) {
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
        int numberActive = 0;
        if (checklist.size()) {
            numberActive += rockstarFindSubgroups(
                trees, parts, self.children[LEFT], checklist, linkLen, nextId);
            numberActive += rockstarFindSubgroups(
                trees, parts, self.children[RIGHT], std::move(checklist), linkLen, nextId);
        }
        self.active = numberActive;
        return numberActive;
    }
}

void rockstarFindSubgroups(DeviceVector<RockstarTree>& trees, DeviceVector<RockstarParticle>& parts,
    float linkLen, int& nextId, bool gpu) {
    Timer tm;
    tm.start();
    for (int i = 0; i < parts.size(); i++) {
        parts[i].subgroup = ROCKSTAR_NO_GROUP;
    }
    for (int i = 0; i < trees.size(); i++) {
        trees[i].active = true;
    }
    int cnt;
    gpu = parts.size() > 4 * 1024;
    if (gpu) {
        DeviceVector<int> leaves;
        rockstarFindNeighbors(
            leaves, trees, trees.size() - 1, Vector<int>(1, trees.size() - 1), linkLen);
        rockstarFindSubgroupsCuda(trees, leaves, parts, linkLen, nextId);
    } else {
        do {
            double pctActive = 0.0;
            for (int i = 0; i < trees.size(); i++) {
                trees[i].lastActive = trees[i].active;
                pctActive += int(trees[i].active) / (double) trees.size();
            }
            int rootId = trees.size() - 1;
            cnt = rockstarFindSubgroups(
                trees, parts, rootId, Vector<int>(1, rootId), linkLen, nextId);
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

std::unordered_map<int, Number> rockstarSubgroupCnts(DeviceVector<RockstarParticle>& parts) {
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

bool rockstarHaloBoundTo(Subgroup const& subhalo, Subgroup const& halo, double scale) {
    DeviceVector<Array<float, NDIM>> xValue;
    DeviceVector<Array<float, NDIM>> yValue;
    for (int i = 0; i < subhalo.parts.size(); i++) {
        Array<float, NDIM> x;
        for (int dim = 0; dim < NDIM; dim++) {
            x[dim] = subhalo.parts[i].xValue[dim];
        }
        yValue.pushBack(x);
    }
    for (int i = 0; i < halo.parts.size(); i++) {
        Array<float, NDIM> x;
        for (int dim = 0; dim < NDIM; dim++) {
            x[dim] = halo.parts[i].xValue[dim];
        }
        xValue.pushBack(x);
    }
    auto const phi =
        bhEvaluatePoints(yValue, xValue, xValue.size() > 4 * 1024 || yValue.size() > 4 * 1024);
    double etot = 0.0;
    for (int i = 0; i < subhalo.parts.size(); i++) {
        float const dvx = subhalo.parts[i].vx - halo.vxb;
        float const dvy = subhalo.parts[i].vy - halo.vyb;
        float const dvz = subhalo.parts[i].vz - halo.vzb;
        float const ekin = 0.5 * sqr(dvx, dvy, dvz);
        etot += ekin + phi[i] / scale;
    }
    return etot < 0.0;
}

bool rockstarHaloBound(Subgroup& halo, Vector<RockstarParticle>& unboundParts, float scale) {
    DeviceVector<Array<float, NDIM>> x;
    for (int i = 0; i < halo.parts.size(); i++) {
        Array<float, NDIM> xValue;
        for (int dim = 0; dim < NDIM; dim++) {
            xValue[dim] = halo.parts[i].xValue[dim];
        }
        x.pushBack(xValue);
    }
    auto y = x;
    auto phi = bhEvaluatePotential(x, x.size() > 4 * 1024);
    auto phiTot = 0.0;
    auto kinTot = 0.0;
    int nunbound = 0;
    int onparts = halo.parts.size();
    for (int i = 0; i < halo.parts.size(); i++) {
        float const dvx = halo.parts[i].vx - halo.vxb;
        float const dvy = halo.parts[i].vy - halo.vyb;
        float const dvz = halo.parts[i].vz - halo.vzb;
        float const ekin = 0.5 * sqr(dvx, dvy, dvz);
        kinTot += ekin;
        phiTot += phi[i] / scale;
        if (ekin + phi[i] / scale > 0.0) {
            nunbound++;
            unboundParts.push_back(halo.parts[i]);
            halo.parts[i] = halo.parts.back();
            halo.parts.popBack();
            i--;
        }
    }
    halo.t = kinTot;
    halo.w = 0.5 * phiTot;
    // PRINT( "%e %e\n", halo.T, halo.W );
    bool const rc = (nunbound * 2 < onparts) && (onparts - nunbound) >= ROCKSTAR_MIN_GROUP;
    if (!rc) {
        for (int i = 0; i < halo.parts.size(); i++) {
            unboundParts.push_back(halo.parts[i]);
        }
        halo.parts = decltype(halo.parts)();
    }
    // if( !rc ) {
    //		PRINT( "%e\n", halo.T / (halo.W));
    //	}
    return rc;
}

Vector<int> rockstarAllSubgroups(Subgroup const& sg, Vector<Subgroup>& subgroups) {
    Vector<int> sgs;
    if (sg.parts.size() >= ROCKSTAR_MIN_GROUP) {
        sgs.push_back(sg.id);
    }
    if (sg.children.size()) {
        for (int i = 0; i < sg.children.size(); i++) {
            int l;
            for (l = 0; l < subgroups.size(); l++) {
                if (subgroups[l].id == sg.children[i]) {
                    break;
                }
            }
            auto theseSgs = rockstarAllSubgroups(subgroups[l], subgroups);
            sgs.insert(sgs.end(), theseSgs.begin(), theseSgs.end());
        }
    }
    return sgs;
}

void rockstarSubgroupStatistics(Subgroup& sg, bool gpu) {
    DeviceVector<RockstarTree> trees;
    DeviceVector<RockstarParticle> parts;
    for (int i = 0; i < sg.parts.size(); i++) {
        if (sg.parts[i].subgroup == sg.id) {
            parts.pushBack(sg.parts[i]);
        }
    }
    float sigma2X = 0.0;
    float sigma2V = 0.0;
    float xcom = 0.0;
    float ycom = 0.0;
    float zcom = 0.0;
    float vxcom = 0.0;
    float vycom = 0.0;
    float vzcom = 0.0;
    int nValue = 0;
    for (int i = 0; i < parts.size(); i++) {
        xcom += parts[i].x;
        ycom += parts[i].y;
        zcom += parts[i].z;
        vxcom += parts[i].vx;
        vycom += parts[i].vy;
        vzcom += parts[i].vz;
        nValue++;
    }
    xcom /= parts.size();
    ycom /= parts.size();
    zcom /= parts.size();
    vxcom /= parts.size();
    vycom /= parts.size();
    vzcom /= parts.size();
    for (int i = 0; i < parts.size(); i++) {
        sigma2X += sqr(parts[i].x - xcom);
        sigma2X += sqr(parts[i].y - ycom);
        sigma2X += sqr(parts[i].z - zcom);
        sigma2V += sqr(parts[i].vx - vxcom);
        sigma2V += sqr(parts[i].vy - vycom);
        sigma2V += sqr(parts[i].vz - vzcom);
        nValue++;
    }
    if (parts.size() >= ROCKSTAR_MIN_GROUP) {
        float xfac = 1.0 / sqrt(sigma2X);
        float vfac = 1.0 / sqrt(sigma2V);
        for (int i = 0; i < parts.size(); i++) {
            parts[i].x *= xfac;
            parts[i].y *= xfac;
            parts[i].z *= xfac;
            parts[i].vx *= vfac;
            parts[i].vy *= vfac;
            parts[i].vz *= vfac;
        }
        int const rootId = rockstarFormTree(trees, parts);

        float const linkLen = rockstarFindLinkLen(trees, parts, rootId, ROCKSTAR_FF, gpu);
    }
    sg.hostPartCnt = parts.size();
}

Vector<RockstarParticle> rockstarGatherHaloParts(
    Vector<Subgroup> const& subgroups, Subgroup const& sg) {
    Vector<RockstarParticle> xValue;
    for (int i = 0; i < sg.parts.size(); i++) {
        xValue.push_back(sg.parts[i]);
    }
    for (int i = 0; i < sg.children.size(); i++) {
        auto const tmp = rockstarGatherHaloParts(subgroups, subgroups[sg.children[i]]);
        xValue.insert(xValue.end(), tmp.begin(), tmp.end());
    }
    return xValue;
}

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
    float sigmaX = sqrt(sigma2X);
    float sigmaV = sqrt(sigma2V);
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
    int const rootId = rockstarFormTree(trees, parts);
    float const linkLen = rockstarFindLinkLen(trees, parts, rootId, ROCKSTAR_FF, gpu);
    rockstarFindSubgroups(trees, parts, linkLen, nextId, gpu);
    auto groupCnts = rockstarSubgroupCnts(parts);

    for (auto i = groupCnts.begin(); i != groupCnts.end(); i++) {
        DeviceVector<RockstarParticle> theseParts;
        int j = 0;
        while (j < parts.size()) {
            if (parts[j].subgroup == i->first) {
                theseParts.pushBack(parts[j]);
                parts[j] = parts.back();
                parts.popBack();
            } else {
                j++;
            }
        }
        auto theseGroups = rockstarSeeds(theseParts, nextId, rfac, vfac, scale, gpu, depth + 1);
        subgroups.insert(subgroups.end(), theseGroups.begin(), theseGroups.end());
    }

    auto const findSigmas = [rfac, vfac, scale](Subgroup& sg) {
        float sigma2V = 0.0;
        float sigma2X = 0.0;
        Array<float, 2 * NDIM> xValue;
        auto const ndim = pow(getOptions().numberParticles, 1.0 / NDIM);
        DeviceVector<RockstarParticle>& theseParts = sg.parts;
        for (int dim = 0; dim < 2 * NDIM; dim++) {
            xValue[dim] = 0.0;
        }
        for (int j = 0; j < theseParts.size(); j++) {
            for (int dim = 0; dim < 2 * NDIM; dim++) {
                xValue[dim] += theseParts[j].xValue[dim];
            }
        }
        for (int dim = 0; dim < 2 * NDIM; dim++) {
            xValue[dim] /= theseParts.size();
        }
        for (int j = 0; j < theseParts.size(); j++) {
            auto const& p = theseParts[j];
            float const dx = p.x - xValue[XDIM];
            float const dy = p.y - xValue[YDIM];
            float const dz = p.z - xValue[ZDIM];
            float const dx2 = sqr(dx, dy, dz);
            sigma2X += dx2;
        }
        sigma2X /= theseParts.size();
        double sigxNorm = sqrt(sigma2X / theseParts.size());
        if (sigxNorm < sg.sigmaXMin) {
            sg.sigmaXMin = sigxNorm;
            sg.x = xValue[XDIM];
            sg.y = xValue[YDIM];
            sg.z = xValue[ZDIM];
        }
        sg.vxb = xValue[NDIM + XDIM];
        sg.vyb = xValue[NDIM + YDIM];
        sg.vzb = xValue[NDIM + ZDIM];
        sigma2X = 0.0;
        struct RadialParticle {
            double r;
            float vx;
            float vy;
            float vz;
        };
        Vector<RadialParticle> radii;
        for (int j = 0; j < theseParts.size(); j++) {
            auto const& p = theseParts[j];
            double const dx = p.x - sg.x;
            double const dy = p.y - sg.y;
            double const dz = p.z - sg.z;
            double const dx2 = sqr(dx, dy, dz);
            sigma2X += dx2;
            RadialParticle p0;
            p0.r = sqrt(dx2);
            p0.vx = p.vx;
            p0.vy = p.vy;
            p0.vz = p.vz;
            radii.push_back(p0);
        }
        std::sort(radii.begin(), radii.end(),
            [](RadialParticle a, RadialParticle b) { return a.r < b.r; });
        double mass = 0.0;
        double volume = 0.0;
        double const omegaM = getOptions().omegaM;
        double const x =
            ((omegaM / (sqr(scale) * scale)) / (omegaM / (sqr(scale) * scale)) + (1.0 - omegaM)) -
            1.0;
        double y = (18.0 * M_PI * M_PI + 82.0 * x - 39.0 * sqr(x)) / (1.0 + x);
        double rVir = pow(theseParts.size() / (4.0 / 3.0 * M_PI * y), 1.0 / 3.0) / ndim;

        sigma2X /= theseParts.size();
        sg.sigma2X = sigma2X;
        sg.rVir = rVir;
        int n;
        float rfrac = 0.1;
        n = 0;
        sg.vxc = 0.0;
        sg.vyc = 0.0;
        sg.vzc = 0.0;
        int rmax = std::max(ROCKSTAR_MIN_GROUP, (int) ((radii.size() + 0.5) / 10));
        for (int j = 0; j < rmax; j++) {
            sg.vxc += radii[j].vx;
            sg.vyc += radii[j].vy;
            sg.vzc += radii[j].vz;
        }
        sg.vxc /= rmax;
        sg.vyc /= rmax;
        sg.vzc /= rmax;
        for (int j = 0; j < theseParts.size(); j++) {
            auto const& p = theseParts[j];
            float const dvx = p.vx - sg.vxc;
            float const dvy = p.vy - sg.vyc;
            float const dvz = p.vz - sg.vzc;
            float const dv2 = sqr(dvx, dvy, dvz);
            sigma2V += dv2;
        }
        sigma2V /= theseParts.size();
        sg.sigma2V = sigma2V;
        float vcircMax = 0.0;
        double h = getOptions().hubble * constants::kH0 * getOptions().codeToS;
        float const numberParticles = getOptions().numberParticles;
        for (int n = 1; n < radii.size(); n++) {
            float vcirc = sqrt(3.0 * getOptions().omegaM * sqr(h) * n /
                (8.0 * M_PI * numberParticles * radii[n].r / rfac));
            vcircMax = std::max(vcircMax, vcirc);
        }
        sg.vcircMax = vcircMax * vfac;
        y *= (1.0 + x);
        sg.rDyn = sqrt(2.0) * vcircMax / h / sqrt(y) * rfac;
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
            if (!foundMerge) {
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
    for (int i = 0; i < depth; i++) {
        //		PRINT("\t");
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
        sg.sigma2V *= sigmaV * sigmaV;
        sg.sigma2X *= sigmaX * sigmaX;
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
        std::function<Vector<RockstarParticle>(Subgroup&)> const unbind =
            [&unbind, &subgroups, scale](Subgroup& grp) {
                Vector<RockstarParticle> unbounds;
                for (int ci = 0; ci < grp.children.size(); ci++) {
                    auto tmp = unbind(subgroups[grp.children[ci]]);
                    for (auto const& p : tmp) {
                        grp.parts.pushBack(p);
                    }
                }
                rockstarHaloBound(grp, unbounds, scale);
                return unbounds;
            };

        int bound = 0;
        int unbound = 0;
        int rebound = 0;
        Vector<RockstarParticle> unbounds;
        for (int k = 0; k < subgroups.size(); k++) {
            int const psize = subgroups[k].parts.size();
            if (subgroups[k].parent == -1) {
                auto tmp = unbind(subgroups[k]);
                unbounds.insert(unbounds.end(), tmp.begin(), tmp.end());
            }
            bound += psize;
        }
        int k = 0;
        while (k < subgroups.size()) {
            if (subgroups[k].parts.size() == 0) {
                subgroups[k] = subgroups.back();
                subgroups.pop_back();
            } else {
                k++;
            }
        }

        unbound += unbounds.size();
        bound -= unbound;
        for (int k = 0; k < subgroups.size(); k++) {
            auto& sgA = subgroups[k];
            float maxMass = 0.0;
            int minIndex = -1;
            for (int l = 0; l < subgroups.size(); l++) {
                auto& sgB = subgroups[l];
                if (sgA.parts.size() < sgB.parts.size()) {
                    if (sgB.parts.size() > maxMass) {
                        float const dx = sgB.x - sgA.x;
                        float const dy = sgB.y - sgA.y;
                        float const dz = sgB.z - sgA.z;
                        float const r = sqrt(sqr(dx, dy, dz));
                        if (r < sgB.rVir) {
                            maxMass = sgB.parts.size();
                            minIndex = l;
                        }
                    }
                }
            }
            if (minIndex != -1) {
                auto& sgB = subgroups[minIndex];
                sgA.parent = minIndex;
                sgB.children.push_back(k);
            } else {
                sgA.parent = -1;
            }
        }
        int haloCount = 0;
        int subhaloCount = 0;
        for (int k = 0; k < subgroups.size(); k++) {
            if (subgroups[k].parent == -1) {
                haloCount++;
            } else {
                subhaloCount++;
            }
        }
        PRINT("%i %i\n", haloCount, subhaloCount);
    }
    return subgroups;
}

void rockstarFindSubgroups(DeviceVector<RockstarParticle>& parts, float scale) {
    int nextId = 1;
    rockstarSeeds(parts, nextId, 1.0, 1.0, scale, false);
}

Vector<RockstarRecord> rockstarFindSubgroups(Vector<LightconeEntry> const& parts, double tauMax) {
    DeviceVector<RockstarParticle> rockParts(parts.size());
    Vector<RockstarRecord> recs;
    double xc = 0.0;
    double yc = 0.0;
    double zc = 0.0;
    for (int i = 0; i < parts.size(); i++) {
        auto& pi = parts[i];
        xc += pi.x.toDouble();
        yc += pi.y.toDouble();
        zc += pi.z.toDouble();
    }

    xc /= parts.size();
    yc /= parts.size();
    zc /= parts.size();
    auto const thisTau = sqrt(sqr(xc, yc, zc));
    auto const z0 = getOptions().z0;
    auto const a0 = 1.0 / (z0 + 1.0);
    //	double scale = cosmos_tau_to_scale(a0, tau_max - this_tau);
    double scale = 1.0;
    //	PRINT( "SCALE = %e %e %e\n", a0, this_tau, scale);
    for (int i = 0; i < parts.size(); i++) {
        auto& pi = parts[i];
        auto& pj = rockParts[i];
        pj.x = pi.x.toDouble() - xc;
        pj.y = pi.y.toDouble() - yc;
        pj.z = pi.z.toDouble() - zc;
        pj.vx = pi.vx;
        pj.vy = pi.vy;
        pj.vz = pi.vz;
    }
    int nextId = 1;
    auto subgroups = rockstarSeeds(rockParts, nextId, 1.0, 1.0, scale, false);
    if (subgroups.size()) {
        std::function<DeviceVector<RockstarParticle>(Subgroup const&)> const gatherSubparticles =
            [&gatherSubparticles, &subgroups](Subgroup const& grp) {
                DeviceVector<RockstarParticle> parts;
                for (int i = 0; i < grp.parts.size(); i++) {
                    parts.pushBack(grp.parts[i]);
                }
                for (int ci = 0; ci < grp.children.size(); ci++) {
                    auto tmp = gatherSubparticles(subgroups[grp.children[ci]]);
                    for (int i = 0; i < tmp.size(); i++) {
                        parts.pushBack(tmp[i]);
                    }
                }
                return parts;
            };
        for (int k = 0; k < subgroups.size(); k++) {
            for (int iValue = 0; iValue < 2; iValue++) {
                auto& grp = subgroups[k];
                DeviceVector<RockstarParticle>* partsPtr;
                DeviceVector<RockstarParticle> haloParts;
                if (iValue == 0) {
                    partsPtr = &grp.parts;
                } else {
                    if (grp.children.size() == 0) {
                        continue;
                    }
                    haloParts = gatherSubparticles(grp);
                    partsPtr = &haloParts;
                }
                auto& parts = *partsPtr;
                DeviceVector<RockstarTree> trees;
                int const rootId = rockstarFormTree(trees, parts);
                Vector<RockstarParticle> parts500;
                rockstarFindAllLinkLens(trees, parts, rootId);
                auto const ndim = pow(getOptions().numberParticles, 1.0 / NDIM);
                double const ll200 = pow(200, -1.0 / 3.0) / ndim;
                double const ll500 = pow(500, -1.0 / 3.0) / ndim;
                double const ll2500 = pow(2500, -1.0 / 3.0) / ndim;
                double m200 = 0.0;
                double m500 = 0.0;
                double m2500 = 0.0;
                for (int i = 0; i < parts.size(); i++) {
                    double const dist = sqrt(parts[i].minDist2);
                    if (dist < ll200) {
                        m200 += 1.0;
                        if (dist < ll500) {
                            parts500.push_back(parts[i]);
                            m500 += 1.0;
                            if (dist < ll2500) {
                                m2500 += 1.0;
                            }
                        }
                    }
                }
                Vector<double> radii;
                double rsValue, rho0, mvir, rvir, rKlypin, jx, jy, jz, jValue, etotValue, lambda,
                    lambdaB, deltaX, deltaV, toW, rcomMax, rvmax;
                double x = grp.x;
                double y = grp.y;
                double z = grp.z;
                double vx = grp.vxc;
                double vy = grp.vyc;
                double vz = grp.vzc;
                double xcom = 0.0;
                double ycom = 0.0;
                double zcom = 0.0;
                double sigmaV = 0.0;
                double sigmaVb = 0.0;
                double sigmaX = 0.0;
                double vxb = 0.0;
                double vyb = 0.0;
                double vzb = 0.0;
                jx = 0.0;
                jy = 0.0;
                jz = 0.0;
                etotValue = grp.t + grp.w;
                for (int i = 0; i < parts.size(); i++) {
                    double const dx = parts[i].x - x;
                    double const dy = parts[i].y - y;
                    double const dz = parts[i].z - z;
                    sigmaX += sqr(dx, dy, dz);
                    xcom += parts[i].x;
                    ycom += parts[i].y;
                    zcom += parts[i].z;
                    double const dvx = parts[i].vx - vx;
                    double const dvy = parts[i].vy - vy;
                    double const dvz = parts[i].vz - vz;
                    vxb += parts[i].vx;
                    vyb += parts[i].vy;
                    vzb += parts[i].vz;
                    sigmaV += sqr(dvx, dvy, dvz);
                    double const r = sqrt(sqr(dx, dy, dz));
                    radii.push_back(r);
                    jx += dy * dvz - dz * dvy;
                    jy -= dx * dvz - dz * dvx;
                    jz += dz * dvy - dy * dvz;
                }
                vxb /= parts.size();
                vyb /= parts.size();
                vzb /= parts.size();
                for (int i = 0; i < parts.size(); i++) {
                    double const dvx = parts[i].vx - vxb;
                    double const dvy = parts[i].vy - vyb;
                    double const dvz = parts[i].vz - vzb;
                    sigmaVb += sqr(dvx, dvy, dvz);
                }
                sigmaX /= parts.size();
                sigmaV /= parts.size();
                sigmaVb /= parts.size();
                sigmaV = sqrt(sigmaV);
                sigmaX = sqrt(sigmaX);
                sigmaVb = sqrt(sigmaVb);
                xcom /= parts.size();
                ycom /= parts.size();
                zcom /= parts.size();
                Array<Array<double, NDIM>, NDIM> ixx, ixx500;
                for (int i = 0; i < NDIM; i++) {
                    for (int j = 0; j < NDIM; j++) {
                        ixx[i][j] = 0.0;
                        ixx500[i][j] = 0.0;
                    }
                }
                rcomMax = 0.0;
                for (int i = 0; i < parts.size(); i++) {
                    double const dx = parts[i].x - xcom;
                    double const dy = parts[i].y - ycom;
                    double const dz = parts[i].z - zcom;
                    rcomMax = std::max(rcomMax, sqrt(sqr(dx, dy, dz)));
                    ixx[XDIM][XDIM] += dx * dx;
                    ixx[XDIM][YDIM] += dx * dy;
                    ixx[XDIM][ZDIM] += dx * dz;
                    ixx[YDIM][XDIM] += dy * dx;
                    ixx[YDIM][YDIM] += dy * dy;
                    ixx[YDIM][ZDIM] += dy * dz;
                    ixx[ZDIM][XDIM] += dz * dx;
                    ixx[ZDIM][YDIM] += dz * dy;
                    ixx[ZDIM][ZDIM] += dz * dz;
                }
                double xcom500 = 0.0;
                double ycom500 = 0.0;
                double zcom500 = 0.0;
                for (int i = 0; i < parts500.size(); i++) {
                    xcom500 += parts500[i].x;
                    ycom500 += parts500[i].y;
                    zcom500 += parts500[i].z;
                }
                xcom500 /= parts500.size();
                ycom500 /= parts500.size();
                zcom500 /= parts500.size();
                double rcommax500 = 0.0;
                for (int i = 0; i < parts500.size(); i++) {
                    double const dx = parts500[i].x - xcom500;
                    double const dy = parts500[i].y - ycom500;
                    double const dz = parts500[i].z - zcom500;
                    rcommax500 = std::max(rcommax500, sqrt(sqr(dx, dy, dz)));
                    ixx500[XDIM][XDIM] += dx * dx;
                    ixx500[XDIM][YDIM] += dx * dy;
                    ixx500[XDIM][ZDIM] += dx * dz;
                    ixx500[YDIM][XDIM] += dy * dx;
                    ixx500[YDIM][YDIM] += dy * dy;
                    ixx500[YDIM][ZDIM] += dy * dz;
                    ixx500[ZDIM][XDIM] += dz * dx;
                    ixx500[ZDIM][YDIM] += dz * dy;
                    ixx500[ZDIM][ZDIM] += dz * dz;
                }
                for (int i = 0; i < NDIM; i++) {
                    for (int j = 0; j < NDIM; j++) {
                        ixx[i][j] /= parts.size();
                        ixx500[i][j] /= parts500.size();
                    }
                }
                auto const lambda2 = findEigenpair(ixx, rcomMax);
                auto const lambda0 = findEigenpair(ixx, 0.0);
                auto const lambda1 = findEigenpair(ixx, 0.5 * (lambda2.first + lambda0.first));
                double bOA = sqrt(lambda1.first / lambda2.first);
                double cOA = sqrt(lambda0.first / lambda2.first);
                Array<double, NDIM> aValue, a500;
                for (int dim = 0; dim < NDIM; dim++) {
                    aValue[dim] = lambda2.second[dim] * sqrt(lambda2.first);
                }
                double bOA500, cOA500;
                if (parts500.size() >= ROCKSTAR_MIN_GROUP) {
                    auto const lambda2 = findEigenpair(ixx500, rcommax500);
                    auto const lambda0 = findEigenpair(ixx500, 0.0);
                    auto const lambda1 =
                        findEigenpair(ixx500, 0.5 * (lambda2.first + lambda0.first));
                    bOA500 = sqrt(lambda1.first / lambda2.first);
                    cOA500 = sqrt(lambda0.first / lambda2.first);
                    for (int dim = 0; dim < NDIM; dim++) {
                        a500[dim] = lambda2.second[dim] * sqrt(lambda2.first);
                    }
                } else {
                    bOA500 = 0.0;
                    cOA500 = 0.0;
                    for (int dim = 0; dim < NDIM; dim++) {
                        a500[dim] = 0.0;
                    }
                }
                deltaX = sqrt(sqr(xcom - grp.x, ycom - grp.y, zcom - grp.z));
                deltaV = sqrt(sqr(grp.vxb - grp.vxc, grp.vyb - grp.vyc, grp.vzb - grp.vzc));
                jValue = sqrt(sqr(jx, jy, jz));
                std::sort(radii.begin(), radii.end());
                cosmosNfwFit(radii, rsValue, rho0);
                double vcircMax = 0.0;
                for (int i = 0; i < radii.size(); i++) {
                    double menc = i + 1.0;
                    double vcirc = sqrt(getOptions().gm / radii[i] * menc);
                    if (vcirc > vcircMax) {
                        rvmax = radii[i];
                        vcircMax = vcirc;
                    }
                }
                mvir = parts.size();
                //	PRINT("%e %e %e %e\n", mvir, m200, m500, m2500);
                double rmax = radii.back();
                double const omegaM = getOptions().omegaM;
                x = ((omegaM / (sqr(scale) * scale)) / (omegaM / (sqr(scale) * scale)) +
                        (1.0 - omegaM)) -
                    1.0;
                y = (18.0 * M_PI * M_PI + 82.0 * x - 39.0 * sqr(x)) / (1.0 + x);
                rvir = pow(mvir / (4.0 / 3.0 * M_PI * y), 1.0 / 3.0) / ndim;
                rKlypin = cosmosKlypinFit(vcircMax, radii.back(), mvir);
                double const g = getOptions().gm;
                lambda = jValue * sqrt(fabs(etotValue)) / g / pow(mvir, 2.5);
                lambdaB = jValue / sqrt(2.0 * g * sqr(mvir) * mvir * rvir);
                toW = grp.t / grp.w;
                double const convMass = getOptions().codeToG * getOptions().hubble / constants::kM0;
                double const convLen =
                    getOptions().codeToCm * getOptions().hubble / constants::kMpcToCm * 1000.0;
                double const convPos =
                    getOptions().codeToCm * getOptions().hubble / constants::kMpcToCm;
                double const convVel = getOptions().codeToCm / getOptions().codeToS / 100000.0;
                double const convAng = convMass * convPos * convVel;
                double const convEne = convMass * sqr(convVel);
                x += xc;
                y += yc;
                z += zc;
                auto const thisTau = sqrt(sqr(x, y, z));
                auto const z0 = getOptions().z0;
                auto const a0 = 1.0 / (z0 + 1.0);
                double a = cosmosTauToScale(a0, tauMax - thisTau);
                double zValue = 1.0 / a - 1.0;
                mvir *= convMass;
                m200 *= convMass;
                m500 *= convMass;
                m2500 *= convMass;
                rmax *= convLen;
                rvir *= convLen;
                rvmax *= convLen;
                sigmaX *= convLen;
                deltaX *= convLen;
                rsValue *= convLen;
                rKlypin *= convLen;
                sigmaV *= convVel;
                sigmaVb *= convVel;
                vx *= convVel;
                vy *= convVel;
                vz *= convVel;
                vxb *= convVel;
                vyb *= convVel;
                vzb *= convVel;
                deltaV *= convVel;
                vcircMax *= convVel;
                etotValue *= convEne;
                x = grp.x * convPos;
                y = grp.y * convPos;
                z = grp.z * convPos;
                jx *= convAng;
                jy *= convAng;
                jz *= convAng;
                for (int dim = 0; dim < NDIM; dim++) {
                    aValue[dim] *= convLen;
                }
                RockstarRecord rec;

                rec.id.haloId = k;
                rec.id.sys = iValue;
                rec.x = x;
                rec.y = y;
                rec.z = z;
                rec.mvir = mvir;
                rec.e = etotValue;
                rec.jx = jx;
                rec.jy = jy;
                rec.jz = jz;
                rec.pid = grp.parent;
                rec.m200 = m200;
                rec.m500 = m500;
                rec.m2500 = m2500;
                double const f16max = std::numeric_limits<Float16>::max();
                if (rsValue > f16max) {
                    PRINT("mvir %i rmax = %e Rs = %e RKly = %e f16max = %e\n", parts.size(), rmax,
                        rsValue, rKlypin, f16max);
                }
                rec.vxc = double2float16(vx);
                rec.vyc = double2float16(vy);
                rec.vzc = double2float16(vz);
                rec.vxb = double2float16(vxb);
                rec.vyb = double2float16(vyb);
                rec.vzb = double2float16(vzb);
                rec.sigVb = double2float16(sigmaVb);
                rec.rmax = double2float16(rmax);
                rec.rvir = double2float16(rvir);
                rec.rvmax = double2float16(rvmax);
                rec.rs = double2float16(rsValue);
                rec.rkly = double2float16(rKlypin);
                rec.sigX = double2float16(sigmaX);
                rec.deltaX = double2float16(deltaX);
                rec.toW = double2float16(toW);
                rec.sigV = double2float16(sigmaV);
                rec.vcircMax = double2float16(vcircMax);
                rec.deltaV = double2float16(deltaV);
                rec.boa = double2float16(bOA);
                rec.coa = double2float16(cOA);
                rec.ax = double2float16(aValue[XDIM]);
                rec.ay = double2float16(aValue[YDIM]);
                rec.az = double2float16(aValue[ZDIM]);
                rec.boa500 = double2float16(bOA500);
                rec.coa500 = double2float16(cOA500);
                rec.ax500 = double2float16(a500[XDIM]);
                rec.ay500 = double2float16(a500[YDIM]);
                rec.az500 = double2float16(a500[ZDIM]);
                rec.lambda = double2float16(lambda);
                rec.lambdaB = double2float16(lambdaB);
                rec.zValue = double2float16(zValue);
                recs.push_back(rec);
            }
        }
    }
    return recs;
}

Vector<Subgroup> rockstarFindSubgroups(Vector<ParticleData> const& parts, float scale) {
    DeviceVector<RockstarParticle> rockParts;
    Array<Fixed32, NDIM> x0;
    for (int dim = 0; dim < NDIM; dim++) {
        x0[dim] = parts[0].x[dim];
    }
    for (int i = 0; i < parts.size(); i++) {
        Array<float, NDIM> x;
        for (int dim = 0; dim < NDIM; dim++) {
            x[dim] = distance(parts[i].x[dim], x0[dim]);
        }
        RockstarParticle part;
        part.x = x[XDIM];
        part.y = x[YDIM];
        part.z = x[ZDIM];
        part.vx = parts[i].v[XDIM];
        part.vy = parts[i].v[YDIM];
        part.vz = parts[i].v[ZDIM];
        part.index = parts[i].index;
        rockParts.pushBack(part);
    }
    int nextId = 1;
    return rockstarSeeds(rockParts, nextId, 1.0, 1.0, scale, false);
}
