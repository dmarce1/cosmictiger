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

#include <stack>

#include <cosmictiger/gravity.hpp>
#include <cosmictiger/group_tree.hpp>
#include <cosmictiger/groups_find.hpp>

HPX_PLAIN_ACTION(groupsFind, GroupsFindAction);

void atomicMin(std::atomic<GroupInt>& minValue, GroupInt value) {
    GroupInt prevValue = minValue;
    while (prevValue > value && !minValue.compare_exchange_strong(prevValue, value)) {
    }
}

hpx::future<size_t> groupsFindFork(
    TreeId self, Vector<TreeId> checklist, double linkLen, bool threadme) {
    static std::atomic<int> nthreads(0);
    hpx::future<size_t> rc;
    GroupTreeNode const* selfPtr = groupTreeGetNode(self);
    bool remote = false;
    if (self.proc != hpxRank()) {
        threadme = true;
        remote = true;
    } else if (threadme) {
        threadme = selfPtr->partRange.second - selfPtr->partRange.first > MIN_KICK_THREAD_PARTS;
        if (threadme) {
            if (nthreads++ < KICK_OVERSUBSCRIPTION * hpxHardwareConcurrency() ||
                (selfPtr->procRange.second - selfPtr->procRange.first > 1)) {
                threadme = true;
            } else {
                threadme = false;
                nthreads--;
            }
        }
    }
    if (!threadme) {
        rc = groupsFind(self, std::move(checklist), linkLen);
    } else if (remote) {
        ASSERT(selfPtr->procRange.first >= 0);
        ASSERT(selfPtr->procRange.first < hpxSize());
        rc = hpx::async<GroupsFindAction>(
            hpxLocalities()[selfPtr->procRange.first], self, std::move(checklist), linkLen);
    } else {
        rc = hpx::async(
            [self, linkLen](Vector<TreeId> checklist) {
                auto rc = groupsFind(self, std::move(checklist), linkLen);
                nthreads--;
                return rc;
            },
            std::move(checklist));
    }
    return rc;
}

struct LeafWorkspace {
    Vector<Fixed32> xValue;
    Vector<Fixed32> yValue;
    Vector<Fixed32> zValue;
    Vector<GroupInt> g;
};

static thread_local std::stack<LeafWorkspace> leafWorkspaces;
static thread_local std::stack<Vector<TreeId>> lists;

static Vector<TreeId> getList() {
    if (lists.empty()) {
        lists.push(Vector<TreeId>());
    }
    auto list = std::move(lists.top());
    lists.pop();
    return std::move(list);
}

static void cleanupList(Vector<TreeId>&& list) {
    lists.push(std::move(list));
}

static LeafWorkspace getLeafWorkspace() {
    if (leafWorkspaces.empty()) {
        leafWorkspaces.push(LeafWorkspace());
    }
    auto list = std::move(leafWorkspaces.top());
    leafWorkspaces.pop();
    return std::move(list);
}

static void cleanupLeafWorkspace(LeafWorkspace&& list) {
    leafWorkspaces.push(std::move(list));
}

hpx::future<size_t> groupsFind(TreeId self, Vector<TreeId> checklist, double linkLen) {
    GroupTreeNode const* selfPtr = groupTreeGetNode(self);
    bool threadLeft = true;
    Vector<TreeId> nextlist = getList();
    Vector<TreeId> leaflist = getList();
    nextlist.resize(0);
    leaflist.resize(0);
    auto const selfBox = selfPtr->box.pad(linkLen * 1.001);
    bool const iamleaf = selfPtr->children[LEFT].index == -1;
    do {
        for (int ci = 0; ci < checklist.size(); ci++) {
            GroupTreeNode const* otherPtr = groupTreeGetNode(checklist[ci]);
            if (otherPtr->lastActive) {
                if (selfBox.periodicIntersection(otherPtr->box).volume() > 0) {
                    if (otherPtr->children[LEFT].index == -1) {
                        leaflist.push_back(checklist[ci]);
                    } else {
                        nextlist.push_back(otherPtr->children[LEFT]);
                        nextlist.push_back(otherPtr->children[RIGHT]);
                    }
                }
            }
        }
        std::swap(nextlist, checklist);
        nextlist.resize(0);
    } while (iamleaf && checklist.size());
    if (selfPtr->children[LEFT].index == -1) {
        LeafWorkspace ws = getLeafWorkspace();
        Vector<Fixed32>& xValue2 = ws.xValue;
        Vector<Fixed32>& yValue2 = ws.yValue;
        Vector<Fixed32>& zValue2 = ws.zValue;
        Vector<GroupInt>& g = ws.g;
        if (leaflist.size()) {
            auto const myRng = selfPtr->partRange;
            float const linkLen2 = sqr(linkLen);
            bool foundAnyLink = false;
            int totalSize = 0;
            for (int i = 0; i < leaflist.size(); i++) {
                GroupTreeNode const* otherPtr = groupTreeGetNode(leaflist[i]);
                if (otherPtr != selfPtr) {
                    auto const& otherRng = otherPtr->partRange;
                    int const otherSize = otherRng.second - otherRng.first;
                    totalSize += otherSize;
                }
            }
            xValue2.resize(totalSize);
            yValue2.resize(totalSize);
            zValue2.resize(totalSize);
            g.resize(totalSize);
            totalSize = 0;
            for (int i = 0; i < leaflist.size(); i++) {
                GroupTreeNode const* otherPtr = groupTreeGetNode(leaflist[i]);
                if (otherPtr != selfPtr) {
                    auto const& otherRng = otherPtr->partRange;
                    int otherSize = otherRng.second - otherRng.first;
                    particlesGlobalReadPosAndGroup(otherPtr->globalPartRange(), xValue2.data(),
                        yValue2.data(), zValue2.data(), g.data(), totalSize);
                    for (int i = totalSize; i < totalSize + otherSize; i++) {
                        Array<double, NDIM> x;
                        x[XDIM] = xValue2[i].toDouble();
                        x[YDIM] = yValue2[i].toDouble();
                        x[ZDIM] = zValue2[i].toDouble();
                        if (!selfBox.periodicContains(x)) {
                            int const j = totalSize + otherSize - 1;
                            xValue2[i] = xValue2[j];
                            yValue2[i] = yValue2[j];
                            zValue2[i] = zValue2[j];
                            g[i] = g[j];
                            otherSize--;
                            i--;
                        }
                    }
                    totalSize += otherSize;
                }
            }
            for (PartInt k = myRng.first; k < myRng.second; k++) {
                SimdInt sinkX = particlesPos(XDIM, k).raw();
                SimdInt sinkY = particlesPos(YDIM, k).raw();
                SimdInt sinkZ = particlesPos(ZDIM, k).raw();
                for (int j = 0; j < totalSize; j += SIMD_FLOAT_SIZE) {
                    static SimdFloat const toFloatFactor(kFixed2float);
                    int const lmax = std::min(totalSize, j + SIMD_FLOAT_SIZE);
                    SimdInt srcX;
                    SimdInt srcY;
                    SimdInt srcZ;
                    for (int l = j; l < lmax; l++) {
                        int const l0 = l - j;
                        srcX[l0] = xValue2[l].raw();
                        srcY[l0] = yValue2[l].raw();
                        srcZ[l0] = zValue2[l].raw();
                    }
                    SimdFloat const x = SimdFloat(srcX - sinkX) * toFloatFactor;
                    SimdFloat const y = SimdFloat(srcY - sinkY) * toFloatFactor;
                    SimdFloat const z = SimdFloat(srcZ - sinkZ) * toFloatFactor;
                    SimdFloat const dist = sqr(x, y, z);
                    SimdFloat const lt = dist < linkLen2;
                    for (int l = j; l < lmax; l++) {
                        int const l0 = l - j;
                        if (lt[l0]) {
                            auto& grp = particlesGroup(k);
                            GroupInt const startGroup = particlesGroup(k);
                            if ((GroupInt) grp == NO_GROUP) {
                                grp = particlesGroupInit(k);
                                foundAnyLink = true;
                            }
                            if (grp > g[l]) {
                                grp = g[l];
                                foundAnyLink = true;
                            }
                        }
                    }
                }
            }
            bool foundLink;
            do {
                foundLink = false;
                for (PartInt j = myRng.first; j < myRng.second; j++) {
                    SimdInt sinkX = particlesPos(XDIM, j).raw();
                    SimdInt sinkY = particlesPos(YDIM, j).raw();
                    SimdInt sinkZ = particlesPos(ZDIM, j).raw();
                    for (PartInt k = j + 1; k < myRng.second; k += SIMD_FLOAT_SIZE) {
                        static SimdFloat const toFloatFactor(kFixed2float);
                        int const lmax = std::min(myRng.second, k + SIMD_FLOAT_SIZE);
                        SimdInt srcX;
                        SimdInt srcY;
                        SimdInt srcZ;
                        for (int l = k; l < lmax; l++) {
                            int const l0 = l - k;
                            srcX[l0] = particlesPos(XDIM, l).raw();
                            srcY[l0] = particlesPos(YDIM, l).raw();
                            srcZ[l0] = particlesPos(ZDIM, l).raw();
                        }
                        SimdFloat const x = SimdFloat(srcX - sinkX) * toFloatFactor;
                        SimdFloat const y = SimdFloat(srcY - sinkY) * toFloatFactor;
                        SimdFloat const z = SimdFloat(srcZ - sinkZ) * toFloatFactor;
                        SimdFloat const dist = sqr(x, y, z);
                        SimdFloat const lt = dist < linkLen2;
                        for (int l = k; l < lmax; l++) {
                            if (lt[l - k] == 1.0) {
                                auto& grpa = particlesGroup(l);
                                auto& grpb = particlesGroup(j);
                                if ((GroupInt) grpa == NO_GROUP) {
                                    grpa = particlesGroupInit(l);
                                    foundLink = true;
                                    foundAnyLink = true;
                                }
                                if ((GroupInt) grpb == NO_GROUP) {
                                    grpb = particlesGroupInit(j);
                                    foundLink = true;
                                    foundAnyLink = true;
                                }
                                if (grpa != grpb) {
                                    if (grpa < grpb) {
                                        grpb = (GroupInt) grpa;
                                    } else {
                                        grpa = (GroupInt) grpb;
                                    }
                                    foundLink = true;
                                    foundAnyLink = true;
                                }
                            }
                        }
                    }
                }
            } while (foundLink);
            cleanupList(std::move(nextlist));
            cleanupList(std::move(leaflist));
            cleanupLeafWorkspace(std::move(ws));
            if (foundAnyLink) {
                groupTreeSetActive(self, true);
                return hpx::make_ready_future((size_t) 1);
            } else {
                groupTreeSetActive(self, false);
                return hpx::make_ready_future((size_t) 0);
            }
        } else {
            groupTreeSetActive(self, false);
            cleanupList(std::move(nextlist));
            cleanupList(std::move(leaflist));
            cleanupLeafWorkspace(std::move(ws));
            return hpx::make_ready_future((size_t) 0);
        }
    } else {
        checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
        cleanupList(std::move(nextlist));
        cleanupList(std::move(leaflist));
        if (checklist.size()) {
            GroupTreeNode const* cl = groupTreeGetNode(selfPtr->children[LEFT]);
            GroupTreeNode const* cr = groupTreeGetNode(selfPtr->children[RIGHT]);
            std::array<hpx::future<size_t>, NCHILD> futs;
            futs[RIGHT] = groupsFindFork(selfPtr->children[RIGHT], checklist, linkLen, true);
            futs[LEFT] =
                groupsFindFork(selfPtr->children[LEFT], std::move(checklist), linkLen, false);
            if (futs[LEFT].is_ready() && futs[RIGHT].is_ready()) {
                auto rcl = futs[LEFT].get();
                auto rcr = futs[RIGHT].get();
                auto const tot = rcl + rcr;
                groupTreeSetActive(self, tot != 0);
                return hpx::make_ready_future(tot);
            } else {
                return hpx::when_all(futs.begin(), futs.end())
                    .then([self](hpx::future<std::vector<hpx::future<size_t>>> futfut) {
                        auto futs = futfut.get();
                        auto const tot = futs[LEFT].get() + futs[RIGHT].get();
                        groupTreeSetActive(self, tot != 0);
                        return tot;
                    });
            }
        } else {
            groupTreeSetActive(self, false);
            return hpx::make_ready_future((size_t) 0);
        }
    }
}
