/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <stack>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/analysis/group_tree.hpp>
#include <cosmictiger/analysis/groups_find.hpp>
#include <cosmictiger/gravity/gravity.hpp>

HPX_PLAIN_ACTION(groupsFind, GroupsFindAction);

void atomicMin(std::atomic<GroupInteger>& minValue, GroupInteger value) {
    GroupInteger prevValue = minValue;
    while (prevValue > value && !minValue.compare_exchange_strong(prevValue, value)) {
    }
}

hpx::future<size_t> groupsFindFork(
    TreeId self, Vector<TreeId> checklist, double linkingLength, bool threadme) {
    static std::atomic<int> numberOfThreads(0);
    hpx::future<size_t> rc;
    GroupTreeNode const* selfPtr = groupTreeGetNode(self);
    bool remote = false;
    if (self.proc != hpxRank()) {
        threadme = true;
        remote = true;
    } else if (threadme) {
        threadme =
            selfPtr->particleRange.second - selfPtr->particleRange.first > kMinKickThreadParts;
        if (threadme) {
            if (numberOfThreads++ < kKickOversubscription * hpxHardwareConcurrency() ||
                (selfPtr->processRange.second - selfPtr->processRange.first > 1)) {
                threadme = true;
            } else {
                threadme = false;
                numberOfThreads--;
            }
        }
    }
    if (!threadme) {
        rc = groupsFind(self, std::move(checklist), linkingLength);
    } else if (remote) {
        ASSERT(selfPtr->processRange.first >= 0);
        ASSERT(selfPtr->processRange.first < hpxSize());
        rc = hpx::async<GroupsFindAction>(hpxLocalities()[selfPtr->processRange.first], self,
            std::move(checklist), linkingLength);
    } else {
        rc = hpx::async(hpx::annotated_function(
                            [self, linkingLength](Vector<TreeId> checklist) {
                                auto rc = groupsFind(self, std::move(checklist), linkingLength);
                                numberOfThreads--;
                                return rc;
                            },
                            "analysis.groups.find"),
            std::move(checklist));
    }
    return rc;
}

struct LeafWorkspace {
    Vector<Position> xValue;
    Vector<Position> yValue;
    Vector<Position> zValue;
    Vector<GroupInteger> g;
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

hpx::future<size_t> groupsFind(TreeId self, Vector<TreeId> checklist, double linkingLength) {
    GroupTreeNode const* selfPtr = groupTreeGetNode(self);
    bool threadLeft = true;
    Vector<TreeId> nextlist = getList();
    Vector<TreeId> leaflist = getList();
    nextlist.resize(0);
    leaflist.resize(0);
    auto const selfBox = selfPtr->box.pad(linkingLength * 1.001);
    bool const iamleaf = selfPtr->children[kLeftChild].index == -1;
    do {
        for (int ci = 0; ci < checklist.size(); ci++) {
            GroupTreeNode const* otherPtr = groupTreeGetNode(checklist[ci]);
            if (otherPtr->lastActive) {
                if (selfBox.periodicIntersection(otherPtr->box).volume() > 0) {
                    if (otherPtr->children[kLeftChild].index == -1) {
                        leaflist.push_back(checklist[ci]);
                    } else {
                        nextlist.push_back(otherPtr->children[kLeftChild]);
                        nextlist.push_back(otherPtr->children[kRightChild]);
                    }
                }
            }
        }
        std::swap(nextlist, checklist);
        nextlist.resize(0);
    } while (iamleaf && checklist.size());
    if (selfPtr->children[kLeftChild].index == -1) {
        LeafWorkspace ws = getLeafWorkspace();
        Vector<Position>& xValue2 = ws.xValue;
        Vector<Position>& yValue2 = ws.yValue;
        Vector<Position>& zValue2 = ws.zValue;
        Vector<GroupInteger>& g = ws.g;
        if (leaflist.size()) {
            auto const myRng = selfPtr->particleRange;
            float const linkingLengthSquared = sqr(linkingLength);
            bool foundAnyLink = false;
            int totalSize = 0;
            for (int i = 0; i < leaflist.size(); i++) {
                GroupTreeNode const* otherPtr = groupTreeGetNode(leaflist[i]);
                if (otherPtr != selfPtr) {
                    auto const& otherRng = otherPtr->particleRange;
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
                    auto const& otherRng = otherPtr->particleRange;
                    int otherSize = otherRng.second - otherRng.first;
                    particlesGlobalReadPosAndGroup(otherPtr->globalParticleRange(), xValue2.data(),
                        yValue2.data(), zValue2.data(), g.data(), totalSize);
                    for (int i = totalSize; i < totalSize + otherSize; i++) {
                        Array<double, kDimensions> x;
                        x[kXDimension] = xValue2[i].toDouble();
                        x[kYDimension] = yValue2[i].toDouble();
                        x[kZDimension] = zValue2[i].toDouble();
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
            for (ParticleInteger k = myRng.first; k < myRng.second; k++) {
                SimdPosition sinkX =
                    simdBroadcast<SimdPosition>(particlesPos(kXDimension, k).raw());
                SimdPosition sinkY =
                    simdBroadcast<SimdPosition>(particlesPos(kYDimension, k).raw());
                SimdPosition sinkZ =
                    simdBroadcast<SimdPosition>(particlesPos(kZDimension, k).raw());
                for (int j = 0; j < totalSize; j += kSimdFloatSize) {
                    static SimdFloat const toFloatFactor =
                        simdBroadcast<SimdFloat>(kPositionToFloat);
                    int const lmax = std::min(totalSize, j + kSimdFloatSize);
                    SimdPosition sourceX = sinkX;
                    SimdPosition sourceY = sinkY;
                    SimdPosition sourceZ = sinkZ;
                    for (int l = j; l < lmax; l++) {
                        int const l0 = l - j;
                        simdSetLane(sourceX, l0, xValue2[l].raw());
                        simdSetLane(sourceY, l0, yValue2[l].raw());
                        simdSetLane(sourceZ, l0, zValue2[l].raw());
                    }
                    SimdFloat const x =
                        simdCast<SimdFloat>(simdPositionDifference(sourceX, sinkX)) * toFloatFactor;
                    SimdFloat const y =
                        simdCast<SimdFloat>(simdPositionDifference(sourceY, sinkY)) * toFloatFactor;
                    SimdFloat const z =
                        simdCast<SimdFloat>(simdPositionDifference(sourceZ, sinkZ)) * toFloatFactor;
                    SimdFloat const dist = sqr(x, y, z);
                    auto const lt = dist < linkingLengthSquared;
                    for (int l = j; l < lmax; l++) {
                        int const l0 = l - j;
                        if (lt[l0]) {
                            auto& grp = particlesGroup(k);
                            GroupInteger const startGroup = particlesGroup(k);
                            if ((GroupInteger) grp == kNoParticleGroup) {
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
                for (ParticleInteger j = myRng.first; j < myRng.second; j++) {
                    SimdPosition sinkX =
                        simdBroadcast<SimdPosition>(particlesPos(kXDimension, j).raw());
                    SimdPosition sinkY =
                        simdBroadcast<SimdPosition>(particlesPos(kYDimension, j).raw());
                    SimdPosition sinkZ =
                        simdBroadcast<SimdPosition>(particlesPos(kZDimension, j).raw());
                    for (ParticleInteger k = j + 1; k < myRng.second; k += kSimdFloatSize) {
                        static SimdFloat const toFloatFactor =
                            simdBroadcast<SimdFloat>(kPositionToFloat);
                        int const lmax = std::min(myRng.second, k + kSimdFloatSize);
                        SimdPosition sourceX = sinkX;
                        SimdPosition sourceY = sinkY;
                        SimdPosition sourceZ = sinkZ;
                        for (int l = k; l < lmax; l++) {
                            int const l0 = l - k;
                            simdSetLane(sourceX, l0, particlesPos(kXDimension, l).raw());
                            simdSetLane(sourceY, l0, particlesPos(kYDimension, l).raw());
                            simdSetLane(sourceZ, l0, particlesPos(kZDimension, l).raw());
                        }
                        SimdFloat const x =
                            simdCast<SimdFloat>(simdPositionDifference(sourceX, sinkX)) *
                            toFloatFactor;
                        SimdFloat const y =
                            simdCast<SimdFloat>(simdPositionDifference(sourceY, sinkY)) *
                            toFloatFactor;
                        SimdFloat const z =
                            simdCast<SimdFloat>(simdPositionDifference(sourceZ, sinkZ)) *
                            toFloatFactor;
                        SimdFloat const dist = sqr(x, y, z);
                        auto const lt = dist < linkingLengthSquared;
                        for (int l = k; l < lmax; l++) {
                            if (lt[l - k]) {
                                auto& grpa = particlesGroup(l);
                                auto& grpb = particlesGroup(j);
                                if ((GroupInteger) grpa == kNoParticleGroup) {
                                    grpa = particlesGroupInit(l);
                                    foundLink = true;
                                    foundAnyLink = true;
                                }
                                if ((GroupInteger) grpb == kNoParticleGroup) {
                                    grpb = particlesGroupInit(j);
                                    foundLink = true;
                                    foundAnyLink = true;
                                }
                                if (grpa != grpb) {
                                    if (grpa < grpb) {
                                        grpb = (GroupInteger) grpa;
                                    } else {
                                        grpa = (GroupInteger) grpb;
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
            GroupTreeNode const* cl = groupTreeGetNode(selfPtr->children[kLeftChild]);
            GroupTreeNode const* cr = groupTreeGetNode(selfPtr->children[kRightChild]);
            std::array<hpx::future<size_t>, kChildCount> futs;
            futs[kRightChild] =
                groupsFindFork(selfPtr->children[kRightChild], checklist, linkingLength, true);
            futs[kLeftChild] = groupsFindFork(
                selfPtr->children[kLeftChild], std::move(checklist), linkingLength, false);
            if (futs[kLeftChild].is_ready() && futs[kRightChild].is_ready()) {
                auto rcl = futs[kLeftChild].get();
                auto rcr = futs[kRightChild].get();
                auto const tot = rcl + rcr;
                groupTreeSetActive(self, tot != 0);
                return hpx::make_ready_future(tot);
            } else {
                return hpx::when_all(futs.begin(), futs.end())
                    .then(hpx::annotated_function(
                        [self](hpx::future<std::vector<hpx::future<size_t>>> futfut) {
                            auto futs = futfut.get();
                            auto const tot = futs[kLeftChild].get() + futs[kRightChild].get();
                            groupTreeSetActive(self, tot != 0);
                            return tot;
                        },
                        "analysis.groups.combineChildren"));
            }
        } else {
            groupTreeSetActive(self, false);
            return hpx::make_ready_future((size_t) 0);
        }
    }
}
