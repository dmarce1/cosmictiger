/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <limits>

#include <cosmictiger/core/assert.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/analysis/bh.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/simd.hpp>

int bhSort(DeviceVector<Array<float, kDimensions>>& parts, Vector<int>& sortOrder, int begin,
    int end, float xm, int xdim) {
    int lo = begin;
    int hi = end;
    float xmid(xm);
    while (lo < hi) {
        if (parts[lo][xdim] >= xmid) {
            while (lo != hi) {
                hi--;
                if (parts[hi][xdim] < xmid) {
                    std::swap(sortOrder[hi], sortOrder[lo]);
                    auto tmp = parts[hi];
                    parts[hi] = parts[lo];
                    parts[lo] = tmp;
                    break;
                }
            }
        }
        lo++;
    }
    return hi;
}

void bhCreateTree(DeviceVector<BhTreeNode>& nodes, DeviceVector<int>& sinkBuckets, int self,
    DeviceVector<Array<float, kDimensions>>& parts, Vector<int>& sortOrder, Range<float> box,
    int begin, int end, int bucketSize, int depth = 0) {
    ALWAYS_ASSERT(begin < end);
    // Recompute tight bounds after every split, including the count-based
    // fallback for coincident points or a midpoint rounded onto a boundary.
    Array<double, kDimensions> sum{};
    for (int dim = 0; dim < kDimensions; ++dim) {
        box.begin[dim] = std::numeric_limits<float>::max();
        box.end[dim] = -std::numeric_limits<float>::max();
    }
    for (int i = begin; i < end; ++i) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            sum[dim] += parts[i][dim];
            box.begin[dim] = std::min(box.begin[dim], parts[i][dim]);
            box.end[dim] = std::max(box.end[dim], parts[i][dim]);
        }
    }
    auto& node = nodes[self];
    node.parts = {begin, end};
    node.mass = end - begin;
    for (int dim = 0; dim < kDimensions; ++dim) {
        node.pos[dim] = sum[dim] / node.mass;
    }
    // ROCKSTAR Appendix B, Eq. (B2): equal-mass cell variance about its COM.
    // The maximum radius supplies r_max in Eqs. (B1) and (B3); see bh.hpp.
    double maximumSquaredRadius = 0.0;
    double sumSquaredRadius = 0.0;
    for (int i = begin; i < end; ++i) {
        double squaredRadius = 0.0;
        for (int dim = 0; dim < kDimensions; ++dim) {
            double const displacement = double(parts[i][dim]) - node.pos[dim];
            squaredRadius += displacement * displacement;
        }
        maximumSquaredRadius = std::max(maximumSquaredRadius, squaredRadius);
        sumSquaredRadius += squaredRadius;
    }
    // Round the enclosing radius outward; underestimating it would invalidate
    // the nearest-sink and compact-support guards in the acceptance criterion.
    node.radius = std::nextafter(
        float(std::sqrt(maximumSquaredRadius)), std::numeric_limits<float>::infinity());
    node.meanSquaredRadius =
        std::nextafter(float(sumSquaredRadius / node.mass), std::numeric_limits<float>::infinity());
    if (end - begin <= bucketSize) {
        sinkBuckets.pushBack(self);
        return;
    }
    int const dimension = box.longestDim();
    float const midpoint = float(0.5 * (double(box.begin[dimension]) + box.end[dimension]));
    int middle = bhSort(parts, sortOrder, begin, end, midpoint, dimension);
    // Every child must contain fewer particles. Limit highly unbalanced spatial
    // recursion too; equal-position clouds then terminate in ordinary buckets.
    if (middle == begin || middle == end || depth >= 32) {
        middle = begin + (end - begin) / 2;
    }
    int const left = nodes.size();
    int const right = left + 1;
    node.children[kLeftChild] = left;
    node.children[kRightChild] = right;
    nodes.resize(nodes.size() + kChildCount);
    bhCreateTree(
        nodes, sinkBuckets, left, parts, sortOrder, box, begin, middle, bucketSize, depth + 1);
    bhCreateTree(
        nodes, sinkBuckets, right, parts, sortOrder, box, middle, end, bucketSize, depth + 1);
}

void bhTreeEvaluate(DeviceVector<BhTreeNode> const& nodes, DeviceVector<int>& sinkBuckets,
    DeviceVector<float>& phi, DeviceVector<Array<float, kDimensions>>& parts, float relativeError) {
    float const h = getOptions().gravity.supportRadius;
    float const gm = getOptions().gravity.gm;
    float const hinv = 1.0 / h;
    float const h2 = h * h;
    SimdFloat const tiny = simdBroadcast<SimdFloat>(1e-20);
    int const numberOfThreads =
        std::max(std::min((int) parts.size() / 512, 2 * (int) hpxHardwareConcurrency()), 1);
    Vector<hpx::future<void>> futs;
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([proc, numberOfThreads, &sinkBuckets, &nodes, &phi, &parts, h, gm,
                                      hinv, h2, relativeError, tiny]() {
            for (int i = proc; i < sinkBuckets.size(); i += numberOfThreads) {
                auto const& sinkBucket = sinkBuckets[i];
                Vector<int> checklist(1, 0);
                Vector<int> nextlist;
                Vector<BhSource> sourcelist;
                auto const myparts = nodes[sinkBucket].parts;
                auto const mypos = nodes[sinkBucket].pos;
                auto const myradius = nodes[sinkBucket].radius;
                for (int i = myparts.first; i < myparts.second; i++) {
                    phi[i] = selfPhi() * hinv;
                }
                while (checklist.size()) {
                    nextlist.resize(0);
                    for (int ci = 0; ci < checklist.size(); ci++) {
                        auto const& node = nodes[checklist[ci]];
                        float const dx = mypos[kXDimension] - node.pos[kXDimension];
                        float const dy = mypos[kYDimension] - node.pos[kYDimension];
                        float const dz = mypos[kZDimension] - node.pos[kZDimension];
                        float const r2 = sqr(dx, dy, dz);
                        if (detail::bhAcceptMonopole(r2, node, myradius, h, relativeError)) {
                            BhSource source;
                            source.x = node.pos;
                            source.m = node.mass;
                            sourcelist.push_back(source);
                        } else if (node.children[kLeftChild] == -1) {
                            for (int i = node.parts.first; i < node.parts.second; i++) {
                                BhSource source;
                                source.x = parts[i];
                                source.m = 1.0f;
                                sourcelist.push_back(source);
                            }
                        } else {
                            nextlist.push_back(node.children[kLeftChild]);
                            nextlist.push_back(node.children[kRightChild]);
                        }
                    }
                    std::swap(nextlist, checklist);
                }
                int const maxi = roundUp((int) sourcelist.size(), kSimdFloatSize);
                while (sourcelist.size() < maxi) {
                    BhSource src;
                    src.m = 0.0f;
                    for (int dim = 0; dim < kDimensions; dim++) {
                        src.x[dim] = 0.f;
                    }
                    sourcelist.push_back(src);
                }
                for (int i = myparts.first; i < myparts.second; i++) {
                    Array<SimdFloat, kDimensions> xValue{};
                    for (int dim = 0; dim < kDimensions; dim++) {
                        xValue[dim] = parts[i][dim];
                    }
                    for (int j = 0; j < sourcelist.size(); j += kSimdFloatSize) {
                        Array<SimdFloat, kDimensions> yValue{};
                        SimdFloat mValue{};
                        for (int k = 0; k < kSimdFloatSize; k++) {
                            auto const& src = sourcelist[j + k];
                            for (int dim = 0; dim < kDimensions; dim++) {
                                simdSetLane(yValue[dim], k, src.x[dim]);
                            }
                            simdSetLane(mValue, k, src.m);
                        }
                        Array<SimdFloat, kDimensions> dx{};
                        for (int dim = 0; dim < kDimensions; dim++) {
                            dx[dim] = xValue[dim] - yValue[dim];    // 3
                        }
                        SimdFloat const separationSquared =
                            sqr(dx[kXDimension], dx[kYDimension], dx[kZDimension]);
                        SimdFloat const r2 = simdMax(separationSquared, tiny);
                        auto const farFlag = separationSquared > h2;
                        SimdFloat rinv1{};
                        if (simdAllOf(farFlag)) {     // 7/8
                            rinv1 = simdRsqrt(r2);    // 5
                        } else {
                            SimdFloat const r = simdSqrt(separationSquared);
                            SimdFloat const rinv1Far = simdBroadcast<SimdFloat>(1) /
                                (r + simdBroadcast<SimdFloat>(1e-30));    // 5
                            SimdFloat const q = r * hinv;
                            SimdFloat const rinv1Near =
                                radial_kernel::potential(selectedGravityKernel(), q) * hinv;
                            rinv1 = simdSelect(farFlag, rinv1Far, rinv1Near);    // 4
                        }
                        phi[i] -= simdSum(mValue * rinv1);    // 1
                    }
                    phi[i] *= gm;
                }
            }
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());
}

void bhTreeEvaluatePoint(DeviceVector<BhTreeNode> const& nodes, Array<float, kDimensions> pt,
    float& phi, DeviceVector<Array<float, kDimensions>>& parts, float relativeError) {
    float const h = getOptions().gravity.supportRadius;
    float const gm = getOptions().gravity.gm;
    float const hinv = 1.0f / h;
    float const h2 = h * h;
    SimdFloat const tiny = simdBroadcast<SimdFloat>(1e-20);
    int const numberOfThreads =
        std::max(std::min((int) parts.size() / 512, 2 * (int) hpxHardwareConcurrency()), 1);
    Vector<hpx::future<void>> futs;
    Vector<int> checklist(1, 0);
    Vector<int> nextlist;
    Vector<BhSource> sourcelist;
    phi = 0.0;
    while (checklist.size()) {
        nextlist.resize(0);
        for (int ci = 0; ci < checklist.size(); ci++) {
            auto const& node = nodes[checklist[ci]];
            float const dx = pt[kXDimension] - node.pos[kXDimension];
            float const dy = pt[kYDimension] - node.pos[kYDimension];
            float const dz = pt[kZDimension] - node.pos[kZDimension];
            float const r2 = sqr(dx, dy, dz);
            if (detail::bhAcceptMonopole(r2, node, 0.0f, h, relativeError)) {
                BhSource source;
                source.x = node.pos;
                source.m = node.mass;
                sourcelist.push_back(source);
            } else if (node.children[kLeftChild] == -1) {
                for (int i = node.parts.first; i < node.parts.second; i++) {
                    BhSource source;
                    source.x = parts[i];
                    source.m = 1.0f;
                    sourcelist.push_back(source);
                }
            } else {
                nextlist.push_back(node.children[kLeftChild]);
                nextlist.push_back(node.children[kRightChild]);
            }
        }
        std::swap(nextlist, checklist);
    }
    int const maxi = roundUp((int) sourcelist.size(), kSimdFloatSize);
    while (sourcelist.size() < maxi) {
        BhSource src;
        src.m = 0.0f;
        for (int dim = 0; dim < kDimensions; dim++) {
            src.x[dim] = 0.f;
        }
        sourcelist.push_back(src);
    }
    Array<SimdFloat, kDimensions> xValue{};
    for (int dim = 0; dim < kDimensions; dim++) {
        xValue[dim] = pt[dim];
    }
    for (int j = 0; j < sourcelist.size(); j += kSimdFloatSize) {
        Array<SimdFloat, kDimensions> yValue{};
        SimdFloat mValue{};
        for (int k = 0; k < kSimdFloatSize; k++) {
            auto const& src = sourcelist[j + k];
            for (int dim = 0; dim < kDimensions; dim++) {
                simdSetLane(yValue[dim], k, src.x[dim]);
            }
            simdSetLane(mValue, k, src.m);
        }
        Array<SimdFloat, kDimensions> dx{};
        for (int dim = 0; dim < kDimensions; dim++) {
            dx[dim] = xValue[dim] - yValue[dim];    // 3
        }
        SimdFloat const separationSquared = sqr(dx[kXDimension], dx[kYDimension], dx[kZDimension]);
        SimdFloat const r2 = simdMax(separationSquared, tiny);
        auto const farFlag = separationSquared > h2;
        SimdFloat rinv1{};
        if (simdAllOf(farFlag)) {     // 7/8
            rinv1 = simdRsqrt(r2);    // 5
        } else {
            SimdFloat const r = simdSqrt(separationSquared);
            SimdFloat const rinv1Far =
                simdBroadcast<SimdFloat>(1) / (r + simdBroadcast<SimdFloat>(1e-30f));    // 5
            SimdFloat const q = r * hinv;
            SimdFloat const rinv1Near = simdSelect(separationSquared > simdBroadcast<SimdFloat>(0),
                radial_kernel::potential(selectedGravityKernel(), q) * hinv,
                simdBroadcast<SimdFloat>(0));
            rinv1 = simdSelect(farFlag, rinv1Far, rinv1Near);    // 4
        }
        phi -= simdSum(mValue * rinv1);    // 1
    }
    phi *= gm;
}

DeviceVector<float> bhEvaluatePotential(DeviceVector<Array<float, kDimensions>>& x, bool gpu) {
    gpu = gpu && getOptions().compute.cuda;
#ifndef USE_CUDA
    gpu = false;
#endif
    if (x.size() <= 1) {
        DeviceVector<float> potential(x.size());
        if (x.size() == 1) {
            potential[0] = 0.0f;
        }
        return potential;
    }
    DeviceVector<BhTreeNode> nodes(1);
    Range<float> box;
    for (int dim = 0; dim < kDimensions; dim++) {
        box.begin[dim] = std::numeric_limits<float>::max();
        box.end[dim] = -std::numeric_limits<float>::max();
    }
    for (auto const& pos : x) {
        for (int dim = 0; dim < kDimensions; dim++) {
            auto const thisX = pos[dim];
            box.begin[dim] = std::min(box.begin[dim], thisX);
            box.end[dim] = std::max(box.end[dim], thisX);
        }
    }
    DeviceVector<float> rpot;
    DeviceVector<int> sinkBuckets;
    Vector<int> sortOrder;
    sortOrder.reserve(x.size());
    for (int i = 0; i < x.size(); i++) {
        sortOrder.push_back(i);
    }
    DeviceVector<float> pot(x.size());
    bhCreateTree(nodes, sinkBuckets, 0, x, sortOrder, box, 0, x.size(), kBhBucketSize);
    if (!gpu) {
        bhTreeEvaluate(nodes, sinkBuckets, pot, x, kBhMaximumRelativePotentialError);
    } else {
#ifdef USE_CUDA
        Timer tm1;
        tm1.start();
        pot = bhEvaluatePotentialGpu(nodes, x, sinkBuckets, kBhMaximumRelativePotentialError,
            getOptions().gravity.supportRadius, getOptions().gravity.gm);
        tm1.stop();
        //		PRINT("1 %e\n", tm1.read());
#endif
    }
    rpot.resize(x.size());
    for (int i = 0; i < sortOrder.size(); i++) {
        rpot[sortOrder[i]] = pot[i];
    }
    return rpot;
}

DeviceVector<float> bhEvaluatePoints(DeviceVector<Array<float, kDimensions>>& y,
    DeviceVector<Array<float, kDimensions>>& x, bool gpu) {
    gpu = gpu && getOptions().compute.cuda;
#ifndef USE_CUDA
    gpu = false;
#endif
    if (x.size() == 0 || y.size() == 0) {
        DeviceVector<float> potential(y.size());
        for (int i = 0; i < potential.size(); ++i) {
            potential[i] = 0.0f;
        }
        return potential;
    }
    DeviceVector<BhTreeNode> nodes(1);
    Range<float> box;
    for (int dim = 0; dim < kDimensions; dim++) {
        box.begin[dim] = std::numeric_limits<float>::max();
        box.end[dim] = -std::numeric_limits<float>::max();
    }
    for (auto const& pos : x) {
        for (int dim = 0; dim < kDimensions; dim++) {
            auto const thisX = pos[dim];
            box.begin[dim] = std::min(box.begin[dim], thisX);
            box.end[dim] = std::max(box.end[dim], thisX);
        }
    }
    Vector<float> rpot;
    DeviceVector<int> sinkBuckets;
    Vector<int> sortOrder;
    sortOrder.reserve(x.size());
    for (int i = 0; i < x.size(); i++) {
        sortOrder.push_back(i);
    }
    DeviceVector<float> pot(y.size());
    bhCreateTree(nodes, sinkBuckets, 0, x, sortOrder, box, 0, x.size(), kBhBucketSize);
    if (gpu) {
#ifdef USE_CUDA
        pot = bhEvaluatePotentialPointsGpu(nodes, x, y, kBhMaximumRelativePotentialError,
            getOptions().gravity.supportRadius, getOptions().gravity.gm);
#endif
    } else {
        for (int i = 0; i < y.size(); i++) {
            bhTreeEvaluatePoint(nodes, y[i], pot[i], x, kBhMaximumRelativePotentialError);
        }
    }
    return pot;
}
