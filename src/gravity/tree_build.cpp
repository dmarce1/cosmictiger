/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <limits>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/tree_build.hpp>
#include <cosmictiger/gravity/tree_multipole.hpp>
#include <cosmictiger/math/math.hpp>

constexpr bool kVerbose = true;

TreeCreateReturn treeBuild(TreeBuildRequest request, TreeBuildPolicy const& policy) {
    auto params = request.params;
    auto const key = request.key;
    auto const processRange = request.processRange;
    auto const particleRange = request.particleRange;
    auto const box = request.box;
    auto const depth = request.depth;
    auto const localRoot = request.localRoot;
    bool const boundedGeometry = policy.boundedGeometry;
    auto const sources = fmmSourceView(policy.sourceStore);
    FlopCounter<FlopCount> flops = 0;
    double const h = getOptions().gravity.supportRadius;
    int bucketSize = getOptions().compute.bucketSize;
    TreeCreateReturn rc;
    if (depth >= kMaxDepth) {
        THROW_ERROR("%s\n", "Maximum depth exceeded\n");
    }
    if (localRoot) {
        size_t cnt = particleRange.second - particleRange.first;
        int numberOfThreads = 1;
        while (numberOfThreads < 4 * hpxHardwareConcurrency()) {
            numberOfThreads *= 2;
            cnt /= 2;
        }
        params.parallelParticleThreshold = cnt;
    }
    Array<TreeId, kChildCount> children;
    Array<Position, kDimensions>& x = rc.pos;
    MultipoleBuffer& multi = rc.multi;
    auto& rbox = rc.box;
    float& radius = rc.radius;
    Array<double, kDimensions> xcValue;
    double r;
    int const index = treeAllocateNode();
    bool isleaf = true;
    auto const numberOfParticles = particleRange.second - particleRange.first;
    flops += policy.subdivisionFlops;
    static SimdFloat const toFloatFactor = kPositionToFloat;
    if (processRange.second - processRange.first > 1 || numberOfParticles > bucketSize ||
        policy.extraSubdivision) {
        isleaf = false;
        int xdim = box.longestDim();
        flops += kDimensions;
        auto leftBox = box;
        auto rightBox = box;
        auto leftRange = processRange;
        auto rightRange = processRange;
        auto leftParticleRange = particleRange;
        auto rightParticleRange = particleRange;
        bool leftLocalRoot = false;
        bool rightLocalRoot = false;
        if (processRange.second - processRange.first > 1) {
            TreeBuildRequest left{
                params, key << 1, leftRange, leftParticleRange, leftBox, depth + 1, false};
            TreeBuildRequest right{
                params, (key << 1) + 1, rightRange, rightParticleRange, rightBox, depth + 1, false};
            policy.splitDomains(policy.context, left, right);
            leftBox = left.box;
            rightBox = right.box;
            leftRange = left.processRange;
            rightRange = right.processRange;
            leftLocalRoot = left.localRoot;
            rightLocalRoot = right.localRoot;
        } else {
            double xmid = 0.5 * (box.end[xdim] + box.begin[xdim]);
            flops += 2;
            ParticleInteger mid = fmmSourcePartition(
                policy.sourceStore, sources, particleRange, Position(xmid), xdim);
            addCpuFlops(1);
            if (boundedGeometry && (mid == particleRange.first || mid == particleRange.second)) {
                // Empty midpoint branches needlessly consume depth and arena
                // slots. Retain the same kd partition but tighten its split to
                // the occupied raw-coordinate extent, including sub-double Q64
                // separations. Coincident positions use an equality-plane tie.
                Array<PositionInteger, kDimensions> lower;
                Array<PositionInteger, kDimensions> upper;
                for (int dim = 0; dim < kDimensions; ++dim) {
                    lower[dim] = upper[dim] = sources.positions[dim][particleRange.first].raw();
                }
                for (ParticleInteger i = particleRange.first + 1; i < particleRange.second; ++i) {
                    for (int dim = 0; dim < kDimensions; ++dim) {
                        auto const value = sources.positions[dim][i].raw();
                        lower[dim] = std::min(lower[dim], value);
                        upper[dim] = std::max(upper[dim], value);
                    }
                }
                xdim = 0;
                for (int dim = 1; dim < kDimensions; ++dim) {
                    if (upper[dim] - lower[dim] > upper[xdim] - lower[xdim]) {
                        xdim = dim;
                    }
                }
                Position split;
                if (upper[xdim] != lower[xdim]) {
                    auto const span = upper[xdim] - lower[xdim];
                    split.raw() = lower[xdim] + span / 2 + span % 2;
                    mid =
                        fmmSourcePartition(policy.sourceStore, sources, particleRange, split, xdim);
                } else {
                    split.raw() = lower[xdim];
                    mid = particleRange.first + numberOfParticles / 2;
                }
                xmid = split.toDouble();
            }
            leftParticleRange.second = rightParticleRange.first = mid;
            leftBox.end[xdim] = rightBox.begin[xdim] = xmid;
        }
        auto futr = policy.fork(policy.context,
            {params, (key << 1) + 1, rightRange, rightParticleRange, rightBox, depth + 1,
                rightLocalRoot},
            true, true);
        hpx::future<TreeCreateReturn> futl;
        try {
            futl = policy.fork(policy.context,
                {params, key << 1, leftRange, leftParticleRange, leftBox, depth + 1, leftLocalRoot},
                false, false);
        } catch (...) {
            futr.wait();
            throw;
        }
        // Either child can fail after its sibling starts. Finish both before a
        // failed cell releases its source slice or permits arena cleanup.
        futl.wait();
        futr.wait();
        auto const rcl = futl.get();
        auto const rcr = futr.get();
        rc.maxGravitySupportRadius =
            std::max(rcl.maxGravitySupportRadius, rcr.maxGravitySupportRadius);
        auto const& xl = rcl.pos;
        auto const& xr = rcr.pos;
        auto const& ml = rcl.multi;
        auto const& mr = rcr.multi;
        double const& rlValue = rcl.radius;
        double const& rrValue = rcr.radius;
        auto const& boxl = rcl.box;
        auto const& boxr = rcr.box;
        for (int dim = 0; dim < kDimensions; dim++) {
            if (ml[0] != 0.0 && mr[0] != 0.0) {
                rbox.begin[dim] = std::min(boxl.begin[dim], boxr.begin[dim]);
                rbox.end[dim] = std::max(boxl.end[dim], boxr.end[dim]);
            } else if (ml[0] != 0.0) {
                rbox = boxl;
            } else if (mr[0] != 0.0) {
                rbox = boxr;
            } else {
                rbox.begin[dim] = rbox.end[dim] = 0.5 * (box.begin[dim] + box.end[dim]);
                flops += 2;
            }
        }
        Array<double, kDimensions> xlValue;
        Array<double, kDimensions> xrValue;
        for (int dim = 0; dim < kDimensions; dim++) {
            xlValue[dim] = xl[dim].toDouble();
            xrValue[dim] = xr[dim].toDouble();
        }
        flops += 2 * kDimensions;
        Array<double, kDimensions> xc1, xc2;
        double radius1 = 0.0, radius2;
        bool tried1 = false;
        // Global nodes have no rank-local particle range to rescan.
        bool const canRescan = processRange.second - processRange.first == 1 &&
            processRange.first == hpxRank() && particleRange.first >= 0 && numberOfParticles > 0 &&
            numberOfParticles <= 64 * getOptions().compute.bucketSize;
        flops += canRescan ? 1 : 0;
        if (canRescan && (rbox.end[xdim] - rbox.begin[xdim] <= 0.5)) {
            tried1 = true;
            for (int dim = 0; dim < kDimensions; dim++) {
                xc1[dim] = 0.5 * (rbox.begin[dim] + rbox.end[dim]);
            }
            Array<SimdPosition, kDimensions> xc{};
            for (int dim = 0; dim < kDimensions; dim++) {
                xc[dim] = simdBroadcast<SimdPosition>(Position(xc1[dim]).raw());
            }
            flops += 9;
            SimdFloat r2Max = simdBroadcast<SimdFloat>(0.0);
            for (ParticleInteger i = particleRange.first; i < particleRange.second;
                i += kSimdFloatSize) {
                SimdPosition x{};
                Array<SimdFloat, kDimensions> dx{};
                int const maxj = std::min(particleRange.second - i, kSimdFloatSize);
                for (int dim = 0; dim < kDimensions; dim++) {
                    for (int j = 0; j < maxj; j++) {
                        simdSetLane(x, j, sources.positions[dim][i + j].raw());
                    }
                    for (int j = maxj; j < kSimdFloatSize; j++) {
                        simdSetLane(x, j, xc[dim][0]);
                    }
                    dx[dim] =
                        simdCast<SimdFloat>(simdPositionDifference(x, xc[dim])) * toFloatFactor;
                }
                auto const r2 = sqr(dx[kXDimension], dx[kYDimension], dx[kZDimension]);
                r2Max = simdMax(r2Max, r2);
            }
            float r2 = 0.0f;
            for (int i = 0; i < kSimdFloatSize; i++) {
                r2 = std::max(r2, static_cast<float>(r2Max[i]));
            }
            flops += FlopCount(roundUp(numberOfParticles, ParticleInteger(kSimdFloatSize))) * 8 + 1;
            radius1 = sqrt(r2);
            //	PRINT("%e\n", radius);
        }
        double rr;
        double rl;
        Array<double, kDimensions> n;
        float rValue;
        double norminv = 0.0;
        for (int dim = 0; dim < kDimensions; dim++) {
            n[dim] = xlValue[dim] - xrValue[dim];
            norminv += sqr(n[dim]);
        }
        flops += 12 + (norminv > 0.0 ? 2 : 0);
        norminv = norminv > 0.0 ? 1.0 / std::sqrt(norminv) : 0.0;
        for (int dim = 0; dim < kDimensions; dim++) {
            n[dim] *= norminv;
            n[dim] = std::abs(n[dim]);
        }
        r = 0.0;
        if (mr[0] != 0.0) {
            if (ml[0] != 0.0) {
                if (norminv > 0.0) {
                    for (int dim = 0; dim < kDimensions; dim++) {
                        double const xmax = std::max(
                            xlValue[dim] + n[dim] * rlValue, xrValue[dim] + n[dim] * rrValue);
                        double const xmin = std::min(
                            xlValue[dim] - n[dim] * rlValue, xrValue[dim] - n[dim] * rrValue);
                        xc2[dim] = (xmax + xmin) * 0.5;
                        r += sqr((xmax - xmin) * 0.5);
                    }
                    flops += 42;
                } else {
                    xc2 = xlValue;
                    r = std::max(sqr(rlValue), sqr(rrValue));
                    flops += 2;
                }
            } else {
                xc2 = xrValue;
                r = rrValue * rrValue;
                flops += 1;
            }
        } else {
            if (ml[0] != 0.0) {
                xc2 = xlValue;
                r = rlValue * rlValue;
                flops += 1;
            } else {
                xc2 = rbox.begin;
            }
        }
        radius2 = std::sqrt(r);
        flops += 1;
        if (tried1 && radius1 < radius2) {
            xcValue = xc1;
            radius = radius1;
        } else {
            xcValue = xc2;
            radius = radius2;
        }
        r = 0.0;
        for (int dim = 0; dim < kDimensions; dim++) {
            double const span = (rbox.end[dim] - rbox.begin[dim]);
            ALWAYS_ASSERT(span >= 0.0);
            r += sqr(span * 0.5);
        }
        r = std::sqrt(r);
        flops += 13;
        if (r < radius) {
            radius = r;
            for (int dim = 0; dim < kDimensions; dim++) {
                xcValue[dim] = (rbox.begin[dim] + rbox.end[dim]) * 0.5;
            }
            flops += 6;
        }

        flops += 3;
        if (boundedGeometry) {
            policy.constrainCenter(policy.context, xcValue, x);
            double leftDistanceSquared = 0.0;
            double rightDistanceSquared = 0.0;
            for (int dim = 0; dim < kDimensions; ++dim) {
                leftDistanceSquared += sqr(unwrappedDoubleDistance(xl[dim], x[dim]));
                rightDistanceSquared += sqr(unwrappedDoubleDistance(xr[dim], x[dim]));
            }
            // Bound the children about the actual stored center. This also
            // protects sub-double Q64 extents and center-clamping roundoff.
            double const enclosingRadius = std::max(std::sqrt(leftDistanceSquared) + rlValue,
                std::sqrt(rightDistanceSquared) + rrValue);
            radius = std::nextafter(float(enclosingRadius), std::numeric_limits<float>::infinity());
        } else {
            for (int dim = 0; dim < kDimensions; dim++) {
                x[dim] = xcValue[dim];
            }
        }
        flops += fmmDispatch([&](auto orderTag) {
            return treeComputeMultipole<decltype(orderTag)::value>(
                multi, ml, mr, xlValue, xrValue, xcValue);
        });
        children[kLeftChild] = rcl.id;
        children[kRightChild] = rcr.id;
    } else {
        // All species use the same fixed comoving gravity support. Gas smoothing
        // lengths belong only to the hydrodynamic tree and neighbor searches.
        rc.maxGravitySupportRadius = numberOfParticles > 0 ? float(h) : 0.0f;
        Array<double, kDimensions>& xminValue = rbox.begin;
        Array<double, kDimensions>& xmaxValue = rbox.end;
        for (int dim = 0; dim < kDimensions; dim++) {
            xminValue[dim] = box.end[dim];
            xmaxValue[dim] = box.begin[dim];
            if (numberOfParticles == 0) {
                xminValue[dim] = xmaxValue[dim] = 0.5 * (box.begin[dim] + box.end[dim]);
                flops += 2;
            }
        }
        for (ParticleInteger i = particleRange.first; i < particleRange.second; i++) {
            for (int dim = 0; dim < kDimensions; dim++) {
                auto const position = sources.positions[dim][i];
                double lower = position.toDouble();
                double upper = lower;
                if (boundedGeometry) {
                    lower = positionDomainCoordinate(position);
                    if (upper < 1.0 && Position(upper).raw() < position.raw()) {
                        upper = std::nextafter(upper, 1.0);
                    }
                }
                xmaxValue[dim] = std::max(xmaxValue[dim], upper);
                xminValue[dim] = std::min(xminValue[dim], lower);
            }
        }
        for (int dim = 0; dim < kDimensions; dim++) {
            xcValue[dim] = (xmaxValue[dim] + xminValue[dim]) * 0.5;
        }
        if (boundedGeometry) {
            policy.constrainCenter(policy.context, xcValue, x);
        }
        r = 0.0;
        flops += 6 + FlopCount(numberOfParticles) * 15;
        for (ParticleInteger i = particleRange.first; i < particleRange.second; i++) {
            double thisRadius = 0.0;
            for (int dim = 0; dim < kDimensions; dim++) {
                double const delta = boundedGeometry ?
                    unwrappedDoubleDistance(sources.positions[dim][i], x[dim]) :
                    sources.positions[dim][i].toDouble() - xcValue[dim];
                thisRadius += sqr(delta);
            }
            r = std::max(r, thisRadius);
        }
        radius = std::sqrt(r);
        if (boundedGeometry) {
            radius = std::nextafter(radius, std::numeric_limits<float>::infinity());
        }
        flops += 4;    // Square root and three center-to-fixed conversions below.
        children[kLeftChild].index = children[kRightChild].index = -1;
        flops += fmmDispatch([&](auto orderTag) {
            return treeComputeMultipole<decltype(orderTag)::value>(
                multi, particleRange, xcValue, sources);
        });
        for (int dim = 0; dim < kDimensions; dim++) {
            x[dim] = xcValue[dim];
        }
    }
    TreeNode node;
    node.radius = radius;
    node.maxGravitySupportRadius = rc.maxGravitySupportRadius;
    node.children = children;
    node.localRoot = localRoot;
    node.particleRange = particleRange;
    node.sourceStore = policy.sourceStore;
    node.processRange = policy.representedProcessRange;
    node.pos = x;
    node.depth = depth;
    for (int dim = 0; dim < kDimensions; dim++) {
        node.box.begin[dim] = rbox.begin[dim];
        node.box.end[dim] = rbox.end[dim];
        // Stored bounds have fewer fractional bits than particle positions.
        // Round outward so box-based acceptance cannot exclude an extremum.
        if (node.box.begin[dim].toDouble() > rbox.begin[dim]) {
            node.box.begin[dim].raw()--;
        }
        if (node.box.end[dim].toDouble() < rbox.end[dim]) {
            node.box.end[dim].raw()++;
        }
    }
    flops += 4 * kDimensions;    // Bounds to/from fixed-point for outward rounding.
    bool const global = processRange.second - processRange.first > 1;
    node.leaf = isleaf;
    treeStoreNode(index, node, multi);
    rc.id.index = index;
    rc.id.proc = hpxRank();
    addCpuFlops(flops);
    return rc;
}
