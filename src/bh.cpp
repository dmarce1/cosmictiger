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

#include <fenv.h>

#include <cosmictiger/assert.hpp>
#include <cosmictiger/bh.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/simd.hpp>
#include <cosmictiger/timer.hpp>

int bhSort(DeviceVector<Array<float, NDIM>>& parts, Vector<int>& sortOrder, int begin, int end,
    float xm, int xdim) {
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
    DeviceVector<Array<float, NDIM>>& parts, Vector<int>& sortOrder, Range<float> box, int begin,
    int end, int bucketSize, int depth = 0) {
    /*feenableexcept (FE_DIVBYZERO);
     feenableexcept (FE_INVALID);
     feenableexcept (FE_OVERFLOW);*/
    auto* nodePtr = &nodes[self];
    Array<float, NDIM> com;
    float radius;
    if (end - begin <= bucketSize) {
        nodePtr->parts.first = begin;
        nodePtr->parts.second = end;
        sinkBuckets.pushBack(self);
        nodePtr->mass = end - begin;
        for (int dim = 0; dim < NDIM; dim++) {
            com[dim] = 0.0;
        }
        if (nodePtr->mass > 0.0) {
            for (int i = begin; i < end; i++) {
                for (int dim = 0; dim < NDIM; dim++) {
                    com[dim] += parts[i][dim];
                }
            }
            float const massinv = 1.0f / nodePtr->mass;
            for (int dim = 0; dim < NDIM; dim++) {
                com[dim] *= massinv;
            }
        }
        radius = 0.0;
        for (int i = begin; i < end; i++) {
            float r2 = 0.0;
            for (int dim = 0; dim < NDIM; dim++) {
                r2 += sqr(com[dim] - parts[i][dim]);
            }
            radius = std::max(radius, r2);
        }
        radius = std::sqrt(radius);
        nodePtr->radius = radius;
        nodePtr->pos = com;
    } else {
        int const xdim = box.longestDim();
        float const xmid = (box.begin[xdim] + box.end[xdim]) * 0.5;
        int const mid = bhSort(parts, sortOrder, begin, end, xmid, xdim);
        auto boxl = box;
        auto boxr = box;
        boxl.end[xdim] = boxr.begin[xdim] = xmid;
        nodePtr->children[LEFT] = nodes.size();
        nodePtr->children[RIGHT] = nodes.size() + 1;
        nodes.resize(nodes.size() + NCHILD);
        nodePtr = &nodes[self];
        bhCreateTree(nodes, sinkBuckets, nodePtr->children[LEFT], parts, sortOrder, boxl, begin,
            mid, bucketSize, depth + 1);
        nodePtr = &nodes[self];
        bhCreateTree(nodes, sinkBuckets, nodePtr->children[RIGHT], parts, sortOrder, boxr, mid, end,
            bucketSize, depth + 1);
        nodePtr = &nodes[self];
        auto const& childl = nodes[nodePtr->children[LEFT]];
        auto const& childr = nodes[nodePtr->children[RIGHT]];
        nodePtr->mass = childl.mass + childr.mass;
        if (childl.mass == 0.0) {
            for (int dim = 0; dim < NDIM; dim++) {
                com[dim] = childr.pos[dim];
            }
            radius = childr.radius;
        } else if (childr.mass == 0.0) {
            for (int dim = 0; dim < NDIM; dim++) {
                com[dim] = childl.pos[dim];
            }
            radius = childl.radius;
        } else {
            float const massinv = 1.0f / nodePtr->mass;
            for (int dim = 0; dim < NDIM; dim++) {
                com[dim] =
                    (childl.mass * childl.pos[dim] + childr.mass * childr.pos[dim]) * massinv;
            }
            float r2l = 0.0;
            float r2r = 0.0;
            float r2 = 0.0;
            for (int dim = 0; dim < NDIM; dim++) {
                r2l += sqr(com[dim] - childl.pos[dim]);
                r2r += sqr(com[dim] - childr.pos[dim]);
            }
            r2 = std::max(r2,
                (float) sqr(box.begin[XDIM] - com[XDIM], box.begin[YDIM] - com[YDIM],
                    box.begin[ZDIM] - com[ZDIM]));
            r2 = std::max(r2,
                (float) sqr(box.begin[XDIM] - com[XDIM], box.begin[YDIM] - com[YDIM],
                    box.end[ZDIM] - com[ZDIM]));
            r2 = std::max(r2,
                (float) sqr(box.begin[XDIM] - com[XDIM], box.end[YDIM] - com[YDIM],
                    box.begin[ZDIM] - com[ZDIM]));
            r2 = std::max(r2,
                (float) sqr(box.begin[XDIM] - com[XDIM], box.end[YDIM] - com[YDIM],
                    box.end[ZDIM] - com[ZDIM]));
            r2 = std::max(r2,
                (float) sqr(box.end[XDIM] - com[XDIM], box.begin[YDIM] - com[YDIM],
                    box.begin[ZDIM] - com[ZDIM]));
            r2 = std::max(r2,
                (float) sqr(box.end[XDIM] - com[XDIM], box.begin[YDIM] - com[YDIM],
                    box.end[ZDIM] - com[ZDIM]));
            r2 = std::max(r2,
                (float) sqr(box.end[XDIM] - com[XDIM], box.end[YDIM] - com[YDIM],
                    box.begin[ZDIM] - com[ZDIM]));
            r2 = std::max(r2,
                (float) sqr(box.end[XDIM] - com[XDIM], box.end[YDIM] - com[YDIM],
                    box.end[ZDIM] - com[ZDIM]));
            radius = std::sqrt(r2);
        }
        nodePtr->radius = radius;
        nodePtr->pos = com;
    }
}

void bhTreeEvaluate(DeviceVector<BhTreeNode> const& nodes, DeviceVector<int>& sinkBuckets,
    DeviceVector<float>& phi, DeviceVector<Array<float, NDIM>>& parts, float theta) {
    float const h = getOptions().gravitySupportRadius;
    float const gm = getOptions().gm;
    float const hinv = 1.0 / h;
    float const h2 = h * h;
    SimdFloat const tiny(1e-20);
    int const numberOfThreads =
        std::max(std::min((int) parts.size() / 512, 2 * (int) hpxHardwareConcurrency()), 1);
    Vector<hpx::future<void>> futs;
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async([proc, numberOfThreads, &sinkBuckets, &nodes, &phi, &parts, h, gm,
                                      hinv, h2, theta, tiny]() {
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
                float const thetainv = 1.0f / theta;
                while (checklist.size()) {
                    nextlist.resize(0);
                    for (int ci = 0; ci < checklist.size(); ci++) {
                        auto const& node = nodes[checklist[ci]];
                        float const dx = mypos[XDIM] - node.pos[XDIM];
                        float const dy = mypos[YDIM] - node.pos[YDIM];
                        float const dz = mypos[ZDIM] - node.pos[ZDIM];
                        float const r2 = sqr(dx, dy, dz);
                        if (r2 > sqr(thetainv * (node.radius + myradius))) {
                            BhSource source;
                            source.x = node.pos;
                            source.m = node.mass;
                            sourcelist.push_back(source);
                        } else if (node.children[LEFT] == -1) {
                            for (int i = node.parts.first; i < node.parts.second; i++) {
                                BhSource source;
                                source.x = parts[i];
                                source.m = 1.0f;
                                sourcelist.push_back(source);
                            }
                        } else {
                            nextlist.push_back(node.children[LEFT]);
                            nextlist.push_back(node.children[RIGHT]);
                        }
                    }
                    std::swap(nextlist, checklist);
                }
                int const maxi = roundUp((int) sourcelist.size(), SIMD_FLOAT_SIZE);
                while (sourcelist.size() < maxi) {
                    BhSource src;
                    src.m = 0.0f;
                    for (int dim = 0; dim < NDIM; dim++) {
                        src.x[dim] = 0.f;
                    }
                    sourcelist.push_back(src);
                }
                for (int i = myparts.first; i < myparts.second; i++) {
                    Array<SimdFloat, NDIM> xValue;
                    for (int dim = 0; dim < NDIM; dim++) {
                        xValue[dim] = parts[i][dim];
                    }
                    for (int j = 0; j < sourcelist.size(); j += SIMD_FLOAT_SIZE) {
                        Array<SimdFloat, NDIM> yValue;
                        SimdFloat mValue;
                        for (int k = 0; k < SIMD_FLOAT_SIZE; k++) {
                            auto const& src = sourcelist[j + k];
                            for (int dim = 0; dim < NDIM; dim++) {
                                yValue[dim][k] = src.x[dim];
                            }
                            mValue[k] = src.m;
                        }
                        Array<SimdFloat, NDIM> dx;
                        for (int dim = 0; dim < NDIM; dim++) {
                            dx[dim] = xValue[dim] - yValue[dim];    // 3
                        }
                        SimdFloat const r2 = max(sqr(dx[XDIM], dx[YDIM], dx[ZDIM]), tiny);    // 5
                        SimdFloat const farFlag = r2 > h2;                                    // 1
                        SimdFloat rinv1;
                        if (farFlag.sum() == SIMD_FLOAT_SIZE) {    // 7/8
                            rinv1 = rsqrt(r2);                     // 5
                        } else {
                            SimdFloat const r = sqrt(r2);    // 4
                            SimdFloat const rinv1Far =
                                SimdFloat(1) / (r + SimdFloat(1e-30));    // 5
                            SimdFloat const q = r * hinv;
                            SimdFloat const rinv1Near =
                                radial_kernel::potential(selectedGravityKernel(), q) * hinv;
                            auto const nearFlag = (SimdFloat(1) - farFlag);       // 1
                            rinv1 = farFlag * rinv1Far + nearFlag * rinv1Near;    // 4
                        }
                        phi[i] -= (mValue * rinv1).sum();    // 1
                    }
                    phi[i] *= gm;
                }
            }
        }));
    }
    hpx::wait_all(futs.begin(), futs.end());
}

void bhTreeEvaluatePoint(DeviceVector<BhTreeNode> const& nodes, Array<float, NDIM> pt, float& phi,
    DeviceVector<Array<float, NDIM>>& parts, float theta) {
    float const h = getOptions().gravitySupportRadius;
    float const gm = getOptions().gm;
    float const hinv = 1.0f / h;
    float const h2 = h * h;
    SimdFloat const tiny(1e-20);
    int const numberOfThreads =
        std::max(std::min((int) parts.size() / 512, 2 * (int) hpxHardwareConcurrency()), 1);
    Vector<hpx::future<void>> futs;
    Vector<int> checklist(1, 0);
    Vector<int> nextlist;
    Vector<BhSource> sourcelist;
    phi = 0.0;
    float const thetainv = 1.0f / theta;
    while (checklist.size()) {
        nextlist.resize(0);
        for (int ci = 0; ci < checklist.size(); ci++) {
            auto const& node = nodes[checklist[ci]];
            float const dx = pt[XDIM] - node.pos[XDIM];
            float const dy = pt[YDIM] - node.pos[YDIM];
            float const dz = pt[ZDIM] - node.pos[ZDIM];
            float const r2 = sqr(dx, dy, dz);
            if (r2 > sqr(thetainv * node.radius)) {
                BhSource source;
                source.x = node.pos;
                source.m = node.mass;
                sourcelist.push_back(source);
            } else if (node.children[LEFT] == -1) {
                for (int i = node.parts.first; i < node.parts.second; i++) {
                    BhSource source;
                    source.x = parts[i];
                    source.m = 1.0f;
                    sourcelist.push_back(source);
                }
            } else {
                nextlist.push_back(node.children[LEFT]);
                nextlist.push_back(node.children[RIGHT]);
            }
        }
        std::swap(nextlist, checklist);
    }
    int const maxi = roundUp((int) sourcelist.size(), SIMD_FLOAT_SIZE);
    while (sourcelist.size() < maxi) {
        BhSource src;
        src.m = 0.0f;
        for (int dim = 0; dim < NDIM; dim++) {
            src.x[dim] = 0.f;
        }
        sourcelist.push_back(src);
    }
    Array<SimdFloat, NDIM> xValue;
    for (int dim = 0; dim < NDIM; dim++) {
        xValue[dim] = pt[dim];
    }
    for (int j = 0; j < sourcelist.size(); j += SIMD_FLOAT_SIZE) {
        Array<SimdFloat, NDIM> yValue;
        SimdFloat mValue;
        for (int k = 0; k < SIMD_FLOAT_SIZE; k++) {
            auto const& src = sourcelist[j + k];
            for (int dim = 0; dim < NDIM; dim++) {
                yValue[dim][k] = src.x[dim];
            }
            mValue[k] = src.m;
        }
        Array<SimdFloat, NDIM> dx;
        for (int dim = 0; dim < NDIM; dim++) {
            dx[dim] = xValue[dim] - yValue[dim];    // 3
        }
        SimdFloat const separationSquared = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
        SimdFloat const r2 = max(separationSquared, tiny);
        SimdFloat const farFlag = r2 > h2;    // 1
        SimdFloat rinv1;
        if (farFlag.sum() == SIMD_FLOAT_SIZE) {    // 7/8
            rinv1 = rsqrt(r2);                     // 5
        } else {
            SimdFloat const r = sqrt(r2);                                         // 4
            SimdFloat const rinv1Far = SimdFloat(1) / (r + SimdFloat(1e-30f));    // 5
            SimdFloat const q = r * hinv;
            SimdFloat const rinv1Near = radial_kernel::potential(selectedGravityKernel(), q) *
                hinv * (separationSquared > SimdFloat(0));
            auto const nearFlag = (SimdFloat(1) - farFlag);       // 1
            rinv1 = farFlag * rinv1Far + nearFlag * rinv1Near;    // 4
        }
        phi -= (mValue * rinv1).sum();    // 1
    }
    phi *= gm;
}

DeviceVector<float> bhEvaluatePotential(DeviceVector<Array<float, NDIM>>& x, bool gpu) {
    ALWAYS_ASSERT(x.size() > 1);
    DeviceVector<BhTreeNode> nodes(1);
    Range<float> box;
    for (int dim = 0; dim < NDIM; dim++) {
        box.begin[dim] = std::numeric_limits<float>::max();
        box.end[dim] = -std::numeric_limits<float>::max();
    }
    for (auto const& pos : x) {
        for (int dim = 0; dim < NDIM; dim++) {
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
    bhCreateTree(nodes, sinkBuckets, 0, x, sortOrder, box, 0, x.size(), BH_BUCKET_SIZE);
    if (!gpu) {
        bhTreeEvaluate(nodes, sinkBuckets, pot, x, 0.85);
    } else {
        Timer tm1;
        tm1.start();
        pot = bhEvaluatePotentialGpu(
            nodes, x, sinkBuckets, 0.85, getOptions().gravitySupportRadius, getOptions().gm);
        tm1.stop();
        //		PRINT("1 %e\n", tm1.read());
    }
    rpot.resize(x.size());
    for (int i = 0; i < sortOrder.size(); i++) {
        rpot[sortOrder[i]] = pot[i];
    }
    return rpot;
}

DeviceVector<float> bhEvaluatePoints(
    DeviceVector<Array<float, NDIM>>& y, DeviceVector<Array<float, NDIM>>& x, bool gpu) {
    ALWAYS_ASSERT(x.size() > 1);
    DeviceVector<BhTreeNode> nodes(1);
    Range<float> box;
    for (int dim = 0; dim < NDIM; dim++) {
        box.begin[dim] = std::numeric_limits<float>::max();
        box.end[dim] = -std::numeric_limits<float>::max();
    }
    for (auto const& pos : x) {
        for (int dim = 0; dim < NDIM; dim++) {
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
    bhCreateTree(nodes, sinkBuckets, 0, x, sortOrder, box, 0, x.size(), BH_BUCKET_SIZE);
    if (gpu) {
        pot = bhEvaluatePotentialPointsGpu(
            nodes, x, y, 0.85, getOptions().gravitySupportRadius, getOptions().gm);
    } else {
        for (int i = 0; i < y.size(); i++) {
            bhTreeEvaluatePoint(nodes, y[i], pot[i], x, 0.85);
        }
    }
    return pot;
}
