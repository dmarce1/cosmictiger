/* CosmicTiger - periodic global FMM policy; BSL-1.0. */
#pragma once

#include <cosmictiger/gravity/dehnen/periodic.hpp>

#ifdef COSMICTIGER_FMM_INSTANCE_ORDER
#include <cosmictiger/gravity/dehnen/ewald_kernels.hpp>
#include <cosmictiger/gravity/gravity_cpu.hpp>
#include <cosmictiger/gravity/kick_walk.hpp>

template <int order>
void cpuGravityEwald(FmmExpansion<float, order>& lValue, Vector<TreeId> const& list, TreeId self,
    bool doPhi, FmmGeometry const& geometry) {
    constexpr int kLocalCoefficients = FmmOrder<order>::kLocalCoefficients_;
    FlopCounter<FlopCount> flops = 0;
    if (list.size()) {
        Vector<TreeNode const*> treePtrs(list.size());
        TreeNode const* selfPtr = treeGetNode(self);
        int const numberOfSources = roundUp((int) list.size(), SIMD_FLOAT_SIZE) / SIMD_FLOAT_SIZE;
        for (int i = 0; i < list.size(); i++) {
            treePtrs[i] = treeGetNode(list[i]);
        }
        static thread_local detail::SimdAlignedVector<FmmMultipole<SimdFloat, order>> mValue;
        static thread_local detail::SimdAlignedVector<Array<SimdPosition, NDIM>> yValue;
        detail::gatherSourceMultipoles<order>(treePtrs, mValue, yValue);
        Array<SimdPosition, NDIM> x{};
        for (int dim = 0; dim < NDIM; dim++) {
            x[dim] = simdBroadcast<SimdPosition>(selfPtr->mpos->pos[dim].raw());
        }
        FmmExpansion<SimdFloat, order> l0;
        l0 = simdBroadcast<SimdFloat>(0.0f);
        for (int j = 0; j < numberOfSources; j++) {
            Array<SimdFloat, NDIM> dx{};
            for (int dim = 0; dim < NDIM; dim++) {
                dx[dim] = fmmPositionDifference(x[dim], yValue[j][dim], dim, geometry);
            }
            Array<SimdFloat, NDIM> unwrappedPosition{};
            for (int lane = 0; lane < SIMD_FLOAT_SIZE; lane++) {
                int const sourceIndex =
                    std::min(j * SIMD_FLOAT_SIZE + lane, int(treePtrs.size()) - 1);
                for (int dim = 0; dim < NDIM; dim++) {
                    simdSetLane(unwrappedPosition[dim], lane,
                        selfPtr->mpos->pos[dim].toDouble() -
                            treePtrs[sourceIndex]->mpos->pos[dim].toDouble());
                }
            }
            FmmExpansion<SimdFloat, order> d;
            flops += FlopCount(SIMD_FLOAT_SIZE) * ewaldGreensFunction(d, dx, unwrappedPosition);
            flops += FlopCount(SIMD_FLOAT_SIZE) * m2L(l0, mValue[j], d, doPhi);
        }
        // Fixed displacement and kernel work is batched outside the source loop.
        flops += FlopCount(numberOfSources) * SIMD_FLOAT_SIZE * (3 + 9);
        for (int i = 0; i < kLocalCoefficients; i++) {
            lValue[i] += simdSum(l0[i]);
        }
        flops += SIMD_FLOAT_SIZE * kLocalCoefficients;
    }
    addCpuFlops(flops);
}

template <int order>
FlopCount dehnenBackgroundImpl(KickParameters const& params, ExpansionBuffer& l,
    Array<Position, NDIM>& pos, TreeId self, Vector<TreeId> const& echecklist) {
    using Expansion = FmmExpansion<float, order>;
    constexpr int kLocalCoefficients = FmmOrder<order>::kLocalCoefficients_;
    constexpr int kLocalDegree = FmmOrder<order>::kLocalDegree_;
    auto& expansion = l.template expansion<order>();
    auto const* selfPtr = treeGetNode(self);
    FlopCounter<FlopCount> flops = 0;
    if (!params.forceOnly && self.proc == 0 && self.index == 0 && echecklist.size() == 1 &&
        echecklist[0] == self) {
        // The Ewald operators contain only the harmonic part. Add the uniform
        // background once, using canonical coordinates rather than periodic
        // nearest images. Its source trace is needed only for the potential
        // gauge, so compute that scalar here instead of storing it in every
        // source multipole.
        Array<float, NDIM> displacement;
        for (int dim = 0; dim < NDIM; dim++) {
            displacement[dim] = unwrappedDoubleDistance(selfPtr->pos[dim], pos[dim]);
        }
        expansion = l2L(expansion, displacement, params.doPhi);
        pos = selfPtr->pos;
        double const secondMoment = params.doPhi ? detail::kickBackgroundSecondMoment(pos) : 0.0;
        Expansion background;
        background = 0.0f;
        initializeBackgroundExpansion<order>(
            background, fmmMultiPos<order>(selfPtr->mpos).multi, secondMoment);
        for (int i = 0; i < kLocalCoefficients; i++) {
            expansion[i] += background[i];
        }
        flops +=
            detail::cartesian::L2lKernel<kLocalDegree, true>::kFlops_ + 26 + kLocalCoefficients;
    }
    return flops;
}

template <int order>
FlopCount dehnenPeriodicImpl(KickParameters const& params, ExpansionBuffer& l, TreeId self,
    Vector<TreeId>& echecklist, KickInteractionCounts& counts) {
    auto& expansion = l.template expansion<order>();
    auto const* selfPtr = treeGetNode(self);
    auto workspace = detail::kickAcquireLists();
    auto& nextlist = workspace.nextlist;
    auto& cclist = workspace.cclist;
    auto& leaflist = workspace.leaflist;
    SimdFloat const thetainv = simdBroadcast<SimdFloat>(fmmInverseOpeningAngle(params.theta));
    bool const openLargerCell = params.openLargerCell;
    Array<TreeNode const*, SIMD_FLOAT_SIZE> otherPtrs;
    SimdFloat selfRadius = selfPtr->radius;
    Array<SimdPosition, NDIM> selfPos{};
    Array<SimdPosition, NDIM> otherPos{};
    Array<SimdFloat, NDIM> dx{};
    SimdFloat otherRadius{};
    SimdFloat otherLeaf{};
    Range<SimdDouble8> otherBox{};
    for (int dim = 0; dim < NDIM; ++dim) {
        selfPos[dim] = simdBroadcast<SimdPosition>(selfPtr->pos[dim].raw());
    }
    FlopCounter<FlopCount> flops = 0;
    {
        nextlist.resize(0);
        cclist.resize(0);
        leaflist.resize(0);
        auto& checklist = echecklist;
        do {
            for (int ci = 0; ci < checklist.size(); ci += SIMD_FLOAT_SIZE) {
                int const maxci = std::min((int) checklist.size(), ci + SIMD_FLOAT_SIZE);
                int const maxi = maxci - ci;
                for (int i = ci; i < maxci; i++) {
                    otherPtrs[i - ci] = treeGetNode(checklist[i]);
                }
                for (int i = 0; i < maxi; i++) {
                    for (int dim = 0; dim < NDIM; dim++) {
                        simdSetLane(otherPos[dim], i, otherPtrs[i]->pos[dim].raw());
                    }
                    simdSetLane(otherRadius, i, otherPtrs[i]->radius);
                    simdSetLane(otherLeaf, i, otherPtrs[i]->leaf);
                }
                for (int i = maxi; i < SIMD_FLOAT_SIZE; i++) {
                    for (int dim = 0; dim < NDIM; dim++) {
                        simdSetLane(otherPos[dim], i, 0.f);
                    }
                    simdSetLane(otherRadius, i, 0.f);
                    simdSetLane(otherLeaf, i, 0);
                    for (int dim = 0; dim < NDIM; dim++) {
                        simdSetLane(otherBox.begin[dim], i, 0.0);
                        simdSetLane(otherBox.end[dim], i, 0.0);
                    }
                }
                for (int dim = 0; dim < NDIM; dim++) {
                    dx[dim] =
                        simdCast<SimdFloat>(simdPositionDifference(selfPos[dim], otherPos[dim])) *
                        kPositionToFloat;
                }
                SimdFloat r2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
                r2 = simdMax(r2,
                    simdMax(sqr(simdBroadcast<SimdFloat>(0.5) - (selfRadius + otherRadius)),
                        simdBroadcast<SimdFloat>(0)));
                SimdFloat const dcc = (selfRadius + otherRadius) * thetainv;
                auto const cc = r2 > sqr(dcc);
                flops += SIMD_FLOAT_SIZE * 14;
                for (int i = 0; i < maxi; i++) {
                    if (cc[i]) {
                        cclist.push_back(checklist[ci + i]);
                    } else if (otherLeaf[i]) {
                        leaflist.push_back(checklist[ci + i]);
                    } else {
                        detail::kickOpenOrDefer(openLargerCell, selfPtr, otherPtrs[i],
                            checklist[ci + i], nextlist, leaflist, counts);
                    }
                    if (!cc[i]) {
                        ALWAYS_ASSERT(!otherLeaf[i] || !selfPtr->leaf);
                    }
                }
                counts.acceptanceTests += maxi;
            }
            std::swap(checklist, nextlist);
            nextlist.resize(0);
        } while (checklist.size() && (selfPtr->leaf || openLargerCell));
        counts.cellCellEwald += cclist.size();
        cpuGravityEwald<order>(expansion, cclist, self, params.doPhi);
        if (!selfPtr->leaf) {
            checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
        }
    }
    detail::kickReleaseLists(std::move(workspace));
    return flops;
}
#endif
