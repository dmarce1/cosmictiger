/* CosmicTiger - periodic global FMM policy; BSL-1.0. */
#pragma once

#include <cosmictiger/gravity/dehnen/periodic.hpp>

#ifdef COSMICTIGER_FMM_INSTANCE_ORDER
#include <cosmictiger/gravity/fmm_kernels/cartesian_ewald_operators.hpp>
#include <cosmictiger/gravity/gravity_cpu.hpp>
#include <cosmictiger/gravity/kick_walk.hpp>

template <int order>
void cpuGravityEwald(cosmictiger::gravity::cartesian::Expansion<float, order>& lValue,
    Vector<TreeId> const& list, TreeId self, bool doPhi, FmmGeometry const& geometry) {
    constexpr int kLocalCoefficients = FmmOrder<order>::kLocalCoefficients_;
    FlopCounter<FlopCount> flops = 0;
    if (list.size()) {
        Vector<TreeNode const*> treePtrs(list.size());
        TreeNode const* selfPtr = treeGetNode(self);
        int const numberOfSources = roundUp((int) list.size(), kSimdFloatSize) / kSimdFloatSize;
        for (int i = 0; i < list.size(); i++) {
            treePtrs[i] = treeGetNode(list[i]);
        }
        static thread_local detail::SimdAlignedVector<
            cosmictiger::gravity::cartesian::Multipole<SimdFloat, order>>
            mValue;
        static thread_local detail::SimdAlignedVector<Array<SimdPosition, kDimensions>> yValue;
        detail::gatherSourceMultipoles<order>(treePtrs, mValue, yValue);
        Array<SimdPosition, kDimensions> x{};
        for (int dim = 0; dim < kDimensions; dim++) {
            x[dim] = simdBroadcast<SimdPosition>(selfPtr->mpos->pos[dim].raw());
        }
        cosmictiger::gravity::cartesian::Expansion<SimdFloat, order> l0;
        l0 = simdBroadcast<SimdFloat>(0.0f);
        for (int j = 0; j < numberOfSources; j++) {
            Array<SimdFloat, kDimensions> dx{};
            for (int dim = 0; dim < kDimensions; dim++) {
                dx[dim] = fmmPositionDifference(x[dim], yValue[j][dim], dim, geometry);
            }
            Array<SimdFloat, kDimensions> unwrappedPosition{};
            for (int lane = 0; lane < kSimdFloatSize; lane++) {
                int const sourceIndex =
                    std::min(j * kSimdFloatSize + lane, int(treePtrs.size()) - 1);
                for (int dim = 0; dim < kDimensions; dim++) {
                    simdSetLane(unwrappedPosition[dim], lane,
                        selfPtr->mpos->pos[dim].toDouble() -
                            treePtrs[sourceIndex]->mpos->pos[dim].toDouble());
                }
            }
            cosmictiger::gravity::cartesian::Green<SimdFloat, order> d{};
            flops += FlopCount(kSimdFloatSize) *
                cosmictiger::gravity::cartesian::evaluateEwaldGreen(d, dx, unwrappedPosition);
            flops += FlopCount(kSimdFloatSize) *
                cosmictiger::gravity::cartesian::M2L(l0, mValue[j], d, doPhi);
        }
        // Fixed displacement and kernel work is batched outside the source loop.
        flops += FlopCount(numberOfSources) * kSimdFloatSize * (3 + 9);
        for (int i = 0; i < kLocalCoefficients; i++) {
            lValue[i] += simdSum(l0[i]);
        }
        flops += kSimdFloatSize * kLocalCoefficients;
    }
    addCpuFlops(flops);
}

template <int order>
FlopCount dehnenBackgroundImpl(KickParameters const& params, ExpansionBuffer& l,
    Array<Position, kDimensions>& pos, TreeId self, Vector<TreeId> const& echecklist) {
    using Expansion = cosmictiger::gravity::cartesian::Expansion<float, order>;
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
        Array<float, kDimensions> displacement;
        for (int dim = 0; dim < kDimensions; dim++) {
            displacement[dim] = unwrappedDoubleDistance(selfPtr->pos[dim], pos[dim]);
        }
        Expansion translated{};
        if (!params.doPhi) {
            translated[0] = expansion[0];
        }
        cosmictiger::gravity::cartesian::L2L(
            translated, expansion, fmmScaleDisplacement(displacement), params.doPhi);
        expansion = translated;
        pos = selfPtr->pos;
        double const secondMoment = params.doPhi ? detail::kickBackgroundSecondMoment(pos) : 0.0;
        Expansion background;
        background = 0.0f;
        initializeBackgroundExpansion<order>(
            background, fmmMultiPos<order>(selfPtr->mpos).multi, secondMoment);
        for (int i = 0; i < kLocalCoefficients; i++) {
            expansion[i] += background[i];
        }
        flops += cosmictiger::gravity::cartesian::detail::L2L<kLocalDegree, true>::kFlops_ + 26 +
            kLocalCoefficients;
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
    Array<TreeNode const*, kSimdFloatSize> otherPtrs;
    SimdFloat selfRadius = selfPtr->radius;
    Array<SimdPosition, kDimensions> selfPos{};
    Array<SimdPosition, kDimensions> otherPos{};
    Array<SimdFloat, kDimensions> dx{};
    SimdFloat otherRadius{};
    SimdFloat otherLeaf{};
    Range<SimdDouble8> otherBox{};
    for (int dim = 0; dim < kDimensions; ++dim) {
        selfPos[dim] = simdBroadcast<SimdPosition>(selfPtr->pos[dim].raw());
    }
    FlopCounter<FlopCount> flops = 0;
    {
        nextlist.resize(0);
        cclist.resize(0);
        leaflist.resize(0);
        auto& checklist = echecklist;
        do {
            for (int ci = 0; ci < checklist.size(); ci += kSimdFloatSize) {
                int const maxci = std::min((int) checklist.size(), ci + kSimdFloatSize);
                int const maxi = maxci - ci;
                for (int i = ci; i < maxci; i++) {
                    otherPtrs[i - ci] = treeGetNode(checklist[i]);
                }
                for (int i = 0; i < maxi; i++) {
                    for (int dim = 0; dim < kDimensions; dim++) {
                        simdSetLane(otherPos[dim], i, otherPtrs[i]->pos[dim].raw());
                    }
                    simdSetLane(otherRadius, i, otherPtrs[i]->radius);
                    simdSetLane(otherLeaf, i, otherPtrs[i]->leaf);
                }
                for (int i = maxi; i < kSimdFloatSize; i++) {
                    for (int dim = 0; dim < kDimensions; dim++) {
                        simdSetLane(otherPos[dim], i, 0.f);
                    }
                    simdSetLane(otherRadius, i, 0.f);
                    simdSetLane(otherLeaf, i, 0);
                    for (int dim = 0; dim < kDimensions; dim++) {
                        simdSetLane(otherBox.begin[dim], i, 0.0);
                        simdSetLane(otherBox.end[dim], i, 0.0);
                    }
                }
                for (int dim = 0; dim < kDimensions; dim++) {
                    dx[dim] =
                        simdCast<SimdFloat>(simdPositionDifference(selfPos[dim], otherPos[dim])) *
                        kPositionToFloat;
                }
                SimdFloat r2 = sqr(dx[kXDimension], dx[kYDimension], dx[kZDimension]);
                r2 = simdMax(r2,
                    simdMax(sqr(simdBroadcast<SimdFloat>(0.5) - (selfRadius + otherRadius)),
                        simdBroadcast<SimdFloat>(0)));
                SimdFloat const dcc = (selfRadius + otherRadius) * thetainv;
                auto const cc = r2 > sqr(dcc);
                flops += kSimdFloatSize * 14;
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
