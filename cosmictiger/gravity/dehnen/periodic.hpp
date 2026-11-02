/* CosmicTiger - periodic global FMM policy; BSL-1.0. */
#pragma once

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/gravity/kick.hpp>

namespace detail {
double kickBackgroundSecondMoment(Array<Position, kDimensions> center);
}

template <int order>
void cpuGravityEwald(cosmictiger::gravity::cartesian::Expansion<float, order>&,
    Vector<TreeId> const&, TreeId, bool, FmmGeometry const& geometry = {});
template <int order>
FlopCounts dehnenBackgroundImpl(KickParameters const&, ExpansionBuffer&,
    Array<Position, kDimensions>&, TreeId, Vector<TreeId> const&);
template <int order>
FlopCounts dehnenPeriodicImpl(
    KickParameters const&, ExpansionBuffer&, TreeId, Vector<TreeId>&, KickInteractionCounts&);
FlopCounts dehnenBackground(int, KickParameters const&, ExpansionBuffer&,
    Array<Position, kDimensions>&, TreeId, Vector<TreeId> const&);
FlopCounts dehnenPeriodic(
    int, KickParameters const&, ExpansionBuffer&, TreeId, Vector<TreeId>&, KickInteractionCounts&);
