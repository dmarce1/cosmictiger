/* CosmicTiger - periodic global FMM policy; BSL-1.0. */
#pragma once

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/math/vector3.hpp>

namespace detail {
double kickBackgroundSecondMoment(Vector3<Position> center);
}

template <int order>
void cpuGravityEwald(cosmictiger::gravity::cartesian::tracefree::Expansion<float, order>&,
    Vector<TreeId> const&, TreeId, bool, FmmGeometry const& geometry = {});
template <int order>
FlopCounts globalFmmBackgroundImpl(
    KickParameters const&, ExpansionBuffer&, Vector3<Position>&, TreeId, Vector<TreeId> const&);
template <int order>
FlopCounts globalFmmPeriodicImpl(
    KickParameters const&, ExpansionBuffer&, TreeId, Vector<TreeId>&, KickInteractionCounts&);
FlopCounts globalFmmBackground(int, KickParameters const&, ExpansionBuffer&, Vector3<Position>&,
    TreeId, Vector<TreeId> const&);
FlopCounts globalFmmPeriodic(
    int, KickParameters const&, ExpansionBuffer&, TreeId, Vector<TreeId>&, KickInteractionCounts&);
