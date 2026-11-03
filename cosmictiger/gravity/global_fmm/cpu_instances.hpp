/* CosmicTiger - periodic global FMM policy; BSL-1.0. */
#pragma once

#include <cosmictiger/gravity/global_fmm/ewald_tables.hpp>
#include <cosmictiger/gravity/global_fmm/periodic_cpu.hpp>
#include <cosmictiger/math/vector3.hpp>

template void cpuGravityEwald<COSMICTIGER_FMM_INSTANCE_ORDER>(
    cosmictiger::gravity::cartesian::tracefree::Expansion<float, COSMICTIGER_FMM_INSTANCE_ORDER>&,
    Vector<TreeId> const&, TreeId, bool, FmmGeometry const&);
template FlopCounts globalFmmBackgroundImpl<COSMICTIGER_FMM_INSTANCE_ORDER>(
    KickParameters const&, ExpansionBuffer&, Vector3<Position>&, TreeId, Vector<TreeId> const&);
template FlopCounts globalFmmPeriodicImpl<COSMICTIGER_FMM_INSTANCE_ORDER>(
    KickParameters const&, ExpansionBuffer&, TreeId, Vector<TreeId>&, KickInteractionCounts&);
template void ewaldFillTables<COSMICTIGER_FMM_INSTANCE_ORDER>(EwaldConstants&);
