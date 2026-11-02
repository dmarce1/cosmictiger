/* CosmicTiger - periodic global FMM policy; BSL-1.0. */
#pragma once

#include <cosmictiger/gravity/dehnen/ewald_tables.hpp>
#include <cosmictiger/gravity/dehnen/periodic_cpu.hpp>
#include <cosmictiger/math/vector3.hpp>

template void cpuGravityEwald<COSMICTIGER_FMM_INSTANCE_ORDER>(
    cosmictiger::gravity::cartesian::Expansion<float, COSMICTIGER_FMM_INSTANCE_ORDER>&,
    Vector<TreeId> const&, TreeId, bool, FmmGeometry const&);
template FlopCounts dehnenBackgroundImpl<COSMICTIGER_FMM_INSTANCE_ORDER>(
    KickParameters const&, ExpansionBuffer&, Vector3<Position>&, TreeId, Vector<TreeId> const&);
template FlopCounts dehnenPeriodicImpl<COSMICTIGER_FMM_INSTANCE_ORDER>(
    KickParameters const&, ExpansionBuffer&, TreeId, Vector<TreeId>&, KickInteractionCounts&);
template void ewaldFillTables<COSMICTIGER_FMM_INSTANCE_ORDER>(EwaldConstants&);
