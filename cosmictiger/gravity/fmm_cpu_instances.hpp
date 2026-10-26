/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

// Explicit instantiations of the order-templated CPU code for the order named
// by COSMICTIGER_FMM_INSTANCE_ORDER (one translation unit per order, generated
// by CMake from src/gravity/fmm_instance_cpu.cpp.in). The dispatchers in gravity.cpp,
// kick.cpp, tree.cpp and ewald_indices.cpp call these through fmmDispatch().

#ifndef COSMICTIGER_FMM_INSTANCE_ORDER
#error "COSMICTIGER_FMM_INSTANCE_ORDER must name the instantiated FMM order"
#endif

#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/gravity_cpu.hpp>
#include <cosmictiger/gravity/kick_walk.hpp>
#include <cosmictiger/gravity/tree_multipole.hpp>

static_assert(COSMICTIGER_FMM_INSTANCE_ORDER >= kFmmMinOrder &&
    COSMICTIGER_FMM_INSTANCE_ORDER <= kFmmMaxOrder);

// CPU gravity operators (gravity_cpu.hpp).
template void cpuGravityCc<COSMICTIGER_FMM_INSTANCE_ORDER>(
    cosmictiger::gravity::cartesian::Expansion<float, COSMICTIGER_FMM_INSTANCE_ORDER>&,
    Vector<TreeId> const&, TreeId, bool, FmmGeometry const&);
template void cpuGravityCp<COSMICTIGER_FMM_INSTANCE_ORDER>(
    cosmictiger::gravity::cartesian::Expansion<float, COSMICTIGER_FMM_INSTANCE_ORDER>&,
    Vector<TreeId> const&, TreeId, bool, FmmGeometry const&);
template void cpuGravityPc<COSMICTIGER_FMM_INSTANCE_ORDER>(
    ForceVectors&, int, TreeId, Vector<TreeId> const&, FmmGeometry const&);

// CPU tree walk (kick_walk.hpp).
template KickReturn kickWalk<COSMICTIGER_FMM_INSTANCE_ORDER>(KickParameters, ExpansionBuffer,
    Array<Position, NDIM>, TreeId, Vector<TreeId>, Vector<TreeId>, std::shared_ptr<KickWorkspace>,
    KickContext const&);

// Tree multipole construction (tree_multipole.hpp).
template FlopCount treeComputeMultipole<COSMICTIGER_FMM_INSTANCE_ORDER>(MultipoleBuffer&,
    MultipoleBuffer const&, MultipoleBuffer const&, Array<double, NDIM> const&,
    Array<double, NDIM> const&, Array<double, NDIM> const&);
template FlopCount treeComputeMultipole<COSMICTIGER_FMM_INSTANCE_ORDER>(
    MultipoleBuffer&, Pair<ParticleInteger>, Array<double, NDIM> const&, FmmSourceView const&);
