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
// kick.cpp and tree.cpp call these through fmmDispatch().

#ifndef COSMICTIGER_FMM_INSTANCE_ORDER
#error "COSMICTIGER_FMM_INSTANCE_ORDER must name the instantiated FMM order"
#endif

#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/gravity_cpu.hpp>
#include <cosmictiger/gravity/kick_walk.hpp>
#include <cosmictiger/gravity/tree_multipole.hpp>
#include <cosmictiger/math/vector3.hpp>

static_assert(COSMICTIGER_FMM_INSTANCE_ORDER >= kFmmMinOrder &&
    COSMICTIGER_FMM_INSTANCE_ORDER <= kFmmMaxOrder);

// CPU gravity operators (gravity_cpu.hpp).
template void cpuGravityCc<COSMICTIGER_FMM_INSTANCE_ORDER>(
    cosmictiger::gravity::cartesian::tracefree::Expansion<float, COSMICTIGER_FMM_INSTANCE_ORDER>&,
    Vector<TreeId> const&, TreeId, bool, FmmGeometry const&);
template void cpuGravityCp<COSMICTIGER_FMM_INSTANCE_ORDER>(
    cosmictiger::gravity::cartesian::tracefree::Expansion<float, COSMICTIGER_FMM_INSTANCE_ORDER>&,
    Vector<TreeId> const&, TreeId, bool, FmmGeometry const&);
template void cpuGravityPc<COSMICTIGER_FMM_INSTANCE_ORDER>(
    ForceVectors&, int, TreeId, Vector<TreeId> const&, FmmGeometry const&);

// CPU tree walk (kick_walk.hpp).
template KickReturn kickWalk<COSMICTIGER_FMM_INSTANCE_ORDER>(KickParameters, ExpansionBuffer,
    Vector3<Position>, TreeId, Vector<TreeId>, std::shared_ptr<KickWorkspace>, KickContext const&);

// Tree multipole construction (tree_multipole.hpp).
template FlopCounts treeComputeMultipole<COSMICTIGER_FMM_INSTANCE_ORDER>(MultipoleBuffer&,
    MultipoleBuffer const&, MultipoleBuffer const&, Vector3<double> const&, Vector3<double> const&,
    Vector3<double> const&);
template FlopCounts treeComputeMultipole<COSMICTIGER_FMM_INSTANCE_ORDER>(
    MultipoleBuffer&, Pair<ParticleInteger>, Vector3<double> const&, FmmSourceView const&);
