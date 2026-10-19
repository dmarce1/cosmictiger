/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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

#pragma once

// Explicit instantiations of the order-templated CPU code for the order named
// by COSMICTIGER_FMM_INSTANCE_ORDER (one translation unit per order, generated
// by CMake from src/gravity/fmm_instance_cpu.cpp.in). The dispatchers in gravity.cpp,
// kick.cpp, tree.cpp and ewald_indices.cpp call these through fmmDispatch().

#ifndef COSMICTIGER_FMM_INSTANCE_ORDER
#error "COSMICTIGER_FMM_INSTANCE_ORDER must name the instantiated FMM order"
#endif

#include <cosmictiger/fmm/ewald_tables.hpp>
#include <cosmictiger/fmm/fmm_order.hpp>
#include <cosmictiger/gravity/gravity_cpu.hpp>
#include <cosmictiger/gravity/kick_walk.hpp>
#include <cosmictiger/gravity/tree_multipole.hpp>

static_assert(COSMICTIGER_FMM_INSTANCE_ORDER >= kFmmMinOrder &&
    COSMICTIGER_FMM_INSTANCE_ORDER <= kFmmMaxOrder);

// CPU gravity operators (gravity_cpu.hpp).
template void cpuGravityCc<COSMICTIGER_FMM_INSTANCE_ORDER>(GravityCcType,
    FmmExpansion<float, COSMICTIGER_FMM_INSTANCE_ORDER>&, Vector<TreeId> const&, TreeId, bool);
template void cpuGravityCp<COSMICTIGER_FMM_INSTANCE_ORDER>(
    FmmExpansion<float, COSMICTIGER_FMM_INSTANCE_ORDER>&, Vector<TreeId> const&, TreeId, bool);
template void cpuGravityPc<COSMICTIGER_FMM_INSTANCE_ORDER>(
    ForceVectors&, int, TreeId, Vector<TreeId> const&);

// CPU tree walk (kick_walk.hpp).
template KickReturn kickWalk<COSMICTIGER_FMM_INSTANCE_ORDER>(KickParameters, ExpansionBuffer,
    Array<Fixed32, NDIM>, TreeId, Vector<TreeId>, Vector<TreeId>, std::shared_ptr<KickWorkspace>);

// Tree multipole construction (tree_multipole.hpp).
template int treeComputeMultipole<COSMICTIGER_FMM_INSTANCE_ORDER>(MultipoleBuffer&,
    MultipoleBuffer const&, MultipoleBuffer const&, Array<double, NDIM> const&,
    Array<double, NDIM> const&, Array<double, NDIM> const&);
template int treeComputeMultipole<COSMICTIGER_FMM_INSTANCE_ORDER>(
    MultipoleBuffer&, Pair<ParticleInteger>, Array<double, NDIM> const&);

// Ewald tables (ewald_tables.hpp).
template void ewaldFillTables<COSMICTIGER_FMM_INSTANCE_ORDER>(EwaldConstants&);
