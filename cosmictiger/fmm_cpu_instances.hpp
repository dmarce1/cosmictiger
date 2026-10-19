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
// by CMake from src/fmm_instance_cpu.cpp.in). The dispatchers in gravity.cpp,
// kick.cpp, tree.cpp and ewald_indices.cpp call these through fmmDispatch().

#ifndef COSMICTIGER_FMM_INSTANCE_ORDER
#error "COSMICTIGER_FMM_INSTANCE_ORDER must name the instantiated FMM order"
#endif

#include <cosmictiger/fmm_order.hpp>

static_assert(COSMICTIGER_FMM_INSTANCE_ORDER >= kFmmMinOrder &&
    COSMICTIGER_FMM_INSTANCE_ORDER <= kFmmMaxOrder);

// Order-templated definitions and their explicit instantiations are added here
// by the CPU conversion (gravity_cpu.hpp, kick_walk.hpp, tree_multipole.hpp,
// ewald_tables.hpp).
