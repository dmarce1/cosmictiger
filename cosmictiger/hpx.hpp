/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

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


#include <cosmictiger/containers.hpp>

#ifndef __CUDACC__

#include <hpx/hpx.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/algorithms/fill.hpp>
#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/serialization/unordered_map.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/hpx_finalize.hpp>
#include <hpx/include/run_as.hpp>
#include <hpx/modules/executors.hpp>

#define HPX_PRIORITY_HI hpx::execution::parallel_executor(hpx::threads::thread_priority::high)
#define PAR_EXECUTION_POLICY hpx::execution::par(hpx::execution::task)
#define hpx_copy hpx::copy
#define hpx_fill hpx::fill


const vector<hpx::id_type>& hpx_localities();
const vector<hpx::id_type>& hpx_children();
void hpx_init();

using mutex_type = hpx::mutex;
using spinlock_type = hpx::spinlock;
using shared_mutex_type = hpx::shared_mutex;

#endif


int hpx_rank();
int hpx_size();
void hpx_yield();
int hpx_hardware_concurrency();
