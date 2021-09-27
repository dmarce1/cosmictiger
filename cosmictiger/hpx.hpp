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
#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/serialization/unordered_map.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/hpx_finalize.hpp>


#if (HPX_VERSION_FULL < ((1<<16) | (6<<8)))
#define HPX_EARLY
#include <hpx/runtime/threads/run_as_os_thread.hpp>
#else
#define HPX_LATE
#include <hpx/include/run_as.hpp>
#endif

#ifdef HPX_EARLY
#define PAR_EXECUTION_POLICY hpx::parallel::execution::par(hpx::parallel::execution::task)
#define hpx_copy hpx::parallel::copy
#else
#define PAR_EXECUTION_POLICY hpx::execution::par(hpx::execution::task)
#define hpx_copy hpx::copy
#endif

#define HPX_PRIORITY_HI hpx::launch::async(hpx::threads::thread_priority_critical)
#define HPX_PRIORITY_NORMAL hpx::launch::async(hpx::threads::thread_priority_normal)
#define HPX_PRIORITY_LO hpx::launch::async(hpx::threads::thread_priority_low)

const vector<hpx::id_type>& hpx_localities();
const vector<hpx::id_type>& hpx_children();
void hpx_init();

using mutex_type = hpx::lcos::local::mutex;
using spinlock_type = hpx::lcos::local::spinlock;
using shared_mutex_type = hpx::lcos::local::shared_mutex;

#endif


int hpx_rank();
int hpx_size();
void hpx_yield();
int hpx_hardware_concurrency();
