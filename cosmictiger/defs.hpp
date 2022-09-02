
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


#ifndef DEFS_HPP_
#define DEFS_HPP_

#include <cosmictiger/profiler.hpp>



//#define LONG_LONG_PART_INT

#define TREEPM


#define COUNT_FLOPS

#ifndef NDEBUG

#define CHECK_BOUNDS
#define DOMAINS_CHECK
#endif





#define NDIM 3
#define XDIM 0
#define YDIM 1
#define ZDIM 2


#define NCHILD 2
#define LEFT 0
#define RIGHT 1



#define BH_BUCKET_SIZE 64
#define DOMAIN_REBOUND_ITERS 20



#define USE_CONFORMAL_TIME


#define GROUP_WAVES 8
#define GROUP_BUCKET_SIZE 256

#define MAX_LOAD_IMBALANCE 0.005
#define CUDA_MAX_MEM 0.3
#define GPU_MIN_LOAD (1.0/32.0)
#define CUDA_KICK_OVERSUBSCRIPTION 2
#define CUDA_KICK_PARTS_MAX (16*1024)
#define HEAP_SIZE (8ULL*1024ULL*1024ULL*1024ULL)
#define L2FETCH 64
#define STACK_SIZE (32*1024)
#define KICK_WORKSPACE_PART_SIZE 20 // In % of total mem
#define MIN_KICK_PC_WARP 8
#define MIN_KICK_WARP 16
#define UNORDERED_SET_SIZE 1024
#define UNORDERED_MAP_SIZE 1024
#define WARP_SIZE 32
#define CUDA_CHECKLIST_SIZE 2048
#define CUDA_STACK_SIZE 32767
#define EWALD_REAL_CUTOFF (2.6)
#define EWALD_REAL_CUTOFF2 (EWALD_REAL_CUTOFF*EWALD_REAL_CUTOFF)
#define KICK_OVERSUBSCRIPTION 1
#define MAX_DEPTH 64
#define MAX_PARTICLES_PER_PARCEL (8*1024*1024)
#define MAX_RUNG 16
#define MIN_CP_PARTS 25
#define MIN_KICK_THREAD_PARTS (1024)
#define MIN_PC_PARTS 47
#define MIN_SORT_THREAD_PARTS (64*1024)


#define CUDA_MAX_DEPTH 64
#define NEXTLIST_SIZE 4196
#define LEAFLIST_SIZE 16384
#define PARTLIST_SIZE 4196
#define MULTLIST_SIZE 16384
#define DCHECKS_SIZE (2*16384)
#define ECHECKS_SIZE 16384



#define NTREES_MIN (2*1024*1024)
#define NSPH_TREES_MIN (2*1024*1024)
#define PART_CACHE_SIZE 1024
#define SINK_BIAS float(1.5)
#define SORT_OVERSUBSCRIPTION 2
#define TREE_CACHE_SIZE 1024
#define SPH_TREE_CACHE_SIZE 1024
#define TREE_NODE_ALLOCATION_SIZE 8



#define LORDER ORDER
#define MORDER (LORDER-1)



#endif /* DEFS_HPP_ */
