/*
 * defs.hpp
 *
 *  Created on: Jul 17, 2021
 *      Author: dmarce1
 */

#ifndef DEFS_HPP_
#define DEFS_HPP_


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

#define BH_BUCKET_SIZE 8
#define BH_CUDA_MIN 512
#define DOMAIN_REBOUND_ITERS 20

#ifdef USE_CUDA
#define SINK_BUCKET_SIZE 160
#define SOURCE_BUCKET_SIZE 160
#else
#define SINK_BUCKET_SIZE 90
#define SOURCE_BUCKET_SIZE 90
#endif

#define GROUP_WAVES 8
#define GROUP_BUCKET_SIZE 25

#define CUDA_MAX_MEM 0.4
#define GPU_MIN_LOAD (1.0/50.0)
#define CUDA_KICK_OVERSUBSCRIPTION 2
#define CUDA_KICK_PARTS_MAX (12*1024)
#define HEAP_SIZE 1
#define L2FETCH 64
#define STACK_SIZE (16*1024)
#define KICK_WORKSPACE_PART_SIZE 20 // In % of total mem
#define KICK_PP_MAX (32*12)
#define MIN_KICK_PC_WARP 8
#define MIN_KICK_WARP 16
#define UNORDERED_SET_SIZE 1024
#define UNORDERED_MAP_SIZE 1024
#define WARP_SIZE 32
#define CUDA_CHECKLIST_SIZE 2048
#define CUDA_STACK_SIZE 32767
#define SELF_PHI float(-35.0/16.0)
#define ANALYTIC_BLOCK_SIZE 128
#define EWALD_DIST float(0.25)
#define EWALD_REAL_CUTOFF2 (2.6*2.6)
#define KICK_OVERSUBSCRIPTION 8
#define MAX_DEPTH 64
#define MAX_PARTICLES_PER_PARCEL (16*1024*1024)
#define MAX_RUNG 64
#define MIN_CP_PARTS 25
#define MIN_KICK_THREAD_PARTS (1024)
#define MIN_PC_PARTS 47
#define MIN_SORT_THREAD_PARTS (64*1024)


#define CUDA_MAX_DEPTH 64
#define NEXTLIST_SIZE 2048
#define LEAFLIST_SIZE 16384
#define PARTLIST_SIZE 2048
#define MULTLIST_SIZE 16384
#define DCHECKS_SIZE 16384
#define ECHECKS_SIZE 8192



#define NTREES_MIN (2*1024*1024)
#define PART_CACHE_SIZE 1024
#define SINK_BIAS float(1.5)
#define SORT_OVERSUBSCRIPTION 8
#define TREE_CACHE_SIZE 1024
#define TREE_NODE_ALLOCATION_SIZE 4



#define LORDER ORDER
#define MORDER (LORDER-1)
#define EWALD_DIST2 float(EWALD_DIST*EWALD_DIST)



#endif /* DEFS_HPP_ */
