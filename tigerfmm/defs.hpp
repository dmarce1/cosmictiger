/*
 * defs.hpp
 *
 *  Created on: Jul 17, 2021
 *      Author: dmarce1
 */

#ifndef DEFS_HPP_
#define DEFS_HPP_

//#define CHECK_BOUNDS

#define NDIM 3
#define XDIM 0
#define YDIM 1
#define ZDIM 2

#define NCHILD 2
#define LEFT 0
#define RIGHT 1

#define SINK_BUCKET_SIZE 100
#define SOURCE_BUCKET_SIZE 100
#define EWALD_DIST float(0.25)
#define EWALD_REAL_CUTOFF2 (2.6*2.6)
#define KICK_OVERSUBSCRIPTION 8
#define MAX_DEPTH 64
#define MAX_PARTICLES_PER_PARCEL (16*1024*1024)
#define MIN_KICK_THREAD_PARTS (1024)
#define MIN_SORT_THREAD_PARTS (1024)
#define NTREES_MIN (2*1024*1024)
#define ORDER 6
#define PAR_EXECUTION_POLICY hpx::parallel::execution::par(hpx::parallel::execution::task)
#define PART_CACHE_SIZE 1024
#define SINK_BIAS float(1.5)
#define SIMD_FLOAT_SIZE 8
#define SIMD_INT_SIZE 8
#define SORT_OVERSUBSCRIPTION 4
#define TREE_CACHE_SIZE 1024
#define TREE_NODE_ALLOCATION_SIZE 8


#define LORDER ORDER
#define MORDER (LORDER-1)
#define EWALD_DIST2 float(EWALD_DIST*EWALD_DIST)


#endif /* DEFS_HPP_ */
