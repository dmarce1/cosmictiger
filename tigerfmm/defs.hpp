/*
 * defs.hpp
 *
 *  Created on: Jul 17, 2021
 *      Author: dmarce1
 */

#ifndef DEFS_HPP_
#define DEFS_HPP_


#define NDIM 3
#define XDIM 0
#define YDIM 1
#define ZDIM 2

#define MAX_PARTICLES_PER_PARCEL (16*1024*1024)
#define PAR_EXECUTION_POLICY hpx::parallel::execution::par(hpx::parallel::execution::task)

#define ORDER 6

#define LORDER ORDER
#define MORDER (LORDER-1)

#define EWALD_REAL_CUTOFF2 (2.6*2.6)

#endif /* DEFS_HPP_ */
