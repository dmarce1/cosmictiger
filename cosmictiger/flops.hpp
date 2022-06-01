/*
 * flops.hpp
 *
 *  Created on: Jun 1, 2022
 *      Author: dmarce1
 */

#ifndef FLOPS_HPP_
#define FLOPS_HPP_


void reset_flops();
void reset_gpu_flops();
double flops_per_second();
double get_gpu_flops();
void add_cpu_flops(int count);
#ifdef __CUDACC__
__device__
void add_gpu_flops(int count);
#endif

#endif /* FLOPS_HPP_ */
