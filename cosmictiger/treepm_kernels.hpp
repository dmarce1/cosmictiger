/*
 * treepm_kernels.hpp
 *
 *  Created on: Sep 3, 2022
 *      Author: dmarce1
 */

#ifndef TREEPM_KERNELS_HPP_
#define TREEPM_KERNELS_HPP_

#pragma once

#include <cosmictiger/defs.hpp>

CUDA_EXPORT inline array<float, LORDER> green_kernel(float r, float rsinv, float rsinv2) {
	array<float, LORDER> d;
	const float nr = float(-0.5) * r * rsinv2;
	const float rinv = 1.f / r;
	float exp0 = expf(-float(0.25) * rsinv2 * sqr(r));
	float erfc0 = erfcf(float(0.5) * rsinv * r);
	const float expfactor = float(1.0 / 1.77245385e+00) * rsinv * exp0;
	float e0 = expfactor * rinv;
	d[0] = -erfc0 * rinv;
	for (int l = 1; l < LORDER; l++) {
		d[l] = fmaf(-float(2 * l - 1) * d[l - 1], rinv, e0);
		e0 *= nr;
	}
	return d;
	/*const float rinv1 = rinv;
	const float rinv2 = rinv1 * rinv1;
	const float d0 = erf0 * rinv;
	const float d1 = fmaf(float(-1) * d0, rinv, e0);
	e0 *= n8r;
	const float d2 = fmaf(float(-3) * d1, rinv, e0);
	e0 *= n8r;
	const float d3 = fmaf(float(-5) * d2, rinv, e0);
	e0 *= n8r;
	const float d4 = fmaf(T(-7) * d3, rinv, e0);*/

}

#endif /* TREEPM_KERNELS_HPP_ */
