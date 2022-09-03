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
}

CUDA_EXPORT inline void green_direct(float& phi, float& f, float r, float r2, float rinv, float rsinv, float rsinv2) {
	const float erfc0 = erfcf(0.5f * r * rsinv);
	const float exp0 = expf(-0.25f * r2 * rsinv2);
	const float cons = .5641895835f;
	phi = erfc0 * rinv;
	f = (erfc0 + cons * r * rsinv * exp0) * sqr(rinv) * rinv;
}

#endif /* TREEPM_KERNELS_HPP_ */
