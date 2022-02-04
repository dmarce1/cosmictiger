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

#define __KERNEL_CU__
#include <cosmictiger/options.hpp>
#include <cosmictiger/kernel.hpp>

#include <cosmictiger/safe_io.hpp>

void kernel_set_type(int type) {
	kernel_type = type;
}

void kernel_output() {
	FILE* fp = fopen("kernel.txt", "wt");
	for (double q = 0.0; q <= 1.0; q += 0.0001) {
		fprintf(fp, "%e %e %e %e %e\n", q, kernelW(q), dkernelW_dq(q), kernelFqinv(q), kernelPot(q));
	}
	fclose(fp);
}

double kernel_stddev(std::function<double(double)> W) {
	double sum = 0.f;
	int N = 10000;
	double dq = 1.0 / N;
	for (int i = 0; i < N; i++) {
		double q = (i + 0.5) * dq;
		sum += sqr(sqr(q)) * W(q) * dq;
	}
	sum *= 4.0 * M_PI;
	return sqrt(sum);
}

void kernel_adjust_options(options& opts) {
	double h0 = 4.743416e-01;
	double h;
	switch (kernel_type) {
	case KERNEL_CUBIC_SPLINE:
	case KERNEL_QUARTIC_SPLINE:
	case KERNEL_QUINTIC_SPLINE:
	case KERNEL_WENDLAND_C2:
	case KERNEL_WENDLAND_C4:
	case KERNEL_WENDLAND_C6:
		break;
	default:
		PRINT("Error ! Unknown kernel!\n");
	}
	h = kernel_stddev(kernelW<double>);
	PRINT( "kernel width = %e\n", h0/h);
	opts.neighbor_number *= pow(h0 / h, 3);
	opts.sph_bucket_size = opts.neighbor_number * 8.0 / M_PI;
	opts.cfl *= h / h0;
	opts.hsoft *= h0 / h;
	opts.eta *= sqrt(h / h0);
	PRINT("Setting:\n");
	PRINT("Neighbor number       = %e\n", opts.neighbor_number);
	PRINT("SPH Bucket size       = %i\n", opts.sph_bucket_size);
	PRINT("Dark matter softening = 1/%e of mean separation.\n", 1.0 / opts.parts_dim / opts.hsoft);
	PRINT("CFL = %f\n", opts.cfl);
	PRINT("eta = %f\n", opts.eta);
}
