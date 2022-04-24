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


void kernel_set_type(double n) {
}

void kernel_output() {
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

	opts.sph_bucket_size = 8.0 / M_PI * opts.neighbor_number;

}
