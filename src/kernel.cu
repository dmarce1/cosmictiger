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

void kernel_set_type(int type) {
	kernel_type = type;
}

void kernel_output() {
	FILE* fp = fopen("kernel.txt", "wt");
	for (double q = 0.0; q <= 1.0; q += 0.01) {
		fprintf(fp, "%e %e %e %e %e\n", q, kernelW(q), dkernelW_dq(q), kernelFqinv(q), kernelPot(q));
	}
	fclose(fp);

}
