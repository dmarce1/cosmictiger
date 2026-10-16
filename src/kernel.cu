/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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
#include <functional>

#include <cosmictiger/kernel.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/safe_io.hpp>

void kernelSetType(int type) {}

void kernelOutput() {
    FILE* fp = fopen("kernel.txt", "wt");
    for (double q = 0.0; q <= 1.0; q += 0.0001) {
        fprintf(fp, "%e %e %e %e %e\n", q, kernelW(q), dkernelWDq(q), kernelFqinv(q), kernelPot(q));
    }
    fclose(fp);
}

double kernelStddev(std::function<double(double)> w) {
    double sum = 0.f;
    int n = 10000;
    double dq = 1.0 / n;
    for (int i = 0; i < n; i++) {
        double q = (i + 0.5) * dq;
        sum += sqr(sqr(q)) * w(q) * dq;
    }
    sum *= 4.0 * M_PI;
    return sqrt(sum);
}

void kernelAdjustOptions(Options& opts) {
    double h0 = 4.743416e-01;
    double h;
    h = kernelStddev(kernelW<double>);
    PRINT("kernel width = %e\n", h0 / h);
    //	opts.neighbor_number *= pow(h0 / h, 3);
    opts.sphBucketSize = 8.0 / M_PI * opts.sphNeighborNumber;
#ifdef KERNEL_CUBIC_SPLINE
    opts.cfl = 0.20;
    opts.eta = 0.17;
    opts.darkMatterNeighborNumber = 60;
    opts.sphNeighborNumber = 60;
#endif
#ifdef KERNEL_QUARTIC_SPLINE
    opts.cfl = 0.16;
    opts.eta = 0.15;
    opts.darkMatterNeighborNumber = 117;
    opts.sphNeighborNumber = 117;
#endif
#ifdef KERNEL_QUINTIC_SPLINE
    opts.cfl = 0.13;
    opts.eta = 0.14;
    opts.darkMatterNeighborNumber = 203;
    opts.sphNeighborNumber = 203;
#endif
    PRINT("Setting:\n");
    PRINT("SPH Bucket size       = %i\n", opts.sphBucketSize);
    PRINT("CFL = %f\n", opts.cfl);
    PRINT("eta = %f\n", opts.eta);
}
