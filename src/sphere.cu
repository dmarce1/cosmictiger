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

#include <cosmictiger/sphere.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/math.hpp>

#define BLOCK_SIZE 32

struct sphere_t {
	array<fixed32, NDIM> x;
	float r;
};

__global__ void cuda_spheres_find(fixed32* X, fixed32* Y, fixed32* Z, pair<part_int>* ranges, sphere_t* spheres, int* index, int N) {
	int i = atomicAdd(index, 1);
	while (i < N) {
		const auto& range = ranges[i];
		auto& sphere = spheres[i];

		double xmax = 0.f;
		double ymax = 0.f;
		double zmax = 0.f;
		double xmin = 1.f;
		double ymin = 1.f;
		double zmin = 1.f;
		for (part_int j = range.first; j < range.second; j++) {
			const auto x = X[j].to_double();
			const auto y = Y[j].to_double();
			const auto z = Z[j].to_double();
			xmax = max(xmax, x);
			ymax = max(ymax, y);
			zmax = max(zmax, z);
			xmin = min(xmin, x);
			ymin = min(ymin, y);
			zmin = min(zmin, z);
		}
		fixed32 x0 = (xmax + xmin)*0.5;
		fixed32 y0 = (ymax + ymin)*0.5;
		fixed32 z0 = (zmax + zmin)*0.5;
		float rmax = 0.f;
		for (part_int j = range.first; j < range.second; j++) {
			const auto& x = X[j];
			const auto& y = Y[j];
			const auto& z = Z[j];
			const auto dx = distance(x, x0);
			const auto dy = distance(y, y0);
			const auto dz = distance(z, z0);
			rmax = fmaxf(rmax, sqrtf(sqr(dx, dy, dz)));
		}
		__syncthreads();
		sphere.r = rmax;
		sphere.x[XDIM] = x0;
		sphere.x[YDIM] = y0;
		sphere.x[ZDIM] = z0;
		i = atomicAdd(index, 1);
	}

}

void sphere_to_gpu(const vector<tree_node*>& tree_nodes) {
	PRINT( "%i spheres\n", tree_nodes.size());
	pair<part_int>* ranges;
	sphere_t* spheres;
	int* index;
	CUDA_CHECK(cudaMallocManaged(&ranges, sizeof(pair<part_int> ) * tree_nodes.size()));
	CUDA_CHECK(cudaMallocManaged(&spheres, sizeof(sphere_t) * tree_nodes.size()));
	CUDA_CHECK(cudaMallocManaged(&index, sizeof(int)));
	ALWAYS_ASSERT(ranges);
	for (int i = 0; i < tree_nodes.size(); i++) {
		ranges[i] = tree_nodes[i]->part_range;
	}
	*index = 0;
	int nblocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) cuda_spheres_find, BLOCK_SIZE, 0));
	nblocks *= cuda_smp_count();
	nblocks = std::max(std::min((int) nblocks, (int) (tree_nodes.size() / BLOCK_SIZE)),1);
	auto stream = cuda_get_stream();
	cuda_spheres_find<<<nblocks,BLOCK_SIZE,0,stream>>>(&particles_pos(XDIM,0),&particles_pos(YDIM,0),&particles_pos(ZDIM,0), ranges, spheres, index, tree_nodes.size());
	while (cudaStreamQuery(stream) != cudaSuccess) {
		hpx_yield();
	}
	cuda_end_stream(stream);
	for (int i = 0; i < tree_nodes.size(); i++) {
		tree_nodes[i]->pos = spheres[i].x;
		tree_nodes[i]->radius = spheres[i].r;
	}
	CUDA_CHECK(cudaFree(index));
	CUDA_CHECK(cudaFree(ranges));
	CUDA_CHECK(cudaFree(spheres));

}
