#include <cosmictiger/defs.hpp>
#ifdef TREEPM
#include <cosmictiger/treepm.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/cuda_reduce.hpp>

#define BLOCK_SIZE 32

__global__ void treepm_compute_density_kernel(int Nres, const fixed32* X, const fixed32* Y, const fixed32* Z, const pair<part_int>* ranges,
		range<int64_t> chain_box, range<int64_t> int_box, range<int64_t> rho_box, float* rho) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	float myrho = 0.f;
	__syncthreads();
	array<int64_t, NDIM> span, I, J;
	for (int dim = 0; dim < NDIM; dim++) {
		span[dim] = rho_box.end[dim] - rho_box.begin[dim];
	}
	I[ZDIM] = bid % (span[ZDIM] * span[YDIM]) + rho_box.begin[ZDIM];
	I[YDIM] = (bid / span[ZDIM]) % span[YDIM] + rho_box.begin[YDIM];
	I[XDIM] = bid / (span[ZDIM] * span[YDIM]) + rho_box.begin[XDIM];
	for (J[XDIM] = CLOUD_MIN; J[XDIM] < CLOUD_MAX; J[XDIM]++) {
		for (J[YDIM] = CLOUD_MIN; J[YDIM] < CLOUD_MAX; J[YDIM]++) {
			for (J[ZDIM] = CLOUD_MIN; J[ZDIM] < CLOUD_MAX; J[ZDIM]++) {
				const auto K = J + I;
				if (int_box.contains(K)) {
					const auto& rng = ranges[chain_box.index(K)];
					for (part_int i = rng.first + tid; i < rng.second; i += BLOCK_SIZE) {
						const float x = X[i].to_float() * Nres - I[XDIM];
						const float y = Y[i].to_float() * Nres - I[YDIM];
						const float z = Z[i].to_float() * Nres - I[ZDIM];
						myrho += cloud_weight(x) * cloud_weight(y) * cloud_weight(z);
					}
				}
			}
		}
	}
	shared_reduce_add<float, BLOCK_SIZE>(myrho);
	if (tid == 0) {
		rho[bid] = myrho;
	}
}

device_vector<float> treepm_compute_density_local(int Nres, const device_vector<pair<part_int>>& chain_mesh, range<int64_t> int_box, range<int64_t> chain_box,
		range<int64_t> rho_box) {
	cuda_set_device();
	device_vector<float> rho;
	rho.resize(rho_box.volume());
	treepm_compute_density_kernel<<<chain_box.volume(), BLOCK_SIZE>>>( Nres, &particles_pos(XDIM,0), &particles_pos(YDIM,0), &particles_pos(ZDIM,0), chain_mesh.data(), int_box, chain_box, rho_box, rho.data());
	CUDA_CHECK(cudaDeviceSynchronize());
	return rho;
}
#endif
