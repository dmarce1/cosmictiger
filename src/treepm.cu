#include <cosmictiger/defs.hpp>
#ifdef TREEPM
#include <cosmictiger/treepm.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/cuda_reduce.hpp>

#define BLOCK_SIZE 32

__managed__ array<float*, NDIM + 1> fields;
__managed__ range<int> int_box;
__managed__ int Nres;
__managed__ float Ninv;

void treepm_allocate_fields(int Nres_) {
	Nres = Nres_;
	Ninv = 1.0f / Nres;
	auto ibox = treepm_get_fourier_box(Nres);
	for (int dim = 0; dim < NDIM; dim++) {
		int_box.begin[dim] = ibox.begin[dim] + CLOUD_MIN;
		int_box.end[dim] = ibox.end[dim] + CLOUD_MAX;
	}
	const auto vol = int_box.volume();
	for (int dim = 0; dim < NDIM + 1; dim++) {
		CUDA_CHECK(cudaMallocManaged(&fields[dim], sizeof(float) * vol));
	}
}

void treepm_set_field(int dim, const vector<float>& values) {
	ALWAYS_ASSERT(values.size() == int_box.volume());
	memcpy(fields[dim], values.data(), values.size() * sizeof(float));
}

CUDA_EXPORT float treepm_get_field(int dim, float x, float y, float z) {
	float xi = x * Nres;
	float yi = y * Nres;
	float zi = z * Nres;
	int i0 = xi;
	int j0 = yi;
	int k0 = zi;
	float res = 0.f;
	array<int, NDIM> I;
	for (int i = i0 + CLOUD_MIN; i <= i0 + CLOUD_MAX; i++) {
		I[XDIM] = i;
		for (int j = j0 + CLOUD_MIN; j <= j0 + CLOUD_MAX; j++) {
			I[YDIM] = j;
			for (int k = k0 + CLOUD_MIN; k <= k0 + CLOUD_MAX; k++) {
				I[ZDIM] = k;
				res += cloud_weight(xi - i) * cloud_weight(yi - j) * cloud_weight(zi - k) * fields[dim][int_box.index(I)];
			}
		}
	}
	return res;
}

void treepm_free_fields() {
	for (int dim = 0; dim < NDIM + 1; dim++) {
		CUDA_CHECK(cudaFree(fields[dim]));
	}
}

__global__ void treepm_compute_density_kernel(int Nres, const fixed32* X, const fixed32* Y, const fixed32* Z, const pair<part_int>* ranges,
		range<int64_t> int_box, range<int64_t> chain_box, range<int64_t> rho_box, float* rho) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	float myrho = 0.f;
	__syncthreads();
	array<int64_t, NDIM> span, I, J;
	for (int dim = 0; dim < NDIM; dim++) {
		span[dim] = rho_box.end[dim] - rho_box.begin[dim];
	}
	I[ZDIM] = bid % span[ZDIM] + rho_box.begin[ZDIM];
	I[YDIM] = (bid / span[ZDIM]) % span[YDIM] + rho_box.begin[YDIM];
	I[XDIM] = bid / (span[ZDIM] * span[YDIM]) + rho_box.begin[XDIM];
	for (J[XDIM] = CLOUD_MIN; J[XDIM] <= CLOUD_MAX; J[XDIM]++) {
		for (J[YDIM] = CLOUD_MIN; J[YDIM] <= CLOUD_MAX; J[YDIM]++) {
			for (J[ZDIM] = CLOUD_MIN; J[ZDIM] <= CLOUD_MAX; J[ZDIM]++) {
				const auto K = I - J;
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
	treepm_compute_density_kernel<<<rho_box.volume(), BLOCK_SIZE>>>( Nres, &particles_pos(XDIM,0), &particles_pos(YDIM,0), &particles_pos(ZDIM,0), chain_mesh.data(), int_box, chain_box, rho_box, rho.data());
	CUDA_CHECK(cudaDeviceSynchronize());
	double rhosum = 0.0;
	for( int i = 0; i < rho.size(); i++) {
		rhosum += rho[i];
	}
	PRINT( "rhosum = %e\n", rhosum);
	return rho;
}
#endif
