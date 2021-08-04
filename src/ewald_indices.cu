#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/safe_io.hpp>

#include <algorithm>

#define NREAL 147
#define NFOUR 92

struct ewald_constants {
	array<array<float, NDIM>, NREAL> real_indices;
	array<array<float, NDIM>, NFOUR> four_indices;
	array<tensor_trless_sym<float, LORDER>, NFOUR> four_expanse;
};

ewald_constants ec;
__device__ ewald_constants* ec_dev;

__global__ void set_ewald_constants(ewald_constants* consts) {
	ec_dev = consts;
}

void ewald_const::init_gpu() {
	int n2max = 10;
	int nmax = std::sqrt(n2max) + 1;
	array<float, NDIM> this_h;
	int count = 0;
	for (int i = -nmax; i <= nmax; i++) {
		for (int j = -nmax; j <= nmax; j++) {
			for (int k = -nmax; k <= nmax; k++) {
				const int i2 = i * i + j * j + k * k;
				if (i2 <= n2max) {
					this_h[0] = i;
					this_h[1] = j;
					this_h[2] = k;
					ec.real_indices[count++] = this_h;
				}
			}
		}
	}
	const auto sort_func = [](const array<float,NDIM>& a, const array<float,NDIM>& b) {
		const auto a2 = sqr(a[0],a[1],a[2]);
		const auto b2 = sqr(b[0],b[1],b[2]);
		return a2 > b2;
	};
	std::sort(ec.real_indices.begin(), ec.real_indices.end(), sort_func);
//	PRINT("nreal = %i\n", count);
	n2max = 8;
	nmax = std::sqrt(n2max) + 1;
	count = 0;
	for (int i = -nmax; i <= nmax; i++) {
		for (int j = -nmax; j <= nmax; j++) {
			for (int k = -nmax; k <= nmax; k++) {
				if (i * i + j * j + k * k <= n2max) {
					this_h[0] = i;
					this_h[1] = j;
					this_h[2] = k;
					const auto hdot = sqr(this_h[0]) + sqr(this_h[1]) + sqr(this_h[2]);
					if (hdot > 0) {
						ec.four_indices[count++] = this_h;
					}
				}
			}
		}
	}
	std::sort(ec.four_indices.begin(), ec.four_indices.end(), sort_func);
//	PRINT("nfour = %i\n", count);
	count = 0;
	for (int i = 0; i < NFOUR; i++) {
		array<float, NDIM> h = ec.four_indices[i];
		auto D0 = vector_to_sym_tensor<float, LORDER>(h);
		const float h2 = sqr(h[0]) + sqr(h[1]) + sqr(h[2]);                     // 5 OP
		const float c0 = -1.0 / h2 * exp(-M_PI * M_PI * h2 / 4.0) / M_PI;
		array<int, NDIM> n;
		const int signs[4] = { 1, -1, -1, 1 };
		for (n[0] = 0; n[0] < LORDER; n[0]++) {
			for (n[1] = 0; n[1] < LORDER - n[0]; n[1]++) {
				for (n[2] = 0; n[2] < LORDER - n[0] - n[1]; n[2]++) {
					const int n0 = n[0] + n[1] + n[2];
					D0(n) *= signs[n0 % 4] * pow(2.0 * M_PI, n0) * c0;
				}
			}
		}
		ec.four_expanse[count++] = D0.detraceD();
	}
	cuda_set_device();
	ewald_constants* dev;
	CUDA_CHECK(cudaMalloc(&dev, sizeof(ewald_constants)));
	CUDA_CHECK(cudaMemcpy(dev, &ec, sizeof(ewald_constants), cudaMemcpyHostToDevice));
	set_ewald_constants<<<1,1>>>(dev);
	CUDA_CHECK(cudaDeviceSynchronize());
}

CUDA_EXPORT int ewald_const::nfour() {
	return NFOUR;
}

CUDA_EXPORT int ewald_const::nreal() {
	return NREAL;
}

CUDA_EXPORT const array<float, NDIM>& ewald_const::real_index(int i) {
#ifdef __CUDA_ARCH__
	return ec_dev->real_indices[i];
#else
	return ec.real_indices[i];
#endif
}

CUDA_EXPORT const array<float, NDIM>& ewald_const::four_index(int i) {
#ifdef __CUDA_ARCH__
	return ec_dev->four_indices[i];
#else
	return ec.four_indices[i];
#endif
}

CUDA_EXPORT const tensor_trless_sym<float, LORDER>& ewald_const::four_expansion(int i) {
#ifdef __CUDA_ARCH__
	return ec_dev->four_expanse[i];
#else
	return ec.four_expanse[i];
#endif
}
