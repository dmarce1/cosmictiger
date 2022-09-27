#include <stdio.h>
#include <utility>
#include <cmath>
#define ORDER 8
#define USE_CUDA
#include <cosmictiger/complex.hpp>
#include <cosmictiger/spherical_fmm.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/simd.hpp>
#include <cuda.h>
#include <cosmictiger/fmm_kernels.hpp>

using real = float;

double Pm(int l, int m, double x) {
	if (m > l) {
		return 0.0;
	} else if (l == 0) {
		return 1.0;
	} else if (l == m) {
		return -(2 * l - 1) * Pm(l - 1, l - 1, x) * sqrt(1.0 - x * x);
	} else {
		return ((2 * l - 1) * x * Pm(l - 1, m, x) - (l - 1 + m) * Pm(l - 2, m, x)) / (l - m);
	}
}

CUDA_EXPORT constexpr int index(int l, int m) {
	return l * (l + 1) / 2 + m;
}

template<class T>
CUDA_EXPORT constexpr T nonepow(int m) {
	return m % 2 == 0 ? T(1) : T(-1);
}

template<class T, int P>
struct spherical_expansion: public array<complex<T>, (P + 1) * (P + 2) / 2> {
	CUDA_EXPORT
	inline complex<T> operator()(int n, int m) const {
		if (m >= 0) {
			return (*this)[index(n, m)];
		} else {
			return (*this)[index(n, -m)].conj() * nonepow<T>(m);
		}
	}
	void print() const {
		for (int l = 0; l <= P; l++) {
			for (int m = 0; m <= l; m++) {
				printf("%e + i %e  ", (*this)(l, m).real(), (*this)(l, m).imag());
			}
			printf("\n");
		}
	}
};

#define MBITS 23

template<int P>
constexpr int multi_bits(int n = 1) {
	if (n == P + 1) {
		return 0;
	} else {
		return (2 * n + 1) * (MBITS - n + 1) + multi_bits<P>(n + 1);
	}
}

template<int BITS>
class bitstream {
	static constexpr int N = ((BITS - 1) / CHAR_BIT) + 1;
	array<unsigned char, N> bits;
	mutable int nextbit;
	mutable int byte;CUDA_EXPORT
	void next() {
		if (nextbit == CHAR_BIT) {
			byte++;
			nextbit = 0;
		} else {
			nextbit++;
			if (nextbit == CHAR_BIT) {
				byte++;
				nextbit = 0;
			}
		}
//		ALWAYS_ASSERT(byte < N);
	}
public:
	CUDA_EXPORT
	int read_bits(int count) const {
		int res = 0;
		if (nextbit == CHAR_BIT) {
			nextbit = 0;
			byte++;
		}
		if (count <= CHAR_BIT - nextbit) {
			res = bits[byte] >> nextbit;
			res &= (1 << count) - 1;
			nextbit += count;
			return res;
		} else {
			int n = 0;
			res = bits[byte] >> nextbit;
			n += CHAR_BIT - nextbit;
			nextbit = 0;
			byte++;
			while (count >= CHAR_BIT + n) {
				res |= ((int) bits[byte]) << n;
				byte++;
				n += CHAR_BIT;
			}
			if (count - n > 0) {
				res |= (((int) bits[byte]) & ((1 << (count - n)) - 1)) << (n);
				nextbit = count - n;
			}
			return res;
		}
	}
	void write_bits(int i, int count) {
		if (nextbit == CHAR_BIT) {
			nextbit = 0;
			byte++;
		}
		bits[byte] = ((i << nextbit) & 0xFF) | (bits[byte] & ((1 << nextbit) - 1));
		const int m = CHAR_BIT - nextbit;
		nextbit = count < m ? nextbit + count : 0;
		byte += count < m ? 0 : 1;
		count -= m;
		if (count > 0) {
			i >>= m;
			while (count >= CHAR_BIT) {
				bits[byte] = i & 0xFF;
				count -= CHAR_BIT;
				i >>= CHAR_BIT;
				byte++;
			}
		}
		if (count > 0) {
			bits[byte] = i;
			nextbit += count;
		}
	}
	bitstream() {
		reset();
	}
	CUDA_EXPORT
	void reset() {
		nextbit = CHAR_BIT;
		byte = -1;
	}
};

/*template<class T, int P>
 class compressed_multipole {
 static constexpr int N = multi_bits<P>();
 float r;
 float mass;
 mutable bitstream<N> bits;
 public:
 void compress(const spherical_expansion<T, P>& O, T scale) {
 constexpr Ylm_max_array<P> norms;
 bits.reset();
 float rpow = 1.0;
 float m = O[index(0, 0)].real();
 float minv = 1.0f / m;
 for (int n = 1; n <= P; n++) {
 for (int m = -n; m <= n; m++) {
 float value = m >= 0 ? O[index(n, m)].real() : O[index(n, -m)].imag();
 value *= rpow;
 value *= minv;
 int sgn = value > 0.0 ? 0 : 1;
 value = fabs(value) / norms(n, abs(m));
 //				if (value >= 1.0) {
 //					printf("%e %e\n", value, norms(n, abs(m)));
 //				}
 int i = 0;
 for (int j = 0; j < MBITS - n; j++) {
 i <<= 1;
 if (value >= 0.5) {
 i |= 1;
 }
 value = fmod(2.0 * value, 1.0);
 }
 i <<= 1;
 i |= sgn;
 bits.write_bits(i, MBITS - n + 1);
 }
 rpow /= scale;
 }
 mass = m;
 r = scale;
 }
 CUDA_EXPORT
 spherical_expansion<T, P> decompress() const {
 constexpr Ylm_max_array<P> norms;
 bits.reset();
 float rpow = 1.0;
 spherical_expansion<T, P> O;
 O[0].real() = mass;
 O[0].imag() = 0.f;
 for (int n = 1; n <= P; n++) {
 for (int m = -n; m <= n; m++) {
 int i = bits.read_bits(MBITS - n + 1);
 int sgn = i & 1;
 i >>= 1;
 float value = (sgn ? -1.0f : 1.0f) * (float) i / (float) (1 << (MBITS - n));
 value *= rpow * mass * norms(n, abs(m));
 if (m >= 0) {
 O[index(n, m)].real() = value;
 } else {
 O[index(n, -m)].imag() = value;
 }
 }
 rpow *= r;
 }
 return O;
 }
 };
 */

double factorial(int n) {
	return n == 0 ? 1.0 : n * factorial(n - 1);
}

CUDA_EXPORT constexpr double const_sqrt_helper(double a, double xn, int iter) {
	if (iter == 5) {
		return xn;
	} else {
		return const_sqrt_helper(a, 0.5 * (xn + a / xn), iter + 1);
	}
}

CUDA_EXPORT constexpr double const_sqrt(double a) {
	return const_sqrt_helper(a, 0.5, 0);

}

CUDA_EXPORT constexpr double Ylm_xy0(int l, int m, double z) {
	if (m > l) {
		return 0.0;
	} else if (l == 0) {
		return 1.0;
	} else if (l == m) {
		return const_sqrt(1.0 - z * z) * Ylm_xy0(l - 1, l - 1, z) / (2 * l);
	} else {
		return ((2 * l - 1) * z * Ylm_xy0(l - 1, m, z) - Ylm_xy0(l - 2, m, z)) / (l * l - m * m);
	}
}

CUDA_EXPORT constexpr double const_abs(double a) {
	return a > 0.0 ? a : -a;
}

CUDA_EXPORT constexpr double const_max(double a, double b) {
	return a > b ? a : b;
}

struct Ylm_max {
	CUDA_EXPORT
	constexpr double operator()(int l, int m) {
		constexpr int N = 64;
		double next = 0.0;
		constexpr int ix = 0;
		constexpr int iy = N;
		for (int iz = 0; iz <= N; iz++) {
			next = const_max(next, Ylm_xy0(l, m, (double) iz / N));
		}
		return next;
	}
};

template<int P>
struct Ylm_max_array {
	double a[P + 1][P + 1];CUDA_EXPORT
	constexpr Ylm_max_array() :
			a() {
		for (int i = 0; i <= P; i++)
			for (int j = 0; j <= i; j++) {
				a[i][j] = 1.001 * Ylm_max()(i, j);
			}
	}
	CUDA_EXPORT
	constexpr double operator()(int l, int m) const {
		return a[l][m];
	}

};

template<class T, int P>
spherical_expansion<T, P> spherical_regular_harmonic(T x, T y, T z) {
	const T r2 = x * x + y * y + z * z;
	const complex<T> R = complex<T>(x, y);
	spherical_expansion<T, P> Y;
	Y[index(0, 0)] = complex<T>(T(1), T(0));
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
			Y[index(m, m)] = Y[index(m - 1, m - 1)] * R / T(2 * m);
		}
		if (m + 1 <= P) {
			Y[index(m + 1, m)] = z * Y[index(m, m)];
		}
		for (int n = m + 2; n <= P; n++) {
			const T inv = T(1) / (T(n * n) - T(m * m));
			Y[index(n, m)] = inv * (T(2 * n - 1) * z * Y[index(n - 1, m)] - r2 * Y[index(n - 2, m)]);
		}
	}
	return Y;
}

template<class T, int P>
spherical_expansion<T, P> spherical_singular_harmonic(T x, T y, T z) {
	const T r2 = x * x + y * y + z * z;
	const T r2inv = T(1) / r2;
	complex<T> R = complex<T>(x, y);
	spherical_expansion<T, P> O;
	O[index(0, 0)] = complex<T>(sqrt(r2inv), T(0));
	R *= r2inv;
	z *= r2inv;
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
			O[index(m, m)] = O[index(m - 1, m - 1)] * R * T(2 * m - 1);
		}
		if (m + 1 <= P) {
			O[index(m + 1, m)] = T(2 * m + 1) * z * O[index(m, m)];
		}
		for (int n = m + 2; n <= P; n++) {
			O[index(n, m)] = (T(2 * n - 1) * z * O[index(n - 1, m)] - T((n - 1) * (n - 1) - m * m) * r2inv * O[index(n - 2, m)]);
		}
	}
	return O;
}

template<class T, int P>
void spherical_expansion_M2M(spherical_expansion<T, P>& M, T x, T y, T z) {
	const auto Y = spherical_regular_harmonic<T, P>(-x, -y, -z);
	const auto M0 = M;
	for (int n = P; n >= 0; n--) {
		for (int m = 0; m <= n; m++) {
			M[index(n, m)] = complex<T>(T(0), T(0));
			for (int k = 0; k <= n; k++) {
				for (int l = -k; l <= k; l++) {
					if (abs(m - l) > n - k) {
						continue;
					}
					if (-abs(m - l) < k - n) {
						continue;
					}
					M[index(n, m)] += Y(k, l) * M0(n - k, m - l);
				}
			}
		}
	}
}

template<class T, int P>
void spherical_expansion_L2L(spherical_expansion<T, P>& L, T x, T y, T z) {
	const auto Y = spherical_regular_harmonic<T, P>(-x, -y, -z);
	const auto L0 = L;
	for (int n = 0; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			L[index(n, m)] = complex<T>(T(0), T(0));
			for (int k = 0; k <= P - n; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					L[index(n, m)] += Y(k, l).conj() * L0(n + k, m + l);
				}
			}
		}
	}
}

void random_unit(real& x, real& y, real& z) {
	const real theta = acos(2 * rand1() - 1.0);
	const real phi = rand1() * 2.0 * M_PI;
	x = cos(phi) * sin(theta);
	y = sin(phi) * sin(theta);
	z = cos(theta);
}

void random_vector(real& x, real& y, real& z) {
	do {
		x = 2 * rand1() - 1;
		y = 2 * rand1() - 1;
		z = 2 * rand1() - 1;
	} while (sqr(x, y, z) > 1);
}

template<class T, int P>
spherical_expansion<T, P> spherical_expansion_ref_M2L(spherical_expansion<T, P - 1> M, T x, T y, T z) {
	const auto O = spherical_singular_harmonic<T, P>(x, y, z);
	spherical_expansion<T, P> L;
	int count = 0;
	for (int n = 0; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			L[index(n, m)] = complex<T>(T(0), T(0));
			const int kmax = std::min(P - n, P - 1);
			for (int k = 0; k <= kmax; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					L[index(n, m)] += M(k, l).conj() * O(n + k, m + l);
					count += 9;
				}
			}
		}
	}
//spherical_inv_rotate_to_z(L, x, y, z);

//	printf( "%i\n", count);
	return L;
}

#include <fenv.h>

template<int P>
std::pair<real, real> test_M2L(real theta = 0.5) {
	real err = 0.0;
	int N = 10000;
	timer tm1, tm2;
	tm1.start();
	feenableexcept (FE_DIVBYZERO);
	feenableexcept (FE_INVALID);
	feenableexcept (FE_OVERFLOW);

	float err2 = 0.0;
	for (int i = 0; i < N; i++) {
		real x0, x1, x2, y0, y1, y2, z0, z1, z2;
		random_vector(x0, y0, z0);
		random_vector(x0, y0, z0);
		random_vector(x0, y0, z0);
			random_unit(x1, y1, z1);
		random_vector(x2, y2, z2);
		/*	x0 = 0.0;
		 y0 = 0.0;
		 z0 = 0.0;
		 x1 = 1.0;
		 y1 = 1.0;
		 z1 = 0.0;
		 x2 = 0.0;
		 y2 = 0.0;
		 z2 = 0.0;*/

		x1 /= 0.5 * theta;
		y1 /= 0.5 * theta;
		z1 /= 0.5 * theta;
		///	x1 = z1 = 0.0;
		//x2 = y2 = z2 = 0.0;
		//z2 = x2 = y2 = 0.0;
		//	z0 = x0 = y0 = 0.0;

		auto M0 = spherical_regular_harmonic<real, P - 1>(x0, y0, z0);
		array<real, P * P> M;
		for (int l = 0; l < P; l++) {
			for (int m = 0; m <= l; m++) {
				if (m) {
					M[l * (l + 1) - m] = M0[index(l, m)].imag();
				}
				M[l * (l + 1) + m] = M0[index(l, m)].real();
			}
		}
//		compressed_multipole<real, P - 1> Mc;
//		Mc.compress(M, 1.0);
		array<float, (P + 1) * (P + 1)> L0;
		array<float, 4> L3;
		for (int n = 0; n < (P + 1) * (P + 1); n++) {
			L0[n] = 0.0;
		}
		for (int n = 0; n < 4; n++) {
			L3[n] = 0.0;
		}
		M2L<real>(L0, M, x1, y1, z1);
		/*	M2L<real>(L3, M, x1, y1, z1);
		 for (int n = 0; n < 4; n++) {
		 printf("%e %e %e\n", L0[n]/ L3[n], L0[n], L3[n]);
		 }
		 printf("\n");
		 abort();*/
		auto H1 = spherical_regular_harmonic<real, P - 1>(x0, y0, z0);
		array<real, P * P> H2;
		regular_harmonic<real>(H2, x0, y0, z0);
		spherical_expansion_M2M(H1, x1, y1, z1);
		M2M(H2, x1, y1, z1);
		for (int l = 0; l < P; l++) {
			for (int m = -l; m <= l; m++) {
				auto h1 = m >= 0 ? H1[index(l, abs(m))].real() : H1[index(l, abs(m))].imag();
				auto h2 = H2[l * (l + 1) + m];
				printf("%i %i %e %e %e\n", l, m, h1, h2, h1 - h2);
			}
		}
		abort();
		auto L = spherical_expansion_ref_M2L<real, P>(M0, x1, y1, z1);
		spherical_expansion<real, P> L2;
		for (int l = 0; l <= P; l++) {
			for (int m = 0; m <= l; m++) {
				if (m != 0) {
					L2[index(l, m)].imag() = L0[l * (l + 1) - m];
				}
				L2[index(l, m)].real() = L0[l * (l + 1) + m];
			}
		}
		spherical_expansion_L2L(L, x2, y2, z2);
		spherical_expansion_L2L(L2, x2, y2, z2);
//			L.print();
//			printf("\n");

//			L2.print();
		//	abort();
		for (int l = 0; l <= P; l++) {
			float norm = 0.0;
			for (int m = 0; m <= l; m++) {
				norm += L(l, m).norm();
			}
			norm /= l + 1;
			for (int m = 0; m <= l; m++) {
				err2 += abs(L2(l, m).real() - L(l, m).real()) / norm;
				err2 += abs(L2(l, m).imag() - L(l, m).imag()) / norm;
			}
		}
		const real dx = (x2 + x1) - x0;
		const real dy = (y2 + y1) - y0;
		const real dz = (z2 + z1) - z0;
		const real r = sqrt(sqr(dx, dy, dz));
		const real ap = 1.0 / r;
		const real ax = -dx / (r * r * r);
		const real ay = -dy / (r * r * r);
		const real az = -dz / (r * r * r);
		const real np = L2(0, 0).real();
		const real nx = -L2(1, 1).real();
		const real ny = -L2(1, 1).imag();
		const real nz = -L2(1, 0).real();
		const real ag = sqrt(sqr(ax, ay, az));
		const real ng = sqrt(sqr(nx, ny, nz));
		//printf("%e %e %e | %e %e %e\n", nx, ny, nz, ax, ay, az);
//		abort();
		err += sqr((ag - ng) / ag);
	}
	tm1.stop();
	err = sqrt(err / N);
	return std::make_pair(err, err2 / N / 2.0);
}
template<int NMAX, int N = 3>
struct run_tests {
	void operator()() {
		auto a = test_M2L<N>();
		printf("%i %e %e\n", N, a.first, a.second);
		run_tests<NMAX, N + 1> run;
		run();
	}
};

template<int NMAX>
struct run_tests<NMAX, NMAX> {
	void operator()() {

	}
};

#define BLOCK_SIZE 32

#define BLOCK_SIZE 32

__global__ void test_old(multipole<float>* M, expansion<float>* Lptr, float* x, float* y, float* z, int N) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	auto& L = *Lptr;
	const int b = (size_t) bid * N / gridDim.x;
	const int e = (size_t)(bid + 1) * N / gridDim.x;
	expansion<float> D;
	expansion<float> L1;
	for (int i = b + tid; i < e; i += BLOCK_SIZE) {
		array<float, 3> X;
		X[XDIM] = x[i];
		X[YDIM] = y[i];
		X[ZDIM] = z[i];
		greens_function(D, X, true);
		M2L(L1, M[i], D, true);
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			L[i] += L1[i];
		}
	}

}

template<int P>
__global__ void test_new(array<float, P * P>* M, array<float, (P + 1) * (P + 1)>* Lptr, float* x, float* y, float* z, int N) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	auto& L = *Lptr;
	const int b = (size_t) bid * N / gridDim.x;
	const int e = (size_t)(bid + 1) * N / gridDim.x;
	array<float, (P + 1) * (P + 1)> L1;
	for (int i = b + tid; i < e; i += BLOCK_SIZE) {
		M2L<float>(L, M[i], x[i], y[i], z[i]);
	}

}
template<int P>
void speed_test(int N, int nblocks) {
	float* x, *y, *z;
	expansion<float>* Lc;
	array<float, (P + 1) * (P + 1)>* Ls;
	array<float, P * P>* Ms;
	multipole<float>* Mc;
	CUDA_CHECK(cudaMallocManaged(&Ls, sizeof(array<float, (P + 1) * (P + 1)> )));
	CUDA_CHECK(cudaMallocManaged(&Lc, sizeof(expansion<float> )));
	CUDA_CHECK(cudaMallocManaged(&Ms, N * sizeof(array<float, P * P> )));
	CUDA_CHECK(cudaMallocManaged(&Mc, N * sizeof(expansion<float> )));
	CUDA_CHECK(cudaMallocManaged(&x, sizeof(float) * N));
	CUDA_CHECK(cudaMallocManaged(&y, sizeof(float) * N));
	CUDA_CHECK(cudaMallocManaged(&z, sizeof(float) * N));
	for (int i = 0; i < N; i++) {
		x[i] = 2.0 * rand1() - 1.0;
		y[i] = 2.0 * rand1() - 1.0;
		z[i] = 2.0 * rand1() - 1.0;
	}
	for (int j = 0; j < N; j++) {
		spherical_expansion<float, P - 1> m;
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			(Mc)[j][i] = 2.0 * rand1() - 1.0;
		}
	}
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < P * P; i++) {
			Ms[j][i] = rand1();
		}
//		Ms[j].compress(m, 1.0);
	}
	int sblocks, cblocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&cblocks, (const void*) test_old, WARP_SIZE, 0));
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&sblocks, (const void*) test_new<P>, WARP_SIZE, 0));
	cblocks *= 82;
	sblocks *= 82;
	timer tms, tmc;
	tmc.start();
	test_old<<<cblocks,BLOCK_SIZE>>>(Mc,Lc,x,y,z,N);
	CUDA_CHECK(cudaDeviceSynchronize());
	tmc.stop();
	tms.start();
	test_new<P> <<<sblocks,BLOCK_SIZE>>>(Ms,Ls,x,y,z,N);
	CUDA_CHECK(cudaDeviceSynchronize());
	tms.stop();
	printf("Spherical = %e Cartesian = %e\n", tms.read(), tmc.read());
	CUDA_CHECK(cudaFree(Lc));
	CUDA_CHECK(cudaFree(Ls));
	CUDA_CHECK(cudaFree(Mc));
	CUDA_CHECK(cudaFree(Ms));
	CUDA_CHECK(cudaFree(x));
	CUDA_CHECK(cudaFree(y));
	CUDA_CHECK(cudaFree(z));
}

int main() {
	/*constexpr int nbits = 20;
	 bitstream<20> bits;
	 bits.write_bits(5, 5);
	 bits.write_bits(232323, 20);
	 bits.write_bits(9, 5);
	 bits.reset();
	 printf("%i\n", bits.read_bits(5));
	 printf("%i\n", bits.read_bits(20));
	 printf("%i\n", bits.read_bits(5));*/
	//speed_test<7>(2 * 1024 * 1024, 100);
	run_tests<13, 7> run;
	run();
//	constexpr int P = 7;
//	printf( "%i %i\n", sizeof(spherical_expansion<float,P-1>), sizeof(compressed_multipole<float,P-1>));
//printf("%e %e\n", Brot(10, -3, 1), brot<float, 10, -3, 1>::value);
	/*printf("err = %e\n", test_M2L<5>());
	 printf("err = %e\n", test_M2L<6>());
	 printf("err = %e\n", test_M2L<20>());*/

	/*constexpr int P = 5;
	 spherical_expansion<float, P> O;
	 float x = 1.0;
	 float y = -1.0;
	 float z = 1.0;
	 auto L = spherical_regular_harmonic<float, P>(x, y, z);
	 for (int l = 0; l <= P; l++) {
	 for (int m = 0; m <= l; m++) {
	 printf("%i %i %e %e\n", l, m, L(l, m).real(), L(l, m).imag());
	 }
	 }
	 spherical_rotate_to_z_regular(L, x, y, z);
	 spherical_inv_rotate_to_z_regular(L, x, y, z);
	 printf("\n");
	 for (int l = 0; l <= P; l++) {
	 for (int m = 0; m <= l; m++) {
	 printf("%i %i %e %e\n", l, m, L(l, m).real(), L(l, m).imag());
	 }
	 }
	 */
}
