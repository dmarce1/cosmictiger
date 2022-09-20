#include <stdio.h>
#include <utility>
#include <cmath>
#include <cosmictiger/complex.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/timer.hpp>

using real = float;

constexpr int index(int l, int m) {
	return l * (l + 1) / 2 + m;
}

template<class T>
constexpr T nonepow(int m) {
	return m % 2 == 0 ? T(1) : T(-1);
}

template<class T, int P>
struct spherical_expansion: public array<complex<T>, (P + 1) * (P + 2) / 2> {
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

template<class T, int N, int M, int L, bool TERM = (L < -N) || (L > N)>
struct brot {
	static constexpr T value =
			(M == 0) ?
					T(0.5) * (brot<T, N - 1, 0, L - 1>::value - brot<T, N - 1, 0, L + 1>::value) :
					(M > 0 ?
							T(0.5) * (brot<T, N - 1, M - 1, L - 1>::value + brot<T, N - 1, M - 1, L + 1>::value + T(2) * brot<T, N - 1, M - 1, L>::value) :
							T(0.5) * (brot<T, N - 1, M + 1, L - 1>::value + brot<T, N - 1, M + 1, L + 1>::value - T(2) * brot<T, N - 1, M + 1, L>::value));
};

template<class T, bool TERM>
struct brot<T, 0, 0, 0, TERM> {
	static constexpr T value = T(1);
};

template<class T, int N, int M, int L>
struct brot<T, N, M, L, true> {
	static constexpr T value = T(0);
};

double Brot(int n, int m, int l) {
	if (n == 0 && m == 0 && l == 0) {
		return 1.0;
	} else if (abs(l) > n) {
		return 0.0;
	} else if (m == 0) {
		return 0.5 * (Brot(n - 1, m, l - 1) - Brot(n - 1, m, l + 1));
	} else if (m > 0) {
		return 0.5 * (Brot(n - 1, m - 1, l - 1) + Brot(n - 1, m - 1, l + 1) + 2.0 * Brot(n - 1, m - 1, l));
	} else {
		return 0.5 * (Brot(n - 1, m + 1, l - 1) + Brot(n - 1, m + 1, l + 1) - 2.0 * Brot(n - 1, m + 1, l));
	}
}

template<class T, int P, int N, int M, int L, bool SING, bool TERM = (L > N)>
struct spherical_swap_xz_l {
	void operator()(spherical_expansion<T, P>& O, const spherical_expansion<T, P>& A) {
		spherical_swap_xz_l<T, P, N, M, L + 1, SING> nextl;
		O[index(N, M)] += A(N,L) * brot<T, N, SING ? M : L, SING ? L : M>::value;
		nextl(O, A);
	}
};

template<class T, int P, int N, int M, int L, bool SING>
struct spherical_swap_xz_l<T, P, N, M, L, SING, true> {
	void operator()(spherical_expansion<T, P>& O, const spherical_expansion<T, P>& A) {
	}
};

template<class T, int P, int N, int M, bool SING, bool TERM = (M > N)>
struct spherical_swap_xz_m {
	void operator()(spherical_expansion<T, P>& O, const spherical_expansion<T, P>& A) {
		int n = N;
		int m = M;
		O[index(n, m)] = 0.0;

		spherical_swap_xz_m<T, P, N, M + 1, SING> nextm;
		spherical_swap_xz_l<T, P, N, M, -N, SING> nextl;
		nextl(O, A);
		nextm(O, A);
	}
};

template<class T, int P, int N, int M, bool SING>
struct spherical_swap_xz_m<T, P, N, M, SING, true> {
	void operator()(spherical_expansion<T, P>& O, const spherical_expansion<T, P>& A) {
	}
};

template<class T, int P, int N, bool SING, bool TERM = (N > P)>
struct spherical_swap_xz_n {
	void operator()(spherical_expansion<T, P>& O, const spherical_expansion<T, P>& A) {

		spherical_swap_xz_m<T, P, N, 0, SING> nextm;
		spherical_swap_xz_n<T, P, N + 1, SING> nextn;
		nextm(O, A);
		nextn(O, A);
	}
};

template<class T, int P, int N, bool SING>
struct spherical_swap_xz_n<T, P, N, SING, true> {
	void operator()(spherical_expansion<T, P>& O, const spherical_expansion<T, P>& A) {
	}
};

template<class T, int P>
void spherical_swap_xz_singular(spherical_expansion<T, P>& O) {
	const auto A = O;
	spherical_swap_xz_n<T, P, 1, true> run;
	run(O, A);
}

template<class T, int P>
void spherical_swap_xz_regular(spherical_expansion<T, P>& O) {
	const auto A = O;
	spherical_swap_xz_n<T, P, 1, false> run;
	run(O, A);
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
				const int lmin = std::max(-k, m - n + k);
				const int lmax = std::min(k, m + n - k);
				for (int l = lmin; l <= lmax; l++) {
					M[index(n, m)] += Y(k, l) * M0(n - k, m - l);
				}
			}
		}
	}
}

template<class T, int P>
void spherical_rotate_z(spherical_expansion<T, P>& O, T phi) {
	for (int l = 0; l <= P; l++) {
		for (int m = 0; m <= l; m++) {
			O[index(l, m)] *= expc(complex<T>(0, -m * phi));
		}
	}
}

template<class T, int P>
spherical_expansion<T, P> spherical_expansion_M2L(spherical_expansion<T, P - 1> M, T x, T y, T z) {
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

template<class T, int P>
void spherical_rotate_to_z_regular(spherical_expansion<T, P>& O, T x, T y, T z) {
	const T phi = atan2(x, y);
	const T theta = atan2(sqrt(x * x + y * y), z);
	spherical_rotate_z(O, -phi);
	spherical_swap_xz_regular(O);
	spherical_rotate_z(O, theta);
	spherical_swap_xz_regular(O);
}

template<class T, int P>
void spherical_inv_rotate_to_z_singular(spherical_expansion<T, P>& O, T x, T y, T z) {
	const T phi = atan2(x, y);
	const T theta = atan2(sqrt(x * x + y * y), z);
	spherical_swap_xz_singular(O);
	spherical_rotate_z(O, -theta);
	spherical_swap_xz_singular(O);
	spherical_rotate_z(O, phi);
}

double factorial(int n) {
	return n == 0 ? 1.0 : n * factorial(n - 1);
}

template<class T, int P>
spherical_expansion<T, P> spherical_expansion_rot_M2L(spherical_expansion<T, P - 1> M, T x, T y, T z) {
	spherical_rotate_to_z_regular(M, x, y, z);
	const T r = sqrt(x * x + y * y + z * z);
	//	const auto O = spherical_singular_harmonic<T, P>(x,y,z);
	spherical_expansion<T, P> L;
	int count = 0;
	for (int n = 0; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			L[index(n, m)] = complex<T>(T(0), T(0));
			const int kmax = std::min(P - n, P - 1);
			for (int k = m; k <= kmax; k++) {
				L[index(n, m)] += nonepow<T>(m) * M(k, m).conj() * factorial(n + k) * pow(r, -k - n - 1);
			}
		}
	}
	spherical_inv_rotate_to_z_singular(L, x, y, z);

//	printf( "%i\n", count);
	return L;
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

template<int P>
real test_M2L(real theta = 0.5) {
	real err = 0.0;
	int N = 10000;
	timer tm1, tm2;
	tm1.start();
	for (int i = 0; i < N; i++) {
		real x0, x1, x2, y0, y1, y2, z0, z1, z2;
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
		x2 = y2 = z2 = 0.0;
		auto M = spherical_regular_harmonic<real, P - 1>(x0, y0, z0);
		auto L = spherical_expansion_rot_M2L<real, P>(M, x1, y1, z1);
		spherical_expansion_L2L(L, x2, y2, z2);
		const real dx = (x2 + x1) - x0;
		const real dy = (y2 + y1) - y0;
		const real dz = (z2 + z1) - z0;
		const real r = sqrt(sqr(dx, dy, dz));
		const real ap = 1.0 / r;
		const real ax = -dx / (r * r * r);
		const real ay = -dy / (r * r * r);
		const real az = -dz / (r * r * r);
		const real np = L(0, 0).real();
		const real nx = -L(1, 1).real();
		const real ny = -L(1, 1).imag();
		const real nz = -L(1, 0).real();
		const real ag = sqrt(sqr(ax, ay, az));
		const real ng = sqrt(sqr(nx, ny, nz));

		err += sqr((ag - ng) / ag);
	}
	tm1.stop();
	err = sqrt(err / N);
	PRINT("%i %e %e\n", P, err, tm1.read());
	return err;
}

int main() {
	//printf("%e %e\n", Brot(10, -3, 1), brot<float, 10, -3, 1>::value);
	test_M2L<7>();

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
