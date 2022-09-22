#include <stdio.h>
#include <utility>
#include <cmath>
#define ORDER 8
#define USE_CUDA
#include <cosmictiger/complex.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/simd.hpp>
#include <cuda.h>
#include <cosmictiger/fmm_kernels.hpp>

using real = float;

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

template<int N, int M, int L, bool TERM = (L < -N) || (L > N)>
struct brot {
	static constexpr real value =
			(M == 0) ?
					real(0.5) * (brot<N - 1, 0, L - 1>::value - brot<N - 1, 0, L + 1>::value) :
					(M > 0 ?
							real(0.5) * (brot<N - 1, M - 1, L - 1>::value + brot<N - 1, M - 1, L + 1>::value + real(2) * brot<N - 1, M - 1, L>::value) :
							real(0.5) * (brot<N - 1, M + 1, L - 1>::value + brot<N - 1, M + 1, L + 1>::value - real(2) * brot<N - 1, M + 1, L>::value));
};

template<bool TERM>
struct brot<0, 0, 0, TERM> {
	static constexpr real value = 1;
};

template<int N, int M, int L>
struct brot<N, M, L, true> {
	static constexpr real value = 0;
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

template<class T, bool zero, bool one, bool none>
struct accumulate {
	static constexpr int nops = 2;CUDA_EXPORT
	inline void operator()(T& A, T coeff, const T& B) const {
		A += B * coeff;
	}
};

template<class T>
struct accumulate<T, true, false, false> {
	static constexpr int nops = 0;CUDA_EXPORT
	inline void operator()(T& A, T coeff, const T& B) const {
	}
};

template<class T>
struct accumulate<T, false, true, false> {
	static constexpr int nops = 1;CUDA_EXPORT
	inline void operator()(T& A, T coeff, const T& B) const {
		A += B;
	}
};

template<class T>
struct accumulate<T, false, false, true> {
	static constexpr int nops = 1;CUDA_EXPORT
	inline void operator()(T& A, T coeff, const T& B) const {
		A -= B;
	}
};

template<class T>
CUDA_EXPORT inline constexpr bool close2(T a, T b) {
	return (a > b - T(1e-6)) && (a < b + T(1e-6));
}

template<class T, int P, int N, int M, int L, bool SING, bool UPPER, bool NOEVENHI, bool TERM = UPPER ? (L > P - N || L > N) : (L > N)>
struct spherical_swap_xz_l {
	static constexpr auto co1 = brot<N, SING ? M : L, SING ? L : M>::value;
	static constexpr auto co2 = brot<N, SING ? M : -L, SING ? -L : M>::value;
	static constexpr auto sgn = L % 2 == 0 ? 1 : -1;
	static constexpr auto cor = L == 0 ? co1 : co1 + sgn * co2;
	static constexpr auto coi = L == 0 ? real(0) : co1 - sgn * co2;
	using ltype = spherical_swap_xz_l<T, P, N, M, L + 1 + (NOEVENHI && N ==P), SING, UPPER,NOEVENHI>;
	using artype = accumulate<T, close2(cor, real(0)), close2(cor, real(1)), close2(cor, real(-1))>;
	using aitype = accumulate<T, close2(coi, real(0)), close2(coi, real(1)), close2(coi, real(-1))>;
	static constexpr int nops = ltype::nops + artype::nops + aitype::nops;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& O, const spherical_expansion<T, P>& A) const {
		constexpr ltype nextl;
		constexpr artype real;
		constexpr aitype imag;
		real(O[index(N, M)].real(), cor, A(N, L).real());
		imag(O[index(N, M)].imag(), coi, A(N, L).imag());
		nextl(O, A);
	}
};

template<class T, int P, int N, int M, int L, bool SING, bool UPPER, bool NOEVENHI>
struct spherical_swap_xz_l<T, P, N, M, L, SING, UPPER, NOEVENHI, true> {
	static constexpr int nops = 0;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& O, const spherical_expansion<T, P>& A) const {
	}
};

template<class T, int P, int N, int M, bool SING, bool LOWER, bool UPPER, bool NOEVENHI, bool TERM = LOWER ? (M > P - N || M > N) : (M > N)>
struct spherical_swap_xz_m {
	static constexpr int LB = (NOEVENHI && N == P) ? (((P + N) / 2) % 2 == 1 ? 1 : 0) : 0;
	using mtype = spherical_swap_xz_m<T, P, N, M + 1, SING, LOWER, UPPER, NOEVENHI>;
	using ltype = spherical_swap_xz_l<T, P, N, M, LB, SING, UPPER, NOEVENHI>;
	static constexpr int nops = mtype::nops + ltype::nops + 1;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& O, const spherical_expansion<T, P>& A) const {
		O[index(N, M)] = T(0);
		constexpr mtype nextm;
		constexpr ltype nextl;
		nextl(O, A);
		nextm(O, A);
	}
};

template<class T, int P, int N, int M, bool SING, bool LOWER, bool UPPER, bool NOEVENHI>
struct spherical_swap_xz_m<T, P, N, M, SING, LOWER, UPPER, NOEVENHI, true> {
	static constexpr int nops = 0;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& O, const spherical_expansion<T, P>& A) const {
	}
};

template<class T, int P, int N, bool SING, bool LOWER, bool UPPER, bool NOEVENHI, bool TERM = (N > P)>
struct spherical_swap_xz_n {
	using mtype = spherical_swap_xz_m<T, P, N, 0, SING, LOWER, UPPER, NOEVENHI>;
	using ltype = spherical_swap_xz_n<T, P, N + 1, SING, LOWER, UPPER, NOEVENHI>;
	static constexpr int nops = mtype::nops + ltype::nops;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& O, const spherical_expansion<T, P>& A) const {
		constexpr mtype nextm;
		constexpr ltype nextn;
		nextm(O, A);
		nextn(O, A);
	}
};

template<class T, int P, int N, bool SING, bool LOWER, bool UPPER, bool NOEVENHI>
struct spherical_swap_xz_n<T, P, N, SING, LOWER, UPPER, NOEVENHI, true> {
	static constexpr int nops = 0;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& O, const spherical_expansion<T, P>& A) const {
	}
};

template<class T, bool REAL, bool IMAG>
struct rotate_z_mult {
	static constexpr int nops = 6;CUDA_EXPORT
	inline void operator()(complex<T>& O, const complex<T>& R) const {
		O *= R;
	}
};

template<class T>
struct rotate_z_mult<T, true, false> {
	static constexpr int nops = 2;CUDA_EXPORT
	inline void operator()(complex<T>& O, const complex<T>& R) const {
		O.imag() = O.real() * R.imag();
		O.real() *= R.real();
	}
};

template<class T>
struct rotate_z_mult<T, false, true> {
	static constexpr int nops = 3;CUDA_EXPORT
	inline void operator()(complex<T>& O, const complex<T>& R) const {
		O.real() = -O.imag() * R.imag();
		O.imag() *= R.real();
	}
};

template<class T, int P, int L, int M, bool NOEVENHI, bool ODD, bool TERM = (M > L)>
struct spherical_rotate_z_m {
	using mtype =spherical_rotate_z_m<T, P, L, M + 1, NOEVENHI, ODD>;
	using optype = rotate_z_mult<T,NOEVENHI && (L >= P - 1 && (M % 2 != ((P + L) / 2) % 2)), false>;
	static constexpr int nops = mtype::nops + 6;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& O, complex<T>* R) const {
		constexpr mtype nextm;
		O[index(L, M)] *= R[M];
		nextm(O, R);
	}
};

template<class T, int P, int L, int M>
struct spherical_rotate_z_m<T, P, L, M, false, true, false> {
	using mtype = spherical_rotate_z_m<T, P, L, M + 1, false, true>;
	using optype = rotate_z_mult<T,(L == P) && (M % 2 == 0), (L == P) && (M % 2 != 0)>;
	static constexpr int nops = mtype::nops + optype::nops;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& O, complex<T>* R) const {
		constexpr mtype nextm;
		constexpr optype op;
		op(O[index(L, M)], R[M]);
		nextm(O, R);
	}
};

template<class T, int P, int L, int M, bool NOEVENHI, bool ODD>
struct spherical_rotate_z_m<T, P, L, M, NOEVENHI, ODD, true> {
	static constexpr int nops = 0;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& O, complex<T>* R) const {
	}
};

template<class T, int P, int L, bool NOEVENHI, bool ODD, bool TERM = (L > P)>
struct spherical_rotate_z_l {
	using ltype = spherical_rotate_z_l<T, P, L + 1, NOEVENHI, ODD>;
	using mtype = spherical_rotate_z_m<T, P, L, (NOEVENHI && L == P) ? (((P + L) / 2) % 2 == 1 ? 1 : 0) : 0, NOEVENHI,ODD>;
	static constexpr int nops = ltype::nops + mtype::nops;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& O, complex<T>* R) const {
		constexpr ltype nextl;
		constexpr mtype nextm;
		nextm(O, R);
		nextl(O, R);
	}
};

template<class T, int P, int L, bool NOEVENHI, bool ODD>
struct spherical_rotate_z_l<T, P, L, NOEVENHI, ODD, true> {
	static constexpr int nops = 0;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& O, complex<T>*) const {
	}
};

template<class T, int P, bool NOEVENHI, bool ODD>
struct spherical_rotate_z {
	using ltype = spherical_rotate_z_l<T, P, 1, NOEVENHI,ODD>;
	static constexpr int nops = 6 * (P - 1) + ltype::nops;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& O, const complex<T>& R0) const {
		array<complex<T>, P + 1> R;
		R[0] = complex<T>(1, 0);
		for (int n = 1; n <= P; n++) {
			R[n] = R[n - 1] * R0;
		}
		constexpr ltype run;
		run(O, R.data());
	}
};

template<class T, int P>
struct spherical_rotate_to_z_regular {
	using xz_type = spherical_swap_xz_n<T, P, 1, false, false, false, false>;
	using rot_type =spherical_rotate_z<T, P, false, false>;
	using xz_type2 =spherical_swap_xz_n<T, P, 1, false, true, false, false>;
	static constexpr int nops = 5 + (xz_type::nops + xz_type2::nops + 2 * rot_type::nops);CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& O, T x, T y, T z, T R, T Rinv, T rinv) const {
		constexpr xz_type xz;
		constexpr xz_type2 trunc_xz;
		rot_type rot;
		rot(O, complex<T>(y * Rinv, x * Rinv));
		auto A = O;
		xz(O, A);
		rot(O, complex<T>(z * rinv, -R * rinv));
		A = O;
		trunc_xz(O, A);
	}
};

template<class T, int P>
struct spherical_inv_rotate_to_z_singular {
	using truncxz_type =spherical_swap_xz_n<T, P, 1, true, false, true, false>;
	using xz_type =spherical_swap_xz_n<T, P, 1, true, false, false, true>;
	using rtype1 = spherical_rotate_z<T, P, false,true>;
	using rtype2 = spherical_rotate_z<T, P, true, false>;
	static constexpr int nops = 5 + rtype1::nops + rtype2::nops + xz_type::nops + truncxz_type::nops;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& O, T x, T y, T z, T R, T Rinv, T rinv) const {
		constexpr truncxz_type trunc_xz;
		constexpr xz_type xz;
		constexpr rtype1 rot1;
		constexpr rtype2 rot2;
		auto A = O;
		trunc_xz(O, A);
		rot2(O, complex<T>(z * rinv, R * rinv));
		A = O;
		xz(O, A);
//		O.print();
//		abort();
		rot1(O, complex<T>(y * Rinv, -x * Rinv));
	}
};

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

double factorial(int n) {
	return n == 0 ? 1.0 : n * factorial(n - 1);
}

template<class T, int N>
struct facto {
	CUDA_EXPORT
	constexpr T operator()() const {
		constexpr facto<T, N - 1> f;
		return T(N) * f();
	}
};

template<class T>
struct facto<T, 0> {
	CUDA_EXPORT
	constexpr T operator()() const {
		return T(1);
	}
};

template<class T, int P, int N, int M, int K, bool TERM = (K > P - N || K > (P - 1))>
struct spherical_expansion_M2L_k {
	static constexpr int nops = 5 + spherical_expansion_M2L_k<T, P, N, M, K + 1>::nops;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& L, const spherical_expansion<T, P - 1>& O, const T* rpow) const {
		constexpr spherical_expansion_M2L_k<T, P, N, M, K + 1> next;
		constexpr facto<real, N + K> factorial;
		constexpr real c0 = ((M % 2 == 0 ? (1) : (-1)) * factorial());
		L[index(N, M)] += O(K, M) * (c0 * rpow[K + N + 1]);
		next(L, O, rpow);
	}
};

template<class T, int P, int N, int K>
struct spherical_expansion_M2L_k<T, P, N, 0, K, false> {
	static constexpr int M = 0;
	static constexpr int nops = 3 + spherical_expansion_M2L_k<T, P, N, M, K + 1>::nops;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& L, const spherical_expansion<T, P - 1>& O, const T* rpow) const {
		constexpr spherical_expansion_M2L_k<T, P, N, M, K + 1> next;
		constexpr facto<real, N + K> factorial;
		constexpr real c0 = ((M % 2 == 0 ? (1) : (-1)) * factorial());
		L[index(N, M)].real() += O(K, M).real() * (c0 * rpow[K + N + 1]);
		next(L, O, rpow);
	}
};

template<class T, int P, int N, int M, int K>
struct spherical_expansion_M2L_k<T, P, N, M, K, true> {
	static constexpr int nops = 0;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& L, const spherical_expansion<T, P - 1>&, const T*) const {
	}
};

template<class T, int P, int N, int M, bool TERM = (M > N)>
struct spherical_expansion_M2L_m {
	static constexpr int nops = spherical_expansion_M2L_m<T, P, N, M + 1>::nops + spherical_expansion_M2L_k<T, P, N, M, M>::nops;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& L, const spherical_expansion<T, P - 1>& O, const T* rpow) const {
		constexpr spherical_expansion_M2L_m<T, P, N, M + 1> nextm;
		constexpr spherical_expansion_M2L_k<T, P, N, M, M> nextk;
		L[index(N, M)] = complex<T>(T(0), T(0));
		nextk(L, O, rpow);
		nextm(L, O, rpow);
	}
};

template<class T, int P, int N, int M>
struct spherical_expansion_M2L_m<T, P, N, M, true> {
	static constexpr int nops = 0;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& L, const spherical_expansion<T, P - 1>&, const T*) const {
	}
};

template<class T, int P, int N, bool TERM = (N > P)>
struct spherical_expansion_M2L_n {
	static constexpr int nops = spherical_expansion_M2L_n<T, P, N + 1>::nops + spherical_expansion_M2L_m<T, P, N, 0>::nops;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& L, const spherical_expansion<T, P - 1>& M, const T* rpow) const {
		constexpr spherical_expansion_M2L_n<T, P, N + 1> nextn;
		constexpr spherical_expansion_M2L_m<T, P, N, 0> nextm;
		nextm(L, M, rpow);
		nextn(L, M, rpow);
	}
};

template<class T, int P, int N>
struct spherical_expansion_M2L_n<T, P, N, true> {
	static constexpr int nops = 0;CUDA_EXPORT
	inline void operator()(spherical_expansion<T, P>& L, const spherical_expansion<T, P - 1>& M, const T*) const {
	}
};

template<class T, int P>
struct spherical_expansion_M2L_type {
	static constexpr int nops = (P + 1) + 24 + spherical_rotate_to_z_regular<T, P - 1>::nops + spherical_inv_rotate_to_z_singular<T, P>::nops
			+ spherical_expansion_M2L_n<T, P, 0>::nops;CUDA_EXPORT
	inline spherical_expansion<T, P> operator()(spherical_expansion<T, P - 1> M, T x, T y, T z) const {
		const T R2 = (x * x + y * y);
		const T R = sqrt(x * x + y * y);
		const T Rinv = T(1) / R;
		const T r = sqrt(z * z + R2);
		const T rinv = T(1) / r;
		constexpr spherical_rotate_to_z_regular<T, P - 1> rot;
		constexpr spherical_inv_rotate_to_z_singular<T, P> inv_rot;
		rot(M, x, y, z, R, Rinv, rinv);
		spherical_expansion<T, P> L;
		array<T, P + 2> rpow;
		rpow[0] = T(1);
		for (int i = 1; i < P + 2; i++) {
			rpow[i] = rpow[i - 1] * rinv;
		}
		constexpr spherical_expansion_M2L_n<T, P, 0> run;
		run(L, M, rpow.data());
		inv_rot(L, x, y, z, R, Rinv, rinv);
		return L;
	}
};

template<class T, int P>
CUDA_EXPORT spherical_expansion<T, P> spherical_expansion_M2L(spherical_expansion<T, P - 1> M, T x, T y, T z) {
	constexpr spherical_expansion_M2L_type<T, P> run;
	return run(M, x, y, z);
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
//		x2 = y2 = z2 = 0.0;
		auto M = spherical_regular_harmonic<real, P - 1>(x0, y0, z0);
		auto L = spherical_expansion_M2L<real, P>(M, x1, y1, z1);
		//				auto L2 = spherical_expansion_ref_M2L<real, P>(M, x1, y1, z1);
		//	 L.print();
		//	 printf( "\n");
		for (int n = 0; n <= P; n++) {
			for (int m = 0; m <= n; m++) {
//				 L2[index(n,m)] -= L[index(n,m)];
			}
		}
		/*	 L2.print();
		 abort();*/
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
	return err;
}
template<int NMAX, int N = 3>
struct run_tests {
	void operator()() {
		printf("%i %i %e\n", N, spherical_expansion_M2L_type<float, N>::nops, test_M2L<N>());
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

__global__ void test_old(multipole<float>* M, expansion<float>* Lptr, float* x, float* y, float* z, int N) {
	const int tid = threadIdx.x;
	const int bid = threadIdx.x;
	auto& L = *Lptr;
	const int b = bid * N / BLOCK_SIZE;
	const int e = (bid + 1) * N / BLOCK_SIZE;
	expansion<float> D;
	expansion<float> L1;
	for (int i = b; i < e; i++) {
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
__global__ void test_new(spherical_expansion<float, P - 1>* M, spherical_expansion<float, P>* Lptr, float* x, float* y, float* z, int N) {
	const int tid = threadIdx.x;
	const int bid = threadIdx.x;
	auto& L = *Lptr;
	const int b = bid * N / BLOCK_SIZE;
	const int e = (bid + 1) * N / BLOCK_SIZE;
	expansion<float> D;
	expansion<float> L1;
	for (int i = b; i < e; i++) {
		auto L1 = spherical_expansion_M2L<float, P>(M[i], x[i], y[i], z[i]);
		for (int l = 0; l <= P; l++) {
			for (int m = 0; m <= l; m++) {
				L[index(l, m)] += (L1, l, m);
			}
		}
	}

}

template<int P>
void speed_test(int N, int nblocks) {
	float* x, *y, *z;
	spherical_expansion<float, P>* Ls;
	expansion<float>* Lc;
	spherical_expansion<float, P - 1>* Ms;
	multipole<float>* Mc;
	CUDA_CHECK(cudaMallocManaged(&Ls, sizeof(spherical_expansion<float, P> )));
	CUDA_CHECK(cudaMallocManaged(&Lc, sizeof(expansion<float> )));
	CUDA_CHECK(cudaMallocManaged(&Ms, N*sizeof(spherical_expansion<float, P> )));
	CUDA_CHECK(cudaMallocManaged(&Mc, N*sizeof(expansion<float> )));
	CUDA_CHECK(cudaMallocManaged(&x, sizeof(float) * N));
	CUDA_CHECK(cudaMallocManaged(&y, sizeof(float) * N));
	CUDA_CHECK(cudaMallocManaged(&z, sizeof(float) * N));
	for (int i = 0; i < N; i++) {
		x[i] = 2.0 * rand1() - 1.0;
		y[i] = 2.0 * rand1() - 1.0;
		z[i] = 2.0 * rand1() - 1.0;
	}
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			(Mc)[j][i] = 2.0 * rand1() - 1.0;
		}
	}
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < Ls->size(); i++) {
			(Ms)[j][i] = 2.0 * rand1() - 1.0;
		}
	}
	int sblocks, cblocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&cblocks, (const void*) test_old, WARP_SIZE, 0));
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&sblocks, (const void*) test_new<P>, WARP_SIZE, 0));
	timer tms, tmc;
	tmc.start();
	test_old<<<cblocks,BLOCK_SIZE>>>(Mc,Lc,x,y,z,N);
	CUDA_CHECK(cudaDeviceSynchronize());
	tmc.stop();
	tms.start();
	test_new<<<sblocks,BLOCK_SIZE>>>(Ms,Ls,x,y,z,N);
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

	speed_test<7>(4*1024 * 1024, 100);
//	run_tests<11, 3> run;
//	run();
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
