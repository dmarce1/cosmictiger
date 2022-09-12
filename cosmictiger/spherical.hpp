#pragma once

#include <cosmictiger/containers.hpp>
#include <cosmictiger/complex.hpp>

template<class T, int P>
class LegendreP {
	array<T, P> p;
	CUDA_EXPORT inline LegendreP(T x) {
		if (P > 0) {
			p[0] = T(1);
		}
		if (P > 1) {
			p[1] = x;
		}
		for (int n = 1; n < P - 1; n++) {
			p[n + 1] = (T(2 * n + 1) * x * p[n] - T(n) * p[n - 1]) / T(n + 1);
		}
	}
	CUDA_EXPORT inline T operator()(int n) {
		return p[n];
	}
	CUDA_EXPORT inline LegendreP& operator=(const LegendreP& other) {
		p = other.p;
		return p;
	}

};

template<class T, int P>
class AssociatedLegendre {
	array<T, P * (P + 1) / 2> pm;
	CUDA_EXPORT inline AssociatedLegendre(T x) {
		LegendreP p(x);
		const T y = sqrt(T(1) - sqr(x));
		for (int n = 0; n < P; n++) {
			pm[n * (n + 1) / 2] = p[n];
		}
		for (int m = 1; m < P; m++) {
			pm[m * (m + 1) / 2 + m] = -T(2 * m - 1) * y * pm[(m - 1) * m / 2 + m - 1];
			for (int n = m + 1; n < P - 1; n++) {
				pm[(n + 1) * (n + 2) / 2 + m] = (T(2 * n + 1) * x * pm[n * (n + 1) / 2 + m] - T(n + m) * pm[n * (n - 1) / 2 + m]) / T(n - m + 1);
			}
		}
	}
	CUDA_EXPORT inline T operator()(int n, int m) {
		return pm[n * (n + 1) / 2 + m];
	}
	CUDA_EXPORT inline AssociatedLegendre& operator=(const AssociatedLegendre& other) {
		pm = other.pm;
		return pm;
	}

};

template<class T, int P>
class SphericalY {
	array<complex<T>, P * (P + 1) / 2> y;
	CUDA_EXPORT inline SphericalY(T x, T y, T z) {
		const T R2 = sqr(x) + sqr(y);
		const T Rinv = rsqrt(R2 + T(1e-20));
		const T r2 = R2 + sqr(z);
		const T rinv = rsqrt(r2 + T(1e-20));
		const T cos0 = z * rinv;
		const T cosphi = x * Rinv;
		const T sinphi = y * Rinv;
		complex<T> expiphi = complex<T>(cosphi, sinphi);
		AssociatedLegendre Pm(cos0);
		for (int m = 0; m < P; m++) {
			complex<T> R = complex<T>(T(1), T(0));
			for (int n = m; n < P; n++) {
				y[n * (n + 1) / 2 + m] = R * Pm(n, m) * sqrt(factorial(n - m) / factorial(n + m));
			}
			R *= expiphi;
		}
	}
	CUDA_EXPORT inline T operator()(int n, int m) {
		return m >= 0 ? y[n * (n + 1) / 2 + m] : y[n * (n + 1) / 2 - m].conj();
	}
	CUDA_EXPORT inline SphericalY& operator=(const SphericalY& other) {
		y = other.y;
		return y;
	}

};

template<class T, int P>
struct spherical_expansion: public array<complex<T>, P * (P + 1) / 2> {
	CUDA_EXPORT inline complex<T> operator()(int n, int m) {
		return (*this)[n * (n + 1) / 2 + m];
	}
	CUDA_EXPORT inline complex<T> operator()(int n, int m) const {
		return m >= 0 ? (*this)[n * (n + 1) / 2 + m] : (*this)[n * (n + 1) / 2 - m].conj();
	}
};

template<class T, int P>
CUDA_EXPORT inline spherical_expansion<T, P> sphericalP2M(T x, T y, T z) {
	spherical_expansion<T, P> M;
	const T r2 = sqr(x, y, z);
	const T r = sqrt(r2);
	SphericalY Y(x, y, z);
	T rpow = T(1);
	for (int n = 0; n < P; n++) {
		for (int m = 0; m <= m; m++) {
			M(n, m) = rpow * Y(n, -m);
		}
		rpow *= r;
	}
	return M;
}

template<class T>
CUDA_EXPORT inline complex<T> ipow(int n) {
	complex<T> I(T(0), T(1));
	complex<T> y = I(T(1), T(1));
	for (int i = 0; i < abs(n % 4); i++) {
		y *= I;
	}
}

template<class T, int P>
CUDA_EXPORT inline spherical_expansion<T, P> sphericalM2M(const spherical_expansion<T, P>& O, T x, T y, T z) {
	spherical_expansion<T, P> M;
	const T r2 = sqr(x, y, z);
	const T r = sqrt(r2);
	SphericalY Y(x, y, z);
	T rpow = T(1);
	const auto A = [](int n, int m) {
		return rsqrt(T(factorial(n+m)*factorial(n-m)));
	};
	for (int j = 0; j < P; j++) {
		for (int k = 0; k <= j; k++) {
			M(j, k) = complex<T>(T(0), T(0));
			const T Ajk = A(j, k);
			for (int n = 0; n <= j; n++) {
				for (int m = k - j; m <= n; m++) {
					const T Anm = A(n, m);
					const T Anmjk = A(j - n, k - m);
					M(j, k) += O(j - n, k - m) * ipow<T>(k - abs(m) - abs(k - m)) * Anm * Anmjk / Ajk * rpow * Y(n, -m);
				}
			}
		}
		rpow *= r;
	}
	return M;
}

template<class T>
CUDA_EXPORT inline T nonepow(int n) {
	return n % 2 ? T(-1) : T(1);
}

template<class T, int P, int Q>
CUDA_EXPORT inline spherical_expansion<T, P> sphericalM2L(const spherical_expansion<T, Q>& O, T x, T y, T z) {
	spherical_expansion<T, P> L;
	const T r2 = sqr(x, y, z);
	const T r = sqrt(r2);
	SphericalY Y(x, y, z);
	T rinv = T(1) / r;
	T rpow0 = rinv;
	const auto A = [](int n, int m) {
		return rsqrt(T(factorial(n+m)*factorial(n-m)));
	};
	for (int j = 0; j < P; j++) {
		for (int k = 0; k <= j; k++) {
			const T Ajk = A(j, k);
			L(j, k) = complex<T>(T(0), T(0));
			T rpow = rpow0;
			const int nmax = j < (P - Q) ? Q : P - j;
			for (int n = 0; n < nmax; n++) {
				for (int m = k - j; m <= n; m++) {
					const T Anm = A(n, m);
					const T Anmjk = A(j + n, m - k);
					L(j, k) += O(n, m) * ipow<T>(k - abs(m) - abs(k - m)) * Anm * Ajk / Anmjk * nonepow<T>(n) * rpow * Y(j + n, m - k);
				}
				rpow *= rinv;
			}
		}
		rpow0 *= rinv;
	}
	return L;
}

template<class T, int P, int Q = P>
CUDA_EXPORT inline spherical_expansion<T, P> sphericalL2L(const spherical_expansion<T, P>& O, T x, T y, T z) {
	spherical_expansion<T, Q> L;
	const T r2 = sqr(x, y, z);
	const T r = sqrt(r2);
	SphericalY Y(x, y, z);
	const auto A = [](int n, int m) {
		return rsqrt(T(factorial(n+m)*factorial(n-m)));
	};
	for (int j = 0; j < Q; j++) {
		for (int k = 0; k <= j; k++) {
			const T Ajk = A(j, k);
			L(j, k) = complex<T>(T(0), T(0));
			T rpow = T(1);
			for (int n = j; n < P; n++) {
				for (int m = j - n + k; m <= n; m++) {
					const T Anm = A(n, m);
					const T Anmjk = A(n - j, m - k);
					L(j, k) += O(n, m) * ipow<T>(abs(m) - k - abs(m - k)) * Anmjk * Ajk / Anm * nonepow<T>(n + j) * rpow * Y(n - j, m - k);
				}
				rpow *= r;
			}
		}
	}
	return L;
}

