#include <stdio.h>
#include <utility>
#include <cmath>
#include <cosmictiger/complex.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/math.hpp>

constexpr int index(int l, int m) {
	return l * (l + 1) / 2 + m;
}

template<class T>
T nonepow(int m) {
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
spherical_expansion<T, P> spherical_expansion_M2L(const spherical_expansion<T, P>& M, T x, T y, T z) {
	const auto O = spherical_singular_harmonic<T, P>(x, y, z);
	spherical_expansion<T, P> L;
	for (int n = 0; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			L[index(n, m)] = complex<T>(T(0), T(0));
			for (int k = 0; k <= P - n; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					L[index(n, m)] += M(k, l).conj() * O(n + k, m + l);
				}
			}
		}
	}
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

void random_unit(float& x, float& y, float& z) {
	const float theta = acos(2 * rand1() - 1.0);
	const float phi = rand1() * 2.0 * M_PI;
	x = cos(phi) * sin(theta);
	y = sin(phi) * sin(theta);
	z = cos(theta);
}

void random_vector(float& x, float& y, float& z) {
	do {
		x = 2 * rand1() - 1;
		y = 2 * rand1() - 1;
		z = 2 * rand1() - 1;
	} while (sqr(x, y, z) > 1);
}

template<int P>
float test_M2L(float theta, int N = 10000) {
	float err = 0.0;
	for (int i = 0; i < N; i++) {
		float x0, x1, x2, y0, y1, y2, z0, z1, z2;
		random_vector(x0, y0, z0);
		random_unit(x1, y1, z1);
		random_vector(x2, y2, z2);
		x1 /= 0.5 * theta;
		y1 /= 0.5 * theta;
		z1 /= 0.5 * theta;
		auto M = spherical_regular_harmonic<float, P>(x0, y0, z0);
		auto L = spherical_expansion_M2L<float, P>(M, x1, y1, z1);
		spherical_expansion_L2L(L, x2, y2, z2);
		const float dx = (x2 + x1) - x0;
		const float dy = (y2 + y1) - y0;
		const float dz = (z2 + z1) - z0;
		const float r = sqrt(sqr(dx, dy, dz));
		const float ap = 1.0 / r;
		const float ax = -dx / (r * r * r);
		const float ay = -dy / (r * r * r);
		const float az = -dz / (r * r * r);
		const float np = L(0, 0).real();
		const float nx = -L(1, 1).real();
		const float ny = -L(1, 1).imag();
		const float nz = -L(1, 0).real();
		const float ag = sqrt(sqr(ax, ay, az));
		const float ng = sqrt(sqr(nx, ny, nz));
		err += sqr((ag - ng) / ag);

	}
	err = sqrt(err / N);
	return err;
}

int main() {
	for (float theta = 0.2; theta < 0.95; theta += 0.05) {
		print("%e | ", theta);
		print(" %e", test_M2L<2>(theta));
		print(" %e", test_M2L<3>(theta));
		print(" %e", test_M2L<4>(theta));
		print(" %e", test_M2L<5>(theta));
		print(" %e", test_M2L<6>(theta));
		print(" %e", test_M2L<7>(theta));
		print(" %e", test_M2L<8>(theta));
		print(" %e", test_M2L<9>(theta));
		print(" %e", test_M2L<10>(theta));
		print(" %e", test_M2L<11>(theta));
		print(" %e", test_M2L<12>(theta));
		print(" %e", test_M2L<13>(theta));
		print(" %e", test_M2L<14>(theta));
		print(" %e", test_M2L<15>(theta));
		print(" %e", test_M2L<16>(theta));
		print("\n");
	}
}
