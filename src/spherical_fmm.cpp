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
spherical_expansion<T, P> spherical_expansion_M2L(const spherical_expansion<T, P - 1>& M, T x, T y, T z) {
	const auto O = spherical_singular_harmonic<T, P>(x, y, z);
	spherical_expansion<T, P> L;
	for (int n = 0; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			L[index(n, m)] = complex<T>(T(0), T(0));
			const int kmax = std::min(P - n, P - 1);
			for (int k = 0; k <= kmax; k++) {
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
		auto M = spherical_regular_harmonic<float, P - 1>(x0, y0, z0);
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

template<class T, int N, bool INV, int STRIDE = 1, int OFFSET = 0, bool NORMALIZED = false>
struct FFT {
	void operator()(array<complex<T>, STRIDE * N>& F) {
		const auto& index = [](int n) {
			return n * STRIDE + OFFSET;
		};
		if (!NORMALIZED && INV) {
			for (int n = 0; n < N; n++) {
				F[n] /= N;
			}
		}
		FFT<T, N / 2, INV, 2 * STRIDE, OFFSET, true> fft_even;
		FFT<T, N / 2, INV, 2 * STRIDE, OFFSET + STRIDE, true> fft_odd;
		fft_even(F);
		fft_odd(F);
		const auto F0 = F;
		const float sgn = INV ? T(1.0) : T(-1.0);
		for (int i = 0; i < N / 2; i++) {
			const auto R = F0[index(2 * i + 1)] * complex<T>(cos(2.0 * M_PI * i / N), sgn * sin(2.0 * M_PI * i / N));
			F[index(i)] = F0[index(2 * i)] + R;
			F[index(i + N / 2)] = F0[index(2 * i)] - R;
		}
	}
};

template<class T, bool INV, int STRIDE, int OFFSET, bool NORMALIZED>
struct FFT<T, 1, INV, STRIDE, OFFSET, NORMALIZED> {
	void operator()(array<complex<T>, STRIDE>& F) {
	}
};

template<class T, int N, int M, bool INV>
struct FFT2 {
	void operator()(array<array<complex<T>, M>, N>& F) {
		array<array<complex<T>, N>, M> G;
		FFT<T, M, INV> fftn;
		FFT<T, N, INV> fftm;
		for (int n = 0; n < N; n++) {
			fftn(F[n]);
		}
		for (int n = 0; n < N; n++) {
			for (int m = 0; m < M; m++) {
				G[m][n] = F[n][m];
			}
		}
		for (int m = 0; m < M; m++) {
			fftm(G[m]);
		}
		for (int n = 0; n < N; n++) {
			for (int m = 0; m < M; m++) {
				F[n][m] = G[m][n];
			}
		}
	}
};

template<class T, int N>
void FFT_c2r_symmetric(array<complex<T>, N / 2 + 1>& A, array<complex<T>, N / 2 + 1>& B, array<float, N>& A0, array<float, N>& B0) {
	array<complex<T>, N> C;
	complex<T> I(T(0), T(1));
	for (int n = 0; n <= N / 2; n++) {
		C[n] = A[n] + I * B[n];
	}
	for (int n = -1; n > -N / 2; n--) {
		C[N + n] = (A[-n].conj() + I * (B[-n].conj()));
	}
	FFT<T, N, false> fft;
	fft(C);
	for (int n = 0; n < N; n++) {
		A0[n] = C[n].real();
		B0[n] = C[n].imag();
	}
}

template<class T, int N>
void FFT_spherical_expansion(array<complex<T>, N / 2>& A, array<complex<T>, N / 2>& B) {
	array<complex<T>, N> Z;
	const auto I = complex<T>(T(0), T(1));
	Z[0] = A[0] + I * B[0];
	for (int n = 1; n < N / 2; n++) {
		Z[n] = A[n] + I * B[n];
		Z[N - n] = A[n].conj() * nonepow<T>(n) + I * (B[n].conj() * nonepow<T>(n));
	}
	Z[N / 2] = 0.0;
	FFT<T, N, false> fft;
	fft(Z);
	for (int n = 0; n < N / 2; n++) {
		const auto even = T(0.5) * (Z[n] + Z[n + N / 2]);
		const auto odd = T(0.5) * (Z[n] - Z[n + N / 2]);
		A[n].real() = even.real();
		A[n].imag() = odd.imag();
		B[n].imag() = even.imag();
		B[n].real() = odd.real();
		B[n] /= I;
	}
}

template<class T, int N>
void FFT_spherical_expansion_inv(array<complex<T>, N / 2>& A, array<complex<T>, N / 2>& B) {
	array<complex<T>, N> Z;
	const auto I = complex<T>(T(0), T(1));
	for (int n = 0; n < N / 2; n++) {
		Z[n] = A[n] + I * B[n];
		Z[n + N / 2] = A[n].conj() + I * (B[n].conj());
	}
//	Z[N / 2] = 0.0;
	FFT<T, N, true> fft_inv;
	fft_inv(Z);
	for (int n = 0; n < N / 2; n++) {
		const auto sym = T(0.5) * (Z[n] + Z[(N - n) % (N)].conj());
		const auto ant = T(0.5) * (Z[n] - Z[(N - n) % (N)].conj());
		if (n % 2 == 0) {
			A[n] = sym;
			B[n] = ant / I;
		} else {
			B[n] = sym / I;
			A[n] = ant;
		}
	}
}

template<class T, int P>
spherical_expansion<T, P> fourier_M2L(const spherical_expansion<T, P - 1>& Mx, T x, T y, T z) {
	const auto Gx = spherical_singular_harmonic<T, P>(x, y, z);
	spherical_expansion<T, P> Lx;
	constexpr int N = (P + 1);
	array<array<complex<T>, N>, 2 * N> Gk;
	array<array<complex<T>, N>, 2 * N> Mk;
	array<array<complex<T>, N>, 2 * N> Lk;
	for (int n = 0; n < 2 * N; n++) {
		for (int m = 0; m < N; m++) {
			Gk[n][m] = Mk[n][m] = complex<T>(T(0), T(0));
		}
	}
	for (int n = 0; n < N; n++) {
		const int nnn = (2 * N - n) % (2 * N);
		for (int m = 0; m <= n; m++) {
			Gk[nnn][m] = Gx(n, -m);
		}
	}
	for (int n = 0; n < P; n++) {
		for (int m = 0; m <= n; m++) {
			Mk[n][m] = Mx(n, m).conj();
		}
	}
	for (int n = 0; n < 2 * N; n++) {
		FFT_spherical_expansion<T, 2 * N>(Gk[n], Mk[n]);
	}
	FFT<T, 2 * N, false> fft;
	FFT<T, 2 * N, true> fft_inv;
	for (int m = 0; m < N; m++) {
		array<complex<T>, 2 * N> col;
		for (int n = 0; n < 2 * N; n++) {
			col[n] = Gk[n][m];
		}
		fft(col);
		for (int n = 0; n < 2 * N; n++) {
			Gk[n][m] = col[n];
		}
		for (int n = 0; n < 2 * N; n++) {
			col[n] = Mk[n][m];
		}
		fft(col);
		for (int n = 0; n < 2 * N; n++) {
			Mk[n][m] = col[n];
		}
	}
	for (int m = 0; m < N; m++) {
		array<complex<T>, 2 * N> col;
		for (int n = 0; n < 2 * N; n++) {
			col[n] = Gk[n][m] * Mk[n][m];
		}
		fft_inv(col);
		for (int n = 0; n < 2 * N; n++) {
			Lk[n][m] = col[n];
		}
	}
	for (int n = 0; n < 2 * N; n += 2) {
		FFT_spherical_expansion_inv<T, 2 * N>(Lk[n], Lk[n + 1]);
	}
	for (int n = 0; n < 2 * N; n++) {
		for (int m = 0; m < N; m++) {
			print("%e + i%e   ", Lk[n][m].real(), Lk[n][m].imag());
		}
		printf("\n");
	}
	for (int n = 0; n < N; n++) {
		for (int m = 0; m <= n; m++) {
			Lx[index(n, m)] = Lk[(2 * N - n) % (2 * N)][m].conj() * nonepow<T>(m);
		}
	}
	return Lx;
}

int main() {

	constexpr int N = 8;
	array<complex<float>, N / 2> A;
	array<complex<float>, N / 2> B;
	array<complex<float>, N> A1;
	array<complex<float>, N> B1;
	A[0] = rand1();
	B[0] = rand1();
	A1[0] = A[0];
	B1[0] = B[0];
	A1[N / 2] = A[0];
	B1[N / 2] = B[0];
	for (int n = 1; n < N / 2; n++) {
		A[n] = complex<float>(rand1(), rand1());
		B[n] = complex<float>(rand1(), rand1());
		A1[n] = A[n];
		A1[(n + N / 2) % (N)] = A[n].conj();
		B1[n] = B[n];
		B1[(n + N / 2) % (N)] = B[n].conj();
	}
	FFT<float, N, true> fft_inv;
	FFT_spherical_expansion_inv<float, N>(A, B);
	fft_inv(A1);
	fft_inv(B1);
	for (int i = 0; i < N / 2; i++) {
		print("%e %e %e %e \n", A1[i].real(), A1[i].imag(), A[i].real(), A[i].imag());
	}
	for (int i = 0; i < N / 2; i++) {
		print("%e %e %e %e \n", B1[i].real(), B1[i].imag(), B[i].real(), B[i].imag());
	}

	constexpr int P = 3;
	float theta = 0.7;
	float x0, x1, x2, y0, y1, y2, z0, z1, z2;
	random_vector(x0, y0, z0);
	random_unit(x1, y1, z1);
	random_vector(x2, y2, z2);
	x1 /= 0.5 * theta;
	y1 /= 0.5 * theta;
	z1 /= 0.5 * theta;
	auto M = spherical_regular_harmonic<float, P - 1>(x0, y0, z0);
	auto L0 = spherical_expansion_M2L<float, P>(M, x1, y1, z1);
	auto L1 = fourier_M2L<float, P>(M, x1, y1, z1);
	printf("\n");
	L0.print();
	printf("\n");
	L1.print();
	double err = 0.0;
	for (int l = 0; l <= P; l++) {
		for (int m = 0; m <= l; m++) {
			err += (L1(l, m) - L0(l, m)).norm();
		}
	}
	err = sqrt(err / ((P + 1) * (P + 1)));
	PRINT("err = %e\n", err);

}
