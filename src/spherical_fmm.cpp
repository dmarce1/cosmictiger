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

constexpr int lcd_run(int number, int n, int lcd) {
	return n == 1 ? lcd : lcd_run(number, n - 1, number % n == 0 ? n : lcd);
}

constexpr int lcd(int number) {
	return lcd_run(number, number, number);
}

constexpr double twiddle_toler = 1e-10;

constexpr float const_sin_generate(float x, int n, float coeff, float xpow) {
	return (
			(coeff * xpow < twiddle_toler && coeff * xpow > -twiddle_toler) ?
					coeff * xpow : coeff * xpow + const_sin_generate(x, n + 2, coeff / ((n + 1) * (n + 2)), -x * xpow * x));
}

constexpr float const_sin(float phi) {
	return phi > M_PI ? const_sin(phi - 2.0 * M_PI) : (phi < -M_PI ? const_sin(phi + 2.0 * M_PI) : const_sin_generate(phi, 1, 1.0, phi));
}
constexpr float const_cos_generate(float x, int n, float coeff, float xpow) {
	return (
			(coeff * xpow < twiddle_toler && coeff * xpow > -twiddle_toler) ?
					coeff * xpow : coeff * xpow + const_cos_generate(x, n + 2, coeff / ((n + 1) * (n + 2)), -x * xpow * x));
}

constexpr float const_cos(float phi) {
	return phi > M_PI ? const_cos(phi - 2.0 * M_PI) : (phi < -M_PI ? const_cos(phi + 2.0 * M_PI) : const_cos_generate(phi, 0, 1.0, 1.0));
}

template<class T, int N, int radix, int n, int m, int l, int sgn, bool term>
struct apply_twiddles {
	static constexpr int nops = apply_twiddles<T, N, radix, n, m + 1, l, sgn, m + 1 == radix>::nops + 8;
	void operator()(complex<T>& F, const complex<T>* F0) const {
		constexpr T phi = (sgn < 0 ? -1.0 : 1.0) * 2.0 * M_PI * (((n + l * N / radix) * m) % N) / N;
		constexpr T s = const_sin(phi);
		constexpr T c = const_cos(phi);
		constexpr complex<T> Tw(c, s);
		F += F0[m] * Tw;
		apply_twiddles<T, N, radix, n, m + 1, l, sgn, m + 1 == radix> f;
		f(F, F0);
	}
};

template<class T, int N, int radix, int n, int m, int l, int sgn>
struct apply_twiddles<T, N, radix, n, m, l, sgn, true> {
	static constexpr int nops = 0;
	void operator()(complex<T>& F, const complex<T>* F0) const {
	}
};

template<class T, int N, int radix, int n, int l, int s, int sgn, bool term>
struct loop_dfts {
	static constexpr int nops = apply_twiddles<T, N, radix, n, 1, l, sgn, false>::nops + loop_dfts<T, N, radix, n, l + 1, s, sgn, l + 1 == radix>::nops;
	void operator()(complex<T>* F, const complex<T>* F0) {
		auto f = F0[0];
		apply_twiddles<T, N, radix, n, 1, l, sgn, false> tw;
		tw(f, F0);
		F[s * (n + l * N / radix)] = f;
		loop_dfts<T, N, radix, n, l + 1, s, sgn, l + 1 == radix> loop;
		loop(F, F0);
	}
};

template<class T, int N, int radix, int n, int l, int s, int sgn>
struct loop_dfts<T, N, radix, n, l, s, sgn, true> {
	static constexpr int nops = 0;
	void operator()(complex<T>* F, const complex<T>* F0) {
	}
};

template<class T, int N, int radix, int n, int s, int sgn, bool term>
struct fft_helper {
	static constexpr int nops = loop_dfts<T, N, radix, n, 0, s, sgn, false>::nops + fft_helper<T, N, radix, n + 1, s, sgn, n + 1 == N / radix>::nops;
	void operator()(complex<T>* F, const complex<T>* F0) const {
		loop_dfts<T, N, radix, n, 0, s, sgn, false> loop;
		loop(F, F0 + n * radix);
		fft_helper<T, N, radix, n + 1, s, sgn, n + 1 == N / radix> next;
		next(F, F0);
	}
};

template<class T, int N, int radix, int n, int s, int sgn>
struct fft_helper<T, N, radix, n, s, sgn, true> {
	static constexpr int nops = 0;
	void operator()(complex<T>* F, const complex<T>* F0) const {
	}
};

template<class T, int N, int sgn, int S = 1, int radix = N % 4 == 0 ? 4 : lcd(N)>
struct ctfft_run {
	static constexpr int nops = radix * ctfft_run<T, N / radix, sgn, radix * S, lcd(N / radix)>::nops + fft_helper<T, N, radix, 0, S, sgn, false>::nops;
	void operator()(complex<T>* F) {
		for (int n = 0; n < radix; n++) {
			ctfft_run<T, N / radix, sgn, radix * S, lcd(N / radix)> func;
			func(F + n * S);
		}
		complex<T> F0[N];
		for (int n = 0; n < N; n++) {
			F0[n] = F[n * S];
		}
		fft_helper<T, N, radix, 0, S, sgn, false> run;
		run(F, &F0[0]);
	}
};

template<class T, int N, int sgn, int S, int k, bool term>
struct radix4_help {
	static constexpr int nops = radix4_help<T, N, sgn, S, k + 1, k + 1 == N / 4>::nops + 28;
	void operator()(complex<T>* F, complex<T>* F0) {
		constexpr complex<T> I(0, sgn);
		const auto u1 = F0[2 * k];
		const auto u2 = F0[2 * k + N / 2];
		const auto tw10 = expc(complex<T>(0, sgn * 2.0 * M_PI * (k) / N));
		const auto tw30 = expc(complex<T>(0, 3 * sgn * 2.0 * M_PI * (k) / N));
		const auto t1 = F0[4 * k + 1] * tw10; // 6
		const auto t2 = F0[4 * k + 3] * tw30; // 6
		const auto sum = t1 + t2; // 2
		const auto dif = t1 - t2; // 2
		F[k * S] = u1 + sum; // 2
		F[k * S + (N / 2) * S] = u1 - sum; // 2
		F[k * S + (N / 4) * S] = u2 + I * dif; // 4
		F[k * S + 3 * (N / 4) * S] = u2 - I * dif; // 4
		radix4_help<T, N, sgn, S, k + 1, k + 1 == N / 4> next;
		next(F, F0);
	}
};

template<class T, int sgn, int S>
struct radix4_help<T, 8, sgn, S, 0, false> {
	static constexpr int nops = radix4_help<T, 8, sgn, S, 1, false>::nops + 16;
	void operator()(complex<T>* F, complex<T>* F0) {
		constexpr complex<T> I(0, sgn);
		const auto u1 = F0[0];
		const auto u2 = F0[4];
		const auto t1 = F0[1]; // 6
		const auto t2 = F0[3]; // 6
		const auto sum = t1 + t2; // 2
		const auto dif = t1 - t2; // 2
		F[0] = u1 + sum; // 2
		F[4 * S] = u1 - sum; // 2
		F[2 * S] = u2 + I * dif; // 4
		F[6 * S] = u2 - I * dif; // 4
		radix4_help<T, 8, sgn, S, 1, false> next;
		next(F, F0);
	}
};

template<class T, int N, int sgn, int S, int k>
struct radix4_help<T, N, sgn, S, k, true> {
	static constexpr int nops = 0;
	void operator()(complex<T>* F, complex<T>* F0) {
	}
};

template<class T, int N, int sgn, int S>
struct ctfft_run<T, N, sgn, S, 4> {
	static constexpr int nops = ctfft_run<T, N / 2, sgn, 2 * S>::nops + 2 * ctfft_run<T, N / 4, sgn, 4 * S>::nops + radix4_help<T, N, sgn, S, 0, false>::nops;
	void operator()(complex<T>* F) {
		ctfft_run<T, N / 2, sgn, 2 * S> bigfft;
		ctfft_run<T, N / 4, sgn, 4 * S> smallfft;
		bigfft(F);
		smallfft(F + S);
		smallfft(F + 3 * S);
		array<complex<T>, N> F0;
		for (int i = 0; i < N; i++) {
			F0[i] = F[S * i];
		}
		radix4_help<T, N, sgn, S, 0, false> run;
		run(F, F0.data());
	}
};

template<class T, int sgn, int S>
struct ctfft_run<T, 2, sgn, S, 2> {
	static constexpr int nops = 4;
	void operator()(complex<T>* F) {
		const auto dif = F[0] - F[S];
		F[0] += F[S];
		F[S] = dif;
	}
};

template<class T, int sgn, int S>
struct ctfft_run<T, 4, sgn, S, 2> {
	static constexpr int nops = 16;
	void operator()(complex<T>* F) {
		constexpr complex<T> I(T(0), T(-sgn));
		auto dif = F[0] - F[2 * S];
		F[0] += F[2 * S];
		F[2 * S] = dif;
		const auto sum = F[S] + F[3 * S];
		dif = F[S] - F[3 * S];
		auto u1 = F[0];
		auto u2 = F[2 * S];
		F[0] = u1 + sum;
		F[2 * S] = u1 - sum;
		F[S].real() = u2.real() - sgn * dif.imag();
		F[S].imag() = u2.imag() + sgn * dif.real();
		F[3 * S].real() = u2.real() + sgn * dif.imag();
		F[3 * S].imag() = u2.imag() - sgn * dif.real();
	}
};

template<class T, int N, int sgn, int S>
struct ctfft_run<T, N, sgn, S, 1> {
	static constexpr int nops = 0;
	void operator()(complex<T>* F) {

	}
};

template<class T, int sgn, int S>
struct ctfft_run<T, 1, sgn, S, 4> {
	void operator()(complex<T>* F) {

	}
};

template<class T, int sgn, int S>
struct ctfft_run<T, 2, sgn, S, 4> {
	static constexpr int nops = 4;
	void operator()(complex<T>* F) {
		const auto dif = F[0] - F[S];
		F[0] += F[S];
		F[S] = dif;
	}
};

template<class T, int N>
int ctfft(complex<T>* F, bool inv = false) {
	int nops = 0;
	if (inv) {
		const T ninv = T(1) / N;
		for (int i = 0; i < N; i++) {
			F[i] *= ninv;
		}
		nops += N;
	}
	if (inv) {
		nops += ctfft_run<T, N, 1, 1>::nops;
		ctfft_run<T, N, 1, 1> f;
		f(F);
	} else {
		nops += ctfft_run<T, N, 1, -1>::nops;
		ctfft_run<T, N, -1, 1> f;
		f(F);

	}
	return nops;
}

template<class T, int N, bool INV, int STRIDE = 1, int OFFSET = 0, bool NORMALIZED = false>
struct FFT {
	int operator()(array<complex<T>, STRIDE * N>& F) {
		return ctfft<T, N>(F.data(), INV);
	}
};

template<class T, bool INV, int STRIDE, int OFFSET, bool NORMALIZED>
struct FFT<T, 1, INV, STRIDE, OFFSET, NORMALIZED> {
	int operator()(array<complex<T>, STRIDE>& F) {
		return 0;
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
void FFT_c2r_symmetric(array<complex<T>, N / 2 + 1>& A, array<complex<T>, N / 2 + 1>& B, array<real, N>& A0, array<real, N>& B0) {
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
int FFT_spherical_expansion(array<complex<T>, N / 2>& A, array<complex<T>, N / 2>& B) {
	int nops;
	array<complex<T>, N> Z;
	constexpr auto I = complex<T>(T(0), T(1));
	nops += 2;
	Z[0] = A[0] + I * B[0];
	for (int n = 1; n < N / 2; n++) {
		nops += 4;
		Z[n] = A[n] + I * B[n];
		Z[N - n] = A[n].conj() * nonepow<T>(n) + I * (B[n].conj() * nonepow<T>(n));
	}
	Z[N / 2] = 0.0;
	FFT<T, N, false> fft;
	nops += fft(Z);
	for (int n = 0; n < N / 2; n++) {
		const auto even = T(0.5) * (Z[n] + Z[n + N / 2]);
		const auto odd = T(0.5) * (Z[n] - Z[n + N / 2]);
		A[n].real() = even.real();
		A[n].imag() = odd.imag();
		B[n].imag() = even.imag();
		B[n].real() = odd.real();
		B[n] /= I;
		nops += 9;
	}
	return nops;
}

template<class T, int N>
int FFT_spherical_expansion_inv(array<complex<T>, N / 2>& A, array<complex<T>, N / 2>& B) {
	int nops;
	array<complex<T>, N> Z;
	const auto I = complex<T>(T(0), T(1));
	for (int n = 0; n < N / 2; n++) {
		Z[n] = A[n] + I * B[n];
		Z[n + N / 2] = A[n].conj() + I * (B[n].conj());
		nops += 4;
	}
//	Z[N / 2] = 0.0;
	FFT<T, N, true> fft_inv;
	nops += fft_inv(Z);
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
		nops += 9;
	}
	return nops;
}

template<class T, int P>
spherical_expansion<T, P> fourier_M2L(const spherical_expansion<T, P - 1>& Mx, T x, T y, T z) {
	int nops = 0;
	const auto Gx = spherical_singular_harmonic<T, P>(x, y, z);
	spherical_expansion<T, P> Lx;
	static constexpr int N = (P + 1);
	array<array<complex<T>, N>, 2 * N> Gk;
	array<array<complex<T>, N>, 2 * N> Mk;
	array<array<complex<T>, N>, 2 * N> Lk;
	for (int n = 0; n < 2 * N; n++) {
		for (int m = 0; m < N; m++) {
			Gk[n][m] = Mk[n][m] = complex<T>(T(0), T(0));
		}
	}
	int lnnn = -1;
	int nnn;
	for (int n = 0; n < N; n++) {
		lnnn = nnn;
		nnn = (2 * N - n) % (2 * N);
		for (int m = 0; m <= n; m++) {
			Gk[nnn][m] = Gx(n, -m);
			nops++;
		}
		if (n % 2 == 1) {
			nops += FFT_spherical_expansion<T, 2 * N>(Gk[lnnn], Gk[nnn]);
		} else if (n == N - 1) {
			nops += FFT_spherical_expansion<T, 2 * N>(Gk[nnn], Gk[1]);
		}
	}
	for (int n = 0; n < P; n++) {
		for (int m = 0; m <= n; m++) {
			Mk[n][m] = Mx(n, m).conj();
			nops++;
		}
		if (n % 2 == 1) {
			nops += FFT_spherical_expansion<T, 2 * N>(Mk[n], Mk[n - 1]);
		} else if (n == P - 1) {
			nops += FFT_spherical_expansion<T, 2 * N>(Mk[n], Mk[n + 1]);
		}
	}
	FFT<T, 2 * N, false> fft;
	FFT<T, 2 * N, true> fft_inv;
	for (int m = 0; m < N; m++) {
		array<complex<T>, 2 * N> col;
		for (int n = 0; n < 2 * N; n++) {
			col[n] = Gk[n][m];
		}
		nops += fft(col);
		for (int n = 0; n < 2 * N; n++) {
			Gk[n][m] = col[n];
		}
		for (int n = 0; n < 2 * N; n++) {
			col[n] = Mk[n][m];
		}
		nops += fft(col);
		for (int n = 0; n < 2 * N; n++) {
			Mk[n][m] = col[n];
		}
	}
	for (int m = 0; m < N; m++) {
		array<complex<T>, 2 * N> col;
		for (int n = 0; n < 2 * N; n++) {
			col[n] = Gk[n][m] * Mk[n][m];
			nops++;
		}
		fft_inv(col);
		for (int n = 0; n < 2 * N; n++) {
			Lk[n][m] = col[n];
		}
	}
	for (int n = (N / 2) * 2; n < 2 * N; n += 2) {
		FFT_spherical_expansion_inv<T, 2 * N>(Lk[n], Lk[n + 1]);
	}

	for (int n = 0; n < N; n++) {
		for (int m = 0; m <= n; m++) {
			Lx[index(n, m)] = Lk[(2 * N - n) % (2 * N)][m].conj() * nonepow<T>(m);
		}
	}
//	printf( "%i\n", nops);
	return Lx;

}

template<int P>
real test_M2L(real theta = 0.5) {
	real err = 0.0;
	int N = 10000;
	timer tm1, tm2;
	for (int i = 0; i < N; i++) {
		real x0, x1, x2, y0, y1, y2, z0, z1, z2;
		random_vector(x0, y0, z0);
		random_unit(x1, y1, z1);
		random_vector(x2, y2, z2);
		x1 /= 0.5 * theta;
		y1 /= 0.5 * theta;
		z1 /= 0.5 * theta;
		auto M = spherical_regular_harmonic<real, P - 1>(x0, y0, z0);
		tm1.start();
		auto L = spherical_expansion_M2L<real, P>(M, x1, y1, z1);
		tm1.stop();
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

	}
	err = 0.0;
	for (int i = 0; i < N; i++) {
		real x0, x1, x2, y0, y1, y2, z0, z1, z2;
		random_vector(x0, y0, z0);
		random_unit(x1, y1, z1);
		random_vector(x2, y2, z2);
		x1 /= 0.5 * theta;
		y1 /= 0.5 * theta;
		z1 /= 0.5 * theta;
		auto M = spherical_regular_harmonic<real, P - 1>(x0, y0, z0);
		tm2.start();
		auto L = fourier_M2L<real, P>(M, x1, y1, z1);
		tm2.stop();
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
	err = sqrt(err / N);
	PRINT("%i %e %e %e\n", P, err, tm1.read(), tm2.read());
	return err;
}

int main() {

	test_M2L<7>();
	/*
	 constexpr int N = 10;
	 vector<complex<real>> A(N);
	 for (int i = 0; i < N; i++) {
	 A[i].real() = cos(2.0 * M_PI * (i - N / 2) / N);
	 A[i].imag() = 0.0;
	 }
	 FFT<real, N, false> test;
	 ctfft<real, N>(A);
	 for (int i = 0; i < N; i++) {
	 print("%e %e\n", A[i].real(), A[i].imag());
	 }*/

}
