#pragma once
#include <tigerfmm/tensor.hpp>
#include <tigerfmm/cuda.hpp>
#include <tigerfmm/ewald_indices.hpp>
#include <tigerfmm/math.hpp>
template<class T>
using expansion = tensor_trless_sym<T,4>;
template<class T>
using expansion2 = tensor_trless_sym<T,2>;
template<class T>
using multipole = tensor_trless_sym<T,3>;
#define EXPANSION_SIZE 17
#define MULTIPOLE_SIZE 10


template<class T>
CUDA_EXPORT
inline int greens_function(tensor_trless_sym<T, 4>& D, array<T, NDIM> X) {
	auto r2 = sqr(X[0], X[1], X[2]);
	r2 = sqr(X[0], X[1], X[2]);
	const T r = sqrt(r2);
	const T rinv1 = -(r > T(0)) / max(r, T(1e-20));
	const T rinv2 = -rinv1 * rinv1;
	const T rinv3 = -rinv2 * rinv1;
	const T rinv4 = -rinv2 * rinv2;
	X[0] *= rinv1;
	X[1] *= rinv1;
	X[2] *= rinv1;
	const T x000 = T(1);
	const T& x100 = X[0];
	const T& x010 = X[1];
	const T& x001 = X[2];
	const T x002 = x001 * x001;
	const T x011 = x010 * x001;
	const T x020 = x010 * x010;
	const T x101 = x100 * x001;
	const T x110 = x100 * x010;
	const T x200 = x100 * x100;
	const T x003 = x002 * x001;
	const T x012 = x011 * x001;
	const T x021 = x011 * x010;
	const T x030 = x020 * x010;
	const T x102 = x101 * x001;
	const T x111 = x110 * x001;
	const T x120 = x110 * x010;
	const T x201 = x101 * x100;
	const T x210 = x110 * x100;
	const T x300 = x200 * x100;
	T& D000 = D[0];
	T& D001 = D[10];
	T& D002 = D[16];
	T& D010 = D[2];
	T& D011 = D[12];
	T& D020 = D[5];
	T& D021 = D[15];
	T& D030 = D[9];
	T& D100 = D[1];
	T& D101 = D[11];
	T& D110 = D[4];
	T& D111 = D[14];
	T& D120 = D[8];
	T& D200 = D[3];
	T& D201 = D[13];
	T& D210 = D[7];
	T& D300 = D[6];
	T x_2_1_000 = x002;
	T x_3_1_001 = x003;
	T x_3_1_010 = x012;
	T x_3_1_100 = x102;
	x_2_1_000 += x020;
	x_3_1_010 += x030;
	x_2_1_000 += x200;
	x_3_1_010 += x210;
	x_3_1_001 += x021;
	x_3_1_100 += x120;
	x_3_1_001 += x201;
	x_3_1_100 += x300;
	D000 = x000;
	D001 = x001;
	D002 = T(3.00000000e+00) * x002;
	D010 = x010;
	D011 = T(3.00000000e+00) * x011;
	D020 = T(3.00000000e+00) * x020;
	D021 = T(1.50000000e+01) * x021;
	D030 = T(1.50000000e+01) * x030;
	D100 = x100;
	D101 = T(3.00000000e+00) * x101;
	D110 = T(3.00000000e+00) * x110;
	D111 = T(1.50000000e+01) * x111;
	D120 = T(1.50000000e+01) * x120;
	D200 = T(3.00000000e+00) * x200;
	D201 = T(1.50000000e+01) * x201;
	D210 = T(1.50000000e+01) * x210;
	D300 = T(1.50000000e+01) * x300;
	D002 -= x_2_1_000;
	D200 -= x_2_1_000;
	D020 -= x_2_1_000;
	D201 = fmaf(T(-3.00000000e+00), x_3_1_001, D201);
	D021 = fmaf(T(-3.00000000e+00), x_3_1_001, D021);
	D210 = fmaf(T(-3.00000000e+00), x_3_1_010, D210);
	D030 = fmaf(T(-9.00000000e+00), x_3_1_010, D030);
	D300 = fmaf(T(-9.00000000e+00), x_3_1_100, D300);
	D120 = fmaf(T(-3.00000000e+00), x_3_1_100, D120);
	D000 *= rinv1;
	D001 *= rinv2;
	D002 *= rinv3;
	D010 *= rinv2;
	D011 *= rinv3;
	D020 *= rinv3;
	D021 *= rinv4;
	D030 *= rinv4;
	D100 *= rinv2;
	D101 *= rinv3;
	D110 *= rinv3;
	D111 *= rinv4;
	D120 *= rinv4;
	D200 *= rinv3;
	D201 *= rinv4;
	D210 *= rinv4;
	D300 *= rinv4;
	return 98;
}
template<class T>
CUDA_EXPORT int ewald_greens_function(tensor_trless_sym<T,4> &D, array<T, NDIM> X) {
	ewald_const econst;
	int flops = 7;
	T r = sqrt(fmaf(X[0], X[0], fmaf(X[1], X[1], sqr(X[2]))));
	const T fouroversqrtpi = T(2.25675833e+00);
	tensor_sym<T, 4> Dreal;
	tensor_trless_sym<T,4> Dfour;
	Dreal = 0.0f;
	Dfour = 0.0f;
	D = 0.0f;
	const auto realsz = econst.nreal();
	const T zero_mask = r > T(0);
	int icnt = 0;
	for (int i = 0; i < realsz; i++) {
		const auto n = econst.real_index(i);
		array<T, NDIM> dx;
		for (int dim = 0; dim < NDIM; dim++) {
			dx[dim] = X[dim] - n[dim];
		}
		T r2 = fmaf(dx[0], dx[0], fmaf(dx[1], dx[1], sqr(dx[2])));
		if (anytrue(r2 < (EWALD_REAL_CUTOFF2))) {
			icnt++;
			const T r = sqrt(r2);
			const T n8r = T(-8) * r;
			const T rinv = (r > T(0)) / max(r, 1.0e-20);
			T exp0;
			T erfc0;
			erfcexp(2.f * r, &erfc0, &exp0);
			const T expfactor = fouroversqrtpi * exp0;
			T e0 = expfactor * rinv;
			const T rinv0 = T(1);
			const T rinv1 = rinv;
			const T d0 = -erfc0 * rinv;
			const T d1 = fmaf(T(-1) * d0, rinv, e0);
			e0 *= n8r;
			const T d2 = fmaf(T(-3) * d1, rinv, e0);
			e0 *= n8r;
			const T d3 = fmaf(T(-5) * d2, rinv, e0);
			const T Drinvpow_0_0 = d0 * rinv0;
			const T Drinvpow_1_0 = d1 * rinv0;
			const T Drinvpow_1_1 = d1 * rinv1;
			const T Drinvpow_2_0 = d2 * rinv0;
			const T Drinvpow_2_1 = d2 * rinv1;
			const T Drinvpow_3_0 = d3 * rinv0;
			array<T,NDIM> dxrinv;
			dxrinv[0] = dx[0] * rinv;
			dxrinv[1] = dx[1] * rinv;
			dxrinv[2] = dx[2] * rinv;
			const T x000 = T(1);
			const T& x100 = dxrinv[0];
			const T& x010 = dxrinv[1];
			const T& x001 = dxrinv[2];
			const T x002 = x001 * x001;
			const T x011 = x010 * x001;
			const T x020 = x010 * x010;
			const T x101 = x100 * x001;
			const T x110 = x100 * x010;
			const T x200 = x100 * x100;
			const T x003 = x002 * x001;
			const T x012 = x011 * x001;
			const T x021 = x011 * x010;
			const T x030 = x020 * x010;
			const T x102 = x101 * x001;
			const T x111 = x110 * x001;
			const T x120 = x110 * x010;
			const T x201 = x101 * x100;
			const T x210 = x110 * x100;
			const T x300 = x200 * x100;
			T x_2_1_000 = x002;
			T x_3_1_001 = x003;
			T x_3_1_010 = x012;
			T x_3_1_100 = x102;
			x_2_1_000 += x020;
			x_3_1_010 += x030;
			x_2_1_000 += x200;
			x_3_1_010 += x210;
			x_3_1_001 += x021;
			x_3_1_100 += x120;
			x_3_1_001 += x201;
			x_3_1_100 += x300;
			x_2_1_000 *= Drinvpow_1_1;
			x_3_1_001 *= Drinvpow_2_1;
			x_3_1_010 *= Drinvpow_2_1;
			x_3_1_100 *= Drinvpow_2_1;
			Dreal[0] = fmaf(x000, Drinvpow_0_0, Dreal[0]);
			Dreal[11] = fmaf(x210, Drinvpow_3_0, Dreal[11]);
			Dreal[1] = fmaf(x100, Drinvpow_1_0, Dreal[1]);
			Dreal[12] += x_3_1_001;
			Dreal[2] = fmaf(x010, Drinvpow_1_0, Dreal[2]);
			Dreal[12] = fmaf(x201, Drinvpow_3_0, Dreal[12]);
			Dreal[3] = fmaf(x001, Drinvpow_1_0, Dreal[3]);
			Dreal[13] += x_3_1_100;
			Dreal[4] = fmaf(x200, Drinvpow_2_0, Dreal[4]);
			Dreal[13] = fmaf(x120, Drinvpow_3_0, Dreal[13]);
			Dreal[4] += x_2_1_000;
			Dreal[14] = fmaf(x111, Drinvpow_3_0, Dreal[14]);
			Dreal[5] = fmaf(x110, Drinvpow_2_0, Dreal[5]);
			Dreal[15] = fmaf(x102, Drinvpow_3_0, Dreal[15]);
			Dreal[6] = fmaf(x101, Drinvpow_2_0, Dreal[6]);
			Dreal[15] += x_3_1_100;
			Dreal[7] = fmaf(x020, Drinvpow_2_0, Dreal[7]);
			Dreal[16] = fmaf(T(3.00000000e+00), x_3_1_010, Dreal[16]);
			Dreal[7] += x_2_1_000;
			Dreal[16] = fmaf(x030, Drinvpow_3_0, Dreal[16]);
			Dreal[8] = fmaf(x011, Drinvpow_2_0, Dreal[8]);
			Dreal[17] += x_3_1_001;
			Dreal[9] += x_2_1_000;
			Dreal[17] = fmaf(x021, Drinvpow_3_0, Dreal[17]);
			Dreal[9] = fmaf(x002, Drinvpow_2_0, Dreal[9]);
			Dreal[18] += x_3_1_010;
			Dreal[10] = fmaf(T(3.00000000e+00), x_3_1_100, Dreal[10]);
			Dreal[18] = fmaf(x012, Drinvpow_3_0, Dreal[18]);
			Dreal[10] = fmaf(x300, Drinvpow_3_0, Dreal[10]);
			Dreal[19] = fmaf(x003, Drinvpow_3_0, Dreal[19]);
			Dreal[11] += x_3_1_010;
			Dreal[19] = fmaf(T(3.00000000e+00), x_3_1_001, Dreal[19]);
		}
	}
	flops += icnt * 157;
	const auto foursz = econst.nfour();
	for (int i = 0; i < foursz; i++) {
		const auto &h = econst.four_index(i);
		const auto& D0 = econst.four_expansion(i);
		const T hdotx = fmaf(h[0], X[0], fmaf(h[1], X[1], h[2] * X[2]));
		T cn, sn;
		sincos(T(2.0 * M_PI) * hdotx, &sn, &cn);
		Dfour[0] = fmaf(cn, D0[0], Dfour[0]);
		Dfour[9] = fmaf(sn, D0[9], Dfour[9]);
		Dfour[1] = fmaf(sn, D0[1], Dfour[1]);
		Dfour[10] = fmaf(sn, D0[10], Dfour[10]);
		Dfour[2] = fmaf(sn, D0[2], Dfour[2]);
		Dfour[11] = fmaf(cn, D0[11], Dfour[11]);
		Dfour[3] = fmaf(cn, D0[3], Dfour[3]);
		Dfour[12] = fmaf(cn, D0[12], Dfour[12]);
		Dfour[4] = fmaf(cn, D0[4], Dfour[4]);
		Dfour[13] = fmaf(sn, D0[13], Dfour[13]);
		Dfour[5] = fmaf(cn, D0[5], Dfour[5]);
		Dfour[14] = fmaf(sn, D0[14], Dfour[14]);
		Dfour[6] = fmaf(sn, D0[6], Dfour[6]);
		Dfour[15] = fmaf(sn, D0[15], Dfour[15]);
		Dfour[7] = fmaf(sn, D0[7], Dfour[7]);
		Dfour[16] = fmaf(cn, D0[16], Dfour[16]);
		Dfour[8] = fmaf(sn, D0[8], Dfour[8]);
	}
	const T& Dreal000 = Dreal[0];
	const T& Dreal001 = Dreal[3];
	const T& Dreal002 = Dreal[9];
	const T& Dreal003 = Dreal[19];
	const T& Dreal010 = Dreal[2];
	const T& Dreal011 = Dreal[8];
	const T& Dreal012 = Dreal[18];
	const T& Dreal020 = Dreal[7];
	const T& Dreal021 = Dreal[17];
	const T& Dreal030 = Dreal[16];
	const T& Dreal100 = Dreal[1];
	const T& Dreal101 = Dreal[6];
	const T& Dreal102 = Dreal[15];
	const T& Dreal110 = Dreal[5];
	const T& Dreal111 = Dreal[14];
	const T& Dreal120 = Dreal[13];
	const T& Dreal200 = Dreal[4];
	const T& Dreal201 = Dreal[12];
	const T& Dreal210 = Dreal[11];
	const T& Dreal300 = Dreal[10];
	T& D000 = D[0];
	T& D001 = D[10];
	T& D002 = D[16];
	T& D010 = D[2];
	T& D011 = D[12];
	T& D020 = D[5];
	T& D021 = D[15];
	T& D030 = D[9];
	T& D100 = D[1];
	T& D101 = D[11];
	T& D110 = D[4];
	T& D111 = D[14];
	T& D120 = D[8];
	T& D200 = D[3];
	T& D201 = D[13];
	T& D210 = D[7];
	T& D300 = D[6];
	T Dreal_2_1_000 = Dreal002;
	T Dreal_3_1_001 = Dreal003;
	T Dreal_3_1_010 = Dreal012;
	T Dreal_3_1_100 = Dreal102;
	Dreal_2_1_000 += Dreal020;
	Dreal_3_1_010 += Dreal030;
	Dreal_2_1_000 += Dreal200;
	Dreal_3_1_010 += Dreal210;
	Dreal_3_1_001 += Dreal021;
	Dreal_3_1_100 += Dreal120;
	Dreal_3_1_001 += Dreal201;
	Dreal_3_1_100 += Dreal300;
	D000 = Dreal000;
	D001 = Dreal001;
	D002 = Dreal002;
	D010 = Dreal010;
	D011 = Dreal011;
	D020 = Dreal020;
	D021 = Dreal021;
	D030 = Dreal030;
	D100 = Dreal100;
	D101 = Dreal101;
	D110 = Dreal110;
	D111 = Dreal111;
	D120 = Dreal120;
	D200 = Dreal200;
	D201 = Dreal201;
	D210 = Dreal210;
	D300 = Dreal300;
	D021 = fmaf(T(-2.00000000e-01), Dreal_3_1_001, D021);
	D201 = fmaf(T(-2.00000000e-01), Dreal_3_1_001, D201);
	D030 = fmaf(T(-6.00000000e-01), Dreal_3_1_010, D030);
	D210 = fmaf(T(-2.00000000e-01), Dreal_3_1_010, D210);
	D120 = fmaf(T(-2.00000000e-01), Dreal_3_1_100, D120);
	D300 = fmaf(T(-6.00000000e-01), Dreal_3_1_100, D300);
	flops += 74 * foursz + 104;
	D = D + Dfour;
	expansion<T> D1;
	greens_function(D1,X);
	D(0, 0, 0) = T(7.85398163e-01) + D(0, 0, 0); 
	for (int i = 0; i < EXPANSION_SIZE; i++) {
	D[i] -= D1[i];
		D[i] *= zero_mask;
	}
	D[0] += 2.837291e+00 * (T(1) - zero_mask);
	if ( LORDER > 2) {
		D[3] += T(-4.18879020e+00) * (T(1) - zero_mask);
		D[5] += T(-4.18879020e+00) * (T(1) - zero_mask);
		D[EXPANSION_SIZE - 1] += T(-4.18879020e+00) * (T(1) - zero_mask);
	}
	return flops;
}


template<class T>
CUDA_EXPORT
inline int M2L(tensor_trless_sym<T, 2>& L, const tensor_trless_sym<T, 3>& M, const tensor_trless_sym<T, 4>& D, bool do_phi) {
	const T& M000 =  (M[0]);
	const T& M001 =  (M[6]);
	const T& M002 =  (M[9]);
	const T& M010 =  (M[2]);
	const T& M011 =  (M[8]);
	const T& M020 =  (M[5]);
	const T& M100 =  (M[1]);
	const T& M101 =  (M[7]);
	const T& M110 =  (M[4]);
	const T& M200 =  (M[3]);
	const T& D000 =  (D[0]);
	const T& D001 =  (D[10]);
	const T& D002 =  (D[16]);
	const T D003 = -(D[13]+(D[15]));
	const T& D010 =  (D[2]);
	const T& D011 =  (D[12]);
	const T D012 = -(D[7]+(D[9]));
	const T& D020 =  (D[5]);
	const T& D021 =  (D[15]);
	const T& D030 =  (D[9]);
	const T& D100 =  (D[1]);
	const T& D101 =  (D[11]);
	const T D102 = -(D[6]+(D[8]));
	const T& D110 =  (D[4]);
	const T& D111 =  (D[14]);
	const T& D120 =  (D[8]);
	const T& D200 =  (D[3]);
	const T& D201 =  (D[13]);
	const T& D210 =  (D[7]);
	const T& D300 =  (D[6]);
	if( do_phi ) {
		L[0] = fmaf(M000, D000, L[0]);
		L[0] = fmaf(M001, D001, L[0]);
		L[0] = fmaf(T(5.00000000e-01) * M002, D002, L[0]);
		L[0] = fmaf(M010, D010, L[0]);
		L[0] = fmaf(M011, D011, L[0]);
		L[0] = fmaf(T(5.00000000e-01) * M020, D020, L[0]);
		L[0] = fmaf(M100, D100, L[0]);
		L[0] = fmaf(M101, D101, L[0]);
		L[0] = fmaf(M110, D110, L[0]);
		L[0] = fmaf(T(5.00000000e-01) * M200, D200, L[0]);
	}
	L[1] = fmaf(T(5.00000000e-01) * M200, D300, L[1]);
	L[2] = fmaf(M011, D021, L[2]);
	L[1] = fmaf(M110, D210, L[1]);
	L[2] = fmaf(M010, D020, L[2]);
	L[1] = fmaf(M101, D201, L[1]);
	L[2] = fmaf(T(5.00000000e-01) * M002, D012, L[2]);
	L[1] = fmaf(M100, D200, L[1]);
	L[2] = fmaf(M001, D011, L[2]);
	L[1] = fmaf(T(5.00000000e-01) * M020, D120, L[1]);
	L[2] = fmaf(M000, D010, L[2]);
	L[1] = fmaf(M011, D111, L[1]);
	L[3] = fmaf(M000, D001, L[3]);
	L[1] = fmaf(M010, D110, L[1]);
	L[3] = fmaf(T(5.00000000e-01) * M200, D201, L[3]);
	L[1] = fmaf(T(5.00000000e-01) * M002, D102, L[1]);
	L[3] = fmaf(M110, D111, L[3]);
	L[1] = fmaf(M001, D101, L[1]);
	L[3] = fmaf(M101, D102, L[3]);
	L[1] = fmaf(M000, D100, L[1]);
	L[3] = fmaf(M100, D101, L[3]);
	L[2] = fmaf(T(5.00000000e-01) * M020, D030, L[2]);
	L[3] = fmaf(T(5.00000000e-01) * M020, D021, L[3]);
	L[2] = fmaf(T(5.00000000e-01) * M200, D210, L[2]);
	L[3] = fmaf(M011, D012, L[3]);
	L[2] = fmaf(M110, D120, L[2]);
	L[3] = fmaf(M010, D011, L[3]);
	L[2] = fmaf(M101, D111, L[2]);
	L[3] = fmaf(T(5.00000000e-01) * M002, D003, L[3]);
	L[2] = fmaf(M100, D110, L[2]);
	L[3] = fmaf(M001, D002, L[3]);
	return 78 + do_phi * 23;
}


template<class T>
CUDA_EXPORT
inline int M2L(tensor_trless_sym<T, 4>& L, const tensor_trless_sym<T, 3>& M, const tensor_trless_sym<T, 4>& D, bool do_phi) {
	const T& M000 =  (M[0]);
	const T& M001 =  (M[6]);
	const T& M002 =  (M[9]);
	const T& M010 =  (M[2]);
	const T& M011 =  (M[8]);
	const T& M020 =  (M[5]);
	const T& M100 =  (M[1]);
	const T& M101 =  (M[7]);
	const T& M110 =  (M[4]);
	const T& M200 =  (M[3]);
	const T& D000 =  (D[0]);
	const T& D001 =  (D[10]);
	const T& D002 =  (D[16]);
	const T D003 = -(D[13]+(D[15]));
	const T& D010 =  (D[2]);
	const T& D011 =  (D[12]);
	const T D012 = -(D[7]+(D[9]));
	const T& D020 =  (D[5]);
	const T& D021 =  (D[15]);
	const T& D030 =  (D[9]);
	const T& D100 =  (D[1]);
	const T& D101 =  (D[11]);
	const T D102 = -(D[6]+(D[8]));
	const T& D110 =  (D[4]);
	const T& D111 =  (D[14]);
	const T& D120 =  (D[8]);
	const T& D200 =  (D[3]);
	const T& D201 =  (D[13]);
	const T& D210 =  (D[7]);
	const T& D300 =  (D[6]);
	if( do_phi ) {
		L[0] = fmaf(M000, D000, L[0]);
		L[0] = fmaf(M001, D001, L[0]);
		L[0] = fmaf(T(5.00000000e-01) * M002, D002, L[0]);
		L[0] = fmaf(M010, D010, L[0]);
		L[0] = fmaf(M011, D011, L[0]);
		L[0] = fmaf(T(5.00000000e-01) * M020, D020, L[0]);
		L[0] = fmaf(M100, D100, L[0]);
		L[0] = fmaf(M101, D101, L[0]);
		L[0] = fmaf(M110, D110, L[0]);
		L[0] = fmaf(T(5.00000000e-01) * M200, D200, L[0]);
	}
	L[1] = fmaf(M000, D100, L[1]);
	L[5] = fmaf(M100, D120, L[5]);
	L[1] = fmaf(M001, D101, L[1]);
	L[6] = fmaf(M000, D300, L[6]);
	L[1] = fmaf(T(5.00000000e-01) * M002, D102, L[1]);
	L[7] = fmaf(M000, D210, L[7]);
	L[1] = fmaf(M010, D110, L[1]);
	L[8] = fmaf(M000, D120, L[8]);
	L[1] = fmaf(M011, D111, L[1]);
	L[9] = fmaf(M000, D030, L[9]);
	L[1] = fmaf(T(5.00000000e-01) * M200, D300, L[1]);
	L[10] = fmaf(M000, D001, L[10]);
	L[1] = fmaf(M110, D210, L[1]);
	L[10] = fmaf(M001, D002, L[10]);
	L[1] = fmaf(M101, D201, L[1]);
	L[10] = fmaf(T(5.00000000e-01) * M002, D003, L[10]);
	L[1] = fmaf(M100, D200, L[1]);
	L[10] = fmaf(M010, D011, L[10]);
	L[1] = fmaf(T(5.00000000e-01) * M020, D120, L[1]);
	L[10] = fmaf(M011, D012, L[10]);
	L[2] = fmaf(M011, D021, L[2]);
	L[10] = fmaf(T(5.00000000e-01) * M020, D021, L[10]);
	L[2] = fmaf(T(5.00000000e-01) * M002, D012, L[2]);
	L[10] = fmaf(M100, D101, L[10]);
	L[2] = fmaf(T(5.00000000e-01) * M020, D030, L[2]);
	L[10] = fmaf(T(5.00000000e-01) * M200, D201, L[10]);
	L[2] = fmaf(M100, D110, L[2]);
	L[10] = fmaf(M101, D102, L[10]);
	L[2] = fmaf(M101, D111, L[2]);
	L[10] = fmaf(M110, D111, L[10]);
	L[2] = fmaf(M110, D120, L[2]);
	L[11] = fmaf(M100, D201, L[11]);
	L[2] = fmaf(T(5.00000000e-01) * M200, D210, L[2]);
	L[11] = fmaf(M010, D111, L[11]);
	L[2] = fmaf(M010, D020, L[2]);
	L[11] = fmaf(M001, D102, L[11]);
	L[2] = fmaf(M001, D011, L[2]);
	L[11] = fmaf(M000, D101, L[11]);
	L[2] = fmaf(M000, D010, L[2]);
	L[12] = fmaf(M000, D011, L[12]);
	L[3] = fmaf(M000, D200, L[3]);
	L[12] = fmaf(M001, D012, L[12]);
	L[3] = fmaf(M001, D201, L[3]);
	L[12] = fmaf(M010, D021, L[12]);
	L[3] = fmaf(M010, D210, L[3]);
	L[12] = fmaf(M100, D111, L[12]);
	L[3] = fmaf(M100, D300, L[3]);
	L[13] = fmaf(M000, D201, L[13]);
	L[4] = fmaf(M000, D110, L[4]);
	L[14] = fmaf(M000, D111, L[14]);
	L[4] = fmaf(M001, D111, L[4]);
	L[15] = fmaf(M000, D021, L[15]);
	L[4] = fmaf(M010, D120, L[4]);
	L[16] = fmaf(M000, D002, L[16]);
	L[4] = fmaf(M100, D210, L[4]);
	L[16] = fmaf(M001, D003, L[16]);
	L[5] = fmaf(M000, D020, L[5]);
	L[16] = fmaf(M010, D012, L[16]);
	L[5] = fmaf(M001, D021, L[5]);
	L[16] = fmaf(M100, D102, L[16]);
	L[5] = fmaf(M010, D030, L[5]);
	return 140 + do_phi * 23;
}


template<class T>
CUDA_EXPORT
tensor_trless_sym<T, 3> P2M(array<T, NDIM>& X) {
	tensor_trless_sym<T, 3> M;
	X[0] = -X[0];
	X[1] = -X[1];
	X[2] = -X[2];
	T& M000 = M[0];
	T& M001 = M[6];
	T& M002 = M[9];
	T& M010 = M[2];
	T& M011 = M[8];
	T& M020 = M[5];
	T& M100 = M[1];
	T& M101 = M[7];
	T& M110 = M[4];
	T& M200 = M[3];
	const T x000 = T(1);
	const T& x100 = X[0];
	const T& x010 = X[1];
	const T& x001 = X[2];
	const T x002 = x001 * x001;
	const T x011 = x010 * x001;
	const T x020 = x010 * x010;
	const T x101 = x100 * x001;
	const T x110 = x100 * x010;
	const T x200 = x100 * x100;
	T x_2_1_000 = x002;
	x_2_1_000 += x020;
	x_2_1_000 += x200;
	M000 = x000;
	M001 = x001;
	M002 = x002;
	M010 = x010;
	M011 = x011;
	M020 = x020;
	M100 = x100;
	M101 = x101;
	M110 = x110;
	M200 = x200;
	return M;
/* FLOPS = 11*/
}


template<class T>
CUDA_EXPORT
tensor_trless_sym<T, 3> M2M(const tensor_trless_sym<T,3>& Ma, array<T, NDIM>& X) {
	tensor_sym<T, 3> Mb;
	tensor_trless_sym<T, 3> Mc;
	X[0] = -X[0];
	X[1] = -X[1];
	X[2] = -X[2];
	const T& Ma000 =  (Ma[0]);
	const T& Ma001 =  (Ma[6]);
	const T& Ma002 =  (Ma[9]);
	const T& Ma010 =  (Ma[2]);
	const T& Ma011 =  (Ma[8]);
	const T& Ma020 =  (Ma[5]);
	const T& Ma100 =  (Ma[1]);
	const T& Ma101 =  (Ma[7]);
	const T& Ma110 =  (Ma[4]);
	const T& Ma200 =  (Ma[3]);
	const T& Mb000 = Mb[0];
	const T& Mb001 = Mb[3];
	const T& Mb002 = Mb[9];
	const T& Mb010 = Mb[2];
	const T& Mb011 = Mb[8];
	const T& Mb020 = Mb[7];
	const T& Mb100 = Mb[1];
	const T& Mb101 = Mb[6];
	const T& Mb110 = Mb[5];
	const T& Mb200 = Mb[4];
	T& Mc000 = Mc[0];
	T& Mc001 = Mc[6];
	T& Mc002 = Mc[9];
	T& Mc010 = Mc[2];
	T& Mc011 = Mc[8];
	T& Mc020 = Mc[5];
	T& Mc100 = Mc[1];
	T& Mc101 = Mc[7];
	T& Mc110 = Mc[4];
	T& Mc200 = Mc[3];
	const T x000 = T(1);
	const T& x100 = X[0];
	const T& x010 = X[1];
	const T& x001 = X[2];
	const T x002 = x001 * x001;
	const T x011 = x010 * x001;
	const T x020 = x010 * x010;
	const T x101 = x100 * x001;
	const T x110 = x100 * x010;
	const T x200 = x100 * x100;
	Mb[0] = Ma000;
	Mb[1] = Ma100;
	Mb[2] = Ma010;
	Mb[3] = Ma001;
	Mb[4] = Ma200;
	Mb[5] = Ma110;
	Mb[6] = Ma101;
	Mb[7] = Ma020;
	Mb[8] = Ma011;
	Mb[9] = Ma002;
	Mb[1] = fmaf( x100, Ma000, Mb[1]);
	Mb[6] = fmaf( x100, Ma001, Mb[6]);
	Mb[2] = fmaf( x010, Ma000, Mb[2]);
	Mb[6] = fmaf( x001, Ma100, Mb[6]);
	Mb[3] = fmaf( x001, Ma000, Mb[3]);
	Mb[7] = fmaf( x020, Ma000, Mb[7]);
	Mb[4] = fmaf(T(2.00000000e+00) * x100, Ma100, Mb[4]);
	Mb[7] = fmaf(T(2.00000000e+00) * x010, Ma010, Mb[7]);
	Mb[4] = fmaf( x200, Ma000, Mb[4]);
	Mb[8] = fmaf( x011, Ma000, Mb[8]);
	Mb[5] = fmaf( x110, Ma000, Mb[5]);
	Mb[8] = fmaf( x010, Ma001, Mb[8]);
	Mb[5] = fmaf( x100, Ma010, Mb[5]);
	Mb[8] = fmaf( x001, Ma010, Mb[8]);
	Mb[5] = fmaf( x010, Ma100, Mb[5]);
	Mb[9] = fmaf(T(2.00000000e+00) * x001, Ma001, Mb[9]);
	Mb[6] = fmaf( x101, Ma000, Mb[6]);
	Mb[9] = fmaf( x002, Ma000, Mb[9]);
	T Mb_2_1_000 = Mb002;
	Mb_2_1_000 += Mb020;
	Mb_2_1_000 += Mb200;
	Mc000 = Mb000;
	Mc001 = Mb001;
	Mc002 = Mb002;
	Mc010 = Mb010;
	Mc011 = Mb011;
	Mc020 = Mb020;
	Mc100 = Mb100;
	Mc101 = Mb101;
	Mc110 = Mb110;
	Mc200 = Mb200;
	return Mc;
/* FLOPS = 47*/
}
template<class T>
CUDA_EXPORT
tensor_trless_sym<T, 4> L2L(const tensor_trless_sym<T, 4>& La, const array<T, NDIM>& X, bool do_phi) {
	tensor_trless_sym<T, 4> Lb;
//	const T x000 = T(1);
	const T& x100 = X[0];
	const T& x010 = X[1];
	const T& x001 = X[2];
	const T x002 = x001 * x001;
	const T x011 = x010 * x001;
	const T x020 = x010 * x010;
	const T x101 = x100 * x001;
	const T x110 = x100 * x010;
	const T x200 = x100 * x100;
	const T x003 = x002 * x001;
	const T x012 = x011 * x001;
	const T x021 = x011 * x010;
	const T x030 = x020 * x010;
	const T x102 = x101 * x001;
	const T x111 = x110 * x001;
	const T x120 = x110 * x010;
	const T x201 = x101 * x100;
	const T x210 = x110 * x100;
	const T x300 = x200 * x100;
	const T& La000 =  (La[0]);
	const T& La001 =  (La[10]);
	const T& La002 =  (La[16]);
	const T La003 = -(La[13]+(La[15]));
	const T& La010 =  (La[2]);
	const T& La011 =  (La[12]);
	const T La012 = -(La[7]+(La[9]));
	const T& La020 =  (La[5]);
	const T& La021 =  (La[15]);
	const T& La030 =  (La[9]);
	const T& La100 =  (La[1]);
	const T& La101 =  (La[11]);
	const T La102 = -(La[6]+(La[8]));
	const T& La110 =  (La[4]);
	const T& La111 =  (La[14]);
	const T& La120 =  (La[8]);
	const T& La200 =  (La[3]);
	const T& La201 =  (La[13]);
	const T& La210 =  (La[7]);
	const T& La300 =  (La[6]);
	Lb = La;
	if( do_phi ) {
		Lb[0] = fmaf( x001, La001, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x002, La002, Lb[0]);
		Lb[0] = fmaf(T(1.66666667e-01) * x003, La003, Lb[0]);
		Lb[0] = fmaf( x010, La010, Lb[0]);
		Lb[0] = fmaf( x011, La011, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x012, La012, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x020, La020, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x021, La021, Lb[0]);
		Lb[0] = fmaf(T(1.66666667e-01) * x030, La030, Lb[0]);
		Lb[0] = fmaf( x100, La100, Lb[0]);
		Lb[0] = fmaf( x101, La101, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x102, La102, Lb[0]);
		Lb[0] = fmaf( x110, La110, Lb[0]);
		Lb[0] = fmaf( x111, La111, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x120, La120, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x200, La200, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x201, La201, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x210, La210, Lb[0]);
		Lb[0] = fmaf(T(1.66666667e-01) * x300, La300, Lb[0]);
	}
	Lb[1] = fmaf( x100, La200, Lb[1]);
	Lb[4] = fmaf( x001, La111, Lb[4]);
	Lb[1] = fmaf( x010, La110, Lb[1]);
	Lb[5] = fmaf( x100, La120, Lb[5]);
	Lb[1] = fmaf( x001, La101, Lb[1]);
	Lb[5] = fmaf( x010, La030, Lb[5]);
	Lb[1] = fmaf(T(5.00000000e-01) * x200, La300, Lb[1]);
	Lb[5] = fmaf( x001, La021, Lb[5]);
	Lb[1] = fmaf( x110, La210, Lb[1]);
	Lb[10] = fmaf( x100, La101, Lb[10]);
	Lb[1] = fmaf( x101, La201, Lb[1]);
	Lb[10] = fmaf( x010, La011, Lb[10]);
	Lb[1] = fmaf(T(5.00000000e-01) * x020, La120, Lb[1]);
	Lb[10] = fmaf( x001, La002, Lb[10]);
	Lb[1] = fmaf( x011, La111, Lb[1]);
	Lb[10] = fmaf(T(5.00000000e-01) * x200, La201, Lb[10]);
	Lb[1] = fmaf(T(5.00000000e-01) * x002, La102, Lb[1]);
	Lb[10] = fmaf( x110, La111, Lb[10]);
	Lb[2] = fmaf( x100, La110, Lb[2]);
	Lb[10] = fmaf( x101, La102, Lb[10]);
	Lb[2] = fmaf( x010, La020, Lb[2]);
	Lb[10] = fmaf(T(5.00000000e-01) * x020, La021, Lb[10]);
	Lb[2] = fmaf( x001, La011, Lb[2]);
	Lb[10] = fmaf( x011, La012, Lb[10]);
	Lb[2] = fmaf(T(5.00000000e-01) * x200, La210, Lb[2]);
	Lb[10] = fmaf(T(5.00000000e-01) * x002, La003, Lb[10]);
	Lb[2] = fmaf( x110, La120, Lb[2]);
	Lb[11] = fmaf( x100, La201, Lb[11]);
	Lb[2] = fmaf( x101, La111, Lb[2]);
	Lb[11] = fmaf( x010, La111, Lb[11]);
	Lb[2] = fmaf(T(5.00000000e-01) * x020, La030, Lb[2]);
	Lb[11] = fmaf( x001, La102, Lb[11]);
	Lb[2] = fmaf( x011, La021, Lb[2]);
	Lb[12] = fmaf( x100, La111, Lb[12]);
	Lb[2] = fmaf(T(5.00000000e-01) * x002, La012, Lb[2]);
	Lb[12] = fmaf( x010, La021, Lb[12]);
	Lb[3] = fmaf( x100, La300, Lb[3]);
	Lb[12] = fmaf( x001, La012, Lb[12]);
	Lb[3] = fmaf( x010, La210, Lb[3]);
	Lb[16] = fmaf( x100, La102, Lb[16]);
	Lb[3] = fmaf( x001, La201, Lb[3]);
	Lb[16] = fmaf( x010, La012, Lb[16]);
	Lb[4] = fmaf( x100, La210, Lb[4]);
	Lb[16] = fmaf( x001, La003, Lb[16]);
	Lb[4] = fmaf( x010, La120, Lb[4]);
	return Lb;
/* FLOPS = 174*/
}
template<class T>
CUDA_EXPORT
tensor_trless_sym<T, 2> L2P(const tensor_trless_sym<T, 4>& La, const array<T, NDIM>& X, bool do_phi) {
	tensor_trless_sym<T, 2> Lb;
//	const T x000 = T(1);
	const T& x100 = X[0];
	const T& x010 = X[1];
	const T& x001 = X[2];
	const T x002 = x001 * x001;
	const T x011 = x010 * x001;
	const T x020 = x010 * x010;
	const T x101 = x100 * x001;
	const T x110 = x100 * x010;
	const T x200 = x100 * x100;
	const T x003 = x002 * x001;
	const T x012 = x011 * x001;
	const T x021 = x011 * x010;
	const T x030 = x020 * x010;
	const T x102 = x101 * x001;
	const T x111 = x110 * x001;
	const T x120 = x110 * x010;
	const T x201 = x101 * x100;
	const T x210 = x110 * x100;
	const T x300 = x200 * x100;
	const T& La000 =  (La[0]);
	const T& La001 =  (La[10]);
	const T& La002 =  (La[16]);
	const T La003 = -(La[13]+(La[15]));
	const T& La010 =  (La[2]);
	const T& La011 =  (La[12]);
	const T La012 = -(La[7]+(La[9]));
	const T& La020 =  (La[5]);
	const T& La021 =  (La[15]);
	const T& La030 =  (La[9]);
	const T& La100 =  (La[1]);
	const T& La101 =  (La[11]);
	const T La102 = -(La[6]+(La[8]));
	const T& La110 =  (La[4]);
	const T& La111 =  (La[14]);
	const T& La120 =  (La[8]);
	const T& La200 =  (La[3]);
	const T& La201 =  (La[13]);
	const T& La210 =  (La[7]);
	const T& La300 =  (La[6]);
	Lb(0,0,0) = La(0,0,0);
	Lb(1,0,0) = La(1,0,0);
	Lb(0,1,0) = La(0,1,0);
	Lb(0,0,1) = La(0,0,1);
	if( do_phi ) {
		Lb[0] = fmaf( x001, La001, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x002, La002, Lb[0]);
		Lb[0] = fmaf(T(1.66666667e-01) * x003, La003, Lb[0]);
		Lb[0] = fmaf( x010, La010, Lb[0]);
		Lb[0] = fmaf( x011, La011, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x012, La012, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x020, La020, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x021, La021, Lb[0]);
		Lb[0] = fmaf(T(1.66666667e-01) * x030, La030, Lb[0]);
		Lb[0] = fmaf( x100, La100, Lb[0]);
		Lb[0] = fmaf( x101, La101, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x102, La102, Lb[0]);
		Lb[0] = fmaf( x110, La110, Lb[0]);
		Lb[0] = fmaf( x111, La111, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x120, La120, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x200, La200, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x201, La201, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x210, La210, Lb[0]);
		Lb[0] = fmaf(T(1.66666667e-01) * x300, La300, Lb[0]);
	}
	Lb[1] = fmaf( x100, La200, Lb[1]);
	Lb[2] = fmaf( x101, La111, Lb[2]);
	Lb[1] = fmaf( x010, La110, Lb[1]);
	Lb[2] = fmaf(T(5.00000000e-01) * x020, La030, Lb[2]);
	Lb[1] = fmaf( x001, La101, Lb[1]);
	Lb[2] = fmaf( x011, La021, Lb[2]);
	Lb[1] = fmaf(T(5.00000000e-01) * x200, La300, Lb[1]);
	Lb[2] = fmaf(T(5.00000000e-01) * x002, La012, Lb[2]);
	Lb[1] = fmaf( x110, La210, Lb[1]);
	Lb[3] = fmaf( x100, La101, Lb[3]);
	Lb[1] = fmaf( x101, La201, Lb[1]);
	Lb[3] = fmaf( x010, La011, Lb[3]);
	Lb[1] = fmaf(T(5.00000000e-01) * x020, La120, Lb[1]);
	Lb[3] = fmaf( x001, La002, Lb[3]);
	Lb[1] = fmaf( x011, La111, Lb[1]);
	Lb[3] = fmaf(T(5.00000000e-01) * x200, La201, Lb[3]);
	Lb[1] = fmaf(T(5.00000000e-01) * x002, La102, Lb[1]);
	Lb[3] = fmaf( x110, La111, Lb[3]);
	Lb[2] = fmaf( x100, La110, Lb[2]);
	Lb[3] = fmaf( x101, La102, Lb[3]);
	Lb[2] = fmaf( x010, La020, Lb[2]);
	Lb[3] = fmaf(T(5.00000000e-01) * x020, La021, Lb[3]);
	Lb[2] = fmaf( x001, La011, Lb[2]);
	Lb[3] = fmaf( x011, La012, Lb[3]);
	Lb[2] = fmaf(T(5.00000000e-01) * x200, La210, Lb[2]);
	Lb[3] = fmaf(T(5.00000000e-01) * x002, La003, Lb[3]);
	Lb[2] = fmaf( x110, La120, Lb[2]);
	return Lb;
/* FLOPS = 138*/
}
#ifdef EXPANSION_CU
__managed__ char Ldest1[23] = { 1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,4,4};
__managed__ float factor1[23] = { 1.00000000e+00,1.00000000e+00,1.00000000e+00,5.00000000e-01,1.00000000e+00,1.00000000e+00,5.00000000e-01,1.00000000e+00,5.00000000e-01,1.00000000e+00,1.00000000e+00,1.00000000e+00,5.00000000e-01,1.00000000e+00,1.00000000e+00,5.00000000e-01,1.00000000e+00,5.00000000e-01,1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00};
__managed__ char xsrc1[23] = { 1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,1,2};
__managed__ char Lsrc1[23] = { 4,5,6,10,11,12,13,14,15,5,7,8,11,13,14,16,17,18,10,11,12,11,13};
__managed__ char Ldest2[22] = { 4,5,5,5,10,10,10,10,10,10,10,10,10,11,11,11,12,12,12,16,16,16};
__managed__ float factor2[22] = { 1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00,5.00000000e-01,1.00000000e+00,1.00000000e+00,5.00000000e-01,1.00000000e+00,5.00000000e-01,1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00};
__managed__ char xsrc2[22] = { 3,1,2,3,1,2,3,4,5,6,7,8,9,1,2,3,1,2,3,1,2,3};
__managed__ char Lsrc2[22] = { 14,13,16,17,6,8,9,12,14,15,17,18,19,12,14,15,14,17,18,15,18,19};
__managed__ float phi_factor[19] = { 1.00000000e+00,5.00000000e-01,1.66666672e-01,1.00000000e+00,1.00000000e+00,5.00000000e-01,5.00000000e-01,5.00000000e-01,1.66666672e-01,1.00000000e+00,1.00000000e+00,5.00000000e-01,1.00000000e+00,1.00000000e+00,5.00000000e-01,5.00000000e-01,5.00000000e-01,5.00000000e-01,1.66666672e-01};
__managed__ char phi_Lsrc[19] = { 3,9,19,2,8,18,7,17,16,1,6,15,5,14,13,4,12,11,10};
#ifdef __CUDA_ARCH__
template<class T>
__device__
tensor_trless_sym<T, 4> L2L_cuda(const tensor_trless_sym<T, 4>& La, const array<T, NDIM>& X, bool do_phi) {
	const int tid = threadIdx.x;
	tensor_trless_sym<T, 4> Lb;
	tensor_sym<T, 4> Lc;
	Lb = 0.0f;
	for( int i = tid; i < EXPANSION_SIZE; i += KICK_BLOCK_SIZE ) {
		Lb[i] = La[i];
	}
	tensor_sym<T,4> dx;
	dx[0] = T(1);
	dx[1] = X[0];
	dx[2] = X[1];
	dx[3] = X[2];
	dx[9]= dx[3] * dx[3];
	dx[8]= dx[2] * dx[3];
	dx[7]= dx[2] * dx[2];
	dx[6]= dx[1] * dx[3];
	dx[5]= dx[1] * dx[2];
	dx[4]= dx[1] * dx[1];
	dx[19]= dx[9] * dx[3];
	dx[18]= dx[8] * dx[3];
	dx[17]= dx[8] * dx[2];
	dx[16]= dx[7] * dx[2];
	dx[15]= dx[6] * dx[3];
	dx[14]= dx[5] * dx[3];
	dx[13]= dx[5] * dx[2];
	dx[12]= dx[6] * dx[1];
	dx[11]= dx[5] * dx[1];
	dx[10]= dx[4] * dx[1];
	Lc[0] =  (La[0]);
	Lc[3] =  (La[10]);
	Lc[9] =  (La[16]);
	Lc[19] = -(La[13]+(La[15]));
	Lc[2] =  (La[2]);
	Lc[8] =  (La[12]);
	Lc[18] = -(La[7]+(La[9]));
	Lc[7] =  (La[5]);
	Lc[17] =  (La[15]);
	Lc[16] =  (La[9]);
	Lc[1] =  (La[1]);
	Lc[6] =  (La[11]);
	Lc[15] = -(La[6]+(La[8]));
	Lc[5] =  (La[4]);
	Lc[14] =  (La[14]);
	Lc[13] =  (La[8]);
	Lc[4] =  (La[3]);
	Lc[12] =  (La[13]);
	Lc[11] =  (La[7]);
	Lc[10] =  (La[6]);
	for( int i = tid; i < 22; i+=KICK_BLOCK_SIZE) {
		Lb[Ldest1[i]] = fmaf(factor1[i] * dx[xsrc1[i]], Lc[Lsrc1[i]], Lb[Ldest1[i]]);
		Lb[Ldest2[i]] = fmaf(factor2[i] * dx[xsrc2[i]], Lc[Lsrc2[i]], Lb[Ldest2[i]]);
	}
	if( tid == 0 ) {
		Lb[Ldest1[22]] = fmaf(factor1[22] * dx[xsrc1[22]], Lc[Lsrc1[22]], Lb[Ldest1[22]]);
	}
	if( do_phi ) {
		for( int i = tid; i < 19; i+=KICK_BLOCK_SIZE) {
			Lb[0] = fmaf(phi_factor[i] * dx[phi_Lsrc[i]], Lc[phi_Lsrc[i]], Lb[0]);
		}
	}
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			Lb[i] += __shfl_xor_sync(0xffffffff, Lb[i], P);
		}
	}
	return Lb;
/* FLOPS = 205 + do_phi * 76*/
}
#endif
#endif
