#pragma once
#include <tigerfmm/tensor.hpp>
#include <tigerfmm/cuda.hpp>
#include <tigerfmm/ewald_indices.hpp>
#include <tigerfmm/math.hpp>
template<class T>
using expansion = tensor_trless_sym<T,3>;
template<class T>
using expansion2 = tensor_trless_sym<T,2>;
template<class T>
using multipole = tensor_trless_sym<T,2>;
#define EXPANSION_SIZE 10
#define MULTIPOLE_SIZE 5


template<class T>
CUDA_EXPORT
inline int greens_function(tensor_trless_sym<T, 3>& D, array<T, NDIM> X) {
	auto r2 = sqr(X[0], X[1], X[2]);
	r2 = sqr(X[0], X[1], X[2]);
	const T r = sqrt(r2);
	const T rinv1 = -(r > T(0)) / max(r, T(1e-20));
	const T rinv2 = -rinv1 * rinv1;
	const T rinv3 = -rinv2 * rinv1;
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
	T& D000 = D[0];
	T& D001 = D[6];
	T& D002 = D[9];
	T& D010 = D[2];
	T& D011 = D[8];
	T& D020 = D[5];
	T& D100 = D[1];
	T& D101 = D[7];
	T& D110 = D[4];
	T& D200 = D[3];
	T x_2_1_000 = x002;
	x_2_1_000 += x020;
	x_2_1_000 += x200;
	D000 = x000;
	D001 = x001;
	D002 = T(3.00000000e+00) * x002;
	D010 = x010;
	D011 = T(3.00000000e+00) * x011;
	D020 = T(3.00000000e+00) * x020;
	D100 = x100;
	D101 = T(3.00000000e+00) * x101;
	D110 = T(3.00000000e+00) * x110;
	D200 = T(3.00000000e+00) * x200;
	D002 -= x_2_1_000;
	D200 -= x_2_1_000;
	D020 -= x_2_1_000;
	D000 *= rinv1;
	D001 *= rinv2;
	D002 *= rinv3;
	D010 *= rinv2;
	D011 *= rinv3;
	D020 *= rinv3;
	D100 *= rinv2;
	D101 *= rinv3;
	D110 *= rinv3;
	D200 *= rinv3;
	return 54;
}
template<class T>
CUDA_EXPORT int ewald_greens_function(tensor_trless_sym<T,3> &D, array<T, NDIM> X) {
	ewald_const econst;
	int flops = 7;
	T r = sqrt(fmaf(X[0], X[0], fmaf(X[1], X[1], sqr(X[2]))));
	const T fouroversqrtpi = T(2.25675833e+00);
	tensor_sym<T, 3> Dreal;
	tensor_trless_sym<T,3> Dfour;
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
			const T Drinvpow_0_0 = d0 * rinv0;
			const T Drinvpow_1_0 = d1 * rinv0;
			const T Drinvpow_1_1 = d1 * rinv1;
			const T Drinvpow_2_0 = d2 * rinv0;
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
			T x_2_1_000 = x002;
			x_2_1_000 += x020;
			x_2_1_000 += x200;
			x_2_1_000 *= Drinvpow_1_1;
			Dreal[0] = fmaf(x000, Drinvpow_0_0, Dreal[0]);
			Dreal[6] = fmaf(x101, Drinvpow_2_0, Dreal[6]);
			Dreal[1] = fmaf(x100, Drinvpow_1_0, Dreal[1]);
			Dreal[7] = fmaf(x020, Drinvpow_2_0, Dreal[7]);
			Dreal[2] = fmaf(x010, Drinvpow_1_0, Dreal[2]);
			Dreal[7] += x_2_1_000;
			Dreal[3] = fmaf(x001, Drinvpow_1_0, Dreal[3]);
			Dreal[8] = fmaf(x011, Drinvpow_2_0, Dreal[8]);
			Dreal[4] = fmaf(x200, Drinvpow_2_0, Dreal[4]);
			Dreal[9] = fmaf(x002, Drinvpow_2_0, Dreal[9]);
			Dreal[4] += x_2_1_000;
			Dreal[9] += x_2_1_000;
			Dreal[5] = fmaf(x110, Drinvpow_2_0, Dreal[5]);
		}
	}
	flops += icnt * 93;
	const auto foursz = econst.nfour();
	for (int i = 0; i < foursz; i++) {
		const auto &h = econst.four_index(i);
		const auto& D0 = econst.four_expansion(i);
		const T hdotx = fmaf(h[0], X[0], fmaf(h[1], X[1], h[2] * X[2]));
		T cn, sn;
		sincos(T(2.0 * M_PI) * hdotx, &sn, &cn);
		Dfour[0] = fmaf(cn, D0[0], Dfour[0]);
		Dfour[5] = fmaf(cn, D0[5], Dfour[5]);
		Dfour[1] = fmaf(sn, D0[1], Dfour[1]);
		Dfour[6] = fmaf(sn, D0[6], Dfour[6]);
		Dfour[2] = fmaf(sn, D0[2], Dfour[2]);
		Dfour[7] = fmaf(cn, D0[7], Dfour[7]);
		Dfour[3] = fmaf(cn, D0[3], Dfour[3]);
		Dfour[8] = fmaf(cn, D0[8], Dfour[8]);
		Dfour[4] = fmaf(cn, D0[4], Dfour[4]);
		Dfour[9] = fmaf(cn, D0[9], Dfour[9]);
	}
	const T& Dreal000 = Dreal[0];
	const T& Dreal001 = Dreal[3];
	const T& Dreal002 = Dreal[9];
	const T& Dreal010 = Dreal[2];
	const T& Dreal011 = Dreal[8];
	const T& Dreal020 = Dreal[7];
	const T& Dreal100 = Dreal[1];
	const T& Dreal101 = Dreal[6];
	const T& Dreal110 = Dreal[5];
	const T& Dreal200 = Dreal[4];
	T& D000 = D[0];
	T& D001 = D[6];
	T& D002 = D[9];
	T& D010 = D[2];
	T& D011 = D[8];
	T& D020 = D[5];
	T& D100 = D[1];
	T& D101 = D[7];
	T& D110 = D[4];
	T& D200 = D[3];
	T Dreal_2_1_000 = Dreal002;
	Dreal_2_1_000 += Dreal020;
	Dreal_2_1_000 += Dreal200;
	D000 = Dreal000;
	D001 = Dreal001;
	D002 = Dreal002;
	D010 = Dreal010;
	D011 = Dreal011;
	D020 = Dreal020;
	D100 = Dreal100;
	D101 = Dreal101;
	D110 = Dreal110;
	D200 = Dreal200;
	flops += 60 * foursz + 58;
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
inline int M2L(tensor_trless_sym<T, 2>& L, const tensor_trless_sym<T, 2>& M, const tensor_trless_sym<T, 3>& D, bool do_phi) {
	const T& M000 =  (M[0]);
	const T& M001 =  (M[3]);
	const T& M010 =  (M[2]);
	const T& M100 =  (M[1]);
	const T& D000 =  (D[0]);
	const T& D001 =  (D[6]);
	const T& D002 =  (D[9]);
	const T& D010 =  (D[2]);
	const T& D011 =  (D[8]);
	const T& D020 =  (D[5]);
	const T& D100 =  (D[1]);
	const T& D101 =  (D[7]);
	const T& D110 =  (D[4]);
	const T& D200 =  (D[3]);
	if( do_phi ) {
		L[0] = fmaf(M000, D000, L[0]);
		L[0] = fmaf(M001, D001, L[0]);
		L[0] = fmaf(M010, D010, L[0]);
		L[0] = fmaf(M100, D100, L[0]);
	}
	L[1] = fmaf(M000, D100, L[1]);
	L[2] = fmaf(M010, D020, L[2]);
	L[1] = fmaf(M001, D101, L[1]);
	L[2] = fmaf(M100, D110, L[2]);
	L[1] = fmaf(M010, D110, L[1]);
	L[3] = fmaf(M000, D001, L[3]);
	L[1] = fmaf(M100, D200, L[1]);
	L[3] = fmaf(M001, D002, L[3]);
	L[2] = fmaf(M000, D010, L[2]);
	L[3] = fmaf(M010, D011, L[3]);
	L[2] = fmaf(M001, D011, L[2]);
	L[3] = fmaf(M100, D101, L[3]);
	return 24 + do_phi * 8;
}


template<class T>
CUDA_EXPORT
inline int M2L(tensor_trless_sym<T, 3>& L, const tensor_trless_sym<T, 2>& M, const tensor_trless_sym<T, 3>& D, bool do_phi) {
	const T& M000 =  (M[0]);
	const T& M001 =  (M[3]);
	const T& M010 =  (M[2]);
	const T& M100 =  (M[1]);
	const T& D000 =  (D[0]);
	const T& D001 =  (D[6]);
	const T& D002 =  (D[9]);
	const T& D010 =  (D[2]);
	const T& D011 =  (D[8]);
	const T& D020 =  (D[5]);
	const T& D100 =  (D[1]);
	const T& D101 =  (D[7]);
	const T& D110 =  (D[4]);
	const T& D200 =  (D[3]);
	if( do_phi ) {
		L[0] = fmaf(M000, D000, L[0]);
		L[0] = fmaf(M001, D001, L[0]);
		L[0] = fmaf(M010, D010, L[0]);
		L[0] = fmaf(M100, D100, L[0]);
	}
	L[1] = fmaf(M100, D200, L[1]);
	L[4] = fmaf(M000, D110, L[4]);
	L[1] = fmaf(M010, D110, L[1]);
	L[5] = fmaf(M000, D020, L[5]);
	L[1] = fmaf(M001, D101, L[1]);
	L[6] = fmaf(M001, D002, L[6]);
	L[1] = fmaf(M000, D100, L[1]);
	L[6] = fmaf(M100, D101, L[6]);
	L[2] = fmaf(M000, D010, L[2]);
	L[6] = fmaf(M010, D011, L[6]);
	L[2] = fmaf(M001, D011, L[2]);
	L[6] = fmaf(M000, D001, L[6]);
	L[2] = fmaf(M010, D020, L[2]);
	L[7] = fmaf(M000, D101, L[7]);
	L[2] = fmaf(M100, D110, L[2]);
	L[8] = fmaf(M000, D011, L[8]);
	L[3] = fmaf(M000, D200, L[3]);
	L[9] = fmaf(M000, D002, L[9]);
	return 36 + do_phi * 8;
}


template<class T>
CUDA_EXPORT
tensor_trless_sym<T, 2> P2M(array<T, NDIM>& X) {
	tensor_trless_sym<T, 2> M;
	X[0] = -X[0];
	X[1] = -X[1];
	X[2] = -X[2];
	T& M000 = M[0];
	T& M001 = M[3];
	T& M010 = M[2];
	T& M100 = M[1];
	const T x000 = T(1);
	const T& x100 = X[0];
	const T& x010 = X[1];
	const T& x001 = X[2];
	M000 = x000;
	M001 = x001;
	M010 = x010;
	M100 = x100;
	return M;
/* FLOPS = 3*/
}


template<class T>
CUDA_EXPORT
tensor_trless_sym<T, 2> M2M(const tensor_trless_sym<T,2>& Ma, array<T, NDIM>& X) {
	tensor_sym<T, 2> Mb;
	tensor_trless_sym<T, 2> Mc;
	X[0] = -X[0];
	X[1] = -X[1];
	X[2] = -X[2];
	const T& Ma000 =  (Ma[0]);
	const T& Ma001 =  (Ma[3]);
	const T& Ma010 =  (Ma[2]);
	const T& Ma100 =  (Ma[1]);
	const T& Mb000 = Mb[0];
	const T& Mb001 = Mb[3];
	const T& Mb010 = Mb[2];
	const T& Mb100 = Mb[1];
	T& Mc000 = Mc[0];
	T& Mc001 = Mc[3];
	T& Mc010 = Mc[2];
	T& Mc100 = Mc[1];
	const T x000 = T(1);
	const T& x100 = X[0];
	const T& x010 = X[1];
	const T& x001 = X[2];
	Mb[0] = Ma000;
	Mb[1] = Ma100;
	Mb[2] = Ma010;
	Mb[3] = Ma001;
	Mb[1] = fmaf( x100, Ma000, Mb[1]);
	Mb[3] = fmaf( x001, Ma000, Mb[3]);
	Mb[2] = fmaf( x010, Ma000, Mb[2]);
	Mc000 = Mb000;
	Mc001 = Mb001;
	Mc010 = Mb010;
	Mc100 = Mb100;
	return Mc;
/* FLOPS = 6*/
}
template<class T>
CUDA_EXPORT
tensor_trless_sym<T, 3> L2L(const tensor_trless_sym<T, 3>& La, const array<T, NDIM>& X, bool do_phi) {
	tensor_trless_sym<T, 3> Lb;
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
	const T& La000 =  (La[0]);
	const T& La001 =  (La[6]);
	const T& La002 =  (La[9]);
	const T& La010 =  (La[2]);
	const T& La011 =  (La[8]);
	const T& La020 =  (La[5]);
	const T& La100 =  (La[1]);
	const T& La101 =  (La[7]);
	const T& La110 =  (La[4]);
	const T& La200 =  (La[3]);
	Lb = La;
	if( do_phi ) {
		Lb[0] = fmaf( x001, La001, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x002, La002, Lb[0]);
		Lb[0] = fmaf( x010, La010, Lb[0]);
		Lb[0] = fmaf( x011, La011, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x020, La020, Lb[0]);
		Lb[0] = fmaf( x100, La100, Lb[0]);
		Lb[0] = fmaf( x101, La101, Lb[0]);
		Lb[0] = fmaf( x110, La110, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x200, La200, Lb[0]);
	}
	Lb[1] = fmaf( x100, La200, Lb[1]);
	Lb[2] = fmaf( x001, La011, Lb[2]);
	Lb[1] = fmaf( x010, La110, Lb[1]);
	Lb[6] = fmaf( x100, La101, Lb[6]);
	Lb[1] = fmaf( x001, La101, Lb[1]);
	Lb[6] = fmaf( x010, La011, Lb[6]);
	Lb[2] = fmaf( x100, La110, Lb[2]);
	Lb[6] = fmaf( x001, La002, Lb[6]);
	Lb[2] = fmaf( x010, La020, Lb[2]);
	return Lb;
/* FLOPS = 45*/
}
template<class T>
CUDA_EXPORT
tensor_trless_sym<T, 2> L2P(const tensor_trless_sym<T, 3>& La, const array<T, NDIM>& X, bool do_phi) {
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
	const T& La000 =  (La[0]);
	const T& La001 =  (La[6]);
	const T& La002 =  (La[9]);
	const T& La010 =  (La[2]);
	const T& La011 =  (La[8]);
	const T& La020 =  (La[5]);
	const T& La100 =  (La[1]);
	const T& La101 =  (La[7]);
	const T& La110 =  (La[4]);
	const T& La200 =  (La[3]);
	Lb(0,0,0) = La(0,0,0);
	Lb(1,0,0) = La(1,0,0);
	Lb(0,1,0) = La(0,1,0);
	Lb(0,0,1) = La(0,0,1);
	if( do_phi ) {
		Lb[0] = fmaf( x001, La001, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x002, La002, Lb[0]);
		Lb[0] = fmaf( x010, La010, Lb[0]);
		Lb[0] = fmaf( x011, La011, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x020, La020, Lb[0]);
		Lb[0] = fmaf( x100, La100, Lb[0]);
		Lb[0] = fmaf( x101, La101, Lb[0]);
		Lb[0] = fmaf( x110, La110, Lb[0]);
		Lb[0] = fmaf(T(5.00000000e-01) * x200, La200, Lb[0]);
	}
	Lb[1] = fmaf( x100, La200, Lb[1]);
	Lb[2] = fmaf( x001, La011, Lb[2]);
	Lb[1] = fmaf( x010, La110, Lb[1]);
	Lb[3] = fmaf( x100, La101, Lb[3]);
	Lb[1] = fmaf( x001, La101, Lb[1]);
	Lb[3] = fmaf( x010, La011, Lb[3]);
	Lb[2] = fmaf( x100, La110, Lb[2]);
	Lb[3] = fmaf( x001, La002, Lb[3]);
	Lb[2] = fmaf( x010, La020, Lb[2]);
	return Lb;
/* FLOPS = 45*/
}
#ifdef EXPANSION_CU
__managed__ char Ldest1[5] = { 1,1,1,2,2};
__managed__ float factor1[5] = { 1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00};
__managed__ char xsrc1[5] = { 1,2,3,1,2};
__managed__ char Lsrc1[5] = { 4,5,6,5,7};
__managed__ char Ldest2[4] = { 2,6,6,6};
__managed__ float factor2[4] = { 1.00000000e+00,1.00000000e+00,1.00000000e+00,1.00000000e+00};
__managed__ char xsrc2[4] = { 3,1,2,3};
__managed__ char Lsrc2[4] = { 8,6,8,9};
__managed__ float phi_factor[9] = { 1.00000000e+00,5.00000000e-01,1.00000000e+00,1.00000000e+00,5.00000000e-01,1.00000000e+00,1.00000000e+00,1.00000000e+00,5.00000000e-01};
__managed__ char phi_Lsrc[9] = { 3,9,2,8,7,1,6,5,4};
#ifdef __CUDA_ARCH__
template<class T>
__device__
tensor_trless_sym<T, 3> L2L_cuda(const tensor_trless_sym<T, 3>& La, const array<T, NDIM>& X, bool do_phi) {
	const int tid = threadIdx.x;
	tensor_trless_sym<T, 3> Lb;
	tensor_sym<T, 3> Lc;
	Lb = 0.0f;
	for( int i = tid; i < EXPANSION_SIZE; i += KICK_BLOCK_SIZE ) {
		Lb[i] = La[i];
	}
	tensor_sym<T,3> dx;
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
	Lc[0] =  (La[0]);
	Lc[3] =  (La[6]);
	Lc[9] =  (La[9]);
	Lc[2] =  (La[2]);
	Lc[8] =  (La[8]);
	Lc[7] =  (La[5]);
	Lc[1] =  (La[1]);
	Lc[6] =  (La[7]);
	Lc[5] =  (La[4]);
	Lc[4] =  (La[3]);
	for( int i = tid; i < 4; i+=KICK_BLOCK_SIZE) {
		Lb[Ldest1[i]] = fmaf(factor1[i] * dx[xsrc1[i]], Lc[Lsrc1[i]], Lb[Ldest1[i]]);
		Lb[Ldest2[i]] = fmaf(factor2[i] * dx[xsrc2[i]], Lc[Lsrc2[i]], Lb[Ldest2[i]]);
	}
	if( tid == 0 ) {
		Lb[Ldest1[4]] = fmaf(factor1[4] * dx[xsrc1[4]], Lc[Lsrc1[4]], Lb[Ldest1[4]]);
	}
	if( do_phi ) {
		for( int i = tid; i < 9; i+=KICK_BLOCK_SIZE) {
			Lb[0] = fmaf(phi_factor[i] * dx[phi_Lsrc[i]], Lc[phi_Lsrc[i]], Lb[0]);
		}
	}
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			Lb[i] += __shfl_xor_sync(0xffffffff, Lb[i], P);
		}
	}
	return Lb;
/* FLOPS = 42 + do_phi * 36*/
}
#endif
#endif
