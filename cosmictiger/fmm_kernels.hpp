/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/
#pragma once
#include <cosmictiger/tensor.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/ewald_indices.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/treepm_kernels.hpp>
template<class T>
using expansion = tensor_sym<T,4>;
template<class T>
using expansion2 = tensor_sym<T,2>;
template<class T>
using multipole = tensor_sym<T,3>;
#define EXPANSION_SIZE 20
#define MULTIPOLE_SIZE 10


template<class T>
CUDA_EXPORT
inline int greens_function(tensor_sym<T, 4>& D, array<T, NDIM> dx, float rsinv, float rsinv2) {
	D = T(0);
	T r2 = fmaf(dx[0], dx[0], fmaf(dx[1], dx[1], sqr(dx[2])));
	const T r = sqrt(r2);
	const T rinv = 1.f / r;
	const T rinv0 = T(1);
	const T rinv1 = rinv;
	const auto d = green_kernel( r, rsinv, rsinv2 );
	const T Drinvpow_0_0 = d[0] * rinv0;
	const T Drinvpow_1_0 = d[1] * rinv0;
	const T Drinvpow_1_1 = d[1] * rinv1;
	const T Drinvpow_2_0 = d[2] * rinv0;
	const T Drinvpow_2_1 = d[2] * rinv1;
	const T Drinvpow_3_0 = d[3] * rinv0;
	array<T,NDIM> dxrinv;
	dxrinv[0] = dx[0] * rinv;
	dxrinv[1] = dx[1] * rinv;
	dxrinv[2] = dx[2] * rinv;
	T x[20];
	x[0] = T(1);
	x[1] = dxrinv[0];
	x[2] = dxrinv[1];
	x[3] = dxrinv[2];
	x[9] = x[3] * x[3];
	x[8] = x[2] * x[3];
	x[7] = x[2] * x[2];
	x[6] = x[1] * x[3];
	x[5] = x[1] * x[2];
	x[4] = x[1] * x[1];
	x[19] = x[9] * x[3];
	x[18] = x[8] * x[3];
	x[17] = x[8] * x[2];
	x[16] = x[7] * x[2];
	x[15] = x[6] * x[3];
	x[14] = x[5] * x[3];
	x[13] = x[5] * x[2];
	x[12] = x[6] * x[1];
	x[11] = x[5] * x[1];
	x[10] = x[4] * x[1];
	D[0] = fmaf(x[0], Drinvpow_0_0, D[0]);
	D[3] = fmaf(x[3], Drinvpow_1_0, D[3]);
	D[9] = fmaf(x[9], Drinvpow_2_0, D[9]);
	D[9] = fmaf(x[0], Drinvpow_1_1, D[9]);
	D[19] = fmaf(x[19], Drinvpow_3_0, D[19]);
	D[19] = fmaf(T(3.000000000e+00), x[3]*Drinvpow_2_1, D[19]);
	D[2] = fmaf(x[2], Drinvpow_1_0, D[2]);
	D[8] = fmaf(x[8], Drinvpow_2_0, D[8]);
	D[18] = fmaf(x[18], Drinvpow_3_0, D[18]);
	D[18] = fmaf(x[2], Drinvpow_2_1, D[18]);
	D[7] = fmaf(x[7], Drinvpow_2_0, D[7]);
	D[7] = fmaf(x[0], Drinvpow_1_1, D[7]);
	D[17] = fmaf(x[17], Drinvpow_3_0, D[17]);
	D[17] = fmaf(x[3], Drinvpow_2_1, D[17]);
	D[16] = fmaf(x[16], Drinvpow_3_0, D[16]);
	D[16] = fmaf(T(3.000000000e+00), x[2]*Drinvpow_2_1, D[16]);
	D[1] = fmaf(x[1], Drinvpow_1_0, D[1]);
	D[6] = fmaf(x[6], Drinvpow_2_0, D[6]);
	D[15] = fmaf(x[15], Drinvpow_3_0, D[15]);
	D[15] = fmaf(x[1], Drinvpow_2_1, D[15]);
	D[5] = fmaf(x[5], Drinvpow_2_0, D[5]);
	D[14] = fmaf(x[14], Drinvpow_3_0, D[14]);
	D[13] = fmaf(x[13], Drinvpow_3_0, D[13]);
	D[13] = fmaf(x[1], Drinvpow_2_1, D[13]);
	D[4] = fmaf(x[4], Drinvpow_2_0, D[4]);
	D[4] = fmaf(x[0], Drinvpow_1_1, D[4]);
	D[12] = fmaf(x[12], Drinvpow_3_0, D[12]);
	D[12] = fmaf(x[3], Drinvpow_2_1, D[12]);
	D[11] = fmaf(x[11], Drinvpow_3_0, D[11]);
	D[11] = fmaf(x[2], Drinvpow_2_1, D[11]);
	D[10] = fmaf(x[10], Drinvpow_3_0, D[10]);
	D[10] = fmaf(T(3.000000000e+00), x[1]*Drinvpow_2_1, D[10]);
	return 0; 
}


template<class T>
CUDA_EXPORT
inline int M2L(tensor_sym<T, 2>& L, const tensor_sym<T, 3>& M, const tensor_sym<T, 4>& D, bool do_phi) {
	if( do_phi ) {
		L[0] *= T(2.000000000e+00);
		L[0] = fmaf(M[9], D[9], L[0]);
		L[0] = fmaf(M[7], D[7], L[0]);
		L[0] = fmaf(M[4], D[4], L[0]);
		L[0] *= T(5.000000000e-01);
		L[0] = fmaf(M[0], D[0], L[0]);
		L[0] = fmaf(M[3], D[3], L[0]);
		L[0] = fmaf(M[2], D[2], L[0]);
		L[0] = fmaf(M[8], D[8], L[0]);
		L[0] = fmaf(M[1], D[1], L[0]);
		L[0] = fmaf(M[6], D[6], L[0]);
		L[0] = fmaf(M[5], D[5], L[0]);
	}
	L[1] *= T(2.000000000e+00);
	L[2] *= T(2.000000000e+00);
	L[1] = fmaf(M[9], D[15], L[1]);
	L[2] = fmaf(M[9], D[18], L[2]);
	L[1] = fmaf(M[7], D[13], L[1]);
	L[2] = fmaf(M[7], D[16], L[2]);
	L[1] = fmaf(M[4], D[10], L[1]);
	L[2] = fmaf(M[4], D[11], L[2]);
	L[1] *= T(5.000000000e-01);
	L[2] *= T(5.000000000e-01);
	L[1] = fmaf(M[0], D[1], L[1]);
	L[2] = fmaf(M[0], D[2], L[2]);
	L[1] = fmaf(M[3], D[6], L[1]);
	L[2] = fmaf(M[3], D[8], L[2]);
	L[1] = fmaf(M[2], D[5], L[1]);
	L[2] = fmaf(M[2], D[7], L[2]);
	L[1] = fmaf(M[8], D[14], L[1]);
	L[2] = fmaf(M[8], D[17], L[2]);
	L[1] = fmaf(M[1], D[4], L[1]);
	L[2] = fmaf(M[1], D[5], L[2]);
	L[1] = fmaf(M[6], D[12], L[1]);
	L[2] = fmaf(M[6], D[14], L[2]);
	L[1] = fmaf(M[5], D[11], L[1]);
	L[2] = fmaf(M[5], D[13], L[2]);
	L[3] *= T(2.000000000e+00);
	L[3] = fmaf(M[9], D[19], L[3]);
	L[3] = fmaf(M[7], D[17], L[3]);
	L[3] = fmaf(M[4], D[12], L[3]);
	L[3] *= T(5.000000000e-01);
	L[3] = fmaf(M[0], D[3], L[3]);
	L[3] = fmaf(M[3], D[9], L[3]);
	L[3] = fmaf(M[2], D[8], L[3]);
	L[3] = fmaf(M[8], D[18], L[3]);
	L[3] = fmaf(M[1], D[6], L[3]);
	L[3] = fmaf(M[6], D[15], L[3]);
	L[3] = fmaf(M[5], D[14], L[3]);
	return 66 + do_phi * 22;
}


template<class T>
CUDA_EXPORT
inline int M2L(tensor_sym<T, 4>& L, const tensor_sym<T, 3>& M, const tensor_sym<T, 4>& D, bool do_phi) {
	if( do_phi ) {
		L[0] *= T(2.000000000e+00);
		L[0] = fmaf(M[9], D[9], L[0]);
		L[0] = fmaf(M[7], D[7], L[0]);
		L[0] = fmaf(M[4], D[4], L[0]);
		L[0] *= T(5.000000000e-01);
		L[0] = fmaf(M[0], D[0], L[0]);
		L[0] = fmaf(M[3], D[3], L[0]);
		L[0] = fmaf(M[2], D[2], L[0]);
		L[0] = fmaf(M[8], D[8], L[0]);
		L[0] = fmaf(M[1], D[1], L[0]);
		L[0] = fmaf(M[6], D[6], L[0]);
		L[0] = fmaf(M[5], D[5], L[0]);
	}
	L[1] *= T(2.000000000e+00);
	L[4] = fmaf(M[0], D[4], L[4]);
	L[1] = fmaf(M[9], D[15], L[1]);
	L[4] = fmaf(M[3], D[12], L[4]);
	L[1] = fmaf(M[7], D[13], L[1]);
	L[4] = fmaf(M[2], D[11], L[4]);
	L[1] = fmaf(M[4], D[10], L[1]);
	L[4] = fmaf(M[1], D[10], L[4]);
	L[1] *= T(5.000000000e-01);
	L[5] = fmaf(M[0], D[5], L[5]);
	L[1] = fmaf(M[0], D[1], L[1]);
	L[5] = fmaf(M[3], D[14], L[5]);
	L[1] = fmaf(M[3], D[6], L[1]);
	L[5] = fmaf(M[2], D[13], L[5]);
	L[1] = fmaf(M[2], D[5], L[1]);
	L[5] = fmaf(M[1], D[11], L[5]);
	L[1] = fmaf(M[8], D[14], L[1]);
	L[6] = fmaf(M[0], D[6], L[6]);
	L[1] = fmaf(M[1], D[4], L[1]);
	L[6] = fmaf(M[3], D[15], L[6]);
	L[1] = fmaf(M[6], D[12], L[1]);
	L[6] = fmaf(M[2], D[14], L[6]);
	L[1] = fmaf(M[5], D[11], L[1]);
	L[6] = fmaf(M[1], D[12], L[6]);
	L[2] *= T(2.000000000e+00);
	L[7] = fmaf(M[0], D[7], L[7]);
	L[2] = fmaf(M[9], D[18], L[2]);
	L[7] = fmaf(M[3], D[17], L[7]);
	L[2] = fmaf(M[7], D[16], L[2]);
	L[7] = fmaf(M[2], D[16], L[7]);
	L[2] = fmaf(M[4], D[11], L[2]);
	L[7] = fmaf(M[1], D[13], L[7]);
	L[2] *= T(5.000000000e-01);
	L[8] = fmaf(M[0], D[8], L[8]);
	L[2] = fmaf(M[0], D[2], L[2]);
	L[8] = fmaf(M[3], D[18], L[8]);
	L[2] = fmaf(M[3], D[8], L[2]);
	L[8] = fmaf(M[2], D[17], L[8]);
	L[2] = fmaf(M[2], D[7], L[2]);
	L[8] = fmaf(M[1], D[14], L[8]);
	L[2] = fmaf(M[8], D[17], L[2]);
	L[9] = fmaf(M[0], D[9], L[9]);
	L[2] = fmaf(M[1], D[5], L[2]);
	L[9] = fmaf(M[3], D[19], L[9]);
	L[2] = fmaf(M[6], D[14], L[2]);
	L[9] = fmaf(M[2], D[18], L[9]);
	L[2] = fmaf(M[5], D[13], L[2]);
	L[9] = fmaf(M[1], D[15], L[9]);
	L[3] *= T(2.000000000e+00);
	L[10] = fmaf(M[0], D[10], L[10]);
	L[3] = fmaf(M[9], D[19], L[3]);
	L[11] = fmaf(M[0], D[11], L[11]);
	L[3] = fmaf(M[7], D[17], L[3]);
	L[12] = fmaf(M[0], D[12], L[12]);
	L[3] = fmaf(M[4], D[12], L[3]);
	L[13] = fmaf(M[0], D[13], L[13]);
	L[3] *= T(5.000000000e-01);
	L[14] = fmaf(M[0], D[14], L[14]);
	L[3] = fmaf(M[0], D[3], L[3]);
	L[15] = fmaf(M[0], D[15], L[15]);
	L[3] = fmaf(M[3], D[9], L[3]);
	L[16] = fmaf(M[0], D[16], L[16]);
	L[3] = fmaf(M[2], D[8], L[3]);
	L[17] = fmaf(M[0], D[17], L[17]);
	L[3] = fmaf(M[8], D[18], L[3]);
	L[18] = fmaf(M[0], D[18], L[18]);
	L[3] = fmaf(M[1], D[6], L[3]);
	L[19] = fmaf(M[0], D[19], L[19]);
	L[3] = fmaf(M[6], D[15], L[3]);
	L[3] = fmaf(M[5], D[14], L[3]);
	return 134 + do_phi * 22;
}


template<class T>
CUDA_EXPORT
tensor_sym<T, 3> P2M(array<T, NDIM>& X) {
	tensor_sym<T, 3> M;
	X[0] *= -T(1);
	X[1] *= -T(1);
	X[2] *= -T(1);
	M[0] = T(1);
	M[1] = X[0];
	M[2] = X[1];
	M[3] = X[2];
	M[9] = M[3] * M[3];
	M[8] = M[2] * M[3];
	M[7] = M[2] * M[2];
	M[6] = M[1] * M[3];
	M[5] = M[1] * M[2];
	M[4] = M[1] * M[1];
	return M;
}


template<class T>
CUDA_EXPORT
tensor_sym<T, 3> M2M(const tensor_sym<T,3>& Ma, array<T, NDIM>& X) {
	auto Mb = Ma;
	X[0] *= -T(1);
	X[1] *= -T(1);
	X[2] *= -T(1);
	T x[10];
	x[0] = T(1);
	x[1] = X[0];
	x[2] = X[1];
	x[3] = X[2];
	x[9] = x[3] * x[3];
	x[8] = x[2] * x[3];
	x[7] = x[2] * x[2];
	x[6] = x[1] * x[3];
	x[5] = x[1] * x[2];
	x[4] = x[1] * x[1];
	Mb[1] = fmaf( x[1], Ma[0], Mb[1]);
	Mb[2] = fmaf( x[2], Ma[0], Mb[2]);
	Mb[3] = fmaf( x[3], Ma[0], Mb[3]);
	Mb[4] *= T(5.000000000e-01);
	Mb[4] = fmaf( x[1], Ma[1], Mb[4]);
	Mb[4] *= T(2.000000000e+00);
	Mb[4] = fmaf( x[4], Ma[0], Mb[4]);
	Mb[5] = fmaf( x[5], Ma[0], Mb[5]);
	Mb[5] = fmaf( x[1], Ma[2], Mb[5]);
	Mb[5] = fmaf( x[2], Ma[1], Mb[5]);
	Mb[6] = fmaf( x[6], Ma[0], Mb[6]);
	Mb[6] = fmaf( x[1], Ma[3], Mb[6]);
	Mb[6] = fmaf( x[3], Ma[1], Mb[6]);
	Mb[7] *= T(5.000000000e-01);
	Mb[7] = fmaf( x[2], Ma[2], Mb[7]);
	Mb[7] *= T(2.000000000e+00);
	Mb[7] = fmaf( x[7], Ma[0], Mb[7]);
	Mb[8] = fmaf( x[8], Ma[0], Mb[8]);
	Mb[8] = fmaf( x[2], Ma[3], Mb[8]);
	Mb[8] = fmaf( x[3], Ma[2], Mb[8]);
	Mb[9] *= T(5.000000000e-01);
	Mb[9] = fmaf( x[3], Ma[3], Mb[9]);
	Mb[9] *= T(2.000000000e+00);
	Mb[9] = fmaf( x[9], Ma[0], Mb[9]);
	return Mb;

}

template<class T>
CUDA_EXPORT
#ifdef __CUDACC__
__noinline__
#endif
tensor_sym<T, 4> L2L(const tensor_sym<T, 4>& La, array<T,NDIM> X, bool do_phi) {
	tensor_sym<T, 4> Lb;
	T x[20];
	x[0] = T(1);
	x[1] = X[0];
	x[2] = X[1];
	x[3] = X[2];
	x[9] = x[3] * x[3];
	x[8] = x[2] * x[3];
	x[7] = x[2] * x[2];
	x[6] = x[1] * x[3];
	x[5] = x[1] * x[2];
	x[4] = x[1] * x[1];
	x[19] = x[9] * x[3];
	x[18] = x[8] * x[3];
	x[17] = x[8] * x[2];
	x[16] = x[7] * x[2];
	x[15] = x[6] * x[3];
	x[14] = x[5] * x[3];
	x[13] = x[5] * x[2];
	x[12] = x[6] * x[1];
	x[11] = x[5] * x[1];
	x[10] = x[4] * x[1];
	Lb = La;
	if( do_phi ) {
		Lb[0] *= T(6.000000000e+00);
		Lb[0] = fmaf( x[10], La[10], Lb[0]);
		Lb[0] = fmaf( x[19], La[19], Lb[0]);
		Lb[0] = fmaf( x[16], La[16], Lb[0]);
		Lb[0] *= T(3.333333333e-01);
		Lb[0] = fmaf( x[9], La[9], Lb[0]);
		Lb[0] = fmaf( x[11], La[11], Lb[0]);
		Lb[0] = fmaf( x[12], La[12], Lb[0]);
		Lb[0] = fmaf( x[4], La[4], Lb[0]);
		Lb[0] = fmaf( x[13], La[13], Lb[0]);
		Lb[0] = fmaf( x[15], La[15], Lb[0]);
		Lb[0] = fmaf( x[17], La[17], Lb[0]);
		Lb[0] = fmaf( x[7], La[7], Lb[0]);
		Lb[0] = fmaf( x[18], La[18], Lb[0]);
		Lb[0] *= T(5.000000000e-01);
		Lb[0] = fmaf( x[1], La[1], Lb[0]);
		Lb[0] = fmaf( x[6], La[6], Lb[0]);
		Lb[0] = fmaf( x[5], La[5], Lb[0]);
		Lb[0] = fmaf( x[14], La[14], Lb[0]);
		Lb[0] = fmaf( x[8], La[8], Lb[0]);
		Lb[0] = fmaf( x[2], La[2], Lb[0]);
		Lb[0] = fmaf( x[3], La[3], Lb[0]);
	}
	Lb[1] *= T(2.000000000e+00);
	Lb[1] = fmaf( x[9], La[15], Lb[1]);
	Lb[1] = fmaf( x[7], La[13], Lb[1]);
	Lb[1] = fmaf( x[4], La[10], Lb[1]);
	Lb[1] *= T(5.000000000e-01);
	Lb[1] = fmaf( x[3], La[6], Lb[1]);
	Lb[1] = fmaf( x[2], La[5], Lb[1]);
	Lb[1] = fmaf( x[8], La[14], Lb[1]);
	Lb[1] = fmaf( x[1], La[4], Lb[1]);
	Lb[1] = fmaf( x[6], La[12], Lb[1]);
	Lb[1] = fmaf( x[5], La[11], Lb[1]);
	Lb[2] *= T(2.000000000e+00);
	Lb[2] = fmaf( x[9], La[18], Lb[2]);
	Lb[2] = fmaf( x[7], La[16], Lb[2]);
	Lb[2] = fmaf( x[4], La[11], Lb[2]);
	Lb[2] *= T(5.000000000e-01);
	Lb[2] = fmaf( x[3], La[8], Lb[2]);
	Lb[2] = fmaf( x[2], La[7], Lb[2]);
	Lb[2] = fmaf( x[8], La[17], Lb[2]);
	Lb[2] = fmaf( x[1], La[5], Lb[2]);
	Lb[2] = fmaf( x[6], La[14], Lb[2]);
	Lb[2] = fmaf( x[5], La[13], Lb[2]);
	Lb[3] *= T(2.000000000e+00);
	Lb[3] = fmaf( x[9], La[19], Lb[3]);
	Lb[3] = fmaf( x[7], La[17], Lb[3]);
	Lb[3] = fmaf( x[4], La[12], Lb[3]);
	Lb[3] *= T(5.000000000e-01);
	Lb[3] = fmaf( x[3], La[9], Lb[3]);
	Lb[3] = fmaf( x[2], La[8], Lb[3]);
	Lb[3] = fmaf( x[8], La[18], Lb[3]);
	Lb[3] = fmaf( x[1], La[6], Lb[3]);
	Lb[3] = fmaf( x[6], La[15], Lb[3]);
	Lb[3] = fmaf( x[5], La[14], Lb[3]);
	Lb[4] = fmaf( x[3], La[12], Lb[4]);
	Lb[4] = fmaf( x[2], La[11], Lb[4]);
	Lb[4] = fmaf( x[1], La[10], Lb[4]);
	Lb[5] = fmaf( x[3], La[14], Lb[5]);
	Lb[5] = fmaf( x[2], La[13], Lb[5]);
	Lb[5] = fmaf( x[1], La[11], Lb[5]);
	Lb[6] = fmaf( x[3], La[15], Lb[6]);
	Lb[6] = fmaf( x[2], La[14], Lb[6]);
	Lb[6] = fmaf( x[1], La[12], Lb[6]);
	Lb[7] = fmaf( x[3], La[17], Lb[7]);
	Lb[7] = fmaf( x[2], La[16], Lb[7]);
	Lb[7] = fmaf( x[1], La[13], Lb[7]);
	Lb[8] = fmaf( x[3], La[18], Lb[8]);
	Lb[8] = fmaf( x[2], La[17], Lb[8]);
	Lb[8] = fmaf( x[1], La[14], Lb[8]);
	Lb[9] = fmaf( x[3], La[19], Lb[9]);
	Lb[9] = fmaf( x[2], La[18], Lb[9]);
	Lb[9] = fmaf( x[1], La[15], Lb[9]);
	return Lb;
/* FLOPS = 115 + do_phi * 42*/
}

template<class T>
CUDA_EXPORT
#ifdef __CUDACC__
__noinline__
#endif
tensor_sym<T, 2> L2P(const tensor_sym<T, 4>& La, array<T,NDIM> X, bool do_phi) {
	tensor_sym<T, 2> Lb;
	T x[20];
	x[0] = T(1);
	x[1] = X[0];
	x[2] = X[1];
	x[3] = X[2];
	x[9] = x[3] * x[3];
	x[8] = x[2] * x[3];
	x[7] = x[2] * x[2];
	x[6] = x[1] * x[3];
	x[5] = x[1] * x[2];
	x[4] = x[1] * x[1];
	x[19] = x[9] * x[3];
	x[18] = x[8] * x[3];
	x[17] = x[8] * x[2];
	x[16] = x[7] * x[2];
	x[15] = x[6] * x[3];
	x[14] = x[5] * x[3];
	x[13] = x[5] * x[2];
	x[12] = x[6] * x[1];
	x[11] = x[5] * x[1];
	x[10] = x[4] * x[1];
	Lb[0] = La[0];
	Lb[1] = La[1];
	Lb[2] = La[2];
	Lb[3] = La[3];
	if( do_phi ) {
		Lb[0] *= T(6.000000000e+00);
		Lb[0] = fmaf( x[10], La[10], Lb[0]);
		Lb[0] = fmaf( x[19], La[19], Lb[0]);
		Lb[0] = fmaf( x[16], La[16], Lb[0]);
		Lb[0] *= T(3.333333333e-01);
		Lb[0] = fmaf( x[9], La[9], Lb[0]);
		Lb[0] = fmaf( x[11], La[11], Lb[0]);
		Lb[0] = fmaf( x[12], La[12], Lb[0]);
		Lb[0] = fmaf( x[4], La[4], Lb[0]);
		Lb[0] = fmaf( x[13], La[13], Lb[0]);
		Lb[0] = fmaf( x[15], La[15], Lb[0]);
		Lb[0] = fmaf( x[17], La[17], Lb[0]);
		Lb[0] = fmaf( x[7], La[7], Lb[0]);
		Lb[0] = fmaf( x[18], La[18], Lb[0]);
		Lb[0] *= T(5.000000000e-01);
		Lb[0] = fmaf( x[1], La[1], Lb[0]);
		Lb[0] = fmaf( x[6], La[6], Lb[0]);
		Lb[0] = fmaf( x[5], La[5], Lb[0]);
		Lb[0] = fmaf( x[14], La[14], Lb[0]);
		Lb[0] = fmaf( x[8], La[8], Lb[0]);
		Lb[0] = fmaf( x[2], La[2], Lb[0]);
		Lb[0] = fmaf( x[3], La[3], Lb[0]);
	}
	Lb[1] *= T(2.000000000e+00);
	Lb[1] = fmaf( x[9], La[15], Lb[1]);
	Lb[1] = fmaf( x[7], La[13], Lb[1]);
	Lb[1] = fmaf( x[4], La[10], Lb[1]);
	Lb[1] *= T(5.000000000e-01);
	Lb[1] = fmaf( x[3], La[6], Lb[1]);
	Lb[1] = fmaf( x[2], La[5], Lb[1]);
	Lb[1] = fmaf( x[8], La[14], Lb[1]);
	Lb[1] = fmaf( x[1], La[4], Lb[1]);
	Lb[1] = fmaf( x[6], La[12], Lb[1]);
	Lb[1] = fmaf( x[5], La[11], Lb[1]);
	Lb[2] *= T(2.000000000e+00);
	Lb[2] = fmaf( x[9], La[18], Lb[2]);
	Lb[2] = fmaf( x[7], La[16], Lb[2]);
	Lb[2] = fmaf( x[4], La[11], Lb[2]);
	Lb[2] *= T(5.000000000e-01);
	Lb[2] = fmaf( x[3], La[8], Lb[2]);
	Lb[2] = fmaf( x[2], La[7], Lb[2]);
	Lb[2] = fmaf( x[8], La[17], Lb[2]);
	Lb[2] = fmaf( x[1], La[5], Lb[2]);
	Lb[2] = fmaf( x[6], La[14], Lb[2]);
	Lb[2] = fmaf( x[5], La[13], Lb[2]);
	Lb[3] *= T(2.000000000e+00);
	Lb[3] = fmaf( x[9], La[19], Lb[3]);
	Lb[3] = fmaf( x[7], La[17], Lb[3]);
	Lb[3] = fmaf( x[4], La[12], Lb[3]);
	Lb[3] *= T(5.000000000e-01);
	Lb[3] = fmaf( x[3], La[9], Lb[3]);
	Lb[3] = fmaf( x[2], La[8], Lb[3]);
	Lb[3] = fmaf( x[8], La[18], Lb[3]);
	Lb[3] = fmaf( x[1], La[6], Lb[3]);
	Lb[3] = fmaf( x[6], La[15], Lb[3]);
	Lb[3] = fmaf( x[5], La[14], Lb[3]);
	return Lb;
/* FLOPS = 79 + do_phi * 42*/
}
static __device__ char Ldest1[45] = { 1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9};
static __constant__ float factor1[45] = { float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(5.000000000e-01),float(1.000000000e+00),float(1.000000000e+00),float(5.000000000e-01),float(1.000000000e+00),float(5.000000000e-01),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(5.000000000e-01),float(1.000000000e+00),float(1.000000000e+00),float(5.000000000e-01),float(1.000000000e+00),float(5.000000000e-01),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(5.000000000e-01),float(1.000000000e+00),float(1.000000000e+00),float(5.000000000e-01),float(1.000000000e+00),float(5.000000000e-01),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00),float(1.000000000e+00)};
static __constant__ char xsrc1[45] = { 1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3};
static __constant__ char Lsrc1[45] = { 4,5,6,10,11,12,13,14,15,5,7,8,11,13,14,16,17,18,6,8,9,12,14,15,17,18,19,10,11,12,11,13,14,12,14,15,13,16,17,14,17,18,15,18,19};
static __constant__ float phi_factor[19] = { float(1.000000000e+00),float(5.000000000e-01),float(1.666666716e-01),float(1.000000000e+00),float(1.000000000e+00),float(5.000000000e-01),float(5.000000000e-01),float(5.000000000e-01),float(1.666666716e-01),float(1.000000000e+00),float(1.000000000e+00),float(5.000000000e-01),float(1.000000000e+00),float(1.000000000e+00),float(5.000000000e-01),float(5.000000000e-01),float(5.000000000e-01),float(5.000000000e-01),float(1.666666716e-01)};
static __constant__ char phi_Lsrc[19] = { 3,9,19,2,8,18,7,17,16,1,6,15,5,14,13,4,12,11,10};
#ifdef __CUDACC__
template<class T>
__device__
tensor_sym<T, 4> L2L_cuda(const tensor_sym<T, 4>& La, array<T,NDIM> X, bool do_phi) {
	const int tid = threadIdx.x;
	tensor_sym<T, 4> Lb;
	for( int i = 0; i < EXPANSION_SIZE; i ++ ) {
		Lb[i] = 0.0f;
	}
	for( int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE ) {
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
	for( int i = tid; i < 45; i+=WARP_SIZE) {
		Lb[Ldest1[i]] = fmaf(factor1[i] * dx[xsrc1[i]], La[Lsrc1[i]], Lb[Ldest1[i]]);
	}
	if( do_phi ) {
		for( int i = tid; i < 19; i+=WARP_SIZE) {
			Lb[0] = fmaf(phi_factor[i] * dx[phi_Lsrc[i]], La[phi_Lsrc[i]], Lb[0]);
		}
	}
	for (int P = warpSize / 2; P >= 1; P /= 2) {
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			Lb[i] += __shfl_xor_sync(0xffffffff, Lb[i], P);
		}
	}
	return Lb;
/* FLOPS = 19 + do_phi * 76*/
}
#endif
