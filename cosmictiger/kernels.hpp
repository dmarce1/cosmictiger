#pragma once

CUDA_EXPORT inline void green_direct(float& phi, float& f, float r, float r2, float rinv, float rsinv, float rsinv2) {
	const float q = r * rsinv;
	float q2 = sqr(q);
	f = phi = 0.f;
	if (q2 < 1.f) {
		phi = rinv;
		f = rinv * sqr(rinv);
		float y;
		y = float(-2.46093750e-01);
		y = fma( y, q2, float(1.50390625e+00) );
		y = fma( y, q2, float(-3.86718750e+00) );
		y = fma( y, q2, float(5.41406250e+00) );
		y = fma( y, q2, float(-4.51171875e+00) );
		y = fma( y, q2, float(2.70703125e+00) );
		phi -= y * rsinv;
		y = float(2.46093750e+00);
		y = fma( y, q2, float(-1.20312500e+01) );
		y = fma( y, q2, float(2.32031250e+01) );
		y = fma( y, q2, float(-2.16562500e+01) );
		y = fma( y, q2, float(9.02343750e+00) );
		f -= y * sqr(rsinv) * rsinv;
	}
}


CUDA_EXPORT inline array<float, 7> green_kernel(float r, float rsinv, float rsinv2, bool do_phi) {
	array<float, 7> d;
	float q0 = 1.f;
	float q1 = r * rsinv;
	float& q = q1;
	float qinv = 1.f / q1;
	float q2 = sqr(q1);
	float q3 = q1 * q2;
	float q4 = q1 * q3;
	float q5 = q1 * q4;
	float q6 = q1 * q5;
	float rsinv1 = rsinv;
	float rsinv3 = rsinv * rsinv2;
	float rsinv4 = rsinv * rsinv3;
	float rsinv5 = rsinv * rsinv4;
	float rsinv6 = rsinv * rsinv5;
	float rsinv7 = rsinv * rsinv6;
	float rinv1 = 1.f / r;
	float rinv2 = rinv1 * rinv1;
	float rinv3 = rinv1 * rinv2;
	float rinv4 = rinv1 * rinv3;
	float rinv5 = rinv1 * rinv4;
	float rinv6 = rinv1 * rinv5;
	float rinv7 = rinv1 * rinv6;
	if (q2 < 1.f) {
		float y;
		float z;
		if( do_phi ) {
			y = float(-2.46093750e-01);
			y = fma( y, q2, float(1.50390625e+00) );
			y = fma( y, q2, float(-3.86718750e+00) );
			y = fma( y, q2, float(5.41406250e+00) );
			y = fma( y, q2, float(-4.51171875e+00) );
			y = fma( y, q2, float(2.70703125e+00) );
			y *= q0 * rsinv1;
			d[0] = fmaf( float(-1), rinv1, y);
		}
		y = float(-2.46093750e+00);
		y = fma( y, q2, float(1.20312500e+01) );
		y = fma( y, q2, float(-2.32031250e+01) );
		y = fma( y, q2, float(2.16562500e+01) );
		y = fma( y, q2, float(-9.02343750e+00) );
		y *= q1 * rsinv2;
		d[1] = fmaf( float(1), rinv2, y);
		y = float(-1.96875000e+01);
		y = fma( y, q2, float(7.21875000e+01) );
		y = fma( y, q2, float(-9.28125000e+01) );
		y = fma( y, q2, float(4.33125000e+01) );
		y *= q2 * rsinv3;
		d[2] = fmaf( float(-3), rinv3, y);
		y = float(-1.18125000e+02);
		y = fma( y, q2, float(2.88750000e+02) );
		y = fma( y, q2, float(-1.85625000e+02) );
		y *= q3 * rsinv4;
		d[3] = fmaf( float(15), rinv4, y);
		y = float(-4.72500000e+02);
		y = fma( y, q2, float(5.77500000e+02) );
		y *= q4 * rsinv5;
		d[4] = fmaf( float(-105), rinv5, y);
		y = float(-9.45000000e+02);
		y *= q5 * rsinv6;
		d[5] = fmaf( float(945), rinv6, y);
		y = 0.f;
		y *= q6 * rsinv7;
		d[6] = fmaf( float(-10395), rinv7, y);
	} else {
		d[0] = 0.f;
		d[1] = 0.f;
		d[2] = 0.f;
		d[3] = 0.f;
		d[4] = 0.f;
		d[5] = 0.f;
		d[6] = 0.f;
	}
	 return d;
}


CUDA_EXPORT inline float green_phi0(float nparts, float rs) {
	return float(4.83321947e-01) * sqr(rs) * (nparts - 1) +  float(0.00000000e+00) / rs;
}


CUDA_EXPORT inline float green_rho(float q) {
	float y;
	float q2 = sqr(q);
	y = float(2.15418702e+00);
	y = fma( y, q2, float(-8.61674809e+00) );
	y = fma( y, q2, float(1.29251221e+01) );
	y = fma( y, q2, float(-8.61674809e+00) );
	y = fma( y, q2, float(2.15418702e+00) );
	return y;
}


CUDA_EXPORT inline void gsoft(float& f, float& phi, float q2, float hinv, float h2inv, float h3inv, bool do_phi) {
	q2 *= h2inv;
	f = phi = 0.f;
	if (q2 < 1.f) {
		float y;
		if( do_phi ) {
			y = float(-2.46093750e-01);
			y = fma( y, q2, float(1.50390625e+00) );
			y = fma( y, q2, float(-3.86718750e+00) );
			y = fma( y, q2, float(5.41406250e+00) );
			y = fma( y, q2, float(-4.51171875e+00) );
			y = fma( y, q2, float(2.70703125e+00) );
			phi = y * hinv;
		}
		y = float(2.46093750e+00);
		y = fma( y, q2, float(-1.20312500e+01) );
		y = fma( y, q2, float(2.32031250e+01) );
		y = fma( y, q2, float(-2.16562500e+01) );
		y = fma( y, q2, float(9.02343750e+00) );
		f = y * h3inv;
	}
}


CUDA_EXPORT inline float self_phi() {
	float f, phi;
	gsoft(f, phi, 0.0f, 1.f, 1.f, 1.f, true);
	return phi;
}


#define CLOUD_MIN -2
#define CLOUD_MAX 3

inline CUDA_EXPORT float cloud_weight(float x) {
	float y;
	x = abs(x);
	const float x2 = x * x;
	y = 0.0f;
	if( x < float(1.00000000e+00) ) {
		y = float(-8.33333333e-02);
		y = fma( y, x, float(2.50000000e-01) );
		y *= x;
		y = fma( y, x, float(-5.00000000e-01) );
		y *= x;
		y = fma( y, x, float(5.50000000e-01) );
	} else if( x < float(2.00000000e+00) ) {
		y = float(4.16666667e-02);
		y = fma( y, x, float(-3.75000000e-01) );
		y = fma( y, x, float(1.25000000e+00) );
		y = fma( y, x, float(-1.75000000e+00) );
		y = fma( y, x, float(6.25000000e-01) );
		y = fma( y, x, float(4.25000000e-01) );
	} else if( x < float(3.00000000e+00) ) {
		y = float(-8.33333333e-03);
		y = fma( y, x, float(1.25000000e-01) );
		y = fma( y, x, float(-7.50000000e-01) );
		y = fma( y, x, float(2.25000000e+00) );
		y = fma( y, x, float(-3.37500000e+00) );
		y = fma( y, x, float(2.02500000e+00) );
	}
	return y;
}

inline CUDA_EXPORT float cloud_filter(float kh) {
	const double s = sinc(0.5 * kh);
	return pow(s, -6);
}

