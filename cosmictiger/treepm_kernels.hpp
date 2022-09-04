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


CUDA_EXPORT inline array<float, 6> green_kernel(float r, float rsinv, float rsinv2) {
	array<float, 6> d;
	float q0 = 1.f;
	float q1 = r * rsinv;
	float& q = q1;
	float qinv = 1.f / q1;
	float q2 = sqr(q1);
	float q3 = q1 * q2;
	float q4 = q1 * q3;
	float q5 = q1 * q4;
	float rsinv1 = rsinv;
	float rsinv3 = rsinv * rsinv2;
	float rsinv4 = rsinv * rsinv3;
	float rsinv5 = rsinv * rsinv4;
	float rsinv6 = rsinv * rsinv5;
	float rinv1 = 1.f / r;
	float rinv2 = rinv1 * rinv1;
	float rinv3 = rinv1 * rinv2;
	float rinv4 = rinv1 * rinv3;
	float rinv5 = rinv1 * rinv4;
	float rinv6 = rinv1 * rinv5;
	if (q2 < 1.f) {
		float y;
		float z;
		y = float(-2.46093750e-01);
		y = fma( y, q2, float(1.50390625e+00) );
		y = fma( y, q2, float(-3.86718750e+00) );
		y = fma( y, q2, float(5.41406250e+00) );
		y = fma( y, q2, float(-4.51171875e+00) );
		y = fma( y, q2, float(2.70703125e+00) );
		y *= q0 * rsinv1;
		d[0] = fmaf( float(-1), rinv1, y);
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
	} else {
		d[0] = 0.f;
		d[1] = 0.f;
		d[2] = 0.f;
		d[3] = 0.f;
		d[4] = 0.f;
		d[5] = 0.f;
	}
	 return d;
}


CUDA_EXPORT inline double green_filter(double k) {
	if(k > double(2.51327412e+01)) {
		return 0.f;
	}
	double y;
	double q2 = sqr(k);
	y = double(1.82297231e-119);
	y = fma( y, q2, double(-1.20534929e-115) );
	y = fma( y, q2, double(7.58164703e-112) );
	y = fma( y, q2, double(-4.53079227e-108) );
	y = fma( y, q2, double(2.56895921e-104) );
	y = fma( y, q2, double(-1.38004489e-100) );
	y = fma( y, q2, double(7.01338813e-97) );
	y = fma( y, q2, double(-3.36642630e-93) );
	y = fma( y, q2, double(1.52364454e-89) );
	y = fma( y, q2, double(-6.49072576e-86) );
	y = fma( y, q2, double(2.59758845e-82) );
	y = fma( y, q2, double(-9.74615186e-79) );
	y = fma( y, q2, double(3.42089930e-75) );
	y = fma( y, q2, double(-1.12068661e-71) );
	y = fma( y, q2, double(3.41809417e-68) );
	y = fma( y, q2, double(-9.68004268e-65) );
	y = fma( y, q2, double(2.53810719e-61) );
	y = fma( y, q2, double(-6.14221940e-58) );
	y = fma( y, q2, double(1.36725804e-54) );
	y = fma( y, q2, double(-2.78920640e-51) );
	y = fma( y, q2, double(5.19350231e-48) );
	y = fma( y, q2, double(-8.78740592e-45) );
	y = fma( y, q2, double(1.34447310e-41) );
	y = fma( y, q2, double(-1.84999499e-38) );
	y = fma( y, q2, double(2.27549384e-35) );
	y = fma( y, q2, double(-2.48483927e-32) );
	y = fma( y, q2, double(2.39041538e-29) );
	y = fma( y, q2, double(-2.00794892e-26) );
	y = fma( y, q2, double(1.45777092e-23) );
	y = fma( y, q2, double(-9.03817968e-21) );
	y = fma( y, q2, double(4.71792979e-18) );
	y = fma( y, q2, double(-2.03814567e-15) );
	y = fma( y, q2, double(7.13350985e-13) );
	y = fma( y, q2, double(-1.96884872e-10) );
	y = fma( y, q2, double(4.13458231e-08) );
	y = fma( y, q2, double(-6.28456511e-06) );
	y = fma( y, q2, double(6.41025641e-04) );
	y = fma( y, q2, double(-3.84615385e-02) );
	y = fma( y, q2, double(1.00000000e+00) );
	return y;
}


CUDA_EXPORT inline float green_phi0(float nparts, float rs) {
	return float(4.83321947e-01) * sqr(rs) * (nparts - 1) +  float(2.70703125e+00) / rs;
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

