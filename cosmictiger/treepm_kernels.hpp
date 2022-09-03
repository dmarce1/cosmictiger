#pragma once

CUDA_EXPORT inline void green_direct(float& phi, float& f, float r, float r2, float rinv, float rsinv, float rsinv2) {
	const float q = r * rsinv;
	float q2 = sqr(q);
	f = phi = 0.f;
	if (q2 < 1.f) {
		phi = rinv;
		f = rinv * sqr(rinv);
		float y;
		y = float(-3.00000000e+00);
		y = fma( y, q, float(1.50000000e+01) );
		y = fma( y, q, float(-2.80000000e+01) );
		y = fma( y, q, float(2.10000000e+01) );
		y *= q;
		y = fma( y, q, float(-7.00000000e+00) );
		y *= q;
		y = fma( y, q, float(3.00000000e+00) );
		phi -= y * rsinv;
		y = float(2.10000000e+01);
		y = fma( y, q, float(-9.00000000e+01) );
		y = fma( y, q, float(1.40000000e+02) );
		y = fma( y, q, float(-8.40000000e+01) );
		y *= q;
		y = fma( y, q, float(1.40000000e+01) );
		f -= y * sqr(rsinv) * rsinv;
	}
}


CUDA_EXPORT inline array<float, 5> green_kernel(float r, float rsinv, float rsinv2) {
	array<float, 5> d;
	float q0 = 1.f;
	float q1 = r * rsinv;
	float& q = q1;
	float qinv = 1.f / q1;
	float q2 = sqr(q1);
	float q3 = q1 * q2;
	float q4 = q1 * q3;
	float rsinv1 = rsinv;
	float rsinv3 = rsinv * rsinv2;
	float rsinv4 = rsinv * rsinv3;
	float rsinv5 = rsinv * rsinv4;
	float rinv1 = 1.f / r;
	float rinv2 = rinv1 * rinv1;
	float rinv3 = rinv1 * rinv2;
	float rinv4 = rinv1 * rinv3;
	float rinv5 = rinv1 * rinv4;
	if (q2 < 1.f) {
		float y;
		float z;
		y = float(-3.00000000e+00);
		y = fma( y, q, float(1.50000000e+01) );
		y = fma( y, q, float(-2.80000000e+01) );
		y = fma( y, q, float(2.10000000e+01) );
		y *= q;
		y = fma( y, q, float(-7.00000000e+00) );
		y *= q;
		y = fma( y, q, float(3.00000000e+00) );
		y *= q0 * rsinv1;
		d[0] = fmaf( float(-1), rinv1, y);
		y = float(-2.10000000e+01);
		y = fma( y, q, float(9.00000000e+01) );
		y = fma( y, q, float(-1.40000000e+02) );
		y = fma( y, q, float(8.40000000e+01) );
		y *= q;
		y = fma( y, q, float(-1.40000000e+01) );
		y *= q1 * rsinv2;
		d[1] = fmaf( float(1), rinv2, y);
		y = float(-1.05000000e+02);
		y = fma( y, q, float(3.60000000e+02) );
		y = fma( y, q, float(-4.20000000e+02) );
		y = fma( y, q, float(1.68000000e+02) );
		y *= q2 * rsinv3;
		d[2] = fmaf( float(-3), rinv3, y);
		y = float(-3.15000000e+02);
		y = fma( y, q, float(7.20000000e+02) );
		z = y;
		y = float(-4.20000000e+02);
		y *= qinv;
		y += z;
		y *= q3 * rsinv4;
		d[3] = fmaf( float(15), rinv4, y);
		y = 0.f;
		z = y;
		y = float(4.20000000e+02);
		y *= qinv;
		y = fma( y, qinv, float(-3.15000000e+02) );
		y *= qinv;
		y += z;
		y *= q4 * rsinv5;
		d[4] = fmaf( float(-105), rinv5, y);
	} else {
		d[0] = 0.f;
		d[1] = 0.f;
		d[2] = 0.f;
		d[3] = 0.f;
		d[4] = 0.f;
	}
	 return d;
}


CUDA_EXPORT inline double green_filter(double k) {
	if(k > double(2.51327412e+01)) {
		return 0.f;
	}
	double y;
	double q2 = sqr(k);
	y = double(9.48923400e-120);
	y = fma( y, q2, double(-6.28509920e-116) );
	y = fma( y, q2, double(3.96048274e-112) );
	y = fma( y, q2, double(-2.37128693e-108) );
	y = fma( y, q2, double(1.34721142e-104) );
	y = fma( y, q2, double(-7.25248816e-101) );
	y = fma( y, q2, double(3.69392015e-97) );
	y = fma( y, q2, double(-1.77725363e-93) );
	y = fma( y, q2, double(8.06388443e-90) );
	y = fma( y, q2, double(-3.44428664e-86) );
	y = fma( y, q2, double(1.38227000e-82) );
	y = fma( y, q2, double(-5.20175847e-79) );
	y = fma( y, q2, double(1.83162884e-75) );
	y = fma( y, q2, double(-6.02082567e-72) );
	y = fma( y, q2, double(1.84304164e-68) );
	y = fma( y, q2, double(-5.23990914e-65) );
	y = fma( y, q2, double(1.37968904e-61) );
	y = fma( y, q2, double(-3.35402405e-58) );
	y = fma( y, q2, double(7.50280597e-55) );
	y = fma( y, q2, double(-1.53875730e-51) );
	y = fma( y, q2, double(2.88187259e-48) );
	y = fma( y, q2, double(-4.90725265e-45) );
	y = fma( y, q2, double(7.56078496e-42) );
	y = fma( y, q2, double(-1.04842885e-38) );
	y = fma( y, q2, double(1.30066849e-35) );
	y = fma( y, q2, double(-1.43398701e-32) );
	y = fma( y, q2, double(1.39440897e-29) );
	y = fma( y, q2, double(-1.18564603e-26) );
	y = fma( y, q2, double(8.72817885e-24) );
	y = fma( y, q2, double(-5.49875267e-21) );
	y = fma( y, q2, double(2.92433665e-18) );
	y = fma( y, q2, double(-1.29138706e-15) );
	y = fma( y, q2, double(4.64038418e-13) );
	y = fma( y, q2, double(-1.32250949e-10) );
	y = fma( y, q2, double(2.89062789e-08) );
	y = fma( y, q2, double(-4.62500463e-06) );
	y = fma( y, q2, double(5.05050505e-04) );
	y = fma( y, q2, double(-3.33333333e-02) );
	y = fma( y, q2, double(1.00000000e+00) );
	return y;
}


CUDA_EXPORT inline float green_phi0(float nparts, float rs) {
	return float(4.18879020e-01) * sqr(rs) * (nparts - 1) +  float(3.00000000e+00) / rs;
}

