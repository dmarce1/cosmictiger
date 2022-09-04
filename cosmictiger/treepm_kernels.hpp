#pragma once

CUDA_EXPORT inline void green_direct(float& phi, float& f, float r, float r2, float rinv, float rsinv, float rsinv2) {
	const float q = r * rsinv;
	float q2 = sqr(q);
	f = phi = 0.f;
	if (q2 < 1.f) {
		phi = rinv;
		f = rinv * sqr(rinv);
		float y;
		y = float(2.25585937e-01);
		y = fma( y, q2, float(-1.59960937e+00) );
		y = fma( y, q2, float(4.88769531e+00) );
		y = fma( y, q2, float(-8.37890625e+00) );
		y = fma( y, q2, float(8.79785156e+00) );
		y = fma( y, q2, float(-5.86523437e+00) );
		y = fma( y, q2, float(2.93261719e+00) );
		phi -= y * rsinv;
		y = float(-2.70703125e+00);
		y = fma( y, q2, float(1.59960937e+01) );
		y = fma( y, q2, float(-3.91015625e+01) );
		y = fma( y, q2, float(5.02734375e+01) );
		y = fma( y, q2, float(-3.51914062e+01) );
		y = fma( y, q2, float(1.17304687e+01) );
		f -= y * sqr(rsinv) * rsinv;
	}
}


CUDA_EXPORT inline array<float, 7> green_kernel(float r, float rsinv, float rsinv2) {
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
		y = float(2.25585937e-01);
		y = fma( y, q2, float(-1.59960937e+00) );
		y = fma( y, q2, float(4.88769531e+00) );
		y = fma( y, q2, float(-8.37890625e+00) );
		y = fma( y, q2, float(8.79785156e+00) );
		y = fma( y, q2, float(-5.86523437e+00) );
		y = fma( y, q2, float(2.93261719e+00) );
		y *= q0 * rsinv1;
		d[0] = fmaf( float(-1), rinv1, y);
		y = float(2.70703125e+00);
		y = fma( y, q2, float(-1.59960937e+01) );
		y = fma( y, q2, float(3.91015625e+01) );
		y = fma( y, q2, float(-5.02734375e+01) );
		y = fma( y, q2, float(3.51914062e+01) );
		y = fma( y, q2, float(-1.17304687e+01) );
		y *= q1 * rsinv2;
		d[1] = fmaf( float(1), rinv2, y);
		y = float(2.70703125e+01);
		y = fma( y, q2, float(-1.27968750e+02) );
		y = fma( y, q2, float(2.34609375e+02) );
		y = fma( y, q2, float(-2.01093750e+02) );
		y = fma( y, q2, float(7.03828125e+01) );
		y *= q2 * rsinv3;
		d[2] = fmaf( float(-3), rinv3, y);
		y = float(2.16562500e+02);
		y = fma( y, q2, float(-7.67812500e+02) );
		y = fma( y, q2, float(9.38437500e+02) );
		y = fma( y, q2, float(-4.02187500e+02) );
		y *= q3 * rsinv4;
		d[3] = fmaf( float(15), rinv4, y);
		y = float(1.29937500e+03);
		y = fma( y, q2, float(-3.07125000e+03) );
		y = fma( y, q2, float(1.87687500e+03) );
		y *= q4 * rsinv5;
		d[4] = fmaf( float(-105), rinv5, y);
		y = float(5.19750000e+03);
		y = fma( y, q2, float(-6.14250000e+03) );
		y *= q5 * rsinv6;
		d[5] = fmaf( float(945), rinv6, y);
		y = float(1.03950000e+04);
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
	return float(4.18879020e-01) * sqr(rs) * (nparts - 1) +  float(2.93261719e+00) / rs;
}


CUDA_EXPORT inline float green_rho(float q) {
	float y;
	float q2 = sqr(q);
	y = float(-2.80044313e+00);
	y = fma( y, q2, float(1.40022156e+01) );
	y = fma( y, q2, float(-2.80044313e+01) );
	y = fma( y, q2, float(2.80044313e+01) );
	y = fma( y, q2, float(-1.40022156e+01) );
	y = fma( y, q2, float(2.80044313e+00) );
	return y;
}

