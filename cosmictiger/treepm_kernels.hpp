#pragma once

CUDA_EXPORT inline void green_direct(float& phi, float& f, float r, float r2, float rinv, float rsinv, float rsinv2) {
	const float q = r * rsinv;
	float q2 = sqr(q);
	f = phi = 0.f;
	if (q2 < 1.f) {
		phi = rinv;
		f = rinv * sqr(rinv);
		float y;
		y = float(2.73437500e-01);
		y = fmaf( y, q2, float(-1.40625000e+00) );
		y = fmaf( y, q2, float(2.95312500e+00) );
		y = fmaf( y, q2, float(-3.28125000e+00) );
		y = fmaf( y, q2, float(2.46093750e+00) );
		phi -= y * rsinv;
		y = float(-2.18750000e+00);
		y = fmaf( y, q2, float(8.43750000e+00) );
		y = fmaf( y, q2, float(-1.18125000e+01) );
		y = fmaf( y, q2, float(6.56250000e+00) );
		f -= y * sqr(rsinv) * rsinv;
	}
}


CUDA_EXPORT inline array<float, 5> green_kernel(float r, float rsinv, float rsinv2) {
	array<float, 5> d;
	float q0 = 1.f;
	float q1 = r * rsinv;
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
		y = float(2.73437500e-01);
		y = fmaf( y, q2, float(-1.40625000e+00) );
		y = fmaf( y, q2, float(2.95312500e+00) );
		y = fmaf( y, q2, float(-3.28125000e+00) );
		y = fmaf( y, q2, float(2.46093750e+00) );
		y *= q0 * rsinv1;
		d[0] = fmaf( float(-1), rinv1, y);
		y = float(2.18750000e+00);
		y = fmaf( y, q2, float(-8.43750000e+00) );
		y = fmaf( y, q2, float(1.18125000e+01) );
		y = fmaf( y, q2, float(-6.56250000e+00) );
		y *= q1 * rsinv2;
		d[1] = fmaf( float(1), rinv2, y);
		y = float(1.31250000e+01);
		y = fmaf( y, q2, float(-3.37500000e+01) );
		y = fmaf( y, q2, float(2.36250000e+01) );
		y *= q2 * rsinv3;
		d[2] = fmaf( float(-3), rinv3, y);
		y = float(5.25000000e+01);
		y = fmaf( y, q2, float(-6.75000000e+01) );
		y *= q3 * rsinv4;
		d[3] = fmaf( float(15), rinv4, y);
		y = float(1.05000000e+02);
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
	y = double(1.4418053700055368e-118);
	y = std::fma( y, q2, double(-9.3140626902970444e-115) );
	y = std::fma( y, q2, double(5.7206973043154297e-111) );
	y = std::fma( y, q2, double(-3.3363106678919343e-107) );
	y = std::fma( y, q2, double(1.8449797993568184e-103) );
	y = std::fma( y, q2, double(-9.6603142293670616e-100) );
	y = std::fma( y, q2, double(4.7818555435515801e-96) );
	y = std::fma( y, q2, double(-2.2340829099277647e-92) );
	y = std::fma( y, q2, double(9.8344329695942630e-89) );
	y = std::fma( y, q2, double(-4.0714552493903858e-85) );
	y = std::fma( y, q2, double(1.5821675099287922e-81) );
	y = std::fma( y, q2, double(-5.7590897361086492e-78) );
	y = std::fma( y, q2, double(1.9592423282268047e-74) );
	y = std::fma( y, q2, double(-6.2147166651443749e-71) );
	y = std::fma( y, q2, double(1.8333414162183147e-67) );
	y = std::fma( y, q2, double(-5.0160221147931506e-64) );
	y = std::fma( y, q2, double(1.2690535950293141e-60) );
	y = std::fma( y, q2, double(-2.9594329836340786e-57) );
	y = std::fma( y, q2, double(6.3391054509173625e-54) );
	y = std::fma( y, q2, double(-1.2424646683825093e-50) );
	y = std::fma( y, q2, double(2.2190418977331087e-47) );
	y = std::fma( y, q2, double(-3.5948478743266922e-44) );
	y = std::fma( y, q2, double(5.2556675922554241e-41) );
	y = std::fma( y, q2, double(-6.8954358810366441e-38) );
	y = std::fma( y, q2, double(8.0676599808239474e-35) );
	y = std::fma( y, q2, double(-8.3580957401361275e-32) );
	y = std::fma( y, q2, double(7.6058671235194367e-29) );
	y = std::fma( y, q2, double(-6.0238467618284041e-26) );
	y = std::fma( y, q2, double(4.1082634915670421e-23) );
	y = std::fma( y, q2, double(-2.3827928251083090e-20) );
	y = std::fma( y, q2, double(1.1580373130027721e-17) );
	y = std::fma( y, q2, double(-4.6321492520111073e-15) );
	y = std::fma( y, q2, double(1.4915520591476050e-12) );
	y = std::fma( y, q2, double(-3.7587111890517630e-10) );
	y = std::fma( y, q2, double(7.1415512591981810e-08) );
	y = std::fma( y, q2, double(-9.7125097125094957e-06) );
	y = std::fma( y, q2, double(8.7412587412586881e-04) );
	y = std::fma( y, q2, double(-4.5454545454545359e-02) );
	y = std::fma( y, q2, double(1.0000000000000000e+00) );
	return y;
}


CUDA_EXPORT inline float green_phi0(float nparts, float rs) {
	return float(5.71198664e-01) * sqr(rs) * (nparts - 1) +  float(2.46093750e+00) / rs;
}

