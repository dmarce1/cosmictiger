#include <cosmictiger/constants.hpp>
#include <cosmictiger/cosmology.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/math.hpp>

#include <cmath>

double cosmos_growth_factor(double omega_m, float a) {
	const double omega_l = 1.f - omega_m;
	const double a3 = a * sqr(a);
	const double deninv = 1.f / (omega_m + a3 * omega_l);
	const double Om = omega_m * deninv;
	const double Ol = a3 * omega_l * deninv;
	return a * 2.5 * Om / (pow(Om, 4.f / 7.f) - Ol + (1.f + 0.5f * Om) * (1.f + 0.014285714f * Ol));
}

double cosmos_dadt(double a) {
	const auto H = constants::H0 * get_options().code_to_s * get_options().hubble;
	const auto omega_m = get_options().omega_m;
	const auto omega_r = get_options().omega_r;
	const auto omega_lambda = 1.0 - omega_m - omega_r;
	return H * a * std::sqrt(omega_r / (a * a * a * a) + omega_m / (a * a * a) + omega_lambda);
}

double cosmos_dadtau(double a) {
	return a * cosmos_dadt(a);
}

double cosmos_ainv(double a0, double a1) {
	double a = a0;
	double ainv = 0.0;
	const int N = 16 * 1024;
	const double da = (a1 - a0) / N;
	double t = 0.0;
	for (int i = 0; i < N; i++) {
		double dadt = cosmos_dadt(a);
		ainv += 0.5 * da / (a * dadt);
		t += 0.5 * da / dadt;
		a += da;
		dadt = cosmos_dadt(a);
		ainv += 0.5 * da / (a * dadt);
		t += 0.5 * da / dadt;
	}
	return ainv / t;
}

double cosmos_time(double a0, double a1) {
	double a = a0;
	double t = 0.0;
	const int N = 1024 * 1024;
	const double loga0 = log(a0);
	const double loga1 = log(a1);
	const double dloga = (loga1 - loga0) / N;
	double loga = loga0;
	a = exp(loga);
	double dadt = cosmos_dadt(a);
	for (int i = 0; i < N; i++) {
		t += 0.5 * a / dadt * dloga;
		loga = loga0 + dloga * (i + 1);
		a = exp(loga);
		dadt = cosmos_dadt(a);
		t += 0.5 * a / dadt * dloga;
	}
	return t;
}

double cosmos_conformal_time(double a0, double a1) {
	double a = a0;
	double t = 0.0;
	const int N = 1024 * 1024;
	const double loga0 = log(a0);
	const double loga1 = log(a1);
	const double dloga = (loga1 - loga0) / N;
	double loga = loga0;
	a = exp(loga);
	double dadt = cosmos_dadtau(a);
	for (int i = 0; i < N; i++) {
		t += 0.5 * a / dadt * dloga;
		loga = loga0 + dloga * (i + 1);
		a = exp(loga);
		dadt = cosmos_dadtau(a);
		t += 0.5 * a / dadt * dloga;
	}
	return t;
}
