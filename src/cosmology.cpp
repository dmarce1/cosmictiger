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


double cosmos_time(double a0, double a1) {
	double a = a0;
	double t = 0.0;
	while (a < a1) {
		const double dadt1 = cosmos_dadt(a);
		const double dt = (a / dadt1) * 1.e-5;
		const double dadt2 = cosmos_dadt(a + dadt1 * dt);
		a += 0.5 * (dadt1 + dadt2) * dt;
		t += dt;
	}
	return t;
}

double cosmos_conformal_time(double a0, double a1) {
	double a = a0;
	double t = 0.0;
	while (a < a1) {
		const double dadt1 = cosmos_dadtau(a);
		const double dt = (a / dadt1) * 1.e-5;
		const double dadt2 = cosmos_dadtau(a + dadt1 * dt);
		a += 0.5 * (dadt1 + dadt2) * dt;
		t += dt;
	}
	return t;
}
