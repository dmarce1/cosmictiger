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

#include <cosmictiger/constants.hpp>
#include <cosmictiger/cosmology.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/math.hpp>

#include <cmath>

void cosmos_update(double& adotdot, double& adot, double& a, double dt) {
	const double omega_m = get_options().omega_m;
	const double omega_r = get_options().omega_r;
	const double omega_lam = get_options().omega_lam;
	const double omega_k = get_options().omega_k;
	const auto H = constants::H0 * get_options().code_to_s * get_options().hubble;
	const double adot0 = adot;
	adotdot = a * sqr(H) * (-omega_r / (sqr(sqr(a))) - 0.5 * omega_m / (sqr(a) * a) + omega_lam);
	const double adotdot0 = adotdot;
	a += 0.5 * adot0 * dt;
	adot += 0.5 * adotdot0 * dt;
	adotdot = a * sqr(H) * (-omega_r / (sqr(sqr(a))) - 0.5 * omega_m / (sqr(a) * a) + omega_lam);
	a += adot * dt;
	adot += adotdot * dt;
	a -= 0.5 * adot0 * dt;
	adot -= 0.5 * adotdot0 * dt;

}

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
	const auto omega_k = get_options().omega_k;
	const auto omega_lam = get_options().omega_lam;
	return H * a * std::sqrt(omega_r / (a * a * a * a) + omega_m / (a * a * a) + omega_k / (a * a) + omega_lam);
}

double cosmos_dadtau(double a) {
	return a * cosmos_dadt(a);
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
