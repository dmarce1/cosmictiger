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
#include <cosmictiger/flops.hpp>

#include <cmath>

void cosmos_update0(double& adotdot, double& adot, double& a, double dt) {
	const double omega_m = get_options().omega_m;
	const double omega_r = get_options().omega_r;
	const double omega_lam = get_options().omega_lam;
	const double omega_k = get_options().omega_k;
	const auto H = constants::H0 * get_options().code_to_s * get_options().hubble;
	const double adot0 = adot;

	adotdot = a * sqr(H) * (-omega_r / (sqr(sqr(a))) - 0.5 * omega_m / (sqr(a) * a) + omega_lam);
	const double da1 = adot * a * dt;
	const double dadot1 = adotdot * a * dt;
	a += da1;
	adot += dadot1;

	adotdot = a * sqr(H) * (-omega_r / (sqr(sqr(a))) - 0.5 * omega_m / (sqr(a) * a) + omega_lam);
	const double da2 = adot * a * dt;
	const double dadot2 = adotdot * a * dt;
	a += -da1 + 0.25 * (da1 + da2);
	adot += -dadot1 + 0.25 * (dadot1 + dadot2);

	adotdot = a * sqr(H) * (-omega_r / (sqr(sqr(a))) - 0.5 * omega_m / (sqr(a) * a) + omega_lam);
	const double da3 = adot * a * dt;
	const double dadot3 = adotdot * a * dt;
	a += -0.25 * (da1 + da2) + (1.0 / 6.0) * (da1 + da2 + 4.0 * da3);
	adot += -0.25 * (dadot1 + dadot2) + (1.0 / 6.0) * (dadot1 + dadot2 + 4.0 * dadot3);

}

void cosmos_update(double& adotdot, double& adot, double& a, double dt0) {
	constexpr int N = 128;
	const double dt = dt0 / N;
	for (int i = 0; i < N; i++) {
		cosmos_update0(adotdot, adot, a, dt);
	}
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
	add_cpu_flops(27);
	return H * a * std::sqrt(omega_r / (a * a * a * a) + omega_m / (a * a * a) + omega_k / (a * a) + omega_lam);
}

double cosmos_dadtau(double a) {
	return a * cosmos_dadt(a);
}

double cosmos_tau_to_scale(double a0, double t1) {
	constexpr int N = 1024;
	double a = a0;
	double dt = t1 / N;
	a += a * cosmos_dadt(a) * 0.5 * dt;
	for (int i = 1; i < N; i++) {
		a += a * cosmos_dadt(a) * dt;
	}
	a += a * cosmos_dadt(a) * 0.5 * dt;
	return a;
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
	int N = 2;
	double t = 0.0;
	double tlast;
	do {
		tlast = t;
		t = 0.0;
		double a = a0;
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
		N *= 2;
	} while (fabs(tlast - t) > 1e-9 * t);
	return t;
}

#define Power(a,b) pow(a,b)
#define Log(a) log(a)

void cosmos_NFW_fit(vector<double> radii, double& Rs, double& rho_nfw) {
	double err;
	std::sort(radii.begin(), radii.end());
	double rmax = radii.back();
	double npart = radii.size();
	int nbin = std::max(1, std::min(50, (int) npart / 15));
	vector<double> rbnd(nbin + 1);
	rbnd[0] = 0.0;
	for (int n = 1; n < nbin; n++) {
		double j = (double) n / nbin * npart;
		double k = j - (int) j;
		double r0 = radii[j] * (1.0 - k) + radii[j + 1] * k;
		rbnd[n] = r0;
	}
	rbnd[nbin] = rmax;
	double rho0 = npart / (4.0 / 3.0 * M_PI * sqr(rmax) * rmax);
	double c = 10.0;
	do {
		double f = 0.0;
		double dfdc = 0.0;
		for (int i = 0; i < nbin; i++) {
			double dr = (rbnd[i + 1] - rbnd[i]);
			if (dr > 0.0) {
				double vol = 4.0 / 3.0 * M_PI * (sqr(rbnd[i + 1]) * rbnd[i + 1] - sqr(rbnd[i]) * rbnd[i]);
				double rho = npart / nbin / vol;
				double x = 0.5 * (rbnd[i] + rbnd[i + 1]) / rmax;
				f +=
						(-2 * c * rho0 * (c * (2 + c * (3 + c * x)) - 2 * Power(1 + c, 2) * Log(1 + c))
								* (c * (c * (1 + c) * rho0 + 3 * rho * x + 3 * c * rho * Power(x, 2) * (2 + c * x))
										- 3 * (1 + c) * rho * x * Power(1 + c * x, 2) * Log(1 + c)))
								/ (9. * Power(x, 2) * Power(1 + c * x, 5) * Power(-c + (1 + c) * Log(1 + c), 3));
				dfdc += (2
						* (3 * Power(c, 6) * (1 + c) * Power(rho0 + c * rho0 * x, 2)
								+
								Power(c, 4) * rho0 * (1 + c * x) * (6 * rho * x * Power(1 + c * x, 3) + rho0 * (9 + c * (16 + x + c * (7 - c * x))))
										* (c - (1 + c) * Log(1 + c))
								+
								Power(c, 2) * rho0
										* (-2 * Power(1 + c, 3) * rho0 * (-3 + 2 * c * x) - 3 * rho * x * Power(1 + c * x, 3) * (-5 - 3 * c + (-1 + c) * c * x))
										* Power(c - (1 + c) * Log(1 + c), 2)
								+ 6 * Power(1 + c, 2) * rho * rho0 * x * Power(1 + c * x, 2) * (-1 + 2 * c * x) * Power(-c + (1 + c) * Log(1 + c), 3)))
						/ (9. * (1 + c) * Power(x, 2) * Power(1 + c * x, 6) * Power(c - (1 + c) * Log(1 + c), 4));
			}
		}
		double dc = -f / dfdc;
		dc = std::min(dc, 0.1 * c);
		dc = std::max(dc, -0.1 * c);
		c += dc;
		err = fabs(dc/c);
	//	PRINT( "%i %e %e\n", npart, c, err);
	} while (err > 1.0e-3);
	Rs = rmax / c;
	rho_nfw = sqr(c) * c * rho0 / (3.0 * (log(1 + c) - c / (1 + c)));
}

