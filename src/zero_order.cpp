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

#include <cosmictiger/options.hpp>
#include <cosmictiger/zero_order.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/safe_io.hpp>

#include <vector>

void zero_order_universe::compute_matter_fractions(double& Oc, double& Ob, double a) const {
	double omega_m = params.omega_b + params.omega_c;
	double omega_r = params.omega_gam + params.omega_nu;
	double Om = omega_m / (omega_r / a + omega_m + (a * a * a) * ((double) 1.0 - omega_m - omega_r));
	Ob = params.omega_b * Om / omega_m;
	Oc = params.omega_c * Om / omega_m;
}

void zero_order_universe::compute_radiation_fractions(double& Ogam, double& Onu, double a) const {
	double omega_m = params.omega_b + params.omega_c;
	double omega_r = params.omega_gam + params.omega_nu;
	double Or = omega_r / (omega_r + a * omega_m + (a * a * a * a) * ((double) 1.0 - omega_m - omega_r));
	Ogam = params.omega_gam * Or / omega_r;
	Onu = params.omega_nu * Or / omega_r;
}

double zero_order_universe::conformal_time_to_scale_factor(double taumax) {
	taumax *= constants::H0 / cosmic_constants::H0;
	double dlogtau = 1.0e-3;
	double a = amin;
	double logtaumax = log(taumax);
	const auto hubble = [this](double a) {
		return hubble_function(a,params.hubble, params.omega_c + params.omega_b, params.omega_gam + params.omega_nu);
	};
	double logtaumin = log(1.f / (a * hubble(a)));
	int N = (logtaumax - logtaumin) / dlogtau + 1;
	dlogtau = (logtaumax - logtaumin) / N;
	for (int i = 0; i < N; i++) {
		double logtau = logtaumin + (double) i * dlogtau;
		double tau = exp(logtau);
		double a0 = a;
		a += tau * a * a * hubble(a) * dlogtau;
		logtau = logtaumin + (double) (i + 1) * dlogtau;
		tau = exp(logtau);
		a = 0.75f * a0 + 0.25f * (a + tau * a * a * hubble(a) * dlogtau);
		logtau = logtaumin + ((double) i + 0.5f) * dlogtau;
		tau = exp(logtau);
		a = 1.f / 3.f * a0 + 2.f / 3.f * (a + tau * a * a * hubble(a) * dlogtau);
	}
	return a;
}

double zero_order_universe::redshift_to_density(double z) const {
	const double a = 1.0 / (1.0 + z);
	double omega_m = params.omega_b + params.omega_c;
	double omega_r = params.omega_gam + params.omega_nu;
	const double omega_l = 1.0 - omega_m - omega_r;
	const double H2 = sqr(params.hubble * constants::H0) * (omega_r / (a * a * a * a) + omega_m / (a * a * a) + omega_l);
	return omega_m * 3.0 * H2 / (8.0 * M_PI * constants::G);
}

double zero_order_universe::scale_factor_to_conformal_time(double a) {
	double amax = a;
	double dloga = 1e-2;
	double logamin = logf(amin);
	double logamax = logf(amax);
	int N = (logamax - logamin) / dloga + 1;
	dloga = (logamax - logamin) / (double) N;
	const auto hubble = [this](double a) {
		return hubble_function(a,params.hubble, params.omega_c + params.omega_b, params.omega_gam + params.omega_nu);
	};
	double tau = 1.f / (amin * hubble(amin));
	for (int i = 0; i < N; i++) {
		double loga = logamin + (double) i * dloga;
		double a = exp(loga);
		double tau0 = tau;
		tau += dloga / (a * hubble(a));
		loga = logamin + (double) (i + 1) * dloga;
		a = exp(loga);
		tau = 0.75f * tau0 + 0.25f * (tau + dloga / (a * hubble(a)));
		loga = logamin + ((double) i + 0.5f) * dloga;
		a = exp(loga);
		tau = (1.f / 3.f) * tau0 + (2.f / 3.f) * (tau + dloga / (a * hubble(a)));
	}
	tau *= cosmic_constants::H0 / constants::H0;
	return tau;
}

double zero_order_universe::redshift_to_time(double z) const {
	double amax = 1.f / (1.f + z);
	double dloga = 1e-3;
	double logamin = log(amin);
	double logamax = log(amax);
	int N = (logamax - logamin) / dloga + 1;
	dloga = (logamax - logamin) / (double) N;
	double t = 0.0;
	const auto hubble = [this](double a) {
		return hubble_function(a,params.hubble, params.omega_c + params.omega_b, params.omega_gam + params.omega_nu);
	};
	for (int i = 0; i < N; i++) {
		double loga = logamin + (double) i * dloga;
		double a = exp(loga);
		double t0 = t;
		t += dloga / hubble(a);
		loga = logamin + (double) (i + 1) * dloga;
		a = exp(loga);
		t = 0.75f * t0 + 0.25f * (t + dloga / hubble(a));
		loga = logamin + ((double) i + 0.5f) * dloga;
		a = exp(loga);
		t = (1.f / 3.f) * t0 + (2.f / 3.f) * (t + dloga / hubble(a));
	}
	t *= cosmic_constants::H0 / constants::H0;
	return t;
}

void create_zero_order_universe(zero_order_universe* uni_ptr, std::function<double(double)> fxe, double amax, cosmic_params cpars) {
	zero_order_universe& uni = *uni_ptr;
	uni.xe = fxe;
	using namespace constants;
	uni.params = cpars;
	double omega_b = cpars.omega_b;
	double omega_c = cpars.omega_c;
	double omega_gam = cpars.omega_gam;
	double omega_nu = cpars.omega_nu;
	double omega_m = cpars.omega_b + omega_c;
	double omega_r = omega_gam + omega_nu;
	double Theta = cpars.Theta;
	double littleh = cpars.hubble;
	double Neff = cpars.Neff;
	double Y = cpars.Y;
	double amin = Theta * Tcmb / (0.07 * 1e6 * evtoK);
	double logamin = log(amin);
	double logamax = log(amax);
	int N = 4 * 1024;
	double dloga = (logamax - logamin) / N;
	std::vector<double> K(N + 1);
	std::vector<double> thomson(N + 1);
	std::vector<double> sound_speed2(N + 1);

	PRINT("\t\tParameters:\n");
	PRINT("\t\t\t h                 = %f\n", littleh);
	PRINT("\t\t\t omega_m           = %f\n", omega_m);
	PRINT("\t\t\t omega_r           = %f\n", omega_r);
	PRINT("\t\t\t omega_lambda      = %f\n", 1 - omega_r - omega_m);
	PRINT("\t\t\t omega_b           = %f\n", omega_b);
	PRINT("\t\t\t omega_c           = %f\n", omega_c);
	PRINT("\t\t\t omega_gam         = %f\n", omega_gam);
	PRINT("\t\t\t omega_nu          = %f\n", omega_nu);
	PRINT("\t\t\t Neff              = %f\n", Neff);
	PRINT("\t\t\t temperature today = %f\n\n", 2.73 * Theta);

	dloga = (logamax - logamin) / N;
	const auto cosmic_hubble = [=](double a) {
		using namespace cosmic_constants;
		return littleh * cosmic_constants::H0 * sqrt(omega_r / (a * a * a * a) + omega_m / (a * a * a) + (1 - omega_r - omega_m));
	};
	auto cgs_hubble = [=](double a) {
		return constants::H0 / cosmic_constants::H0 * cosmic_hubble(a);
	};

	const auto rho_baryon = [=](double a) {
		using namespace constants;
		return 3.0 * pow(littleh * H0, 2) / (8.0 * M_PI * G) * omega_b / (a * a * a);
	};

	const auto T_radiation = [=](double a) {
		using namespace constants;
		return Tcmb * Theta / a;
	};

	double loga;
	double a;

	double rho_b, nH, nHe, ne, Tgas, Trad;
	double hubble = cgs_hubble(amin);

	rho_b = rho_baryon(amin);
	double nnuc = rho_b / mh;
	nHe = Y * nnuc / 4;
	nH = (1 - Y) * nnuc;
	ne = fxe(amin) * nH;
	Trad = T_radiation(amin);
	Tgas = Trad;
	double n = nH + nHe;
	double P = kb * (n + ne) * Tgas;
	double t = 0.0;
	double dt;
	for (int i = -10 / dloga; i <= 0; i++) {
		loga = logamin + i * dloga;
		dt = dloga / cgs_hubble(exp(loga));
		t += dt;
	}
	a = exp(logamin);
	PRINT("%e \n", P);
//	print_time(t);
//	PRINT(
//			", redshift %.0f: Big Bang nucleosynthesis has ended. The Universe is dominated by radiation at a temperature of %8.2e K."
//					" \n   Its total matter density is %.1f \% times the density of air at sea level.\n", 1 / a - 1,
//			Trad, 100 * rho_b * omega_m / omega_b / 1.274e-3);
	double mu = (nH + 4 * nHe) * mh / (nH + nHe + ne);
	double sigmaC = mu / me * c * (8.0 / 3.0) * omega_gam / (a * omega_m) * sigma_T * ne / hubble;
	double sigmaT = c * sigma_T * ne / hubble;
	thomson[0] = sigmaT;
	double P1, P2;
	double rho1, rho2;
	double cs2;
	P1 = P2 = P;
	rho1 = rho2 = rho_b;
	K[0] = P / std::pow(rho_b, 5.0 / 3.0);
	double a1;
	for (int i = 1; i <= N; i++) {
		loga = logamin + i * dloga;
		a = exp(loga);
		a1 = exp(loga + dloga);
//		PRINT("%e %e %e %e %e %e\n", a, nH, nHp, nHe, nHep, nHepp);
		P2 = P1;
		P1 = P;
		rho2 = rho1;
		rho1 = rho_b;
		hubble = cgs_hubble(a);
		nH /= rho_b;
		nHe /= rho_b;
		ne /= rho_b;
		rho_b = rho_baryon(a);
		nH *= rho_b;
		nHe *= rho_b;
		ne *= rho_b;
		Trad = T_radiation(a);
		double dt = dloga / hubble;
		const double gamma = 1.0 - 1.0 / sqrt(2.0);
		ne = fxe(a) * nH;
		mu = (nH + 4 * nHe) * mh / (nH + nHe + ne);
		sigmaC = mu / me * c * (8.0 / 3.0) * omega_gam / (a * omega_m) * sigma_T * ne / hubble;
		const double dTgasdT1 = ((Tgas + gamma * dloga * sigmaC * Trad) / (1 + gamma * dloga * (2 + sigmaC)) - Tgas) / (gamma * dloga);
		const double T1 = Tgas + (1 - 2 * gamma) * dTgasdT1 * dloga;
		const double dTgasdT2 = ((T1 + gamma * dloga * sigmaC * Trad) / (1 + gamma * dloga * (2 + sigmaC)) - T1) / (gamma * dloga);
		Tgas += 0.5 * (dTgasdT1 + dTgasdT2) * dloga;
		ne = fxe(a) * nH;
		n = nH + nHe;
		P = kb * (n + ne) * Tgas;
		sigmaT = c * sigma_T * ne / hubble;
		t += dt;
		if (i == 1) {
			cs2 = (P - P1) / (rho_b - rho1);
		} else {
			cs2 = (P - P2) / (rho_b - rho2);
		}
		sound_speed2[i - 1] = cs2 / (c * c);
		thomson[i] = sigmaT;
		K[i] = P / std::pow(rho_b,5.0/3.0);

		//	printf("%e %e %e %e |  %e %e %e \n", a, ne / (nH + 2 * nHe), thomson[i], sound_speed2[i - 1], P, P1, P2);
	}
	cs2 = (P - P1) / (rho_b - rho1);
	sound_speed2[N - 1] = cs2 / c;
//	print_time(t);
	uni.amin = amin;
	uni.amax = amax;
	build_interpolation_function(&uni.K, K, (double) amin, (double) amax);
	build_interpolation_function(&uni.sigma_T, thomson, (double) amin, (double) amax);
	build_interpolation_function(&uni.cs2, sound_speed2, (double) amin, (double) amax);
//	uni.hubble = std::move(cosmic_hubble);
}

