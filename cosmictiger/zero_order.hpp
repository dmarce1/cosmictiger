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

#pragma once

#include <cosmictiger/constants.hpp>
#include <cosmictiger/interp.hpp>

#include <functional>

struct cosmic_params {
	double omega_b, omega_c, omega_gam, omega_nu;
	double Neff, Y, Theta, hubble;
};


inline double hubble_function(double a, double littleh, double omega_m, double omega_r) {
	return littleh * cosmic_constants::H0
			* sqrt(omega_r / (a * a * a * a) + omega_m / (a * a * a) + (1 - omega_r - omega_m));
}

struct zero_order_universe {
	double amin;
	double amax;
	cosmic_params params;
	interp_functor<double> sigma_T;
	interp_functor<double> cs2;
	interp_functor<double> K;
	std::function<double(double)> xe;


	void compute_matter_fractions(double& Oc, double& Ob, double a) const;


	void compute_radiation_fractions(double& Ogam, double& Onu, double a) const;


	double conformal_time_to_scale_factor(double taumax);


	double scale_factor_to_conformal_time(double a);


	double redshift_to_time(double z) const;


	double redshift_to_density(double z) const;
};


void create_zero_order_universe(zero_order_universe* uni_ptr, std::function<double(double)> fxe, double amax, cosmic_params);

class cosmos {
	double a;
	double t;
	double tau;
	double omega_m;
	double omega_r;
	double omega_lam;
	double H;
public:
	cosmos(double omega_c, double omega_b, double omega_gam, double omega_nu, double h_, double a_) {
		H = constants::H0 * h_;
		omega_m = omega_c + omega_b;
		omega_r = omega_gam + omega_nu;
		omega_lam = 1.0 - omega_r - omega_m;
		a = a_;
		t = tau = 0.0;
	}
	double advance(double dtau0) {
		const double beta[3] = { 1.0, 0.25, 2.0 / 3.0 };

		double tau1 = tau + dtau0;
		while (tau < tau1) {
			double a0 = a;
			double t0 = t;
			double dtau = std::min(std::abs(1.0e-2/a / (H * sqrt(omega_m / (a * a * a) + omega_r / (a * a * a * a) + omega_lam))) * 0.01, tau1 - tau);
			for (int rk = 0; rk < 3; rk++) {
				double da = (H * sqrt(omega_m / (a * a * a) + omega_r / (a * a * a * a) + omega_lam))* a *a  * dtau;
				double dt = a * dtau;
				a = (1.0 - beta[rk]) * a0 + beta[rk] * (a + da);
				t = (1.0 - beta[rk]) * t0 + beta[rk] * (t + dt);
			}
			tau += dtau;
		}
		return t;
	}
	double scale() const {
		return a;
	}
	double time() const {
		return t;
	}

};

