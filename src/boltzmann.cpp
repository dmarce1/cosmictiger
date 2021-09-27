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

#include <cosmictiger/boltzmann.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/hpx.hpp>
#include <vector>

#include <cmath>

void einstein_boltzmann_init(cos_state* uptr, const zero_order_universe* uni_ptr, double k, double normalization, double a, double ns) {
	cos_state& U = *uptr;
	const zero_order_universe& uni = *uni_ptr;
	double Oc, Ob, Ogam, Onu, Or;
	uni.compute_radiation_fractions(Ogam, Onu, a);
	uni.compute_matter_fractions(Oc, Ob, a);
	Or = Ogam + Onu;
	double hubble = hubble_function(a, uni.params.hubble, uni.params.omega_b + uni.params.omega_c, uni.params.omega_gam + uni.params.omega_nu);
	double eps = k / (a * hubble);
	double C = normalization * pow(eps, -1.5f) * pow(eps, (ns - 1.f) * 0.5f);
	double Rnu = Onu / Or;
	U[taui] = (double) 1.0 / (a * hubble);
	U[deltanui] = U[deltagami] = -(double) 2.0 / (double) 3.0 * C * eps * eps;
	U[deltabi] = U[deltaci] = (double) 3.0 / (double) 4.0 * U[deltagami];
	U[thetabi] = U[thetagami] = -C / (double) 18.0 * eps * eps * eps;
	U[thetanui] = ((double) 23 + (double) 4 * Rnu) / ((double) 15 + (double) 4 * Rnu) * U[thetagami];
	U[N2i] = (double) 0.5 * ((double) 4.0 * C) / ((double) 3.0 * ((double) 15 + (double) 4 * Rnu)) * eps * eps;
	U[hdoti] = (double) (double) 2.0 * C * eps * eps;
	U[G0i] = U[G1i] = U[G2i] = U[F2i] = (double) 0.0;
	for (int l = 3; l < LMAX; l++) {
		U[FLi + l] = (double) 0.0;
		U[NLi + l] = (double) 0.0;
		U[GLi + l] = (double) 0.0;
	}
	U[etai] = ((double) 0.5 * U[hdoti] - ((double) 1.5 * (Oc * U[deltaci] + Ob * U[deltabi]) + (double) 1.5 * (Ogam * U[deltagami] + Onu * U[deltanui])))
			/ (eps * eps);
}

void einstein_boltzman_kernel(int myindex, cos_state* states, const zero_order_universe* uni_ptr, double* ks, double amin, double amax, double norm, int nmax,
		bool cont, double ns) {
	if (!cont) {
		einstein_boltzmann_init(states + myindex, uni_ptr, ks[myindex], norm, amin, ns);
	}
	einstein_boltzmann(states + myindex, uni_ptr, ks[myindex], amin, amax);
}

void einstein_boltzmann(cos_state* uptr, const zero_order_universe *uni_ptr, double k, double amin, double amax) {
	const auto &uni = *uni_ptr;
	if (amin < uni.amin && amax > uni.amax) {
		THROW_ERROR("out of range error in einstein_boltzmann\n");
	}
	cos_state& U = *uptr;
	cos_state U0;
	double loga = log(amin);
	double logamax = log(amax);
	double omega_m = uni.params.omega_b + uni.params.omega_c;
	double omega_r = uni.params.omega_gam + uni.params.omega_nu;
	while (loga < logamax) {
		double Oc, Ob, Ogam, Onu, Or;
		double a = exp(loga);
		double hubble = hubble_function(a, uni.params.hubble, omega_m, omega_r);
		double eps = k / (a * hubble);
		uni.compute_radiation_fractions(Ogam, Onu, a);
		uni.compute_matter_fractions(Oc, Ob, a);
		Or = Ogam + Onu;
		double cs2 = uni.cs2(a);
		double lambda_i = 0.0;
		lambda_i = std::max((double) lambda_i, sqrt(((double) LMAX + (double) 1.0) / ((double) LMAX + (double) 3.0)) * eps);
		lambda_i = std::max((double) lambda_i, sqrt((double) 3.0 * pow(eps, 4) + (double) 8.0 * eps * eps * Or) / sqrt((double) 5) / eps);
		double lambda_r = (eps + sqrt(eps * eps + (double) 4.0 * cs2 * pow(eps, (double) 4))) / ((double) 2.0 * eps);
		double dloga_i = (double) 2.0 * (double) 1.73 / lambda_i;
		double dloga_r = (double) 2.0 * (double) 2.51 / lambda_r;
		double dloga = std::min(std::min((double) 5e-2, std::min((double) 0.9 * dloga_i, (double) 0.1 * dloga_r)), logamax - loga);
		double loga0 = loga;
		const auto compute_explicit =
				[&](int step) {
					U0 = U;
					cos_state dudt;
					constexpr double beta[3] = {1, 0.25, (2.0 / 3.0)};
					constexpr double tm[3] = {0, 1, 0.5};
					for (int i = 0; i < 3; i++) {
						loga = loga0 + (double) 0.5 * (tm[i] + step) * dloga;
						a = exp(loga);
						hubble = hubble_function(a, uni.params.hubble, omega_m,
								omega_r);
						eps = k / (a * hubble);
						uni.compute_radiation_fractions(Ogam,Onu,a);
						uni.compute_matter_fractions(Oc,Ob,a);
						Or = Ogam + Onu;
						cs2 = uni.cs2(a);
						dudt[taui] = (double) 1.0 / (a * hubble);
						dudt[etai] = ((double) 1.5 * ((Ob * U[thetabi]) + ((double) 4.0 / (double) 3.0) * (Ogam * U[thetagami] + Onu * U[thetanui])) / eps);
						double factor = ((a * omega_m) + (double) 4 * a * a * a * a * ((double) 1 - omega_m - omega_r))
						/ ((double) 2 * a * omega_m + (double) 2 * omega_r + (double) 2 * a * a * a * a * ((double) 1 - omega_m - omega_r));
						dudt[hdoti] =
						(-factor * U[hdoti] - ((double) 3.0 * (Oc * U[deltaci] + Ob * U[deltabi]) + (double) 6.0 * (Ogam * U[deltagami] + Onu * U[deltanui])));
						dudt[deltaci] = -(double) 0.5 * U[hdoti];
						dudt[deltabi] = -eps * U[thetabi] - (double) 0.5 * U[hdoti];
						dudt[deltagami] = -(double) 4.0 / (double) 3.0 * eps * U[thetagami] - ((double) 2.0 / (double) 3.0) * U[hdoti];
						dudt[deltanui] = -(double) 4.0 / (double) 3.0 * eps * U[thetanui] - ((double) 2.0 / (double) 3.0) * U[hdoti];
						dudt[thetabi] = -U[thetabi] + cs2 * eps * U[deltabi];
						dudt[thetagami] = eps * ((double) 0.25 * U[deltagami] - (double) 0.5 * U[F2i]);
						dudt[thetanui] = eps * ((double) 0.25 * U[deltanui] - (double) 0.5 * U[N2i]);
						dudt[F2i] = ((double) 8.0 / (double) 15.0) * eps * U[thetagami] + ((double) 4.0 / (double) 15.0) * U[hdoti] + ((double) 8.0 / (double) 5.0) * dudt[etai]
						- ((double) 3.0 / (double) 5.0) * eps * U[FLi + 3];
						dudt[N2i] = ((double) 8.0 / (double) 15.0) * eps * U[thetanui] + ((double) 4.0 / (double) 15.0) * U[hdoti] + ((double) 8.0 / (double) 5.0) * dudt[etai]
						- ((double) 3.0 / (double) 5.0) * eps * U[NLi + 3];
						dudt[GLi + 0] = -eps * U[GLi + 1];
						dudt[GLi + 1] = eps / (double) (3) * (U[GLi + 0] - (double) 2 * U[GLi + 2]);
						dudt[GLi + 2] = eps / (double) (5) * ((double) 2 * U[GLi + 1] - (double) 3 * U[GLi + 3]);
						for (int l = 3; l < LMAX - 1; l++) {
							dudt[FLi + l] = eps / (double) (2 * l + 1) * ((double) l * U[FLi - 1 + l] - (double) (l + 1) * U[FLi + 1 + l]);
							dudt[NLi + l] = eps / (double) (2 * l + 1) * ((double) l * U[NLi - 1 + l] - (double) (l + 1) * U[NLi + 1 + l]);
							dudt[GLi + l] = eps / (double) (2 * l + 1) * ((double) l * U[GLi - 1 + l] - (double) (l + 1) * U[GLi + 1 + l]);
						}
						dudt[FLi + LMAX - 1] = (eps * U[FLi + LMAX - 2]) / (double) (2 * LMAX - 1);
						dudt[NLi + LMAX - 1] = (eps * U[NLi + LMAX - 2]) / (double) (2 * LMAX - 1);
						dudt[GLi + LMAX - 1] = (eps * U[GLi + LMAX - 2]) / (double) (2 * LMAX - 1);
						for (int f = 0; f < NFIELD; f++) {
							U[f] = ((double) 1 - beta[i]) * U0[f] + beta[i] * (U[f] + dudt[f] * dloga * (double) 0.5);
						}
					}
				};

		auto compute_implicit_dudt =
				[&](double loga, double dloga) {
					a = exp(loga);
					double thetab = U[thetabi];
					double thetagam = U[thetagami];
					double F2 = U[F2i];
					double G0 = U[G0i];
					double G1 = U[G1i];
					double G2 = U[G2i];
					double thetab0 = thetab;
					double thetagam0 = thetagam;
					double F20 = F2;
					double G00 = G0;
					double G10 = G1;
					double G20 = G2;
					double sigma = uni.sigma_T(a);

					thetab = -((-(double) 3 * Ob * thetab0 - (double) 3 * dloga * Ob * sigma * thetab0 - (double) 4 * dloga * Ogam * sigma * thetagam0)
							/ ((double) 3 * Ob + (double) 3 * dloga * Ob * sigma + (double) 4 * dloga * Ogam * sigma));
					thetagam = -((-(double) 3 * dloga * Ob * sigma * thetab0 - (double) 3 * Ob * thetagam0 - (double) 4 * dloga * Ogam * sigma * thetagam0)
							/ ((double) 3 * Ob + (double) 3 * dloga * (double) Ob * sigma + (double) 4 * dloga * Ogam * sigma));
					F2 = -((-(double) 10 * F20 - (double) 4 * dloga * F20 * sigma - dloga * G00 * sigma - dloga * G20 * sigma)
							/ (((double) 1 + dloga * sigma) * ((double) 10 + (double) 3 * dloga * sigma)));
					G0 = -((-(double) 10 * G00 - (double) 5 * dloga * F20 * sigma - (double) 8 * dloga * G00 * sigma - (double) 5 * dloga * G20 * sigma)
							/ (((double) 1 + dloga * sigma) * ((double) 10 + (double) 3 * dloga * sigma)));
					G1 = G10 / ((double) 1 + dloga * sigma);
					G2 = -((-(double) 10 * G20 - dloga * F20 * sigma - dloga * G00 * sigma - (double) 4 * dloga * G20 * sigma)
							/ (((double) 1 + dloga * sigma) * ((double) 10 + (double) 3 * dloga * sigma)));
					std::array<double, NFIELD> dudt;
					for (int f = 0; f < NFIELD; f++) {
						dudt[f] = (double) 0.0;
					}
					dudt[thetabi] = (thetab - thetab0) / dloga;
					dudt[thetagami] = (thetagam - thetagam0) / dloga;
					dudt[F2i] = (F2 - F20) / dloga;
					dudt[G0i] = (G0 - G00) / dloga;
					dudt[G1i] = (G1 - G10) / dloga;
					dudt[G2i] = (G2 - G20) / dloga;
					for (int l = 3; l < LMAX - 1; l++) {
						dudt[GLi + l] = U[GLi + l] * ((double) 1 / ((double) 1 + dloga * sigma) - (double) 1) / dloga;
						dudt[FLi + l] = U[FLi + l] * ((double) 1 / ((double) 1 + dloga * sigma) - (double) 1) / dloga;
					}
					dudt[GLi + LMAX - 1] = U[GLi + LMAX - 1]
					* ((double) 1 / ((double) 1 + (sigma + (double) LMAX / (U[taui] * a * hubble) / ((double) 2 * (double) LMAX - (double) 1))) - (double) 1) / dloga;
					dudt[FLi + LMAX - 1] = U[FLi + LMAX - 1]
					* ((double) 1 / ((double) 1 + (sigma + (double) LMAX / (U[taui] * a * hubble) / ((double) 2 * (double) LMAX - (double) 1))) - (double) 1) / dloga;
					return dudt;
				};

		compute_explicit(0);
		double gamma = (double) 1.0 - (double) 1.0 / sqrt((double) 2);

		auto dudt1 = compute_implicit_dudt(loga + gamma * dloga, gamma * dloga);
		for (int f = 0; f < NFIELD; f++) {
			U[f] += dudt1[f] * ((double) 1.0 - (double) 2.0 * gamma) * dloga;
		}
		auto dudt2 = compute_implicit_dudt(loga + ((double) 1.0 - gamma) * dloga, gamma * dloga);
		for (int f = 0; f < NFIELD; f++) {
			U[f] += (dudt1[f] * ((double) -0.5 + (double) 2.0 * gamma) + dudt2[f] * (double) 0.5) * dloga;
		}

		compute_explicit(1);

		loga = loga0 + dloga;
	}
}

void einstein_boltzmann_interpolation_function(interp_functor<double>* m_k_func, interp_functor<double>* vel_k_func, cos_state* U, zero_order_universe* uni,
		double kmin, double kmax, double norm, int N, double astart, double astop, bool cont, double ns) {
	double dlogk = 1.0e-2;
	double logkmin = log(kmin) - dlogk;
	double logkmax = log(kmax) + dlogk;
	dlogk = (logkmax - logkmin) / (double) (N - 1);
	vector<double> ks(N);
	for (int i = 0; i < N; i++) {
		ks[i] = exp(logkmin + (double) i * dlogk);
	}
	vector<hpx::future<void>> futs;
	for (int block = 0; block < N; block++) {
		futs.push_back(hpx::async([block,&U,&uni,&ks,astart,astop,norm,N,cont,ns]() {
			einstein_boltzman_kernel(block, U, uni, ks.data(), astart, astop, norm, N, cont, ns);
		}));
	}
	hpx::wait_all(futs.begin(),futs.end());

	std::vector<double> m_k(N), vel_k(N);
	double oc = uni->params.omega_c;
	double ob = uni->params.omega_b;
	double om = oc + ob;
	oc /= om;
	ob /= om;
	const auto hubble = [uni](double a) {
		return hubble_function(a,uni->params.hubble, uni->params.omega_c + uni->params.omega_b, uni->params.omega_gam + uni->params.omega_nu);
	};
	double H = hubble(astop);
	for (int i = 0; i < N; i++) {
		double k = ks[i];
		double eps = k / (astop * H);
		m_k[i] = sqr(oc * U[i][deltaci] + ob * U[i][deltabi]);
		vel_k[i] = sqr((ob * (eps * U[i][thetabi] + (double) 0.5 * U[i][hdoti]) + oc * ((double) 0.5 * U[i][hdoti])));
		//	printf("%e %e\n", k, vel_k[i]);
	}
	build_interpolation_function(m_k_func, m_k, (double) exp(logkmin), (double) exp(logkmax));
	build_interpolation_function(vel_k_func, vel_k, (double) exp(logkmin), (double) exp(logkmax));
}

