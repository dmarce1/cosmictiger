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
#include <cosmictiger/chemistry.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/constants.hpp>
#include <fenv.h>

#define NRATES 20

const double Ktoev = double(1. / 11604.45);

array<double, NRATES> chemistry_rates(double T) {
	array<double, NRATES> K;
	double Tev = T * Ktoev;
	double lnT = log(Tev);
	double log10T = lnT / log(10);
	{
		const auto c0 = -32.71396786;
		const auto c1 = 13.536556;
		const auto c2 = -5.73932875;
		const auto c3 = 1.56315498;
		const auto c4 = -0.2877056;
		const auto c5 = 3.48255977e-2;
		const auto c6 = -2.63197617e-3;
		const auto c7 = 1.11954395e-4;
		const auto c8 = -2.03914985e-6;
		double k = c8;
		k = k * lnT + c7;
		k = k * lnT + c6;
		k = k * lnT + c5;
		k = k * lnT + c4;
		k = k * lnT + c3;
		k = k * lnT + c2;
		k = k * lnT + c1;
		k = k * lnT + c0;
		K[1] = exp(k);
	}
	{
		const auto c0 = -28.6130338;
		const auto c1 = -0.72411256;
		const auto c2 = -2.02604473e-2;
		const auto c3 = -2.38086188e-3;
		const auto c4 = -3.21260521e-4;
		const auto c5 = -1.42150291e-5;
		const auto c6 = 4.98910892e-6;
		const auto c7 = 5.75561414e-7;
		const auto c8 = -1.85676704e-8;
		const auto c9 = -3.07113524e-9;
		const auto lnT = std::log(T);
		double k = c9;
		k = k * lnT + c8;
		k = k * lnT + c7;
		k = k * lnT + c6;
		k = k * lnT + c5;
		k = k * lnT + c4;
		k = k * lnT + c3;
		k = k * lnT + c2;
		k = k * lnT + c1;
		k = k * lnT + c0;
		K[2] = std::exp(k);
	}

	{
		const auto c0 = -44.09864886;
		const auto c1 = 23.91596563;
		const auto c2 = -10.7532302;
		const auto c3 = 3.05803875;
		const auto c4 = -0.56851189;
		const auto c5 = 6.79539123e-2;
		const auto c6 = -5.00905610e-3;
		const auto c7 = 2.06723616e-4;
		const auto c8 = -3.64916141e-6;
		const auto lnT = std::log(T);
		double k = c8;
		k = k * lnT + c7;
		k = k * lnT + c6;
		k = k * lnT + c5;
		k = k * lnT + c4;
		k = k * lnT + c3;
		k = k * lnT + c2;
		k = k * lnT + c1;
		k = k * lnT + c0;
		K[3] = exp(k);

	}

	{
		K[4] = 3.925E-13 * std::pow(Tev, -0.6353);
		K[4] += 1.544E-9 * std::pow(Tev, -1.5) * std::exp(-48.596 / T) * (0.3 + std::exp(8.10 / T));
	}
	{
		const auto c0 = -68.71040990;
		const auto c1 = 43.93347633;
		const auto c2 = -18.4806699;
		const auto c3 = 4.70162649;
		const auto c4 = -0.76924663;
		const auto c5 = 8.113042E-2;
		const auto c6 = -5.32402063E-3;
		const auto c7 = 1.97570531E-4;
		const auto c8 = -3.16558106E-6;
		const auto lnT = std::log(T);
		double k = c8;
		k = k * lnT + c7;
		k = k * lnT + c6;
		k = k * lnT + c5;
		k = k * lnT + c4;
		k = k * lnT + c3;
		k = k * lnT + c2;
		k = k * lnT + c1;
		k = k * lnT + c0;
		K[5] = exp(k);
	}

	{

		K[6] = 3.36e-10 * std::pow(T, -0.5) * std::pow(T / 1000, -0.2) / (1.0 + std::pow(T / 1e6, 0.7));
	}

	{
		if (T < 6000) {
			const auto c0 = 1.429e-18;
			const auto c1 = std::pow(T, 0.762);
			const auto c2 = std::pow(T, 0.1523 * log10(T));
			const auto c3 = std::pow(T, -3.247e-2 * log10(T) * log10(T));
			K[7] = c0 * c1 * c2 * c3;
		} else {
			const auto c0 = 3.802e-17;
			const auto c1 = std::pow(T, 0.1998 * log10(T));
			const auto c2 = std::pow(10, 4.0415e-5 * std::pow(log10(T), 6));
			const auto c3 = std::pow(10, -5.447e-3 * std::pow(log10(T), 4));
			K[7] = c0 * c1 * c2 * c3;
		}
	}
	{
		if (T > 0.1) {
			const auto c0 = -20.06913897;
			const auto c1 = 0.22898;
			const auto c2 = 3.5998377E-2;
			const auto c3 = -4.55512E-3;
			const auto c4 = -3.10511544E-4;
			const auto c5 = 1.0732940E-4;
			const auto c6 = -8.36671960E-6;
			const auto c7 = 2.23830623E-7;
			const auto lnT = std::log(T);
			double k = c7;
			k = k * lnT + c6;
			k = k * lnT + c5;
			k = k * lnT + c4;
			k = k * lnT + c3;
			k = k * lnT + c2;
			k = k * lnT + c1;
			k = k * lnT + c0;
			K[8] = exp(k);
		} else {
			K[8] = 1.428e-9;
		}
	}
	{
		if (T < 6700) {
			K[9] = 1.85e-23 * std::pow(T, 1.8);
		} else {
			K[9] = 5.81e-16 * std::pow(T / 56200.0, -0.6657 * std::log10(T / 56200.0));
		}
	}
	{
		K[10] = 6.4E-10;
	}
	{
		const auto c0 = -24.24914687;
		const auto c1 = 3.40082444;
		const auto c2 = -3.89800396;
		const auto c3 = 2.04558782;
		const auto c4 = -0.541618285;
		const auto c5 = 8.41077503E-2;
		const auto c6 = -7.87902615E-3;
		const auto c7 = 4.13839842E-4;
		const auto c8 = -9.36345888E-6;
		const auto lnT = std::log(T);
		double k = c8;
		k = k * lnT + c7;
		k = k * lnT + c6;
		k = k * lnT + c5;
		k = k * lnT + c4;
		k = k * lnT + c3;
		k = k * lnT + c2;
		k = k * lnT + c1;
		k = k * lnT + c0;
		K[11] = exp(k);

	}

	{
		K[12] = 5.6e-11 * sqrt(T) * exp(-102124 / T);
	}

	{
		K[13] = 1.067e-10 * std::pow(T, 2.012) * std::exp(-(4.463 / T) * std::pow(1 + 0.2472 * T, 3.512));
	}

	{

		const auto c0 = -18.01849334;
		const auto c1 = 2.3608522;
		const auto c2 = -0.28274430;
		const auto c3 = 1.62331664E-2;
		const auto c4 = -3.36501203E-2;
		const auto c5 = 1.17832978E-2;
		const auto c6 = -1.65619470E-3;
		const auto c7 = 1.06827520E-4;
		const auto c8 = -2.63128581E-6;
		const auto lnT = std::log(T);
		double k = c8;
		k = k * lnT + c7;
		k = k * lnT + c6;
		k = k * lnT + c5;
		k = k * lnT + c4;
		k = k * lnT + c3;
		k = k * lnT + c2;
		k = k * lnT + c1;
		k = k * lnT + c0;
		K[14] = std::exp(k);
	}

	{
		if (T > 0.1) {
			const auto c0 = -20.37260896;
			const auto c1 = 1.13944933;
			const auto c2 = -0.14210135;
			const auto c3 = 8.4644554E-3;
			const auto c4 = -1.4327641E-3;
			const auto c5 = 2.0122503E-4;
			const auto c6 = 8.6639632E-5;
			const auto c7 = -2.5850097E-5;
			const auto c8 = 2.4555012E-6;
			const auto c9 = -8.0683825E-8;
			const auto lnT = std::log(T);
			double k = c9;
			k = k * lnT + c9;
			k = k * lnT + c7;
			k = k * lnT + c6;
			k = k * lnT + c5;
			k = k * lnT + c4;
			k = k * lnT + c3;
			k = k * lnT + c2;
			k = k * lnT + c1;
			k = k * lnT + c0;
			K[15] = std::exp(k);
		} else {
			K[15] = 2.5634E-9 * std::pow(T, 1.78186);
		}

	}
	{
		K[16] = 7e-8 * std::pow(T / 100, -0.5);
	}
	{
		if (T < 1.719) {
			K[17] = 2.291e-10 * std::pow(T, -0.4);
		} else {
			K[17] = 8.4258e-10 * std::pow(T, -1.4) * std::exp(-1.301 / T);
		}
	}

	{
		if (T < 617) {
			K[18] = 1e-8;
		} else {
			K[18] = 1.32e-6 * std::pow(T, -0.76);
		}
	}
	{
		K[19] = 5e-7 * std::sqrt(100 / T);
	}
	return K;
}

species_t species_t::fractions_to_number_density(double rho) const {
	species_t N;
	double tot = 0.f;
	rho *= constants::avo;
	for (int i = 0; i < NSPECIES; i++) {
		tot += n[i];
		N.n[i] = n[i] * rho;
	}
	double totinv = 1.f / tot;
	for (int i = 0; i < NSPECIES; i++) {
		N.n[i] *= totinv;
	}
	N.He *= 0.25;
	N.Hep *= 0.25;
	N.Hepp *= 0.25;
	N.H2 *= 0.5;
	return N;
}
species_t species_t::number_density_to_fractions() const {
	species_t f;
	f = *this;
	f.He *= 4.0;
	f.Hep *= 4.0;
	f.Hepp *= 4.0;
	f.H2 *= 2.0;
	double rho = 0.f;
	for (int i = 0; i < NSPECIES; i++) {
		rho += f.n[i];
	}
	double rhoinv = 1.f / rho;
	for (int i = 0; i < NSPECIES; i++) {
		f.n[i] *= rhoinv;
	}
	return f;
}

static double rand1() {
	return ((double) rand() + 0.5) / (double) RAND_MAX;
}

species_t chemistry_update(species_t species, double rho, double T, double dt) {
	species_t N = species.fractions_to_number_density(rho);

	const auto N0 = N;
	double Ntot = N.H + N.Hp + N.Hn + N.He + N.Hep + N.Hepp + N.H2;
	double Hall = N.H + N.Hn + N.Hp + 2.0 * (N.H2);
	double Heall = N.He + N.Hep + N.Hepp;
	feenableexcept (FE_DIVBYZERO);
	feenableexcept (FE_INVALID);
	feenableexcept (FE_OVERFLOW);

#define Power(a, b) pow(a, b)
#define Sqrt(a) sqrt(a)
#define Rule(a,b) a = b
#define List(...) __VA_ARGS__

	auto k = chemistry_rates(T);

	const auto compute_ne =
			[N0,Heall,Hall,k, dt](species_t& N, double ne) {
				double K1 = k[1];
				double K2 = k[2];
				double K3 = k[3];
				double K4 = k[4];
				double K5 = k[5];
				double K6 = k[6];
				double K7 = k[7];
				double K8 = k[8];
				double K9 = k[9];
				double K10 = k[10];
				double K11 = k[11];
				double K12 = k[12];
				double K13 = k[13];
				double K14 = k[14];
				double K15 = k[15];
				double K16 = k[16];
				double K17 = k[17];
				double K18 = k[18];
				double K19 = k[19];

				double& H = N.H;
				double& Hp = N.Hp;
				double& Hn = N.Hn;
				double& He = N.He;
				double& Hep = N.Hep;
				double& Hepp = N.Hepp;
				double& H2 = N.H2;
				double H0 = N0.H;
				double Hp0 = N0.Hp;
				double Hn0 = N0.Hn;
				double He0 = N0.He;
				double Hep0 = N0.Hep;
				double Hepp0 = N0.Hepp;
				double H20 = N0.H2;
				Hn = 0.0;
				double Hp1, Hp2, H21, H22;
				Rule(Hp,(Hp0 - 2*dt*H20*K1*ne + dt*Hall*K1*ne - dt*Hn*K1*ne)/(1 + dt*K1*ne + dt*K2*ne));
				H = Hall - Hp - Hn - 2 * H2;
				Hn = (H * K7 * ne) / (Hp * K16 + H * K8 + K14 * ne);
				H = Hall - Hp - Hn - 2 * H2;
				List(List(Rule(H2,(H20 + dt*H*Hn*K8)/(1 + dt*Hp*K11 + dt*H*K13 + dt*K12*ne))));
				H = Hall - Hp - Hn - 2 * H2;
				Rule(Hep,
						-((-Hep0 - dt*Heall*K3*ne + dt*Hepp0*K3*ne - dt*Hep0*K6*ne - dt*Hepp0*K6*ne - Power(dt,2)*Heall*K3*K6*Power(ne,2))/ (1 + dt*K3*ne + dt*K4*ne + dt*K5*ne + dt*K6*ne + Power(dt,2)*K3*K5*Power(ne,2) + Power(dt,2)*K3*K6*Power(ne,2) + Power(dt,2)*K4*K6*Power(ne,2))));
				Rule(Hepp,
						-((-Hepp0 - dt*Hepp0*K3*ne - dt*Hepp0*K4*ne - dt*Hep0*K5*ne - dt*Hepp0*K5*ne - Power(dt,2)*Heall*K3*K5*Power(ne,2))/ (1 + dt*K3*ne + dt*K4*ne + dt*K5*ne + dt*K6*ne + Power(dt,2)*K3*K5*Power(ne,2) + Power(dt,2)*K3*K6*Power(ne,2) + Power(dt,2)*K4*K6*Power(ne,2))));
				He = Heall - Hep - Hepp;
//				PRINT( "%e %e %e %e %e %e %e %e\n", H, Hp, Hn, H2, He, Hep, Hepp, ne);
				H = std::max(H,0.0);
				Hp = std::max(Hp,0.0);
				Hn = std::max(Hn,0.0);
				H2 = std::max(H2,0.0);
				He = std::max(He,0.0);
				Hep = std::max(Hep,0.0);
				Hepp = std::max(Hepp,0.0);
				return Hp - Hn + Hep + Hepp;
			};

	double ne_max = Hall + 4 * Heall;
	double ne_min = 0.0;
	do {
		double ne_mid = 0.5 * (ne_max + ne_min);
		double f_max = compute_ne(N, ne_max) - ne_max;
		double f_mid = compute_ne(N, ne_mid) - ne_min;
		if (f_max * f_mid < 0.0) {
			ne_min = ne_mid;
		} else {
			ne_max = ne_mid;
		}
		auto fracs = N.number_density_to_fractions();
		//	PRINT("%e %e %e %e %e %e %e %e %e %e\n", ne_min, ne_max, fracs.H, fracs.Hp, fracs.Hn, fracs.H2, fracs.He, fracs.Hep, fracs.Hepp, (N.Hp-N.Hn+2*N.Hepp+N.Hepp) / (Hall + 4*Heall));
	} while ((ne_max - ne_min) / ne_max > 1e-6);

	double Hfac = Hall / (N.H + N.Hn + N.Hp + 2.0 * (N.H2));
	double Hefac = Heall / (N.He + N.Hep + N.Hepp);
	N.H *= Hfac;
	N.Hp *= Hfac;
	N.Hn *= Hfac;
	N.H2 *= Hfac;
	N.He *= Hefac;
	N.Hep *= Hefac;
	N.Hepp *= Hefac;
	return N.number_density_to_fractions();
}

void chemistry_test() {
	species_t N;
	N.H = 0.71;
	N.H2 = 0.02;
	N.Hp = 0.01;
	N.Hn = 0.01;
	N.He = 0.23;
	N.Hep = 0.01;
	N.Hepp = 0.01;
/*	for( double T = 1000.0; T < 1e8; T *= 1.1) {
		auto k = chemistry_rates(T);
		PRINT( "%e %e %e %e %e\n", T, k[7], k[16], k[8], k[14]);
	}
	return;*/
	double m = 0.0;
	while(1) {
		double T = pow(10.0,3.0 + rand1()*5.0);
		double rho = pow(10.0, -(20.0+6.0*rand1()));
		N.H = rand1();
		N.H2 = 0.0;
		N.Hp = rand1();
		N.Hn = 0.0;
		N.He = rand1();
		N.Hep = rand1();
		N.Hepp = rand1();
		double t0 = N.H + 2 * N.H2 + N.Hp + N.Hn + N.He + N.Hep + N.Hepp;
		N.H /= t0;
		N.H2 /= t0;
		N.Hp /= t0;
		N.Hn /= t0;
		N.He /= t0;
		N.Hep /= t0;
		N.Hepp /= t0;
	//	print("-----------------------------------------------------------\n");
	//	print("%e %e %e %e %e %e %e\n", N.H, N.Hp, N.Hn, N.H2, N.He, N.Hep, N.Hepp);
		N = chemistry_update(N, rho, T, 1.0e16);
		if( m < N.H2) {
			m = N.H2;
		//	PRINT( "%e %e %e \n", rho, T, N.H2);
		}
		print("%e %e %e %e %e %e %e\n", N.H, N.Hp, N.Hn, N.H2, N.He, N.Hep, N.Hepp);
	}
}

