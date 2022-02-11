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
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/hpx.hpp>
#include <fenv.h>

#define NRATES 20
#define NCOOL 15

const float Ktoev = float(1. / 11604.45);

float cooling_rate(float T, float z, species_t N) {
	float K1, K3, K5;
	T = std::max(1000.0f, T);

	{
		array<float, NRATES> K;
		float Tev = T * Ktoev;
		float lnT = log(Tev);
		float log10T = lnT / log(10);
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
			float k = c8;
			k = k * lnT + c7;
			k = k * lnT + c6;
			k = k * lnT + c5;
			k = k * lnT + c4;
			k = k * lnT + c3;
			k = k * lnT + c2;
			k = k * lnT + c1;
			k = k * lnT + c0;
			K1 = expf(k);
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
			float k = c8;
			k = k * lnT + c7;
			k = k * lnT + c6;
			k = k * lnT + c5;
			k = k * lnT + c4;
			k = k * lnT + c3;
			k = k * lnT + c2;
			k = k * lnT + c1;
			k = k * lnT + c0;
			K3 = expf(k);

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
			float k = c8;
			k = k * lnT + c7;
			k = k * lnT + c6;
			k = k * lnT + c5;
			k = k * lnT + c4;
			k = k * lnT + c3;
			k = k * lnT + c2;
			k = k * lnT + c1;
			k = k * lnT + c0;
			K5 = expf(k);
		}
	}

	const float ne = N.Hp - N.Hn + N.Hep + 2 * N.Hepp;
	const float T5 = T * 1e-5;
	array<float, NCOOL> C;
	float Tinv = 1.0 / T;
	float tmp = 1.0 / (1.0 + sqrt(T5));
	float tmp2 = pow(T, -.1687);
	C[0] = 0.f;
	C[1] = 7.5e-19 * expf(-118348 * Tinv) * ne * N.H;
	C[2] = 9.1e-27 * expf(-13179 * Tinv) * tmp2 * sqr(ne) * N.He;
	C[3] = 5.54e-17 * expf(-473638 * Tinv) * pow(T, -.397) * ne * N.Hep;
	C[4] = 2.18e-11 * K1 * ne * N.H;
	C[5] = 3.94e-11 * K3 * ne * N.He;
	C[6] = 8.72e-11 * K5 * ne * N.Hep;
	C[7] = 5.01e-27 * tmp * tmp2 * expf(-55338 * Tinv) * sqr(ne) * N.Hep;
	C[8] = 8.7e-27 * sqrt(T) * pow(T / 1000, -.2) / (1.0 + pow(T * 1e-6, .7)) * ne * N.Hp;
	C[9] = 1.55e-26 * pow(T, 0.3647) * ne * N.Hp;
	C[10] = 1.24e-13 * pow(T, -1.5) * (1 + .3 * exp(-94000 * Tinv)) * exp(-470000 / T) * ne * N.Hp;
	C[11] = 3.48e-26 * sqrt(T) * pow(T / 1000, -.2) / (1.0 + pow(T * 1e-6, .7)) * ne * N.Hepp;
	float Qn = powf(N.H2, 0.77) + 1.2 * powf(N.H, 0.77);
	float LrL, LrH;
	float x = log10f(T / 10000);
	double LvL;
	if (T > 4031) {
		LrL = 1.38e-22 * expf(-9243 * Tinv);
	} else {
		LrL = pow(10.0, -22.90 - 0.553 * x - 1.48 * sqr(x));
	}
	LrL *= Qn;
	if (T > 1087) {
		LrH = 3.9e-19 * expf(-6118 * Tinv);
	} else {
		LrH = pow(10, -19.24 + 0.474 * x - 1.247 * x * x);
	}
	if (T > 1635) {
		LvL = 1e-12 * sqrtf(T) * exp(-1000 / T);
	} else {
		LvL = 1.4e-13 * exp(T / 125 - sqr(T / 577));
	}
	LvL *= N.H * 8.18e-13;
	double LvH = 1.1e-13 * exp(-6744 / T);
	C[12] = N.H2 * (LrH * LrL / (LrL + LrH) + LvH * LvL / (LvL + LvH));
	C[13] = 1.43e-27 * sqrt(T) * (1.1 + 0.34 * expf(-sqr(5.5 - log10f(T)) / 3.0)) * ne * (N.Hp + N.Hep + N.Hepp);
	C[14] = 5.65e-36 * sqr(sqr(1 + z)) * (T - 2.73 * (1 + z)) * ne;
	float total = 0.0f;
	for (int i = 0; i < 15; i++) {
		total += C[i];
	}
	return total;
}

array<float, NRATES> chemistry_rates(float T) {
	T = std::max(1000.0f, T);
	array<float, NRATES> K;
	float Tev = T * Ktoev;
	float lnT = log(Tev);
	float log10T = lnT / log(10);
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
		float k = c8;
		k = k * lnT + c7;
		k = k * lnT + c6;
		k = k * lnT + c5;
		k = k * lnT + c4;
		k = k * lnT + c3;
		k = k * lnT + c2;
		k = k * lnT + c1;
		k = k * lnT + c0;
		K[1] = expf(k);
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
		float k = c9;
		k = k * lnT + c8;
		k = k * lnT + c7;
		k = k * lnT + c6;
		k = k * lnT + c5;
		k = k * lnT + c4;
		k = k * lnT + c3;
		k = k * lnT + c2;
		k = k * lnT + c1;
		k = k * lnT + c0;
		K[2] = expf(k);
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
		float k = c8;
		k = k * lnT + c7;
		k = k * lnT + c6;
		k = k * lnT + c5;
		k = k * lnT + c4;
		k = k * lnT + c3;
		k = k * lnT + c2;
		k = k * lnT + c1;
		k = k * lnT + c0;
		K[3] = expf(k);

	}

	{
		K[4] = 3.925E-13 * std::pow(Tev, -0.6353);
		K[4] += 1.544E-9 * std::pow(Tev, -1.5) * (0.3 * expf(-48.596 / Tev) + expf(-40.496 / Tev));
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
		float k = c8;
		k = k * lnT + c7;
		k = k * lnT + c6;
		k = k * lnT + c5;
		k = k * lnT + c4;
		k = k * lnT + c3;
		k = k * lnT + c2;
		k = k * lnT + c1;
		k = k * lnT + c0;
		K[5] = expf(k);
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
			float k = c7;
			k = k * lnT + c6;
			k = k * lnT + c5;
			k = k * lnT + c4;
			k = k * lnT + c3;
			k = k * lnT + c2;
			k = k * lnT + c1;
			k = k * lnT + c0;
			K[8] = expf(k);
		} else {
			K[8] = 1.428e-9;
		}
	}
	/*{
	 if (T < 6700) {
	 K[9] = 1.85e-23 * std::pow(T, 1.8);
	 } else {
	 K[9] = 5.81e-16 * std::pow(T / 56200.0, -0.6657 * std::log10(T / 56200.0));
	 }
	 }
	 {
	 K[10] = 6.4E-10;
	 }*/
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
		float k = c8;
		k = k * lnT + c7;
		k = k * lnT + c6;
		k = k * lnT + c5;
		k = k * lnT + c4;
		k = k * lnT + c3;
		k = k * lnT + c2;
		k = k * lnT + c1;
		k = k * lnT + c0;
		K[11] = expf(k);

	}

	{
		K[12] = 5.6e-11 * sqrt(T) * expf(-102124 / T);
	}

	{
//		K[13] = 1.067e-10 * std::pow(T, 2.012) * expf(-(4.463 / T) * std::pow(1 + 0.2472 * T, 3.512));
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
		float k = c8;
		k = k * lnT + c7;
		k = k * lnT + c6;
		k = k * lnT + c5;
		k = k * lnT + c4;
		k = k * lnT + c3;
		k = k * lnT + c2;
		k = k * lnT + c1;
		k = k * lnT + c0;
		K[14] = expf(k);
	}

	/*	{
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
	 float k = c9;
	 k = k * lnT + c9;
	 k = k * lnT + c7;
	 k = k * lnT + c6;
	 k = k * lnT + c5;
	 k = k * lnT + c4;
	 k = k * lnT + c3;
	 k = k * lnT + c2;
	 k = k * lnT + c1;
	 k = k * lnT + c0;
	 K[15] = expf(k);
	 } else {
	 K[15] = 2.5634E-9 * std::pow(T, 1.78186);
	 }

	 }*/
	{
		K[16] = 7e-8 * std::pow(T / 100, -0.5);
	}
	/*{
	 if (T < 1.719) {
	 K[17] = 2.291e-10 * std::pow(T, -0.4);
	 } else {
	 K[17] = 8.4258e-10 * std::pow(T, -1.4) * expf(-1.301 / T);
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
	 }*/
	//PRINT( "%e %e %e %e %e %e\n", K[1], K[2], K[3], K[4], K[5], K[6]);
	return K;
}

species_t species_t::fractions_to_number_density(float rho) const {
	species_t N;
	float tot = 0.f;
	rho *= constants::avo;
	for (int i = 0; i < NSPECIES; i++) {
		tot += n[i];
		N.n[i] = n[i] * rho;
	}
	float totinv = 1.f / tot;
	for (int i = 0; i < NSPECIES; i++) {
		N.n[i] *= totinv;
	}
	N.He *= 0.25;
	N.Hep *= 0.25;
	N.Hepp *= 0.25;
	N.H2 *= 0.5;
	return N;
}
void radiation_cross_sections(float z, float& sigma20, float& sigma21, float& sigma22, float& sigmaH, float& sigmaHe, float& sigmaHep) {
	float k0[6] = { 5.6e-13, 2 * 4.8e-15, 3.2e-13, 3.9e-24, 6.4e-24, 8.6e-26 };
	float alpha[6] = { 0.43, 0.30, 0.43, .43, .43, .3 };
	float beta[6] = { 1.0 / 1.95, 1.0 / 2.6, 1.0 / 1.95, 1 / 1.95, 1 / 2.1, 1 / 2.7 };
	float z0[6] = { 2.3, 2.3, 2.3, 2.3, 2.3, 2.3 };
	float z1[6] = { 0, 0, 0, 0, 0, 0 };
	float gamma[6] = { 0, 0, 0, 0, 0, 0 };
	float res[6];
	for (int i = 0; i < 6; i++) {
		float pwr = beta[i] * sqr(z - z0[i]);
		if (pwr < 50.0) {
			res[i] = k0[i] * pow(1 + z, alpha[i]) * expf(-pwr / (1 + gamma[i] * sqr(z + z1[i])));
		} else {
			res[i] = 0.0;
		}
	}
	sigma20 = res[0];
	sigma21 = res[2];
	sigma22 = res[1];
	sigmaH = res[3];
	sigmaHe = res[4];
	sigmaHep = res[5];
}

species_t chemistry_update(species_t species, float rho, float& T0, float z, float dt) {
	species_t N = species.fractions_to_number_density(rho);

	const auto N0 = N;
	float Ntot = N.H + N.Hp + N.Hn + N.He + N.Hep + N.Hepp + N.H2;
	float Hall = N.H + N.Hn + N.Hp + 2.0 * (N.H2);
//	PRINT("@!1111111111111111$ %e %e %e %e\n", N.H, N.Hn, N.Hp, N.H2);
	float Heall = N.He + N.Hep + N.Hepp;
	feenableexcept (FE_DIVBYZERO);
	feenableexcept (FE_INVALID);
	feenableexcept (FE_OVERFLOW);
	float oldT;
#define Power(a, b) pow(a, b)
#define Sqrt(a) sqrt(a)
#define Rule(a,b) a = b
#define List(...) __VA_ARGS__
	float sigma20, sigma21, sigma22, sigmaH, sigmaHe, sigmaHep;
	radiation_cross_sections(z, sigma20, sigma21, sigma22, sigmaH, sigmaHe, sigmaHep);
	const float oldN = N.H + 2.0 * N.Hp + N.H2 + N.He + 2.0 * N.Hep + 3.0 * N.Hepp;
	float newN;
	const auto test_temp =
			[N0,&newN,oldN,Heall,Hall,dt, sigma20, sigma21, sigma22, sigmaH, sigmaHe, sigmaHep,&N, z, T0](float T) {
				auto k = chemistry_rates(T);
				const auto compute_ne =
				[N0,Heall,Hall,k, dt, sigma20, sigma21, sigma22, sigmaH, sigmaHe, sigmaHep, T0, T](species_t& N, float ne) {
					float K1 = k[1];
					float K2 = k[2];
					float K3 = k[3];
					float K4 = k[4];
					float K5 = k[5];
					float K6 = k[6];
					float K7 = k[7];
					float K8 = k[8];
					float K9 = k[9];
					float K10 = k[10];
					float K11 = k[11];
					float K12 = k[12];
					float K13 = k[13];
					float K14 = k[14];
					float K15 = k[15];
					float K16 = k[16];
					float K17 = k[17];
					float K18 = k[18];
					float K19 = k[19];

					float& H = N.H;
					float& Hp = N.Hp;
					float& Hn = N.Hn;
					float& He = N.He;
					float& Hep = N.Hep;
					float& Hepp = N.Hepp;
					float& H2 = N.H2;
					float H0 = N0.H;
					float Hp0 = N0.Hp;
					float Hn0 = N0.Hn;
					float He0 = N0.He;
					float Hep0 = N0.Hep;
					float Hepp0 = N0.Hepp;
					float H20 = N0.H2;
					float Hp1, Hp2, H21, H22;
					Hn = (H * K7 * ne) / (Hp * K16 + H * K8 + K14 * ne);
					H = Hall - Hp - Hn - 2 * H2;
					List(List(Rule(Hp,(Hp0 - 2*dt*H2*K1*ne + dt*Hall*K1*ne - dt*Hn*K1*ne - 2*dt*H2*sigma20 + dt*Hall*sigma20 - dt*Hn*sigma20)/(1 + dt*K1*ne + dt*K2*ne + dt*sigma20))));
					H = Hall - Hp - Hn - 2 * H2;
					Hn = (H * K7 * ne) / (Hp * K16 + H * K8 + K14 * ne);
					H = Hall - Hp - Hn - 2 * H2;
					float H200 = H2;
					List(List(Rule(H2,(H20*Hp*K16 + H*H20*K8 + H20*K14*ne + dt*Power(H,2)*K7*K8*ne + 2*dt*H*H20*K7*K8*ne)/
											(Hp*K16 + dt*Power(Hp,2)*K11*K16 + H*K8 + dt*H*Hp*K11*K8 + K14*ne + dt*Hp*K11*K14*ne + dt*Hp*K12*K16*ne + dt*H*K12*K8*ne + 2*dt*H*K7*K8*ne + dt*K12*K14*Power(ne,2)))));
					H = Hall - Hp - Hn - 2 * H2;
					Hn = (H * K7 * ne) / (Hp * K16 + H * K8 + K14 * ne);
					H = Hall - Hp - Hn - 2 * H2;
					List(List(Rule(Hep,-((Hepp0*(dt*K3*ne - dt*K6*ne + dt*sigma21) + (1 + dt*K6*ne)*(-Hep0 - dt*Heall*K3*ne - dt*Heall*sigma21))/
													(-((dt*K3*ne - dt*K6*ne + dt*sigma21)*(-(dt*K5*ne) - dt*sigma22)) + (1 + dt*K6*ne)*(1 + dt*K3*ne + dt*K4*ne + dt*K5*ne + dt*sigma21 + dt*sigma22)))),
									Rule(Hepp,-((-Hepp0 - dt*Hepp0*K3*ne - dt*Hepp0*K4*ne - dt*Hep0*K5*ne - dt*Hepp0*K5*ne - Power(dt,2)*Heall*K3*K5*Power(ne,2) - dt*Hepp0*sigma21 - Power(dt,2)*Heall*K5*ne*sigma21 - dt*Hep0*sigma22 - dt*Hepp0*sigma22 -
															Power(dt,2)*Heall*K3*ne*sigma22 - Power(dt,2)*Heall*sigma21*sigma22)/
													(1 + dt*K3*ne + dt*K4*ne + dt*K5*ne + dt*K6*ne + Power(dt,2)*K3*K5*Power(ne,2) + Power(dt,2)*K3*K6*Power(ne,2) + Power(dt,2)*K4*K6*Power(ne,2) + dt*sigma21 + Power(dt,2)*K5*ne*sigma21 +
															Power(dt,2)*K6*ne*sigma21 + dt*sigma22 + Power(dt,2)*K3*ne*sigma22 + Power(dt,2)*sigma21*sigma22)))));
					He = Heall - Hep - Hepp;
					H = std::max(H,0.0f);
					Hp = std::max(Hp,0.0f);
					Hn = std::max(Hn,0.0f);
					H2 = std::max(H2,0.0f);
					He = std::max(He,0.0f);
					Hep = std::max(Hep,0.0f);
					Hepp = std::max(Hepp,0.0f);
					return Hp - Hn + Hep + 2 * Hepp;
				};
				float ne_max = Hall + 4 * Heall;
				float ne_min = 0.0;
				bool dotop = true;
				bool dobottom = true;
				float f_max, f_mid;
				for (int iter = 0; iter < 20; iter++) {
					float ne_mid = 0.5 * (ne_max + ne_min);
					if (dotop) {
						f_max = compute_ne(N, ne_max) - ne_max;
					}
					if (dobottom) {
						f_mid = compute_ne(N, ne_mid) - ne_mid;
					}
					if (f_max * f_mid < 0.0) {
						ne_min = ne_mid;
						dobottom = true;
						dotop = false;
					} else {
						ne_max = ne_mid;
						dobottom = false;
						dotop = true;
					}
				}

				newN = N.H + 2.0 * N.Hp + N.H2 + N.He + 2.0 * N.Hep + 3.0 * N.Hepp;
				float dedt_cool = cooling_rate(T,z, N) - N.H * sigmaH - N.He * sigmaHe - N.Hep * sigmaHep;
				return 1.5 * constants::kb *(newN * T - oldN * T0) + dt * dedt_cool;
			};

	float Tmin = 1e3;
	float Tmax = 1e8;
	float Tmid;
	for (int i = 0; i < 20; i++) {
		Tmid = sqrt(Tmax * Tmin);
		float fmax = test_temp(Tmax);
		float fmid = test_temp(Tmid);
		if (std::copysign(1.0, fmax) != std::copysign(1.0, fmid)) {
			Tmin = Tmid;
		} else {
			Tmax = Tmid;
		}
		//	PRINT("%e %e %e %e\n", T0, Tmin, Tmid, Tmax);
	}
	if (oldN / newN > 18. / 5. || oldN / newN < 5. / 18.) {
		PRINT("%e %e %e\n", oldN / newN, oldN, newN);
		PRINT("%e %e %e %e %e %e %e\n", N0.H, N0.Hp, N0.Hn, N0.H2, N0.He, N0.Hep, N0.Hepp);
		PRINT("%e %e %e %e %e %e %e\n", N.H, N.Hp, N.Hn, N.H2, N.He, N.Hep, N.Hepp);
		abort();
	}
	float Hfac = Hall / (N.H + N.Hn + N.Hp + 2.0 * (N.H2));
	float Hefac = Heall / (N.He + N.Hep + N.Hepp);
	N.H *= Hfac;
	N.Hp *= Hfac;
	N.Hn *= Hfac;
	N.H2 *= Hfac;
	N.He *= Hefac;
	N.Hep *= Hefac;
	N.Hepp *= Hefac;
	T0 = Tmid;
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
	/*for (float T = 1000.0; T < 1e8; T *= 1.1) {
	 auto k = chemistry_rates(T);
	 float K1 = k[1];
	 float K2 = k[2];
	 float K3 = k[3];
	 float K4 = k[4];
	 float K5 = k[5];
	 float K6 = k[6];
	 float K7 = k[7];
	 float K8 = k[8];
	 float K9 = k[9];
	 float K10 = k[10];
	 float K11 = k[11];
	 float K12 = k[12];
	 float K13 = k[13];
	 float K14 = k[14];
	 float K15 = k[15];
	 float K16 = k[16];
	 float K17 = k[17];
	 float K18 = k[18];
	 float K19 = k[19];

	 PRINT("%e %e %e %e %e %e %e %e %e %e %e %e \n", T, K1, K2, K3, K4, K5, K6, K7, K8, K11, K14, K16);
	 }*/
	float m = 0.0;

	for (int i = 0; true; i++) {
		float T = pow(10.0, 3.0 + rand1() * 5.0);
		float z = rand1() * 5;
		float rho = pow(10.0, -(24.0 + 3.0 * rand1())) * (z + 1);
		N.H = rand1();
		N.H2 = 0.0;
		N.Hp = rand1();
		N.Hn = 0.0;
		N.He = rand1();
		N.Hep = rand1();
		N.Hepp = rand1();
//		PRINT("!!!!!!!!!!!!!!!!!!!!!! %e %e %e %e\n", N.H + N.H2 + N.Hp + N.Hn, N.He, N.Hep, N.Hepp);
		float h0 = (N.H + N.H2 + N.Hp + N.Hn) / .75;
		float he0 = (N.He + N.Hep + N.Hepp) / .25;
		N.H /= h0;
		N.H2 /= h0;
		N.Hp /= h0;
		N.Hn /= h0;
		N.He /= he0;
		N.Hep /= he0;
		N.Hepp /= he0;
//		PRINT("!!!!!!!!!!!!!!!!!!!!!! %e %e %e %e  %e\n", N.H + N.H2 + N.Hp + N.Hn, N.He, N.Hep, N.Hepp, he0);
		//	print("-----------------------------------------------------------\n");
		float T0 = T;
		N = chemistry_update(N, rho, T, z, 1e15);
		print("%e %e %e %e %e %e %e %e %e %e\n", z, T0, T, N.H, N.Hp, N.Hn, N.H2, N.He, N.Hep, N.Hepp);
		float sigma20, sigma21, sigma22;
		//radiation_cross_sections(z, sigma20, sigma21, sigma22);
		if (m < N.H2) {
			m = N.H2;
			//	PRINT( "%e %e %e \n", rho, T, N.H2);
		}
		//	print("%e %e %e %e\n", z, sigma20, sigma21, sigma22);
	}
}

HPX_PLAIN_ACTION (chemistry_do_step);

static float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6), 1.0
		/ (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
		/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24), 1.0
		/ (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };

void chemistry_do_step(float a, int minrung, float t0, float adot, int dir) {




	vector<hpx::future<void>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<chemistry_do_step_action>(c, a, minrung, t0, adot, dir));
	}
	int nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc, nthreads, a, minrung,t0,dir,adot]() {
			const part_int b = (size_t) proc * sph_particles_size() / nthreads;
			const part_int e = (size_t) (proc+1) * sph_particles_size() / nthreads;
			vector<chem_attribs> chems;
			const float mass = get_options().sph_mass;
			const int N = get_options().neighbor_number;
			for( part_int i = b; i < e; i++) {
				int rung = sph_particles_rung(i);
				if( rung >= minrung ) {
					chem_attribs chem;
					float T = sph_particles_temperature(i,a);
					if( T > 1e7) {
						PRINT( "T-------------> %e\n", T);
					}
					chem.Hp = sph_particles_Hp(i);
					chem.Hn = sph_particles_Hn(i);
					chem.H2 = sph_particles_H2(i);
					chem.Hep = sph_particles_Hep(i);
					chem.Hepp = sph_particles_Hepp(i);
					chem.K = sph_particles_ent(i);
					if( dir == 1 ) {
						double cv = 1.5 + 0.5* chem.H2 / (1. - .75 * get_options().Y - 0.5 * chem.H2);
						double gamma = 1. + 1. / cv;
						double dt = rung_dt[rung] * t0;
						chem.K *= exp((5. - 3.*gamma)*adot/a*dt);
					}
					chem.rho = mass * float(3.0f / 4.0f / M_PI * N) * powf(sph_particles_smooth_len(i),-3);
			//		PRINT( "%e\n", chem.rho);
					chem.dt = 0.5f * t0 * rung_dt[rung];
					chems.push_back(chem);
				}
			}
			cuda_chemistry_step(chems, a);
			int j = 0;
			for( part_int i = b; i < e; i++) {
				int rung = sph_particles_rung(i);
				if( rung >= minrung ) {
					chem_attribs chem = chems[j++];
					//		PRINT( "%e %e %e %e %e %e\n", chem.Hp, chem.Hn, chem.H2, chem.Hep, chem.Hepp, chem.K);
					if( dir == -1 ) {
						double cv = 1.5 + 0.5* chem.H2 / (1. - .75 * get_options().Y - 0.5 * chem.H2);
						double gamma = 1. + 1. / cv;
						double dt = rung_dt[rung] * t0;
						chem.K *= exp((5.-3.*gamma)*adot/a*dt);
		//				PRINT( "%e %e %e %e\n", cv,(5.f-3.f*gamma)*adot/a*dt, chem.H2, exp((5.-3.*gamma)*adot/a*dt));
					}
					sph_particles_Hp(i) = chem.Hp;
					sph_particles_Hn(i) = chem.Hn;
					sph_particles_H2(i) = chem.H2;
					sph_particles_Hep(i) = chem.Hep;
					sph_particles_Hepp(i) = chem.Hepp;
					sph_particles_ent(i) = chem.K;
					sph_particles_tcool(i) = chem.tcool;
					if( chem.K < 0.0 ) {
						PRINT( "Chem routines made negative entropy!\n");
					}
				}
			}

		}));
	}

	hpx::wait_all(futs.begin(), futs.end());
}
