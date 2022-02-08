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
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/options.hpp>

struct chemistry_params {
	double code_to_cm;
	double code_to_g;
	double code_to_s;
	float a;
	float Hefrac;
	float dt;
};

#define CHEM_BLOCK_SIZE 32

__device__ float test_electron_fraction(float ne, species_t N0, species_t& N, float T, float dt, float z, float K1, float K2, float K3, float K4, float K5,
		float K6, float K7, float K8, float K9, float K11, float K12, float K14, float K16, float sigma20, float sigma21, float sigma22, int& flops) {
	float& H = N.H;
	float& Hp = N.Hp;
	float& Hn = N.Hn;
	float& He = N.He;
	float& Hep = N.Hep;
	float& Hepp = N.Hepp;
	float& H2 = N.H2;
	float Hp0 = N0.Hp;
	float Hep0 = N0.Hep;
	float Hepp0 = N0.Hepp;
	float H20 = N0.H2;
	float Hall = H + Hp + 2.f * H2 + Hn;																																							// 4
	float Heall = He + Hep + Hepp;																																									// 2
	N = N0;
#define Power(a, b) powf(a, b)
#define Sqrt(a) sqrtf(a)
#define Rule(a,b) a = b
#define List(...) __VA_ARGS__
//	PRINT("in %e %e %e %e %e %e %e\n", H, Hp, Hn, H2, He, Hep, Hepp);
	Hn = (H * K7 * ne) / (Hp * K16 + H * K8 + K14 * ne);																																	  // 11
	H = Hall - Hp - Hn - 2 * H2;																																										// 4
	List(
			List( Rule(Hp, (Hp0 - 2 * dt * H2 * K1 * ne + dt * Hall * K1 * ne - dt * Hn * K1 * ne - 2 * dt * H2 * sigma20 + dt * Hall * sigma20 - dt * Hn * sigma20) / (1 + dt * K1 * ne + dt * K2 * ne + dt * sigma20)))); // 35
	H = Hall - Hp - Hn - 2 * H2;																																										// 4
	Hn = (H * K7 * ne) / (Hp * K16 + H * K8 + K14 * ne);																																	// 11
	H = Hall - Hp - Hn - 2 * H2;
	float H200 = H2;
	List(
			List(Rule(H2,(H20*Hp*K16 + H*H20*K8 + H20*K14*ne + dt*sqr(H)*K7*K8*ne + 2*dt*H*H20*K7*K8*ne)/ (Hp*K16 + dt*sqr(Hp)*K11*K16 + H*K8 + dt*H*Hp*K11*K8 + K14*ne + dt*Hp*K11*K14*ne + dt*Hp*K12*K16*ne + dt*H*K12*K8*ne + 2*dt*H*K7*K8*ne + dt*K12*K14*sqr(ne))))); //62

	if (isnan(H2)) {
		H2 = H200;
		PRINT("%e %e %e %e %e %e %e \n", dt,H,K7,K8,ne, H*H, K7*K8*ne );
		__trap();
	}
	H = Hall - Hp - Hn - 2 * H2;																																										// 4
	Hn = (H * K7 * ne) / (Hp * K16 + H * K8 + K14 * ne);																																	// 11
	H = Hall - Hp - Hn - 2 * H2;																																										// 4
	Hep = -((Hepp0 * (dt * K3 * ne - dt * K6 * ne + dt * sigma21) + (1 + dt * K6 * ne) * (-Hep0 - dt * Heall * K3 * ne - dt * Heall * sigma21))
			/ (-((dt * K3 * ne - dt * K6 * ne + dt * sigma21) * (-(dt * K5 * ne) - dt * sigma22))
					+ (1 + dt * K6 * ne) * (1 + dt * K3 * ne + dt * K4 * ne + dt * K5 * ne + dt * sigma21 + dt * sigma22)));             // 59
	Hepp = -((-Hepp0 - dt * Hepp0 * K3 * ne - dt * Hepp0 * K4 * ne - dt * Hep0 * K5 * ne - dt * Hepp0 * K5 * ne - sqr(dt) * Heall * K3 * K5 * Power(ne, 2)
			- dt * Hepp0 * sigma21 - sqr(dt) * Heall * K5 * ne * sigma21 - dt * Hep0 * sigma22 - dt * Hepp0 * sigma22
			- sqr(dt) * Heall * K3 * ne * sigma22 - sqr(dt) * Heall * sigma21 * sigma22)
			/ (1 + dt * K3 * ne + dt * K4 * ne + dt * K5 * ne + dt * K6 * ne + sqr(dt) * K3 * K5 * Power(ne, 2) + sqr(dt) * K3 * K6 * Power(ne, 2)
					+ sqr(dt) * K4 * K6 * Power(ne, 2) + dt * sigma21 + sqr(dt) * K5 * ne * sigma21 + sqr(dt) * K6 * ne * sigma21 + dt * sigma22
					+ sqr(dt) * K3 * ne * sigma22 + sqr(dt) * sigma21 * sigma22));												//112
	He = Heall - Hep - Hepp;																																											// 2
//	PRINT("out %e\n", dt);
/*	if (fmaxf(H, 0.0f) + fmaxf(Hp, 0.0f) + fmaxf(Hn, 0.0f) + 2 * fmaxf(H2, 0.0f) <= 0.0) {
		PRINT("ZEROS ONE %e %e %e %e %e %e %e\n", N.H, N.Hp, N.Hn, N.H2, N.He, N.Hep, N.Hepp);
		__trap();
	}*/
	H = fmaxf(H, 0.0f);																																													// 1
	Hp = fmaxf(Hp, 0.0f);																																												// 1
	Hn = fmaxf(Hn, 0.0f);																																												// 1
	H2 = fmaxf(H2, 0.0f);																																												// 1
	He = fmaxf(He, 0.0f);																																												// 1
	Hep = fmaxf(Hep, 0.0f);																																												// 1
	Hepp = fmaxf(Hepp, 0.0f);																																											// 1
/*	if (H + Hp + Hn + 2 * H2 <= 0.0) {
		PRINT("ZEROS TWO %e %e %e %e %e %e %e\n", N.H, N.Hp, N.Hn, N.H2, N.He, N.Hep, N.Hepp);
		__trap();
	}*/
	flops += 332;
	return Hp - Hn + Hep + 2 * Hepp - ne;
}

__device__ float test_temperature(species_t N0, species_t& N, float T0, float T, float dt, float z, int& flops) {

	T = fmaxf(1e3f, T);							// 1
	T = fminf(1e8f, T);							// 1
	float Tev = T / constants::evtoK;					// 4
	float Tevinv = 1.f / Tev;					// 4
	float lnTev = log(Tev);						// 8
	float log10T = log10(T);						// 8
	float T3 = T * 1e-3f;					// 1
	float sqrtTinv = rsqrtf(T);
	float sqrtT = sqrtf(T);
	float Tinv = 1.f / T;
	float T6 = T * 1e-6f;					// 1
	float K1, K2, K3, K4, K5, K6, K7, K8, K9, K11, K12, K14, K16;
	{
		const float c0 = -32.71396786f;
		const float c1 = 13.536556f;
		const float c2 = -5.73932875f;
		const float c3 = 1.56315498f;
		const float c4 = -0.2877056f;
		const float c5 = 3.48255977e-2f;
		const float c6 = -2.63197617e-3f;
		const float c7 = 1.11954395e-4f;
		const float c8 = -2.03914985e-6f;
		float k = c8;
		k = fmaf(k, lnTev, c7);
		k = fmaf(k, lnTev, c6);
		k = fmaf(k, lnTev, c5);
		k = fmaf(k, lnTev, c4);
		k = fmaf(k, lnTev, c3);
		k = fmaf(k, lnTev, c2);
		k = fmaf(k, lnTev, c1);
		k = fmaf(k, lnTev, c0);
		K1 = expf(k);
	}
	{
		const float c0 = -28.6130338f;
		const float c1 = -0.72411256f;
		const float c2 = -2.02604473e-2f;
		const float c3 = -2.38086188e-3f;
		const float c4 = -3.21260521e-4f;
		const float c5 = -1.42150291e-5f;
		const float c6 = 4.98910892e-6f;
		const float c7 = 5.75561414e-7f;
		const float c8 = -1.85676704e-8f;
		const float c9 = -3.07113524e-9f;
		float k = c9;
		k = fmaf(k, lnTev, c8);
		k = fmaf(k, lnTev, c7);
		k = fmaf(k, lnTev, c6);
		k = fmaf(k, lnTev, c5);
		k = fmaf(k, lnTev, c4);
		k = fmaf(k, lnTev, c3);
		k = fmaf(k, lnTev, c2);
		k = fmaf(k, lnTev, c1);
		k = fmaf(k, lnTev, c0);
		K2 = expf(k);
	}
	{
		const float c0 = -44.09864886f;
		const float c1 = 23.91596563f;
		const float c2 = -10.7532302f;
		const float c3 = 3.05803875f;
		const float c4 = -0.56851189f;
		const float c5 = 6.79539123e-2f;
		const float c6 = -5.00905610e-3f;
		const float c7 = 2.06723616e-4f;
		const float c8 = -3.64916141e-6f;
		float k = c8;
		k = fmaf(k, lnTev, c7);
		k = fmaf(k, lnTev, c6);
		k = fmaf(k, lnTev, c5);
		k = fmaf(k, lnTev, c4);
		k = fmaf(k, lnTev, c3);
		k = fmaf(k, lnTev, c2);
		k = fmaf(k, lnTev, c1);
		k = fmaf(k, lnTev, c0);
		K3 = expf(k);
	}
	{
		K4 = 3.925E-13f * powf(Tev, -0.6353f);																					// 5
		K4 += 1.544E-9f * pow(Tev, -1.5f) * (0.3f * expf(-48.596f * Tevinv) + expf(-40.496f * Tevinv));    // 18
	}
	{
		const float c0 = -68.71040990f;
		const float c1 = 43.93347633f;
		const float c2 = -18.4806699f;
		const float c3 = 4.70162649f;
		const float c4 = -0.76924663f;
		const float c5 = 8.113042E-2f;
		const float c6 = -5.32402063E-3f;
		const float c7 = 1.97570531E-4f;
		const float c8 = -3.16558106E-6f;
		float k = c8;
		k = fmaf(k, lnTev, c7);
		k = fmaf(k, lnTev, c6);
		k = fmaf(k, lnTev, c5);
		k = fmaf(k, lnTev, c4);
		k = fmaf(k, lnTev, c3);
		k = fmaf(k, lnTev, c2);
		k = fmaf(k, lnTev, c1);
		k = fmaf(k, lnTev, c0);
		K5 = expf(k);
	}
	{

		K6 = 3.36e-10f * sqrtTinv * pow(T3, -0.2f) / (1.0f + powf(T6, 0.7));
	}
	{
		if (T < 6000.f) {																// 1
			const float c0 = 1.429e-18f;
			const float c1 = powf(T, 0.762f);									// 8
			const float c2 = powf(T, 0.1523f * log10T);						// 9
			const float c3 = powf(T, -3.247e-2f * sqr(log10T));		   // 10
			K7 = c0 * c1 * c2 * c3;											    	// 3
			flops += 31;
		} else {
			const float c0 = 3.802e-17;
			const float c1 = powf(T, 0.1998f * log10T);						// 9
			const float c2 = powf(10.f, 4.0415e-5f * pow(log10T, 6));   // 17
			const float c3 = powf(10.f, -5.447e-3f * pow(log10T, 4));   // 17
			K7 = c0 * c1 * c2 * c3;                                   // 3
			flops += 47;
		}
	}
	{
		if (T > 0.1) {
			const float c0 = -20.06913897f;
			const float c1 = 0.22898f;
			const float c2 = 3.5998377E-2f;
			const float c3 = -4.55512E-3f;
			const float c4 = -3.10511544E-4f;
			const float c5 = 1.0732940E-4f;
			const float c6 = -8.36671960E-6f;
			const float c7 = 2.23830623E-7f;
			float k = c7;
			k = fmaf(k, lnTev, c6);
			k = fmaf(k, lnTev, c5);
			k = fmaf(k, lnTev, c4);
			k = fmaf(k, lnTev, c3);
			k = fmaf(k, lnTev, c2);
			k = fmaf(k, lnTev, c1);
			k = fmaf(k, lnTev, c0);
			K8 = expf(k);
			flops += 22;
		} else {
			K8 = 1.428e-9f;
		}
	}
	{
		const float c0 = -24.24914687f;
		const float c1 = 3.40082444f;
		const float c2 = -3.89800396f;
		const float c3 = 2.04558782f;
		const float c4 = -0.541618285f;
		const float c5 = 8.41077503E-2f;
		const float c6 = -7.87902615E-3f;
		const float c7 = 4.13839842E-4f;
		const float c8 = -9.36345888E-6f;
		float k = c8;
		k = fmaf(k, lnTev, c7);
		k = fmaf(k, lnTev, c6);
		k = fmaf(k, lnTev, c5);
		k = fmaf(k, lnTev, c4);
		k = fmaf(k, lnTev, c3);
		k = fmaf(k, lnTev, c2);
		k = fmaf(k, lnTev, c1);
		k = fmaf(k, lnTev, c0);
		K11 = expf(k);
	}
	{
		K12 = 5.6e-11f * sqrtT * expf(-102124.f * Tinv); // 14
	}
	{

		const float c0 = -18.01849334f;
		const float c1 = 2.3608522f;
		const float c2 = -0.28274430f;
		const float c3 = 1.62331664E-2f;
		const float c4 = -3.36501203E-2f;
		const float c5 = 1.17832978E-2f;
		const float c6 = -1.65619470E-3f;
		const float c7 = 1.06827520E-4f;
		const float c8 = -2.63128581E-6f;
		float k = c8;
		k = fmaf(k, lnTev, c7);
		k = fmaf(k, lnTev, c6);
		k = fmaf(k, lnTev, c5);
		k = fmaf(k, lnTev, c4);
		k = fmaf(k, lnTev, c3);
		k = fmaf(k, lnTev, c2);
		k = fmaf(k, lnTev, c1);
		k = fmaf(k, lnTev, c0);
		K14 = expf(k);
	}
	{
		K16 = 7e-7f * sqrtTinv; // 5
	}

	float cool = 0.f;
	float ne = N.Hp - N.Hn + N.Hep + 2 * N.Hepp;														// 4
	float T5 = T * 1e-5f;																					// 1
	float tmp1 = 1.0f / (1.0f + sqrtf(T5));															// 9
	float tmp2 = powf(T, -.1687f);																		// 8
	float tmp3 = powf(T3, -.2f);																			// 8
	cool = fmaf(7.5e-19f, expf(-118348.f * Tinv) * ne * N.H, cool);									// 13
	cool = fmaf(9.1e-27f, expf(-13179.f * Tinv) * tmp2 * sqr(ne) * N.He, cool);               // 15
	cool = fmaf(5.54e-17f, expf(-473638.f * Tinv) * powf(T, -.397f) * ne * N.Hep, cool);     // 22
	cool = fmaf(2.18e-11f, K1 * ne * N.H, cool);																// 4
	cool = fmaf(3.94e-11f, K3 * ne * N.He, cool);															// 4
	cool = fmaf(8.72e-11f, K5 * ne * N.Hep, cool);															// 4
	cool = fmaf(5.01e-27f, tmp1 * tmp2 * expf(-55338.f * Tinv) * sqr(ne) * N.Hep, cool);		// 16
	cool = fmaf(8.7e-27f, sqrtT * tmp3 / (1.0f + powf(T6, .7f)) * ne * N.Hp, cool);				// 18
	cool = fmaf(1.55e-26f, powf(T, 0.3647f) * ne * N.Hp, cool);											// 12
	cool = fmaf(1.24e-13f, powf(T, -1.5f) * (1.f + .3f * expf(-94000.f * Tinv)) * expf(-470000.f * Tinv) * ne * N.Hp, cool); // 34
	cool = fmaf(3.48e-26f, sqrtT * tmp3 / (1.0 + pow(T6, .7f)) * ne * N.Hepp, cool);          // 18
	float Qn = powf(N.H2, 0.77f) + 1.2 * powf(N.H, 0.77f);										// 18
	float LrL, LrH;
	float x = log10T - 4.f; 																				// 1
	float x2 = sqr(x);																						// 1
	double LvL;
	if (T > 4031.f) {																							// 1
		LrL = 1.38e-22f * expf(-9243.f * Tinv);
		flops += 10;
	} else {
		LrL = pow(10.0f, -22.90f - 0.553f * x - 1.48f * x2);
		flops += 12;
	}
	LrL *= Qn;																									// 1
	if (T > 1087.f) {																							// 1
		LrH = 3.9e-19f * expf(-6118.f * Tinv);
		flops += 10;
	} else {
		LrH = powf(10.f, -19.24f + 0.474f * x - 1.247 * x2);
		flops += 12;
	}
	if (T > 1635.f) {																							// 1
		LvL = 1e-12f * sqrtT * expf(-1000.f * Tinv);
		flops += 11;
	} else {
		LvL = 1.4e-13f * exp(T * (1.f / 125.f) - sqr(T * (1.f / 577.f)));
		flops += 13;
	}
	LvL *= N.H * 8.18e-13f;																					// 2
	float LvH = 1.1e-13f * exp(-6744.f * Tinv);														// 10
	cool = fmaf(N.H2, (LrH * LrL / (LrL + LrH) + LvH * LvL / (LvL + LvH)), cool);					// 15
	cool = fmaf(1.43e-27f, sqrtT * (fmaf(0.34f, expf(-sqr(5.5f - log10T) * (1.f / 3.f)), 1.1f)) * ne * (N.Hp + N.Hep + N.Hepp), cool); // 21
	cool = fmaf(5.65e-36f, sqr(sqr(1.f + z)) * (fmaf(-2.73f, (1 + z), T)) * ne, cool);			// 10
	constexpr float expmax = 88.0f;
	float tmp4 = sqr(z - 2.3f);																						// 1
	float tmp5 = powf(1.f + z, 0.43f);																				// 9
	float tmp6 = powf(1.f + z, 0.3f);																				// 9
	float tmp7 = tmp5 * expf(-fminf((1.f / 1.95f) * tmp4, expmax));											// 12
	float sigma20 = 5.6e-13f * tmp7;																					// 1
	float sigma22 = 9.6e-15f * tmp6 * expf(-fminf((1.f / 2.6f) * tmp4, expmax));							// 12
	float sigma21 = 3.2e-13f * tmp7;																					// 1
	float sigmaH = 3.9e-24f * tmp7;																					// 1
	float sigmaHe = 6.4e-24f * tmp5 * expf(-fminf((1.f / 2.1f) * tmp4, expmax));							// 12
	float sigmaHep = 8.6e-26f * tmp6 * expf(-fminf((1.f / 2.7f) * tmp4, expmax));						// 12
	float heat = sigmaH * N.H + sigmaHe * N.He + sigmaHep * N.Hep;											// 5

	float dedt = heat - cool;																							// 1

	flops += 625;

	float n0 = N.H + 2.0 * N.Hp + N.H2 + N.He + 2.0 * N.Hep + 3.0 * N.Hepp;								// 8
	float cv0 = (1.5f * (1.0f - N.H2) + 2.5f * N.H2);															// 4

	float Htot = N.H + N.Hp + N.Hn + 2.f * N.H2;																	// 4
	float Hetot = N.He + N.Hep + N.Hepp;																			// 2
	float ne_max = Htot + 4.f * Hetot;																				// 2
	float ne_min = ne_max * 1e-7f;																					// 1
	for (int i = 0; i < 28; i++) {
		float ne_mid = sqrtf(ne_max * ne_min);
		float fe_max, fe_mid;
		//	PRINT( "%e ", N.H + 2 * N.H2 + N.Hn + N.Hp);
		if (i == 0) {
			fe_max = test_electron_fraction(ne_max, N0, N, T, dt, z, K1, K2, K3, K4, K5, K6, K7, K8, K9, K11, K12, K14, K16, sigma20, sigma21, sigma22, flops);
		}
		fe_mid = test_electron_fraction(ne_mid, N0, N, T, dt, z, K1, K2, K3, K4, K5, K6, K7, K8, K9, K11, K12, K14, K16, sigma20, sigma21, sigma22, flops);
		//	PRINT( "%e\n", N.H + 2 * N.H2 + N.Hn + N.Hp);
		if (copysignf(1.f, fe_max) != copysignf(1.f, fe_mid)) {
			ne_min = ne_mid;
		} else {
			ne_max = ne_mid;
			fe_max = fe_mid;
		}
		flops += 8;
	}

	float n1 = N.H + 2.0 * N.Hp + N.H2 + N.He + 2.0 * N.Hep + 3.0 * N.Hepp;											// 8
	float cv1 = (1.5f * (1.0f - N.H2) + 2.5f * N.H2);																		// 4
	return constants::kb * (cv1 * n1 * T - cv0 * n0 * T0) - dt * dedt;												// 8
}

__global__ void chemistry_kernel(chemistry_params params, chem_attribs* chems, int nchems, float dt, int* next_index, double* total_flops) {
	int index;
	index = atomicAdd(next_index, 1);
	const double code_to_energy_density = params.code_to_g / (params.code_to_cm * sqr(params.code_to_s));		// 7
	const double code_to_density = pow(params.code_to_cm, -3) * params.code_to_g;										// 10
	dt *= params.a * params.code_to_s;																												// 1
	int flops = 18;
	double myflops = 0.0;
	while (index < nchems) {
		chem_attribs& attr = chems[index];
		species_t N;
		species_t N0;
		N.H = 1.f - params.Hefrac - attr.Hp - 2.f * attr.H2;															// 4
		N.Hp = attr.Hp;
		N.Hn = attr.Hn;
		N.H2 = attr.H2;
		N.He = params.Hefrac - attr.Hep - attr.Hepp;																		// 2
		N.Hep = attr.Hep;
		N.Hepp = attr.Hepp;
		const float rho = attr.rho * code_to_density;
		const float rhoavo = rho * constants::avo;																		// 1
		N.H *= rhoavo;																												// 1
		N.Hp *= rhoavo;																											// 1
		N.Hn *= rhoavo;																											// 1
		N.H2 *= 0.5f * rhoavo;																									// 2
		N.He *= 0.25f * rhoavo;																									// 2
		N.Hep *= 0.25f * rhoavo;																								// 2
		N.Hepp *= 0.25f * rhoavo;																								// 2
		float n0 = N.H + N.Hp + N.Hn + N.H2 + N.He + N.Hep + N.Hepp;									// 8
		float n = N.H + 2.f * N.Hp + N.H2 + N.He + 2.f * N.Hep + 3.f * N.Hepp;									// 8
		float cv = (1.5f + N.H2 / n0);																// 4
		float gamma = 1.f + 1.f / cv;																							// 5
		cv *= float(constants::kb);																							// 1
		float K = attr.K;												// 11
//		PRINT("%e\n", K);
		K *= pow(params.a, (1.f / 3.f) * gamma - 5.f);												// 11
		//PRINT("%e %e \n", K, (code_to_energy_density * pow(code_to_density, -gamma)));
		K *= (code_to_energy_density * pow(code_to_density, -gamma));												// 11
		//	PRINT("%e\n", K);
		float energy = float((double) K * pow((double) rho, (double) gamma) / ((double) gamma - 1.0));																			// 9
		float T0 = energy / (n * cv);																							// 5
//		PRINT("%e %e %e %e %e %e %e\n", T0, energy, K, rho, gamma, n, cv);
		float Tmax = 1e8f;
		float Tmin = 1e3f;
		N0 = N;
		float z = 1.f / params.a - 1.f;																						// 2
		float Tmid;
		for (int i = 0; i < 28; i++) {
			float f_mid, f_min;
			Tmid = sqrtf(Tmax * Tmin);																							// 5
			N = N.number_density_to_fractions();
//			PRINT("%e %e %e %e %e %e %e %e\n", Tmid, N.H, N.Hp, N.Hn, N.H2, N.He, N.Hep, N.Hepp);
			N = N0;
			if (i == 0) {
				f_min = test_temperature(N0, N, T0, Tmin, dt, z, flops);
			}
			f_mid = test_temperature(N0, N, T0, Tmid, dt, z, flops);
			if (copysignf(1.f, f_mid) != copysignf(1.f, f_min)) {
				Tmax = Tmid;
			} else {
				Tmin = Tmid;
				f_min = f_mid;
			}

			flops += 7;
		}
		float T = Tmid;

		n0 = N.H + N.H2 + N.He + N.Hep + N.Hepp + N.Hp + N.Hn;
		n = N.H + 2.f * N.Hp + N.H2 + N.He + 2.f * N.Hep + 3.f * N.Hepp;											// 8
		const float rhoavoinv = 1.f / rhoavo;																				// 4
		cv = (1.5f + N.H2 / n0);																		// 4
		N.H *= rhoavoinv;																											// 1
		N.Hp *= rhoavoinv;																										// 1
		N.Hn *= rhoavoinv;																										// 1
		N.H2 *= 2.f * rhoavoinv;																										// 1
		N.He *= 4.f * rhoavoinv;																										// 1
		N.Hep *= 4.f * rhoavoinv;																										// 1
		N.Hepp *= 4.f * rhoavoinv;																										// 1
		gamma = 1.f + 1.f / cv;																									// 5
		energy = cv * n * T;																										 	// 1
		K = energy * powf(rho, -gamma) * (gamma - 1.f);																	// 12
		K *= (pow(code_to_density, gamma) / code_to_energy_density);													// 12
		K *= pow(params.a, -(1.f / 3.f) * gamma + 5.f);																	// 11
		attr.H2 = N.H2;
		attr.Hep = N.Hep;
		attr.Hepp = N.Hepp;
		attr.Hn = N.Hn;
		attr.Hp = N.Hp;
		attr.K = K;
		flops += 136;
		myflops += flops;
		flops = 0;
		index = atomicAdd(next_index, 1);
	}
	atomicAdd(total_flops, myflops);
}

void cuda_chemistry_step(vector<chem_attribs>& chems, float scale, float dt) {
	timer tm;
	tm.start();
	static const auto opts = get_options();
	double* flops;
	int* index;
	chem_attribs* dev_chems;
	chemistry_params params;
	int nblocks;
	auto stream = cuda_get_stream();
	CUDA_CHECK(cudaMalloc(&dev_chems, sizeof(chem_attribs) * chems.size()));
	CUDA_CHECK(cudaMallocManaged(&flops, sizeof(double)));
	CUDA_CHECK(cudaMallocManaged(&index, sizeof(int)));
	CUDA_CHECK(cudaMemcpyAsync(dev_chems, chems.data(), sizeof(chem_attribs) * chems.size(), cudaMemcpyHostToDevice));
	*index = 0;
	params.Hefrac = opts.Y;
	params.a = scale;
	params.code_to_cm = opts.code_to_cm;
	params.code_to_g = opts.code_to_g;
	params.code_to_s = opts.code_to_s;
	params.dt = dt;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) chemistry_kernel, CHEM_BLOCK_SIZE, 0));
	nblocks *= cuda_smp_count();
	chemistry_kernel<<<nblocks,CHEM_BLOCK_SIZE>>>(params, dev_chems, chems.size(), dt, index, flops);
	cuda_stream_synchronize(stream);
	CUDA_CHECK(cudaMemcpy(chems.data(), dev_chems, sizeof(chem_attribs) * chems.size(), cudaMemcpyDeviceToHost));
	double myflops = *flops;
	CUDA_CHECK(cudaFree(flops));
	CUDA_CHECK(cudaFree(index));
	CUDA_CHECK(cudaFree(dev_chems));
	cuda_end_stream(stream);
	tm.stop();
	PRINT("Cuda Chemistry took %e at %e GFLOPs\n", tm.read(), myflops / 1024 / 1024 / 1024 / tm.read());
}

void test_cuda_chemistry_kernel() {
	static const auto opts = get_options();
	const double code_to_energy_density = opts.code_to_g / (opts.code_to_cm * sqr(opts.code_to_s));		// 7
	const double code_to_density = pow(opts.code_to_cm, -3) * opts.code_to_g;										// 10
	PRINT("%e %e %e %e %e\n", opts.code_to_g, opts.code_to_cm, opts.code_to_s, code_to_energy_density, code_to_density);
	const int N = 1;
	vector<chem_attribs> chem0;
	vector<chem_attribs> chems;
	float z = 0.5;
	for (int i = 0; i < N; i++) {
		float T = pow(10.0, 3.0 + rand1() * 5.0);
//		PRINT("%e\n", T);
		float rho = pow(10.0, -(24.0 + 3.0 * rand1())) * (z + 1);
		species_t N;
		N.H = rand1();
		N.H2 = rand1() * 0.1;
		N.Hp = rand1();
		N.Hn = 0.0;
		N.He = rand1();
		N.Hep = rand1();
		N.Hepp = rand1();
		float h0 = (N.H + 2.0 * N.H2 + N.Hp + N.Hn) / (1.0 - opts.Y);
		float he0 = (N.He + N.Hep + N.Hepp) / opts.Y;
		N.H /= h0;
		N.H2 /= h0;
		N.Hp /= h0;
		N.Hn /= h0;
		N.He /= he0;
		N.Hep /= he0;
		N.Hepp /= he0;
		chem_attribs chem;
		float n0 = N.H + N.Hp + 0.5f * N.H2 + N.Hn + 0.25f * N.He + 0.25f * N.Hep + 0.25f * N.Hepp;
		float n = (N.H + 2.f * N.Hp + 0.5f * N.H2 + 0.25f * N.He + 0.5f * N.Hep + .75f * N.Hepp) * rho * constants::avo;
		float cv = (1.5 + 0.5f * N.H2 / n0);
		float gamma = 1.0 + 1.0 / cv;
		cv *= constants::kb;
		float energy = cv * n * T;
		double K = (double) energy * (gamma - 1.0) * pow((double) rho, -gamma);
		K /= code_to_energy_density;
		K *= pow(code_to_density, gamma);
//		PRINT("!!!!!!!!!!!1 %e %e %e %e\n", energy, gamma, rho, K);
		rho /= code_to_density;
		chem.H2 = N.H2;
		chem.Hp = N.Hp;
		chem.Hep = N.Hep;
		chem.Hepp = N.Hepp;
		chem.rho = rho;
		chem.K = K;
		chem.Hn = N.Hn;
		chem.rho = rho;
		chems.push_back(chem);
		chem0.push_back(chem);
	}
	PRINT("Starting\n");
	cuda_chemistry_step(chems, 1.0, 1e15 / opts.code_to_s);

	for (int i = 0; i < N; i++) {
		PRINT( "%e %e %e %e\n", chems[i].Hp, chems[i].Hn, chems[i].H2, chems[i].Hep, chems[i].Hepp);
	}
}

CUDA_EXPORT
species_t species_t::number_density_to_fractions() const {
	species_t f;
	f = *this;
	f.He *= 4.0f;
	f.Hep *= 4.0f;
	f.Hepp *= 4.0f;
	f.H2 *= 2.0f;
	float rho = 0.f;
	for (int i = 0; i < NSPECIES; i++) {
		rho += f.n[i];
	}
	float rhoinv = 1.f / rho;
	for (int i = 0; i < NSPECIES; i++) {
		f.n[i] *= rhoinv;
	}
	return f;
}

