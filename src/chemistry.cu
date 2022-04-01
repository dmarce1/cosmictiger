/*
 CosmicTiger - A Cosmological N-Body code
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
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/options.hpp>

struct chemistry_params {
	double code_to_cm;
	double code_to_g;
	double code_to_s;
	bool stars;
	float a;
};

#define CHEM_BLOCK_SIZE 64

#define NCOOL 14

__managed__ array<double, NCOOL> cooling_totals;

__device__ float test_electron_fraction(float ne, species_t N0, species_t& N, float T, float dt, float z, float K1, float K2, float K3, float K4, float K5,
		float K6, float K7, float K8, float K9, float K11, float K12, float K14, float K16, float sigma20, float sigma21, float sigma22, int& flops) {
//	const int tid = threadIdx.x;
//	auto tm = clock64();
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
	Hn = (H * K7 * ne) / fmaf(Hp, K16, fmaf(H, K8, K14 * ne));																															// 11
	Hp = Hall - (H + fmaf(2.f, H2, Hn));
	Hp = fmaxf(Hp, 0.0f);																																												// 1
// 4
	const float twoH2 = 2.f * H2;																																										// 1
	const float tmp1 = (twoH2 - Hall + Hn);																																						// 2
	const float Hpnum = (Hp0 - dt * tmp1 * fmaf(ne, K1, sigma20));																															// 5
	const float Hpden = fmaf(dt, fmaf(ne, K1 + K2, sigma20), 1.f);																															// 5
	Hp = Hpnum / Hpden;																																													// 4
	H = Hall - (Hp + fmaf(2.f, H2, Hn));																																							// 4
	Hn = (H * K7 * ne) / fmaf(Hp, K16, fmaf(H, K8, K14 * ne));																															  // 11
	H = Hall - (Hp + fmaf(2.f, H2, Hn));																																							// 4
	const float tmp2 = K7 * K8;																																										// 1
	const float H2num = fmaf(H20, fmaf(Hp, K16, fmaf(H, K8, K14 * ne)), dt * H * tmp2 * ne * fmaf(2.f, H20, H));														// 13
	const float K14ne = K14 * ne;																																										// 1
	const float tmp3 = Hp * K16 + fmaf(H, K8, K14ne);																																			// 4
	const float H2den = fmaf(dt, fmaf(tmp3, (fmaf(Hp, K11, ne * K12)), ne * 2.f * H * tmp2), tmp3);																			// 10
	H2 = H2num / H2den;																																													// 4
	H = Hall - (Hp + fmaf(2.f, H2, Hn));																																							// 4
	Hn = (H * K7 * ne) / fmaf(Hp, K16, fmaf(H, K8, K14 * ne));																															 // 11
	H = Hall - (Hp + fmaf(2.f, H2, Hn));
	const float K6ne = K6 * ne;																																										// 1
	const float tmp4 = fmaf(dt, K6ne, 1.f);																																						// 2
	const float tmp5 = fmaf(K3, ne, sigma21 - K6ne);																																			// 3
	const float Hep_num = fmaf(-Hepp0, dt * tmp5, tmp4 * (fmaf(dt, fmaf(Heall, K3 * ne, Heall * sigma21), Hep0)));													    // 11
	const float K345 = K3 + K4 + K5;																																									// 2
	const float Hep_den = fmaf(tmp4, fmaf(dt, fmaf(K345, ne, sigma21 + sigma22), 1.f), (sqr(dt) * tmp5 * fmaf(K5, ne, sigma22)));									// 12
	Hep = Hep_num / Hep_den;																																											// 4
	const float Hepp_num = fmaf(dt,
			fmaf(dt, Heall * fmaf(K5, ne * fmaf(K3, ne, sigma21), fmaf(K3, ne, sigma21) * sigma22),
					fmaf(ne, fmaf(Hepp0, K345, Hep0 * K5), fmaf(Hepp0, sigma21, (Hep0 + Hepp0) * sigma22))), Hepp0);														// 22
	const float Hepp_den = fmaf(dt,
			(dt * fmaf(K3, sqr(ne) * (K5 + K6), fmaf(ne, fmaf(K6, fmaf(K4, ne, sigma21), K5 * sigma21), fmaf(K3, ne, sigma21) * sigma22))
					+ fmaf((K345 + K6), ne, sigma21) + sigma22), 1.f);																														// 21
	Hepp = (Hepp_num / Hepp_den);																																										// 4
	He = Heall - Hep - Hepp;																																											// 2
	H = fmaxf(H, 0.0f);																																													// 1
	Hp = fmaxf(Hp, 0.0f);																																												// 1
	Hn = fmaxf(Hn, 0.0f);																																												// 1
	H2 = fmaxf(H2, 0.0f);																																												// 1
	He = fmaxf(He, 0.0f);																																												// 1
	Hep = fmaxf(Hep, 0.0f);																																												// 1
	Hepp = fmaxf(Hepp, 0.0f);																																											// 1
	flops += 200;
//	tm = clock64() - tm;
//	if (tid == 0) {
//		atomicAdd(&timer3, (double) tm);
//	}
	return Hp - Hn + Hep + 2 * Hepp - ne;
}

__device__ float test_temperature(species_t N0, species_t& N, float T0, float T, float dt, float z, int& flops, float* dedt_ptr, bool add_totals) {
//	const int tid = threadIdx.x;
//	auto tm = clock64();
	const auto oT = T;
	T = fmaxf(TMIN, T);							// 1
	T = fminf(TMAX, T);							// 1
	const float Tev = T / constants::evtoK;					// 4
	const float Tevinv = 1.f / Tev;					// 4
	const float lnTev = log(Tev);						// 8
	const float log10T = log10(T);						// 8
	const float T3 = T * 1e-3f;					// 1
	const float sqrtTinv = rsqrtf(T);
	const float sqrtT = sqrtf(T);
	const float Tinv = 1.f / T;
	const float T6 = T * 1e-6f;					// 1
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
		K4 += 1.544E-9f * powf(Tev, -1.5f) * (0.3f * expf(-48.596f * Tevinv) + expf(-40.496f * Tevinv));    // 18
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
		K6 = 3.36e-10f * sqrtTinv * powf(T3, -0.2f) / (1.0f + powf(T6, 0.7));
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
			const float c2 = powf(10.f, 4.0415e-5f * powf(log10T, 6));   // 17
			const float c3 = powf(10.f, -5.447e-3f * powf(log10T, 4));   // 17
			K7 = c0 * c1 * c2 * c3;                                   // 3
			flops += 47;
		}
	}
	{
		if (Tev > 0.1) {
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
	T = oT;
	float cool = 0.f;
	const float ne = N.Hp - N.Hn + N.Hep + 2 * N.Hepp;														// 4
	const float T5 = T * 1e-5f;																					// 1
	const float tmp1 = 1.0f / (1.0f + sqrtf(T5));															// 9
	const float tmp2 = powf(T, -.1687f);																		// 8
	const float tmp3 = powf(T3, -.2f);																			// 8
	const float tmp4 = tmp3 / (1.0f + powf(T6, .7f));
	array<float, NCOOL> C;
	C[0] = 7.5e-19f * tmp1 * expf(-118348.f * Tinv) * ne * N.H;									// 13
	C[1] = 9.1e-27f * tmp1 * expf(-13179.f * Tinv) * tmp2 * sqr(ne) * N.He;               // 15
	C[2] = 5.54e-17f * tmp1 * expf(-473638.f * Tinv) * powf(T, -.397f) * ne * N.Hep;     // 22
	C[3] = 1.27e-21f * sqrtT * tmp1 * expf(-157809.1f * Tinv) * ne * N.H;																// 4
	C[4] = 9.38e-22f * sqrtT * tmp1 * expf(-285335.4f * Tinv) * ne * N.He;																// 4
	C[5] = 4.95e-22f * sqrtT * tmp1 * expf(-631515.f * Tinv) * ne * N.Hep;																// 4
	C[6] = 5.01e-27f * tmp1 * tmp2 * expf(-55338.f * Tinv) * sqr(ne) * N.Hep;		// 16
	C[7] = 8.7e-27f * sqrtT * tmp4 * ne * N.Hp;				// 18
	C[8] = 1.55e-26f * powf(T, 0.3647f) * ne * N.Hep;											// 12
	C[9] = 1.24e-13f * powf(T, -1.5f) * (1.f + .3f * expf(-94000.f * Tinv)) * expf(-470000.f * Tinv) * ne * N.Hep; // 34
	C[10] = 3.48e-26f * sqrtT * tmp4 * ne * N.Hepp;          // 18
	const float Qn = powf(N.H2, 0.77f) + 1.2 * powf(N.H, 0.77f);										// 18
	float LrL, LrH;
	const float x = log10T - 4.f; 																				// 1
	const float x2 = sqr(x);																						// 1
	float LvL;
	if (T > 4031.f) {																							// 1
		LrL = 1.38e-22f * expf(-9243.f * Tinv);
		flops += 10;
	} else {
		LrL = powf(10.0f, -22.90f - 0.553f * x - 1.148f * x2);
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
		LvL = 1.4e-13f * expf(T * (1.f / 125.f) - sqr(T * (1.f / 577.f)));
		flops += 13;
	}
	LvL *= N.H * 8.18e-13f;																					// 2
	const float LvH = 1.1e-13f * expf(-6744.f * Tinv);														// 10
	C[11] = N.H2 * (LrH * LrL / (LrL + LrH) + LvH * LvL / (LvL + LvH));					// 15
	C[12] = 1.43e-27f * sqrtT * (fmaf(0.34f, expf(-sqr(5.5f - log10T) * (1.f / 3.f)), 1.1f)) * ne * (N.Hp + N.Hep + N.Hepp); // 21
	C[13] = 5.65e-36f * sqr(sqr(1.f + z)) * (fmaf(-2.73f, (1 + z), T)) * ne;			// 10
	constexpr float expmax = 88.0f;
	const float tmp5 = powf(1.f + z, 0.43f);																				// 9
	const float tmp6 = powf(1.f + z, 0.3f);																				// 9
	const float tmp8 = sqr(z - 2.3f);																						// 1
	const float tmp7 = tmp5 * expf(-fminf((1.f / 1.95f) * tmp8, expmax));											// 12
	const float sigma20 = 5.6e-13f * tmp7;																					// 1
	const float sigma22 = 9.6e-15f * tmp6 * expf(-fminf((1.f / 2.6f) * tmp8, expmax));							// 12
	const float sigma21 = 3.2e-13f * tmp7;																					// 1
	const float sigmaH = 3.9e-24f * tmp7;																					// 1
	const float sigmaHe = 6.4e-24f * tmp5 * expf(-fminf((1.f / 2.1f) * tmp8, expmax));							// 12
	const float sigmaHep = 8.6e-26f * tmp6 * expf(-fminf((1.f / 2.7f) * tmp8, expmax));						// 12
	const float heat = sigmaH * N.H + sigmaHe * N.He + sigmaHep * N.Hep;											// 5
	for (int i = 0; i < NCOOL; i++) {
		cool += C[i];
		if (add_totals) {
			atomicAdd(&cooling_totals[i], C[i] * dt);
		}
	}
	const float dedt = heat - cool;																							// 1

	flops += 625;

	const float n0 = N.H + 2.0 * N.Hp + N.H2 + N.He + 2.0 * N.Hep + 3.0 * N.Hepp;								// 8
	const float cv0 = (1.5f * (1.0f - N.H2) + 2.5f * N.H2);															// 4

	const float Htot = N.H + N.Hp + N.Hn + 2.f * N.H2;																	// 4
	const float Hetot = N.He + N.Hep + N.Hepp;																			// 2
	float ne_max = Htot + 4.f * Hetot;																				// 2
	float ne_min = ne_max * 1e-7f;																					// 1
	for (int i = 0; i < 28; i++) {
		float ne_mid = sqrtf(ne_max * ne_min);
		float fe_max, fe_mid;
		if (i == 0) {
			fe_max = test_electron_fraction(ne_max, N0, N, T, dt, z, K1, K2, K3, K4, K5, K6, K7, K8, K9, K11, K12, K14, K16, sigma20, sigma21, sigma22, flops);
		}
		fe_mid = test_electron_fraction(ne_mid, N0, N, T, dt, z, K1, K2, K3, K4, K5, K6, K7, K8, K9, K11, K12, K14, K16, sigma20, sigma21, sigma22, flops);
		if (copysignf(1.f, fe_max) != copysignf(1.f, fe_mid)) {
			ne_min = ne_mid;
		} else {
			ne_max = ne_mid;
			fe_max = fe_mid;
		}
		flops += 8;
	}

	const float n1 = N.H + 2.0 * N.Hp + N.H2 + N.He + 2.0 * N.Hep + 3.0 * N.Hepp;											// 8
	const float cv1 = (1.5f + N.H2);																		// 4
	*dedt_ptr = dedt;
	return constants::kb * (cv1 * n1 * T - cv0 * n0 * T0) - dt * dedt;												// 8
}

__global__ void chemistry_kernel(chemistry_params params, chem_attribs* chems, int nchems, int* next_index, double* total_flops) {
//	const int tid = threadIdx.x;
//	auto tm = clock64();
	int index;
	index = atomicAdd(next_index, 1);
	const double code_to_energy_density = params.code_to_g / (params.code_to_cm * sqr(params.code_to_s));		// 7
	const double code_to_energy = sqr(params.code_to_cm) / sqr(params.code_to_s);		// 7
	const double code_to_density = pow(params.code_to_cm, -3) * params.code_to_g;										// 10
	int flops = 18;
	double myflops = 0.0;
	while (index < nchems) {
		chem_attribs& attr = chems[index];
		species_t N;
		species_t N0;
		N.H = 1.0 - ((double) attr.He + (double) attr.Hep + (double) attr.Hepp + (double) attr.Hp + (double) attr.Hn + (double) attr.H2);							// 4
		N.Hp = attr.Hp;
		N.Hn = attr.Hn;
		N.H2 = attr.H2;
		N.He = attr.He;																	// 2
		N.Hep = attr.Hep;
		N.Hepp = attr.Hepp;
		double dt = (double) attr.dt * (double) params.a;
		dt *= (double) params.code_to_s;																												// 1
		double rho = (double) attr.rho * (double) code_to_density * pow((double) params.a, -3.0);
		rho *= 1.0f - attr.cold_mass;
		const double rhoavo = rho * constants::avo;																		// 1
		N.H *= rhoavo;																												// 1
		N.Hp *= rhoavo;																											// 1
		N.Hn *= rhoavo;																											// 1
		N.H2 *= 0.5 * rhoavo;																									// 2
		N.He *= 0.25 * rhoavo;																									// 2
		N.Hep *= 0.25 * rhoavo;																								// 2
		N.Hepp *= 0.25 * rhoavo;																								// 2
		double n0 = (double) N.H + (double) N.Hp + (double) N.Hn + (double) N.H2 + (double) N.He + (double) N.Hep + (double) N.Hepp;									// 8
		double n = (double) N.H + 2.0 * (double) N.Hp + (double) N.H2 + (double) N.He + 2.0 * (double) N.Hep + 3.0 * (double) N.Hepp;									// 8
		double cv = (1.50 + (double) N.H2 / n0);																// 4
		double gamma = 1.0 + 1.0 / cv;																							// 5
		cv *= constants::kb;																							// 1
		double eint = attr.eint;
		eint *= code_to_energy;
		eint /= sqr(params.a);
		double T0 = (eint * rho) / (n * cv);
		double Tmax = TMAX;
		float Tmin = TMIN;
		N0 = N;
		double z = 1.0 / (double) params.a - 1.0;																						// 2
		double Tmid;
		float dedt;
		float dedt0;
		float dT = T0 * 1e-3f;
		test_temperature(N0, N, T0, T0, dt, z, flops, &dedt0, false);
		test_temperature(N0, N, T0, T0 + dT, dt, z, flops, &dedt, false);
		if (params.stars && (dedt0 < 0.0f && (-dedt / (T0 + dT) > -dedt0 / T0))) {
			float hot_mass = 1.f - attr.cold_mass;
			const float tcool = -eint * rho / dedt0 / params.a;
			float factor = fmaxf(expf(-dt / tcool), 0.5f);
			hot_mass *= factor;
			attr.cold_mass = 1.f - hot_mass;
			if (hot_mass < 0.0 || hot_mass > 1.0f || attr.cold_mass < 0.f || attr.cold_mass > 1.f) {
				PRINT("cold mass error --------- %e %e %e\n", hot_mass, attr.cold_mass, factor);
				__trap();
			}
			if (hot_mass < 1.f) {
				//		PRINT("%e %e %e %e\n", hot_mass, attr.cold_mass, dt/tcool, T0);
			}
			attr.eint *= factor;
			const double rhoavoinv = 1.0 / rhoavo;																				// 4
			N.H *= (double) rhoavoinv;																											// 1
			N.Hp *= (double) rhoavoinv;																										// 1
			N.Hn *= (double) rhoavoinv;																										// 1
			N.H2 *= 2.0 * (double) rhoavoinv;																										// 1
			N.He *= 4.0 * (double) rhoavoinv;																										// 1
			N.Hep *= 4.0 * (double) rhoavoinv;																										// 1
			N.Hepp *= 4.0 * (double) rhoavoinv;																										// 1
			attr.H2 = N.H2;
			attr.Hep = N.Hep;
			attr.Hepp = N.Hepp;
			attr.Hn = N.Hn;
			attr.Hp = N.Hp;
			attr.He = N.He;
		} else {
			for (int i = 0; i < 28; i++) {
				float f_mid, f_max;
				Tmid = sqrtf(Tmax * Tmin);																							// 5
				N = N.number_density_to_fractions();
				N = N0;
				if (i == 0) {
					f_max = test_temperature(N0, N, T0, Tmax, dt, z, flops, &dedt, false);
				}
				f_mid = test_temperature(N0, N, T0, Tmid, dt, z, flops, &dedt, false);
				if (copysignf(1.f, f_mid) != copysignf(1.f, f_max)) {
					Tmin = Tmid;
				} else {
					Tmax = Tmid;
					f_max = f_mid;
				}

				flops += 7;
			}
			Tmid = sqrtf(Tmax * Tmin);
			test_temperature(N0, N, T0, T0, dt, z, flops, &dedt0, false);
			test_temperature(N0, N, T0, Tmid, dt, z, flops, &dedt, true);
			dedt = 0.5f * dedt + 0.5f * dedt0;
			float T = Tmid;
			n0 = (double) N.H + (double) N.H2 + (double) N.He + (double) N.Hep + (double) N.Hepp + (double) N.Hp + (double) N.Hn;
			n = (double) N.H + 2.0 * (double) N.Hp + (double) N.H2 + (double) N.He + 2.0 * (double) N.Hep + 3.0 * (double) N.Hepp;										// 8
			const double rhoavoinv = 1.0 / rhoavo;																				// 4
			cv = (1.5 + N.H2 / n0);																		// 4
			N.H *= (double) rhoavoinv;																											// 1
			N.Hp *= (double) rhoavoinv;																										// 1
			N.Hn *= (double) rhoavoinv;																										// 1
			N.H2 *= 2.0 * (double) rhoavoinv;																										// 1
			N.He *= 4.0 * (double) rhoavoinv;																										// 1
			N.Hep *= 4.0 * (double) rhoavoinv;																										// 1
			N.Hepp *= 4.0 * (double) rhoavoinv;																										// 1
			gamma = 1.0 + 1.0 / cv;																									// 5
			cv *= constants::kb;
			eint = cv * n * T / rho;																										 	// 1
			eint *= sqr(params.a);
			eint /= code_to_energy;
			attr.H2 = N.H2;
			attr.Hep = N.Hep;
			attr.Hepp = N.Hepp;
			attr.Hn = N.Hn;
			attr.Hp = N.Hp;
			attr.He = N.He;
			attr.eint = eint;
		}
		//PRINT( "%e\n", eint);
		flops += 136;
		myflops += flops;
		flops = 0;
		index = atomicAdd(next_index, 1);
	}
	atomicAdd(total_flops, myflops);
}

void cuda_chemistry_step(vector<chem_attribs>& chems, float scale) {
	timer tm;
	static const auto opts = get_options();
	double* flops;
	int* index;
	chem_attribs* dev_chems;
	chemistry_params params;
	int nblocks;
	auto stream = cuda_get_stream();
	for (int i = 0; i < NCOOL; i++) {
		cooling_totals[i] = 0.f;
	}
	CUDA_CHECK(cudaMalloc(&dev_chems, sizeof(chem_attribs) * chems.size()));
	CUDA_CHECK(cudaMallocManaged(&flops, sizeof(double)));
	CUDA_CHECK(cudaMallocManaged(&index, sizeof(int)));
	CUDA_CHECK(cudaMemcpyAsync(dev_chems, chems.data(), sizeof(chem_attribs) * chems.size(), cudaMemcpyHostToDevice, stream));
	*index = 0;
	params.a = scale;
	params.code_to_cm = opts.code_to_cm;
	params.stars = opts.stars;
	params.code_to_g = opts.code_to_g;
	params.code_to_s = opts.code_to_s;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) chemistry_kernel, CHEM_BLOCK_SIZE, 0));
	nblocks *= cuda_smp_count();
	nblocks = std::max(1, nblocks / hpx_hardware_concurrency());
	tm.start();
	chemistry_kernel<<<nblocks,CHEM_BLOCK_SIZE, 0, stream>>>(params, dev_chems, chems.size(), index, flops);
	cuda_stream_synchronize(stream);
	tm.stop();
	CUDA_CHECK(cudaMemcpyAsync(chems.data(), dev_chems, sizeof(chem_attribs) * chems.size(), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaFree(flops));
	CUDA_CHECK(cudaFree(index));
	cuda_stream_synchronize(stream);
	CUDA_CHECK(cudaFree(dev_chems));
	cuda_end_stream(stream);
	double ctot = 0.0;
	for (int i = 0; i < NCOOL; i++) {
		ctot += cooling_totals[i];
	}
	if (ctot > 0.0) {
		for (int i = 0; i < NCOOL; i++) {
			//		PRINT("C%i = %e\n", i, cooling_totals[i] / ctot);
		}
	}
}

void test_cuda_chemistry_kernel() {
	/*	static const auto opts = get_options();
	 const double code_to_energy_density = opts.code_to_g / (opts.code_to_cm * sqr(opts.code_to_s));		// 7
	 const double code_to_density = pow(opts.code_to_cm, -3) * opts.code_to_g;
	 const double Y = get_options().Y0;		// 10
	 PRINT("%e %e %e %e %e\n", opts.code_to_g, opts.code_to_cm, opts.code_to_s, code_to_energy_density, code_to_density);
	 const int N = 10000000;
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
	 float h0 = (N.H + 2.0 * N.H2 + N.Hp + N.Hn) / (1.0 - opts.Y0);
	 float he0 = (N.He + N.Hep + N.Hepp) / opts.Y0;
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
	 chem.dt = 1e15 / opts.code_to_s;
	 chem.He = N.He;
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
	 cuda_chemistry_step(chems, 1.0);

	 for (int i = 0; i < N; i++) {
	 //PRINT("%e %e | %e %e %e | %e %e %e | \n", chem0[i].K, chems[i].K, 1 - Y - chem0[i].Hp - chem0[i].H2 * 2 - chem0[i].Hn, chem0[i].Hp, chem0[i].H2,
	 //1 - Y - chems[i].Hp - chems[i].H2 * 2 - chems[i].Hn, chems[i].Hp, chems[i].H2);
	 }*/
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

