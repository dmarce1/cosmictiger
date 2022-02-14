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

#include <cosmictiger/containers.hpp>

#define NSPECIES 7
#define SPECIE_H 0
#define SPECIE_HP 1
#define SPECIE_HN 2
#define SPECIE_H2 3
#define SPECIE_H2P 4
#define SPECIE_HEP 5
#define SPECIE_HEPP 6

#define TMAX 1e9f

class frac_real {
	union {
		uint16_t bits;
		struct {
			uint16_t ex :6;
			uint16_t mn :10;
		};
	};
	static constexpr float factor = 1023.49999;
public:
	CUDA_EXPORT
	operator float() const {
		float x = 1.f + (float) mn / (float) factor;
		float b2y = powf(2.f, -ex);
		//	PRINT("%i %i to %e\n", mn, ex, x * b2y);
		return x * b2y;
	}
	CUDA_EXPORT
	frac_real& operator=(float z) {
		if (z != 0.f) {
			ex = floor(-log2(z));
			mn = factor * (z * powf(2.f, ex) - 1.f);
		} else {
			ex = 63;
			mn = 0;
		}
		//	PRINT("%e to %i and %i\n", z, mn, ex);
		return *this;
	}
};

struct species_t {
	union {
		float n[NSPECIES];
		struct {
			float H;
			float Hp;
			float Hn;
			float He;
			float Hep;
			float Hepp;
			float H2;
		};
	};
	species_t fractions_to_number_density(float rho) const;CUDA_EXPORT
	species_t number_density_to_fractions() const;
};

struct chem_attribs {
	float Hp;
	float H2;
	float Hn;
	float He;
	float Hep;
	float Hepp;
	float rho;
	float K;
	float dt;
	float tcool;
};

void chemistry_test();
void cuda_chemistry_step(vector<chem_attribs>& chems, float scale);
void test_cuda_chemistry_kernel();
void chemistry_do_step(float, int, float, float, int);
