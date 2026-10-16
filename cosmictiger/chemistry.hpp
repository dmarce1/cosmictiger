/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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

#define TMAX 1e10f
#define TMIN 100.f

struct SpeciesT {
    union {
        float n[NSPECIES];
        struct {
            float h;
            float hp;
            float hn;
            float he;
            float hep;
            float hepp;
            float h2;
        };
    };
    SpeciesT fractionsToNumberDensity(float rho) const;
    CUDA_EXPORT
    SpeciesT numberDensityToFractions() const;
};

struct ChemAttribs {
    float h;
    float hp;
    float h2;
    float hn;
    float he;
    float hep;
    float hepp;
    float rho;
    float eint;
    float dt;
    float tcool;
    float coldMass;
};

void chemistryTest();
double cudaChemistryStep(Vector<ChemAttribs>& chems, float scale);
void testCudaChemistryKernel();
Pair<double> chemistryDoStep(float, int, float, float, int);
