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

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>

// Expansion-order conventions shared by the STF kernels, the Ewald derivative
// generator, and the CUDA kick workspace.
//
// The build selects the FMM order (COSMICTIGER_WITH_ORDER, the legacy ORDER
// macro). It equals the multipole degree p plus two: source multipoles carry
// degree p, local expansions carry degree p + 1, and the Ewald derivative
// tensors carry ranks zero through p + 1, so their "order" template argument
// is p + 2. Order 3 (dipoles with quadratic expansions) is the smallest order
// whose local expansion can hold the periodic uniform-background quadratic;
// order 10 is degree-8 multipoles with degree-9 expansions.
// Coefficient counts use compact rank-major STF storage: a degree-d harmonic
// tensor stores (d + 1)^2 coefficients, and local expansions append one
// constant Laplacian (the periodic background trace).

#ifndef ORDER
#error \
    "ORDER (the FMM order, multipole degree + 2) must be defined; configure COSMICTIGER_WITH_ORDER"
#endif

inline constexpr int kFmmLegacyOrder = ORDER;
inline constexpr int kFmmMultipoleDegree = ORDER - 2;
inline constexpr int kFmmLocalDegree = ORDER - 1;
inline constexpr int kFmmEwaldTensorOrder = ORDER;

static_assert(kFmmMultipoleDegree >= 1 && kFmmMultipoleDegree <= 8,
    "Supported FMM orders are 3 through 10 (multipole degrees 1 through 8)");
static_assert(kFmmLocalDegree == kFmmMultipoleDegree + 1, "Local degree is always p + 1");
static_assert(kFmmEwaldTensorOrder == LORDER, "LORDER must equal the legacy expansion order");

CUDA_EXPORT constexpr int fmmHarmonicCoefficientCount(int degree) {
    return (degree + 1) * (degree + 1);
}

CUDA_EXPORT constexpr int fmmMultipoleCoefficientCount(int multipoleDegree) {
    return fmmHarmonicCoefficientCount(multipoleDegree);
}

// The index of the constant Laplacian stored after the harmonic coefficients.
CUDA_EXPORT constexpr int fmmLocalTraceIndex(int localDegree) {
    return fmmHarmonicCoefficientCount(localDegree);
}

CUDA_EXPORT constexpr int fmmLocalCoefficientCount(int localDegree) {
    return fmmHarmonicCoefficientCount(localDegree) + 1;
}

// A positioned multipole stores the coefficients plus three Fixed32 positions.
CUDA_EXPORT constexpr int fmmPositionedMultipoleBytes(int multipoleDegree) {
    return 4 * (fmmMultipoleCoefficientCount(multipoleDegree) + NDIM);
}

CUDA_EXPORT constexpr int fmmLocalExpansionBytes(int localDegree) {
    return 4 * fmmLocalCoefficientCount(localDegree);
}

inline constexpr int kFmmMultipoleCoefficients = fmmMultipoleCoefficientCount(kFmmMultipoleDegree);
inline constexpr int kFmmLocalHarmonicCoefficients = fmmHarmonicCoefficientCount(kFmmLocalDegree);
inline constexpr int kFmmLocalTraceIndex = fmmLocalTraceIndex(kFmmLocalDegree);
inline constexpr int kFmmLocalCoefficients = fmmLocalCoefficientCount(kFmmLocalDegree);
inline constexpr int kFmmPositionedMultipoleBytes =
    fmmPositionedMultipoleBytes(kFmmMultipoleDegree);
inline constexpr int kFmmLocalExpansionBytes = fmmLocalExpansionBytes(kFmmLocalDegree);

static_assert(fmmPositionedMultipoleBytes(1) == 28 && fmmPositionedMultipoleBytes(5) == 156 &&
        fmmPositionedMultipoleBytes(8) == 336,
    "Positioned multipole byte sizes");
static_assert(fmmLocalExpansionBytes(2) == 40 && fmmLocalExpansionBytes(6) == 200 &&
        fmmLocalExpansionBytes(9) == 404,
    "Local expansion byte sizes");
