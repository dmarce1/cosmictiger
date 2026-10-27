/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <array>
#include <vector>

#include <cosmictiger/core/flops.hpp>

namespace pmmm {

// Real spherical coefficient indices, not Fourier real/imaginary components.
// The same real weights multiply complex spatial Fourier amplitudes without
// conjugating them. Harmonic conjugation is already included in the weights.
struct ContractionTerm {
    int local;
    int source;
    int green;
    double weight;
};

// Application order q uses source degree q-2, local degree q-1 and keeps only
// terms whose combined source/local degree is at most q-1.
std::vector<ContractionTerm> contractionTerms(int order);

// Add mass*U(-offset) to moments. Offset is relative to the source cell center,
// already divided by the cell width. Radial second moments are separate.
// Arithmetic routines return source-operation counts for caller-side batching;
// FMA counts twice and other arithmetic or elementary function results once.
FlopCount addMoments(int order, double* moments, double mass, std::array<double, 3> const& offset);

// Overwrite the harmonic Green coefficients. boxOffset is target minus source
// center in the unit periodic box, on the minimum-image branch [-0.5,0.5]^3.
// Degree-d coefficients are multiplied by cellWidth^d. Near entries exclude
// only the primary free-space interaction; the origin is valid when near=true.
// The uniform-background trace is supplied separately by the caller.
FlopCount green(int order, double* coefficients, std::array<double, 3> const& boxOffset,
    double cellWidth, bool near);

// Overwrite [potential, accelerationX, accelerationY, accelerationZ]. The
// target offset is in cell units; localTrace is the positive trace coefficient
// of the polynomial -localTrace*|cellOffset|^2/2. Accelerations are converted
// from derivatives in cell units to derivatives in unit-box coordinates.
FlopCount evaluate(int order, std::array<double, 4>& field, double const* coefficients,
    double localTrace, std::array<double, 3> const& cellOffset, double cellWidth);

}    // namespace pmmm
