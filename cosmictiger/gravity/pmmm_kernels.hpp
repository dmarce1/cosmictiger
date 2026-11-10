/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <array>
#include <vector>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/math/vector3.hpp>

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

// Overwrite the harmonic Green coefficients. boxOffset is target minus source
// center in the unit periodic box, on the minimum-image branch [-0.5,0.5]^3.
// Degree-d coefficients are multiplied by cellWidth^d. Near entries exclude
// only the primary free-space interaction; the origin is valid when near=true.
// The uniform-background trace is supplied separately by the caller.
FlopCount green(
    int order, double* coefficients, Vector3<double> const& boxOffset, double cellWidth, bool near);

}    // namespace pmmm
