/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

// Symbol directory (periodic spherical Ewald operators):
// M: source multipole; L: target local expansion; G: periodic Green harmonics.
// x, y, z: minimum-image target-minus-source displacement in unit-box coordinates.
// sourceTrace, localTrace, greenTrace: independent radial/background terms.
// The primary free-space singularity is handled separately from the periodic correction.

#pragma once

#include <cosmictiger/gravity/kernels/spherical/ewald_green.hpp>
#include <cosmictiger/gravity/kernels/spherical/ewald_l2l.hpp>
#include <cosmictiger/gravity/kernels/spherical/ewald_l2p.hpp>
#include <cosmictiger/gravity/kernels/spherical/ewald_m2l.hpp>
#include <cosmictiger/gravity/kernels/spherical_kernels.hpp>
