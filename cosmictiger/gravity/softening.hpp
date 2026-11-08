/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/kernel.hpp>

// Effective radius of the selected gravity softening kernel (derived from the
// kernel table's support-to-effective-radius ratio, no source equation number):
//
//   h̃_G = h_G / (h/h̃)
//
// The division is done in double against the double gravity table; the float
// overload narrows the quotient once.
//
// Symbol directory:
//   h_G -> compact gravity support (gravity.smoothingLength)
//   h/h̃ -> kernel support per effective radius of the selected gravity kernel
//   h̃_G -> gravity effective radius used by the timestep criteria (return value)
// Code names: h/h̃ -> selectedGravityKernel().smoothingLengthPerEffectiveRadius
CUDA_EXPORT inline double gravitySofteningEffectiveRadius(double h_G) {
    return h_G / selectedGravityKernel().smoothingLengthPerEffectiveRadius;
}

CUDA_EXPORT inline float gravitySofteningEffectiveRadius(float h_G) {
    return float(gravitySofteningEffectiveRadius(double(h_G)));
}
