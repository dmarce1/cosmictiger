/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <type_traits>

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/math/vector3.hpp>

// Preserve the exact 3D ABI; lower-dimensional records contain only active axes.
template <class Real>
using ParticleVector = std::conditional_t<kSpatialDimensions == 3, Vector3<Real>,
    math::Vector<Real, kSpatialDimensions>>;

// Particle species stored in particlesType(). Kept apart from particles.hpp so
// lightweight headers shared with CUDA kernels can name a species.
inline constexpr int kDarkMatterType = 0;
inline constexpr int kSphType = 1;
inline constexpr int kStarType = 2;
