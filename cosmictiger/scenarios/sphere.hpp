/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef SPHERE_HPP_
#define SPHERE_HPP_

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>

// Spreads each shell's points evenly over the unit sphere by damped, softened
// mutual repulsion. Entry i holds one shell; its sizes on entry are the point
// counts. Runs on the GPU when compute.cuda=true and on the CPU otherwise.
void solveSphereSurfaceProblem(Vector<Array<Vector<float>, kDimensions>>& x);
void solveSphereSurfaceProblemCpu(Vector<Array<Vector<float>, kDimensions>>& x);
#ifdef USE_CUDA
void solveSphereSurfaceProblemCuda(Vector<Array<Vector<float>, kDimensions>>& x);
#endif

namespace detail {

// Repulsion softening length in units of the mean spacing sqrt(4 pi / n). The
// softened core's potential is finite, so a softening near the spacing (0.9
// before 2026-10-19) lets about 40% of the points settle in coincident pairs.
constexpr float kSphereSurfaceSoftening = 0.5f;
// Velocity damping rate of the relaxation.
constexpr float kSphereSurfaceDamping = 2.0f;
// Step factor in dt = eta min(sqrt(h / gmax), 1 / damping).
constexpr float kSphereSurfaceEta = 0.25f;
// Relaxation stops once kinetic over potential energy falls below this.
constexpr float kSphereSurfaceTolerance = 1e-10f;
constexpr int kSphereSurfaceMinIterations = 100;
// Point i draws its initial position from random stream i % kSphereSurfaceStreams
// seeded with that stream index; this is the CUDA solver's block size.
constexpr int kSphereSurfaceStreams = 512;

// Advances one linear congruential stream and returns a deviate in (-1, 1).
CUDA_EXPORT inline float sphereSurfaceRandom(unsigned& state) {
    constexpr unsigned kMultiplier = 1103515245;
    constexpr unsigned kIncrement = 12345;
    constexpr unsigned kModulus = 1U << 31;
    state = (kMultiplier * state + kIncrement) % kModulus;
    return 2.f * ((state + 0.5f) * (1.f / kModulus)) - 1.f;
}

}    // namespace detail

#endif /* SPHERE_HPP_ */
