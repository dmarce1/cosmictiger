/* CosmicTiger - streamed mesh-field kick consumer; BSL-1.0. */
#pragma once

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/math/vector3.hpp>

// Consume one completed, gm-scaled field for an owned sink without retaining a
// force array. Particle membership is fixed by the caller's current ladder
// stage; only this owned sink's velocity/rung and requested diagnostics change.
// Traditional hydro sinks receive the acceleration only; hydro owns their kicks
// and rungs. Traditional sources must contain every owned particle even when
// minRung selects only a subset of the sinks.
//
// Particle-mesh solvers add the per-particle long acceleration: the closing
// half-kick uses parameters.closingBaseTimestep (the previous outer interval)
// when it is positive, the long half-kicks use longCloseTimestep and
// longOpenTimestep, and the saved/sampled acceleration is the long plus short
// total. Gas receives the long half-kicks directly; their diagnostic
// corrections share the dark-matter correction channels.
//
// Accumulate source-operation counts in flopCount when supplied, so callers can
// submit a whole sink batch once; otherwise submit this particle's count here.
// precisionCounts optionally receives the same work split by arithmetic precision.
// Supplying either output defers global submission to the caller.
KickReturn particleMeshApplyField(ParticleInteger index, Vector3<float> const& acceleration,
    float potential, KickParameters const& parameters, FlopCount* flopCount = nullptr,
    FlopCounts* precisionCounts = nullptr);

// Pure PM inner kick: every selected sink receives a zero short field, so the
// rung proposal comes from hydro alone and the long half-kicks are applied.
KickReturn particleMeshZeroFieldKick(KickParameters const& parameters);
