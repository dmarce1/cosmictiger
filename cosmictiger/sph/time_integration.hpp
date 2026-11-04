/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/hierarchical_state.hpp>

struct KickParameters;

// These collective calls run only after gravity sorting/migration has completed.
// Begin freezes the event-time predictor and projects inactive source positions;
// end restores those positions before the next gravity rung can drift.
void sphHierarchyBeginEvent(double time, int minimumRung, double baseTimestep, bool fullGravity,
    KickParameters const& gravityParameters);
void sphHierarchyEndEvent();
void sphHierarchyCaptureRungs();
void sphHierarchyOpenRung(int rung);
void sphHierarchyRecordDrift(int rung, double endTime, double inverseScale);

bool sphHierarchyEventActive();
bool sphHierarchyFullGravity();
float sphHierarchyPredictedVelocity(int dimension, ParticleInteger gas);
double sphHierarchyPredictedEntropy(ParticleInteger gas);
Vector3<float> sphHierarchyGravityPrediction(ParticleInteger gas);

// Called once for each active gas target, before the gravity closing kick.
// oldHydro must be captured before the shared SPH kernel publishes new rates.
void sphHierarchyCloseTarget(
    ParticleInteger gas, Vector3<float> const& oldHydro, Vector3<float> const& newHydro);
// Called after the gravity closing kick and before its opening kick. Only a
// full-source gravity event can synchronize this predictor to the live momentum.
void sphHierarchySynchronizeTarget(ParticleInteger gas, Vector3<float> const& velocity);
int sphHierarchyKickTarget(ParticleInteger gas, Vector3<float> const& oldHydro,
    Vector3<float> const& newHydro, Vector3<float> const& leafGravity);
