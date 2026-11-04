/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/math/vector3.hpp>

// Extra state exists only with hierarchical gravity. The ordinary SPH acceleration
// remains the full predictor derivative; subtracting hydroAcceleration recovers
// the coarse gravitational predictor without a separate stored gravity vector.
struct SphHierarchicalState {
    Vector3<float> hydroAcceleration = Vector3<float>(0.0f);
    Vector3<float> predictedVelocity = Vector3<float>(0.0f);
    double predictionTime = 0.0;
    double positionTime = 0.0;
    double inverseDriftScale = 1.0;
    double entropyTime = 0.0;
    int minimumHydroRung = 0;
    bool initialized = false;

    template <typename Archive>
    void serialize(Archive&& archive, unsigned) {
        archive & hydroAcceleration;
        archive & predictedVelocity;
        archive & predictionTime;
        archive & positionTime;
        archive & inverseDriftScale;
        archive & entropyTime;
        archive & minimumHydroRung;
        archive & initialized;
    }
};
