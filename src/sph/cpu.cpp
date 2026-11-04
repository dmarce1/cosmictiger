/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <cosmictiger/sph/cpu.hpp>

SphRunReturn sphRunCpuNode(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    switch (parameters.runType) {
    case kSphRunSmoothingLengths:
        return sphCpuSmoothingLengths(parameters, data, targetNodeIndex);
    case kSphRunDensityAndGradients:
        return sphCpuDensityAndGradients(parameters, data, targetNodeIndex);
    case kSphRunHydro:
        return sphCpuHydro(parameters, data, targetNodeIndex);
    case kSphRunRungs:
        return sphCpuRungs(parameters, data, targetNodeIndex);
    default:
        ALWAYS_ASSERT(false);
        return {};
    }
}
