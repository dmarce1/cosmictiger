/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <cosmictiger/sph/sph_cpu.hpp>

SphRunReturn sphRunCpuNode(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    switch (parameters.runType) {
    case kSphRunPrehydro1:
        return sphCpuPrehydro1(parameters, data, targetNodeIndex);
    case kSphRunPrehydro2:
        return sphCpuPrehydro2(parameters, data, targetNodeIndex);
    case kSphRunHydro:
        return sphCpuHydro(parameters, data, targetNodeIndex);
    case kSphRunRungs:
        return sphCpuRungs(parameters, data, targetNodeIndex);
    default:
        ALWAYS_ASSERT(false);
        return {};
    }
}
