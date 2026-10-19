/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#include <cosmictiger/sph/sph_cpu.hpp>

SphRunReturn sphRunCpuNode(
    SphRunParameters const& parameters, SphRunData const& data, int targetNodeIndex) {
    switch (parameters.runType) {
    case SPH_RUN_PREHYDRO1:
        return sphCpuPrehydro1(parameters, data, targetNodeIndex);
    case SPH_RUN_PREHYDRO2:
        return sphCpuPrehydro2(parameters, data, targetNodeIndex);
    case SPH_RUN_HYDRO:
        return sphCpuHydro(parameters, data, targetNodeIndex);
    case SPH_RUN_RUNGS:
        return sphCpuRungs(parameters, data, targetNodeIndex);
    default:
        ALWAYS_ASSERT(false);
        return {};
    }
}
