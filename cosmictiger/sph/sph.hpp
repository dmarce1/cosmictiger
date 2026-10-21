/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef SPH_HPP_
#define SPH_HPP_

#include <atomic>

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/stack_vector.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/range.hpp>

#ifndef __CUDACC__
struct SphTreeNeighborReturn {
    Fixed32Range innerBox;
    Fixed32Range outerBox;
    CUDA_EXPORT
    SphTreeNeighborReturn() {
        for (int dim = 0; dim < NDIM; dim++) {
            innerBox.begin[dim] = 1.9;
            innerBox.end[dim] = -0.9;
            outerBox.begin[dim] = 1.9;
            outerBox.end[dim] = -0.9;
        }
    }
    CUDA_EXPORT
    SphTreeNeighborReturn& operator+=(SphTreeNeighborReturn const& other) {
        for (int dim = 0; dim < NDIM; dim++) {
            innerBox.begin[dim] = std::min(innerBox.begin[dim], other.innerBox.begin[dim]);
            innerBox.end[dim] = std::max(innerBox.end[dim], other.innerBox.end[dim]);
            outerBox.begin[dim] = std::min(outerBox.begin[dim], other.outerBox.begin[dim]);
            outerBox.end[dim] = std::max(outerBox.end[dim], other.outerBox.end[dim]);
        }
        return *this;
    }
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & innerBox;
        arc & outerBox;
    }
};
#endif

#define SPH_TREE_NEIGHBOR_BOXES 0
#define SPH_TREE_NEIGHBOR_NEIGHBORS 1
#define SPH_INTERACTIONS_I 1
#define SPH_INTERACTIONS_J 2
#define SPH_INTERACTIONS_IJ 3
#define SPH_SET_ACTIVE 1
#define SPH_SET_ALL 4

struct SphTreeNeighborParameters {
    int runType = 0;
    float supportRadiusMultiplier = 1.0f;
    int minimumRung = 0;
    int innerSelection = 0;
    int outerSelection = 0;
    template <typename T>
    void serialize(T&& arc, unsigned) {
        arc & innerSelection;
        arc & outerSelection;
        arc & runType;
        arc & supportRadiusMultiplier;
        arc & minimumRung;
    }
};

struct SphRunReturn {
    float minimumSupportRadius = std::numeric_limits<float>::max();
    float maximumSupportRadius = 0.0f;
    double floatingPointOperations = 0.0;
    int maximumHydroRung = 0;
    int maximumGravityRung = 0;
    int maximumRung = 0;
    float maximumSignalSpeed = 0.0f;
    bool needsAnotherPass = false;
    float inverseSoundTimestep = 0.0f;
    float inverseViscosityTimestep = 0.0f;
    float inverseAccelerationTimestep = 0.0f;
    template <typename Archive>
    void serialize(Archive&& arc, unsigned) {
        arc & needsAnotherPass;
        arc & floatingPointOperations;
        arc & inverseSoundTimestep;
        arc & inverseViscosityTimestep;
        arc & inverseAccelerationTimestep;
        arc & minimumSupportRadius;
        arc & maximumSupportRadius;
        arc & maximumHydroRung;
        arc & maximumGravityRung;
        arc & maximumRung;
        arc & maximumSignalSpeed;
    }
    SphRunReturn& operator+=(SphRunReturn const& other) {
        floatingPointOperations += other.floatingPointOperations;
        maximumSupportRadius = std::max(maximumSupportRadius, other.maximumSupportRadius);
        minimumSupportRadius = std::min(minimumSupportRadius, other.minimumSupportRadius);
        maximumHydroRung = std::max(maximumHydroRung, other.maximumHydroRung);
        maximumGravityRung = std::max(maximumGravityRung, other.maximumGravityRung);
        maximumRung = std::max(maximumRung, other.maximumRung);
        maximumSignalSpeed = std::max(maximumSignalSpeed, other.maximumSignalSpeed);
        inverseSoundTimestep = std::max(inverseSoundTimestep, other.inverseSoundTimestep);
        inverseViscosityTimestep =
            std::max(inverseViscosityTimestep, other.inverseViscosityTimestep);
        inverseAccelerationTimestep =
            std::max(inverseAccelerationTimestep, other.inverseAccelerationTimestep);
        needsAnotherPass = needsAnotherPass || other.needsAnotherPass;
        return *this;
    }
};

struct SphRunParameters {
    int runType;
    int phase;
    int minimumRung;
    float baseTimestep;
    float scaleFactor;      // Cosmological scale factor; one in isolated test problems.
    float expansionRate;    // Expansion rate; zero in isolated test problems.
    float courantFactor;
    float minimumSupportRadius;
    float maximumSupportRadius;
    float uniformAccelerationY;    // Optional uniform force along y.
    float zPlaneAcceleration;      // Magnitude of the force toward z = 0.5.
    float maximumTimestep = 1e30f;
    float viscosityAlpha;
    float viscosityBeta;
    float viscosityEpsilon;
    bool balsaraSwitch;
    float velocityDampingRate;    // Velocity damping used while relaxing the gas sphere.
    SphRunParameters() {
        auto const opts = getOptions();
        minimumSupportRadius = opts.hydro.minSupportRadius;
        maximumSupportRadius = opts.hydro.maxSupportRadius;
        velocityDampingRate = opts.hydro.damping;
        uniformAccelerationY = opts.hydro.yAcceleration;
        zPlaneAcceleration = opts.hydro.zPlaneAcceleration;
        viscosityAlpha = opts.hydro.viscosityAlpha;
        viscosityBeta = opts.hydro.viscosityBeta;
        viscosityEpsilon = opts.hydro.viscosityEpsilon;
        balsaraSwitch = opts.hydro.balsaraSwitch;
    }
    template <typename Archive>
    void serialize(Archive&& arc, unsigned) {
        arc & viscosityAlpha;
        arc & viscosityBeta;
        arc & viscosityEpsilon;
        arc & balsaraSwitch;
        arc & velocityDampingRate;
        arc & minimumSupportRadius;
        arc & maximumSupportRadius;
        arc & expansionRate;
        arc & maximumTimestep;
        arc & uniformAccelerationY;
        arc & baseTimestep;
        arc & phase;
        arc & runType;
        arc & minimumRung;
        arc & scaleFactor;
        arc & courantFactor;
        arc & zPlaneAcceleration;
    }
    CUDA_EXPORT
    float midplaneAccelerationZ(Fixed32 const z) const {
        Fixed32 const middle(0.5);
        return z < middle ? zPlaneAcceleration : (z > middle ? -zPlaneAcceleration : 0.0f);
    }
};

#define SPH_RUN_PREHYDRO1 0
#define SPH_RUN_PREHYDRO2 1
#define SPH_RUN_HYDRO 2
#define SPH_RUN_RUNGS 3

SphRunReturn sphRun(SphRunParameters params, bool cuda = false);
#ifndef __CUDACC__
hpx::future<SphTreeNeighborReturn> sphTreeNeighbor(
    SphTreeNeighborParameters params, TreeId self, Vector<TreeId> checklist, int level = 0);
#endif

#endif /* SPH_HPP_ */
