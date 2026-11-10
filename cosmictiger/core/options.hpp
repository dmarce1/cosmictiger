/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cstdint>
#include <stddef.h>
#include <string>

#include <cosmictiger/core/build_config.hpp>
#include <cosmictiger/gravity/build_config.hpp>

// Public data groups mirror the CLI/INI category names. Radius aliases are
// resolved during parsing; canonical radii and unit conversions are stored below.
struct Run {
    size_t darkMatterParticleCount{};
    size_t gasParticleCount{};
    std::string scenario{gravity::kBuildHasGravity ? "cosmology" : "sod"};
    double endTime{};
    std::string test{};
    bool verbose{};
    int maxIterations{};
    int numberOfSteps{};
    int minRung{};
    // Hierarchical gravity merges its lowest occupied rung into the next one when
    // the finer rungs hold at least this fraction of the coarse solve set
    // (GADGET-4's N1 > N0/3, including cost ties; docs/hierarchical_gravity.md).
    double rungMergeFraction{1.0 / 3.0};
    // run.test=force: build the tree once at this opening angle and sweep only
    // the kick's angle upward from it; zero rebuilds the tree at every angle.
    double treeOpeningAngle{};
    // Retain the serialized fields; gravity uses the sole hierarchical ladder.
    bool hierarchicalGravity{integration::kBuildUsesHierarchical};
    std::string timeIntegrator{integration::kBuildTimeIntegrator};
};

struct Compute {
    std::string positionType{"fixed32"};
    int bucketSize{};
    // Zero input derives the SPH leaf size from hydro.neighborNumber; startup
    // stores the resolved positive size independently of the gravity bucket.
    int sphBucketSize{};
    // Rebuild the SPH octree when this fraction of all particles is active.
    double sphTreeRebuildFraction{1.0 / 3.0};
    int treeCacheLineSize{};
    int treeAllocLineSize{};
    int particleCacheLineSize{};
};

struct Drift {
    bool cuda{};
};

struct Sort {
    bool cuda{};
};

struct Checkpoint {
    int readIndex{};
    int intervalSeconds{};
};

struct Initial {
    bool cuda{};
    bool closePack{};
    // Derived geometry and mass-normalization population, never public options.
    int latticeSide{};
    int darkMatterLatticeSide{};
    int gasLatticeSide{};
    size_t numberOfParticles{};
    // Derived FFT mesh. numberOfParticles is the reference population used by
    // code mass/length units: max(darkMatterParticleCount, gasParticleCount).
    int fourierSize{};
    double darkMatterMass{};
    double gasMass{};
};

// Equal even lattices retain exact spectral phase sampling. Other resolutions
// use the existing interpolation-corrected LPT sampling on a common even FFT.
inline bool interpolatedInitialConditions(Initial const& initial, bool useGlass) {
    return useGlass ||
        (initial.darkMatterLatticeSide != 0 &&
            initial.darkMatterLatticeSide != initial.fourierSize) ||
        (initial.gasLatticeSide != 0 && initial.gasLatticeSide != initial.fourierSize);
}

struct Output {
    bool silo{};
    int siloInterval{};
    std::string directory{};
    std::string units{"code"};
    std::string frame{"comoving"};
    double xMin{};
    double xMax{};
    double yMin{};
    double yMax{};
    double zMin{};
    double zMax{};
};

struct Analysis {
    bool powerCuda{};
    bool groupsCuda{};
    bool tracersCuda{};
    bool sliceCuda{};
    bool viewsCuda{};
    bool power{};
    bool groups{};
    bool tracers{};
    int tracerCount{};
    bool slice{};
    bool views{};
    int viewSize{};
    int sliceResolution{};
    double sliceSize{};
    double linkingLength{};
    int minGroupSize{};
    // Legacy request for both diagnostic fields; independent requests below
    // permit retaining either field without allocating the other.
    bool saveForce{};
    bool saveAcceleration{};
    bool savePotential{};
};

/// Test whether either the combined-force or acceleration-only output requests stored
/// accelerations.
inline bool saveParticleAcceleration(Analysis const& analysis) {
    return analysis.saveForce || analysis.saveAcceleration;
}

/// Test whether either the combined-force or potential-only output requests stored
/// potentials.
inline bool saveParticlePotential(Analysis const& analysis) {
    return analysis.saveForce || analysis.savePotential;
}

struct Lightcone {
    bool cuda{};
    bool enabled{};
    std::string directory{};
    double linkingLength{};
    int mapSize{};
};

struct Gravity {
    bool cuda{};
    bool errorCuda{};
    // CUDA kick shared-memory policy; zero selects the automatic choice.
    int cudaKickResidency{};            // resident kick blocks per multiprocessor
    int cudaKickSinkTile{};             // shared sink-force tile; -1 keeps forces in global memory
    int cudaKickParticleTile{};         // staged particle sources
    int cudaKickMultipoleTile{};        // staged positioned multipoles
    bool cudaKickSharedExpansion{};     // keep the active local expansion in shared memory
    int cudaKickHandoffBlocks{};        // pins the CPU/GPU handoff divisor; zero derives it
    int cudaKickFineHandoffBlocks{};    // finer handoff divisor for the near-cell walks; zero
                                        // (or below the divisor above) keeps that one
    int cudaKickListReserve{};    // pool entries reserved per kick list at block start; zero grows
                                  // on demand
    bool enabled{};
    // Coarse-step cadence for sampled direct-force comparisons; zero disables.
    int errorInterval{};
    // Evenly spaced global particle indices, capped at the particle count.
    int errorSampleSize{100};
    int meshSize{};    // Full-particle PMMM side; any positive integer, scaled per stage.
    // Coarsest hierarchical stage mesh and the granularity of particle-domain
    // cuts; a divisor of meshSize. Zero chooses a compatible divisor near meshSize/8.
    int pmmmCoarseMeshSize{};
    int pmmmNearWidth{2};    // Direct neighbor cube radius for periodic PMMM, one to four cells.
    // Near field of the PMMM solver: fmm, or its direct particle-pair fallback pp.
    // The field keeps its historical serialized name.
    int pmmmBatchParticles{4096};    // Target owned sink count per ready near-field batch.
    // PMMM outer interval: the FFT far field is solved only at synchronized
    // outer boundaries, each particle's near-cube membership is frozen for the
    // interval, and the near field advances on the inner rungs (see docs/pmmm.md).
    bool pmmmOuterInterval{true};
    // Potential expansion degree + 1 of the far and near fields; template/storage
    // convention.
    int fmmOrder{gravity::kBuildFmmOrder};
    std::string kernelFamily{"ferrers"};
    int kernelDegree{4};
    // Near-field FMM opening angle, derived from the neighbor cube geometry as
    // sqrt(3)/(pmmmNearWidth+1); not a configured option.
    double openingAngle{};
    // Open only the larger of two rejected cells instead of both.
    bool openLargerCell{};
    // Fixed comoving compact support in box units; effectiveRadius is 2 sigma.
    double smoothingLength{};
    double timestepEta{};
    double gm{};
};

// PMMM with its FFT far field on an outer interval and frozen near membership
// inside; the inner rung clock subdivides that interval.
inline bool usesOuterInterval(Gravity const& gravityOptions) {
    return gravityOptions.pmmmOuterInterval;
}

/// Test whether the selected time integrator is hierarchical.
inline bool usesHierarchicalIntegration(Run const& run) {
    return run.timeIntegrator == "hierarchical";
}

/// Test whether the selected time integrator is global.
inline bool usesGlobalIntegration(Run const& run) {
    return run.timeIntegrator == "global";
}

struct Cosmology {
    double startRedshift{};
    double endRedshift{};
    // Derived from the density budget; not a configurable input.
    double omegaK{};
    double omegaLambda{};
    double omegaB{};
    double omegaC{};
    double nEff{};
    double thetaCmb{};
    double heliumFraction{};
    double sigma8{};
    double hubble{};
    double spectralIndex{};
    double omegaM{};
    double omegaR{};
    double omegaGamma{};
    double omegaNu{};
};

struct Units {
    // Canonical code-unit conversions; units.massSolar is converted to grams at startup.
    double massG{};
    double lengthCm{};
    double timeSeconds{};
};

struct Hydro {
    bool cuda{};
    bool enabled{};
    std::string method{"sph"};
    std::string kernelFamily{"spline"};
    int kernelDegree{3};
    // Pairs use beta_ij = vonNeumannRichtmyerCoefficient alpha_ij with Rosswog-limited alpha.
    double vonNeumannRichtmyerCoefficient{};
    double viscosityEpsilon{};
    double conductionCoefficient{0.0};
    double minimumBulkViscosityCoefficient{0.01};
    double maximumBulkViscosityCoefficient{2.0};
    double viscosityDecay{0.2};
    double balsaraSoundSpeedCoefficient{1.0e-4};
    double heatCapacityRatio{};
    double cfl{};
    double neighborNumber{42.0};
    // Canonical support bounds in box units, normalized from input particle spacings.
    double minSmoothingLength{};
    double maxSmoothingLength{};
    // Optional uniform external acceleration in the y direction.
    double yAcceleration{};
    // Velocity damping during the self-gravitating gas relaxation test.
    double damping{};
};

struct Chemistry {
    bool cuda{};
    // Compatibility group name: GADGET-4 equilibrium atomic cooling.
    bool enabled{false};
    // GADGET-4 InitGasTemp; zero retains the initial-condition thermal energy.
    double initializationTemperature{10000.0};
    // Physical CGS erg/g, independent of cosmological storage factors.
    double minimumSpecificEnergy{0.0};
};

struct StarFormation {
    bool cuda{};
    bool enabled{false};
    double beta{0.1};
    double evaporationFactor{1000.0};
    double starFormationTimeGyr{2.1};
    double coldTemperature{1000.0};
    double supernovaTemperature{1.0e8};
    // Hydrogen nuclei/cm^3; zero derives the threshold as in GADGET-4.
    double criticalPhysicalDensity{0.0};
    double criticalOverDensity{57.7};
    std::uint64_t seed{42};
};

struct CosmologyScenario {
    std::string gadget4Restart{};
    bool useGlass{};
    bool twoLpt{};
};

struct PlummerScenario {
    double radius{};
};

struct KelvinHelmholtzScenario {
    std::string packing{"CV/C"};
};

// Fixed central potential for the orbiting gas disc.
struct DiscScenario {
    double centralAcceleration{};
    double centralSmoothingLength{};
};

struct RayleighTaylorScenario {
    // Magnitude of the external acceleration toward the plane z = 0.5.
    double zPlaneAcceleration{};
};

// Options read only by the run.scenario problem of the same name.
struct Scenario {
    CosmologyScenario cosmology{};
    PlummerScenario plummer{};
    KelvinHelmholtzScenario kelvinHelmholz{};
    DiscScenario disc{};
    RayleighTaylorScenario rayleighTaylor{};
};

struct Options {
    std::string configuration;
    Run run;
    Compute compute;
    Drift drift;
    Sort sort;
    Checkpoint checkpoint;
    Initial initial;
    Scenario scenario;
    Output output;
    Analysis analysis;
    Lightcone lightcone;
    Gravity gravity;
    Cosmology cosmology;
    Units units;
    Hydro hydro;
    Chemistry chemistry;
    StarFormation starFormation;

    /// Archive `run.hierarchicalGravity`, `run.timeIntegrator`, `hydro.enabled`,
    /// `gravity.enabled`, `compute.sphBucketSize`,
    /// `hydro.vonNeumannRichtmyerCoefficient` and the remaining fields in transport
    /// order.
    template <typename Archive>
    void serialize(Archive&& arc, unsigned) {
        // Preserve the historical scalar wire order independently of grouping.
        int retiredLightconeMinGroup = 20;
        bool retiredGlassSelector = run.scenario == "glass";
        bool retiredPlummerSelector = run.scenario == "plummer";
        arc & run.hierarchicalGravity;
        arc & run.timeIntegrator;
        arc & hydro.enabled;
        arc & gravity.enabled;
        arc & compute.sphBucketSize;
        arc & hydro.vonNeumannRichtmyerCoefficient;
        arc & hydro.viscosityEpsilon;
        arc & hydro.minimumBulkViscosityCoefficient;
        arc & hydro.maximumBulkViscosityCoefficient;
        arc & hydro.balsaraSoundSpeedCoefficient;
        arc & hydro.viscosityDecay;
        arc & hydro.neighborNumber;
        arc & hydro.minSmoothingLength;
        arc & hydro.maxSmoothingLength;
        arc & hydro.damping;
        arc & initial.darkMatterMass;
        arc & initial.gasMass;
        arc & hydro.cfl;
        arc & hydro.heatCapacityRatio;
        arc & hydro.yAcceleration;
        arc & scenario.disc.centralAcceleration;
        arc & scenario.disc.centralSmoothingLength;
        arc & initial.closePack;
        arc & initial.numberOfParticles;
        arc & scenario.cosmology.useGlass;
        arc & retiredGlassSelector;
        arc & scenario.cosmology.gadget4Restart;
        arc & run.minRung;
        arc & run.treeOpeningAngle;
        arc & retiredPlummerSelector;
        arc & scenario.plummer.radius;
        arc & cosmology.omegaK;
        arc & cosmology.omegaLambda;
        arc & compute.bucketSize;
        arc & run.numberOfSteps;
        arc & initial.fourierSize;
        arc & gravity.cuda;
        arc & compute.positionType;
        arc & lightcone.enabled;
        arc & analysis.saveForce;
        arc & analysis.power;
        arc & analysis.groups;
        arc & analysis.tracers;
        arc & analysis.slice;
        arc & analysis.views;
        arc & output.silo;
        arc & output.siloInterval;
        arc & output.directory;
        arc & output.units;
        arc & output.frame;
        arc & output.xMin;
        arc & output.xMax;
        arc & output.yMin;
        arc & output.yMax;
        arc & output.zMin;
        arc & output.zMax;
        arc & scenario.cosmology.twoLpt;
        arc & lightcone.directory;
        arc & analysis.tracerCount;
        arc & initial.latticeSide;
        arc & compute.treeCacheLineSize;
        arc & compute.treeAllocLineSize;
        arc & compute.particleCacheLineSize;
        arc & gravity.cudaKickResidency;
        arc & gravity.cudaKickSinkTile;
        arc & gravity.cudaKickParticleTile;
        arc & gravity.cudaKickMultipoleTile;
        arc & gravity.cudaKickSharedExpansion;
        arc & gravity.cudaKickHandoffBlocks;
        arc & gravity.cudaKickFineHandoffBlocks;
        arc & gravity.cudaKickListReserve;
        arc & gravity.fmmOrder;
        arc & checkpoint.readIndex;
        arc & checkpoint.intervalSeconds;
        arc & run.maxIterations;
        arc & retiredLightconeMinGroup;
        arc & analysis.minGroupSize;
        arc & analysis.viewSize;
        arc & analysis.sliceResolution;
        arc & lightcone.mapSize;
        arc & lightcone.linkingLength;
        arc & gravity.openingAngle;
        arc & gravity.openLargerCell;
        arc & analysis.sliceSize;
        arc & analysis.linkingLength;
        arc & gravity.smoothingLength;
        arc & gravity.gm;
        arc & gravity.timestepEta;
        arc & units.timeSeconds;
        arc & units.lengthCm;
        arc & units.massG;
        arc & cosmology.omegaM;
        arc & cosmology.omegaR;
        arc & cosmology.startRedshift;
        arc & cosmology.endRedshift;
        arc & cosmology.hubble;
        arc & cosmology.sigma8;
        arc & cosmology.omegaB;
        arc & cosmology.omegaC;
        arc & cosmology.omegaGamma;
        arc & cosmology.omegaNu;
        arc & cosmology.thetaCmb;
        arc & cosmology.spectralIndex;
        arc & cosmology.heliumFraction;
        arc & cosmology.nEff;
        arc & configuration;
        arc & run.test;
        arc & run.scenario;
        arc & run.endTime;
        arc & gravity.kernelFamily;
        arc & gravity.kernelDegree;
        arc & hydro.kernelFamily;
        arc & hydro.kernelDegree;
        arc & chemistry.enabled;
        arc & chemistry.initializationTemperature;
        arc & chemistry.minimumSpecificEnergy;
        arc & run.verbose;
        arc & scenario.rayleighTaylor.zPlaneAcceleration;
        arc & hydro.method;
        arc & gravity.meshSize;
        arc & gravity.pmmmNearWidth;
        arc & analysis.saveAcceleration;
        arc & analysis.savePotential;
        arc & gravity.pmmmBatchParticles;
        arc & gravity.pmmmCoarseMeshSize;
        arc & gravity.pmmmOuterInterval;
        arc & gravity.errorInterval;
        arc & gravity.errorSampleSize;
        arc & starFormation.enabled;
        arc & starFormation.beta;
        arc & starFormation.evaporationFactor;
        arc & starFormation.starFormationTimeGyr;
        arc & starFormation.coldTemperature;
        arc & starFormation.supernovaTemperature;
        arc & starFormation.criticalPhysicalDensity;
        arc & starFormation.criticalOverDensity;
        arc & starFormation.seed;
        arc & hydro.cuda;
        arc & drift.cuda;
        arc & sort.cuda;
        arc & initial.cuda;
        arc & gravity.errorCuda;
        arc & analysis.powerCuda;
        arc & analysis.groupsCuda;
        arc & analysis.tracersCuda;
        arc & analysis.sliceCuda;
        arc & analysis.viewsCuda;
        arc & lightcone.cuda;
        arc & chemistry.cuda;
        arc & starFormation.cuda;
        arc & compute.sphTreeRebuildFraction;
        arc & run.rungMergeFraction;
        arc & hydro.conductionCoefficient;
        arc & run.darkMatterParticleCount;
        arc & run.gasParticleCount;
        arc & initial.darkMatterLatticeSide;
        arc & initial.gasLatticeSide;
        arc & scenario.kelvinHelmholz.packing;
    }
};

// Halo and lightcone samples represent dark matter alone. The mass conversion
// also supports fixtures that set an explicit reference mass normalization.
inline double darkMatterReferenceCount(Options const& options) {
    if (!options.hydro.enabled) {
        return options.initial.numberOfParticles;
    }
    double const fraction =
        options.cosmology.omegaM > 0.0 ? options.cosmology.omegaC / options.cosmology.omegaM : 1.0;
    return options.initial.darkMatterMass > 0.0 ?
        options.initial.numberOfParticles * fraction / options.initial.darkMatterMass :
        0.0;
}

// Device resources are shared, but each module selects its own execution backend.
inline bool usesCuda(Options const& options) {
    return options.gravity.cuda || options.gravity.errorCuda || options.hydro.cuda ||
        options.drift.cuda || options.sort.cuda || options.initial.cuda ||
        options.analysis.powerCuda || options.analysis.groupsCuda || options.analysis.tracersCuda ||
        options.analysis.sliceCuda || options.analysis.viewsCuda || options.lightcone.cuda ||
        options.chemistry.cuda || options.starFormation.cuda;
}

bool processOptions(int argc, char* argv[]);
Options const& getOptions();
void setOptions(Options const& opts);
