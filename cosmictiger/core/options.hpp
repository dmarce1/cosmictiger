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
    std::string scenario{"cosmology"};
    double endTime{};
    std::string test{};
    bool verbose{};
    int maxIterations{};
    int numberOfSteps{};
    int minRung{};
    // run.test=force: build the tree once at this opening angle and sweep only
    // the kick's angle upward from it; zero rebuilds the tree at every angle.
    double treeOpeningAngle{};
    // Retain the historical boolean while the selector distinguishes all three modes.
    bool hierarchicalGravity{integration::kBuildUsesHierarchical};
    std::string timeIntegrator{integration::kBuildTimeIntegrator};
};

struct Compute {
    std::string positionType{"fixed32"};
    bool cuda{};
    int bucketSize{};
    // Zero input derives the SPH leaf size from hydro.neighborNumber; startup
    // stores the resolved positive size independently of the gravity bucket.
    int sphBucketSize{};
    int treeCacheLineSize{};
    int treeAllocLineSize{};
    int particleCacheLineSize{};
    // CUDA kick shared-memory policy; zero selects the automatic choice.
    int cudaKickResidency{};            // resident kick blocks per multiprocessor
    int cudaKickSinkTile{};             // shared sink-force tile; -1 keeps forces in global memory
    int cudaKickParticleTile{};         // staged particle sources
    int cudaKickMultipoleTile{};        // staged positioned multipoles
    bool cudaKickSharedExpansion{};     // keep the active local expansion in shared memory
    int cudaKickHandoffBlocks{};        // pins the CPU/GPU handoff divisor; zero derives it
    int cudaKickFineHandoffBlocks{};    // handoff divisor once a node carries no Ewald checks;
                                        // zero (or below the divisor above) keeps that one
    int cudaKickListReserve{};    // pool entries reserved per kick list at block start; zero grows
                                  // on demand
};

struct Checkpoint {
    int readIndex{};
    int intervalSeconds{};
};

struct Initial {
    std::string gadget4Restart{};
    bool useGlass{};
    bool createGlass{};
    bool plummer{};
    double plummerRadius{};
    bool closePack{};
    bool twoLpt{};
    int particlesPerDimension{};
    size_t numberOfParticles{};
    // Derived initial-condition mesh and particle population.
    int fourierSize{};
    double darkMatterMass{};
    double gasMass{};
};

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

inline bool saveParticleAcceleration(Analysis const& analysis) {
    return analysis.saveForce || analysis.saveAcceleration;
}

inline bool saveParticlePotential(Analysis const& analysis) {
    return analysis.saveForce || analysis.savePotential;
}

struct Lightcone {
    bool enabled{};
    std::string directory{};
    double linkingLength{};
    int mapSize{};
};

struct Gravity {
    bool enabled{};
    // Coarse-step cadence for sampled direct-force comparisons; zero disables.
    int errorInterval{};
    // Evenly spaced global particle indices, capped at the particle count.
    int errorSampleSize{100};
    // Selected algorithms retain their historical serialized field names.
    std::string method{gravity::kBuildGravityMethod};
    int meshSize{};    // Requested full-particle PMMM side; scale by rung before FFT rounding.
    int pmmmMaxPrimeFactor{5};    // Inclusive FFT mesh prime-factor limit; zero is unrestricted.
    int pmmmNearWidth{2};    // Direct neighbor cube radius for periodic PMMM, one to four cells.
    int pmmmOrder{gravity::kBuildFmmOrder};    // Potential expansion degree + 1, shared by far and
                                               // near fields.
    std::string pmmmNearMethod{gravity::kBuildPmmmNearMethod};
    int pmmmBatchParticles{4096};             // Target owned sink count per ready near-field batch.
    int fmmOrder{gravity::kBuildFmmOrder};    // Potential expansion degree + 1; template/storage convention.
    std::string kernelFamily{"ferrers"};
    int kernelDegree{4};
    double openingAngle{};
    // Global cosmological FMM redshift:theta pairs; empty retains the scalar angle.
    std::string openingSchedule{};
    // Open only the larger of two rejected cells instead of both.
    bool openLargerCell{};
    // Fixed comoving compact support in box units; effectiveRadius is 2 sigma.
    double supportRadius{};
    double timestepEta{};
    double gm{};
};

inline bool usesMeshGravity(Gravity const& gravityOptions) {
    return gravityOptions.method == "pmmm";
}

inline bool usesHierarchicalIntegration(Run const& run) {
    return run.timeIntegrator == "hierarchical";
}

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
    bool enabled{};
    std::string kernelFamily{"spline"};
    int kernelDegree{3};
    double viscosityAlpha{};
    double viscosityBeta{};
    double viscosityEpsilon{};
    bool balsaraSwitch{true};
    // Pressure force and reversible work: monaghan (Monaghan 1992, the regression baseline)
    // or variational (compatible density-energy SPH with grad-H, docs/variational_sph.md).
    std::string pressureOperator{"monaghan"};
    std::string viscosityEvolution{"morrisAndMonaghan"};
    double viscosityAlphaMin{0.1};
    double viscosityDecay{0.2};
    double viscosityAlphaMax{2.0};
    double viscosityDecayCD{0.05};
    double viscosityAlphaMaxPrice{1.0};
    double viscosityDecayPrice{0.1};
    // Cullen-Dehnen and Price et al. share the gradient-based shock detector and its state.
    bool viscosityGradientSwitch() const {
        return viscosityEvolution == "cullenAndDehnen" || viscosityEvolution == "priceEtAl";
    }
    // Morris-Monaghan and Cullen-Dehnen use beta_ij = viscosityBeta alpha_ij; the fixed
    // Monaghan coefficients and Price et al. use beta_ij = viscosityBeta.
    bool viscosityBetaFollowsAlpha() const {
        return viscosityEvolution == "morrisAndMonaghan" || viscosityEvolution == "cullenAndDehnen";
    }
    double gamma{};
    double cfl{};
    double neighborNumber{42.0};
    // Canonical support bounds in box units, normalized from input particle spacings.
    double minSupportRadius{};
    double maxSupportRadius{};
    // Optional uniform external acceleration in the y direction.
    double yAcceleration{};
    // Magnitude of the external acceleration toward the plane z = 0.5.
    double zPlaneAcceleration{};
    // Fixed central potential for the orbiting-disc gas test.
    double centralAcceleration{};
    double centralSupportRadius{};
    // Velocity damping during the self-gravitating gas relaxation test.
    double damping{};
};

struct Chemistry {
    // Source half steps follow each active gas particle's hydrodynamic timestep.
    bool enabled{false};
    std::string network{"atomic6"};
    std::string initialization{"recombination"};
    std::string radiationFile{""};
    double initialTemperature{1000.0};
    double hydrogenIonFraction{2.0e-4};
    double heliumIonFraction{0.0};
    double heliumDoubleIonFraction{0.0};
    double hydrogenMinusFraction{0.0};
    double molecularFraction{0.0};
    double molecularIonFraction{0.0};
    double relativeTolerance{1.0e-4};
    double abundanceAbsoluteTolerance{1.0e-12};
    double energyAbsoluteTolerance{1.0e-8};
    int maxSubsteps{4096};
};

struct StarFormation {
    bool enabled{false};
    double beta{0.1};
    double evaporationFactor{1000.0};
    double starFormationTimeGyr{2.1};
    double coldTemperature{1000.0};
    double supernovaTemperature{1.0e8};
    std::uint64_t seed{1};
    double metalYield{0.02};
    bool winds{false};
    double windMassLoading{2.0};
    double windEnergyFraction{0.25};
    double windDecouplingMyr{50.0};
    double windRecoupleDensityFraction{0.1};
    std::string windDirection{"isotropic"};
};

struct Options {
    std::string configuration;
    Run run;
    Compute compute;
    Checkpoint checkpoint;
    Initial initial;
    Output output;
    Analysis analysis;
    Lightcone lightcone;
    Gravity gravity;
    Cosmology cosmology;
    Units units;
    Hydro hydro;
    Chemistry chemistry;
    StarFormation starFormation;

    template <typename Archive>
    void serialize(Archive&& arc, unsigned) {
        // Preserve the historical scalar wire order independently of grouping.
        int retiredLightconeMinGroup = 20;
        arc & run.hierarchicalGravity;
        arc & run.timeIntegrator;
        arc & hydro.enabled;
        arc & gravity.enabled;
        arc & compute.sphBucketSize;
        arc & hydro.viscosityAlpha;
        arc & hydro.viscosityBeta;
        arc & hydro.viscosityEpsilon;
        arc & hydro.balsaraSwitch;
        arc & hydro.viscosityEvolution;
        arc & hydro.viscosityAlphaMin;
        arc & hydro.viscosityDecay;
        arc & hydro.viscosityAlphaMax;
        arc & hydro.viscosityDecayCD;
        arc & hydro.viscosityAlphaMaxPrice;
        arc & hydro.viscosityDecayPrice;
        arc & hydro.neighborNumber;
        arc & hydro.minSupportRadius;
        arc & hydro.maxSupportRadius;
        arc & hydro.damping;
        arc & initial.darkMatterMass;
        arc & initial.gasMass;
        arc & hydro.cfl;
        arc & hydro.gamma;
        arc & hydro.yAcceleration;
        arc & hydro.centralAcceleration;
        arc & hydro.centralSupportRadius;
        arc & initial.closePack;
        arc & initial.numberOfParticles;
        arc & initial.useGlass;
        arc & initial.createGlass;
        arc & initial.gadget4Restart;
        arc & run.minRung;
        arc & run.treeOpeningAngle;
        arc & initial.plummer;
        arc & initial.plummerRadius;
        arc & cosmology.omegaK;
        arc & cosmology.omegaLambda;
        arc & compute.bucketSize;
        arc & run.numberOfSteps;
        arc & initial.fourierSize;
        arc & compute.cuda;
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
        arc & initial.twoLpt;
        arc & lightcone.directory;
        arc & analysis.tracerCount;
        arc & initial.particlesPerDimension;
        arc & compute.treeCacheLineSize;
        arc & compute.treeAllocLineSize;
        arc & compute.particleCacheLineSize;
        arc & compute.cudaKickResidency;
        arc & compute.cudaKickSinkTile;
        arc & compute.cudaKickParticleTile;
        arc & compute.cudaKickMultipoleTile;
        arc & compute.cudaKickSharedExpansion;
        arc & compute.cudaKickHandoffBlocks;
        arc & compute.cudaKickFineHandoffBlocks;
        arc & compute.cudaKickListReserve;
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
        arc & gravity.supportRadius;
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
        arc & chemistry.network;
        arc & chemistry.initialization;
        arc & chemistry.radiationFile;
        arc & chemistry.initialTemperature;
        arc & chemistry.hydrogenIonFraction;
        arc & chemistry.heliumIonFraction;
        arc & chemistry.heliumDoubleIonFraction;
        arc & chemistry.hydrogenMinusFraction;
        arc & chemistry.molecularFraction;
        arc & chemistry.molecularIonFraction;
        arc & chemistry.relativeTolerance;
        arc & chemistry.abundanceAbsoluteTolerance;
        arc & chemistry.energyAbsoluteTolerance;
        arc & chemistry.maxSubsteps;
        arc & run.verbose;
        arc & hydro.zPlaneAcceleration;
        arc & gravity.method;
        arc & gravity.meshSize;
        arc & gravity.pmmmNearWidth;
        arc & gravity.pmmmNearMethod;
        arc & analysis.saveAcceleration;
        arc & analysis.savePotential;
        arc & gravity.pmmmBatchParticles;
        arc & gravity.pmmmOrder;
        arc & gravity.openingSchedule;
        arc & gravity.pmmmMaxPrimeFactor;
        arc & gravity.errorInterval;
        arc & gravity.errorSampleSize;
        arc & starFormation.enabled;
        arc & starFormation.beta;
        arc & starFormation.evaporationFactor;
        arc & starFormation.starFormationTimeGyr;
        arc & starFormation.coldTemperature;
        arc & starFormation.supernovaTemperature;
        arc & starFormation.seed;
        arc & starFormation.metalYield;
        arc & starFormation.winds;
        arc & starFormation.windMassLoading;
        arc & starFormation.windEnergyFraction;
        arc & starFormation.windDecouplingMyr;
        arc & starFormation.windRecoupleDensityFraction;
        arc & starFormation.windDirection;
        arc & hydro.pressureOperator;
    }
};

bool processOptions(int argc, char* argv[]);
Options const& getOptions();
void setOptions(Options const& opts);
