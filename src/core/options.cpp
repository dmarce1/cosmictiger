/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>

#include <hpx/modules/program_options.hpp>

#include <cosmictiger/chemistry/primordial_eos.hpp>
#include <cosmictiger/chemistry/recombination.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/diagnostic_output.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/options_metadata.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/core/startup.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/gravity/opening_schedule.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/io/view.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/position.hpp>
#include <cosmictiger/scenarios/scenario.hpp>

#define SHOW(label, opt) show(label, opts.opt)

namespace detail {

static std::vector<std::string> commandLineArguments;
static bool displayOptions = false;

radial_kernel::KernelSpec kernelSpec(std::string const& family, int degree) {
    radial_kernel::Family kind;
    if (family == "spline") {
        kind = radial_kernel::Family::Spline;
    } else if (family == "wendland") {
        kind = radial_kernel::Family::Wendland;
    } else if (family == "ferrers") {
        kind = radial_kernel::Family::Ferrers;
    } else {
        throw std::invalid_argument("kernelFamily must be spline, wendland, or ferrers");
    }
    radial_kernel::KernelSpec const spec{kind, degree};
    if (!radial_kernel::isValid(spec)) {
        throw std::invalid_argument("kernelDegree must be 2..9 for spline, 2/5/8 for "
                                    "wendland, or 2/4/6/8 for ferrers (density polynomial degree)");
    }
    return spec;
}

bool isFiniteOption(double value) {
    // Keep option validation effective when the solver is built with -ffast-math.
    constexpr std::uint64_t kExponentMask = UINT64_C(0x7ff0000000000000);
    return (std::bit_cast<std::uint64_t>(value) & kExponentMask) != kExponentMask;
}

bool isFiniteFloatOption(double value) {
    constexpr double kLargestFloat = std::numeric_limits<float>::max();
    return isFiniteOption(value) && value >= -kLargestFloat && value <= kLargestFloat;
}

bool isPositiveFloatOption(double value) {
    // Device fast math flushes subnormal floats, so require a positive normal.
    return isFiniteFloatOption(value) && value >= std::numeric_limits<float>::min();
}

}    // namespace detail

void show(char const* name, bool opt) {
    if (!detail::displayOptions) {
        return;
    }
    PRINT("%-38s: %c\n", name, opt ? 'T' : 'F');
}

void show(char const* name, int opt) {
    if (!detail::displayOptions) {
        return;
    }
    PRINT("%-38s: %i\n", name, opt);
}

void show(char const* name, double opt) {
    if (!detail::displayOptions) {
        return;
    }
    PRINT("%-38s: %e\n", name, opt);
}

void show(char const* name, std::string opt) {
    if (!detail::displayOptions) {
        return;
    }
    PRINT("%-38s: %s\n", name, opt.c_str());
}

Options globalOpts;

HPX_PLAIN_ACTION(setOptions, SetOptionsAction);

Options const& getOptions() {
    return globalOpts;
}

void setOptions(Options const& opts) {
#if !COSMICTIGER_WITH_SPH
    if (opts.hydro.enabled || opts.chemistry.enabled) {
        throw std::invalid_argument("Hydrodynamics and chemistry require COSMICTIGER_WITH_SPH=ON");
    }
#endif
    if (opts.compute.positionType != (kPositionBits == 64 ? "fixed64" : "fixed32")) {
        throw std::invalid_argument(
            "compute.positionType differs from this locality's application backend");
    }
    if (opts.gravity.method != gravity::kBuildGravityMethod ||
        opts.gravity.pmmmNearMethod != gravity::kBuildPmmmNearMethod ||
        opts.gravity.pmmmOrder != gravity::kBuildFmmOrder) {
        throw std::invalid_argument(
            "Gravity solver, PMMM near method, and expansion order must match this "
            "executable's build configuration");
    }
    if (opts.run.hierarchicalGravity !=
        (integration::kBuildUsesHierarchical && opts.gravity.enabled)) {
        throw std::invalid_argument(
            "Time integrator must match this executable's build configuration");
    }
    setFmmOrder(opts.gravity.fmmOrder);
    terminal::setVerbose(opts.run.verbose);
    std::vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<SetOptionsAction>(c, opts));
    }
    auto const gravitySpec =
        detail::kernelSpec(opts.gravity.kernelFamily, opts.gravity.kernelDegree);
    auto const hydroSpec = detail::kernelSpec(opts.hydro.kernelFamily, opts.hydro.kernelDegree);
#ifdef USE_CUDA
    if (opts.compute.cuda) {
        int count = 0;
        auto const status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess || count == 0) {
            throw std::runtime_error("compute.cuda=true requires an available CUDA device");
        }
        cudaSetDevice();
    }
#endif
    setKernelSelections(gravitySpec, hydroSpec, opts.compute.cuda);
    globalOpts = opts;
    setStackTraceOutputPath(outputPath("stack_trace.txt"));
    for (auto& future : futs) {
        future.get();
    }
}

void recordCommandLineArguments(int argc, char* argv[]) {
    detail::commandLineArguments.clear();
    for (int i = 0; i < argc; i++) {
        detail::commandLineArguments.emplace_back(argv[i]);
    }
}

std::vector<std::string> const& getCommandLineArguments() {
    return detail::commandLineArguments;
}

std::vector<OptionsMetadataEntry> getOptionsMetadata(Options const& options) {
    // Undo storage normalization without undoing current runtime overrides or caps.
    // Public effective radius is 2 sigma in mean box particle spacings.
    auto const gravityTable = radial_kernel::makeKernelTable(
        detail::kernelSpec(options.gravity.kernelFamily, options.gravity.kernelDegree));
    auto const hydroTable = radial_kernel::makeKernelTable(
        detail::kernelSpec(options.hydro.kernelFamily, options.hydro.kernelDegree));
    double const effectiveRadius = options.gravity.supportRadius *
        std::pow(double(options.initial.numberOfParticles), 1.0 / kDimensions) /
        gravityTable.supportRadiusPerEffectiveRadius;
    double const timestepEta = options.gravity.timestepEta;
    return {
        {"build.timeIntegrator", std::string(integration::kBuildTimeIntegrator)},
        {"build.gravityMethod", std::string(gravity::kBuildGravityMethod)},
        {"build.pmmmNearMethod", std::string(gravity::kBuildPmmmNearMethod)},
        {"build.fmmOrder", gravity::kBuildFmmOrder},
        {"configuration", options.configuration},
        {"run.test", options.run.test},
        {"run.verbose", options.run.verbose},
        {"run.scenario", options.run.scenario},
        {"run.endTime", options.run.endTime},
        {"run.maxIterations", options.run.maxIterations},
        {"run.numberOfSteps", options.run.numberOfSteps},
        {"run.minRung", options.run.minRung},
        {"run.treeOpeningAngle", options.run.treeOpeningAngle},
        {"compute.cuda", options.compute.cuda},
        {"compute.positionType", options.compute.positionType},
        {"compute.bucketSize", options.compute.bucketSize},
        {"compute.sphBucketSize", options.compute.sphBucketSize},
        {"compute.cudaKickResidency", options.compute.cudaKickResidency},
        {"compute.cudaKickSinkTile", options.compute.cudaKickSinkTile},
        {"compute.cudaKickParticleTile", options.compute.cudaKickParticleTile},
        {"compute.cudaKickMultipoleTile", options.compute.cudaKickMultipoleTile},
        {"compute.cudaKickSharedExpansion", options.compute.cudaKickSharedExpansion},
        {"compute.cudaKickHandoffBlocks", options.compute.cudaKickHandoffBlocks},
        {"compute.cudaKickFineHandoffBlocks", options.compute.cudaKickFineHandoffBlocks},
        {"compute.cudaKickListReserve", options.compute.cudaKickListReserve},
        {"checkpoint.readIndex", options.checkpoint.readIndex},
        {"checkpoint.intervalSeconds", options.checkpoint.intervalSeconds},
        {"initial.gadget4Restart", options.initial.gadget4Restart},
        {"initial.useGlass", options.initial.useGlass},
        {"initial.createGlass", options.initial.createGlass},
        {"initial.plummer", options.initial.plummer},
        {"initial.plummerRadius", options.initial.plummerRadius},
        {"initial.closePack", options.initial.closePack},
        {"initial.twoLpt", options.initial.twoLpt},
        {"initial.particlesPerDimension", options.initial.particlesPerDimension},
        {"output.silo", options.output.silo},
        {"output.siloInterval", options.output.siloInterval},
        {"output.directory", options.output.directory},
        {"output.units", options.output.units},
        {"output.frame", options.output.frame},
        {"output.xMin", options.output.xMin},
        {"output.xMax", options.output.xMax},
        {"output.yMin", options.output.yMin},
        {"output.yMax", options.output.yMax},
        {"output.zMin", options.output.zMin},
        {"output.zMax", options.output.zMax},
        {"analysis.power", options.analysis.power},
        {"analysis.groups", options.analysis.groups},
        {"analysis.tracers", options.analysis.tracers},
        {"analysis.tracerCount", options.analysis.tracerCount},
        {"analysis.slice", options.analysis.slice},
        {"analysis.views", options.analysis.views},
        {"analysis.viewSize", options.analysis.viewSize},
        {"analysis.sliceResolution", options.analysis.sliceResolution},
        {"analysis.saveForce", options.analysis.saveForce},
        {"analysis.saveAcceleration", options.analysis.saveAcceleration},
        {"analysis.savePotential", options.analysis.savePotential},
        {"lightcone.enabled", options.lightcone.enabled},
        {"lightcone.directory", options.lightcone.directory},
        {"lightcone.linkingLength", options.lightcone.linkingLength},
        {"lightcone.mapSize", options.lightcone.mapSize},
        {"gravity.enabled", options.gravity.enabled},
        {"gravity.method", options.gravity.method},
        {"gravity.meshSize", options.gravity.meshSize},
        {"gravity.pmmmNearWidth", options.gravity.pmmmNearWidth},
        {"gravity.pmmmNearMethod", options.gravity.pmmmNearMethod},
        {"gravity.pmmmBatchParticles", options.gravity.pmmmBatchParticles},
        {"gravity.pmmmOrder", options.gravity.pmmmOrder},
        {"gravity.fmmOrder", options.gravity.fmmOrder},
        {"gravity.kernelFamily", options.gravity.kernelFamily},
        {"gravity.kernelDegree", options.gravity.kernelDegree},
        {"gravity.openingAngle", options.gravity.openingAngle},
        {"gravity.openingSchedule", options.gravity.openingSchedule},
        {"gravity.openLargerCell", options.gravity.openLargerCell},
        {"gravity.softening", effectiveRadius},
        {"gravity.effectiveRadius", effectiveRadius},
        {"gravity.supportRadius",
            options.gravity.supportRadius *
                std::pow(double(options.initial.numberOfParticles), 1.0 / kDimensions)},
        {"gravity.timestepEta", timestepEta},
        {"cosmology.startRedshift", options.cosmology.startRedshift},
        {"cosmology.endRedshift", options.cosmology.endRedshift},
        {"cosmology.omegaLambda", options.cosmology.omegaLambda},
        {"cosmology.omegaB", options.cosmology.omegaB},
        {"cosmology.omegaC", options.cosmology.omegaC},
        {"cosmology.nEff", options.cosmology.nEff},
        {"cosmology.thetaCmb", options.cosmology.thetaCmb},
        {"cosmology.heliumFraction", options.cosmology.heliumFraction},
        {"cosmology.sigma8", options.cosmology.sigma8},
        {"cosmology.hubble", options.cosmology.hubble},
        {"cosmology.spectralIndex", options.cosmology.spectralIndex},
        {"units.massSolar", options.units.massG / constants::kM0},
        {"chemistry.enabled", options.chemistry.enabled},
        {"chemistry.network", options.chemistry.network},
        {"chemistry.initialization", options.chemistry.initialization},
        {"chemistry.radiationFile", options.chemistry.radiationFile},
        {"chemistry.synchronizedRung", options.chemistry.synchronizedRung},
        {"chemistry.initialTemperature", options.chemistry.initialTemperature},
        {"chemistry.hydrogenIonFraction", options.chemistry.hydrogenIonFraction},
        {"chemistry.heliumIonFraction", options.chemistry.heliumIonFraction},
        {"chemistry.heliumDoubleIonFraction", options.chemistry.heliumDoubleIonFraction},
        {"chemistry.hydrogenMinusFraction", options.chemistry.hydrogenMinusFraction},
        {"chemistry.molecularFraction", options.chemistry.molecularFraction},
        {"chemistry.molecularIonFraction", options.chemistry.molecularIonFraction},
        {"chemistry.relativeTolerance", options.chemistry.relativeTolerance},
        {"chemistry.abundanceAbsoluteTolerance", options.chemistry.abundanceAbsoluteTolerance},
        {"chemistry.energyAbsoluteTolerance", options.chemistry.energyAbsoluteTolerance},
        {"chemistry.maxSubsteps", options.chemistry.maxSubsteps},
        {"hydro.enabled", options.hydro.enabled},
        {"hydro.kernelFamily", options.hydro.kernelFamily},
        {"hydro.kernelDegree", options.hydro.kernelDegree},
        {"hydro.yAcceleration", options.hydro.yAcceleration},
        {"hydro.zPlaneAcceleration", options.hydro.zPlaneAcceleration},
        {"hydro.viscosityAlpha", options.hydro.viscosityAlpha},
        {"hydro.viscosityBeta", options.hydro.viscosityBeta},
        {"hydro.viscosityEpsilon", options.hydro.viscosityEpsilon},
        {"hydro.balsaraSwitch", options.hydro.balsaraSwitch},
        {"hydro.viscosityEvolution", options.hydro.viscosityEvolution},
        {"hydro.viscosityAlphaMin", options.hydro.viscosityAlphaMin},
        {"hydro.viscosityDecay", options.hydro.viscosityDecay},
        {"hydro.viscosityAlphaMax", options.hydro.viscosityAlphaMax},
        {"hydro.viscosityDecayCD", options.hydro.viscosityDecayCD},
        {"hydro.viscosityBetaMode", options.hydro.viscosityBetaMode},
        {"hydro.viscosityBetaOverAlpha", options.hydro.viscosityBetaOverAlpha},
        {"hydro.gamma", options.hydro.gamma},
        {"hydro.centralAcceleration", options.hydro.centralAcceleration},
        {"hydro.centralSoftening",
            options.hydro.centralSupportRadius / gravityTable.supportRadiusPerEffectiveRadius},
        {"hydro.centralEffectiveRadius",
            options.hydro.centralSupportRadius / gravityTable.supportRadiusPerEffectiveRadius},
        {"hydro.centralSupportRadius", options.hydro.centralSupportRadius},
        {"hydro.minEffectiveRadius",
            options.hydro.minSupportRadius * options.initial.particlesPerDimension /
                hydroTable.supportRadiusPerEffectiveRadius},
        {"hydro.maxEffectiveRadius",
            options.hydro.maxSupportRadius * options.initial.particlesPerDimension /
                hydroTable.supportRadiusPerEffectiveRadius},
        {"hydro.minSupportRadius",
            options.hydro.minSupportRadius * options.initial.particlesPerDimension},
        {"hydro.maxSupportRadius",
            options.hydro.maxSupportRadius * options.initial.particlesPerDimension},
        {"hydro.neighborNumber", options.hydro.neighborNumber},
        {"hydro.cfl", options.hydro.cfl},
    };
}

bool processOptions(int argc, char* argv[]) {
    Options opts{};
    double gravityEffectiveRadius;
    double centralEffectiveRadius;
    std::string omegaLambdaInput;
    namespace po = hpx::program_options;
    bool rc;
    po::options_description runOpts("Run control");
    runOpts.add_options()("help", "produce help message")("configuration",
        po::value<std::string>(&opts.configuration)->default_value(""),
        "INI configuration file; explicit command-line values take precedence")("run.test",
        po::value<std::string>(&opts.run.test)->default_value(""),
        "name of developer regression to run (requires BUILD_TESTING=ON)")("run.verbose",
        po::value<bool>(&opts.run.verbose)->default_value(false),
        "print all resolved options and detailed internal diagnostics")("run.scenario",
        po::value<std::string>(&opts.run.scenario)->default_value("cosmology"),
        "physical scenario: plummer, sod, blast, rayleighTaylor, kelvinHelmholtz, disc, "
        "polytropicStar, hydroWave, cosmology, glass")("run.endTime",
        po::value<double>(&opts.run.endTime)->default_value(0.0),
        "total evolution time (conformal code time for cosmology); positive values override "
        "endRedshift, 0 uses the scenario duration")("run.maxIterations",
        po::value<int>(&opts.run.maxIterations)->default_value(1000000),
        "maximum number of time steps")("run.numberOfSteps",
        po::value<int>(&opts.run.numberOfSteps)->default_value(64),
        "number of super time steps")("run.minRung",
        po::value<int>(&opts.run.minRung)->default_value(0), "minimum starting rung")(
        "run.treeOpeningAngle", po::value<double>(&opts.run.treeOpeningAngle)->default_value(0.0),
        "run.test=force: build the tree once at this opening angle and sweep only the kick's "
        "angle upward from it; 0 rebuilds the tree at every angle");

    po::options_description computeOpts("Compute");
    computeOpts.add_options()("compute.positionType",
        po::value<std::string>(&opts.compute.positionType)
            ->default_value(kPositionBits == 64 ? "fixed64" : "fixed32"),
        "position storage: fixed32 (Q0.32) or fixed64 (Q0.64)")
#ifdef USE_CUDA
        ("compute.cuda", po::value<bool>(&opts.compute.cuda)->default_value(true), "use CUDA")
#else
        ("compute.cuda", po::value<bool>(&opts.compute.cuda)->default_value(false),
            "use CUDA (not enabled for this build)")
#endif
            ("compute.bucketSize", po::value<int>(&opts.compute.bucketSize)->default_value(128),
                "gravity tree leaf bucket size")("compute.sphBucketSize",
                po::value<int>(&opts.compute.sphBucketSize)->default_value(0),
                "SPH tree leaf bucket size; 0 uses floor(8/pi * hydro.neighborNumber)")(
                "compute.cudaKickResidency",
                po::value<int>(&opts.compute.cudaKickResidency)->default_value(0),
                "resident CUDA kick blocks per multiprocessor; 0 uses the register limit")(
                "compute.cudaKickSinkTile",
                po::value<int>(&opts.compute.cudaKickSinkTile)->default_value(0),
                "CUDA kick shared sink-force tile; 0 automatic, -1 global force storage")(
                "compute.cudaKickParticleTile",
                po::value<int>(&opts.compute.cudaKickParticleTile)->default_value(0),
                "CUDA kick staged particle sources per tile; 0 automatic")(
                "compute.cudaKickMultipoleTile",
                po::value<int>(&opts.compute.cudaKickMultipoleTile)->default_value(0),
                "CUDA kick staged multipoles per tile; 0 automatic")(
                "compute.cudaKickSharedExpansion",
                po::value<bool>(&opts.compute.cudaKickSharedExpansion)->default_value(true),
                "keep the active CUDA kick local expansion in shared memory")(
                "compute.cudaKickHandoffBlocks",
                po::value<int>(&opts.compute.cudaKickHandoffBlocks)->default_value(0),
                "CUDA block count dividing the particles for the CPU/GPU handoff threshold; 0 "
                "derives it from the particle count (two blocks per multiprocessor at 32768 "
                "particles, growing as the cube root to six at 884736 and above)")(
                "compute.cudaKickFineHandoffBlocks",
                po::value<int>(&opts.compute.cudaKickFineHandoffBlocks)->default_value(0),
                "CUDA block count dividing the particles for the handoff of a node that carries "
                "no Ewald checks (nodes that still carry them keep the threshold of "
                "compute.cudaKickHandoffBlocks); 0, or a count below that divisor, applies that "
                "divisor to every node")("compute.cudaKickListReserve",
                po::value<int>(&opts.compute.cudaKickListReserve)->default_value(4096),
                "entries per CUDA kick interaction list in the per-launch slab handed to each "
                "block (pool growth only on overflow); 0 grows every list from the pool");

    po::options_description checkpointOpts("Checkpoints");
    checkpointOpts.add_options()("checkpoint.readIndex",
        po::value<int>(&opts.checkpoint.readIndex)->default_value(-1),
        "checkpoint index to restart; -1 starts a new run")("checkpoint.intervalSeconds",
        po::value<int>(&opts.checkpoint.intervalSeconds)->default_value(1000000000),
        "elapsed wall time between checkpoints, in seconds");

    po::options_description initialOpts("Initial conditions");
    initialOpts.add_options()("initial.gadget4Restart",
        po::value<std::string>(&opts.initial.gadget4Restart)->default_value(""),
        "Gadget4 input file for power-spectrum analysis")("initial.useGlass",
        po::value<bool>(&opts.initial.useGlass)->default_value(false),
        "initialize from a glass file")("initial.createGlass",
        po::value<bool>(&opts.initial.createGlass)->default_value(false), "create a glass file")(
        "initial.plummer", po::value<bool>(&opts.initial.plummer)->default_value(false),
        "initialize the dark-matter Plummer model")("initial.plummerRadius",
        po::value<double>(&opts.initial.plummerRadius)->default_value(1.0e-2), "Plummer radius")(
        "initial.closePack", po::value<bool>(&opts.initial.closePack)->default_value(false),
        "use a close-packed particle lattice")("initial.twoLpt",
        po::value<bool>(&opts.initial.twoLpt)->default_value(false),
        "use second-order Lagrangian initial conditions")("initial.particlesPerDimension",
        po::value<int>(&opts.initial.particlesPerDimension)->default_value(128),
        "particle lattice size along each dimension, per species");

    po::options_description outputOpts("Particle output");
    outputOpts.add_options()("output.silo",
        po::value<bool>(&opts.output.silo)->default_value(false),
        "write particle Silo files for the configured box")("output.siloInterval",
        po::value<int>(&opts.output.siloInterval)->default_value(1),
        "number of synchronized coarse steps between particle Silo snapshots")("output.directory",
        po::value<std::string>(&opts.output.directory)->default_value("./output"),
        "directory for all scenario outputs; relative paths start at the working directory")(
        "output.units", po::value<std::string>(&opts.output.units)->default_value("code"),
        "particle Silo units: code, cgs, cosmological, or both (cgs and cosmological)")(
        "output.frame", po::value<std::string>(&opts.output.frame)->default_value("comoving"),
        "particle Silo frame: comoving, lab, or both; converted velocities are peculiar")(
        "output.xMin", po::value<double>(&opts.output.xMin)->default_value(0.0),
        "minimum x coordinate of the Silo box, in [0, 1]")("output.xMax",
        po::value<double>(&opts.output.xMax)->default_value(1.0),
        "maximum x coordinate of the Silo box, in [0, 1]")("output.yMin",
        po::value<double>(&opts.output.yMin)->default_value(0.0),
        "minimum y coordinate of the Silo box, in [0, 1]")("output.yMax",
        po::value<double>(&opts.output.yMax)->default_value(1.0),
        "maximum y coordinate of the Silo box, in [0, 1]")("output.zMin",
        po::value<double>(&opts.output.zMin)->default_value(0.0),
        "minimum z coordinate of the Silo box, in [0, 1]")("output.zMax",
        po::value<double>(&opts.output.zMax)->default_value(1.0),
        "maximum z coordinate of the Silo box, in [0, 1]");

    po::options_description analysisOpts("Analysis");
    analysisOpts.add_options()("analysis.power",
        po::value<bool>(&opts.analysis.power)->default_value(false),
        "compute the mass power spectrum")("analysis.groups",
        po::value<bool>(&opts.analysis.groups)->default_value(false), "find particle groups")(
        "analysis.tracers", po::value<bool>(&opts.analysis.tracers)->default_value(false),
        "write tracer particles to Silo")("analysis.tracerCount",
        po::value<int>(&opts.analysis.tracerCount)->default_value(1000000),
        "number of tracer particles")("analysis.slice",
        po::value<bool>(&opts.analysis.slice)->default_value(false),
        "write a projected slice through the volume")("analysis.views",
        po::value<bool>(&opts.analysis.views)->default_value(false),
        "write instantaneous HEALPix maps")("analysis.viewSize",
        po::value<int>(&opts.analysis.viewSize)->default_value(1024),
        "view HEALPix Nside")("analysis.sliceResolution",
        po::value<int>(&opts.analysis.sliceResolution)->default_value(4096), "slice resolution")(
        "analysis.saveForce", po::value<bool>(&opts.analysis.saveForce)->default_value(false),
        "retain both per-particle acceleration and potential diagnostics")(
        "analysis.saveAcceleration",
        po::value<bool>(&opts.analysis.saveAcceleration)->default_value(false),
        "retain per-particle acceleration diagnostics")("analysis.savePotential",
        po::value<bool>(&opts.analysis.savePotential)->default_value(false),
        "retain per-particle potential diagnostics");

    po::options_description lightconeOpts("Lightcone analysis");
    lightconeOpts.add_options()("lightcone.enabled",
        po::value<bool>(&opts.lightcone.enabled)->default_value(false),
        "enable lightcone analysis")("lightcone.directory",
        po::value<std::string>(&opts.lightcone.directory)->default_value("./lightcone"),
        "directory for lightcone output")("lightcone.linkingLength",
        po::value<double>(&opts.lightcone.linkingLength)->default_value(0.28),
        "lightcone group-finder linking length")("lightcone.mapSize",
        po::value<int>(&opts.lightcone.mapSize)->default_value(-1),
        "lightcone HEALPix Nside; -1 chooses automatically");

    po::options_description gravityOpts("Gravity");
    gravityOpts.add_options()("gravity.enabled",
        po::value<bool>(&opts.gravity.enabled)->default_value(true), "enable gravity")(
        "gravity.meshSize", po::value<int>(&opts.gravity.meshSize)->default_value(0),
        "Full-particle PMMM cells per axis; required for PMMM builds")("gravity.pmmmBatchParticles",
        po::value<int>(&opts.gravity.pmmmBatchParticles)->default_value(4096),
        "target sink particles per ready PMMM batch; oversized cells run alone")(
        "gravity.pmmmNearWidth", po::value<int>(&opts.gravity.pmmmNearWidth)->default_value(2),
        "PMMM neighbor cube radius in cells: 1 through 4")("gravity.openingAngle",
        po::value<double>(&opts.gravity.openingAngle)->default_value(0.8),
        "global FMM opening angle; PMMM derives sqrt(3)/(pmmmNearWidth+1)")(
        "gravity.openingSchedule",
        po::value<std::string>(&opts.gravity.openingSchedule)->default_value(""),
        "global cosmological FMM z:theta pairs, comma separated; none selects a constant angle")(
        "gravity.openLargerCell",
        po::value<bool>(&opts.gravity.openLargerCell)->default_value(true),
        "tree walk opens only the larger of two rejected cells instead of both")(
        "gravity.softening",
        po::value<double>(&gravityEffectiveRadius)->default_value(0.01876988456953534),
        "legacy alias for gravity.effectiveRadius")("gravity.effectiveRadius", po::value<double>(),
        "fixed comoving effective radius (2 sigma) in initial reference spacings; overrides "
        "supportRadius")("gravity.supportRadius", po::value<double>(),
        "compact support radius in initial reference spacings; ignored when effectiveRadius is "
        "specified")("gravity.kernelFamily",
        po::value<std::string>(&opts.gravity.kernelFamily)->default_value("ferrers"),
        "density kernel family: spline, wendland, or ferrers")("gravity.kernelDegree",
        po::value<int>(&opts.gravity.kernelDegree)->default_value(4),
        "density degree: spline 2..9; wendland 2,5,8; ferrers 2,4,6,8")("gravity.timestepEta",
        po::value<double>(&opts.gravity.timestepEta)->default_value(0.20401599335954483),
        "prefactor C in dt = C sqrt(effectiveRadius/acceleration), effectiveRadius = 2 sigma");

    po::options_description cosmologyOpts("Cosmology");
    cosmologyOpts.add_options()("cosmology.startRedshift",
        po::value<double>(&opts.cosmology.startRedshift)->default_value(49.0), "starting redshift")(
        "cosmology.endRedshift", po::value<double>(&opts.cosmology.endRedshift)->default_value(0.0),
        "ending redshift on the expanding branch; ignored when run.endTime is positive")(
        "cosmology.omegaLambda", po::value<std::string>(&omegaLambdaInput)->default_value("auto"),
        "signed dark-energy density parameter; auto derives a spatially flat universe")(
        "cosmology.omegaB", po::value<double>(&opts.cosmology.omegaB)->default_value(0.049389),
        "baryon density parameter")("cosmology.omegaC",
        po::value<double>(&opts.cosmology.omegaC)->default_value(0.26503),
        "cold-dark-matter density parameter")("cosmology.nEff",
        po::value<double>(&opts.cosmology.nEff)->default_value(3.046),
        "effective number of neutrino species")("cosmology.thetaCmb",
        po::value<double>(&opts.cosmology.thetaCmb)->default_value(2.7255 / 2.73),
        "CMB temperature divided by 2.73 K")("cosmology.heliumFraction",
        po::value<double>(&opts.cosmology.heliumFraction)->default_value(0.2454006),
        "primordial helium mass fraction")("cosmology.sigma8",
        po::value<double>(&opts.cosmology.sigma8)->default_value(0.8607),
        "power-spectrum normalization; 0 initializes a homogeneous lattice")("cosmology.hubble",
        po::value<double>(&opts.cosmology.hubble)->default_value(0.6732),
        "dimensionless Hubble parameter")("cosmology.spectralIndex",
        po::value<double>(&opts.cosmology.spectralIndex)->default_value(0.96605),
        "primordial spectral index");

    po::options_description unitsOpts("Units");
    unitsOpts.add_options()("units.massSolar",
        po::value<double>(&opts.units.massG)->default_value(1.e9 / .6732),
        "particle mass unit in solar masses");

    po::options_description hydroOpts("Monaghan (1992) SPH");
    hydroOpts.add_options()("hydro.enabled",
        po::value<bool>(&opts.hydro.enabled)->default_value(false),
        "use density/internal-energy SPH")("hydro.kernelFamily",
        po::value<std::string>(&opts.hydro.kernelFamily)->default_value("spline"),
        "density kernel family: spline, wendland, or ferrers")("hydro.kernelDegree",
        po::value<int>(&opts.hydro.kernelDegree)->default_value(3),
        "density degree: spline 2..9; wendland 2,5,8; ferrers 2,4,6,8")("hydro.yAcceleration",
        po::value<double>(&opts.hydro.yAcceleration)->default_value(0.0),
        "uniform gravitational acceleration in the y direction")("hydro.zPlaneAcceleration",
        po::value<double>(&opts.hydro.zPlaneAcceleration)->default_value(0.0),
        "nonnegative acceleration magnitude toward the plane z=0.5")("hydro.viscosityAlpha",
        po::value<double>(&opts.hydro.viscosityAlpha)->default_value(1.0),
        "linear coefficient alpha when viscosityEvolution=none")("hydro.viscosityBeta",
        po::value<double>(&opts.hydro.viscosityBeta)->default_value(2.0),
        "fixed quadratic coefficient beta when viscosityBetaMode=constant")(
        "hydro.viscosityEpsilon",
        po::value<double>(&opts.hydro.viscosityEpsilon)->default_value(0.01),
        "viscosity regularization relative to the calibrated radius sqrt(5/6) effectiveRadius")(
        "hydro.balsaraSwitch", po::value<bool>(&opts.hydro.balsaraSwitch)->default_value(true),
        "suppress artificial viscosity in shear flows with the Balsara switch")(
        "hydro.viscosityEvolution",
        po::value<std::string>(&opts.hydro.viscosityEvolution)->default_value("morrisAndMonaghan"),
        "viscosity evolution: none, morrisAndMonaghan, or cullenAndDehnen")(
        "hydro.viscosityAlphaMin",
        po::value<double>(&opts.hydro.viscosityAlphaMin)->default_value(0.1),
        "Morris-Monaghan alpha floor and smooth-flow initial value")("hydro.viscosityDecay",
        po::value<double>(&opts.hydro.viscosityDecay)->default_value(0.2),
        "Morris-Monaghan C1 in tau=h/(C1 c)")("hydro.viscosityAlphaMax",
        po::value<double>(&opts.hydro.viscosityAlphaMax)->default_value(2.0),
        "maximum alpha for Cullen-Dehnen viscosity evolution")("hydro.viscosityDecayCD",
        po::value<double>(&opts.hydro.viscosityDecayCD)->default_value(0.05),
        "Cullen-Dehnen decay coefficient ell in tau=H/(2 ell vSignal)")("hydro.viscosityBetaMode",
        po::value<std::string>(&opts.hydro.viscosityBetaMode)->default_value("ratio"),
        "quadratic coefficient: ratio follows alpha, constant uses viscosityBeta")(
        "hydro.viscosityBetaOverAlpha",
        po::value<double>(&opts.hydro.viscosityBetaOverAlpha)->default_value(2.0),
        "beta/alpha ratio when viscosityBetaMode=ratio")("hydro.gamma",
        po::value<double>(&opts.hydro.gamma)->default_value(5.0 / 3.0),
        "ideal-gas adiabatic index")("hydro.centralAcceleration",
        po::value<double>(&opts.hydro.centralAcceleration)->default_value(0.0),
        "magnitude of the central force")("hydro.centralSoftening",
        po::value<double>(&centralEffectiveRadius)->default_value(0.005477225575051661),
        "legacy alias for hydro.centralEffectiveRadius")("hydro.centralEffectiveRadius",
        po::value<double>(),
        "central-force effective radius (2 sigma) in box units; overrides centralSupportRadius")(
        "hydro.centralSupportRadius", po::value<double>(),
        "central-force support radius in box units; uses the selected gravity kernel")(
        "hydro.minSupportRadius",
        po::value<double>(&opts.hydro.minSupportRadius)->default_value(1.0e-6),
        "minimum support radius in interparticle spacings")("hydro.minEffectiveRadius",
        po::value<double>(),
        "minimum effective radius (2 sigma) in interparticle spacings; overrides "
        "minSupportRadius")("hydro.maxSupportRadius",
        po::value<double>(&opts.hydro.maxSupportRadius)->default_value(1000.0),
        "maximum kernel support radius in interparticle spacings; capped at half the box")(
        "hydro.maxEffectiveRadius", po::value<double>(),
        "maximum effective radius (2 sigma) in interparticle spacings; overrides "
        "maxSupportRadius")("hydro.neighborNumber",
        po::value<double>(&opts.hydro.neighborNumber)->default_value(42),
        "kernel-weighted neighbor target within the support sphere, including self")("hydro.cfl",
        po::value<double>(&opts.hydro.cfl)->default_value(0.22821773229381923),
        "Courant time-step factor using effective radius (2 sigma)");

    po::options_description chemistryOpts("Primordial chemistry");
    chemistryOpts.add_options()("chemistry.enabled",
        po::value<bool>(&opts.chemistry.enabled)->default_value(false),
        "evolve primordial thermochemistry (requires hydro.enabled)")("chemistry.network",
        po::value<std::string>(&opts.chemistry.network)->default_value("atomic6"),
        "atomic6 or molecular9; optically thin low-density gas")("chemistry.initialization",
        po::value<std::string>(&opts.chemistry.initialization)->default_value("recombination"),
        "recombination fills unspecified initial T/H+/He+/He++; manual uses fixed defaults")(
        "chemistry.radiationFile",
        po::value<std::string>(&opts.chemistry.radiationFile)->default_value(""),
        "optional versioned redshift/photo-rate table; empty means no imposed UV")(
        "chemistry.synchronizedRung",
        po::value<int>(&opts.chemistry.synchronizedRung)->default_value(4),
        "fixed common timestep rung; fail if hydro/gravity requires a finer step")(
        "chemistry.initialTemperature",
        po::value<double>(&opts.chemistry.initialTemperature)->default_value(1000.0),
        "initial gas temperature in kelvin")("chemistry.hydrogenIonFraction",
        po::value<double>(&opts.chemistry.hydrogenIonFraction)->default_value(2.0e-4),
        "initial H+ fraction per H nucleus")("chemistry.heliumIonFraction",
        po::value<double>(&opts.chemistry.heliumIonFraction)->default_value(0.0),
        "initial He+ fraction per He nucleus")("chemistry.heliumDoubleIonFraction",
        po::value<double>(&opts.chemistry.heliumDoubleIonFraction)->default_value(0.0),
        "initial He++ fraction per He nucleus")("chemistry.hydrogenMinusFraction",
        po::value<double>(&opts.chemistry.hydrogenMinusFraction)->default_value(0.0),
        "initial H- fraction per H nucleus")("chemistry.molecularFraction",
        po::value<double>(&opts.chemistry.molecularFraction)->default_value(0.0),
        "initial fraction of H nuclei in H2")("chemistry.molecularIonFraction",
        po::value<double>(&opts.chemistry.molecularIonFraction)->default_value(0.0),
        "initial fraction of H nuclei in H2+")("chemistry.relativeTolerance",
        po::value<double>(&opts.chemistry.relativeTolerance)->default_value(1.0e-4),
        "local relative source tolerance (minimum 1e-5); not a global error guarantee")(
        "chemistry.abundanceAbsoluteTolerance",
        po::value<double>(&opts.chemistry.abundanceAbsoluteTolerance)->default_value(1.0e-12),
        "absolute fractional-nuclei tolerance")("chemistry.energyAbsoluteTolerance",
        po::value<double>(&opts.chemistry.energyAbsoluteTolerance)->default_value(1.0e-8),
        "absolute thermal-energy tolerance, eV per H nucleus")("chemistry.maxSubsteps",
        po::value<int>(&opts.chemistry.maxSubsteps)->default_value(4096),
        "maximum accepted source substeps per half step");

    po::options_description commandOpts("CosmicTiger options");
    commandOpts.add(runOpts)
        .add(computeOpts)
        .add(checkpointOpts)
        .add(initialOpts)
        .add(outputOpts)
        .add(analysisOpts)
        .add(lightconeOpts)
        .add(gravityOpts)
        .add(cosmologyOpts)
        .add(unitsOpts)
        .add(hydroOpts)
        .add(chemistryOpts);
    terminal::detailPrint("Processing options\n");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, commandOpts), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << "Build configuration (fixed at compile time):\n"
                  << "  Time integrator: " << integration::kBuildTimeIntegrator << "\n"
                  << "  Gravity solver: " << gravity::kBuildGravityMethod << "\n"
                  << "  PMMM near method: " << gravity::kBuildPmmmNearMethod << "\n"
                  << "  FMM/PMMM expansion order: " << gravity::kBuildFmmOrder << "\n\n";
        std::cout << commandOpts << "\n";
        rc = false;
    } else {
        if (!opts.configuration.empty()) {
            std::ifstream cfgFs{opts.configuration};
            if (cfgFs) {
                po::store(po::parse_config_file(cfgFs, commandOpts), vm);
                rc = true;
            } else {
                THROW_ERROR("Cannot open configuration file %s\n", opts.configuration.c_str());
            }
        } else {
            rc = true;
        }
    }

    if (rc) {
        po::notify(vm);
    }
    if (!rc) {
        return false;
    }
#ifndef COSMICTIGER_BUILD_TESTING
    if (!opts.run.test.empty()) {
        THROW_ERROR("run.test requires a build configured with BUILD_TESTING=ON\n");
    }
#endif
    terminal::setVerbose(opts.run.verbose);
    auto const isExplicit = [&](char const* key) { return vm.count(key) && !vm[key].defaulted(); };
    if (opts.gravity.meshSize < 0) {
        THROW_ERROR("gravity.meshSize must be nonnegative\n");
    }
    if (opts.gravity.pmmmBatchParticles <= 0) {
        THROW_ERROR("gravity.pmmmBatchParticles must be positive\n");
    }
    if (opts.gravity.pmmmNearWidth < 1 || opts.gravity.pmmmNearWidth > 4) {
        THROW_ERROR("gravity.pmmmNearWidth must be between 1 and 4\n");
    }
    if (usesMeshGravity(opts.gravity)) {
        opts.gravity.openingAngle = std::sqrt(3.0) / (opts.gravity.pmmmNearWidth + 1);
        if (opts.gravity.meshSize < 2 * opts.gravity.pmmmNearWidth + 1) {
            THROW_ERROR("gravity.meshSize must be at least 2*gravity.pmmmNearWidth+1 "
                        "for a PMMM build\n");
        }
        size_t const meshSize = static_cast<size_t>(opts.gravity.meshSize);
        size_t const maximumCells = std::min(size_t(std::numeric_limits<ptrdiff_t>::max()),
            std::numeric_limits<size_t>::max() / (8 * sizeof(float)));
        if (meshSize > maximumCells / meshSize / meshSize) {
            THROW_ERROR("gravity.meshSize cubed exceeds mesh indexing/allocation limits\n");
        }
    }
    if (opts.initial.plummer && opts.initial.createGlass) {
        THROW_ERROR("initial.plummer and initial.createGlass cannot both be enabled\n");
    }
    if (opts.initial.plummer || opts.initial.createGlass) {
        std::string const legacy = opts.initial.plummer ? "plummer" : "glass";
        if (isExplicit("run.scenario") && opts.run.scenario != legacy) {
            THROW_ERROR("initial.plummer/createGlass conflicts with run.scenario\n");
        }
        opts.run.scenario = legacy;
    }
    auto const& scenario = scenarios::definition(opts.run.scenario);
    opts.initial.plummer = opts.run.scenario == "plummer";
    opts.initial.createGlass = opts.run.scenario == "glass";
    if (opts.run.test.empty()) {
        if (!isExplicit("gravity.enabled")) {
            opts.gravity.enabled = scenario.gravity;
        }
        if (!isExplicit("hydro.enabled")) {
            opts.hydro.enabled = scenario.hydro;
        }
        if (scenario.fixedTime && !opts.hydro.enabled) {
            THROW_ERROR("run.scenario=%s requires hydro.enabled=true\n", scenario.name);
        }
        if (!opts.gravity.enabled && !opts.hydro.enabled) {
            THROW_ERROR("A scenario requires gravity.enabled and/or hydro.enabled\n");
        }
        if (!isExplicit("run.numberOfSteps")) {
            opts.run.numberOfSteps = scenario.numberOfSteps;
        }
        if (!isExplicit("output.directory")) {
            opts.output.directory = "./output/" + opts.run.scenario;
        }
    }
    auto const openingSchedule = gravity::parseOpeningSchedule(opts.gravity.openingSchedule);
    if (!openingSchedule.empty() &&
        (usesMeshGravity(opts.gravity) || opts.run.scenario != "cosmology")) {
        THROW_ERROR("gravity.openingSchedule requires an FMM build and "
                    "run.scenario=cosmology\n");
    }
    bool const automaticLambda = omegaLambdaInput == "auto";
    if (!automaticLambda) {
        try {
            size_t consumed = 0;
            opts.cosmology.omegaLambda = std::stod(omegaLambdaInput, &consumed);
            if (consumed != omegaLambdaInput.size()) {
                throw std::invalid_argument("trailing input");
            }
        } catch (std::exception const&) {
            THROW_ERROR("cosmology.omegaLambda must be a finite signed number or auto\n");
        }
    }
    if (!isExplicit("lightcone.directory")) {
        opts.lightcone.directory = opts.output.directory + "/lightcone";
    }
    if (!detail::isFiniteOption(opts.run.endTime) || opts.run.endTime < 0.0 ||
        opts.run.maxIterations < 0) {
        THROW_ERROR("run.endTime and run.maxIterations must be finite and nonnegative\n");
    }
    if (scenario.fixedTime && !opts.run.test.empty() && opts.run.test != "checkpointState") {
        THROW_ERROR("run.scenario and run.test select separate execution modes\n");
    }
    auto const gravityTable = radial_kernel::makeKernelTable(
        detail::kernelSpec(opts.gravity.kernelFamily, opts.gravity.kernelDegree));
    auto const hydroTable = radial_kernel::makeKernelTable(
        detail::kernelSpec(opts.hydro.kernelFamily, opts.hydro.kernelDegree));
    // Resolve representations after both CLI and INI parsing. Only explicit
    // effective radii override support radii; defaulted legacy aliases do not.
    auto const resolveSupportRadius = [&](char const* effectiveKey, char const* supportKey,
                                          char const* legacyKey, double defaultSupportRadius,
                                          double supportRadiusPerEffectiveRadius) {
        auto const explicitOption = [&](char const* key) {
            return key != nullptr && vm.count(key) && !vm[key].defaulted();
        };
        char const* selectedEffectiveKey = explicitOption(effectiveKey) ? effectiveKey :
            explicitOption(legacyKey)                                   ? legacyKey :
                                                                          nullptr;
        double supportRadius = defaultSupportRadius;
        if (selectedEffectiveKey != nullptr) {
            if (selectedEffectiveKey == effectiveKey && explicitOption(legacyKey)) {
                PRINT("Warning: %s is ignored because %s is specified.\n", legacyKey, effectiveKey);
            }
            if (explicitOption(supportKey)) {
                PRINT("Warning: %s is ignored because %s is specified.\n", supportKey,
                    selectedEffectiveKey);
            }
            supportRadius = vm[selectedEffectiveKey].as<double>() * supportRadiusPerEffectiveRadius;
        } else if (explicitOption(supportKey)) {
            supportRadius = vm[supportKey].as<double>();
        }
        if (!detail::isFiniteOption(supportRadius) || supportRadius <= 0.0) {
            THROW_ERROR("%s / %s%s%s must give a finite positive support radius\n", effectiveKey,
                supportKey, legacyKey ? " / " : "", legacyKey ? legacyKey : "");
        }
        return supportRadius;
    };
    opts.gravity.supportRadius = resolveSupportRadius("gravity.effectiveRadius",
        "gravity.supportRadius", "gravity.softening",
        gravityEffectiveRadius * gravityTable.supportRadiusPerEffectiveRadius,
        gravityTable.supportRadiusPerEffectiveRadius);
    opts.hydro.centralSupportRadius = resolveSupportRadius("hydro.centralEffectiveRadius",
        "hydro.centralSupportRadius", "hydro.centralSoftening",
        centralEffectiveRadius * gravityTable.supportRadiusPerEffectiveRadius,
        gravityTable.supportRadiusPerEffectiveRadius);
    opts.hydro.minSupportRadius =
        resolveSupportRadius("hydro.minEffectiveRadius", "hydro.minSupportRadius", nullptr,
            opts.hydro.minSupportRadius, hydroTable.supportRadiusPerEffectiveRadius);
    opts.hydro.maxSupportRadius =
        resolveSupportRadius("hydro.maxEffectiveRadius", "hydro.maxSupportRadius", nullptr,
            opts.hydro.maxSupportRadius, hydroTable.supportRadiusPerEffectiveRadius);
    double const selfNeighborFloor =
        4.0 * std::numbers::pi / 3.0 * radial_kernel::value(hydroTable, 0.0);
    if (opts.output.siloInterval <= 0) {
        THROW_ERROR("output.siloInterval must be positive\n");
    }
    if (opts.output.directory.empty()) {
        THROW_ERROR("output.directory must not be empty\n");
    }
    if (opts.output.units != "code" && opts.output.units != "cgs" &&
        opts.output.units != "cosmological" && opts.output.units != "both") {
        THROW_ERROR("output.units must be code, cgs, cosmological, or both\n");
    }
    if (opts.output.frame != "comoving" && opts.output.frame != "lab" &&
        opts.output.frame != "both") {
        THROW_ERROR("output.frame must be comoving, lab, or both\n");
    }
    if (opts.output.units == "code" && opts.output.frame != "comoving") {
        THROW_ERROR("output.units=code requires output.frame=comoving because native fields "
                    "retain their evolution normalization; choose output.units=cgs, "
                    "cosmological, or both for lab-frame output\n");
    }
    double const outputMin[] = {opts.output.xMin, opts.output.yMin, opts.output.zMin};
    double const outputMax[] = {opts.output.xMax, opts.output.yMax, opts.output.zMax};
    char const* const outputAxes[] = {"x", "y", "z"};
    for (int dim = 0; dim < kDimensions; dim++) {
        if (!detail::isFiniteOption(outputMin[dim]) || !detail::isFiniteOption(outputMax[dim]) ||
            outputMin[dim] < 0.0 || outputMax[dim] > 1.0 || outputMin[dim] >= outputMax[dim]) {
            THROW_ERROR("output.%sMin and output.%sMax must be finite and satisfy "
                        "0 <= min < max <= 1\n",
                outputAxes[dim], outputAxes[dim]);
        }
    }
    if (opts.analysis.tracers && opts.analysis.tracerCount <= 0) {
        THROW_ERROR("analysis.tracerCount must be positive when analysis.tracers is enabled\n");
    }
    if (opts.initial.particlesPerDimension <= 0) {
        THROW_ERROR("initial.particlesPerDimension must be positive\n");
    }
    if (opts.run.numberOfSteps <= 0) {
        THROW_ERROR("run.numberOfSteps must be positive\n");
    }
    if (opts.compute.positionType != "fixed32" && opts.compute.positionType != "fixed64") {
        THROW_ERROR("compute.positionType must be fixed32 or fixed64\n");
    }
    if (opts.compute.positionType != (kPositionBits == 64 ? "fixed64" : "fixed32")) {
        THROW_ERROR("compute.positionType differs from the selected application backend\n");
    }
    if (opts.compute.bucketSize <= 0) {
        THROW_ERROR("compute.bucketSize must be positive\n");
    }
    if (opts.compute.sphBucketSize < 0) {
        THROW_ERROR("compute.sphBucketSize must be nonnegative (0 selects automatic sizing)\n");
    }
    if (opts.compute.bucketSize > MAX_BUCKET_SIZE) {
        THROW_ERROR("compute.bucketSize must not exceed the compile-time maximum %i "
                    "(COSMICTIGER_WITH_MAX_BUCKET)\n",
            int(MAX_BUCKET_SIZE));
    }
    if (opts.compute.cudaKickResidency < 0 || opts.compute.cudaKickSinkTile < -1 ||
        opts.compute.cudaKickParticleTile < 0 || opts.compute.cudaKickMultipoleTile < 0 ||
        opts.compute.cudaKickHandoffBlocks < 0 || opts.compute.cudaKickFineHandoffBlocks < 0 ||
        opts.compute.cudaKickListReserve < 0) {
        THROW_ERROR("compute.cudaKick* tile and residency options must be nonnegative "
                    "(cudaKickSinkTile may be -1 for global force storage)\n");
    }
    if (!detail::isFiniteOption(opts.hydro.yAcceleration) ||
        !detail::isFiniteOption(opts.hydro.centralAcceleration) ||
        !detail::isFiniteOption(opts.hydro.centralSupportRadius) ||
        opts.hydro.centralSupportRadius <= 0.0) {
        THROW_ERROR("hydro.yAcceleration and hydro.centralAcceleration must be finite; "
                    "hydro.centralSoftening must be finite and positive\n");
    }
    if (!detail::isFiniteOption(opts.hydro.zPlaneAcceleration) ||
        opts.hydro.zPlaneAcceleration < 0.0) {
        THROW_ERROR("hydro.zPlaneAcceleration must be finite and nonnegative\n");
    }
    if (!detail::isFiniteOption(opts.hydro.viscosityAlpha) || opts.hydro.viscosityAlpha < 0.0 ||
        !detail::isFiniteOption(opts.hydro.viscosityBeta) || opts.hydro.viscosityBeta < 0.0 ||
        !detail::isFiniteOption(opts.hydro.viscosityEpsilon) ||
        opts.hydro.viscosityEpsilon <= 0.0) {
        THROW_ERROR("hydro.viscosityAlpha and hydro.viscosityBeta must be finite and nonnegative; "
                    "hydro.viscosityEpsilon must be finite and positive\n");
    }
    if (opts.hydro.viscosityEvolution != "none" &&
        opts.hydro.viscosityEvolution != "morrisAndMonaghan" &&
        opts.hydro.viscosityEvolution != "cullenAndDehnen") {
        THROW_ERROR(
            "hydro.viscosityEvolution must be none, morrisAndMonaghan, or cullenAndDehnen\n");
    }
    if (opts.hydro.viscosityBetaMode != "ratio" && opts.hydro.viscosityBetaMode != "constant") {
        THROW_ERROR("hydro.viscosityBetaMode must be ratio or constant\n");
    }
    if (!detail::isPositiveFloatOption(opts.hydro.viscosityAlphaMin) ||
        !detail::isPositiveFloatOption(opts.hydro.viscosityDecay) ||
        !detail::isFiniteFloatOption(opts.hydro.viscosityBetaOverAlpha) ||
        opts.hydro.viscosityBetaOverAlpha < 0.0) {
        THROW_ERROR("hydro.viscosityAlphaMin and hydro.viscosityDecay must be finite and positive "
                    "in single precision; hydro.viscosityBetaOverAlpha must be finite and "
                    "nonnegative in single precision\n");
    }
    if (!detail::isPositiveFloatOption(opts.hydro.viscosityAlphaMax) ||
        !detail::isPositiveFloatOption(opts.hydro.viscosityDecayCD)) {
        THROW_ERROR(
            "hydro.viscosityAlphaMax and hydro.viscosityDecayCD must be finite and positive "
            "in single precision\n");
    }
    if (opts.hydro.viscosityBetaMode == "ratio" &&
        !detail::isFiniteFloatOption(opts.hydro.viscosityBetaOverAlpha *
            (opts.hydro.viscosityEvolution == "morrisAndMonaghan" ?
                    opts.hydro.viscosityAlphaMin :
                    (opts.hydro.viscosityEvolution == "cullenAndDehnen" ?
                            opts.hydro.viscosityAlphaMax :
                            opts.hydro.viscosityAlpha)))) {
        THROW_ERROR("the initial quadratic viscosity coefficient must be finite in single "
                    "precision\n");
    }
    if (!detail::isFiniteOption(opts.hydro.gamma) || opts.hydro.gamma <= 1.0) {
        THROW_ERROR("hydro.gamma must be finite and greater than one\n");
    }
    if (!detail::isFiniteOption(opts.hydro.cfl) || opts.hydro.cfl <= 0.0 || opts.hydro.cfl > 1.0) {
        THROW_ERROR("hydro.cfl must be finite and in (0, 1]\n");
    }
    if (!detail::isFiniteOption(opts.hydro.neighborNumber) ||
        opts.hydro.neighborNumber <= selfNeighborFloor) {
        THROW_ERROR("hydro.neighborNumber must be finite and greater than the selected kernel "
                    "self-neighbor floor\n");
    }
    if (!detail::isFiniteOption(opts.hydro.minSupportRadius) ||
        !detail::isFiniteOption(opts.hydro.maxSupportRadius) ||
        opts.hydro.minSupportRadius <= 0.0 ||
        opts.hydro.maxSupportRadius < opts.hydro.minSupportRadius) {
        THROW_ERROR(
            "hydro support radii must be finite with 0 < minSupportRadius <= maxSupportRadius\n");
    }
    if (!detail::isFiniteFloatOption(opts.hydro.viscosityAlpha) ||
        !detail::isFiniteFloatOption(opts.hydro.viscosityBeta) ||
        !detail::isPositiveFloatOption(opts.hydro.viscosityEpsilon)) {
        THROW_ERROR("hydro.viscosityAlpha, hydro.viscosityBeta and hydro.viscosityEpsilon must "
                    "be representable in single precision, with positive viscosityEpsilon\n");
    }
    if (!detail::isFiniteFloatOption(opts.hydro.gamma) ||
        static_cast<float>(opts.hydro.gamma) <= 1.0f) {
        THROW_ERROR("hydro.gamma must remain finite and greater than one in single precision\n");
    }
    if (!detail::isPositiveFloatOption(opts.hydro.cfl)) {
        THROW_ERROR("hydro.cfl must remain finite and positive in single precision\n");
    }
    if (!detail::isFiniteFloatOption(opts.hydro.neighborNumber) ||
        static_cast<float>(opts.hydro.neighborNumber) <= static_cast<float>(selfNeighborFloor)) {
        THROW_ERROR("hydro.neighborNumber must remain finite and above the selected kernel "
                    "self-neighbor floor in single precision\n");
    }
    if (!detail::isFiniteFloatOption(opts.hydro.yAcceleration) ||
        !detail::isFiniteFloatOption(opts.hydro.centralAcceleration) ||
        !detail::isPositiveFloatOption(opts.hydro.centralSupportRadius)) {
        THROW_ERROR("hydro.yAcceleration, hydro.centralAcceleration and hydro.centralSoftening "
                    "must be representable in single precision, with positive centralSoftening\n");
    }
    if (opts.hydro.zPlaneAcceleration != 0.0 &&
        !detail::isPositiveFloatOption(opts.hydro.zPlaneAcceleration)) {
        THROW_ERROR("nonzero hydro.zPlaneAcceleration must remain finite and positive in "
                    "single precision\n");
    }
    if (!detail::isPositiveFloatOption(
            opts.hydro.minSupportRadius / opts.initial.particlesPerDimension) ||
        !detail::isPositiveFloatOption(
            std::min(opts.hydro.maxSupportRadius / opts.initial.particlesPerDimension, 0.5))) {
        THROW_ERROR("hydro support radii must remain finite and positive in single precision "
                    "after normalization by initial.particlesPerDimension\n");
    }
    opts.run.hierarchicalGravity = integration::kBuildUsesHierarchical && opts.gravity.enabled;
    opts.hydro.damping = 0.0;
    if (opts.compute.sphBucketSize == 0) {
        // Retain the historical SPH bucket heuristic, using the final parsed
        // neighbor target rather than the independently tuned gravity bucket.
        double const bucketSize = 8.0 / std::numbers::pi * opts.hydro.neighborNumber;
        if (bucketSize > std::numeric_limits<int>::max()) {
            THROW_ERROR("compute.sphBucketSize automatic value exceeds the integer range; "
                        "set a positive compute.sphBucketSize\n");
        }
        opts.compute.sphBucketSize = std::max(1, static_cast<int>(bucketSize));
    }
    opts.hydro.minSupportRadius /= opts.initial.particlesPerDimension;
    opts.hydro.maxSupportRadius =
        std::min(opts.hydro.maxSupportRadius / opts.initial.particlesPerDimension, 0.5);
    if (opts.hydro.minSupportRadius > 0.5) {
        THROW_ERROR("hydro.minSupportRadius cannot exceed half the periodic box after "
                    "normalization by initial.particlesPerDimension\n");
    }
    if (opts.run.test == "sphGravity") {
        opts.hydro.enabled = true;
    }
#if !COSMICTIGER_WITH_SPH
    if (opts.hydro.enabled || opts.chemistry.enabled) {
        THROW_ERROR("hydro.enabled=true or chemistry.enabled=true requires "
                    "COSMICTIGER_WITH_SPH=ON\n");
    }
#endif
    if (opts.hydro.enabled && integration::kBuildUsesHierarchical) {
        THROW_ERROR("SPH requires ordinary block-timestep leapfrog; rebuild with "
                    "COSMICTIGER_TIME_INTEGRATOR=leapfrog\n");
    }
    if (opts.hydro.enabled && opts.gravity.enabled && usesMeshGravity(opts.gravity)) {
        THROW_ERROR("SPH currently requires an FMM build when gravity.enabled=true; "
                    "PMMM does not yet support SPH\n");
    }
    if (opts.chemistry.enabled) {
        if (!opts.hydro.enabled ||
            (opts.run.scenario != "cosmology" && opts.run.test != "sphGravity") ||
            opts.initial.createGlass || opts.initial.plummer) {
            THROW_ERROR("chemistry.enabled requires cosmological SPH initial conditions\n");
        }
        if (opts.chemistry.network != "atomic6" && opts.chemistry.network != "molecular9") {
            THROW_ERROR("chemistry.network must be atomic6 or molecular9\n");
        }
        if (opts.chemistry.synchronizedRung < 1 || opts.chemistry.synchronizedRung >= kMaxRung) {
            THROW_ERROR("chemistry.synchronizedRung must be in [1, MAX_RUNG)\n");
        }
        if (!detail::isPositiveFloatOption(opts.chemistry.relativeTolerance) ||
            opts.chemistry.relativeTolerance < 1.0e-5 || opts.chemistry.relativeTolerance > 0.1 ||
            !detail::isPositiveFloatOption(opts.chemistry.abundanceAbsoluteTolerance) ||
            !detail::isPositiveFloatOption(opts.chemistry.energyAbsoluteTolerance) ||
            opts.chemistry.maxSubsteps < 1) {
            THROW_ERROR("Invalid chemistry temperature, tolerances or work bound\n");
        }
        if (!detail::isFiniteOption(opts.cosmology.heliumFraction) ||
            opts.cosmology.heliumFraction < 0.0 || opts.cosmology.heliumFraction >= 1.0) {
            THROW_ERROR("Chemistry requires helium mass fraction in [0,1)\n");
        }
    }
    if (opts.hydro.enabled) {
        if (opts.initial.createGlass || opts.initial.plummer ||
            !opts.initial.gadget4Restart.empty()) {
            THROW_ERROR("SPH cannot use the dark-matter-only glass, Plummer or Gadget loaders\n");
        }
        if (opts.lightcone.enabled || opts.analysis.groups) {
            THROW_ERROR(
                "The lightcone and halo finders require equal-mass dark matter particles\n");
        }
    }
    if (!detail::isPositiveFloatOption(opts.hydro.centralSupportRadius)) {
        THROW_ERROR(
            "hydro.centralSoftening support must remain finite and positive in single precision\n");
    }
    if (!detail::isPositiveFloatOption(opts.gravity.timestepEta)) {
        THROW_ERROR("gravity.timestepEta must remain finite and positive in single precision "
                    "after normalization\n");
    }
    opts.initial.numberOfParticles =
        sqr(opts.initial.particlesPerDimension) * opts.initial.particlesPerDimension;
    opts.initial.fourierSize = opts.initial.particlesPerDimension;
    if (opts.initial.closePack) {
        opts.initial.numberOfParticles *= 2;
        //		opts.Nfour *= 2;
    }
    opts.compute.treeCacheLineSize = 65536 / sizeof(TreeNode);
    opts.compute.treeAllocLineSize = 16 * opts.compute.treeCacheLineSize;
    opts.compute.particleCacheLineSize = 131072 / (sizeof(Position) * kDimensions);
    opts.analysis.saveForce = opts.analysis.saveForce || opts.run.test == "force";
    if (!detail::isFiniteOption(opts.cosmology.omegaB) || opts.cosmology.omegaB < 0.0 ||
        !detail::isFiniteOption(opts.cosmology.omegaC) || opts.cosmology.omegaC < 0.0 ||
        !(opts.cosmology.omegaB + opts.cosmology.omegaC > 0.0) ||
        !detail::isFiniteOption(opts.cosmology.nEff) || opts.cosmology.nEff < 0.0 ||
        !detail::isFiniteOption(opts.cosmology.thetaCmb) || !(opts.cosmology.thetaCmb > 0.0) ||
        !detail::isFiniteOption(opts.cosmology.hubble) || !(opts.cosmology.hubble > 0.0) ||
        !detail::isFiniteOption(opts.cosmology.heliumFraction) ||
        opts.cosmology.heliumFraction < 0.0 || !(opts.cosmology.heliumFraction < 1.0) ||
        !detail::isFiniteOption(opts.cosmology.omegaLambda) ||
        !detail::isFiniteOption(opts.cosmology.spectralIndex) ||
        !detail::isFiniteOption(opts.cosmology.sigma8) || opts.cosmology.sigma8 < 0.0 ||
        !detail::isFiniteOption(opts.cosmology.startRedshift) ||
        opts.cosmology.startRedshift <= -1.0 ||
        !detail::isFiniteOption(opts.cosmology.endRedshift) || opts.cosmology.endRedshift <= -1.0) {
        THROW_ERROR("Invalid cosmology: densities and nEff must be nonnegative, total matter, "
                    "temperature and Hubble rate positive, sigma8 nonnegative, redshifts "
                    "finite and greater than -1, and heliumFraction in [0,1)\n");
    }
    opts.cosmology.omegaM = opts.cosmology.omegaB + opts.cosmology.omegaC;
    opts.initial.darkMatterMass =
        opts.hydro.enabled ? opts.cosmology.omegaC / opts.cosmology.omegaM : 1.0;
    opts.initial.gasMass = opts.hydro.enabled ? opts.cosmology.omegaB / opts.cosmology.omegaM : 0.0;
    if (!detail::isPositiveFloatOption(opts.initial.darkMatterMass) ||
        (opts.hydro.enabled &&
            (!detail::isPositiveFloatOption(opts.initial.gasMass) ||
                !detail::isPositiveFloatOption(opts.cosmology.omegaB) ||
                !detail::isPositiveFloatOption(opts.cosmology.omegaC)))) {
        THROW_ERROR("hydro.enabled requires finite positive gas and dark-matter class masses "
                    "from cosmology.omegaB and cosmology.omegaC\n");
    }

    opts.units.massG *= constants::kM0;
    opts.units.lengthCm = pow(opts.units.massG * (8.0 * std::numbers::pi) *
            opts.initial.numberOfParticles * constants::kG /
            (3.0 * opts.cosmology.omegaM * sqr(constants::kH0 * opts.cosmology.hubble)),
        1.0 / 3.0);
    opts.units.timeSeconds = opts.units.lengthCm / constants::kC;
    double h = constants::kH0 * opts.units.timeSeconds;
    opts.gravity.gm = opts.cosmology.omegaM * 3.0 * sqr(h * opts.cosmology.hubble) /
        (8.0 * std::numbers::pi) / opts.initial.numberOfParticles;
    terminal::detailPrint("box_size = %e Mpc\n", opts.units.lengthCm / constants::kMpcToCm);
    opts.analysis.sliceSize =
        std::min((20.0 / opts.cosmology.hubble) / (opts.units.lengthCm / constants::kMpcToCm), 1.0);
    auto const nside1 =
        pow(std::numbers::pi * opts.initial.numberOfParticles / 10000.0 / 12.0, 0.5);
    int nside = 1;
    while (nside < nside1 / 2.0) {
        nside *= 2;
    }
    if (opts.lightcone.mapSize < 0) {
        opts.lightcone.mapSize = nside;
    }
    terminal::detailPrint("lc map size = %e\n", nside1);

    // The photon density depends only on T_CMB and H_0. Adding massless species
    // changes the neutrino density without changing the photons.
    opts.cosmology.omegaGamma = 32.0 * std::numbers::pi / 3.0 * constants::kG * constants::kSigma *
        std::pow(constants::kH0, -2) * std::pow(constants::kC, -3) *
        std::pow(constants::kTcmb * opts.cosmology.thetaCmb, 4) *
        std::pow(opts.cosmology.hubble, -2);
    opts.cosmology.omegaNu = opts.cosmology.omegaGamma * opts.cosmology.nEff * (7.0 / 8.0) *
        std::pow(4.0 / 11.0, 4.0 / 3.0);
    opts.cosmology.omegaR = opts.cosmology.omegaGamma + opts.cosmology.omegaNu;
    if (automaticLambda) {
        opts.cosmology.omegaLambda = 1.0 - opts.cosmology.omegaM - opts.cosmology.omegaR;
        opts.cosmology.omegaK = 0.0;
    } else {
        opts.cosmology.omegaK =
            1.0 - opts.cosmology.omegaM - opts.cosmology.omegaR - opts.cosmology.omegaLambda;
        // Density closure at floating-point roundoff denotes a flat model.
        // A residual curvature would change a critical Lambda=0 model's fate.
        double const closureScale = std::max({1.0, std::abs(opts.cosmology.omegaM),
            std::abs(opts.cosmology.omegaR), std::abs(opts.cosmology.omegaLambda)});
        if (std::abs(opts.cosmology.omegaK) <=
            8.0 * std::numeric_limits<double>::epsilon() * closureScale) {
            opts.cosmology.omegaK = 0.0;
        }
    }
    if (opts.chemistry.enabled) {
        if (opts.chemistry.initialization != "recombination" &&
            opts.chemistry.initialization != "manual") {
            THROW_ERROR("chemistry.initialization must be recombination or manual\n");
        }
        if (opts.chemistry.initialization == "recombination") {
            if (!detail::isFiniteOption(opts.cosmology.startRedshift) ||
                opts.cosmology.startRedshift <= -1.0 || std::abs(opts.cosmology.omegaK) > 1.0e-12) {
                THROW_ERROR("Native chemistry initialization requires finite startRedshift > -1 "
                            "and a spatially flat cosmology\n");
            }
            primordial::RecombinationParameters parameters;
            parameters.hubble = opts.cosmology.hubble;
            parameters.omegaB = opts.cosmology.omegaB;
            parameters.omegaC = opts.cosmology.omegaC;
            parameters.omegaRadiation = opts.cosmology.omegaR;
            parameters.heliumMassFraction = opts.cosmology.heliumFraction;
            parameters.cmbTemperature = constants::kTcmb * opts.cosmology.thetaCmb;
            double const scaleFactor = 1.0 / (1.0 + opts.cosmology.startRedshift);
            primordial::RecombinationOptions historyOptions;
            historyOptions.maxScaleFactor = std::max(historyOptions.maxScaleFactor, scaleFactor);
            primordial::RecombinationHistory const history(parameters, historyOptions);
            if (!isExplicit("chemistry.initialTemperature")) {
                opts.chemistry.initialTemperature = history.gasTemperature(scaleFactor);
            }
            if (!isExplicit("chemistry.hydrogenIonFraction")) {
                opts.chemistry.hydrogenIonFraction = history.hydrogenIonizedFraction(scaleFactor);
            }
            if (!isExplicit("chemistry.heliumIonFraction")) {
                opts.chemistry.heliumIonFraction = history.heliumSinglyIonizedFraction(scaleFactor);
            }
            if (!isExplicit("chemistry.heliumDoubleIonFraction")) {
                opts.chemistry.heliumDoubleIonFraction =
                    history.heliumDoublyIonizedFraction(scaleFactor);
            }
            // Resolve options on restart as well for checkpoint compatibility.
            // Restarts restore particle state and never call particle initialization.
        }
        primordial::Network const network = opts.chemistry.network == "molecular9" ?
            primordial::Network::Molecular9 :
            primordial::Network::Atomic6;
        if (!detail::isPositiveFloatOption(opts.chemistry.initialTemperature) ||
            opts.chemistry.initialTemperature < primordial::minimumTemperature(network) ||
            opts.chemistry.initialTemperature > primordial::maximumTemperature(network)) {
            THROW_ERROR("Initial chemistry temperature %g K is outside %s domain [%g, %g] K; "
                        "set chemistry.initialTemperature explicitly or select "
                        "chemistry.initialization=manual\n",
                opts.chemistry.initialTemperature, opts.chemistry.network.c_str(),
                double(primordial::minimumTemperature(network)),
                double(primordial::maximumTemperature(network)));
        }
        double const fractions[] = {opts.chemistry.hydrogenIonFraction,
            opts.chemistry.heliumIonFraction, opts.chemistry.heliumDoubleIonFraction,
            opts.chemistry.hydrogenMinusFraction, opts.chemistry.molecularFraction,
            opts.chemistry.molecularIonFraction};
        for (double const fraction : fractions) {
            if (!detail::isFiniteOption(fraction) || fraction < 0.0 || fraction > 1.0) {
                THROW_ERROR("Initial chemistry fractions must be finite and in [0,1]\n");
            }
        }
        if (fractions[0] + fractions[3] + fractions[4] + fractions[5] > 1.0 ||
            fractions[1] + fractions[2] > 1.0 ||
            (opts.chemistry.network == "atomic6" &&
                fractions[3] + fractions[4] + fractions[5] != 0.0)) {
            THROW_ERROR("Initial chemistry fractions violate nuclear closure/network\n");
        }
    }
    // ROCKSTAR Sec. III.1 and Table 2: outer FOF linking length b=0.28.
    // Behroozi, Wechsler & Wu (2013), https://arxiv.org/html/1110.4372v3
    opts.analysis.linkingLength = pow(opts.initial.numberOfParticles, -1.0 / kDimensions) * 0.28;
    opts.analysis.minGroupSize = 20;
    if (opts.initial.createGlass) {
        opts.gravity.gm = 1.0 / opts.initial.numberOfParticles;
        opts.units.lengthCm = constants::kMpcToCm;
        opts.cosmology.hubble = 1.0;
    }
    if (opts.initial.particlesPerDimension % 2 == 1) {
        THROW_ERROR("initial.particlesPerDimension must be an even number\n");
    }
    opts.gravity.supportRadius *= pow(opts.initial.numberOfParticles, -1.0 / kDimensions);
    if (opts.initial.plummer) {
        opts.gravity.supportRadius = gravityTable.supportRadiusPerEffectiveRadius *
            1.5015907655628273 *
            pow((4.0 / 3.0 * std::numbers::pi * pow(opts.initial.plummerRadius, 3.0)) /
                    opts.initial.numberOfParticles,
                1.0 / 3.0);
    }
    if (opts.gravity.enabled && !detail::isPositiveFloatOption(opts.gravity.supportRadius)) {
        THROW_ERROR("gravity.effectiveRadius / gravity.supportRadius (gravity.softening) must "
                    "remain finite and positive in single precision "
                    "after normalization when gravity.enabled is true\n");
    }
    if (usesMeshGravity(opts.gravity) &&
        opts.gravity.supportRadius > double(opts.gravity.pmmmNearWidth) / opts.gravity.meshSize) {
        THROW_ERROR("PMMM requires the normalized gravity.supportRadius to be no larger than "
                    "gravity.pmmmNearWidth/gravity.meshSize\n");
    }
    detail::displayOptions = terminal::verbose() || !opts.run.test.empty();
    if (hpxRank() == 0 && opts.run.test.empty()) {
        PRINT(
            "%.*s\n", int(cosmictiger::kStartupScreen.size()), cosmictiger::kStartupScreen.data());
    }
    if (detail::displayOptions) {
        terminal::section("Resolved options");
    }
    SHOW("chemistry.enabled", chemistry.enabled);
    SHOW("chemistry.network", chemistry.network);
    SHOW("chemistry.initialization", chemistry.initialization);
    SHOW("chemistry.radiationFile", chemistry.radiationFile);
    SHOW("chemistry.synchronizedRung", chemistry.synchronizedRung);
    SHOW("chemistry.initialTemperature", chemistry.initialTemperature);
    SHOW("chemistry.hydrogenIonFraction", chemistry.hydrogenIonFraction);
    SHOW("chemistry.heliumIonFraction", chemistry.heliumIonFraction);
    SHOW("chemistry.heliumDoubleIonFraction", chemistry.heliumDoubleIonFraction);
    SHOW("chemistry.hydrogenMinusFraction", chemistry.hydrogenMinusFraction);
    SHOW("chemistry.molecularFraction", chemistry.molecularFraction);
    SHOW("chemistry.molecularIonFraction", chemistry.molecularIonFraction);
    SHOW("chemistry.relativeTolerance", chemistry.relativeTolerance);
    SHOW("chemistry.abundanceAbsoluteTolerance", chemistry.abundanceAbsoluteTolerance);
    SHOW("chemistry.energyAbsoluteTolerance", chemistry.energyAbsoluteTolerance);
    SHOW("chemistry.maxSubsteps", chemistry.maxSubsteps);
    SHOW("hydro.enabled", hydro.enabled);
    SHOW("gravity.enabled", gravity.enabled);
    show("build.timeIntegrator", std::string(integration::kBuildTimeIntegrator));
    SHOW("build.gravityMethod", gravity.method);
    SHOW("gravity.meshSize", gravity.meshSize);
    SHOW("gravity.pmmmNearWidth", gravity.pmmmNearWidth);
    SHOW("build.pmmmNearMethod", gravity.pmmmNearMethod);
    SHOW("gravity.pmmmBatchParticles", gravity.pmmmBatchParticles);
    SHOW("run.hierarchicalGravity", run.hierarchicalGravity);
    SHOW("hydro.viscosityAlpha", hydro.viscosityAlpha);
    SHOW("hydro.viscosityBeta", hydro.viscosityBeta);
    SHOW("hydro.viscosityEpsilon", hydro.viscosityEpsilon);
    SHOW("hydro.balsaraSwitch", hydro.balsaraSwitch);
    SHOW("hydro.viscosityEvolution", hydro.viscosityEvolution);
    SHOW("hydro.viscosityAlphaMin", hydro.viscosityAlphaMin);
    SHOW("hydro.viscosityDecay", hydro.viscosityDecay);
    SHOW("hydro.viscosityAlphaMax", hydro.viscosityAlphaMax);
    SHOW("hydro.viscosityDecayCD", hydro.viscosityDecayCD);
    SHOW("hydro.viscosityBetaMode", hydro.viscosityBetaMode);
    SHOW("hydro.viscosityBetaOverAlpha", hydro.viscosityBetaOverAlpha);
    SHOW("hydro.gamma", hydro.gamma);
    SHOW("hydro.yAcceleration", hydro.yAcceleration);
    SHOW("hydro.zPlaneAcceleration", hydro.zPlaneAcceleration);
    SHOW("hydro.neighborNumber", hydro.neighborNumber);
    show(
        "hydro.minSupportRadius", opts.hydro.minSupportRadius * opts.initial.particlesPerDimension);
    show(
        "hydro.maxSupportRadius", opts.hydro.maxSupportRadius * opts.initial.particlesPerDimension);
    show("hydro.minEffectiveRadius",
        opts.hydro.minSupportRadius * opts.initial.particlesPerDimension /
            hydroTable.supportRadiusPerEffectiveRadius);
    show("hydro.maxEffectiveRadius",
        opts.hydro.maxSupportRadius * opts.initial.particlesPerDimension /
            hydroTable.supportRadiusPerEffectiveRadius);
    SHOW("hydro.cfl", hydro.cfl);
    terminal::detailPrint("Simulation Options\n");
    terminal::detailPrint("code_to_M_solar = %e\n", opts.units.massG / 1.98e33);
    terminal::detailPrint(
        "Box size = %e cm/h\n", opts.units.lengthCm / constants::kMpcToCm * opts.cosmology.hubble);
    if (opts.initial.plummer) {
        opts.gravity.gm = 1.0 / opts.initial.numberOfParticles;
        // The model's default angle; an explicit gravity.openingAngle takes precedence.
        if (vm.count("gravity.openingAngle") == 0 || vm["gravity.openingAngle"].defaulted()) {
            opts.gravity.openingAngle = 0.4;
        }
    }
#ifdef CHECK_MUTUAL_SORT
    opts.analysis.groups = true;
#endif

    if (usesMeshGravity(opts.gravity)) {
        // Scenario defaults must not replace the PMMM geometry-derived angle.
        opts.gravity.openingAngle = std::sqrt(3.0) / (opts.gravity.pmmmNearWidth + 1);
    } else {
        opts.gravity.openingAngle = gravity::openingScheduleInitialTheta(
            openingSchedule, opts.cosmology.startRedshift, opts.gravity.openingAngle);
    }

    SHOW("checkpoint.intervalSeconds", checkpoint.intervalSeconds);
    SHOW("units.lengthCm", units.lengthCm);
    SHOW("units.massG", units.massG);
    SHOW("units.timeSeconds", units.timeSeconds);
    SHOW("compute.cuda", compute.cuda);
    SHOW("compute.positionType", compute.positionType);
    SHOW("build.fmmOrder", gravity.fmmOrder);
    SHOW("compute.cudaKickResidency", compute.cudaKickResidency);
    SHOW("compute.cudaKickSinkTile", compute.cudaKickSinkTile);
    SHOW("compute.cudaKickParticleTile", compute.cudaKickParticleTile);
    SHOW("compute.cudaKickMultipoleTile", compute.cudaKickMultipoleTile);
    SHOW("compute.cudaKickSharedExpansion", compute.cudaKickSharedExpansion);
    SHOW("compute.cudaKickHandoffBlocks", compute.cudaKickHandoffBlocks);
    SHOW("compute.cudaKickFineHandoffBlocks", compute.cudaKickFineHandoffBlocks);
    SHOW("compute.cudaKickListReserve", compute.cudaKickListReserve);
    SHOW("lightcone.enabled", lightcone.enabled);
    SHOW("analysis.groups", analysis.groups);
    SHOW("analysis.power", analysis.power);
    SHOW("analysis.slice", analysis.slice);
    SHOW("analysis.tracers", analysis.tracers);
    SHOW("analysis.views", analysis.views);
    SHOW("output.silo", output.silo);
    SHOW("output.siloInterval", output.siloInterval);
    SHOW("output.directory", output.directory);
    SHOW("output.units", output.units);
    SHOW("output.frame", output.frame);
    SHOW("output.xMin", output.xMin);
    SHOW("output.xMax", output.xMax);
    SHOW("output.yMin", output.yMin);
    SHOW("output.yMax", output.yMax);
    SHOW("output.zMin", output.zMin);
    SHOW("output.zMax", output.zMax);
    SHOW("gravity.timestepEta", gravity.timestepEta);
    SHOW("gravity.gm", gravity.gm);
    show("gravity.softening",
        opts.gravity.supportRadius *
            std::pow(double(opts.initial.numberOfParticles), 1.0 / kDimensions) /
            gravityTable.supportRadiusPerEffectiveRadius);
    show("gravity.effectiveRadius",
        opts.gravity.supportRadius *
            std::pow(double(opts.initial.numberOfParticles), 1.0 / kDimensions) /
            gravityTable.supportRadiusPerEffectiveRadius);
    show("gravity.supportRadius",
        opts.gravity.supportRadius *
            std::pow(double(opts.initial.numberOfParticles), 1.0 / kDimensions));
    show("hydro.centralEffectiveRadius",
        opts.hydro.centralSupportRadius / gravityTable.supportRadiusPerEffectiveRadius);
    show("hydro.centralSupportRadius", opts.hydro.centralSupportRadius);
    SHOW("gravity.kernelFamily", gravity.kernelFamily);
    SHOW("gravity.kernelDegree", gravity.kernelDegree);
    SHOW("hydro.kernelFamily", hydro.kernelFamily);
    SHOW("hydro.kernelDegree", hydro.kernelDegree);
    SHOW("cosmology.hubble", cosmology.hubble);
    SHOW("lightcone.linkingLength", lightcone.linkingLength);
    SHOW("lightcone.mapSize", lightcone.mapSize);
    SHOW("analysis.linkingLength", analysis.linkingLength);
    SHOW("run.maxIterations", run.maxIterations);
    SHOW("run.treeOpeningAngle", run.treeOpeningAngle);
    SHOW("analysis.minGroupSize", analysis.minGroupSize);
    SHOW("cosmology.omegaB", cosmology.omegaB);
    SHOW("cosmology.omegaC", cosmology.omegaC);
    SHOW("cosmology.omegaGamma", cosmology.omegaGamma);
    SHOW("cosmology.omegaLambda", cosmology.omegaLambda);
    SHOW("cosmology.omegaM", cosmology.omegaM);
    SHOW("cosmology.omegaNu", cosmology.omegaNu);
    SHOW("cosmology.omegaR", cosmology.omegaR);
    SHOW("cosmology.omegaK", cosmology.omegaK);
    SHOW("compute.particleCacheLineSize", compute.particleCacheLineSize);
    SHOW("initial.particlesPerDimension", initial.particlesPerDimension);
    SHOW("checkpoint.readIndex", checkpoint.readIndex);
    SHOW("analysis.saveForce", analysis.saveForce);
    SHOW("analysis.saveAcceleration", analysis.saveAcceleration);
    SHOW("analysis.savePotential", analysis.savePotential);
    SHOW("cosmology.sigma8", cosmology.sigma8);
    SHOW("analysis.sliceResolution", analysis.sliceResolution);
    SHOW("analysis.sliceSize", analysis.sliceSize);
    SHOW("cosmology.thetaCmb", cosmology.thetaCmb);
    SHOW("gravity.openingAngle", gravity.openingAngle);
    SHOW("gravity.openingSchedule", gravity.openingSchedule);
    SHOW("analysis.tracerCount", analysis.tracerCount);
    SHOW("compute.treeCacheLineSize", compute.treeCacheLineSize);
    SHOW("initial.twoLpt", initial.twoLpt);
    SHOW("analysis.viewSize", analysis.viewSize);
    SHOW("cosmology.startRedshift", cosmology.startRedshift);
    SHOW("cosmology.endRedshift", cosmology.endRedshift);

    SHOW("cosmology.spectralIndex", cosmology.spectralIndex);
    SHOW("cosmology.heliumFraction", cosmology.heliumFraction);
    SHOW("cosmology.nEff", cosmology.nEff);
    SHOW("configuration", configuration);
    SHOW("run.test", run.test);
    SHOW("run.verbose", run.verbose);
    SHOW("run.scenario", run.scenario);
    SHOW("run.endTime", run.endTime);
    SHOW("run.numberOfSteps", run.numberOfSteps);
    SHOW("run.minRung", run.minRung);
    SHOW("compute.bucketSize", compute.bucketSize);
    SHOW("compute.sphBucketSize", compute.sphBucketSize);
#ifndef USE_CUDA
    if (opts.compute.cuda) {
        THROW_ERROR("This executable was compiled without CUDA support\n");
    }
#endif
    setOptions(opts);
    if (opts.run.test.empty()) {
        outputPath(".");
    }
    if (hpxRank() == 0 && opts.run.test.empty()) {
        detail::displayOptions = true;
        terminal::section("Run configuration");
        SHOW("run.scenario", run.scenario);
        if (!opts.configuration.empty()) {
            SHOW("configuration", configuration);
        }
        PRINT("%-38s: %zu\n", "initial.numberOfParticles", size_t(opts.initial.numberOfParticles));
        PRINT("%-38s: %i localities, %zu worker threads here\n", "Runtime", hpxSize(),
            size_t(hpx::get_num_worker_threads()));
        SHOW("compute.cuda", compute.cuda);
        SHOW("compute.positionType", compute.positionType);
        show("build.timeIntegrator", std::string(integration::kBuildTimeIntegrator));
        SHOW("gravity.enabled", gravity.enabled);
        if (opts.gravity.enabled) {
            SHOW("build.gravityMethod", gravity.method);
            if (usesMeshGravity(opts.gravity)) {
                SHOW("gravity.meshSize", gravity.meshSize);
                SHOW("gravity.pmmmNearWidth", gravity.pmmmNearWidth);
                SHOW("build.pmmmNearMethod", gravity.pmmmNearMethod);
                SHOW("gravity.pmmmBatchParticles", gravity.pmmmBatchParticles);
                PRINT("%-38s: periodic multipole mesh, %s particle timesteps\n", "PMMM gravity",
                    integration::kBuildUsesHierarchical ? "hierarchical" : "ordinary block");
            }
            SHOW("build.fmmOrder", gravity.fmmOrder);
            SHOW("gravity.openingAngle", gravity.openingAngle);
            if (!opts.gravity.openingSchedule.empty()) {
                SHOW("gravity.openingSchedule", gravity.openingSchedule);
            }
            SHOW("gravity.kernelFamily", gravity.kernelFamily);
            SHOW("gravity.kernelDegree", gravity.kernelDegree);
        }
        SHOW("hydro.enabled", hydro.enabled);
        if (opts.hydro.enabled) {
            SHOW("compute.sphBucketSize", compute.sphBucketSize);
            SHOW("hydro.kernelFamily", hydro.kernelFamily);
            SHOW("hydro.kernelDegree", hydro.kernelDegree);
            SHOW("hydro.balsaraSwitch", hydro.balsaraSwitch);
            SHOW("hydro.viscosityEvolution", hydro.viscosityEvolution);
            SHOW("hydro.viscosityAlphaMin", hydro.viscosityAlphaMin);
            SHOW("hydro.viscosityDecay", hydro.viscosityDecay);
            SHOW("hydro.viscosityAlphaMax", hydro.viscosityAlphaMax);
            SHOW("hydro.viscosityDecayCD", hydro.viscosityDecayCD);
            SHOW("hydro.viscosityBetaMode", hydro.viscosityBetaMode);
            SHOW("hydro.viscosityBetaOverAlpha", hydro.viscosityBetaOverAlpha);
        }
        SHOW("chemistry.enabled", chemistry.enabled);
        if (opts.chemistry.enabled) {
            SHOW("chemistry.network", chemistry.network);
        }
        SHOW("run.maxIterations", run.maxIterations);
        SHOW("run.numberOfSteps", run.numberOfSteps);
        SHOW("run.minRung", run.minRung);
        SHOW("run.endTime", run.endTime);
        terminal::section("Output");
        SHOW("output.directory", output.directory);
        SHOW("output.silo", output.silo);
        if (opts.output.silo) {
            SHOW("output.siloInterval", output.siloInterval);
            PRINT("%-38s: %s, %s\n", "Silo coordinates", opts.output.units.c_str(),
                opts.output.frame.c_str());
        }
        PRINT("\n  Internal diagnostics: --run.verbose=true\n");
        detail::displayOptions = terminal::verbose();
    }
    viewConfigure();
    return rc;
}
