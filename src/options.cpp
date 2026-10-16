/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>

#include <hpx/modules/program_options.hpp>

#include <cosmictiger/constants.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/options_metadata.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/view.hpp>

#define SHOW(label, opt) show(label, opts.opt)

namespace detail {

static std::vector<std::string> commandLineArguments;

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
    PRINT("%-34s: %c\n", name, opt ? 'T' : 'F');
}

void show(char const* name, int opt) {
    PRINT("%-34s: %i\n", name, opt);
}

void show(char const* name, double opt) {
    PRINT("%-34s: %e\n", name, opt);
}

void show(char const* name, std::string opt) {
    PRINT("%-34s: %s\n", name, opt.c_str());
}

Options globalOpts;

HPX_PLAIN_ACTION(setOptions, SetOptionsAction);

Options const& getOptions() {
    return globalOpts;
}

void setOptions(Options const& opts) {
    std::vector<hpx::future<void>> futs;
    for (auto c : hpxChildren()) {
        futs.push_back(hpx::async<SetOptionsAction>(c, opts));
    }
    globalOpts = opts;
    hpx::wait_all(futs.begin(), futs.end());
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
    // In particular, Plummer softening is reported in mean box particle spacings,
    // and the maximum SPH support remains capped at half the periodic box.
    double const softening =
        options.hsoft * std::pow(double(options.numberOfParticles), 1.0 / NDIM) / selfPhi();
    double const timestepEta = options.eta * sqrt(selfPhi());
    return {
        {"configuration", options.configFile},
        {"run.test", options.test},
        {"run.maxIterations", options.maxIter},
        {"run.numberOfSteps", options.numberOfSteps},
        {"run.minRung", options.minrung},
        {"compute.cuda", options.cuda},
        {"compute.bucketSize", options.bucketSize},
        {"checkpoint.readIndex", options.readCheck},
        {"checkpoint.intervalSeconds", options.checkFreq},
        {"initial.gadget4Restart", options.gadget4Restart},
        {"initial.useGlass", options.useGlass},
        {"initial.createGlass", options.createGlass},
        {"initial.plummer", options.plummer},
        {"initial.plummerRadius", options.plummerR},
        {"initial.closePack", options.closePack},
        {"initial.usePowerFile", options.usePowerFile},
        {"initial.twoLpt", options.twolpt},
        {"initial.particlesPerDimension", options.particlesPerDimension},
        {"output.silo", options.outputSilo},
        {"output.siloInterval", options.outputSiloInterval},
        {"output.directory", options.outputDirectory},
        {"output.units", options.outputUnits},
        {"output.frame", options.outputFrame},
        {"output.xMin", options.outputXMin},
        {"output.xMax", options.outputXMax},
        {"output.yMin", options.outputYMin},
        {"output.yMax", options.outputYMax},
        {"output.zMin", options.outputZMin},
        {"output.zMax", options.outputZMax},
        {"analysis.power", options.doPower},
        {"analysis.groups", options.doGroups},
        {"analysis.tracers", options.doTracers},
        {"analysis.tracerCount", options.tracerCount},
        {"analysis.slice", options.doSlice},
        {"analysis.views", options.doViews},
        {"analysis.viewSize", options.viewSize},
        {"analysis.sliceResolution", options.sliceRes},
        {"lightcone.enabled", options.doLc},
        {"lightcone.directory", options.lcDir},
        {"lightcone.linkingLength", options.lcB},
        {"lightcone.mapSize", options.lcMapSize},
        {"gravity.enabled", options.gravity},
        {"gravity.openingAngle", options.theta},
        {"gravity.softening", softening},
        {"gravity.timestepEta", timestepEta},
        {"cosmology.startRedshift", options.z0},
        {"cosmology.endRedshift", options.z1},
        {"cosmology.omegaK", options.omegaK},
        {"cosmology.omegaLambda", options.omegaLam},
        {"cosmology.omegaB", options.omegaB},
        {"cosmology.omegaC", options.omegaC},
        {"cosmology.nEff", options.neff},
        {"cosmology.thetaCmb", options.thetaCmb},
        {"cosmology.heliumFraction", options.y0},
        {"cosmology.sigma8", options.sigma8},
        {"cosmology.hubble", options.hubble},
        {"cosmology.spectralIndex", options.ns},
        {"units.massSolar", options.codeToG / constants::kM0},
        {"hydro.enabled", options.sph},
        {"hydro.chemistry", options.chem},
        {"hydro.conduction", options.conduction},
        {"hydro.diffusion", options.diffusion},
        {"hydro.yReflect", options.yreflect},
        {"hydro.yAcceleration", options.yAcceleration},
        {"hydro.viscosityAlpha", options.alpha0},
        {"hydro.viscosityBeta", options.beta},
        {"hydro.viscosityEpsilon", options.viscosityEpsilon},
        {"hydro.gamma", options.gamma},
        {"hydro.centralAcceleration", options.centralAcceleration},
        {"hydro.centralSoftening", options.hcentral},
        {"hydro.minSupportRadius", options.hmin * options.particlesPerDimension},
        {"hydro.maxSupportRadius", options.hmax * options.particlesPerDimension},
        {"hydro.neighborNumber", options.sphNeighborNumber},
        {"hydro.cfl", options.cfl},
    };
}

bool processOptions(int argc, char* argv[]) {
    Options opts{};
    namespace po = hpx::program_options;
    bool rc;
    po::options_description runOpts("Run control");
    runOpts.add_options()("help", "produce help message")("configuration",
        po::value<std::string>(&opts.configFile)->default_value(""),
        "INI configuration file; explicit command-line values take precedence")(
        "run.test", po::value<std::string>(&opts.test)->default_value(""), "name of test to run")(
        "run.maxIterations", po::value<int>(&opts.maxIter)->default_value(1000000),
        "maximum number of time steps")("run.numberOfSteps",
        po::value<int>(&opts.numberOfSteps)->default_value(64), "number of super time steps")(
        "run.minRung", po::value<int>(&opts.minrung)->default_value(0), "minimum starting rung");

    po::options_description computeOpts("Compute");
    computeOpts.add_options()
#ifdef USE_CUDA
        ("compute.cuda", po::value<bool>(&opts.cuda)->default_value(true), "use CUDA")
#else
        ("compute.cuda", po::value<bool>(&opts.cuda)->default_value(false),
            "use CUDA (not enabled for this build)")
#endif
            ("compute.bucketSize", po::value<int>(&opts.bucketSize)->default_value(128),
                "tree leaf bucket size");

    po::options_description checkpointOpts("Checkpoints");
    checkpointOpts.add_options()("checkpoint.readIndex",
        po::value<int>(&opts.readCheck)->default_value(-1),
        "checkpoint index to restart; -1 starts a new run")("checkpoint.intervalSeconds",
        po::value<int>(&opts.checkFreq)->default_value(1000000000),
        "elapsed wall time between checkpoints, in seconds");

    po::options_description initialOpts("Initial conditions");
    initialOpts.add_options()("initial.gadget4Restart",
        po::value<std::string>(&opts.gadget4Restart)->default_value(""),
        "Gadget4 input file for power-spectrum analysis")("initial.useGlass",
        po::value<bool>(&opts.useGlass)->default_value(false),
        "initialize from a glass file")("initial.createGlass",
        po::value<bool>(&opts.createGlass)->default_value(false), "create a glass file")(
        "initial.plummer", po::value<bool>(&opts.plummer)->default_value(false),
        "initialize the dark-matter Plummer model")("initial.plummerRadius",
        po::value<double>(&opts.plummerR)->default_value(1.0e-2), "Plummer radius")(
        "initial.closePack", po::value<bool>(&opts.closePack)->default_value(false),
        "use a close-packed particle lattice")("initial.usePowerFile",
        po::value<bool>(&opts.usePowerFile)->default_value(true),
        "read power.init, evenly spaced in log wave number")("initial.twoLpt",
        po::value<bool>(&opts.twolpt)->default_value(false),
        "use second-order Lagrangian initial conditions")("initial.particlesPerDimension",
        po::value<int>(&opts.particlesPerDimension)->default_value(128),
        "particle lattice size along each dimension, per species");

    po::options_description outputOpts("Particle output");
    outputOpts.add_options()("output.silo", po::value<bool>(&opts.outputSilo)->default_value(false),
        "write particle Silo files for the configured box")("output.siloInterval",
        po::value<int>(&opts.outputSiloInterval)->default_value(1),
        "number of synchronized coarse steps between particle Silo snapshots")("output.directory",
        po::value<std::string>(&opts.outputDirectory)->default_value("./output"),
        "directory for particle Silo files; relative paths start at the working directory")(
        "output.units", po::value<std::string>(&opts.outputUnits)->default_value("code"),
        "particle Silo units: code, cgs, cosmological, or both (cgs and cosmological)")(
        "output.frame", po::value<std::string>(&opts.outputFrame)->default_value("comoving"),
        "particle Silo frame: comoving, lab, or both; converted velocities are peculiar")(
        "output.xMin", po::value<double>(&opts.outputXMin)->default_value(0.0),
        "minimum x coordinate of the Silo box, in [0, 1]")("output.xMax",
        po::value<double>(&opts.outputXMax)->default_value(1.0),
        "maximum x coordinate of the Silo box, in [0, 1]")("output.yMin",
        po::value<double>(&opts.outputYMin)->default_value(0.0),
        "minimum y coordinate of the Silo box, in [0, 1]")("output.yMax",
        po::value<double>(&opts.outputYMax)->default_value(1.0),
        "maximum y coordinate of the Silo box, in [0, 1]")("output.zMin",
        po::value<double>(&opts.outputZMin)->default_value(0.0),
        "minimum z coordinate of the Silo box, in [0, 1]")("output.zMax",
        po::value<double>(&opts.outputZMax)->default_value(1.0),
        "maximum z coordinate of the Silo box, in [0, 1]");

    po::options_description analysisOpts("Analysis");
    analysisOpts.add_options()("analysis.power",
        po::value<bool>(&opts.doPower)->default_value(false),
        "compute the mass power spectrum")("analysis.groups",
        po::value<bool>(&opts.doGroups)->default_value(false), "find particle groups")(
        "analysis.tracers", po::value<bool>(&opts.doTracers)->default_value(false),
        "write tracer particles to Silo")("analysis.tracerCount",
        po::value<int>(&opts.tracerCount)->default_value(1000000), "number of tracer particles")(
        "analysis.slice", po::value<bool>(&opts.doSlice)->default_value(false),
        "write a projected slice through the volume")("analysis.views",
        po::value<bool>(&opts.doViews)->default_value(false), "write instantaneous HEALPix maps")(
        "analysis.viewSize", po::value<int>(&opts.viewSize)->default_value(1024),
        "view HEALPix Nside")("analysis.sliceResolution",
        po::value<int>(&opts.sliceRes)->default_value(4096), "slice resolution");

    po::options_description lightconeOpts("Lightcone analysis");
    lightconeOpts.add_options()("lightcone.enabled",
        po::value<bool>(&opts.doLc)->default_value(false), "enable lightcone analysis")(
        "lightcone.directory", po::value<std::string>(&opts.lcDir)->default_value("./lightcone"),
        "directory for lightcone output")("lightcone.linkingLength",
        po::value<double>(&opts.lcB)->default_value(0.28), "lightcone group-finder linking length")(
        "lightcone.mapSize", po::value<int>(&opts.lcMapSize)->default_value(-1),
        "lightcone HEALPix Nside; -1 chooses automatically");

    po::options_description gravityOpts("Gravity");
    gravityOpts.add_options()(
        "gravity.enabled", po::value<bool>(&opts.gravity)->default_value(true), "enable gravity")(
        "gravity.openingAngle", po::value<double>(&opts.theta)->default_value(0.8),
        "opening angle for test problems")("gravity.softening",
        po::value<double>(&opts.hsoft)->default_value(1.0 / 80.0),
        "dark-matter softening in interparticle spacings")("gravity.timestepEta",
        po::value<double>(&opts.eta)->default_value(0.25), "gravity time-step criterion");

    po::options_description cosmologyOpts("Cosmology");
    cosmologyOpts.add_options()("cosmology.startRedshift",
        po::value<double>(&opts.z0)->default_value(49.0), "starting redshift")(
        "cosmology.endRedshift", po::value<double>(&opts.z1)->default_value(0.0),
        "ending redshift")("cosmology.omegaK", po::value<double>(&opts.omegaK)->default_value(0.0),
        "curvature density parameter; recomputed from the density budget")("cosmology.omegaLambda",
        po::value<double>(&opts.omegaLam)->default_value(-1.0),
        "dark-energy density parameter; negative chooses a flat universe")("cosmology.omegaB",
        po::value<double>(&opts.omegaB)->default_value(0.049389), "baryon density parameter")(
        "cosmology.omegaC", po::value<double>(&opts.omegaC)->default_value(0.26503),
        "cold-dark-matter density parameter")("cosmology.nEff",
        po::value<double>(&opts.neff)->default_value(3.046),
        "effective number of neutrino species")("cosmology.thetaCmb",
        po::value<double>(&opts.thetaCmb)->default_value(2.7255 / 2.73),
        "CMB temperature divided by 2.73 K")("cosmology.heliumFraction",
        po::value<double>(&opts.y0)->default_value(0.2454006),
        "primordial helium mass fraction")("cosmology.sigma8",
        po::value<double>(&opts.sigma8)->default_value(0.8607), "power-spectrum normalization")(
        "cosmology.hubble", po::value<double>(&opts.hubble)->default_value(0.6732),
        "dimensionless Hubble parameter")("cosmology.spectralIndex",
        po::value<double>(&opts.ns)->default_value(0.96605), "primordial spectral index");

    po::options_description unitsOpts("Units");
    unitsOpts.add_options()("units.massSolar",
        po::value<double>(&opts.codeToG)->default_value(1.e9 / .6732),
        "particle mass unit in solar masses");

    po::options_description hydroOpts("Monaghan (1992) SPH");
    hydroOpts.add_options()("hydro.enabled", po::value<bool>(&opts.sph)->default_value(false),
        "use density/internal-energy SPH")("hydro.chemistry",
        po::value<bool>(&opts.chem)->default_value(false),
        "legacy chemistry; true is unsupported in the Monaghan SPH baseline")("hydro.conduction",
        po::value<bool>(&opts.conduction)->default_value(false),
        "legacy conduction; true is unsupported in the Monaghan SPH baseline")("hydro.diffusion",
        po::value<bool>(&opts.diffusion)->default_value(false),
        "legacy diffusion; true is unsupported in the Monaghan SPH baseline")("hydro.yReflect",
        po::value<bool>(&opts.yreflect)->default_value(false),
        "legacy reflecting y boundary; true is currently unsupported")("hydro.yAcceleration",
        po::value<double>(&opts.yAcceleration)->default_value(0.0),
        "uniform gravitational acceleration in the y direction")("hydro.viscosityAlpha",
        po::value<double>(&opts.alpha0)->default_value(1.0),
        "constant linear artificial-viscosity coefficient alpha")("hydro.viscosityBeta",
        po::value<double>(&opts.beta)->default_value(2.0),
        "constant quadratic artificial-viscosity coefficient beta")("hydro.viscosityEpsilon",
        po::value<double>(&opts.viscosityEpsilon)->default_value(0.01),
        "viscosity denominator regularization eta^2 / h_Monaghan^2")("hydro.gamma",
        po::value<double>(&opts.gamma)->default_value(5.0 / 3.0),
        "ideal-gas adiabatic index")("hydro.centralAcceleration",
        po::value<double>(&opts.centralAcceleration)->default_value(0.0),
        "magnitude of the central force")("hydro.centralSoftening",
        po::value<double>(&opts.hcentral)->default_value(0.01),
        "softening length for the central force")("hydro.minSupportRadius",
        po::value<double>(&opts.hmin)->default_value(1.0e-6),
        "minimum kernel support radius (2 h_Monaghan), in interparticle spacings")(
        "hydro.maxSupportRadius", po::value<double>(&opts.hmax)->default_value(1000.0),
        "maximum kernel support radius in interparticle spacings; capped at half the box")(
        "hydro.neighborNumber", po::value<double>(&opts.sphNeighborNumber)->default_value(60),
        "kernel-weighted neighbor target within the support sphere, including self")(
        "hydro.cfl", po::value<double>(&opts.cfl)->default_value(0.25), "Courant time-step factor");

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
        .add(hydroOpts);
    PRINT("Processing options\n");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, commandOpts), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << commandOpts << "\n";
        rc = false;
    } else {
        if (!opts.configFile.empty()) {
            std::ifstream cfgFs{opts.configFile};
            if (cfgFs) {
                po::store(po::parse_config_file(cfgFs, commandOpts), vm);
                rc = true;
            } else {
                THROW_ERROR("Cannot open configuration file %s\n", opts.configFile.c_str());
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
    if (opts.outputSiloInterval <= 0) {
        THROW_ERROR("output.siloInterval must be positive\n");
    }
    if (opts.outputDirectory.empty()) {
        THROW_ERROR("output.directory must not be empty\n");
    }
    if (opts.outputUnits != "code" && opts.outputUnits != "cgs" &&
        opts.outputUnits != "cosmological" && opts.outputUnits != "both") {
        THROW_ERROR("output.units must be code, cgs, cosmological, or both\n");
    }
    if (opts.outputFrame != "comoving" && opts.outputFrame != "lab" && opts.outputFrame != "both") {
        THROW_ERROR("output.frame must be comoving, lab, or both\n");
    }
    if (opts.outputUnits == "code" && opts.outputFrame != "comoving") {
        THROW_ERROR("output.units=code requires output.frame=comoving because native fields "
                    "retain their evolution normalization; choose output.units=cgs, "
                    "cosmological, or both for lab-frame output\n");
    }
    double const outputMin[] = {opts.outputXMin, opts.outputYMin, opts.outputZMin};
    double const outputMax[] = {opts.outputXMax, opts.outputYMax, opts.outputZMax};
    char const* const outputAxes[] = {"x", "y", "z"};
    for (int dim = 0; dim < NDIM; dim++) {
        if (!detail::isFiniteOption(outputMin[dim]) || !detail::isFiniteOption(outputMax[dim]) ||
            outputMin[dim] < 0.0 || outputMax[dim] > 1.0 || outputMin[dim] >= outputMax[dim]) {
            THROW_ERROR("output.%sMin and output.%sMax must be finite and satisfy "
                        "0 <= min < max <= 1\n",
                outputAxes[dim], outputAxes[dim]);
        }
    }
    if (opts.doTracers && opts.tracerCount <= 0) {
        THROW_ERROR("analysis.tracerCount must be positive when analysis.tracers is enabled\n");
    }
    if (opts.particlesPerDimension <= 0) {
        THROW_ERROR("initial.particlesPerDimension must be positive\n");
    }
    if (opts.numberOfSteps <= 0) {
        THROW_ERROR("run.numberOfSteps must be positive\n");
    }
    if (opts.bucketSize <= 0) {
        THROW_ERROR("compute.bucketSize must be positive\n");
    }
    if (opts.chem || opts.conduction || opts.diffusion) {
        THROW_ERROR("hydro.chemistry, hydro.conduction and hydro.diffusion are detached from SPH; "
                    "leave them false\n");
    }
    if (opts.yreflect) {
        THROW_ERROR("hydro.yReflect=true is unsupported in the Monaghan SPH baseline\n");
    }
    if (!detail::isFiniteOption(opts.yAcceleration) ||
        !detail::isFiniteOption(opts.centralAcceleration) ||
        !detail::isFiniteOption(opts.hcentral) || opts.hcentral <= 0.0) {
        THROW_ERROR("hydro.yAcceleration and hydro.centralAcceleration must be finite; "
                    "hydro.centralSoftening must be finite and positive\n");
    }
    if (!detail::isFiniteOption(opts.alpha0) || opts.alpha0 < 0.0 ||
        !detail::isFiniteOption(opts.beta) || opts.beta < 0.0 ||
        !detail::isFiniteOption(opts.viscosityEpsilon) || opts.viscosityEpsilon <= 0.0) {
        THROW_ERROR("hydro.viscosityAlpha and hydro.viscosityBeta must be finite and nonnegative; "
                    "hydro.viscosityEpsilon must be finite and positive\n");
    }
    if (!detail::isFiniteOption(opts.gamma) || opts.gamma <= 1.0) {
        THROW_ERROR("hydro.gamma must be finite and greater than one\n");
    }
    if (!detail::isFiniteOption(opts.cfl) || opts.cfl <= 0.0 || opts.cfl > 1.0) {
        THROW_ERROR("hydro.cfl must be finite and in (0, 1]\n");
    }
    if (!detail::isFiniteOption(opts.sphNeighborNumber) || opts.sphNeighborNumber <= 32.0 / 3.0) {
        THROW_ERROR(
            "hydro.neighborNumber must be finite and greater than 32/3 (the self-kernel floor)\n");
    }
    if (!detail::isFiniteOption(opts.hmin) || !detail::isFiniteOption(opts.hmax) ||
        opts.hmin <= 0.0 || opts.hmax < opts.hmin) {
        THROW_ERROR(
            "hydro support radii must be finite with 0 < minSupportRadius <= maxSupportRadius\n");
    }
    if (!detail::isFiniteFloatOption(opts.alpha0) || !detail::isFiniteFloatOption(opts.beta) ||
        !detail::isPositiveFloatOption(opts.viscosityEpsilon)) {
        THROW_ERROR("hydro.viscosityAlpha, hydro.viscosityBeta and hydro.viscosityEpsilon must "
                    "be representable in single precision, with positive viscosityEpsilon\n");
    }
    if (!detail::isFiniteFloatOption(opts.gamma) || static_cast<float>(opts.gamma) <= 1.0f) {
        THROW_ERROR("hydro.gamma must remain finite and greater than one in single precision\n");
    }
    if (!detail::isPositiveFloatOption(opts.cfl)) {
        THROW_ERROR("hydro.cfl must remain finite and positive in single precision\n");
    }
    if (!detail::isFiniteFloatOption(opts.sphNeighborNumber) ||
        static_cast<float>(opts.sphNeighborNumber) <= static_cast<float>(32.0 / 3.0)) {
        THROW_ERROR("hydro.neighborNumber must remain finite and above 32/3 in single precision\n");
    }
    if (!detail::isFiniteFloatOption(opts.yAcceleration) ||
        !detail::isFiniteFloatOption(opts.centralAcceleration) ||
        !detail::isPositiveFloatOption(opts.hcentral)) {
        THROW_ERROR("hydro.yAcceleration, hydro.centralAcceleration and hydro.centralSoftening "
                    "must be representable in single precision, with positive centralSoftening\n");
    }
    if (!detail::isPositiveFloatOption(opts.hmin / opts.particlesPerDimension) ||
        !detail::isPositiveFloatOption(std::min(opts.hmax / opts.particlesPerDimension, 0.5))) {
        THROW_ERROR("hydro support radii must remain finite and positive in single precision "
                    "after normalization by initial.particlesPerDimension\n");
    }
    opts.alpha1 = opts.alpha0;
    opts.alphaDecay = 0.1;
    opts.viscType = 2;
    opts.glass = 0;
    opts.stars = false;
    opts.vsoft = false;
    opts.htime = true;
    opts.damping = 0.0;
    opts.sigma8C = opts.sigma8;
    opts.sphBucketSize = opts.bucketSize;
    opts.hmin /= opts.particlesPerDimension;
    opts.hmax = std::min(opts.hmax / opts.particlesPerDimension, 0.5);
    if (opts.hmin > 0.5) {
        THROW_ERROR("hydro.minSupportRadius cannot exceed half the periodic box after "
                    "normalization by initial.particlesPerDimension\n");
    }
    bool const hydroTest = opts.test == "sphGravity" || opts.test == "sod" ||
        opts.test == "blast" || opts.test == "helmholtz" || opts.test == "rt" ||
        opts.test == "disc" || opts.test == "star" || opts.test == "plummer";
    if (hydroTest) {
        opts.sph = true;
        opts.chem = false;
        opts.stars = false;
        opts.conduction = false;
        opts.diffusion = false;
        opts.gravity = opts.test == "star" || opts.test == "plummer";
    }
    if (opts.sph) {
        opts.htime = false;
        if (!opts.cuda && opts.test != "sphGravity") {
            THROW_ERROR("SPH requires the CUDA backend\n");
        }
        if (opts.createGlass || opts.plummer || !opts.gadget4Restart.empty()) {
            THROW_ERROR("SPH cannot use the dark-matter-only glass, Plummer or Gadget loaders\n");
        }
        if (opts.doLc || opts.doGroups) {
            THROW_ERROR(
                "The lightcone and halo finders require equal-mass dark matter particles\n");
        }
        if (opts.stars && !opts.chem) {
            THROW_ERROR("Star formation requires hydro.chemistry=true\n");
        }
    } else {
        opts.chem = opts.stars = opts.conduction = opts.diffusion = false;
    }
    opts.hsoft *= selfPhi();
    opts.eta /= sqrt(selfPhi());
    if (!detail::isPositiveFloatOption(opts.eta)) {
        THROW_ERROR("gravity.timestepEta must remain finite and positive in single precision "
                    "after normalization\n");
    }
    opts.numberOfParticles = sqr(opts.particlesPerDimension) * opts.particlesPerDimension;
    opts.nfour = opts.particlesPerDimension;
    if (opts.closePack) {
        opts.numberOfParticles *= 2;
        //		opts.Nfour *= 2;
    }
    opts.treeCacheLineSize = 65536 / sizeof(TreeNode);
    opts.treeAllocLineSize = 16 * opts.treeCacheLineSize;
    opts.particleCacheLineSize = 131072 / (sizeof(Fixed32) * NDIM);
    opts.saveForce = opts.test == "force";
    opts.omegaM = opts.omegaB + opts.omegaC;
    opts.darkMatterMass = opts.sph ? opts.omegaC / opts.omegaM : 1.0;
    opts.sphMass = opts.sph ? opts.omegaB / opts.omegaM : 0.0;
    opts.rho0B = opts.numberOfParticles * opts.omegaB / opts.omegaM;
    opts.rho0C = opts.numberOfParticles * opts.omegaC / opts.omegaM;

    opts.codeToG *= constants::kM0;
    opts.codeToCm = pow(opts.codeToG * (8.0 * M_PI) * opts.numberOfParticles * constants::kG /
            (3.0 * opts.omegaM * sqr(constants::kH0 * opts.hubble)),
        1.0 / 3.0);
    opts.codeToS = opts.codeToCm / constants::kC;
    double h = constants::kH0 * opts.codeToS;
    opts.gm = opts.omegaM * 3.0 * sqr(h * opts.hubble) / (8.0 * M_PI) / opts.numberOfParticles;
    PRINT("box_size = %e Mpc\n", opts.codeToCm / constants::kMpcToCm);
    opts.sliceSize = std::min((20.0 / opts.hubble) / (opts.codeToCm / constants::kMpcToCm), 1.0);
    double const neff = 3.086;
    double const thetaValue = 1.0;
    auto const nside1 = pow(M_PI * opts.numberOfParticles / 10000.0 / 12.0, 0.5);
    int nside = 1;
    while (nside < nside1 / 2.0) {
        nside *= 2;
    }
    if (opts.lcMapSize < 0) {
        opts.lcMapSize = nside;
    }
    PRINT("lc map size = %e\n", nside1);

    double omegaR = 32.0 * M_PI / 3.0 * constants::kG * constants::kSigma *
        (1 + neff * (7. / 8.0) * std::pow(4. / 11., 4. / 3.)) * std::pow(constants::kH0, -2) *
        std::pow(constants::kC, -3) * std::pow(2.73 * thetaValue, 4) * std::pow(opts.hubble, -2);
    opts.omegaNu = omegaR * opts.neff / (8.0 / 7.0 * std::pow(11.0 / 4.0, 4.0 / 3.0) + opts.neff);
    opts.omegaGam = omegaR - opts.omegaNu;
    opts.omegaR = omegaR;
    if (opts.omegaLam < 0.0) {
        opts.omegaLam = 1.0 - opts.omegaM - opts.omegaR;
        opts.omegaK = 0.0;
    } else {
        opts.omegaK = 1.0 - opts.omegaM - opts.omegaR - opts.omegaLam;
    }
    opts.linkingLength = pow(opts.numberOfParticles, -1.0 / NDIM) * 0.28;
    opts.minGroup = 20;
    opts.lcMinGroup = 20;
    if (opts.createGlass) {
        opts.gm = 1.0 / opts.numberOfParticles;
        opts.codeToCm = constants::kMpcToCm;
        opts.hubble = 1.0;
    }
    if (opts.particlesPerDimension % 2 == 1) {
        THROW_ERROR("initial.particlesPerDimension must be an even number\n");
    }
    opts.hsoft *= pow(opts.numberOfParticles, -1.0 / NDIM);
    if (opts.plummer) {
        opts.hsoft = selfPhi() *
            pow((4.0 / 3.0 * M_PI * pow(opts.plummerR, 3.0)) / opts.numberOfParticles, 1.0 / 3.0);
    }
    if (opts.gravity && !detail::isPositiveFloatOption(opts.hsoft)) {
        THROW_ERROR("gravity.softening must remain finite and positive in single precision "
                    "after normalization when gravity.enabled is true\n");
    }
    SHOW("hydro.enabled", sph);
    SHOW("run.hierarchicalGravity", htime);
    SHOW("hydro.chemistry", chem);
    SHOW("hydro.stars", stars);
    SHOW("hydro.viscosityAlpha", alpha0);
    SHOW("hydro.viscosityBeta", beta);
    SHOW("hydro.viscosityEpsilon", viscosityEpsilon);
    SHOW("hydro.gamma", gamma);
    SHOW("hydro.neighborNumber", sphNeighborNumber);
    show("hydro.minSupportRadius", opts.hmin * opts.particlesPerDimension);
    show("hydro.maxSupportRadius", opts.hmax * opts.particlesPerDimension);
    SHOW("hydro.cfl", cfl);
    PRINT("Simulation Options\n");
    PRINT("code_to_M_solar = %e\n", opts.codeToG / 1.98e33);
    PRINT("Box size = %e cm/h\n", opts.codeToCm / constants::kMpcToCm * opts.hubble);
    if (opts.plummer) {
        opts.gm = 1.0 / opts.numberOfParticles;
        opts.theta = 0.4;
    }
#ifdef CHECK_MUTUAL_SORT
    opts.doGroups = true;
#endif

    SHOW("checkpoint.intervalSeconds", checkFreq);
    SHOW("units.lengthCm", codeToCm);
    SHOW("units.massG", codeToG);
    SHOW("units.timeSeconds", codeToS);
    SHOW("compute.cuda", cuda);
    SHOW("lightcone.enabled", doLc);
    SHOW("analysis.groups", doGroups);
    SHOW("analysis.power", doPower);
    SHOW("analysis.slice", doSlice);
    SHOW("analysis.tracers", doTracers);
    SHOW("analysis.views", doViews);
    SHOW("output.silo", outputSilo);
    SHOW("output.siloInterval", outputSiloInterval);
    SHOW("output.directory", outputDirectory);
    SHOW("output.units", outputUnits);
    SHOW("output.frame", outputFrame);
    SHOW("output.xMin", outputXMin);
    SHOW("output.xMax", outputXMax);
    SHOW("output.yMin", outputYMin);
    SHOW("output.yMax", outputYMax);
    SHOW("output.zMin", outputZMin);
    SHOW("output.zMax", outputZMax);
    SHOW("gravity.timestepEta", eta);
    SHOW("gravity.gm", gm);
    SHOW("gravity.softening", hsoft);
    SHOW("cosmology.hubble", hubble);
    SHOW("lightcone.linkingLength", lcB);
    SHOW("lightcone.minGroupSize", lcMinGroup);
    SHOW("lightcone.mapSize", lcMapSize);
    SHOW("analysis.linkingLength", linkingLength);
    SHOW("run.maxIterations", maxIter);
    SHOW("analysis.minGroupSize", minGroup);
    SHOW("cosmology.omegaB", omegaB);
    SHOW("cosmology.omegaC", omegaC);
    SHOW("cosmology.omegaGamma", omegaGam);
    SHOW("cosmology.omegaM", omegaM);
    SHOW("cosmology.omegaNu", omegaNu);
    SHOW("cosmology.omegaR", omegaR);
    SHOW("compute.particleCacheLineSize", particleCacheLineSize);
    SHOW("initial.particlesPerDimension", particlesPerDimension);
    SHOW("checkpoint.readIndex", readCheck);
    SHOW("analysis.saveForce", saveForce);
    SHOW("cosmology.sigma8", sigma8);
    SHOW("analysis.sliceResolution", sliceRes);
    SHOW("analysis.sliceSize", sliceSize);
    SHOW("cosmology.thetaCmb", thetaCmb);
    SHOW("gravity.openingAngle", theta);
    SHOW("analysis.tracerCount", tracerCount);
    SHOW("compute.treeCacheLineSize", treeCacheLineSize);
    SHOW("initial.twoLpt", twolpt);
    SHOW("initial.usePowerFile", usePowerFile);
    SHOW("analysis.viewSize", viewSize);
    SHOW("cosmology.startRedshift", z0);
    SHOW("cosmology.endRedshift", z1);

    SHOW("cosmology.spectralIndex", ns);
    SHOW("cosmology.heliumFraction", y0);
    SHOW("cosmology.nEff", neff);
    SHOW("configuration", configFile);
    SHOW("run.test", test);
#ifndef USE_CUDA
    if (opts.cuda) {
        THROW_ERROR("This executable was compiled without CUDA support\n");
    }
#endif
    setOptions(opts);
    viewConfigure();
    return rc;
}
