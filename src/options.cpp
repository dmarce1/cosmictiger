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
#include <stdexcept>

#include <hpx/modules/program_options.hpp>

#include <cosmictiger/constants.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/fmm_order.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/kernel.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/options_metadata.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/view.hpp>

#define SHOW(label, opt) show(label, opts.opt)

namespace detail {

static std::vector<std::string> commandLineArguments;

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
        throw std::invalid_argument("kernelDegree must be 3..9 for spline, 2/5/8 for "
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
    setFmmOrder(opts.gravity.fmmOrder);
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
    // Public effective radius is 2 sigma in mean box particle spacings.
    auto const gravityTable = radial_kernel::makeKernelTable(
        detail::kernelSpec(options.gravity.kernelFamily, options.gravity.kernelDegree));
    auto const hydroTable = radial_kernel::makeKernelTable(
        detail::kernelSpec(options.hydro.kernelFamily, options.hydro.kernelDegree));
    double const effectiveRadius = options.gravity.supportRadius *
        std::pow(double(options.initial.numberOfParticles), 1.0 / NDIM) /
        gravityTable.supportRadiusPerEffectiveRadius;
    double const timestepEta = options.gravity.timestepEta;
    return {
        {"configuration", options.configuration},
        {"run.test", options.run.test},
        {"run.maxIterations", options.run.maxIterations},
        {"run.numberOfSteps", options.run.numberOfSteps},
        {"run.minRung", options.run.minRung},
        {"run.treeOpeningAngle", options.run.treeOpeningAngle},
        {"compute.cuda", options.compute.cuda},
        {"compute.bucketSize", options.compute.bucketSize},
        {"compute.cudaKickResidency", options.compute.cudaKickResidency},
        {"compute.cudaKickSinkTile", options.compute.cudaKickSinkTile},
        {"compute.cudaKickParticleTile", options.compute.cudaKickParticleTile},
        {"compute.cudaKickMultipoleTile", options.compute.cudaKickMultipoleTile},
        {"compute.cudaKickSharedExpansion", options.compute.cudaKickSharedExpansion},
        {"compute.cudaKickHandoffBlocks", options.compute.cudaKickHandoffBlocks},
        {"compute.cudaKickListReserve", options.compute.cudaKickListReserve},
        {"checkpoint.readIndex", options.checkpoint.readIndex},
        {"checkpoint.intervalSeconds", options.checkpoint.intervalSeconds},
        {"initial.gadget4Restart", options.initial.gadget4Restart},
        {"initial.useGlass", options.initial.useGlass},
        {"initial.createGlass", options.initial.createGlass},
        {"initial.plummer", options.initial.plummer},
        {"initial.plummerRadius", options.initial.plummerRadius},
        {"initial.closePack", options.initial.closePack},
        {"initial.usePowerFile", options.initial.usePowerFile},
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
        {"lightcone.enabled", options.lightcone.enabled},
        {"lightcone.directory", options.lightcone.directory},
        {"lightcone.linkingLength", options.lightcone.linkingLength},
        {"lightcone.mapSize", options.lightcone.mapSize},
        {"gravity.enabled", options.gravity.enabled},
        {"gravity.fmmOrder", options.gravity.fmmOrder},
        {"gravity.kernelFamily", options.gravity.kernelFamily},
        {"gravity.kernelDegree", options.gravity.kernelDegree},
        {"gravity.openingAngle", options.gravity.openingAngle},
        {"gravity.openLargerCell", options.gravity.openLargerCell},
        {"gravity.softening", effectiveRadius},
        {"gravity.effectiveRadius", effectiveRadius},
        {"gravity.supportRadius",
            options.gravity.supportRadius *
                std::pow(double(options.initial.numberOfParticles), 1.0 / NDIM)},
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
        {"hydro.viscosityAlpha", options.hydro.viscosityAlpha},
        {"hydro.viscosityBeta", options.hydro.viscosityBeta},
        {"hydro.viscosityEpsilon", options.hydro.viscosityEpsilon},
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
    namespace po = hpx::program_options;
    bool rc;
    po::options_description runOpts("Run control");
    runOpts.add_options()("help", "produce help message")("configuration",
        po::value<std::string>(&opts.configuration)->default_value(""),
        "INI configuration file; explicit command-line values take precedence")("run.test",
        po::value<std::string>(&opts.run.test)->default_value(""), "name of test to run")(
        "run.maxIterations", po::value<int>(&opts.run.maxIterations)->default_value(1000000),
        "maximum number of time steps")("run.numberOfSteps",
        po::value<int>(&opts.run.numberOfSteps)->default_value(64),
        "number of super time steps")("run.minRung",
        po::value<int>(&opts.run.minRung)->default_value(0), "minimum starting rung")(
        "run.treeOpeningAngle", po::value<double>(&opts.run.treeOpeningAngle)->default_value(0.0),
        "run.test=force: build the tree once at this opening angle and sweep only the kick's "
        "angle upward from it; 0 rebuilds the tree at every angle");

    po::options_description computeOpts("Compute");
    computeOpts.add_options()
#ifdef USE_CUDA
        ("compute.cuda", po::value<bool>(&opts.compute.cuda)->default_value(true), "use CUDA")
#else
        ("compute.cuda", po::value<bool>(&opts.compute.cuda)->default_value(false),
            "use CUDA (not enabled for this build)")
#endif
            ("compute.bucketSize", po::value<int>(&opts.compute.bucketSize)->default_value(128),
                "tree leaf bucket size")("compute.cudaKickResidency",
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
                "fixed CUDA block count for the CPU/GPU handoff threshold; 0 derives it")(
                "compute.cudaKickListReserve",
                po::value<int>(&opts.compute.cudaKickListReserve)->default_value(4096),
                "entries reserved per CUDA kick interaction list at block start; 0 grows on "
                "demand");

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
        "use a close-packed particle lattice")("initial.usePowerFile",
        po::value<bool>(&opts.initial.usePowerFile)->default_value(true),
        "read power.init, evenly spaced in log wave number")("initial.twoLpt",
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
        "directory for particle Silo files; relative paths start at the working directory")(
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
        po::value<int>(&opts.analysis.sliceResolution)->default_value(4096), "slice resolution");

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
        "gravity.fmmOrder", po::value<int>(&opts.gravity.fmmOrder)->default_value(kFmmDefaultOrder),
        "FMM order (multipole degree plus two), 3 to 9; every order is compiled in")(
        "gravity.openingAngle", po::value<double>(&opts.gravity.openingAngle)->default_value(0.8),
        "opening angle for test problems")("gravity.openLargerCell",
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
        "density degree: spline 3..9; wendland 2,5,8; ferrers 2,4,6,8")("gravity.timestepEta",
        po::value<double>(&opts.gravity.timestepEta)->default_value(0.20401599335954483),
        "prefactor C in dt = C sqrt(effectiveRadius/acceleration), effectiveRadius = 2 sigma");

    po::options_description cosmologyOpts("Cosmology");
    cosmologyOpts.add_options()("cosmology.startRedshift",
        po::value<double>(&opts.cosmology.startRedshift)->default_value(49.0), "starting redshift")(
        "cosmology.endRedshift", po::value<double>(&opts.cosmology.endRedshift)->default_value(0.0),
        "ending redshift")("cosmology.omegaLambda",
        po::value<double>(&opts.cosmology.omegaLambda)->default_value(-1.0),
        "dark-energy density parameter; negative chooses a flat universe")("cosmology.omegaB",
        po::value<double>(&opts.cosmology.omegaB)->default_value(0.049389),
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
        "power-spectrum normalization")("cosmology.hubble",
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
        "density degree: spline 3..9; wendland 2,5,8; ferrers 2,4,6,8")("hydro.yAcceleration",
        po::value<double>(&opts.hydro.yAcceleration)->default_value(0.0),
        "uniform gravitational acceleration in the y direction")("hydro.viscosityAlpha",
        po::value<double>(&opts.hydro.viscosityAlpha)->default_value(1.0),
        "constant linear artificial-viscosity coefficient alpha")("hydro.viscosityBeta",
        po::value<double>(&opts.hydro.viscosityBeta)->default_value(2.0),
        "constant quadratic artificial-viscosity coefficient beta")("hydro.viscosityEpsilon",
        po::value<double>(&opts.hydro.viscosityEpsilon)->default_value(0.01),
        "viscosity regularization relative to the calibrated radius sqrt(5/6) effectiveRadius")(
        "hydro.gamma", po::value<double>(&opts.hydro.gamma)->default_value(5.0 / 3.0),
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
        "atomic6 or molecular9; optically thin low-density gas")("chemistry.radiationFile",
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
    PRINT("Processing options\n");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, commandOpts), vm);
    po::notify(vm);
    if (vm.count("help")) {
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
    double const selfNeighborFloor = 4.0 * M_PI / 3.0 * radial_kernel::value(hydroTable, 0.0);
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
    for (int dim = 0; dim < NDIM; dim++) {
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
    if (opts.compute.bucketSize <= 0) {
        THROW_ERROR("compute.bucketSize must be positive\n");
    }
    if (opts.gravity.fmmOrder < kFmmMinOrder || opts.gravity.fmmOrder > kFmmMaxOrder) {
        THROW_ERROR("gravity.fmmOrder must be between %i and %i\n", kFmmMinOrder, kFmmMaxOrder);
    }
    if (opts.compute.bucketSize > MAX_BUCKET_SIZE) {
        THROW_ERROR("compute.bucketSize must not exceed the compile-time maximum %i "
                    "(COSMICTIGER_WITH_MAX_BUCKET)\n",
            int(MAX_BUCKET_SIZE));
    }
    if (opts.compute.cudaKickResidency < 0 || opts.compute.cudaKickSinkTile < -1 ||
        opts.compute.cudaKickParticleTile < 0 || opts.compute.cudaKickMultipoleTile < 0 ||
        opts.compute.cudaKickHandoffBlocks < 0 || opts.compute.cudaKickListReserve < 0) {
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
    if (!detail::isFiniteOption(opts.hydro.viscosityAlpha) || opts.hydro.viscosityAlpha < 0.0 ||
        !detail::isFiniteOption(opts.hydro.viscosityBeta) || opts.hydro.viscosityBeta < 0.0 ||
        !detail::isFiniteOption(opts.hydro.viscosityEpsilon) ||
        opts.hydro.viscosityEpsilon <= 0.0) {
        THROW_ERROR("hydro.viscosityAlpha and hydro.viscosityBeta must be finite and nonnegative; "
                    "hydro.viscosityEpsilon must be finite and positive\n");
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
    if (!detail::isPositiveFloatOption(
            opts.hydro.minSupportRadius / opts.initial.particlesPerDimension) ||
        !detail::isPositiveFloatOption(
            std::min(opts.hydro.maxSupportRadius / opts.initial.particlesPerDimension, 0.5))) {
        THROW_ERROR("hydro support radii must remain finite and positive in single precision "
                    "after normalization by initial.particlesPerDimension\n");
    }
    opts.run.hierarchicalGravity = true;
    opts.hydro.damping = 0.0;
    opts.compute.sphBucketSize = opts.compute.bucketSize;
    opts.hydro.minSupportRadius /= opts.initial.particlesPerDimension;
    opts.hydro.maxSupportRadius =
        std::min(opts.hydro.maxSupportRadius / opts.initial.particlesPerDimension, 0.5);
    if (opts.hydro.minSupportRadius > 0.5) {
        THROW_ERROR("hydro.minSupportRadius cannot exceed half the periodic box after "
                    "normalization by initial.particlesPerDimension\n");
    }
    bool const hydroTest = opts.run.test == "sphGravity" || opts.run.test == "sod" ||
        opts.run.test == "blast" || opts.run.test == "helmholtz" || opts.run.test == "rt" ||
        opts.run.test == "disc" || opts.run.test == "star" || opts.run.test == "plummer";
    if (hydroTest) {
        opts.hydro.enabled = true;
        opts.gravity.enabled = opts.run.test == "star" || opts.run.test == "plummer";
    }
    if (opts.chemistry.enabled) {
        if (!opts.hydro.enabled || (hydroTest && opts.run.test != "sphGravity") ||
            opts.initial.createGlass || opts.initial.plummer) {
            THROW_ERROR("chemistry.enabled requires cosmological SPH initial conditions\n");
        }
        if (opts.chemistry.network != "atomic6" && opts.chemistry.network != "molecular9") {
            THROW_ERROR("chemistry.network must be atomic6 or molecular9\n");
        }
        if (opts.chemistry.synchronizedRung < 1 || opts.chemistry.synchronizedRung >= MAX_RUNG) {
            THROW_ERROR("chemistry.synchronizedRung must be in [1, MAX_RUNG)\n");
        }
        if (!detail::isPositiveFloatOption(opts.chemistry.initialTemperature) ||
            !detail::isPositiveFloatOption(opts.chemistry.relativeTolerance) ||
            opts.chemistry.relativeTolerance < 1.0e-5 || opts.chemistry.relativeTolerance > 0.1 ||
            !detail::isPositiveFloatOption(opts.chemistry.abundanceAbsoluteTolerance) ||
            !detail::isPositiveFloatOption(opts.chemistry.energyAbsoluteTolerance) ||
            opts.chemistry.maxSubsteps < 1) {
            THROW_ERROR("Invalid chemistry temperature, tolerances or work bound\n");
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
        if (!detail::isFiniteOption(opts.cosmology.heliumFraction) ||
            opts.cosmology.heliumFraction < 0.0 || opts.cosmology.heliumFraction >= 1.0) {
            THROW_ERROR("Chemistry requires helium mass fraction in [0,1)\n");
        }
    }
    if (opts.hydro.enabled) {
        opts.run.hierarchicalGravity = false;
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
    opts.compute.particleCacheLineSize = 131072 / (sizeof(Fixed32) * NDIM);
    opts.analysis.saveForce = opts.run.test == "force";
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
    opts.units.lengthCm =
        pow(opts.units.massG * (8.0 * M_PI) * opts.initial.numberOfParticles * constants::kG /
                (3.0 * opts.cosmology.omegaM * sqr(constants::kH0 * opts.cosmology.hubble)),
            1.0 / 3.0);
    opts.units.timeSeconds = opts.units.lengthCm / constants::kC;
    double h = constants::kH0 * opts.units.timeSeconds;
    opts.gravity.gm = opts.cosmology.omegaM * 3.0 * sqr(h * opts.cosmology.hubble) / (8.0 * M_PI) /
        opts.initial.numberOfParticles;
    PRINT("box_size = %e Mpc\n", opts.units.lengthCm / constants::kMpcToCm);
    opts.analysis.sliceSize =
        std::min((20.0 / opts.cosmology.hubble) / (opts.units.lengthCm / constants::kMpcToCm), 1.0);
    double const neff = 3.086;
    double const thetaValue = 1.0;
    auto const nside1 = pow(M_PI * opts.initial.numberOfParticles / 10000.0 / 12.0, 0.5);
    int nside = 1;
    while (nside < nside1 / 2.0) {
        nside *= 2;
    }
    if (opts.lightcone.mapSize < 0) {
        opts.lightcone.mapSize = nside;
    }
    PRINT("lc map size = %e\n", nside1);

    double omegaR = 32.0 * M_PI / 3.0 * constants::kG * constants::kSigma *
        (1 + neff * (7. / 8.0) * std::pow(4. / 11., 4. / 3.)) * std::pow(constants::kH0, -2) *
        std::pow(constants::kC, -3) * std::pow(2.73 * thetaValue, 4) *
        std::pow(opts.cosmology.hubble, -2);
    opts.cosmology.omegaNu = omegaR * opts.cosmology.nEff /
        (8.0 / 7.0 * std::pow(11.0 / 4.0, 4.0 / 3.0) + opts.cosmology.nEff);
    opts.cosmology.omegaGamma = omegaR - opts.cosmology.omegaNu;
    opts.cosmology.omegaR = omegaR;
    if (opts.cosmology.omegaLambda < 0.0) {
        opts.cosmology.omegaLambda = 1.0 - opts.cosmology.omegaM - opts.cosmology.omegaR;
        opts.cosmology.omegaK = 0.0;
    } else {
        opts.cosmology.omegaK =
            1.0 - opts.cosmology.omegaM - opts.cosmology.omegaR - opts.cosmology.omegaLambda;
    }
    // ROCKSTAR Sec. III.1 and Table 2: outer FOF linking length b=0.28.
    // Behroozi, Wechsler & Wu (2013), https://arxiv.org/html/1110.4372v3
    opts.analysis.linkingLength = pow(opts.initial.numberOfParticles, -1.0 / NDIM) * 0.28;
    opts.analysis.minGroupSize = 20;
    if (opts.initial.createGlass) {
        opts.gravity.gm = 1.0 / opts.initial.numberOfParticles;
        opts.units.lengthCm = constants::kMpcToCm;
        opts.cosmology.hubble = 1.0;
    }
    if (opts.initial.particlesPerDimension % 2 == 1) {
        THROW_ERROR("initial.particlesPerDimension must be an even number\n");
    }
    opts.gravity.supportRadius *= pow(opts.initial.numberOfParticles, -1.0 / NDIM);
    if (opts.initial.plummer) {
        opts.gravity.supportRadius = gravityTable.supportRadiusPerEffectiveRadius *
            1.5015907655628273 *
            pow((4.0 / 3.0 * M_PI * pow(opts.initial.plummerRadius, 3.0)) /
                    opts.initial.numberOfParticles,
                1.0 / 3.0);
    }
    if (opts.gravity.enabled && !detail::isPositiveFloatOption(opts.gravity.supportRadius)) {
        THROW_ERROR("gravity.effectiveRadius / gravity.supportRadius (gravity.softening) must "
                    "remain finite and positive in single precision "
                    "after normalization when gravity.enabled is true\n");
    }
    SHOW("chemistry.enabled", chemistry.enabled);
    SHOW("chemistry.network", chemistry.network);
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
    SHOW("run.hierarchicalGravity", run.hierarchicalGravity);
    SHOW("hydro.viscosityAlpha", hydro.viscosityAlpha);
    SHOW("hydro.viscosityBeta", hydro.viscosityBeta);
    SHOW("hydro.viscosityEpsilon", hydro.viscosityEpsilon);
    SHOW("hydro.gamma", hydro.gamma);
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
    PRINT("Simulation Options\n");
    PRINT("code_to_M_solar = %e\n", opts.units.massG / 1.98e33);
    PRINT(
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

    SHOW("checkpoint.intervalSeconds", checkpoint.intervalSeconds);
    SHOW("units.lengthCm", units.lengthCm);
    SHOW("units.massG", units.massG);
    SHOW("units.timeSeconds", units.timeSeconds);
    SHOW("compute.cuda", compute.cuda);
    SHOW("gravity.fmmOrder", gravity.fmmOrder);
    SHOW("compute.cudaKickResidency", compute.cudaKickResidency);
    SHOW("compute.cudaKickSinkTile", compute.cudaKickSinkTile);
    SHOW("compute.cudaKickParticleTile", compute.cudaKickParticleTile);
    SHOW("compute.cudaKickMultipoleTile", compute.cudaKickMultipoleTile);
    SHOW("compute.cudaKickSharedExpansion", compute.cudaKickSharedExpansion);
    SHOW("compute.cudaKickHandoffBlocks", compute.cudaKickHandoffBlocks);
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
        opts.gravity.supportRadius * std::pow(double(opts.initial.numberOfParticles), 1.0 / NDIM) /
            gravityTable.supportRadiusPerEffectiveRadius);
    show("gravity.effectiveRadius",
        opts.gravity.supportRadius * std::pow(double(opts.initial.numberOfParticles), 1.0 / NDIM) /
            gravityTable.supportRadiusPerEffectiveRadius);
    show("gravity.supportRadius",
        opts.gravity.supportRadius * std::pow(double(opts.initial.numberOfParticles), 1.0 / NDIM));
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
    SHOW("cosmology.omegaM", cosmology.omegaM);
    SHOW("cosmology.omegaNu", cosmology.omegaNu);
    SHOW("cosmology.omegaR", cosmology.omegaR);
    SHOW("compute.particleCacheLineSize", compute.particleCacheLineSize);
    SHOW("initial.particlesPerDimension", initial.particlesPerDimension);
    SHOW("checkpoint.readIndex", checkpoint.readIndex);
    SHOW("analysis.saveForce", analysis.saveForce);
    SHOW("cosmology.sigma8", cosmology.sigma8);
    SHOW("analysis.sliceResolution", analysis.sliceResolution);
    SHOW("analysis.sliceSize", analysis.sliceSize);
    SHOW("cosmology.thetaCmb", cosmology.thetaCmb);
    SHOW("gravity.openingAngle", gravity.openingAngle);
    SHOW("analysis.tracerCount", analysis.tracerCount);
    SHOW("compute.treeCacheLineSize", compute.treeCacheLineSize);
    SHOW("initial.twoLpt", initial.twoLpt);
    SHOW("initial.usePowerFile", initial.usePowerFile);
    SHOW("analysis.viewSize", analysis.viewSize);
    SHOW("cosmology.startRedshift", cosmology.startRedshift);
    SHOW("cosmology.endRedshift", cosmology.endRedshift);

    SHOW("cosmology.spectralIndex", cosmology.spectralIndex);
    SHOW("cosmology.heliumFraction", cosmology.heliumFraction);
    SHOW("cosmology.nEff", cosmology.nEff);
    SHOW("configuration", configuration);
    SHOW("run.test", run.test);
#ifndef USE_CUDA
    if (opts.compute.cuda) {
        THROW_ERROR("This executable was compiled without CUDA support\n");
    }
#endif
    setOptions(opts);
    viewConfigure();
    return rc;
}
