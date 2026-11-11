# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

"""Check the grouped CLI, INI precedence, and SPH parameter rejection."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import configparser
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

# Each case runs two HPX worker threads; ctest reserves PROCESSORS to match.
kConcurrentCases = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--near-method", choices=("fmm",), default="fmm",
                        help="fixed PMMM-FMM solver used by the fixtures")
    parser.add_argument("--order", type=int, choices=(0, *range(3, 10)), required=True,
                        help="compiled expansion order; 0 for an SPH-only build without gravity")
    parser.add_argument("--time-integrator", choices=("traditional", "hierarchical", "global"), required=True)
    parser.add_argument("--without-sph", action="store_true")
    parser.add_argument("--without-chemistry", action="store_true")
    parser.add_argument("--with-star-formation", action="store_true")
    parser.add_argument("--without-entropy-sources", action="store_true",
                        help="Compiled cooling/star capabilities cannot yet couple to SPH entropy")
    parser.add_argument("--concurrent-cases", type=int, choices=range(1, kConcurrentCases + 1),
                        default=kConcurrentCases, help="Concurrent two-thread application cases")
    parser.add_argument("--without-rockstar", action="store_true")
    parser.add_argument("--hydro-methods-only", action="store_true",
                        help="Check hydrodynamics method selection without the full option suite")
    parser.add_argument("--startup-options-only", action="store_true",
                        help="Check the complete option dump during normal nonverbose startup")
    parser.add_argument("--retired-viscosity-only", action="store_true",
                        help="Check that retired viscosity selectors are rejected")
    args = parser.parse_args()
    # PMMM is the only gravity solver; order zero marks an SPH-only build.
    gravity_compiled = args.order != 0
    if not gravity_compiled and (not args.startup_options_only or args.without_sph):
        parser.error("order=0 requires startup-options-only and SPH")
    chemistryAvailable = not args.without_sph and not args.without_chemistry
    starFormationAvailable = not args.without_sph and args.with_star_formation
    coupledSourcesAvailable = not args.without_entropy_sources
    args.workdir.mkdir(parents=True, exist_ok=True)
    unused_view = args.workdir / "view.txt"
    unused_view.unlink(missing_ok=True)
    for name in ("configured snapshots", "cli snapshots", "units startup", "output"):
        directory = args.workdir / name
        if directory.exists():
            shutil.rmtree(directory)
    command = [str(args.executable.resolve()), "--hpx:threads=2", "--hpx:bind=none"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")

    cuda_keys = ("gravity.cuda", "gravity.error.cuda", "hydro.cuda", "drift.cuda", "sort.cuda",
                 "initial.cuda", "analysis.power.cuda", "analysis.groups.cuda",
                 "analysis.tracers.cuda", "analysis.slice.cuda", "analysis.views.cuda",
                 "lightcone.cuda", "chemistry.cuda", "starFormation.cuda")
    cuda_tunings = ("kickResidency", "kickSinkTile", "kickParticleTile", "kickMultipoleTile",
                    "kickSharedExpansion", "kickHandoffBlocks", "kickFineHandoffBlocks",
                    "kickListReserve")

    hydro_rejected = False
    hydro_error = ""

    def run(name, options, success=True, pmmm_mesh=8, selectors=True):
        if selectors:
            selected = {}
            if not gravity_compiled:
                selected["gravity.enabled"] = "false"
            options = options + [f"--{key}={value}" for key, value in selected.items()
                                 if not any(option.startswith(f"--{key}=") for option in options)]
        # Most cases inspect unrelated options; supply the required mesh size.
        # Mesh/INI-specific cases can disable this fixture default explicitly.
        if (gravity_compiled and pmmm_mesh is not None and
                not any(option.startswith("--gravity.meshSize=") for option in options)):
            options = options + [f"--gravity.meshSize={pmmm_mesh}"]
        result = subprocess.run(command + options, cwd=args.workdir, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
                                env=environment)
        (args.workdir / f"{name}.log").write_text(result.stdout)
        if (result.returncode == 0) != success:
            raise RuntimeError(f"{name} returned {result.returncode}:\n{result.stdout}")
        if result.returncode == 0 and "--help" not in options:
            assertResolvedOptions(result.stdout)
        return result.stdout

    def runAll(cases):
        """Run independent (name, options, success) cases concurrently.

        Each launch is dominated by runtime startup, so overlapping a few of them
        shortens the suite; the outputs are returned in case order.
        """
        with ThreadPoolExecutor(args.concurrent_cases) as pool:
            return list(pool.map(lambda case: run(*case), cases))

    def assertResolvedOptions(output):
        if "Resolved options\n" not in output:
            raise RuntimeError("successful startup did not print its resolved options")
        names = []
        for line in output.split("Resolved options\n", 1)[1].splitlines():
            match = re.match(r"^([A-Za-z][A-Za-z0-9_.]*)\s*:", line)
            if match:
                names.append(match[1])
            elif names:
                break
        if names != sorted(names) or len(names) != len(set(names)):
            raise RuntimeError("resolved options were not alphabetical and unique")
        missing = option_names - set(names)
        if missing:
            raise RuntimeError(f"resolved output omitted registered options: {sorted(missing)}")

    def assertNumericSetting(output, key, expected):
        match = re.search(r"^" + re.escape(key) + r"\s*:\s*(\S+)", output, re.MULTILINE)
        if match is None or not math.isclose(float(match.group(1)), expected, rel_tol=1e-5):
            raise RuntimeError(f"incorrect {key}: expected {expected}")

    help_text = run("help", ["--help"])
    if not re.search(r"^\s*Time integrators:.*\b" + args.time_integrator + r"\b",
                     help_text, re.MULTILINE):
        raise RuntimeError("help did not identify the compiled time integrator")
    if not re.search(r"^\s*Equilibrium cooling: " + ("enabled" if chemistryAvailable else "disabled") + r"\s*$",
                     help_text, re.MULTILINE):
        raise RuntimeError("help did not identify the compiled equilibrium cooling capability")
    if not re.search(r"^\s*Star formation: " + ("enabled" if starFormationAvailable else "disabled") + r"\s*$",
                     help_text, re.MULTILINE):
        raise RuntimeError("help did not identify the compiled star formation capability")
    help_option_names = re.findall(r"^\s+--([A-Za-z][A-Za-z0-9_.]*)", help_text, re.MULTILINE)
    option_names = set(help_option_names)
    if help_option_names != sorted(option_names):
        raise RuntimeError("help options were not alphabetical and unique")
    required = {"help", "configuration", "run.scenario", "run.endTime", "run.verbose",
                "run.darkMatterParticleCount", "run.gasParticleCount", "compute.bucketSize",
                "compute.sphBucketSize", "compute.sphTreeRebuildFraction", "compute.positionType",
                "hydro.enabled", "hydro.method", "hydro.viscosityCoefficientMax", "hydro.vonNeumannRichtmyerCoefficient",
                "hydro.viscosityEpsilon", "hydro.conductionCoefficient",
                "hydro.neighborNumber", "gravity.enabled",
                "gravity.kernelFamily", "gravity.kernelDegree",
                "gravity.meshSize", "gravity.pmmmNearWidth", "gravity.pmmmCoarseMeshSize",
                "gravity.errorInterval", "gravity.errorSampleSize",
                "hydro.kernelFamily", "hydro.kernelDegree",
                "gravity.effectiveRadius", "gravity.smoothingLength",
                "hydro.minSmoothingLength", "hydro.maxSmoothingLength",
                "scenario.disc.centralEffectiveRadius", "scenario.disc.centralSmoothingLength",
                "scenario.rayleighTaylor.zPlaneAcceleration",
                "output.silo", "run.numberOfOutputs", "output.directory",
                "output.units", "output.frame",
                "units.massSolar", "units.boxLength",
                "output.xMin", "output.xMax", "output.yMin", "output.yMax",
                "output.zMin", "output.zMax", "chemistry.initializationTemperature", "chemistry.minimumSpecificEnergy",
                "starFormation.enabled", "starFormation.seed", "starFormation.criticalPhysicalDensity",
                "starFormation.criticalOverDensity",
                "starFormation.evaporationFactor", "starFormation.starFormationTimeGyr"}
    required.update({
                     "hydro.viscosityCoefficientMin",
                     "hydro.viscosityDecay", "hydro.balsaraSoundSpeedCoefficient"})
    required.update({"scenario.cosmology.gadget4Restart", "scenario.cosmology.useGlass",
                     "scenario.cosmology.twoLpt", "scenario.plummer.radius",
                     "scenario.kelvinHelmholz.packing"})
    required.update(cuda_keys)
    required.update({"cosmology.omegaLambda", "cosmology.startRedshift", "lightcone.enabled"})
    required.update("gravity.cuda." + key for key in cuda_tunings)
    if not required <= option_names:
        raise RuntimeError(f"grouped help is missing {sorted(required - option_names)}")
    # Deprecated aliases remain parseable but are not public keys: help, the
    # resolved listing and complete presets omit them.
    deprecated_aliases = {"gravity.softening": "gravity.effectiveRadius"}
    if set(deprecated_aliases) & option_names:
        raise RuntimeError(f"deprecated aliases remain in help: {sorted(set(deprecated_aliases) & option_names)}")

    sph_controls_available = not args.without_sph and not hydro_rejected
    # These fixtures exercise gravity-free midpoint SPH.
    sph_enabled_options = ["--hydro.enabled=true", "--gravity.enabled=false"]
    sph_control_options = sph_enabled_options if sph_controls_available else []

    def assertStartupOptions(output, hydroMethod):
        assertResolvedOptions(output)
        for key, value in (("run.verbose", "F"), ("hydro.method", hydroMethod)):
            if not re.search(r"^" + re.escape(key) + r"\s*:\s*" + value + r"\s*$",
                             output, re.MULTILINE):
                raise RuntimeError(f"normal startup did not print resolved {key}={value}")
        if "Simulation Options\n" in output or "code_to_M_solar =" in output:
            raise RuntimeError("nonverbose option dump enabled internal diagnostics")

    if args.startup_options_only:
        startupDirectory = args.workdir / "units startup"
        hydroMethod = "sph"
        startupPhysics = (["--run.scenario=hydroWave", "--hydro.enabled=true"]
                          if not gravity_compiled else ["--hydro.enabled=false"])
        startupOptions = [
            "--run.verbose=false", *startupPhysics,
            f"--hydro.method={hydroMethod}", "--run.darkMatterParticleCount=512", "--run.gasParticleCount=512",
            "--cosmology.sigma8=0", "--run.maxIterations=0", "--output.silo=false",
            "--analysis.power=false", "--analysis.groups=false", "--analysis.tracers=false",
            "--lightcone.enabled=false", f"--output.directory={startupDirectory.resolve()}",
            "--units.boxLength=20", "--compute.sphBucketSize=0", "--cosmology.omegaLambda=auto",
            "--scenario.disc.centralSmoothingLength=0.02", "--hydro.minSmoothingLength=0.001",
            "--hydro.maxSmoothingLength=1000", "--starFormation.seed=18446744073709551615",
            "--analysis.saveForce=true"]
        output = run("startup_options", startupOptions, selectors=False)
        if not re.search(r"^run\.timeIntegrator\s*:\s*hierarchical\s*$", output, re.MULTILINE):
            raise RuntimeError("the sole integration scheme was not selected automatically")
        for mode in ("hierarchical", "traditional", "global", "unknown"):
            rejected = run("startup_reject_" + mode,
                           startupOptions + [f"--run.timeIntegrator={mode}"],
                           success=False, selectors=False)
            if "unrecognized option" not in rejected or "run.timeIntegrator" not in rejected:
                raise RuntimeError(f"retired integrator {mode} lacked a clear diagnostic")
            retiredConfig = args.workdir / f"retired_integrator_{mode}.ini"
            retiredConfig.write_text(f"[run]\ntimeIntegrator = {mode}\n")
            rejected = run("startup_reject_ini_" + mode,
                           startupOptions + [f"--configuration={retiredConfig.resolve()}"],
                           success=False, selectors=False)
            if "unrecognized option" not in rejected or "timeIntegrator" not in rejected:
                raise RuntimeError(f"retired INI integrator {mode} lacked a clear diagnostic")
        if gravity_compiled:
            rejected = run("startup_reject_unsplit",
                           startupOptions + ["--gravity.pmmmOuterInterval=false"],
                           success=False, selectors=False)
            if "long/short gravity splitting is mandatory" not in rejected:
                raise RuntimeError("unsplit gravity lacked a clear diagnostic")
        assertStartupOptions(output, hydroMethod)
        assertNumericSetting(output, "compute.sphBucketSize", 106)
        dimension = int(re.search(r"^build\.dimensionCount\s*:\s*(\d+)", output, re.MULTILINE)[1])
        population = int(re.search(r"^initial\.numberOfParticles\s*:\s*(\d+)", output, re.MULTILINE)[1])
        # The half-box support cap is reported in reference particle spacings.
        assertNumericSetting(output, "hydro.maxSmoothingLength", 0.5 * population**(1 / dimension))
        assertNumericSetting(output, "units.boxLength", 20)
        assertNumericSetting(output, "scenario.disc.centralSmoothingLength", 0.02)
        assertNumericSetting(output, "hydro.minSmoothingLength", 0.001)
        for key, expected in (("starFormation.seed", "18446744073709551615"),
                              ("analysis.saveForce", "T"), ("analysis.saveAcceleration", "F"),
                              ("analysis.savePotential", "F"), ("help", "F")):
            if not re.search(r"^" + re.escape(key) + r"\s*:\s*" + expected + r"\s*$",
                             output, re.MULTILINE):
                raise RuntimeError(f"startup did not print the resolved {key}")
        omegaLambda = re.search(r"^cosmology\.omegaLambda\s*:\s*(\S+)", output, re.MULTILINE)
        if omegaLambda is None or not math.isfinite(float(omegaLambda[1])):
            raise RuntimeError("startup printed the unresolved automatic cosmological constant")
        print("Nonverbose startup printed every registered option alphabetically with resolved values")
        return

    removed_options = {"gravity.method.near", "run.timeIntegrator", "run.numberOfSteps", "output.siloInterval", "compute.cuda", "hydro.chemistry", "hydro.conduction", "hydro.diffusion",
                       "hydro.yReflect", "gravity.sphSmoothing", "cosmology.omegaK",
                       "initial.usePowerFile", "initial.particlesPerDimension",
                       "initial.darkMatterParticleCount", "initial.gasParticleCount", "gravity.method",
                       "gravity.method.far", "gravity.cloud", "gravity.p3mSplitCells",
                       "gravity.p3mCutoff", "gravity.openingAngle", "gravity.openingSchedule",
                       "gravity.pmmmNearMethod", "gravity.pmmmOrder", "gravity.fmmOrder", "gravity.expansionDegree",
                       "run.hierarchicalGravity", "initial.createGlass", "initial.plummer",
                       "hydro.pressureOperator", "hydro.balsaraSwitch", "chemistry.network", "chemistry.initialization",
                       "chemistry.initialTemperature", "chemistry.radiationFile", "chemistry.radiationTable",
                       "chemistry.hydrogenIonFraction", "chemistry.heliumIonFraction",
                       "chemistry.heliumDoubleIonFraction", "chemistry.hydrogenMinusFraction",
                       "chemistry.molecularFraction", "chemistry.molecularIonFraction",
                       "chemistry.fractionalChangeLimit", "chemistry.electronFractionFloor",
                       "chemistry.maxSubsteps", "chemistry.synchronizedRung",
                       "starFormation.metalYield", "starFormation.winds",
                       "starFormation.windMassLoading", "starFormation.windEnergyFraction",
                       "starFormation.windDecouplingMyr", "starFormation.windRecoupleDensityFraction",
                       "starFormation.windDirection", "hydro.dualEnergyPressureSwitch",
                       "hydro.dualEnergyUpdateSwitch", "hydro.treeRebuildActiveFraction",
                       "hydro.viscosityAlpha", "hydro.viscosityEvolution",
                       "hydro.minEffectiveRadius", "hydro.maxEffectiveRadius"}
    removed_options.update({"initial.gadget4Restart", "initial.useGlass", "initial.twoLpt", "initial.plummerRadius",
                            "compute.sphGhosts"})
    removed_options.update("compute.cuda" + key[0].upper() + key[1:] for key in cuda_tunings)
    if removed_options & option_names:
        raise RuntimeError(f"removed options remain in help: {sorted(removed_options & option_names)}")
    for name in option_names - {"help", "configuration"}:
        if not re.fullmatch(r"[a-z][A-Za-z0-9]*(?:\.[a-z][A-Za-z0-9]*)+", name):
            raise RuntimeError(f"option does not use groupName.optionName: {name}")

    # Mix section-style and dotted keys. Choosing a checkpoint serialization
    # regression exercises full option setup without evolving a simulation.
    config = args.workdir / "grouped.ini"
    config.write_text("""analysis.tracerCount = 41
[run]
test = checkpointState
maxIterations = 7
darkMatterParticleCount = 512
gasParticleCount = 512
[compute]
bucketSize = 64
sphBucketSize = 73
sphTreeRebuildFraction = 0.5
[hydro]
enabled = false
viscosityCoefficientMax = 0.75
vonNeumannRichtmyerCoefficient = 1.5
viscosityEpsilon = 0.02
conductionCoefficient = 0.3
viscosityCoefficientMin = 0.15
viscosityDecay = 0.3
heatCapacityRatio = 1.4
neighborNumber = 48
cfl = 0.3
[scenario.rayleighTaylor]
zPlaneAcceleration = 0.25
[lightcone]
mapSize = 8
""")
    configured = run("config", [f"--configuration={config.resolve()}",
                                "--run.maxIterations=3", "--hydro.viscosityCoefficientMax=1.25"] + sph_control_options)
    expected = {"run.maxIterations": 3.0, "run.darkMatterParticleCount": 512,
                "run.gasParticleCount": 512 if sph_controls_available else 0,
                "analysis.tracerCount": 41.0, "hydro.viscosityCoefficientMax": 1.25,
                "hydro.vonNeumannRichtmyerCoefficient": 1.5, "hydro.viscosityEpsilon": 0.02,
                "hydro.conductionCoefficient": 0.3,
                "hydro.viscosityCoefficientMin": 0.15, "hydro.viscosityDecay": 0.3,
                "hydro.heatCapacityRatio": 1.4, "hydro.neighborNumber": 48.0,
                "hydro.cfl": 0.3, "scenario.rayleighTaylor.zPlaneAcceleration": 0.25, "lightcone.mapSize": 8.0,
                "hydro.minSmoothingLength": 1.0e-6, "hydro.maxSmoothingLength": 4.0,
                "gravity.kernelDegree": 4.0,
                "compute.bucketSize": 64.0, "compute.sphBucketSize": 73.0,
                "compute.sphTreeRebuildFraction": 0.5}
    for name, value in expected.items():
        match = re.search(r"^" + re.escape(name) + r"\s*:\s*(\S+)", configured, re.MULTILINE)
        if match is None or abs(float(match.group(1)) - value) > 1.0e-6:
            raise RuntimeError(f"configured {name} did not equal {value}")
    if "Checkpoint state and mode validation passed" not in configured:
        raise RuntimeError("configuration run did not finish the checkpoint regression")

    metadata_options = ["--run.test=checkpointState",
                        "--hydro.enabled=false", "--run.darkMatterParticleCount=512", "--run.gasParticleCount=512"]

    if args.retired_viscosity_only:
        # The Rosswog-limited Morris-Monaghan model is the only viscosity; no selector remains.
        for selector in ("none", "morrisAndMonaghan", "huEtal", "priceEtAl", "cullenAndDehnen"):
            rejected = run("retired_viscosity_selector_" + selector, metadata_options + [
                "--hydro.viscosityEvolution=" + selector], success=False)
            if "hydro.viscosityEvolution" not in rejected:
                raise RuntimeError(f"retired selector {selector} lacked a diagnostic")
        print("Retired viscosity selectors rejected")
        return

    for key in cuda_keys:
        if not re.search(r"^" + re.escape(key) + r"\s*:\s*F\s*$", configured, re.MULTILINE):
            raise RuntimeError(f"{key} did not default to CPU execution")
    for key in ("sort.cuda", "analysis.tracers.cuda", "analysis.slice.cuda", "analysis.views.cuda"):
        rejected = run("unsupported_" + key, metadata_options + [f"--{key}=true"], success=False)
        if key not in rejected or "not implemented" not in rejected:
            raise RuntimeError(f"unsupported {key} did not fail explicitly")

    def assertHydroMethod(output, method):
        if not re.search(r"^hydro\.method\s*:\s*" + method + r"\s*$", output, re.MULTILINE):
            raise RuntimeError(f"hydro.method did not resolve to {method}")
        assertResolvedOptions(output)

    assertHydroMethod(configured, "sph")
    output = run("hydro_method_sph", metadata_options + ["--hydro.method=sph"])
    assertHydroMethod(output, "sph")
    method_config = args.workdir / "hydro_method.ini"
    method_config.write_text("[hydro]\nmethod = sph\n")
    output = run("hydro_method_ini", metadata_options + [f"--configuration={method_config.resolve()}"])
    assertHydroMethod(output, "sph")
    invalid_method_config = args.workdir / "hydro_method_invalid.ini"
    invalid_method_config.write_text("[hydro]\nmethod = unknown\n")
    for label, options in (("cli", ["--hydro.method=unknown"]),
                           ("ini", [f"--configuration={invalid_method_config.resolve()}"])):
        rejected = run("hydro_method_invalid_" + label, metadata_options + options, success=False)
        if "hydro.method must be sph" not in rejected:
            raise RuntimeError("invalid hydro.method lacked the allowed-method diagnostic")

    position_type = re.search(r"^compute\.positionType\s*:\s*(fixed32|fixed64)\s*$",
                              configured, re.MULTILINE)[1]
    compiled_integrators = re.search(r"^\s*Time integrators:([^\n]*)", help_text, re.MULTILINE)[1].split()
    if args.hydro_methods_only:
        print("Hydrodynamics method validation passed")
        return

    for key, value in (("gravity.method.near", "unknown"),
                       ("run.timeIntegrator", "unknown"),
                       ("run.timeIntegrator", "traditional"),
                       ("run.timeIntegrator", "global"),
                       ("gravity.pmmmOuterInterval", "false"),
                       ("gravity.expansionDegree", "1"),
                       ("gravity.expansionDegree", "9")):
        output = run("selection_invalid_" + key + "_" + value,
                     metadata_options + [f"--{key}={value}"], success=False)
        if key not in output:
            raise RuntimeError(f"invalid {key} lacked a selection diagnostic")
    degree = re.search(r"Potential expansion degree: (\d+)", help_text)
    if degree is None or int(degree[1]) != args.order - 1:
        raise RuntimeError("help omitted the build's single potential degree")
    selected_options = []
    selection_config = args.workdir / "algorithm_selection.ini"
    selection_config.write_text("[gravity]\nexpansionDegree = 6\n")
    output = run("removed_degree_ini", metadata_options + [
        f"--configuration={selection_config.resolve()}"], success=False, selectors=False)
    if "expansionDegree" not in output:
        raise RuntimeError("removed INI degree lacked a diagnostic")
    for key, label, choices in (
            ("run.timeIntegrator", "Time integrators", ("traditional", "hierarchical", "global")),):
        compiled = re.search(r"^\s*" + label + r":([^\n]*)", help_text, re.MULTILINE)
        count = sum(choice in compiled[1].split() for choice in choices) if compiled else 0
        arguments = [option for option in selected_options if not option.startswith(f"--{key}=")]
        output = run("selection_implicit_" + key, metadata_options + arguments,
                     success=count == 1, selectors=False)
        if count > 1 and key not in output:
            raise RuntimeError(f"ambiguous {key} lacked an explicit-selection diagnostic")
    compiled_near = re.search(r"^\s*PMMM near solvers:([^\n]*)", help_text, re.MULTILINE)
    if compiled_near is None or compiled_near[1].split() != ["fmm"]:
        raise RuntimeError("help did not identify PMMM-FMM as the sole gravity solver")
    for source in ("cli", "ini"):
        for value in ("fmm", "pp"):
            retired = args.workdir / f"retired_near_{value}.ini"
            retired.write_text(f"[gravity]\nmethod.near = {value}\n")
            arguments = ([f"--gravity.method.near={value}"] if source == "cli" else
                         [f"--configuration={retired.resolve()}"])
            output = run(f"retired_near_{source}_{value}", metadata_options + arguments, False)
            if "gravity.method.near" not in output and "method.near" not in output:
                raise RuntimeError("retired near solver lacked a diagnostic")

    # A comoving Mpc/h box contains rho_m,0 * L^3. Resolution changes the mass
    # unit, and close packing doubles the reference count at fixed resolution.
    # The constants are CGS values from cosmictiger/math/constants.hpp.
    normalizationOptions = metadata_options + ["--cosmology.hubble=0.7",
                                               "--cosmology.omegaB=0.05",
                                               "--cosmology.omegaC=0.25"]
    mpcToCm = 3.085678e24
    solarMassG = 1.989e33
    # Match kH0 = 1.e7 / kMpcToCm; a separately rounded Mpc changes rho by 2e-4.
    matterDensity = 0.3 * 3.0 * (0.7 * 1.e7 / mpcToCm)**2 / (8.0 * math.pi * 6.67259e-8)
    boxLength = 100.0
    massSolar = matterDensity * (boxLength * mpcToCm / 0.7)**3 / (8**3 * solarMassG)

    def assertNormalization(output, length, mass):
        assertNumericSetting(output, "units.boxLength", length)
        assertNumericSetting(output, "units.massSolar", mass)
        assertNumericSetting(output, "units.lengthCm", length * mpcToCm / 0.7)
        assertNumericSetting(output, "units.massG", mass * solarMassG)

    normalized = run("units_box_cli", normalizationOptions + [f"--units.boxLength={boxLength}"])
    assertNormalization(normalized, boxLength, massSolar)
    normalized = run("units_mass_inverse", normalizationOptions + [f"--units.massSolar={massSolar:.17g}"])
    assertNormalization(normalized, boxLength, massSolar)
    # A small homogeneous lattice skips the Einstein-Boltzmann solve while
    # exercising the normal successful startup and its complete nonverbose dump.
    startupDirectory = args.workdir / "units startup"
    startupOptions = [option for option in normalizationOptions
                      if not option.startswith("--run.test=")]
    startupOutput = run("units_normal_startup", startupOptions + [
        "--run.verbose=false", f"--units.boxLength={boxLength}",
        "--cosmology.sigma8=0", "--run.maxIterations=0", "--output.silo=false",
        "--analysis.power=false", "--analysis.groups=false", "--analysis.tracers=false",
        "--lightcone.enabled=false", f"--output.directory={startupDirectory.resolve()}"])
    if "Run configuration" not in startupOutput:
        raise RuntimeError("normal startup did not print its summary")
    assertStartupOptions(startupOutput, "sph")
    assertNumericSetting(startupOutput, "units.boxLength", boxLength)
    assertNumericSetting(startupOutput, "units.massSolar", massSolar)
    assertNumericSetting(startupOutput, "initial.numberOfParticles", 8**3)
    boxSummary = re.search(r"^units\.boxLength\s*:\s*\S+ Mpc/h \((\S+) Mpc comoving\)$",
                           startupOutput, re.MULTILINE)
    if boxSummary is None or not math.isclose(float(boxSummary.group(1)), boxLength / 0.7,
                                             rel_tol=1e-5):
        raise RuntimeError("normal startup lacked comoving box units and Mpc conversion")
    if not re.search(r"^units\.massSolar\s*:\s*\S+ Msun$", startupOutput, re.MULTILINE):
        raise RuntimeError("normal startup lacked solar-mass units")
    defaultMass = 3.58e9
    defaultLength = (8**3 * defaultMass * solarMassG / matterDensity)**(1.0 / 3.0) * 0.7 / mpcToCm
    assertNormalization(run("units_default_mass", normalizationOptions), defaultLength, defaultMass)
    for resolution, closePack in ((16, False), (8, True), (16, True)):
        count = resolution**3 * (2 if closePack else 1)
        options = [option for option in normalizationOptions
                   if not option.startswith(("--run.darkMatterParticleCount=", "--run.gasParticleCount="))]
        normalized = run(f"units_box_resolution_{resolution}_close_pack_{closePack}", options + [
            f"--run.darkMatterParticleCount={count}", f"--run.gasParticleCount={count}",
            f"--initial.closePack={str(closePack).lower()}", f"--units.boxLength={boxLength}"])
        assertNormalization(normalized, boxLength, massSolar * 8**3 / count)
    if sph_controls_available:
        options = [option for option in normalizationOptions
                   if not option.startswith("--hydro.enabled=")]
        normalized = run("units_box_sph_combined_mass", options + sph_enabled_options + [
            f"--units.boxLength={boxLength}"])
        assertNormalization(normalized, boxLength, massSolar)

    normalizationConfig = args.workdir / "normalization.ini"
    for key, value in (("boxLength", boxLength), ("massSolar", massSolar)):
        normalizationConfig.write_text(f"[units]\n{key} = {value:.17g}\n")
        normalized = run(f"units_{key}_ini", normalizationOptions + [
            f"--configuration={normalizationConfig.resolve()}"])
        assertNormalization(normalized, boxLength, massSolar)
        # Ignore an invalid INI value when the same option is replaced on CLI.
        normalizationConfig.write_text(f"[units]\n{key} = -1\n")
        normalized = run(f"units_{key}_cli_override", normalizationOptions + [
            f"--configuration={normalizationConfig.resolve()}", f"--units.{key}={value:.17g}"])
        assertNormalization(normalized, boxLength, massSolar)
        invalidUnitsCases = []
        for index, invalid in enumerate(("0", "-1", "nan", "inf", "-inf")):
            invalidConfig = args.workdir / f"normalization_{key}_invalid_{index}.ini"
            invalidConfig.write_text(f"[units]\n{key} = {invalid}\n")
            for source, arguments in (
                    ("cli", [f"--units.{key}={invalid}"]),
                    ("ini", [f"--configuration={invalidConfig.resolve()}"])):
                invalidUnitsCases.append((f"units_{key}_invalid_{source}_{index}",
                                          normalizationOptions + arguments, False))
        for rejected in runAll(invalidUnitsCases):
            if f"units.{key}" not in rejected:
                raise RuntimeError(f"invalid units.{key} lacked its option diagnostic")

    for value in ("1e308", "1e-150"):
        rejected = run(f"units_box_out_of_range_{value}", normalizationOptions + [
            f"--units.boxLength={value}"], success=False)
        if "units.boxLength produces out-of-range simulation units" not in rejected:
            raise RuntimeError("unrepresentable box normalization lacked its range diagnostic")
    for name, option, diagnostic in (
            ("glass", "--run.scenario=glass",
             "units.boxLength cannot be used with glass generation or Gadget4 input, "
             "which supply their own length scale"),
            ("gadget", "--scenario.cosmology.gadget4Restart=missing.gadget",
             "scenario.cosmology.gadget4Restart supplies its own units; do not set units.boxLength")):
        rejected = run(f"units_box_unsupported_{name}", normalizationOptions + [
            option, f"--units.boxLength={boxLength}"], success=False)
        # Gadget input rejects the unit conflict before attempting to read its header.
        if diagnostic not in rejected:
            raise RuntimeError(f"box normalization with {name} lacked its unit-conflict diagnostic")

    for source in ("cli", "ini", "box_cli", "mass_cli"):
        if source == "cli":
            arguments = [f"--units.boxLength={boxLength}", f"--units.massSolar={massSolar:.17g}"]
        else:
            lines = ["[units]"]
            arguments = [f"--configuration={normalizationConfig.resolve()}"]
            for key, value in (("boxLength", boxLength), ("massSolar", massSolar)):
                onCli = (source == "box_cli" and key == "boxLength" or
                         source == "mass_cli" and key == "massSolar")
                if onCli:
                    arguments.append(f"--units.{key}={value:.17g}")
                else:
                    lines.append(f"{key} = {value:.17g}")
            normalizationConfig.write_text("\n".join(lines) + "\n")
        rejected = run(f"units_conflicting_{source}", normalizationOptions + arguments, success=False)
        if "units.massSolar" not in rejected or "units.boxLength" not in rejected:
            raise RuntimeError("conflicting normalization options lacked both option names")

    error_defaults = run("gravity_error_defaults", metadata_options)
    assertNumericSetting(error_defaults, "gravity.errorInterval", 0)
    assertNumericSetting(error_defaults, "gravity.errorSampleSize", 100)
    error_config = args.workdir / "gravity_error.ini"
    error_config.write_text("[gravity]\nerrorInterval = 3\nerrorSampleSize = 7\n")
    error_configured = run("gravity_error_config", metadata_options + [
        f"--configuration={error_config.resolve()}"])
    assertNumericSetting(error_configured, "gravity.errorInterval", 3)
    assertNumericSetting(error_configured, "gravity.errorSampleSize", 7)
    error_override = run("gravity_error_cli_override", metadata_options + [
        f"--configuration={error_config.resolve()}", "--gravity.errorInterval=2",
        "--gravity.errorSampleSize=11"])
    assertNumericSetting(error_override, "gravity.errorInterval", 2)
    assertNumericSetting(error_override, "gravity.errorSampleSize", 11)
    for label, options, diagnostic in (
            ("interval", ["--gravity.errorInterval=-1"],
             "gravity.errorInterval must be nonnegative"),
            ("empty", ["--gravity.errorInterval=1", "--gravity.errorSampleSize=0"],
             "gravity.errorSampleSize must be positive"),
            ("negative", ["--gravity.errorInterval=1", "--gravity.errorSampleSize=-2"],
             "gravity.errorSampleSize must be positive"),
            ("disabled_gravity", ["--gravity.errorInterval=1", "--gravity.enabled=false"],
             "gravity.errorInterval requires gravity.enabled=true")):
        rejected = run(f"gravity_error_reject_{label}", metadata_options + options, False)
        if diagnostic not in rejected:
            raise RuntimeError(f"{label}: missing gravity-error option diagnostic")
    error_disabled = run("gravity_error_disabled", metadata_options + [
        "--gravity.errorInterval=0", "--gravity.errorSampleSize=0"])
    assertNumericSetting(error_disabled, "gravity.errorInterval", 0)

    balsara_options = ([option for option in metadata_options if not option.startswith("--hydro.enabled=")]
                       + sph_control_options if sph_controls_available else metadata_options)
    balsara_default = run("balsara_default", balsara_options)
    def assertCompiledSettings(output):
        for key, value in (("run.timeIntegrator", args.time_integrator),):
            if not re.search(r"^" + re.escape(key) + r"\s*:\s*" + value + r"\s*$",
                             output, re.MULTILINE):
                raise RuntimeError(f"compiled metadata did not retain {key}={value}")
        if not re.search(r"^build\.chemistry\s*:\s*" + ("T" if chemistryAvailable else "F") + r"\s*$",
                         output, re.MULTILINE):
            raise RuntimeError("build metadata did not identify the compiled equilibrium cooling capability")
        if not re.search(r"^build\.starFormation\s*:\s*" + ("T" if starFormationAvailable else "F") + r"\s*$",
                         output, re.MULTILINE):
            raise RuntimeError("build metadata did not identify the compiled star formation capability")
        expected_hierarchical = "T" if re.search(
            r"^gravity\.enabled\s*:\s*T\s*$", output, re.MULTILINE) else "F"
        if not re.search(r"^run\.hierarchicalGravity\s*:\s*" + expected_hierarchical + r"\s*$",
                         output, re.MULTILINE):
            raise RuntimeError("hierarchical gravity did not match enabled self-gravity")

    assertCompiledSettings(balsara_default)
    assertCompiledSettings(configured)
    for scenario in ("plummer", "glass"):
        output = run(f"integrator_{scenario}", metadata_options + [f"--run.scenario={scenario}"])
        assertCompiledSettings(output)
    run("gravity_negative_mesh", metadata_options + ["--gravity.meshSize=-1"], success=False)
    if gravity_compiled:
        # All positive mesh sizes are accepted, including smaller near cubes.
        minimum_mesh = {1: 1, 2: 2, 3: 3, 4: 4}
        for near_width, mesh_size in minimum_mesh.items():
            pmmm_output = run(f"pmmm_metadata_{near_width}", metadata_options + [
                f"--gravity.meshSize={mesh_size}", f"--gravity.pmmmNearWidth={near_width}"])
            assertCompiledSettings(pmmm_output)
            for key, expected in (("gravity.meshSize", mesh_size),
                                  ("gravity.pmmmNearWidth", near_width),
                                  ("gravity.pmmmCoarseMeshSize", mesh_size)):
                assertNumericSetting(pmmm_output, key, expected)
            assertNumericSetting(pmmm_output, "gravity.openingAngle", math.sqrt(3) / (near_width + 1))
        # balsara_default runs SPH without gravity, which leaves the coarse mesh
        # unresolved; read the PMMM defaults from the gravity-only fixture.
        assertNumericSetting(error_defaults, "gravity.pmmmNearWidth", 2)
        assertNumericSetting(error_defaults, "gravity.pmmmBatchParticles", 4096)
        assertNumericSetting(error_defaults, "gravity.pmmmCoarseMeshSize", 8)
        assertNumericSetting(error_defaults, "gravity.openingAngle", math.sqrt(3) / 3)
        # Zero resolves to meshSize/8, never below the near-width minimum and
        # never above the mesh itself.
        for mesh_size, near_width, expected in ((64, None, 8), (8, None, 8),
                                                (16, 4, 16), (32, 1, 4), (7, 2, 7),
                                                (12, 2, 12), (30, 2, 10), (120, 2, 15)):
            width_options = [] if near_width is None else [f"--gravity.pmmmNearWidth={near_width}"]
            output = run(f"pmmm_coarse_default_{mesh_size}_{near_width}", metadata_options + [
                f"--gravity.meshSize={mesh_size}"] + width_options)
            assertNumericSetting(output, "gravity.meshSize", mesh_size)
            assertNumericSetting(output, "gravity.pmmmCoarseMeshSize", expected)
        for coarse in (1, 4, 8, 16, 64):
            output = run(f"pmmm_coarse_cli_{coarse}", metadata_options + [
                "--gravity.meshSize=64", f"--gravity.pmmmCoarseMeshSize={coarse}"])
            assertNumericSetting(output, "gravity.pmmmCoarseMeshSize", coarse)
        coarse_config = args.workdir / "pmmm_coarse_mesh.ini"
        coarse_config.write_text("[gravity]\nmeshSize = 64\npmmmCoarseMeshSize = 16\n")
        output = run("pmmm_coarse_ini", metadata_options + [
            f"--configuration={coarse_config.resolve()}"], pmmm_mesh=None)
        assertNumericSetting(output, "gravity.meshSize", 64)
        assertNumericSetting(output, "gravity.pmmmCoarseMeshSize", 16)
        output = run("pmmm_coarse_cli_override", metadata_options + [
            f"--configuration={coarse_config.resolve()}", "--gravity.pmmmCoarseMeshSize=32"],
            pmmm_mesh=None)
        assertNumericSetting(output, "gravity.pmmmCoarseMeshSize", 32)
        # Ignore an invalid INI value when the same option is replaced on CLI.
        coarse_config.write_text("[gravity]\nmeshSize = 64\npmmmCoarseMeshSize = 12\n")
        output = run("pmmm_coarse_invalid_ini_cli_override", metadata_options + [
            f"--configuration={coarse_config.resolve()}", "--gravity.pmmmCoarseMeshSize=16"],
            pmmm_mesh=None)
        assertNumericSetting(output, "gravity.pmmmCoarseMeshSize", 16)
        rejected = run("pmmm_coarse_invalid_ini", metadata_options + [
            f"--configuration={coarse_config.resolve()}"], success=False, pmmm_mesh=None)
        if "gravity.pmmmCoarseMeshSize must be a positive divisor" not in rejected:
            raise RuntimeError("INI coarse mesh rejection lacked its option diagnostic")
        coarse_range = "gravity.pmmmCoarseMeshSize must be a positive divisor of gravity.meshSize"
        for label, options, diagnostic in (
                ("mesh_overflow", ["--gravity.meshSize=2147483647"],
                 "gravity.meshSize cubed exceeds mesh indexing/allocation limits"),
                ("mesh_allocation", ["--gravity.meshSize=1073741824"],
                 "gravity.meshSize cubed exceeds mesh indexing/allocation limits"),
                ("coarse_3", ["--gravity.meshSize=64", "--gravity.pmmmCoarseMeshSize=3"],
                 coarse_range),
                ("coarse_12", ["--gravity.meshSize=64", "--gravity.pmmmCoarseMeshSize=12"],
                 coarse_range),
                ("coarse_above_mesh", ["--gravity.meshSize=16", "--gravity.pmmmCoarseMeshSize=32"],
                 coarse_range),
                ("coarse_negative", ["--gravity.meshSize=64", "--gravity.pmmmCoarseMeshSize=-1"],
                 "gravity.pmmmCoarseMeshSize must be nonnegative")):
            rejected = run(f"pmmm_coarse_reject_{label}", metadata_options + options,
                           success=False)
            if diagnostic not in rejected:
                raise RuntimeError(f"{label}: missing PMMM mesh diagnostic: {diagnostic}")
        # The support radius is normalized by the 8**3 fixture; width/mesh = 0.25.
        radius_options = metadata_options + ["--gravity.meshSize=8"]
        run("pmmm_support_within_width", radius_options + ["--gravity.smoothingLength=1.9"])
        rejected = run("pmmm_support_beyond_width", radius_options + [
            "--gravity.smoothingLength=2.1"], success=False)
        if ("PMMM requires the normalized gravity.smoothingLength to be no larger than "
                "gravity.pmmmNearWidth/gravity.meshSize") not in rejected:
            raise RuntimeError("PMMM support validation lacked its near-width diagnostic")
        for scenario in ("cosmology", "plummer"):
            # The near-field angle is derived from the neighbor cube for every scenario.
            pmmm_output = run(f"pmmm_derived_angle_{scenario}", metadata_options + [
                f"--run.scenario={scenario}"])
            assertCompiledSettings(pmmm_output)
            assertNumericSetting(pmmm_output, "gravity.openingAngle", math.sqrt(3) / 3)
    if gravity_compiled:
        for index, invalid in enumerate((
                [],
                ["--gravity.meshSize=8", "--gravity.pmmmNearWidth=0"],
                ["--gravity.meshSize=16", "--gravity.pmmmNearWidth=5"],
                ["--gravity.meshSize=8", "--gravity.smoothingLength=8"],
                ["--gravity.meshSize=8", "--gravity.pmmmBatchParticles=0"],
                ["--gravity.meshSize=8", "--gravity.pmmmBatchParticles=-1"],
                ["--gravity.meshSize=2147483647"],
                ["--gravity.meshSize=1073741824"])):
            run(f"pmmm_invalid_{index}", metadata_options + invalid,
                success=False, pmmm_mesh=None)
        pmmm_config = args.workdir / "pmmm.ini"
        pmmm_config.write_text("[gravity]\nmeshSize = 16\npmmmNearWidth = 2\n")
        pmmm_output = run("pmmm_ini_cli", metadata_options + [
            f"--configuration={pmmm_config.resolve()}", "--gravity.pmmmNearWidth=1"],
            pmmm_mesh=None)
        assertCompiledSettings(pmmm_output)
        assertNumericSetting(pmmm_output, "gravity.meshSize", 16)
        # Width one lowers the minimum coarse mesh to four, above meshSize/8.
        assertNumericSetting(pmmm_output, "gravity.pmmmCoarseMeshSize", 4)
        assertNumericSetting(pmmm_output, "gravity.pmmmNearWidth", 1)
        assertNumericSetting(pmmm_output, "gravity.openingAngle", math.sqrt(3) / 2)

    sph_options = [option for option in metadata_options
                   if not option.startswith("--hydro.enabled=")]
    if args.without_sph:
        sph_build_error = "COSMICTIGER_ENABLE_SPH=ON"
        for gravity_enabled in ("true", "false"):
            rejected = run(f"disabled_sph_cli_gravity_{gravity_enabled}", sph_options + [
                "--hydro.enabled=true", f"--gravity.enabled={gravity_enabled}"], success=False)
            if sph_build_error not in rejected:
                raise RuntimeError("disabled SPH CLI request lacked build guidance")
        sph_config = args.workdir / "disabled_sph.ini"
        sph_config.write_text("[hydro]\nenabled = true\n")
        rejected = run("disabled_sph_ini", sph_options + [
            f"--configuration={sph_config.resolve()}"], success=False)
        if sph_build_error not in rejected:
            raise RuntimeError("disabled SPH INI request lacked build guidance")
        rejected = run("disabled_sph_scenario", [
            "--run.scenario=sod", "--run.maxIterations=0",
            "--run.darkMatterParticleCount=512", "--run.gasParticleCount=512"], success=False)
        if sph_build_error not in rejected:
            raise RuntimeError("disabled SPH scenario request lacked build guidance")
        forced_sph = [option for option in metadata_options
                      if not option.startswith("--run.test=")]
        rejected = run("disabled_sph_test", forced_sph + [
            "--run.test=sphGravity"], success=False)
        if sph_build_error not in rejected:
            raise RuntimeError("disabled SPH regression request lacked build guidance")
        rejected = run("disabled_sph_chemistry", metadata_options + [
            "--chemistry.enabled=true"], success=False)
        if sph_build_error not in rejected:
            raise RuntimeError("disabled SPH chemistry request lacked build guidance")
    else:
        for gravity_enabled in ("true", "false"):
            # The sole scheme supports SPH with and without self-gravity.
            rejected_here = hydro_rejected
            output = run(f"sph_cli_gravity_{gravity_enabled}", sph_options + [
                "--hydro.enabled=true", f"--gravity.enabled={gravity_enabled}"],
                success=not rejected_here)
            if rejected_here:
                if hydro_rejected and hydro_error not in output:
                    raise RuntimeError("hierarchical hydro rejection lacked guidance")
                continue
            assertCompiledSettings(output)
            if not re.search(r"^hydro\.enabled\s*:\s*T\s*$", output, re.MULTILINE):
                raise RuntimeError("SPH CLI request was not enabled")
        sph_config = args.workdir / "sph_allowed.ini"
        sph_config.write_text("[gravity]\nenabled = false\nmeshSize = 8\n"
                              "[hydro]\nenabled = true\n")
        output = run("sph_ini_allowed", sph_options + [
            f"--configuration={sph_config.resolve()}"], success=not hydro_rejected)
        if hydro_rejected:
            if hydro_error not in output:
                raise RuntimeError("hierarchical hydro INI rejection lacked guidance")
        else:
            assertCompiledSettings(output)
            if not re.search(r"^hydro\.enabled\s*:\s*T\s*$", output, re.MULTILINE):
                raise RuntimeError("SPH INI request was not enabled")
    if sph_controls_available and not coupledSourcesAvailable:
        for field in ("chemistry.enabled", "starFormation.enabled"):
            output = run("midpoint_source_rejected_" + field, sph_options + [
                "--hydro.enabled=true", "--gravity.enabled=true", f"--{field}=true"], success=False)
            expected = "Midpoint SPH does not yet support chemistry or star formation"
            if field == "chemistry.enabled" and not chemistryAvailable:
                expected = "COSMICTIGER_ENABLE_CHEMISTRY=ON"
            if expected not in output:
                raise RuntimeError("midpoint deferred source rejection lacked guidance")
    # Pressure is always variational; neither old selector value is accepted.
    for operator in ("monaghan", "variational"):
        rejected = run("pressure_operator_removed_cli_" + operator,
                       ["--hydro.pressureOperator=" + operator], success=False)
        if "hydro.pressureOperator" not in rejected:
            raise RuntimeError("removed pressure selector lacked a CLI diagnostic")
        removed_config = args.workdir / f"pressure_operator_removed_{operator}.ini"
        removed_config.write_text(f"[hydro]\npressureOperator = {operator}\n")
        rejected = run("pressure_operator_removed_ini_" + operator,
                       [f"--configuration={removed_config.resolve()}"], success=False)
        if "pressureOperator" not in rejected:
            raise RuntimeError("removed pressure selector lacked an INI diagnostic")
    for key in ("hydro.viscosityAlpha", "hydro.viscosityEvolution", "hydro.viscosityDecayCD",
                "hydro.viscosityAlphaMaxPrice", "hydro.viscosityDecayPrice",
                "hydro.viscosityAlphaMaxHu", "hydro.viscosityDecayHu", "hydro.balsaraSwitch",
                "hydro.viscosityAlphaMin", "hydro.viscosityAlphaMax", "hydro.viscosityBeta",
                "hydro.gamma"):
        retired_config = args.workdir / f"retired_{key}.ini"
        retired_config.write_text(f"[hydro]\n{key.split('.')[1]} = 1\n")
        rejected = run("retired_ini_" + key, [f"--configuration={retired_config.resolve()}"],
                       success=False)
        if key.split(".")[1] not in rejected:
            raise RuntimeError(f"retired {key} lacked an INI diagnostic")
    for key in ("hydro.viscosityBetaMode", "hydro.viscosityBetaOverAlpha",
                "hydro.viscosityAlpha", "hydro.viscosityEvolution", "hydro.viscosityDecayCD",
                "hydro.viscosityAlphaMaxPrice", "hydro.viscosityDecayPrice",
                "hydro.viscosityAlphaMaxHu", "hydro.viscosityDecayHu",
                "initial.createGlass", "initial.plummer", "hydro.pressureOperator",
                "hydro.viscosityAlphaMin", "hydro.viscosityAlphaMax", "hydro.viscosityBeta",
                "hydro.gamma"):
        if re.search(r"^" + re.escape(key) + r"\s*:", balsara_default, re.MULTILINE):
            raise RuntimeError(f"removed option {key} is still reported")
    if sph_controls_available:
        viscosity_override = run("variable_viscosity_cli_override", [
            f"--configuration={config.resolve()}",
            "--hydro.viscosityCoefficientMin=0.4"] + sph_control_options)
        match = re.search(r"^hydro\.viscosityCoefficientMin\s*:\s*(\S+)", viscosity_override,
                          re.MULTILINE)
        if match is None or float(match.group(1)) != 0.4:
            raise RuntimeError("CLI hydro.viscosityCoefficientMin did not override INI")

    # Metadata-only cases exercise the storage range without integrating a flow
    # with extreme viscosity. Both alpha bounds are stored as binary16 particle values.
    viscosityStorageKeys = ("hydro.viscosityCoefficientMin", "hydro.viscosityCoefficientMax")
    halfMinimum = 2.0 ** -24
    halfMaximum = 65504.0
    halfOutsideValues = (2.0 ** -25, 65505.0)
    invalidHalfCases = []
    for key in viscosityStorageKeys:
        for bound, value in zip(("below", "above"), halfOutsideValues):
            invalidHalfCases.append((key, (
                f"viscosity_half_{key.split('.')[1]}_{bound}", metadata_options + [
                    f"--{key}={value:.17g}"], False)))
    for (key, _), output in zip(invalidHalfCases,
                               runAll([case for _, case in invalidHalfCases])):
        if key not in output or "binary16 viscosity storage" not in output:
            raise RuntimeError(f"invalid binary16 {key} lacked its storage diagnostic")
    reversed_bounds = run("viscosity_alpha_bounds_reversed", metadata_options + [
        "--hydro.viscosityCoefficientMin=0.5", "--hydro.viscosityCoefficientMax=0.25"], success=False)
    if "hydro.viscosityCoefficientMax must not be below hydro.viscosityCoefficientMin" not in reversed_bounds:
        raise RuntimeError("reversed alpha bounds lacked their diagnostic")

    # Equal bounds at either end of the binary16 range give a valid constant alpha.
    acceptedHalfCases = []
    for bound, value in zip(("minimum", "maximum"), (halfMinimum, halfMaximum)):
        settings = {key: value for key in viscosityStorageKeys}
        acceptedHalfCases.append((settings, (
            f"viscosity_half_constant_{bound}", metadata_options + [
                *[f"--{key}={setting:.17g}" for key, setting in settings.items()]], True)))
    for (settings, _), output in zip(acceptedHalfCases,
                                    runAll([case for _, case in acceptedHalfCases])):
        if "Checkpoint state and mode validation passed" not in output:
            raise RuntimeError("binary16 range case did not finish metadata validation")
        for key, value in settings.items():
            assertNumericSetting(output, key, value)

    verbose_config = args.workdir / "verbose.ini"
    verbose_config.write_text("[run]\nverbose = true\n")
    verbose_output = run("verbose_ini", metadata_options + [
        f"--configuration={verbose_config.resolve()}"])
    if not re.search(r"^run\.verbose\s*:\s*T\s*$", verbose_output, re.MULTILINE):
        raise RuntimeError("INI run.verbose was not applied")
    quiet_output = run("verbose_cli_override", metadata_options + [
        f"--configuration={verbose_config.resolve()}", "--run.verbose=false"])
    if not re.search(r"^run\.verbose\s*:\s*F\s*$", quiet_output, re.MULTILINE):
        raise RuntimeError("CLI run.verbose did not override INI")

    # Selections survive option setup and checkpoints independently.
    for gravity_family, gravity_degree, hydro_family, hydro_degree in (
            ("spline", 9, "wendland", 8), ("wendland", 5, "ferrers", 6),
            ("ferrers", 2, "spline", 4), ("spline", 2, "ferrers", 8),
            ("ferrers", 8, "spline", 2)):
        name = f"kernels_{gravity_family}_{gravity_degree}_{hydro_family}_{hydro_degree}"
        output = run(name, metadata_options + [
            f"--gravity.kernelFamily={gravity_family}", f"--gravity.kernelDegree={gravity_degree}",
            f"--hydro.kernelFamily={hydro_family}", f"--hydro.kernelDegree={hydro_degree}",
            "--hydro.neighborNumber=200", "--gravity.effectiveRadius=0.02", "--gravity.timestepEta=0.15"])
        for key, value in (("gravity.kernelFamily", gravity_family),
                           ("gravity.kernelDegree", str(gravity_degree)),
                           ("hydro.kernelFamily", hydro_family),
                           ("hydro.kernelDegree", str(hydro_degree))):
            if not re.search(r"^" + re.escape(key) + r"\s*:\s*" + value + r"\s*$", output, re.MULTILINE):
                raise RuntimeError(f"independent kernel setting {key} was not applied")
        for key, value in (("gravity.effectiveRadius", 0.02), ("gravity.timestepEta", 0.15)):
            match = re.search(r"^" + re.escape(key) + r"\s*:\s*(\S+)", output, re.MULTILINE)
            if match is None or abs(float(match.group(1)) - value) > 1.e-8:
                raise RuntimeError(f"kernel-independent setting {key} changed meaning")
    runAll([(f"invalid_kernel_{index}", metadata_options + invalid, False) for index, invalid in enumerate((
            ["--gravity.kernelFamily=gaussian"], ["--hydro.kernelFamily=wendland", "--hydro.kernelDegree=9"],
            ["--gravity.kernelFamily=ferrers", "--gravity.kernelDegree=9"],
            ["--hydro.kernelDegree=10"], ["--gravity.kernelDegree=-1"],
            ["--gravity.kernelFamily=spline", "--gravity.kernelDegree=1"],
            ["--hydro.kernelFamily=spline", "--hydro.kernelDegree=1"],
            ["--hydro.kernelFamily=wendland", "--hydro.kernelDegree=8", "--hydro.neighborNumber=12"]))])

    if not starFormationAvailable:
        requested = metadata_options + ["--starFormation.enabled=true"]
        rejected = run("disabled_star_formation", requested, success=False)
        guidance = "COSMICTIGER_ENABLE_SPH=ON" if args.without_sph else "COSMICTIGER_ENABLE_STARS=ON"
        if guidance not in rejected and "pending SPH entropy source conversions" not in rejected:
            raise RuntimeError("disabled star formation request lacked build guidance")
    elif not chemistryAvailable:
        rejected = run("star_formation_without_chemistry_build", [option for option in metadata_options
                        if not option.startswith("--hydro.enabled=")] + [
            "--hydro.enabled=true", "--starFormation.enabled=true", "--chemistry.enabled=true"],
            success=False)
        if "COSMICTIGER_ENABLE_CHEMISTRY=ON" not in rejected and "pending SPH entropy source conversions" not in rejected:
            raise RuntimeError("star formation without compiled chemistry lacked build guidance")
    elif not coupledSourcesAvailable:
        rejected = run("deferred_star_formation", metadata_options + ["--starFormation.enabled=true"],
                       success=False)
        if "pending SPH entropy source conversions" not in rejected:
            raise RuntimeError("compiled star formation lacked deferred-coupling guidance")
    elif not hydro_rejected:
        star_options = [option for option in metadata_options
                        if not option.startswith("--hydro.enabled=")] + [
                            *sph_enabled_options, "--starFormation.enabled=true",
                            "--chemistry.enabled=true", "--chemistry.initializationTemperature=10000"]
        star_config = args.workdir / "star_formation.ini"
        star_config.write_text("[starFormation]\nenabled=true\nbeta=0.15\nseed=18446744073709551615\n"
                               "[chemistry]\nenabled=true\ninitializationTemperature=10000\n")
        star_ini_options = [option for option in star_options
                            if not option.startswith(("--starFormation.enabled=", "--chemistry.enabled="))]
        output = run("star_formation_ini", star_ini_options + [f"--configuration={star_config.resolve()}"])
        assertNumericSetting(output, "starFormation.beta", 0.15)
        if not re.search(r"^starFormation\.seed\s*:\s*18446744073709551615\s*$", output, re.MULTILINE):
            raise RuntimeError("star formation seed lost uint64 precision")
        output = run("star_formation_override", star_options + [
            f"--configuration={star_config.resolve()}", "--starFormation.beta=0.2", "--starFormation.seed=17"])
        assertNumericSetting(output, "starFormation.beta", 0.2)
        assertNumericSetting(output, "starFormation.seed", 17)
        for key, value in (("seed", "-1"), ("seed", "18446744073709551616"),
                           ("beta", "0"), ("beta", "1"), ("beta", "nan"),
                           ("evaporationFactor", "0"), ("starFormationTimeGyr", "-1"),
                           ("coldTemperature", "10000"), ("supernovaTemperature", "0"),
                           ("criticalPhysicalDensity", "-1"), ("criticalPhysicalDensity", "nan"),
                           ("criticalOverDensity", "-1"), ("criticalOverDensity", "inf")):
            rejected = run("star_formation_reject_" + key + "_" + value,
                           star_options + [f"--starFormation.{key}={value}"], success=False)
            if "starFormation" not in rejected:
                raise RuntimeError("invalid star formation setting lacked a diagnostic")
        for label, options, diagnostic in (
                ("no_hydro", metadata_options + ["--starFormation.enabled=true"], "hydro.enabled"),
                ("no_chemistry", [option for option in star_options
                                  if not option.startswith("--chemistry.enabled=")] +
                 ["--chemistry.enabled=false"], "chemistry.enabled=true"),
                ("wrong_gamma", star_options + ["--hydro.heatCapacityRatio=1.4"], "hydro.heatCapacityRatio")):
            rejected = run("star_formation_" + label, options, success=False)
            if diagnostic not in rejected:
                raise RuntimeError("incompatible star formation selection lacked a diagnostic")
        output = run("star_formation_thresholds", star_options + [
            "--starFormation.criticalPhysicalDensity=0.1", "--starFormation.criticalOverDensity=60"])
        assertNumericSetting(output, "starFormation.criticalPhysicalDensity", 0.1)
        assertNumericSetting(output, "starFormation.criticalOverDensity", 60)
        if "Checkpoint state and mode validation passed" not in output:
            raise RuntimeError("GADGET-4 star formation checkpoint state did not round-trip")

    if args.without_chemistry and not args.without_sph:
        chemistryConfig = args.workdir / "disabled_chemistry.ini"
        chemistryConfig.write_text("[chemistry]\nenabled = true\n")
        for label, options in (("cli", ["--chemistry.enabled=true"]),
                               ("ini", [f"--configuration={chemistryConfig.resolve()}"])):
            rejected = run("disabled_chemistry_" + label, metadata_options + options, success=False)
            if "COSMICTIGER_ENABLE_CHEMISTRY=ON" not in rejected and "pending SPH entropy source conversions" not in rejected:
                raise RuntimeError("disabled chemistry request lacked build guidance")
    elif chemistryAvailable and not coupledSourcesAvailable:
        chemistryConfig = args.workdir / "deferred_chemistry.ini"
        chemistryConfig.write_text("[chemistry]\nenabled = true\n")
        for label, options in (("cli", ["--chemistry.enabled=true"]),
                               ("ini", [f"--configuration={chemistryConfig.resolve()}"])):
            rejected = run("deferred_chemistry_" + label, metadata_options + options, success=False)
            if "pending SPH entropy source conversions" not in rejected:
                raise RuntimeError("compiled chemistry lacked deferred-coupling guidance")

    if sph_controls_available:
        # Both checkpoint layouts round-trip: zero retains the existing SPH block,
        # while nonzero acceleration appends the flagged z-plane force parameter.
        hydro_metadata_options = [option for option in metadata_options
                                  if not option.startswith("--hydro.enabled=")] + sph_enabled_options
        for acceleration in (0.0, 0.5):
            output = run(f"z_plane_checkpoint_{acceleration}", hydro_metadata_options + [
                f"--configuration={config.resolve()}",
                f"--scenario.rayleighTaylor.zPlaneAcceleration={acceleration}"])
            assertNumericSetting(output, "scenario.rayleighTaylor.zPlaneAcceleration", acceleration)
            if "Checkpoint state and mode validation passed" not in output:
                raise RuntimeError("z-plane acceleration checkpoint state did not round-trip")

    if chemistryAvailable and coupledSourcesAvailable and not hydro_rejected:
        cooling_options = [option for option in metadata_options
                           if not option.startswith("--hydro.enabled=")] + [
                               *sph_enabled_options, "--chemistry.enabled=true"]
        output = run("gadget4_cooling_defaults", cooling_options)
        assertNumericSetting(output, "chemistry.initializationTemperature", 10000.0)
        assertNumericSetting(output, "chemistry.minimumSpecificEnergy", 0.0)
        if "Checkpoint state and mode validation passed" not in output:
            raise RuntimeError("GADGET-4 cooling checkpoint state did not round-trip")
        for field, value in (("initializationTemperature", 0.0),
                             ("initializationTemperature", 2e9),
                             ("minimumSpecificEnergy", 1e10)):
            output = run(f"gadget4_cooling_override_{field}_{value}",
                         cooling_options + [f"--chemistry.{field}={value}"])
            assertNumericSetting(output, f"chemistry.{field}", value)
        cooling_config = args.workdir / "gadget4_cooling.ini"
        cooling_config.write_text("[chemistry]\ninitializationTemperature=500\nminimumSpecificEnergy=1000\n")
        output = run("gadget4_cooling_precedence", cooling_options + [
            f"--configuration={cooling_config.resolve()}", "--chemistry.initializationTemperature=750"])
        assertNumericSetting(output, "chemistry.initializationTemperature", 750.0)
        assertNumericSetting(output, "chemistry.minimumSpecificEnergy", 1000.0)
        for index, extra in enumerate(("--chemistry.initializationTemperature=-1",
                                       "--chemistry.initializationTemperature=nan",
                                       "--chemistry.minimumSpecificEnergy=-1",
                                       "--chemistry.minimumSpecificEnergy=inf",
                                       "--hydro.heatCapacityRatio=1.4")):
            run(f"gadget4_cooling_invalid_{index}", cooling_options + [extra], success=False)
        for index, removed in enumerate(("--chemistry.network=atomic6",
                                         "--chemistry.initialization=manual",
                                         "--chemistry.maxSubsteps=4096",
                                         "--starFormation.winds=true",
                                         "--starFormation.metalYield=0.02")):
            run(f"gadget4_removed_option_{index}", cooling_options + [removed], success=False)

    # Photon density is fixed by the requested blackbody temperature and H0;
    # neutrinos add radiation without changing that photon density.
    for neff, temperature in ((0.0, 2.7255), (3.046, 2.7255), (6.0, 3.0)):
        output = run(f"radiation_{neff}_{temperature}", metadata_options + [
            f"--cosmology.nEff={neff}", f"--cosmology.thetaCmb={temperature / 2.73:.17g}"])
        hubble = 0.6732 * 1.e7 / mpcToCm
        criticalDensity = 3.0 * hubble**2 / (8.0 * math.pi * 6.67259e-8)
        photons = (4.0 * 5.67051e-5 / 2.99792458e10) * temperature**4 / (
            criticalDensity * 2.99792458e10**2)
        neutrinos = photons * neff * (7.0 / 8.0) * (4.0 / 11.0)**(4.0 / 3.0)
        for key, expected in (("omegaGamma", photons), ("omegaNu", neutrinos),
                              ("omegaR", photons + neutrinos)):
            assertNumericSetting(output, f"cosmology.{key}", expected)

    runAll([(f"invalid_cosmology_{index}", metadata_options + [invalid], False) for index, invalid in enumerate((
            "--cosmology.nEff=-1", "--cosmology.thetaCmb=0", "--cosmology.hubble=0",
            "--cosmology.heliumFraction=1", "--cosmology.sigma8=nan",
            "--cosmology.sigma8=-0.1"))])

    rungValues = ("-0.1", "1.5", "nan", "inf")
    rungOutputs = runAll([(f"invalid_rung_merge_fraction_{index}", metadata_options + [
        f"--run.rungMergeFraction={value}"], False) for index, value in enumerate(rungValues)])
    for value, output in zip(rungValues, rungOutputs):
        if "run.rungMergeFraction must be finite and between 0 and 1" not in output:
            raise RuntimeError(f"invalid run.rungMergeFraction={value} lacked its diagnostic")
    output = run("rung_merge_fraction", metadata_options + ["--run.rungMergeFraction=0.5"])
    assertNumericSetting(output, "run.rungMergeFraction", 0.5)

    rebuildOutputs = runAll([(f"invalid_sph_rebuild_fraction_{index}", metadata_options + [
        f"--compute.sphTreeRebuildFraction={value}"], False)
        for index, value in enumerate(rungValues)])
    for value, output in zip(rungValues, rebuildOutputs):
        if "compute.sphTreeRebuildFraction must be finite and between 0 and 1" not in output:
            raise RuntimeError(f"invalid compute.sphTreeRebuildFraction={value} lacked its diagnostic")
    output = run("sph_rebuild_fraction", metadata_options + ["--compute.sphTreeRebuildFraction=0.25"])
    assertNumericSetting(output, "compute.sphTreeRebuildFraction", 0.25)

    lambda_baseline = run("lambda_baseline", metadata_options)
    def cosmologySetting(output, key):
        match = re.search(r"^cosmology\." + key + r"\s*:\s*(\S+)", output, re.MULTILINE)
        if match is None or not math.isfinite(float(match.group(1))):
            raise RuntimeError(f"missing finite resolved cosmology.{key}")
        return float(match.group(1))

    # Only the word auto requests density closure. Every numeric value,
    # including the former -1 sentinel, is a literal signed Lambda density.
    lambda_auto = run("lambda_auto", metadata_options + ["--cosmology.omegaLambda=auto"])
    lambda_dense = run("lambda_auto_dense", metadata_options + [
        "--cosmology.omegaLambda=auto", "--cosmology.omegaB=0.05", "--cosmology.omegaC=1.95"])
    for output in (lambda_baseline, lambda_auto, lambda_dense):
        total = sum(cosmologySetting(output, key) for key in ("omegaM", "omegaR", "omegaLambda"))
        if not math.isclose(total, 1.0, rel_tol=2.e-6):
            raise RuntimeError("automatic Lambda did not preserve a flat density budget")
        assertNumericSetting(output, "cosmology.omegaK", 0.0)
    if cosmologySetting(lambda_dense, "omegaLambda") >= 0.0:
        raise RuntimeError("automatic Lambda could not derive a negative flat-universe value")
    assertNumericSetting(lambda_auto, "cosmology.omegaLambda",
                         cosmologySetting(lambda_baseline, "omegaLambda"))
    # Negligible radiation lets these literal densities close to roundoff.
    for value in (-1.0, -0.25, 0.0, 0.7):
        output = run(f"lambda_literal_{value}", metadata_options + [
            f"--cosmology.omegaLambda={value}", "--cosmology.omegaB=0.05",
            f"--cosmology.omegaC={0.95 - value!r}", "--cosmology.thetaCmb=0.0001"])
        assertNumericSetting(output, "cosmology.omegaLambda", value)
        assertNumericSetting(output, "cosmology.omegaK", 0.0)
    lambda_config = args.workdir / "signed_lambda.ini"
    lambda_config.write_text("[cosmology]\nomegaLambda = -0.25\n")
    output = run("lambda_ini", metadata_options + [f"--configuration={lambda_config.resolve()}",
                 "--cosmology.omegaB=0.05", "--cosmology.omegaC=1.2",
                 "--cosmology.thetaCmb=0.0001"])
    assertNumericSetting(output, "cosmology.omegaLambda", -0.25)
    assertNumericSetting(output, "cosmology.omegaK", 0.0)
    # Both signs are rejected even for homogeneous ICs or a restart request.
    for name, extra in (("positive", ["--cosmology.omegaLambda=0"]),
                        ("negative", ["--cosmology.omegaLambda=1"]),
                        ("homogeneous", ["--cosmology.omegaLambda=0", "--cosmology.sigma8=0"]),
                        ("restart", ["--cosmology.omegaLambda=1", "--checkpoint.readIndex=1"]),
                        ("ini", [f"--configuration={lambda_config.resolve()}"])):
        output = run(f"nonflat_{name}", metadata_options + extra, success=False)
        if "Nonzero cosmological curvature is not supported" not in output:
            raise RuntimeError(f"nonflat {name} lacked the curvature diagnostic")
    output = run("lambda_cli_override", metadata_options + [
        f"--configuration={lambda_config.resolve()}", "--cosmology.omegaLambda=auto"])
    assertNumericSetting(output, "cosmology.omegaLambda",
                         cosmologySetting(lambda_baseline, "omegaLambda"))
    cosmologyCases = [(f"Lambda {value}", (f"invalid_lambda_{index}", metadata_options + [
        f"--cosmology.omegaLambda={value}"], False)) for index, value in enumerate((
            "automatic", "AUTO", "-1junk", "1.2.3", "nan", "inf", "-inf", "1e1000"))]
    cosmologyCases += [(f"{key}={value}", (f"invalid_{key}_{index}", metadata_options + [
        f"--cosmology.{key}={value}"], False)) for key in ("startRedshift", "endRedshift")
        for index, value in enumerate(("-1", "-1.01", "nan", "inf"))]
    for (label, _), output in zip(cosmologyCases, runAll([case for _, case in cosmologyCases])):
        if "cosmology" not in output.lower():
            raise RuntimeError(f"invalid {label} lacked a cosmology diagnostic")
    homogeneous = run("homogeneous_cosmology", metadata_options + ["--cosmology.sigma8=0"])
    assertNumericSetting(homogeneous, "cosmology.sigma8", 0.0)
    if "Checkpoint state and mode validation passed" not in homogeneous:
        raise RuntimeError("homogeneous cosmology options did not complete checkpoint validation")

    def assertIgnoredOption(output, ignoredKey):
        if not any(ignoredKey in line and "warning" in line.lower() and "ignored" in line.lower()
                   for line in output.splitlines()):
            raise RuntimeError(f"missing warning that {ignoredKey} is ignored")

    def assertNoRadiusWarning(output):
        radiusKeys = ("effectiveRadius", "EffectiveRadius", "smoothingLength", "SmoothingLength",
                      *deprecated_aliases)
        if any("warning" in line.lower() and any(key in line for key in radiusKeys)
               for line in output.splitlines()):
            raise RuntimeError("a single radius representation produced a warning")

    def assertDeprecatedOption(output, aliasKey):
        warnings = [line for line in output.splitlines()
                    if aliasKey in line and "deprecated" in line.lower()]
        if len(warnings) != 1 or deprecated_aliases[aliasKey] not in warnings[0]:
            raise RuntimeError(f"{aliasKey} did not print one deprecation warning naming "
                               f"{deprecated_aliases[aliasKey]}")

    defaults = run("independent_defaults", balsara_options)
    build_dimension = int(
        re.search(r"^build\.dimensionCount\s*:\s*(\d+)", defaults, re.MULTILINE)[1])
    for key, value in (("gravity.kernelDegree", 4),
                       ("hydro.neighborNumber", 42), ("hydro.kernelDegree", 3),
                       ("hydro.conductionCoefficient", 0),
                       ("hydro.viscosityCoefficientMin", 0.01), ("hydro.viscosityCoefficientMax", 2.0),
                       ("hydro.viscosityDecay", 0.2),
                       ("compute.bucketSize", 128), ("compute.sphBucketSize", 106)):
        assertNumericSetting(defaults, key, value)

    # SPH leaf capacity follows its own explicit setting or neighbor-derived
    # automatic value, independently of the gravity bucket and its compile-time cap.
    output = run("sph_bucket_gravity_independent", metadata_options + [
        "--compute.bucketSize=32"])
    assertNumericSetting(output, "compute.bucketSize", 32)
    assertNumericSetting(output, "compute.sphBucketSize", 106)
    output = run("sph_bucket_neighbor_override", metadata_options + [
        "--compute.sphBucketSize=0", "--hydro.neighborNumber=84"])
    assertNumericSetting(output, "hydro.neighborNumber", 84)
    assertNumericSetting(output, "compute.sphBucketSize", 213)
    output = run("sph_bucket_cli_override", metadata_options + [
        f"--configuration={config.resolve()}", "--compute.sphBucketSize=2048"])
    assertNumericSetting(output, "compute.bucketSize", 64)
    assertNumericSetting(output, "compute.sphBucketSize", 2048)

    # Both radius representations describe the same kernel, including kernels
    # whose support-to-effective-radius ratios differ.
    for family, degree, ratio in (("ferrers", 8, math.sqrt(13) / 2),
                                   ("spline", 3, math.sqrt(10 / 3)),
                                   ("spline", 2, math.sqrt(135 / 52)),
                                   ("wendland", 5, math.sqrt(15) / 2)):
        common = metadata_options + [f"--gravity.kernelFamily={family}",
                                      f"--gravity.kernelDegree={degree}"]
        for representation, value in (("effectiveRadius", 0.02), ("smoothingLength", 0.02 * ratio)):
            output = run(f"radius_{family}_{degree}_{representation}", common + [
                f"--gravity.{representation}={value}"])
            assertNumericSetting(output, "gravity.effectiveRadius", 0.02)
            assertNumericSetting(output, "gravity.smoothingLength", 0.02 * ratio)
            assertNoRadiusWarning(output)

    gravityRatio = 1.5
    radiusPairs = (
        ("gravity.effectiveRadius", "gravity.smoothingLength", 0.02, gravityRatio),
        ("scenario.disc.centralEffectiveRadius", "scenario.disc.centralSmoothingLength", 0.02, gravityRatio),
    )
    radiusConfig = args.workdir / "radius.ini"
    for effectiveKey, supportKey, effectiveRadius, ratio in radiusPairs:
        smoothingLength = effectiveRadius * ratio
        conflictingSupport = 2 * smoothingLength
        for key, value in ((effectiveKey, effectiveRadius), (supportKey, smoothingLength)):
            output = run(f"radius_only_{key}", metadata_options + [f"--{key}={value}"])
            assertNumericSetting(output, supportKey, smoothingLength)
            assertNoRadiusWarning(output)

        # Explicit effective radii win across representations regardless of
        # whether the winning value came from the command line or an INI.
        sources = (
            ("both_cli", "", [f"--{effectiveKey}={effectiveRadius}",
                                f"--{supportKey}={conflictingSupport}"]),
            ("effective_cli", f"{supportKey} = {conflictingSupport}\n",
             [f"--{effectiveKey}={effectiveRadius}"]),
            ("effective_ini", f"{effectiveKey} = {effectiveRadius}\n",
             [f"--{supportKey}={conflictingSupport}"]),
            ("both_ini", f"{effectiveKey} = {effectiveRadius}\n"
                         f"{supportKey} = {conflictingSupport}\n", []),
        )
        for sourceName, configText, cliOptions in sources:
            radiusConfig.write_text(configText)
            output = run(f"radius_precedence_{effectiveKey}_{sourceName}", metadata_options +
                         [f"--configuration={radiusConfig.resolve()}"] + cliOptions)
            assertNumericSetting(output, supportKey, smoothingLength)
            assertIgnoredOption(output, supportKey)

        # A losing support radius is ignored before physical validation.
        for invalidValue in ("0", "-1", "nan", "inf"):
            output = run(f"radius_ignored_{supportKey}_{invalidValue}", metadata_options +
                         [f"--{effectiveKey}={effectiveRadius}", f"--{supportKey}={invalidValue}"])
            assertNumericSetting(output, supportKey, smoothingLength)
            assertIgnoredOption(output, supportKey)

        # Ordinary CLI-over-INI precedence still applies within each key.
        for key, value in ((effectiveKey, effectiveRadius), (supportKey, smoothingLength)):
            radiusConfig.write_text(f"{key} = {2 * value}\n")
            output = run(f"radius_same_key_{key}", metadata_options +
                         [f"--configuration={radiusConfig.resolve()}", f"--{key}={value}"])
            assertNumericSetting(output, supportKey, smoothingLength)
            assertNoRadiusWarning(output)

    for effectiveKey, supportKey, aliasKey in (
            ("gravity.effectiveRadius", "gravity.smoothingLength", "gravity.softening"),):
        # Deprecated effective-radius aliases retain their meaning with one
        # deprecation warning, and explicitly supplied aliases also take
        # precedence over support-radius inputs.
        output = run(f"radius_alias_{aliasKey}", metadata_options + [f"--{aliasKey}=0.02"])
        assertNumericSetting(output, effectiveKey, 0.02)
        assertNumericSetting(output, supportKey, 0.02 * gravityRatio)
        assertDeprecatedOption(output, aliasKey)
        if any("warning" in line.lower() and "ignored" in line.lower()
               for line in output.splitlines()):
            raise RuntimeError(f"a lone {aliasKey} produced an ignored-option warning")
        if re.search(r"^" + re.escape(aliasKey) + r"\s*:", output, re.MULTILINE):
            raise RuntimeError(f"resolved options listed deprecated {aliasKey}")
        for winningKey, losingKey in ((effectiveKey, aliasKey), (aliasKey, supportKey)):
            for winningSource in ("cli", "ini", "both_cli"):
                if winningSource == "cli":
                    radiusConfig.write_text(f"{losingKey} = 0.03\n")
                    cliOptions = [f"--{winningKey}=0.02"]
                elif winningSource == "ini":
                    radiusConfig.write_text(f"{winningKey} = 0.02\n")
                    cliOptions = [f"--{losingKey}=0.03"]
                else:
                    radiusConfig.write_text("")
                    cliOptions = [f"--{winningKey}=0.02", f"--{losingKey}=0.03"]
                output = run(f"radius_alias_precedence_{winningKey}_{winningSource}", metadata_options +
                             [f"--configuration={radiusConfig.resolve()}"] + cliOptions)
                assertNumericSetting(output, effectiveKey, 0.02)
                assertNumericSetting(output, supportKey, 0.02 * gravityRatio)
                assertIgnoredOption(output, losingKey)
                assertDeprecatedOption(output, aliasKey)

    # Defaulted effective radii and defaulted support bounds are fallbacks, not
    # explicitly selected representations that can conflict with one another.
    output = run("radius_defaults", metadata_options)
    assertNoRadiusWarning(output)
    radiusKeys = [key for effectiveKey, supportKey, _, _ in radiusPairs
                  for key in (effectiveKey, supportKey)]
    runAll([(f"invalid_{key}_{value}", metadata_options + [f"--{key}={value}"], False)
            for key in radiusKeys + ["hydro.minSmoothingLength", "hydro.maxSmoothingLength"]
            for value in ("0", "-1", "nan", "inf")])

    def assertOutputSettings(output, outputUnits, outputFrame):
        for key, expectedValue in (("output.units", outputUnits), ("output.frame", outputFrame)):
            match = re.search(r"^" + re.escape(key) + r"\s*:\s*(\S+)", output, re.MULTILINE)
            if match is None or match.group(1) != expectedValue:
                raise RuntimeError(f"configured {key} did not equal {expectedValue}")

    # An obsolete view.txt must have no effect, even with no output options.
    unused_view.write_text("0 0.125 0 1 0 1\n0.5 1 0 1 0 1\n")
    disabled = run("silo_default_disabled", metadata_options)
    if "Silo particle output: disabled" not in disabled or "view.txt" in disabled:
        raise RuntimeError("obsolete view.txt affected the default output configuration")
    assertOutputSettings(disabled, "code", "comoving")

    output_config = args.workdir / "output.ini"
    output_config.write_text("[run]\nnumberOfOutputs = 7\n[output]\nsilo = true\n"
                             "units = cosmological\nframe = lab\n"
                             "directory = configured snapshots\nxMin = 0.25\nxMax = 0.75\n")
    configured = run("silo_ini", metadata_options + [f"--configuration={output_config.resolve()}"])
    assertOutputSettings(configured, "cosmological", "lab")
    if "Silo particle output: x [0.25, 0.75], y [0, 1], z [0, 1]" not in configured:
        raise RuntimeError("INI Silo bounds were not applied")
    if "7 equally spaced outputs including endpoints" not in configured:
        raise RuntimeError("INI Silo cadence was not applied")
    if not (args.workdir / "configured snapshots").is_dir():
        raise RuntimeError("relative Silo output directory was not created")

    cli_directory = (args.workdir / "cli snapshots").resolve()
    overridden = run("silo_cli", metadata_options + [f"--configuration={output_config.resolve()}",
                                                    "--output.xMin=0.5", "--run.numberOfOutputs=3",
                                                    "--output.units=both", "--output.frame=both",
                                                    f"--output.directory={cli_directory}"])
    assertOutputSettings(overridden, "both", "both")
    if "Silo particle output: x [0.5, 0.75], y [0, 1], z [0, 1]" not in overridden:
        raise RuntimeError("CLI bounds did not override INI bounds")
    if "3 equally spaced outputs including endpoints" not in overridden or not cli_directory.is_dir():
        raise RuntimeError("CLI output cadence or directory did not override the INI")

    disabled = run("silo_disabled", metadata_options + ["--output.silo=false"])
    if "Silo particle output: disabled" not in disabled or "view.txt" in disabled:
        raise RuntimeError("explicitly disabling Silo did not suppress output")
    bounds_only = run("silo_bounds_only", metadata_options + ["--output.xMax=0.5"])
    if "Silo particle output: disabled" not in bounds_only:
        raise RuntimeError("specifying bounds silently enabled output")
    default_output = run("silo_default_directory", metadata_options + ["--output.silo=true"])
    if "Silo particle output directory: ./output" not in default_output:
        raise RuntimeError("Silo did not default to ./output")
    if not (args.workdir / "output").is_dir():
        raise RuntimeError("default Silo output directory was not created")
    unused_view.unlink()

    for outputUnits in ("cgs", "cosmological", "both"):
        for outputFrame in ("comoving", "lab", "both"):
            output = run(f"units_{outputUnits}_{outputFrame}", metadata_options +
                         [f"--output.units={outputUnits}", f"--output.frame={outputFrame}"])
            assertOutputSettings(output, outputUnits, outputFrame)
            if "Checkpoint state and mode validation passed" not in output:
                raise RuntimeError("unit selection changed checkpoint metadata compatibility")

    blocker = args.workdir / "silo_directory_blocker"
    blocker.write_text("This is a file, not a directory.\n")
    output = run("silo_directory_blocked", metadata_options + ["--output.silo=true",
                 f"--output.directory={blocker.resolve()}"], success=False)
    if "directory" not in output.lower():
        raise RuntimeError("invalid Silo directory failed without a directory diagnostic")

    # Complete editable examples must track every public option as new controls
    # are introduced. Loading each through the executable also verifies comments
    # and empty-string defaults without running its physical problem.
    root = Path(__file__).resolve().parents[1]
    optional_keys = {"compute.positionType", "compute.bucketSize",
                     "gravity.smoothingLength",
                     "units.boxLength",
                     "scenario.disc.centralSmoothingLength"}
    public_keys = set(option_names) - {"help", "configuration"}
    preset_curvatures = {"critical.ini": 0.0}
    checked_presets = set()
    local_presets = {root / "configurations/lcdm_present.ini"}
    preset_paths = [root / "default.ini", *sorted((root / "configurations").rglob("*.ini"))]
    for path in preset_paths:
        relative_path = path.relative_to(root)
        case_name = "_".join(relative_path.with_suffix("").parts)
        ini = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
        ini.optionxform = str
        with path.open() as stream:
            ini.read_file(stream)
        preset_optional_keys = optional_keys
        actual_keys = {f"{section}.{key}" for section in ini.sections() for key in ini[section]}
        active_optional_keys = actual_keys & preset_optional_keys
        expected_keys = (public_keys - preset_optional_keys) | active_optional_keys
        if actual_keys != expected_keys:
            raise RuntimeError(f"{relative_path} option coverage differs: {sorted(actual_keys ^ expected_keys)}")
        # Build-selected position width/bucket size and radius alternatives may
        # remain commented. All three cooling controls must be explicit.
        documented_alternatives = set(active_optional_keys)
        section = None
        for line in path.read_text().splitlines():
            section_match = re.fullmatch(r"\s*\[([^]]+)\]\s*", line)
            if section_match:
                section = section_match.group(1)
            alternative_match = re.match(
                r"\s*[#;]\s*([A-Za-z][A-Za-z0-9]*(?:\.[A-Za-z][A-Za-z0-9]*)*)\s*=", line)
            if section and alternative_match:
                documented_alternatives.add(f"{section}.{alternative_match.group(1)}")
        if not preset_optional_keys <= documented_alternatives:
            raise RuntimeError(f"{relative_path} is missing optional settings: "
                               f"{sorted(preset_optional_keys - documented_alternatives)}")
        if (path not in local_presets and
                (ini["output"]["units"] != "code" or ini["output"]["frame"] != "comoving")):
            raise RuntimeError(f"{relative_path} must preserve native output defaults")
        metadata_directory = args.workdir.resolve() / "output" / "options_presets" / case_name
        runtime_overrides = [*[f"--{key}=false" for key in cuda_keys],
                             f"--compute.positionType={position_type}",
                             f"--output.directory={metadata_directory}",
                             f"--lightcone.directory={metadata_directory / 'lightcone'}"]
        if args.without_rockstar:
            runtime_overrides.extend(["--analysis.groups=false", "--lightcone.enabled=false"])
        preset_options = [f"--configuration={path}", "--chemistry.enabled=false",
                          "--starFormation.enabled=false"] + runtime_overrides
        if args.time_integrator == "hierarchical":
            # Validate gravity options and serialization with hydro disabled
            # even for presets whose physical problem uses imposed forces only.
            preset_options.append("--gravity.enabled=true")
        if args.without_sph and ini["run"]["scenario"] not in ("cosmology", "plummer", "glass"):
            # Keep validating every example's option coverage and syntax when its
            # physical scenario is unavailable in the compiled executable.
            preset_options.append("--run.scenario=cosmology")
        if (ini["run"]["scenario"] == "kelvinHelmholtz" and build_dimension != 2 and
                ini.has_option("scenario.kelvinHelmholz", "packing")):
            # The Kelvin-Helmholtz preset targets the 2D build (make.sh); its CEV/E
            # packing has nonuniform site neighborhoods in 3D. Other builds check the
            # preset's option coverage with the default uniform CV/C packing.
            preset_options.append("--scenario.kelvinHelmholz.packing=CV/C")
        output = run(f"complete_{case_name}", metadata_options + preset_options)
        if "Checkpoint state and mode validation passed" not in output:
            raise RuntimeError(f"{relative_path} did not finish metadata validation")
        if (ini["run"]["scenario"] == "cosmology" and coupledSourcesAvailable and chemistryAvailable and
                starFormationAvailable and not hydro_rejected):
            enabled_options = [option for option in metadata_options
                               if not option.startswith("--hydro.enabled=")]
            # Check the preset's hydro, cooling and star formation switches
            # independently of gravity and its force-error sampling.
            enabled = run(f"enabled_physics_{case_name}", enabled_options +
                          [f"--configuration={path}", "--gravity.enabled=false",
                           "--gravity.errorInterval=0"] + runtime_overrides)
            for section in ("hydro", "chemistry", "starFormation"):
                key = section + ".enabled"
                expected = "T" if ini.getboolean(section, "enabled") else "F"
                if not re.search(r"^" + re.escape(key) + r"\s*:\s*" + expected + r"\s*$",
                                 enabled, re.MULTILINE):
                    raise RuntimeError(f"{relative_path} did not preserve {key}")
        if path.parent == root / "configurations" and path.name in preset_curvatures:
            assertNumericSetting(output, "cosmology.omegaK", preset_curvatures[path.name])
            if path.name == "critical.ini":
                assertNumericSetting(output, "cosmology.omegaLambda", 0.0)
            checked_presets.add(path.name)
        assertNoRadiusWarning(output)

    if checked_presets != set(preset_curvatures):
        raise RuntimeError(f"missing cosmology presets: {sorted(set(preset_curvatures) - checked_presets)}")

    invalid = [
        ("legacy_key", "--parts_dim=8", "parts_dim"),
        ("legacy_config", "--run.configFile=missing_grouped_options.ini", "run.configFile"),
        ("legacy_help", "--run.help", "run.help"),
        ("missing_config", "--configuration=missing_grouped_options.ini", "Cannot open configuration file"),
        ("silo_directory_empty", "--output.directory=", "output.directory"),
        ("silo_units_unknown", "--output.units=si", "output.units"),
        ("silo_units_empty", "--output.units=", "output.units"),
        ("silo_frame_unknown", "--output.frame=proper", "output.frame"),
        ("silo_frame_empty", "--output.frame=", "output.frame"),
        ("silo_code_lab", "--output.frame=lab", "requires output.frame=comoving"),
        ("silo_code_both", "--output.frame=both", "requires output.frame=comoving"),
        ("silo_nan", "--output.xMin=nan", "output.xMin"),
        ("silo_inf", "--output.yMax=inf", "output.yMin"),
        ("silo_negative", "--output.zMin=-0.1", "output.zMin"),
        ("silo_outside", "--output.xMax=1.1", "output.xMin"),
        ("silo_empty", "--output.yMin=1", "output.yMin"),
        ("silo_reversed", "--output.zMax=-0.1", "output.zMin"),
        ("gamma_nan", "--hydro.heatCapacityRatio=nan", "hydro.heatCapacityRatio"),
        ("gamma_inf", "--hydro.heatCapacityRatio=inf", "hydro.heatCapacityRatio"),
        ("conduction_negative", "--hydro.conductionCoefficient=-0.1", "hydro.conductionCoefficient"),
        ("conduction_nan", "--hydro.conductionCoefficient=nan", "hydro.conductionCoefficient"),
        ("conduction_inf", "--hydro.conductionCoefficient=inf", "hydro.conductionCoefficient"),
        ("conduction_overflow", "--hydro.conductionCoefficient=1e100", "single precision"),
        ("conduction_underflow", "--hydro.conductionCoefficient=1e-100", "single precision"),
        ("alpha_max_nan", "--hydro.viscosityCoefficientMax=nan", "hydro.viscosityCoefficientMax"),
        ("balsara_negative", "--hydro.balsaraSoundSpeedCoefficient=-1",
         "hydro.balsaraSoundSpeedCoefficient"),
        ("balsara_nan", "--hydro.balsaraSoundSpeedCoefficient=nan",
         "hydro.balsaraSoundSpeedCoefficient"),
        ("balsara_overflow", "--hydro.balsaraSoundSpeedCoefficient=1e100",
         "hydro.balsaraSoundSpeedCoefficient"),
        ("epsilon_inf", "--hydro.viscosityEpsilon=inf", "hydro.viscosityEpsilon"),
        ("cfl_nan", "--hydro.cfl=nan", "hydro.cfl"),
        ("neighbors_inf", "--hydro.neighborNumber=inf", "hydro.neighborNumber"),
        ("z_plane_negative", "--scenario.rayleighTaylor.zPlaneAcceleration=-0.5", "scenario.rayleighTaylor.zPlaneAcceleration"),
        ("z_plane_nan", "--scenario.rayleighTaylor.zPlaneAcceleration=nan", "scenario.rayleighTaylor.zPlaneAcceleration"),
        ("z_plane_inf", "--scenario.rayleighTaylor.zPlaneAcceleration=inf", "scenario.rayleighTaylor.zPlaneAcceleration"),
        ("z_plane_overflow", "--scenario.rayleighTaylor.zPlaneAcceleration=1e100", "single precision"),
        ("z_plane_underflow", "--scenario.rayleighTaylor.zPlaneAcceleration=1e-100", "single precision"),
        ("outputs_zero", "--run.numberOfOutputs=0", "run.numberOfOutputs"),
        ("outputs_one", "--run.numberOfOutputs=1", "run.numberOfOutputs"),
        ("end_time_negative", "--run.endTime=-1", "run.endTime"),
        ("end_time_nan", "--run.endTime=nan", "run.endTime"),
        ("end_time_inf", "--run.endTime=inf", "run.endTime"),
        ("scenario_unknown", "--run.scenario=unknown", "scenario"),
        ("position_type", "--compute.positionType=float64", "compute.positionType"),
        ("bucket_zero", "--compute.bucketSize=0", "compute.bucketSize"),
        ("sph_bucket_negative", "--compute.sphBucketSize=-1", "compute.sphBucketSize"),
        ("sph_bucket_auto_overflow", "--hydro.neighborNumber=1e9", "compute.sphBucketSize"),
        ("eta_zero", "--gravity.timestepEta=0", "gravity.timestepEta"),
        ("eta_negative", "--gravity.timestepEta=-0.1", "gravity.timestepEta"),
        ("eta_nan", "--gravity.timestepEta=nan", "gravity.timestepEta"),
        ("eta_inf", "--gravity.timestepEta=inf", "gravity.timestepEta"),
        ("eta_underflow", "--gravity.timestepEta=1e-100", "gravity.timestepEta"),
        ("effective_radius_zero", "--gravity.effectiveRadius=0", "gravity.effectiveRadius"),
        ("effective_radius_negative", "--gravity.effectiveRadius=-0.1", "gravity.effectiveRadius"),
        ("effective_radius_nan", "--gravity.effectiveRadius=nan", "gravity.effectiveRadius"),
        ("effective_radius_inf", "--gravity.effectiveRadius=inf", "gravity.effectiveRadius"),
        ("effective_radius_underflow", "--gravity.effectiveRadius=1e-100", "gravity.effectiveRadius"),
        ("softening_deprecated_zero", "--gravity.softening=0", "gravity.softening"),
        ("smoothing_obsolete", "--gravity.sphSmoothing=true", "gravity.sphSmoothing"),
        ("adaptive_option_removed", "--gravity.softeningMode=adaptive", "gravity.softeningMode"),
        ("alpha_max_overflow", "--hydro.viscosityCoefficientMax=1e100", "binary16 viscosity storage"),
        ("epsilon_underflow", "--hydro.viscosityEpsilon=1e-100", "single precision"),
        ("gamma_rounding", "--hydro.heatCapacityRatio=1.00000001", "single precision"),
        # The float cubic table gives a self-neighbor floor of 10.666666236 (float
        # 10.666666031): this value passes the double check and rounds onto the floor.
        ("neighbors_rounding", "--hydro.neighborNumber=10.6666664", "single precision"),
        ("cfl_underflow", "--hydro.cfl=1e-100", "single precision"),
        ("support_underflow", "--hydro.minSmoothingLength=1e-100", "single precision"),
        ("viscosity_alpha_max_hu_removed", "--hydro.viscosityAlphaMaxHu=1",
         "hydro.viscosityAlphaMaxHu"),
        ("viscosity_decay_hu_removed", "--hydro.viscosityDecayHu=1", "hydro.viscosityDecayHu"),
        ("viscosity_evolution_removed", "--hydro.viscosityEvolution=none",
         "hydro.viscosityEvolution"),
        ("viscosity_alpha_removed", "--hydro.viscosityAlpha=1", "hydro.viscosityAlpha"),
        ("viscosity_decay_cd_removed", "--hydro.viscosityDecayCD=1", "hydro.viscosityDecayCD"),
        ("viscosity_alpha_max_price_removed", "--hydro.viscosityAlphaMaxPrice=1",
         "hydro.viscosityAlphaMaxPrice"),
        ("viscosity_decay_price_removed", "--hydro.viscosityDecayPrice=1",
         "hydro.viscosityDecayPrice"),
        ("beta_mode_removed", "--hydro.viscosityBetaMode=ratio", "hydro.viscosityBetaMode"),
        ("viscosity_alpha_min_renamed", "--hydro.viscosityAlphaMin=0.1", "hydro.viscosityAlphaMin"),
        ("viscosity_alpha_max_renamed", "--hydro.viscosityAlphaMax=1", "hydro.viscosityAlphaMax"),
        ("viscosity_beta_renamed", "--hydro.viscosityBeta=2", "hydro.viscosityBeta"),
        ("gamma_renamed", "--hydro.gamma=1.4", "hydro.gamma"),
        ("z_plane_moved", "--hydro.zPlaneAcceleration=0.5", "hydro.zPlaneAcceleration"),
        ("central_acceleration_moved", "--hydro.centralAcceleration=1",
         "hydro.centralAcceleration"),
        ("plummer_radius_moved", "--initial.plummer.radius=0.02", "initial.plummer.radius"),
        ("two_lpt_moved", "--initial.cosmology.twoLpt=true", "initial.cosmology.twoLpt"),
        ("alpha_floor_negative", "--hydro.viscosityCoefficientMin=-0.1", "hydro.viscosityCoefficientMin"),
        ("alpha_floor_underflow", "--hydro.viscosityCoefficientMin=1e-100", "single precision"),
        ("viscosity_decay_zero", "--hydro.viscosityDecay=0", "hydro.viscosityDecay"),
        ("viscosity_decay_overflow", "--hydro.viscosityDecay=1e100", "single precision"),
        ("beta_overflow", "--hydro.vonNeumannRichtmyerCoefficient=1e100", "single precision"),
        ("beta_nan", "--hydro.vonNeumannRichtmyerCoefficient=nan", "hydro.vonNeumannRichtmyerCoefficient"),
        ("alpha_max", "--hydro.viscosityCoefficientMax=-1", "hydro.viscosityCoefficientMax"),
        ("beta", "--hydro.vonNeumannRichtmyerCoefficient=-1", "hydro.vonNeumannRichtmyerCoefficient"),
        ("epsilon", "--hydro.viscosityEpsilon=0", "hydro.viscosityEpsilon"),
        ("gamma", "--hydro.heatCapacityRatio=1", "hydro.heatCapacityRatio"),
        ("central_effective_radius", "--scenario.disc.centralEffectiveRadius=0",
         "scenario.disc.centralEffectiveRadius"),
        ("central_softening_removed", "--hydro.centralSoftening=0", "hydro.centralSoftening"),
        ("reflection", "--hydro.yReflect=true", "hydro.yReflect"),
        ("cfl_zero", "--hydro.cfl=0", "hydro.cfl"),
        ("cfl_large", "--hydro.cfl=1.1", "hydro.cfl"),
        ("neighbors", "--hydro.neighborNumber=10", "hydro.neighborNumber"),
        ("min_radius", "--hydro.minSmoothingLength=0", "hydro.minSmoothingLength"),
        ("max_radius", "--hydro.maxSmoothingLength=1e-7", "smoothing lengths"),
        ("chemistry", "--hydro.chemistry=true", "hydro.chemistry"),
        ("conduction", "--hydro.conduction=true", "hydro.conduction"),
        ("diffusion", "--hydro.diffusion=true", "hydro.diffusion"),
    ]
    for name, option, diagnostic in invalid:
        output = run(name, metadata_options + [option], success=False)
        if diagnostic not in output:
            raise RuntimeError(f"{name} failed without its expected diagnostic: {diagnostic}")
    removedCases = []
    for option in sorted(removed_options):
        value = {"cosmology.omegaK": "0.1", "gravity.cloud": "pcs",
                 "gravity.method": "pmmm", "gravity.method.far": "pmmm",
                 "gravity.pmmmNearMethod": args.near_method, "gravity.method.near": "pp",
                 "gravity.fmmOrder": str(args.order), "gravity.pmmmOrder": str(args.order),
                 "gravity.p3mSplitCells": "1.25", "gravity.p3mCutoff": "4.5",
                 "gravity.openingAngle": "0.8", "gravity.openingSchedule": "none",
                 "run.timeIntegrator": args.time_integrator, "run.numberOfSteps": "4",
                 "output.siloInterval": "1"
                 }.get(option, "false")
        group, name = option.split(".", 1)
        # Numeric legacy booleans must not abbreviate the plummer scenario category.
        values = [value, "0", "1"] if option == "initial.plummer" else [value]
        for value in values:
            # Concurrent cases need their own INI rather than one rewritten file.
            removed_config = args.workdir / f"removed_{option}_{value}.ini"
            removed_config.write_text(f"[{group}]\n{name} = {value}\n")
            for source, arguments in (
                    ("cli", [f"--{option}={value}"]),
                    ("ini", [f"--configuration={removed_config.resolve()}"])):
                removedCases.append((source, option, (f"removed_{source}_{option}_{value}", arguments, False)))
    for (source, option, _), output in zip(removedCases, runAll([case for _, _, case in removedCases])):
        if option not in output or not re.search(r"unrecogni[sz]ed option|is ambiguous and matches", output):
            raise RuntimeError(f"removed {source} option {option} was not rejected explicitly")
    output = run("support_half_box", metadata_options + [
                                      "--hydro.minSmoothingLength=5"], success=False)
    if "half the periodic box" not in output:
        raise RuntimeError("oversized minimum smoothing length was not rejected explicitly")
    print("Grouped CLI, complete INIs, Silo units/frames, and parameter bounds passed")


if __name__ == "__main__":
    main()
