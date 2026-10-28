# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

"""Check the grouped CLI, INI precedence, and SPH parameter rejection."""

import argparse
import configparser
import math
import os
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--gravity-method", choices=("fmm", "pmmm"), required=True)
    parser.add_argument("--near-method", choices=("pp", "fmm"), required=True)
    parser.add_argument("--order", type=int, choices=range(3, 10), required=True)
    parser.add_argument("--time-integrator", choices=("leapfrog", "hierarchical"), required=True)
    parser.add_argument("--without-sph", action="store_true")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    unused_view = args.workdir / "view.txt"
    unused_view.unlink(missing_ok=True)
    for name in ("configured snapshots", "cli snapshots", "output"):
        directory = args.workdir / name
        if directory.exists():
            shutil.rmtree(directory)
    command = [str(args.executable.resolve()), "--hpx:threads=2", "--hpx:bind=none"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")

    def run(name, options, success=True, pmmm_mesh=8):
        # Most cases inspect unrelated options; supply the required PMMM mesh.
        # Mesh/INI-specific cases can disable this fixture default explicitly.
        if (args.gravity_method == "pmmm" and pmmm_mesh is not None and
                not any(option.startswith("--gravity.meshSize=") for option in options)):
            options = options + [f"--gravity.meshSize={pmmm_mesh}"]
        result = subprocess.run(command + options, cwd=args.workdir, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
                                env=environment)
        (args.workdir / f"{name}.log").write_text(result.stdout)
        if (result.returncode == 0) != success:
            raise RuntimeError(f"{name} returned {result.returncode}:\n{result.stdout}")
        return result.stdout

    def assertNumericSetting(output, key, expected):
        match = re.search(r"^" + re.escape(key) + r"\s*:\s*(\S+)", output, re.MULTILINE)
        if match is None or not math.isclose(float(match.group(1)), expected, rel_tol=1e-5):
            raise RuntimeError(f"incorrect {key}: expected {expected}")

    help_text = run("help", ["--help"])
    if not re.search(r"^\s*Time integrator:\s*" + args.time_integrator + r"\s*$",
                     help_text, re.MULTILINE):
        raise RuntimeError("help did not identify the compiled time integrator")
    option_names = set(re.findall(r"^\s+--([A-Za-z][A-Za-z0-9_.]*)", help_text, re.MULTILINE))
    required = {"help", "configuration", "run.scenario", "run.endTime", "run.verbose",
                "initial.particlesPerDimension", "compute.cuda", "compute.bucketSize",
                "compute.sphBucketSize", "compute.positionType",
                "hydro.enabled", "hydro.viscosityAlpha", "hydro.viscosityBeta",
                "hydro.viscosityEpsilon", "hydro.balsaraSwitch", "hydro.neighborNumber", "gravity.enabled",
                "gravity.kernelFamily", "gravity.kernelDegree",
                "gravity.meshSize", "gravity.pmmmNearWidth",
                "gravity.openingAngle", "gravity.openingSchedule",
                "hydro.kernelFamily", "hydro.kernelDegree",
                "gravity.effectiveRadius", "gravity.supportRadius",
                "hydro.minEffectiveRadius", "hydro.maxEffectiveRadius",
                "hydro.minSupportRadius", "hydro.maxSupportRadius",
                "hydro.centralEffectiveRadius", "hydro.centralSupportRadius",
                "hydro.zPlaneAcceleration",
                "output.silo", "output.siloInterval", "output.directory",
                "output.units", "output.frame",
                "output.xMin", "output.xMax", "output.yMin", "output.yMax",
                "output.zMin", "output.zMax", "chemistry.initialization"}
    required.update({"hydro.viscosityEvolution", "hydro.viscosityAlphaMin",
                     "hydro.viscosityDecay", "hydro.viscosityAlphaMax",
                     "hydro.viscosityDecayCD", "hydro.viscosityBetaMode",
                     "hydro.viscosityBetaOverAlpha"})
    if not required <= option_names:
        raise RuntimeError(f"grouped help is missing {sorted(required - option_names)}")
    removed_options = {"hydro.chemistry", "hydro.conduction", "hydro.diffusion",
                       "hydro.yReflect", "gravity.sphSmoothing", "cosmology.omegaK",
                       "initial.usePowerFile", "gravity.cloud", "gravity.method",
                       "gravity.pmmmNearMethod", "gravity.pmmmOrder", "gravity.fmmOrder",
                       "run.timeIntegrator", "run.hierarchicalGravity"}
    if removed_options & option_names:
        raise RuntimeError(f"removed options remain in help: {sorted(removed_options & option_names)}")
    for name in option_names - {"help", "configuration"}:
        if not re.fullmatch(r"[a-z][A-Za-z0-9]*\.[a-z][A-Za-z0-9]*", name):
            raise RuntimeError(f"option does not use groupName.optionName: {name}")

    # Mix section-style and dotted keys. Choosing a checkpoint serialization
    # regression exercises full option setup without evolving a simulation.
    config = args.workdir / "grouped.ini"
    config.write_text("""analysis.tracerCount = 41
[run]
test = checkpointState
maxIterations = 7
[initial]
particlesPerDimension = 8
[compute]
cuda = false
bucketSize = 64
sphBucketSize = 73
[hydro]
enabled = false
viscosityAlpha = 0.75
viscosityBeta = 1.5
viscosityEpsilon = 0.02
balsaraSwitch = false
viscosityEvolution = none
viscosityAlphaMin = 0.15
viscosityDecay = 0.3
viscosityAlphaMax = 1.5
viscosityDecayCD = 0.1
viscosityBetaMode = constant
viscosityBetaOverAlpha = 2.5
zPlaneAcceleration = 0.25
gamma = 1.4
neighborNumber = 48
cfl = 0.3
[lightcone]
mapSize = 8
""")
    configured = run("config", [f"--configuration={config.resolve()}",
                                "--run.maxIterations=3", "--hydro.viscosityAlpha=1.25"])
    expected = {"run.maxIterations": 3.0, "initial.particlesPerDimension": 8.0,
                "analysis.tracerCount": 41.0, "hydro.viscosityAlpha": 1.25,
                "hydro.viscosityBeta": 1.5, "hydro.viscosityEpsilon": 0.02,
                "hydro.viscosityAlphaMin": 0.15, "hydro.viscosityDecay": 0.3,
                "hydro.viscosityAlphaMax": 1.5, "hydro.viscosityDecayCD": 0.1,
                "hydro.viscosityBetaOverAlpha": 2.5,
                "hydro.gamma": 1.4, "hydro.neighborNumber": 48.0,
                "hydro.cfl": 0.3, "hydro.zPlaneAcceleration": 0.25, "lightcone.mapSize": 8.0,
                "hydro.minSupportRadius": 1.0e-6, "hydro.maxSupportRadius": 4.0,
                "gravity.kernelDegree": 4.0,
                "compute.bucketSize": 64.0, "compute.sphBucketSize": 73.0}
    for name, value in expected.items():
        match = re.search(r"^" + re.escape(name) + r"\s*:\s*(\S+)", configured, re.MULTILINE)
        if match is None or abs(float(match.group(1)) - value) > 1.0e-6:
            raise RuntimeError(f"configured {name} did not equal {value}")
    if "Checkpoint state and mode validation passed" not in configured:
        raise RuntimeError("configuration run did not finish the checkpoint regression")

    metadata_options = ["--run.test=checkpointState", "--compute.cuda=false",
                        "--hydro.enabled=false", "--initial.particlesPerDimension=8"]

    if not re.search(r"^hydro\.balsaraSwitch\s*:\s*F\s*$", configured, re.MULTILINE):
        raise RuntimeError("INI hydro.balsaraSwitch=false was not applied")
    balsara_default = run("balsara_default", metadata_options)
    def assertCompiledSettings(output):
        for key, value in (("build.gravityMethod", args.gravity_method),
                           ("build.pmmmNearMethod", args.near_method),
                           ("build.timeIntegrator", args.time_integrator)):
            if not re.search(r"^" + re.escape(key) + r"\s*:\s*" + value + r"\s*$",
                             output, re.MULTILINE):
                raise RuntimeError(f"compiled metadata did not retain {key}={value}")
        assertNumericSetting(output, "build.fmmOrder", args.order)
        expected_hierarchical = "T" if args.time_integrator == "hierarchical" else "F"
        if not re.search(r"^run\.hierarchicalGravity\s*:\s*" + expected_hierarchical + r"\s*$",
                         output, re.MULTILINE):
            raise RuntimeError("gravity stepping did not match the compiled time integrator")

    assertCompiledSettings(balsara_default)
    assertCompiledSettings(configured)
    for scenario in ("plummer", "glass"):
        output = run(f"integrator_{scenario}", metadata_options + [f"--run.scenario={scenario}"])
        assertCompiledSettings(output)
    if args.gravity_method == "fmm":
        assertNumericSetting(balsara_default, "gravity.openingAngle", 0.8)
        scalar_angle = run("fmm_scalar_angle", metadata_options + ["--gravity.openingAngle=0.73"])
        assertNumericSetting(scalar_angle, "gravity.openingAngle", 0.73)
        schedule_config = args.workdir / "opening_schedule.ini"
        schedule_config.write_text("[gravity]\nopeningSchedule = 2:0.61,49:0.41,20:0.51\n"
                                   "openingAngle = 0.73\n[cosmology]\nstartRedshift = 30\n")
        schedule_output = run("opening_schedule_ini", metadata_options + [
            f"--configuration={schedule_config.resolve()}"])
        assertNumericSetting(schedule_output, "gravity.openingAngle", 0.51)
        if not re.search(r"^gravity\.openingSchedule\s*:\s*2:0\.61,49:0\.41,20:0\.51\s*$",
                         schedule_output, re.MULTILINE):
            raise RuntimeError("INI opening schedule was not retained in option metadata")
        schedule_output = run("opening_schedule_cli_override", metadata_options + [
            f"--configuration={schedule_config.resolve()}", "--cosmology.startRedshift=46",
            "--gravity.openingSchedule=20:0.55,49:0.44,2:0.66"])
        assertNumericSetting(schedule_output, "gravity.openingAngle", 0.44)
        schedule_output = run("opening_schedule_empty_override", metadata_options + [
            f"--configuration={schedule_config.resolve()}", "--gravity.openingSchedule=none"])
        assertNumericSetting(schedule_output, "gravity.openingAngle", 0.73)
        schedule_output = run("opening_schedule_tie", metadata_options + [
            "--cosmology.startRedshift=34.5", "--gravity.openingSchedule=20:0.51,49:0.41"])
        assertNumericSetting(schedule_output, "gravity.openingAngle", 0.41)
    for index, schedule in enumerate((
            "49", "49:0.41,", "49:0.41:0.51", "49:0.41,49:0.51",
            "nan:0.41", "49:nan", "inf:0.41", "49:inf", "-1:0.41",
            "49:0", "49:-0.41")):
        rejected = run(f"opening_schedule_invalid_{index}", metadata_options + [
            f"--gravity.openingSchedule={schedule}"], success=False)
        if "gravity.openingSchedule" not in rejected:
            raise RuntimeError("invalid opening schedule lacked an option-specific diagnostic")
    unsupported_schedules = [("plummer", ["--run.scenario=plummer"])]
    if args.gravity_method == "pmmm":
        unsupported_schedules.append(("pmmm", []))
    for name, options in unsupported_schedules:
        rejected = run(f"opening_schedule_unsupported_{name}", metadata_options + options + [
            "--gravity.openingSchedule=49:0.41,20:0.51"], success=False)
        if "gravity.openingSchedule" not in rejected:
            raise RuntimeError(f"opening schedule was not explicitly rejected for {name}")
    run("gravity_negative_mesh", metadata_options + ["--gravity.meshSize=-1"], success=False)
    if args.gravity_method == "pmmm":
        for near_width in (1, 2, 3, 4):
            mesh_size = 2 * near_width + 1
            pmmm_output = run(f"pmmm_metadata_{near_width}", metadata_options + [
                f"--gravity.meshSize={mesh_size}", f"--gravity.pmmmNearWidth={near_width}"])
            assertCompiledSettings(pmmm_output)
            for key, expected in (("gravity.meshSize", mesh_size),
                                  ("gravity.pmmmNearWidth", near_width)):
                assertNumericSetting(pmmm_output, key, expected)
            assertNumericSetting(pmmm_output, "gravity.openingAngle", math.sqrt(3) / (near_width + 1))
        assertNumericSetting(balsara_default, "gravity.pmmmNearWidth", 2)
        assertNumericSetting(balsara_default, "gravity.pmmmBatchParticles", 4096)
        assertNumericSetting(balsara_default, "gravity.openingAngle", math.sqrt(3) / 3)
        for scenario in ("cosmology", "plummer"):
            pmmm_output = run(f"pmmm_derived_angle_{scenario}", metadata_options + [
                f"--run.scenario={scenario}", "--gravity.openingAngle=0.99"])
            assertCompiledSettings(pmmm_output)
            assertNumericSetting(pmmm_output, "gravity.openingAngle", math.sqrt(3) / 3)
    for selected_field in ("saveAcceleration", "savePotential"):
        output = run(f"diagnostic_{selected_field}", metadata_options + [
            f"--analysis.{selected_field}=true"])
        for field in ("saveForce", "saveAcceleration", "savePotential"):
            expected = "T" if field == selected_field else "F"
            if not re.search(r"^analysis\." + field + r"\s*:\s*" + expected + r"\s*$",
                             output, re.MULTILINE):
                raise RuntimeError("independent saved field option enabled the wrong diagnostic")
    if args.gravity_method == "pmmm":
        for index, invalid in enumerate((
                [], ["--gravity.meshSize=4"],
                ["--gravity.meshSize=3", "--gravity.pmmmNearWidth=0"],
                ["--gravity.meshSize=11", "--gravity.pmmmNearWidth=5"],
                ["--gravity.meshSize=6", "--gravity.pmmmNearWidth=3"],
                ["--gravity.meshSize=8", "--gravity.pmmmNearWidth=4"],
                ["--gravity.meshSize=2", "--gravity.pmmmNearWidth=1"],
                ["--gravity.meshSize=7", "--gravity.supportRadius=8"],
                ["--gravity.meshSize=7", "--gravity.pmmmBatchParticles=0"],
                ["--gravity.meshSize=7", "--gravity.pmmmBatchParticles=-1"],
                ["--gravity.meshSize=2147483647"])):
            run(f"pmmm_invalid_{index}", metadata_options + invalid,
                success=False, pmmm_mesh=None)
        pmmm_config = args.workdir / "pmmm.ini"
        pmmm_config.write_text("[gravity]\nmeshSize = 7\npmmmNearWidth = 2\nopeningAngle = 0.1\n")
        pmmm_output = run("pmmm_ini_cli", metadata_options + [
            f"--configuration={pmmm_config.resolve()}", "--gravity.pmmmNearWidth=1",
            "--gravity.openingAngle=0.9"], pmmm_mesh=None)
        assertCompiledSettings(pmmm_output)
        assertNumericSetting(pmmm_output, "gravity.meshSize", 7)
        assertNumericSetting(pmmm_output, "gravity.pmmmNearWidth", 1)
        assertNumericSetting(pmmm_output, "gravity.openingAngle", math.sqrt(3) / 2)

    sph_options = [option for option in metadata_options
                   if not option.startswith("--hydro.enabled=")]
    sph_gravity_error = "SPH currently requires an FMM build when gravity.enabled=true"
    if args.without_sph:
        sph_build_error = "COSMICTIGER_WITH_SPH=ON"
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
            "--run.scenario=sod", "--compute.cuda=false", "--run.maxIterations=0",
            "--initial.particlesPerDimension=8"], success=False)
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
    elif args.time_integrator == "hierarchical":
        sph_integrator_error = "COSMICTIGER_TIME_INTEGRATOR=leapfrog"
        for gravity_enabled in ("true", "false"):
            rejected = run(f"hierarchical_sph_gravity_{gravity_enabled}", sph_options + [
                "--hydro.enabled=true", f"--gravity.enabled={gravity_enabled}"], success=False)
            if sph_integrator_error not in rejected:
                raise RuntimeError("hierarchical SPH lacked integrator build guidance")
        forced_sph = [option for option in metadata_options
                      if not option.startswith("--run.test=")]
        rejected = run("hierarchical_forced_sph_rejected", forced_sph + [
            "--run.test=sphGravity"], success=False)
        if sph_integrator_error not in rejected:
            raise RuntimeError("hierarchical SPH restriction was bypassed through run.test")
        rejected = run("hierarchical_scenario_sph_rejected", [
            "--run.scenario=sod", "--compute.cuda=false",
            "--initial.particlesPerDimension=8"], success=False)
        if sph_integrator_error not in rejected:
            raise RuntimeError("hierarchical SPH restriction was bypassed through scenario defaults")
        sph_config = args.workdir / "hierarchical_sph.ini"
        sph_config.write_text("[hydro]\nenabled = true\n")
        rejected = run("hierarchical_ini_sph_rejected", sph_options + [
            f"--configuration={sph_config.resolve()}"], success=False)
        if sph_integrator_error not in rejected:
            raise RuntimeError("hierarchical SPH restriction was bypassed through an INI file")
    elif args.gravity_method == "pmmm":
        rejected = run("pmmm_sph_rejected", sph_options + [
            "--hydro.enabled=true", "--gravity.enabled=true"], success=False)
        if sph_gravity_error not in rejected:
            raise RuntimeError("PMMM with SPH lacked the solver compatibility diagnostic")
        forced_sph = [option for option in metadata_options
                      if not option.startswith("--run.test=")]
        rejected = run("pmmm_forced_sph_rejected", forced_sph + [
            "--run.test=sphGravity", "--gravity.enabled=true"], success=False)
        if sph_gravity_error not in rejected:
            raise RuntimeError("PMMM bypassed the SPH restriction through run.test")
        rejected = run("pmmm_scenario_sph_rejected", [
            "--run.scenario=sod", "--compute.cuda=false", "--gravity.enabled=true",
            "--initial.particlesPerDimension=8"], success=False)
        if sph_gravity_error not in rejected:
            raise RuntimeError("PMMM bypassed the SPH restriction through scenario defaults")
        sph_config = args.workdir / "pmmm_sph.ini"
        sph_config.write_text("[gravity]\nenabled = true\nmeshSize = 8\n"
                              "[hydro]\nenabled = true\n")
        rejected = run("pmmm_ini_sph_rejected", sph_options + [
            f"--configuration={sph_config.resolve()}"], success=False)
        if sph_gravity_error not in rejected:
            raise RuntimeError("PMMM bypassed the SPH restriction through an INI file")
        output = run("pmmm_hydro_without_gravity", sph_options + [
            "--hydro.enabled=true", "--gravity.enabled=false"])
        assertCompiledSettings(output)
    else:
        output = run("fmm_sph_allowed", sph_options + [
            "--gravity.enabled=true", "--hydro.enabled=true"])
        assertCompiledSettings(output)
    if not re.search(r"^hydro\.balsaraSwitch\s*:\s*T\s*$", balsara_default, re.MULTILINE):
        raise RuntimeError("Balsara switch must default to enabled")
    balsara_override = run("balsara_cli_override", [f"--configuration={config.resolve()}",
                                                 "--hydro.balsaraSwitch=true"])
    if not re.search(r"^hydro\.balsaraSwitch\s*:\s*T\s*$", balsara_override, re.MULTILINE):
        raise RuntimeError("CLI hydro.balsaraSwitch did not override INI")

    for key, expected_value in (("hydro.viscosityEvolution", "none"),
                                ("hydro.viscosityBetaMode", "constant")):
        if not re.search(r"^" + re.escape(key) + r"\s*:\s*" + expected_value + r"\s*$",
                         configured, re.MULTILINE):
            raise RuntimeError(f"INI {key} was not applied")
    for key, expected_value in (("hydro.viscosityEvolution", "morrisAndMonaghan"),
                                ("hydro.viscosityBetaMode", "ratio")):
        if not re.search(r"^" + re.escape(key) + r"\s*:\s*" + expected_value + r"\s*$",
                         balsara_default, re.MULTILINE):
            raise RuntimeError(f"unexpected default for {key}")
    viscosity_override = run("variable_viscosity_cli_override", [
        f"--configuration={config.resolve()}", "--hydro.viscosityEvolution=cullenAndDehnen",
        "--hydro.viscosityBetaMode=ratio", "--hydro.viscosityBetaOverAlpha=3"])
    for key, expected_value in (("hydro.viscosityEvolution", "cullenAndDehnen"),
                                ("hydro.viscosityBetaMode", "ratio"),
                                ("hydro.viscosityBetaOverAlpha", "3")):
        match = re.search(r"^" + re.escape(key) + r"\s*:\s*(\S+)",
                          viscosity_override, re.MULTILINE)
        if match is None or (float(match.group(1)) != 3 if expected_value == "3"
                             else match.group(1) != expected_value):
            raise RuntimeError(f"CLI {key} did not override INI")

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
            "--hydro.neighborNumber=200", "--gravity.softening=0.02", "--gravity.timestepEta=0.15"])
        for key, value in (("gravity.kernelFamily", gravity_family),
                           ("gravity.kernelDegree", str(gravity_degree)),
                           ("hydro.kernelFamily", hydro_family),
                           ("hydro.kernelDegree", str(hydro_degree))):
            if not re.search(r"^" + re.escape(key) + r"\s*:\s*" + value + r"\s*$", output, re.MULTILINE):
                raise RuntimeError(f"independent kernel setting {key} was not applied")
        for key, value in (("gravity.softening", 0.02), ("gravity.timestepEta", 0.15)):
            match = re.search(r"^" + re.escape(key) + r"\s*:\s*(\S+)", output, re.MULTILINE)
            if match is None or abs(float(match.group(1)) - value) > 1.e-8:
                raise RuntimeError(f"kernel-independent setting {key} changed meaning")
    for index, invalid in enumerate((
            ["--gravity.kernelFamily=gaussian"], ["--hydro.kernelFamily=wendland", "--hydro.kernelDegree=9"],
            ["--gravity.kernelFamily=ferrers", "--gravity.kernelDegree=9"],
            ["--hydro.kernelDegree=10"], ["--gravity.kernelDegree=-1"],
            ["--gravity.kernelFamily=spline", "--gravity.kernelDegree=1"],
            ["--hydro.kernelFamily=spline", "--hydro.kernelDegree=1"],
            ["--hydro.kernelFamily=wendland", "--hydro.kernelDegree=8", "--hydro.neighborNumber=12"])):
        run(f"invalid_kernel_{index}", metadata_options + invalid, success=False)

    chemistry_fields = ("initialTemperature", "hydrogenIonFraction",
                        "heliumIonFraction", "heliumDoubleIonFraction")

    if not args.without_sph and args.time_integrator == "leapfrog":
        # Both checkpoint layouts round-trip: zero retains the existing SPH block,
        # while nonzero acceleration appends the flagged z-plane force parameter.
        hydro_metadata_options = [option for option in metadata_options
                                  if not option.startswith("--hydro.enabled=")] + [
                                      "--hydro.enabled=true"]
        if args.gravity_method == "pmmm":
            hydro_metadata_options.append("--gravity.enabled=false")
        for acceleration in (0.0, 0.5):
            output = run(f"z_plane_checkpoint_{acceleration}", hydro_metadata_options + [
                f"--configuration={config.resolve()}",
                f"--hydro.zPlaneAcceleration={acceleration}"])
            assertNumericSetting(output, "hydro.zPlaneAcceleration", acceleration)
            if "Checkpoint state and mode validation passed" not in output:
                raise RuntimeError("z-plane acceleration checkpoint state did not round-trip")

        chemistry_options = [option for option in metadata_options
                             if not option.startswith("--hydro.enabled=")] + [
                                 "--hydro.enabled=true", "--chemistry.enabled=true",
                                 "--cosmology.startRedshift=49"]
        if args.gravity_method == "pmmm":
            chemistry_options.append("--gravity.enabled=false")

        def chemistrySettings(output):
            result = {}
            for field in chemistry_fields:
                match = re.search(r"^chemistry\." + field + r"\s*:\s*(\S+)", output, re.MULTILINE)
                if match is None or not math.isfinite(float(match.group(1))):
                    raise RuntimeError(f"missing finite resolved chemistry.{field}")
                result[field] = float(match.group(1))
            return result

        native = chemistrySettings(run("chemistry_native_defaults", chemistry_options))
        if not (10 < native["initialTemperature"] < 100 and
                1e-5 < native["hydrogenIonFraction"] < 1e-3):
            raise RuntimeError(f"Native chemistry defaults are not a cold residual plasma: {native}")
        earlier = chemistrySettings(run("chemistry_native_earlier_epoch", [
            option for option in chemistry_options if not option.startswith("--cosmology.startRedshift=")
        ] + ["--cosmology.startRedshift=99"]))
        if earlier["initialTemperature"] <= native["initialTemperature"]:
            raise RuntimeError("Native chemistry initialization ignored the requested starting epoch")
        restarted = chemistrySettings(run("chemistry_restart_resolution",
                                          chemistry_options + ["--checkpoint.readIndex=1"]))
        if restarted != native:
            raise RuntimeError("Restart option resolution changed native initialization metadata")
        manual = run("chemistry_manual_defaults",
                     chemistry_options + ["--chemistry.initialization=manual"])
        for field, value in zip(chemistry_fields, (1000.0, 2e-4, 0.0, 0.0)):
            assertNumericSetting(manual, f"chemistry.{field}", value)
        for field, value in zip(chemistry_fields, (777.0, 0.01, 0.1, 0.02)):
            output = run(f"chemistry_override_{field}", chemistry_options + [f"--chemistry.{field}={value}"])
            for checked in chemistry_fields:
                assertNumericSetting(output, f"chemistry.{checked}",
                                     value if checked == field else native[checked])
        chemistry_config = args.workdir / "chemistry_initialization.ini"
        chemistry_config.write_text("[chemistry]\ninitialTemperature = 500\nhydrogenIonFraction = 0.01\n")
        output = run("chemistry_override_precedence", chemistry_options + [
            f"--configuration={chemistry_config.resolve()}", "--chemistry.initialTemperature=750"])
        for field, value in (("initialTemperature", 750), ("hydrogenIonFraction", 0.01),
                             ("heliumIonFraction", native["heliumIonFraction"]),
                             ("heliumDoubleIonFraction", native["heliumDoubleIonFraction"])):
            assertNumericSetting(output, f"chemistry.{field}", value)
        molecular_native = chemistrySettings(run(
            "chemistry_molecular_native", chemistry_options + ["--chemistry.network=molecular9"]))
        if molecular_native != native:
            raise RuntimeError("Cold molecular initialization differs from the native atomic history")
        for label, extra in (("unknown_mode", ["--chemistry.initialization=unknown"]),
                             ("molecular_hot", ["--chemistry.network=molecular9",
                                                "--chemistry.initialTemperature=6001"])):
            output = run(f"chemistry_{label}", chemistry_options + extra, success=False)
            if "chemistry" not in output.lower():
                raise RuntimeError(f"Invalid chemistry initialization {label} lacked a diagnostic")
        output = run("chemistry_molecular_below_floor", [
            option for option in chemistry_options if not option.startswith("--cosmology.startRedshift=")
        ] + ["--cosmology.startRedshift=10", "--chemistry.network=molecular9"], success=False)
        if "chemistry" not in output.lower():
            raise RuntimeError("Native molecular initialization below 10 K lacked a diagnostic")

    # Photon density is fixed by the requested blackbody temperature and H0;
    # neutrinos add radiation without changing that photon density.
    for neff, temperature in ((0.0, 2.7255), (3.046, 2.7255), (6.0, 3.0)):
        output = run(f"radiation_{neff}_{temperature}", metadata_options + [
            f"--cosmology.nEff={neff}", f"--cosmology.thetaCmb={temperature / 2.73:.17g}"])
        hubble = 0.6732 * 1.e7 / 3.086e24
        criticalDensity = 3.0 * hubble**2 / (8.0 * math.pi * 6.67259e-8)
        photons = (4.0 * 5.67051e-5 / 2.99792458e10) * temperature**4 / (
            criticalDensity * 2.99792458e10**2)
        neutrinos = photons * neff * (7.0 / 8.0) * (4.0 / 11.0)**(4.0 / 3.0)
        for key, expected in (("omegaGamma", photons), ("omegaNu", neutrinos),
                              ("omegaR", photons + neutrinos)):
            assertNumericSetting(output, f"cosmology.{key}", expected)

    for index, invalid in enumerate((
            "--cosmology.nEff=-1", "--cosmology.thetaCmb=0", "--cosmology.hubble=0",
            "--cosmology.heliumFraction=1", "--cosmology.sigma8=nan",
            "--cosmology.sigma8=-0.1")):
        run(f"invalid_cosmology_{index}", metadata_options + [invalid], success=False)

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
    if cosmologySetting(lambda_dense, "omegaLambda") >= 0.0:
        raise RuntimeError("automatic Lambda could not derive a negative flat-universe value")
    assertNumericSetting(lambda_auto, "cosmology.omegaLambda",
                         cosmologySetting(lambda_baseline, "omegaLambda"))
    for value in (-1.0, -0.25, 0.0):
        output = run(f"lambda_literal_{value}", metadata_options + [
            f"--cosmology.omegaLambda={value}"])
        assertNumericSetting(output, "cosmology.omegaLambda", value)
    lambda_config = args.workdir / "signed_lambda.ini"
    lambda_config.write_text("[cosmology]\nomegaLambda = -0.25\n")
    output = run("lambda_ini", metadata_options + [f"--configuration={lambda_config.resolve()}"])
    assertNumericSetting(output, "cosmology.omegaLambda", -0.25)
    output = run("lambda_cli_override", metadata_options + [
        f"--configuration={lambda_config.resolve()}", "--cosmology.omegaLambda=auto"])
    assertNumericSetting(output, "cosmology.omegaLambda",
                         cosmologySetting(lambda_baseline, "omegaLambda"))
    for index, value in enumerate(("automatic", "AUTO", "-1junk", "1.2.3", "nan", "inf",
                                   "-inf", "1e1000")):
        output = run(f"invalid_lambda_{index}", metadata_options + [
            f"--cosmology.omegaLambda={value}"], success=False)
        if "cosmology" not in output.lower():
            raise RuntimeError(f"invalid Lambda {value} lacked a cosmology diagnostic")
    for key in ("startRedshift", "endRedshift"):
        for index, value in enumerate(("-1", "-1.01", "nan", "inf")):
            output = run(f"invalid_{key}_{index}", metadata_options + [
                f"--cosmology.{key}={value}"], success=False)
            if "cosmology" not in output.lower():
                raise RuntimeError(f"invalid {key}={value} lacked a cosmology diagnostic")
    homogeneous = run("homogeneous_cosmology", metadata_options + ["--cosmology.sigma8=0"])
    assertNumericSetting(homogeneous, "cosmology.sigma8", 0.0)
    if "Checkpoint state and mode validation passed" not in homogeneous:
        raise RuntimeError("homogeneous cosmology options did not complete checkpoint validation")

    def assertIgnoredOption(output, ignoredKey):
        if not any(ignoredKey in line and "warning" in line.lower() and "ignored" in line.lower()
                   for line in output.splitlines()):
            raise RuntimeError(f"missing warning that {ignoredKey} is ignored")

    def assertNoRadiusWarning(output):
        radiusKeys = ("effectiveRadius", "EffectiveRadius", "supportRadius", "SupportRadius",
                      "gravity.softening", "hydro.centralSoftening")
        if any("warning" in line.lower() and any(key in line for key in radiusKeys)
               for line in output.splitlines()):
            raise RuntimeError("a single radius representation produced a warning")

    defaults = run("independent_defaults", metadata_options)
    for key, value in (("gravity.kernelDegree", 4),
                       ("hydro.neighborNumber", 42), ("hydro.kernelDegree", 3),
                       ("hydro.viscosityAlphaMin", 0.1), ("hydro.viscosityDecay", 0.2),
                       ("hydro.viscosityAlphaMax", 2), ("hydro.viscosityDecayCD", 0.05),
                       ("hydro.viscosityBetaOverAlpha", 2),
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
        for representation, value in (("effectiveRadius", 0.02), ("supportRadius", 0.02 * ratio)):
            output = run(f"radius_{family}_{degree}_{representation}", common + [
                f"--gravity.{representation}={value}"])
            assertNumericSetting(output, "gravity.effectiveRadius", 0.02)
            assertNumericSetting(output, "gravity.supportRadius", 0.02 * ratio)
            assertNoRadiusWarning(output)

    gravityRatio = 1.5
    hydroRatio = math.sqrt(10 / 3)
    radiusPairs = (
        ("gravity.effectiveRadius", "gravity.supportRadius", 0.02, gravityRatio),
        ("hydro.centralEffectiveRadius", "hydro.centralSupportRadius", 0.02, gravityRatio),
        ("hydro.minEffectiveRadius", "hydro.minSupportRadius", 0.01, hydroRatio),
        ("hydro.maxEffectiveRadius", "hydro.maxSupportRadius", 0.1, hydroRatio),
    )
    radiusConfig = args.workdir / "radius.ini"
    for effectiveKey, supportKey, effectiveRadius, ratio in radiusPairs:
        supportRadius = effectiveRadius * ratio
        conflictingSupport = 2 * supportRadius
        for key, value in ((effectiveKey, effectiveRadius), (supportKey, supportRadius)):
            output = run(f"radius_only_{key}", metadata_options + [f"--{key}={value}"])
            assertNumericSetting(output, supportKey, supportRadius)
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
            assertNumericSetting(output, supportKey, supportRadius)
            assertIgnoredOption(output, supportKey)

        # A losing support radius is ignored before physical validation.
        for invalidValue in ("0", "-1", "nan", "inf"):
            output = run(f"radius_ignored_{supportKey}_{invalidValue}", metadata_options +
                         [f"--{effectiveKey}={effectiveRadius}", f"--{supportKey}={invalidValue}"])
            assertNumericSetting(output, supportKey, supportRadius)
            assertIgnoredOption(output, supportKey)

        # Ordinary CLI-over-INI precedence still applies within each key.
        for key, value in ((effectiveKey, effectiveRadius), (supportKey, supportRadius)):
            radiusConfig.write_text(f"{key} = {2 * value}\n")
            output = run(f"radius_same_key_{key}", metadata_options +
                         [f"--configuration={radiusConfig.resolve()}", f"--{key}={value}"])
            assertNumericSetting(output, supportKey, supportRadius)
            assertNoRadiusWarning(output)

    for effectiveKey, supportKey, aliasKey in (
            ("gravity.effectiveRadius", "gravity.supportRadius", "gravity.softening"),
            ("hydro.centralEffectiveRadius", "hydro.centralSupportRadius", "hydro.centralSoftening")):
        # Legacy effective-radius aliases retain their meaning, and explicitly
        # supplied aliases also take precedence over support-radius inputs.
        output = run(f"radius_alias_{aliasKey}", metadata_options + [f"--{aliasKey}=0.02"])
        assertNumericSetting(output, effectiveKey, 0.02)
        assertNumericSetting(output, supportKey, 0.02 * gravityRatio)
        assertNoRadiusWarning(output)
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

    # Defaulted legacy aliases and defaulted support bounds are fallbacks, not
    # explicitly selected representations that can conflict with one another.
    output = run("radius_defaults", metadata_options)
    assertNoRadiusWarning(output)
    for effectiveKey, supportKey, _, _ in radiusPairs:
        for key in (effectiveKey, supportKey):
            for value in ("0", "-1", "nan", "inf"):
                run(f"invalid_{key}_{value}", metadata_options + [f"--{key}={value}"], success=False)

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
    output_config.write_text("[output]\nsilo = true\nsiloInterval = 7\n"
                             "units = cosmological\nframe = lab\n"
                             "directory = configured snapshots\nxMin = 0.25\nxMax = 0.75\n")
    configured = run("silo_ini", metadata_options + [f"--configuration={output_config.resolve()}"])
    assertOutputSettings(configured, "cosmological", "lab")
    if "Silo particle output: x [0.25, 0.75], y [0, 1], z [0, 1]" not in configured:
        raise RuntimeError("INI Silo bounds were not applied")
    if "every 7 synchronized coarse steps" not in configured:
        raise RuntimeError("INI Silo cadence was not applied")
    if not (args.workdir / "configured snapshots").is_dir():
        raise RuntimeError("relative Silo output directory was not created")

    cli_directory = (args.workdir / "cli snapshots").resolve()
    overridden = run("silo_cli", metadata_options + [f"--configuration={output_config.resolve()}",
                                                    "--output.xMin=0.5", "--output.siloInterval=3",
                                                    "--output.units=both", "--output.frame=both",
                                                    f"--output.directory={cli_directory}"])
    assertOutputSettings(overridden, "both", "both")
    if "Silo particle output: x [0.5, 0.75], y [0, 1], z [0, 1]" not in overridden:
        raise RuntimeError("CLI bounds did not override INI bounds")
    if "every 3 synchronized coarse steps" not in overridden or not cli_directory.is_dir():
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
    optional_keys = {"gravity.effectiveRadius", "gravity.supportRadius",
                     "hydro.centralEffectiveRadius", "hydro.centralSupportRadius",
                     "hydro.minEffectiveRadius", "hydro.maxEffectiveRadius",
                     *(f"chemistry.{field}" for field in chemistry_fields)}
    expected_keys = option_names - {"help", "configuration"} - optional_keys
    preset_curvatures = {"lcdm_present.ini": 0.0,
                         "closed_no_dark_energy.ini": -1.000092093902909,
                         "omega_k_negative.ini": -0.1, "omega_k_positive.ini": 0.1,
                         "critical.ini": 0.0,
                         "subcritical_no_dark_energy.ini": 0.6864420835574896}
    checked_presets = set()
    for path in [root / "default.ini", *sorted((root / "configurations").glob("*.ini"))]:
        ini = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
        ini.optionxform = str
        with path.open() as stream:
            ini.read_file(stream)
        actual_keys = {f"{section}.{key}" for section in ini.sections() for key in ini[section]}
        if actual_keys != expected_keys:
            raise RuntimeError(f"{path.name} option coverage differs: {sorted(actual_keys ^ expected_keys)}")
        # Radius alternatives and manual chemistry overrides remain commented,
        # preserving one radius representation and native initialization.
        documented_alternatives = set()
        section = None
        for line in path.read_text().splitlines():
            section_match = re.fullmatch(r"\s*\[([^]]+)\]\s*", line)
            if section_match:
                section = section_match.group(1)
            alternative_match = re.match(r"\s*[#;]\s*([A-Za-z][A-Za-z0-9]*)\s*=", line)
            if section and alternative_match:
                documented_alternatives.add(f"{section}.{alternative_match.group(1)}")
        if not optional_keys <= documented_alternatives:
            raise RuntimeError(f"{path.name} is missing commented optional settings: "
                               f"{sorted(optional_keys - documented_alternatives)}")
        if ini["output"]["units"] != "code" or ini["output"]["frame"] != "comoving":
            raise RuntimeError(f"{path.name} must preserve native output defaults")
        preset_options = [f"--configuration={path}", "--chemistry.enabled=false"]
        if args.without_sph and ini["run"]["scenario"] not in ("cosmology", "plummer", "glass"):
            # Keep validating every example's option coverage and syntax when its
            # physical scenario is unavailable in the compiled executable.
            preset_options.append("--run.scenario=cosmology")
        output = run(f"complete_{path.stem}", metadata_options + preset_options)
        if "Checkpoint state and mode validation passed" not in output:
            raise RuntimeError(f"{path.name} did not finish metadata validation")
        if path.name in preset_curvatures:
            assertNumericSetting(output, "cosmology.omegaK", preset_curvatures[path.name])
            if path.name in {"closed_no_dark_energy.ini", "critical.ini",
                             "subcritical_no_dark_energy.ini"}:
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
        ("silo_interval_zero", "--output.siloInterval=0", "output.siloInterval"),
        ("silo_interval_negative", "--output.siloInterval=-1", "output.siloInterval"),
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
        ("gamma_nan", "--hydro.gamma=nan", "hydro.gamma"),
        ("gamma_inf", "--hydro.gamma=inf", "hydro.gamma"),
        ("alpha_nan", "--hydro.viscosityAlpha=nan", "hydro.viscosityAlpha"),
        ("epsilon_inf", "--hydro.viscosityEpsilon=inf", "hydro.viscosityEpsilon"),
        ("cfl_nan", "--hydro.cfl=nan", "hydro.cfl"),
        ("neighbors_inf", "--hydro.neighborNumber=inf", "hydro.neighborNumber"),
        ("z_plane_negative", "--hydro.zPlaneAcceleration=-0.5", "hydro.zPlaneAcceleration"),
        ("z_plane_nan", "--hydro.zPlaneAcceleration=nan", "hydro.zPlaneAcceleration"),
        ("z_plane_inf", "--hydro.zPlaneAcceleration=inf", "hydro.zPlaneAcceleration"),
        ("z_plane_overflow", "--hydro.zPlaneAcceleration=1e100", "single precision"),
        ("z_plane_underflow", "--hydro.zPlaneAcceleration=1e-100", "single precision"),
        ("steps_zero", "--run.numberOfSteps=0", "run.numberOfSteps"),
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
        ("softening_zero", "--gravity.softening=0", "gravity.softening"),
        ("softening_negative", "--gravity.softening=-0.1", "gravity.softening"),
        ("softening_nan", "--gravity.softening=nan", "gravity.softening"),
        ("softening_inf", "--gravity.softening=inf", "gravity.softening"),
        ("softening_underflow", "--gravity.softening=1e-100", "gravity.softening"),
        ("smoothing_obsolete", "--gravity.sphSmoothing=true", "gravity.sphSmoothing"),
        ("adaptive_option_removed", "--gravity.softeningMode=adaptive", "gravity.softeningMode"),
        ("alpha_overflow", "--hydro.viscosityAlpha=1e100", "single precision"),
        ("epsilon_underflow", "--hydro.viscosityEpsilon=1e-100", "single precision"),
        ("gamma_rounding", "--hydro.gamma=1.00000001", "single precision"),
        ("neighbors_rounding", "--hydro.neighborNumber=10.66666667", "single precision"),
        ("cfl_underflow", "--hydro.cfl=1e-100", "single precision"),
        ("support_underflow", "--hydro.minSupportRadius=1e-100", "single precision"),
        ("viscosity_evolution", "--hydro.viscosityEvolution=unknown", "hydro.viscosityEvolution"),
        ("viscosity_alpha_max", "--hydro.viscosityAlphaMax=0", "hydro.viscosityAlphaMax"),
        ("viscosity_decay_cd", "--hydro.viscosityDecayCD=0", "hydro.viscosityDecayCD"),
        ("variable_viscosity_mode", "--hydro.viscosityBetaMode=unknown", "hydro.viscosityBetaMode"),
        ("alpha_floor_zero", "--hydro.viscosityAlphaMin=0", "hydro.viscosityAlphaMin"),
        ("alpha_floor_underflow", "--hydro.viscosityAlphaMin=1e-100", "single precision"),
        ("viscosity_decay_zero", "--hydro.viscosityDecay=0", "hydro.viscosityDecay"),
        ("viscosity_decay_overflow", "--hydro.viscosityDecay=1e100", "single precision"),
        ("beta_ratio_negative", "--hydro.viscosityBetaOverAlpha=-1", "hydro.viscosityBetaOverAlpha"),
        ("beta_ratio_overflow", "--hydro.viscosityBetaOverAlpha=1e100", "single precision"),
        ("beta_ratio_nan", "--hydro.viscosityBetaOverAlpha=nan", "hydro.viscosityBetaOverAlpha"),
        ("alpha", "--hydro.viscosityAlpha=-1", "hydro.viscosityAlpha"),
        ("beta", "--hydro.viscosityBeta=-1", "hydro.viscosityBeta"),
        ("epsilon", "--hydro.viscosityEpsilon=0", "hydro.viscosityEpsilon"),
        ("gamma", "--hydro.gamma=1", "hydro.gamma"),
        ("central_softening", "--hydro.centralSoftening=0", "hydro.centralSoftening"),
        ("reflection", "--hydro.yReflect=true", "hydro.yReflect"),
        ("cfl_zero", "--hydro.cfl=0", "hydro.cfl"),
        ("cfl_large", "--hydro.cfl=1.1", "hydro.cfl"),
        ("neighbors", "--hydro.neighborNumber=10", "hydro.neighborNumber"),
        ("min_radius", "--hydro.minSupportRadius=0", "hydro.minSupportRadius"),
        ("max_radius", "--hydro.maxSupportRadius=1e-7", "support radii"),
        ("chemistry", "--hydro.chemistry=true", "hydro.chemistry"),
        ("conduction", "--hydro.conduction=true", "hydro.conduction"),
        ("diffusion", "--hydro.diffusion=true", "hydro.diffusion"),
    ]
    for name, option, diagnostic in invalid:
        output = run(name, metadata_options + [option], success=False)
        if diagnostic not in output:
            raise RuntimeError(f"{name} failed without its expected diagnostic: {diagnostic}")
    removed_config = args.workdir / "removed_option.ini"
    for option in sorted(removed_options):
        value = {"cosmology.omegaK": "0.1", "gravity.cloud": "pcs",
                 "gravity.method": args.gravity_method, "gravity.pmmmNearMethod": args.near_method,
                 "gravity.fmmOrder": str(args.order), "gravity.pmmmOrder": str(args.order),
                 "run.timeIntegrator": args.time_integrator
                 }.get(option, "false")
        group, name = option.split(".")
        removed_config.write_text(f"[{group}]\n{name} = {value}\n")
        for source, arguments in (
                ("cli", [f"--{option}={value}"]),
                ("ini", [f"--configuration={removed_config.resolve()}"])):
            output = run(f"removed_{source}_{option}", arguments, success=False)
            if option not in output or not re.search(r"unrecogni[sz]ed option", output):
                raise RuntimeError(f"removed {source} option {option} was not rejected explicitly")
    if not args.without_sph and args.gravity_method == "fmm" and args.time_integrator == "leapfrog":
        for parameter in ("omegaB", "omegaC"):
            for value in ("0", "-0.1", "nan", "inf"):
                output = run(f"invalid_class_mass_{parameter}_{value}", [
                    "--run.test=sphGravity", "--compute.cuda=false", "--hydro.enabled=true",
                    "--initial.particlesPerDimension=8", f"--cosmology.{parameter}={value}"],
                    success=False)
                # Negative and nonfinite densities fail the common cosmology guard;
                # zero is allowed there but cannot supply an SPH particle class.
                diagnostic = "class masses" if value == "0" else "Invalid cosmology"
                if diagnostic not in output:
                    raise RuntimeError(f"invalid {parameter} did not produce {diagnostic!r}")
    output = run("support_half_box", metadata_options + [
                                      "--hydro.minSupportRadius=5"], success=False)
    if "half the periodic box" not in output:
        raise RuntimeError("oversized minimum support radius was not rejected explicitly")
    print("Grouped CLI, complete INIs, Silo units/frames, and parameter bounds passed")


if __name__ == "__main__":
    main()
