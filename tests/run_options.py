# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

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

    def run(name, options, success=True):
        result = subprocess.run(command + options, cwd=args.workdir, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
                                env=environment)
        (args.workdir / f"{name}.log").write_text(result.stdout)
        if (result.returncode == 0) != success:
            raise RuntimeError(f"{name} returned {result.returncode}:\n{result.stdout}")
        return result.stdout

    help_text = run("help", ["--help"])
    option_names = set(re.findall(r"^\s+--([A-Za-z][A-Za-z0-9_.]*)", help_text, re.MULTILINE))
    required = {"help", "configuration", "run.scenario", "run.endTime",
                "initial.particlesPerDimension", "compute.cuda",
                "hydro.enabled", "hydro.viscosityAlpha", "hydro.viscosityBeta",
                "hydro.viscosityEpsilon", "hydro.neighborNumber", "gravity.enabled",
                "gravity.kernelFamily", "gravity.kernelDegree",
                "hydro.kernelFamily", "hydro.kernelDegree",
                "gravity.effectiveRadius", "gravity.supportRadius",
                "hydro.minEffectiveRadius", "hydro.maxEffectiveRadius",
                "hydro.minSupportRadius", "hydro.maxSupportRadius",
                "hydro.centralEffectiveRadius", "hydro.centralSupportRadius",
                "output.silo", "output.siloInterval", "output.directory", "output.units", "output.frame",
                "output.xMin", "output.xMax", "output.yMin", "output.yMax",
                "output.zMin", "output.zMax"}
    if not required <= option_names:
        raise RuntimeError(f"grouped help is missing {sorted(required - option_names)}")
    removed_options = {"hydro.chemistry", "hydro.conduction", "hydro.diffusion",
                       "hydro.yReflect", "gravity.sphSmoothing", "cosmology.omegaK"}
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
[hydro]
enabled = false
viscosityAlpha = 0.75
viscosityBeta = 1.5
viscosityEpsilon = 0.02
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
                "hydro.gamma": 1.4, "hydro.neighborNumber": 48.0,
                "hydro.cfl": 0.3, "lightcone.mapSize": 8.0,
                "hydro.minSupportRadius": 1.0e-6, "hydro.maxSupportRadius": 4.0,
                "gravity.kernelDegree": 4.0}
    for name, value in expected.items():
        match = re.search(r"^" + re.escape(name) + r"\s*:\s*(\S+)", configured, re.MULTILINE)
        if match is None or abs(float(match.group(1)) - value) > 1.0e-6:
            raise RuntimeError(f"configured {name} did not equal {value}")
    if "Checkpoint state and mode validation passed" not in configured:
        raise RuntimeError("configuration run did not finish the checkpoint regression")

    metadata_options = ["--run.test=checkpointState", "--compute.cuda=false",
                        "--hydro.enabled=false", "--initial.particlesPerDimension=8"]

    # Selections survive option setup and checkpoints independently.
    for gravity_family, gravity_degree, hydro_family, hydro_degree in (
            ("spline", 9, "wendland", 8), ("wendland", 5, "ferrers", 6),
            ("ferrers", 2, "spline", 4)):
        output = run(f"kernels_{gravity_family}_{hydro_family}", metadata_options + [
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
            ["--hydro.kernelFamily=wendland", "--hydro.kernelDegree=8", "--hydro.neighborNumber=12"])):
        run(f"invalid_kernel_{index}", metadata_options + invalid, success=False)

    def assertNumericSetting(output, key, expected):
        match = re.search(r"^" + re.escape(key) + r"\s*:\s*(\S+)", output, re.MULTILINE)
        if match is None or not math.isclose(float(match.group(1)), expected, rel_tol=1e-5):
            raise RuntimeError(f"incorrect {key}: expected {expected}")

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
                       ("hydro.neighborNumber", 42), ("hydro.kernelDegree", 3)):
        assertNumericSetting(defaults, key, value)

    # Both radius representations describe the same kernel, including kernels
    # whose support-to-effective-radius ratios differ.
    for family, degree, ratio in (("ferrers", 8, math.sqrt(13) / 2),
                                   ("spline", 3, math.sqrt(10 / 3)),
                                   ("wendland", 5, math.sqrt(15) / 2)):
        common = metadata_options + [f"--gravity.kernelFamily={family}",
                                      f"--gravity.kernelDegree={degree}"]
        for representation, value in (("effectiveRadius", 0.02), ("supportRadius", 0.02 * ratio)):
            output = run(f"radius_{family}_{representation}", common + [
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
                     "hydro.minEffectiveRadius", "hydro.maxEffectiveRadius"}
    expected_keys = option_names - {"help", "configuration"} - optional_keys
    for path in [root / "default.ini", *sorted((root / "configurations").glob("*.ini"))]:
        ini = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
        ini.optionxform = str
        with path.open() as stream:
            ini.read_file(stream)
        actual_keys = {f"{section}.{key}" for section in ini.sections() for key in ini[section]}
        if actual_keys != expected_keys:
            raise RuntimeError(f"{path.name} option coverage differs: {sorted(actual_keys ^ expected_keys)}")
        # Alternatives stay commented so loading a complete example does not
        # accidentally select two representations and trigger a warning.
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
            raise RuntimeError(f"{path.name} is missing commented radius alternatives: "
                               f"{sorted(optional_keys - documented_alternatives)}")
        if ini["output"]["units"] != "code" or ini["output"]["frame"] != "comoving":
            raise RuntimeError(f"{path.name} must preserve native output defaults")
        output = run(f"complete_{path.stem}", metadata_options + [f"--configuration={path}",
                     "--chemistry.enabled=false"])
        if "Checkpoint state and mode validation passed" not in output:
            raise RuntimeError(f"{path.name} did not finish metadata validation")
        assertNoRadiusWarning(output)

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
        ("steps_zero", "--run.numberOfSteps=0", "run.numberOfSteps"),
        ("end_time_negative", "--run.endTime=-1", "run.endTime"),
        ("end_time_nan", "--run.endTime=nan", "run.endTime"),
        ("end_time_inf", "--run.endTime=inf", "run.endTime"),
        ("scenario_unknown", "--run.scenario=unknown", "scenario"),
        ("bucket_zero", "--compute.bucketSize=0", "compute.bucketSize"),
        ("eta_zero", "--gravity.timestepEta=0", "gravity.timestepEta"),
        ("eta_negative", "--gravity.timestepEta=-0.1", "gravity.timestepEta"),
        ("eta_nan", "--gravity.timestepEta=nan", "gravity.timestepEta"),
        ("eta_inf", "--gravity.timestepEta=inf", "gravity.timestepEta"),
        ("eta_underflow", "--gravity.timestepEta=1e-100", "gravity.timestepEta"),
        ("softening_zero", "--gravity.softening=0", "gravity.softening"),
        ("softening_negative", "--gravity.softening=-0.1", "gravity.softening"),
        ("softening_nan", "--gravity.softening=nan", "gravity.softening"),
        ("softening_inf", "--gravity.softening=inf", "gravity.softening"),
        ("softening_underflow", "--gravity.softening=1e-37", "gravity.softening"),
        ("smoothing_obsolete", "--gravity.sphSmoothing=true", "gravity.sphSmoothing"),
        ("adaptive_option_removed", "--gravity.softeningMode=adaptive", "gravity.softeningMode"),
        ("alpha_overflow", "--hydro.viscosityAlpha=1e100", "single precision"),
        ("epsilon_underflow", "--hydro.viscosityEpsilon=1e-100", "single precision"),
        ("gamma_rounding", "--hydro.gamma=1.00000001", "single precision"),
        ("neighbors_rounding", "--hydro.neighborNumber=10.66666667", "single precision"),
        ("cfl_underflow", "--hydro.cfl=1e-100", "single precision"),
        ("support_underflow", "--hydro.minSupportRadius=1e-100", "single precision"),
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
        output = run(name, [option], success=False)
        if diagnostic not in output:
            raise RuntimeError(f"{name} failed without its expected diagnostic: {diagnostic}")
    removed_config = args.workdir / "removed_option.ini"
    for option in sorted(removed_options):
        value = "0.1" if option == "cosmology.omegaK" else "false"
        group, name = option.split(".")
        removed_config.write_text(f"[{group}]\n{name} = {value}\n")
        for source, arguments in (
                ("cli", [f"--{option}={value}"]),
                ("ini", [f"--configuration={removed_config.resolve()}"])):
            output = run(f"removed_{source}_{option}", arguments, success=False)
            if option not in output or not re.search(r"unrecogni[sz]ed option", output):
                raise RuntimeError(f"removed {source} option {option} was not rejected explicitly")
    for parameter in ("omegaB", "omegaC"):
        for value in ("0", "-0.1", "nan", "inf"):
            output = run(f"invalid_class_mass_{parameter}_{value}", [
                "--run.test=sphGravity", "--compute.cuda=false", "--hydro.enabled=true",
                "--initial.particlesPerDimension=8", f"--cosmology.{parameter}={value}"],
                success=False)
            if "class masses" not in output:
                raise RuntimeError(f"invalid {parameter} did not produce a class-mass diagnostic")
    output = run("support_half_box", ["--initial.particlesPerDimension=8",
                                      "--hydro.minSupportRadius=5"], success=False)
    if "half the periodic box" not in output:
        raise RuntimeError("oversized minimum support radius was not rejected explicitly")
    print("Grouped CLI, complete INIs, Silo units/frames, and parameter bounds passed")


if __name__ == "__main__":
    main()
