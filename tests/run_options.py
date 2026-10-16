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
    required = {"help", "configuration", "initial.particlesPerDimension", "compute.cuda",
                "hydro.enabled", "hydro.viscosityAlpha", "hydro.viscosityBeta",
                "hydro.viscosityEpsilon", "hydro.neighborNumber", "gravity.enabled",
                "output.silo", "output.siloInterval", "output.directory", "output.units", "output.frame",
                "output.xMin", "output.xMax", "output.yMin", "output.yMax",
                "output.zMin", "output.zMax"}
    if not required <= option_names:
        raise RuntimeError(f"grouped help is missing {sorted(required - option_names)}")
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
                "hydro.minSupportRadius": 1.0e-6, "hydro.maxSupportRadius": 4.0}
    for name, value in expected.items():
        match = re.search(r"^" + re.escape(name) + r"\s*:\s*(\S+)", configured, re.MULTILINE)
        if match is None or abs(float(match.group(1)) - value) > 1.0e-6:
            raise RuntimeError(f"configured {name} did not equal {value}")
    if "Checkpoint state and mode validation passed" not in configured:
        raise RuntimeError("configuration run did not finish the checkpoint regression")

    metadata_options = ["--run.test=checkpointState", "--compute.cuda=false",
                        "--hydro.enabled=false", "--initial.particlesPerDimension=8"]

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
    expected_keys = option_names - {"help", "configuration"}
    for path in [root / "default.ini", *sorted((root / "configurations").glob("*.ini"))]:
        ini = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
        ini.optionxform = str
        with path.open() as stream:
            ini.read_file(stream)
        actual_keys = {f"{section}.{key}" for section in ini.sections() for key in ini[section]}
        if actual_keys != expected_keys:
            raise RuntimeError(f"{path.name} option coverage differs: {sorted(actual_keys ^ expected_keys)}")
        if ini["output"]["units"] != "code" or ini["output"]["frame"] != "comoving":
            raise RuntimeError(f"{path.name} must preserve native output defaults")
        output = run(f"complete_{path.stem}", metadata_options + [f"--configuration={path}"])
        if "Checkpoint state and mode validation passed" not in output:
            raise RuntimeError(f"{path.name} did not finish metadata validation")

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
        ("reflection", "--hydro.yReflect=true", "unsupported"),
        ("cfl_zero", "--hydro.cfl=0", "hydro.cfl"),
        ("cfl_large", "--hydro.cfl=1.1", "hydro.cfl"),
        ("neighbors", "--hydro.neighborNumber=10", "hydro.neighborNumber"),
        ("min_radius", "--hydro.minSupportRadius=0", "support radii"),
        ("max_radius", "--hydro.maxSupportRadius=1e-7", "support radii"),
        ("chemistry", "--hydro.chemistry=true", "detached"),
        ("conduction", "--hydro.conduction=true", "detached"),
        ("diffusion", "--hydro.diffusion=true", "detached"),
    ]
    for name, option, diagnostic in invalid:
        output = run(name, [option], success=False)
        if diagnostic not in output:
            raise RuntimeError(f"{name} failed without its expected diagnostic: {diagnostic}")
    output = run("support_half_box", ["--initial.particlesPerDimension=8",
                                      "--hydro.minSupportRadius=5"], success=False)
    if "half the periodic box" not in output:
        raise RuntimeError("oversized minimum support radius was not rejected explicitly")
    print("Grouped CLI, complete INIs, Silo units/frames, and parameter bounds passed")


if __name__ == "__main__":
    main()
