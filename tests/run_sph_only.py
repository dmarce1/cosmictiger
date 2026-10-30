# CosmicTiger - gravity-free SPH application regression; BSL-1.0.

"""Check SPH-only startup, imposed forces, evolution, and exact restart."""

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import struct

from application_selection import application_options

from run_checkpoint import checkManifest, checkpoint_position_bytes, fnv1a64, hydro_particle_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--position-type", choices=("fixed32", "fixed64"), required=True)
    parser.add_argument("--reference", type=Path,
                        help="gravity-enabled traditional/SPH executable for evolution comparison")
    args = parser.parse_args()
    base = args.workdir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")
    common = ["--hpx:threads=2", "--hpx:bind=none", "--compute.cuda=false",
              "--chemistry.enabled=false", "--analysis.groups=false",
              "--analysis.views=false", "--analysis.power=false", "--lightcone.enabled=false",
              "--output.directory=.", "--output.silo=false",
              "--run.verbose=true", "--checkpoint.intervalSeconds=0"]
    selected = application_options(args.executable.resolve(), hydro=True)
    common += [f"--{key}={value}" for key, value in selected.items()]

    def prepare(name):
        directory = base / name
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()
        return directory

    def execute(directory, options, success=True, executable=None, omitChemistryDefault=False):
        particleCount = ([] if any(option.startswith("--initial.particlesPerDimension=")
                                    for option in options)
                         else ["--initial.particlesPerDimension=8"])
        defaults = [option for option in common
                    if not (omitChemistryDefault and option.startswith("--chemistry.enabled="))]
        command = [str((executable or args.executable).resolve()), *defaults, *particleCount, *options]
        result = subprocess.run(command, cwd=directory, env=environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        (directory / "run.log").write_text(result.stdout)
        if (result.returncode == 0) != success:
            raise RuntimeError(f"unexpected exit {result.returncode}; see {directory / 'run.log'}\n"
                               + result.stdout[-6000:])
        return result.stdout

    def reject(name, options, diagnostic, omitChemistryDefault=False):
        output = execute(prepare(name), options, success=False, omitChemistryDefault=omitChemistryDefault)
        if diagnostic not in output:
            raise RuntimeError(f"{name}: missing diagnostic {diagnostic!r}")

    helpText = execute(prepare("help"), ["--help"])
    if not re.search(r"Gravity solvers:\s*none\s*$", helpText, re.MULTILINE):
        raise RuntimeError("help does not identify the gravity-free build")
    if "Potential expansion degrees:" in helpText:
        raise RuntimeError("SPH-only help advertises an unused expansion order")
    chemistryDisabled = re.search(r"Particle chemistry:\s*disabled\s*$", helpText, re.MULTILINE) is not None
    if chemistryDisabled:
        reject("chemistry_cli", ["--chemistry.enabled=true"], "COSMICTIGER_ENABLE_CHEMISTRY=ON",
               omitChemistryDefault=True)
        chemistryConfig = base / "chemistry.ini"
        chemistryConfig.write_text("[chemistry]\nenabled = true\n")
        reject("chemistry_ini", [f"--configuration={chemistryConfig}"], "COSMICTIGER_ENABLE_CHEMISTRY=ON",
               omitChemistryDefault=True)
    metadata = execute(prepare("metadata"), ["--run.test=checkpointState", "--hydro.enabled=false"])
    execute(prepare("saved_diagnostics"), ["--run.test=sphOnlyDiagnostics", "--hydro.enabled=true"])
    for key, value in (("gravity.method.far", "none"), ("gravity.expansionDegree", "0"),
                       ("run.timeIntegrator", selected["run.timeIntegrator"]),
                       ("compute.positionType", args.position_type), ("gravity.enabled", "F")):
        if not re.search(r"^" + re.escape(key) + r"\s*:\s*" + value + r"\s*$",
                         metadata, re.MULTILINE):
            raise RuntimeError(f"incorrect build metadata for {key}")
    reject("gravity_cli", ["--gravity.enabled=true"], "gravity-enabled build")
    config = base / "gravity.ini"
    config.write_text("[gravity]\nenabled = true\n")
    reject("gravity_ini", [f"--configuration={config}"], "gravity-enabled build")
    execute(prepare("gravity_cli_override"), [f"--configuration={config}",
            "--gravity.enabled=false", "--run.test=checkpointState", "--hydro.enabled=false"])
    for scenario in ("plummer", "glass"):
        reject(scenario, [f"--run.scenario={scenario}"], "gravity-enabled build")
    reject("hydro_disabled", ["--run.scenario=sod", "--hydro.enabled=false"], "requires hydro.enabled")
    reject("gravity_error", ["--gravity.errorInterval=1"], "requires gravity.enabled=true")
    reject("opening_schedule", ["--gravity.openingSchedule=50:0.7"], "requires an FMM build")
    opposite = "fixed64" if args.position_type == "fixed32" else "fixed32"
    reject("wrong_positions", [f"--compute.positionType={opposite}"], "this build uses")

    duration = 1.0e-5

    def evolve(name, scenario, steps, extra=(), executable=None):
        directory = prepare(name)
        options = [f"--run.scenario={scenario}", f"--run.endTime={duration}",
                   f"--run.maxIterations={steps}", "--analysis.saveForce=true", *extra]
        output = execute(directory, options, executable=executable)
        if not re.search(r"^gravity.enabled\s*:\s*F\s*$", output, re.MULTILINE):
            raise RuntimeError(f"{name}: scenario defaults enabled gravity")
        if not re.search(r"^hydro.enabled\s*:\s*T\s*$", output, re.MULTILINE):
            raise RuntimeError(f"{name}: scenario defaults disabled SPH")
        if "Do kick\n" in output:
            raise RuntimeError(f"{name}: invoked a gravity kick")
        rows = [list(map(float, line.split())) for line in (directory / "energy.dat").read_text().splitlines()
                if line.strip() and not line.startswith("#")]
        if not rows or any(not math.isfinite(value) for row in rows for value in row):
            raise RuntimeError(f"{name}: missing or nonfinite energy diagnostics")
        if any(row[7] != 0.0 for row in rows):
            raise RuntimeError(f"{name}: nonzero self-gravity potential energy")
        checkpoint = directory / f"checkpoint.{steps}"
        checkManifest(checkpoint, steps, scenario=scenario, endTime=duration, gravity=False)
        width = checkpoint_position_bytes((checkpoint / "checkpoint.0.dat").read_bytes())
        if width != (8 if args.position_type == "fixed64" else 4):
            raise RuntimeError("wrong checkpoint position width")
        if (directory / "soft.txt").exists() and executable is None:
            raise RuntimeError("SPH-only startup produced a gravity kernel diagnostic")
        if (directory / "ewald.txt").exists() and executable is None:
            raise RuntimeError("SPH-only startup initialized Ewald gravity")
        print(f"{name}: hydro evolution and zero self-gravity diagnostics passed")
        return directory, options

    for scenario in ("sod", "blast", "rayleighTaylor", "disc", "polytropicStar"):
        extra = ["--initial.particlesPerDimension=16"] if scenario == "disc" else []
        if scenario == "rayleighTaylor":
            extra.append("--hydro.zPlaneAcceleration=0.5")
        if scenario == "disc":
            extra.append("--hydro.centralAcceleration=1")
        evolve(scenario, scenario, 1, extra)
    first, options = evolve("wave_first", "hydroWave", 1)
    if chemistryDisabled:
        incompatibleChemistry = prepare("chemistry_restart_rejected")
        copiedChemistry = incompatibleChemistry / "checkpoint.1"
        shutil.copytree(first / "checkpoint.1", copiedChemistry)
        # This synthetic fixture only advertises chemistry in the header. An
        # unavailable capability must be rejected before any chemistry payload.
        rankPath = copiedChemistry / "checkpoint.0.dat"
        payload = bytearray(rankPath.read_bytes())
        flags = struct.unpack_from("=I", payload, 12)[0]
        struct.pack_into("=I", payload, 12, flags | (1 << 11))
        rankPath.write_bytes(payload)
        manifest = copiedChemistry / "checkpoint.meta"
        records = manifest.read_text().splitlines()
        records[1] = f"0 {len(payload)} {fnv1a64(payload):016x}"
        manifest.write_text("\n".join(records) + "\n")
        output = execute(incompatibleChemistry, [*options, "--checkpoint.readIndex=1"], success=False)
        if "Chemistry checkpoints require" not in output or "Done reading checkpoint" in output:
            raise RuntimeError("chemistry checkpoint was not rejected before loading particle state")
        print("chemistry: disabled CLI, INI, and synthetic checkpoint header rejected")
    full, _ = evolve("wave_full", "hydroWave", 2)
    resumed = prepare("wave_restart")
    shutil.copytree(first / "checkpoint.1", resumed / "checkpoint.1")
    execute(resumed, [*options, "--checkpoint.readIndex=1"])
    checkManifest(resumed / "checkpoint.2", 2, scenario="hydroWave", endTime=duration, gravity=False)
    if hydro_particle_payload(full / "checkpoint.2") != hydro_particle_payload(resumed / "checkpoint.2"):
        raise RuntimeError("restarted SPH state differs from uninterrupted evolution")
    print("hydroWave: exact restart passed")
    incompatible = prepare("gravity_restart_rejected")
    copied = incompatible / "checkpoint.1"
    shutil.copytree(first / "checkpoint.1", copied)
    scenarioMetadata = copied / "scenario.meta"
    fields = scenarioMetadata.read_text().splitlines()
    fields[-1] = "1"
    scenarioMetadata.write_text("\n".join(fields) + "\n")
    manifest = copied / "checkpoint.meta"
    records = manifest.read_text().splitlines()
    payload = scenarioMetadata.read_bytes()
    records[-1] = f"SCENARIO {len(payload)} {fnv1a64(payload):016x}"
    manifest.write_text("\n".join(records) + "\n")
    output = execute(incompatible, [*options, "--checkpoint.readIndex=1"], success=False)
    if "gravity.enabled differs" not in output:
        raise RuntimeError("gravity-enabled checkpoint was not rejected at compatibility check")

    cosmology = prepare("cosmology")
    output = execute(cosmology, ["--run.scenario=cosmology", "--run.maxIterations=1",
                                "--initial.particlesPerDimension=4", "--cosmology.sigma8=0",
                                "--cosmology.startRedshift=49", "--cosmology.endRedshift=48.9"])
    if not re.search(r"^gravity.enabled\s*:\s*F\s*$", output, re.MULTILINE):
        raise RuntimeError("cosmology defaults enabled self-gravity")
    if not (cosmology / "energy.txt").is_file() or not list(cosmology.glob("checkpoint.*/checkpoint.meta")):
        raise RuntimeError("cosmological SPH failed to produce diagnostics and a checkpoint")
    print("cosmology: gravity-free SPH evolution passed")
    if args.reference:
        reference, _ = evolve("wave_gravity_build", "hydroWave", 2,
                              ["--gravity.enabled=false"], args.reference)
        if hydro_particle_payload(full / "checkpoint.2") != hydro_particle_payload(reference / "checkpoint.2"):
            raise RuntimeError("SPH-only evolution differs from the gravity-disabled reference")
        print("hydroWave: gravity-disabled reference matches exactly")


if __name__ == "__main__":
    main()
