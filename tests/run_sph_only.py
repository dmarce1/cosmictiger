# CosmicTiger - gravity-free SPH application regression; BSL-1.0.

"""Check SPH-only startup, imposed forces, evolution, and restart."""

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import struct

from application_selection import application_options
from checkpoint_format import decode_checkpoint, require

from run_checkpoint import (checkManifest, checkpoint_position_bytes, fnv1a64,
                            hydro_particle_payload, check_velocity_roundoff_restart)


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
    common = ["--hpx:threads=2", "--hpx:bind=none", "--gravity.cuda=false",
              "--chemistry.enabled=false", "--analysis.groups=false",
              "--analysis.views=false", "--lightcone.enabled=false",
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
        particleCount = ([] if any(option.startswith(("--run.darkMatterParticleCount=", "--run.gasParticleCount="))
                                    for option in options)
                         else ["--run.darkMatterParticleCount=512", "--run.gasParticleCount=512"])
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
    if re.search(r"^\s*Cosmology:\s*$", helpText, re.MULTILINE):
        raise RuntimeError("SPH-only help advertises unavailable cosmology options")
    scenarioHelp = re.search(r"^\s*--run\.scenario\b.*?(?=^\s*--|\Z)",
                             helpText, re.MULTILINE | re.DOTALL)
    if scenarioHelp is None or "cosmology" in scenarioHelp[0]:
        raise RuntimeError("SPH-only scenario help is missing or advertises cosmology")
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
    for key, value in (("gravity.expansionDegree", "0"),
                       ("run.timeIntegrator", selected["run.timeIntegrator"]),
                       ("run.scenario", "sod"),
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
    reject("cosmology_cli", ["--run.scenario=cosmology"], "does not include run.scenario=cosmology")
    cosmologyConfig = base / "cosmology.ini"
    cosmologyConfig.write_text("[run]\nscenario = cosmology\n")
    reject("cosmology_ini", [f"--configuration={cosmologyConfig}"],
           "does not include run.scenario=cosmology")
    # Shared hydro presets retain inert cosmology values for cross-build use.
    hydroConfig = base / "hydro_cosmology_defaults.ini"
    hydroConfig.write_text("[run]\nscenario = sod\n[cosmology]\nsigma8 = 0.81\n")
    execute(prepare("hydro_cosmology_defaults"), [f"--configuration={hydroConfig}",
            "--run.test=checkpointState", "--hydro.enabled=false"])
    reject("power", ["--analysis.power=true"], "analysis.power requires a gravity-enabled build")
    for fftTest in ("fft1", "fft2"):
        reject(fftTest, [f"--run.test={fftTest}"], f"test {fftTest} is not known")
    reject("hydro_disabled", ["--run.scenario=sod", "--hydro.enabled=false"], "requires hydro.enabled")
    reject("gravity_error", ["--gravity.errorInterval=1"], "requires gravity.enabled=true")
    reject("near_method", ["--gravity.method.near=pp"], "gravity.method.near")
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

    reject("polytrope_without_gravity", ["--run.scenario=polytropicStar"], "requires gravity.enabled=true")
    for scenario in ("sod", "blast", "rayleighTaylor", "disc"):
        extra = ["--run.darkMatterParticleCount=4096", "--run.gasParticleCount=4096"] if scenario == "disc" else []
        if scenario == "rayleighTaylor":
            extra.append("--scenario.rayleighTaylor.zPlaneAcceleration=0.5")
        if scenario == "disc":
            extra.append("--scenario.disc.centralAcceleration=1")
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
        # Pair reduction can change sub-float cancellation in transverse
        # momentum when restart rebuilds the tree. Measured initial differences
        # were 1.69e-21 and 6.78e-21, only 0.249 componentwise float eps; all other
        # state remained byte-identical. Do not relax those exact fields.
        errors = check_velocity_roundoff_restart(full / "checkpoint.2", resumed / "checkpoint.2")
        print(f"hydroWave: velocity-roundoff restart passed; maximum errors={errors}")
    else:
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

    if args.reference:
        reference, _ = evolve("wave_gravity_build", "hydroWave", 2,
                              ["--gravity.enabled=false"], args.reference)
        if hydro_particle_payload(full / "checkpoint.2") != hydro_particle_payload(reference / "checkpoint.2"):
            raise RuntimeError("SPH-only evolution differs from the gravity-disabled reference")
        print("hydroWave: gravity-disabled reference matches exactly")


if __name__ == "__main__":
    main()
