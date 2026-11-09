# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

"""Exercise published checkpoint integrity and restart compatibility."""

import argparse
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import struct
import sys

from application_selection import cpu_backend_options
from checkpoint_format import Header, decode_checkpoint, read_header, require


def checkpoint_position_bytes(payload):
    """Read the position width from a current versioned checkpoint header."""
    return 8 if read_header(payload).flags & (1 << 15) else 4


def fnv1a64(payload):
    checksum = 0xCBF29CE484222325
    for byte in payload:
        checksum = ((checksum ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return checksum


def runSimulation(command, workdir, logName, expectSuccess=True, timeout=240):
    logPath = workdir / logName
    with logPath.open("w") as log:
        with subprocess.Popen(command, cwd=workdir, text=True, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                returnCode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"simulation timed out; see {logPath}")
    output = logPath.read_text(errors="replace")
    if (returnCode == 0) != expectSuccess:
        raise RuntimeError(f"{logName} exited with {returnCode}; see {logPath}:\n"
                           + output[-8000:])
    if not expectSuccess and "Done reading checkpoint" in output:
        raise RuntimeError(f"{logName} loaded a checkpoint that should have been rejected")
    return output


def prepareCase(baseWorkdir, name, sourceCheckpoint=None):
    workdir = baseWorkdir / name
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    if sourceCheckpoint is not None:
        shutil.copytree(sourceCheckpoint, workdir / sourceCheckpoint.name)
    return workdir


def checkFreshOutput(options, baseWorkdir, baseline):
    """Reject occupied checkpoint output before changing any prior run files."""
    def snapshot(directory):
        result = {}
        for path in directory.rglob("*"):
            name = path.relative_to(directory).as_posix()
            if path.is_symlink():
                result[name] = ("symlink", os.readlink(path))
            elif path.is_file():
                result[name] = ("file", path.read_bytes(), path.stat().st_mtime_ns)
            else:
                result[name] = ("directory",)
        return result

    def reject(workdir, outputDirectory, command):
        before = snapshot(outputDirectory)
        output = runSimulation(command + ["--run.maxIterations=0",
                                           "--checkpoint.intervalSeconds=1000000000"],
                               workdir, "fresh_start.log", expectSuccess=False)
        if ("conflicting checkpoint path" not in output.lower() or
                "--output.directory" not in output or "--checkpoint.readIndex" not in output):
            raise RuntimeError("fresh run lacked actionable checkpoint-output guidance")
        if snapshot(outputDirectory) != before:
            raise RuntimeError("rejected fresh run changed existing output or wrote simulation files")

    # Reuse the actual completed output, including progress and diagnostics.
    # Keep invocation logs/profiles in a separate working directory.
    workdir = prepareCase(baseWorkdir, "fresh_existing_output")
    command = [option for option in options if not option.startswith("--output.directory=")]
    reject(workdir, baseline, command + [f"--output.directory={baseline}"])
    for name, entry in (("incomplete_output", "checkpoint.999.incomplete"),
                        ("numbered_file_output", "checkpoint.999")):
        workdir = prepareCase(baseWorkdir, name)
        outputDirectory = workdir / "preserved output"
        outputDirectory.mkdir()
        checkpoint = outputDirectory / entry
        if entry.endswith(".incomplete"):
            checkpoint.mkdir()
            (checkpoint / "preserve.txt").write_text("unfinished checkpoint\n")
        else:
            checkpoint.write_text("occupied checkpoint path\n")
        (outputDirectory / "progress.txt").write_text("prior run progress\n")
        command = [option for option in options if not option.startswith("--output.directory=")]
        command += ["--output.directory=preserved output"]
        reject(workdir, outputDirectory, command)
    print("fresh output: existing checkpoints and staging paths rejected before output changes")


def checkManifest(checkpointDir, iteration, scenario="cosmology", endTime=0.0, gravity=True,
                  testFixture=False, ranks=1):
    manifest = checkpointDir / "checkpoint.meta"
    if not manifest.is_file():
        raise RuntimeError("completed checkpoint has no manifest")
    lines = manifest.read_text().splitlines()
    if len(lines) != ranks + (1 if testFixture else 2) or lines[0].split() != [
            "COSMICTIGER_CHECKPOINT", "1" if testFixture else "2", "3", str(iteration), str(ranks)]:
        raise RuntimeError(f"unexpected checkpoint manifest: {lines}")
    for expected_rank in range(ranks):
        rank, byteCount, checksumText = lines[expected_rank + 1].split()
        if rank != str(expected_rank) or not re.fullmatch(r"[0-9a-fA-F]{1,16}", checksumText):
            raise RuntimeError(f"invalid rank record in checkpoint manifest: {lines[expected_rank + 1]}")
        payload = (checkpointDir / f"checkpoint.{expected_rank}.dat").read_bytes()
        if int(byteCount) != len(payload) or int(checksumText, 16) != fnv1a64(payload):
            raise RuntimeError("checkpoint manifest size or checksum differs from rank file")
        checkpoint_position_bytes(payload)
    if not testFixture:
        scenarioLabel, scenarioBytes, scenarioChecksum = lines[ranks + 1].split()
        if scenarioLabel != "SCENARIO" or not re.fullmatch(r"[0-9a-fA-F]{1,16}", scenarioChecksum):
            raise RuntimeError(f"invalid scenario record in checkpoint manifest: {lines[ranks + 1]}")
        scenarioPayload = (checkpointDir / "scenario.meta").read_bytes()
        if int(scenarioBytes) != len(scenarioPayload) or int(scenarioChecksum, 16) != fnv1a64(scenarioPayload):
            raise RuntimeError("checkpoint manifest size or checksum differs from scenario metadata")
        scenarioLines = scenarioPayload.decode().splitlines()
        if (len(scenarioLines) != 4 or scenarioLines[0] != "COSMICTIGER_SCENARIO 1" or
                scenarioLines[1] != scenario or float(scenarioLines[2]) != endTime or
                scenarioLines[3] != str(int(gravity))):
            raise RuntimeError(f"unexpected checkpoint scenario metadata: {scenarioLines}")
    checkpoint_position_bytes(payload)


def hydro_particle_payload(checkpoint, expected_count=None):
    """Read the all-gas physical payload, excluding native driver timing/padding."""
    state = decode_checkpoint((checkpoint / "checkpoint.0.dat").read_bytes())
    if not state.hydro or state.gas_count != state.count or state.count <= 0:
        raise RuntimeError("restart fixture is not all gas")
    if expected_count is not None and state.count != expected_count:
        raise RuntimeError(f"unexpected gas count in {checkpoint}: {state.count}")
    return state.physical


def check_velocity_roundoff_restart(reference, resumed, ranks=1):
    """Require exact discrete state and float state within roundoff of the continuous run.

    Every rank file is compared; the decomposition must be the same, so each rank holds the
    same particles in the same order in both runs."""
    pairs = [[decode_checkpoint((path / f"checkpoint.{rank}.dat").read_bytes(), ranks=ranks)
              for path in (reference, resumed)] for rank in range(ranks)]
    scale = max(abs(value) for pair in pairs for state in pair for value in state.velocities)
    errors = [0.0, 0.0, 0.0]
    for first, second in pairs:
        rankErrors = check_rank_velocity_roundoff_restart(first, second, scale)
        errors = [max(a, b) for a, b in zip(errors, rankErrors)]
    return errors


def check_rank_velocity_roundoff_restart(first, second, scale):
    require(first.hydro and second.hydro and first.count == first.gas_count ==
            second.count == second.gas_count, "restart changed its all-gas population")
    require((first.integer_bytes, first.position_bytes) ==
            (second.integer_bytes, second.position_bytes), "restart changed particle layout")
    for field in ("tau", "integer_time", "global_timestep", "global_hydro_timestep", "iteration"):
        require(getattr(first.driver, field) == getattr(second.driver, field),
                f"restart changed driver {field}")
    # Particle ordering, rungs, flags, index maps and entropy context stay exact.
    positions = 2 * first.integer_bytes
    require(first.physical[:positions] == second.physical[:positions],
            "restart changed the particle counts")
    for field in ("types", "rungs", "gas_indices", "gravity_indices", "gas_flags",
                  "proposed_rungs", "cooling", "stars", "frozen_cells", "entropy_initialized",
                  "thermal_scale_factor"):
        require(getattr(first, field, None) == getattr(second, field, None),
                f"restart changed {field}")
    # Densities, supports and entropies are sums over neighbors of quantities that moved
    # by the roundoff allowed below; they agree to float roundoff, not bit for bit. The
    # binary16 bulk viscosity coefficient may round to the neighboring code.
    for field, tolerance in (("entropies", 1.0e-5), ("smoothing_lengths", 1.0e-5),
                             ("densities", 1.0e-5), ("bulk_viscosity_coefficients", 2.0e-3)):
        old, new = list(getattr(first, field)), list(getattr(second, field))
        require(len(old) == len(new), f"restart changed the length of {field}")
        if old:
            fieldScale = max(abs(value) for value in old + new)
            error = max(abs(a - b) for a, b in zip(old, new))
            require(error <= tolerance * fieldScale,
                    f"restart changed {field} by {error / fieldScale:.3e} of its scale")
    # A drift after a kick whose velocity differs by float roundoff can round a
    # fixed-point coordinate to the neighboring unit. Allow one unit, modulo the
    # periodic wrap, and nothing more.
    period = 1 << (8 * first.position_bytes)
    for axis in range(3):
        for old, new in zip(first.positions[axis], second.positions[axis]):
            step = (new - old) % period
            require(min(step, period - step) <= 1,
                    "restart moved a position by more than one fixed-point unit")
    # Pair impulses are accumulated in float compensated sums (SphCompensatedSum) in an
    # order set by worker scheduling. A component that cancels exactly, such as the
    # transverse axes of the wave, retains only roundoff of single pair impulses, so its
    # own maximum is no scale for that roundoff. Bound every axis by the largest velocity
    # component of the run, over every rank, instead, as tests/run_sph_dimensions.py does.
    errors = []
    for axis in range(3):
        old, new = first.velocities[axis::3], second.velocities[axis::3]
        error = max(abs(a - b) for a, b in zip(old, new))
        require(error <= 4 * 2 ** -23 * scale,
                "restart velocity exceeded float accumulation tolerance of the velocity scale")
        errors.append(error)
    return errors



def checkViscosityRestarts(executable, baseWorkdir):
    """Check both viscosity models; the shear limiter follows the model."""
    configuration = Path(__file__).resolve().parents[1] / "configurations/hydro_wave.ini"
    common = [str(executable), f"--configuration={configuration}", "--hpx:threads=2",
              "--hpx:bind=none", "--run.darkMatterParticleCount=512", "--run.gasParticleCount=512",
              *[f"--{key}={value}" for key, value in cpu_backend_options().items()],
              "--run.endTime=0.00001", "--run.verbose=true", "--checkpoint.intervalSeconds=0",
              "--output.directory=.", "--output.silo=false", "--analysis.views=false",
              "--analysis.power=false", "--analysis.groups=false", "--lightcone.enabled=false"]
    # Equal alpha bounds give constant viscosity; the default bounds evolve alpha.
    for model in ("constant", "evolving"):
        name = f"viscosity_{model}"
        options = common + (["--hydro.viscosityCoefficientMin=1", "--hydro.viscosityCoefficientMax=1",
                             "--hydro.conductionCoefficient=0.3"] if model == "constant" else [])
        reference = prepareCase(baseWorkdir, name)
        runSimulation(options + ["--run.maxIterations=2"], reference, "run.log")
        checkpoint = reference / "checkpoint.1"
        if not checkpoint.is_dir():
            raise RuntimeError(f"{name} did not save a completed first step")
        for path in (checkpoint, reference / "checkpoint.2"):
            state = decode_checkpoint((path / "checkpoint.0.dat").read_bytes())
            if len(state.bulk_viscosity_coefficients) != state.gas_count:
                raise RuntimeError(f"{name} has an incorrect viscosity payload")
        resumed = prepareCase(baseWorkdir, name + "_restart", checkpoint)
        output = runSimulation(options + ["--run.maxIterations=1", "--checkpoint.readIndex=1"],
                               resumed, "restart.log")
        if "Done reading checkpoint" not in output:
            raise RuntimeError(f"{name} did not load the checkpoint")
        # Pair reduction order can leave tiny transverse velocity residuals.
        # Keep all nonvelocity bytes (including viscosity state) and clocks
        # exact, and bound only velocity accumulation at float precision.
        restart_errors = check_velocity_roundoff_restart(
            reference / "checkpoint.2", resumed / "checkpoint.2")
        checkManifest(resumed / "checkpoint.2", 2, scenario="hydroWave", endTime=1.e-5,
                      gravity=False)
        mutations = {"ceiling": "--hydro.viscosityCoefficientMax=" + (
            "1.5" if model == "constant" else "1"),
                     "beta": "--hydro.vonNeumannRichtmyerCoefficient=3",
                     "balsara": "--hydro.balsaraSoundSpeedCoefficient=0.01"}
        if model == "constant":
            mutations.update({"conduction_off": "--hydro.conductionCoefficient=0",
                              "conduction_changed": "--hydro.conductionCoefficient=0.6"})
        if model == "evolving":
            mutations.update({"floor": "--hydro.viscosityCoefficientMin=0.2",
                              "decay": "--hydro.viscosityDecay=0.3"})
        for label, changed in mutations.items():
            directory = prepareCase(baseWorkdir, name + "_reject_" + label, checkpoint)
            changed_key = changed.split("=", 1)[0] + "="
            changed_options = [option for option in options
                               if not option.startswith(changed_key)]
            output = runSimulation(changed_options + ["--run.maxIterations=0",
                                   "--checkpoint.readIndex=1", changed],
                                   directory, "restart.log", expectSuccess=False)
            diagnostics = (
                "Checkpoint physics, particle layout or integrator differs",
                "SPH checkpoint parameters differ",
                "SPH checkpoint variable-viscosity parameters differ",
                "SPH checkpoint hydro.conductionCoefficient differs",
            )
            if not any(diagnostic in output for diagnostic in diagnostics):
                raise RuntimeError(f"{name}/{label} lacked a viscosity compatibility diagnostic")
        print(f"{name}: velocity-roundoff restart, exact committed state, and "
              f"viscosity compatibility passed; maximum errors={restart_errors}")


def checkVariationalPressureRestart(executable, baseWorkdir):
    """Restart variational SPH and reject retired pressure and viscosity layouts.

    Rebuilt grad-H corrections must preserve every nonvelocity payload byte.
    Tiny velocity residuals may reflect conservative pair reduction ordering.
    """
    configuration = Path(__file__).resolve().parents[1] / "configurations/hydro_wave.ini"
    options = [str(executable), f"--configuration={configuration}", "--hpx:threads=2",
               "--hpx:bind=none", "--run.darkMatterParticleCount=512", "--run.gasParticleCount=512",
               *[f"--{key}={value}" for key, value in cpu_backend_options().items()],
               "--run.endTime=0.00001", "--checkpoint.intervalSeconds=0",
               "--output.directory=.", "--output.silo=false", "--analysis.views=false",
               "--analysis.power=false", "--analysis.groups=false", "--lightcone.enabled=false"]
    name = "variational_pressure"
    reference = prepareCase(baseWorkdir, name)
    runSimulation(options + ["--run.maxIterations=2"], reference, "run.log")
    checkpoint = reference / "checkpoint.1"
    if not checkpoint.is_dir():
        raise RuntimeError(f"{name} did not save a completed first step")
    resumed = prepareCase(baseWorkdir, name + "_restart", checkpoint)
    output = runSimulation(options + ["--run.maxIterations=1", "--checkpoint.readIndex=1"],
                           resumed, "restart.log")
    if "Done reading checkpoint" not in output:
        raise RuntimeError(f"{name} did not load the checkpoint")
    restart_errors = check_velocity_roundoff_restart(
        reference / "checkpoint.2", resumed / "checkpoint.2")

    def rejectRetired(label, alter, diagnostic):
        directory = prepareCase(baseWorkdir, f"{name}_reject_{label}", checkpoint)
        checkpoint_dir = directory / checkpoint.name
        rank_file = checkpoint_dir / "checkpoint.0.dat"
        payload = bytearray(rank_file.read_bytes())
        # Validate the current native header before changing it.
        checkpoint_position_bytes(payload)
        physics_offset = Header.reserved.offset
        physics_code = struct.unpack_from("@I", payload, physics_offset)[0]
        struct.pack_into("@I", payload, physics_offset, alter(physics_code))
        rank_file.write_bytes(payload)
        # Keep integrity valid so rejection specifically tests the retired layout.
        manifest = checkpoint_dir / "checkpoint.meta"
        lines = manifest.read_text().splitlines()
        lines[1] = f"0 {len(payload)} {fnv1a64(payload):016x}"
        manifest.write_text("\n".join(lines) + "\n")
        output = runSimulation(options + ["--run.maxIterations=0", "--checkpoint.readIndex=1"],
                               directory, "restart.log", expectSuccess=False)
        if diagnostic not in output:
            raise RuntimeError(f"retired {label} checkpoint lacked its diagnostic")

    def retirePressure(physics_code):
        if physics_code & 0xff != 2:
            raise RuntimeError("SPH checkpoint did not record the variational formulation")
        return (physics_code & ~0xff) | 1

    rejectRetired("pressure", retirePressure,
                  "Checkpoint SPH pressure operator (monaghan) is no longer supported")
    rejectRetired("viscosity", lambda physics_code: physics_code & ~(1 << 25),
                  "Checkpoint uses the retired SPH viscosity layout")
    print(f"{name}: velocity-roundoff restart and retired-layout rejection passed; "
          f"maximum errors={restart_errors}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    sph_mode = parser.add_mutually_exclusive_group()
    sph_mode.add_argument("--viscosity-only", action="store_true")
    sph_mode.add_argument("--without-sph", action="store_true")
    parser.add_argument("--fresh-output-only", action="store_true")
    args = parser.parse_args()
    if args.viscosity_only and args.fresh_output_only:
        parser.error("--fresh-output-only cannot be combined with --viscosity-only")
    executable = args.executable.resolve()
    baseWorkdir = args.workdir.resolve()
    baseWorkdir.mkdir(parents=True, exist_ok=True)

    if args.viscosity_only:
        checkViscosityRestarts(executable, baseWorkdir)
        checkVariationalPressureRestart(executable, baseWorkdir)
        return 0

    options = [str(executable), "--hpx:threads=2", "--hpx:bind=none",
               "--run.timeIntegrator=traditional", "--gravity.meshSize=8",
               "--run.darkMatterParticleCount=512", "--run.gasParticleCount=512", "--hydro.enabled=false",
               "--gravity.cuda=false", "--analysis.views=false",
               "--lightcone.enabled=false", "--analysis.power=false",
               "--analysis.groups=false", "--gravity.kernelDegree=8", "--output.directory=.",
               # Checkpoint publication and integrity need no linear-power calculation.
               "--cosmology.sigma8=0"]
    baseline = prepareCase(baseWorkdir, "baseline")
    # The first step crosses this redshift. Its checkpoint must keep a numbered
    # restart path, while the second step must produce a final checkpoint.
    (baseline / "checkpoint.txt").write_text("48.9\n")
    (baseline / "checkpoint.z.42.0").write_text("unrelated redshift alias name\n")
    baselineOutput = runSimulation(options + ["--run.maxIterations=2",
                                              "--checkpoint.intervalSeconds=1000000000"],
                                   baseline, "run.log")
    if "Checkpoint saved:" not in baselineOutput:
        raise RuntimeError("normal exit did not finish a checkpoint")
    checkpoints = [path for path in baseline.glob("checkpoint.*")
                   if path.is_dir() and re.fullmatch(r"checkpoint\.[0-9]+", path.name)]
    if not checkpoints:
        raise RuntimeError("normal exit did not publish a numbered checkpoint")
    triggeredCheckpoint = baseline / "checkpoint.1"
    if triggeredCheckpoint not in checkpoints:
        raise RuntimeError("redshift-triggered checkpoint has no numbered restart path")
    checkpoint = max(checkpoints, key=lambda path: int(path.name.split(".")[1]))
    iteration = int(checkpoint.name.split(".")[1])
    if iteration != 2:
        raise RuntimeError("two-step run did not publish final checkpoint.2")
    if list(baseline.glob("checkpoint.*.incomplete")):
        raise RuntimeError("completed checkpoint left a staging directory")
    checkManifest(triggeredCheckpoint, 1)
    checkManifest(checkpoint, iteration)
    checkFreshOutput(options, baseWorkdir, baseline)

    def restart(name, mutate=None, expectSuccess=True):
        workdir = prepareCase(baseWorkdir, name, checkpoint)
        copiedCheckpoint = workdir / checkpoint.name
        if mutate is not None:
            mutate(copiedCheckpoint)
        output = runSimulation(options + [f"--checkpoint.readIndex={iteration}",
                                          "--run.maxIterations=0",
                                          "--checkpoint.intervalSeconds=1000000000"],
                               workdir, "restart.log", expectSuccess=expectSuccess)
        if expectSuccess and "Done reading checkpoint" not in output:
            raise RuntimeError(f"{name} did not load the checkpoint")
        if not expectSuccess and not re.search(r"checkpoint|manifest", output, re.IGNORECASE):
            raise RuntimeError(f"{name} failed without a checkpoint diagnostic")
        return output

    restart("current")
    if args.fresh_output_only:
        print("fresh output: new runs succeed and explicit checkpoint restarts remain supported")
        return 0

    def leaveOnlyStagingDirectory(checkpointDir):
        checkpointDir.rename(checkpointDir.with_name(checkpointDir.name + ".incomplete"))

    restart("staging_only", leaveOnlyStagingDirectory, expectSuccess=False)

    def removeManifest(checkpointDir):
        (checkpointDir / "checkpoint.meta").unlink()

    restart("missing_manifest", removeManifest, expectSuccess=False)

    def removeScenarioMetadata(checkpointDir):
        (checkpointDir / "scenario.meta").unlink()

    restart("missing_scenario", removeScenarioMetadata, expectSuccess=False)

    def corruptScenarioMetadata(checkpointDir):
        # Whitespace is semantically acceptable, but must invalidate the hash.
        path = checkpointDir / "scenario.meta"
        path.write_bytes(path.read_bytes() + b"\n")

    restart("corrupt_scenario", corruptScenarioMetadata, expectSuccess=False)

    def changeRankCount(checkpointDir):
        manifest = checkpointDir / "checkpoint.meta"
        lines = manifest.read_text().splitlines()
        fields = lines[0].split()
        fields[-1] = "2"
        lines[0] = " ".join(fields)
        manifest.write_text("\n".join(lines) + "\n")

    restart("wrong_rank_count", changeRankCount, expectSuccess=False)

    def corruptPayload(checkpointDir):
        rankFile = checkpointDir / "checkpoint.0.dat"
        payload = bytearray(rankFile.read_bytes())
        payload[len(payload) // 2] ^= 0x01
        rankFile.write_bytes(payload)

    restart("corrupt_payload", corruptPayload, expectSuccess=False)

    def truncatePayload(checkpointDir):
        rankFile = checkpointDir / "checkpoint.0.dat"
        payload = rankFile.read_bytes()
        rankFile.write_bytes(payload[:-16])

    restart("truncated_payload", truncatePayload, expectSuccess=False)

    def unsupportedVersion(checkpointDir):
        rankFile = checkpointDir / "checkpoint.0.dat"
        payload = bytearray(rankFile.read_bytes())
        payload[8:12] = (4).to_bytes(4, byteorder=sys.byteorder)
        rankFile.write_bytes(payload)
        manifest = checkpointDir / "checkpoint.meta"
        lines = manifest.read_text().splitlines()
        lines[1] = f"0 {len(payload)} {fnv1a64(payload):016x}"
        manifest.write_text("\n".join(lines) + "\n")

    restart("unsupported_version", unsupportedVersion, expectSuccess=False)

    def retiredParticleMesh(checkpointDir):
        rankFile = checkpointDir / "checkpoint.0.dat"
        payload = bytearray(rankFile.read_bytes())
        flags = int.from_bytes(payload[12:16], byteorder=sys.byteorder)
        payload[12:16] = (flags | (1 << 16)).to_bytes(4, byteorder=sys.byteorder)
        rankFile.write_bytes(payload)
        # Keep the manifest valid so this exercises solver compatibility,
        # rather than stopping at the earlier corruption check.
        manifest = checkpointDir / "checkpoint.meta"
        lines = manifest.read_text().splitlines()
        lines[1] = f"0 {len(payload)} {fnv1a64(payload):016x}"
        manifest.write_text("\n".join(lines) + "\n")

    output = restart("retired_pm", retiredParticleMesh, expectSuccess=False)
    if "Scalar particle-mesh (pm, p3m) checkpoints are no longer supported" not in output:
        raise RuntimeError("retired PM checkpoint was not rejected before reading its payload")

    def downgradeToVersionOne(checkpointDir):
        removeManifest(checkpointDir)
        rankFile = checkpointDir / "checkpoint.0.dat"
        with rankFile.open("r+b") as stream:
            stream.seek(8)
            stream.write((1).to_bytes(4, byteorder=sys.byteorder))

    # Retain the current scenario sidecar while checking legacy payload decoding.
    # Unlabelled old checkpoints are explicitly rejected by missing_scenario.
    legacy_compatible = checkpoint_position_bytes((checkpoint / "checkpoint.0.dat").read_bytes()) == 4
    restart("version_one", downgradeToVersionOne, expectSuccess=legacy_compatible)
    print(f"checkpoint {iteration}: redshift and final publication, payload version 3, "
          "scenario manifest v2, restart, corruption rejection, and labelled version 1 decoding passed")
    if not args.without_sph:
        checkViscosityRestarts(executable, baseWorkdir)
        checkVariationalPressureRestart(executable, baseWorkdir)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
