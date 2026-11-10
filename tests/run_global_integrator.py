"""Check raw global steps, synchronized ranks, endpoint clipping, and restart state."""

import argparse
import math
import re
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True

from application_selection import application_arguments, compiled_algorithms
from checkpoint_format import decode_checkpoint, require
from run_checkpoint import check_velocity_roundoff_restart, fnv1a64, prepareCase, runSimulation


def checkpoint_state(directory, iteration, rank=0, integer_bytes=None):
    checkpoint = directory / f"checkpoint.{iteration}"
    path = checkpoint / f"checkpoint.{rank}.dat"
    ranks = int((checkpoint / "checkpoint.meta").read_text().splitlines()[0].split()[4])
    decoded = decode_checkpoint(path.read_bytes(), ranks=ranks)
    header, driver = decoded.header, decoded.driver
    require(header.flags & (1 << 18) and header.flags & (1 << 19),
            f"{path}: missing global checkpoint flags")
    require(decoded.algorithms[0] == 1 and decoded.algorithms[4] == 2,
            f"{path}: invalid global algorithm selection")
    require(integer_bytes is None or integer_bytes == decoded.integer_bytes,
            f"{path}: requested particle-index width differs from the checkpoint")
    require(driver.global_timestep > 0 and driver.integer_time == 0 and driver.max_rung == 0,
            f"{path}: invalid global clock")
    require(driver.iteration == iteration and driver.step == iteration,
            f"{path}: driver iteration differs from checkpoint name")
    require(not any(decoded.rungs), "global particles retained block rungs")
    if decoded.hydro and decoded.algorithms[1] in (0, 1):
        require(driver.global_hydro_timestep > 0,
                f"{path}: midpoint global hydro bound was not persisted")
    return driver, driver.global_timestep, decoded.physical


def check_hydro_restart_roundoff(reference, resumed):
    """Keep clocks/nonvelocity bytes exact and use the shared velocity-scale float bound."""
    states = [decode_checkpoint((path / "checkpoint.0.dat").read_bytes())
              for path in (reference, resumed)]

    def ordered_float(value):
        # Map binary32 values to consecutive integers across both signs, treating
        # signed zero equally; a double-precision math.ulp is not the stored ULP.
        bits = struct.unpack("=i", struct.pack("=f", value))[0]
        return -0x80000000 - bits if bits < 0 else bits

    maximum_ulps = 0
    for index, (old, new) in enumerate(zip(states[0].velocities, states[1].velocities)):
        require(math.isfinite(old) and math.isfinite(new),
                f"restart velocity {index} is nonfinite")
        ulps = abs(ordered_float(old) - ordered_float(new))
        maximum_ulps = max(maximum_ulps, ulps)
    # Near cancellation, a tiny absolute reduction error can span many ULPs of
    # the residual. Bound against the velocity axis scale, not that residual;
    # retain local ULP distance as evidence rather than widening a ULP cutoff.
    errors = check_velocity_roundoff_restart(reference, resumed)
    return maximum_ulps, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--without-sph", action="store_true")
    parser.add_argument("--mpi-exec")
    parser.add_argument("--particle-integer-bytes", type=int, choices=(4, 8))
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--hpx-argument", action="append", default=[])
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    available = compiled_algorithms(str(executable))
    if not ('global' in available['integrator']):
        raise RuntimeError('global integrator is not compiled')
    common = {"hpx:threads": 2, "hpx:bind": "none", "gravity.cuda": "false",
              "run.darkMatterParticleCount": 512, "run.gasParticleCount": 512, "scenario.cosmology.useGlass": "false",
              "initial.closePack": "false", "chemistry.enabled": "false",
              "analysis.groups": "false", "analysis.views": "false",
              "analysis.tracers": "false", "analysis.power": "false",
              "lightcone.enabled": "false", "output.silo": "false",
              "output.directory": ".", "checkpoint.intervalSeconds": 0,
              # Global integration deliberately accepts a floor beyond the rung
              # table: its raw timestep must neither index nor obey that floor.
              "run.numberOfOutputs": 2, "run.minRung": 20}
    cases = []
    if not args.without_sph:
        cases.append(("hydro", False, True, None, None))
    # SPH has no self-gravity coupling in the PMMM solver: gravity cases are
    # dark matter only, hydro runs are gravity free.
    if "pmmm" in available["far"]:
        for near in available["near"]:
            cases.append(("pmmm_" + near, True, False, "pmmm", near))

    def run(directory, options, selections, restart=None, mpi=False, success=True):
        command = [str(executable), *args.hpx_argument] + selections
        command += [f"--{key}={value}" for key, value in options.items()]
        if restart is not None:
            command.append(f"--checkpoint.readIndex={restart}")
        if mpi:
            command = [args.mpi_exec, args.mpi_numproc_flag, "2", *args.mpi_preflag,
                       command[0], *args.mpi_postflag, *command[1:]]
            command += ["--hpx:localities=2", "--hpx:ini=hpx.parcel.mpi.enable=1"]
        return runSimulation(command, directory, "run.log", expectSuccess=success)

    def read_state(directory, iteration, rank=0):
        return checkpoint_state(directory, iteration, rank, args.particle_integer_bytes)

    def checkpoint_iterations(directory):
        return sorted(int(path.name.split(".")[1]) for path in directory.glob("checkpoint.*")
                      if path.is_dir() and re.fullmatch(r"checkpoint\.[0-9]+", path.name))

    for label, gravity, hydro, far, near in cases:
        selections = application_arguments(executable, near=near, integrator="global", hydro=hydro)
        options = dict(common, **{"run.scenario": "hydroWave" if hydro else "plummer",
                                  "gravity.enabled": str(gravity).lower(),
                                  "hydro.enabled": str(hydro).lower(), "hydro.cfl": 0.123,
                                  "gravity.meshSize": 8, "run.endTime": 0.5,
                                  "run.maxIterations": 2})
        baseline = prepareCase(root, label)
        run(baseline, options, selections)
        if not (checkpoint_iterations(baseline) == [1, 2]):
            raise RuntimeError(f'{label}: iteration limit was not exact')
        first, timestep, _ = read_state(baseline, 1)
        final, _, physical = read_state(baseline, 2)
        if not (0 < timestep < 0.5 and math.isclose(first.tau, timestep, rel_tol=1e-14)):
            raise RuntimeError('Check failed: 0 < timestep < 0.5 and math.isclose(first.tau, timestep, rel_tol=1e-14)')
        exponent = math.log2(timestep / 0.5)
        if not (abs(exponent - round(exponent)) > 1e-05):
            raise RuntimeError(f'{label}: dyadic timestep {timestep}')
        resumed = prepareCase(root, label + "_restart", baseline / "checkpoint.1")
        run(resumed, dict(options, **{"run.maxIterations": 1}), selections, restart=1)
        if not (checkpoint_iterations(resumed) == [1, 2]):
            raise RuntimeError(f'{label}: restart ran an extra iteration')
        continued, next_timestep, continued_physical = read_state(resumed, 2)
        require(continued.tau == final.tau, f"{label}: restart changed the exact clock")
        restart_kind = "exact"
        if continued_physical != physical:
            require(hydro, f"{label}: restart changed exact collisionless particle state")
            # Pair reductions need not retain their floating addition order when
            # a restart rebuilds the SPH tree. Keep all nonvelocity bytes exact;
            # use the existing four-float-epsilon bound per velocity axis, which
            # also handles tiny transverse residuals from cancellation.
            maximum_ulps, errors = check_hydro_restart_roundoff(
                baseline / "checkpoint.2", resumed / "checkpoint.2")
            restart_kind = (f"velocity-roundoff (max local ULP distance={maximum_ulps}; "
                            f"errors={errors})")
        if not (math.isclose(continued.tau - first.tau, next_timestep, rel_tol=1e-14)):
            raise RuntimeError('Check failed: math.isclose(continued.tau - first.tau, next_timestep, rel_tol=1e-14)')

        # A short run must land on the exact requested endpoint, including a
        # non-dyadic final interval, without leaving a tiny rounding remainder.
        endpoint = timestep * 1.37
        clipped = prepareCase(root, label + "_endpoint")
        run(clipped, dict(options, **{"run.endTime": endpoint,
                                     "run.maxIterations": 8}), selections)
        checkpoints = checkpoint_iterations(clipped)
        ended, last_dt, _ = read_state(clipped, checkpoints[-1])
        if not (ended.tau == endpoint and last_dt <= timestep * 1.01):
            raise RuntimeError(label)
        if not (len(checkpoints) <= 3):
            raise RuntimeError(f'{label}: extra tiny terminal intervals')

        if label == cases[0][0]:
            # Rehash each damaged file so validation reaches the clock reader,
            # independently of the existing checksum-corruption regression.
            clock_error = "Invalid global timestep checkpoint extension"
            hydro_error = "Invalid global hydro timestep checkpoint extension"
            mutations = [("zero_dt", first.global_extension_offset + 8, "d", 0.0, clock_error),
                         ("nan_dt", first.global_extension_offset + 8, "d", math.nan, clock_error),
                         ("integer_phase", first.integer_phase_offset, "Q", 1, clock_error)]
            if hydro and far in (None, "fmm"):
                mutations += [
                    ("zero_hydro_dt", first.global_extension_offset + 16, "d", 0.0, hydro_error),
                    ("nan_hydro_dt", first.global_extension_offset + 16, "d", math.nan, hydro_error)]
            for suffix, offset, encoding, value, expected_error in mutations:
                rejected = prepareCase(root, label + "_" + suffix, baseline / "checkpoint.1")
                directory = rejected / "checkpoint.1"
                path = directory / "checkpoint.0.dat"
                data = bytearray(path.read_bytes())
                struct.pack_into("=" + encoding, data, offset, value)
                path.write_bytes(data)
                manifest = directory / "checkpoint.meta"
                lines = manifest.read_text().splitlines()
                lines[1] = f"0 {len(data)} {fnv1a64(data):016x}"
                manifest.write_text("\n".join(lines) + "\n")
                output = run(rejected, dict(options, **{"run.maxIterations": 0}),
                             selections, restart=1, success=False)
                if expected_error not in output:
                    raise RuntimeError(suffix)
            print("global checkpoint clock: invalid gravity/hydro intervals and integer phase rejected")

        if args.mpi_exec and label == cases[0][0]:
            distributed = prepareCase(root, label + "_mpi")
            run(distributed, dict(options, **{"run.maxIterations": 1}), selections, mpi=True)
            states = [read_state(distributed, 1, rank) for rank in range(2)]
            if not (states[0][1] == states[1][1]):
                raise RuntimeError('ranks selected different timesteps')
            if not (math.isclose(states[0][1], timestep, rel_tol=2e-05)):
                raise RuntimeError('distributed minimum differs from serial minimum')
        print(f"{label}: raw dt={timestep:.12g}, {restart_kind} restart and exact endpoint passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
