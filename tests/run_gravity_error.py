# CosmicTiger - sampled gravity accuracy evolution regression; BSL-1.0.

"""Check direct-force sampling cadence, restart, and unchanged particle evolution."""

import argparse
import csv
import math
import struct
from pathlib import Path

from application_selection import application_options
import sys

sys.dont_write_bytecode = True

from checkpoint_format import decode_checkpoint
from run_checkpoint import fnv1a64, prepareCase, runSimulation


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def readState(checkpoint, ranks):
    """Read dark-matter physics, excluding driver wall clocks and native padding."""
    manifest = (checkpoint / "checkpoint.meta").read_text().splitlines()
    iteration = int(checkpoint.name.split(".")[1])
    require(manifest[0].split() == ["COSMICTIGER_CHECKPOINT", "2", "3",
                                   str(iteration), str(ranks)], "invalid checkpoint manifest")
    particles = []
    state = None
    for rank in range(ranks):
        data = (checkpoint / f"checkpoint.{rank}.dat").read_bytes()
        require(manifest[rank + 1].split() == [str(rank), str(len(data)),
                                              f"{fnv1a64(data):016x}"],
                "checkpoint checksum differs from manifest")
        decoded = decode_checkpoint(data, ranks=ranks)
        header, driver = decoded.header, decoded.driver
        require(not header.flags & ((1 << 0) | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 11)),
                "fixture enabled optional particle fields")
        require(driver.iteration == iteration, "checkpoint driver layout mismatch")
        if state is None:
            state = driver
        else:
            require((state.tau, state.integer_time, state.step) ==
                    (driver.tau, driver.integer_time, driver.step), "locality clocks disagree")
        scale = 2 ** (8 * decoded.position_bytes)
        particles.extend((tuple(decoded.positions[dim][i] / scale for dim in range(3)),
                          decoded.velocities[3 * i:3 * i + 3], decoded.rungs[i])
                         for i in range(decoded.count))
    return state, sorted(particles)


def compareState(expected, actual):
    firstState, first = expected
    secondState, second = actual
    require((firstState.tau, firstState.integer_time, firstState.iteration) ==
            (secondState.tau, secondState.integer_time, secondState.iteration),
            "diagnostics changed the evolution clock")
    require(len(first) == len(second), "diagnostics changed the particle count")
    for left, right in zip(first, second):
        require(left[2] == right[2], "diagnostics changed a particle rung")
        for a, b in zip((*left[0], *left[1]), (*right[0], *right[1])):
            require(math.isclose(a, b, rel_tol=3.0e-6, abs_tol=3.0e-8),
                    f"diagnostics changed particle state: {a} != {b}")


def readErrors(directory, steps, sampleCount, duration):
    with (directory / "gravity_error.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    require([int(row["coarse_step"]) for row in rows] == steps,
            f"incorrect coarse-step diagnostic cadence: {rows}")
    for row, step in zip(rows, steps):
        require(int(row["sample_count"]) == sampleCount, "incorrect global sample count")
        require(math.isclose(float(row["time"]), step * duration / 5, abs_tol=1.0e-13),
                "diagnostic time differs from the pre-kick coarse boundary")
        require(float(row["scale_factor"]) == 1.0, "incorrect plummer scale factor")
        for key in ("rms_absolute", "rms_relative", "max_absolute", "max_relative"):
            value = float(row[key])
            require(math.isfinite(value) and value >= 0.0, f"invalid {key}: {value}")
        reference = float(row["rms_reference"])
        require(math.isfinite(reference) and reference > 0.0, "invalid RMS direct acceleration")
        require(int(row["zero_reference_count"]) == 0, "fixture has a zero direct acceleration")
        require(math.isclose(float(row["rms_relative"]),
                             float(row["rms_absolute"]) / reference, rel_tol=1.0e-13,
                             abs_tol=1.0e-15), "relative RMS normalization is inconsistent")
        require(float(row["rms_relative"]) < 0.02,
                "sampled gravity exceeds the fixture's 2% relative RMS accuracy bound")
        require(float(row["max_relative"]) < 0.1,
                "sampled gravity exceeds the fixture's 10% maximum relative accuracy bound")
        require(float(row["rms_absolute"]) <= float(row["max_absolute"]) * (1.0 + 1.0e-12),
                "RMS absolute error exceeds the maximum")
    return fields, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--gravity-method", choices=("fmm", "pmmm"), required=True)
    parser.add_argument("--time-integrator", choices=("traditional", "hierarchical", "global"))
    parser.add_argument("--position-type", choices=("fixed32", "fixed64"), default="fixed32")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--mpi-exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--disable-tcp", action="store_true")
    parser.add_argument("--ranks", type=int, default=2)
    args = parser.parse_args()
    executable = args.executable.resolve()
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    ranks = args.ranks if args.mpi_exec else 1
    duration = 1.0e-5
    common = {"hpx:threads": 2, "hpx:bind": "none", "run.scenario": "plummer",
              "gravity.cuda": str(args.cuda).lower(),
              "gravity.error.cuda": str(args.cuda).lower(),
              "compute.positionType": args.position_type,
              "hydro.enabled": "false", "chemistry.enabled": "false", "gravity.enabled": "true",
              "run.darkMatterParticleCount": 64, "run.gasParticleCount": 64, "initial.closePack": "false",
              "scenario.cosmology.useGlass": "false", "run.endTime": duration, "run.numberOfSteps": 5,
              "run.minRung": 1, "run.maxIterations": 100, "analysis.groups": "false",
              "analysis.tracers": "false", "analysis.views": "false", "analysis.power": "false",
              "lightcone.enabled": "false", "output.silo": "false", "output.directory": ".",
              "checkpoint.intervalSeconds": 0}
    common.update(application_options(executable, far=args.gravity_method,
                                      integrator=args.time_integrator))
    if args.gravity_method == "pmmm":
        common["gravity.meshSize"] = 8
        if args.cuda:
            # The PP CUDA path requires enough candidate pairs to amortize a
            # launch. The eight-cell power-of-two mesh with width two gives each
            # cell a 5**3 neighbor cube; 512 particles and a 4096-particle batch
            # keep the candidate pairs on the GPU rather than falling back to CPU.
            common.update({"run.darkMatterParticleCount": 512, "run.gasParticleCount": 512, "gravity.meshSize": 8,
                           "gravity.pmmmNearWidth": 2, "gravity.pmmmBatchParticles": 4096})
    elif args.cuda:
        common["run.verbose"] = "true"
    if ranks > 1:
        common["hpx:localities"] = ranks

    def run(directory, changes):
        options = dict(common, **changes)
        command = [str(executable)] + [f"--{key}={value}" for key, value in options.items()]
        if args.mpi_exec:
            command = ([args.mpi_exec, args.mpi_numproc_flag, str(ranks), *args.mpi_preflag,
                        command[0], *args.mpi_postflag, *command[1:]])
            command.append("--hpx:ini=hpx.parcel.mpi.enable=1")
            if args.disable_tcp:
                command.append("--hpx:ini=hpx.parcel.tcp.enable=0")
        return runSimulation(command, directory, "run.log")

    def checkpoints(directory):
        return sorted((path for path in directory.glob("checkpoint.*")
                       if path.is_dir() and path.name.split(".")[-1].isdigit()),
                      key=lambda path: int(path.name.split(".")[1]))

    baseline = prepareCase(workdir, "baseline")
    run(baseline, {})
    require(not (baseline / "gravity_error.csv").exists(), "disabled diagnostics wrote output")
    baselineState = readState(checkpoints(baseline)[-1], ranks)
    particleCount = common["run.darkMatterParticleCount"]
    require(len(baselineState[1]) == particleCount,
            f"fixture did not initialize {particleCount} particles")
    sampled = prepareCase(workdir, "sampled")
    sampleOptions = {"gravity.errorInterval": 2, "gravity.errorSampleSize": 7}
    sampledLog = run(sampled, sampleOptions)
    referenceBackend = "CUDA" if args.cuda else "CPU"
    require(f"using {referenceBackend} Ewald table" in sampledLog,
            f"sampled evolution did not select the {referenceBackend} reference backend")
    if args.cuda and args.gravity_method == "fmm":
        require("CUDA kick arena (" in sampledLog,
                "sampled FMM evolution did not execute the CUDA kick path")
    fields, rows = readErrors(sampled, [0, 2, 4], 7, duration)
    sampledCheckpoints = checkpoints(sampled)
    compareState(baselineState, readState(sampledCheckpoints[-1], ranks))
    require(any(float(row["max_absolute"]) > 0.0 for row in rows),
            "fixture produced no measurable comparison error")

    candidates = [(path, readState(path, ranks)[0]) for path in sampledCheckpoints]
    source, sourceState = next((path, state) for path, state in candidates
                              if state.step == 1 and state.integer_time != 0)
    resumed = prepareCase(workdir, "resumed", source)
    # Resume an existing report, retaining only measurements preceding the checkpoint.
    with (resumed / "gravity_error.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in rows if float(row["time"]) < sourceState.tau)
    run(resumed, dict(sampleOptions, **{"checkpoint.readIndex": sourceState.iteration}))
    readErrors(resumed, [0, 2, 4], 7, duration)
    compareState(baselineState, readState(checkpoints(resumed)[-1], ranks))

    capped = prepareCase(workdir, "capped")
    run(capped, {"gravity.errorInterval": 100, "gravity.errorSampleSize": 1000,
                 "run.darkMatterParticleCount": 64, "run.gasParticleCount": 64, "run.maxIterations": 1})
    readErrors(capped, [0], 64, duration)
    print(f"Gravity error sampling passed: cadence, 7/{particleCount} samples, restart, unchanged physics, "
          f"and count cap ({args.gravity_method}, {'CUDA' if args.cuda else 'CPU'}, "
          f"{args.position_type}, {ranks} rank(s))")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, struct.error, StopIteration) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
