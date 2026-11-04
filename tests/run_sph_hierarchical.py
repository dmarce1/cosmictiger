"""Exercise chronological midpoint SPH with hierarchical gravity and retained trees."""

import argparse
import itertools
import json
import math
import struct
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from application_selection import application_arguments, compiled_algorithms, cpu_backend_options
from checkpoint_format import decode_checkpoint, require
from run_checkpoint import prepareCase, runSimulation
from run_sph_octree import checkpoint_state, validate_events
from sph_midpoint_checks import check_midpoint_state, validate_midpoints


def read_states(directory, iteration, ranks):
    return [decode_checkpoint((directory / f"checkpoint.{iteration}" /
                              f"checkpoint.{rank}.dat").read_bytes(), ranks=ranks)
            for rank in range(ranks)]



def comparison_records(state):
    records = []
    for particle in range(state.count):
        gas = state.gas_indices[particle]
        records.append((*(axis[particle] for axis in state.positions),
                        *state.velocities[3 * particle:3 * particle + 3],
                        state.entropies[gas], state.smoothing_lengths[gas],
                        state.rungs[particle]))
    return sorted(records)


def compare_restart(reference, resumed, *, position_units=0, accumulation_units=4,
                    entropy_accumulation_units=2, viscosity_accumulation_ulps=0,
                    position_period=2 ** 32):
    require(reference[2:] == resumed[2:], "restart changed particle counts or driver clocks")
    first, second = reference[0], resumed[0]
    require(len(first) == len(second), "restart changed the particle population")
    # Restart reconstructs the octree and can change float neighbor accumulation
    # order. Only force/density-derived float fields may differ at this level;
    # rungs, flags and hydro timestep bounds stay exact. Positive
    # viscosity heating can amplify small source changes in cold gas. Constrain
    # entropy locally with an explicit case-specific allowance and globally
    # with the original accumulation bound; the default local limit stays tight.
    # CUDA rebuilding can perturb the live velocity by float accumulation
    # rounding. Two subsequent half-drifts can then move fixed32 positions by
    # two units. This explicit GPU regression tolerance is not an error bound;
    # CPU positions remain exact. Match uniquely within the periodic allowance
    # so canonical xyz sorting cannot exchange particles in equal-x columns.
    maximum_position_units = 0
    maximum_entropy_ulps = 0
    maximum_viscosity_ulps = 0
    if position_units:
        require(position_units == 2 and position_period % 4 == 0,
                "unsupported restart position tolerance")
        cells = position_period // 4
        grid = {}
        for row in second:
            key = (row[0], *(coordinate // 4 for coordinate in row[1:4]))
            grid.setdefault(key, []).append(row)
        matched = []
        for row in first:
            candidates = []
            for offset in itertools.product((-1, 0, 1), repeat=3):
                key = (row[0], *((coordinate // 4 + shift) % cells
                                for coordinate, shift in zip(row[1:4], offset)))
                for candidate in grid.get(key, ()):
                    differences = [min(abs(old - new), position_period - abs(old - new))
                                   for old, new in zip(row[1:4], candidate[1:4])]
                    if max(differences) <= position_units:
                        candidates.append((key, candidate, max(differences)))
            require(len(candidates) == 1,
                    "GPU restart lacks a unique particle match within two position units")
            key, candidate, difference = candidates[0]
            grid[key].remove(candidate)
            matched.append(candidate)
            maximum_position_units = max(maximum_position_units, difference)
        require(not any(grid.values()), "GPU restart left unmatched particles")
        second = tuple(matched)
    particle_fields = (4, 5, 6)
    # Version 130000 packs flags and proposal into one byte after density, then
    # viscosity history. Removed forces/rates must not shift discrete fields
    # into the float-tolerated set. Viscosity history remains exact here.
    gas_fields = (0, 1, 2)
    scales = {}
    for field in particle_fields:
        scales[("particle", field)] = max(abs(row[field]) for row in first + second)
    for field in gas_fields:
        scales[("gas", field)] = max(abs(row[-1][field]) for row in first + second if row[0] == 1)
    errors = {key: 0.0 for key in scales}
    for left, right in zip(first, second):
        require(left[0] == right[0] and (position_units or left[1:4] == right[1:4]) and
                left[7] == right[7],
                "restart changed exact particle positions, species or rungs")
        require(len(left[-1]) == len(right[-1]), "restart changed gas state layout")
        for field in particle_fields:
            key = ("particle", field)
            errors[key] = max(errors[key], abs(left[field] - right[field]))
        for field, (old, new) in enumerate(zip(left[-1], right[-1])):
            if field in gas_fields:
                if field == 0:
                    maximum_entropy_ulps = max(maximum_entropy_ulps, abs(
                        struct.unpack("=I", struct.pack("=f", old))[0] -
                        struct.unpack("=I", struct.pack("=f", new))[0]))
                    require(abs(old - new) <= entropy_accumulation_units * 2 ** -23 *
                            max(abs(old), abs(new)),
                            "restart changed entropy beyond local float rounding")
                key = ("gas", field)
                errors[key] = max(errors[key], abs(old - new))
            elif field == 6 and viscosity_accumulation_ulps:
                # Optional alpha-only bound; divergence histories and timestamps
                # at later fields retain their exact restart requirement.
                ulps = abs(struct.unpack("=I", struct.pack("=f", old))[0] -
                           struct.unpack("=I", struct.pack("=f", new))[0])
                maximum_viscosity_ulps = max(maximum_viscosity_ulps, ulps)
                require(ulps <= viscosity_accumulation_ulps,
                        "restart changed alpha beyond its explicit float rounding allowance")
            else:
                require(old == new, f"restart changed exact gas field {field}")
    for key, error in errors.items():
        require(error <= accumulation_units * 2 ** -23 * scales[key],
                f"restart changed {key} beyond float accumulation tolerance: {error}")
    return {"maximum_position_units": maximum_position_units,
            "maximum_entropy_ulps": maximum_entropy_ulps,
            "maximum_viscosity_ulps": maximum_viscosity_ulps,
            "maximum_accumulation_float_units": max(
                (error / (2 ** -23 * scales[key]) if scales[key] else 0)
                for key, error in errors.items()),
            **{f"{kind}_{field}": error for (kind, field), error in errors.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--cuda", action="store_true", help="use CUDA for gravity and SPH")
    parser.add_argument("--gravity-cuda", action="store_true")
    parser.add_argument("--hydro-cuda", action="store_true")
    parser.add_argument("--mpi-exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--hpx-argument", action="append", default=[])
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ranks = 2 if args.mpi_exec else 1
    available = compiled_algorithms(str(executable))
    require("fmm" in available["far"] and "hierarchical" in available["integrator"],
            "hierarchical SPH regression requires FMM and hierarchical integration")
    common = {"hpx:threads": args.threads, "hpx:bind": "none", **cpu_backend_options(),
              "initial.particlesPerDimension": 8, "initial.useGlass": "false",
              "initial.closePack": "false", "hydro.enabled": "true", "gravity.enabled": "true",
              "chemistry.enabled": "false", "starFormation.enabled": "false",
              "analysis.groups": "false", "analysis.tracers": "false", "analysis.views": "false",
              "analysis.power": "false", "analysis.slice": "false", "lightcone.enabled": "false",
              "output.silo": "false", "output.directory": ".", "checkpoint.intervalSeconds": 0,
              "run.verbose": "true", "compute.sphBucketSize": 16,
              "hydro.viscosityEvolution": "morrisAndMonaghan"}
    common["gravity.cuda"] = "true" if args.cuda or args.gravity_cuda else "false"
    common["hydro.cuda"] = "true" if args.cuda or args.hydro_cuda else "false"

    def run(directory, options, integrator="hierarchical", restart=None):
        command = [str(executable), *args.hpx_argument, *application_arguments(
            executable, far="fmm", integrator=integrator, hydro=True)]
        command += [f"--{key}={value}" for key, value in dict(common, **options).items()]
        if restart is not None:
            command.append(f"--checkpoint.readIndex={restart}")
        if ranks == 2:
            command = [args.mpi_exec, args.mpi_numproc_flag, "2", *args.mpi_preflag,
                       command[0], *args.mpi_postflag, *command[1:],
                       "--hpx:localities=2", "--hpx:ini=hpx.parcel.mpi.enable=1"]
        return runSimulation(command, directory, "run.log")

    if ranks == 2:
        fixture = prepareCase(root, "hierarchy_migration")
        output = run(fixture, {"run.test": "sphTree"})
        require("SPH hierarchical migration regression passed: crossed=2 prefix=7" in output,
                "two-rank midpoint gas migration fixture did not finish")
        require("SPH hierarchical out-of-domain rebuild regression passed: "
                "active=6 total=16 inactive_outside=10" in output,
                "two-rank hierarchy did not rebuild around inactive out-of-domain gas")
        require("SPH common-boundary rung closure regression passed: ranks=2 passes=7 inactive=6"
                in output, "two-rank inactive-neighbor rung closure fixture did not finish")

    # A localized blast supplies hot fine-rung gas and cold coarse-rung sources.
    # Stop before the endpoint so restart resumes the boundary gravity ladder
    # with all particle positions at the same chronological time.
    options = {"run.scenario": "blast", "run.endTime": 0.005,
               "run.numberOfSteps": 1, "run.maxIterations": 8,
               "initial.particlesPerDimension": 16}
    directory = prepareCase(root, "blast")
    output = run(directory, options)
    require("SPH midpoint:" in output, "blast driver did not execute midpoint hydro")
    events = validate_events(output, "hierarchical_blast", "hierarchical")
    previous_entropy = 0.0
    rungs = set()
    for iteration in range(1, options["run.maxIterations"] + 1):
        states = read_states(directory, iteration, ranks)
        for state in states:
            check_midpoint_state(state)
            rungs.update(state.rungs)
        entropy = math.fsum(value for state in states for value in state.entropies)
        require(entropy >= previous_entropy * (1.0 - 1e-7),
                "positive viscosity evolution decreased accumulated entropy")
        previous_entropy = entropy
    require(len(rungs) >= 2, "blast failed to exercise multiple gas rungs")
    midpoint_events = validate_midpoints(output, "hierarchical_blast", options["run.endTime"],
                                        states[0].driver.tau)
    final = checkpoint_state(directory, options["run.maxIterations"], ranks)
    restart_iteration = options["run.maxIterations"] - 1
    resumed = prepareCase(root, "blast_restart", directory / f"checkpoint.{restart_iteration}")
    run(resumed, dict(options, **{"run.maxIterations": 1}), restart=restart_iteration)
    continued = checkpoint_state(resumed, options["run.maxIterations"], ranks)
    gpu = args.cuda or args.gravity_cuda or args.hydro_cuda
    # Retained-vs-rebuilt density/support sums differ by 1--3 float ULPs.
    # In the cold blast shell this perturbs positive heating by 52--54 local
    # entropy ULPs, while hot-cell errors stay within the global 4/8-eps bound.
    # Repeating the rebuilt CPU case with one and two threads reproduced entropy
    # exactly. MM alpha changes by at most one ULP in this comparison. Scope
    # both explicit allowances to this conditioned blast case; histories stay exact.
    restart_errors = compare_restart(final, continued, position_units=2 if gpu else 0,
                                     accumulation_units=8 if gpu else 4,
                                     entropy_accumulation_units=64, viscosity_accumulation_ulps=2,
                                     position_period=2 ** (8 * states[0].position_bytes))
    for state in read_states(resumed, options["run.maxIterations"], ranks):
        check_midpoint_state(state)
    evidence = {"blast": {**events, "rungs": sorted(rungs), "midpoints": midpoint_events,
                           "entropy": previous_entropy, "ranks": ranks,
                           "exact_restart": final[0] == continued[0],
                           "restart_maximum_absolute_errors": restart_errors}}

    if "traditional" in available["integrator"]:
        # Forced uniform rungs reduce both schemes to the same midpoint composition.
        # A short density-wave interval keeps gas on the configured minimum rung.
        options = {"run.scenario": "hydroWave", "run.endTime": 0.001,
                   "run.numberOfSteps": 1, "run.maxIterations": 2, "run.minRung": 3}
        states = []
        for integrator in ("hierarchical", "traditional"):
            directory = prepareCase(root, "uniform_" + integrator)
            uniform_output = run(directory, options, integrator)
            localities = read_states(directory, options["run.maxIterations"], ranks)
            require(all(set(state.rungs) == {3} for state in localities),
                    "uniform comparison did not remain on rung three")
            validate_midpoints(uniform_output, f"uniform_{integrator}", options["run.endTime"],
                               localities[0].driver.tau, 3)
            states.append(localities)
        require(all(state.driver.tau == states[0][0].driver.tau
                    for localities in states for state in localities), "uniform clocks diverged")
        first, second = [sorted(row for state in localities for row in comparison_records(state))
                         for localities in states]
        require(len(first) == len(second), "uniform integrators changed the particle population")
        position_unit = float(2 ** (8 * states[0][0].position_bytes))
        maximum_error = 0.0
        for left, right in zip(first, second):
            for dimension in range(3):
                difference = abs(left[dimension] - right[dimension]) / position_unit
                require(min(difference, 1.0 - difference) <= 2e-7,
                        "uniform hierarchical/traditional positions diverged")
            for left_value, right_value in zip(left[3:-1], right[3:-1]):
                error = abs(left_value - right_value)
                maximum_error = max(maximum_error, error)
                require(math.isclose(left_value, right_value, rel_tol=3e-5, abs_tol=2e-7),
                        "uniform hierarchical/traditional hydro state diverged")
        evidence["uniform"] = {"maximum_absolute_state_error": maximum_error}
    (root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("Hierarchical SPH midpoint timing, positive entropy, retained topology and restart passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
