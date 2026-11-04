"""Exercise shared SPH rates with hierarchical gravity and explicit source clocks."""

import argparse
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from application_selection import application_arguments, compiled_algorithms, cpu_backend_options
from checkpoint_format import decode_checkpoint, require
from run_checkpoint import prepareCase, runSimulation
from run_sph_octree import checkpoint_state, validate_events


def read_states(directory, iteration, ranks):
    return [decode_checkpoint((directory / f"checkpoint.{iteration}" /
                              f"checkpoint.{rank}.dat").read_bytes(), ranks=ranks)
            for rank in range(ranks)]


def check_clocks(state, base_timestep):
    require(state.hierarchy_offset == 1000 and len(state.hierarchy) == state.gas_count,
            "hierarchical checkpoint omitted the predictor state")
    future = 0
    for gas, values in enumerate(state.hierarchy):
        require(all(math.isfinite(value) for value in values), "nonfinite hierarchical predictor")
        prediction, position, inverse_scale, entropy, minimum_rung, initialized = values[6:]
        particle = state.gravity_indices[gas]
        rung = state.rungs[particle]
        half_step = math.ldexp(base_timestep, -rung - 1)
        require(initialized and inverse_scale == 1.0, "fixed-background drift metadata is invalid")
        require(prediction <= state.driver.tau <= position + 1e-14,
                "predictor or stored position is on the wrong side of the driver clock")
        require(math.isclose(position - entropy, half_step, rel_tol=2e-12, abs_tol=1e-14),
                "positive hydro half-step does not match the recorded drift endpoint")
        require(0 <= minimum_rung <= rung, "gravity selected a rung below its hydro bound")
        require(state.entropies[gas] > 0, "nonpositive entropy")
        future += position > state.driver.tau + 1e-14
    return future


def comparison_records(state):
    records = []
    for particle in range(state.count):
        gas = state.gas_indices[particle]
        records.append((*(axis[particle] for axis in state.positions),
                        *state.velocities[3 * particle:3 * particle + 3],
                        state.entropies[gas], state.support_radii[gas],
                        state.rungs[particle]))
    return sorted(records)


def compare_restart(reference, resumed):
    require(reference[2:] == resumed[2:], "restart changed particle counts or driver clocks")
    first, second = reference[0], resumed[0]
    require(len(first) == len(second), "restart changed the particle population")
    # Restart reconstructs the octree and can change float neighbor accumulation
    # order. Only force/density-derived float fields may differ at this level;
    # positions, entropy, entropy rate, rungs, flags and every stage clock stay exact.
    particle_fields = (4, 5, 6)
    gas_fields = (1, 2, 3, 4, 6, 9, 10, 11, 12, 13, 14)
    scales = {}
    for field in particle_fields:
        scales[("particle", field)] = max(abs(row[field]) for row in first + second)
    for field in gas_fields:
        scales[("gas", field)] = max(abs(row[-1][field]) for row in first + second if row[0] == 1)
    errors = {key: 0.0 for key in scales}
    for left, right in zip(first, second):
        require(left[:4] == right[:4] and left[7] == right[7],
                "restart changed exact particle positions, species or rungs")
        require(len(left[-1]) == len(right[-1]), "restart changed gas state layout")
        for field in particle_fields:
            key = ("particle", field)
            errors[key] = max(errors[key], abs(left[field] - right[field]))
        for field, (old, new) in enumerate(zip(left[-1], right[-1])):
            if field in gas_fields:
                key = ("gas", field)
                errors[key] = max(errors[key], abs(old - new))
            else:
                require(old == new, f"restart changed exact gas field {field}")
    for key, error in errors.items():
        require(error <= 4 * 2 ** -23 * scales[key],
                f"restart changed {key} beyond float accumulation tolerance: {error}")
    return {f"{kind}_{field}": error for (kind, field), error in errors.items()}


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
              "run.verbose": "true", "compute.sphBucketSize": 16, "hydro.viscosityEvolution": "none"}
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
                "two-rank hierarchical state migration fixture did not finish")
        require("SPH hierarchical out-of-domain rebuild regression passed: "
                "active=6 total=16 inactive_outside=10" in output,
                "two-rank hierarchy did not rebuild around inactive out-of-domain gas")

    # A localized blast supplies hot fine-rung gas and cold coarse-rung sources.
    # Stop before the endpoint so the checkpoint contains pending positive hydro
    # half-kicks and source positions stored ahead of the event clock.
    options = {"run.scenario": "blast", "run.endTime": 0.005,
               "run.numberOfSteps": 1, "run.maxIterations": 8,
               "initial.particlesPerDimension": 16}
    directory = prepareCase(root, "blast")
    output = run(directory, options)
    events = validate_events(output, "hierarchical_blast", "hierarchical")
    previous_entropy = 0.0
    maximum_future = 0
    rungs = set()
    for iteration in range(1, options["run.maxIterations"] + 1):
        states = read_states(directory, iteration, ranks)
        maximum_future = max(maximum_future,
                             sum(check_clocks(state, options["run.endTime"]) for state in states))
        for state in states:
            rungs.update(state.rungs)
        entropy = math.fsum(value for state in states for value in state.entropies)
        require(entropy >= previous_entropy * (1.0 - 1e-7),
                "positive viscosity evolution decreased accumulated entropy")
        previous_entropy = entropy
    require(len(rungs) >= 2 and maximum_future > 0,
            "blast failed to exercise slow sources at future positions")
    final = checkpoint_state(directory, options["run.maxIterations"], ranks)
    restart_iteration = options["run.maxIterations"] - 1
    resumed = prepareCase(root, "blast_restart", directory / f"checkpoint.{restart_iteration}")
    run(resumed, dict(options, **{"run.maxIterations": 1}), restart=restart_iteration)
    continued = checkpoint_state(resumed, options["run.maxIterations"], ranks)
    restart_errors = compare_restart(final, continued)
    for state in read_states(resumed, options["run.maxIterations"], ranks):
        check_clocks(state, options["run.endTime"])
    evidence = {"blast": {**events, "rungs": sorted(rungs), "maximum_future_sources": maximum_future,
                           "entropy": previous_entropy, "ranks": ranks,
                           "exact_restart": final[0] == continued[0],
                           "restart_maximum_absolute_errors": restart_errors}}

    if "traditional" in available["integrator"]:
        # Forced uniform rungs reduce both schemes to the same KDK composition.
        # A short density-wave interval keeps gas on the configured minimum rung.
        options = {"run.scenario": "hydroWave", "run.endTime": 0.001,
                   "run.numberOfSteps": 1, "run.maxIterations": 2, "run.minRung": 3}
        states = []
        for integrator in ("hierarchical", "traditional"):
            directory = prepareCase(root, "uniform_" + integrator)
            run(directory, options, integrator)
            localities = read_states(directory, options["run.maxIterations"], ranks)
            require(all(set(state.rungs) == {3} for state in localities),
                    "uniform comparison did not remain on rung three")
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
    print("Hierarchical SPH clocks, positive entropy, future-source prediction and restart passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
