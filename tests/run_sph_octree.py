"""Validate retained SPH octrees, chronological midpoint kicks, and restarts.

The deterministic application fixture tests sparse activity, periodic drift and
unchanged membership. With MPI it also crosses both rank boundaries without
migration and checks neighbors against particle distances. Evolution cases check
the driver's actual inclusive one-third decisions independently of whichever
rungs the physical scenario happens to select.
"""

import argparse
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

from application_selection import application_arguments, compiled_algorithms, cpu_backend_options
from checkpoint_format import decode_checkpoint, require
from run_checkpoint import prepareCase, runSimulation, check_velocity_roundoff_restart
from sph_midpoint_checks import check_midpoint_state, validate_midpoints


EVENT = re.compile(r"SPH octree (rebuild|retain): active=(\d+) total=(\d+)"
                   r"(?: reason=(threshold|invalidated))?")


def validate_events(output, label, integrator, fraction=1 / 3):
    events = EVENT.findall(output)
    require(events, f"{label}: no SPH octree lifecycle diagnostics")
    counts = {"rebuild": 0, "retain": 0}
    for operation, active_text, total_text, reason in events:
        active, total = int(active_text), int(total_text)
        require(0 <= active <= total and total > 0, f"{label}: invalid active/total counts")
        threshold = 3 * active >= total if fraction == 1 / 3 else active / total >= fraction
        counts[operation] += 1
        if operation == "retain":
            require(not threshold and not reason,
                    f"{label}: retained the octree at or above fraction {fraction}")
        else:
            require(reason in ("threshold", "invalidated"), f"{label}: missing rebuild reason")
            if reason == "threshold":
                require(threshold, f"{label}: threshold rebuild below fraction {fraction}")
        if integrator == "global":
            require(active == total and operation == "rebuild",
                    f"{label}: global integration did not rebuild its fully active octree")
    require(counts["rebuild"] > 0, f"{label}: never constructed an octree")
    return counts


def checkpoint_state(directory, iteration, ranks):
    records = []
    clocks = []
    species = [0, 0]
    payloads = []
    for rank in range(ranks):
        path = directory / f"checkpoint.{iteration}" / f"checkpoint.{rank}.dat"
        state = decode_checkpoint(path.read_bytes(), ranks=ranks)
        require(state.hydro and not state.has_cooling and not state.has_stars,
                f"{path}: expected ideal-gas SPH without additional sources")
        require(140000 <= state.gas_version < 141000, f"{path}: obsolete SPH rate checkpoint")
        clocks.append((state.driver.tau, state.driver.integer_time,
                       state.driver.global_timestep, state.driver.global_hydro_timestep,
                       state.driver.iteration))
        payloads.append(state.physical)
        for particle, kind in enumerate(state.types):
            require(kind in (0, 1), f"{path}: unexpected particle species")
            species[kind] += 1
            gas = state.gas[state.gas_indices[particle]] if kind == 1 else ()
            if kind == 1:
                require(gas[0] > 0 and gas[1] > 0,
                        f"{path}: nonpositive entropy or support radius")
            records.append((kind, *(axis[particle] for axis in state.positions),
                            *state.velocities[3 * particle:3 * particle + 3],
                            state.rungs[particle], gas))
        require(all(math.isfinite(value) for value in state.velocities),
                f"{path}: nonfinite velocity")
    require(all(clock == clocks[0] for clock in clocks),
            f"{directory}: locality clocks differ")
    # Particle and gas-array indices may change when a restart rebuilds the tree.
    # Compare exact physical/stage values in canonical particle order as well.
    return tuple(sorted(records)), tuple(payloads), clocks[0], tuple(species)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--time-integrator", action="append", choices=("global", "traditional"))
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
    available = compiled_algorithms(str(executable))
    integrators = args.time_integrator or [name for name in ("global", "traditional")
                                          if name in available["integrator"]]
    require(integrators, "global or traditional integration is required")
    common = {"hpx:threads": args.threads, "hpx:bind": "none", **cpu_backend_options(),
              "run.darkMatterParticleCount": 512, "run.gasParticleCount": 512, "scenario.cosmology.useGlass": "false",
              "initial.closePack": "false", "hydro.enabled": "true",
              # SPH has no self-gravity coupling in the PMMM solver.
              "gravity.enabled": "false", "chemistry.enabled": "false",
              "starFormation.enabled": "false", "analysis.groups": "false",
              "analysis.tracers": "false", "analysis.views": "false",
              "analysis.power": "false", "analysis.slice": "false",
              "lightcone.enabled": "false", "output.silo": "false",
              "output.directory": ".", "checkpoint.intervalSeconds": 0,
              "run.verbose": "true", "compute.sphBucketSize": 16,
              "hydro.viscosityCoefficientMin": 1, "hydro.viscosityCoefficientMax": 1}
    common["gravity.cuda"] = "true" if args.cuda or args.gravity_cuda else "false"
    common["hydro.cuda"] = "true" if args.cuda or args.hydro_cuda else "false"

    def run(directory, options, integrator, *, ranks=1, restart=None):
        command = [str(executable), *args.hpx_argument,
                   *application_arguments(executable, integrator=integrator, hydro=True)]
        command += [f"--{key}={value}" for key, value in dict(common, **options).items()]
        if restart is not None:
            command.append(f"--checkpoint.readIndex={restart}")
        if ranks == 2:
            command = [args.mpi_exec, args.mpi_numproc_flag, "2", *args.mpi_preflag,
                       command[0], *args.mpi_postflag, *command[1:],
                       "--hpx:localities=2", "--hpx:ini=hpx.parcel.mpi.enable=1"]
        return runSimulation(command, directory, "run.log")

    evidence = {"fixtures": [], "evolution": {}}
    fixture = prepareCase(root, "tree_geometry")
    output = run(fixture, {"run.test": "sphTree"}, integrators[0])
    require("SPH tree arena regression passed" in output, "serial octree fixture did not finish")
    evidence["fixtures"].append("serial: periodic/signed drift, membership, support growth, "
                                "threshold, midpoint target selection and live sources")
    if args.mpi_exec:
        fixture = prepareCase(root, "tree_geometry_mpi")
        output = run(fixture, {"run.test": "sphTree"}, integrators[0], ranks=2)
        marker = "SPH octree distributed retain regression passed: active=2 total=16 crossed=2"
        require(marker in output, "two-rank sparse crossing fixture did not finish")
        evidence["fixtures"].append("two ranks: 2/16 active, both domain boundaries crossed without migration")

    for integrator in integrators:
        cases = [("wave", {"run.scenario": "hydroWave", "run.endTime": 0.5,
                           "run.numberOfSteps": 1, "run.maxIterations": 8}, 1)]
        if args.mpi_exec:
            cases.append(("wave_mpi", dict(cases[-1][1]), 2))
        for name, options, ranks in cases:
            label = f"{integrator}_{name}"
            reference = prepareCase(root, label)
            output = run(reference, options, integrator, ranks=ranks)
            require("SPH midpoint:" in output, f"{label}: driver did not execute midpoint hydro")
            counts = validate_events(output, label, integrator)
            final_iteration = options["run.maxIterations"]
            final, payloads, clock, species = checkpoint_state(reference, final_iteration, ranks)
            previous_time = 0.0
            global_intervals = []
            for iteration in range(1, final_iteration + 1):
                state = decode_checkpoint((reference / f"checkpoint.{iteration}" /
                                           "checkpoint.0.dat").read_bytes(), ranks=ranks)
                check_midpoint_state(state)
                global_intervals.append((previous_time, state.driver.tau))
                previous_time = state.driver.tau
            base_timestep = state.driver.tau_max / state.header.steps
            counts["midpoints"] = validate_midpoints(output, label, base_timestep,
                state.driver.tau, global_intervals=global_intervals if integrator == "global" else None)
            require(species[1] > 0 and (name.startswith("wave") or species[0] > 0),
                    f"{label}: did not exercise the requested species")
            # One resumed event is sufficient to exercise tree reconstruction and
            # reading the current checkpoint without the removed gravity arrays.
            restart_iteration = final_iteration - 1
            resumed = prepareCase(root, label + "_restart",
                                  reference / f"checkpoint.{restart_iteration}")
            restart_output = run(resumed, dict(options, **{"run.maxIterations": 1}),
                                 integrator, ranks=ranks, restart=restart_iteration)
            require("Done reading checkpoint" in restart_output,
                    f"{label}: checkpoint was not loaded")
            continued, continued_payloads, continued_clock, continued_species = checkpoint_state(
                resumed, final_iteration, ranks)
            require(clock == continued_clock and species == continued_species,
                    f"{label}: restart changed particle counts or simulation clock")
            if name.startswith("wave"):
                # Pair reduction order leaves transverse residuals below 0.12
                # float eps in this wave; a following drift may round a position to
                # the neighboring fixed-point unit. Every other payload byte is exact.
                counts["restart_maximum_velocity_errors"] = check_velocity_roundoff_restart(
                    reference / f"checkpoint.{final_iteration}",
                    resumed / f"checkpoint.{final_iteration}", ranks=ranks)
            else:
                require(final == continued,
                        f"{label}: restart changed exact particle/stage values")
            counts["restart_payload_identical"] = payloads == continued_payloads
            counts["particles"] = sum(species)
            counts["gas"] = species[1]
            counts["ranks"] = ranks
            evidence["evolution"][label] = counts
            restart_kind = "velocity-roundoff" if name == "wave" else "exact physical"
            print(f"{label}: {counts['rebuild']} rebuilds, {counts['retain']} retained events, "
                  f"{restart_kind} restart passed", flush=True)
    (root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("SPH octree application regression passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
