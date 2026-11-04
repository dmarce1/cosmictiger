"""Check comparable self-gravity/SPH forces and timestep convergence in a gas star."""

import argparse
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from application_selection import application_arguments, compiled_algorithms, cpu_backend_options
from checkpoint_format import decode_checkpoint, require
from run_checkpoint import prepareCase, runSimulation
from run_sph_hierarchical import compare_restart
from run_sph_octree import checkpoint_state, validate_events


def physical_records(state, base_timestep):
    """Predict checkpoint midpoint velocities to the common position time."""
    records = []
    position_unit = float(2 ** (8 * state.position_bytes))
    for particle in range(state.count):
        gas = state.gas_indices[particle]
        acceleration = state.gas[gas][2:5]
        if state.hierarchy_offset:
            hierarchy = state.hierarchy[gas]
            elapsed = state.driver.tau - hierarchy[6]
            velocity = [hierarchy[3 + dimension] + acceleration[dimension] * elapsed
                        for dimension in range(3)]
        else:
            half_step = math.ldexp(base_timestep, -state.rungs[particle] - 1)
            velocity = [state.velocities[3 * particle + dimension] +
                        acceleration[dimension] * half_step for dimension in range(3)]
        records.append((*(axis[particle] / position_unit for axis in state.positions), *velocity))
    return sorted(records)


def rms_errors(first, second):
    require(len(first) == len(second), "convergence cases changed particle count")
    return tuple(math.sqrt(math.fsum((left[dimension] - right[dimension]) ** 2
                                    for left, right in zip(first, second)
                                    for dimension in dimensions) / len(first))
                 for dimensions in (range(3), range(3, 6)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    available = compiled_algorithms(str(executable))
    require("fmm" in available["far"] and "hierarchical" in available["integrator"],
            "self-gravitating SPH regression requires FMM and hierarchical integration")
    common = {"hpx:threads": args.threads, "hpx:bind": "none", **cpu_backend_options(),
              "initial.particlesPerDimension": 8, "initial.useGlass": "false",
              "hydro.enabled": "true", "gravity.enabled": "true",
              "chemistry.enabled": "false", "starFormation.enabled": "false",
              "analysis.groups": "false", "analysis.tracers": "false", "analysis.views": "false",
              "analysis.power": "false", "analysis.slice": "false", "lightcone.enabled": "false",
              "output.silo": "false", "output.directory": ".", "checkpoint.intervalSeconds": 1000000,
              "run.verbose": "false", "compute.sphBucketSize": 16,
              "hydro.viscosityEvolution": "none", "run.scenario": "polytropicStar",
              "run.numberOfSteps": 2}

    def run(label, options, integrator="hierarchical", *, completed=False, source_checkpoint=None):
        directory = prepareCase(root, label, source_checkpoint)
        command = [str(executable), *application_arguments(
            executable, far="fmm", integrator=integrator, hydro=True)]
        command += [f"--{key}={value}" for key, value in dict(common, **options).items()]
        output = runSimulation(command, directory, "run.log")
        if completed:
            require("L2 Error =" in output, "terminal closing skipped the final star diagnostics")
        iteration = options["run.maxIterations"] + options.get("checkpoint.readIndex", 0)
        checkpoint = directory / f"checkpoint.{iteration}" / "checkpoint.0.dat"
        return decode_checkpoint(checkpoint.read_bytes())

    # The star's pressure is initialized against its self-gravity. These forces
    # have comparable magnitude even though the code-unit gravitational constant
    # is small. Natural timesteps exercise multiple gravity source sets.
    probe = run("nonuniform", {"run.maxIterations": 2})
    require(len(set(probe.rungs)) > 1, "self-gravitating probe did not use multiple rungs")
    hydro_maximum = max(abs(value) for state in probe.hierarchy for value in state[:3])
    gravity_maximum = max(abs(probe.gas[gas][2 + dimension] - probe.hierarchy[gas][dimension])
                          for gas in range(probe.gas_count) for dimension in range(3))
    ratio = gravity_maximum / hydro_maximum
    require(0.05 < ratio < 20, "regression did not exercise comparable gravity and hydro forces")
    evidence = {"force_ratio": ratio, "natural_rungs": sorted(set(probe.rungs)), "convergence": {}}

    # Hold the endpoint and gas realization fixed, forcing uniform positive steps
    # shorter than the natural timestep. Stop before a stage transition; the
    # checkpoint reader predicts the final pending midpoint state to tau=10000.
    records = {}
    for rung in (4, 5, 6, 7):
        state = run(f"rung_{rung}", {"run.endTime": 40000, "run.minRung": rung,
                                   "run.maxIterations": 2 ** (rung - 1)})
        require(state.driver.tau == 10000 and set(state.rungs) == {rung},
                "convergence fixture did not keep a common endpoint and uniform rung")
        require(all(math.isfinite(value) for value in state.velocities) and
                all(value > 0 for value in state.entropies), "invalid gas state")
        records[rung] = physical_records(state, 20000)
        energy = state.driver.energies
        relative_energy_change = abs(energy[0] + energy[2] + energy[3] - state.driver.energy0) / abs(
            state.driver.energy0)
        momentum_fraction = math.sqrt(energy[5] ** 2 + energy[6] ** 2 + energy[8] ** 2) / energy[9]
        # The first star stage contains damping, so energy is bounded, not
        # expected to be conserved exactly. The force pair still conserves net momentum.
        require(relative_energy_change < 0.01, "self-gravitating energy changed by more than one percent")
        require(momentum_fraction < 1e-5, "self-gravitating gas acquired excessive net momentum")
        evidence["convergence"][str(rung)] = {"relative_energy_change": relative_energy_change,
                                                "momentum_fraction": momentum_fraction}
    errors = [rms_errors(records[rung], records[7]) for rung in (4, 5, 6)]
    for index, rung in enumerate((4, 5, 6)):
        evidence["convergence"][str(rung)].update(zip(("rms_position_error", "rms_velocity_error"), errors[index]))
    for previous, current in zip(errors, errors[1:]):
        require(all(fine < 0.6 * coarse for fine, coarse in zip(current, previous)),
                "self-gravitating position/velocity errors did not decrease with smaller timesteps")
    if "traditional" in available["integrator"]:
        state = run("traditional_rung_5", {"run.endTime": 40000, "run.minRung": 5,
                                          "run.maxIterations": 16}, "traditional")
        agreement = rms_errors(records[5], physical_records(state, 20000))
        require(agreement[0] < 1e-9 and agreement[1] < 1e-13,
                "uniform-rung hierarchical and traditional self-gravity/SPH composition differs")
        evidence["traditional_agreement"] = dict(zip(("rms_position_error", "rms_velocity_error"), agreement))
    terminal = run("terminal_floor", {"run.endTime": 40000, "run.minRung": 4,
                                      "run.maxIterations": 32}, completed=True)
    require(terminal.driver.tau == 40000 and set(terminal.rungs) == {4},
            "terminal floor fixture changed endpoint or rung")
    evidence["terminal_minimum_rung"] = 4
    # The expanding main driver has its own rung-merging and drift-factor ledger.
    # This floor puts the cold dark matter one rung below the gas, so the existing
    # one-third rule merges rung nine into ten within these eight physical events.
    mixed_options = {"run.scenario": "cosmology", "run.numberOfSteps": 64,
                     "cosmology.sigma8": 0, "initial.particlesPerDimension": 4,
                     "initial.closePack": "false", "run.minRung": 9,
                     "run.maxIterations": 8, "checkpoint.intervalSeconds": 0,
                     "run.verbose": "true"}
    mixed = run("cosmology", mixed_options)
    mixed_directory = root / "cosmology"
    output = (mixed_directory / "run.log").read_text()
    require("Merging rung 9 into rung 10" in output,
            "cosmological hierarchy did not exercise the gravity rung merge")
    lifecycle = validate_events(output, "hierarchical_cosmology", "hierarchical")
    scales, drift_factors, minimum_rungs = [], [], []
    maximum_gravity = 0.0
    maximum_acceleration = 0.0
    for iteration in range(1, 9):
        state = decode_checkpoint((mixed_directory / f"checkpoint.{iteration}" /
                                   "checkpoint.0.dat").read_bytes())
        require(state.types.count(0) == 64 and state.types.count(1) == 64,
                "mixed cosmological particle populations changed")
        scales.append(state.driver.a)
        minimum_rungs.append(state.driver.min_rung)
        base_timestep = state.driver.tau_max / state.header.steps
        for gas, hierarchy in enumerate(state.hierarchy):
            prediction, position, inverse_scale, entropy, minimum_rung, initialized = hierarchy[6:]
            rung = state.rungs[state.gravity_indices[gas]]
            half_step = math.ldexp(base_timestep, -rung - 1)
            tolerance = 64 * sys.float_info.epsilon * base_timestep
            require(initialized and all(math.isfinite(value) for value in hierarchy),
                    "cosmological predictor is uninitialized or nonfinite")
            require(inverse_scale > 0 and prediction <= state.driver.tau + tolerance and
                    state.driver.tau <= position + tolerance,
                    "cosmological source clock or drift factor is invalid")
            require(math.isclose(position - entropy, half_step, rel_tol=2e-12, abs_tol=tolerance),
                    "cosmological hydro midpoint disagrees with its drift endpoint")
            require(0 <= minimum_rung <= rung and state.entropies[gas] > 0,
                    "cosmological gas has an invalid rung bound or entropy")
            drift_factors.append(inverse_scale)
            maximum_acceleration = max(maximum_acceleration, *(abs(value) for value in state.gas[gas][2:5]))
            maximum_gravity = max(maximum_gravity, *(abs(state.gas[gas][2 + dimension] - hierarchy[dimension])
                                                    for dimension in range(3)))
        require(all(math.isfinite(value) for value in state.velocities),
                "cosmological particle velocity is nonfinite")
    require(scales[-1] > scales[0] and min(drift_factors) < max(drift_factors) and
            max(minimum_rungs) == 10 and maximum_gravity > 0 and maximum_acceleration > 0,
            "cosmological case missed expansion, rung merging, or gravity")
    run("cosmology_restart", dict(mixed_options, **{"run.maxIterations": 1, "checkpoint.readIndex": 7}),
        source_checkpoint=mixed_directory / "checkpoint.7")
    restart_errors = compare_restart(checkpoint_state(mixed_directory, 8, 1),
                                     checkpoint_state(root / "cosmology_restart", 8, 1))
    evidence["cosmology"] = {**lifecycle, "scale_range": [scales[0], scales[-1]],
                            "inverse_drift_scale_range": [min(drift_factors), max(drift_factors)],
                            "minimum_rungs": minimum_rungs, "maximum_gravity": maximum_gravity,
                            "maximum_acceleration": maximum_acceleration,
                            "restart_maximum_errors": restart_errors}
    (root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("Self-gravitating SPH convergence, energy, momentum, cosmological clocks and restart passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
