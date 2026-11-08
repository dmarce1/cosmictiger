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
from sph_midpoint_checks import check_midpoint_state, validate_midpoints
from run_sph_octree import checkpoint_state, validate_events


def physical_records(state):
    """Positions at tau and stored momenta before the pending gravity boundary."""
    records = []
    position_unit = float(2 ** (8 * state.position_bytes))
    for particle in range(state.count):
        velocity = state.velocities[3 * particle:3 * particle + 3]
        records.append((*(axis[particle] / position_unit for axis in state.positions), *velocity))
    return sorted(records)


def rms_errors(first, second):
    require(len(first) == len(second), "convergence cases changed particle count")
    return tuple(math.sqrt(math.fsum((left[dimension] - right[dimension]) ** 2
                                    for left, right in zip(first, second)
                                    for dimension in dimensions) / len(first))
                 for dimensions in (range(3), range(3, 6)))


def hydro_wave_records(state, side):
    # The wave moves only in x. Equal-x particles in different transverse
    # columns can exchange order after float force accumulation, so raw xyz
    # sorting is not a particle correspondence. Label the unchanged y/z lattice
    # columns, then use the ordered x planes within each column.
    columns = {}
    for row in physical_records(state):
        column = tuple(round(row[dimension] * side - 0.5) for dimension in (1, 2))
        require(all(0 <= index < side for index in column), "wave column left its lattice")
        require(all(abs(row[dimension] - (index + 0.5) / side) < 1e-7
                    for dimension, index in zip((1, 2), column)),
                "wave acquired transverse motion, invalidating particle correspondence")
        columns.setdefault(column, []).append(row)
    require(len(columns) == side * side and all(len(rows) == side for rows in columns.values()),
            "wave refinement changed its transverse particle columns")
    records = []
    for column in sorted(columns):
        rows = sorted(columns[column])
        require(all(right[0] - left[0] > 1e-5 for left, right in zip(rows, rows[1:])),
                "wave planes became coincident, invalidating particle correspondence")
        records.extend(rows)
    return records


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
              "run.darkMatterParticleCount": 512, "run.gasParticleCount": 512, "initial.cosmology.useGlass": "false",
              "hydro.enabled": "true", "gravity.enabled": "true",
              "chemistry.enabled": "false", "starFormation.enabled": "false",
              "analysis.groups": "false", "analysis.tracers": "false", "analysis.views": "false",
              "analysis.power": "false", "analysis.slice": "false", "lightcone.enabled": "false",
              "output.silo": "false", "output.directory": ".", "checkpoint.intervalSeconds": 1000000,
              "run.verbose": "true", "compute.sphBucketSize": 16,
              "run.scenario": "polytropicStar",
              "hydro.viscosityAlphaMin": 0, "hydro.viscosityAlphaMax": 0,
              "hydro.viscosityBeta": 0,
              "run.numberOfSteps": 2}

    def run(label, options, integrator="hierarchical", *, completed=False, source_checkpoint=None):
        directory = prepareCase(root, label, source_checkpoint)
        command = [str(executable), *application_arguments(
            executable, far="fmm", integrator=integrator, hydro=True)]
        command += [f"--{key}={value}" for key, value in dict(common, **options).items()]
        output = runSimulation(command, directory, "run.log")
        require("SPH midpoint:" in output, f"{label}: driver did not execute midpoint hydro")
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
    # Derived short-step accounting with initially zero velocities:
    #   v(h)/h = G/2 + H + O(h), x(h) = x(0) + h²(G+H)/2 + O(h³).
    # Two first-step runs eliminate the unknown initial position:
    #   C = [x(h)-x(h/2)]/[h²-(h/2)²], W=v(h)/h;
    #   G ≈ 4C-2W, H ≈ 2(W-C).
    # These finite-step estimates establish comparable force scales; they are
    # not an accuracy test of either force. No force/rate array is checkpointed.
    first_step = run("force_balance", {"run.endTime": 40000, "run.minRung": 4,
                                       "run.maxIterations": 1})
    step = 20000 / 16
    require(first_step.driver.tau == step and set(first_step.rungs) == {4},
            "force-balance fixture did not perform one uniform step")
    half_step = run("force_balance_half", {"run.endTime": 40000, "run.minRung": 5,
                                           "run.maxIterations": 1})
    require(half_step.driver.tau == step / 2 and set(half_step.rungs) == {5},
            "force-balance refinement did not halve its uniform first interval")
    coarse, fine = physical_records(first_step), physical_records(half_step)
    require(len(coarse) == len(fine), "force-balance refinement changed the gas population")
    unused = set(range(len(fine)))
    gravity_maximum = hydro_maximum = 0.0
    maximum_displacement = 0.0
    for row in coarse:
        matches = [index for index in unused
                   if sum((row[dimension] - fine[index][dimension]) ** 2
                          for dimension in range(3)) < 1e-6]
        require(len(matches) == 1, "force-balance runs lack a unique nearby particle match")
        other = fine[matches[0]]
        unused.remove(matches[0])
        for dimension in range(3):
            displacement = row[dimension] - other[dimension]
            maximum_displacement = max(maximum_displacement, abs(displacement))
            coefficient = displacement / (step ** 2 - (step / 2) ** 2)
            momentum_rate = row[dimension + 3] / step
            gravity_maximum = max(gravity_maximum, abs(4 * coefficient - 2 * momentum_rate))
            hydro_maximum = max(hydro_maximum, abs(2 * (momentum_rate - coefficient)))
    require(maximum_displacement > 32 / 2 ** (8 * first_step.position_bytes),
            "force-balance displacement is hidden by fixed-point rounding")
    require(hydro_maximum > 0, "force-balance fixture did not evaluate midpoint hydro")
    ratio = gravity_maximum / hydro_maximum
    require(0.05 < ratio < 20, "regression did not exercise comparable gravity and hydro forces")
    evidence = {"estimated_force_ratio": ratio,
                "force_balance_displacement": maximum_displacement,
                "natural_rungs": sorted(set(probe.rungs)), "convergence": {}}


    # Hold the endpoint and gas realization fixed, forcing uniform positive steps
    # shorter than the natural timestep. Stop before a stage transition; the
    # positions are already at tau=10000. The relaxation stage includes an
    # explicit velocity-damping source, so this case checks convergence but does
    # not establish second order. Stored momenta retain a gravity half-kick.
    records = {}
    for rung in (4, 5, 6, 7):
        state = run(f"rung_{rung}", {"run.endTime": 40000, "run.minRung": rung,
                                   "run.maxIterations": 2 ** (rung - 1)})
        require(state.driver.tau == 10000 and set(state.rungs) == {rung},
                "convergence fixture did not keep a common endpoint and uniform rung")
        check_midpoint_state(state)
        validate_midpoints((root / f"rung_{rung}" / "run.log").read_text(), f"rung_{rung}",
                           20000, state.driver.tau, rung)
        records[rung] = physical_records(state)
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
        evidence["convergence"][str(rung)].update(zip(("rms_position_error", "rms_staggered_velocity_difference"), errors[index]))
    for previous, current in zip(errors, errors[1:]):
        require(current[0] < 0.6 * previous[0],
                "damped star position errors did not decrease with timestep refinement")
    evidence["position_orders"] = [math.log2(previous[0] / current[0])
                                   for previous, current in zip(errors, errors[1:])]
    if "traditional" in available["integrator"]:
        state = run("traditional_rung_5", {"run.endTime": 40000, "run.minRung": 5,
                                          "run.maxIterations": 16}, "traditional")
        agreement = rms_errors(records[5], physical_records(state))
        require(agreement[0] < 1e-9 and agreement[1] < 1e-13,
                "uniform-rung hierarchical and traditional self-gravity/SPH composition differs")
        evidence["traditional_agreement"] = dict(zip(("rms_position_error", "rms_velocity_error"), agreement))
    terminal = run("terminal_floor", {"run.endTime": 40000, "run.minRung": 4,
                                      "run.maxIterations": 32}, completed=True)
    require(terminal.driver.tau == 40000 and set(terminal.rungs) == {4},
            "terminal floor fixture changed endpoint or rung")
    evidence["terminal_minimum_rung"] = 4
    # Separate temporal and spatial error using one unchanged gas realization,
    # a common endpoint, fixed uniform rungs, and zero viscosity/damping. The
    # density-wave scenario supplies a pressure-dominated inviscid test; the
    # star above independently exercises comparable gravity and pressure.
    wave_records = {}
    for rung in (4, 5, 6, 7):
        state = run(f"wave_order_{rung}", {"run.scenario": "hydroWave", "run.endTime": 0.1,
                    "run.numberOfSteps": 1, "run.minRung": rung,
                    "run.maxIterations": 2 ** (rung - 1)})
        require(math.isclose(state.driver.tau, 0.05, rel_tol=2e-12) and
                set(state.rungs) == {rung},
                "inviscid refinement changed the endpoint or selected a different rung")
        check_midpoint_state(state)
        validate_midpoints((root / f"wave_order_{rung}" / "run.log").read_text(),
                           f"wave_order_{rung}", 0.1, state.driver.tau, rung)
        wave_records[rung] = hydro_wave_records(state, 8)
    wave_errors = [rms_errors(wave_records[rung], wave_records[7])[0] for rung in (4, 5, 6)]
    wave_differences = [rms_errors(wave_records[rung], wave_records[rung + 1])[0]
                        for rung in (4, 5, 6)]
    wave_ratios = [coarse / fine for coarse, fine in zip(wave_differences, wave_differences[1:])]
    require(all(3.2 < ratio < 5.0 for ratio in wave_ratios),
            f"undamped midpoint position differences did not show second order: {wave_ratios}")
    evidence["inviscid_pressure_order"] = {"reference_rung": 7,
        "position_errors_against_reference": wave_errors,
        "successive_position_differences": wave_differences,
        "refinement_ratios": wave_ratios, "observed_orders": [math.log2(ratio) for ratio in wave_ratios]}
    # The expanding main driver has its own rung-merging and drift integration.
    # A tight hydro CFL puts the cold gas one rung above the dark-matter floor.
    # The gravity work criterion N_n <= 3 N_(n+1) then promotes rung nine into ten;
    # this criterion is independent of the SPH octree rebuild fraction.
    mixed_options = {"run.scenario": "cosmology", "run.numberOfSteps": 64,
                     "cosmology.sigma8": 0, "run.darkMatterParticleCount": 64, "run.gasParticleCount": 64,
                     "initial.closePack": "false", "run.minRung": 9,
                     "hydro.cfl": 5e-6,
                     "run.maxIterations": 8, "checkpoint.intervalSeconds": 0,
                     "run.verbose": "true"}
    mixed = run("cosmology", mixed_options)
    mixed_directory = root / "cosmology"
    output = (mixed_directory / "run.log").read_text()
    require("Merging rung 9 into rung 10" in output,
            "cosmological hierarchy did not exercise the gravity rung merge")
    lifecycle = validate_events(output, "hierarchical_cosmology", "hierarchical")
    scales, minimum_rungs = [], []
    for iteration in range(1, 9):
        state = decode_checkpoint((mixed_directory / f"checkpoint.{iteration}" /
                                   "checkpoint.0.dat").read_bytes())
        require(state.types.count(0) == 64 and state.types.count(1) == 64,
                "mixed cosmological particle populations changed")
        scales.append(state.driver.a)
        minimum_rungs.append(state.driver.min_rung)
        base_timestep = state.driver.tau_max / state.header.steps
        check_midpoint_state(state)
    midpoint_events = validate_midpoints(output, "hierarchical_cosmology", base_timestep,
                                        mixed.driver.tau)
    require(scales[-1] > scales[0] and max(minimum_rungs) == 10,
            "cosmological case missed expansion or rung merging")
    run("cosmology_restart", dict(mixed_options, **{"run.maxIterations": 1, "checkpoint.readIndex": 7}),
        source_checkpoint=mixed_directory / "checkpoint.7")
    restart_errors = compare_restart(checkpoint_state(mixed_directory, 8, 1),
                                     checkpoint_state(root / "cosmology_restart", 8, 1))
    evidence["cosmology"] = {**lifecycle, "scale_range": [scales[0], scales[-1]],
                            "midpoints": midpoint_events, "minimum_rungs": minimum_rungs,
                            "restart_maximum_errors": restart_errors}
    evidence["boundary_thermal_scale"] = {}
    for integrator in (name for name in ("hierarchical", "traditional")
                       if name in available["integrator"]):
        label = "thermal_scale_" + integrator
        thermal_options = dict(mixed_options, **{"hydro.gamma": 4 / 3, "hydro.cfl": 0.1,
                                                "run.maxIterations": 2})
        current = run(label, thermal_options, integrator)
        previous = decode_checkpoint((root / label / "checkpoint.1" /
                                      "checkpoint.0.dat").read_bytes())
        require(set(previous.rungs) == {9} and set(current.rungs) == {9},
                "thermal-scale fixture did not retain uniform rung nine")
        check_midpoint_state(current)
        validate_midpoints((root / label / "run.log").read_text(), label,
                           current.driver.tau_max / current.header.steps,
                           current.driver.tau, 9)
        scale = previous.driver.a
        gamma = previous.header.physics[2]
        gas_mass = previous.header.physics[1]
        # Derived from cosmictiger/sph/entropy.hpp and the energy units:
        #   u = A ρ^(γ−1) a^(5−3γ)/(γ−1), E_thermal = Σ m u/a².
        # Symbols: γ -> gamma; A -> entropy; ρ -> density; m -> gas_mass;
        # a -> conversion_scale in u, scale in the boundary energy units.
        # No hydro event intervenes at this gravity boundary, so checkpoint
        # one's gas state determines checkpoint two's thermal diagnostic.
        def thermal_energy(conversion_scale):
            return math.fsum(gas_mass * entropy * density ** (gamma - 1) *
                             conversion_scale ** (5 - 3 * gamma) / (gamma - 1)
                             for entropy, density in zip(previous.entropies, previous.densities)) / scale ** 2
        expected = thermal_energy(scale)
        stale = thermal_energy(previous.thermal_scale_factor)
        relative_error = abs(current.driver.energies[0] - expected) / expected
        stale_relative_error = abs(stale - expected) / expected
        require(relative_error < 4e-7,
                f"{label}: boundary energy used a stale entropy conversion scale")
        require(stale_relative_error > 5e-5,
                f"{label}: fixture cannot distinguish the obsolete midpoint conversion")
        evidence["boundary_thermal_scale"][integrator] = {
            "relative_error": relative_error, "stale_relative_error": stale_relative_error}
    (root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("Midpoint SPH position convergence, energy, momentum, cosmological timing and restart passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
