#!/usr/bin/env python3
# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# Distributed under the Boost Software License, Version 1.0.

"""Inspect saved-glass pair separations, noise, and full paired particle motion."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmictiger_mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from analyze_glass_growth import growth, metadata_at, read_xyz, structure_factor, wavevectors, write_json
from analyze_linear_growth import Background, mode_key, read_csv, write_csv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadataFor(path):
    metadata = metadata_at(path / "linear_growth_metadata.json")
    for key, expected in dict(side=8, species=1, load_kind="file", softening_mode="fixed",
                              kernel_family="ferrers", kernel_degree=4, pressure_enabled=False).items():
        if metadata.get(key) != expected:
            raise ValueError(f"unexpected {key}: {metadata.get(key)!r}")
    support = float(metadata["baseline_support_dm"])
    if not math.isfinite(support) or support <= 0.:
        raise ValueError("invalid fixed support")
    if not math.isclose(support, float(metadata["configured_support_radius_box"]), rel_tol=2.e-7):
        raise ValueError("configured and measured fixed support differ")
    return metadata


def rms(vector):
    return float(np.sqrt(np.mean(np.sum(vector * vector, axis=1))))


def distances(points, spacing, support):
    tree = cKDTree(points, boxsize=1.)
    nearest, indices = tree.query(points, k=2)
    nearest = nearest[:, 1]
    smallest = int(np.argmin(nearest))
    closestNeighbor = int(indices[smallest, 1])
    if closestNeighbor == smallest:
        closestNeighbor = int(indices[smallest, 0])
    result = dict(minimum_separation_spacing=float(nearest[smallest] / spacing),
                  minimum_separation_support=float(nearest[smallest] / support),
                  closest_particle=smallest, closest_neighbor=closestNeighbor,
                  nearest_mean_spacing=float(nearest.mean() / spacing),
                  nearest_median_spacing=float(np.median(nearest) / spacing),
                  nearest_percentile_5_spacing=float(np.quantile(nearest, .05) / spacing))
    for name, radius in (("support", support), ("0p1_spacing", .1 * spacing), ("0p5_spacing", .5 * spacing)):
        pairs = tree.query_pairs(radius, output_type="ndarray")
        difference = points[pairs[:, 0]] - points[pairs[:, 1]]
        difference -= np.rint(difference)
        result["pairs_below_" + name] = int(np.count_nonzero(np.sum(difference * difference, axis=1) < radius**2))
    return result


def initial(args):
    metadata = metadataFor(args.input)
    reference = read_xyz(args.glass)
    count = metadata["side"]**3
    if len(reference) != count:
        raise ValueError("glass has the wrong particle count")
    values = np.full((count, 9), np.nan)
    seen = set()
    for row in read_csv(args.input / "linear_growth_particles_initial.csv"):
        particle = int(row["particle"])
        if particle not in range(count) or int(row["species"]) != 0 or particle in seen:
            raise ValueError("invalid or duplicate initial particle")
        seen.add(particle)
        fields = [float(row[prefix + axis]) for prefix in ("reference_", "", "acceleration_") for axis in "xyz"]
        if not all(math.isfinite(value) for value in fields):
            raise ValueError("nonfinite initial particle")
        values[particle] = fields
    if len(seen) != count or not np.array_equal(values[:, :3], reference):
        raise ValueError("missing initial particle or different saved glass")
    if not np.allclose(values[:, 3:6], reference, rtol=0., atol=2.**-32):
        raise ValueError("initial positions do not match saved glass")
    force = values[:, 6:9]
    if not math.isclose(rms(force), metadata["baseline_force_rms"], rel_tol=1.e-12, abs_tol=1.e-15):
        raise ValueError("initial particle forces disagree with metadata")
    modes = wavevectors(4)
    factors = structure_factor(values[:, 3:6], modes)
    summary = dict(metadata=metadata, glass_sha256=digest(args.glass), particles=count,
                   support_over_spacing=metadata["baseline_support_dm"] * metadata["side"],
                   force_rms=rms(force), force_max=float(np.linalg.norm(force, axis=1).max()),
                   net_acceleration_norm=float(np.linalg.norm(force.sum(axis=0))),
                   **distances(values[:, 3:6], 1. / metadata["side"], metadata["baseline_support_dm"]))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "initial_summary.json", summary)
    write_csv(args.output / "initial_structure_factor.csv", [dict(mode_x=mode[0], mode_y=mode[1], mode_z=mode[2],
              mode_norm_squared=sum(value * value for value in mode), structure_factor=float(value))
              for mode, value in zip(modes, factors)])
    print(json.dumps(dict(initial=str(args.input), force_rms=summary["force_rms"],
                         minimum_separation_spacing=summary["minimum_separation_spacing"])))


def particleStates(path, requests, reference):
    count = len(reference)
    names = ("positive", "negative", "zero")
    states = {(run, step, name): np.full((count, 12), np.nan)
              for run, request in enumerate(requests) for step in range(int(request["steps"]) + 1) for name in names}
    seen = set()
    columns = [prefix + axis for prefix in ("", "u_", "realized_u_", "v_") for axis in "xyz"]
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            run, step, particle = (int(row[key]) for key in ("run_index", "step", "particle"))
            key = (run, step, row["trajectory"])
            fullKey = key + (particle,)
            if key not in states or particle not in range(count) or int(row["species"]) != 0 or fullKey in seen:
                raise ValueError("unexpected, duplicate, or invalid particle sample")
            seen.add(fullKey)
            request = requests[run]
            expectedScale = float(request["a_initial"]) * math.exp(step * math.log(
                float(request["a_final"]) / float(request["a_initial"])) / int(request["steps"]))
            if row["label"] != request["label"] or not math.isclose(float(row["a"]), expectedScale, rel_tol=2.e-13, abs_tol=1.e-15):
                raise ValueError("particle sample label or scale factor disagrees with request")
            fields = np.asarray([float(row[column]) for column in columns])
            if not np.all(np.isfinite(fields)) or np.any(fields[:3] < 0.) or np.any(fields[:3] >= 1.):
                raise ValueError("nonfinite fields or invalid periodic position")
            states[key][particle] = fields
    if len(seen) != count * len(states):
        raise ValueError("missing particle samples")
    for key, values in states.items():
        positionError = values[:, :3] - np.mod(reference + values[:, 3:6], 1.)
        positionError -= np.rint(positionError)
        realized = values[:, :3] - reference
        realized -= np.rint(realized)
        if np.max(np.abs(positionError)) > 2.**-32 + 2.e-15 or not np.allclose(realized, values[:, 6:9], rtol=0., atol=2.e-15):
            raise ValueError("particle identity, realized displacement, or position mismatch")
        if key[1] == 0:
            request = requests[key[0]]
            sine = np.sin(2. * np.pi * reference @ np.asarray(mode_key(request)))
            sign = {"positive": 1., "negative": -1., "zero": 0.}[key[2]]
            spacing = count**(-1. / 3.)
            expectedU = sign * sine[:, None] * np.asarray([float(request[f"u_dm_{axis}"]) for axis in "xyz"]) * spacing
            expectedV = sign * sine[:, None] * np.asarray([float(request[f"v_dm_{axis}"]) for axis in "xyz"]) * spacing
            if not np.allclose(values[:, 3:6], expectedU, rtol=2.e-12, atol=1.e-15) or not np.allclose(values[:, 9:12], expectedV, rtol=2.e-12, atol=1.e-15):
                raise ValueError("initial full particle field differs from requested sinusoid")
    return states


def evolution(args):
    metadata = metadataFor(args.input)
    if not metadata.get("paired_evolution"):
        raise ValueError("paired trajectories are required")
    if not math.isclose(metadata["baseline_support_dm"] * metadata["side"], .025, rel_tol=2.e-7):
        raise ValueError("evolution diagnostic requires fixed support Delta/40")
    reference = read_xyz(args.glass)
    if len(reference) != metadata["side"]**3:
        raise ValueError("wrong saved glass particle count")
    requests = read_csv(args.requests)
    if not requests:
        raise ValueError("no evolution requests")
    states = particleStates(args.input / "linear_growth_particles.csv", requests, reference)
    args.output.mkdir(parents=True, exist_ok=True)
    growth(argparse.Namespace(metadata=args.input / "linear_growth_metadata.json", requests=args.requests,
           evolution=args.input / "linear_growth_evolution.csv", baseline=args.input / "linear_growth_baseline_evolution.csv",
           density=args.input / "linear_growth_density.csv", output=args.output / "growth"))
    projected = {(int(row["run_index"]), int(row["step"]), int(row["axis"])): row
                 for row in read_csv(args.input / "linear_growth_evolution.csv")}
    densityRows = {(int(row["run_index"]), int(row["step"])): row
                    for row in read_csv(args.input / "linear_growth_density.csv")}
    background = Background(metadata)
    modes = wavevectors(4)
    initialStructure = structure_factor(reference, modes)
    spacing, support = 1. / metadata["side"], metadata["baseline_support_dm"]
    pairs, fields, noise, summaries = [], [], [], []
    for run, request in enumerate(requests):
        mode = mode_key(request)
        direction = np.asarray(mode, dtype=float) / np.linalg.norm(mode)
        phase = 2. * np.pi * reference @ np.asarray(mode)
        basis = np.column_stack((np.ones(len(reference)), np.sin(phase), np.cos(phase)))
        initialU = .5 * (states[(run, 0, "positive")][:, 3:6] - states[(run, 0, "negative")][:, 3:6])
        start, end = float(request["a_initial"]), float(request["a_final"])
        solution = background.growing_solution(1., start, end)
        firstGrowth = solution.sol(math.log(start))[0]
        rowsForRun = []
        for step in range(int(request["steps"]) + 1):
            scale = start * math.exp(step * math.log(end / start) / int(request["steps"]))
            continuum = solution.sol(math.log(scale))
            growthFactor = float(continuum[0] / firstGrowth)
            rate = float(continuum[1] / continuum[0])
            plus, minus, zero = (states[(run, step, name)] for name in ("positive", "negative", "zero"))
            for name, values in (("positive", plus), ("negative", minus), ("zero", zero)):
                pairs.append(dict(run_index=run, mode_x=mode[0], mode_y=mode[1], mode_z=mode[2], step=step, a=scale,
                                  trajectory=name, **distances(values[:, :3], spacing, support)))
            differenceU = .5 * (plus[:, 3:6] - minus[:, 3:6])
            differenceV = .5 * (plus[:, 9:12] - minus[:, 9:12])
            evenU = .5 * (plus[:, 3:6] + minus[:, 3:6]) - zero[:, 3:6]
            expectedU = initialU * growthFactor
            expectedV = expectedU * rate
            coefficients = np.linalg.lstsq(basis, differenceU, rcond=None)[0]
            velocityCoefficients = np.linalg.lstsq(basis, differenceV, rcond=None)[0]
            residual = differenceU - basis @ coefficients
            for axis in range(3):
                stored = projected[(run, step, axis)]
                if not np.allclose([coefficients[1, axis] / spacing, coefficients[2, axis] / spacing,
                                    velocityCoefficients[1, axis] / spacing, velocityCoefficients[2, axis] / spacing],
                                   [float(stored[key]) for key in ("u_sin", "u_cos", "v_sin", "v_cos")], rtol=2.e-10, atol=2.e-13):
                    raise ValueError("full particle field does not reproduce stored projection")
            actualDensity = .5 * (np.exp(-2j * np.pi * (plus[:, :3] @ np.asarray(mode))).mean() -
                                     np.exp(-2j * np.pi * (minus[:, :3] @ np.asarray(mode))).mean())
            storedDensity = densityRows[(run, step)]
            if abs(actualDensity - complex(float(storedDensity["density_diff_real"]), float(storedDensity["density_diff_imag"]))) > 2.e-15:
                raise ValueError("full particle positions do not reproduce stored Eulerian density")
            originalAmplitude = np.linalg.norm([float(request[f"u_dm_{axis}"]) for axis in "xyz"]) * spacing * growthFactor
            row = dict(run_index=run, mode_x=mode[0], mode_y=mode[1], mode_z=mode[2], step=step, a=scale,
                       differential_displacement_rms_ratio=rms(differenceU) / rms(expectedU),
                       differential_velocity_rms_ratio=rms(differenceV) / rms(expectedV),
                       full_displacement_error_relative_rms=rms(differenceU - expectedU) / rms(expectedU),
                       full_velocity_error_relative_rms=rms(differenceV - expectedV) / rms(expectedV),
                       projected_displacement_ratio=float(coefficients[1] @ direction / originalAmplitude),
                       projected_velocity_ratio=float(velocityCoefficients[1] @ direction / (originalAmplitude * rate)),
                       nonmode_displacement_relative_rms=rms(residual) / originalAmplitude,
                       baseline_displacement_rms_spacing=rms(zero[:, 3:6]) / spacing,
                       baseline_displacement_max_spacing=float(np.linalg.norm(zero[:, 3:6], axis=1).max() / spacing),
                       baseline_velocity_rms_spacing=rms(zero[:, 9:12]) / spacing,
                       even_displacement_relative_rms=rms(evenU) / originalAmplitude)
            fields.append(row)
            rowsForRun.append(row)
            structure = structure_factor(zero[:, :3], modes)
            for wave, initialValue, value in zip(modes, initialStructure, structure):
                noise.append(dict(run_index=run, step=step, a=scale, mode_x=wave[0], mode_y=wave[1], mode_z=wave[2],
                                  mode_norm_squared=sum(component**2 for component in wave),
                                  initial_structure_factor=float(initialValue), structure_factor=float(value),
                                  structure_factor_ratio=float(value / initialValue)))
        runPairs = [row for row in pairs if row["run_index"] == run]
        final = dict(rowsForRun[-1])
        final.update(minimum_separation_spacing=min(row["minimum_separation_spacing"] for row in runPairs),
                     minimum_separation_support=min(row["minimum_separation_support"] for row in runPairs),
                     max_pairs_below_support=max(row["pairs_below_support"] for row in runPairs),
                     max_pairs_below_0p1_spacing=max(row["pairs_below_0p1_spacing"] for row in runPairs),
                     max_pairs_below_0p5_spacing=max(row["pairs_below_0p5_spacing"] for row in runPairs))
        finalNoise = [row for row in noise if row["run_index"] == run and row["step"] == int(request["steps"])]
        for maximum in (2, 4):
            subset = [row for row in finalNoise if row["mode_norm_squared"] <= maximum**2]
            first = np.mean([row["initial_structure_factor"] for row in subset])
            last = np.mean([row["structure_factor"] for row in subset])
            final[f"noise_mean_structure_norm_le_{maximum}"] = float(last)
            final[f"noise_mean_structure_ratio_norm_le_{maximum}"] = float(last / first)
        summaries.append(final)
    write_csv(args.output / "pair_tracks.csv", pairs)
    write_csv(args.output / "field_tracks.csv", fields)
    write_csv(args.output / "noise_tracks.csv", noise)
    write_csv(args.output / "particle_summary.csv", summaries)
    write_json(args.output / "particle_summary.json", dict(metadata=metadata, glass_sha256=digest(args.glass),
               requests_sha256=digest(args.requests), noise_modes=modes, runs=summaries,
               scope="Actual full particle positions and paired fields; pressure-free short evolution; not production block timestep driver"))
    plot(fields, pairs, noise, args.output)
    print(json.dumps(dict(analyzed=str(args.input), runs=len(summaries),
                         minimum_separation_spacing=min(row["minimum_separation_spacing"] for row in summaries))))


def plot(fields, pairs, noise, output):
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for run in sorted({row["run_index"] for row in fields}):
        rows = [row for row in fields if row["run_index"] == run]
        time = [row["a"] for row in rows]
        nearest = [min(row["minimum_separation_spacing"] for row in pairs if row["run_index"] == run and row["step"] == step)
                   for step in range(len(rows))]
        axes[0].plot(time, nearest, "o-", markersize=3, label=f"n={rows[0]['mode_x']}")
        axes[1].plot(time, [row["baseline_displacement_rms_spacing"] for row in rows], "o-", markersize=3)
        axes[2].plot(time, [row["nonmode_displacement_relative_rms"] for row in rows], "o-", markersize=3)
    for axis in axes:
        axis.set(xlabel="Scale factor a", xscale="log")
        axis.grid(alpha=.2)
    axes[0].set_ylabel(r"Closest pair / $\Delta$ (all trajectories)")
    axes[1].set_ylabel(r"Zero-load drift RMS / $\Delta$")
    axes[2].set_ylabel("Differential non-mode motion / fluid target")
    axes[0].legend()
    figure.suptitle(r"Fixed support $H=\Delta/40$: short glass evolution")
    figure.savefig(output / "particle_diagnostics.png", dpi=170)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    for maximum in (2, 4):
        selected = [row for row in noise if row["run_index"] == 0 and row["mode_norm_squared"] <= maximum**2]
        times = sorted({row["a"] for row in selected})
        values = [np.mean([row["structure_factor"] for row in selected if row["a"] == scale]) for scale in times]
        axis.plot(times, values, "o-", label=rf"$0<|n|\leq {maximum}$")
    axis.set(xlabel="Scale factor a", ylabel="Mean zero-load structure factor S(k)", xscale="log", yscale="log")
    axis.grid(alpha=.2)
    axis.legend()
    figure.savefig(output / "baseline_noise.png", dpi=170)
    plt.close(figure)


def compare(args):
    summaries = {}
    for name in ("evolution_glass", "control_16", "control_half"):
        path = args.archive / "analysis" / name
        summaries[name] = dict(particles=json.loads((path / "particle_summary.json").read_text()),
                               growth=json.loads((path / "growth/growth_summary.json").read_text()))
    primary = summaries["evolution_glass"]
    for name, case in summaries.items():
        if case["particles"]["glass_sha256"] != primary["particles"]["glass_sha256"]:
            raise ValueError("evolution controls use different glasses")
        for key in ("side", "species", "backend", "theta", "softening_mode", "kernel_family", "kernel_degree",
                    "baseline_support_dm", "omega_m", "omega_r", "omega_k", "omega_lambda"):
            if case["particles"]["metadata"][key] != primary["particles"]["metadata"][key]:
                raise ValueError(f"mismatched evolution control setting {key}")
        expectedModes = [(1, 0, 0), (2, 0, 0), (3, 0, 0)] if name == "evolution_glass" else [(3, 0, 0)]
        if [mode_key(row) for row in case["growth"]["runs"]] != expectedModes:
            raise ValueError("incorrect evolution control modes")
        for row in case["growth"]["runs"]:
            expectedSteps = 16 if name == "control_16" else 8
            expectedAmplitude = .0005 if name == "control_half" else .001
            if (row["steps"] != expectedSteps or row["initial_amplitude_spacing"] != expectedAmplitude or
                    not math.isclose(row["a_initial"], .02) or not math.isclose(row["a"], .1)):
                raise ValueError("incorrect evolution control interval, steps, or amplitude")
    reference = summaries["evolution_glass"]["growth"]["runs"][-1]
    particleReference = summaries["evolution_glass"]["particles"]["runs"][-1]
    controls = {}
    for name in ("control_16", "control_half"):
        row = summaries[name]["growth"]["runs"][0]
        particleRow = summaries[name]["particles"]["runs"][0]
        controls[name] = dict(density_relative_change=row["density_parallel_ratio"] / reference["density_parallel_ratio"] - 1.,
                             phase_absolute_change=row["density_phase_ratio"] - reference["density_phase_ratio"],
                             projected_velocity_relative_change=particleRow["projected_velocity_ratio"] /
                             particleReference["projected_velocity_ratio"] - 1.,
                             nonmode_displacement_absolute_change=particleRow["nonmode_displacement_relative_rms"] -
                             particleReference["nonmode_displacement_relative_rms"],
                             baseline_drift_absolute_change_spacing=particleRow["baseline_displacement_rms_spacing"] -
                             particleReference["baseline_displacement_rms_spacing"],
                             minimum_separation_absolute_change_spacing=particleRow["minimum_separation_spacing"] -
                             particleReference["minimum_separation_spacing"],
                             noise_power_relative_change_norm_le_4=particleRow["noise_mean_structure_norm_le_4"] /
                             particleReference["noise_mean_structure_norm_le_4"] - 1.)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "comparison.json", dict(cases=summaries, controls=controls))
    print(json.dumps(controls))


def selfCheck(args):
    points = np.asarray([[.01, .2, .2], [.99, .2, .2], [.5, .6, .6]])
    result = distances(points, 1., .025)
    if not math.isclose(result["minimum_separation_spacing"], .02, abs_tol=1.e-14) or result["pairs_below_support"] != 1:
        raise AssertionError("periodic boundary pair was not detected")
    modes = wavevectors(4)
    if not np.allclose(structure_factor(points, modes), structure_factor(np.mod(points + [.2, .3, .4], 1.), modes), rtol=2.e-12, atol=2.e-13):
        raise AssertionError("structure factor is not translation invariant")
    reference = np.asarray([[x, y, z] for x in (.125, .625) for y in (.125, .625) for z in (.125, .625)])
    request = dict(label="synthetic", mode_x=1, mode_y=0, mode_z=0, a_initial=.02, a_final=.04, steps=1,
                   u_dm_x=.001, u_dm_y=0., u_dm_z=0., v_dm_x=.00099, v_dm_y=0., v_dm_z=0.)
    rows = []
    for step in range(2):
        for name, sign in (("positive", 1.), ("negative", -1.), ("zero", 0.)):
            for particle, point in enumerate(reference):
                displacement = np.asarray([sign * .0005 * math.sin(2. * np.pi * point[0]) * (1. + .1 * step), 0., 0.])
                position = np.floor(np.mod(point + displacement, 1.) * 2.**32) / 2.**32
                realized = position - point
                realized -= np.rint(realized)
                row = dict(label="synthetic", run_index=0, step=step, a=.02 * 2.**step,
                           trajectory=name, particle=particle, species=0)
                for prefix, values in (("", position), ("u_", displacement), ("realized_u_", realized), ("v_", .99 * displacement)):
                    row.update({prefix + axis: float(value) for axis, value in zip("xyz", values)})
                rows.append(row)
    with tempfile.TemporaryDirectory(prefix="cosmictiger_fixed_particle_check_") as temporary:
        path = Path(temporary) / "particles.csv"
        write_csv(path, rows)
        particleStates(path, [request], reference)
        for label, changed in (("missing particle", rows[:-1]),
                               ("wrong fixed32 position", [dict(rows[0], x=rows[0]["x"] + .001)] + rows[1:]),
                               ("wrong initial velocity", [dict(rows[0], v_x=rows[0]["v_x"] * 2.)] + rows[1:])):
            write_csv(path, changed)
            try:
                particleStates(path, [request], reference)
            except ValueError:
                pass
            else:
                raise AssertionError("accepted " + label)
    print("Six checks passed: periodic pair count; translation-invariant S(k); full synthetic particle load; rejection of missing particles, wrong coordinates, and wrong initial velocities.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("initial")
    command.add_argument("input", type=Path)
    command.add_argument("glass", type=Path)
    command.add_argument("output", type=Path)
    command.set_defaults(function=initial)
    command = commands.add_parser("evolution")
    command.add_argument("input", type=Path)
    command.add_argument("requests", type=Path)
    command.add_argument("glass", type=Path)
    command.add_argument("output", type=Path)
    command.set_defaults(function=evolution)
    command = commands.add_parser("compare")
    command.add_argument("archive", type=Path)
    command.add_argument("output", type=Path)
    command.set_defaults(function=compare)
    command = commands.add_parser("self-check")
    command.set_defaults(function=selfCheck)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
