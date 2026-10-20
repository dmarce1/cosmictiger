#!/usr/bin/env python3
# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# Distributed under the Boost Software License, Version 1.0.

"""Prepare and analyze a paired glass/lattice linear-growth experiment.

A Fourier mode is not an eigenmode of an individual glass. The primary growth
observable is the central difference of the actual Eulerian density Fourier
coefficient, not an eigenvalue of a small projected force-response matrix.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import struct

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmictiger_mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from analyze_linear_growth import (Background, mode_key, normalize_metadata,
                                   read_csv, response_matrices, species_index, write_csv)


def metadata_at(path):
    with path.open() as source:
        return normalize_metadata(json.load(source))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_glass(path):
    content = path.read_bytes()
    if len(content) < 4:
        raise ValueError("glass header is missing")
    side, = struct.unpack("=i", content[:4])
    if side <= 0 or len(content) != 4 + 12 * side**3:
        raise ValueError("glass file size does not match its native int dimension")
    points = np.frombuffer(content, dtype="=u4", offset=4).reshape(3, side**3).T
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("glass has duplicate fixed32 positions")
    return side, points.astype(np.float64) / 2.**32


def read_xyz(path):
    points = np.loadtxt(path)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("load must be a finite N by 3 coordinate table")
    if np.any(points < 0.) or np.any(points >= 1.):
        raise ValueError("load coordinates must lie in [0,1)")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("load has duplicate positions")
    return points


def convert(args):
    side, points = read_glass(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, points, fmt="%.17g")
    if not np.array_equal(read_xyz(args.output), points):
        raise RuntimeError("coordinate text conversion lost fixed32 precision")
    print(json.dumps(dict(side=side, particles=len(points),
                          source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                          output=str(args.output))))


def wavevectors(maximum):
    # Retain one member of each conjugate pair. A shell does not count +/- twice.
    return [wave for wave in itertools.product(range(-maximum, maximum + 1), repeat=3)
            if 0 < sum(x*x for x in wave) <= maximum**2 and
            next(x for x in wave if x != 0) > 0]


def structure_factor(points, modes):
    phases = 2. * math.pi * np.asarray(modes) @ points.T
    return np.abs(np.exp(-1j * phases).sum(axis=1))**2 / len(points)


def quality(args):
    points = read_xyz(args.input)
    side = round(len(points)**(1. / 3.))
    if side**3 != len(points):
        raise ValueError("quality comparison currently requires N cubed particles")
    args.output.mkdir(parents=True, exist_ok=True)
    random = np.random.default_rng(args.seed).random(points.shape)
    lattice = np.asarray(list(itertools.product(range(side), repeat=3)), dtype=float) / side
    modes = wavevectors(args.max_mode)
    values = {"glass": structure_factor(points, modes),
              "poisson": structure_factor(random, modes),
              "lattice": structure_factor(lattice, modes)}
    norms = np.asarray([sum(x*x for x in mode) for mode in modes])
    shells = []
    for norm in sorted(set(norms)):
        mask = norms == norm
        row = dict(mode_norm_squared=int(norm), k_over_nyquist=2. * math.sqrt(norm) / side,
                   independent_modes=int(mask.sum()))
        for name, array in values.items():
            row.update({f"{name}_mean": float(array[mask].mean()),
                        f"{name}_minimum": float(array[mask].min()),
                        f"{name}_maximum": float(array[mask].max())})
        shells.append(row)
    nearest = {name: cKDTree(load, boxsize=1.).query(load, k=2)[0][:, 1] * side
               for name, load in (("glass", points), ("poisson", random), ("lattice", lattice))}
    summary = dict(particles=len(points), side=side, poisson_seed=args.seed,
                   coordinate_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                   poisson_structure_factor_expectation=1.,
                   low_mode_cutoff=args.low_mode, low_mode_independent_modes=int((norms <= args.low_mode**2).sum()))
    for name, distances in nearest.items():
        summary[name] = dict(nearest_spacing_minimum=float(distances.min()),
                             nearest_spacing_mean=float(distances.mean()),
                             nearest_spacing_median=float(np.median(distances)),
                             low_mode_mean_structure_factor=float(values[name][norms <= args.low_mode**2].mean()))
    write_csv(args.output / "structure_factor_shells.csv", shells)
    write_csv(args.output / "structure_factor_modes.csv", [dict(mode_x=mode[0], mode_y=mode[1], mode_z=mode[2],
              **{name: float(array[index]) for name, array in values.items()}) for index, mode in enumerate(modes)])
    write_json(args.output / "glass_quality.json", summary)
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    selected = points[:, 2] < .25
    axes[0].scatter(points[selected, 0], points[selected, 1], s=10, c=points[selected, 2], cmap="viridis")
    axes[0].set(xlabel="x / L", ylabel="y / L", title="Glass slice: 0 ≤ z/L < 0.25", xlim=(0, 1), ylim=(0, 1), aspect="equal")
    for name, color in (("glass", "C0"), ("poisson", "C1")):
        axes[1].hist(nearest[name], bins=np.linspace(0., 1.5, 31), density=True,
                     alpha=.55, label=name, color=color)
        axes[2].plot([row["k_over_nyquist"] for row in shells],
                     [row[f"{name}_mean"] for row in shells], "o-", label=name, color=color)
    axes[1].axvline(1., color=".3", linestyle=":", label="cubic lattice")
    axes[1].set(xlabel="Nearest-particle distance / Δ", ylabel="Probability density")
    axes[1].legend()
    axes[2].axhline(1., color=".4", linestyle=":", label="Poisson expectation")
    axes[2].set(xlabel="k / kNy", ylabel="Shell mean S(k)", yscale="log")
    axes[2].legend()
    figure.suptitle("Particle-load quality before sinusoidal perturbations")
    figure.savefig(args.output / "glass_quality.png", dpi=170)
    plt.close(figure)
    print(json.dumps(summary))


def response(args):
    metadata = metadata_at(args.metadata)
    if metadata["species_count"] != 1:
        raise ValueError("glass response comparison currently requires one particle class")
    rows = read_csv(args.input)
    matrices = response_matrices(rows, 1)
    if not matrices:
        raise ValueError("response file has no matrices")
    summaries = []
    for (mode, amplitude), matrix in sorted(matrices.items()):
        direction = np.asarray(mode, dtype=float)
        direction /= np.linalg.norm(direction)
        vector = matrix @ direction
        longitudinal = direction @ vector
        residual = np.zeros((3, 3))
        entries = [row for row in rows if mode_key(row) == mode and float(row["amplitude_spacing"]) == amplitude]
        for row in entries:
            residual[int(row["output_axis"]), int(row["input_axis"])] = float(row["response_residual_rms"])
        if not np.all(np.isfinite(residual)) or np.any(residual < 0.):
            raise ValueError("nonfinite or negative response residual")
        # No cross-component residual covariance is recorded. Triangle inequality
        # gives a bound for diagonal input; it is exact for axis-aligned input.
        residual_bound = float(np.abs(direction) @ np.linalg.norm(residual, axis=0))
        other = [(other_amplitude, other_matrix) for (other_mode, other_amplitude), other_matrix in matrices.items()
                 if other_mode == mode and other_amplitude != amplitude]
        change = None if not other else max(float(np.linalg.norm(matrix - other_matrix) /
                                                  max(np.linalg.norm(matrix), 1.e-30)) for _, other_matrix in other)
        summaries.append(dict(mode_x=mode[0], mode_y=mode[1], mode_z=mode[2],
                              k_over_nyquist=2. * np.linalg.norm(mode) / metadata["n_per_dimension"],
                              amplitude_spacing=amplitude, longitudinal_sine=float(longitudinal.real),
                              longitudinal_cosine=float(longitudinal.imag),
                              transverse_response_norm=float(np.linalg.norm(vector - direction * longitudinal)),
                              outside_mode_response_rms_bound=residual_bound,
                              outside_mode_bound_is_exact=bool(np.count_nonzero(direction) == 1),
                              amplitude_matrix_relative_change=change))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "response_summary.csv", summaries)
    write_json(args.output / "response_summary.json", dict(metadata=metadata, modes=summaries))
    print(json.dumps(summaries))


def parse_modes(text):
    result = [tuple(int(component) for component in mode.split(",")) for mode in text.split(";")]
    if any(len(mode) != 3 or not any(mode) for mode in result) or len(set(result)) != len(result):
        raise ValueError("mode list must contain unique nonzero integer triples")
    return result


def requests(args):
    metadata = metadata_at(args.metadata)
    if not (0. < args.a_initial < args.a_final and args.steps > 0 and 0. < args.amplitude < .1):
        raise ValueError("invalid evolution amplitude or interval")
    solution = Background(metadata).growing_solution(1., args.a_initial, args.a_final)
    initial = solution.sol(math.log(args.a_initial))
    rate = initial[1] / initial[0]
    rows = []
    for mode in parse_modes(args.modes):
        if max(abs(value) for value in mode) >= metadata["n_per_dimension"] // 2:
            raise ValueError("mode component reaches or exceeds particle Nyquist")
        direction = np.asarray(mode, dtype=float) / np.linalg.norm(mode)
        row = dict(label=args.label, mode_x=mode[0], mode_y=mode[1], mode_z=mode[2],
                   a_initial=args.a_initial, a_final=args.a_final, steps=args.steps)
        for species, name in enumerate(("dm", "gas")):
            for field, factor in (("u", 1.), ("v", rate)):
                for axis, component in zip("xyz", direction):
                    row[f"{field}_{name}_{axis}"] = float(args.amplitude * factor * component) if species < metadata["species_count"] else 0.
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)
    print(json.dumps(dict(requests=len(rows), continuum_initial_growth_rate=float(rate), output=str(args.output))))


def indexed_samples(path, requests_, species_count, axis_rows, fields):
    samples = {}
    for row in read_csv(path):
        run, step, species = (int(row[key]) for key in ("run_index", "step", "species"))
        axis = int(row["axis"]) if axis_rows else 0
        if run not in range(len(requests_)) or species not in range(species_count) or axis not in range(3 if axis_rows else 1):
            raise ValueError(f"invalid sample indices in {path}")
        request = requests_[run]
        steps = int(request["steps"])
        if step not in range(steps + 1) or row["label"] != request["label"]:
            raise ValueError(f"unexpected step or mismatched request label in {path}")
        initial, final = float(request["a_initial"]), float(request["a_final"])
        expected_a = initial * math.exp(step * math.log(final / initial) / steps)
        if not math.isclose(float(row["a"]), expected_a, rel_tol=2.e-13, abs_tol=1.e-15):
            raise ValueError(f"incorrect scale factor in {path}")
        if not all(math.isfinite(float(row[key])) for key in fields):
            raise ValueError(f"nonfinite sample in {path}")
        key = (run, step, species, axis)
        if key in samples:
            raise ValueError(f"duplicate sample in {path}: {key}")
        samples[key] = row
    expected = {(run, step, species, axis)
                for run, request in enumerate(requests_)
                for step in range(int(request["steps"]) + 1)
                for species in range(species_count)
                for axis in range(3 if axis_rows else 1)}
    if samples.keys() != expected:
        raise ValueError(f"missing or unexpected rows in {path}")
    return samples


def growth(args):
    metadata = metadata_at(args.metadata)
    if not metadata.get("paired_evolution", False):
        raise ValueError("growth analysis requires paired positive/negative/zero trajectories")
    requested = read_csv(args.requests)
    if not requested:
        raise ValueError("no evolution requests")
    species_count = metadata["species_count"]
    fractions = np.asarray(metadata["mass_fractions"])
    if len(fractions) != species_count or np.any(fractions <= 0.) or not np.isclose(fractions.sum(), 1.):
        raise ValueError("invalid class mass fractions")
    fields = ("u_sin", "u_cos", "v_sin", "v_cos", "residual_rms")
    displacement = indexed_samples(args.evolution, requested, species_count, True, fields)
    baseline = indexed_samples(args.baseline, requested, species_count, True,
                               fields + ("u_rms", "v_rms", "even_u_rms", "even_v_rms"))
    density = indexed_samples(args.density, requested, species_count, False,
                              ("density_diff_real", "density_diff_imag", "baseline_real", "baseline_imag", "even_real", "even_imag"))
    args.output.mkdir(parents=True, exist_ok=True)
    background = Background(metadata)
    traces, summaries = [], []
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for run, request in enumerate(requested):
        mode = mode_key(request)
        direction = np.asarray(mode, dtype=float) / np.linalg.norm(mode)
        initial_a, final_a = float(request["a_initial"]), float(request["a_final"])
        solution = background.growing_solution(1., initial_a, final_a)
        initial_growth = solution.sol(math.log(initial_a))[0]
        initial_u = np.asarray([[float(request[f"u_{name}_{axis}"]) for axis in "xyz"] for name in ("dm", "gas")[:species_count]])
        initial_v = np.asarray([[float(request[f"v_{name}_{axis}"]) for axis in "xyz"] for name in ("dm", "gas")[:species_count]])
        initial_longitudinal = float(fractions @ initial_u @ direction)
        if initial_longitudinal == 0.:
            raise ValueError("growth ratio requires nonzero initial longitudinal displacement")
        initial_density = sum(fractions[species] * complex(float(density[(run, 0, species, 0)]["density_diff_real"]),
                             float(density[(run, 0, species, 0)]["density_diff_imag"])) for species in range(species_count))
        if abs(initial_density) < 1.e-15:
            raise ValueError("initial paired density signal vanishes")
        run_rows = []
        for step in range(int(request["steps"]) + 1):
            a = float(density[(run, step, 0, 0)]["a"])
            continuum = float(solution.sol(math.log(a))[0] / initial_growth)
            u, v, cos = (np.zeros((species_count, 3)) for _ in range(3))
            residual_squared = baseline_squared = even_squared = 0.
            differential_density = baseline_density = even_density = 0j
            for species in range(species_count):
                for axis in range(3):
                    sample = displacement[(run, step, species, axis)]
                    zero = baseline[(run, step, species, axis)]
                    u[species, axis], v[species, axis] = float(sample["u_sin"]), float(sample["v_sin"])
                    cos[species, axis] = float(sample["u_cos"])
                    residual_squared += fractions[species] * float(sample["residual_rms"])**2
                    baseline_squared += fractions[species] * float(zero["u_rms"])**2
                    even_squared += fractions[species] * float(zero["even_u_rms"])**2
                sample = density[(run, step, species, 0)]
                for real, imag, destination in (("density_diff_real", "density_diff_imag", "difference"),
                                                ("baseline_real", "baseline_imag", "baseline"),
                                                ("even_real", "even_imag", "even")):
                    value = fractions[species] * complex(float(sample[real]), float(sample[imag]))
                    if destination == "difference":
                        differential_density += value
                    elif destination == "baseline":
                        baseline_density += value
                    else:
                        even_density += value
            if step == 0 and (not np.allclose(u, initial_u, rtol=1.e-10, atol=1.e-12) or
                              not np.allclose(v, initial_v, rtol=1.e-10, atol=1.e-12)):
                raise ValueError(f"initial projected state does not match request {run}")
            ratio = differential_density / (initial_density * continuum)
            target_u = abs(initial_longitudinal) * continuum
            row = dict(run_index=run, label=request["label"], mode_x=mode[0], mode_y=mode[1], mode_z=mode[2],
                       step=step, a=a, continuum_growth=continuum,
                       density_parallel_ratio=float(ratio.real), density_phase_ratio=float(ratio.imag),
                       density_amplitude_ratio=float(abs(ratio)),
                       baseline_density_amplitude=float(abs(baseline_density)),
                       even_density_relative_to_signal=float(abs(even_density) / (abs(initial_density) * continuum)),
                       displacement_parallel_ratio=float(fractions @ u @ direction / (initial_longitudinal * continuum)),
                       displacement_cosine_relative=float(np.linalg.norm(fractions @ cos) / target_u),
                       nonmode_displacement_relative_rms=math.sqrt(residual_squared) / target_u,
                       baseline_displacement_rms_spacing=math.sqrt(baseline_squared),
                       even_displacement_relative_rms=math.sqrt(even_squared) / target_u)
            traces.append(row)
            run_rows.append(row)
        final = dict(run_rows[-1])
        final.update(initial_density_amplitude=float(abs(initial_density)), initial_amplitude_spacing=abs(initial_longitudinal),
                     steps=int(request["steps"]), a_initial=initial_a,
                     maximum_even_density_relative=max(row["even_density_relative_to_signal"] for row in run_rows),
                     maximum_nonmode_displacement_relative_rms=max(row["nonmode_displacement_relative_rms"] for row in run_rows))
        summaries.append(final)
        label = f"{request['label']} k={mode}"
        for axis, key in zip(axes, ("density_parallel_ratio", "density_phase_ratio", "nonmode_displacement_relative_rms")):
            axis.plot([row["a"] for row in run_rows], [row[key] for row in run_rows], "o-", markersize=3, label=label)
    for axis in axes:
        axis.set(xlabel="Scale factor a", xscale="log")
        axis.grid(alpha=.2)
    axes[0].axhline(1., color=".4", linestyle=":")
    axes[0].set_ylabel("Paired density / continuum (in phase)")
    axes[1].set_ylabel("Paired density / continuum (quadrature)")
    axes[2].set_ylabel("Displacement outside fitted mode / target")
    axes[0].legend(fontsize=7)
    figure.suptitle("Paired particle growth: actual Eulerian density and mode mixing")
    figure.savefig(args.output / "glass_growth.png", dpi=170)
    plt.close(figure)
    write_csv(args.output / "growth_tracks.csv", traces)
    write_csv(args.output / "growth_summary.csv", summaries)
    write_json(args.output / "growth_summary.json", dict(metadata=metadata, runs=summaries))
    print(json.dumps(summaries))


def comparison(args):
    """Plot already-validated complete run summaries, never partial raw traces."""
    def read_summary(directory, filename):
        with (args.input / directory / "analysis" / filename).open() as source:
            return json.load(source)

    axis_rows, orientation_rows, growth_rows = {}, {}, {}
    summary = dict(amplitude_spacing=args.amplitude, axis_response={}, equal_norm_orientation={}, growth={})
    orientation_order = ((3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 2, 2), (2, 1, 2), (2, 2, 1))
    baseline_metadata = None
    common_settings = ("n_per_dimension", "species_count", "kernel_family", "kernel_degree", "softening_mode",
                       "neighbor_number", "theta", "backend", "omega_m", "omega_r", "omega_k", "omega_lambda")
    def verify_settings(metadata):
        nonlocal baseline_metadata
        settings = {key: metadata[key] for key in common_settings}
        if baseline_metadata is None:
            baseline_metadata = settings
        elif settings != baseline_metadata:
            raise ValueError("comparison contains mismatched particle counts, force settings, or background")

    for load in ("lattice", "glass"):
        response_data = read_summary(f"adaptive_{load}", "response_summary.json")
        verify_settings(response_data["metadata"])
        axis_rows[load] = [row for row in response_data["modes"] if row["amplitude_spacing"] == args.amplitude and
                           sum(value != 0 for value in mode_key(row)) == 1]
        expected_axes = {(n if axis == 0 else 0, n if axis == 1 else 0, n if axis == 2 else 0)
                         for n in (1, 2, 3) for axis in range(3)}
        if {mode_key(row) for row in axis_rows[load]} != expected_axes or len(axis_rows[load]) != 9:
            raise ValueError(f"comparison needs exactly all nine axis modes at the selected amplitude for {load}")
        diagonal_data = read_summary(f"orientation_{load}", "response_summary.json")
        verify_settings(diagonal_data["metadata"])
        combined = axis_rows[load] + [row for row in diagonal_data["modes"] if row["amplitude_spacing"] == args.amplitude]
        orientation_rows[load] = {mode_key(row): row for row in combined if mode_key(row) in orientation_order}
        if set(orientation_rows[load]) != set(orientation_order):
            raise ValueError(f"missing equal-wavelength orientation measurements for {load}")
        values = np.array([orientation_rows[load][mode]["longitudinal_sine"] for mode in orientation_order])
        summary["equal_norm_orientation"][load] = dict(minimum=float(values.min()), maximum=float(values.max()),
                                                        mean=float(values.mean()), relative_span=float(np.ptp(values) / values.mean()))
        summary["axis_response"][load] = []
        for n in (1, 2, 3):
            values = np.array([row["longitudinal_sine"] for row in axis_rows[load] if sum(value*value for value in mode_key(row)) == n*n])
            summary["axis_response"][load].append(dict(k_over_nyquist=2.*n/response_data["metadata"]["n_per_dimension"],
                mean=float(values.mean()), minimum=float(values.min()), maximum=float(values.max())))
        growth_directory = "evolution_glass" if load == "glass" else "adaptive_lattice"
        growth_data = read_summary(growth_directory, "growth_summary.json")
        verify_settings(growth_data["metadata"])
        growth_rows[load] = growth_data["runs"]
        if not growth_rows[load]:
            raise ValueError(f"missing complete growth runs for {load}")
        summary["growth"][load] = growth_rows[load]
    comparable = lambda row: (mode_key(row), row["initial_amplitude_spacing"], row["a_initial"], row["a"], row["steps"])
    if sorted(comparable(row) for row in growth_rows["glass"]) != sorted(comparable(row) for row in growth_rows["lattice"]):
        raise ValueError("growth comparison does not have matched modes, amplitudes, intervals, and steps")
    args.output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for load, color, marker in (("lattice", "C1", "s"), ("glass", "C0", "o")):
        rows = summary["axis_response"][load]
        mean = np.array([row["mean"] for row in rows])
        errors = np.array([[row["mean"] - row["minimum"] for row in rows],
                           [row["maximum"] - row["mean"] for row in rows]])
        axes[0].errorbar([row["k_over_nyquist"] for row in rows], mean, yerr=errors,
                         color=color, marker=marker, capsize=3, label=load)
        axes[1].plot(range(len(orientation_order)), [orientation_rows[load][mode]["longitudinal_sine"] for mode in orientation_order],
                     color=color, marker=marker, label=load)
        rows = sorted(growth_rows[load], key=lambda row: sum(value*value for value in mode_key(row)))
        side = read_summary(f"adaptive_{load}", "response_summary.json")["metadata"]["n_per_dimension"]
        axes[2].plot([2.*np.linalg.norm(mode_key(row))/side for row in rows],
                     [row["density_parallel_ratio"] for row in rows], color=color, marker=marker, label=load)
    for axis in axes:
        axis.grid(alpha=.2)
    axes[0].axhline(1., color=".4", linestyle=":", label="continuum")
    axes[0].set(xlabel="k / kNy", ylabel="Longitudinal force response", title="Three axis directions: mean and range")
    axes[0].legend()
    axes[1].set_xticks(range(len(orientation_order)), [str(mode).replace(" ", "") for mode in orientation_order], rotation=35, ha="right")
    axes[1].set(xlabel="Wavevector n", ylabel="Longitudinal force response", title="Equal wavelength: |n| = 3")
    axes[2].axhline(1., color=".4", linestyle=":")
    intervals = {(row["a_initial"], row["a"]) for row in growth_rows["glass"]}
    interval = f"a = {next(iter(intervals))[0]:g} → {next(iter(intervals))[1]:g}" if len(intervals) == 1 else "matched intervals"
    directions = "x-axis modes" if all(mode_key(row)[1:] == (0, 0) for row in growth_rows["glass"]) else "sampled modes"
    axes[2].set(xlabel="k / kNy", ylabel="Final paired density / continuum", title=f"Growth: {interval}, {directions}")
    degree = int(baseline_metadata["kernel_degree"])
    kernel = (f"Ferrers F{degree // 2}" if baseline_metadata["kernel_family"] == "ferrers" and degree % 2 == 0
              else f"{baseline_metadata['kernel_family']} degree {degree}")
    neighbors = f", {baseline_metadata['neighbor_number']:g} weighted neighbors" if baseline_metadata["softening_mode"] == "adaptive" else ""
    particles = baseline_metadata["species_count"] * baseline_metadata["n_per_dimension"]**3
    figure.suptitle(f"Generated glass versus cubic lattice: {baseline_metadata['softening_mode']} {kernel}{neighbors}, {particles} particles")
    figure.savefig(args.output / "glass_lattice_comparison.png", dpi=180)
    plt.close(figure)
    write_json(args.output / "glass_lattice_comparison.json", summary)
    print(json.dumps(summary["equal_norm_orientation"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("convert", help="convert native glass.bin to exact fixed32 text coordinates")
    command.add_argument("input", type=Path)
    command.add_argument("output", type=Path)
    command.set_defaults(function=convert)
    command = commands.add_parser("quality", help="measure periodic load quality against seeded Poisson and cubic lattice")
    command.add_argument("input", type=Path)
    command.add_argument("output", type=Path)
    command.add_argument("--seed", type=int, default=48619)
    command.add_argument("--max-mode", type=int, default=4)
    command.add_argument("--low-mode", type=int, default=2)
    command.set_defaults(function=quality)
    command = commands.add_parser("response", help="project measured response without assuming Fourier eigenmodes")
    command.add_argument("input", type=Path)
    command.add_argument("metadata", type=Path)
    command.add_argument("output", type=Path)
    command.set_defaults(function=response)
    command = commands.add_parser("requests", help="write continuum-growing-mode evolution requests")
    command.add_argument("metadata", type=Path)
    command.add_argument("output", type=Path)
    command.add_argument("--modes", default="1,0,0;3,0,0")
    command.add_argument("--a-initial", type=float, default=.02)
    command.add_argument("--a-final", type=float, default=.1)
    command.add_argument("--steps", type=int, default=8)
    command.add_argument("--amplitude", type=float, default=.001)
    command.add_argument("--label", default="continuum")
    command.set_defaults(function=requests)
    command = commands.add_parser("growth", help="analyze complete paired density and displacement traces")
    command.add_argument("metadata", type=Path)
    command.add_argument("requests", type=Path)
    command.add_argument("evolution", type=Path)
    command.add_argument("baseline", type=Path)
    command.add_argument("density", type=Path)
    command.add_argument("output", type=Path)
    command.set_defaults(function=growth)
    command = commands.add_parser("comparison", help="compare complete validated glass and lattice summaries")
    command.add_argument("input", type=Path, help="parent of adaptive_glass, adaptive_lattice, orientation_glass, orientation_lattice, evolution_glass")
    command.add_argument("output", type=Path)
    command.add_argument("--amplitude", type=float, default=.005)
    command.set_defaults(function=comparison)
    args = parser.parse_args()
    if args.command == "quality" and not (1 <= args.low_mode <= args.max_mode):
        parser.error("quality requires 1 <= low-mode <= max-mode")
    args.function(args)


if __name__ == "__main__":
    main()
