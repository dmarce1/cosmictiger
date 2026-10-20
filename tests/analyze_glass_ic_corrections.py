#!/usr/bin/env python3
# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# Distributed under the Boost Software License, Version 1.0.

"""Calibrate and validate horizon-specific scalar glass IC corrections.

The glass's projected force response supplies an approximate growing velocity,
not a glass eigenmode. Full paired trajectories measure the actual response.
All target ratios use the ORIGINAL continuum IC, never the corrected initial
density, so a larger seed cannot hide missing growth by changing the target.
"""

import argparse
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

from analyze_glass_growth import indexed_samples, metadata_at, write_json
from analyze_linear_growth import Background, mode_key, read_csv, write_csv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_metadata(metadata, load):
    expected = dict(side=8, species=1, kernel_family="ferrers", kernel_degree=4,
                    softening_mode="adaptive", neighbor_number=24,
                    pressure_enabled=False)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"unexpected {key}: {metadata.get(key)!r}, expected {value!r}")
    if load == "glass" and metadata.get("load_kind") != "file":
        raise ValueError("glass analysis requires the saved glass load")
    if load == "lattice" and metadata.get("load_kind") != "lattice":
        raise ValueError("lattice analysis requires a lattice load")


def check_matching_metadata(first, second):
    for key in ("side", "species", "mass_dm", "mass_gas", "gm", "kernel_family",
                "kernel_degree", "softening_mode", "neighbor_number", "theta",
                "backend", "omega_m", "omega_r", "omega_k", "omega_lambda",
                "pressure_enabled"):
        if first[key] != second[key]:
            raise ValueError(f"mismatched baseline setting {key}")


def request_for(source, correction, label, steps=None, amplitude_scale=1.):
    row = dict(source)
    row["label"] = label
    if steps is not None:
        row["steps"] = steps
    for axis in "xyz":
        displacement = (float(source[f"u_dm_{axis}"]) *
                        correction["displacement_multiplier"] * amplitude_scale)
        row[f"u_dm_{axis}"] = displacement
        row[f"v_dm_{axis}"] = displacement * correction["initial_growth_rate"]
    return row


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = args.baseline
    metadata = metadata_at(baseline / "evolution_glass/linear_growth_metadata.json")
    validate_metadata(metadata, "glass")
    requests = read_csv(baseline / "requests_8.csv")
    response_path = baseline / "adaptive_glass/analysis/response_summary.json"
    responses = json.loads(response_path.read_text())
    check_matching_metadata(metadata, responses["metadata"])
    if [mode_key(row) for row in requests] != [(1, 0, 0), (2, 0, 0), (3, 0, 0)]:
        raise ValueError("expected the three archived x-axis continuum requests")
    background = Background(metadata)
    models = []
    for request in requests:
        mode = mode_key(request)
        candidates = [row for row in responses["modes"]
                      if mode_key(row) == mode and row["amplitude_spacing"] == .005]
        if len(candidates) != 1:
            raise ValueError(f"missing or duplicate initial response for {mode}")
        response = candidates[0]
        eigenvalue = float(response["longitudinal_sine"])
        start, end = float(request["a_initial"]), float(request["a_final"])
        continuum = background.growing_solution(1., start, end)
        effective = background.growing_solution(eigenvalue, start, end)
        c0, ct = continuum.sol(math.log(start)), continuum.sol(math.log(end))
        e0, et = effective.sol(math.log(start)), effective.sol(math.log(end))
        multiplier = float((ct[0] / c0[0]) / (et[0] / e0[0]))
        rate = float(e0[1] / e0[0])
        rate_ratio = float(rate / (c0[1] / c0[0]))
        models.append(dict(mode_x=mode[0], mode_y=mode[1], mode_z=mode[2],
                           k_over_axis_nyquist=response["k_over_nyquist"],
                           effective_longitudinal_response=eigenvalue,
                           displacement_multiplier=multiplier,
                           velocity_per_displacement_multiplier=rate_ratio,
                           total_initial_velocity_multiplier=multiplier * rate_ratio,
                           initial_growth_rate=rate,
                           continuum_initial_growth_rate=float(c0[1] / c0[0]),
                           model_final_velocity_ratio_at_matched_density=float(
                               (et[1] / et[0]) / (ct[1] / ct[0])),
                           a_initial=start, a_final=end,
                           continuum_growth=float(ct[0] / c0[0]),
                           initial_amplitude_spacing=float(request["u_dm_x"])))
    write_csv(args.output / "correction_model.csv", models)
    write_json(args.output / "correction_model.json", dict(
        interpretation="Approximate scalar model; not glass eigenmodes", metadata=metadata,
        source_response_sha256=digest(response_path), corrections=models))
    write_csv(args.output / "requests_model.csv",
              [request_for(request, model, "model") for request, model in zip(requests, models)])
    references = read_csv(args.output / "reference_lattice.csv")
    references = {mode_key(row): {key: float(value) for key, value in row.items()}
                  for row in references}
    write_csv(args.output / "requests_lattice.csv",
              [request_for(request, references[mode_key(request)], "lattice_corrected")
               for request in requests])
    print(json.dumps(dict(prepared=len(models), output=str(args.output))))


def load_samples(directory, requests):
    fields = ("u_sin", "u_cos", "v_sin", "v_cos", "residual_rms")
    displacement = indexed_samples(directory / "linear_growth_evolution.csv", requests, 1, True, fields)
    baseline = indexed_samples(directory / "linear_growth_baseline_evolution.csv", requests, 1, True,
                               fields + ("u_rms", "v_rms", "even_u_rms", "even_v_rms"))
    density = indexed_samples(directory / "linear_growth_density.csv", requests, 1, False,
                              ("density_diff_real", "density_diff_imag", "baseline_real",
                               "baseline_imag", "even_real", "even_imag"))
    for run, request in enumerate(requests):
        actual_u = [float(displacement[(run, 0, 0, axis)]["u_sin"]) for axis in range(3)]
        actual_v = [float(displacement[(run, 0, 0, axis)]["v_sin"]) for axis in range(3)]
        expected_u = [float(request[f"u_dm_{axis}"]) for axis in "xyz"]
        expected_v = [float(request[f"v_dm_{axis}"]) for axis in "xyz"]
        if (not np.allclose(actual_u, expected_u, rtol=1.e-10, atol=1.e-12) or
                not np.allclose(actual_v, expected_v, rtol=1.e-10, atol=1.e-12)):
            raise ValueError(f"initial projected state does not match request {run}")
    return displacement, baseline, density


def complex_field(row, prefix):
    return complex(float(row[prefix + "_real"]), float(row[prefix + "_imag"]))


def analyze(args):
    if not math.isfinite(args.target_scale) or args.target_scale <= 0.:
        raise ValueError("target scale must be finite and positive")
    baseline_directory = args.baseline / ("evolution_glass" if args.load == "glass" else "adaptive_lattice")
    baseline_requests = read_csv(args.baseline / "requests_8.csv")
    baseline_metadata = metadata_at(baseline_directory / "linear_growth_metadata.json")
    metadata = metadata_at(args.input / "linear_growth_metadata.json")
    validate_metadata(metadata, args.load)
    check_matching_metadata(metadata, baseline_metadata)
    if not metadata.get("paired_evolution") or not baseline_metadata.get("paired_evolution"):
        raise ValueError("correction validation requires paired positive/negative/zero trajectories")
    glass_sha256 = None
    if args.load == "glass":
        glass_sha256 = digest(args.input.parent / "glass.txt")
        if glass_sha256 != digest(args.baseline / "glass.txt"):
            raise ValueError("corrected and original runs use different saved glass coordinates")
    requests = read_csv(args.requests)
    if not requests:
        raise ValueError("empty correction request file")
    displacement, baseline, density = load_samples(args.input, requests)
    _, _, original_density = load_samples(baseline_directory, baseline_requests)
    baseline_indices = {mode_key(row): index for index, row in enumerate(baseline_requests)}
    if len(baseline_indices) != len(baseline_requests):
        raise ValueError("duplicate baseline modes")
    background = Background(metadata)
    tracks, summaries = [], []
    for run, request in enumerate(requests):
        mode = mode_key(request)
        original_run = baseline_indices[mode]
        original_request = baseline_requests[original_run]
        start, end = float(request["a_initial"]), float(request["a_final"])
        if (start, end) != (float(original_request["a_initial"]), float(original_request["a_final"])):
            raise ValueError("correction and original target intervals differ")
        direction = np.asarray(mode, dtype=float) / np.linalg.norm(mode)
        original_u = np.asarray([float(original_request[f"u_dm_{axis}"]) for axis in "xyz"])
        original_amplitude = float(original_u @ direction) * args.target_scale
        initial_density = complex_field(original_density[(original_run, 0, 0, 0)], "density_diff") * args.target_scale
        if original_amplitude <= 0. or abs(initial_density) < 1.e-15:
            raise ValueError("invalid original target amplitude")
        solution = background.growing_solution(1., start, end)
        initial_growth = solution.sol(math.log(start))[0]
        run_rows = []
        for step in range(int(request["steps"]) + 1):
            sample = density[(run, step, 0, 0)]
            scale = float(sample["a"])
            continuum_state = solution.sol(math.log(scale))
            growth = float(continuum_state[0] / initial_growth)
            rate = float(continuum_state[1] / continuum_state[0])
            target_u = original_amplitude * growth
            target_density = initial_density * growth
            ratio = complex_field(sample, "density_diff") / target_density
            u = np.asarray([float(displacement[(run, step, 0, axis)]["u_sin"]) for axis in range(3)])
            v = np.asarray([float(displacement[(run, step, 0, axis)]["v_sin"]) for axis in range(3)])
            u_cos = np.asarray([float(displacement[(run, step, 0, axis)]["u_cos"]) for axis in range(3)])
            v_cos = np.asarray([float(displacement[(run, step, 0, axis)]["v_cos"]) for axis in range(3)])
            outside = math.sqrt(sum(float(displacement[(run, step, 0, axis)]["residual_rms"])**2 for axis in range(3)))
            zero_rms = math.sqrt(sum(float(baseline[(run, step, 0, axis)]["u_rms"])**2 for axis in range(3)))
            even_rms = math.sqrt(sum(float(baseline[(run, step, 0, axis)]["even_u_rms"])**2 for axis in range(3)))
            row = dict(run_index=run, label=request["label"], mode_x=mode[0], mode_y=mode[1], mode_z=mode[2],
                       step=step, a=scale, continuum_growth=growth,
                       density_parallel_ratio=float(ratio.real), density_phase_ratio=float(ratio.imag),
                       density_amplitude_ratio=float(abs(ratio)),
                       displacement_parallel_ratio=float(u @ direction / target_u),
                       velocity_parallel_ratio=float(v @ direction / (target_u * rate)),
                       displacement_cosine_relative=float(u_cos @ direction / target_u),
                       velocity_cosine_relative=float(v_cos @ direction / (target_u * rate)),
                       nonmode_displacement_relative_rms=outside / target_u,
                       baseline_displacement_rms_spacing=zero_rms,
                       even_displacement_relative_rms=even_rms / target_u,
                       even_density_relative_to_target=abs(complex_field(sample, "even")) / abs(target_density),
                       baseline_density_amplitude=abs(complex_field(sample, "baseline")))
            run_rows.append(row)
            tracks.append(row)
        final = dict(run_rows[-1])
        final.update(a_initial=start, a_final=end, steps=int(request["steps"]), target_scale=args.target_scale,
                     original_target_amplitude_spacing=original_amplitude,
                     initial_displacement_multiplier=run_rows[0]["displacement_parallel_ratio"],
                     initial_total_velocity_multiplier=run_rows[0]["velocity_parallel_ratio"],
                     initial_velocity_per_displacement_multiplier=(run_rows[0]["velocity_parallel_ratio"] /
                                                                  run_rows[0]["displacement_parallel_ratio"]),
                     maximum_even_density_relative=max(row["even_density_relative_to_target"] for row in run_rows),
                     maximum_nonmode_displacement_relative_rms=max(row["nonmode_displacement_relative_rms"] for row in run_rows))
        summaries.append(final)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "tracks.csv", tracks)
    write_csv(args.output / "summary.csv", summaries)
    write_json(args.output / "summary.json", dict(metadata=metadata, load=args.load,
               target="Original continuum IC differential density and displacement; not corrected IC",
               velocity_observable="Lagrangian sine projection of d(displacement)/dln(a); not Eulerian density derivative",
               target_scale=args.target_scale, requests_sha256=digest(args.requests),
               glass_sha256=glass_sha256, runs=summaries))
    plot_tracks(tracks, args.output, args.load)
    print(json.dumps(dict(analyzed=len(summaries), output=str(args.output), final_density=[row["density_parallel_ratio"] for row in summaries])))


def plot_tracks(tracks, output, load):
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for run in sorted({row["run_index"] for row in tracks}):
        rows = [row for row in tracks if row["run_index"] == run]
        for axis, key in zip(axes, ("density_parallel_ratio", "velocity_parallel_ratio", "nonmode_displacement_relative_rms")):
            axis.plot([row["a"] for row in rows], [row[key] for row in rows], "o-", markersize=3,
                      label=f"n={rows[0]['mode_x']}")
    for axis in axes:
        axis.set(xlabel="Scale factor a", xscale="log")
        axis.grid(alpha=.2)
    for axis in axes[:2]:
        axis.axhline(1., color=".4", linestyle=":")
    axes[0].set_ylabel("Density / original continuum target")
    axes[1].set_ylabel("Projected velocity / continuum target")
    axes[2].set_ylabel("Non-mode displacement / target")
    axes[0].legend()
    figure.suptitle(f"F2 / 24: {load} correction validation")
    figure.savefig(output / "validation.png", dpi=170)
    plt.close(figure)


def calibrate(args):
    models = read_csv(args.archive / "correction_model.csv")
    model_metadata = json.loads((args.archive / "correction_model.json").read_text())["metadata"]
    measurement_data = json.loads(args.model.read_text())
    validate_metadata(measurement_data["metadata"], "glass")
    check_matching_metadata(measurement_data["metadata"], model_metadata)
    if measurement_data["load"] != "glass" or measurement_data["target_scale"] != 1.:
        raise ValueError("model calibration requires the full-amplitude glass target")
    summaries = measurement_data["runs"]
    measured = {mode_key(row): row for row in summaries}
    if (len(measured) != len(summaries) or len(models) != len(measured) or
            {mode_key(row) for row in models} != set(measured)):
        raise ValueError("model calibration requires one complete matching measurement per mode")
    original = read_csv(args.baseline / "requests_8.csv")
    corrected = []
    for model in models:
        row = {key: float(value) for key, value in model.items()}
        mode = mode_key(model)
        row.update(mode_x=mode[0], mode_y=mode[1], mode_z=mode[2])
        measurement = measured[mode]
        if (measurement["target_scale"] != 1. or measurement["a_initial"] != row["a_initial"] or
                measurement["a_final"] != row["a_final"]):
            raise ValueError("model calibration used a different target")
        if not math.isclose(measurement["initial_displacement_multiplier"], row["displacement_multiplier"], rel_tol=1.e-9):
            raise ValueError("model calibration measurement used a different displacement")
        if not math.isclose(measurement["initial_velocity_per_displacement_multiplier"],
                            row["velocity_per_displacement_multiplier"], rel_tol=1.e-9):
            raise ValueError("model calibration used a different velocity")
        if not math.isclose(measurement["initial_total_velocity_multiplier"],
                            row["total_initial_velocity_multiplier"], rel_tol=1.e-9):
            raise ValueError("model calibration used a different total initial velocity")
        ratio = measurement["density_parallel_ratio"]
        if not .25 < ratio < 4.:
            raise ValueError("refusing an ill-conditioned empirical rescaling")
        row["model_displacement_multiplier"] = row["displacement_multiplier"]
        row["model_measured_final_density_ratio"] = ratio
        row["displacement_multiplier"] /= ratio
        row["total_initial_velocity_multiplier"] = row["displacement_multiplier"] * row["velocity_per_displacement_multiplier"]
        corrected.append(row)
    write_csv(args.archive / "correction_calibrated.csv", corrected)
    write_json(args.archive / "correction_calibrated.json", dict(
        interpretation="Empirical in-phase density match at a_final; approximate initial growing velocity retained",
        model_validation_sha256=digest(args.model), corrections=corrected))
    mapping = {mode_key(row): row for row in corrected}
    rows = [request_for(row, mapping[mode_key(row)], "calibrated") for row in original]
    write_csv(args.archive / "requests_calibrated.csv", rows)
    shortest = original[-1]
    write_csv(args.archive / "requests_control_16.csv", [request_for(shortest, mapping[mode_key(shortest)], "calibrated_16", steps=16)])
    write_csv(args.archive / "requests_control_half.csv", [request_for(shortest, mapping[mode_key(shortest)], "calibrated_half", amplitude_scale=.5)])
    print(json.dumps(dict(calibrated=len(rows), output=str(args.archive))))


def compare(args):
    model = {mode_key(row): row for row in read_csv(args.archive / "correction_model.csv")}
    calibrated = {mode_key(row): row for row in read_csv(args.archive / "correction_calibrated.csv")}
    lattice = {mode_key(row): row for row in read_csv(args.archive / "reference_lattice.csv")}
    measurements = {}
    for name in ("model_glass", "calibrated_glass", "corrected_lattice", "control_16_glass", "control_half_glass"):
        measurements[name] = json.loads((args.archive / name / "analysis/summary.json").read_text())["runs"]
    by_mode = {name: {mode_key(row): row for row in rows} for name, rows in measurements.items()}
    rows = []
    for mode in sorted(model):
        m, c, l = model[mode], calibrated[mode], lattice[mode]
        mg, cg, cl = (by_mode[name][mode] for name in ("model_glass", "calibrated_glass", "corrected_lattice"))
        rows.append(dict(mode_x=mode[0], mode_y=mode[1], mode_z=mode[2], k_over_axis_nyquist=float(m["k_over_axis_nyquist"]),
                         glass_model_displacement=float(m["displacement_multiplier"]), glass_calibrated_displacement=float(c["displacement_multiplier"]),
                         glass_rate_ratio=float(c["velocity_per_displacement_multiplier"]), glass_total_velocity=float(c["total_initial_velocity_multiplier"]),
                         lattice_displacement=float(l["displacement_multiplier"]), lattice_rate_ratio=float(l["velocity_per_displacement_multiplier"]),
                         lattice_total_velocity=float(l["total_initial_velocity_multiplier"]), model_glass_final_density=mg["density_parallel_ratio"],
                         calibrated_glass_final_density=cg["density_parallel_ratio"], corrected_lattice_final_density=cl["density_parallel_ratio"],
                         calibrated_glass_final_phase=cg["density_phase_ratio"], calibrated_glass_final_velocity=cg["velocity_parallel_ratio"],
                         corrected_lattice_final_velocity=cl["velocity_parallel_ratio"], glass_nonmode_relative_rms=cg["nonmode_displacement_relative_rms"]))
    shortest = by_mode["calibrated_glass"][(3, 0, 0)]
    controls = {}
    for name in ("control_16_glass", "control_half_glass"):
        row = by_mode[name][(3, 0, 0)]
        controls[name] = {key: row[key] / shortest[key] - 1. for key in
                          ("density_parallel_ratio", "velocity_parallel_ratio")}
        controls[name]["phase_absolute_difference"] = row["density_phase_ratio"] - shortest["density_phase_ratio"]
    write_csv(args.archive / "comparison.csv", rows)
    write_json(args.archive / "comparison.json", dict(corrections=rows, controls=controls, measurements=measurements))
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    k = [row["k_over_axis_nyquist"] for row in rows]
    for load, marker in (("glass", "o"), ("lattice", "s")):
        displacement_key = "glass_calibrated_displacement" if load == "glass" else "lattice_displacement"
        axes[0].plot(k, [row[displacement_key] for row in rows], marker + "-", label=load)
        axes[1].plot(k, [row[load + "_rate_ratio"] for row in rows], marker + "-")
        density_key = "calibrated_glass_final_density" if load == "glass" else "corrected_lattice_final_density"
        axes[2].plot(k, [100. * (row[density_key] - 1.) for row in rows], marker + "-",
                     label="glass calibrated" if load == "glass" else "lattice corrected")
    axes[2].plot(k, [100. * (row["model_glass_final_density"] - 1.) for row in rows], "o--",
                 color=".5", label="glass scalar prediction")
    for axis in axes:
        axis.set_xlabel(r"$k/k_{\rm Ny}$")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Initial displacement multiplier")
    axes[1].set_ylabel("Initial velocity / displacement factor")
    axes[2].set_ylabel("Final in-phase density error (%)")
    axes[0].legend()
    axes[2].legend(fontsize=8)
    axes[2].axhline(0., color=".4", linestyle=":")
    figure.suptitle("F2 / 24 IC corrections: z=49 to z=9, 512 particles")
    figure.savefig(args.archive / "correction_comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps(dict(compared=len(rows), controls=controls)))


def self_check(args):
    """Exercise original-target normalization against an analytic continuum mode."""
    source = Path(__file__).resolve().parents[1] / "docs/glass_growth_results"
    metadata = metadata_at(source / "evolution_glass/linear_growth_metadata.json")
    original_request = read_csv(source / "requests_8.csv")[0]
    original_request["steps"] = 2
    background = Background(metadata)
    start, end = float(original_request["a_initial"]), float(original_request["a_final"])
    solution = background.growing_solution(1., start, end)
    initial = solution.sol(math.log(start))
    rate = float(initial[1] / initial[0])
    boost = 1.7
    correction = dict(displacement_multiplier=boost, initial_growth_rate=rate)
    corrected_request = request_for(original_request, correction, "synthetic_boost")

    def fixture(directory, request, amplitude):
        directory.mkdir(parents=True)
        write_json(directory / "linear_growth_metadata.json", metadata)
        evolution, zeros, densities = [], [], []
        for step in range(3):
            scale = start * math.exp(step * math.log(end / start) / 2)
            state = solution.sol(math.log(scale))
            growth, growth_rate = float(state[0] / initial[0]), float(state[1] / state[0])
            for axis in range(3):
                row = dict(label=request["label"], run_index=0, step=step, a=scale, species=0, axis=axis,
                           u_sin=amplitude * .001 * growth if axis == 0 else 0., u_cos=0.,
                           v_sin=amplitude * .001 * growth * growth_rate if axis == 0 else 0., v_cos=0., residual_rms=0.)
                evolution.append(row)
                zero = dict(row)
                zero.update(u_sin=0., v_sin=0., u_rms=0., v_rms=0., even_u_rms=0., even_v_rms=0.)
                zeros.append(zero)
            delta = complex(-.0004, .00002) * amplitude * growth
            densities.append(dict(label=request["label"], run_index=0, step=step, a=scale, species=0,
                                  density_diff_real=delta.real, density_diff_imag=delta.imag,
                                  baseline_real=0., baseline_imag=0., even_real=0., even_imag=0.))
        write_csv(directory / "linear_growth_evolution.csv", evolution)
        write_csv(directory / "linear_growth_baseline_evolution.csv", zeros)
        write_csv(directory / "linear_growth_density.csv", densities)

    checks = []
    with tempfile.TemporaryDirectory(prefix="cosmictiger_glass_ic_check_") as temporary:
        root = Path(temporary)
        baseline, actual, output = root / "baseline", root / "actual", root / "analysis"
        fixture(baseline / "evolution_glass", original_request, 1.)
        write_csv(baseline / "requests_8.csv", [original_request])
        (baseline / "glass.txt").write_bytes((source / "glass.txt").read_bytes())
        (root / "glass.txt").write_bytes((source / "glass.txt").read_bytes())
        fixture(actual, corrected_request, boost)
        request_path = root / "requests.csv"
        write_csv(request_path, [corrected_request])
        options = argparse.Namespace(baseline=baseline, input=actual, requests=request_path,
                                     output=output, load="glass", target_scale=1.)
        analyze(options)
        summary = json.loads((output / "summary.json").read_text())["runs"][0]
        for key in ("density_parallel_ratio", "displacement_parallel_ratio", "velocity_parallel_ratio"):
            if not math.isclose(summary[key], boost, rel_tol=2.e-12):
                raise AssertionError(f"original target lost the deliberate boost in {key}")
        checks.append("original-target density, displacement and velocity retain a 1.7 boost")
        options.target_scale = .5
        analyze(options)
        half = json.loads((output / "summary.json").read_text())["runs"][0]
        if not math.isclose(half["density_parallel_ratio"], 2. * boost, rel_tol=2.e-12):
            raise AssertionError("half-amplitude target was not scaled independently of the corrected seed")
        checks.append("halving the original target doubles its measured correction ratio")
        options.target_scale = 1.
        analyze(options)
        models = [dict(mode_x=1, mode_y=0, mode_z=0, displacement_multiplier=boost,
                       initial_growth_rate=rate, velocity_per_displacement_multiplier=1.,
                       total_initial_velocity_multiplier=boost, a_initial=start, a_final=end)]
        write_csv(root / "correction_model.csv", models)
        write_json(root / "correction_model.json", dict(metadata=metadata, corrections=models))
        calibrate(argparse.Namespace(archive=root, baseline=baseline, model=output / "summary.json"))
        calibrated = read_csv(root / "correction_calibrated.csv")[0]
        if not math.isclose(float(calibrated["displacement_multiplier"]), 1., rel_tol=2.e-12):
            raise AssertionError("known empirical rescaling did not recover the original continuum seed")
        checks.append("empirical rescaling recovers the analytically correct unit seed")
        for label, path, transform in (
                ("missing density row", actual / "linear_growth_density.csv", lambda rows: rows[:-1]),
                ("mismatched request label", request_path,
                 lambda rows: [dict(rows[0], label="wrong_label")]),
                ("wrong initial velocity", actual / "linear_growth_evolution.csv",
                 lambda rows: [dict(rows[0], v_sin=float(rows[0]["v_sin"]) * 1.1)] + rows[1:])):
            content = path.read_bytes()
            write_csv(path, transform(read_csv(path)))
            try:
                analyze(options)
            except ValueError:
                checks.append("rejects " + label)
            else:
                raise AssertionError("accepted " + label)
            finally:
                path.write_bytes(content)
        path = actual / "linear_growth_metadata.json"
        invalid = dict(metadata, neighbor_number=12)
        write_json(path, invalid)
        try:
            analyze(options)
        except ValueError:
            checks.append("rejects mismatched gravity metadata")
        else:
            raise AssertionError("accepted mismatched gravity metadata")
    result = dict(passed=len(checks), checks=checks)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("baseline", type=Path)
    command.add_argument("output", type=Path)
    command.set_defaults(function=prepare)
    command = commands.add_parser("analyze")
    command.add_argument("baseline", type=Path)
    command.add_argument("input", type=Path)
    command.add_argument("requests", type=Path)
    command.add_argument("output", type=Path)
    command.add_argument("--load", choices=("glass", "lattice"), default="glass")
    command.add_argument("--target-scale", type=float, default=1.)
    command.set_defaults(function=analyze)
    command = commands.add_parser("calibrate")
    command.add_argument("baseline", type=Path)
    command.add_argument("archive", type=Path)
    command.add_argument("model", type=Path, help="validated model summary.json")
    command.set_defaults(function=calibrate)
    command = commands.add_parser("compare")
    command.add_argument("archive", type=Path)
    command.set_defaults(function=compare)
    command = commands.add_parser("self-check")
    command.add_argument("--output", type=Path)
    command.set_defaults(function=self_check)
    args = parser.parse_args()
    if hasattr(args, "target_scale") and (not math.isfinite(args.target_scale) or args.target_scale <= 0.):
        parser.error("target-scale must be finite and positive")
    args.function(args)


if __name__ == "__main__":
    main()
