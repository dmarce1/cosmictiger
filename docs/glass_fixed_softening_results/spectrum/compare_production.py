#!/usr/bin/env python3
"""Compare independent Ewald forces and analytic Jacobian with raw production data."""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np


def read_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_rows(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def rms(field):
    return float(np.sqrt(np.mean(np.sum(field**2, axis=1))))


def complete(directory):
    command = json.loads((directory / "command.json").read_text())
    if command.get("return_code") != 0:
        raise ValueError(f"Production run is incomplete or failed: {directory}")
    return command


def initial_comparison(directory, output):
    complete(directory)
    path = directory / "linear_growth_particles_initial.csv"
    raw = sorted(read_rows(path), key=lambda row: int(row["particle"]))
    reference_path = output / "glass_forces.csv"
    reference = sorted(read_rows(reference_path), key=lambda row: int(row["particle"]))
    positions = np.array([[float(row[axis]) for axis in "xyz"] for row in raw])
    expected_positions = np.array([[float(row[axis]) for axis in "xyz"] for row in reference])
    if len(raw) != len(reference) or not np.array_equal(positions, expected_positions):
        raise ValueError("Production and Ewald particles do not have identical positions/order")
    production = np.array([[float(row[f"acceleration_{axis}"]) for axis in "xyz"] for row in raw])
    expected = np.array([[float(row[f"acceleration_{axis}"]) for axis in "xyz"] for row in reference])
    difference = production - expected
    rows = []
    for index in range(len(raw)):
        row = dict(particle=index)
        for component, axis in enumerate("xyz"):
            row[f"production_{axis}"] = production[index, component]
            row[f"ewald_{axis}"] = expected[index, component]
            row[f"difference_{axis}"] = difference[index, component]
        rows.append(row)
    write_rows(output / "production_initial_comparison.csv", rows)
    summary = dict(production=fingerprint(path), reference=fingerprint(reference_path),
                   particles=len(raw), production_force_rms=rms(production),
                   ewald_force_rms=rms(expected), difference_rms=rms(difference),
                   relative_difference_rms=float(np.linalg.norm(difference) / np.linalg.norm(expected)),
                   maximum_component_difference=float(np.max(np.abs(difference))))
    (output / "production_initial_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return positions


def response_comparison(directory, output, positions):
    command = complete(directory)
    repository = Path(__file__).resolve().parents[3]
    specification = importlib.util.spec_from_file_location(
        "independent_ewald", repository / "tests/analyze_glass_stability.py")
    independent = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(independent)
    _, jacobian = independent.ewald(positions)
    path = directory / "linear_growth_response_particles.csv"
    groups = {}
    for row in read_rows(path):
        key = tuple(int(row[f"mode_{axis}"]) for axis in "xyz") + (
            float(row["amplitude_spacing"]), int(row["input_species"]), int(row["input_axis"]))
        groups.setdefault(key, []).append(row)
    modes = [tuple(int(value) for value in mode.split(",")) for mode in
             command["environment"]["COSMICTIGER_LINEAR_GROWTH_MODES"].split(";")]
    amplitudes = [float(value) for value in
                  command["environment"]["COSMICTIGER_LINEAR_GROWTH_AMPLITUDES"].split(",")]
    expected = {mode + (amplitude, 0, axis)
                for mode in modes for amplitude in amplitudes for axis in range(3)}
    if set(groups) != expected:
        raise ValueError("Missing, extra, or mismatched production response groups")
    comparisons = []
    projections = []
    matrices = {}
    perturbation_fields = {}
    count = len(positions)
    support = .025 / round(count**(1. / 3.))
    for key, rows in groups.items():
        mode, amplitude, input_axis = key[:3], key[3], key[5]
        if len(rows) != count * 3:
            raise ValueError(f"Incomplete response group {key}")
        values = {name: np.full((count, 3), np.nan) for name in (
            "acceleration_positive", "acceleration_negative", "displacement_positive", "displacement_negative")}
        for row in rows:
            index, axis = int(row["particle"]), int(row["output_axis"])
            if int(row["output_species"]) != 0:
                raise ValueError("Only pure-DM response is supported")
            for name, field in values.items():
                if np.isfinite(field[index, axis]):
                    raise ValueError("Duplicate particle response component")
                field[index, axis] = float(row[name])
        if not all(np.isfinite(field).all() for field in values.values()):
            raise ValueError("Nonfinite or missing particle fields")
        plus_positions = positions + values["displacement_positive"]
        minus_positions = positions + values["displacement_negative"]
        for points in (plus_positions, minus_positions):
            separation = points[:, None, :] - points[None, :, :]
            separation -= np.rint(separation)
            distance = np.linalg.norm(separation, axis=2)
            np.fill_diagonal(distance, np.inf)
            if distance.min() <= support:
                raise ValueError("A perturbed pair enters the compact kernel support")
        displacement = .5 * (values["displacement_positive"] - values["displacement_negative"])
        production = .5 * (values["acceleration_positive"] - values["acceleration_negative"])
        linear = (jacobian @ displacement.ravel()).reshape(count, 3)
        phase = 2. * np.pi * (positions @ mode)
        basis = np.column_stack((np.ones(count), np.sin(phase), np.cos(phase)))
        fit_displacement = np.linalg.lstsq(basis, displacement, rcond=None)[0]
        realized_amplitude = float(fit_displacement[1, input_axis])
        normalization = 4. * np.pi * realized_amplitude
        fit = {name: np.linalg.lstsq(basis, field / normalization, rcond=None)[0]
               for name, field in (("production", production), ("linear", linear))}
        row = dict(mode_x=mode[0], mode_y=mode[1], mode_z=mode[2],
                   amplitude_spacing=amplitude, input_axis=input_axis,
                   realized_amplitude=realized_amplitude,
                   linear_response_rms=rms(linear),
                   production_minus_linear_relative_norm=float(np.linalg.norm(production - linear) / np.linalg.norm(linear)),
                   production_minus_linear_rms=rms(production - linear))
        comparisons.append(row)
        perturbation_fields[key] = (plus_positions, minus_positions, production, linear)
        matrix_key = mode + (amplitude,)
        matrix = matrices.setdefault(matrix_key, {
            name: np.full((3, 3), np.nan) for name in ("production", "linear")})
        for axis in range(3):
            projected = {name: fit[name][1, axis] for name in fit}
            projections.append(dict(
                mode_x=mode[0], mode_y=mode[1], mode_z=mode[2],
                amplitude_spacing=amplitude, input_axis=input_axis, output_axis=axis,
                **{f"{name}_sine": projected[name] for name in fit},
                **{f"{name}_cosine": fit[name][2, axis] for name in fit}))
            for name in fit:
                matrix[name][axis, input_axis] = projected[name]
        print(f"Compared mode={mode}, amplitude={amplitude}, axis={input_axis}: "
              f"production/linear={row['production_minus_linear_relative_norm']:.4e}", flush=True)
    exact_selected = []
    for amplitude in amplitudes:
        worst = max((row for row in comparisons if row["amplitude_spacing"] == amplitude),
                    key=lambda row: row["production_minus_linear_relative_norm"])
        key = tuple(worst[f"mode_{axis}"] for axis in "xyz") + (amplitude, 0, worst["input_axis"])
        plus_positions, minus_positions, production, linear = perturbation_fields[key]
        ewald_positive, _ = independent.ewald(plus_positions, withJacobian=False)
        ewald_negative, _ = independent.ewald(minus_positions, withJacobian=False)
        ewald = .5 * (ewald_positive - ewald_negative)
        row = dict(worst)
        row.update(
            selection="Largest production-minus-J*displacement relative norm at this amplitude",
            production_minus_exact_ewald_relative_norm=float(np.linalg.norm(production - ewald) / np.linalg.norm(ewald)),
            exact_finite_amplitude_minus_linear_relative_norm=float(np.linalg.norm(ewald - linear) / np.linalg.norm(linear)),
            production_minus_exact_ewald_rms=rms(production - ewald))
        exact_selected.append(row)
        print(f"Exact central-force check {key}: "
              f"production/Ewald={row['production_minus_exact_ewald_relative_norm']:.4e}; "
              f"finite/linear={row['exact_finite_amplitude_minus_linear_relative_norm']:.4e}", flush=True)
    write_rows(output / "production_response_field_comparison.csv", comparisons)
    write_rows(output / "production_response_projection_comparison.csv", projections)
    write_rows(output / "production_response_exact_selected.csv", exact_selected)
    longitudinal = []
    for key, values in matrices.items():
        direction = np.array(key[:3], dtype=float)
        direction /= np.linalg.norm(direction)
        row = dict(mode_x=key[0], mode_y=key[1], mode_z=key[2], amplitude_spacing=key[3])
        for name, matrix in values.items():
            row[f"{name}_longitudinal_response"] = float(direction @ matrix @ direction)
        longitudinal.append(row)
    write_rows(output / "production_response_longitudinal_comparison.csv", longitudinal)
    # Verify independently fitted production projections agree with the C++ output.
    reported_path = directory / "linear_growth_response.csv"
    reported = {}
    for row in read_rows(reported_path):
        key = tuple(int(row[f"mode_{axis}"]) for axis in "xyz") + (
            float(row["amplitude_spacing"]), int(row["input_axis"]), int(row["output_axis"]))
        reported[key] = row
    projection_difference = 0.
    for row in projections:
        key = tuple(row[f"mode_{axis}"] for axis in "xyz") + (
            row["amplitude_spacing"], row["input_axis"], row["output_axis"])
        for name in ("sine", "cosine"):
            field = "response_sin" if name == "sine" else "response_cos"
            projection_difference = max(projection_difference,
                                        abs(row[f"production_{name}"] - float(reported[key][field])))
    if projection_difference > 1.e-10:
        raise RuntimeError("Independent projection does not reproduce production projection")
    summary = dict(
        production_particle_input=fingerprint(path), production_projection_input=fingerprint(reported_path),
        particles=count, perturbations=len(comparisons),
        method="Analytic initial J times actual realized half-difference displacement for all perturbations; exact Ewald central forces at actual +/- positions for the largest production-versus-linear relative error at each amplitude",
        maximum_production_projection_reconstruction_difference=projection_difference,
        by_amplitude={str(amplitude): {
            name: max(row[name] for row in comparisons if row["amplitude_spacing"] == amplitude)
            for name in ("production_minus_linear_relative_norm",)}
            for amplitude in amplitudes},
        exact_selected=exact_selected,
        longitudinal=longitudinal)
    (output / "production_response_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial", type=Path)
    parser.add_argument("response", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--initial-only", action="store_true")
    arguments = parser.parse_args()
    positions = initial_comparison(arguments.initial, arguments.output)
    if not arguments.initial_only:
        response_comparison(arguments.response, arguments.output, positions)


if __name__ == "__main__":
    main()
