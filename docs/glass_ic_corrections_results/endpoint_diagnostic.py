#!/usr/bin/env python3
"""Infer a two-target endpoint fit from two measured initial-state families.

This is a linear-superposition prediction, not an independently evolved IC.
It deliberately mixes the numerical growing and decaying responses. It is
provided to illustrate why matching density and velocity at one epoch differs
from preparing approximately growing initial conditions.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np


def main():
    archive = Path(__file__).resolve().parent
    root = archive.parents[1]
    sys.path.insert(0, str(root / "tests"))
    from analyze_glass_ic_corrections import (check_matching_metadata, complex_field,
                                             load_samples, validate_metadata)
    from analyze_glass_growth import metadata_at, write_json
    from analyze_linear_growth import Background, mode_key, read_csv, write_csv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=archive.parent / "glass_growth_results")
    parser.add_argument("--input", type=Path, default=archive,
                        help="Complete model_glass and corrected_lattice raw run directories")
    parser.add_argument("--output", type=Path, default=archive)
    arguments = parser.parse_args()
    original_requests_path = arguments.baseline / "requests_8.csv"
    original_requests = read_csv(original_requests_path)
    expected_modes = [(1, 0, 0), (2, 0, 0), (3, 0, 0)]
    if [mode_key(request) for request in original_requests] != expected_modes:
        raise ValueError("Expected the three archived x-axis continuum requests")
    sources = {original_requests_path}
    rows = []
    matrices = []

    for load, original_name, model_name, requests_name in (
            ("glass", "evolution_glass", "model_glass", "requests_model.csv"),
            ("lattice", "adaptive_lattice", "corrected_lattice", "requests_lattice.csv")):
        original_directory = arguments.baseline / original_name
        model_directory = arguments.input / model_name
        requests_path = archive / requests_name
        model_requests = read_csv(requests_path)
        sources.add(requests_path)
        if [mode_key(request) for request in model_requests] != expected_modes:
            raise ValueError(f"Unexpected or incomplete {load} model requests")
        original_metadata = metadata_at(original_directory / "linear_growth_metadata.json")
        model_metadata = metadata_at(model_directory / "linear_growth_metadata.json")
        validate_metadata(original_metadata, load)
        validate_metadata(model_metadata, load)
        check_matching_metadata(original_metadata, model_metadata)
        if not (original_metadata.get("paired_evolution") and model_metadata.get("paired_evolution")):
            raise ValueError("Both state families must use paired positive, negative and zero runs")
        original_evolution, _, original_density = load_samples(original_directory, original_requests)
        model_evolution, _, model_density = load_samples(model_directory, model_requests)
        for directory in (original_directory, model_directory):
            for filename in ("linear_growth_metadata.json", "linear_growth_evolution.csv",
                             "linear_growth_baseline_evolution.csv", "linear_growth_density.csv"):
                sources.add(directory / filename)
        background = Background(original_metadata)

        for run, (original_request, model_request) in enumerate(zip(original_requests, model_requests)):
            mode = mode_key(original_request)
            for key in ("a_initial", "a_final", "steps"):
                if float(original_request[key]) != float(model_request[key]):
                    raise ValueError(f"Endpoint fit requires matched {key}")
            direction = np.asarray(mode, dtype=float) / np.linalg.norm(mode)
            original_u = np.asarray([float(original_request[f"u_dm_{axis}"]) for axis in "xyz"])
            original_v = np.asarray([float(original_request[f"v_dm_{axis}"]) for axis in "xyz"])
            model_u = np.asarray([float(model_request[f"u_dm_{axis}"]) for axis in "xyz"])
            model_v = np.asarray([float(model_request[f"v_dm_{axis}"]) for axis in "xyz"])
            amplitude = float(original_u @ direction)
            initial_velocity = float(original_v @ direction)
            if amplitude <= 0. or initial_velocity <= 0.:
                raise ValueError("Original continuum state must have positive longitudinal amplitudes")
            for vector in (original_u, original_v, model_u, model_v):
                if not np.allclose(vector, direction * (vector @ direction), rtol=1.e-12, atol=1.e-15):
                    raise ValueError("Scalar endpoint fit requires longitudinal sine-only initial states")
            start, end = (float(original_request[key]) for key in ("a_initial", "a_final"))
            solution = background.growing_solution(1., start, end)
            first = solution.sol(math.log(start))
            last = solution.sol(math.log(end))
            growth = float(last[0] / first[0])
            initial_rate = float(first[1] / first[0])
            final_rate = float(last[1] / last[0])
            if not math.isclose(initial_velocity / amplitude, initial_rate, rel_tol=1.e-11):
                raise ValueError("Original velocity is not the consistent continuum growing velocity")
            density_target = complex_field(original_density[(run, 0, 0, 0)], "density_diff") * growth
            velocity_target = amplitude * growth * final_rate
            if abs(density_target) < 1.e-15:
                raise ValueError("Original continuum density target vanishes")
            initial_matrix = np.asarray([[1., float(model_u @ direction / amplitude)],
                                         [1., float(model_v @ direction / initial_velocity)]])
            endpoint_matrix = np.zeros((2, 2))
            density_phase = np.zeros(2)
            velocity_quadrature = np.zeros(2)
            for column, (request, evolution, density) in enumerate((
                    (original_request, original_evolution, original_density),
                    (model_request, model_evolution, model_density))):
                final_step = int(request["steps"])
                density_ratio = complex_field(density[(run, final_step, 0, 0)], "density_diff") / density_target
                velocity = np.asarray([float(evolution[(run, final_step, 0, axis)]["v_sin"])
                                       for axis in range(3)])
                quadrature = np.asarray([float(evolution[(run, final_step, 0, axis)]["v_cos"])
                                         for axis in range(3)])
                endpoint_matrix[:, column] = (density_ratio.real, velocity @ direction / velocity_target)
                density_phase[column] = density_ratio.imag
                velocity_quadrature[column] = quadrature @ direction / velocity_target
            transition = np.linalg.solve(initial_matrix.T, endpoint_matrix.T).T
            target = np.ones(2)
            correction = np.linalg.solve(transition, target)
            weights = np.linalg.solve(initial_matrix, correction)
            recovery = endpoint_matrix @ weights
            if not np.allclose(recovery, target, rtol=1.e-10, atol=1.e-10):
                raise RuntimeError("Endpoint-map algebra did not recover its fitted target")
            row = dict(load=load, mode_x=mode[0], mode_y=mode[1], mode_z=mode[2],
                       k_over_axis_nyquist=2. * np.linalg.norm(mode) / original_metadata["side"],
                       a_initial=start, a_final=end,
                       predicted_initial_displacement_multiplier=float(correction[0]),
                       predicted_initial_total_velocity_multiplier=float(correction[1]),
                       predicted_velocity_per_displacement_multiplier=float(correction[1] / correction[0]),
                       reverses_initial_displacement=bool(correction[0] < 0.),
                       predicted_final_density_inphase=float(recovery[0]),
                       predicted_final_projected_velocity=float(recovery[1]),
                       predicted_final_density_phase=float(density_phase @ weights),
                       predicted_final_velocity_quadrature=float(velocity_quadrature @ weights),
                       initial_family_condition=float(np.linalg.cond(initial_matrix)),
                       initial_family_determinant=float(np.linalg.det(initial_matrix)),
                       endpoint_map_condition=float(np.linalg.cond(transition)),
                       endpoint_map_determinant=float(np.linalg.det(transition)),
                       original_family_weight=float(weights[0]), model_family_weight=float(weights[1]),
                       baseline_final_density=float(endpoint_matrix[0, 0]),
                       model_final_density=float(endpoint_matrix[0, 1]),
                       baseline_final_projected_velocity=float(endpoint_matrix[1, 0]),
                       model_final_projected_velocity=float(endpoint_matrix[1, 1]))
            rows.append(row)
            matrices.append(dict(load=load, mode=list(mode), initial_states=initial_matrix.tolist(),
                                 measured_endpoints=endpoint_matrix.tolist(),
                                 inferred_transition=transition.tolist(),
                                 measured_density_phases=density_phase.tolist(),
                                 measured_velocity_quadratures=velocity_quadrature.tolist()))

    arguments.output.mkdir(parents=True, exist_ok=True)
    write_csv(arguments.output / "endpoint_diagnostic.csv", rows)
    source_hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(sources)}
    write_json(arguments.output / "endpoint_diagnostic.json", dict(
        scope="Linear-superposition prediction from two paired measured state families; no independent endpoint-fit simulation",
        recommendation="Diagnostic only; this endpoint fit mixes growing and decaying responses and is not the recommended glass IC",
        initial_coordinates="A = u_initial/u_continuum_initial; B = (du/dln a)_initial/(f1_initial*u_continuum_initial)",
        endpoint_coordinates="In-phase actual paired Eulerian density/original continuum target; longitudinal sine-projected du/dln a/original continuum projected velocity target",
        velocity_caveat="The projected velocity observable is not the Eulerian density derivative; the fit also does not remove density phase or mode leakage",
        approximation_caveat="Linear superposition is inferred at finite perturbation amplitude; exact algebraic target recovery is not an independent dynamics validation",
        rows=rows, matrices=matrices, source_sha256=source_hashes))
    print(json.dumps(dict(output=str(arguments.output), predicted=rows), allow_nan=False))


if __name__ == "__main__":
    main()
