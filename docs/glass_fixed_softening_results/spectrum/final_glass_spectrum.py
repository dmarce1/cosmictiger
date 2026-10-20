#!/usr/bin/env python3
"""Freeze the final unperturbed glass and compare its full Newtonian spectrum."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np


def read_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evolution", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    command = json.loads((arguments.evolution / "command.json").read_text())
    marker = "Linear growth diagnostic complete (CPU, 1 species, N=8)"
    captured_success = command.get("return_code") == 0
    completed_uncaptured_exit = (
        command.get("runner_session_exit_code") == 143 and
        "child_return_code" in command and command["child_return_code"] is None and
        command.get("completion_marker_present") is True and
        marker in (arguments.evolution / "run.log").read_text())
    if not captured_success and not completed_uncaptured_exit:
        raise ValueError("Production evolution is incomplete or failed")
    metadata = json.loads((arguments.evolution / "linear_growth_metadata.json").read_text())
    if metadata["species"] != 1:
        raise ValueError("Only equal-mass single-species loads are supported")
    side = metadata["side"]
    count = side**3
    support = metadata["baseline_support_dm"]
    requests = read_rows(arguments.evolution / "requests.csv")
    path = arguments.evolution / "linear_growth_particles.csv"
    initial = sorted(read_rows(arguments.evolution / "linear_growth_particles_initial.csv"),
                     key=lambda row: int(row["particle"]))
    reference = np.array([[float(row[axis]) for axis in "xyz"] for row in initial])
    if reference.shape != (count, 3):
        raise ValueError("Initial particle reference is incomplete")
    rows = read_rows(path)
    seen = set()
    groups = {}
    maximum_coordinate_residual = 0.
    for row in rows:
        run = int(row["run_index"])
        step = int(row["step"])
        particle = int(row["particle"])
        trajectory = row["trajectory"]
        if run not in range(len(requests)) or particle not in range(count):
            raise ValueError("Invalid particle or run index")
        request = requests[run]
        steps = int(request["steps"])
        if step not in range(steps + 1) or trajectory not in ("positive", "negative", "zero"):
            raise ValueError("Invalid step or trajectory")
        key = (run, step, trajectory, particle)
        if key in seen or int(row["species"]) != 0 or row["label"] != request["label"]:
            raise ValueError("Duplicate or incorrectly identified particle record")
        seen.add(key)
        expected_scale = math.exp(math.log(float(request["a_initial"])) + step / steps *
                                  math.log(float(request["a_final"]) / float(request["a_initial"])))
        if abs(float(row["a"]) - expected_scale) > 1.e-12:
            raise ValueError("Wrong scale factor in particle record")
        for field in ("x", "y", "z", "u_x", "u_y", "u_z", "realized_u_x", "realized_u_y",
                      "realized_u_z", "v_x", "v_y", "v_z"):
            if not math.isfinite(float(row[field])):
                raise ValueError("Nonfinite particle field")
        position = np.array([float(row[axis]) for axis in "xyz"])
        displacement = np.array([float(row[f"u_{axis}"]) for axis in "xyz"])
        realized = np.array([float(row[f"realized_u_{axis}"]) for axis in "xyz"])
        if np.any(position < 0.) or np.any(position >= 1.):
            raise ValueError("Unwrapped position in periodic particle record")
        residual = position - reference[particle] - displacement
        residual -= np.rint(residual)
        maximum_coordinate_residual = max(maximum_coordinate_residual, float(np.max(np.abs(residual))))
        realized_residual = position - reference[particle] - realized
        realized_residual -= np.rint(realized_residual)
        if np.max(np.abs(realized_residual)) > 2.e-15:
            raise ValueError("Recorded realized displacement disagrees with coordinates")
        if row["trajectory"] == "zero" and int(row["step"]) == int(requests[run]["steps"]):
            if abs(float(row["a"]) - float(requests[run]["a_final"])) > 1.e-12:
                raise ValueError("Final baseline row has the wrong scale factor")
            groups.setdefault(run, []).append(row)
    expected_rows = sum((int(request["steps"]) + 1) * 3 * count for request in requests)
    if len(seen) != expected_rows or maximum_coordinate_residual > 2.**-32 + 2.e-15:
        raise ValueError("Incomplete particle fields or fixed-point displacement mismatch")
    if set(groups) != set(range(len(requests))):
        raise ValueError("Missing final unperturbed trajectory")
    final_positions = {}
    for run, rows in groups.items():
        rows.sort(key=lambda row: int(row["particle"]))
        if len(rows) != count or [int(row["particle"]) for row in rows] != list(range(count)):
            raise ValueError("Missing or duplicate baseline particle")
        final_positions[run] = np.array([[float(row[axis]) for axis in "xyz"] for row in rows])
    chosen_run = 0
    positions = final_positions[chosen_run]
    repeat_discrepancy = 0.
    for other in final_positions.values():
        difference = other - positions
        difference -= np.rint(difference)
        repeat_discrepancy = max(repeat_discrepancy, float(np.max(np.linalg.norm(difference, axis=1))))
    coordinate_path = arguments.output / "glass_final_coordinates.txt"
    np.savetxt(coordinate_path, positions, fmt="%.17g")
    repository = Path(__file__).resolve().parents[3]
    specification = importlib.util.spec_from_file_location(
        "independent_ewald", repository / "tests/analyze_glass_stability.py")
    independent = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(independent)
    final, final_values = independent.analyze("glass_final", positions, arguments.output, support, False)
    initial = json.loads((arguments.output / "glass_summary.json").read_text())
    with (arguments.output / "glass_eigenvalues.csv").open() as source:
        initial_values = np.array([float(row["eigenvalue"]) for row in csv.DictReader(source)])
    comparison = dict(
        production_particles_path=str(path.resolve()),
        production_particles_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        final_coordinates_sha256=hashlib.sha256(coordinate_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        independent_ewald_script_sha256=hashlib.sha256(Path(specification.origin).read_bytes()).hexdigest(),
        chosen_run=chosen_run, a_final=float(requests[chosen_run]["a_final"]),
        baseline_repeats=len(final_positions), maximum_repeat_position_difference=repeat_discrepancy,
        raw_validation=dict(rows=len(seen), expected_rows=expected_rows,
                            maximum_coordinate_displacement_residual=maximum_coordinate_residual),
        production_exit_status=dict(captured_success=captured_success,
                                    completion_with_uncaptured_child_exit=completed_uncaptured_exit,
                                    runner_session_exit_code=command.get("runner_session_exit_code"),
                                    child_return_code=command.get("child_return_code", command.get("return_code"))),
        interpretation="Frozen full Jacobians at initial and final unperturbed loads, not the full time-dependent propagator",
        ewald_convergence="Uses the partition already verified on initial glass and lattice; final translation, symmetry, trace and inactive-support checks repeated",
        initial=initial, final=final)
    (arguments.output / "glass_initial_final_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    figure, axes = independent.plt.subplots(1, 2, figsize=(10., 4.), constrained_layout=True)
    for label, values in (("Initial glass", initial_values), ("Final unperturbed glass", final_values)):
        axes[0].hist(values, bins=70, density=True, histtype="step", label=label)
        axes[1].plot(values, (np.arange(len(values)) + 1.) / len(values), label=label)
    axes[0].set_ylabel("Eigenvalue density")
    axes[1].set_ylabel("Cumulative fraction")
    for axis in axes:
        axis.set_xlabel(r"Normalized force-Jacobian eigenvalue $\lambda$")
        axis.axvline(1., color="gray", linestyle="--", linewidth=.8)
        axis.grid(alpha=.2)
        axis.legend()
    figure.suptitle(f"Frozen Newtonian spectra of the {count}-particle glass")
    figure.savefig(arguments.output / "glass_initial_final_spectrum.png", dpi=180)
    independent.plt.close(figure)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
