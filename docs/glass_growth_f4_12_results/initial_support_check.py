#!/usr/bin/env python3
"""Independently solve the initial periodic number constraint for both loads.

Run with one thread from any directory:
    python3 -B initial_support_check.py

Requires numpy and scipy. No production kernel or support-solver code is used.
"""

import hashlib
import json
import math
import os
from fractions import Fraction
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.optimize import brentq


def statistics(values):
    return {
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
        "standard_deviation": float(np.std(values)),
    }


def self_count(exponent):
    # (4*pi/3)*Gamma(p+5/2)/(pi**(3/2)*Gamma(p+1)).
    # The equivalent factorial expression avoids gamma roundoff here.
    return float(Fraction(
        4 * math.factorial(2 * exponent + 4),
        3 * 4 ** (exponent + 2) * math.factorial(exponent + 2)
        * math.factorial(exponent),
    ))


def metadata_for(archive, load):
    direct = archive / f"adaptive_{load}" / "linear_growth_metadata.json"
    summary = archive / f"adaptive_{load}" / "analysis" / "response_summary.json"
    path = direct if direct.exists() else summary
    data = json.loads(path.read_text())
    return path, data if path == direct else data["metadata"]


def evaluate(coordinates, side, exponent, target, metadata):
    difference = coordinates[:, None, :] - coordinates[None, :, :]
    difference -= np.rint(difference)
    radii = np.linalg.norm(difference, axis=-1) * side
    floor = self_count(exponent)
    assert target > floor

    def count(radius, support):
        return floor * np.sum(np.maximum(0.0, 1 - (radius / support) ** 2) ** exponent)

    support = np.array([
        brentq(lambda value: count(radius, value) - target, 1e-6,
               0.5 * side, xtol=1e-13)
        for radius in radii
    ])
    q_squared = (radii / support[:, None]) ** 2
    derivative = np.sum(
        floor * 2 * exponent * q_squared
        * np.maximum(0.0, 1 - q_squared) ** (exponent - 1), axis=1,
    )
    other_count = np.sum(radii < support[:, None], axis=1) - 1
    assert np.min(derivative) > 0
    residual = np.array([count(radius, value) - target
                         for radius, value in zip(radii, support)])
    maximum_residual = float(np.max(np.abs(residual)))
    assert maximum_residual < 1e-10

    # Production metadata stores radii in box units, not spacing units.
    assert metadata["side"] == side
    assert metadata["species"] == 1
    assert metadata["kernel_family"] == "ferrers"
    assert metadata["kernel_degree"] == 2 * exponent
    assert metadata["neighbor_number"] == target
    assert metadata["softening_mode"] == "adaptive"
    expected = {
        "mean": metadata["baseline_support_dm"] * side,
        "minimum": metadata["baseline_min_support"] * side,
        "maximum": metadata["baseline_max_support"] * side,
    }
    measured = statistics(support)
    errors = {key: measured[key] - value for key, value in expected.items()}
    # Production stores each solved radius as float; independent solve is double.
    tolerance = 5e-7
    assert max(abs(value) for value in errors.values()) < tolerance
    return {
        "ferrers_exponent": exponent,
        "density_polynomial_degree": 2 * exponent,
        "weighted_target": target,
        "self_weight": floor,
        "self_fraction_of_target": floor / target,
        "support_over_spacing": measured,
        "effective_width_2sigma_over_spacing": statistics(
            2 * support / math.sqrt(2 * exponent + 5)),
        "actual_other_neighbors_inside_support": statistics(other_count),
        "constraint_derivative_dC_dlnH": statistics(derivative),
        "maximum_absolute_constraint_residual": maximum_residual,
        "metadata_support_over_spacing": expected,
        "independent_minus_metadata_support_over_spacing": errors,
        "metadata_absolute_agreement_tolerance": tolerance,
        "metadata_agreement_passed": True,
    }


def main():
    archive = Path(__file__).resolve().parent
    previous = archive.parent / "glass_growth_results"
    glass_path = archive / "glass.txt"
    coordinates = np.loadtxt(glass_path)
    side = 8
    assert coordinates.shape == (side ** 3, 3)
    assert np.all((coordinates >= 0) & (coordinates < 1))
    grid = np.indices((side, side, side)).reshape(3, -1).T / side
    result = {
        "description": "Independent analytic periodic all-pairs number-constraint solve",
        "kernel": "w(q) proportional to (1-q^2)^p for q<1; q=r/H",
        "constraint": "C=(4*pi/3)*sum_j w(r_ij/H), including self",
        "effective_width": "2sigma=2H/sqrt(2p+5)",
        "particle_count": side ** 3,
        "side": side,
        "spacing": 1.0 / side,
        "glass_sha256": hashlib.sha256(glass_path.read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "measurements": {},
    }
    for exponent, target, source in ((2, 24, previous), (4, 12, archive)):
        for load, points in (("lattice", grid), ("glass", coordinates)):
            path, metadata = metadata_for(source, load)
            measurement = evaluate(points, side, exponent, target, metadata)
            measurement["metadata_source"] = os.path.relpath(path, archive)
            measurement["metadata_source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            result["measurements"][f"f{exponent}_{target}_{load}"] = measurement
    result["width_ratios_f4_12_over_f2_24"] = {
        load: result["measurements"][f"f4_12_{load}"]["effective_width_2sigma_over_spacing"]["mean"]
        / result["measurements"][f"f2_24_{load}"]["effective_width_2sigma_over_spacing"]["mean"]
        for load in ("lattice", "glass")
    }
    output = archive / "initial_support_check.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Four independent support solves agree with production metadata: {output}")


if __name__ == "__main__":
    main()
