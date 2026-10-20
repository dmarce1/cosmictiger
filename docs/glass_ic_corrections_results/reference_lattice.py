#!/usr/bin/env python3
"""Extract the earlier matching pure-DM lattice IC correction reference.

Independently verify the archived corrections with the Riccati equation for
f = d ln(D) / d ln(a). This differs from the archived analyzer's direct
second-order D equation. The regular smooth-radiation growing branch is used.
Run from any directory; the CSV and JSON are written beside this script.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

from scipy.integrate import solve_ivp


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def growth(eigenvalue, metadata, a_initial, a_final):
    matter = metadata["omega_m"]
    radiation = metadata["omega_r"]
    curvature = metadata["omega_k"]
    dark_energy = metadata["omega_lambda"]
    equality = radiation / matter
    seed_scale = min(a_initial * 1.e-8, equality * 1.e-8)
    seed_f = 1.5 * eigenvalue * seed_scale / equality

    def rhs(log_scale, state):
        scale = math.exp(log_scale)
        # Multiplying H^2 by a^4 avoids unnecessarily large early-time terms.
        total = (radiation + matter * scale + curvature * scale**2 +
                 dark_energy * scale**4)
        friction = (matter * scale + 2. * curvature * scale**2 +
                    4. * dark_energy * scale**4) / (2. * total)
        source = 1.5 * matter * scale / total
        logarithmic_growth, rate = state
        return rate, source * eigenvalue - rate * rate - friction * rate

    solution = solve_ivp(rhs, (math.log(seed_scale), math.log(a_final)),
                         (0., seed_f), method="DOP853", rtol=2.e-12,
                         atol=2.e-14, dense_output=True)
    if not solution.success:
        raise RuntimeError(solution.message)
    initial = solution.sol(math.log(a_initial))
    final = solution.sol(math.log(a_final))
    return dict(growth=math.exp(final[0] - initial[0]),
                initial_rate=float(initial[1]), final_rate=float(final[1]))


def main():
    output_directory = Path(__file__).resolve().parent
    repository = output_directory.parents[1]
    source_directory = repository / "docs/linear_growth_results/adaptive_dm_n8"
    metadata_path = source_directory / "linear_growth_metadata.json"
    corrections_path = source_directory / "linear_growth_corrections.csv"
    parameters_path = source_directory / "analysis_parameters.json"
    metadata = json.loads(metadata_path.read_text())
    parameters = json.loads(parameters_path.read_text())
    assert (metadata["side"], metadata["species"], metadata["kernel_family"],
            metadata["kernel_degree"], metadata["softening_mode"],
            metadata["neighbor_number"]) == (8, 1, "ferrers", 4, "adaptive", 24)
    a_initial = parameters["a_initial"]
    a_final = parameters["a_final"]
    assert (a_initial, a_final) == (.02, .1)
    continuum = growth(1., metadata, a_initial, a_final)
    with corrections_path.open(newline="") as source:
        source_rows = list(csv.DictReader(source))
    rows = []
    maximum_discrepancy = 0.
    for mode in (1, 2, 3):
        source = next(row for row in source_rows
                      if (int(row["mode_x"]), int(row["mode_y"]),
                          int(row["mode_z"])) == (mode, 0, 0))
        eigenvalue = float(source["eigenvalue"])
        lattice = growth(eigenvalue, metadata, a_initial, a_final)
        independent_displacement = continuum["growth"] / lattice["growth"]
        independent_rate_ratio = lattice["initial_rate"] / continuum["initial_rate"]
        displacement = float(source["initial_displacement_multiplier"])
        rate_ratio = float(source["initial_velocity_per_displacement_multiplier"])
        displacement_error = independent_displacement / displacement - 1.
        rate_error = independent_rate_ratio / rate_ratio - 1.
        maximum_discrepancy = max(maximum_discrepancy,
                                  abs(displacement_error), abs(rate_error))
        rows.append(dict(
            mode_x=mode, mode_y=0, mode_z=0,
            k_over_axis_nyquist=float(source["k_over_axis_nyquist"]),
            force_eigenvalue=eigenvalue,
            displacement_multiplier=displacement,
            velocity_per_displacement_multiplier=rate_ratio,
            total_initial_velocity_multiplier=displacement * rate_ratio,
            initial_growth_rate=float(source["initial_growth_rate"]),
            continuum_initial_growth_rate=float(source["continuum_initial_growth_rate"]),
            final_velocity_ratio_at_matched_density=float(
                source["final_velocity_ratio_at_matched_density"]),
            independent_displacement_multiplier=independent_displacement,
            independent_velocity_per_displacement_multiplier=independent_rate_ratio,
            independent_displacement_relative_difference=displacement_error,
            independent_rate_ratio_relative_difference=rate_error))
    if maximum_discrepancy > 1.e-8:
        raise RuntimeError(f"Independent integration mismatch: {maximum_discrepancy}")
    with (output_directory / "reference_lattice.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        source="Earlier pure-DM adaptive lattice calibration; not the mixed gas/DM table",
        particles=metadata["side"]**3,
        a_initial=a_initial, a_final=a_final,
        kernel="(1-q^2)^2", weighted_neighbors=24,
        adaptive_conservation_terms=True,
        force_backend=metadata["backend"],
        target="particle-center density amplitude at a_final",
        reference="same-background continuum growing IC",
        velocity_definition="p=a^2 H d(Psi)/dln(a); multiplier also applies to peculiar velocity at fixed a",
        correction_equations=dict(
            displacement="R = G_continuum / G_lattice",
            velocity_per_displacement="f_lattice(a_initial) / f_continuum(a_initial)",
            total_initial_velocity="R * f_lattice(a_initial) / f_continuum(a_initial)"),
        continuum_growth=continuum,
        independent_check="DOP853 integration of f'=1.5 Omega_m lambda-f^2-(2+dlnH/dlna)f and lnD'=f",
        maximum_independent_relative_difference=maximum_discrepancy,
        input_sha256={str(path.relative_to(repository)): file_hash(path)
                      for path in (metadata_path, corrections_path, parameters_path)},
        rows=rows)
    (output_directory / "reference_lattice.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
