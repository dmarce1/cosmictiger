#!/usr/bin/env python3
# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# Distributed under the GNU General Public License, version 2 or later.

"""Recreate the study comparison from the accompanying measured CSV tables."""

import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmictiger_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CASES = (("adaptive_mixed_n8", "Adaptive gas + DM, N=8", "o-"),
         ("adaptive_dm_n8", "Adaptive DM, N=8", "s-"),
         ("fixed_mixed_n8", "Fixed gas + DM, matched support", "D-"),
         ("adaptive_mixed_n16", "Adaptive gas + DM, N=16", "^"))


def read_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_rows(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    figure.suptitle("Measured pressure-free particle growth: Ferrers F2 gravity\n"
                   "Axis modes; density correction targets z=9 from z=49", fontsize=12)
    all_rows = []
    quality = []
    case_rows = {}
    for case, label, style in CASES:
        rows = read_rows(ROOT / case / "linear_growth_corrections.csv")
        metadata = json.loads((ROOT / case / "linear_growth_metadata.json").read_text())
        case_rows[case] = rows
        all_rows.extend(dict(case=case, **row) for row in rows)
        axis_rows = [row for row in rows if int(row["mode_y"]) == int(row["mode_z"]) == 0]
        axis_rows.sort(key=lambda row: float(row["k_over_axis_nyquist"]))
        x = [float(row["k_over_axis_nyquist"]) for row in axis_rows]
        for axis, field in ((axes[0, 0], "eigenvalue"),
                            (axes[0, 1], "initial_displacement_multiplier"),
                            (axes[1, 0], "initial_velocity_per_displacement_multiplier")):
            axis.plot(x, [float(row[field]) for row in axis_rows], style, label=label)
        if case == "fixed_mixed_n8":
            axes[0, 0].plot(x, [float(row["fixed_continuum_transfer_at_same_support"])
                               for row in axis_rows], "k--", label="Fixed continuum W(kH)")
        if case == "adaptive_mixed_n8":
            for species, style in (("dm", "o-"), ("gas", "s-")):
                axes[1, 1].plot(x, [float(row[f"{species}_longitudinal_eigenvector"])
                                   for row in axis_rows], style, label=species.upper())
        entry = dict(case=case, side=metadata["side"], species=metadata["species"],
                     backend=metadata["backend"], theta=metadata["theta"],
                     baseline_force_rms=metadata["baseline_force_rms"])
        for field in ("mass_symmetry_relative_error", "cosine_relative_norm",
                      "amplitude_relative_difference", "eigenvalue_amplitude_difference"):
            values = [float(row[field]) for row in rows if field in row and math.isfinite(float(row[field]))]
            entry["maximum_" + field] = max(values) if values else "not_measured"
        quality.append(entry)
    axes[0, 0].set_ylabel(r"Density-branch force eigenvalue $\lambda$")
    axes[0, 1].set_ylabel("Initial density amplitude multiplier")
    axes[1, 0].set_ylabel(r"Velocity / displacement correction $f_\lambda/f_1$")
    axes[1, 1].set_ylabel("Growing-mode species displacement / total")
    for axis in axes.flat:
        axis.axhline(1., color="gray", linewidth=.7)
        axis.set_xlabel(r"$|\mathbf{k}|/(\pi N/L)$")
        axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize=7)
    axes[1, 1].legend(fontsize=8)
    figure.savefig(ROOT / "linear_growth_comparison.png", dpi=180)
    figure.savefig(ROOT / "linear_growth_comparison.pdf")
    plt.close(figure)
    write_rows(ROOT / "all_corrections.csv", all_rows)
    write_rows(ROOT / "response_quality.csv", quality)
    mixed8 = next(row for row in case_rows["adaptive_mixed_n8"]
                  if tuple(int(row[f"mode_{axis}"]) for axis in "xyz") == (1, 0, 0))
    mixed16 = next(row for row in case_rows["adaptive_mixed_n16"]
                   if tuple(int(row[f"mode_{axis}"]) for axis in "xyz") == (2, 0, 0))
    checks = dict(same_k_spacing_n8_mode1_eigenvalue=float(mixed8["eigenvalue"]),
                  same_k_spacing_n16_mode2_eigenvalue=float(mixed16["eigenvalue"]),
                  same_k_spacing_eigenvalue_difference=float(mixed16["eigenvalue"]) - float(mixed8["eigenvalue"]))
    (ROOT / "grid_comparison.json").write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    main()
