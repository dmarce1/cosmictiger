#!/usr/bin/env python3
"""Compare the two archived, validated one-class glass/lattice experiments."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmictiger_mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path):
    return json.loads(path.read_text())


def mode(row):
    return tuple(int(row[f"mode_{axis}"]) for axis in "xyz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("current", type=Path)
    args = parser.parse_args()
    roots = (args.previous, args.current)
    summaries = [read_json(root / "glass_lattice_comparison.json") for root in roots]
    metadata = [read_json(root / "adaptive_lattice" / "linear_growth_metadata.json") for root in roots]
    fixed_keys = ("side", "species", "mass_dm", "mass_gas", "gm", "softening_mode", "theta", "backend",
                  "omega_m", "omega_r", "omega_k", "omega_lambda", "pressure_enabled")
    if any(metadata[0][key] != metadata[1][key] for key in fixed_keys):
        raise ValueError("experiments differ in more than the intended kernel and neighbor target")
    if [(item["kernel_family"], item["kernel_degree"], item["neighbor_number"]) for item in metadata] != [
            ("ferrers", 4, 24), ("ferrers", 8, 12)]:
        raise ValueError("expected old F2/24 and new F4/12 metadata")
    if (args.previous / "glass.txt").read_bytes() != (args.current / "glass.txt").read_bytes():
        raise ValueError("glass coordinates differ between experiments")
    rows = []
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for load, color, marker in (("lattice", "C1", "s"), ("glass", "C0", "o")):
        old_growth, new_growth = [{mode(row): row for row in summary["growth"][load]} for summary in summaries]
        if set(old_growth) != set(new_growth):
            raise ValueError("growth wavevectors differ between experiments")
        for wave in sorted(old_growth):
            old, new = old_growth[wave], new_growth[wave]
            if any(old[key] != new[key] for key in ("initial_amplitude_spacing", "a_initial", "a", "steps")):
                raise ValueError("growth intervals, amplitudes, or step counts differ")
            old_deficit = 1. - old["density_parallel_ratio"]
            new_deficit = 1. - new["density_parallel_ratio"]
            rows.append(dict(load=load, mode_x=wave[0], mode_y=wave[1], mode_z=wave[2],
                             k_over_nyquist=2. * np.linalg.norm(wave) / metadata[0]["side"],
                             previous_density_ratio=old["density_parallel_ratio"],
                             current_density_ratio=new["density_parallel_ratio"],
                             density_ratio_change=new["density_parallel_ratio"] - old["density_parallel_ratio"],
                             previous_growth_deficit=old_deficit, current_growth_deficit=new_deficit,
                             fraction_of_previous_deficit_remaining=new_deficit / old_deficit,
                             previous_nonmode_relative_rms=old["nonmode_displacement_relative_rms"],
                             current_nonmode_relative_rms=new["nonmode_displacement_relative_rms"]))
        for summary, label, style in zip(summaries, ("F2 / 24", "F4 / 12"), ("--", "-")):
            force = summary["axis_response"][load]
            growth = sorted(summary["growth"][load], key=mode)
            axes[0].plot([row["k_over_nyquist"] for row in force], [row["mean"] for row in force],
                         color=color, marker=marker, linestyle=style, label=f"{load}: {label}")
            for axis, key in zip(axes[1:], ("density_parallel_ratio", "nonmode_displacement_relative_rms")):
                axis.plot([2. * np.linalg.norm(mode(row)) / metadata[0]["side"] for row in growth],
                          [row[key] for row in growth], color=color, marker=marker, linestyle=style)
    for axis in axes:
        axis.set_xlabel("k / kNy")
        axis.grid(alpha=.2)
    for axis in axes[:2]:
        axis.axhline(1., color=".4", linestyle=":")
    axes[0].set(ylabel="Longitudinal force response", title="Mean over three coordinate axes")
    axes[0].legend(fontsize=8)
    axes[1].set(ylabel="Final paired density / continuum", title="Growth: a = 0.02 → 0.1, x-axis modes")
    axes[2].set(ylabel="Displacement outside mode / continuum target", title="Differential motion outside fitted mode")
    figure.suptitle("Narrower adaptive softening: same 512-particle glass and lattice")
    figure.savefig(args.current / "kernel_comparison.png", dpi=180)
    plt.close(figure)
    (args.current / "kernel_comparison.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    with (args.current / "kernel_comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    controls = []
    for load in ("lattice", "glass"):
        main_row = next(row for row in summaries[1]["growth"][load] if mode(row) == (3, 0, 0))
        cases = [("main_8_steps", main_row)]
        for directory, name in ((f"control_16_{load}", "16_steps"), (f"control_half_{load}", "half_amplitude")):
            data = read_json(args.current / directory / "analysis" / "growth_summary.json")
            control_keys = fixed_keys + ("kernel_family", "kernel_degree", "neighbor_number")
            if any(data["metadata"][key] != metadata[1][key] for key in control_keys):
                raise ValueError("control force settings or background differ from main run")
            runs = data["runs"]
            if len(runs) != 1 or mode(runs[0]) != mode(main_row):
                raise ValueError("control must contain exactly the shortest x-axis mode")
            control = runs[0]
            if any(control[key] != main_row[key] for key in ("a_initial", "a")):
                raise ValueError("control interval differs from main run")
            expected = (16, .001) if name == "16_steps" else (8, .0005)
            if (control["steps"], control["initial_amplitude_spacing"]) != expected:
                raise ValueError("unexpected control step count or amplitude")
            cases.append((name, control))
        for name, control in cases:
            shift = control["density_parallel_ratio"] - main_row["density_parallel_ratio"]
            controls.append(dict(load=load, case=name, steps=control["steps"],
                                 initial_amplitude_spacing=control["initial_amplitude_spacing"],
                                 density_parallel_ratio=control["density_parallel_ratio"],
                                 absolute_shift_from_main=shift,
                                 relative_shift_from_main=shift / main_row["density_parallel_ratio"],
                                 density_phase_ratio=control["density_phase_ratio"],
                                 nonmode_displacement_relative_rms=control["nonmode_displacement_relative_rms"],
                                 even_displacement_relative_rms=control["even_displacement_relative_rms"]))
    (args.current / "growth_controls.json").write_text(json.dumps(controls, indent=2, allow_nan=False) + "\n")
    with (args.current / "growth_controls.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(controls[0]))
        writer.writeheader()
        writer.writerows(controls)
    orientations = []
    for load in ("lattice", "glass"):
        main_row = next(row for row in summaries[1]["growth"][load] if mode(row) == (3, 0, 0))
        data = read_json(args.current / f"oblique_{load}" / "analysis" / "growth_summary.json")
        control_keys = fixed_keys + ("kernel_family", "kernel_degree", "neighbor_number")
        if any(data["metadata"][key] != metadata[1][key] for key in control_keys):
            raise ValueError("oblique force settings or background differ from axis run")
        if len(data["runs"]) != 1 or mode(data["runs"][0]) != (2, 2, 1):
            raise ValueError("expected one (2,2,1) oblique growth run")
        tilted = data["runs"][0]
        if any(tilted[key] != main_row[key] for key in ("a_initial", "a", "initial_amplitude_spacing", "steps")):
            raise ValueError("axis and oblique growth settings differ")
        response = read_json(args.current / f"adaptive_{load}" / "analysis" / "response_summary.json")["modes"]
        for label, row, directory in (("axis", main_row, f"adaptive_{load}"), ("oblique", tilted, f"oblique_{load}")):
            force = next(item for item in response if mode(item) == mode(row) and item["amplitude_spacing"] == .005)
            with (args.current / directory / "linear_growth_evolution.csv").open() as source:
                final = [sample for sample in csv.DictReader(source) if int(sample["run_index"]) == row["run_index"] and
                         int(sample["step"]) == row["steps"] and int(sample["species"]) == 0]
            if len(final) != 3 or {int(sample["axis"]) for sample in final} != {0, 1, 2}:
                raise ValueError("missing complete final displacement vector")
            displacement = np.asarray([float(sample["u_sin"]) for sample in sorted(final, key=lambda sample: int(sample["axis"]))])
            direction = np.asarray(mode(row), dtype=float) / np.linalg.norm(mode(row))
            transverse = np.linalg.norm(displacement - np.dot(displacement, direction) * direction)
            target = row["initial_amplitude_spacing"] * row["continuum_growth"]
            orientations.append(dict(load=load, direction=label, mode_x=mode(row)[0], mode_y=mode(row)[1], mode_z=mode(row)[2],
                                     longitudinal_force_response=force["longitudinal_sine"],
                                     transverse_force_response=force["transverse_response_norm"],
                                     density_parallel_ratio=row["density_parallel_ratio"],
                                     density_phase_ratio=row["density_phase_ratio"],
                                     transverse_sine_displacement_relative=float(transverse / target),
                                     nonmode_displacement_relative_rms=row["nonmode_displacement_relative_rms"]))
    (args.current / "oblique_comparison.json").write_text(json.dumps(orientations, indent=2, allow_nan=False) + "\n")
    with (args.current / "oblique_comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(orientations[0]))
        writer.writeheader()
        writer.writerows(orientations)
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
