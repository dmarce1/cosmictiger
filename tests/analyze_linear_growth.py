#!/usr/bin/env python3
# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# Distributed under the Boost Software License, Version 1.0.

"""Analyze measured pressure-free particle growth and prepare IC corrections.

The C++ linearGrowth diagnostic measures g = 4*pi*G*rho_comoving*M*u.
This script solves the cosmological linear ODE independently of the particle
integrator. Corrections refer to the mass-weighted particle-center density;
they are not an extended-density deconvolution or a second-order LPT scheme.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmictiger_mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import spherical_jn


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def species_index(value):
    return {"dm": 0, "dark_matter": 0, "gas": 1}.get(value, int(value) if value.isdigit() else -1)


def mode_key(row):
    return tuple(int(row[f"mode_{axis}"]) for axis in "xyz")


def ferrers_f2_transfer(argument):
    if abs(argument) < 1.e-4:
        return 1. - argument**2 / 18. + argument**4 / 792.
    return 105. * spherical_jn(3, argument) / argument**3


def existing_ic_growth(matter, scale):
    omega_m = matter / (matter + scale**3 * (1. - matter))
    omega_lambda = 1. - omega_m
    return scale * 2.5 * omega_m / (omega_m**(4. / 7.) - omega_lambda +
                                   (1. + .5 * omega_m) * (1. + omega_lambda / 70.))


def normalize_metadata(metadata):
    metadata = dict(metadata)
    metadata["species_count"] = int(metadata.get("species_count", metadata.get("species", 1)))
    metadata["n_per_dimension"] = int(metadata.get("n_per_dimension", metadata.get("side")))
    if "mass_fractions" not in metadata:
        masses = [float(metadata["mass_dm"])]
        if metadata["species_count"] == 2:
            masses.append(float(metadata["mass_gas"]))
        metadata["mass_fractions"] = [mass / sum(masses) for mass in masses]
    return metadata


class Background:
    def __init__(self, metadata):
        self.matter = float(metadata["omega_m"])
        self.radiation = float(metadata["omega_r"])
        self.curvature = float(metadata["omega_k"])
        self.dark_energy = float(metadata["omega_lambda"])
        if self.matter <= 0 or self.radiation < 0:
            raise ValueError("growth requires positive matter and nonnegative radiation")

    def coefficients(self, log_a):
        a = math.exp(log_a)
        terms = np.array([self.radiation / a**4, self.matter / a**3,
                          self.curvature / a**2, self.dark_energy])
        total = float(terms.sum())
        if total <= 0:
            raise ValueError("background H squared must be positive")
        friction = 2.0 + float(terms @ np.array([-4., -3., -2., 0.])) / (2.0 * total)
        source = 1.5 * terms[1] / total
        return friction, source

    def growing_solution(self, eigenvalue, a_initial, a_final):
        # The regular smooth-radiation solution is a generalized Meszaros branch.
        # Radiation perturbations and baryon pressure are deliberately excluded.
        if self.radiation > 0:
            a_equal = self.radiation / self.matter
            seed_a = min(a_initial * 1.e-6, a_equal * 1.e-6)
            first_order = 1.5 * eigenvalue * seed_a / a_equal
            seed = (1.0 + first_order, first_order)
        else:
            if 1.0 + 24.0 * eigenvalue < 0:
                raise ValueError("no real power-law growing branch for this eigenvalue")
            exponent = (-1.0 + math.sqrt(1.0 + 24.0 * eigenvalue)) / 4.0
            seed_a = a_initial * 1.e-5
            seed = (1.0, exponent)

        def rhs(log_a, state):
            friction, source = self.coefficients(log_a)
            return (state[1], source * eigenvalue * state[0] - friction * state[1])

        solution = solve_ivp(rhs, (math.log(seed_a), math.log(a_final)), seed,
                             rtol=2.e-11, atol=2.e-13, dense_output=True)
        if not solution.success:
            raise RuntimeError(solution.message)
        return solution

    def matrix_solution(self, matrix, initial_u, initial_v, a_initial, a_final):
        dimension = len(initial_u)

        def rhs(log_a, state):
            friction, source = self.coefficients(log_a)
            return np.concatenate((state[dimension:],
                                   source * matrix @ state[:dimension] - friction * state[dimension:]))

        solution = solve_ivp(rhs, (math.log(a_initial), math.log(a_final)),
                             np.concatenate((initial_u, initial_v)), rtol=2.e-11,
                             atol=2.e-13, dense_output=True)
        if not solution.success:
            raise RuntimeError(solution.message)
        return solution


def response_matrices(rows, species_count):
    grouped = {}
    for row in rows:
        key = (mode_key(row), float(row["amplitude_spacing"]))
        grouped.setdefault(key, []).append(row)
    result = {}
    dimension = 3 * species_count
    for key, entries in grouped.items():
        matrix = np.full((dimension, dimension), np.nan + 1j * np.nan)
        for row in entries:
            output = 3 * species_index(row["output_species"]) + int(row["output_axis"])
            input_ = 3 * species_index(row["input_species"]) + int(row["input_axis"])
            if min(input_, output) < 0:
                raise ValueError("unknown species identifier")
            if np.isfinite(matrix[output, input_]):
                raise ValueError(f"duplicate response matrix entry {key}, {output}, {input_}")
            matrix[output, input_] = float(row["response_sin"]) + 1j * float(row["response_cos"])
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"incomplete or nonfinite matrix for {key}")
        result[key] = matrix
    return result


def analyze_modes(metadata, matrices, a_initial, a_final):
    species_count = int(metadata["species_count"])
    fractions = np.asarray(metadata["mass_fractions"][:species_count], dtype=float)
    if np.any(fractions <= 0) or abs(fractions.sum() - 1.0) > 1.e-9:
        raise ValueError("mass fractions must be positive and sum to one")
    weights = np.repeat(np.sqrt(fractions), 3)
    background = Background(metadata)
    continuum = background.growing_solution(1., a_initial, a_final)
    continuum_start = continuum.sol(math.log(a_initial))
    continuum_end = continuum.sol(math.log(a_final))
    continuum_growth = continuum_end[0] / continuum_start[0]
    f_continuum_start = continuum_start[1] / continuum_start[0]
    f_continuum_end = continuum_end[1] / continuum_end[0]
    # initialize.cpp currently uses this flat matter+Lambda approximation even
    # when the runtime background contains radiation. Keep that difference visible.
    legacy_omega_m = background.matter / (background.matter +
                                         a_initial**3 * (1.0 - background.matter))
    legacy_initial_rate = legacy_omega_m**(5. / 9.)
    legacy_hubble_squared = background.matter / a_initial**3 + 1.0 - background.matter
    runtime_hubble_squared = (background.radiation / a_initial**4 +
                              background.matter / a_initial**3 +
                              background.curvature / a_initial**2 + background.dark_energy)
    legacy_hubble_ratio = math.sqrt(legacy_hubble_squared / runtime_hubble_squared)
    # The existing power spectrum convention normalizes D to unity at a=1.
    # Keep the separate old-background correction available without folding it
    # into the measured force correction R in the primary plots and columns.
    continuum_to_today = background.growing_solution(1., a_initial, max(1., a_final))
    continuum_initial_normalized = (continuum_to_today.sol(math.log(a_initial))[0] /
                                    continuum_to_today.sol(0.)[0])
    legacy_initial_normalized = (existing_ic_growth(background.matter, a_initial) /
                                 existing_ic_growth(background.matter, 1.))
    background_displacement_ratio = continuum_initial_normalized / legacy_initial_normalized
    summaries = []
    modes = {}
    for mode in sorted({key[0] for key in matrices}, key=lambda value: (sum(x*x for x in value), value)):
        amplitudes = sorted(key[1] for key in matrices if key[0] == mode)
        amplitude = amplitudes[0]
        measured = matrices[(mode, amplitude)]
        weighted = weights[:, None] * measured / weights[None, :]
        # On this inversion-symmetric lattice the exact matrix is real symmetric
        # in mass-weighted coordinates. Report, rather than conceal, the residuals.
        symmetric = (weighted.real + weighted.real.T) * .5
        values, vectors = np.linalg.eigh(symmetric)
        direction = np.asarray(mode, dtype=float)
        direction /= np.linalg.norm(direction)
        density_projection = np.concatenate([fraction * direction for fraction in fractions])
        physical_vectors = vectors / weights[:, None]
        overlap = np.abs(density_projection @ physical_vectors)**2
        selected = int(np.argmax(overlap))
        eigenvalue = float(values[selected])
        eigenvector = physical_vectors[:, selected]
        eigenvector /= density_projection @ eigenvector
        growing = background.growing_solution(eigenvalue, a_initial, a_final)
        start = growing.sol(math.log(a_initial))
        end = growing.sol(math.log(a_final))
        growth = end[0] / start[0]
        f_start = start[1] / start[0]
        f_end = end[1] / end[0]
        correction = continuum_growth / growth
        scale = max(np.linalg.norm(weighted), 1.e-30)
        amplitude_difference = math.nan
        eigenvalue_amplitude_difference = math.nan
        if len(amplitudes) > 1:
            previous = matrices[(mode, amplitudes[1])]
            amplitude_difference = float(np.linalg.norm(measured - previous) / max(np.linalg.norm(measured), 1.e-30))
            previous_weighted = weights[:, None] * previous.real / weights[None, :]
            previous_values, previous_vectors = np.linalg.eigh((previous_weighted + previous_weighted.T) * .5)
            previous_overlaps = np.abs(density_projection @ (previous_vectors / weights[:, None]))**2
            eigenvalue_amplitude_difference = abs(eigenvalue - float(previous_values[np.argmax(previous_overlaps)]))
        shared = np.tile(direction, species_count)
        summary = dict(mode_x=mode[0], mode_y=mode[1], mode_z=mode[2],
                       k_over_axis_nyquist=2. * np.linalg.norm(mode) / int(metadata["n_per_dimension"]),
                       amplitude_spacing=amplitude, eigenvalue=eigenvalue,
                       eds_growth_exponent=(-1. + math.sqrt(max(0., 1. + 24. * eigenvalue))) / 4.,
                       initial_growth_rate=f_start, continuum_initial_growth_rate=f_continuum_start,
                       existing_ic_initial_growth_rate=legacy_initial_rate,
                       existing_ic_hubble_ratio_to_runtime=legacy_hubble_ratio,
                       continuum_initial_displacement_vs_existing_ic=background_displacement_ratio,
                       final_growth_rate=f_end, continuum_final_growth_rate=f_continuum_end,
                       initial_displacement_multiplier=correction,
                       initial_velocity_per_displacement_multiplier=f_start / f_continuum_start,
                       growth_rate_ratio_to_existing_ic=f_start / legacy_initial_rate,
                       corrected_momentum_per_displacement_vs_existing_ic=f_start /
                           (legacy_initial_rate * legacy_hubble_ratio),
                       target_matched_momentum_multiplier_vs_existing_ic=correction * f_start /
                           (legacy_initial_rate * legacy_hubble_ratio),
                       target_matched_displacement_for_spectrum_normalized_at_a1=correction * background_displacement_ratio,
                       target_matched_momentum_for_spectrum_normalized_at_a1=correction * background_displacement_ratio *
                           f_start / (legacy_initial_rate * legacy_hubble_ratio),
                       final_velocity_ratio_at_matched_density=f_end / f_continuum_end,
                       shared_longitudinal_response=float(density_projection @ measured.real @ shared),
                       cosine_relative_norm=float(np.linalg.norm(weighted.imag) / scale),
                       mass_symmetry_relative_error=float(np.linalg.norm(weighted - weighted.conj().T) / scale),
                       amplitude_relative_difference=amplitude_difference,
                       eigenvalue_amplitude_difference=eigenvalue_amplitude_difference,
                       selected_density_overlap=float(overlap[selected]),
                       transverse_mode_norm=float(np.linalg.norm(eigenvector -
                           np.concatenate([direction * (direction @ eigenvector[3*s:3*s+3])
                                           for s in range(species_count)]))))
        for index, value in enumerate(values):
            summary[f"eigenvalue_{index}"] = float(value)
        if metadata.get("kernel_family") == "ferrers" and int(metadata.get("kernel_degree", 0)) == 4:
            support = float(metadata["baseline_support_dm"])
            argument = 2. * math.pi * float(np.linalg.norm(mode)) * support
            summary["k_support"] = argument
            summary["support_over_spacing"] = support * int(metadata["n_per_dimension"])
            summary["fixed_continuum_transfer_at_same_support"] = ferrers_f2_transfer(argument)
        for species in range(species_count):
            name = ("dm", "gas")[species]
            summary[f"{name}_longitudinal_eigenvector"] = float(direction @ eigenvector[3*species:3*species+3])
            for axis, component in zip("xyz", eigenvector[3*species:3*species+3]):
                summary[f"{name}_eigenvector_{axis}"] = float(component)
        summaries.append(summary)
        modes[mode] = dict(matrix=measured.real, eigenvector=eigenvector, direction=direction,
                           growing=growing, continuum=continuum, f_start=f_start,
                           f_continuum=f_continuum_start, correction=correction,
                           projection=density_projection, summary=summary)
    return summaries, modes, background


def prepare_evolution(modes, selected_modes, amplitude, steps, a_initial, a_final):
    requests = []
    for mode in selected_modes:
        if mode not in modes:
            raise ValueError(f"requested unmeasured mode {mode}")
        result = modes[mode]
        dimension = len(result["eigenvector"])
        shared = np.tile(result["direction"], dimension // 3)
        cases = (("continuum", shared, result["f_continuum"]),
                 ("growing_mode", result["eigenvector"], result["f_start"]),
                 ("target_matched", result["correction"] * result["eigenvector"], result["f_start"]))
        for label, displacement, rate in cases:
            displacement = amplitude * displacement
            derivative = rate * displacement
            row = dict(label=label, mode_x=mode[0], mode_y=mode[1], mode_z=mode[2],
                       a_initial=a_initial, a_final=a_final, steps=steps)
            for species in range(2):
                for field, values in (("u", displacement), ("v", derivative)):
                    for axis_index, axis in enumerate("xyz"):
                        index = 3 * species + axis_index
                        row[f"{field}_{('dm', 'gas')[species]}_{axis}"] = float(values[index]) if index < dimension else 0.
            requests.append(row)
    return requests


def analyze_evolution(path, requests, modes, background, output_directory):
    rows = read_csv(path)
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["run_index"]), []).append(row)
    if set(grouped) != set(range(len(requests))):
        raise ValueError("evolution file does not contain exactly the requested runs")
    summaries = []
    tracks = []
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    figure.suptitle("Pressure-free particle evolution with recomputed gravity\n"
                   "Particle RK4 versus independently integrated linear response", fontsize=11)
    for run_index, entries in sorted(grouped.items()):
        request = requests[run_index]
        mode = mode_key(request)
        result = modes[mode]
        dimension = len(result["eigenvector"])
        initial_u = np.array([float(request[f"u_{('dm', 'gas')[index//3]}_{'xyz'[index%3]}"])
                              for index in range(dimension)])
        initial_v = np.array([float(request[f"v_{('dm', 'gas')[index//3]}_{'xyz'[index%3]}"])
                              for index in range(dimension)])
        a_initial, a_final = float(request["a_initial"]), float(request["a_final"])
        steps = int(request["steps"])
        solution = background.matrix_solution(result["matrix"], initial_u, initial_v, a_initial, a_final)
        grouped_steps = {}
        for row in entries:
            grouped_steps.setdefault(int(row["step"]), []).append(row)
        if set(grouped_steps) != set(range(steps + 1)):
            raise ValueError(f"run {run_index} has missing or unexpected evolution steps")
        maximum_u_error = maximum_v_error = maximum_cosine = maximum_residual = 0.
        run_tracks = []
        for step, values in sorted(grouped_steps.items()):
            state_u = np.full(dimension, np.nan)
            state_v = np.full(dimension, np.nan)
            cosine = np.zeros(dimension)
            seen_components = set()
            expected_a = a_initial * math.exp(step * math.log(a_final / a_initial) / steps)
            for row in values:
                index = 3 * species_index(row["species"]) + int(row["axis"])
                if index < 0 or index >= dimension or index in seen_components:
                    raise ValueError(f"duplicate or invalid component in run {run_index}, step {step}")
                seen_components.add(index)
                if row["label"] != request["label"]:
                    raise ValueError(f"evolution label does not match request {run_index}")
                if not math.isclose(float(row["a"]), expected_a, rel_tol=2.e-13, abs_tol=1.e-15):
                    raise ValueError(f"evolution scale does not match run {run_index}, step {step}")
                state_u[index] = float(row["u_sin"])
                state_v[index] = float(row["v_sin"])
                cosine[index] = float(row["u_cos"])
                if not all(math.isfinite(float(row[key])) for key in
                           ("u_sin", "u_cos", "v_sin", "v_cos", "residual_rms")):
                    raise ValueError("nonfinite evolution sample")
            if seen_components != set(range(dimension)):
                raise ValueError("incomplete evolution sample")
            if step == 0 and (not np.allclose(state_u, initial_u, rtol=1.e-10, atol=1.e-12) or
                              not np.allclose(state_v, initial_v, rtol=1.e-10, atol=1.e-12)):
                raise ValueError(f"initial evolution state does not match request {run_index}")
            a = float(values[0]["a"])
            expected = solution.sol(math.log(a))
            u_scale = max(np.linalg.norm(expected[:dimension]), 1.e-30)
            v_scale = max(np.linalg.norm(expected[dimension:]), 1.e-30)
            u_error = float(np.linalg.norm(state_u - expected[:dimension]) / u_scale)
            v_error = float(np.linalg.norm(state_v - expected[dimension:]) / v_scale)
            maximum_u_error = max(maximum_u_error, u_error)
            maximum_v_error = max(maximum_v_error, v_error)
            maximum_cosine = max(maximum_cosine, float(np.linalg.norm(cosine) / u_scale))
            maximum_residual = max(maximum_residual,
                                   math.sqrt(sum(float(row["residual_rms"])**2 for row in values)) / u_scale)
            continuum = result["continuum"]
            continuum_state = continuum.sol(math.log(a))
            continuum_growth = continuum_state[0] / continuum.sol(math.log(a_initial))[0]
            initial_target = float(result["projection"] @ initial_u)
            if request["label"].startswith("target_matched"):
                initial_target /= result["correction"]
            row = dict(run_index=run_index, label=request["label"], mode_x=mode[0],
                       mode_y=mode[1], mode_z=mode[2], step=step, a=a,
                       measured_total_displacement=float(result["projection"] @ state_u),
                       measured_total_log_derivative=float(result["projection"] @ state_v),
                       ode_total_displacement=float(result["projection"] @ expected[:dimension]),
                       continuum_target_displacement=initial_target * continuum_growth,
                       continuum_target_log_derivative=initial_target * continuum_growth *
                           continuum_state[1] / continuum_state[0],
                       displacement_relative_error=u_error, derivative_relative_error=v_error)
            tracks.append(row)
            run_tracks.append(row)
        final = run_tracks[-1]
        summaries.append(dict(run_index=run_index, label=request["label"], mode_x=mode[0],
                              mode_y=mode[1], mode_z=mode[2], steps=int(request["steps"]),
                              initial_total_displacement=float(result["projection"] @ initial_u),
                              max_displacement_relative_error=maximum_u_error,
                              max_derivative_relative_error=maximum_v_error,
                              max_cosine_relative_norm=maximum_cosine,
                              max_nonmode_displacement_relative_rms=maximum_residual,
                              final_density_ratio_to_continuum=final["measured_total_displacement"] /
                                  final["continuum_target_displacement"],
                              final_momentum_ratio_to_continuum=final["measured_total_log_derivative"] /
                                  final["continuum_target_log_derivative"]))
        x_values = [row["a"] / a_initial for row in run_tracks]
        axes[0].plot(x_values, [row["measured_total_displacement"] / row["continuum_target_displacement"]
                               for row in run_tracks], label=f"{mode} {request['label'].replace('_', ' ')}")
        axes[1].semilogy(x_values[1:], [max(row["displacement_relative_error"], 1.e-12)
                                       for row in run_tracks[1:]])
    axes[0].axhline(1., color="black", linewidth=.8)
    axes[0].set(xlabel=r"$a/a_i$", ylabel="Particle-center density / continuum target")
    axes[1].set(xlabel=r"$a/a_i$", ylabel="Particle vs linear ODE displacement error")
    axes[0].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=.25)
    figure.savefig(output_directory / "linear_growth_evolution.png", dpi=180)
    figure.savefig(output_directory / "linear_growth_evolution.pdf")
    plt.close(figure)
    write_csv(output_directory / "evolution_comparison.csv", tracks)
    write_csv(output_directory / "evolution_summary.csv", summaries)
    return summaries


def plot_corrections(summaries, output_directory, metadata, a_initial, a_final):
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    figure.suptitle(f"Pressure-free particle growth: {metadata.get('softening_mode', 'measured')} gravity, "
                   f"N={metadata['n_per_dimension']}, {metadata['species_count']} species\n"
                   f"Density target: z={1. / a_final - 1.:g} from z={1. / a_initial - 1.:g}", fontsize=11)
    directions = {}
    for row in summaries:
        mode = mode_key(row)
        divisor = math.gcd(*mode)
        direction = tuple(value // divisor for value in mode)
        directions.setdefault(direction, []).append(row)
    for direction, rows in sorted(directions.items()):
        rows.sort(key=lambda row: row["k_over_axis_nyquist"])
        x_values = [row["k_over_axis_nyquist"] for row in rows]
        label = str(direction)
        axes[0].plot(x_values, [row["eigenvalue"] for row in rows], "o-", label=label)
        axes[1].plot(x_values, [row["initial_displacement_multiplier"] for row in rows], "o-", label=label)
        axes[2].plot(x_values, [row["initial_velocity_per_displacement_multiplier"] for row in rows], "o-", label=label)
    for axis in axes:
        axis.axhline(1., color="black", linewidth=.8)
        axis.set_xlabel(r"$|\mathbf{k}|/(\pi N/L)$")
        axis.grid(alpha=.25)
    axes[0].set_ylabel(r"Growing-branch force eigenvalue $\lambda$")
    axes[1].set_ylabel("Initial displacement multiplier")
    axes[2].set_ylabel(r"Velocity / displacement correction $f_\lambda/f_1$")
    axes[0].legend(title="Mode direction", fontsize=8)
    figure.savefig(output_directory / "linear_growth_corrections.png", dpi=180)
    figure.savefig(output_directory / "linear_growth_corrections.pdf")
    plt.close(figure)


def self_check():
    # Exact EdS power laws test normalization and velocity/amplitude conventions.
    metadata = dict(omega_m=1., omega_r=0., omega_k=0., omega_lambda=0.)
    background = Background(metadata)
    for eigenvalue in (.1, .7, 1., 1.3):
        exponent = (-1. + math.sqrt(1. + 24. * eigenvalue)) / 4.
        solution = background.growing_solution(eigenvalue, .02, .2)
        initial = solution.sol(math.log(.02))
        final = solution.sol(math.log(.2))
        np.testing.assert_allclose(final[0] / initial[0], 10.**exponent, rtol=3.e-9)
        np.testing.assert_allclose(initial[1] / initial[0], exponent, rtol=3.e-9)
    # With smooth radiation and no Lambda, lambda=1 has exact regular Meszaros
    # solution 1 + 3*a/(2*a_eq), including its logarithmic derivative.
    metadata.update(omega_m=.9999, omega_r=.0001)
    background = Background(metadata)
    solution = background.growing_solution(1., .02, .2)
    for a in (.02, .2):
        expected_derivative = 1.5 * a * background.matter / background.radiation
        np.testing.assert_allclose(solution.sol(math.log(a)),
                                   [1. + expected_derivative, expected_derivative], rtol=3.e-9)
    # The independent radial integral fixes the single-kernel normalization;
    # convolving two extended clouds instead would give its square.
    from scipy.integrate import quad
    for argument in (.001, .1, 1., 3., 6.):
        radial = 105. / 8. * quad(lambda radius: radius**2 * (1. - radius**2)**2 *
                                 np.sinc(argument * radius / math.pi), 0., 1., epsabs=1.e-13)[0]
        np.testing.assert_allclose(ferrers_f2_transfer(argument), radial, atol=3.e-14, rtol=3.e-14)
    print("Linear-growth analysis self-checks passed (EdS, regular Meszaros, Ferrers Fourier kernel).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response", type=Path, nargs="?")
    parser.add_argument("metadata", type=Path, nargs="?")
    parser.add_argument("output_directory", type=Path, nargs="?")
    parser.add_argument("--a-initial", type=float, default=.02)
    parser.add_argument("--a-final", type=float, default=.2)
    parser.add_argument("--evolution-modes", default="1,0,0;3,0,0")
    parser.add_argument("--evolution-amplitude", type=float, default=.002)
    parser.add_argument("--evolution-steps", type=int, default=32)
    parser.add_argument("--evolution", type=Path)
    parser.add_argument("--evolution-requests", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        if args.response is None:
            return
    if None in (args.response, args.metadata, args.output_directory):
        parser.error("response, metadata and output_directory are required")
    if not 0. < args.a_initial < args.a_final:
        parser.error("require 0 < a_initial < a_final")
    metadata = normalize_metadata(json.loads(args.metadata.read_text()))
    args.output_directory.mkdir(parents=True, exist_ok=True)
    matrices = response_matrices(read_csv(args.response), int(metadata["species_count"]))
    summaries, modes, background = analyze_modes(metadata, matrices, args.a_initial, args.a_final)
    write_csv(args.output_directory / "linear_growth_corrections.csv", summaries)
    plot_corrections(summaries, args.output_directory, metadata, args.a_initial, args.a_final)
    selected_modes = [tuple(int(value) for value in mode.split(","))
                      for mode in args.evolution_modes.split(";") if mode]
    requests = prepare_evolution(modes, selected_modes, args.evolution_amplitude,
                                 args.evolution_steps, args.a_initial, args.a_final)
    request_path = args.output_directory / "linear_growth_evolution_requests.csv"
    write_csv(request_path, requests)
    if args.evolution:
        if args.evolution_requests:
            requests = read_csv(args.evolution_requests)
        analyze_evolution(args.evolution, requests, modes, background, args.output_directory)
    (args.output_directory / "analysis_parameters.json").write_text(json.dumps(dict(
        a_initial=args.a_initial, a_final=args.a_final,
        response_file=str(args.response.resolve()), metadata_file=str(args.metadata.resolve()),
        evolution_file=str(args.evolution.resolve()) if args.evolution else None,
        evolution_requests_file=str(args.evolution_requests.resolve()) if args.evolution_requests else
            (str(request_path.resolve()) if selected_modes else None),
        correction_target="mass-weighted particle-center density at a_final",
        growth_scope="pressure-free matter with smooth radiation background; linear only",
        matrix_symmetrization="real mass-weighted Hermitian part; residuals in CSV",
        velocity_definition="stored p=a^2 H du/dln(a)"), indent=2) + "\n")
    print(f"Analyzed {len(summaries)} modes" +
          (f"; wrote {request_path}" if selected_modes else "; no evolution requests generated"))


if __name__ == "__main__":
    main()
