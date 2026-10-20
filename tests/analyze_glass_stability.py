#!/usr/bin/env python3
"""Independent periodic Newtonian force Jacobian for a saved particle load.

The force is the Ewald solution of Poisson's equation with the mean density
removed. Equal masses sum to one in a unit periodic box and G=1. The reported
Jacobian eigenvalues are normalized by 4*pi*G*rho. Compact gravitational
softening is inactive provided every pair is farther apart than its support.

All BLAS libraries are restricted to one thread before NumPy is imported.
This is an instantaneous, frozen-load linearization, not an equilibrium or
time-integrated stability proof for a glass with nonzero residual force.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmictiger_mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.special import erfc


def reciprocalModes(alpha, cutoff):
    # One of each +/- pair, with the corresponding factor of two in weights.
    indices = np.array([index for index in itertools.product(
        range(-cutoff, cutoff + 1), repeat=3)
        if index > (0, 0, 0)], dtype=float)
    wavevectors = 2. * np.pi * indices
    squares = np.einsum("ka,ka->k", wavevectors, wavevectors)
    weights = 8. * np.pi * np.exp(-squares / (4. * alpha**2)) / squares
    return wavevectors, weights


def ewald(positions, alpha=5., realCutoff=1, reciprocalCutoff=10,
          withJacobian=True):
    """Return attraction acceleration and d(acceleration)/d(position).

    Real-space pair force F(r)=-A(r)r, with
      A=erfc(alpha*r)/r^3+2*alpha*exp(-alpha^2*r^2)/(sqrt(pi)*r^2).
    Its derivative is B*r*r^T-A*I, where
      B=(3*A+4*alpha^3*exp(-alpha^2*r^2)/sqrt(pi))/r^2.
    Reciprocal force and derivative follow by differentiating the cosine
    Fourier potential. Self particles and their periodic images exert no net
    force and cannot change their own relative coordinates under displacement.
    """
    count = len(positions)
    mass = 1. / count
    differences = positions[:, None, :] - positions[None, :, :]
    differences -= np.rint(differences)
    pairForce = np.zeros((count, count, 3))
    pairTensor = np.zeros((count, count, 3, 3)) if withJacobian else None
    diagonal = np.arange(count)
    for shift in itertools.product(range(-realCutoff, realCutoff + 1), repeat=3):
        displacement = differences + shift
        squared = np.einsum("ija,ija->ij", displacement, displacement)
        # Exclude all self-image terms: the source image follows its particle.
        squared[diagonal, diagonal] = np.inf
        radius = np.sqrt(squared)
        gaussian = np.exp(-alpha**2 * squared)
        radial = erfc(alpha * radius) / radius**3
        radial += 2. * alpha / np.sqrt(np.pi) * gaussian / squared
        pairForce -= radial[:, :, None] * displacement
        if withJacobian:
            outer = (3. * radial + 4. * alpha**3 / np.sqrt(np.pi) * gaussian) / squared
            pairTensor += (outer[:, :, None, None] *
                            displacement[:, :, :, None] * displacement[:, :, None, :])
            for axis in range(3):
                pairTensor[:, :, axis, axis] -= radial
    forces = mass * pairForce.sum(axis=1)
    wavevectors, weights = reciprocalModes(alpha, reciprocalCutoff)
    phase = positions @ wavevectors.T
    sine = np.sin(phase)
    cosine = np.cos(phase)
    sourceSine = sine.sum(axis=0)
    sourceCosine = cosine.sum(axis=0)
    forces -= mass * ((sine * sourceCosine - cosine * sourceSine) * weights) @ wavevectors
    if not withJacobian:
        return forces, None
    for first in range(3):
        for second in range(first, 3):
            tensorWeight = weights * wavevectors[:, first] * wavevectors[:, second]
            reciprocalTensor = -((cosine * tensorWeight) @ cosine.T +
                                  (sine * tensorWeight) @ sine.T)
            pairTensor[:, :, first, second] += reciprocalTensor
            if second != first:
                pairTensor[:, :, second, first] += reciprocalTensor
    pairTensor[diagonal, diagonal] = 0.
    # Acceleration_i=sum_j m*F(x_i-x_j); hence off-diagonal blocks are -m*T.
    blocks = -mass * pairTensor
    blocks[diagonal, diagonal] = mass * pairTensor.sum(axis=1)
    jacobian = blocks.transpose(0, 2, 1, 3).reshape(3 * count, 3 * count)
    return forces, jacobian


def csvWrite(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def participation(vector):
    weights = np.sum(vector.reshape(-1, 3)**2, axis=1)
    weights /= weights.sum()
    count = len(weights)
    return dict(participating_particles=float(1. / np.sum(weights**2)),
                participation_fraction=float(1. / (count * np.sum(weights**2))),
                maximum_particle_weight=float(weights.max()),
                top_ten_particle_weight=float(np.sort(weights)[-10:].sum()),
                largest_particle_index=int(np.argmax(weights)))


def analyze(name, positions, output, support, checks):
    count = len(positions)
    spacing = count**(-1. / 3.)
    relative = positions[:, None, :] - positions[None, :, :]
    relative -= np.rint(relative)
    distances = np.linalg.norm(relative, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    if nearest.min() <= support:
        raise ValueError("Compact softening is active; point Newtonian analysis is invalid")
    forces, jacobian = ewald(positions)
    normalized = jacobian / (4. * np.pi)
    # eigensolver consumes its own copy; preserve unsymmetrized J for checks.
    values, vectors = eigh(normalized, check_finite=True, driver="evr")
    translations = np.tile(np.eye(3), (count, 1)) / math.sqrt(count)
    symmetry = np.linalg.norm(normalized - normalized.T) / np.linalg.norm(normalized)
    translationError = np.linalg.norm(normalized @ translations)
    expectedTrace = count - 1.
    trace = float(np.trace(normalized))
    if symmetry > 1.e-12 or translationError > 1.e-12 or abs(trace - expectedTrace) > 1.e-8:
        raise RuntimeError(f"Jacobian checks failed for {name}")
    rows = []
    for index, value in enumerate(values):
        row = dict(index=index, eigenvalue=float(value))
        row.update(participation(vectors[:, index]))
        rows.append(row)
    csvWrite(output / f"{name}_eigenvalues.csv", rows)
    csvWrite(output / f"{name}_forces.csv", [dict(
        particle=index, x=point[0], y=point[1], z=point[2],
        acceleration_x=force[0], acceleration_y=force[1], acceleration_z=force[2])
        for index, (point, force) in enumerate(zip(positions, forces))])
    fastest = vectors[:, -1].reshape(-1, 3)
    csvWrite(output / f"{name}_fastest_mode.csv", [dict(
        particle=index, x=point[0], y=point[1], z=point[2],
        mode_x=vector[0], mode_y=vector[1], mode_z=vector[2],
        weight=float(vector @ vector))
        for index, (point, vector) in enumerate(zip(positions, fastest))])
    summary = dict(
        particles=count, spacing=spacing, softening_support=support,
        minimum_separation=float(nearest.min()),
        minimum_separation_over_spacing=float(nearest.min() / spacing),
        minimum_separation_over_support=float(nearest.min() / support),
        force_rms=float(np.sqrt(np.mean(np.sum(forces**2, axis=1)))),
        force_max=float(np.max(np.linalg.norm(forces, axis=1))),
        net_force_norm=float(np.linalg.norm(forces.sum(axis=0))),
        normalized_jacobian_symmetry_relative_error=float(symmetry),
        translation_residual_norm=float(translationError),
        trace=trace, expected_trace=expectedTrace,
        trace_absolute_error=abs(trace - expectedTrace),
        eigenvalue_sum=float(values.sum()),
        eigenvalue_minimum=float(values[0]), eigenvalue_maximum=float(values[-1]),
        negative_eigenvalues_below_minus_1e_8=int(np.sum(values < -1.e-8)),
        translation_or_near_zero_eigenvalues=int(np.sum(np.abs(values) <= 1.e-8)),
        positive_eigenvalues_above_1e_8=int(np.sum(values > 1.e-8)),
        eigenvalues_above_continuum_1=int(np.sum(values > 1. + 1.e-8)),
        eigenvalue_quantiles={str(quantile): float(np.quantile(values, quantile))
                             for quantile in (0., .01, .1, .25, .5, .75, .9, .99, 1.)},
        fastest_mode_eds_growth_exponent=float((np.sqrt(1. + 24. * values[-1]) - 1.) / 4.),
        fastest_mode_participation=participation(vectors[:, -1]))
    if checks:
        convergence = []
        for alpha, realCutoff, reciprocalCutoff in ((4., 2, 9), (6., 1, 12)):
            otherForces, otherJacobian = ewald(positions, alpha, realCutoff, reciprocalCutoff)
            forceDifference = otherForces - forces
            jacobianDifference = (otherJacobian - jacobian) / (4. * np.pi)
            convergence.append(dict(
                alpha=alpha, real_cutoff=realCutoff, reciprocal_cutoff=reciprocalCutoff,
                force_difference_rms=float(np.sqrt(np.mean(np.sum(forceDifference**2, axis=1)))),
                force_difference_max=float(np.max(np.abs(forceDifference))),
                normalized_jacobian_difference_frobenius=float(np.linalg.norm(jacobianDifference)),
                normalized_jacobian_difference_max=float(np.max(np.abs(jacobianDifference)))))
        summary["ewald_convergence"] = convergence
        rng = np.random.default_rng(194791)
        direction = rng.normal(size=(count, 3))
        direction -= direction.mean(axis=0)
        direction /= np.sqrt(np.mean(np.sum(direction**2, axis=1)))
        prediction = (normalized @ direction.ravel()).reshape(-1, 3)
        finiteDifferences = []
        for relativeStep in (1.e-4, 1.e-5, 1.e-6):
            step = relativeStep * spacing
            plus, _ = ewald(positions + step * direction, withJacobian=False)
            minus, _ = ewald(positions - step * direction, withJacobian=False)
            measured = (plus - minus) / (2. * step * 4. * np.pi)
            finiteDifferences.append(dict(
                step_over_spacing=relativeStep,
                derivative_relative_error=float(np.linalg.norm(measured - prediction) /
                                                np.linalg.norm(prediction)),
                derivative_max_absolute_error=float(np.max(np.abs(measured - prediction)))))
        summary["force_finite_difference"] = finiteDifferences
        if max(row["normalized_jacobian_difference_frobenius"] for row in convergence) > 1.e-8:
            raise RuntimeError(f"Ewald convergence failed for {name}")
        if finiteDifferences[-1]["derivative_relative_error"] > 1.e-7:
            raise RuntimeError(f"Force derivative check failed for {name}")
    (output / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("glass", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--support-spacing", type=float, default=1. / 40.)
    parser.add_argument("--plot-only", action="store_true",
                        help="Regenerate the figure from the existing archived eigenvalue CSVs")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    glass = np.loadtxt(arguments.glass)
    if glass.ndim != 2 or glass.shape[1] != 3 or not np.isfinite(glass).all():
        raise ValueError("Glass must be a finite N by3 coordinate table")
    count = len(glass)
    side = round(count**(1. / 3.))
    if side**3 != count:
        raise ValueError("Comparison requires a cubic particle count")
    lattice = np.array(list(itertools.product(range(side), repeat=3)), dtype=float) / side
    support = arguments.support_spacing / side
    results = {}
    spectra = {}
    sourceHash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    calculationHash = sourceHash
    if arguments.plot_only:
        archived = json.loads((arguments.output / "summary.json").read_text())
        results = archived["results"]
        calculationHash = archived.get("calculation_script_sha256", archived["script_sha256"])
        for name in ("glass", "lattice"):
            with (arguments.output / f"{name}_eigenvalues.csv").open() as source:
                spectra[name] = np.array([float(row["eigenvalue"]) for row in csv.DictReader(source)])
    else:
        for name, positions in (("glass", glass), ("lattice", lattice)):
            print(f"Analyzing {name}, N={count}", flush=True)
            results[name], spectra[name] = analyze(name, positions, arguments.output, support, True)
            print(json.dumps(results[name]), flush=True)
    figure, axes = plt.subplots(1, 3, figsize=(14., 4.2), constrained_layout=True)
    for name, values in spectra.items():
        axes[0].hist(values, bins=70, histtype="step", density=True, label=name)
        axes[1].plot(values, (np.arange(len(values)) + 1.) / len(values), label=name)
        with (arguments.output / f"{name}_eigenvalues.csv").open() as source:
            records = list(csv.DictReader(source))
        axes[2].scatter(values, [float(row["participation_fraction"]) for row in records],
                        s=3., alpha=.55, label=name)
    axes[0].set_ylabel("Eigenvalue density")
    axes[1].set_ylabel("Cumulative fraction")
    axes[2].set_ylabel("Participation fraction")
    for axis in axes:
        axis.set_xlabel(r"Normalized force-Jacobian eigenvalue $\lambda$")
        axis.axvline(0., color="gray", linewidth=.8)
        axis.axvline(1., color="gray", linestyle="--", linewidth=.8)
        axis.grid(alpha=.2)
        axis.legend()
    figure.suptitle(f"Periodic Newtonian linearization: {count} particles, compact support Delta/40")
    figure.savefig(arguments.output / "stability_spectrum.png", dpi=180)
    plt.close(figure)
    summary = dict(
        glass_path=str(arguments.glass.resolve()),
        glass_sha256=hashlib.sha256(arguments.glass.read_bytes()).hexdigest(),
        script_sha256=sourceHash, calculation_script_sha256=calculationHash,
        plot_regenerated_from_archived_eigenvalues=arguments.plot_only,
        normalization="Unit box, equal masses summing to1, G=1; force Jacobian divided by4*pi",
        ewald=dict(alpha=5., real_cube_cutoff=1, reciprocal_cube_cutoff=10,
                   background="Uniform mean density removed; zero Fourier mode excluded"),
        interpretation="Instantaneous frozen-load Jacobian; nonzero residual forces mean glass is not exactly stationary",
        trace_rule="trace(J/(4*pi*G*rho))=N-1; three translation modes vanish",
        results=results)
    (arguments.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
