# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# Distributed under the Boost Software License, Version 1.0.

"""Check a bounded, cold two-species sinusoidal gravity response."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def norm(matrix):
    return math.sqrt(sum(value * value for row in matrix for value in row))


def runDiagnostic(executable, workdir):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("COSMICTIGER_LINEAR_GROWTH_")}
    environment["COSMICTIGER_LINEAR_GROWTH_MODES"] = "1,0,0"
    environment["COSMICTIGER_LINEAR_GROWTH_AMPLITUDES"] = "0.01,0.005"
    command = [str(executable), "--hpx:threads=2", "--hpx:bind=none",
               "--run.test=linearGrowth", "--initial.particlesPerDimension=8",
               "--initial.closePack=false", "--initial.useGlass=false",
               "--hydro.enabled=true", "--compute.cuda=false",
               "--gravity.kernelFamily=ferrers", "--gravity.kernelDegree=4",
               "--gravity.supportRadius=1.5",
               "--gravity.openingAngle=0.1", "--cosmology.omegaB=0.049389",
               "--cosmology.omegaC=0.26503", "--analysis.views=false",
               "--lightcone.enabled=false", "--analysis.power=false",
               "--analysis.groups=false"]
    logPath = workdir / "run.log"
    with logPath.open("w") as log:
        with subprocess.Popen(command, cwd=workdir, env=environment, text=True, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                returnCode = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"linear-growth diagnostic timed out; see {logPath}")
    require(returnCode == 0,
            f"linear-growth diagnostic exited with {returnCode}; see {logPath}:\n"
            + logPath.read_text(errors="replace")[-8000:])


def readMetadata(workdir):
    metadata = json.loads((workdir / "linear_growth_metadata.json").read_text())
    for key, value in metadata.items():
        if isinstance(value, (int, float)):
            require(math.isfinite(value), f"nonfinite metadata {key}={value}")
    expected = {"side": 8, "species": 2, "kernel_family": "ferrers", "kernel_degree": 4,
                "softening_mode": "fixed", "neighbor_number": 0, "theta": 0.1,
                "backend": "cpu", "pressure_enabled": False}
    for key, value in expected.items():
        require(metadata.get(key) == value,
                f"metadata {key}={metadata.get(key)!r}, expected {value!r}")
    masses = [metadata["mass_dm"], metadata["mass_gas"]]
    require(all(math.isfinite(mass) and mass > 0.0 for mass in masses),
            f"invalid class masses {masses}")
    fractions = [mass / sum(masses) for mass in masses]
    require(0.14 < fractions[1] < 0.18, f"unexpected gas mass fraction {fractions[1]}")
    support = metadata["configured_support_radius_box"]
    require(math.isclose(support, 1.5 / 8, rel_tol=1.0e-12),
            f"unexpected configured support {support}")
    for key in ("baseline_support_dm", "baseline_support_gas",
                "baseline_min_support", "baseline_max_support"):
        require(math.isclose(metadata[key], support, rel_tol=1.0e-12),
                f"{key} differs from fixed gravity support: {metadata[key]} != {support}")
    require(metadata["baseline_force_rms"] >= 0.0, "negative baseline force RMS")
    return fractions, support


def readMatrices(workdir, support):
    amplitudes = (0.01, 0.005)
    entries = {amplitude: {} for amplitude in amplitudes}
    with (workdir / "linear_growth_response.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            numeric = {key: float(value) for key, value in row.items()}
            require(all(math.isfinite(value) for value in numeric.values()),
                    "response contains nonfinite values")
            require(tuple(numeric[f"mode_{axis}"] for axis in "xyz") == (1, 0, 0),
                    "response contains an unexpected wavevector")
            amplitude = numeric["amplitude_spacing"]
            require(amplitude in entries, f"unexpected amplitude {amplitude}")
            outputSpecies, outputAxis = numeric["output_species"], numeric["output_axis"]
            inputSpecies, inputAxis = numeric["input_species"], numeric["input_axis"]
            require(outputSpecies in (0, 1) and inputSpecies in (0, 1) and
                    outputAxis in (0, 1, 2) and inputAxis in (0, 1, 2),
                    "response contains an invalid matrix index")
            index = (int(3 * outputSpecies + outputAxis), int(3 * inputSpecies + inputAxis))
            require(index not in entries[amplitude], f"duplicate response entry {index}")
            entries[amplitude][index] = (numeric["response_sin"], numeric["response_cos"])
            for key in ("mean_support_dm", "mean_support_gas", "min_support", "max_support"):
                require(math.isclose(numeric[key], support, rel_tol=1.0e-12),
                        f"perturbation changed fixed gravity support: {key}={numeric[key]}")
            realized = numeric["realized_amplitude"] * 8
            require(abs(realized / amplitude - 1.0) < 1.0e-4,
                    f"unexpected realized displacement amplitude {realized}")
    matrices = {}
    for amplitude in amplitudes:
        require(len(entries[amplitude]) == 36,
                f"amplitude {amplitude} has {len(entries[amplitude])} entries, expected 36")
        matrices[amplitude] = tuple(
            [[entries[amplitude][(row, column)][component] for column in range(6)]
             for row in range(6)] for component in range(2))
    return matrices


def fixedKernelResponse(support):
    # Fourier response of the normalized F2 density, W(q) = 105/(32*pi)*(1-q^2)^2,
    # at the tested one-cycle box mode. Simpson integration is independent of
    # the production force tables and includes the spherical volume measure.
    intervals = 1024
    waveSupport = 2.0 * math.pi * support
    total = 0.0
    for index in range(intervals + 1):
        radius = index / intervals
        argument = waveSupport * radius
        sinc = math.sin(argument) / argument if argument else 1.0
        integrand = 105.0 / 8.0 * radius**2 * (1.0 - radius**2)**2 * sinc
        weight = 1 if index in (0, intervals) else 4 if index % 2 else 2
        total += weight * integrand
    return total / (3.0 * intervals)


def checkMatrices(matrices, fractions, support):
    continuum = fixedKernelResponse(support)
    densities = []
    for amplitude, (sine, cosine) in matrices.items():
        responseNorm = norm(sine)
        require(responseNorm > 0.0, "zero linear-response matrix")
        quadrature = norm(cosine) / responseNorm
        require(quadrature < 0.005, f"quadrature response is too large: {quadrature}")
        weighted = [[fractions[row // 3] * sine[row][column] for column in range(6)]
                    for row in range(6)]
        antisymmetric = [[weighted[row][column] - weighted[column][row]
                          for column in range(6)] for row in range(6)]
        symmetry = norm(antisymmetric) / norm(weighted)
        require(symmetry < 0.005, f"mass-weighted reciprocity residual is {symmetry}")
        # Both species receive the same longitudinal displacement. Project the
        # resulting acceleration onto the total matter density perturbation.
        density = sum(fractions[species] * (sine[3 * species][0] + sine[3 * species][3])
                      for species in range(2))
        # Allow the coarse two-species lattice and finite opening angle to differ
        # slightly from the continuum response of the fixed kernel.
        require(abs(density - continuum) < 0.005,
                f"density response {density} differs from fixed-kernel continuum {continuum}")
        densities.append(density)
        print(f"amplitude={amplitude:g}: density response={density:.8f}, "
              f"quadrature={quadrature:.3g}, reciprocity residual={symmetry:.3g}")
    larger, smaller = matrices[0.01][0], matrices[0.005][0]
    difference = [[larger[row][column] - smaller[row][column] for column in range(6)]
                  for row in range(6)]
    relativeDifference = norm(difference) / norm(smaller)
    require(relativeDifference < 0.005,
            f"centered-response amplitude convergence residual is {relativeDifference}")
    require(abs(densities[0] - densities[1]) < 0.005,
            "density projection did not converge with displacement amplitude")
    print(f"central-difference amplitude residual={relativeDifference:.3g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    arguments = parser.parse_args()
    executable = arguments.executable.resolve(strict=True)
    workdir = arguments.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    # Only remove files owned by this diagnostic, preventing stale-output passes.
    for name in ("linear_growth_response.csv", "linear_growth_metadata.json",
                 "linear_growth_evolution.csv"):
        (workdir / name).unlink(missing_ok=True)
    runDiagnostic(executable, workdir)
    fractions, support = readMetadata(workdir)
    checkMatrices(readMatrices(workdir, support), fractions, support)
    print("Linear-growth regression passed (fixed cold two-species CPU gravity, N=8)")


if __name__ == "__main__":
    main()
