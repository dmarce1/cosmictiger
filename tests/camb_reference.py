#!/usr/bin/env python3
"""Generate and compare external CAMB references for the linear Boltzmann solver.

Generating CAMB references requires a Python environment containing CAMB, NumPy
and SciPy. Running the native driver and comparing saved files use only Python's
standard library. CAMB is never linked into CosmicTiger.
All wavenumbers are in Mpc^-1 and all power spectra are in Mpc^3.

CAMB's native adiabatic normalization has eta_MB -> -1. The CSV density
transfers are negated to match CosmicTiger's test convention eta_MB -> +1.
No amplitude fitting, sigma8 renormalization, or gauge conversion is applied.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time


# Limit every library before importing any numerical extension.
for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"


SOURCES = {
    "transfer_conventions":
        "https://camb.readthedocs.io/en/latest/transfer_variables.html",
    "parameters": "https://camb.readthedocs.io/en/latest/model.html",
    "evolution_output": "https://camb.readthedocs.io/en/latest/results.html",
    "reference_source":
        "https://github.com/cmbant/CAMB/tree/1.6.5",
}
COSMOLOGY = {
    "h": 0.6732,
    "omega_b": 0.049389,
    "omega_c": 0.26503,
    "tcmb_kelvin": 2.7255,
    "helium_mass_fraction": 0.2454006,
    "scalar_amplitude": 2.1e-9,
    "scalar_index": 0.966,
    "pivot_mpc": 0.05,
}


def writeCsv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def writeJson(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parameters(camb, nEff, kmax, redshifts, accuracy, lateRadiationTruncation):
    config = camb.CAMBparams()
    config.set_classes(recombination_model="Recfast")
    config.set_cosmology(
        H0=100 * COSMOLOGY["h"],
        ombh2=COSMOLOGY["omega_b"] * COSMOLOGY["h"] ** 2,
        omch2=COSMOLOGY["omega_c"] * COSMOLOGY["h"] ** 2,
        omk=0,
        mnu=0,
        num_massive_neutrinos=0,
        nnu=nEff,
        standard_neutrino_neff=3.046,
        TCMB=COSMOLOGY["tcmb_kelvin"],
        YHe=COSMOLOGY["helium_mass_fraction"],
    )
    config.InitPower.set_params(
        As=COSMOLOGY["scalar_amplitude"],
        ns=COSMOLOGY["scalar_index"],
        pivot_scalar=COSMOLOGY["pivot_mpc"],
    )
    config.Reion.Reionization = False
    config.Recomb.RECFAST_Hswitch = True
    config.Recomb.RECFAST_Heswitch = 6
    config.Recomb.RECFAST_fudge = 1.125
    config.Recomb.RECFAST_fudge_He = 0.86
    config.WantCls = False
    config.Want_CMB = False
    config.DoLensing = False
    config.Evolve_baryon_cs = False
    config.Evolve_delta_xe = False
    config.Evolve_delta_Ts = False
    config.NonLinear = camb.model.NonLinear_none
    config.set_accuracy(
        AccuracyBoost=accuracy,
        lAccuracyBoost=accuracy,
        DoLateRadTruncation=lateRadiationTruncation,
    )
    config.set_matter_power(
        redshifts=sorted(redshifts, reverse=True),
        kmax=kmax * 1.05,
        k_per_logint=50,
        nonlinear=False,
        silent=True,
    )
    return config


def generate(arguments):
    import camb
    import numpy as np
    import scipy

    if not 0 < arguments.hierarchy_accuracy <= 5:
        raise ValueError(
            "Use 0 < hierarchy accuracy <= 5: CAMB 1.6.5's high-precision "
            "transfer hierarchy uses 45*boost moments and has a 256-moment limit"
        )
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if arguments.k_values:
        ks = np.asarray(arguments.k_values, dtype=float)
    else:
        ks = np.geomspace(arguments.kmin, arguments.kmax, arguments.count)
    if np.any(~np.isfinite(ks)) or np.any(ks <= 0):
        raise ValueError("Wavenumbers must be finite and positive")
    ks = np.unique(ks)
    redshifts = sorted(set(arguments.redshifts), reverse=True)
    allRows = []
    for nEff in arguments.n_eff:
        started = time.monotonic()
        name = f"neff_{nEff:g}".replace(".", "_")
        config = parameters(
            camb, nEff, float(ks[-1]), redshifts, arguments.accuracy,
            arguments.late_radiation_truncation,
        )
        results = camb.get_results(config)
        variables = ["delta_cdm", "delta_baryon", "delta_nonu"]
        evolution = results.get_redshift_evolution(
            ks, redshifts, vars=variables,
            lAccuracyBoost=arguments.hierarchy_accuracy,
        )
        power = results.get_matter_power_interpolator(
            nonlinear=False, var1="delta_nonu", var2="delta_nonu",
            hubble_units=False, k_hunit=False, silent=True,
        )
        # This independent superhorizon call verifies the signed normalization.
        probeK = 0.001
        probeZ = 1e6
        probe = results.get_redshift_evolution(
            probeK, [probeZ], vars=["etak", "delta_cdm"],
            lAccuracyBoost=arguments.hierarchy_accuracy,
        )[0]
        eta = float(probe[0]) / probeK
        if not math.isclose(eta, -1, rel_tol=1e-5, abs_tol=1e-8):
            raise AssertionError(f"Unexpected native CAMB normalization: eta={eta}")
        rows = []
        for ik, k in enumerate(ks):
            primordial = float(config.scalar_power(float(k)))
            for iz, z in enumerate(redshifts):
                cdm, baryon, cb = -evolution[ik, iz, :]
                weighted = (
                    COSMOLOGY["omega_c"] * cdm + COSMOLOGY["omega_b"] * baryon
                ) / (COSMOLOGY["omega_c"] + COSMOLOGY["omega_b"])
                if not math.isclose(float(cb), float(weighted), rel_tol=3e-7):
                    raise AssertionError("CAMB delta_nonu disagrees with weighted CDM+baryons")
                rows.append({
                    "n_eff": nEff,
                    "z": z,
                    "k_mpc": float(k),
                    "delta_cdm": float(cdm),
                    "delta_baryon": float(baryon),
                    "delta_cb": float(cb),
                    "p_cb_mpc3": float(power.P(z, float(k))),
                    "p_cb_from_transfer_mpc3":
                        2 * math.pi ** 2 * primordial / float(k) ** 3 * float(cb) ** 2,
                })
        writeCsv(output / f"camb_{name}.csv", rows)
        allRows.extend(rows)
        backgroundZ = np.unique(np.concatenate((
            np.geomspace(1, 1e7, 161) - 1,
            np.asarray([0, 49, 100, 500, 1000, 1100, 1500, 3000, 9990]),
        )))
        thermal = results.get_background_redshift_evolution(
            backgroundZ, vars=["x_e", "opacity", "cs2b", "T_b"], format="dict",
        )
        thermalRows = []
        for index, z in enumerate(backgroundZ):
            thermalRows.append({
                "z": float(z),
                "a": float(1 / (1 + z)),
                "h_km_s_mpc": float(results.hubble_parameter(float(z))),
                **{key: float(value[index]) for key, value in thermal.items()},
            })
        writeCsv(output / f"camb_background_{name}.csv", thermalRows)
        configPath = output / f"camb_parameters_{name}.txt"
        configPath.write_text(str(config).rstrip() + "\n")
        agreement = max(
            abs(row["p_cb_from_transfer_mpc3"] / row["p_cb_mpc3"] - 1)
            for row in rows
        )
        metadata = {
            "camb_version": camb.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "python_version": platform.python_version(),
            "cosmology": {**COSMOLOGY, "n_eff": nEff, "sum_mnu_ev": 0},
            "normalization":
                "eta_MB -> +1; density transfers are minus native CAMB unit-curvature output",
            "native_camb_superhorizon_probe": {
                "k_mpc": probeK, "z": probeZ, "eta": eta, "delta_cdm": float(probe[1]),
            },
            "omega_gamma": float(results.get_Omega("photon")),
            "omega_massless_neutrinos": float(results.get_Omega("neutrino")),
            "num_massive_neutrinos": int(config.num_nu_massive),
            "reionization": False,
            "recombination": "RECFAST; Hswitch=True, Heswitch=6, fudge=1.125, fudge_He=.86",
            "baryon_thermal_perturbations": False,
            "accuracy_boost": arguments.accuracy,
            "evolution_hierarchy_boost": arguments.hierarchy_accuracy,
            "late_radiation_truncation": arguments.late_radiation_truncation,
            "maximum_direct_transfer_vs_camb_power_relative_difference": agreement,
            "seconds": time.monotonic() - started,
            "thread_limit": 1,
            "sources": SOURCES,
            "config_sha256": hashlib.sha256(configPath.read_bytes()).hexdigest(),
        }
        writeJson(output / f"camb_metadata_{name}.json", metadata)
        print(f"{name}: {len(rows)} modes/epochs; CAMB power/transfer consistency "
              f"{agreement:.3g}; {metadata['seconds']:.2f}s", flush=True)
    writeCsv(output / "camb_reference.csv", allRows)


def readRows(path):
    with path.open(newline="") as source:
        return [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(source)]


def runSolver(arguments):
    """Run the standalone C++ test executable using only files/processes."""
    reference = readRows(arguments.reference)
    ks = sorted({
        row["k_mpc"] for row in reference
        if math.isclose(row["n_eff"], arguments.n_eff, rel_tol=0, abs_tol=1e-12)
        and arguments.min_k <= row["k_mpc"] <= arguments.max_k
    })
    if not ks:
        raise ValueError("No reference modes match the requested model and k range")
    output = arguments.output_dir.resolve()
    rawOutput = output / "raw"
    rawOutput.mkdir(parents=True, exist_ok=True)
    rows = []
    executions = []
    executable = arguments.executable.resolve()
    metadata = {
        "executable": str(executable),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "native_recombination": True,
        "recombination_max_log_step": arguments.recombination_max_log_step,
        "recombination_relative_tolerance": arguments.recombination_rtol,
        "n_eff": arguments.n_eff,
        "maximum_multipole": arguments.max_l,
        "relative_tolerance": arguments.rtol,
        "thread_limit": 1,
        "normalization": "eta_MB -> +1; no fitted amplitude",
        "cosmology": COSMOLOGY,
        "requested_wavenumbers": ks,
        "complete": False,
    }
    repository = Path(__file__).resolve().parent.parent
    metadata["source_sha256_at_run"] = {
        name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
        for name in (
            "src/cosmology/boltzmann.cpp",
            "cosmictiger/cosmology/boltzmann.hpp",
            "src/cosmology/zero_order.cpp",
            "cosmictiger/cosmology/zero_order.hpp",
            "cosmictiger/math/interpolation.hpp",
            "cosmictiger/math/constants.hpp",
            "tests/boltzmann_reference.cpp",
        )
    }
    for name in (
        "cosmictiger/chemistry/recombination.hpp",
        "src/chemistry/recombination.cpp",
        "cosmictiger/cosmology/native_recombination.hpp",
    ):
        metadata["source_sha256_at_run"][name] = hashlib.sha256(
            (repository / name).read_bytes()
        ).hexdigest()
    started = time.monotonic()

    def saveProgress():
        metadata["seconds"] = time.monotonic() - started
        metadata["executions"] = executions
        metadata["rows"] = len(rows)
        if rows:
            writeCsv(output / "transfers.csv", rows)
        writeJson(output / "run_metadata.json", metadata)

    saveProgress()
    # Preserve executable bytes throughout the campaign even if a parallel
    # development build updates the original path.
    with tempfile.TemporaryDirectory(prefix="boltzmann_reference_") as temporary:
        snapshot = Path(temporary)
        copiedExecutable = snapshot / "boltzmann_reference"
        shutil.copy2(executable, copiedExecutable)
        for k in ks:
            command = [
                str(copiedExecutable), str(arguments.n_eff), repr(k),
                str(arguments.max_l), str(arguments.rtol),
                str(arguments.recombination_max_log_step), str(arguments.recombination_rtol),
            ]
            modeStart = time.monotonic()
            log = rawOutput / f"k_{k:.10e}.txt"
            try:
                completed = subprocess.run(
                    command, text=True, capture_output=True, check=False,
                    timeout=arguments.timeout,
                )
            except subprocess.TimeoutExpired as exception:
                stdout = exception.stdout or b""
                stderr = exception.stderr or b""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                log.write_text(stdout + stderr)
                executions.append({
                    "k_mpc": k,
                    "seconds": time.monotonic() - modeStart,
                    "failure": "timeout",
                    "log": str(log.relative_to(output)),
                })
                saveProgress()
                raise RuntimeError(f"Solver timed out; see {log}") from exception
            log.write_text(completed.stdout + completed.stderr)
            executions.append({
                "k_mpc": k,
                "seconds": time.monotonic() - modeStart,
                "returncode": completed.returncode,
                "log": str(log.relative_to(output)),
            })
            if completed.returncode:
                saveProgress()
                raise RuntimeError(f"Solver exited {completed.returncode}; see {log}")
            marker = "n_eff,z,k_mpc,"
            location = completed.stdout.find(marker)
            if location < 0:
                raise ValueError(f"Missing transfer CSV header in {log}")
            modeRows = [
                {key: float(value) for key, value in row.items()}
                for row in csv.DictReader(io.StringIO(completed.stdout[location:]))
            ]
            if len(modeRows) != 2 or {row["z"] for row in modeRows} != {49, 0}:
                raise ValueError(f"Expected exactly redshifts 49 and 0 in {log}")
            for row in modeRows:
                primordial = COSMOLOGY["scalar_amplitude"] * (
                    k / COSMOLOGY["pivot_mpc"]
                ) ** (COSMOLOGY["scalar_index"] - 1)
                row["p_cb_mpc3"] = (
                    2 * math.pi ** 2 * primordial / k ** 3 * row["delta_cb"] ** 2
                )
                rows.append(row)
            saveProgress()
    metadata["complete"] = True
    saveProgress()
    print(f"n_eff={arguments.n_eff:g}, lmax={arguments.max_l}, "
          f"rtol={arguments.rtol:g}: {len(rows)} rows in {metadata['seconds']:.2f}s")


def compare(arguments):
    reference = readRows(arguments.reference)
    candidate = readRows(arguments.candidate)
    metrics = ["delta_cdm", "delta_baryon", "delta_cb"]
    if all("p_cb_mpc3" in row for row in candidate):
        metrics.append("p_cb_mpc3")
    differences = []
    for row in candidate:
        matches = [
            entry for entry in reference
            if all(math.isclose(row[key], entry[key], rel_tol=1e-9, abs_tol=1e-12)
                   for key in ("n_eff", "z", "k_mpc"))
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one reference for {row}")
        ref = matches[0]
        difference = {key: row[key] for key in ("n_eff", "z", "k_mpc")}
        for key in metrics:
            if not math.isfinite(row[key]) or not math.isfinite(ref[key]) or ref[key] == 0:
                raise ValueError(f"Invalid comparison for {key}: {row[key]}, {ref[key]}")
            difference[key + "_relative_error"] = row[key] / ref[key] - 1
        differences.append(difference)
    if not differences:
        raise ValueError("Candidate CSV has no rows")
    report = {
        "reference": str(arguments.reference.resolve()),
        "candidate": str(arguments.candidate.resolve()),
        "rows": len(differences),
        "reference_rows": len(reference),
        "tolerance": arguments.tolerance,
        "normalization_fit": False,
        "metrics": {},
    }
    for key in metrics:
        worst = max(differences, key=lambda item: abs(item[key + "_relative_error"]))
        report["metrics"][key] = {
            "maximum_absolute_relative_error": abs(worst[key + "_relative_error"]),
            "worst_mode": worst,
        }
    report["passed"] = all(
        item["maximum_absolute_relative_error"] <= arguments.tolerance
        for item in report["metrics"].values()
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    writeCsv(arguments.output_dir / "differences.csv", differences)
    writeJson(arguments.output_dir / "comparison.json", report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


def probeGrowth(arguments):
    """Compare the CDM+baryon displacement derivative with an exact CAMB RHS."""
    import camb
    from camb import symbolic

    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = parameters(camb, arguments.n_eff, max(arguments.k_values), [49, 0], 3, False)
    results = camb.get_results(config)
    # CAMB's covariant hdot is hdot_MB/6. H is the conformal Hubble rate.
    evolution = results.get_redshift_evolution(
        arguments.k_values, [49, 0],
        vars=["delta_nonu", "v_baryon_cdm", "H", 6 * symbolic.hdot / symbolic.H],
        lAccuracyBoost=5,
    )
    baryonFraction = COSMOLOGY["omega_b"] / (COSMOLOGY["omega_b"] + COSMOLOGY["omega_c"])
    reference = []
    for ik, k in enumerate(arguments.k_values):
        for iz, z in enumerate((49, 0)):
            delta, velocity, expansion, hPrime = evolution[ik, iz]
            # Negate native CAMB perturbations to obtain eta_MB -> +1.
            hPrime, velocity, delta = -float(hPrime), -float(velocity), -float(delta)
            derivative = -0.5 * hPrime - baryonFraction * k / expansion * velocity
            reference.append({
                "n_eff": arguments.n_eff, "z": z, "k_mpc": k,
                "delta_cb": delta, "h_prime": hPrime,
                "theta_b_over_k": velocity, "conformal_hubble_mpc": float(expansion),
                "delta_cb_prime": float(derivative),
                "growth_cb": float(derivative / delta),
            })
    writeCsv(output / "camb_growth_probe.csv", reference)
    nativeParameters = json.loads(arguments.native_parameters.read_text())
    omegaMatter = nativeParameters["omegaB"] + nativeParameters["omegaC"]
    omegaRadiation = nativeParameters["omegaGam"] + nativeParameters["omegaNu"]
    comparisons = []
    for candidate in arguments.candidate:
        native = readRows(candidate)
        for ref in reference:
            matching = [
                row for row in native
                if all(math.isclose(row[key], ref[key], rel_tol=1e-10, abs_tol=1e-12)
                       for key in ("n_eff", "z", "k_mpc"))
            ]
            if len(matching) != 1:
                raise ValueError(f"Expected one native row for {ref} in {candidate}")
            row = matching[0]
            a = 1 / (1 + row["z"])
            # Same physical c and flat background as cosmic_constants::kH0.
            expansion = COSMOLOGY["h"] * 100 / 299792.458 * math.sqrt(
                omegaRadiation / a**2 + omegaMatter / a
                + (1 - omegaRadiation - omegaMatter) * a**2
            )
            derivative = -0.5 * row["h_prime"] - baryonFraction * row["k_mpc"] / expansion * row["theta_b_over_k"]
            comparisons.append({
                "candidate": str(candidate), "z": row["z"], "k_mpc": row["k_mpc"],
                "delta_cb_prime_camb": ref["delta_cb_prime"],
                "delta_cb_prime_native": derivative,
                "delta_cb_prime_relative_error": derivative / ref["delta_cb_prime"] - 1,
                "growth_cb_camb": ref["growth_cb"],
                "growth_cb_native": derivative / row["delta_cb"],
                "growth_cb_relative_error": derivative / row["delta_cb"] / ref["growth_cb"] - 1,
            })
    writeCsv(output / "growth_comparison.csv", comparisons)
    summary = {
        str(candidate): {
            key: max(abs(row[key]) for row in comparisons if row["candidate"] == str(candidate))
            for key in ("delta_cb_prime_relative_error", "growth_cb_relative_error")
        }
        for candidate in arguments.candidate
    }
    writeJson(output / "growth_comparison.json", {
        "camb_version": camb.__version__, "accuracy_boost": 3, "hierarchy_boost": 5,
        "formula": "d(delta_cb)/dln(a) = -h_prime/2 - f_b (k/Hc) theta_b_over_k",
        "camb_metric": "-6*camb.symbolic.hdot/camb.symbolic.H (eta_MB -> +1)",
        "native_parameters_sha256": hashlib.sha256(arguments.native_parameters.read_bytes()).hexdigest(),
        "summary": summary,
    })
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generator = commands.add_parser("generate")
    generator.add_argument("--output-dir", type=Path, required=True)
    generator.add_argument("--n-eff", type=float, nargs="+", default=[3.046, 0])
    generator.add_argument("--redshifts", type=float, nargs="+", default=[49, 0])
    generator.add_argument("--kmin", type=float, default=1e-4)
    generator.add_argument("--kmax", type=float, default=1)
    generator.add_argument("--count", type=int, default=25)
    generator.add_argument("--k-values", type=float, nargs="+")
    generator.add_argument("--accuracy", type=float, default=2)
    generator.add_argument("--hierarchy-accuracy", type=float, default=4)
    generator.add_argument("--late-radiation-truncation", action="store_true")
    generator.set_defaults(function=generate)
    runner = commands.add_parser("run")
    runner.add_argument("--executable", type=Path, required=True)
    runner.add_argument("--recombination-max-log-step", type=float, default=0.01)
    runner.add_argument("--recombination-rtol", type=float, default=1e-7)
    runner.add_argument("--reference", type=Path, required=True)
    runner.add_argument("--n-eff", type=float, required=True)
    runner.add_argument("--max-l", type=int, default=64)
    runner.add_argument("--rtol", type=float, default=1e-6)
    runner.add_argument("--min-k", type=float, default=1e-4)
    runner.add_argument("--max-k", type=float, default=1)
    runner.add_argument("--timeout", type=float, default=300)
    runner.add_argument("--output-dir", type=Path, required=True)
    runner.set_defaults(function=runSolver)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--reference", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--output-dir", type=Path, required=True)
    comparison.add_argument("--tolerance", type=float, default=0.005)
    comparison.set_defaults(function=compare)
    probe = commands.add_parser("probe-growth")
    probe.add_argument("--n-eff", type=float, default=3.046)
    probe.add_argument("--k-values", type=float, nargs="+", default=[0.01, 0.1, 1, 10])
    probe.add_argument("--candidate", type=Path, nargs="+", required=True)
    probe.add_argument("--native-parameters", type=Path, required=True)
    probe.add_argument("--output-dir", type=Path, required=True)
    probe.set_defaults(function=probeGrowth)
    arguments = parser.parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
