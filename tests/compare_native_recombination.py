#!/usr/bin/env python3
"""Compare native homogeneous recombination with independent RECFAST/CAMB.

The RECFAST comparison uses archived independent numerical reference data.
Native and reference histories use xe=n_e/n_H and temperature in kelvin.
"""

import argparse
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

import camb_reference


def referenceCamb(arguments):
    import camb
    import numpy as np

    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Exact RECFAST z nodes plus denser H/He and logarithmic residual-era probes.
    redshifts = np.unique(np.concatenate((
        np.arange(0, 10001, 10), np.arange(600, 1801, 2),
        np.arange(1500, 8001, 5), np.geomspace(1, 1e8 + 1, 401) - 1,
        np.arange(0, 21), [49, 50, 100, 500, 9990],
    )))
    allRows = []
    for nEff in arguments.n_eff:
        config = camb_reference.parameters(camb, nEff, 1, [49, 0], 3, False)
        results = camb.get_background(config)
        thermal = results.get_background_redshift_evolution(
            redshifts, vars=["x_e", "T_b", "opacity", "cs2b"], format="dict",
        )
        rows = [
            {
                "n_eff": nEff, "z": float(z), "a": float(1 / (1 + z)),
                "x_e": float(thermal["x_e"][index]),
                "T_m": float(thermal["T_b"][index]),
                "opacity": float(thermal["opacity"][index]),
                "cs2b": float(thermal["cs2b"][index]),
            }
            for index, z in enumerate(redshifts)
        ]
        name = f"neff_{nEff:g}".replace(".", "_")
        camb_reference.writeCsv(output / f"{name}.csv", rows)
        (output / f"{name}_parameters.txt").write_text(str(config).rstrip() + "\n")
        allRows.extend(rows)
        print(f"CAMB N_eff={nEff:g}: {len(rows)} ionization/temperature samples")
    camb_reference.writeCsv(output / "reference.csv", allRows)
    (output / "redshifts.txt").write_text("".join(f"{z:.17g}\n" for z in redshifts))
    camb_reference.writeJson(output / "provenance.json", {
        "camb_version": camb.__version__, "thread_limit": 1,
        "reionization": False, "recombination": "RECFAST Hswitch=1, Heswitch=6",
        "redshift_count": len(redshifts),
    })


def runNative(arguments):
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(arguments.executable.resolve()), str(arguments.n_eff),
        str(arguments.max_log_step), str(arguments.redshifts.resolve()), str(arguments.rtol),
    ]
    with tempfile.TemporaryDirectory(prefix="native_recombination_driver_") as temporary:
        snapshot = Path(temporary) / "recombination_reference"
        shutil.copy2(arguments.executable, snapshot)
        executableSha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        result = subprocess.run(
            [str(snapshot), *command[1:]], capture_output=True, text=True, timeout=120, check=False,
        )
    (output / "history.csv").write_text(result.stdout)
    (output / "stderr.log").write_text(result.stderr)
    if result.returncode:
        camb_reference.writeJson(output / "failure.json", {
            "command": command, "returncode": result.returncode,
            "executable_sha256": executableSha,
        })
        raise RuntimeError(f"Native history driver failed; see {output / 'stderr.log'}")
    rows = camb_reference.readRows(output / "history.csv")
    if any(not math.isfinite(row[key]) or row[key] <= 0 for row in rows for key in ("x_e", "T_m")):
        raise ValueError("Native history contains nonpositive or nonfinite states")
    camb_reference.writeJson(output / "provenance.json", {
        "executable": str(arguments.executable.resolve()),
        "executable_sha256": executableSha,
        "command": command, "n_eff": arguments.n_eff, "thread_limit": 1,
        "max_log_step": arguments.max_log_step, "relative_tolerance": arguments.rtol,
        "rows": len(rows),
        "source_sha256_at_run": {
            name: hashlib.sha256((Path(__file__).resolve().parent.parent / name).read_bytes()).hexdigest()
            for name in (
                "cosmictiger/chemistry/recombination.hpp", "src/chemistry/recombination.cpp",
                "cosmictiger/cosmology/native_recombination.hpp", "tests/recombination_reference.cpp",
            )
        },
    })
    print(f"Native N_eff={arguments.n_eff:g}: {len(rows)} ionization/temperature samples")


def compareHistories(arguments):
    candidate = camb_reference.readRows(arguments.candidate)
    reference = camb_reference.readRows(arguments.reference)
    if arguments.integer_redshifts:
        # CAMB RECFAST has a fixed integer-z thermal table. Its natural spline
        # has a sizeable endpoint artifact between z=0 and z=1 without reionization.
        reference = [
            row for row in reference
            if math.isclose(row["z"], round(row["z"]), rel_tol=0, abs_tol=1e-8)
        ]
    differences = []
    for ref in reference:
        matches = [
            row for row in candidate
            if math.isclose(row["n_eff"], ref["n_eff"], abs_tol=1e-10)
            and math.isclose(row["z"], ref["z"], rel_tol=1e-10, abs_tol=1e-8)
        ]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError(f"Duplicate native samples for {ref}")
        row = matches[0]
        difference = {key: ref[key] for key in ("n_eff", "z")}
        for key in ("x_e", "T_m"):
            if not math.isfinite(row[key]) or not math.isfinite(ref[key]) or ref[key] <= 0:
                raise ValueError("Invalid recombination sample")
            difference[key + "_native"] = row[key]
            difference[key + "_reference"] = ref[key]
            difference[key + "_relative_error"] = row[key] / ref[key] - 1
        differences.append(difference)
    if not differences:
        raise ValueError("No common native/reference samples")
    epochs = {
        "all": lambda z: True,
        "early_ionized": lambda z: z > 8000,
        "helium": lambda z: 1800 < z <= 8000,
        "hydrogen": lambda z: 500 < z <= 1800,
        "residual": lambda z: z <= 500,
    }
    summary = {}
    for name, selector in epochs.items():
        subset = [row for row in differences if selector(row["z"])]
        if subset:
            summary[name] = {
                key: max(subset, key=lambda row: abs(row[key + "_relative_error"]))
                for key in ("x_e", "T_m")
            }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    camb_reference.writeCsv(arguments.output_dir / "differences.csv", differences)
    passed = all(
        abs(row[key + "_relative_error"]) <= arguments.tolerance
        for row in differences for key in ("x_e", "T_m")
    )
    camb_reference.writeJson(arguments.output_dir / "comparison.json", {
        "reference": str(arguments.reference.resolve()),
        "candidate": str(arguments.candidate.resolve()),
        "matched_rows": len(differences), "candidate_rows": len(candidate),
        "reference_rows": len(reference), "epoch_maxima": summary,
        "integer_redshifts_only": arguments.integer_redshifts,
        "passed": passed,
        "tolerance": arguments.tolerance,
    })
    for name, values in summary.items():
        print(name, {key: row[key + "_relative_error"] for key, row in values.items()})
    if not passed:
        raise SystemExit(1)


def checkReference(arguments):
    """Standard-library-only CI check against frozen independent RECFAST data."""
    workdir = arguments.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    report = []
    for nEff in (0, 3.046):
        name = f"neff_{nEff:g}".replace(".", "_")
        folder = workdir / name
        folder.mkdir(parents=True, exist_ok=True)
        referencePath = arguments.reference_root / name / "reference.csv"
        referenceRows = camb_reference.readRows(referencePath)
        if len(referenceRows) != 1000 or any(
            row["n_eff"] != nEff or row["z"] != 9990 - 10 * index
            or any(not math.isfinite(row[key]) or row[key] <= 0 for key in ("x_e", "T_m"))
            for index, row in enumerate(referenceRows)
        ):
            raise ValueError("Unexpected frozen RECFAST reference grid or state")
        samples = folder / "redshifts.txt"
        samples.write_text("".join(f"{row['z']:.17g}\n" for row in referenceRows))
        native = subprocess.run(
            [str(arguments.native_executable.resolve()), str(nEff), "0.01", str(samples)],
            capture_output=True, text=True, timeout=120, check=True,
        )
        (folder / "native.csv").write_text(native.stdout)
        (folder / "native.log").write_text(native.stderr)
        nativeRows = camb_reference.readRows(folder / "native.csv")
        if len(nativeRows) != len(referenceRows):
            raise ValueError("Unexpected native reference sample count")
        differences = []
        for reference, row in zip(referenceRows, nativeRows):
            if row["z"] != reference["z"] or row["n_eff"] != nEff or any(
                not math.isfinite(row[key]) or row[key] <= 0 for key in ("x_e", "T_m")
            ):
                raise ValueError("Invalid native reference sample")
            differences.append({
                "z": row["z"], **{
                    key + "_relative_error": row[key] / reference[key] - 1
                    for key in ("x_e", "T_m")
                },
            })
        worst = {
            key: max(differences, key=lambda row: abs(row[key + "_relative_error"]))
            for key in ("x_e", "T_m")
        }
        camb_reference.writeCsv(folder / "differences.csv", differences)
        report.append({
            "n_eff": nEff, "samples": len(nativeRows), "worst": worst,
            "reference": str(referencePath.resolve()),
            "reference_sha256": hashlib.sha256(referencePath.read_bytes()).hexdigest(),
        })
        print(f"N_eff={nEff:g}: max xe difference {worst['x_e']['x_e_relative_error']:.6g}, "
              f"T difference {worst['T_m']['T_m_relative_error']:.6g}")
    passed = all(
        abs(case["worst"][key][key + "_relative_error"]) <= arguments.tolerance
        for case in report for key in ("x_e", "T_m")
    )
    camb_reference.writeJson(workdir / "comparison.json", {
        "cases": report, "tolerance": arguments.tolerance, "passed": passed,
        "scope": "Electron fraction and gas temperature versus archived RECFAST 1.5.2 data",
    })
    if not passed:
        raise AssertionError("Native history exceeds frozen RECFAST regression tolerance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    camb = commands.add_parser("camb")
    camb.add_argument("--n-eff", type=float, nargs="+", default=[0, 3.046])
    camb.add_argument("--output-dir", type=Path, required=True)
    camb.set_defaults(function=referenceCamb)
    native = commands.add_parser("native")
    native.add_argument("--executable", type=Path, required=True)
    native.add_argument("--n-eff", type=float, required=True)
    native.add_argument("--max-log-step", type=float, default=0.01)
    native.add_argument("--rtol", type=float, default=1e-7)
    native.add_argument("--redshifts", type=Path, required=True)
    native.add_argument("--output-dir", type=Path, required=True)
    native.set_defaults(function=runNative)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--reference", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--output-dir", type=Path, required=True)
    comparison.add_argument("--tolerance", type=float, default=0.01)
    comparison.add_argument("--integer-redshifts", action="store_true")
    comparison.set_defaults(function=compareHistories)
    check = commands.add_parser("check-reference")
    check.add_argument("--native-executable", type=Path, required=True)
    check.add_argument("--reference-root", type=Path, required=True)
    check.add_argument("--workdir", type=Path, required=True)
    check.add_argument("--tolerance", type=float, default=0.002)
    check.set_defaults(function=checkReference)
    arguments = parser.parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
