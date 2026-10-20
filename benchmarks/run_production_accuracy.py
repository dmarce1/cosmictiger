#!/usr/bin/env python3
"""Bounded order/angle accuracy screening and qualified paired timing.

accuracy runs one explicit order/angle on clustered, adaptive, or both fixtures.
The baseline is order 7 at theta 0.61; candidates require its summary.json.
timing accepts only explicit ORDER:THETA=summary.json candidates that passed on
both fixtures with the same binary and source state. It runs three alternating
pairs of eleven measured samples. This is a synthetic full-walk comparison,
not a production checkpoint replay or a high-precision reference validation.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

from run_cuda_kick_arena import digest, parse_log


FIXTURES = ("clustered", "adaptive")
BASELINE = {"order": 7, "theta": 0.61}
ERROR_METRICS = (
    "force_relative_l2",
    "force_relative_p99",
    "potential_relative_l2",
    "potential_relative_p99",
    "low_force_absolute_rms",
    "low_force_absolute_max",
)
SETTINGS = {
    "bucket": 184,
    "side": 64,
    "threads": 8,
    "reference_sinks": 1024,
    "effective_radius": 0.01876988456953534,
    "kernel_family": "ferrers",
    "kernel_degree": 4,
    "omp_threads": 1,
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def percentile99(values):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, 99 * len(ordered) // 100)]


def read_reference(path, fixture):
    rows = {}
    fields = ("gx", "gy", "gz", "phi")
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = tuple(int(row[name]) for name in ("x_raw", "y_raw", "z_raw"))
            if row["fixture"] != fixture or key in rows:
                raise RuntimeError(f"Wrong fixture or duplicate reference sink in {path}")
            actual = tuple(float(row[name]) for name in fields)
            reference = tuple(float(row["reference_" + name]) for name in fields)
            if not all(math.isfinite(value) for value in actual + reference):
                raise RuntimeError(f"Nonfinite reference row in {path}")
            rows[key] = (actual, reference)
    if len(rows) != SETTINGS["reference_sinks"]:
        raise RuntimeError(f"Expected 1024 reference sinks in {path}, got {len(rows)}")
    return rows


def error_metrics(reference, actual, low_indices):
    force_norms = [math.hypot(*value[:3]) for value in reference]
    force_errors = [
        math.hypot(*(got[dim] - want[dim] for dim in range(3)))
        for want, got in zip(reference, actual)
    ]
    potential_errors = [abs(got[3] - want[3]) for want, got in zip(reference, actual)]
    force_squared = math.fsum(value * value for value in force_norms)
    potential_squared = math.fsum(value[3] * value[3] for value in reference)
    if force_squared == 0 or potential_squared == 0:
        raise RuntimeError("Zero reference norm prevents relative accuracy comparison")
    reference_rms = math.sqrt(force_squared / len(force_norms))
    low_errors = [force_errors[i] for i in low_indices]
    low_rms = math.sqrt(math.fsum(value * value for value in low_errors) / len(low_errors))
    low_maximum = max(low_errors)
    return {
        "force_relative_l2": math.sqrt(
            math.fsum(value * value for value in force_errors) / force_squared),
        "force_relative_p99": percentile99([
            error / max(norm, 1e-300) for error, norm in zip(force_errors, force_norms)]),
        "potential_relative_l2": math.sqrt(
            math.fsum(value * value for value in potential_errors) / potential_squared),
        "potential_relative_p99": percentile99([
            error / max(abs(want[3]), 1e-300)
            for error, want in zip(potential_errors, reference)]),
        "force_absolute_rms": math.sqrt(
            math.fsum(value * value for value in force_errors) / len(force_errors)),
        "potential_absolute_rms": math.sqrt(
            math.fsum(value * value for value in potential_errors) / len(potential_errors)),
        "low_force_absolute_p99": percentile99(low_errors),
        "low_force_absolute_rms": low_rms,
        "low_force_absolute_max": low_maximum,
        "low_force_rms_over_global_reference_rms": low_rms / reference_rms,
        "low_force_max_over_global_reference_rms": low_maximum / reference_rms,
        "low_force_count": len(low_indices),
        "low_force_reference_max": max(force_norms[i] for i in low_indices),
    }


def evidence(path):
    path = Path(path).resolve()
    result = json.loads(path.read_text())
    if result.get("kind") != "accuracy" or not result.get("complete"):
        raise RuntimeError(f"Not a completed accuracy summary: {path}")
    return result


def compatible(first, second):
    for field in ("binary_sha256", "git_head", "git_diff_sha256", "driver_sha256", "settings"):
        if first[field] != second[field]:
            raise RuntimeError(f"Accuracy evidence differs in {field}")


def checked_rows(record, fixture):
    path = Path(record["reference_csv"])
    if digest(path) != record["reference_csv_sha256"]:
        raise RuntimeError(f"Reference evidence changed: {path}")
    return read_reference(path, fixture)


def compare_accuracy(candidate, baseline):
    if candidate.keys() != baseline.keys():
        raise RuntimeError("Candidate and baseline do not contain the same physical sinks")
    keys = sorted(baseline)
    reference = [baseline[key][1] for key in keys]
    baseline_actual = [baseline[key][0] for key in keys]
    candidate_actual = [candidate[key][0] for key in keys]
    candidate_reference = [candidate[key][1] for key in keys]
    low_indices = sorted(range(len(keys)), key=lambda i: (math.hypot(*reference[i][:3]), keys[i]))
    low_indices = low_indices[:math.ceil(len(keys) / 10)]
    before = error_metrics(reference, baseline_actual, low_indices)
    after = error_metrics(reference, candidate_actual, low_indices)
    discrepancy = error_metrics(reference, candidate_reference, low_indices)
    discrepancy_ratios = {}
    for field in ("force_absolute_rms", "potential_absolute_rms",
                  "low_force_absolute_rms", "low_force_absolute_max"):
        discrepancy_ratios[field] = (
            discrepancy[field] / before[field] if before[field] else
            (0.0 if discrepancy[field] == 0 else None))
    discrepancy_resolved = all(
        value is not None and value < 0.01 for value in discrepancy_ratios.values())
    gates = {field: after[field] <= before[field] for field in ERROR_METRICS}
    return {
        "baseline_errors": before,
        "candidate_errors_against_baseline_reference": after,
        "reference_discrepancy": discrepancy,
        "reference_discrepancy_fraction_of_baseline_error": discrepancy_ratios,
        "reference_discrepancy_resolved": discrepancy_resolved,
        "reference_discrepancy_note": (
            "This measures repeat-reference disagreement, not true reference accuracy."),
        "gates": gates,
        "passed": discrepancy_resolved and all(gates.values()),
    }


def metadata(args):
    repository = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    difference = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--"], cwd=repository)
    binary = Path(args.binary).resolve(strict=True)
    return {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "binary": str(binary),
        "binary_sha256": digest(binary),
        "git_head": head,
        "git_diff_sha256": hashlib.sha256(difference).hexdigest(),
        "driver_sha256": digest(__file__),
        "settings": dict(SETTINGS, cpus=args.cpus),
    }


def prepare_output(path):
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_case(output, info, config, fixture, samples, reference, name):
    directory = output / "runs" / name
    directory.mkdir(parents=True)
    settings = info["settings"]
    command = [
        "taskset", "-c", settings["cpus"], info["binary"],
        "--run.test=stfBenchmark", "--compute.cuda=true",
        f"--initial.particlesPerDimension={settings['side']}",
        f"--compute.bucketSize={settings['bucket']}",
        f"--gravity.fmmOrder={config['order']}",
        f"--gravity.openingAngle={config['theta']:.17g}",
        f"--gravity.effectiveRadius={settings['effective_radius']:.17g}",
        f"--gravity.kernelFamily={settings['kernel_family']}",
        f"--gravity.kernelDegree={settings['kernel_degree']}",
        f"--hpx:threads={settings['threads']}", "--hpx:bind=none",
        "--hydro.enabled=false", "--analysis.views=false", "--analysis.power=false",
        "--analysis.groups=false", "--lightcone.enabled=false",
    ]
    environment = dict(
        os.environ,
        COSMICTIGER_STF_WALK_SAMPLES=str(samples),
        COSMICTIGER_STF_WALK_BUCKET=str(settings["bucket"]),
        COSMICTIGER_STF_WALK_FIXTURE=fixture,
        COSMICTIGER_STF_WALK_REFERENCE=str(reference),
        COSMICTIGER_STF_WALK_EXPORT_FORCES="0",
        OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1",
    )
    logfile = directory / "stdout.log"
    print(f"{name}: order {config['order']}, theta {config['theta']}, {fixture}", flush=True)
    started = time.monotonic()
    with logfile.open("w") as stream:
        result = subprocess.run(command, cwd=directory, env=environment, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=1800)
    if result.returncode:
        raise RuntimeError(f"Benchmark exited {result.returncode}; see {logfile}")
    parsed = parse_log(logfile.read_text(), samples, [fixture])
    parsed.update(config=config, fixture=fixture, command=command, log=str(logfile),
                  wall_seconds=time.monotonic() - started)
    if reference:
        reference_csv = directory / "stf_walk_reference.csv"
        read_reference(reference_csv, fixture)
        parsed.update(reference_csv=str(reference_csv),
                      reference_csv_sha256=digest(reference_csv))
    return parsed


def run_accuracy(args):
    info = metadata(args)
    config = {"order": args.order, "theta": args.theta}
    fixtures = FIXTURES if args.fixture == "both" else (args.fixture,)
    base = evidence(args.baseline) if args.baseline else None
    if base:
        compatible(info, base["metadata"])
        if base["config"] != BASELINE:
            raise RuntimeError("Baseline evidence must be order 7, theta 0.61")
        if not all(fixture in base["fixtures"] for fixture in fixtures):
            raise RuntimeError("Baseline evidence is missing a requested fixture")
    elif config != BASELINE:
        raise RuntimeError("Candidate accuracy requires --baseline")
    output = prepare_output(args.output)
    summary = {
        "kind": "accuracy", "complete": False, "config": config, "metadata": info,
        "baseline_summary": str(Path(args.baseline).resolve()) if args.baseline else None,
        "fixtures": {},
    }
    write_json(output / "metadata.json", info)
    for fixture in fixtures:
        run = run_case(output, info, config, fixture, 1, SETTINGS["reference_sinks"], fixture)
        rows = read_reference(run["reference_csv"], fixture)
        baseline_rows = checked_rows(base["fixtures"][fixture]["run"], fixture) if base else rows
        summary["fixtures"][fixture] = {
            "run": run, "comparison": compare_accuracy(rows, baseline_rows),
        }
        write_json(output / "summary.json", summary)
    summary["complete"] = True
    summary["qualified_for_timing"] = all(
        fixture in summary["fixtures"] and summary["fixtures"][fixture]["comparison"]["passed"]
        for fixture in FIXTURES)
    write_json(output / "summary.json", summary)
    lines = [
        f"Order {config['order']}, theta {config['theta']}; baseline order 7, theta 0.61.",
        "", "| fixture | force L2 | force p99 | potential L2 | potential p99 | low-force absolute RMS | low-force absolute max | passes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for fixture, record in summary["fixtures"].items():
        comparison = record["comparison"]
        errors = comparison["candidate_errors_against_baseline_reference"]
        values = " | ".join(f"{errors[field]:.6g}" for field in ERROR_METRICS)
        lines.append(f"| {fixture} | {values} | {comparison['passed']} |")
    lines += ["", f"Qualified on both fixtures: {summary['qualified_for_timing']}.",
              "Reference discrepancies and strict baseline gates are in summary.json.",
              "Reference disagreement measures reproducibility, not true reference accuracy.",
              "The direct reference evaluates pair contributions in float and accumulates in double."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Accuracy complete: {output / 'summary.md'}", flush=True)


def parse_candidate(text):
    specification, separator, filename = text.partition("=")
    try:
        order_text, theta_text = specification.split(":")
        order, theta = int(order_text), float(theta_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use ORDER:THETA=SUMMARY_JSON") from error
    if not separator or not filename or not math.isfinite(theta):
        raise argparse.ArgumentTypeError("Use ORDER:THETA=SUMMARY_JSON")
    return {"order": order, "theta": theta}, Path(filename).resolve()


def run_timing(args):
    info = metadata(args)
    base = evidence(args.baseline)
    compatible(info, base["metadata"])
    if base["config"] != BASELINE or not base["qualified_for_timing"]:
        raise RuntimeError("Baseline must have completed valid accuracy evidence on both fixtures")
    variants = [("baseline", BASELINE)]
    candidate_evidence = []
    for index, (config, path) in enumerate(args.candidate, 1):
        candidate = evidence(path)
        compatible(info, candidate["metadata"])
        if candidate["config"] != config or config == BASELINE:
            raise RuntimeError(f"Candidate configuration does not match evidence: {path}")
        if not candidate["qualified_for_timing"]:
            raise RuntimeError(f"Candidate has not passed both fixtures: {path}")
        for fixture in FIXTURES:
            current = checked_rows(candidate["fixtures"][fixture]["run"], fixture)
            original = checked_rows(base["fixtures"][fixture]["run"], fixture)
            if not compare_accuracy(current, original)["passed"]:
                raise RuntimeError(f"Candidate does not pass the supplied baseline: {path}")
        variants.append((f"candidate{index}", config))
        candidate_evidence.append({"config": config, "summary": str(path), "sha256": digest(path)})
    output = prepare_output(args.output)
    info.update(baseline_summary=str(Path(args.baseline).resolve()),
                baseline_summary_sha256=digest(args.baseline),
                candidates=candidate_evidence, pairs=3, samples=11)
    write_json(output / "metadata.json", info)
    runs = []
    for pair in range(3):
        sequence = variants if pair % 2 == 0 else list(reversed(variants))
        for label, config in sequence:
            for fixture in FIXTURES:
                run = run_case(output, info, config, fixture, 11, 0,
                               f"pair{pair}_{label}_{fixture}")
                run.update(pair=pair, label=label)
                runs.append(run)
                write_json(output / "runs.json", runs)
    indexed = {
        (run["pair"], run["label"], run["fixture"], row["potential"]): row
        for run in runs for row in run["medians"]
    }
    rows = []
    for label, config in variants:
        for fixture in FIXTURES:
            for potential in (False, True):
                row = dict(label=label, config=config, fixture=fixture, potential=potential)
                for field in ("tree_seconds", "kick_seconds", "total_seconds"):
                    values = [indexed[pair, label, fixture, potential][field] for pair in range(3)]
                    ratios = [
                        indexed[pair, "baseline", fixture, potential][field] / values[pair]
                        for pair in range(3)]
                    row[field] = {
                        "median": statistics.median(values), "min": min(values), "max": max(values),
                        "paired_speedup_median": statistics.median(ratios),
                        "paired_speedup_min": min(ratios), "paired_speedup_max": max(ratios),
                    }
                rows.append(row)
    summary = {"kind": "timing", "complete": True, "metadata": info, "timing": rows}
    write_json(output / "summary.json", summary)
    lines = ["Primary metric: tree plus complete kick; timings exclude initialization and reference work.",
             "", "| setting | fixture | potential | tree + kick ms | paired speedup range | kick ms |",
             "|---|---|---|---:|---|---:|"]
    for row in rows:
        total = row["total_seconds"]
        config = row["config"]
        lines.append(
            f"| o{config['order']} theta {config['theta']} | {row['fixture']} | {row['potential']} | "
            f"{total['median'] * 1000:.3f} | "
            f"{total['paired_speedup_min']:.3f}-{total['paired_speedup_max']:.3f} | "
            f"{row['kick_seconds']['median'] * 1000:.3f} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Timing complete: {output / 'summary.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("accuracy", "timing"):
        child = subparsers.add_parser(name)
        child.add_argument("--binary", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--cpus", default="1-8", help="Exactly eight taskset CPU IDs")
        child.add_argument("--baseline", type=Path, required=name == "timing")
        if name == "accuracy":
            child.add_argument("--order", type=int, choices=(5, 6, 7), required=True)
            child.add_argument("--theta", type=float, required=True)
            child.add_argument("--fixture", choices=(*FIXTURES, "both"), default="both")
        else:
            child.add_argument("--candidate", type=parse_candidate, action="append", required=True,
                               help="Explicit approved ORDER:THETA=ACCURACY_SUMMARY_JSON")
    args = parser.parse_args()
    cpus = set()
    try:
        for part in args.cpus.split(","):
            endpoints = [int(value) for value in part.split("-")]
            if len(endpoints) == 1:
                cpus.add(endpoints[0])
            elif len(endpoints) == 2 and endpoints[0] <= endpoints[1]:
                cpus.update(range(endpoints[0], endpoints[1] + 1))
            else:
                raise ValueError()
    except ValueError:
        parser.error("--cpus must contain CPU IDs or increasing ranges")
    if len(cpus) != 8 or min(cpus) < 0:
        parser.error("--cpus must select exactly eight nonnegative CPU IDs")
    if args.command == "accuracy" and (not math.isfinite(args.theta) or not 0 < args.theta < 1):
        parser.error("--theta must be finite and between zero and one")
    try:
        if args.command == "accuracy":
            run_accuracy(args)
        else:
            run_timing(args)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
