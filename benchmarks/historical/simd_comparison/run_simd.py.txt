#!/usr/bin/env python3
"""Plan an old/new CPU gravity comparison; only --execute starts benchmark runs.

Prepare and compile immutable snapshots with prepare_simd.py first. This runner
uses alternating old/new order, one pinned CPU, and fresh working directories.
Startup, input generation, output clearing, and correctness checks are outside
the C++ harness's timed regions. Results describe these kernel workloads, not
whole-simulation throughput. The default build needs CUDA access for application
startup even though every measured kernel runs on the CPU.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def machine_state(cpu):
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    return {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpuinfo": read_text("/proc/cpuinfo"),
        "load_average": os.getloadavg(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "selected_cpu": cpu,
        "smt_siblings": read_text(topology / "topology/thread_siblings_list"),
        "scaling_governor": read_text(topology / "cpufreq/scaling_governor"),
        "scaling_driver": read_text(topology / "cpufreq/scaling_driver"),
        "scaling_frequency_khz": read_text(topology / "cpufreq/scaling_cur_freq"),
    }


def verify_snapshot(variant):
    source = Path(variant["source_dir"])
    for relative, expected in variant["source_sha256"].items():
        path = source / relative
        if isinstance(expected, dict):
            matches = path.is_symlink() and os.readlink(path) == expected["symlink"]
        else:
            matches = path.is_file() and not path.is_symlink() and digest(path) == expected
        if not matches:
            raise RuntimeError(f"Prepared source changed: {path}; prepare a fresh comparison")
    binary = Path(variant["executable"])
    expected = variant.get("executable_sha256")
    if not expected or not binary.is_file() or digest(binary) != expected:
        raise RuntimeError(f"Missing or changed prepared executable: {binary}")


def read_measurements(directory, batches, repetitions):
    timings = {}
    with (directory / "bench_timings.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["case"]
            sample = int(row["sample"])
            seconds = float(row["seconds"])
            checksum = float(row["checksum"])
            calls = int(row["calls"])
            if calls != repetitions or not math.isfinite(seconds) or seconds <= 0:
                raise RuntimeError(f"Invalid timing in {directory}: {row}")
            if not math.isfinite(checksum):
                raise RuntimeError(f"Non-finite timing checksum in {directory}: {name}")
            samples = timings.setdefault(name, {})
            if sample in samples:
                raise RuntimeError(f"Duplicate sample for {name}")
            samples[sample] = {**row, "seconds": seconds, "calls": calls,
                               "ns_per_call": seconds * 1e9 / calls}
    values = {}
    with (directory / "bench_values.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["case"], int(row["index"]))
            value = float(row["value"])
            if key in values or not math.isfinite(value):
                raise RuntimeError(f"Duplicate/non-finite output in {directory}: {key}")
            values[key] = value
    if not timings or not values or {name for name, _ in values} != set(timings):
        raise RuntimeError(f"Missing benchmark cases or outputs in {directory}")
    for name, samples in timings.items():
        if set(samples) != set(range(batches)):
            raise RuntimeError(f"Incomplete samples for {name} in {directory}")
    return timings, values


def compare_outputs(old, new, absolute_tolerance, relative_tolerance):
    if old.keys() != new.keys():
        raise RuntimeError("Old/new output dimensions or case names differ")
    errors = {}
    failures = []
    for key, old_value in old.items():
        new_value = new[key]
        difference = abs(new_value - old_value)
        scale = max(abs(old_value), abs(new_value))
        limit = absolute_tolerance + relative_tolerance * scale
        case = errors.setdefault(key[0], {"max_abs_error": 0.0, "max_error_over_limit": 0.0})
        case["max_abs_error"] = max(case["max_abs_error"], difference)
        case["max_error_over_limit"] = max(case["max_error_over_limit"], difference / limit)
        if difference > limit:
            failures.append({"case": key[0], "index": key[1], "old": old_value,
                             "new": new_value, "absolute_error": difference, "limit": limit})
    return errors, failures


def summarize(round_data, errors):
    names = set(round_data[0]["old"])
    summaries = []
    for pair in round_data:
        if set(pair["old"]) != names or set(pair["new"]) != names:
            raise RuntimeError("Case set changed across paired rounds")
    for name in sorted(names):
        medians = {variant: [statistics.median(
            sample["ns_per_call"] for sample in pair[variant][name].values())
            for pair in round_data] for variant in ("old", "new")}
        ratios = [new / old for old, new in zip(medians["old"], medians["new"])]
        ratio = statistics.median(ratios)
        summaries.append({
            "case": name,
            "old_median_ns_per_call": statistics.median(medians["old"]),
            "new_median_ns_per_call": statistics.median(medians["new"]),
            "median_paired_new_over_old": ratio,
            "new_time_change_percent": 100 * (ratio - 1),
            "paired_ratio_min": min(ratios),
            "paired_ratio_max": max(ratios),
            "paired_ratio_mad": statistics.median(abs(value - ratio) for value in ratios),
            **errors[name],
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("build/simd_benchmark"))
    parser.add_argument("--execute", action="store_true",
                        help="explicitly authorize this invocation to execute the benchmark")
    parser.add_argument("--rounds", type=positive, default=6,
                        help="even number of paired rounds; order alternates old/new and new/old")
    parser.add_argument("--batches", type=positive, default=5)
    parser.add_argument("--repetitions", type=positive, default=8)
    parser.add_argument("--warmup", type=positive, default=2)
    parser.add_argument("--cpu", type=int, help="logical CPU for affinity (default: first allowed)")
    parser.add_argument("--case", help="optional exact harness case name")
    parser.add_argument("--kernel-family", choices=("ferrers", "spline", "wendland"),
                        default="ferrers")
    parser.add_argument("--kernel-degree", type=positive, default=4)
    parser.add_argument("--absolute-tolerance", type=float, default=2e-4)
    parser.add_argument("--relative-tolerance", type=float, default=2e-4)
    parser.add_argument("--timeout", type=positive, default=1800,
                        help="maximum seconds for one binary invocation")
    parser.add_argument("--output", type=Path, help="new result directory; never overwritten")
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2:
        parser.error("--rounds must be even and at least 2 to balance run order")
    for name in ("absolute_tolerance", "relative_tolerance"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.repetitions > 256 or args.batches > 1000 or args.warmup > 256:
        parser.error("repetitions <=256, batches <=1000, and warmup <=256 are required")
    prepared = args.prepared.resolve()
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported prepared manifest schema")
    allowed = sorted(os.sched_getaffinity(0))
    cpu = allowed[0] if args.cpu is None else args.cpu
    if cpu not in allowed:
        parser.error(f"CPU {cpu} is outside the allowed affinity {allowed}")
    taskset = shutil.which("taskset")
    if not taskset:
        raise RuntimeError("taskset is required for repeatable CPU affinity")
    runtime_arguments = ["--run.test=simdBenchmark", "--compute.cuda=false",
                         "--hydro.enabled=false", "--initial.particlesPerDimension=8",
                         f"--gravity.kernelFamily={args.kernel_family}",
                         f"--gravity.kernelDegree={args.kernel_degree}",
                         "--hpx:threads=1", "--hpx:bind=none"]
    commands = {name: [taskset, "-c", str(cpu), variant["executable"], *runtime_arguments]
                for name, variant in manifest["variants"].items()}
    plan = {"execute": args.execute, "prepared": str(prepared), "cpu": cpu,
            "rounds": args.rounds, "batches": args.batches,
            "repetitions": args.repetitions, "warmup": args.warmup,
            "case": args.case, "commands": commands,
            "order": [["old", "new"] if index % 2 == 0 else ["new", "old"]
                      for index in range(args.rounds)]}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        print("PLAN ONLY: no benchmark or application was executed. Add --execute when ready.")
        return
    if set(manifest["variants"]) != {"old", "new"}:
        raise RuntimeError("Expected exactly old and new variants")
    for variant in manifest["variants"].values():
        verify_snapshot(variant)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    output = (args.output or prepared / "results" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_path, output / "prepared_manifest.json")
    metadata = {"plan": plan, "machine_before": machine_state(cpu), "runs": [],
                "runner_sha256": digest(Path(__file__)),
                "prepared_manifest_sha256": digest(manifest_path),
                "absolute_tolerance": args.absolute_tolerance,
                "relative_tolerance": args.relative_tolerance,
                "notes": ["Timings cover production CPU gravity routines including packing and reductions.",
                          "Startup, dataset construction, and output reset/validation are not timed.",
                          "A ratio below 1 means the new implementation is faster.",
                          "Range and MAD describe paired-round variation, not a confidence interval.",
                          "CP currently computes all expansion coefficients even when do_phi is false.",
                          "Single pinned CPU, single HPX worker; no whole-simulation speedup is implied."]}
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith("COSMICTIGER_BENCH_"):
            del environment[key]
    environment.update({"COSMICTIGER_BENCH_REPETITIONS": str(args.repetitions),
                        "COSMICTIGER_BENCH_BATCHES": str(args.batches),
                        "COSMICTIGER_BENCH_WARMUP": str(args.warmup),
                        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    if args.case:
        environment["COSMICTIGER_BENCH_CASE"] = args.case
    rounds = []
    errors = {}
    try:
        for index, order in enumerate(plan["order"]):
            pair = {}
            pair_values = {}
            for name in order:
                directory = output / f"round_{index:02d}_{name}"
                directory.mkdir()
                run_info = {"round": index, "variant": name, "command": commands[name],
                            "directory": str(directory), "before": machine_state(cpu)}
                metadata["runs"].append(run_info)
                print(f"Round {index + 1}/{args.rounds}: {name}", flush=True)
                with (directory / "application.log").open("w") as log:
                    result = subprocess.run(commands[name], cwd=directory, env=environment,
                                            stdout=log, stderr=subprocess.STDOUT,
                                            timeout=args.timeout, check=False)
                run_info.update({"returncode": result.returncode, "after": machine_state(cpu)})
                if result.returncode:
                    raise RuntimeError(f"{name} failed; see {directory / 'application.log'}")
                pair[name], pair_values[name] = read_measurements(
                    directory, args.batches, args.repetitions)
            round_errors, failures = compare_outputs(pair_values["old"], pair_values["new"],
                                                     args.absolute_tolerance,
                                                     args.relative_tolerance)
            for name in pair["old"]:
                if name not in pair["new"]:
                    raise RuntimeError(f"Missing new-backend case {name}")
                for sample, old_sample in pair["old"][name].items():
                    new_sample = pair["new"][name][sample]
                    for key in ("calls", "interactions", "sinks", "sources", "do_phi"):
                        if old_sample[key] != new_sample[key]:
                            raise RuntimeError(f"Old/new work counts differ: {name}, {key}")
            if failures:
                (output / "numerical_failures.json").write_text(json.dumps(failures, indent=2))
                raise RuntimeError("Numerical comparison failed; no speedup summary produced")
            for name, case_errors in round_errors.items():
                combined = errors.setdefault(name, {key: 0.0 for key in case_errors})
                for key, value in case_errors.items():
                    combined[key] = max(combined[key], value)
            rounds.append(pair)
        summaries = summarize(rounds, errors)
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        metadata["status"] = "complete"
        metadata["summary"] = summaries
        print(f"Results: {output / 'summary.csv'}")
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        raise
    finally:
        metadata["machine_after"] = machine_state(cpu)
        (output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as error:
        print(f"Benchmark setup/run error: {error}", file=sys.stderr)
        sys.exit(1)
