#!/usr/bin/env python3
"""Build or run the opt-in STF microbenchmarks, serially.

Without --build or --run, print the proposed compiler commands only. Builds
never run benchmarks. Timing requires --run and an intact build manifest.
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
import shlex
import shutil
import statistics
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def inputs(root, baseline):
    paths = list((root / "cosmictiger").rglob("*.hpp"))
    paths += [root / "benchmarks/stf_kernels.cpp", root / "benchmarks/stf_kernels_cuda.cu", baseline]
    return {str(path): digest(path) for path in sorted(set(paths))}


def commands(args, root):
    variants = []
    for kind, type_id in [("float", 0), ("simd_float8", 2)]:
        for implementation in ("legacy", "stf"):
            name = f"{implementation}_cpu_{kind}"
            command = [args.cxx, "-std=c++20", "-O3", "-ffast-math", "-march=native",
                       "-DNDEBUG", "-DORDER=7", "-DUSE_CUDA", "-I" + str(root),
                       "-I" + str(args.cuda_root / "include"),
                       f"-DCOSMICTIGER_STF_BENCHMARK_TYPE={type_id}"]
            if implementation == "legacy":
                command += [f'-DCOSMICTIGER_STF_BASELINE_HEADER="{args.baseline_header}"']
            command += [str(root / "benchmarks/stf_kernels.cpp"), "-o", str(args.output / name)]
            variants.append((name, command))
    if not args.skip_cuda:
        for implementation in ("legacy", "stf"):
            name = f"{implementation}_cuda_float"
            command = [str(args.cuda_root / "bin/nvcc"), "-std=c++20", "-O3", "--use_fast_math",
                       "-DNDEBUG", "-DORDER=7", "-DUSE_CUDA", "-I" + str(root),
                       "-arch=" + args.arch, "--expt-relaxed-constexpr", "-rdc=true",
                       "-ccbin", args.cxx, "--compiler-options=-O3,-ffast-math,-march=native"]
            if implementation == "legacy":
                command += [f'-DCOSMICTIGER_STF_BASELINE_HEADER="{args.baseline_header}"']
            command += [str(root / "benchmarks/stf_kernels_cuda.cu"), "-o", str(args.output / name)]
            variants.append((name, command))
    return variants


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def build(args, root, variants):
    path = args.output / "stf_build_manifest.json"
    if path.exists():
        raise RuntimeError("Build manifest already exists; choose a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    source_hashes = inputs(root, args.baseline_header)
    compilers = sorted({command[0] for _, command in variants})
    manifest = {"root": str(root), "baseline_header": str(args.baseline_header),
                "source_sha256": source_hashes, "builds": [],
                "compiler_versions": {compiler: subprocess.check_output(
                    [compiler, "--version"], text=True) for compiler in compilers}}
    save(path, manifest)
    for name, command in variants:
        print("Building " + name, flush=True)
        (args.output / (name + ".command")).write_text(shlex.join(command) + "\n")
        begin = time.monotonic()
        with (args.output / (name + ".build.log")).open("w") as log:
            status = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT).returncode
        binary = args.output / name
        record = {"name": name, "command": command, "exit_code": status,
                  "seconds": time.monotonic() - begin,
                  "binary_sha256": digest(binary) if status == 0 else None}
        manifest["builds"].append(record)
        save(path, manifest)
        if status:
            raise RuntimeError(f"Compiler failed: see {name}.build.log")
        if inputs(root, args.baseline_header) != source_hashes:
            raise RuntimeError("Sources changed during compilation; rebuild in a fresh directory")
    print("Builds complete. Benchmarks have not run.", flush=True)


def read_result(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        key = (row["operator"], int(row["do_phi"]))
        elapsed = float(row["ns_per_interaction"])
        if key in result or not math.isfinite(elapsed) or elapsed <= 0 or not math.isfinite(float(row["checksum"])):
            raise RuntimeError(f"Invalid benchmark row: {path}: {row}")
        result[key] = elapsed
    if len(result) != 10:
        raise RuntimeError(f"Expected ten benchmark rows: {path}")
    return result


def run(args, root, variants):
    manifest = json.loads((args.output / "stf_build_manifest.json").read_text())
    if inputs(root, args.baseline_header) != manifest["source_sha256"]:
        raise RuntimeError("Prepared sources changed; rebuild before collecting measurements")
    built = {record["name"]: record for record in manifest["builds"]}
    for name, command in variants:
        record = built.get(name)
        binary = args.output / name
        if (not record or record["exit_code"] != 0 or record["command"] != command or
                not binary.is_file() or digest(binary) != record["binary_sha256"]):
            raise RuntimeError("Missing, changed, or differently configured benchmark: " + name)
    affinity = sorted(os.sched_getaffinity(0))
    cpu = args.cpu if args.cpu is not None else affinity[0]
    if cpu not in affinity or shutil.which("taskset") is None:
        raise RuntimeError("Choose an allowed --cpu and install taskset for pinned CPU measurements")
    run_dir = args.output / "measurements"
    run_dir.mkdir(exist_ok=False)
    state = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "platform": platform.platform(), "cpu": cpu, "affinity": affinity,
             "load_average": os.getloadavg(), "commands": []}
    groups = ["cpu_float", "cpu_simd_float8"] + ([] if args.skip_cuda else ["cuda_float"])
    ratios = {}
    for group in groups:
        for pair in range(args.pairs):
            results = {}
            order = ("legacy", "stf") if pair % 2 == 0 else ("stf", "legacy")
            for implementation in order:
                name = implementation + "_" + group
                command = [str(args.output / name)]
                if group.startswith("cpu_"):
                    command = ["taskset", "--cpu-list", str(cpu)] + command
                    command += [str(args.cpu_samples), str(args.cpu_rounds), str(args.repetitions)]
                else:
                    command += [str(args.cuda_samples), str(args.cuda_rounds), str(args.repetitions)]
                prefix = run_dir / f"{group}.pair{pair}.{implementation}"
                csv_path = Path(str(prefix) + ".csv")
                stderr_path = Path(str(prefix) + ".stderr")
                print("Running " + prefix.name, flush=True)
                state["commands"].append(command)
                save(run_dir / "run_manifest.json", state)
                with csv_path.open("w") as stdout, stderr_path.open("w") as stderr:
                    subprocess.run(command, cwd=root, stdout=stdout, stderr=stderr, check=True)
                results[implementation] = read_result(csv_path)
            if results["legacy"].keys() != results["stf"].keys():
                raise RuntimeError("Legacy and STF case sets differ")
            for key in results["legacy"]:
                ratios.setdefault((group, *key), []).append({
                    "legacy_ns": results["legacy"][key], "stf_ns": results["stf"][key],
                    "stf_over_legacy": results["stf"][key] / results["legacy"][key]})
    summary = [{"group": key[0], "operator": key[1], "do_phi": key[2], "pairs": values,
                "median_stf_over_legacy": statistics.median(v["stf_over_legacy"] for v in values)}
               for key, values in sorted(ratios.items())]
    save(run_dir / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", action="store_true", help="compile serially; does not run timings")
    parser.add_argument("--run", action="store_true", help="run and summarize verified prepared binaries")
    parser.add_argument("--cxx", default="g++-13")
    parser.add_argument("--cuda-root", type=Path, default=Path("/usr/local/cuda"))
    parser.add_argument("--arch", default="sm_86")
    parser.add_argument("--skip-cuda", action="store_true", help="omit CUDA compilation and timing")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--pairs", type=positive, default=3)
    parser.add_argument("--repetitions", type=positive, default=7)
    parser.add_argument("--cpu-samples", type=positive, default=128)
    parser.add_argument("--cpu-rounds", type=positive, default=1000)
    parser.add_argument("--cuda-samples", type=positive, default=32768)
    parser.add_argument("--cuda-rounds", type=positive, default=100)
    args = parser.parse_args()
    args.baseline_header = args.baseline_header.resolve(strict=True)
    args.output = args.output.resolve()
    args.cuda_root = args.cuda_root.resolve()
    root = Path(__file__).resolve().parents[1]
    variants = commands(args, root)
    if not args.build and not args.run:
        for _, command in variants:
            print(shlex.join(command))
        return
    if args.build:
        build(args, root, variants)
    if args.run:
        run(args, root, variants)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
