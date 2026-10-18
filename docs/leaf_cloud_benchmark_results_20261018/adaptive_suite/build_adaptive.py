#!/usr/bin/env python3
"""Prepare a common adaptive walk harness; compile/link only with --build.

This script never runs benchmarks or changes the existing application objects.
The original comparison manifest supplies provenance for every reused object.
"""

import argparse
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--parent-manifest", type=Path)
    args = parser.parse_args()
    work = Path(__file__).resolve().parent
    comparison = work.parent
    repo = comparison.parent.parent
    parent_path = args.parent_manifest or (
        repo / "docs/leaf_cloud_benchmark_results_20261018/build_manifest.json")
    parent_path = parent_path.resolve(strict=True)
    parent = json.loads(parent_path.read_text())
    build = Path(parent["build_directory"])
    baseline = comparison / "baseline_source"
    source = work / "stf_walk.cpp"
    object_path = work / "stf_walk.cpp.o"
    old_harness = "CMakeFiles/cosmictiger.dir/benchmarks/stf_walk.cpp.o"
    old_device = "CMakeFiles/cosmictiger.dir/cmake_device_link.o"
    changed_sources = ["src/kick.cpp", "src/kick.cu", "tests/fmm_walk.cpp"]
    replacements = {
        "CMakeFiles/cosmictiger.dir/" + path + ".o":
        comparison / "baseline_objects" / (path + ".o")
        for path in changed_sources
    }

    # A dry run only inspects the existing build. Never let this helper rebuild
    # shared objects or silently accept a changed comparison baseline.
    pending = subprocess.check_output(
        ["ninja", "-C", str(build), "-n", "cosmictiger"], text=True)
    if "no work to do" not in pending:
        raise RuntimeError("Current application build is not up to date:\n" + pending)
    checked_inputs = {str(parent_path): digest(parent_path), str(source): digest(source)}
    for relative, expected in parent["shared_object_sha256"].items():
        path = build / relative
        if digest(path) != expected:
            raise RuntimeError("Shared object changed since original comparison: " + str(path))
        checked_inputs[str(path)] = expected
    for relative, before_path in replacements.items():
        expected = parent["replacement_objects"][relative]
        if digest(before_path) != expected["before"]:
            raise RuntimeError("Baseline object changed: " + str(before_path))
        if digest(build / relative) != expected["after"]:
            raise RuntimeError("Current object changed: " + str(build / relative))
        checked_inputs[str(before_path)] = expected["before"]
        checked_inputs[str(build / relative)] = expected["after"]
    for relative, expected in parent["changed_source_sha256"].items():
        for path, version in [(repo / relative, "after"), (baseline / relative, "before")]:
            if digest(path) != expected[version]:
                raise RuntimeError("Comparison source changed: " + str(path))
            checked_inputs[str(path)] = expected[version]

    entries = json.loads((build / "compile_commands.json").read_text())
    entry = next(row for row in entries
                 if row["file"] == str(repo / "benchmarks/stf_walk.cpp"))
    compile_command = shlex.split(entry["command"])
    compile_command = [str(source) if part == entry["file"] else part
                       for part in compile_command]
    for flag in ["-o", "-MF", "-MT"]:
        if flag in compile_command:
            compile_command[compile_command.index(flag) + 1] = (
                str(object_path) + (".d" if flag == "-MF" else ""))
    all_commands = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "commands", "cosmictiger"],
        text=True).splitlines()
    device_template = shlex.split(next(line for line in all_commands if " -dlink " in line))
    link_text = all_commands[-1]
    if not (link_text.startswith(": && ") and link_text.endswith(" && :")):
        raise RuntimeError("Unrecognized Ninja final-link command")
    link_template = shlex.split(link_text[len(": && "):-len(" && :")])
    if old_harness not in link_template or old_harness not in device_template:
        raise RuntimeError("Expected common harness object missing from link commands")
    common_objects = {}
    for part in link_template:
        if not part.endswith(".o") or part in replacements or part in (old_harness, old_device):
            continue
        expected = parent["shared_object_sha256"].get(part)
        if expected is None or digest(build / part) != expected:
            raise RuntimeError("Unverified common link input: " + part)
        common_objects[part] = expected

    commands = [("harness", compile_command)]
    binaries = {}
    for variant in ["before", "after"]:
        object_map = dict(replacements) if variant == "before" else {}
        object_map[old_harness] = object_path
        object_map[old_device] = work / (variant + "_device_link.o")
        device_command = [str(object_map.get(part, part)) for part in device_template]
        link_command = [str(object_map.get(part, part)) for part in link_template]
        binary = work / ("cosmictiger_" + variant)
        link_command[link_command.index("-o") + 1] = str(binary)
        commands += [(variant + "_device_link", device_command),
                     (variant + "_link", link_command)]
        binaries[variant] = {"path": str(binary)}

    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "baseline_commit": parent["baseline_commit"],
        "parent_manifest": str(parent_path),
        "parent_manifest_sha256": digest(parent_path),
        "build_directory": str(build),
        "build_method": "Compile one common adaptive harness; regenerate each CUDA device link and final link with the verified before/after traversal objects.",
        "harness_path": str(source),
        "harness_sha256": digest(source),
        "checked_input_sha256": checked_inputs,
        "shared_object_sha256": common_objects,
        "replacement_objects": {relative: parent["replacement_objects"][relative]
                                for relative in replacements},
        "commands": [{"name": name, "arguments": command, "cwd": str(build)}
                     for name, command in commands],
        "fixtures": ["lattice", "clustered", "adaptive"],
        "fixture_environment": "COSMICTIGER_STF_WALK_FIXTURE=all|lattice|clustered|adaptive",
        "sample_environment": "COSMICTIGER_STF_WALK_SAMPLES=1..100; default 5",
        "histogram": "Occupied leaf depth/count histogram once per selected fixture, after warmup tree timing and before warmup kick timing.",
        "lifecycle": "Both versions use this same benchmark-only harness: stop the memory daemon and wait for the HPX pool once before workloads; wait for pool quiescence after every kick inside its timed interval and before tree cleanup/reuse. Production source is unchanged.",
        "binaries": binaries,
        "completed": False,
    }
    if not args.build:
        for name, command in commands:
            print(name + ": " + shlex.join(command))
        print("Preparation verified; pass --build to compile/link. No timings run.")
        return

    manifest_path = work / "build_manifest.json"
    if manifest_path.exists() or any(Path(row["path"]).exists() for row in binaries.values()):
        raise RuntimeError("Adaptive build output already exists; preserve it before rebuilding")
    (work / "harness_changes.patch").write_text("".join(difflib.unified_diff(
        (repo / "benchmarks/stf_walk.cpp").read_text().splitlines(keepends=True),
        source.read_text().splitlines(keepends=True),
        fromfile="benchmarks/stf_walk.cpp", tofile="adaptive_suite/stf_walk.cpp")))
    save(manifest_path, manifest)
    manifest["executed"] = []
    for name, command in commands:
        with (work / (name + ".build.log")).open("w") as log:
            result = subprocess.run(command, cwd=build, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        manifest["executed"].append({"name": name, "exit_code": result.returncode})
        save(manifest_path, manifest)
        if result.returncode:
            raise RuntimeError("Build failed; see " + str(work / (name + ".build.log")))
        print("Completed " + name, flush=True)
    for path, expected in checked_inputs.items():
        if digest(path) != expected:
            raise RuntimeError("Input changed during build: " + path)
    manifest["harness_object_sha256"] = digest(object_path)
    manifest["device_link_sha256"] = {
        variant: digest(work / (variant + "_device_link.o")) for variant in binaries
    }
    for row in binaries.values():
        row["sha256"] = digest(row["path"])
    manifest["completed"] = True
    save(manifest_path, manifest)
    print("Adaptive comparison binaries prepared. No timings run.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
