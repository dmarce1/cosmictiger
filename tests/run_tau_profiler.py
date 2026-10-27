"""Check ordinary-function samples attributed to HPX task contexts through APEX/TAU."""

import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile

from run_apex_profiler import check_apex, check_samples, require
from run_interrupt_profiler import run_interrupt_fixture


def read_tau_profiles(directory):
    profiles = sorted(path for path in directory.rglob("profile.*")
                      if re.fullmatch(r"profile\.\d+\.\d+\.\d+", path.name))
    require(profiles, "TAU did not produce per-thread profiles")
    rows = []
    row_pattern = re.compile(
        r'^"(?P<name>(?:\\.|[^"\\])*)"\s+'
        r'(?P<calls>\S+)\s+(?P<subroutines>\S+)\s+'
        r'(?P<exclusive>\S+)\s+(?P<inclusive>\S+)\s+(?P<profile_calls>\S+)')
    for profile in profiles:
        lines = profile.read_text().splitlines()
        header = re.fullmatch(r"(\d+)\s+templated_functions(?:_\S+)?", lines[0]) if lines else None
        require(header is not None, f"unrecognized TAU profile header in {profile}")
        remaining = int(header[1])
        for line in lines[1:]:
            if remaining == 0:
                break
            match = row_pattern.match(line)
            if match is None:
                continue
            row = {key: float(value) for key, value in match.groupdict().items() if key != "name"}
            require(all(math.isfinite(value) and value >= 0 for value in row.values()),
                    f"invalid TAU counters in {profile}: {line}")
            row.update(name=match["name"], profile=profile)
            rows.append(row)
            remaining -= 1
        require(remaining == 0, f"TAU profile {profile} is missing {remaining} function rows")
    return profiles, rows


def check_tau(directory):
    profiles, rows = read_tau_profiles(directory)
    matches = {}
    task_functions = {"cosmictiger.apex.parent": "ordinaryParentFunction",
                      "cosmictiger.apex.child": "ordinaryChildFunction",
                      "cosmictiger.apex.migrated": "ordinaryMigratedFunction"}
    for task, function in task_functions.items():
        require(any(row["name"] == task and row["calls"] > 0 for row in rows),
                f"APEX did not forward the {task} task timer to TAU")
        context_pattern = re.compile(r"^\[(?:CONTEXT|INTERMEDIATE)\]\s+" +
                                     re.escape(task) + r"(?:\s|$)")
        task_matches = []
        for row in rows:
            if row["calls"] <= 0:
                continue
            segments = re.split(r"\s+=>\s+", row["name"])
            for index, segment in enumerate(segments):
                # TAU 2.35 calls this intermediate attribution node CONTEXT;
                # older TAU profiles label the same node INTERMEDIATE.
                if not context_pattern.search(segment):
                    continue
                if not any(part.strip() == task for part in segments[:index]):
                    continue
                sample_segments = [part for part in segments[index + 1:] if "[SAMPLE]" in part]
                for other_task, other_function in task_functions.items():
                    if other_task != task:
                        require(not any(re.search(r"\b" + other_function + r"\b", part)
                                        for part in sample_segments),
                                f"TAU attributed {other_function} to the wrong task {task}: "
                                f"{row['profile']}: {row['name']}")
                if row["inclusive"] > 0 and any(
                        re.search(r"\b" + function + r"\b", part) for part in sample_segments):
                    task_matches.append(row)
                    break
        require(task_matches,
                f"TAU did not attribute {function} samples beneath the {task} context; "
                "separate task timers and global function samples are insufficient")
        matches[task] = task_matches
    # Each metric may have its own MULTI_* directory. Count unique rank/context/
    # thread file names, so two metrics on one worker do not fake migration.
    migrated_threads = {row["profile"].name for row in matches["cosmictiger.apex.migrated"]}
    require(len(migrated_threads) >= 2,
            "TAU lost ordinaryMigratedFunction attribution when its HPX context changed workers; "
            f"positive samples appeared in only {sorted(migrated_threads)}")
    return profiles, matches


def run_regression(args, directory):
    require(not any(directory.iterdir()), "TAU regression output directory must be empty")
    if args.apex_config:
        (directory / "apex.conf").write_text(
            "APEX_DISABLE=1\nAPEX_SUSPEND=1\nAPEX_TIME_TOP_LEVEL_OS_THREADS=1\n")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("APEX_", "TAU_"))}
    environment.update({"TAU_EXEC": str(args.tau_exec.resolve()),
                        "COSMICTIGER_PROFILE_DIR": str(directory),
                        "PROFILEDIR": str(directory / "tau"),
                        "APEX_THROTTLE_TIMERS": "0",
                        "HWLOC_COMPONENTS": "-gl", "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1"})
    command = [str(args.launcher.resolve()), str(args.executable.resolve())]
    if args.sampler_disabled:
        command.append("--sampler-disabled")
    if args.interrupt_exit:
        command.append("--interrupt-exit")
    command.extend(("--hpx:threads=2", "--hpx:bind=none"))
    if args.interrupt_exit:
        result = run_interrupt_fixture(command, directory, environment,
                                       getattr(signal, args.interrupt_signal))
    else:
        result = subprocess.run(command, cwd=directory, env=environment,
                                text=True, errors="replace", stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120)
    (directory / "fixture_output.txt").write_text(result.stdout)
    try:
        require(result.returncode == 0, f"TAU fixture failed with status {result.returncode}")
        if args.require_compiled_disabled:
            require(result.stdout.count("APEX_CPU_SAMPLER_COMPILED=0") == 1,
                    "TAU fixture was not compiled with PROFILE_DISABLE")
        full_markers = re.findall(r"^APEX_FULL_THREAD_DESCRIPTION=([01])$",
                                  result.stdout, re.MULTILINE)
        require(len(full_markers) == 1, "fixture did not report its FULL description setting")
        migration_markers = re.findall(r"^APEX_MIGRATION_WORKERS=(\d+),(\d+)$",
                                       result.stdout, re.MULTILINE)
        require(len(migration_markers) == 1 and
                migration_markers[0][0] != migration_markers[0][1],
                "fixture did not confirm HPX task migration between two distinct workers")
        check_apex(directory / "apex", full_markers[0] == "1")
        profiles, contexts = check_tau(directory / "tau")
        if args.sampler_disabled:
            require(not (directory / "cpu_samples.txt").exists() and
                    not list(directory.glob("cosmictiger.profile.*.txt")),
                    "disabled CPU sampling unexpectedly produced reports")
        else:
            check_samples(directory)
    except Exception as exception:
        csv_output = directory / "apex" / "apex_profiles.csv"
        details = csv_output.read_text(errors="replace") if csv_output.is_file() else "<missing CSV>"
        sample_rows = []
        for profile in sorted((directory / "tau").rglob("profile.*")):
            if profile.is_file():
                sample_rows.extend(f"{profile.relative_to(directory)}: {line}"
                                   for line in profile.read_text(errors="replace").splitlines()
                                   if re.search(r"ordinary(?:Hot|Parent|Child|Migrated)Function", line)
                                   or "cosmictiger.apex." in line)
        raise RuntimeError(f"{exception}\nAPEX CSV:\n{details}\n"
                           f"Relevant TAU rows:\n" + "\n".join(sample_rows[:200]) +
                           f"\nFixture output:\n{result.stdout}") from exception
    print(f"TAU profiler regression passed: {len(profiles)} thread profiles; "
          "distinct ordinary functions attributed to their named parent and child contexts; "
          "migrated context samples on both workers; APEX completion/yield counts and task tree; " +
          ("disabled CPU sampler" if args.sampler_disabled else "coexisting CPU sampler"))
    for task, rows in contexts.items():
        print(f"  {task}: {len(rows)} positive contextual sample rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("launcher", type=Path)
    parser.add_argument("tau_exec", type=Path)
    parser.add_argument("--sampler-disabled", action="store_true")
    parser.add_argument("--require-compiled-disabled", action="store_true")
    parser.add_argument("--interrupt-exit", action="store_true")
    parser.add_argument("--interrupt-signal", choices=("SIGINT", "SIGUSR1"), default="SIGINT")
    parser.add_argument("--apex-config", action="store_true",
                        help="check that a late APEX config cannot disable TAU task capture")
    parser.add_argument("--output-dir", type=Path,
                        help="retain evidence in an empty directory instead of a temporary one")
    args = parser.parse_args()
    if args.require_compiled_disabled:
        args.sampler_disabled = True
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)
    with (nullcontext(args.output_dir) if args.output_dir is not None else
          tempfile.TemporaryDirectory(prefix="cosmictiger_tau_profiler_")) as output:
        run_regression(args, Path(output))


if __name__ == "__main__":
    main()
