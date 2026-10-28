"""Check HPX-owned APEX task lifecycles with optional independent CPU sampling."""

import argparse
import csv
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_apex(directory, full_thread_descriptions):
    profile = directory / "apex_profiles.csv"
    require(profile.is_file(), "HPX did not produce the APEX CSV profile")
    with profile.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(rows and {"name", "type", "num samples/calls", "yields"} <= rows[0].keys(),
            "APEX CSV profile is missing timer-count columns")
    expected = {"cosmictiger.apex.parent": 4, "cosmictiger.apex.child": 4,
                "cosmictiger.apex.post": 1, "cosmictiger.apex.continuation": 1,
                "ApexRegressionAction": 1, "cosmictiger.apex.migrated": 1,
                "cosmictiger.apex.lambda": 12}
    for name, count in expected.items():
        matches = [row for row in rows if row["name"] == name and row["type"] == "timer"]
        require(len(matches) == 1, f"missing or duplicate APEX timer for {name}: {matches}")
        require(float(matches[0]["num samples/calls"]) == count,
                f"APEX task completions were lost or resumes became extra calls: {matches[0]}")
        if name != "cosmictiger.apex.child":
            require(float(matches[0]["yields"]) > 0,
                    f"APEX did not observe suspended/resumed contexts for {name}")
    for name in ("cosmictiger.apex.sort", "cosmictiger.apex.copy", "cosmictiger.apex.fill"):
        matches = [row for row in rows if row["name"] == name and row["type"] == "timer"]
        require(len(matches) == 1 and float(matches[0]["num samples/calls"]) >= 1,
                f"parallel algorithm tasks did not retain their phase annotation: {name}: {matches}")
    if full_thread_descriptions:
        for name in ("detail::unannotatedTask(", "detail::unannotatedForkTask("):
            matches = [row for row in rows if name in row["name"] and row["type"] == "timer"]
            require(len(matches) == 1,
                    f"FULL thread descriptions did not resolve {name}: {matches}")
            require(float(matches[0]["num samples/calls"]) == 1 and
                    float(matches[0]["yields"]) > 0,
                    f"unannotated task completion or suspension was lost: {matches[0]}")
    graphs = list(directory.glob("taskgraph.*.dot"))
    if graphs:
        require(len(graphs) == 1, "APEX did not produce one locality task graph")
        graph = graphs[0].read_text()
        require(re.search(r'"cosmictiger\.apex\.parent"\s*->\s*'
                          r'"cosmictiger\.apex\.child"', graph) is not None,
                "APEX task graph lost the async parent/child dependency")
    else:
        # APEX 1.11 writes either a graph or a tree when both are requested.
        with (directory / "apex_tasktree.csv").open(newline="") as stream:
            tree = list(csv.DictReader(stream))
        parents = {(row["process rank"], row["node index"]) for row in tree
                   if row["name"] == "cosmictiger.apex.parent"}
        require(any(row["name"] == "cosmictiger.apex.child" and
                    (row["process rank"], row["parent index"]) in parents for row in tree),
                "APEX task tree lost the async parent/child dependency")


def check_samples(directory):
    report = (directory / "cpu_samples.txt").read_text()
    require("Sampling is still active" not in report, "CPU sample report was not stopped")
    diagnostics = re.search(
        r"Delivered samples: (\d+); dropped samples: (\d+); "
        r"timer overruns \(missing ticks\): (\d+); setup failures: (\d+)", report)
    require(diagnostics is not None, "CPU sampler diagnostics are missing")
    samples, dropped, overruns, failures = map(int, diagnostics.groups())
    require(samples > 0 and dropped == 0 and failures == 0,
            f"CPU sampling failed alongside APEX: {diagnostics[0]}")
    workers = re.search(r"Attached OS workers: (\d+); registered function labels: (\d+)", report)
    require(workers is not None and int(workers[1]) >= 3,
            "CPU sampler did not attach the main thread and both HPX workers")
    rows = re.findall(r"^\s*(\d+)\s+([0-9.e+-]+)\s+([0-9.]+)\s+(.+)$",
                      report, re.MULTILINE)
    require(any("detail::ordinaryHotFunction(" in row[3] for row in rows),
            "CPU samples missed the ordinary unannotated function inside HPX contexts")
    require(sum(int(row[0]) for row in rows) == samples - dropped,
            "CPU function rows do not conserve the delivered sample count")
    for count, seconds, percentage, function in rows:
        require(math.isclose(float(seconds), int(count) * 0.01, rel_tol=1e-6),
                f"wrong 10 ms CPU-time estimate for {function}")
        require(math.isclose(float(percentage), 100 * int(count) / samples, abs_tol=0.011),
                f"wrong CPU sample percentage for {function}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--sampler-disabled", action="store_true")
    parser.add_argument("--reports-disabled", action="store_true")
    args = parser.parse_args()
    executable = args.executable.resolve()
    with tempfile.TemporaryDirectory(prefix="cosmictiger_apex_profiler_") as temporary:
        directory = Path(temporary)
        # Each regression owns its APEX output settings, independent of a user's
        # tracing, sampling, filtering, or throttling settings in the environment.
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("APEX_")}
        environment.update({"APEX_THROTTLE_TIMERS": "0",
                            "APEX_OUTPUT_FILE_PATH": str(directory), "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1", "HWLOC_COMPONENTS": "-gl"})
        if args.reports_disabled:
            environment.update({name: "0" for name in
                                ("APEX_SCREEN_OUTPUT", "APEX_CSV_OUTPUT",
                                 "APEX_TASKTREE_OUTPUT", "APEX_TASKGRAPH_OUTPUT")})
        command = [str(executable)]
        if args.sampler_disabled:
            command.append("--sampler-disabled")
        command.extend(("--hpx:threads=2", "--hpx:bind=none"))
        result = subprocess.run(command, cwd=directory, env=environment,
                                text=True, errors="replace", stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=45)
        try:
            require(result.returncode == 0,
                    f"APEX fixture failed with status {result.returncode}")
            full_markers = re.findall(r"^APEX_FULL_THREAD_DESCRIPTION=([01])$",
                                      result.stdout, re.MULTILINE)
            require(len(full_markers) == 1, "fixture did not report its FULL description setting")
            require("CPU Timers" not in result.stdout and
                    "CosmicTiger statistical CPU profile (" not in result.stdout,
                    "detailed profiler reports leaked to the terminal")
            if args.reports_disabled:
                require(not list(directory.glob("apex*")) and
                        not list(directory.glob("taskgraph.*")),
                        "explicit APEX report opt-outs were ignored")
                require("CPU Timers" not in result.stdout,
                        "explicit APEX screen output opt-out was ignored")
            else:
                check_apex(directory, full_markers[0] == "1")
                require(list(directory.glob("apex_tasktree*")),
                        "APEX task tree was not enabled by default")
            if args.sampler_disabled:
                require(not (directory / "cpu_samples.txt").exists() and
                        not list(directory.glob("cosmictiger.profile.*.txt")),
                        "disabled CPU sampling unexpectedly produced reports")
            else:
                check_samples(directory)
        except Exception as exception:
            csv_output = directory / "apex_profiles.csv"
            details = csv_output.read_text(errors="replace") if csv_output.is_file() else "<missing CSV>"
            raise RuntimeError(f"{exception}\nAPEX CSV:\n{details}\n"
                               f"Fixture output:\n{result.stdout}") from exception
    print("APEX profiler regression passed: async, fork, post, continuation, action, resumed contexts, "
          "task graph, " + ("automatic task naming, " if full_markers[0] == "1" else
                            "automatic task naming skipped (FULL disabled), ") +
          ("disabled CPU sampler" if args.sampler_disabled else "ordinary-function CPU samples"))


if __name__ == "__main__":
    main()
