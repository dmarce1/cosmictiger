"""Original close-pair SPH/gravity stress case, with four bounded CPU runs.

Keep the original small softening, 512 gas particles, mass unit and end time.
Compare hierarchical stepping with three successively refined global steps.
Each process has its own timeout and iteration cap; runs are sequential and no
restart, executable discovery, MPI or CUDA process is started by this runner.
"""

import argparse
import copy
import json
from pathlib import Path
import signal
import subprocess
import sys

from run_hierarchical_sph import run_case, state_difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0 or not 0 < args.timeout <= 120:
        parser.error("threads must be positive and each timeout must be in (0, 120]")
    args.executable = args.executable.resolve()
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.scenario = "hydroWave"
    args.resolution = 8
    args.end_time = 0.05
    args.coarse_steps = 4
    args.max_iterations = 4096
    args.minimum_rung = 0
    args.self_gravity = True
    args.mass_solar = 1e26
    args.gravity_opening_angle = 0.8
    args.gravity_softening = 0.018769884569535343
    args.cuda = False
    args.energy_tolerance = 0.002

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    report = {"application_process_limit": 4, "application_processes_started": 0,
              "physical_parameters": {"particles": 512, "mass_solar": args.mass_solar,
                                      "softening": args.gravity_softening,
                                      "end_time": args.end_time},
              "runs": {}, "comparisons": {}}
    states = {}
    cases = (("hierarchical_default_theta08", "hierarchical", 0.1, 1),
             ("global_default_theta08", "global", 0.025, 1),
             ("global_half_eta_theta08", "global", 0.025, 2),
             ("global_quarter_eta_theta08", "global", 0.025, 4))
    try:
        for name, integrator, cfl, refinement in cases:
            options = copy.copy(args)
            options.gravity_eta = 0.20401599335954482 / refinement
            if not args.analyze_only:
                report["application_processes_started"] += 1
            result, state = run_case(options, name, integrator, cfl)
            report["runs"][name] = result
            states[name] = state
        baseline = report["runs"][cases[0][0]]
        if not baseline["work"]["partial_active_events"]:
            raise RuntimeError("close-pair fixture did not exercise mixed gas rungs")
        if not (baseline["work"].get("fine_to_full_median_sort", 1) < 1 and
                baseline["work"].get("fine_to_full_median_density", 1) < 1):
            raise RuntimeError("fine events did not reduce sorting and density work")
        first, second, third = (states[case[0]] for case in cases[1:])
        coarse = state_difference(first, second)
        fine = state_difference(second, third)
        hierarchy = state_difference(states[cases[0][0]], third)
        report["comparisons"].update(global_first_refinement=coarse,
                                     global_second_refinement=fine,
                                     hierarchical_to_refined_global=hierarchy)
        coarse_error, fine_error = max(coarse.values()), max(fine.values())
        if fine_error > 0.0015 or fine_error > coarse_error:
            raise RuntimeError("global timestep refinement did not converge")
        if max(hierarchy.values()) > 0.005:
            raise RuntimeError("hierarchy differs from the refined same-physics global result")
        report["status"] = "passed"
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        report.update(status="failed", error=str(error))
    finally:
        (args.workdir / "gravity_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
