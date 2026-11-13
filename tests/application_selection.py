"""Choose explicit compiled algorithms for application regression runners."""

import os
import re
import subprocess
from functools import lru_cache


@lru_cache(maxsize=None)
def compiled_algorithms(executable):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([str(executable), "--help", "--gravity.cuda=false",
                             "--hpx:threads=1", "--hpx:bind=none"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=environment, timeout=30, check=True)
    def choices(label):
        match = re.search(r"^\s*" + re.escape(label) + r":([^\n]*)", result.stdout, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"application help omitted compiled {label}")
        return match[1].split()
    # The global FMM is the sole gravity solver.
    return {"far": choices("Gravity solvers"), "near": ["fmm"] if "fmm" in choices("Gravity solvers") else [],
            "integrator": choices("Time integrators")}


def has_gravity(executable):
    return "fmm" in compiled_algorithms(str(executable))["far"]


def application_options(executable, *, near=None, integrator=None, hydro=False):
    compiled = compiled_algorithms(str(executable))
    def select(requested, available, preference, label):
        if requested is not None:
            if requested not in available:
                raise RuntimeError(f"requested regression {label}={requested} is not compiled")
            return requested
        for choice in preference:
            if choice in available:
                return choice
        raise RuntimeError(f"no compiled {label} supports this regression")
    gravity = "fmm" in compiled["far"]
    # Midpoint SPH also runs without self-gravity under the sole scheme.
    integrator = select(integrator, compiled["integrator"], ("hierarchical",), "integrator")
    result = {}
    if gravity:
        select(near, compiled["near"], ("fmm",), "near solver")
    return result


def application_arguments(executable, **options):
    return [f"--{key}={value}" for key, value in application_options(executable, **options).items()]


def cpu_backend_options():
    """Override GPU-enabled presets without changing the selected physics."""
    keys = ("gravity.cuda", "gravity.error.cuda", "hydro.cuda", "drift.cuda", "sort.cuda",
            "initial.cuda", "chemistry.cuda", "starFormation.cuda", "analysis.power.cuda",
            "analysis.groups.cuda", "analysis.tracers.cuda", "analysis.slice.cuda",
            "analysis.views.cuda", "lightcone.cuda")
    return {key: "false" for key in keys}
