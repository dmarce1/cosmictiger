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
    near = choices("PMMM near solvers")
    try:
        short = choices("P3M short solvers")
    except RuntimeError:
        short = []
    return {"far": choices("Gravity solvers"), "near": near, "short": short,
            "integrator": choices("Time integrators")}


def application_options(executable, *, far=None, near=None, integrator=None, hydro=False):
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
    far = select(far, compiled["far"], ("fmm", "pmmm", "p3m", "pm", "none"), "far solver")
    # Hierarchical integration is collisionless only; hydro uses the traditional
    # or global integrator, and the particle-mesh solvers exclude global.
    mesh = far in ("pm", "p3m")
    eligible = [value for value in compiled["integrator"]
                if (value != "hierarchical" or (far != "none" and not hydro))
                and (value != "global" or not mesh)]
    integrator = select(integrator, eligible, ("traditional", "global", "hierarchical"), "integrator")
    result = {"run.timeIntegrator": integrator}
    if far != "none":
        result["gravity.method.far"] = far
    if far == "pmmm":
        result["gravity.method.near"] = select(near, compiled["near"], ("pp", "fmm"), "near solver")
    if far == "p3m":
        result["gravity.method.near"] = select(near, compiled["short"], ("pp", "fmm"), "short solver")
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
