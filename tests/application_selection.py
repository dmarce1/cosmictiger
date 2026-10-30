"""Choose explicit compiled algorithms for application regression runners."""

import os
import re
import subprocess
from functools import lru_cache


@lru_cache(maxsize=None)
def compiled_algorithms(executable):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([str(executable), "--help", "--compute.cuda=false",
                             "--hpx:threads=1", "--hpx:bind=none"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=environment, timeout=30, check=True)
    def choices(label):
        match = re.search(r"^\s*" + re.escape(label) + r":([^\n]*)", result.stdout, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"application help omitted compiled {label}")
        return match[1].split()
    return {"far": choices("Gravity solvers"), "near": choices("PMMM near solvers"),
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
    far = select(far, compiled["far"], ("fmm", "pmmm", "none"), "far solver")
    eligible = [value for value in compiled["integrator"]
                if value != "hierarchical" or (not hydro and far != "none")]
    integrator = select(integrator, eligible, ("traditional", "global", "hierarchical"), "integrator")
    result = {"run.timeIntegrator": integrator}
    if far != "none":
        result["gravity.method.far"] = far
    if far == "pmmm":
        result["gravity.method.near"] = select(near, compiled["near"], ("pp", "fmm"), "near solver")
    return result


def application_arguments(executable, **options):
    return [f"--{key}={value}" for key, value in application_options(executable, **options).items()]
