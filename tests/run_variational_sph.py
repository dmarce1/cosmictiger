# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

"""Validate hydro.pressureOperator=variational in the application.

The driver runs the Sod, Sedov blast, and hydroWave scenarios with both SPH
pressure operators and every hydro-capable integrator, a CFL-halving check, small
SPH cosmologies for two adiabatic indices, the chemistry rejection, and, when
requested, two-locality and CUDA comparisons. Every run uses a fresh directory
under workdir. One line is printed per run and per check; the exit status is
nonzero when a check fails. A KNOWN status marks a failure of the Monaghan
baseline itself, which says nothing about the variational operator.
"""

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

from application_selection import application_arguments, application_options, compiled_algorithms

kRoot = Path(__file__).resolve().parents[1]
kConfigurations = kRoot / "configurations"
kOperators = ("monaghan", "variational")
kHydroIntegrators = ("traditional", "global")
kRunTimeout = 300
kCosmologyTimeout = 240
# Shared by every scenario run: text diagnostics only, no checkpoint before the end.
kCommonOptions = {
    "output.directory": ".",
    "output.silo": "false",
    "analysis.views": "false",
    "analysis.power": "false",
    "analysis.groups": "false",
    "lightcone.enabled": "false",
    "checkpoint.intervalSeconds": "1000000000",
}
kRejectionMessage = "constant-gamma ideal gas"


class RunSpec:
    """One application process: a scenario, backend, operator, integrator, and size."""

    def __init__(self, name, stem, executable, *, backend="cpu", localities=1,
                 operator="monaghan", integrator="traditional", resolution=16,
                 overrides=None, kind="fixed", expectSuccess=True, rawCommand=None,
                 timeout=kRunTimeout):
        self.name = name
        self.stem = stem
        self.executable = executable
        self.backend = backend
        self.localities = localities
        self.operator = operator
        self.integrator = integrator
        self.resolution = resolution
        self.overrides = dict(overrides or {})
        self.kind = kind
        self.expectSuccess = expectSuccess
        self.rawCommand = rawCommand
        self.timeout = timeout


class ProcessRegistry:
    """Track live process groups so an interrupted driver leaves nothing running."""

    def __init__(self):
        self.lock_ = threading.Lock()
        self.groups_ = set()

    def add(self, pid):
        with self.lock_:
            self.groups_.add(pid)

    def discard(self, pid):
        with self.lock_:
            self.groups_.discard(pid)

    def killAll(self):
        with self.lock_:
            groups = list(self.groups_)
        for pid in groups:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


kRegistry = ProcessRegistry()


def killGroup(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


def environmentFor(spec):
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1")
    if spec.backend != "cuda":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def prepareDirectory(base, name):
    directory = base / name
    if directory.parent != base:
        raise RuntimeError(f"case directory {directory} escaped {base}")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    return directory


def buildCommand(spec, mpiExec):
    if spec.rawCommand is not None:
        command = list(spec.rawCommand)
    else:
        command = [str(spec.executable), "--hpx:threads=2", "--hpx:bind=none",
                   f"--configuration={kConfigurations / (spec.stem + '.ini')}"]
        command += application_arguments(spec.executable, far="fmm", integrator=spec.integrator,
                                         hydro=True)
        options = dict(kCommonOptions)
        options.update({
            "compute.cuda": "true" if spec.backend == "cuda" else "false",
            "initial.particlesPerDimension": str(spec.resolution),
            "hydro.pressureOperator": spec.operator,
        })
        options.update(spec.overrides)
        command += [f"--{key}={value}" for key, value in options.items()]
    if spec.localities > 1:
        command = [mpiExec, "-n", str(spec.localities)] + command + [
            f"--hpx:localities={spec.localities}"]
    return command


def runProcess(command, directory, environment, timeout):
    logPath = directory / "run.log"
    start = time.monotonic()
    timedOut = False
    with logPath.open("w") as log:
        with subprocess.Popen(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                              text=True, env=environment, start_new_session=True) as process:
            kRegistry.add(process.pid)
            try:
                returnCode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                killGroup(process)
                returnCode = None
                timedOut = True
            finally:
                kRegistry.discard(process.pid)
    return returnCode, timedOut, time.monotonic() - start, logPath.read_text(errors="replace")


def readTable(path, columns):
    if not path.is_file():
        raise ValueError(f"missing {path.name}")
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if columns is not None and len(fields) != columns:
            raise ValueError(f"{path.name} row has {len(fields)} columns, expected {columns}")
        rows.append([float(value) for value in fields])
    if not rows:
        raise ValueError(f"empty {path.name}")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError(f"nonfinite values in {path.name}")
    return rows


def setting(output, key):
    matches = re.findall(r"^" + re.escape(key) + r"\s*:\s*(\S+)\s*$", output, re.MULTILINE)
    return matches[-1] if matches else None


def failureExcerpt(directory, output):
    """Return the most specific failure line the application left behind."""
    texts = [output]
    stackTrace = directory / "stack_trace.txt"
    if stackTrace.is_file():
        texts.insert(0, stackTrace.read_text(errors="replace"))
    for pattern in (r"Condition \"[^\n]*", r"CosmicTiger failed:[^\n]*", r"Error in [^\n]*",
                    r"[^\n]*(?:error|Error|ERROR)[^\n]*"):
        for text in texts:
            match = re.search(pattern, text)
            if match:
                return match[0].strip()[:300]
    tail = [line for line in output.splitlines() if line.strip()]
    return tail[-1][:300] if tail else "no output"


def lastStep(directory):
    path = directory / "progress.txt"
    if not path.is_file():
        return None
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    scenarioRows = [row for row in rows if re.fullmatch(r"[0-9]+", row[0])]
    if scenarioRows:
        return int(scenarioRows[-1][0])
    # Cosmology rows begin with the runtime and number iterations from zero.
    return sum(1 for row in rows if len(row) > 1 and re.fullmatch(r"[0-9]+", row[1])) or None


def analyzeRun(spec, directory, returnCode, timedOut, wall, output):
    result = {"name": spec.name, "stem": spec.stem, "backend": spec.backend,
              "localities": spec.localities, "operator": spec.operator,
              "integrator": spec.integrator, "resolution": spec.resolution,
              "overrides": spec.overrides, "kind": spec.kind, "wall": wall,
              "returnCode": returnCode, "directory": str(directory)}
    if timedOut:
        result.update(status="timeout", message=f"exceeded {spec.timeout} s")
        return result
    if spec.kind == "reject":
        rejected = returnCode != 0 and kRejectionMessage in output
        result.update(status="rejected" if rejected else "accepted",
                      message=failureExcerpt(directory, output) if returnCode else "exited 0")
        return result
    if returnCode != 0:
        result.update(status="abort", message=failureExcerpt(directory, output),
                      steps=lastStep(directory))
        return result
    try:
        if spec.kind == "migration":
            if "SPH gravity regression passed" not in output:
                raise ValueError("success message missing")
            result.update(status="ok", message=re.search(
                r"SPH gravity regression passed[^\n]*", output)[0])
            return result
        expected = {"hydro.pressureOperator": spec.operator,
                    "run.timeIntegrator": spec.integrator,
                    "compute.cuda": "T" if spec.backend == "cuda" else "F"}
        for key, value in expected.items():
            if setting(output, key) != value:
                raise ValueError(f"resolved {key}={setting(output, key)}, expected {value}")
        runtime = re.search(r"^Runtime\s*:\s*(\d+) localities", output, re.MULTILINE)
        if runtime is None or int(runtime[1]) != spec.localities:
            raise ValueError("resolved locality count differs from the request")
        result["steps"] = lastStep(directory)
        if spec.kind == "cosmology":
            rows = readTable(directory / "energy.txt", 10)
            errors = [row[8] for row in rows]
            result.update(energyRows=len(rows), finalEnergyError=errors[-1],
                          maxEnergyError=max(abs(value) for value in errors),
                          finalScale=rows[-1][1])
        else:
            if not re.search(r":\s+finished at t=", output):
                raise ValueError("scenario did not reach its end time")
            rows = readTable(directory / "energy.dat", 9)
            density = readTable(directory / "rho.dat", 2)
            result.update(energyRows=len(rows), finalEnergyError=rows[-1][8],
                          maxEnergyError=max(abs(row[8]) for row in rows),
                          kinetic=[row[4] for row in rows], thermal=[row[5] for row in rows],
                          maxDensity=max(row[1] for row in density))
            if spec.stem == "sod":
                samples = readTable(directory / "sod.txt", 5)
                if len(samples) != 1000:
                    raise ValueError(f"sod.txt has {len(samples)} samples, expected 1000")
                if any(min(row[1:]) <= 0.0 for row in samples):
                    raise ValueError("Sod density or pressure is nonpositive")
                # Columns: x, SPH density, exact density, SPH pressure, exact pressure.
                result["l1"] = (sum(abs(row[1] - row[2]) for row in samples) /
                                sum(row[2] for row in samples))
        result.update(status="ok", message="")
    except (ValueError, TypeError) as error:
        result.update(status="bad-output", message=str(error))
    return result


def execute(spec, base, mpiExec):
    directory = prepareDirectory(base, spec.name)
    command = buildCommand(spec, mpiExec)
    (directory / "command.txt").write_text(" ".join(command) + "\n")
    returnCode, timedOut, wall, output = runProcess(command, directory, environmentFor(spec),
                                                    spec.timeout)
    result = analyzeRun(spec, directory, returnCode, timedOut, wall, output)
    result["command"] = command
    return result


def formatNumber(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}e}"


def runLine(result):
    fields = [f"RUN {result['name']:<40}", f"{result['backend']:<4}",
              f"loc={result['localities']}", f"op={result['operator']:<11}",
              f"int={result['integrator']:<11}", f"n={result['resolution']:<2}",
              f"status={result['status']:<10}", f"wall={result['wall']:6.1f}s"]
    if result.get("steps") is not None:
        fields.append(f"steps={result['steps']}")
    for key, label in (("l1", "L1"), ("finalEnergyError", "dE_final"),
                       ("maxEnergyError", "max|dE|"), ("maxDensity", "rho_max")):
        if result.get(key) is not None:
            fields.append(f"{label}={formatNumber(result[key])}")
    if result.get("message"):
        fields.append(f"[{result['message']}]")
    return " ".join(fields)


def relativeDifference(a, b):
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale > 0.0 else 0.0


def compareEnergies(reference, other):
    """Final-row relative kinetic/thermal differences and the all-row maximum of
    |difference| normalized by the reference total K+U of that row."""
    count = min(len(reference["kinetic"]), len(other["kinetic"]))
    final = (relativeDifference(other["kinetic"][-1], reference["kinetic"][-1]),
             relativeDifference(other["thermal"][-1], reference["thermal"][-1]))
    rowMaximum = 0.0
    for index in range(count):
        total = abs(reference["kinetic"][index]) + abs(reference["thermal"][index])
        for column in ("kinetic", "thermal"):
            difference = abs(other[column][index] - reference[column][index])
            rowMaximum = max(rowMaximum, difference / total if total > 0.0 else difference)
    return final, rowMaximum, len(reference["kinetic"]) == len(other["kinetic"])


class Validation:
    """Collect checks and print one line for each."""

    def __init__(self):
        self.checks_ = []

    def add(self, name, status, detail):
        self.checks_.append({"name": name, "status": status, "detail": detail})

    def lines(self):
        return [f"CHECK {check['name']:<44} {check['status']:<5} {check['detail']}"
                for check in self.checks_]

    def failures(self):
        return [check for check in self.checks_ if check["status"] == "FAIL"]

    def counts(self):
        counts = {}
        for check in self.checks_:
            counts[check["status"]] = counts.get(check["status"], 0) + 1
        return counts

    def asList(self):
        return list(self.checks_)


def ok(result):
    return result is not None and result["status"] == "ok"


def describe(result):
    if result is None:
        return "not run"
    if ok(result):
        return "ok"
    step = f" after step {result['steps']}" if result.get("steps") is not None else ""
    return f"{result['status']}{step} ({result.get('message', '')})"


def checkSod(validation, results, key, label):
    monaghan, variational = results.get(key + ("monaghan",)), results.get(key + ("variational",))
    if not ok(monaghan) or not ok(variational):
        validation.add(label, "FAIL", f"monaghan {describe(monaghan)}; "
                                      f"variational {describe(variational)}")
        return
    ratio = variational["l1"] / monaghan["l1"]
    passed = ratio < 1.3 and variational["l1"] < 0.25
    validation.add(label, "PASS" if passed else "FAIL",
                   f"L1 var={variational['l1']:.4e} mon={monaghan['l1']:.4e} "
                   f"ratio={ratio:.3f} (<1.3, var<0.25); dE_final var="
                   f"{variational['finalEnergyError']:.3e} mon={monaghan['finalEnergyError']:.3e}")


def checkEnergy(validation, results, key, label, limit=1.e-2):
    """Require the variational run to complete with a bounded energy error.

    When the Monaghan baseline itself aborts, a variational abort is KNOWN; when the
    baseline exceeds the absolute limit, the limit becomes 1.5 times the baseline.
    """
    monaghan, variational = results.get(key + ("monaghan",)), results.get(key + ("variational",))
    if not ok(monaghan):
        if ok(variational):
            passed = variational["maxEnergyError"] < limit
            validation.add(label, "PASS" if passed else "FAIL",
                           f"monaghan baseline {describe(monaghan)}; variational completed, "
                           f"max|dE|={variational['maxEnergyError']:.3e} (<{limit:g}), "
                           f"rho_max={variational['maxDensity']:.4g}")
        else:
            validation.add(label, "KNOWN",
                           f"both operators fail: monaghan {describe(monaghan)}; "
                           f"variational {describe(variational)}")
        return
    if not ok(variational):
        validation.add(label, "FAIL", f"monaghan completed (max|dE|="
                                      f"{monaghan['maxEnergyError']:.3e}); variational "
                                      f"{describe(variational)}")
        return
    threshold = limit if monaghan["maxEnergyError"] <= limit else 1.5 * monaghan["maxEnergyError"]
    passed = variational["maxEnergyError"] < threshold
    validation.add(label, "PASS" if passed else "FAIL",
                   f"max|dE| var={variational['maxEnergyError']:.3e} "
                   f"mon={monaghan['maxEnergyError']:.3e} (<{threshold:.3g}); rho_max "
                   f"var={variational['maxDensity']:.4g} mon={monaghan['maxDensity']:.4g}")


def gpuBusyWithOurJobs():
    """Return a reason to skip CUDA cases, or None when the GPU can be used."""
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                 "--format=csv,noheader"], text=True, capture_output=True,
                                timeout=30, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        return f"nvidia-smi unavailable: {error}"
    ours = [line.strip() for line in result.stdout.splitlines() if "cosmictiger" in line]
    return f"GPU busy with {ours}" if ours else None


def readDefaultCfl(stem):
    text = (kConfigurations / (stem + ".ini")).read_text()
    match = re.search(r"^\s*cfl\s*=\s*([0-9.eE+\-]+)", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"configurations/{stem}.ini has no hydro cfl")
    return float(match[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--cuda", action="store_true",
                        help="repeat Sod and hydroWave on the CUDA backend and compare with CPU")
    parser.add_argument("--cuda-executable", type=Path,
                        help="binary for the CUDA runs; defaults to the main executable")
    parser.add_argument("--mpi-exec", help="run the two-locality comparisons with this launcher")
    parser.add_argument("--quick", action="store_true",
                        help="8^3 runs with a single integrator, suitable for CTest")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2,
                        help="concurrent single-locality CPU processes (2 threads each)")
    args = parser.parse_args()
    executable = args.executable.resolve()
    cudaExecutable = (args.cuda_executable or args.executable).resolve()
    base = args.workdir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    available = compiled_algorithms(str(executable))["integrator"]
    integrators = [value for value in kHydroIntegrators if value in available]
    if not integrators:
        raise RuntimeError("the executable has no hydro-capable integrator")
    matrixIntegrators = integrators[:1] if args.quick else integrators
    sodSize = 8 if args.quick else 16
    blastSizes = (8,) if args.quick else (8, 16)
    waveSizes = (8,) if args.quick else (8, 16)
    defaultCfl = readDefaultCfl("sod")

    cpuSpecs = []
    keys = {}

    def addCpu(name, stem, key, **options):
        cpuSpecs.append(RunSpec(name, stem, executable, **options))
        keys[name] = key

    for integrator in matrixIntegrators:
        for operator in kOperators:
            addCpu(f"sod_n{sodSize}_{integrator}_{operator}", "sod",
                   ("sod", "cpu", sodSize, integrator, operator),
                   operator=operator, integrator=integrator, resolution=sodSize)
            for size in blastSizes:
                addCpu(f"blast_n{size}_{integrator}_{operator}", "blast",
                       ("blast", "cpu", size, integrator, operator),
                       operator=operator, integrator=integrator, resolution=size)
            for size in waveSizes:
                addCpu(f"hydroWave_n{size}_{integrator}_{operator}", "hydro_wave",
                       ("hydroWave", "cpu", size, integrator, operator),
                       operator=operator, integrator=integrator, resolution=size)
    # One coarse interval lets the Courant limit, not the coarse cadence, set every
    # global step, so halving hydro.cfl halves the actual timestep.
    refinementStems = [("sod", "sod", sodSize)]
    if not args.quick:
        refinementStems.append(("hydroWave", "hydro_wave", 16))
    if "global" in integrators:
        for label, stem, size in refinementStems:
            for operator in (kOperators if not args.quick else ("variational",)):
                for factor, tag in ((1.0, "cfl1"), (0.5, "cfl0.5")):
                    addCpu(f"refine_{label}_n{size}_{operator}_{tag}", stem,
                           ("refine", label, size, operator, tag), operator=operator,
                           integrator="global", resolution=size,
                           overrides={"run.numberOfSteps": "1",
                                      "hydro.cfl": repr(defaultCfl * factor)})
    cosmologyCases = [("variational", "5/3")] if args.quick else [
        (operator, gamma) for gamma in ("5/3", "7/5") for operator in kOperators]
    gammas = {"5/3": repr(5.0 / 3.0), "7/5": repr(7.0 / 5.0)}
    for operator, gamma in cosmologyCases:
        # Many short coarse intervals put a diagnostic row on every iteration.
        addCpu(f"cosmology_n8_gamma{gamma.replace('/', '_')}_{operator}", "cosmology",
               ("cosmology", gamma, operator), operator=operator, integrator=integrators[0],
               resolution=8, kind="cosmology", timeout=kCosmologyTimeout,
               overrides={"chemistry.enabled": "false", "starFormation.enabled": "false",
                          "hydro.gamma": gammas[gamma], "run.numberOfSteps": "2048",
                          "run.maxIterations": "5"})
    addCpu("reject_chemistry_variational", "cosmology", ("reject",), operator="variational",
           integrator=integrators[0], resolution=8, kind="reject", expectSuccess=False,
           overrides={"chemistry.enabled": "true", "run.maxIterations": "0"})

    mpiSpecs = []
    waveIntegrator = "traditional" if "traditional" in integrators else integrators[0]
    if args.mpi_exec:
        mpiSpecs.append(RunSpec(f"mpi_hydroWave_n8_{waveIntegrator}_variational", "hydro_wave",
                                executable, localities=2, operator="variational",
                                integrator=waveIntegrator, resolution=8))
        keys[mpiSpecs[-1].name] = ("hydroWave", "mpi", 8, waveIntegrator, "variational")
        # Mirrors the sph_migration_mpi CTest command, with the variational operator.
        migration = [str(executable), "--run.test=sphGravity",
                     "--initial.particlesPerDimension=8", "--hydro.enabled=true",
                     "--compute.cuda=false", "--hpx:threads=2", "--hpx:bind=none"]
        migration += application_arguments(executable, far="fmm", hydro=True)
        migration += ["--hydro.pressureOperator=variational"]
        mpiSpecs.append(RunSpec("mpi_sph_migration_variational", None, executable,
                                localities=2, operator="variational",
                                integrator=application_options(
                                    executable, far="fmm", hydro=True)["run.timeIntegrator"],
                                resolution=8, kind="migration", rawCommand=migration))
        keys[mpiSpecs[-1].name] = ("migration",)

    cudaSpecs = []
    cudaSkip = None
    if args.cuda:
        cudaSkip = gpuBusyWithOurJobs()
        cudaIntegrators = [value for value in matrixIntegrators
                           if value in compiled_algorithms(str(cudaExecutable))["integrator"]]
        if cudaSkip is None:
            for operator in kOperators:
                if "traditional" in cudaIntegrators:
                    name = f"cuda_sod_n{sodSize}_traditional_{operator}"
                    cudaSpecs.append(RunSpec(name, "sod", cudaExecutable, backend="cuda",
                                             operator=operator, integrator="traditional",
                                             resolution=sodSize))
                    keys[name] = ("sod", "cuda", sodSize, "traditional", operator)
                for integrator in cudaIntegrators:
                    for size in waveSizes:
                        name = f"cuda_hydroWave_n{size}_{integrator}_{operator}"
                        cudaSpecs.append(RunSpec(name, "hydro_wave", cudaExecutable,
                                                 backend="cuda", operator=operator,
                                                 integrator=integrator, resolution=size))
                        keys[name] = ("hydroWave", "cuda", size, integrator, operator)

    print(f"variational SPH validation: {len(cpuSpecs)} CPU, {len(mpiSpecs)} MPI, "
          f"{len(cudaSpecs)} CUDA runs; integrators {matrixIntegrators}; default cfl "
          f"{defaultCfl:.6g}; workdir {base}", flush=True)
    resultsByName = {}

    def record(result):
        resultsByName[result["name"]] = result
        print(runLine(result), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(execute, spec, base, args.mpi_exec) for spec in cpuSpecs]
        try:
            for future in concurrent.futures.as_completed(futures):
                record(future.result())
        except BaseException:
            # Stop queued work and the running process groups before the pool joins.
            for future in futures:
                future.cancel()
            kRegistry.killAll()
            raise
    # Two-locality and GPU runs each use the whole four-core budget; run them alone.
    for spec in mpiSpecs + cudaSpecs:
        if spec.backend == "cuda":
            reason = gpuBusyWithOurJobs()
            if reason is not None:
                cudaSkip = reason
                break
        record(execute(spec, base, args.mpi_exec))

    results = {keys[name]: result for name, result in resultsByName.items()}
    validation = Validation()
    for integrator in matrixIntegrators:
        checkSod(validation, results, ("sod", "cpu", sodSize, integrator),
                 f"sod n{sodSize} {integrator} cpu L1")
        for operator in kOperators:
            result = results.get(("sod", "cpu", sodSize, integrator, operator))
            if ok(result) and abs(result["maxEnergyError"]) >= 1.e-2:
                validation.add(f"sod n{sodSize} {integrator} {operator} energy", "FAIL",
                               f"max|dE|={result['maxEnergyError']:.3e} (>=1e-2)")
        for size in blastSizes:
            checkEnergy(validation, results, ("blast", "cpu", size, integrator),
                        f"blast n{size} {integrator} cpu energy")
        for size in waveSizes:
            checkEnergy(validation, results, ("hydroWave", "cpu", size, integrator),
                        f"hydroWave n{size} {integrator} cpu energy")

    if "global" in integrators:
        for label, _, size in refinementStems:
            for operator in (kOperators if not args.quick else ("variational",)):
                coarse = results.get(("refine", label, size, operator, "cfl1"))
                fine = results.get(("refine", label, size, operator, "cfl0.5"))
                name = f"cfl refinement {label} n{size} global {operator}"
                if not ok(coarse) or not ok(fine):
                    validation.add(name, "FAIL", f"cfl {describe(coarse)}; "
                                                 f"cfl/2 {describe(fine)}")
                    continue
                ratio = (abs(fine["finalEnergyError"]) / abs(coarse["finalEnergyError"])
                         if coarse["finalEnergyError"] != 0.0 else math.inf)
                spatial = (f"; L1 cfl={coarse['l1']:.4e} cfl/2={fine['l1']:.4e}"
                           if "l1" in coarse else "")
                # The requirement applies to the variational Sod case; the others inform.
                required = operator == "variational" and label == "sod"
                status = ("PASS" if ratio <= 1.5 else "FAIL") if required else "INFO"
                validation.add(name, status,
                               f"steps {coarse['steps']}->{fine['steps']}; dE_final "
                               f"{coarse['finalEnergyError']:.3e}->{fine['finalEnergyError']:.3e} "
                               f"(ratio {ratio:.3f}{', <=1.5' if required else ''}); max|dE| "
                               f"{coarse['maxEnergyError']:.3e}->{fine['maxEnergyError']:.3e}"
                               + spatial)
    else:
        validation.add("cfl refinement", "SKIP", "global integrator not compiled")

    for operator, gamma in cosmologyCases:
        result = results.get(("cosmology", gamma, operator))
        name = f"cosmology n8 gamma={gamma} {operator}"
        if ok(result) and result["energyRows"] >= 2:
            validation.add(name, "PASS", f"{result['steps']} steps, {result['energyRows']} "
                                         f"energy rows, a={result['finalScale']:.5g}, "
                                         f"max|dE|={result['maxEnergyError']:.3e}")
        else:
            detail = describe(result) if not ok(result) else (
                f"only {result['energyRows']} energy rows")
            validation.add(name, "FAIL", detail)

    rejection = results.get(("reject",))
    validation.add("reject variational with chemistry",
                   "PASS" if rejection and rejection["status"] == "rejected" else "FAIL",
                   describe(rejection) if rejection and rejection["status"] != "rejected"
                   else f"exit {rejection['returnCode']}: {rejection['message']}")

    if args.mpi_exec:
        single = results.get(("hydroWave", "cpu", 8, waveIntegrator, "variational"))
        double = results.get(("hydroWave", "mpi", 8, waveIntegrator, "variational"))
        name = f"mpi hydroWave n8 {waveIntegrator} variational"
        if not ok(single) or not ok(double):
            validation.add(name, "FAIL", f"1 locality {describe(single)}; "
                                         f"2 localities {describe(double)}")
        else:
            (kinetic, thermal), rowMaximum, sameRows = compareEnergies(single, double)
            passed = sameRows and kinetic < 1.e-4 and thermal < 1.e-4
            validation.add(name, "PASS" if passed else "FAIL",
                           f"final rel diff K={kinetic:.2e} U={thermal:.2e} (<1e-4); "
                           f"all-row max |diff|/(K+U)={rowMaximum:.2e}; rows "
                           f"{single['energyRows']}/{double['energyRows']}; max|dE| 1loc="
                           f"{single['maxEnergyError']:.3e} 2loc={double['maxEnergyError']:.3e}")
        migration = results.get(("migration",))
        validation.add("mpi sph migration variational", "PASS" if ok(migration) else "FAIL",
                       migration["message"] if ok(migration) else describe(migration))
    else:
        validation.add("mpi", "SKIP", "no --mpi-exec")

    if args.cuda:
        if cudaSkip is not None and not any(spec.name in resultsByName for spec in cudaSpecs):
            validation.add("cuda", "SKIP", cudaSkip)
        for spec in cudaSpecs:
            stem, _, size, integrator, operator = keys[spec.name]
            gpu = results.get((stem, "cuda", size, integrator, operator))
            cpu = results.get((stem, "cpu", size, integrator, operator))
            name = f"cuda {stem} n{size} {integrator} {operator}"
            if gpu is None:
                validation.add(name, "SKIP", cudaSkip or "not run")
                continue
            if not ok(gpu) or not ok(cpu):
                validation.add(name, "FAIL", f"cuda {describe(gpu)}; cpu {describe(cpu)}")
                continue
            (kinetic, thermal), rowMaximum, sameRows = compareEnergies(cpu, gpu)
            passed = sameRows and kinetic < 1.e-3 and thermal < 1.e-3
            l1 = (f"; L1 cuda={gpu['l1']:.4e} cpu={cpu['l1']:.4e}" if "l1" in gpu else "")
            validation.add(name, "PASS" if passed else "FAIL",
                           f"final rel diff K={kinetic:.2e} U={thermal:.2e} (<1e-3); "
                           f"all-row max |diff|/(K+U)={rowMaximum:.2e}; max|dE| cuda="
                           f"{gpu['maxEnergyError']:.3e} cpu={cpu['maxEnergyError']:.3e}; "
                           f"wall cuda={gpu['wall']:.1f}s cpu={cpu['wall']:.1f}s" + l1)
        for operator in kOperators:
            key = ("sod", "cuda", sodSize, "traditional", operator)
            if key in results and ok(results[key]):
                result = results[key]
                if result["l1"] >= 0.25 or result["maxEnergyError"] >= 1.e-2:
                    validation.add(f"cuda sod n{sodSize} traditional {operator} accuracy",
                                   "FAIL", f"L1={result['l1']:.4e} "
                                           f"max|dE|={result['maxEnergyError']:.3e}")
        checkSod(validation, results, ("sod", "cuda", sodSize, "traditional"),
                 f"sod n{sodSize} traditional cuda L1")
    else:
        validation.add("cuda", "SKIP", "no --cuda")

    for line in validation.lines():
        print(line)
    counts = validation.counts()
    elapsed = time.monotonic() - started
    summary = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
    print(f"SUMMARY: {summary}; {len(resultsByName)} runs in {elapsed:.0f} s wall")
    (base / "results.json").write_text(json.dumps(
        {"runs": [dict(result, kinetic=None, thermal=None) | {
            "finalKinetic": result["kinetic"][-1] if result.get("kinetic") else None,
            "finalThermal": result["thermal"][-1] if result.get("thermal") else None}
            for result in resultsByName.values()],
         "checks": validation.asList(), "seconds": elapsed}, indent=1, default=str))
    return 1 if validation.failures() else 0


def terminate(signalNumber, frame):
    # Application processes run in their own sessions; take them down with the driver.
    kRegistry.killAll()
    sys.exit(128 + signalNumber)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        kRegistry.killAll()
        print(error, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        kRegistry.killAll()
        sys.exit(130)
