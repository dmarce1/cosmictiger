# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# SPDX-License-Identifier: BSL-1.0

"""Check PM evolution, the common timestep, single drift, and restart compatibility."""

import argparse
import ctypes
import math
from pathlib import Path
import re
import struct
import subprocess
import sys

sys.dont_write_bytecode = True

from run_checkpoint import fnv1a64, prepareCase, runSimulation


class Header(ctypes.Structure):
    _fields_ = [("magic", ctypes.c_uint64), ("version", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("physics", ctypes.c_double * 11),
                ("particles_per_species", ctypes.c_uint64), ("steps", ctypes.c_int32),
                ("reserved", ctypes.c_uint32)]


class Driver(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in
                ("a", "adot", "dummy", "tau", "tau_max", "energy0")]
    _fields_ += [("energies", ctypes.c_double * 10)]
    _fields_ += [(name, ctypes.c_int) for name in
                 ("max_rung", "iteration", "min_rung", "step", "bucket_size")]
    _fields_ += [("processed", ctypes.c_size_t)]
    _fields_ += [(name, ctypes.c_double) for name in ("flops", "runtime", "years")]
    _fields_ += [("integer_time", ctypes.c_uint64)]


class Domain(ctypes.Structure):
    _fields_ = [("box", ctypes.c_double * 6), ("midpoint", ctypes.c_double),
                ("rank", ctypes.c_int)]


class DomainEntry(ctypes.Structure):
    _fields_ = [("key", ctypes.c_size_t), ("domain", Domain)]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_rank(path, expected_iteration, ranks):
    data = path.read_bytes()
    header = Header.from_buffer_copy(data)
    require(header.magic == 0x4354535048303031 and header.version == 3,
            f"{path}: unsupported checkpoint header")
    require(header.flags & (1 << 16), f"{path}: PM flag is missing")
    require(not header.flags & ((1 << 7) | (1 << 8) | (1 << 9) | (1 << 11) | (1 << 12)),
            f"{path}: fixture unexpectedly enabled optional checkpoint fields")
    hydro = bool(header.flags & 1)
    position_bytes = 8 if header.flags & (1 << 15) else 4
    offset = ctypes.sizeof(Header)
    extension = struct.unpack_from("=6Q", data, offset)
    require(extension[0] == 1 and extension[-1] == 1, f"{path}: unsupported PM convention")
    offset += 48 + (104 if hydro else 0)
    driver = Driver.from_buffer_copy(data, offset)
    require(driver.iteration == expected_iteration, f"{path}: driver layout/iteration mismatch")
    require(all(math.isfinite(value) for value in
                [driver.a, driver.adot, driver.tau, driver.tau_max, *driver.energies]),
            f"{path}: nonfinite driver state")
    offset += ctypes.sizeof(Driver)

    def parse_particles(integer_bytes):
        cursor = offset

        def take(code, count):
            nonlocal cursor
            values = struct.unpack_from(f"={count}{code}", data, cursor)
            cursor += count * struct.calcsize("=" + code)
            return values

        integer_code = "i" if integer_bytes == 4 else "q"
        count = take(integer_code, 1)[0]
        gas_count = take(integer_code, 1)[0] if hydro else 0
        require(0 <= gas_count <= count <= header.particles_per_species * (2 if hydro else 1),
                f"{path}: invalid particle counts")
        raw_positions = [take("Q" if position_bytes == 8 else "I", count) for _ in range(3)]
        velocities = take("f", 3 * count)
        rungs = take("b", count)
        types = take("b", count) if hydro else (0,) * count
        gas_indices = take(integer_code, count) if hydro else ()
        rows = [{"x": tuple(raw_positions[dim][i] / 2 ** (8 * position_bytes)
                            for dim in range(3)), "v": velocities[3 * i:3 * i + 3],
                 "rung": rungs[i], "type": types[i], "gas": ()} for i in range(count)]
        if hydro:
            gas_header = take("I", 7)
            require(gas_header == (0x43545350, 2, integer_bytes, 8, 12, 3, 4),
                    f"{path}: unsupported gas layout {gas_header}")
            require(take(integer_code, 1)[0] == gas_count, f"{path}: inconsistent gas count")
            thermodynamic = take("f", 2 * gas_count)
            acceleration = take("f", 3 * gas_count)
            gravity_indices = take(integer_code, gas_count)
            scalar_fields = [take("f", gas_count) for _ in range(3)]
            flags = [take("B", gas_count) for _ in range(2)]
            vector_fields = [take("f", gas_count) for _ in range(6)]
            for gas_index, particle in enumerate(gravity_indices):
                require(0 <= particle < count and types[particle] == 1 and
                        gas_indices[particle] == gas_index, f"{path}: broken gas mapping")
                rows[particle]["gas"] = (
                    *thermodynamic[2 * gas_index:2 * gas_index + 2],
                    *acceleration[3 * gas_index:3 * gas_index + 3],
                    *(field[gas_index] for field in scalar_fields),
                    *(field[gas_index] for field in flags),
                    *(field[gas_index] for field in vector_fields))
        rung_count, rung_begin = take("i", 2)
        require(0 <= rung_count <= 64 and 0 <= rung_begin <= count,
                f"{path}: invalid rung stack")
        take("i", rung_count)
        # domainsSave writes native domain records after the particle payload.
        # Their particle-index members follow the executable's integer width.
        class LocalDomain(ctypes.Structure):
            _fields_ = [("box", ctypes.c_double * 6), ("process_range", ctypes.c_int * 2),
                        ("particle_range", (ctypes.c_int32 if integer_bytes == 4
                                            else ctypes.c_int64) * 2),
                        ("depth", ctypes.c_int)]

        domain_count = ctypes.c_size_t.from_buffer_copy(data, cursor).value
        cursor += ctypes.sizeof(ctypes.c_size_t)
        cursor += domain_count * ctypes.sizeof(DomainEntry) + ranks * ctypes.sizeof(LocalDomain)
        require(cursor == len(data), f"{path}: trailing/unparsed checkpoint data")
        require(all(math.isfinite(value) for row in rows for value in (*row["v"], *row["gas"])),
                f"{path}: nonfinite particle state")
        return rows

    errors = []
    for integer_bytes in (4, 8):
        try:
            rows = parse_particles(integer_bytes)
            break
        except (RuntimeError, struct.error, OverflowError, ValueError) as error:
            errors.append(str(error))
    else:
        raise RuntimeError(f"{path}: unable to decode particle layout: {errors}")
    return header, driver, rows


def read_checkpoint(path, ranks):
    iteration = int(path.name.split(".")[1])
    lines = (path / "checkpoint.meta").read_text().splitlines()
    require(lines[0].split() == ["COSMICTIGER_CHECKPOINT", "2", "3", str(iteration), str(ranks)],
            f"{path}: unexpected manifest")
    rows = []
    state = None
    for rank in range(ranks):
        rank_path = path / f"checkpoint.{rank}.dat"
        data = rank_path.read_bytes()
        record = lines[rank + 1].split()
        require(record == [str(rank), str(len(data)), f"{fnv1a64(data):016x}"],
                f"{path}: rank {rank} size/checksum mismatch")
        header, driver, particles = read_rank(rank_path, iteration, ranks)
        if state is None:
            state = driver
        else:
            require((state.tau, state.integer_time, state.max_rung) ==
                    (driver.tau, driver.integer_time, driver.max_rung),
                    f"{path}: locality clocks differ")
        rows.extend(particles)
    require(rows and {row["rung"] for row in rows} == {state.max_rung},
            f"{path}: particle rungs are not globally uniform")
    return header, state, rows


def displacement(first, second):
    return tuple((b - a + 0.5) % 1.0 - 0.5 for a, b in zip(first, second))


def match_rows(before, after):
    require(len(before) == len(after), "particle count changed")
    unused = set(range(len(before)))
    for row in after:
        candidates = [index for index in unused if before[index]["type"] == row["type"]]
        require(candidates, "particle species count changed")
        index = min(candidates, key=lambda i: sum(value * value for value in
                    displacement(before[i]["x"], row["x"])))
        unused.remove(index)
        yield before[index], row


def compare_restart(reference, resumed):
    _, reference_state, reference_rows = reference
    _, resumed_state, resumed_rows = resumed
    require((reference_state.tau, reference_state.integer_time, reference_state.max_rung) ==
            (resumed_state.tau, resumed_state.integer_time, resumed_state.max_rung),
            "restart changed the common clock")
    for old, new in match_rows(reference_rows, resumed_rows):
        require(max(map(abs, displacement(old["x"], new["x"]))) <= 2.0e-8,
                "restart changed particle positions")
        require(old["rung"] == new["rung"] and len(old["gas"]) == len(new["gas"]),
                "restart changed particle/rung layout")
        for a, b in zip((*old["v"], *old["gas"]), (*new["v"], *new["gas"])):
            require(math.isclose(a, b, rel_tol=2.0e-5, abs_tol=1.0e-8),
                    f"restart changed velocity or SPH physical/predictor state: {a} != {b}")


def check_clock(checkpoints):
    for (header, old, _), (_, new, _) in zip(checkpoints, checkpoints[1:]):
        expected = math.ldexp(old.tau_max / header.steps, -new.max_rung)
        require(math.isclose(new.tau - old.tau, expected, rel_tol=1.0e-10, abs_tol=1.0e-20),
                "particle mesh clock did not advance by its common dyadic step")
        require(new.max_rung >= old.max_rung - 1, "timestep grew by more than a factor of two")
        require(old.integer_time % (1 << (64 - new.max_rung)) == 0,
                "timestep coarsened away from an integer-clock synchronization boundary")


def check_single_drift(before, after):
    header, old_state, old_rows = before
    _, state, rows = after
    require(old_state.a == state.a == 1.0, "single-drift fixture must have a static background")
    timestep = state.tau - old_state.tau
    quantum = 2.0 ** (-64 if header.flags & (1 << 15) else -32)
    signal = error = 0.0
    for old, new in match_rows(old_rows, rows):
        moved = displacement(old["x"], new["x"])
        for actual, velocity in zip(moved, new["v"]):
            expected = timestep * velocity
            signal += expected * expected
            error += (actual - expected) ** 2
            require(abs(actual - expected) <= 3.0 * quantum + 2.0e-6 * abs(expected) + 1.0e-15,
                    "particle displacement does not equal exactly one drift with its stored velocity")
    require(signal > len(rows) * quantum * quantum * 100,
            "single-drift fixture has insufficient nonzero velocity")
    require(error < 0.01 * signal, "drift residual is too large")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--cpu", action="store_true")
    backend.add_argument("--cuda", action="store_true")
    parser.add_argument("--mpi-exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--position-type", choices=("fixed32", "fixed64"), default="fixed32")
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ranks = args.ranks if args.mpi_exec else 1
    common = {"hpx:threads": 2, "hpx:bind": "none", "gravity.method": "pm",
              "gravity.meshSize": 8, "gravity.cloud": "pcs", "gravity.enabled": "true",
              "compute.cuda": str(args.cuda).lower(), "compute.positionType": args.position_type,
              "initial.particlesPerDimension": 8, "initial.closePack": "false",
              "initial.useGlass": "false", "chemistry.enabled": "false",
              "run.endTime": 0.00001, "run.numberOfSteps": 4,
              "analysis.groups": "false", "analysis.tracers": "false",
              "analysis.views": "false", "analysis.power": "false", "lightcone.enabled": "false",
              "output.silo": "false", "output.directory": ".", "checkpoint.intervalSeconds": 0}
    if ranks > 1:
        common["hpx:localities"] = ranks

    def run(directory, options, success=True):
        command = [str(executable)] + [f"--{key}={value}" for key, value in options.items()]
        if args.mpi_exec:
            command = ([args.mpi_exec, args.mpi_numproc_flag, str(ranks)]
                       + args.mpi_preflag + command[:1] + args.mpi_postflag + command[1:])
        if ranks > 1:
            command += ["--hpx:ini=hpx.parcel.mpi.enable=1",
                        "--hpx:ini=hpx.parcel.tcp.enable=0"]
        return runSimulation(command, directory, "run.log", expectSuccess=success)

    for hydro in (False, True):
        label = "mixed" if hydro else "dark_matter"
        options = dict(common, **{"hydro.enabled": str(hydro).lower(),
                                  "run.scenario": "cosmology" if hydro else "plummer"})
        baseline = prepareCase(root, label + "_baseline")
        run(baseline, dict(options, **{"run.maxIterations": 3}))
        paths = sorted((path for path in baseline.glob("checkpoint.*") if path.is_dir() and
                        re.fullmatch(r"checkpoint\.[0-9]+", path.name)),
                       key=lambda path: int(path.name.split(".")[1]))
        require([int(path.name.split(".")[1]) for path in paths] == [1, 2, 3, 4],
                f"{label}: expected one checkpoint per step")
        states = [read_checkpoint(path, ranks) for path in paths]
        check_clock(states)
        if not hydro:
            check_single_drift(states[0], states[1])
        else:
            require({row["type"] for row in states[-1][2]} == {0, 1},
                    "mixed-species fixture did not include dark matter and gas")
        source = baseline / "checkpoint.2"
        resumed = prepareCase(root, label + "_resumed", source)
        log = run(resumed, dict(options, **{"checkpoint.readIndex": 2,
                                          "run.maxIterations": 1}))
        require("Done reading checkpoint" in log, "PM restart did not load its checkpoint")
        compare_restart(states[-1], read_checkpoint(resumed / "checkpoint.4", ranks))
        for name, change in (("mesh", {"gravity.meshSize": 10}),
                             ("cloud", {"gravity.cloud": "cic"}),
                             ("backend", {"gravity.method": "fmm"})):
            rejected = prepareCase(root, label + "_reject_" + name, source)
            log = run(rejected, dict(options, **change,
                                    **{"checkpoint.readIndex": 2, "run.maxIterations": 0}), False)
            require("checkpoint" in log.lower(), f"{name} mismatch lacked a checkpoint diagnostic")
        print(f"{label}: globally uniform dyadic steps, restart state, and mesh/cloud/backend "
              "mismatch rejection passed")
    print(f"Particle mesh evolution passed ({ranks} rank(s), "
          f"{'CUDA SPH/drift' if args.cuda else 'CPU SPH/drift'}, {args.position_type})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, struct.error, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
