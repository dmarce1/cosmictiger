# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# SPDX-License-Identifier: BSL-1.0

"""Decode PMMM checkpoints and verify clocks, drift, and restart state for either integrator."""

import ctypes
import math
import struct
import sys
from pathlib import Path

sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_checkpoint import fnv1a64


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


def read_rank(path, expected_iteration, ranks, time_integrator):
    data = path.read_bytes()
    header = Header.from_buffer_copy(data)
    require(header.magic == 0x4354535048303031 and header.version == 3,
            f"{path}: unsupported checkpoint header")
    require(header.flags & (1 << 17), f"{path}: PMMM flag is missing")
    require(bool(header.flags & (1 << 6)) == (time_integrator == "hierarchical"),
            f"{path}: checkpoint time integrator differs from the compiled selection")
    require(not header.flags & (1 << 16), f"{path}: removed PM solver flag is set")
    require(not header.flags & ((1 << 7) | (1 << 8) | (1 << 9) | (1 << 11) | (1 << 12)),
            f"{path}: fixture unexpectedly enabled optional checkpoint fields")
    hydro = bool(header.flags & 1)
    position_bytes = 8 if header.flags & (1 << 15) else 4
    offset = ctypes.sizeof(Header)
    require(bool(header.flags & (1 << 19)) == (time_integrator == "global"),
            f"{path}: global integrator flag differs from the selected mode")
    if header.flags & (1 << 18):
        algorithms = struct.unpack_from("=5Q", data, offset)
        require(algorithms[0] == 1 and algorithms[1] == 2,
                f"{path}: invalid algorithm selection extension")
        require(algorithms[4] == {"traditional": 0, "hierarchical": 1, "global": 2}[time_integrator],
                f"{path}: algorithm selection integrator differs")
        offset += 40
    extension_version = struct.unpack_from("=Q", data, offset)[0]
    require(extension_version in (1, 2, 3, 4), f"{path}: unsupported PMMM version")
    extension_size = {1: 7, 2: 8, 3: 10, 4: 11}[extension_version]
    extension = struct.unpack_from(f"={extension_size}Q", data, offset)
    require(extension[6] == 1, f"{path}: unsupported PMMM convention")
    if extension_version >= 2:
        require(extension[7] in (0, 1), f"{path}: unknown PMMM near solver")
    if extension_version >= 3:
        theta = struct.unpack("=d", struct.pack("=Q", extension[9]))[0]
        require(3 <= extension[8] <= 9 and extension[2] == extension[8],
                f"{path}: PMMM and near FMM orders differ or are invalid")
        width = extension[3]
        require(1 <= width <= 4 and extension[1] >= 2 * width + 1,
                f"{path}: invalid PMMM near width or mesh size")
        require(math.isfinite(theta) and math.isclose(
                    theta, math.sqrt(3.0) / (width + 1), rel_tol=1.0e-15, abs_tol=0.0),
                f"{path}: PMMM opening angle differs from the near-width geometry")
    if extension_version >= 4:
        require(extension[10] == 0 or 2 <= extension[10] <= 2 ** 31 - 1,
                f"{path}: invalid PMMM maximum FFT prime factor")
    offset += 8 * extension_size + (104 if hydro else 0)
    gas_version = 2
    if hydro and header.flags & (1 << 14):
        viscosity = struct.unpack_from("=7d", data, offset)
        require(all(math.isfinite(value) for value in viscosity) and viscosity[4] in (0.0, 1.0),
                f"{path}: invalid variable-viscosity parameters")
        gas_version = 6 if viscosity[4] else 4
        offset += 56
    driver = Driver.from_buffer_copy(data, offset)
    require(driver.iteration == expected_iteration, f"{path}: driver layout/iteration mismatch")
    require(all(math.isfinite(value) for value in
                [driver.a, driver.adot, driver.tau, driver.tau_max, *driver.energies]),
            f"{path}: nonfinite driver state")
    offset += ctypes.sizeof(Driver)
    driver.global_timestep = 0.0
    if header.flags & (1 << 19):
        version, driver.global_timestep = struct.unpack_from("=Qd", data, offset)
        require(version == 1 and math.isfinite(driver.global_timestep) and driver.global_timestep >= 0,
                f"{path}: invalid explicit global timestep")
        offset += 16

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
            require(gas_header == (0x43545350, gas_version, integer_bytes, 8, 12, 3, 4),
                    f"{path}: unsupported gas layout {gas_header}")
            require(take(integer_code, 1)[0] == gas_count, f"{path}: inconsistent gas count")
            thermodynamic = take("f", 2 * gas_count)
            acceleration = take("f", 3 * gas_count)
            gravity_indices = take(integer_code, gas_count)
            scalar_fields = [take("f", gas_count) for _ in range(3)]
            flags = [take("B", gas_count) for _ in range(2)]
            vector_fields = [take("f", gas_count) for _ in range(6)]
            # Production writes two separate arrays: the common two-float
            # viscosity state, then Cullen-Dehnen's one-float extension.
            viscosity = take("f", 2 * gas_count) if gas_version >= 4 else ()
            cullen_dehnen = take("f", gas_count) if gas_version == 6 else ()
            require(sorted(gravity_indices) == [i for i, species in enumerate(types)
                                                if species == 1],
                    f"{path}: missing or duplicate gas mapping")
            for gas_index, particle in enumerate(gravity_indices):
                require(0 <= particle < count and types[particle] == 1 and
                        gas_indices[particle] == gas_index, f"{path}: broken gas mapping")
                rows[particle]["gas"] = (
                    *thermodynamic[2 * gas_index:2 * gas_index + 2],
                    *acceleration[3 * gas_index:3 * gas_index + 3],
                    *(field[gas_index] for field in scalar_fields),
                    *(field[gas_index] for field in flags),
                    *(field[gas_index] for field in vector_fields),
                    *viscosity[2 * gas_index:2 * gas_index + 2],
                    *cullen_dehnen[gas_index:gas_index + 1])
                require(thermodynamic[2 * gas_index] > 0.0 and
                        thermodynamic[2 * gas_index + 1] > 0.0 and
                        scalar_fields[2][gas_index] > 0.0,
                        f"{path}: nonpositive gas energy, support radius, or density")
        rung_count, rung_begin = take("i", 2)
        require(0 <= rung_count <= 64 and 0 <= rung_begin <= count,
                f"{path}: invalid rung stack")
        rung_begins = take("i", rung_count)
        require(all(0 <= begin <= count for begin in rung_begins) and
                list(rung_begins) == sorted(rung_begins) and
                (not rung_begins or rung_begins[-1] <= rung_begin),
                f"{path}: invalid nested rung selections")
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


def read_checkpoint(path, ranks, time_integrator):
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
        header, driver, particles = read_rank(rank_path, iteration, ranks, time_integrator)
        if state is None:
            state = driver
        else:
            require((state.tau, state.integer_time, state.max_rung) ==
                    (driver.tau, driver.integer_time, driver.max_rung),
                    f"{path}: locality clocks differ")
        rows.extend(particles)
    minimum_rung = state.min_rung if time_integrator == "hierarchical" else 0
    require(rows and all(minimum_rung <= row["rung"] <= state.max_rung < 64 for row in rows),
            f"{path}: particle rung lies outside the timestep range")
    if time_integrator == "traditional":
        require(all(row["rung"] == state.max_rung for row in rows),
                f"{path}: ordinary PMMM traditional did not synchronize every particle rung")
    if header.flags & 1:
        require(sum(row["type"] == 0 for row in rows) == header.particles_per_species and
                sum(row["type"] == 1 for row in rows) == header.particles_per_species,
                f"{path}: mixed evolution lost dark matter or gas particles")
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
    require(reference_state.global_timestep == resumed_state.global_timestep,
            "restart changed the pending global interval")
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
        global_step = bool(header.flags & (1 << 19))
        expected = new.global_timestep if global_step else math.ldexp(
            old.tau_max / header.steps, -new.max_rung)
        require(math.isclose(new.tau - old.tau, expected, rel_tol=1.0e-10, abs_tol=1.0e-20),
                "particle mesh clock did not advance by the selected interval")
        if not global_step:
            require(old.integer_time % (1 << (64 - new.max_rung)) == 0,
                    "timestep coarsened away from an integer-clock synchronization boundary")


def match_drift_rows(before, after, base_timestep, active_rung, global_timestep=None):
    """Match identities at the predicted start of each particle's scheduled drift.

    A crowded particle can cross a nearer neighbor during one rung step. Matching
    final positions alone confuses those particles. Backward drift prediction
    disambiguates them; the caller still checks the strict fixed-point residual.
    """
    require(len(before) == len(after), "particle count changed")
    unused = set(range(len(before)))
    for row in after:
        active = row["rung"] >= active_rung
        timestep = (global_timestep if global_timestep is not None else
                    math.ldexp(base_timestep, -row["rung"])) if active else 0.0
        predicted = tuple((position - timestep * velocity) % 1.0
                          for position, velocity in zip(row["x"], row["v"]))
        candidates = [index for index in unused
                      if before[index]["type"] == row["type"] and
                      (before[index]["rung"] >= active_rung) == active]
        require(candidates, "particle species or active-stage membership changed")
        index = min(candidates, key=lambda i: sum(value * value for value in
                    displacement(before[i]["x"], predicted)))
        unused.remove(index)
        yield before[index], row


def check_single_drift(before, after):
    header, old_state, old_rows = before
    _, state, rows = after
    require(old_state.a == state.a == 1.0, "single-drift fixture must have a static background")
    base_timestep = old_state.tau_max / header.steps
    clock_rung = (64 - (old_state.integer_time & -old_state.integer_time).bit_length() + 1
                  if old_state.integer_time else 0)
    # Hierarchical steps drift only the active rungs. Ordinary PMMM traditional
    # assigns every particle the common rung and drifts the entire population.
    hierarchical = bool(header.flags & (1 << 6))
    active_rung = max(clock_rung, state.min_rung) if hierarchical else 0
    quantum = 2.0 ** (-64 if header.flags & (1 << 15) else -32)
    signal = error = 0.0
    global_timestep = state.global_timestep if header.flags & (1 << 19) else None
    for old, new in match_drift_rows(old_rows, rows, base_timestep, active_rung, global_timestep):
        moved = displacement(old["x"], new["x"])
        active = old["rung"] >= active_rung
        if global_timestep is None:
            require(new["rung"] >= old["rung"] - 1,
                    "a particle timestep grew by more than a factor of two")
        if not active:
            require(old["rung"] == new["rung"] and old["v"] == new["v"] and
                    old["x"] == new["x"], "inactive slower particle changed during a fast stage")
        timestep = (global_timestep if global_timestep is not None else
                    math.ldexp(base_timestep, -new["rung"])) if active else 0.0
        for actual, velocity in zip(moved, new["v"]):
            expected = timestep * velocity
            signal += expected * expected
            error += (actual - expected) ** 2
            require(abs(actual - expected) <= 3.0 * quantum + 2.0e-6 * abs(expected) + 1.0e-15,
                    "particle displacement does not equal its one scheduled rung drift")
    require(signal > len(rows) * quantum * quantum * 100,
            "single-drift fixture has insufficient nonzero velocity")
    require(error < 0.01 * signal, "drift residual is too large")
