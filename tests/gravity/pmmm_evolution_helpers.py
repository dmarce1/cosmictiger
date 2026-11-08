# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# SPDX-License-Identifier: BSL-1.0

"""Decode PMMM checkpoints and verify clocks, drift, and restart state for all three integrators."""

import math
import struct
import sys
from pathlib import Path

sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checkpoint_format import Header, decode_checkpoint
from run_checkpoint import fnv1a64


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_rank(path, expected_iteration, ranks, time_integrator):
    decoded = decode_checkpoint(path.read_bytes(), ranks=ranks)
    header, driver, extension = decoded.header, decoded.driver, decoded.pmmm
    require(header.flags & (1 << 17), f"{path}: PMMM flag is missing")
    require(bool(header.flags & (1 << 6)) == (time_integrator == "hierarchical") and
            bool(header.flags & (1 << 19)) == (time_integrator == "global"),
            f"{path}: checkpoint time integrator differs from the selected mode")
    require(not header.flags & ((1 << 7) | (1 << 8) | (1 << 9) | (1 << 11) | (1 << 12)),
            f"{path}: fixture unexpectedly enabled optional checkpoint fields")
    if decoded.algorithms:
        require(decoded.algorithms[1] == 2 and decoded.algorithms[4] ==
                {"traditional": 0, "hierarchical": 1, "global": 2}[time_integrator],
                f"{path}: checkpoint algorithm selection differs")
    require(extension[6] == 1, f"{path}: unsupported PMMM convention")
    if extension[0] >= 2:
        require(extension[7] in (0, 1), f"{path}: unknown PMMM near solver")
    if extension[0] >= 3:
        theta = struct.unpack("=d", struct.pack("=Q", extension[9]))[0]
        require(3 <= extension[8] <= 9 and extension[2] == extension[8],
                f"{path}: PMMM and near FMM orders differ or are invalid")
        width = extension[3]
        require(1 <= width <= 4 and extension[1] >= 2 * width + 1,
                f"{path}: invalid PMMM near width or mesh size")
        require(math.isfinite(theta) and math.isclose(
                    theta, math.sqrt(3.0) / (width + 1), rel_tol=1.0e-15, abs_tol=0.0),
                f"{path}: PMMM opening angle differs from the near-width geometry")
    if extension[0] == 4:
        require(extension[10] == 0 or 2 <= extension[10] <= 2 ** 31 - 1,
                f"{path}: invalid retired PMMM maximum FFT prime factor")
    if extension[0] >= 5:
        # The coarse mesh is a power of two between the near minimum and the mesh.
        coarse = extension[10]
        require(coarse > 0 and coarse & (coarse - 1) == 0 and
                2 * extension[3] + 1 <= coarse <= extension[1],
                f"{path}: invalid PMMM coarse mesh size")
    if decoded.viscosity:
        require(all(decoded.viscosity[index] > 0.0 for index in (0, 1, 3, 4, 5, 6)),
                f"{path}: invalid variable-viscosity parameters")
    require(driver.iteration == expected_iteration, f"{path}: driver layout/iteration mismatch")
    require(all(math.isfinite(value) for value in
                [driver.a, driver.adot, driver.tau, driver.tau_max, *driver.energies]),
            f"{path}: nonfinite driver state")
    scale = 2 ** (8 * decoded.position_bytes)
    rows = []
    for index, kind in enumerate(decoded.types):
        gas = decoded.gas[decoded.gas_indices[index]] if kind == 1 else ()
        if gas:
            require(gas[0] > 0 and gas[1] > 0 and gas[6] > 0,
                    f"{path}: nonpositive gas energy, support radius, or density")
        rows.append({"x": tuple(decoded.positions[dim][index] / scale for dim in range(3)),
                     "v": decoded.velocities[3 * index:3 * index + 3],
                     "rung": decoded.rungs[index], "type": kind, "gas": gas})
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
    if time_integrator == "global":
        require(all(row["rung"] == 0 for row in rows),
                f"{path}: global PMMM did not use rung zero")
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


def match_drift_rows(before, after, base_timestep, active_rung, common_timestep=None):
    """Match identities at the predicted start of each particle's scheduled drift.

    A crowded particle can cross a nearer neighbor during one rung step. Matching
    final positions alone confuses those particles. Backward drift prediction
    disambiguates them; the caller still checks the strict fixed-point residual.
    """
    require(len(before) == len(after), "particle count changed")
    unused = set(range(len(before)))
    for row in after:
        active = row["rung"] >= active_rung
        timestep = (common_timestep if common_timestep is not None else
                    math.ldexp(base_timestep, -row["rung"]) if active else 0.0)
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
    # Hierarchical stages drift active particles by their own rung intervals.
    # Traditional steps kick only active particles but drift every particle by
    # the selected finest clock interval, preserving the individual kick rungs.
    hierarchical = bool(header.flags & (1 << 6))
    global_step = bool(header.flags & (1 << 19))
    active_rung = (max(clock_rung, state.min_rung) if hierarchical else
                   0 if global_step else clock_rung)
    quantum = 2.0 ** (-64 if header.flags & (1 << 15) else -32)
    signal = error = 0.0
    common_timestep = (state.global_timestep if global_step else
                       state.tau - old_state.tau if not hierarchical else None)
    for old, new in match_drift_rows(old_rows, rows, base_timestep, active_rung, common_timestep):
        moved = displacement(old["x"], new["x"])
        active = old["rung"] >= active_rung
        if not global_step:
            require(new["rung"] >= old["rung"] - 1,
                    "a particle timestep grew by more than a factor of two")
        if not active:
            require(old["rung"] == new["rung"] and old["v"] == new["v"],
                    "inactive slower particle received a kick or a new rung")
            if hierarchical:
                require(old["x"] == new["x"],
                        "inactive slower particle drifted during a hierarchical stage")
        timestep = (common_timestep if common_timestep is not None else
                    math.ldexp(base_timestep, -new["rung"]) if active else 0.0)
        for actual, velocity in zip(moved, new["v"]):
            expected = timestep * velocity
            signal += expected * expected
            error += (actual - expected) ** 2
            require(abs(actual - expected) <= 3.0 * quantum + 2.0e-6 * abs(expected) + 1.0e-15,
                    "particle displacement does not equal its scheduled drift")
    require(signal > len(rows) * quantum * quantum * 100,
            "single-drift fixture has insufficient nonzero velocity")
    require(error < 0.01 * signal, "drift residual is too large")
