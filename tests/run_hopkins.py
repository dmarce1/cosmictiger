# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# SPDX-License-Identifier: BSL-1.0

"""Exercise meshless evolution, conservation, restarts, and SPH default dispatch."""

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys

from checkpoint_format import read_prefix
from run_checkpoint import checkManifest, fnv1a64
from run_scenarios import check_bool, check_finite_rows, run_process, setting


METHOD_OFFSETS = {"sph": 0, "mfm": 2032, "mfv": 2064}
CASES = (("hydroWave", 0.04), ("sod", 0.02), ("gresho", 0.04), ("blast", 0.0001))


def particle_rank_state(path, method):
    """Decode one all-gas rank, excluding native driver timing and padding."""
    checkpoint = path.parent
    data = path.read_bytes()
    prefix = read_prefix(data)
    position_bytes = prefix.position_bytes
    position_bits = 8 * position_bytes
    integer_bytes = prefix.integer_bytes
    if prefix.header.flags & ((1 << 8) | (1 << 9)):
        raise RuntimeError(f"fixture requires disabled groups and tracers: {checkpoint}")
    if not prefix.hydro or prefix.has_cooling or prefix.has_stars:
        raise RuntimeError(f"fixture requires all gas without cooling or star formation: {checkpoint}")
    method_code = {"sph": 0, "mfm": 1, "mfv": 2}[method]
    if (prefix.header.reserved >> 25) & 3 != method_code:
        raise RuntimeError(f"checkpoint hydro method differs from {method}: {checkpoint}")

    def integer(offset):
        if offset < 0 or offset + integer_bytes > len(data):
            raise RuntimeError(f"truncated particle integer in {checkpoint}")
        return int.from_bytes(data[offset:offset + integer_bytes], sys.byteorder, signed=True)

    # Decode the current typed prefix and advance in writer order. Never scan
    # particle data for a magic word that could occur in an arbitrary float.
    start = prefix.particle_start
    count = integer(start)
    if count < 0 or integer(start + integer_bytes) != count:
        raise RuntimeError(f"fixture is not entirely gas in {checkpoint}")
    header_offset = start + 2 * integer_bytes + count * (3 * position_bytes + 14 + integer_bytes)
    if header_offset + 28 + integer_bytes > len(data):
        raise RuntimeError(f"truncated gas header in {checkpoint}")
    header = struct.unpack_from("=7I", data, header_offset)
    expected_header = (0x43545350, prefix.gas_version + METHOD_OFFSETS[method], integer_bytes, 8, 12, 3, 4)
    viscosity_version = prefix.gas_version - (1000 if prefix.header.flags & (1 << 6) else 0)
    viscosity_bytes = {2: 0, 4: 8, 6: 12, 8: 12}.get(viscosity_version)
    if header != expected_header or viscosity_bytes is None or integer(header_offset + 28) != count:
        raise RuntimeError(f"unsupported {method} gas layout in {checkpoint}: {header}")
    positions_offset = start + 2 * integer_bytes
    velocities_offset = positions_offset + 3 * position_bytes * count
    rungs_offset = velocities_offset + 12 * count
    gas_offset = header_offset + 28 + integer_bytes
    indices_offset = gas_offset + 20 * count
    density_offset = indices_offset + integer_bytes * count + 8 * count
    mass_offset = gas_offset + count * (58 + integer_bytes)
    mass_bytes = 4 if method == "mfv" else 0
    end = mass_offset + count * (mass_bytes + viscosity_bytes)
    if end > len(data):
        raise RuntimeError(f"truncated gas arrays in {checkpoint}")

    def floats(offset, length):
        return struct.unpack_from(f"={length}f", data, offset)

    gas_mass = prefix.header.physics[1]
    # Match the application conversion of its class mass to stored float precision.
    fixed_mass = struct.unpack("=f", struct.pack("=f", gas_mass))[0]
    masses = floats(mass_offset, count) if method == "mfv" else (fixed_mass,) * count
    velocity_words = floats(velocities_offset, 3 * count)
    velocities = [tuple(velocity_words[3 * integer(indices_offset + i * integer_bytes) + d]
                        for d in range(3)) for i in range(count)]
    rungs = [data[rungs_offset + integer(indices_offset + i * integer_bytes)]
             for i in range(count)]
    if any(rung >= 16 for rung in rungs):
        raise RuntimeError(f"invalid gas rung in {path}")
    thermal = floats(gas_offset, 2 * count)[::2]
    supports = floats(gas_offset, 2 * count)[1::2]
    densities = floats(density_offset, count)
    predicted_offset = gas_offset + count * (34 + integer_bytes)
    pending_work_components = [floats(predicted_offset + 8 * d * count, count)
                               for d in range(3)]
    pending_work = [tuple(pending_work_components[d][i] for d in range(3))
                    for i in range(count)]
    position_format = "Q" if position_bytes == 8 else "I"
    position_words = struct.unpack_from(f"={3 * count}{position_format}", data, positions_offset)
    raw_positions = [tuple(position_words[d * count + integer(indices_offset + i * integer_bytes)]
                           for d in range(3)) for i in range(count)]
    positions = [tuple(math.ldexp(word, -position_bits) for word in position)
                 for position in raw_positions]
    integration_state = []
    pair_cache = []
    if method != "sph":
        if position_bytes != 8:
            raise RuntimeError(f"meshless checkpoints require Fixed64 positions: {path}")
        if end + 12 > len(data):
            raise RuntimeError(f"missing meshless integration state in {path}")
        state_header = struct.unpack_from("=3I", data, end)
        if state_header != (0x484f5053, 7, 472):
            raise RuntimeError(f"unsupported meshless integration state in {path}: {state_header}")
        state_offset = end + 12
        cache_offset = state_offset + 472 * count
        if cache_offset + 32 > len(data):
            raise RuntimeError(f"truncated meshless integration state in {path}")
        # Native AoS records have padding after each bool. Compare every named
        # field exactly without treating that padding as numerical state.
        integration_state = [struct.unpack_from("=Q30d2Q?", data, state_offset + 472 * i) +
                             struct.unpack_from("=15d?", data, state_offset + 472 * i + 272) +
                             struct.unpack_from("=d", data, state_offset + 472 * i + 400) +
                             struct.unpack_from("=3Q4di?", data, state_offset + 472 * i + 408)
                             for i in range(count)]
        for index, state in enumerate(integration_state):
            if (not state[33] or not state[49] or
                    any(not math.isfinite(value) for value in state[1:]) or
                    state[1] <= 0.0 or state[2] <= math.fsum(
                        value * value for value in state[4:7]) / (2.0 * state[1]) or
                    state[3] <= 0.0 or state[12] <= 0.0 or state[50] <= 0.0):
                raise RuntimeError(f"invalid conservative meshless state in {path}")
            if method == "mfm" and state[1] != fixed_mass:
                raise RuntimeError(f"MFM changed a particle's conserved mass in {path}")
            if struct.unpack("=f", struct.pack("=f", state[1] + state[7]))[0] != masses[index]:
                raise RuntimeError(f"conserved and public particle masses disagree in {path}")
            if struct.unpack("=f", struct.pack("=f", state[50]))[0] != supports[index]:
                raise RuntimeError(f"double and public support radii disagree in {path}")
        pending_work = [state[25:28] for state in integration_state]
        cache_header = struct.unpack_from("=4Q", data, cache_offset)
        if cache_header[:3] != (0x484f504341434845, 2, 176):
            raise RuntimeError(f"unsupported pair cache in {path}: {cache_header}")
        cache_end = cache_offset + 32 + 176 * cache_header[3]
        if cache_end > len(data):
            raise RuntimeError(f"truncated pair cache in {path}")
        pair_cache = [struct.unpack_from("=4Q18d", data, cache_offset + 32 + 176 * i)
                      for i in range(cache_header[3])]
        for pair in pair_cache:
            if (pair[0] >= pair[1] or pair[4] >= pair[5] or
                    (pair[2] >= pair[3] and (pair[2] != 0 or pair[3] != 0)) or
                    any(not math.isfinite(value) for value in pair[4:])):
                raise RuntimeError(f"invalid cached pair interval in {path}")
    for name, values in (("mass", masses), ("density", densities),
                         ("internal energy", thermal), ("support", supports)):
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise RuntimeError(f"nonpositive or nonfinite {name} in {checkpoint}")
    if any(not math.isfinite(value) for velocity in velocities for value in velocity):
        raise RuntimeError(f"nonfinite velocity in {checkpoint}")
    if any(not math.isfinite(value) for work in pending_work for value in work):
        raise RuntimeError(f"nonfinite pending gravitational work in {checkpoint}")
    primitive_momentum = tuple(math.fsum(masses[i] * velocities[i][d] for i in range(count))
                               for d in range(3))
    primitive_energy = math.fsum(masses[i] * (thermal[i] + 0.5 * math.fsum(
        value * value for value in velocities[i])) for i in range(count))
    momentum = (tuple(math.fsum(state[4 + d] + state[9 + d] for state in integration_state) for d in range(3))
                if method != "sph" else primitive_momentum)
    energy = (math.fsum(state[2] + state[8] for state in integration_state)
              if method != "sph" else primitive_energy)
    total_mass = (math.fsum(state[1] + state[7] for state in integration_state)
                  if method != "sph" else math.fsum(masses))
    physical_records = {}
    for i, state in enumerate(integration_state):
        particle = integer(indices_offset + i * integer_bytes)
        # Domain/tree sorting may change local array order after restart. Keep
        # every persistent physical field bit-exact, indexed by stable identity,
        # while excluding the two storage-index maps that encode that order.
        pieces = [data[positions_offset + position_bytes * (d * count + particle):
                       positions_offset + position_bytes * (d * count + particle + 1)]
                  for d in range(3)]
        pieces += [data[velocities_offset + 12 * particle:velocities_offset + 12 * (particle + 1)],
                   data[rungs_offset + particle:rungs_offset + particle + 1],
                   data[rungs_offset + count + particle:rungs_offset + count + particle + 1],
                   data[gas_offset + 8 * i:gas_offset + 8 * (i + 1)],
                   data[gas_offset + 8 * count + 12 * i:gas_offset + 8 * count + 12 * (i + 1)]]
        rate_offset = indices_offset + integer_bytes * count
        pieces += [data[rate_offset + 4 * (d * count + i):
                        rate_offset + 4 * (d * count + i + 1)] for d in range(3)]
        flag_offset = density_offset + 4 * count
        pieces += [data[flag_offset + d * count + i:flag_offset + d * count + i + 1]
                   for d in range(2)]
        pieces += [data[predicted_offset + 4 * (d * count + i):
                        predicted_offset + 4 * (d * count + i + 1)] for d in range(6)]
        if mass_bytes:
            pieces.append(data[mass_offset + 4 * i:mass_offset + 4 * (i + 1)])
        viscosity_offset = mass_offset + mass_bytes * count
        pieces.append(data[viscosity_offset + viscosity_bytes * i:
                           viscosity_offset + viscosity_bytes * (i + 1)])
        physical_records[state[0]] = b"".join(pieces)
    clock = (prefix.driver.tau, prefix.driver.global_timestep, prefix.driver.integer_time,
             prefix.driver.max_rung, prefix.driver.iteration, prefix.driver.step)
    if not all(math.isfinite(value) for value in clock):
        raise RuntimeError(f"nonfinite checkpoint clock in {checkpoint}")
    return {"payload": data[start:end], "count": count, "masses": masses,
            "clock": clock, "algorithms": prefix.algorithms,
            "mass": total_mass, "momentum": momentum, "energy": energy,
            "fixed_mass": fixed_mass, "pending_work": pending_work,
            "positions": positions, "raw_positions": raw_positions,
            "position_bits": position_bits, "rungs": rungs,
            "integration_state": integration_state, "pair_cache": pair_cache,
            "physical_records": physical_records,
            "primitive_mass": math.fsum(masses), "primitive_momentum": primitive_momentum,
            "primitive_energy": primitive_energy}


def particle_state(checkpoint, method):
    """Reduce all rank payloads, including the coordinator's persistent pair cache."""
    files = sorted(checkpoint.glob("checkpoint.*.dat"))
    if not files:
        raise RuntimeError(f"missing rank files in {checkpoint}")
    ranks = [particle_rank_state(path, method) for path in files]
    result = {key: math.fsum(rank[key] for rank in ranks)
              for key in ("mass", "energy", "primitive_mass", "primitive_energy")}
    result["count"] = sum(rank["count"] for rank in ranks)
    if result["count"] <= 0:
        raise RuntimeError(f"empty gas checkpoint {checkpoint}")
    for key in ("momentum", "primitive_momentum"):
        result[key] = tuple(math.fsum(rank[key][d] for rank in ranks) for d in range(3))
    for key in ("masses", "pending_work", "positions", "raw_positions", "rungs",
                "integration_state", "pair_cache"):
        result[key] = [value for rank in ranks for value in rank[key]]
    result["payload"] = tuple((path.name, rank["payload"]) for path, rank in zip(files, ranks))
    result["fixed_mass"] = ranks[0]["fixed_mass"]
    if any(rank["fixed_mass"] != result["fixed_mass"] for rank in ranks):
        raise RuntimeError(f"rank gas masses disagree in {checkpoint}")
    result["position_bits"] = ranks[0]["position_bits"]
    if any(rank["position_bits"] != result["position_bits"] for rank in ranks):
        raise RuntimeError(f"rank position representations disagree in {checkpoint}")
    for key in ("clock", "algorithms"):
        result[key] = ranks[0][key]
        if any(rank[key] != result[key] for rank in ranks):
            raise RuntimeError(f"rank {key} values disagree in {checkpoint}")
    identifiers = [state[0] for state in result["integration_state"]]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError(f"duplicate meshless particle identifiers in {checkpoint}")
    result["identifier_ranks"] = {state[0]: rank_index for rank_index, rank in enumerate(ranks)
                                  for state in rank["integration_state"]}
    result["predictor_anchors"] = {state[0]: state[1:7] + state[12:22] + state[28:29] + state[30:32] + state[34:49] for state in result["integration_state"]}
    result["kick_phases"] = [state[31] for state in result["integration_state"]]
    result["physical_records"] = {identifier: values for rank in ranks
                                  for identifier, values in rank["physical_records"].items()}
    return result


def check_manifest(checkpoint, iteration, scenario, duration, localities, gravity=False):
    if localities == 1:
        checkManifest(checkpoint, iteration, scenario, duration, gravity)
        return
    lines = (checkpoint / "checkpoint.meta").read_text().splitlines()
    if (len(lines) != localities + 2 or lines[0].split() != [
            "COSMICTIGER_CHECKPOINT", "2", "3", str(iteration), str(localities)]):
        raise RuntimeError(f"unexpected distributed checkpoint manifest: {lines}")
    for rank, line in enumerate(lines[1:-1]):
        fields = line.split()
        payload = (checkpoint / f"checkpoint.{rank}.dat").read_bytes()
        if (len(fields) != 3 or fields[0] != str(rank) or int(fields[1]) != len(payload) or
                int(fields[2], 16) != fnv1a64(payload)):
            raise RuntimeError(f"invalid rank {rank} checkpoint manifest in {checkpoint}")
    scenario_data = (checkpoint / "scenario.meta").read_bytes()
    fields = lines[-1].split()
    if (len(fields) != 3 or fields[0] != "SCENARIO" or int(fields[1]) != len(scenario_data) or
            int(fields[2], 16) != fnv1a64(scenario_data)):
        raise RuntimeError(f"invalid scenario checkpoint manifest in {checkpoint}")
    fields = scenario_data.decode().splitlines()
    if (len(fields) != 4 or fields[:2] != ["COSMICTIGER_SCENARIO 1", scenario] or
            float(fields[2]) != duration or fields[3] != str(int(gravity))):
        raise RuntimeError(f"unexpected scenario checkpoint metadata in {checkpoint}")


def check_conservation(initial, final, context):
    if initial["count"] != final["count"]:
        raise RuntimeError(f"particle count changed during {context}")
    # Inactive public primitives are predicted from immutable own-step anchors;
    # they are not a second conserved ledger. Test U_anchor + pending transfers,
    # and separately check the float mass view used by gravity/output.
    representations = (("", 1.e-10),) if initial["integration_state"] else (("", 1.e-10), ("primitive_", 5.e-6))
    if abs(final["primitive_mass"] - initial["primitive_mass"]) > 5.e-6 * initial["primitive_mass"]:
        raise RuntimeError(f"{context} float mass view lost total mass")
    for prefix, tolerance in representations:
        relative_mass_error = abs(final[prefix + "mass"] - initial[prefix + "mass"]) / \
            initial[prefix + "mass"]
        relative_energy_error = abs(final[prefix + "energy"] - initial[prefix + "energy"]) / \
            initial[prefix + "energy"]
        momentum_scale = math.sqrt(initial[prefix + "mass"] * initial[prefix + "energy"])
        relative_momentum_error = max(abs(final[prefix + "momentum"][d] -
                                          initial[prefix + "momentum"][d])
                                      for d in range(3)) / momentum_scale
        # Persistent conserved quantities use double precision. Independently
        # check the public float primitive representation at storage precision.
        if max(relative_mass_error, relative_energy_error, relative_momentum_error) > tolerance:
            raise RuntimeError(f"{context} violated {prefix or 'conservative '}conservation: "
                               f"mass={relative_mass_error:.3e}, energy={relative_energy_error:.3e}, "
                               f"momentum={relative_momentum_error:.3e}")
        if not prefix:
            print(f"{context}: mass error {relative_mass_error:.3e}, "
                  f"momentum error {relative_momentum_error:.3e}, "
                  f"energy error {relative_energy_error:.3e}", flush=True)


def check_activity(output, states, context, localities):
    tree_records = [dict(re.findall(r"([A-Za-z]+)=([^\s]+)", match.group(1)))
                    for match in re.finditer(r"Hopkins tree: ([^\n]+)", output)]
    if len(tree_records) < 2:
        raise RuntimeError(f"{context} printed fewer than two hydro tree rows")
    records = []
    for match in re.finditer(r"Hopkins step: ([^\n]+)", output):
        fields = dict(re.findall(r"([A-Za-z]+)=([^\s]+)", match.group(1)))
        try:
            record = {key: int(fields[key]) for key in (
                "active", "total", "minRung", "maxRung", "pairs", "cached", "refreshed", "woken")}
            record["time"] = float(fields["time"])
        except (KeyError, ValueError) as error:
            raise RuntimeError(f"malformed Hopkins activity record: {match.group(0)}") from error
        if (not math.isfinite(record["time"]) or record["active"] > record["total"] or
                min(record[key] for key in record if key != "time") < 0 or
                record["minRung"] > record["maxRung"] or
                record["pairs"] != record["cached"] + record["refreshed"] or
                record["total"] != states[0]["count"]):
            raise RuntimeError(f"inconsistent Hopkins activity record: {match.group(0)}")
        records.append(record)
    if len(records) < len(states):
        raise RuntimeError(f"{context} omitted per-step Hopkins activity records")
    # The tree is rebuilt at events with at least hydro.treeRebuildActiveFraction
    # (default one third, GADGET-4's N1 > N0/3) of the cells active and refit in
    # place otherwise; events below that fraction must therefore reuse the
    # persistent topology. Tree rows and step rows are printed once per event in
    # the same order.
    quiet = [index for index, record in enumerate(records)
             if index >= 1 and index < len(tree_records) and 3 * record["active"] < record["total"]]
    if quiet and not any(int(tree_records[index]["updated"]) > 0 and
                         int(tree_records[index]["sorted"]) == 0 and
                         int(tree_records[index]["rebuilds"]) == 0 for index in quiet):
        raise RuntimeError(f"{context} did not reuse persistent hydro tree topology")
    if not any(len(set(state["rungs"])) > 1 for state in states):
        raise RuntimeError(f"{context} never stored distinct gas rungs")
    if not any(record["active"] < record["total"] and record["cached"] > 0 and
               record["minRung"] < record["maxRung"] for record in records):
        raise RuntimeError(f"{context} never reused a pair flux while gas was inactive")
    if not any(max(state["kick_ages"]) > state["fine_ticks"] for state in states):
        raise RuntimeError(f"{context} never retained a coarse predictor across a fine event")
    retained = refreshed = 0
    for previous, current in zip(states, states[1:]):
        for identifier, anchor in current["predictor_anchors"].items():
            old_anchor = previous["predictor_anchors"][identifier]
            anchor_tick = current["predictor_ticks"][identifier]
            old_tick = previous["predictor_ticks"][identifier]
            if anchor_tick == old_tick:
                if anchor != old_anchor:
                    raise RuntimeError(f"{context} changed an inactive primitive/gradient anchor")
                retained += 1
            else:
                if anchor_tick < old_tick:
                    raise RuntimeError(f"{context} moved a predictor clock backwards")
                refreshed += 1
    if retained == 0 or refreshed == 0:
        raise RuntimeError(f"{context} did not exercise both retained and refreshed predictors")
    if localities > 1 and not any(
            state["identifier_ranks"][pair[0]] != state["identifier_ranks"][pair[1]]
            for state in states for pair in state["pair_cache"]):
        raise RuntimeError(f"{context} did not cache a pair across MPI owners")
    minimum_active = min(record["active"] for record in records)
    maximum_cached = max(record["cached"] for record in records)
    print(f"{context}: minimum active gas {minimum_active}/{states[0]['count']}, "
          f"maximum reused pairs {maximum_cached}, retained predictor records {retained}", flush=True)


def check_uniform_acceleration(destination, mass, acceleration, context, method):
    """Use the independent integrated momentum/work laws at synchronized times."""
    check_finite_rows(destination / "energy.dat")
    rows = [list(map(float, line.split()))
            for line in (destination / "energy.dat").read_text().splitlines() if line.strip()]
    if len(rows) < 2 or rows[0][0] != 0.0 or rows[-1][0] <= 0.0:
        raise RuntimeError(f"{context} did not write evolving synchronized diagnostics")
    initial_momentum = rows[0][2]
    initial_energy = rows[0][6]
    maximum_momentum_error = 0.0
    maximum_energy_error = 0.0
    for row in rows:
        time = row[0]
        expected_momentum = initial_momentum + mass * acceleration * time
        expected_energy = initial_energy + initial_momentum * acceleration * time + \
            0.5 * mass * (acceleration * time)**2
        momentum_error = abs(row[2] - expected_momentum)
        energy_error = abs(row[6] - expected_energy)
        # Diagnostics have six printed decimal places in scientific notation,
        # and the evolved primitive fields use float storage.
        if momentum_error > 5.e-7 * max(1.0, abs(expected_momentum)):
            raise RuntimeError(f"{context} failed the uniform-force momentum law at t={time}: "
                               f"actual={row[2]}, expected={expected_momentum}")
        if energy_error > 2.e-6 * max(1.0, abs(initial_energy)):
            raise RuntimeError(f"{context} failed the uniform-force work law at t={time}: "
                               f"actual={row[6]}, expected={expected_energy}")
        maximum_momentum_error = max(maximum_momentum_error, momentum_error)
        maximum_energy_error = max(maximum_energy_error, energy_error)
    print(f"{context}: uniform-force momentum error {maximum_momentum_error:.3e}, "
          f"work error {maximum_energy_error:.3e}", flush=True)
    snapshots = []
    for fields in (line.split() for line in (destination / "progress.txt").read_text().splitlines()):
        state = particle_state(destination / f"checkpoint.{fields[0]}", method)
        # The position clock may be ahead of the owning conserved anchor.
        # This uniform-force fixture has a single gas rung; compare the stored
        # U at its own source clock, and the diagnostics above at physical time.
        anchors = {record[28] for record in state["integration_state"]}
        if len(anchors) != 1:
            raise RuntimeError(f"{context} uniform-force fixture has asynchronous source anchors")
        anchor_time = anchors.pop()
        if anchor_time > state["clock"][0]:
            raise RuntimeError(f"{context} source anchor exceeds its position clock")
        snapshots.append((anchor_time, state))
    first_time, first = snapshots[0]
    for time, state in snapshots:
        expected_momentum = initial_momentum + mass * acceleration * time
        if abs(state["momentum"][1] - expected_momentum) > 1.e-11 * max(1.0, abs(expected_momentum)):
            raise RuntimeError(f"{context} checkpoint momentum is not at its own source clock")
        elapsed = time - first_time
        expected_energy = first["energy"] + first["momentum"][1] * acceleration * elapsed + \
            0.5 * mass * (acceleration * elapsed)**2
        # MFV also applies the separately discretized work of intercell mass
        # transfer. Its uniform-force energy check uses the diagnostic tolerance
        # above; MFM has no such contribution and can be checked at double precision.
        if method == "mfm" and abs(state["energy"] - expected_energy) > \
                1.e-10 * max(1.0, abs(first["energy"])):
            raise RuntimeError(f"{context} checkpoint energy violates the elapsed source work law")
    print(f"{context}: checkpoint momentum matches the physical clock", flush=True)


def pending_gravity_work_signal(state):
    """Measure source work retained in endpoint ledgers and elapsed live pairs."""
    integrals = {entry[0]: list(entry[22:25]) for entry in state["integration_state"]}
    moments = {entry[0]: list(entry[25:28]) for entry in state["integration_state"]}
    anchors = {entry[0]: entry for entry in state["integration_state"]}
    time = state["clock"][0]
    for pair in state["pair_cache"]:
        interval = pair[5] - pair[4]
        elapsed = max(0.0, min(time, pair[5]) - pair[4])
        for endpoint in pair[:2]:
            age = pair[4] - anchors[endpoint][28]
            for dimension in range(3):
                initial = pair[16 + dimension]
                slope = 2.0 * (pair[19 + dimension] - initial) / interval
                integral = elapsed * (initial + 0.5 * elapsed * slope)
                first_moment = elapsed * elapsed * (0.5 * initial + elapsed * slope / 3.0)
                integrals[endpoint][dimension] += integral
                moments[endpoint][dimension] += age * integral + first_moment
    signal = 0.0
    for identifier, entry in anchors.items():
        force = entry[19:22]
        signal = max(signal, abs(math.fsum(integrals[identifier][d] * force[d]
                                          for d in range(3))))
        duration = time - entry[28]
        if duration > 0.0:
            signal = max(signal, abs(math.fsum(moments[identifier][d] * force[d]
                                              for d in range(3))) / duration)
    return signal


def check_wake_transactions(output, states, context):
    """Check real early closures and both sides of every retired pair integral."""
    records = [dict(re.findall(r"([A-Za-z]+)=([^\s]+)", match.group(1)))
               for match in re.finditer(r"Hopkins step: ([^\n]+)", output)]
    records = [record for record in records if int(record["terminal"]) == 0]
    if not any(max(state["rungs"]) - min(state["rungs"]) >= 3 for state in states):
        raise RuntimeError(f"{context} never spanned four rung levels")
    first_wake = None
    total_woken = truncated = 0
    for step, (previous, current) in enumerate(zip(states, states[1:]), start=1):
        time, tick = previous["clock"][0], previous["tick"]
        before = {entry[0]: entry for entry in previous["integration_state"]}
        after = {entry[0]: entry for entry in current["integration_state"]}
        woken = {identifier for identifier in before
                 if before[identifier][32] > tick and after[identifier][31] == tick}
        record = next((entry for entry in records if math.isclose(float(entry["time"]), time,
                       rel_tol=1.e-13, abs_tol=1.e-15)), None)
        if record is None or int(record["woken"]) != len(woken):
            raise RuntimeError(f"{context} wake log differs from early own-step closures")
        if not woken:
            continue
        if first_wake is None:
            first_wake = step
        total_woken += len(woken)
        pairs_after = {pair[:2]: pair for pair in current["pair_cache"]}
        deposits = {identifier: [0.0] * 5 for identifier in before}
        for pair in previous["pair_cache"]:
            if pairs_after.get(pair[:2]) == pair:
                if pair[0] in woken or pair[1] in woken:
                    raise RuntimeError(f"{context} retained a pair across an endpoint wake")
                continue
            duration = min(time, pair[5]) - pair[4]
            interval = pair[5] - pair[4]
            delta = [duration * (pair[6 + component] + duration / interval *
                     (pair[11 + component] - pair[6 + component])) for component in range(5)]
            if pair[2] < tick < pair[3] and (pair[0] in woken or pair[1] in woken):
                truncated += 1
            for endpoint, sign in ((pair[0], -1.0), (pair[1], 1.0)):
                for component in range(5):
                    deposits[endpoint][component] += sign * delta[component]
        # This fixture has no gravity, damping, or cosmological sources. The
        # independently integrated old pair records must explain every change
        # in U + pending, including the inactive partners of woken particles.
        for identifier in before:
            old, new = before[identifier], after[identifier]
            for component, (anchor, pending) in enumerate(((1, 7), (4, 9), (5, 10), (6, 11), (2, 8))):
                expected = old[anchor] + old[pending] + deposits[identifier][component]
                actual = new[anchor] + new[pending]
                if not math.isclose(actual, expected, rel_tol=2.e-12, abs_tol=1.e-14):
                    raise RuntimeError(f"{context} wake changed particle {identifier} conserved "
                                       f"component {component} outside its elapsed pair deposits")
    if first_wake is None or truncated == 0:
        raise RuntimeError(f"{context} never truncated a live pair at a real wake")
    populations = max(len(set(state["rungs"])) for state in states)
    print(f"{context}: {total_woken} early own-step closures, {truncated} truncated pairs, "
          f"{populations} occupied rungs spanning four levels", flush=True)
    return first_wake


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--mpi-exec", type=Path)
    parser.add_argument("--localities", type=int, default=1)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2,
                        help="HPX worker threads per locality")
    parser.add_argument("--expansion-degree", type=int, default=6)
    parser.add_argument("--async-only", action="store_true",
                        help="run only the individual-rung conservation/restart fixtures")
    parser.add_argument("--wake-only", action="store_true",
                        help="check natural wake-up truncation and restart over two coarse steps")
    parser.add_argument("--global-sources-only", action="store_true",
                        help="check global uniform-source laws and forced restart with unequal final intervals")
    parser.add_argument("--thermal-sources-only", action="store_true",
                        help="check MFM equilibrium cooling and star formation against SPH on a homogeneous lattice")
    parser.add_argument("--matrix-only", action="store_true",
                        help="run gravity-coupled MFM/MFV evolution and restart across FMM/PMMM and traditional/global")
    parser.add_argument("--gravity-method", choices=("fmm", "pmmm"),
                        help="restrict --matrix-only to this far-field solver")
    parser.add_argument("--near-method", choices=("pp", "fmm"),
                        help="restrict --matrix-only PMMM cases to this near-field solver")
    parser.add_argument("--time-integrator", choices=("traditional", "global", "hierarchical"),
                        help="restrict --matrix-only to this integrator")
    parser.add_argument("--cuda", action="store_true",
                        help="use CUDA gravity and drift for --matrix-only; Hopkins hydro remains on the CPU")
    args = parser.parse_args()
    if args.cuda and not args.matrix_only:
        parser.error("--cuda requires --matrix-only")
    if sum((args.matrix_only, args.async_only, args.global_sources_only, args.wake_only,
            args.thermal_sources_only)) > 1:
        parser.error("--matrix-only, --async-only, --wake-only, --global-sources-only and "
                     "--thermal-sources-only are mutually exclusive")
    if not args.matrix_only and any((args.gravity_method, args.near_method, args.time_integrator)):
        parser.error("solver/integrator selectors require --matrix-only")
    if args.localities < 1 or (args.localities > 1 and args.mpi_exec is None):
        parser.error("--localities must be positive and requires --mpi-exec when greater than one")
    command = [str(args.executable.resolve()), f"--hpx:threads={args.threads}", "--hpx:bind=none"]
    if args.localities > 1:
        command = [str(args.mpi_exec.resolve()), "-n", str(args.localities)] + command + [
            f"--hpx:localities={args.localities}"]
    base = args.workdir.resolve()
    base.mkdir(parents=True, exist_ok=True)

    def prepare(name):
        directory = base / name
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()
        return directory

    def options(scenario, duration, method, iterations):
        values = {"run.scenario": scenario, "run.endTime": duration,
                  "run.numberOfSteps": 8, "run.maxIterations": iterations, "run.verbose": "true",
                  "initial.particlesPerDimension": 8,
                  "hydro.enabled": "true", "gravity.enabled": "false",
                  "run.timeIntegrator": "traditional", "gravity.method.far": "fmm",
                  "gravity.expansionDegree": args.expansion_degree, "starFormation.enabled": "false",
                  "chemistry.enabled": "false", "gravity.cuda": "false",
                  "compute.positionType": "fixed64",
                  "analysis.groups": "false", "analysis.tracers": "false",
                  "analysis.power": "false", "analysis.views": "false",
                  "lightcone.enabled": "false", "output.silo": "false",
                  "output.directory": "output", "checkpoint.intervalSeconds": 0}
        if method is not None:
            values["hydro.method"] = method
        return values

    def launch(directory, values, success=True):
        invocation = command + [f"--{key}={value}" for key, value in values.items()]
        if not args.cuda:
            return run_process(invocation, directory, success)
        # Preserve the caller's GPU selection. The shared CPU runner deliberately
        # hides all CUDA devices, so accelerated fixtures need their own launcher.
        log_path = directory / "run.log"
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with log_path.open("w") as log:
            with subprocess.Popen(invocation, cwd=directory, stdout=log,
                                  stderr=subprocess.STDOUT, text=True, env=environment,
                                  start_new_session=True) as process:
                try:
                    result = process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise RuntimeError(f"CUDA Hopkins fixture timed out; see {log_path}")
        output = log_path.read_text()
        if (result == 0) != success:
            raise RuntimeError(f"unexpected exit {result}; see {log_path}\n{output[-6000:]}")
        return output

    def check_restart(name, method, values, reference_output, reference_state,
                      reject_other_methods=False, split_step=2, final_step=4,
                      require_pending_pair=False):
        split_directory = prepare(f"{name}_split")
        launch(split_directory, dict(values, **{"run.maxIterations": split_step}))
        source = split_directory / f"output/checkpoint.{split_step}"
        split_state = particle_state(source, method)
        if require_pending_pair:
            progress = (split_directory / "output/progress.txt").read_text().splitlines()[-1].split()
            # The meshless cache clock has 2^(kMaxRung-1) ticks per base interval.
            tick = round(float(progress[1]) / (values["run.endTime"] /
                         values["run.numberOfSteps"]) * 32768)
            if not any(pair[2] < tick < pair[3] for pair in split_state["pair_cache"]):
                raise RuntimeError(f"{name} restart did not split a cached coarse pair interval")
            if len(set(split_state["rungs"])) < 2:
                raise RuntimeError(f"{name} restart did not contain different gas rungs")
        directory = prepare(f"{name}_restart")
        destination = directory / "output"
        destination.mkdir()
        shutil.copytree(source, destination / source.name)
        output = launch(directory, dict(values, **{
            "run.maxIterations": final_step - split_step, "checkpoint.readIndex": split_step}))
        if "Done reading checkpoint" not in output:
            raise RuntimeError(f"{name} did not load its restart")
        check_manifest(destination / f"checkpoint.{final_step}", final_step,
                       values["run.scenario"], values["run.endTime"], args.localities,
                       values["gravity.enabled"] == "true")
        resumed = particle_state(destination / f"checkpoint.{final_step}", method)
        physical_key = "payload" if args.localities == 1 else "physical_records"
        if resumed[physical_key] != reference_state[physical_key]:
            raise RuntimeError(f"{name} restart changed physical particle state")
        for key in ("integration_state", "pair_cache", "clock", "algorithms"):
            actual, expected = resumed[key], reference_state[key]
            if key == "integration_state" and args.localities > 1:
                actual = {state[0]: state for state in actual}
                expected = {state[0]: state for state in expected}
            if actual != expected:
                raise RuntimeError(f"{name} restart changed persisted {key}")
        resumed_clock = (destination / "progress.txt").read_text().splitlines()[-1].split()
        reference_clock = (reference_output / "progress.txt").read_text().splitlines()[-1].split()
        if resumed_clock != reference_clock:
            raise RuntimeError(f"{name} restart changed the physical clock")
        if reject_other_methods:
            for other in ({"sph", "mfm", "mfv"} - {method}):
                rejected_directory = prepare(f"{name}_reject_{other}")
                rejected_output = rejected_directory / "output"
                rejected_output.mkdir()
                shutil.copytree(source, rejected_output / source.name)
                rejected = launch(rejected_directory, dict(values, **{
                    "hydro.method": other, "checkpoint.readIndex": split_step}), False)
                if "Done reading checkpoint" in rejected or "checkpoint" not in rejected.lower():
                    raise RuntimeError(f"{method} checkpoint was not rejected with {other}")
        print(f"{name}: exact physical restart passed", flush=True)
        return split_state

    def checkpoint_states(destination, method, scenario, duration, steps, context, number_of_steps=8):
        states = []
        progress = {int(fields[0]): fields for fields in (
            line.split() for line in (destination / "progress.txt").read_text().splitlines())}
        time = 0.0
        tick = 0
        for step in range(1, steps + 1):
            checkpoint = destination / f"checkpoint.{step}"
            check_manifest(checkpoint, step, scenario, duration, args.localities)
            states.append(particle_state(checkpoint, method))
            # Reconstruct the exact power-of-two time from the printed integer
            # rung; the diagnostic's decimal time has only six-place precision.
            finest_rung = int(progress[step][4])
            timestep = math.ldexp(duration / number_of_steps, -finest_rung)
            fine_ticks = 32768 >> finest_rung
            time += timestep
            tick += fine_ticks
            states[-1]["time"] = time
            states[-1]["timestep"] = timestep
            states[-1]["fine_ticks"] = fine_ticks
            states[-1]["tick"] = tick
            states[-1]["kick_ages"] = [tick - phase
                                       for phase in states[-1]["kick_phases"]]
            states[-1]["predictor_ticks"] = {
                state[0]: tick - age for state, age in zip(
                    states[-1]["integration_state"], states[-1]["kick_ages"])}
            if any(age <= 0 or age > (32768 >> rung) or age > tick
                   for age, rung in zip(states[-1]["kick_ages"], states[-1]["rungs"])):
                raise RuntimeError(f"{context} has a predictor age outside its gas-rung interval")
            if step > 1:
                check_conservation(states[0], states[-1], f"{context} step {step}")
        return states

    if args.thermal_sources_only:
        # MFM couples the GADGET-4 sources to its conserved state. On a homogeneous
        # lattice every pair flux vanishes, so MFM and SPH reduce to the same
        # cooling history and the same stochastic conversion draws; compare them
        # there, then check conversions leave a consistent gas set, the next kick
        # settles the survivors' pairs, and a restart is exact.
        help_text = subprocess.run(
            command + ["--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=60, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="")).stdout
        capabilities = dict(re.findall(r"^\s*(Equilibrium cooling|Star formation): (enabled|disabled)",
                                       help_text, re.MULTILINE))
        if capabilities.get("Equilibrium cooling") != "enabled":
            raise RuntimeError("thermal sources require a build with equilibrium cooling")
        star_formation_available = capabilities.get("Star formation") == "enabled"
        integrators = [value for value in ("traditional", "global") if re.search(
            r"^\s*Time integrators:.*\b" + value + r"\b", help_text, re.MULTILINE)]
        if not integrators:
            raise RuntimeError("thermal sources require the traditional or global integrator")

        def thermal_options(method, integrator, iterations, star_formation):
            values = {"run.scenario": "cosmology", "run.timeIntegrator": integrator,
                      "run.maxIterations": iterations, "run.numberOfSteps": 64, "run.minRung": 0,
                      "run.verbose": "true", "hydro.enabled": "true", "hydro.method": method,
                      "gravity.enabled": "true", "gravity.method.far": "fmm",
                      "gravity.expansionDegree": 6, "gravity.cuda": "false", "hydro.cuda": "false",
                      "drift.cuda": "false", "compute.positionType": "fixed64",
                      "initial.particlesPerDimension": 8, "initial.closePack": "false",
                      "initial.useGlass": "false", "cosmology.sigma8": 0,
                      "cosmology.startRedshift": 49, "cosmology.endRedshift": 0,
                      "chemistry.enabled": "true", "chemistry.initializationTemperature": 10000,
                      "starFormation.enabled": str(star_formation).lower(),
                      "analysis.groups": "false", "analysis.tracers": "false",
                      "analysis.power": "false", "analysis.views": "false",
                      "lightcone.enabled": "false", "output.silo": "false",
                      "output.directory": "output", "checkpoint.intervalSeconds": 0}
            if star_formation:
                # Every lattice cell is eligible and converts with a few-percent
                # chance per step once the saved SFR exists, leaving most gas.
                values.update({"starFormation.criticalOverDensity": 0,
                               "starFormation.criticalPhysicalDensity": 1.e-3,
                               "starFormation.starFormationTimeGyr": 0.01})
            return values

        def thermal_state(path):
            """Read the species and gas thermal arrays; the gas layout carries the method offset."""
            data = path.read_bytes()
            prefix = read_prefix(data)
            reader = prefix.reader
            integer_code = "q" if prefix.integer_bytes == 8 else "i"
            count = reader.take(integer_code)[0]
            gas = reader.take(integer_code)[0]
            reader.skip(3 * prefix.position_bytes * count + 13 * count)
            types = reader.take("b", count)
            reader.skip(prefix.integer_bytes * count)
            if reader.take("I", 7)[0] != 0x43545350 or reader.take(integer_code)[0] != gas:
                raise RuntimeError(f"unexpected gas layout in {path}")
            thermal = reader.take("f", 2 * gas)[::2]
            if gas != types.count(1) or any(not math.isfinite(u) or u <= 0.0 for u in thermal):
                raise RuntimeError(f"invalid gas thermal state in {path}")
            return {"gas": gas, "stars": types.count(2), "mean": math.fsum(thermal) / gas,
                    "physical": data[prefix.particle_start:], "prefix": data[:prefix.driver_offset]}

        def thermal_states(directory, steps):
            return [thermal_state(directory / f"output/checkpoint.{step}/checkpoint.0.dat")
                    for step in range(1, steps + 1)]

        def conversions(output):
            return [int(fields["stars"]) for fields in (
                dict(re.findall(r"([A-Za-z]+)=([^\s]+)", match.group(1)))
                for match in re.finditer(r"Hopkins step: ([^\n]+)", output))]

        def thermal_clock(directory):
            # Keep the step, redshift, scale, expansion rate, timestep, years and
            # rung columns; wall-clock columns differ between runs.
            rows = [line.split() for line in (directory / "output/progress.txt").read_text().splitlines()]
            return [row[1:9] for row in rows if row and row[0] != "runtime"]

        def check_thermal_restart(name, values, reference, steps):
            split_directory = prepare(name + "_split")
            launch(split_directory, dict(values, **{"run.maxIterations": 1}))
            directory = prepare(name + "_restart")
            (directory / "output").mkdir()
            shutil.copytree(split_directory / "output/checkpoint.1", directory / "output/checkpoint.1")
            output = launch(directory, dict(values, **{"run.maxIterations": steps - 1,
                                                       "checkpoint.readIndex": 1}))
            if "Done reading checkpoint" not in output:
                raise RuntimeError(f"{name} did not load its restart")
            resumed = thermal_state(directory / f"output/checkpoint.{steps}/checkpoint.0.dat")
            if (resumed["physical"], resumed["prefix"]) != (reference["physical"], reference["prefix"]):
                raise RuntimeError(f"{name} restart changed the particle, source or meshless state")
            if thermal_clock(directory)[-1] != thermal_clock(split_directory.parent / name)[-1]:
                raise RuntimeError(f"{name} restart changed the physical clock")
            print(f"{name}: exact restart with sources passed", flush=True)

        for integrator in integrators:
            means = {}
            for method in ("mfm", "sph") if integrator == "traditional" else ("mfm",):
                name = f"{method}_cooling_{integrator}"
                directory = prepare(name)
                output = launch(directory, thermal_options(method, integrator, 3, False))
                check_finite_rows(directory / "output/energy.txt")
                states = thermal_states(directory, 3)
                if any(state["stars"] != 0 or state["gas"] != states[0]["gas"] for state in states):
                    raise RuntimeError(f"{name} changed the gas population without star formation")
                means[method] = [state["mean"] for state in states]
                if method == "mfm" and any(record != 0 for record in conversions(output)):
                    raise RuntimeError(f"{name} reported conversions without star formation")
            # Stored a^2 u is constant under adiabatic expansion of the lattice;
            # only the source changes it.
            if abs(means["mfm"][-1] - means["mfm"][0]) <= 1.e-4 * means["mfm"][0]:
                raise RuntimeError(f"MFM {integrator} cooling left the thermal energy unchanged")
            if "sph" in means:
                for step, (mfm, sph) in enumerate(zip(means["mfm"], means["sph"]), start=1):
                    if abs(mfm - sph) > 1.e-3 * sph:
                        raise RuntimeError(f"MFM and SPH cooling disagree at step {step}: {mfm} vs {sph}")
            print(f"mfm_cooling_{integrator}: thermal energy {means['mfm'][0]:.6e} -> "
                  f"{means['mfm'][-1]:.6e}" + (" matches SPH" if "sph" in means else ""), flush=True)
            name = f"mfm_cooling_{integrator}"
            check_thermal_restart(name, thermal_options("mfm", integrator, 3, False),
                                  thermal_states(base / name, 3)[-1], 3)
            if not star_formation_available:
                continue
            steps = 4
            name = f"mfm_star_formation_{integrator}"
            directory = prepare(name)
            values = thermal_options("mfm", integrator, steps, True)
            output = launch(directory, values)
            check_finite_rows(directory / "output/energy.txt")
            formed = conversions(output)
            states = thermal_states(directory, steps)
            first = next((index for index, count in enumerate(formed) if count), None)
            if (first is None or first >= len(formed) - 1 or len(formed) < steps or
                    states[-1]["stars"] != sum(formed) or
                    any(state["gas"] + state["stars"] != states[0]["gas"] + states[0]["stars"]
                        for state in states) or
                    states[-1]["stars"] >= states[0]["gas"] // 2):
                raise RuntimeError(f"{name} conversions {formed} disagree with the stored species")
            if integrator == "traditional":
                reference = prepare(f"sph_star_formation_{integrator}")
                launch(reference, thermal_options("sph", integrator, steps, True))
                sph_states = thermal_states(reference, steps)
                for step, (mfm, sph) in enumerate(zip(states, sph_states), start=1):
                    if abs(mfm["stars"] - sph["stars"]) > max(3, sph["stars"] // 10):
                        raise RuntimeError(f"MFM formed {mfm['stars']} stars by step {step}; SPH formed {sph['stars']}")
            check_thermal_restart(name, values, states[-1], steps)
            print(f"{name}: {states[-1]['stars']} of {states[0]['gas']} cells converted over "
                  f"{steps} steps ({formed})", flush=True)
        print("Hopkins thermal-source regressions passed", flush=True)
        return

    if args.global_sources_only:
        endpoints = {}
        for method in ("mfm", "mfv"):
            name = f"{method}_global_uniform_force"
            values = dict(options("hydroWave", 0.5, method, 1), **{
                "run.timeIntegrator": "global", "run.numberOfSteps": 1,
                "hydro.cfl": 0.123, "hydro.yAcceleration": 1.25, "run.verbose": "true"})
            probe = prepare(name + "_timestep")
            launch(probe, values)
            first = particle_state(probe / "output/checkpoint.1", method)
            # Keep the CFL-selected first step and force a shorter non-dyadic
            # final interval, exercising the persisted previousTimestep source
            # correction rather than equal-duration special cases.
            first_timestep = first["clock"][1]
            if not 0 < first_timestep < values["run.endTime"]:
                raise RuntimeError(f"{name} did not select a raw source timestep")
            endpoint = 1.37 * first_timestep
            endpoints[method] = endpoint
            values.update({"run.endTime": endpoint, "run.maxIterations": 8})
            directory = prepare(name)
            output = launch(directory, values)
            destination = directory / "output"
            steps = sorted(int(path.name.split(".")[1]) for path in
                           destination.glob("checkpoint.*") if path.is_dir())
            final = particle_state(destination / f"checkpoint.{steps[-1]}", method)
            if (not 2 <= len(steps) <= 3 or final["clock"][0] != endpoint or
                    not 0 < final["clock"][1] < first_timestep or
                    any(float(record["time"]) >= endpoint and int(record["terminal"]) == 0
                        for record in (dict(re.findall(r"([A-Za-z]+)=([^\s]+)", match.group(1)))
                        for match in re.finditer(r"Hopkins step: ([^\n]+)", output)))):
                raise RuntimeError(f"{name} did not finish one unequal clipped source interval")
            check_uniform_acceleration(destination, final["count"] * final["fixed_mass"],
                                       1.25, name, method)
            check_restart(name, method, values, destination, final,
                          split_step=1, final_step=steps[-1])
            print(f"{name}: dt first={first_timestep:.12g}, final={final['clock'][1]:.12g}", flush=True)

        name = "mfv_global_central_force"
        directory = prepare(name)
        values = dict(options("hydroWave", endpoints["mfv"], "mfv", 8), **{
            "run.timeIntegrator": "global", "run.numberOfSteps": 1,
            "hydro.cfl": 0.123, "hydro.centralAcceleration": 0.05})
        launch(directory, values)
        destination = directory / "output"
        steps = sorted(int(path.name.split(".")[1]) for path in
                       destination.glob("checkpoint.*") if path.is_dir())
        if not 2 <= len(steps) <= 3:
            raise RuntimeError(f"{name} did not retain an interior restart point")
        final = particle_state(destination / f"checkpoint.{steps[-1]}", "mfv")
        if final["clock"][0] != values["run.endTime"]:
            raise RuntimeError(f"{name} missed its endpoint")
        split = check_restart(name, "mfv", values, destination, final,
                              split_step=1, final_step=steps[-1])
        work_signal = pending_gravity_work_signal(split)
        if work_signal <= 1.e-14:
            raise RuntimeError(f"{name} did not persist nonzero MFV source work")
        print(f"{name}: pending endpoint/live-pair source work {work_signal:.3e}", flush=True)
        print("Hopkins global source-law and forced-restart regressions passed", flush=True)
        return

    if args.matrix_only:
        # Self gravity changes fluid energy. The existing unforced fixtures
        # check all conserved fluid quantities; this matrix instead checks
        # mass conservation, finite physical evolution, solver identity, and
        # exact restarts with live gravity sources, including MFV mass transfer.
        solvers = [("fmm", None), ("pmmm", "pp"), ("pmmm", "fmm")]
        solvers = [(far, near) for far, near in solvers
                   if (args.gravity_method is None or far == args.gravity_method) and
                   (args.near_method is None or near == args.near_method)]
        if not solvers:
            parser.error("the requested far/near solver combination is empty")
        integrators = (args.time_integrator,) if args.time_integrator else ("traditional", "global")
        for far, near in solvers:
            for integrator in integrators:
                for method in ("mfm", "mfv"):
                    if integrator == "hierarchical" and method == "mfv":
                        continue
                    solver = far + ("_" + near if near else "")
                    name = f"{method}_{solver}_{integrator}"
                    directory = prepare(name)
                    values = dict(options("hydroWave", 0.2, method, 3), **{
                        "gravity.enabled": "true", "gravity.method.far": far,
                        "gravity.cuda": str(args.cuda).lower(), "drift.cuda": str(args.cuda).lower(),
                        "gravity.meshSize": 8, "gravity.pmmmNearWidth": 2,
                        "run.timeIntegrator": integrator, "run.numberOfSteps": 1,
                        "hydro.cfl": 0.123, "run.verbose": "true"})
                    if near:
                        values["gravity.method.near"] = near
                    output = launch(directory, values)
                    check_bool(output, "gravity.cuda", args.cuda)
                    check_bool(output, "drift.cuda", args.cuda)
                    check_bool(output, "hydro.cuda", False)
                    for key, expected in (("hydro.method", method),
                                          ("gravity.method.far", far),
                                          ("run.timeIntegrator", integrator)):
                        if setting(output, key) != expected:
                            raise RuntimeError(f"{name} selected the wrong {key}")
                    if near and setting(output, "gravity.method.near") != near:
                        raise RuntimeError(f"{name} selected the wrong near-field solver")
                    destination = directory / "output"
                    if args.cuda and far == "fmm":
                        profile = destination / "profile.txt"
                        if profile.exists():
                            kernels = re.search(r"^kick\.gpu\.kernel\s+(\d+)\s+",
                                                profile.read_text(), re.MULTILINE)
                            if kernels is None or int(kernels[1]) <= 0:
                                raise RuntimeError(f"{name} did not execute a CUDA gravity kernel")
                        elif "CUDA kick arena (" not in output:
                            # PROFILE=OFF builds retain the CUDA launch-arena
                            # message, so acceleration remains observable.
                            raise RuntimeError(f"{name} did not enter the CUDA gravity launch path")
                    for filename in ("energy.dat", "rho.dat", "progress.txt"):
                        check_finite_rows(destination / filename)
                    states = []
                    expected_algorithms = (1, {"fmm": 1, "pmmm": 2}[far],
                                           {None: 0, "pp": 1, "fmm": 2}[near], args.expansion_degree + 1,
                                           {"traditional": 0, "hierarchical": 1, "global": 2}[integrator])
                    for step in range(1, 4):
                        checkpoint = destination / f"checkpoint.{step}"
                        check_manifest(checkpoint, step, "hydroWave", values["run.endTime"],
                                       args.localities, gravity=True)
                        state = particle_state(checkpoint, method)
                        if state["algorithms"] != expected_algorithms:
                            raise RuntimeError(f"{name} checkpoint has wrong compiled solver identity")
                        time, timestep, phase, max_rung, iteration, driver_step = state["clock"]
                        if iteration != step or not 0 < time < values["run.endTime"]:
                            raise RuntimeError(f"{name} checkpoint has invalid physical clock")
                        if states and time <= states[-1]["clock"][0]:
                            raise RuntimeError(f"{name} did not advance its physical clock")
                        if integrator == "global":
                            previous_time = states[-1]["clock"][0] if states else 0.0
                            if (timestep <= 0 or phase != 0 or max_rung != 0 or driver_step != step or
                                    any(state["rungs"]) or any(state["kick_phases"]) or
                                    any(pair[2] or pair[3] or not math.isclose(pair[5], time,
                                        rel_tol=1.e-14, abs_tol=1.e-15) for pair in state["pair_cache"]) or
                                    not math.isclose(time - previous_time, timestep, rel_tol=1.e-14)):
                                raise RuntimeError(f"{name} retained block clocks or an incomplete pair interval")
                        initial_mass = state["fixed_mass"] * state["count"]
                        if abs(state["mass"] - initial_mass) > 1.e-10 * initial_mass:
                            raise RuntimeError(f"{name} changed total conservative mass")
                        if abs(state["primitive_mass"] - initial_mass) > 5.e-6 * initial_mass:
                            raise RuntimeError(f"{name} float mass projection lost total mass")
                        states.append(state)
                    if method == "mfv" and len(set(states[-1]["masses"])) == 1:
                        raise RuntimeError(f"{name} did not exercise MFV mass exchange")
                    if integrator == "global":
                        exponent = math.log2(states[0]["clock"][1] / values["run.endTime"])
                        if abs(exponent - round(exponent)) < 1.e-5:
                            raise RuntimeError(f"{name} rounded its global interval to a block rung")
                        records = [dict(re.findall(r"([A-Za-z]+)=([^\s]+)", match.group(1)))
                                   for match in re.finditer(r"Hopkins step: ([^\n]+)", output)]
                        if len(records) != 3 or any(int(record["active"]) != int(record["total"]) or
                                int(record["minRung"]) != 0 or int(record["maxRung"]) != 0 or
                                int(record["cached"]) != 0 for record in records):
                            raise RuntimeError(f"{name} did not evaluate every gas particle at every global step")
                    check_restart(name, method, values, destination, states[-1],
                                  split_step=1, final_step=3)
                    mass_error = abs(states[-1]["mass"] - states[0]["mass"]) / states[0]["mass"]
                    print(f"{name}: live gravity, mass error {mass_error:.3e}, "
                          f"t={states[-1]['clock'][0]:.12g}", flush=True)
                    if integrator == "global":
                        endpoint = states[0]["clock"][1] * 1.37
                        endpoint_directory = prepare(name + "_endpoint")
                        endpoint_output = launch(endpoint_directory, dict(values, **{
                            "run.endTime": endpoint, "run.maxIterations": 8}))
                        checkpoints = sorted(int(path.name.split(".")[1]) for path in
                            (endpoint_directory / "output").glob("checkpoint.*") if path.is_dir())
                        ended = particle_state(endpoint_directory / "output" /
                                               f"checkpoint.{checkpoints[-1]}", method)
                        if ended["clock"][0] != endpoint or len(checkpoints) > 3:
                            raise RuntimeError(f"{name} did not stop at the exact clipped endpoint")
                        if any(float(record["time"]) >= endpoint and int(record["terminal"]) == 0
                                for record in (dict(re.findall(r"([A-Za-z]+)=([^\s]+)", match.group(1)))
                                for match in re.finditer(r"Hopkins step: ([^\n]+)", endpoint_output))):
                            raise RuntimeError(f"{name} opened an outgoing interval at the endpoint")
                        print(f"{name}: exact non-dyadic endpoint {endpoint:.12g}", flush=True)
        print("Hopkins gravity/integrator matrix regressions passed", flush=True)
        return

    for method in ("mfm", "mfv"):
        name = f"{method}_wake_rungs"
        directory = prepare(name)
        values = dict(options("blast", 0.004, method, 16), **{"run.numberOfSteps": 2})
        output = launch(directory, values)
        destination = directory / "output"
        states = checkpoint_states(destination, method, "blast", values["run.endTime"],
                                   16, name, number_of_steps=2)
        if states[-1]["tick"] != 2 * 32768 or not math.isclose(
                states[-1]["clock"][0], values["run.endTime"], rel_tol=1.e-13):
            raise RuntimeError(f"{name} did not complete two coarse steps")
        check_activity(output, states, name, args.localities)
        split_step = check_wake_transactions(output, states, name)
        check_restart(name, method, values, destination, states[-1], split_step=split_step,
                      final_step=16, require_pending_pair=True)
    if args.wake_only:
        print("Hopkins real-wake regressions passed", flush=True)
        return

    for method in ("mfm", "mfv"):
        name = f"{method}_individual_rungs"
        directory = prepare(name)
        values = dict(options("blast", 0.004, method, 8), **{"run.verbose": "true"})
        output = launch(directory, values)
        destination = directory / "output"
        states = checkpoint_states(destination, method, "blast", values["run.endTime"], 8, name)
        check_activity(output, states, name, args.localities)
        check_restart(name, method, values, destination, states[-1], split_step=1, final_step=8,
                      require_pending_pair=True)
    if args.async_only:
        print("Hopkins individual-rung regressions passed", flush=True)
        return

    references = {}
    for method in ("mfm", "mfv"):
        for scenario, duration in CASES:
            name = f"{method}_{scenario}"
            directory = prepare(name)
            values = options(scenario, duration, method, 4)
            output = launch(directory, values)
            if setting(output, "hydro.method") != method:
                raise RuntimeError(f"{name} selected the wrong solver")
            destination = directory / "output"
            for filename in ("energy.dat", "rho.dat", "progress.txt"):
                check_finite_rows(destination / filename)
            states = checkpoint_states(destination, method, scenario, duration, 4, name)
            initial_mass = states[0]["fixed_mass"] * states[0]["count"]
            if abs(states[-1]["mass"] - initial_mass) > 5.e-6 * initial_mass:
                raise RuntimeError(f"{name} changed mass on its initial step")
            if method == "mfm" and len(set(states[-1]["masses"])) != 1:
                raise RuntimeError(f"{name} did not retain particle masses")
            if scenario == "hydroWave":
                references[method] = (destination, values, states[-1])
                if method == "mfv" and len(set(states[-1]["masses"])) == 1:
                    raise RuntimeError("MFV displaced hydro wave did not exchange particle mass")

    for method, (reference_output, values, reference_state) in references.items():
        check_restart(method, method, values, reference_output, reference_state, True)

    for method in ("mfm", "mfv"):
        name = f"{method}_uniform_force"
        directory = prepare(name)
        values = dict(options("hydroWave", 0.04, method, 4), **{"hydro.yAcceleration": 1.25})
        launch(directory, values)
        destination = directory / "output"
        reference = particle_state(destination / "checkpoint.4", method)
        check_uniform_acceleration(destination, reference["count"] * reference["fixed_mass"],
                                   1.25, name, method)
        check_restart(name, method, values, destination, reference)

    # hydroWave exchanges mass primarily along x; the uniform y force above is
    # transverse to that exchange. A weak central field additionally exercises
    # nonzero endpoint work when MFV mass crosses a potential gradient.
    name = "mfv_central_force"
    directory = prepare(name)
    values = dict(options("hydroWave", 0.04, "mfv", 4), **{"hydro.centralAcceleration": 0.05})
    launch(directory, values)
    destination = directory / "output"
    for filename in ("energy.dat", "rho.dat", "progress.txt"):
        check_finite_rows(destination / filename)
    reference = particle_state(destination / "checkpoint.4", "mfv")
    split = check_restart(name, "mfv", values, destination, reference)
    # Work remains in live pair intervals until settlement; endpoint-only
    # pending moments can therefore be zero at a valid interior restart.
    work_signal = pending_gravity_work_signal(split)
    if work_signal <= 1.e-14:
        raise RuntimeError("MFV forced restart did not exercise pending gravitational work")
    print(f"{name}: pending endpoint/live-pair source work {work_signal:.3e}", flush=True)

    sph_payloads = []
    for method in (None, "sph"):
        directory = prepare("sph_default" if method is None else "sph_explicit")
        launch(directory, options("hydroWave", 0.00001, method, 2))
        sph_payloads.append(particle_state(directory / "output/checkpoint.2", "sph")["payload"])
    if sph_payloads[0] != sph_payloads[1]:
        raise RuntimeError("explicit hydro.method=sph changed default SPH evolution")
    print("Hopkins integration regressions passed; default and explicit SPH agree", flush=True)


if __name__ == "__main__":
    main()
