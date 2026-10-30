"""Check raw global steps, synchronized ranks, endpoint clipping, and exact restarts."""

import argparse
import ctypes
import math
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "gravity/nitadori"))

from application_selection import application_arguments, compiled_algorithms
from pmmm_evolution_helpers import Header, Driver
from run_checkpoint import fnv1a64, hydro_particle_payload, prepareCase, runSimulation


def checkpoint_state(directory, iteration, rank=0, integer_bytes=4):
    path = directory / f"checkpoint.{iteration}" / f"checkpoint.{rank}.dat"
    data = path.read_bytes()
    header = Header.from_buffer_copy(data)
    assert header.flags & (1 << 18) and header.flags & (1 << 19), path
    offset = ctypes.sizeof(Header)
    algorithms = struct.unpack_from("=5Q", data, offset)
    assert algorithms[0] == 1 and algorithms[4] == 2, algorithms
    offset += 40
    if header.flags & (1 << 17):
        version = struct.unpack_from("=Q", data, offset)[0]
        offset += 8 * {1: 7, 2: 8, 3: 10, 4: 11}[version]
    hydro = bool(header.flags & 1)
    if hydro:
        offset += 104
        if header.flags & (1 << 14):
            offset += 56
    driver = Driver.from_buffer_copy(data, offset)
    offset += ctypes.sizeof(Driver)
    driver.global_extension_offset = offset
    driver.integer_phase_offset = offset - ctypes.sizeof(Driver) + Driver.integer_time.offset
    version, timestep = struct.unpack_from("=Qd", data, offset)
    offset += 16
    assert version == 1 and math.isfinite(timestep) and timestep > 0, path
    assert driver.integer_time == 0 and driver.max_rung == 0, path
    assert driver.iteration == iteration and driver.step == iteration, path
    count = int.from_bytes(data[offset:offset + integer_bytes], sys.byteorder, signed=True)
    assert 0 <= count <= 2 * header.particles_per_species, path
    particle_start = offset
    offset += integer_bytes * (2 if hydro else 1)
    position_bytes = 8 if header.flags & (1 << 15) else 4
    offset += count * (3 * position_bytes + 12)
    assert not any(data[offset:offset + count]), "global particles retained block rungs"
    physical = (hydro_particle_payload(path.parent) if hydro and rank == 0 else
                data[particle_start:offset + count])
    return driver, timestep, physical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--without-sph", action="store_true")
    parser.add_argument("--mpi-exec")
    parser.add_argument("--particle-integer-bytes", type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    available = compiled_algorithms(str(executable))
    assert "global" in available["integrator"], "global integrator is not compiled"
    common = {"hpx:threads": 2, "hpx:bind": "none", "compute.cuda": "false",
              "initial.particlesPerDimension": 8, "initial.useGlass": "false",
              "initial.closePack": "false", "chemistry.enabled": "false",
              "analysis.groups": "false", "analysis.views": "false",
              "analysis.tracers": "false", "analysis.power": "false",
              "lightcone.enabled": "false", "output.silo": "false",
              "output.directory": ".", "checkpoint.intervalSeconds": 0,
              "run.numberOfSteps": 1, "run.minRung": 20}
    cases = []
    if not args.without_sph:
        cases.append(("hydro", False, True, None, None))
    for far in (name for name in available["far"] if name != "none"):
        for near in (available["near"] if far == "pmmm" else [None]):
            cases.append((far + ("_" + near if near else ""), True, False, far, near))
            if not args.without_sph:
                cases.append((far + ("_" + near if near else "") + "_hydro",
                              True, True, far, near))

    def run(directory, options, selections, restart=None, mpi=False, success=True):
        command = [str(executable)] + selections
        command += [f"--{key}={value}" for key, value in options.items()]
        if restart is not None:
            command.append(f"--checkpoint.readIndex={restart}")
        if mpi:
            command = [args.mpi_exec, "-n", "2"] + command
            command += ["--hpx:localities=2", "--hpx:ini=hpx.parcel.mpi.enable=1"]
        return runSimulation(command, directory, "run.log", expectSuccess=success)

    def read_state(directory, iteration, rank=0):
        return checkpoint_state(directory, iteration, rank, args.particle_integer_bytes)

    for label, gravity, hydro, far, near in cases:
        selections = application_arguments(executable, far=far, near=near,
                                           integrator="global", hydro=hydro)
        options = dict(common, **{"run.scenario": "hydroWave" if hydro else "plummer",
                                  "gravity.enabled": str(gravity).lower(),
                                  "hydro.enabled": str(hydro).lower(), "hydro.cfl": 0.123,
                                  "gravity.meshSize": 8, "run.endTime": 0.5,
                                  "run.maxIterations": 2})
        baseline = prepareCase(root, label)
        run(baseline, options, selections)
        first, timestep, _ = read_state(baseline, 1)
        final, _, physical = read_state(baseline, 2)
        assert 0 < timestep < 0.5 and math.isclose(first.tau, timestep, rel_tol=1e-14)
        exponent = math.log2(timestep / 0.5)
        assert abs(exponent - round(exponent)) > 1e-5, f"{label}: dyadic timestep {timestep}"
        resumed = prepareCase(root, label + "_restart", baseline / "checkpoint.1")
        run(resumed, dict(options, **{"run.maxIterations": 1}), selections, restart=1)
        continued, next_timestep, continued_physical = read_state(resumed, 2)
        assert continued.tau == final.tau and continued_physical == physical, label
        assert math.isclose(continued.tau - first.tau, next_timestep, rel_tol=1e-14)

        # A short run must land on the exact requested endpoint, including a
        # non-dyadic final interval, without leaving a tiny rounding remainder.
        endpoint = timestep * 1.37
        clipped = prepareCase(root, label + "_endpoint")
        run(clipped, dict(options, **{"run.endTime": endpoint,
                                     "run.maxIterations": 8}), selections)
        checkpoints = sorted(int(path.name.split(".")[1]) for path in clipped.glob("checkpoint.*")
                             if path.is_dir())
        ended, last_dt, _ = read_state(clipped, checkpoints[-1])
        assert ended.tau == endpoint and last_dt <= timestep * 1.01, label
        assert len(checkpoints) <= 3, f"{label}: extra tiny terminal intervals"

        if label == cases[0][0]:
            # Rehash each damaged file so validation reaches the clock reader,
            # independently of the existing checksum-corruption regression.
            mutations = (("zero_dt", first.global_extension_offset + 8, "d", 0.0),
                         ("nan_dt", first.global_extension_offset + 8, "d", math.nan),
                         ("integer_phase", first.integer_phase_offset, "Q", 1))
            for suffix, offset, encoding, value in mutations:
                rejected = prepareCase(root, label + "_" + suffix, baseline / "checkpoint.1")
                directory = rejected / "checkpoint.1"
                path = directory / "checkpoint.0.dat"
                data = bytearray(path.read_bytes())
                struct.pack_into("=" + encoding, data, offset, value)
                path.write_bytes(data)
                manifest = directory / "checkpoint.meta"
                lines = manifest.read_text().splitlines()
                lines[1] = f"0 {len(data)} {fnv1a64(data):016x}"
                manifest.write_text("\n".join(lines) + "\n")
                output = run(rejected, dict(options, **{"run.maxIterations": 0}),
                             selections, restart=1, success=False)
                assert "Invalid global timestep checkpoint extension" in output, suffix
            print("global checkpoint clock: zero/NaN pending intervals and integer phase rejected")

        if args.mpi_exec and label == "hydro":
            distributed = prepareCase(root, label + "_mpi")
            run(distributed, dict(options, **{"run.maxIterations": 1}), selections, mpi=True)
            states = [read_state(distributed, 1, rank) for rank in range(2)]
            assert states[0][1] == states[1][1], "ranks selected different timesteps"
            assert math.isclose(states[0][1], timestep, rel_tol=2e-5), \
                "distributed minimum differs from serial minimum"
        print(f"{label}: raw dt={timestep:.12g}, exact restart and endpoint passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
