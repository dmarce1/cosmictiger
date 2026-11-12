# SPDX-License-Identifier: BSL-1.0
"""Exercise scheduled PMMM boundaries and restart of their frozen force split."""

import argparse
from pathlib import Path
import re
import struct
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint_format import decode_checkpoint, require
from run_checkpoint import prepareCase, runSimulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--mpi-exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--disable-tcp", action="store_true")
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ranks = 2 if args.mpi_exec else 1
    # A 4^3 lattice retains the occupied/empty-cell and near/far schedule cases
    # without spending this control-flow regression on thousands of Ewald M2Ls.
    common = {
        "hpx:threads": 1 if ranks > 1 else 2, "hpx:bind": "none",
        "run.scenario": "cosmology", "run.darkMatterParticleCount": 64,
        "run.gasParticleCount": 64, "run.numberOfOutputs": 3, "run.minRung": 2,
        "run.maxIterations": 400, "run.verbose": "true", "units.boxLength": 20,
        "cosmology.startRedshift": 1, "cosmology.endRedshift": 0.8,
        "cosmology.sigma8": 0, "scenario.cosmology.useGlass": "false",
        "initial.closePack": "false", "gravity.meshSize": 8,
        "gravity.openingSchedule": "1:0.45,0.99:0.6,0.93:0.5",
        "gravity.smoothingLength": 0.01,
        "gravity.cuda": "false", "drift.cuda": "false", "hydro.cuda": "false",
        "hydro.enabled": "false", "chemistry.enabled": "false",
        "starFormation.enabled": "false", "analysis.groups": "false",
        "analysis.tracers": "false", "analysis.views": "false", "analysis.slice": "false",
        "analysis.power": "false", "lightcone.enabled": "false", "output.silo": "false",
        "output.directory": ".", "checkpoint.intervalSeconds": 0,
    }

    def run(directory, overrides=None, success=True):
        options = dict(common, **(overrides or {}))
        command = [str(executable)] + [f"--{key}={value}" for key, value in options.items()]
        if args.mpi_exec:
            command = ([args.mpi_exec, args.mpi_numproc_flag, str(ranks)] + args.mpi_preflag
                       + command[:1] + args.mpi_postflag + command[1:] + [
                           f"--hpx:localities={ranks}", "--hpx:ini=hpx.parcel.mpi.enable=1"])
            if args.disable_tcp:
                command.append("--hpx:ini=hpx.parcel.tcp.enable=0")
        return runSimulation(command, directory, "run.log", expectSuccess=success)

    def read(path):
        decoded = [decode_checkpoint((path / f"checkpoint.{rank}.dat").read_bytes(), ranks=ranks)
                   for rank in range(ranks)]
        angles = [struct.unpack("=d", struct.pack("=Q", record.pmmm[9]))[0]
                  for record in decoded]
        require(len(set(angles)) == 1, "ranks restored different active angles")
        return decoded, angles[0]

    baseline = prepareCase(root, "baseline")
    log = run(baseline)
    require("reached the requested end time" in log, "scheduled run did not finish")
    paths = sorted((path for path in baseline.glob("checkpoint.*")
                    if path.is_dir() and re.fullmatch(r"checkpoint\.[0-9]+", path.name)),
                   key=lambda path: int(path.name.split(".")[1]))
    records = [(path, *read(path)) for path in paths]
    require({angle for _, _, angle in records} == {0.45, 0.6, 0.5},
            "schedule did not apply all angles")
    require(len(re.findall(r"^Opening schedule at z=", log, re.MULTILINE)) == 2,
            "schedule changed outside its two threshold crossings")
    deferred = False
    previous = None
    for path, decoded, angle in records:
        state = decoded[0].driver
        if previous is not None:
            old_state, old_angle = previous
            if state.outer_start == old_state.outer_start:
                require(angle == old_angle, "angle changed within a frozen interval")
            if angle != old_angle:
                require(old_state.integer_time == 0, "angle changed before closing the old interval")
        redshift = 1 / state.a - 1
        requested = 0.5 if redshift <= 0.93 else 0.6 if redshift <= 0.99 else 0.45
        deferred |= state.integer_time != 0 and angle != requested
        previous = state, angle
    require(deferred, "fixture did not cross a threshold inside a frozen interval")
    # Restart both an interior state with the changed angle and a pending boundary.
    candidates = [next((path for path, decoded, angle in records
                        if angle != 0.45 and (decoded[0].driver.integer_time != 0) == inner), None)
                  for inner in (True, False)]
    require(all(candidates), "fixture lacks changed-angle interior/boundary checkpoints")
    candidates.append(next(path for path, decoded, angle in records
                           if decoded[0].driver.integer_time != 0 and angle == 0.45
                           and 1 / decoded[0].driver.a - 1 < 0.99))
    final_path, final_records, _ = records[-1]
    for index, checkpoint in enumerate(candidates):
        restarted = prepareCase(root, f"restart_{index}", checkpoint)
        run(restarted, {"checkpoint.readIndex": checkpoint.name.split(".")[1]})
        resumed, angle = read(restarted / final_path.name)
        require(angle == records[-1][2], "restart lost the active opening angle")
        for reference, actual in zip(final_records, resumed):
            require((reference.physical, reference.rung_stack, reference.domains, reference.local_domains) ==
                    (actual.physical, actual.rung_stack, actual.domains, actual.local_domains),
                    "restart changed particle or domain state")
            require(reference.driver.tau == actual.driver.tau, "restart changed the final clock")
    for name, schedule in (("changed", "1:0.45,0.99:0.61,0.93:0.5"),
                           ("removed", "none")):
        directory = prepareCase(root, name, candidates[0])
        rejected = run(directory, {"checkpoint.readIndex": candidates[0].name.split(".")[1],
                                   "gravity.openingSchedule": schedule}, success=False)
        require("opening schedule" in rejected, "incompatible restart lacked schedule diagnostic")
    for index, extra in enumerate((
            {"gravity.openingSchedule": "1:1.1"},
            {"gravity.openingSchedule": "1:0.5,1:0.6"},
            {"gravity.openingSchedule": "1:nan"},
            {"run.scenario": "plummer"},
            {"gravity.openingSchedule": "1:0.45,0.99:1", "gravity.smoothingLength": 0.75})):
        directory = prepareCase(root, f"invalid_{index}")
        rejected = run(directory, extra, success=False)
        require("openingSchedule" in rejected, "invalid schedule lacked option diagnostic")
    # INI input and CLI clearing both use the public option parser.
    directory = prepareCase(root, "ini")
    config = directory / "schedule.ini"
    config.write_text("[gravity]\nopeningSchedule = 1:0.45,0.99:0.6,0.93:0.5\n")
    saved = common.pop("gravity.openingSchedule")
    run(directory, {"configuration": config, "run.maxIterations": 0})
    directory = prepareCase(root, "cleared")
    run(directory, {"configuration": config, "gravity.openingSchedule": "none",
                    "run.maxIterations": 0})
    common["gravity.openingSchedule"] = saved
    # A flat recollapsing matter/Lambda background crosses both thresholds twice.
    contraction_options = {
        "cosmology.startRedshift": 0, "cosmology.omegaB": 0.05,
        "cosmology.omegaC": 1.95, "cosmology.omegaLambda": "auto",
        "cosmology.thetaCmb": 0.0001, "run.endTime": 140,
        "run.maxIterations": 600,
        "gravity.openingSchedule": "0:0.45,-0.05:0.6,-0.15:0.5",
    }
    directory = prepareCase(root, "contraction")
    log = run(directory, contraction_options)
    require("reached the requested end time" in log, "contracting run did not finish")
    changes = [float(value) for value in re.findall(
        r"^Opening schedule at z=.* -> (\S+)$", log, re.MULTILINE)]
    require(changes == [0.6, 0.5, 0.6, 0.45], "contraction did not reverse the schedule")
    paths = sorted((path for path in directory.glob("checkpoint.*")
                    if path.is_dir() and re.fullmatch(r"checkpoint\.[0-9]+", path.name)),
                   key=lambda path: int(path.name.split(".")[1]))
    checkpoint = next(path for path in paths if read(path)[0][0].driver.adot < 0
                      and read(path)[1] == 0.6)
    restarted = prepareCase(root, "contraction_restart", checkpoint)
    run(restarted, dict(contraction_options, **{"checkpoint.readIndex": checkpoint.name.split(".")[1]}))
    reference, _ = read(paths[-1])
    actual, angle = read(restarted / paths[-1].name)
    require(angle == 0.45, "contracting restart lost reverse traversal")
    for before, after in zip(reference, actual):
        require((before.physical, before.rung_stack, before.domains, before.local_domains) ==
                (after.physical, after.rung_stack, after.domains, after.local_domains),
                "contracting restart changed particle or domain state")
    print("Opening schedule: expansion, contraction, deferred boundaries, restart identity, and validation passed")


if __name__ == "__main__":
    main()
