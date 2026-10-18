"""Production primordial/SPH startup, restart, metadata and backend regression."""
import argparse
import ctypes
from ctypes.util import find_library
from pathlib import Path
import shutil
import struct
import sys

from run_checkpoint import prepareCase, runSimulation, checkManifest


def gasPayload(path):
    payload = (path / "checkpoint.0.dat").read_bytes()
    prefix = struct.pack("=2I", 0x43545350, 3)
    offset = payload.rfind(prefix)
    if offset < 0:
        raise RuntimeError("No versioned chemistry gas payload in checkpoint")
    fields = struct.unpack_from("=7I", payload, offset)
    count = int.from_bytes(payload[offset + 28:offset + 28 + fields[2]], sys.byteorder, signed=True)
    if count != 512:
        raise RuntimeError(f"Unexpected gas count {count}")
    # The tail is explicitly serialized: six signed fractions, eV/H energy,
    # pressure/energy ratio and frozen gamma for each gas particle.
    chemistryStart = offset + 28 + fields[2] + count * (58 + fields[2])
    chemistryEnd = chemistryStart + count * 9 * 4
    chemistry = payload[chemistryStart:chemistryEnd]
    values = struct.unpack(f"={count * 9}f", chemistry)
    for i in range(count):
        words = values[9 * i:9 * i + 9]
        if not (all(abs(x) <= .5 for x in words[:6]) and words[6] > 0 and words[7] > 0 and words[8] > 1):
            raise RuntimeError(f"Invalid chemistry checkpoint state {words}")
    return payload[offset:chemistryEnd]



def checkOutput(directory):
    library = find_library("siloh5") or find_library("silo")
    if not library:
        raise RuntimeError("Cannot locate the Silo library for output verification")
    # Some distribution Silo builds rely on the C++ executable to supply this.
    ctypes.CDLL(find_library("stdc++"), mode=ctypes.RTLD_GLOBAL)
    silo = ctypes.CDLL(library)
    silo.DBOpenReal.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    silo.DBOpenReal.restype = ctypes.c_void_p
    silo.DBInqVarExists.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    silo.DBInqVarExists.restype = ctypes.c_int
    silo.DBClose.argtypes = [ctypes.c_void_p]
    files = list((directory / "output").glob("*.silo"))
    if not files:
        raise RuntimeError("Chemistry run produced no requested Silo output")
    for filename in files:
        handle = silo.DBOpenReal(str(filename).encode(), 5, 1)  # DB_UNKNOWN, DB_READ
        if not handle:
            raise RuntimeError(f"Cannot read {filename}")
        try:
            for system in ("cgs", "cosmological"):
                for frame in ("comoving", "lab"):
                    for field in ("chem_HI", "chem_H2", "chem_electrons", "chem_signed_fraction_0",
                                  "temperature", "mean_molecular_weight", "frozen_gamma"):
                        name = f"{system}/{frame}/{field}".encode()
                        if silo.DBInqVarExists(handle, name) != 1:
                            raise RuntimeError(f"Missing chemistry output field {name!r}")
        finally:
            silo.DBClose(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("power_file", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    executable = args.executable.resolve()
    recfast = executable.parent / "recfast"
    base = args.workdir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    options = [str(executable), "--hpx:threads=2", "--hpx:bind=none",
               "--initial.particlesPerDimension=8", "--hydro.enabled=true",
               f"--compute.cuda={'true' if args.cuda else 'false'}",
               "--analysis.views=false", "--lightcone.enabled=false", "--analysis.power=false",
               "--analysis.groups=false", "--chemistry.enabled=true",
               "--chemistry.synchronizedRung=8", "--checkpoint.intervalSeconds=0"]
    for network in ("atomic6", "molecular9"):
        baseline = prepareCase(base, f"{network}_baseline", args.power_file.resolve(), recfast)
        table = "COSMICTIGER_PRIMORDIAL_RADIATION 1\n" + "\n".join(
            str(z) + " " + " ".join(["0"] * 14) for z in (0, 100)) + "\n"
        (baseline / "radiation.txt").write_text(table)
        command = options + [f"--chemistry.network={network}", "--chemistry.radiationFile=radiation.txt",
                             "--output.silo=true", "--output.units=both", "--output.frame=both"]
        if network == "molecular9":
            command += ["--chemistry.molecularFraction=0.2"]
        runSimulation(command + ["--run.maxIterations=1"], baseline, "run.log")
        checkpoint1 = baseline / "checkpoint.1"
        checkpoint2 = baseline / "checkpoint.2"
        checkManifest(checkpoint1, 1)
        checkManifest(checkpoint2, 2)
        uninterrupted = gasPayload(checkpoint2)
        checkOutput(baseline)
        restarted = prepareCase(base, f"{network}_restart", args.power_file.resolve(), recfast, checkpoint1)
        (restarted / "radiation.txt").write_text(table)
        runSimulation(command + ["--checkpoint.readIndex=1", "--run.maxIterations=0"], restarted, "restart.log")
        restored = gasPayload(restarted / "checkpoint.2")
        if uninterrupted != restored:
            raise RuntimeError(f"{network}: restart changed the gas payload (source phase or state mismatch)")
        altered = prepareCase(base, f"{network}_radiation_changed", args.power_file.resolve(), recfast, checkpoint1)
        (altered / "radiation.txt").write_text(table + "# changed radiation provenance\n")
        output = runSimulation(command + ["--checkpoint.readIndex=1", "--run.maxIterations=0"],
                               altered, "rejected.log", expectSuccess=False)
        if "radiation" not in output.lower():
            raise RuntimeError("Changed radiation table lacked compatibility diagnostic")
        incompatible = prepareCase(base, f"{network}_incompatible", args.power_file.resolve(), recfast, checkpoint1)
        output = runSimulation(options + [f"--chemistry.network={'molecular9' if network == 'atomic6' else 'atomic6'}",
            "--checkpoint.readIndex=1", "--run.maxIterations=0"], incompatible, "rejected.log", expectSuccess=False)
        if "chemistry" not in output.lower():
            raise RuntimeError("Incompatible restart lacked chemistry diagnostic")
        print(f"{network}: production {'CUDA' if args.cuda else 'CPU'} stepping, exact gas restart, incompatible-network rejection passed")
    # No silent extrapolation of the selected molecular cooling fits.
    invalid = prepareCase(base, "invalid_domain", args.power_file.resolve(), recfast)
    output = runSimulation(options + ["--chemistry.network=molecular9", "--chemistry.initialTemperature=10000",
        "--run.maxIterations=0"], invalid, "rejected.log", expectSuccess=False)
    if "chemistry" not in output.lower():
        raise RuntimeError("Unsupported domain lacked chemistry diagnostic")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
