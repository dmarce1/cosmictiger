"""Production primordial/SPH startup, restart, metadata and backend regression."""
import argparse
import csv
import ctypes
from ctypes.util import find_library
import io
import math
from pathlib import Path
import struct
import subprocess
import sys

from run_checkpoint import prepareCase, runSimulation, checkManifest
from application_selection import application_arguments
from gravity.nitadori.pmmm_evolution_helpers import Header, Driver


def globalState(path):
    payload = (path / "checkpoint.0.dat").read_bytes()
    header = Header.from_buffer_copy(payload)
    if (header.magic != 0x4354535048303031 or header.version != 3 or
            not header.flags & (1 << 19) or header.flags & (1 << 17)):
        raise RuntimeError("Expected an FMM global checkpoint")
    offset = ctypes.sizeof(Header)
    if header.flags & (1 << 18):
        algorithms = struct.unpack_from("=5Q", payload, offset)
        if algorithms[0] != 1 or algorithms[1] != 1 or algorithms[4] != 2:
            raise RuntimeError("Invalid global algorithm checkpoint extension")
        offset += 40
    if not header.flags & 1 or not header.flags & (1 << 11):
        raise RuntimeError("Expected SPH chemistry checkpoint extensions")
    offset += 104  # Historical thirteen-double SPH block.
    if header.flags & (1 << 12):
        offset += 8
    if header.flags & (1 << 14):
        offset += 56
    offset += 17 * 8 + 8 + 24  # Chemistry parameters, radiation checksum, CKDKC phase.
    driver = Driver.from_buffer_copy(payload, offset)
    version, timestep = struct.unpack_from("=Qd", payload, offset + ctypes.sizeof(Driver))
    if (version != 1 or driver.integer_time != 0 or not math.isfinite(timestep) or
            timestep < 0 or (driver.iteration > 0 and timestep == 0)):
        raise RuntimeError("Invalid raw global checkpoint phase")
    return driver, timestep


def gasState(path, requireEos=True, globalTimestep=None):
    payload = (path / "checkpoint.0.dat").read_bytes()
    prefix = struct.pack("=I", 0x43545350)
    offset = payload.rfind(prefix)
    if offset < 0:
        raise RuntimeError("No versioned chemistry gas payload in checkpoint")
    fields = struct.unpack_from("=7I", payload, offset)
    if fields[1] not in (11, 13, 15, 17) or fields[2] not in (4, 8) or fields[5:] != (3, 4):
        raise RuntimeError(f"Unsupported chemistry checkpoint layout {fields}")
    count = int.from_bytes(payload[offset + 28:offset + 28 + fields[2]], sys.byteorder, signed=True)
    if count != 512:
        raise RuntimeError(f"Unexpected gas count {count}")
    # The tail is explicitly serialized: six signed fractions, eV/H energy,
    # pressure/energy ratio and frozen gamma, then each particle's pending
    # conformal half-interval and opening heating ledger (two doubles).
    viscosityBytes = {11: 0, 13: 8, 15: 12, 17: 12}[fields[1]]
    chemistryStart = offset + 28 + fields[2] + count * (
        fields[3] + fields[4] + 38 + fields[2] + viscosityBytes)
    chemistryEnd = chemistryStart + count * (9 * 4 + 2 * 8)
    if chemistryEnd > len(payload):
        raise RuntimeError("Truncated chemistry checkpoint state")
    chemistry = payload[chemistryStart:chemistryEnd]
    records = list(struct.iter_unpack("=9f2d", chemistry))
    for i, record in enumerate(records):
        words = record[:9]
        halfInterval, pendingHeating = record[9:]
        if not (math.isfinite(halfInterval) and halfInterval >= 0 and
                math.isfinite(pendingHeating)):
            raise RuntimeError(f"Invalid per-particle chemistry phase {record[9:]}")
        if requireEos and halfInterval <= 0:
            raise RuntimeError("Evolved chemistry checkpoint lost its pending source half")
        if globalTimestep is not None and halfInterval != 0.5 * globalTimestep:
            raise RuntimeError("Chemistry half-interval differs from the exact global timestep")
        if not (all(math.isfinite(x) for x in words) and
                all(abs(x) <= .5 for x in words[:6]) and words[6] > 0):
            raise RuntimeError(f"Invalid chemistry checkpoint state {words}")
        if requireEos and not (words[7] > 0 and words[8] > 1):
            raise RuntimeError(f"Invalid evolved chemistry EOS state {words}")
    return [record[:9] for record in records], payload[offset:chemistryEnd]


def gasPayload(path, globalMode=False):
    timestep = globalState(path)[1] if globalMode else None
    return gasState(path, globalTimestep=timestep)[1]


def nativeInitialReference(executable, directory, redshift):
    redshifts = directory / "initial_redshift.txt"
    redshifts.write_text(f"{redshift:.17g}\n")
    result = subprocess.run([str(executable.resolve()), "3.046", "0.005",
                             str(redshifts), "1e-9"], text=True, capture_output=True,
                            timeout=30, check=True)
    (directory / "native_initial_reference.csv").write_text(result.stdout)
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    if len(rows) != 1 or float(rows[0]["z"]) != redshift:
        raise RuntimeError("Native initialization reference did not return the requested epoch")
    return {key: float(value) for key, value in rows[0].items()}


def checkInitialChemistry(checkpoint, reference):
    # SPH retains its existing mass convention and monatomic ideal-gas EOS.
    # Decode signed minority abundances and infer temperature from the saved
    # eV/H energy, independently of option printing and Silo derived fields.
    helium = 0.2454006 / (4.0 * (1.0 - 0.2454006))
    boltzmann = struct.unpack("=f", struct.pack("=f", 8.617333262145e-5))[0]
    states, _ = gasState(checkpoint, requireEos=False)
    for words in states:
        fractions = [1.0 + value if math.copysign(1.0, value) < 0 else value
                     for value in words[:6]]
        expected = [reference["x_H"], reference["x_He"], reference["x_He_double"], 0, 0, 0]
        if any(not math.isclose(actual, target, rel_tol=2e-6, abs_tol=2e-12)
               for actual, target in zip(fractions, expected)):
            raise RuntimeError(f"Initial checkpoint fractions {fractions} differ from native {expected}")
        electrons = (fractions[0] - fractions[3] + 0.5 * fractions[5] +
                     helium * (fractions[1] + 2.0 * fractions[2]))
        particles = 1.0 + helium - 0.5 * (fractions[4] + fractions[5]) + electrons
        temperature = words[6] / (1.5 * boltzmann * particles)
        if not math.isclose(temperature, reference["T_m"], rel_tol=3e-6):
            raise RuntimeError(f"Initial checkpoint T={temperature} K differs from native {reference['T_m']} K")
    print(f"Native initial checkpoint: {len(states)} gas particles, "
          f"T={reference['T_m']:.9g} K, HII/H={reference['x_H']:.9g}")



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
    files = list(directory.glob("*.silo"))
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
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--recombination-reference", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--time-integrator", choices=("traditional", "global"),
                        default="traditional")
    args = parser.parse_args()
    executable = args.executable.resolve()
    base = args.workdir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    options = [str(executable), "--hpx:threads=2", "--hpx:bind=none",
               "--initial.particlesPerDimension=8", "--hydro.enabled=true",
               f"--compute.cuda={'true' if args.cuda else 'false'}",
               "--analysis.views=false", "--lightcone.enabled=false", "--analysis.power=false",
               "--analysis.groups=false", "--chemistry.enabled=true",
               "--checkpoint.intervalSeconds=0",
               "--cosmology.startRedshift=49", f"--cosmology.thetaCmb={2.7255 / 2.73:.17g}",
               "--output.directory=."]
    options += application_arguments(executable, far="fmm", integrator=args.time_integrator,
                                     hydro=True)
    for name, network in (("native_initial", "atomic6"), ("molecular9_native_initial", "molecular9")):
        initial = prepareCase(base, name)
        reference = nativeInitialReference(args.recombination_reference, initial, 49.0)
        runSimulation(options + [f"--chemistry.network={network}",
                                  "--run.test=nativeChemistryInitialization"],
                      initial, "initial.log")
        checkManifest(initial / "checkpoint.0", 0, testFixture=True)
        checkInitialChemistry(initial / "checkpoint.0", reference)
        print(f"{network}: pristine native initialization passed")
    cases = (("atomic6", "atomic6", []),
             ("molecular9_cold", "molecular9", []),
             ("molecular9", "molecular9",
              ["--chemistry.initialization=manual", "--chemistry.molecularFraction=0.2"]))
    for name, network, initialization in cases:
        baseline = prepareCase(base, f"{name}_baseline")
        table = "COSMICTIGER_PRIMORDIAL_RADIATION 1\n" + "\n".join(
            str(z) + " " + " ".join(["0"] * 14) for z in (0, 100)) + "\n"
        (baseline / "radiation.txt").write_text(table)
        command = options + [f"--chemistry.network={network}", "--chemistry.radiationFile=radiation.txt",
                             "--output.silo=true", "--output.units=both", "--output.frame=both"]
        command += initialization
        runSimulation(command + ["--run.maxIterations=1"], baseline, "run.log")
        checkpoint1 = baseline / "checkpoint.1"
        checkpoint2 = baseline / "checkpoint.2"
        checkManifest(checkpoint1, 1)
        checkManifest(checkpoint2, 2)
        uninterrupted = gasPayload(checkpoint2, args.time_integrator == "global")
        checkOutput(baseline)
        restarted = prepareCase(base, f"{name}_restart", checkpoint1)
        (restarted / "radiation.txt").write_text(table)
        restartCommand = [option for option in command if not option.startswith(
            ("--run.timeIntegrator=", "--gravity.method.far=", "--gravity.method.near=",
             "--gravity.expansionDegree="))]
        runSimulation(restartCommand + ["--checkpoint.readIndex=1", "--run.maxIterations=0"],
                      restarted, "restart.log")
        restored = gasPayload(restarted / "checkpoint.2", args.time_integrator == "global")
        if uninterrupted != restored:
            raise RuntimeError(f"{name}: restart changed the gas payload (source phase or state mismatch)")
        altered = prepareCase(base, f"{name}_radiation_changed", checkpoint1)
        (altered / "radiation.txt").write_text(table + "# changed radiation provenance\n")
        output = runSimulation(command + ["--checkpoint.readIndex=1", "--run.maxIterations=0"],
                               altered, "rejected.log", expectSuccess=False)
        if "radiation" not in output.lower():
            raise RuntimeError("Changed radiation table lacked compatibility diagnostic")
        incompatible = prepareCase(base, f"{name}_incompatible", checkpoint1)
        output = runSimulation(options + [f"--chemistry.network={'molecular9' if network == 'atomic6' else 'atomic6'}",
            "--chemistry.initialization=manual",
            "--checkpoint.readIndex=1", "--run.maxIterations=0"], incompatible, "rejected.log", expectSuccess=False)
        if "chemistry" not in output.lower():
            raise RuntimeError("Incompatible restart lacked chemistry diagnostic")
        print(f"{name}: production {'CUDA' if args.cuda else 'CPU'} {args.time_integrator} "
              "stepping, exact gas restart, incompatible-network rejection passed")
    # No silent extrapolation of the selected molecular cooling fits.
    invalid = prepareCase(base, "invalid_domain")
    output = runSimulation(options + ["--chemistry.network=molecular9", "--chemistry.initialTemperature=10000",
        "--run.maxIterations=0"], invalid, "rejected.log", expectSuccess=False)
    if "chemistry" not in output.lower():
        raise RuntimeError("Unsupported domain lacked chemistry diagnostic")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
