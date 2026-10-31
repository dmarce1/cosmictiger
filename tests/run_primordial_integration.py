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


def gasState(path, requireEos=True, globalTimestep=None, expectedCount=512):
    payload = (path / "checkpoint.0.dat").read_bytes()
    prefix = struct.pack("=I", 0x43545350)
    offset = payload.rfind(prefix)
    if offset < 0:
        raise RuntimeError("No versioned chemistry gas payload in checkpoint")
    fields = struct.unpack_from("=7I", payload, offset)
    if fields[1] not in (51, 53, 55, 57) or fields[2] not in (4, 8) or fields[5:] != (3, 4):
        raise RuntimeError(f"Unsupported chemistry checkpoint layout {fields}")
    count = int.from_bytes(payload[offset + 28:offset + 28 + fields[2]], sys.byteorder, signed=True)
    if count != expectedCount:
        raise RuntimeError(f"Unexpected gas count {count}")
    # The tail is explicitly serialized: nine species counts per H nucleus, eV/H energy,
    # pressure/energy ratio and frozen gamma, then each particle's pending
    # conformal half-interval and opening heating ledger (two doubles).
    viscosityBytes = {51: 0, 53: 8, 55: 12, 57: 12}[fields[1]]
    chemistryStart = offset + 28 + fields[2] + count * (
        fields[3] + fields[4] + 38 + fields[2] + viscosityBytes)
    chemistryEnd = chemistryStart + count * (12 * 4 + 2 * 8)
    if chemistryEnd > len(payload):
        raise RuntimeError("Truncated chemistry checkpoint state")
    chemistry = payload[chemistryStart:chemistryEnd]
    records = list(struct.iter_unpack("=12f2d", chemistry))
    for i, record in enumerate(records):
        words = record[:12]
        halfInterval, pendingHeating = record[12:]
        if not (math.isfinite(halfInterval) and halfInterval >= 0 and
                math.isfinite(pendingHeating)):
            raise RuntimeError(f"Invalid per-particle chemistry phase {record[12:]}")
        if requireEos and halfInterval <= 0:
            raise RuntimeError("Evolved chemistry checkpoint lost its pending source half")
        if globalTimestep is not None and halfInterval != 0.5 * globalTimestep:
            raise RuntimeError("Chemistry half-interval differs from the exact global timestep")
        if not (all(math.isfinite(x) for x in words) and
                all(x >= 0 for x in words[:9]) and words[9] > 0):
            raise RuntimeError(f"Invalid chemistry checkpoint state {words}")
        hydrogen = words[0] + words[1] + words[6] + 2.0 * (words[7] + words[8])
        charge = words[1] + words[3] + 2.0 * words[4] + words[8] - words[6]
        if (abs(hydrogen - 1.0) > 1e-6 or
                abs(words[5] - charge) > 1e-6 * max(words[5], abs(charge), 1e-30)):
            raise RuntimeError(f"Unconserved checkpoint chemistry state {words}")
        if requireEos and not (words[10] > 0 and words[11] > 1):
            raise RuntimeError(f"Invalid evolved chemistry EOS state {words}")
    return [record[:12] for record in records], payload[offset:chemistryEnd]


def gasPayload(path, globalMode=False, expectedCount=512):
    timestep = globalState(path)[1] if globalMode else None
    return gasState(path, globalTimestep=timestep, expectedCount=expectedCount)[1]


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
    # Check all nine saved species counts and infer temperature from the saved
    # eV/H energy, independently of option printing and Silo derived fields.
    helium = 0.2454006 / (4.0 * (1.0 - 0.2454006))
    boltzmann = struct.unpack("=f", struct.pack("=f", 8.617333262145e-5))[0]
    states, _ = gasState(checkpoint, requireEos=False)
    for words in states:
        hp = reference["x_H"]
        hep = helium * reference["x_He"]
        hepp = helium * reference["x_He_double"]
        expected = [1.0 - hp, hp, helium - hep - hepp, hep, hepp,
                    hp + hep + 2.0 * hepp, 0.0, 0.0, 0.0]
        if any(not math.isclose(actual, target, rel_tol=2e-6, abs_tol=2e-12)
               for actual, target in zip(words[:9], expected)):
            raise RuntimeError(f"Initial checkpoint counts {words[:9]} differ from native {expected}")
        particles = sum(words[:9])
        temperature = words[9] / (1.5 * boltzmann * particles)
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
                    fields = ("chem_HI", "chem_H2", "chem_electrons", "temperature",
                              "mean_molecular_weight", "frozen_gamma")
                    fields += tuple(f"chem_number_per_h_{species}" for species in range(9))
                    for field in fields:
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
    parser.add_argument("--cases", default="",
                        help="comma-separated subset of case names to run (default: all)")
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
    selected = {name for name in args.cases.split(",") if name}
    for name, network in (("native_initial", "atomic6"), ("molecular9_native_initial", "molecular9")):
        if selected and name not in selected:
            continue
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
             # Molecule-rich gas needs many source substeps; use 64 particles to
             # bound runtime while retaining the production fractional-change limit.
             ("molecular9", "molecular9",
              ["--chemistry.initialization=manual", "--chemistry.molecularFraction=0.2",
               "--initial.particlesPerDimension=4"]),
             # Hot molecular gas above the former 6000 K ceiling: molecules are
             # retained, dissociate and recombine within the nine-species network.
             # Keep the production fractional-change limit and use 64 particles
             # to bound the integration regression's runtime.
             ("molecular9_hot", "molecular9",
              ["--chemistry.initialization=manual", "--chemistry.initialTemperature=30000",
               "--chemistry.hydrogenIonFraction=0.5", "--chemistry.molecularFraction=0.001",
               "--initial.particlesPerDimension=4"]))
    for name, network, initialization in cases:
        if selected and name not in selected:
            continue
        baseline = prepareCase(base, f"{name}_baseline")
        table = "COSMICTIGER_PRIMORDIAL_RADIATION 1\n" + "\n".join(
            str(z) + " " + " ".join(["0"] * 14) for z in (0, 100)) + "\n"
        (baseline / "radiation.txt").write_text(table)
        command = options + [f"--chemistry.network={network}", "--chemistry.radiationFile=radiation.txt",
                             "--output.silo=true", "--output.units=both", "--output.frame=both"]
        overridden = {option.split("=")[0] for option in initialization}
        command = [option for option in command if option.split("=")[0] not in overridden]
        command += initialization
        runSimulation(command + ["--run.maxIterations=1"], baseline, "run.log")
        checkpoint1 = baseline / "checkpoint.1"
        checkpoint2 = baseline / "checkpoint.2"
        checkManifest(checkpoint1, 1)
        checkManifest(checkpoint2, 2)
        gasCount = next(int(option.split("=")[1]) ** 3 for option in command
                        if option.startswith("--initial.particlesPerDimension="))
        uninterrupted = gasPayload(checkpoint2, args.time_integrator == "global", gasCount)
        checkOutput(baseline)
        restarted = prepareCase(base, f"{name}_restart", checkpoint1)
        (restarted / "radiation.txt").write_text(table)
        restartCommand = [option for option in command if not option.startswith(
            ("--run.timeIntegrator=", "--gravity.method.far=", "--gravity.method.near=",
             "--gravity.expansionDegree="))]
        runSimulation(restartCommand + ["--checkpoint.readIndex=1", "--run.maxIterations=0"],
                      restarted, "restart.log")
        restored = gasPayload(restarted / "checkpoint.2", args.time_integrator == "global", gasCount)
        if uninterrupted != restored:
            raise RuntimeError(f"{name}: restart changed the gas payload (source phase or state mismatch)")
        altered = prepareCase(base, f"{name}_radiation_changed", checkpoint1)
        (altered / "radiation.txt").write_text(table + "# changed radiation provenance\n")
        output = runSimulation(command + ["--checkpoint.readIndex=1", "--run.maxIterations=0"],
                               altered, "rejected.log", expectSuccess=False)
        if "radiation" not in output.lower():
            raise RuntimeError("Changed radiation table lacked compatibility diagnostic")
        # Same particle layout and radiation table as the checkpoint, so the
        # restart reaches the chemistry comparison and fails on the network.
        incompatible = prepareCase(base, f"{name}_incompatible", checkpoint1)
        (incompatible / "radiation.txt").write_text(table)
        swapped = "molecular9" if network == "atomic6" else "atomic6"
        # Keep chemistry enabled with the same radiation table and solver
        # settings; only the network and the initial composition change (the
        # molecular fractions of a manual molecular9 start are invalid for atomic6).
        initialComposition = ("--chemistry.network=", "--chemistry.initialTemperature=",
                              "--chemistry.hydrogenIonFraction=", "--chemistry.heliumIonFraction=",
                              "--chemistry.heliumDoubleIonFraction=",
                              "--chemistry.hydrogenMinusFraction=", "--chemistry.molecularFraction=",
                              "--chemistry.molecularIonFraction=")
        incompatibleCommand = [option for option in restartCommand
                               if not option.startswith(initialComposition)] + [
            f"--chemistry.network={swapped}", "--checkpoint.readIndex=1", "--run.maxIterations=0"]
        output = runSimulation(incompatibleCommand, incompatible, "rejected.log", expectSuccess=False)
        if "Checkpoint chemistry network" not in output:
            raise RuntimeError("Incompatible restart lacked the chemistry network diagnostic")
        print(f"{name}: production {'CUDA' if args.cuda else 'CPU'} {args.time_integrator} "
              "stepping, exact gas restart, incompatible-network rejection passed")
    # No silent extrapolation beyond the shared 1e8 K ceiling of either network.
    for network in ("molecular9", "atomic6"):
        invalid = prepareCase(base, f"invalid_domain_{network}")
        output = runSimulation(options + [f"--chemistry.network={network}",
            "--chemistry.initialization=manual", "--chemistry.initialTemperature=2e8",
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
