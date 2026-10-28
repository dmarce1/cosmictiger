# CosmicTiger - configured build feature regression; BSL-1.0.

"""Check source exclusion in a configured build without compiling or running it."""

import argparse
import json
from pathlib import Path
import re
import shlex
import sys


def readCache(path):
    cache = {}
    for line in path.read_text().splitlines():
        match = re.match(r"([^#/:][^:]*):[^=]+=(.*)$", line)
        if match:
            cache[match.group(1)] = match.group(2)
    return cache


def checkBuild(buildDirectory):
    cache = readCache(buildDirectory / "CMakeCache.txt")

    def setting(name):
        if name not in cache:
            raise ValueError(f"missing {name} in CMakeCache.txt; reconfigure this build")
        return cache[name]

    def enabled(name):
        value = setting(name).upper()
        falseValues = {"", "0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND"}
        return value not in falseValues and not value.endswith("-NOTFOUND")

    sourceDirectory = Path(setting("CMAKE_HOME_DIRECTORY")).resolve()
    gravity = setting("COSMICTIGER_GRAVITY_METHOD")
    near = setting("COSMICTIGER_PMMM_NEAR_METHOD")
    if gravity not in ("fmm", "pmmm") or near not in ("pp", "fmm"):
        raise ValueError("invalid gravity or PMMM near method in CMakeCache.txt")
    sph = enabled("COSMICTIGER_WITH_SPH")
    cuda = enabled("COSMICTIGER_WITH_CUDA")
    testing = enabled("BUILD_TESTING")
    commands = json.loads((buildDirectory / "compile_commands.json").read_text())
    if not isinstance(commands, list) or not commands:
        raise ValueError("compile_commands.json contains no compilation entries")

    sources = set()
    applications = {}
    errors = []
    appNames = {"cosmictiger", "cosmictiger_fixed64"}
    for entry in commands:
        path = (Path(entry["directory"]) / entry["file"]).resolve()
        try:
            source = path.relative_to(sourceDirectory).as_posix()
        except ValueError:
            source = str(path)
        sources.add(source)
        if not cuda and path.suffix == ".cu":
            errors.append(f"CUDA disabled but graph compiles {source}")
        arguments = entry.get("arguments")
        if arguments is None:
            arguments = shlex.split(entry["command"])
        output = entry.get("output", "")
        if not output and "-o" in arguments:
            output = arguments[arguments.index("-o") + 1]
        targetMatch = re.search(r"CMakeFiles[/\\]([^/\\]+)\.dir[/\\]", output)
        target = targetMatch.group(1) if targetMatch else None
        if target in appNames:
            applications.setdefault(target, set()).add(source)
            bits = 64 if target == "cosmictiger_fixed64" else 32
            if source == "src/main.cpp" and f"-DCOSMICTIGER_POSITION_BITS={bits}" not in arguments:
                errors.append(f"{target}: main.cpp lacks its {bits}-bit position definition")

    def reject(predicate, reason):
        for source in sorted(sources):
            if predicate(source):
                errors.append(f"{reason}: {source}")

    otherSolver = "nitadori" if gravity == "fmm" else "dehnen"
    reject(lambda source: source.startswith((f"src/gravity/{otherSolver}/",
                                             f"tests/gravity/{otherSolver}/")),
           f"{gravity} build compiles the other solver")
    if gravity == "pmmm":
        nearPattern = (r"(?:src|tests)/gravity/nitadori/near_pp[^/]*\.(?:cpp|cu)$"
                        if near == "fmm" else
                        r"(?:src|tests)/gravity/nitadori/(?:cell_|ghost_|fmm_batch)[^/]*\.(?:cpp|cu)$")
        reject(lambda source: re.fullmatch(nearPattern, source),
               f"PMMM-{near} build compiles the other near solver")
    if not sph or gravity != "fmm":
        reject(lambda source: source == "tests/kernel_selection_cuda.cu",
               "graph compiles the unavailable SPH gravity regression's CUDA helper")
    if not sph:
        reject(lambda source: source.startswith(("src/sph/", "tests/sph/", "tests/scenarios/")),
               "SPH disabled but graph compiles SPH sources")
        # These are the dispatcher and the collisionless initial conditions.
        collisionlessScenarios = {"scenario.cpp", "cosmology.cpp", "plummer.cpp", "glass.cpp"}
        reject(lambda source: source.startswith("src/scenarios/") and
               Path(source).name not in collisionlessScenarios,
               "SPH disabled but graph compiles a hydrodynamic scenario")
        reject(lambda source: source in {"src/chemistry/primordial_chemistry.cpp",
                                        "src/chemistry/primordial_chemistry.cu"},
               "SPH disabled but graph compiles particle chemistry")
    if not testing:
        reject(lambda source: source.startswith("tests/"),
               "BUILD_TESTING=OFF but graph compiles project tests")

    expectedApps = {"cosmictiger", "cosmictiger_fixed64"}
    if applications.keys() != expectedApps:
        errors.append(f"application targets: expected {sorted(expectedApps)}, "
                      f"found {sorted(applications)}")
    # Check entry points rather than duplicating CMake's complete source lists.
    required = {"src/main.cpp", "src/core/driver.cpp", "src/particles/particles.cpp",
                "src/cosmology/fft.cpp"}
    if gravity == "fmm":
        required.add("src/gravity/dehnen/execution.cpp")
    else:
        required.add("src/gravity/nitadori/solver.cpp")
        required.add("src/gravity/nitadori/" + ("fmm_batch.cpp" if near == "fmm" else "near_pp.cpp"))
    if gravity == "fmm" or near == "fmm":
        required.add("src/gravity/kick.cpp")
    if sph:
        required.add("src/sph/sph.cpp")
    if cuda:
        required.add("src/particles/drift.cu")
    for target in sorted(expectedApps & applications.keys()):
        for missing in sorted(required - applications[target]):
            errors.append(f"{target}: required source missing: {missing}")
    if "src/chemistry/recombination.cpp" not in sources:
        errors.append("shared homogeneous recombination source is missing")
    if errors:
        raise ValueError("Build feature source checks failed:\n  " + "\n  ".join(errors))
    return (f"Build source selection passed: gravity={gravity}, near={near}, "
            f"SPH={sph}, CUDA={cuda}, tests={testing}; "
            f"{len(commands)} compilation entries checked")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("buildDirectory", metavar="build_directory", type=Path)
    args = parser.parse_args()
    print(checkBuild(args.buildDirectory.resolve()))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
