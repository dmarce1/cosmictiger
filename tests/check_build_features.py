# CosmicTiger - configured build feature regression; BSL-1.0.

"""Check the enabled source graph, degree instantiations and application link."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shlex
import sys


FEATURES = ("FMM", "PMMM_PP", "PMMM_FMM", "SPH", "HOPKINS", "CHEMISTRY", "STARS",
            "INTEGRATOR_TRADITIONAL", "INTEGRATOR_HIERARCHICAL", "INTEGRATOR_GLOBAL")
FEATURE_DEFINITIONS = {"COSMICTIGER_ENABLE_" + feature for feature in FEATURES}
FEATURE_DEFINITIONS.update({"ORDER", "FMM_MIN_ORDER", "FMM_MAX_ORDER",
                            "COSMICTIGER_ENABLE_STAR_FORMATION",
                            "COSMICTIGER_POSITION_BITS", "COSMICTIGER_WITH_SPH",
                            "COSMICTIGER_WITH_GRAVITY"})
LEGACY_DEFINITIONS = {"COSMICTIGER_GRAVITY_PMMM", "COSMICTIGER_PMMM_NEAR_FMM",
                      "COSMICTIGER_INTEGRATOR_HIERARCHICAL", "COSMICTIGER_ENABLE_PMMM",
                      "COSMICTIGER_ENABLE_PMMM_NEAR_PP", "COSMICTIGER_ENABLE_PMMM_NEAR_FMM"}
TARGET_PATTERN = re.compile(r"CMakeFiles[/\\]([^/\\]+)\.dir(?:[/\\]|$)", re.MULTILINE)
COSMOLOGY_NAME_PATTERN = re.compile(
    r"(?:^|_)(?:cosmology|boltzmann|recombination|friedmann|fft(?:w|[0-9]+)?|"
    r"linear_power|linear_growth|lpt|power)(?:_|$)")


def registeredTestNames(buildDirectory):
    names, pending, visited = set(), [buildDirectory], set()
    # Follow the current CTest directory graph instead of scanning stale discovery
    # files left by an earlier configuration of this build directory.
    while pending:
        directory = pending.pop().resolve()
        if directory in visited:
            continue
        visited.add(directory)
        path = directory / "CTestTestfile.cmake"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            match = re.match(
                r'^\s*(add_test|include|subdirs)\((?:\[=\[(.*?)\]=\]|"([^"]+)"|([^\s)]+))', line)
            if match is None:
                continue
            value = next(value for value in match.groups()[1:] if value is not None)
            if match[1] == "subdirs":
                pending.append(directory / value)
            elif match[1] == "include":
                names.add(Path(value).name.split("[", 1)[0])
            else:
                names.add(value)
    return names


def readCache(path):
    cache = {}
    for line in path.read_text().splitlines():
        match = re.match(r"([^#/:][^:]*):[^=]+=(.*)$", line)
        if match:
            cache[match.group(1)] = match.group(2)
    return cache


def degreeRange(value):
    match = re.fullmatch(r"([2-8])(?:-([2-8]))?", value)
    if not match:
        raise ValueError("invalid COSMICTIGER_EXPANSION_DEGREE; expected degree 2..8 or range")
    low, high = int(match[1]), int(match[2] or match[1])
    if low > high:
        raise ValueError("invalid COSMICTIGER_EXPANSION_DEGREE; decreasing range")
    return low, high


def hasPmmm(configuration):
    return configuration["PMMM_PP"] or configuration["PMMM_FMM"]


def hasGravity(configuration):
    return configuration["FMM"] or hasPmmm(configuration)


def hasTree(configuration):
    return configuration["FMM"] or configuration["PMMM_FMM"]


def hasChemistry(configuration):
    return configuration["SPH"] and configuration["CHEMISTRY"]


def hasStarFormation(configuration):
    return configuration["SPH"] and configuration["STARS"]


def configurationDefinitions(configuration, bits, degrees):
    low, high = degrees
    if not hasGravity(configuration):
        low = high = -1
    definitions = {"COSMICTIGER_ENABLE_" + key: str(int(configuration[key])) for key in FEATURES}
    definitions["COSMICTIGER_ENABLE_CHEMISTRY"] = str(int(hasChemistry(configuration)))
    definitions["COSMICTIGER_ENABLE_STARS"] = str(int(hasStarFormation(configuration)))
    definitions["COSMICTIGER_ENABLE_STAR_FORMATION"] = str(int(hasStarFormation(configuration)))
    definitions.update({"ORDER": str(high + 1), "FMM_MIN_ORDER": str(low + 1),
                        "FMM_MAX_ORDER": str(high + 1), "COSMICTIGER_POSITION_BITS": str(bits),
                        "COSMICTIGER_WITH_SPH": str(int(configuration["SPH"])),
                        "COSMICTIGER_WITH_GRAVITY": str(int(hasGravity(configuration)))})
    return definitions


def sourceSelectionErrors(sources, configuration):
    errors = []
    collisionlessScenarios = {"scenario.cpp", "cosmology.cpp", "plummer.cpp", "glass.cpp"}
    for source in sorted(sources):
        if not hasGravity(configuration):
            gravitySource = source.startswith(("src/gravity/", "tests/gravity/"))
            cosmologySource = source.startswith(("src/cosmology/", "tests/cosmology/")) or source in {
                "src/chemistry/recombination.cpp", "src/scenarios/cosmology.cpp"}
            cosmologySource = cosmologySource or bool(re.fullmatch(
                r"src/analysis/power(?:_cpu|_cuda)?\.(?:cpp|cu)", source))
            cosmologySource = cosmologySource or (source.startswith("tests/") and
                COSMOLOGY_NAME_PATTERN.search(Path(source).stem) is not None)
            haloSource = re.fullmatch(
                r"src/analysis/(?:bh|compress|group_entry|groups|group_tree|groups_find|healpix|rockstar|rockstar_catalogue|"
                r"lightcone|lightcone_cpu)\.(?:cpp|cu)", source)
            excludedSource = source in {
                "src/tools/groups2halofunc.cpp", "src/tools/halo_catalog_export.cpp",
                "src/tools/healpix2silo.cpp", "src/math/fp16.cpp",
                "src/scenarios/plummer.cpp", "src/scenarios/glass.cpp",
                "tests/analysis_backends.cpp", "tests/analysis_tools.cpp", "tests/bh.cpp",
                "tests/compression.cpp", "tests/fp16.cpp", "tests/groups_catalogue.cpp",
                "tests/halo_catalog_fixture.cpp", "tests/lightcone_cpu.cpp", "tests/lightcone_cuda.cu",
                "tests/lightcone_epoch.cpp", "tests/rockstar.cpp", "tests/rockstar_core.cpp",
                "tests/rockstar_eigenpair.cpp", "tests/rockstar_properties.cpp",
                "tests/softening_flops.cpp", "tests/sph/gravity.cpp"}
            if gravitySource or haloSource or excludedSource or Path(source).name.startswith("fmm_instance_"):
                errors.append(f"gravity disabled but compiles gravity source: {source}")
            if cosmologySource:
                errors.append(f"gravity disabled but compiles cosmology source: {source}")
        for feature, enabled, directory in (("FMM", configuration["FMM"], "global_fmm"),
                                            ("PMMM", hasPmmm(configuration), "pmmm")):
            if not enabled and source.startswith(
                    (f"src/gravity/{directory}/", f"tests/gravity/{directory}/")):
                errors.append(f"{feature} disabled but compiles solver source: {source}")
        for feature, pattern in (
                ("PMMM_PP", r"(?:src|tests)/gravity/pmmm/pp/near_pp[^/]*\.(?:cpp|cu)$"),
                ("PMMM_FMM", r"(?:src|tests)/gravity/pmmm/fmm/(?:cell_|ghost_|fmm_batch)[^/]*\.(?:cpp|cu)$")):
            if not configuration[feature] and re.fullmatch(pattern, source):
                errors.append(f"{feature} disabled but compiles near solver: {source}")
        if not hasPmmm(configuration) and source in {
                "src/gravity/pm/collective_dispatch.cpp", "tests/gravity/pm/fft_flops.cpp",
                "src/particles/particle_mesh_sort.cpp", "tests/particle_mesh_sort.cpp",
                "tests/mesh_domain.cpp", "tests/mesh_domains.cpp"}:
            errors.append(f"PMMM disabled but compiles solver source: {source}")
        if not hasTree(configuration) and source in {
                "src/gravity/gravity.cpp", "src/gravity/kick.cpp", "src/gravity/kick.cu",
                "src/gravity/kick_workspace.cpp", "src/gravity/kick_executor.cpp",
                "src/gravity/kick_interaction_counts.cpp", "src/gravity/source_access.cpp",
                "src/gravity/sources.cpp", "src/gravity/tree_build.cpp", "src/gravity/tree_arena.cpp",
                "tests/gravity/cartesian_kernels.cpp", "tests/gravity/cartesian_kernels_cuda.cu",
                "tests/gravity/cartesian_l2l_cuda.cu", "tests/gravity/fmm_acceptance.cpp",
                "tests/gravity/fmm_acceptance_cuda.cu", "tests/gravity/fmm_geometry.cpp"}:
            errors.append(f"tree algorithms disabled but compiles tree source: {source}")
        if hasGravity(configuration) and source == "tests/sph/no_gravity.cpp":
            errors.append(f"gravity-enabled build compiles SPH-only diagnostic regression: {source}")
        if (not configuration["SPH"] or not configuration["FMM"]) and source == "tests/kernel_selection_cuda.cu":
            errors.append(f"unavailable SPH gravity regression's CUDA helper: {source}")
        if not configuration["HOPKINS"] and source in {
                "src/hydro/hopkins.cpp", "src/scenarios/sound_wave.cpp", "tests/hopkins.cpp"}:
            errors.append(f"Hopkins disabled but compiles source: {source}")
        if not hasChemistry(configuration) and source in {
                "src/chemistry/gadget4_cooling.cpp", "src/chemistry/gadget4_cooling.cu",
                "src/hydro/sph/sph_chemistry.cpp", "tests/sph/chemistry.cpp",
                "tests/gadget4_cooling.cpp"}:
            errors.append(f"equilibrium cooling disabled but compiles source: {source}")
        if not hasStarFormation(configuration) and source in {
                "src/hydro/sph/star_formation.cpp", "src/hydro/sph/star_formation_cuda.cu",
                "tests/sph/star_formation_cuda.cu", "tests/sph/star_formation.cpp", "tests/star_formation_particles.cpp",
                "tests/springel_and_hernquist.cpp", "tests/springel_and_hernquist_cuda.cu"}:
            errors.append(f"star formation disabled but compiles source: {source}")
        if not configuration["SPH"]:
            if source.startswith(("src/hydro/sph/", "tests/sph/", "tests/scenarios/")) or source in {
                    "tests/gresho.cpp", "tests/radial_density_kernel.cpp", "tests/rayleigh_taylor.cpp",
                    "tests/units_output.cpp", "tests/distributed_output.cpp"}:
                errors.append(f"SPH disabled but compiles SPH source: {source}")
            if source.startswith("src/scenarios/") and Path(source).name not in collisionlessScenarios:
                errors.append(f"SPH disabled but compiles a hydrodynamic scenario: {source}")
    return errors


def requiredSources(configuration, cuda):
    required = {"src/core/driver.cpp", "src/core/hpx.cpp", "src/particles/particles.cpp"}
    if hasGravity(configuration):
        required.update({"src/gravity/fmm_order.cpp", "src/gravity/error.cpp", "src/gravity/reference_table.cpp"})
    if configuration["FMM"]:
        required.add("src/gravity/global_fmm/execution.cpp")
    if hasPmmm(configuration):
        required.update({"src/gravity/pmmm/solver.cpp", "src/gravity/pmmm/fft.cpp"})
        if configuration["PMMM_PP"]:
            required.add("src/gravity/pmmm/pp/near_pp.cpp")
        if configuration["PMMM_FMM"]:
            required.add("src/gravity/pmmm/fmm/fmm_batch.cpp")
    if hasTree(configuration):
        required.add("src/gravity/kick.cpp")
    if configuration["SPH"]:
        required.update({"src/hydro/sph/sph.cpp", "src/hydro/sph/sph_hydro_cpu.cpp", "src/hydro/sph/sph_prehydro_cpu.cpp",
                         "src/hydro/sph/sph_rungs_cpu.cpp", "src/hydro/sph/sph_particles.cpp", "src/hydro/sph/sph_tree.cpp"})
        if cuda:
            required.update({"src/hydro/sph/sph.cu", "src/hydro/sph/sph_hydro.cu", "src/hydro/sph/sph_prehydro.cu", "src/hydro/sph/sph_rungs.cu"})
    if configuration["HOPKINS"]:
        required.update({"src/hydro/hopkins.cpp", "src/scenarios/sound_wave.cpp"})
    if hasChemistry(configuration):
        required.add("src/hydro/sph/sph_chemistry.cpp")
    if hasStarFormation(configuration):
        required.update({"src/hydro/sph/star_formation.cpp"})
        if cuda:
            required.add("src/hydro/sph/star_formation_cuda.cu")
    if cuda:
        required.add("src/particles/drift.cu")
        if hasGravity(configuration):
            required.add("src/gravity/error_cuda.cu")
    return required


def checkBuild(buildDirectory):
    cache = readCache(buildDirectory / "CMakeCache.txt")

    def setting(name):
        if name not in cache:
            raise ValueError(f"missing {name} in CMakeCache.txt; reconfigure this build")
        return cache[name]

    def enabled(name):
        value = setting(name).upper()
        return value not in {"", "0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND"} and not value.endswith("-NOTFOUND")

    sourceDirectory = Path(setting("CMAKE_HOME_DIRECTORY")).resolve()
    configuration = {feature: enabled("COSMICTIGER_ENABLE_" + feature) for feature in FEATURES}
    if enabled("COSMICTIGER_ENABLE_STAR_FORMATION") != configuration["STARS"]:
        raise ValueError("COSMICTIGER_ENABLE_STAR_FORMATION must match COSMICTIGER_ENABLE_STARS")
    degrees = degreeRange(setting("COSMICTIGER_EXPANSION_DEGREE"))
    cuda, testing = enabled("COSMICTIGER_WITH_CUDA"), enabled("BUILD_TESTING")
    positionType = setting("COSMICTIGER_POSITION_TYPE")
    if positionType not in ("fixed32", "fixed64"):
        raise ValueError("invalid COSMICTIGER_POSITION_TYPE in CMakeCache.txt")
    bits = 32 if positionType == "fixed32" else 64
    if configuration["HOPKINS"] and not configuration["SPH"]:
        raise ValueError("COSMICTIGER_ENABLE_HOPKINS=ON requires COSMICTIGER_ENABLE_SPH=ON")
    if configuration["HOPKINS"] and bits != 64:
        raise ValueError("COSMICTIGER_ENABLE_HOPKINS=ON requires COSMICTIGER_POSITION_TYPE=fixed64")
    commands = json.loads((buildDirectory / "compile_commands.json").read_text())
    if not isinstance(commands, list) or not commands:
        raise ValueError("compile_commands.json contains no compilation entries")
    sourcesByTarget, definitionsByTarget, errors = {}, {}, []
    for entry in commands:
        path = (Path(entry["directory"]) / entry["file"]).resolve()
        try:
            source = path.relative_to(sourceDirectory).as_posix()
        except ValueError:
            source = str(path)
        if not cuda and (path.suffix == ".cu" or source == "src/cuda/cuda_stream.cpp"):
            errors.append(f"CUDA disabled but graph compiles {source}")
        if not testing and source.startswith("tests/"):
            errors.append(f"BUILD_TESTING=OFF but graph compiles {source}")
        arguments = entry.get("arguments") or shlex.split(entry["command"])
        output = entry.get("output", "")
        if not output and "-o" in arguments:
            output = arguments[arguments.index("-o") + 1]
        match = TARGET_PATTERN.search(output)
        if match is None:
            errors.append(f"cannot identify compilation target for {source}")
            continue
        target = match[1]
        sourcesByTarget.setdefault(target, Counter())[source] += 1
        definitions = {}
        for argument in arguments:
            if argument.startswith("-D"):
                name, _, value = argument[2:].partition("=")
                definitions.setdefault(name, set()).add(value)
        definitionsByTarget.setdefault(target, []).append((source, definitions))
        precisionTest = re.fullmatch(r"gravity_error_cuda_reference_(32|64)_test", target)
        expectedBits = precisionTest[1] if precisionTest else str(bits)
        foundBits = definitions.get("COSMICTIGER_POSITION_BITS")
        if (precisionTest or foundBits is not None) and foundBits != {expectedBits}:
            errors.append(f"{target}: {source} needs COSMICTIGER_POSITION_BITS={expectedBits}, found {sorted(foundBits or set())}")
        for error in sourceSelectionErrors({source}, configuration):
            errors.append(f"{target}: {error}")
    if not hasGravity(configuration):
        targets = set(sourcesByTarget)
        targetDirectories = buildDirectory / "CMakeFiles/TargetDirectories.txt"
        if targetDirectories.exists():
            targets.update(match[1] for match in TARGET_PATTERN.finditer(targetDirectories.read_text()))
        for target in sorted(targets):
            if COSMOLOGY_NAME_PATTERN.search(target):
                errors.append(f"gravity disabled but configures cosmology target: {target}")
        if testing:
            for name in sorted(registeredTestNames(buildDirectory)):
                if COSMOLOGY_NAME_PATTERN.search(name):
                    errors.append(f"gravity disabled but registers cosmology test: {name}")
    coreSources = set(sourcesByTarget.get("cosmictiger_core", {}))
    if not coreSources:
        errors.append("shared cosmictiger_core library has no compilation entries")
    if hasGravity(configuration):
        for target, source in (
                ("cosmictiger_cosmology", "src/cosmology/boltzmann.cpp"),
                ("cosmictiger_fft", "src/cosmology/fft.cpp"),
                ("cosmictiger_power", "src/analysis/power.cpp"),
                ("cosmictiger_recombination", "src/chemistry/recombination.cpp")):
            if source not in sourcesByTarget.get(target, {}):
                errors.append(f"{target}: required library source missing: {source}")
    for source in sorted(coreSources):
        if source.startswith(("src/cosmology/", "src/analysis/")) or source == "src/chemistry/recombination.cpp":
            errors.append(f"shared core compiles physics library source: {source}")
    for source, count in sourcesByTarget.get("cosmictiger_core", {}).items():
        if count != 1:
            errors.append(f"cosmictiger_core compiles {source} {count} times instead of once")
    for source, definitions in definitionsByTarget.get("cosmictiger_core", []):
        for name in sorted((FEATURE_DEFINITIONS | LEGACY_DEFINITIONS) & definitions.keys()):
            errors.append(f"cosmictiger_core: {source} inherits feature definition {name}")
    applications = {target for target, sources in sourcesByTarget.items() if "src/main.cpp" in sources}
    for missing in {"cosmictiger"} - applications:
        errors.append(f"missing application target: {missing}")
    for unexpected in applications - {"cosmictiger"}:
        errors.append(f"unexpected application target: {unexpected}")
    for target in sorted(sourcesByTarget):
        if target.endswith("_objects") and target != "cosmictiger_objects":
            errors.append(f"unexpected physics object target: {target}")
    for source, count in sourcesByTarget.get("cosmictiger_objects", {}).items():
        if count != 1:
            errors.append(f"cosmictiger_objects compiles {source} {count} times instead of once")
    expectedDefinitions = configurationDefinitions(configuration, bits, degrees)
    starCudaSources = set(sourcesByTarget.get("cosmictiger_star_formation_cuda", {}))
    if cuda and hasStarFormation(configuration):
        if sourcesByTarget.get("cosmictiger_star_formation_cuda", {}) != Counter({"src/hydro/sph/star_formation_cuda.cu": 1}):
            errors.append("strict star-formation CUDA target must compile its production source once")
        for source, definitions in definitionsByTarget.get("cosmictiger_star_formation_cuda", []):
            for name, value in expectedDefinitions.items():
                if definitions.get(name) != {value}:
                    errors.append(f"cosmictiger_star_formation_cuda: {source} needs {name}={value}")
    elif starCudaSources:
        errors.append("unavailable strict star-formation CUDA target is compiled")
    for target in ("cosmictiger", "cosmictiger_objects"):
        sources = set(sourcesByTarget.get(target, {}))
        if target == "cosmictiger":
            if sources != {"src/main.cpp"}:
                errors.append("cosmictiger: executable must compile only src/main.cpp")
        else:
            for missing in sorted(requiredSources(configuration, cuda) - (sources | starCudaSources)):
                errors.append(f"{target}: required source missing: {missing}")
            instances = Counter(Path(source).name for source, count in sourcesByTarget.get(target, {}).items()
                                for _ in range(count) if Path(source).name.startswith("fmm_instance_"))
            expectedInstances = Counter()
            if hasTree(configuration):
                for degree in range(degrees[0], degrees[1] + 1):
                    expectedInstances[f"fmm_instance_cpu_{degree + 1}.cpp"] = 1
                    if cuda:
                        expectedInstances[f"fmm_instance_cuda_{degree + 1}.cu"] = 1
            if instances != expectedInstances:
                errors.append(f"{target}: generated FMM instances {dict(instances)}, expected {dict(expectedInstances)}")
            for duplicate in sorted(coreSources & sources):
                errors.append(f"{target}: recompiles shared core source: {duplicate}")
        for source, definitions in definitionsByTarget.get(target, []):
            for name, value in expectedDefinitions.items():
                if definitions.get(name) != {value}:
                    errors.append(f"{target}: {source} needs {name}={value}, found {sorted(definitions.get(name, set()))}")
            for name in sorted(LEGACY_DEFINITIONS & definitions.keys()):
                errors.append(f"{target}: {source} retains legacy selection definition {name}")
    checkedLinks = 0
    linkPath = buildDirectory / "CMakeFiles/cosmictiger.dir/link.txt"
    if linkPath.exists():
        checkedLinks = 1
        linkCommand = linkPath.read_text()
        linkedObjects = {match[1] for match in TARGET_PATTERN.finditer(linkCommand) if match[1].endswith("_objects")}
        if linkedObjects != {"cosmictiger_objects"}:
            errors.append(f"cosmictiger links unexpected object targets {sorted(linkedObjects)}")
        libraries = {Path(argument).name[3:-2] for argument in shlex.split(linkCommand)
                     if Path(argument).name.startswith("lib") and argument.endswith(".a")}
        if "cosmictiger_core" not in libraries:
            errors.append("cosmictiger does not link the shared cosmictiger_core library")
        linkedSources = set().union(*(set(sourcesByTarget.get(library, {})) for library in libraries))
        if hasGravity(configuration) and "src/chemistry/recombination.cpp" not in linkedSources:
            errors.append("cosmictiger lacks shared homogeneous recombination")
        if hasGravity(configuration) and "src/cosmology/fft.cpp" not in linkedSources:
            errors.append("cosmictiger lacks its cosmology FFT library")
        if not hasGravity(configuration):
            for argument in shlex.split(linkCommand):
                if "fftw" in Path(argument).name.lower():
                    errors.append(f"gravity disabled but links FFTW: {argument}")
        if hasChemistry(configuration):
            chemistry = {"src/chemistry/gadget4_cooling.cpp"}
            if cuda:
                chemistry.add("src/chemistry/gadget4_cooling.cu")
            for missing in chemistry - linkedSources:
                errors.append(f"cosmictiger lacks equilibrium cooling source {missing}")
    elif setting("CMAKE_GENERATOR") == "Unix Makefiles":
        errors.append("cosmictiger generated Makefile link command is missing")
    if errors:
        raise ValueError("Build feature checks failed:\n  " + "\n  ".join(errors))
    return (f"Build features passed: one application, {len(coreSources)} shared core sources; "
            f"potential degrees={degrees[0]}-{degrees[1]}, positions={positionType}, CUDA={cuda}, "
            f"tests={testing}; {len(commands)} compilation entries; {checkedLinks} link commands checked")


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
