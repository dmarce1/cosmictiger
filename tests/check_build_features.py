# CosmicTiger - configured build feature regression; BSL-1.0.

"""Check every named variant's compile graph and the generated Makefile links."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shlex
import sys


VARIANT_DEFINITIONS = {"ORDER", "FMM_MIN_ORDER", "FMM_MAX_ORDER",
                       "COSMICTIGER_POSITION_BITS", "COSMICTIGER_WITH_SPH",
                       "COSMICTIGER_INTEGRATOR_HIERARCHICAL", "COSMICTIGER_GRAVITY_PMMM",
                       "COSMICTIGER_PMMM_NEAR_FMM"}
TARGET_PATTERN = re.compile(r"CMakeFiles[/\\]([^/\\]+)\.dir[/\\]")


def readCache(path):
    cache = {}
    for line in path.read_text().splitlines():
        match = re.match(r"([^#/:][^:]*):[^=]+=(.*)$", line)
        if match:
            cache[match.group(1)] = match.group(2)
    return cache


def namedVariants():
    variants = {}
    for integrator in ("leapfrog", "hierarchical"):
        for gravity in ("fmm", "pmmm_pp", "pmmm_fmm"):
            for sph in (False, True):
                if sph and integrator != "leapfrog":
                    continue
                for order in range(3, 10):
                    name = f"cosmictiger_{integrator}_{gravity}{'_sph' if sph else ''}_{order}"
                    variants[name] = (integrator, gravity, sph, order)
    return variants


def configurationDefinitions(configuration, bits):
    integrator, gravity, sph, order = configuration
    return {"ORDER": str(order), "FMM_MIN_ORDER": str(order), "FMM_MAX_ORDER": str(order),
            "COSMICTIGER_POSITION_BITS": str(bits), "COSMICTIGER_WITH_SPH": str(int(sph)),
            "COSMICTIGER_INTEGRATOR_HIERARCHICAL": str(int(integrator == "hierarchical")),
            "COSMICTIGER_GRAVITY_PMMM": str(int(gravity != "fmm")),
            "COSMICTIGER_PMMM_NEAR_FMM": str(int(gravity == "pmmm_fmm"))}


def sourceSelectionErrors(sources, configuration):
    _, gravity, sph, _ = configuration
    errors = []
    otherSolver = "nitadori" if gravity == "fmm" else "dehnen"
    collisionlessScenarios = {"scenario.cpp", "cosmology.cpp", "plummer.cpp", "glass.cpp"}
    for source in sorted(sources):
        if source.startswith((f"src/gravity/{otherSolver}/", f"tests/gravity/{otherSolver}/")):
            errors.append(f"{gravity} compiles the other gravity solver: {source}")
        if gravity.startswith("pmmm_"):
            nearPattern = (r"(?:src|tests)/gravity/nitadori/near_pp[^/]*\.(?:cpp|cu)$"
                           if gravity == "pmmm_fmm" else
                           r"(?:src|tests)/gravity/nitadori/(?:cell_|ghost_|fmm_batch)[^/]*\.(?:cpp|cu)$")
            if re.fullmatch(nearPattern, source):
                errors.append(f"{gravity} compiles the other near solver: {source}")
        if (not sph or gravity != "fmm") and source == "tests/kernel_selection_cuda.cu":
            errors.append(f"unavailable SPH gravity regression's CUDA helper: {source}")
        if not sph:
            if source.startswith(("src/sph/", "tests/sph/", "tests/scenarios/")):
                errors.append(f"SPH disabled but compiles SPH source: {source}")
            if source.startswith("src/scenarios/") and Path(source).name not in collisionlessScenarios:
                errors.append(f"SPH disabled but compiles a hydrodynamic scenario: {source}")
            if source in {"src/chemistry/primordial_chemistry.cpp",
                          "src/chemistry/primordial_chemistry.cu"}:
                errors.append(f"SPH disabled but compiles particle chemistry: {source}")
    return errors


def requiredSources(configuration, cuda):
    _, gravity, sph, _ = configuration
    required = {"src/core/driver.cpp", "src/particles/particles.cpp", "src/cosmology/fft.cpp",
                "src/gravity/error.cpp", "src/gravity/reference_table.cpp"}
    if gravity == "fmm":
        required.add("src/gravity/dehnen/execution.cpp")
    else:
        required.add("src/gravity/nitadori/solver.cpp")
        required.add("src/gravity/nitadori/" +
                     ("fmm_batch.cpp" if gravity == "pmmm_fmm" else "near_pp.cpp"))
    if gravity in ("fmm", "pmmm_fmm"):
        required.add("src/gravity/kick.cpp")
    if sph:
        required.update({"src/sph/sph.cpp", "src/sph/sph_hydro_cpu.cpp",
                         "src/sph/sph_prehydro_cpu.cpp", "src/sph/sph_rungs_cpu.cpp",
                         "src/sph/sph_particles.cpp", "src/sph/sph_tree.cpp",
                         "src/sph/sph_chemistry.cpp"})
        if cuda:
            required.update({"src/sph/sph.cu", "src/sph/sph_hydro.cu",
                             "src/sph/sph_prehydro.cu", "src/sph/sph_rungs.cu"})
    if cuda:
        required.add("src/particles/drift.cu")
        required.add("src/gravity/error_cuda.cu")
    return required


def makefileDefaultErrors(path, expectedTargets, selectedName):
    rules = {}
    for line in path.read_text().replace("\\\n", " ").splitlines():
        if not line or line[0].isspace():
            continue
        target, separator, dependencies = line.partition(":")
        if separator and (target == "all" or target.endswith("/all")):
            rules.setdefault(target, set()).update(dependencies.split())
    if "all" not in rules:
        return ["generated Makefile2 has no root all rule"]

    def ruleName(target):
        return f"CMakeFiles/{target}.dir/all"

    namedApps = set(expectedTargets) - {"cosmictiger"}
    objectTargets = {value[2] for value in expectedTargets.values()}
    namedRules = {ruleName(target) for target in namedApps | objectTargets}
    errors = []
    for eager in sorted(rules["all"] & namedRules):
        errors.append(f"root all eagerly builds named variant target: {eager}")

    pending = ["all"]
    reachable = set()
    while pending:
        target = pending.pop()
        if target in reachable:
            continue
        reachable.add(target)
        pending.extend(rules.get(target, set()) - reachable)
    selectedObjects = {ruleName(selectedName + "_objects")}
    for unexpected in sorted((reachable & namedRules) - selectedObjects):
        errors.append(f"root all reaches an unselected variant target: {unexpected}")
    for missing in sorted(selectedObjects - reachable):
        errors.append(f"root all does not build selected variant objects: {missing}")
    for application in ("cosmictiger",):
        appRule = ruleName(application)
        if appRule not in rules["all"]:
            errors.append(f"root all does not build default application {application}")
        objects = ruleName(expectedTargets[application][2])
        if objects not in rules.get(appRule, set()):
            errors.append(f"{application}: Makefile dependency omits selected objects {objects}")
    return errors


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
    selectedConfiguration = (setting("COSMICTIGER_TIME_INTEGRATOR"),
                             "fmm" if gravity == "fmm" else f"pmmm_{near}",
                             enabled("COSMICTIGER_WITH_SPH"), int(setting("COSMICTIGER_WITH_ORDER")))
    cuda = enabled("COSMICTIGER_WITH_CUDA")
    testing = enabled("BUILD_TESTING")
    positionType = setting("COSMICTIGER_POSITION_TYPE")
    if positionType not in ("fixed32", "fixed64"):
        raise ValueError("invalid COSMICTIGER_POSITION_TYPE in CMakeCache.txt")
    positionBits = 32 if positionType == "fixed32" else 64
    commands = json.loads((buildDirectory / "compile_commands.json").read_text())
    if not isinstance(commands, list) or not commands:
        raise ValueError("compile_commands.json contains no compilation entries")

    variants = namedVariants()
    selectedNames = [name for name, configuration in variants.items()
                     if configuration == selectedConfiguration]
    if len(selectedNames) != 1:
        raise ValueError("default CMake configuration is not a supported named variant")
    selectedName = selectedNames[0]
    expectedTargets = {}
    for base, configuration in variants.items():
        expectedTargets[base] = (configuration, positionBits, base + "_objects")
    expectedTargets["cosmictiger"] = (
        selectedConfiguration, positionBits, selectedName + "_objects")

    sourcesByTarget = {}
    definitionsByTarget = {}
    errors = []
    for entry in commands:
        path = (Path(entry["directory"]) / entry["file"]).resolve()
        try:
            source = path.relative_to(sourceDirectory).as_posix()
        except ValueError:
            source = str(path)
        if not cuda and path.suffix == ".cu":
            errors.append(f"CUDA disabled but graph compiles {source}")
        if not testing and source.startswith("tests/"):
            errors.append(f"BUILD_TESTING=OFF but graph compiles {source}")
        arguments = entry.get("arguments")
        if arguments is None:
            arguments = shlex.split(entry["command"])
        output = entry.get("output", "")
        if not output and "-o" in arguments:
            output = arguments[arguments.index("-o") + 1]
        targetMatch = TARGET_PATTERN.search(output)
        if targetMatch is None:
            errors.append(f"cannot identify compilation target for {source}")
            continue
        target = targetMatch.group(1)
        sourcesByTarget.setdefault(target, Counter())[source] += 1
        definitions = {}
        for argument in arguments:
            if argument.startswith("-D"):
                name, _, value = argument[2:].partition("=")
                definitions.setdefault(name, set()).add(value)
        definitionsByTarget.setdefault(target, []).append((source, definitions))
        compiledPositionBits = definitions.get("COSMICTIGER_POSITION_BITS")
        # The diagnostic's standalone CUDA tests deliberately cover both
        # widths in every build, independently of the application's selection.
        precisionTest = re.fullmatch(r"gravity_error_cuda_reference_(32|64)_test", target)
        expectedPositionBits = precisionTest.group(1) if precisionTest else str(positionBits)
        if ((precisionTest or compiledPositionBits is not None)
                and compiledPositionBits != {expectedPositionBits}):
            errors.append(f"{target}: {source} needs COSMICTIGER_POSITION_BITS={expectedPositionBits}, "
                          f"found {sorted(compiledPositionBits or set())}")

    coreSources = set(sourcesByTarget.get("cosmictiger_core", {}))
    if not coreSources:
        errors.append("shared cosmictiger_core library has no compilation entries")
    for source, count in sourcesByTarget.get("cosmictiger_core", {}).items():
        if count != 1:
            errors.append(f"cosmictiger_core compiles {source} {count} times instead of once")
    for source, definitions in definitionsByTarget.get("cosmictiger_core", []):
        for definition in sorted(VARIANT_DEFINITIONS & definitions.keys()):
            errors.append(f"cosmictiger_core: {source} inherits variant definition {definition}")

    applicationTargets = {target for target, sources in sourcesByTarget.items()
                          if "src/main.cpp" in sources}
    if applicationTargets != expectedTargets.keys():
        for missing in sorted(expectedTargets.keys() - applicationTargets):
            errors.append(f"missing application target: {missing}")
        for unexpected in sorted(applicationTargets - expectedTargets.keys()):
            errors.append(f"unexpected application target: {unexpected}")
    objectTargets = {value[2] for value in expectedTargets.values()}
    for target in sorted(objectTargets - sourcesByTarget.keys()):
        errors.append(f"missing variant object target: {target}")
    checkedObjects = set()
    checkedLinks = 0
    for application, (configuration, bits, objects) in expectedTargets.items():
        expectedDefinitions = configurationDefinitions(configuration, bits)
        for target in (application, objects):
            if target in checkedObjects:
                continue
            if target == objects:
                checkedObjects.add(target)
            sources = set(sourcesByTarget.get(target, {}))
            if target == application:
                if sources != {"src/main.cpp"}:
                    errors.append(f"{target}: executable must compile only src/main.cpp")
            else:
                for missing in sorted(requiredSources(configuration, cuda) - sources):
                    errors.append(f"{target}: required source missing: {missing}")
                _, solver, _, order = configuration
                instances = {Path(source).name for source in sources
                             if Path(source).name.startswith("fmm_instance_")}
                expectedInstances = set()
                if solver in ("fmm", "pmmm_fmm"):
                    expectedInstances.add(f"fmm_instance_cpu_{order}.cpp")
                    if cuda:
                        expectedInstances.add(f"fmm_instance_cuda_{order}.cu")
                if instances != expectedInstances:
                    errors.append(f"{target}: generated FMM instances {sorted(instances)}, "
                                  f"expected {sorted(expectedInstances)}")
                for duplicate in sorted(coreSources & sources):
                    errors.append(f"{target}: recompiles shared core source: {duplicate}")
                for error in sourceSelectionErrors(sources, configuration):
                    errors.append(f"{target}: {error}")
            for source, definitions in definitionsByTarget.get(target, []):
                for name, value in expectedDefinitions.items():
                    if definitions.get(name) != {value}:
                        errors.append(f"{target}: {source} needs {name}={value}, "
                                      f"found {sorted(definitions.get(name, set()))}")
        linkPath = buildDirectory / "CMakeFiles" / (application + ".dir") / "link.txt"
        if not linkPath.exists():
            if setting("CMAKE_GENERATOR") == "Unix Makefiles":
                errors.append(f"{application}: generated Makefile link command is missing")
            continue
        checkedLinks += 1
        linkCommand = linkPath.read_text()
        linkedObjects = {match.group(1) for match in TARGET_PATTERN.finditer(linkCommand)
                         if match.group(1).endswith("_objects")}
        if linkedObjects != {objects}:
            errors.append(f"{application}: links variant objects {sorted(linkedObjects)}, "
                          f"expected {objects}")
        linkedLibraries = {Path(argument).name[3:-2] for argument in shlex.split(linkCommand)
                           if Path(argument).name.startswith("lib") and argument.endswith(".a")}
        if "cosmictiger_core" not in linkedLibraries:
            errors.append(f"{application}: does not link the shared cosmictiger_core library")
        linkedSources = set()
        for library in linkedLibraries:
            linkedSources.update(sourcesByTarget.get(library, {}))
        if "src/chemistry/recombination.cpp" not in linkedSources:
            errors.append(f"{application}: lacks shared homogeneous recombination")
        if configuration[2]:
            chemistrySources = {"src/chemistry/primordial_chemistry.cpp"}
            if cuda:
                chemistrySources.add("src/chemistry/primordial_chemistry.cu")
            for missing in sorted(chemistrySources - linkedSources):
                errors.append(f"{application}: lacks particle chemistry source {missing}")
        for error in sourceSelectionErrors(linkedSources, configuration):
            errors.append(f"{application} linked libraries: {error}")

    # Standalone tests still follow the selected default configuration, while
    # application tests belong to each variant's own object library.
    for target, sources in sourcesByTarget.items():
        if target in expectedTargets or target in objectTargets or not any(
                source.startswith("tests/") for source in sources):
            continue
        for error in sourceSelectionErrors(sources, selectedConfiguration):
            errors.append(f"{target}: {error}")
    if setting("CMAKE_GENERATOR") == "Unix Makefiles":
        makefilePath = buildDirectory / "CMakeFiles/Makefile2"
        if makefilePath.exists():
            errors.extend(makefileDefaultErrors(makefilePath, expectedTargets, selectedName))
        else:
            errors.append("generated Makefile2 is missing")
    if errors:
        raise ValueError("Build feature checks failed:\n  " + "\n  ".join(errors))
    linkSummary = (f"{checkedLinks} Makefile link commands checked" if checkedLinks else
                   "link checking unavailable for this generator")
    if setting("CMAKE_GENERATOR") == "Unix Makefiles":
        linkSummary += f"; default all selects only {selectedName}"
    return (f"Build variants passed: {len(variants)} physics variants, "
            f"{len(expectedTargets)} application targets, {len(coreSources)} shared core sources; "
            f"positions={positionType}, CUDA={cuda}, tests={testing}; "
            f"{len(commands)} compilation entries and {linkSummary}")


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
