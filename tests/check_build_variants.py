# CosmicTiger - enabled build graph checker regression; BSL-1.0.

"""Ensure capability graph checks catch missing sources and disabled algorithms."""

import json
from itertools import product
from pathlib import Path
import tempfile
import subprocess
import unittest

from check_build_features import (FEATURES, LEGACY_DEFINITIONS, checkBuild, configurationDefinitions,
                                  degreeRange, hasChemistry, hasTree, requiredSources)


class BuildFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.buildDirectory = Path(self.temporary.name)
        self.sourceDirectory = self.buildDirectory / "source"
        self.configure()

    def configure(self, **changes):
        self.features = {feature: feature in ("FMM", "SPH", "CHEMISTRY", "INTEGRATOR_TRADITIONAL") for feature in FEATURES}
        self.features.update(changes)
        self.degrees = (4, 6)
        self.commands = []
        self.cache = {"CMAKE_HOME_DIRECTORY": str(self.sourceDirectory), "CMAKE_GENERATOR": "Unix Makefiles",
                      "COSMICTIGER_EXPANSION_DEGREE": "4-6", "COSMICTIGER_POSITION_TYPE": "fixed32",
                      "COSMICTIGER_WITH_CUDA": "OFF", "BUILD_TESTING": "OFF"}
        self.cache.update({"COSMICTIGER_ENABLE_" + key: "ON" if value else "OFF"
                           for key, value in self.features.items()})
        self.addCompilation("cosmictiger_core", "src/core/flops.cpp", {})
        self.addCompilation("cosmictiger_recombination", "src/chemistry/recombination.cpp", {})
        if hasChemistry(self.features):
            self.addCompilation("cosmictiger_sph_chemistry", "src/chemistry/primordial_chemistry.cpp", {})
        self.definitions = configurationDefinitions(self.features, 32, self.degrees)
        for source in sorted(requiredSources(self.features, False)):
            self.addCompilation("cosmictiger_objects", source, self.definitions)
        if hasTree(self.features):
            for degree in range(self.degrees[0], self.degrees[1] + 1):
                self.addCompilation("cosmictiger_objects", f"instances/fmm_instance_cpu_{degree + 1}.cpp",
                                    self.definitions)
        self.addCompilation("cosmictiger", "src/main.cpp", self.definitions)
        self.linkPath = self.buildDirectory / "CMakeFiles/cosmictiger.dir/link.txt"
        self.linkPath.parent.mkdir(parents=True, exist_ok=True)
        self.linkPath.write_text("c++ CMakeFiles/cosmictiger.dir/src/main.cpp.o "
                                "CMakeFiles/cosmictiger_objects.dir/src/core/driver.cpp.o "
                                "libcosmictiger_core.a libcosmictiger_recombination.a\n" +
                                ("libcosmictiger_sph_chemistry.a\n" if hasChemistry(self.features) else ""))

    def addCompilation(self, target, source, definitions):
        self.commands.append({"directory": str(self.buildDirectory),
                              "file": str(self.sourceDirectory / source),
                              "arguments": ["c++", *[f"-D{name}={value}" for name, value in definitions.items()],
                                            "-o", f"CMakeFiles/{target}.dir/{source}.o",
                                            "-c", str(self.sourceDirectory / source)]})

    def check(self):
        (self.buildDirectory / "CMakeCache.txt").write_text("".join(
            f"{name}:STRING={value}\n" for name, value in self.cache.items()))
        (self.buildDirectory / "compile_commands.json").write_text(json.dumps(self.commands))
        return checkBuild(self.buildDirectory)

    def testDefaultRange(self):
        self.assertIn("one application", self.check())
        self.assertIn("potential degrees=4-6", self.check())

    def testCombinedSolvers(self):
        self.configure(PMMM_PP=True, PMMM_FMM=True, INTEGRATOR_HIERARCHICAL=True,
                       INTEGRATOR_GLOBAL=True)
        self.assertIn("Build features passed", self.check())

    def testPmmmPpWithoutTree(self):
        self.configure(FMM=False, PMMM_PP=True, SPH=False)
        self.assertIn("Build features passed", self.check())

    def testPmmmNearFmmWithoutGlobalFmm(self):
        self.configure(FMM=False, PMMM_FMM=True, SPH=False)
        self.assertIn("Build features passed", self.check())

    def testBothPmmmSolversWithoutGlobalFmm(self):
        self.configure(FMM=False, PMMM_PP=True, PMMM_FMM=True, SPH=False)
        self.assertIn("Build features passed", self.check())

    def testSphOnly(self):
        self.configure(FMM=False)
        self.assertIn("Build features passed", self.check())
        self.assertEqual(self.definitions["ORDER"], "0")

    def testSingleDegree(self):
        self.cache["COSMICTIGER_EXPANSION_DEGREE"] = "6"
        self.commands = [entry for entry in self.commands if not any(
            entry["file"].endswith(f"fmm_instance_cpu_{order}.cpp") for order in (5, 6))]
        for entry in self.commands:
            entry["arguments"] = [arg.replace("FMM_MIN_ORDER=5", "FMM_MIN_ORDER=7") for arg in entry["arguments"]]
        self.assertIn("potential degrees=6-6", self.check())

    def testBounds(self):
        self.assertEqual(degreeRange("2-8"), (2, 8))
        for invalid in ("1", "9", "6-4", "4;5;6", "4--6", "4-9", ""):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                degreeRange(invalid)

    def testMissingEnabledSolver(self):
        for feature in ("PMMM_PP", "PMMM_FMM"):
            with self.subTest(feature=feature):
                self.configure(FMM=False, **{feature: True})
                self.commands = [entry for entry in self.commands if not entry["file"].endswith("nitadori/solver.cpp")]
                with self.assertRaisesRegex(ValueError, "required source missing: src/gravity/nitadori/solver.cpp"):
                    self.check()

    def testMissingEnabledNearSolver(self):
        for feature, source in (("PMMM_PP", "near_pp.cpp"), ("PMMM_FMM", "fmm_batch.cpp")):
            with self.subTest(feature=feature):
                self.configure(FMM=False, **{feature: True})
                self.commands = [entry for entry in self.commands if not entry["file"].endswith("nitadori/" + source)]
                with self.assertRaisesRegex(ValueError, "required source missing: src/gravity/nitadori/" + source):
                    self.check()

    def testPmmmSourceLeak(self):
        self.addCompilation("cosmictiger_objects", "src/gravity/nitadori/solver.cpp", self.definitions)
        with self.assertRaisesRegex(ValueError, "PMMM disabled but compiles solver source"):
            self.check()

    def testFmmSourceLeak(self):
        self.configure(FMM=False, PMMM_PP=True)
        self.addCompilation("cosmictiger_objects", "src/gravity/dehnen/execution.cpp", self.definitions)
        with self.assertRaisesRegex(ValueError, "FMM disabled but compiles solver source"):
            self.check()

    def testNearSolverLeak(self):
        for enabled, disabled, source in (("PMMM_PP", "PMMM_FMM", "cell_tree.cpp"),
                                           ("PMMM_FMM", "PMMM_PP", "near_pp.cpp")):
            with self.subTest(disabled=disabled):
                self.configure(FMM=False, **{enabled: True})
                self.addCompilation("cosmictiger_objects", "src/gravity/nitadori/" + source, self.definitions)
                with self.assertRaisesRegex(ValueError, disabled + " disabled but compiles near solver"):
                    self.check()

    def testSphSourceLeak(self):
        self.configure(SPH=False)
        self.addCompilation("cosmictiger_objects", "src/sph/sph.cpp", self.definitions)
        with self.assertRaisesRegex(ValueError, "SPH disabled but compiles SPH source"):
            self.check()

    def testUnlinkedChemistryLeak(self):
        self.configure(SPH=False)
        self.addCompilation("unused_chemistry", "src/chemistry/primordial_chemistry.cpp", {})
        with self.assertRaisesRegex(ValueError, "particle chemistry disabled but compiles source"):
            self.check()

    def testChemistryCapabilityCombinations(self):
        for sph, chemistry in product((False, True), repeat=2):
            with self.subTest(sph=sph, chemistry=chemistry):
                self.configure(SPH=sph, CHEMISTRY=chemistry)
                self.assertIn("Build features passed", self.check())
                self.assertEqual(self.definitions["COSMICTIGER_ENABLE_CHEMISTRY"], str(int(chemistry)))

    def testDisabledChemistrySourceLeak(self):
        for source in ("src/chemistry/primordial_chemistry.cpp", "src/chemistry/primordial_chemistry.cu",
                       "src/sph/sph_chemistry.cpp", "tests/sph/chemistry.cpp",
                       "tests/primordial_rates.cpp", "tests/primordial_solver.cpp", "tests/primordial_units.cpp",
                       "tests/primordial_reference.cpp", "tests/primordial_cuda.cu"):
            with self.subTest(source=source):
                self.configure(CHEMISTRY=False)
                self.addCompilation("unused_chemistry", source, {})
                with self.assertRaisesRegex(ValueError, "particle chemistry disabled but compiles source: " + source):
                    self.check()

    def testMissingSphChemistrySource(self):
        self.commands = [entry for entry in self.commands if not entry["file"].endswith("sph/sph_chemistry.cpp")]
        with self.assertRaisesRegex(ValueError, "required source missing: src/sph/sph_chemistry.cpp"):
            self.check()

    def testSphOnlyGravitySourceLeak(self):
        self.configure(FMM=False)
        self.addCompilation("cosmictiger_objects", "src/gravity/reference_table.cpp", self.definitions)
        with self.assertRaisesRegex(ValueError, "gravity disabled but compiles gravity source"):
            self.check()

    def testTreeLeak(self):
        self.configure(FMM=False, PMMM_PP=True)
        self.addCompilation("cosmictiger_objects", "src/gravity/tree_build.cpp", self.definitions)
        with self.assertRaisesRegex(ValueError, "tree algorithms disabled but compiles tree source"):
            self.check()

    def testMissingDegreeInstantiation(self):
        self.commands = [entry for entry in self.commands if not entry["file"].endswith("fmm_instance_cpu_6.cpp")]
        with self.assertRaisesRegex(ValueError, "generated FMM instances"):
            self.check()

    def testExtraneousDegreeInstantiation(self):
        self.addCompilation("cosmictiger_objects", "instances/fmm_instance_cpu_8.cpp", self.definitions)
        with self.assertRaisesRegex(ValueError, "generated FMM instances"):
            self.check()

    def testDuplicateDegreeInstantiation(self):
        self.addCompilation("cosmictiger_objects", "instances/fmm_instance_cpu_7.cpp", self.definitions)
        with self.assertRaisesRegex(ValueError, "generated FMM instances"):
            self.check()

    def testWrongDefinition(self):
        self.commands[-1]["arguments"].remove("-DCOSMICTIGER_ENABLE_FMM=1")
        with self.assertRaisesRegex(ValueError, "needs COSMICTIGER_ENABLE_FMM=1"):
            self.check()

    def testLegacyDefinition(self):
        for name in sorted(LEGACY_DEFINITIONS):
            with self.subTest(name=name):
                self.configure()
                self.commands[-1]["arguments"].append("-D" + name + "=0")
                with self.assertRaisesRegex(ValueError, "retains legacy selection definition " + name):
                    self.check()

    def testCoreDefinitionLeak(self):
        self.commands[0]["arguments"].append("-DORDER=7")
        with self.assertRaisesRegex(ValueError, "inherits feature definition ORDER"):
            self.check()

    def testDuplicateCoreCompilation(self):
        self.addCompilation("cosmictiger_objects", "src/core/flops.cpp", self.definitions)
        with self.assertRaisesRegex(ValueError, "recompiles shared core source"):
            self.check()

    def testWrongLink(self):
        self.linkPath.write_text(self.linkPath.read_text().replace("cosmictiger_objects", "old_objects"))
        with self.assertRaisesRegex(ValueError, "links unexpected object targets"):
            self.check()

    def testMissingChemistry(self):
        self.linkPath.write_text(self.linkPath.read_text().replace("libcosmictiger_sph_chemistry.a", ""))
        with self.assertRaisesRegex(ValueError, "lacks particle chemistry source"):
            self.check()

    def testMissingRecombinationWithoutChemistry(self):
        self.configure(CHEMISTRY=False)
        self.linkPath.write_text(self.linkPath.read_text().replace("libcosmictiger_recombination.a", ""))
        with self.assertRaisesRegex(ValueError, "lacks shared homogeneous recombination"):
            self.check()

    def testOldNamedApplication(self):
        self.addCompilation("cosmictiger_leapfrog_fmm_7", "src/main.cpp", self.definitions)
        with self.assertRaisesRegex(ValueError, "unexpected application target"):
            self.check()

    def testUnusedNamedObjectTarget(self):
        self.addCompilation("cosmictiger_leapfrog_fmm_7_objects", "src/core/driver.cpp", self.definitions)
        with self.assertRaisesRegex(ValueError, "unexpected physics object target"):
            self.check()

    def testFixed64(self):
        self.cache["COSMICTIGER_POSITION_TYPE"] = "fixed64"
        for entry in self.commands:
            entry["arguments"] = [arg.replace("COSMICTIGER_POSITION_BITS=32", "COSMICTIGER_POSITION_BITS=64")
                                  for arg in entry["arguments"]]
        self.assertIn("positions=fixed64", self.check())

    def testWrongPositionWidth(self):
        self.commands[-1]["arguments"].append("-DCOSMICTIGER_POSITION_BITS=64")
        with self.assertRaisesRegex(ValueError, "needs COSMICTIGER_POSITION_BITS=32"):
            self.check()


class FeatureConfigurationTest(unittest.TestCase):
    def configure(self, values):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "features.cmake"
            source = Path(__file__).resolve().parents[1] / "cmake/features.cmake"
            script.write_text("\n".join(f"set({key} {value} CACHE STRING \"test\")"
                                        for key, value in values.items()) +
                              f'\ninclude("{source}")\n' +
                              'message(STATUS "range=${COSMICTIGER_FMM_MIN_ORDER}-${COSMICTIGER_FMM_MAX_ORDER}")\n'
                              'message(STATUS "gravity=${COSMICTIGER_WITH_GRAVITY}")\n'
                              'message(STATUS "default_gravity=${cosmictiger_default_gravity_method}")\n'
                              'message(STATUS "default_near=${cosmictiger_default_near_method}")\n'
                              'message(STATUS "pmmm_pp=${COSMICTIGER_ENABLE_PMMM_PP}")\n'
                              'message(STATUS "pmmm_fmm=${COSMICTIGER_ENABLE_PMMM_FMM}")\n'
                              'message(STATUS "chemistry=${COSMICTIGER_ENABLE_CHEMISTRY}")\n')
            return subprocess.run(["cmake", "-P", str(script)], text=True, capture_output=True)

    def testDefaultRange(self):
        result = self.configure({})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("range=5-7", result.stdout)
        self.assertIn("pmmm_pp=OFF", result.stdout)
        self.assertIn("pmmm_fmm=OFF", result.stdout)
        self.assertIn("default_gravity=fmm", result.stdout)
        self.assertIn("default_near=none", result.stdout)
        self.assertIn("chemistry=ON", result.stdout)

    def testChemistryCapabilityCombinations(self):
        for sph, chemistry in product(("OFF", "ON"), repeat=2):
            with self.subTest(sph=sph, chemistry=chemistry):
                result = self.configure({"COSMICTIGER_ENABLE_SPH": sph,
                                         "COSMICTIGER_ENABLE_CHEMISTRY": chemistry})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("chemistry=" + chemistry, result.stdout)

    def testSingleAndBoundaryDegrees(self):
        for value, orders in (("2", "3-3"), ("8", "9-9"), ("2-8", "3-9")):
            with self.subTest(value=value):
                result = self.configure({"COSMICTIGER_EXPANSION_DEGREE": value})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("range=" + orders, result.stdout)

    def testInvalidRangesFailBeforeDependencies(self):
        for value in ("1", "9", "6-4", "4-9", "4--6"):
            with self.subTest(value=value):
                result = self.configure({"COSMICTIGER_EXPANSION_DEGREE": value})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("COSMICTIGER_EXPANSION_DEGREE", result.stderr)

    def testRetiredCacheRequiresExplicitMigration(self):
        for key, value in (("COSMICTIGER_WITH_ORDER", "7"), ("COSMICTIGER_WITH_SPH", "OFF"),
                           ("COSMICTIGER_GRAVITY_METHOD", "pmmm"),
                           *product(("COSMICTIGER_ENABLE_PMMM", "COSMICTIGER_ENABLE_PMMM_NEAR_PP",
                                     "COSMICTIGER_ENABLE_PMMM_NEAR_FMM"), ("ON", "OFF"))):
            with self.subTest(key=key, value=value):
                result = self.configure({key: value})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Retired CosmicTiger cache options", result.stderr)
                self.assertIn(key, result.stderr)
                self.assertIn("-U", result.stderr)

    def testNoIntegrator(self):
        result = self.configure({"COSMICTIGER_ENABLE_INTEGRATOR_TRADITIONAL": "OFF"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Enable at least one COSMICTIGER_ENABLE_INTEGRATOR", result.stderr)

    def testGravityCapabilityCombinations(self):
        for fmm, pmmmPp, pmmmFmm in product((False, True), repeat=3):
            with self.subTest(fmm=fmm, pmmmPp=pmmmPp, pmmmFmm=pmmmFmm):
                result = self.configure({"COSMICTIGER_ENABLE_" + feature: "ON" if enabled else "OFF"
                                         for feature, enabled in (("FMM", fmm), ("PMMM_PP", pmmmPp),
                                                                  ("PMMM_FMM", pmmmFmm))})
                self.assertEqual(result.returncode, 0, result.stderr)
                gravity = fmm or pmmmPp or pmmmFmm
                self.assertIn("gravity=" + ("ON" if gravity else "OFF"), result.stdout)
                self.assertIn("range=" + ("5-7" if gravity else "0-0"), result.stdout)
                self.assertIn("default_gravity=" + ("fmm" if fmm else "pmmm" if gravity else "none"), result.stdout)
                self.assertIn("default_near=" + ("pp" if pmmmPp else "fmm" if pmmmFmm else "none"), result.stdout)

    def testNoPhysics(self):
        result = self.configure({"COSMICTIGER_ENABLE_FMM": "OFF", "COSMICTIGER_ENABLE_SPH": "OFF"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Enable at least one", result.stderr)
        self.assertIn("COSMICTIGER_ENABLE_PMMM_PP", result.stderr)
        self.assertIn("COSMICTIGER_ENABLE_PMMM_FMM", result.stderr)

    def testGlobalSphOnly(self):
        result = self.configure({"COSMICTIGER_ENABLE_FMM": "OFF",
                                 "COSMICTIGER_ENABLE_INTEGRATOR_TRADITIONAL": "OFF",
                                 "COSMICTIGER_ENABLE_INTEGRATOR_GLOBAL": "ON"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("range=0-0", result.stdout)


if __name__ == "__main__":
    unittest.main()
