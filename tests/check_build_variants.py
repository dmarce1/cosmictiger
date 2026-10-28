# CosmicTiger - build matrix graph checker regression; BSL-1.0.

"""Ensure the graph checks catch cross-variant configuration and source leaks."""

import json
from pathlib import Path
import tempfile
import unittest

from check_build_features import checkBuild, configurationDefinitions, namedVariants, requiredSources


class BuildVariantsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.buildDirectory = Path(self.temporary.name)
        self.sourceDirectory = self.buildDirectory / "source"
        self.default = "cosmictiger_leapfrog_fmm_sph_7"
        self.commands = []
        cache = {"CMAKE_HOME_DIRECTORY": str(self.sourceDirectory), "CMAKE_GENERATOR": "Unix Makefiles",
                 "COSMICTIGER_TIME_INTEGRATOR": "leapfrog", "COSMICTIGER_GRAVITY_METHOD": "fmm",
                 "COSMICTIGER_PMMM_NEAR_METHOD": "pp", "COSMICTIGER_WITH_ORDER": "7",
                 "COSMICTIGER_POSITION_TYPE": "fixed32",
                 "COSMICTIGER_WITH_SPH": "ON", "COSMICTIGER_WITH_CUDA": "OFF", "BUILD_TESTING": "OFF"}
        (self.buildDirectory / "CMakeCache.txt").write_text("".join(
            f"{name}:STRING={value}\n" for name, value in cache.items()))
        self.addCompilation("cosmictiger_core", "src/core/flops.cpp", {})
        self.addCompilation("cosmictiger_recombination", "src/chemistry/recombination.cpp", {})
        self.addCompilation("cosmictiger_sph_chemistry", "src/chemistry/primordial_chemistry.cpp", {})
        variants = namedVariants()
        for base, configuration in variants.items():
            objects = base + "_objects"
            definitions = configurationDefinitions(configuration, 32)
            for source in sorted(requiredSources(configuration, False)):
                self.addCompilation(objects, source, definitions)
            if configuration[1] in ("fmm", "pmmm_fmm"):
                self.addCompilation(objects,
                                    f"instances/fmm_instance_cpu_{configuration[3]}.cpp",
                                    definitions)
            self.addApplication(base, objects, definitions)
        self.addApplication("cosmictiger", self.default + "_objects",
                            configurationDefinitions(variants[self.default], 32))
        self.makefilePath = self.buildDirectory / "CMakeFiles/Makefile2"
        self.makefilePath.write_text(
            "all: CMakeFiles/cosmictiger.dir/all\n"
            f"CMakeFiles/cosmictiger.dir/all: CMakeFiles/{self.default}_objects.dir/all\n")

    def addCompilation(self, target, source, definitions):
        self.commands.append({"directory": str(self.buildDirectory),
                              "file": str(self.sourceDirectory / source),
                              "arguments": ["c++", *[f"-D{name}={value}"
                                             for name, value in definitions.items()],
                                            "-o", f"CMakeFiles/{target}.dir/{source}.o",
                                            "-c", str(self.sourceDirectory / source)]})

    def addApplication(self, application, objects, definitions):
        self.addCompilation(application, "src/main.cpp", definitions)
        linkPath = self.buildDirectory / "CMakeFiles" / (application + ".dir") / "link.txt"
        linkPath.parent.mkdir(parents=True)
        linkPath.write_text(f"c++ CMakeFiles/{application}.dir/src/main.cpp.o "
                           f"CMakeFiles/{objects}.dir/src/core/driver.cpp.o "
                           "libcosmictiger_core.a libcosmictiger_recombination.a\n")
        if definitions["COSMICTIGER_WITH_SPH"] == "1":
            linkPath.write_text(linkPath.read_text() + "libcosmictiger_sph_chemistry.a\n")

    def check(self):
        (self.buildDirectory / "compile_commands.json").write_text(json.dumps(self.commands))
        return checkBuild(self.buildDirectory)

    def testCompleteMatrix(self):
        self.assertEqual(len(namedVariants()), 63)
        self.assertIn("63 physics variants", self.check())
        self.assertIn("64 Makefile link commands checked", self.check())
        self.assertIn("positions=fixed32", self.check())

    def testFixed64Matrix(self):
        cachePath = self.buildDirectory / "CMakeCache.txt"
        cachePath.write_text(cachePath.read_text().replace("=fixed32", "=fixed64"))
        for command in self.commands:
            command["arguments"] = [argument.replace("COSMICTIGER_POSITION_BITS=32",
                                                       "COSMICTIGER_POSITION_BITS=64")
                                    for argument in command["arguments"]]
        self.assertIn("positions=fixed64", self.check())
        self.assertIn("64 Makefile link commands checked", self.check())

    def testSphFamilies(self):
        variants = namedVariants()
        for gravity in ("fmm", "pmmm_pp", "pmmm_fmm"):
            for order in range(3, 10):
                self.assertIn(f"cosmictiger_leapfrog_{gravity}_sph_{order}", variants)
                self.assertNotIn(f"cosmictiger_hierarchical_{gravity}_sph_{order}", variants)

    def testMissingPmmmSphHydro(self):
        self.commands = [command for command in self.commands if not (
            "cosmictiger_leapfrog_pmmm_pp_sph_7_objects.dir" in " ".join(command["arguments"]) and
            command["file"].endswith("src/sph/sph_hydro_cpu.cpp"))]
        with self.assertRaisesRegex(ValueError, "required source missing: src/sph/sph_hydro_cpu.cpp"):
            self.check()

    def testInvalidPositionType(self):
        cachePath = self.buildDirectory / "CMakeCache.txt"
        cachePath.write_text(cachePath.read_text().replace("=fixed32", "=float64"))
        with self.assertRaisesRegex(ValueError, "invalid COSMICTIGER_POSITION_TYPE"):
            self.check()

    def testWrongIntegratorDefinition(self):
        for command in self.commands:
            if "cosmictiger_hierarchical_pmmm_pp_nosph_7_objects.dir" in " ".join(command["arguments"]):
                command["arguments"].remove("-DCOSMICTIGER_INTEGRATOR_HIERARCHICAL=1")
                command["arguments"].append("-DCOSMICTIGER_INTEGRATOR_HIERARCHICAL=0")
                break
        with self.assertRaisesRegex(ValueError, "needs COSMICTIGER_INTEGRATOR_HIERARCHICAL=1"):
            self.check()

    def testWrongPositionWidth(self):
        for command in self.commands:
            if "cosmictiger_leapfrog_pmmm_pp_nosph_7.dir" in " ".join(command["arguments"]):
                command["arguments"].remove("-DCOSMICTIGER_POSITION_BITS=32")
                command["arguments"].append("-DCOSMICTIGER_POSITION_BITS=64")
                break
        with self.assertRaisesRegex(ValueError, "needs COSMICTIGER_POSITION_BITS=32"):
            self.check()

    def testStandaloneWrongPositionWidth(self):
        self.addCompilation("drift_cpu_test", "src/particles/drift_cpu.cpp",
                            {"COSMICTIGER_POSITION_BITS": "64"})
        with self.assertRaisesRegex(ValueError, "drift_cpu_test: .*needs COSMICTIGER_POSITION_BITS=32"):
            self.check()

    def testStandaloneConflictingPositionWidths(self):
        self.addCompilation("drift_cpu_test", "src/particles/drift_cpu.cpp",
                            {"COSMICTIGER_POSITION_BITS": "32"})
        self.commands[-1]["arguments"].append("-DCOSMICTIGER_POSITION_BITS=64")
        with self.assertRaisesRegex(ValueError, "drift_cpu_test: .*found \\['32', '64'\\]"):
            self.check()

    def testUnwantedCompanion(self):
        self.addApplication(self.default + "_fixed64", self.default + "_objects",
                            configurationDefinitions(namedVariants()[self.default], 64))
        with self.assertRaisesRegex(ValueError, "unexpected application target"):
            self.check()

    def testNearSolverLeak(self):
        configuration = ("leapfrog", "pmmm_pp", False, 7)
        self.addCompilation("cosmictiger_leapfrog_pmmm_pp_nosph_7_objects",
                            "src/gravity/nitadori/cell_tree.cpp",
                            configurationDefinitions(configuration, 32))
        with self.assertRaisesRegex(ValueError, "compiles the other near solver"):
            self.check()

    def testSphSourceLeak(self):
        configuration = ("leapfrog", "fmm", False, 7)
        self.addCompilation("cosmictiger_leapfrog_fmm_nosph_7_objects", "src/sph/sph.cpp",
                            configurationDefinitions(configuration, 32))
        with self.assertRaisesRegex(ValueError, "SPH disabled but compiles SPH source"):
            self.check()

    def testCoreDefinitionLeak(self):
        self.commands[0]["arguments"].append("-DORDER=7")
        with self.assertRaisesRegex(ValueError, "inherits variant definition ORDER"):
            self.check()

    def testCorePositionWidthLeak(self):
        self.commands[0]["arguments"].append("-DCOSMICTIGER_POSITION_BITS=64")
        with self.assertRaisesRegex(ValueError, "inherits variant definition COSMICTIGER_POSITION_BITS"):
            self.check()

    def testDuplicateCoreCompilation(self):
        configuration = ("leapfrog", "fmm", True, 7)
        self.addCompilation(self.default + "_objects", "src/core/flops.cpp",
                            configurationDefinitions(configuration, 32))
        with self.assertRaisesRegex(ValueError, "recompiles shared core source"):
            self.check()

    def testMissingVariant(self):
        self.commands = [command for command in self.commands if
                         "cosmictiger_hierarchical_pmmm_fmm_nosph_9.dir" not in
                         " ".join(command["arguments"])]
        with self.assertRaisesRegex(ValueError, "missing application target"):
            self.check()

    def testWrongLegacyObjects(self):
        linkPath = self.buildDirectory / "CMakeFiles/cosmictiger.dir/link.txt"
        linkPath.write_text(linkPath.read_text().replace(self.default + "_objects",
                           "cosmictiger_hierarchical_pmmm_pp_nosph_3_objects"))
        with self.assertRaisesRegex(ValueError, "links variant objects"):
            self.check()

    def testWrongGeneratedOrder(self):
        for command in self.commands:
            if "cosmictiger_hierarchical_fmm_nosph_9_objects.dir" in " ".join(command["arguments"]) and \
                    command["file"].endswith("fmm_instance_cpu_9.cpp"):
                command["file"] = command["file"].replace("fmm_instance_cpu_9.cpp", "fmm_instance_cpu_7.cpp")
                break
        with self.assertRaisesRegex(ValueError, "generated FMM instances"):
            self.check()

    def testUnnecessaryPmmmPpInstantiation(self):
        configuration = ("leapfrog", "pmmm_pp", False, 7)
        self.addCompilation("cosmictiger_leapfrog_pmmm_pp_nosph_7_objects",
                            "instances/fmm_instance_cpu_7.cpp",
                            configurationDefinitions(configuration, 32))
        with self.assertRaisesRegex(ValueError, "generated FMM instances"):
            self.check()

    def testMissingSphChemistry(self):
        linkPath = self.buildDirectory / "CMakeFiles" / (self.default + ".dir/link.txt")
        linkPath.write_text(linkPath.read_text().replace("libcosmictiger_sph_chemistry.a", ""))
        with self.assertRaisesRegex(ValueError, "lacks particle chemistry source"):
            self.check()

    def testEagerNamedApplication(self):
        self.makefilePath.write_text(self.makefilePath.read_text() +
                                    f"all: CMakeFiles/{self.default}.dir/all\n")
        with self.assertRaisesRegex(ValueError, "root all eagerly builds named variant target"):
            self.check()

    def testEagerSelectedObjects(self):
        self.makefilePath.write_text(self.makefilePath.read_text() +
                                    f"all: CMakeFiles/{self.default}_objects.dir/all\n")
        with self.assertRaisesRegex(ValueError, "root all eagerly builds named variant target"):
            self.check()

    def testIndirectUnselectedObjects(self):
        self.makefilePath.write_text(self.makefilePath.read_text() +
                                    "all: CMakeFiles/cosmictiger_core.dir/all\n"
                                    "CMakeFiles/cosmictiger_core.dir/all: "
                                    "CMakeFiles/cosmictiger_hierarchical_pmmm_pp_nosph_3_objects.dir/all\n")
        with self.assertRaisesRegex(ValueError, "root all reaches an unselected variant target"):
            self.check()

    def testMissingLegacyObjectDependency(self):
        self.makefilePath.write_text(self.makefilePath.read_text().replace(
            f"CMakeFiles/cosmictiger.dir/all: CMakeFiles/{self.default}_objects.dir/all\n", ""))
        with self.assertRaisesRegex(ValueError, "Makefile dependency omits selected objects"):
            self.check()


if __name__ == "__main__":
    unittest.main()
