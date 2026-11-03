# CosmicTiger - enabled build graph checker regression; BSL-1.0.

"""Ensure capability graph checks catch missing sources and disabled algorithms."""

import json
import re
from itertools import product
from pathlib import Path
import tempfile
import subprocess
import unittest

from check_build_features import (FEATURES, LEGACY_DEFINITIONS, checkBuild, configurationDefinitions,
                                  degreeRange, hasChemistry, hasGravity, hasTree, requiredSources,
                                  sourceSelectionErrors)


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
        self.cache["COSMICTIGER_ENABLE_STAR_FORMATION"] = self.cache["COSMICTIGER_ENABLE_STARS"]
        self.addCompilation("cosmictiger_core", "src/core/flops.cpp", {})
        if hasGravity(self.features):
            self.addCompilation("cosmictiger_recombination", "src/chemistry/recombination.cpp", {})
            self.addCompilation("cosmictiger_cosmology", "src/cosmology/boltzmann.cpp", {})
            self.addCompilation("cosmictiger_fft", "src/cosmology/fft.cpp", {})
            self.addCompilation("cosmictiger_power", "src/analysis/power.cpp", {})
        if hasChemistry(self.features):
            self.addCompilation("cosmictiger_sph_chemistry", "src/chemistry/gadget4_cooling.cpp", {})
        bits = 64 if self.features["HOPKINS"] else 32
        self.cache["COSMICTIGER_POSITION_TYPE"] = f"fixed{bits}"
        self.definitions = configurationDefinitions(self.features, bits, self.degrees)
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
                                "libcosmictiger_core.a\n" +
                                ("libcosmictiger_cosmology.a libcosmictiger_fft.a "
                                 "libcosmictiger_power.a libcosmictiger_recombination.a\n"
                                 if hasGravity(self.features) else "") +
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

    def testHopkinsCapabilityCombinations(self):
        for sph, hopkins in ((False, False), (True, False), (True, True)):
            with self.subTest(sph=sph, hopkins=hopkins):
                self.configure(SPH=sph, HOPKINS=hopkins)
                self.assertIn("Build features passed", self.check())
                self.assertEqual(self.definitions["COSMICTIGER_ENABLE_HOPKINS"], str(int(hopkins)))

    def testHopkinsRequiresSph(self):
        self.configure(SPH=False, HOPKINS=True)
        with self.assertRaisesRegex(ValueError, "HOPKINS=ON requires COSMICTIGER_ENABLE_SPH=ON"):
            self.check()

    def testHopkinsRequiresFixed64(self):
        self.configure(HOPKINS=True)
        self.cache["COSMICTIGER_POSITION_TYPE"] = "fixed32"
        with self.assertRaisesRegex(ValueError, "HOPKINS=ON requires COSMICTIGER_POSITION_TYPE=fixed64"):
            self.check()

    def testDisabledHopkinsRejectsImplementationSources(self):
        for source in ("src/hydro/hopkins.cpp", "src/scenarios/sound_wave.cpp", "tests/hopkins.cpp"):
            with self.subTest(source=source):
                self.configure(HOPKINS=False)
                self.addCompilation("unused_hopkins", source, {})
                with self.assertRaisesRegex(ValueError, "Hopkins disabled but compiles source: " + source):
                    self.check()

    def testEnabledHopkinsRequiresRuntimeAndSharedSphSources(self):
        for source in ("src/hydro/hopkins.cpp", "src/scenarios/sound_wave.cpp",
                       "src/hydro/sph/sph_particles.cpp", "src/hydro/sph/sph_tree.cpp"):
            with self.subTest(source=source):
                self.configure(HOPKINS=True)
                self.commands = [entry for entry in self.commands if not entry["file"].endswith(source)]
                with self.assertRaisesRegex(ValueError, "required source missing: " + source):
                    self.check()

    def testHopkinsCompilerCapabilityMustMatch(self):
        self.commands[-1]["arguments"].append("-DCOSMICTIGER_ENABLE_HOPKINS=1")
        with self.assertRaisesRegex(ValueError, "needs COSMICTIGER_ENABLE_HOPKINS=0"):
            self.check()

    def testStarFormationCapabilityCombinations(self):
        for sph, starFormation in product((False, True), repeat=2):
            with self.subTest(sph=sph, starFormation=starFormation):
                self.configure(SPH=sph, STARS=starFormation)
                self.assertIn("Build features passed", self.check())
                self.assertEqual(self.definitions["COSMICTIGER_ENABLE_STARS"], str(int(sph and starFormation)))
                self.assertEqual(self.definitions["COSMICTIGER_ENABLE_STAR_FORMATION"], str(int(sph and starFormation)))

    def testStarsCacheAliasMustMatch(self):
        self.cache["COSMICTIGER_ENABLE_STAR_FORMATION"] = "ON"
        with self.assertRaisesRegex(ValueError, "must match COSMICTIGER_ENABLE_STARS"):
            self.check()

    def testStarsCompilerAliasMustMatch(self):
        self.commands[-1]["arguments"].append("-DCOSMICTIGER_ENABLE_STAR_FORMATION=1")
        with self.assertRaisesRegex(ValueError, "needs COSMICTIGER_ENABLE_STAR_FORMATION=0"):
            self.check()

    def testDisabledStarFormationRejectsEveryImplementationSource(self):
        for source in ("src/hydro/sph/star_formation.cpp", "src/hydro/sph/star_formation_cuda.cu",
                              "tests/sph/star_formation_cuda.cu", "tests/sph/star_formation.cpp", "tests/star_formation_particles.cpp",
                       "tests/springel_and_hernquist.cpp", "tests/springel_and_hernquist_cuda.cu"):
            with self.subTest(source=source):
                self.configure(STARS=False)
                self.addCompilation("unused_star_formation", source, {})
                with self.assertRaisesRegex(ValueError, "star formation disabled but compiles source: " + source):
                    self.check()

    def testEnabledStarFormationRequiresRuntimeSource(self):
        for source in ("src/hydro/sph/star_formation.cpp",):
            with self.subTest(source=source):
                self.configure(STARS=True)
                self.commands = [entry for entry in self.commands if not entry["file"].endswith(source)]
                with self.assertRaisesRegex(ValueError, "required source missing: " + source):
                    self.check()

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
                self.commands = [entry for entry in self.commands if not entry["file"].endswith("pmmm/solver.cpp")]
                with self.assertRaisesRegex(ValueError, "required source missing: src/gravity/pmmm/solver.cpp"):
                    self.check()

    def testMissingEnabledNearSolver(self):
        for feature, source in (("PMMM_PP", "pp/near_pp.cpp"), ("PMMM_FMM", "fmm/fmm_batch.cpp")):
            with self.subTest(feature=feature):
                self.configure(FMM=False, **{feature: True})
                self.commands = [entry for entry in self.commands if not entry["file"].endswith("pmmm/" + source)]
                with self.assertRaisesRegex(ValueError, "required source missing: src/gravity/pmmm/" + source):
                    self.check()

    def testPmmmSourceLeak(self):
        self.addCompilation("cosmictiger_objects", "src/gravity/pmmm/solver.cpp", self.definitions)
        with self.assertRaisesRegex(ValueError, "PMMM disabled but compiles solver source"):
            self.check()

    def testFmmSourceLeak(self):
        self.configure(FMM=False, PMMM_PP=True)
        self.addCompilation("cosmictiger_objects", "src/gravity/global_fmm/execution.cpp", self.definitions)
        with self.assertRaisesRegex(ValueError, "FMM disabled but compiles solver source"):
            self.check()

    def testNearSolverLeak(self):
        for enabled, disabled, source in (("PMMM_PP", "PMMM_FMM", "fmm/cell_tree.cpp"),
                                           ("PMMM_FMM", "PMMM_PP", "pp/near_pp.cpp")):
            with self.subTest(disabled=disabled):
                self.configure(FMM=False, **{enabled: True})
                self.addCompilation("cosmictiger_objects", "src/gravity/pmmm/" + source, self.definitions)
                with self.assertRaisesRegex(ValueError, disabled + " disabled but compiles near solver"):
                    self.check()

    def testSphSourceLeak(self):
        self.configure(SPH=False)
        self.addCompilation("cosmictiger_objects", "src/hydro/sph/sph.cpp", self.definitions)
        with self.assertRaisesRegex(ValueError, "SPH disabled but compiles SPH source"):
            self.check()

    def testUnlinkedChemistryLeak(self):
        self.configure(SPH=False)
        self.addCompilation("unused_chemistry", "src/chemistry/gadget4_cooling.cpp", {})
        with self.assertRaisesRegex(ValueError, "equilibrium cooling disabled but compiles source"):
            self.check()

    def testChemistryCapabilityCombinations(self):
        for sph, chemistry in product((False, True), repeat=2):
            with self.subTest(sph=sph, chemistry=chemistry):
                self.configure(SPH=sph, CHEMISTRY=chemistry)
                self.assertIn("Build features passed", self.check())
                self.assertEqual(self.definitions["COSMICTIGER_ENABLE_CHEMISTRY"], str(int(sph and chemistry)))

    def testDisabledChemistrySourceLeak(self):
        for source in ("src/chemistry/gadget4_cooling.cpp", "src/chemistry/gadget4_cooling.cu",
                       "src/hydro/sph/sph_chemistry.cpp", "tests/sph/chemistry.cpp",
                       "tests/gadget4_cooling.cpp"):
            with self.subTest(source=source):
                self.configure(CHEMISTRY=False)
                self.addCompilation("unused_chemistry", source, {})
                with self.assertRaisesRegex(ValueError, "equilibrium cooling disabled but compiles source: " + source):
                    self.check()

    def testMissingSphChemistrySource(self):
        self.commands = [entry for entry in self.commands if not entry["file"].endswith("sph/sph_chemistry.cpp")]
        with self.assertRaisesRegex(ValueError, "required source missing: src/hydro/sph/sph_chemistry.cpp"):
            self.check()

    def testSphOnlyGravitySourceLeak(self):
        self.configure(FMM=False)
        self.addCompilation("cosmictiger_objects", "src/gravity/reference_table.cpp", self.definitions)
        with self.assertRaisesRegex(ValueError, "gravity disabled but compiles gravity source"):
            self.check()

    def testSphOnlyRejectsCosmologySourcesAndReferences(self):
        for source in ("src/cosmology/boltzmann.cpp", "src/cosmology/zero_order.cpp",
                       "src/cosmology/background.cpp", "src/cosmology/fft.cpp",
                       "src/chemistry/recombination.cpp", "src/scenarios/cosmology.cpp",
                       "src/analysis/power.cpp", "src/analysis/power_cuda.cu",
                       "tests/recombination_reference.cpp", "tests/boltzmann_reference.cpp",
                       "tests/native_recombination.cpp", "tests/friedmann.cpp",
                       "tests/cosmology_evolution.cpp", "tests/fftw_threads.cpp",
                       "tests/linear_power.cpp", "tests/linear_growth.cpp", "tests/lpt.cpp"):
            with self.subTest(source=source):
                self.configure(FMM=False)
                self.cache["BUILD_TESTING"] = "ON"
                self.addCompilation("unused_cosmology", source, {})
                with self.assertRaisesRegex(ValueError, "gravity disabled but compiles cosmology source: " + source):
                    self.check()

    def testSphOnlyRejectsCosmologyTargetsWithoutCompiledSources(self):
        self.configure(FMM=False)
        targetPath = self.buildDirectory / "CMakeFiles/TargetDirectories.txt"
        for name in ("cosmictiger_recombination", "boltzmann_reference", "cosmictiger_cosmology",
                     "cosmictiger_fft", "cosmictiger_power"):
            with self.subTest(name=name):
                targetPath.write_text(f"{self.buildDirectory}/CMakeFiles/{name}.dir\n")
                with self.assertRaisesRegex(ValueError, "gravity disabled but configures cosmology target: " + name):
                    self.check()

    def testSphOnlyRejectsRegisteredCosmologyTests(self):
        self.configure(FMM=False)
        self.cache["BUILD_TESTING"] = "ON"
        registration = self.buildDirectory / "CTestTestfile.cmake"
        for name in ("cosmology_integration", "native_recombination_reference", "fft1_cpu",
                     "linear_growth_analysis_self_check", "power_cpu"):
            with self.subTest(name=name):
                registration.write_text(f'add_test([=[{name}]=] "unused")\n')
                with self.assertRaisesRegex(ValueError, "gravity disabled but registers cosmology test: " + name):
                    self.check()

    def testSphOnlyFollowsActiveCTestDirectories(self):
        self.configure(FMM=False)
        self.cache["BUILD_TESTING"] = "ON"
        registration = self.buildDirectory / "CTestTestfile.cmake"
        registration.write_text('subdirs("tests/cmake")\n')
        discovery = self.buildDirectory / "tests/cmake"
        discovery.mkdir(parents=True)
        (self.buildDirectory / "stale").mkdir()
        (self.buildDirectory / "stale/CTestTestfile.cmake").write_text('add_test(cosmology_stale "unused")\n')
        self.assertIn("Build features passed", self.check())
        (discovery / "CTestTestfile.cmake").write_text('include("recombination_thermal_test[1]_include.cmake")\n')
        with self.assertRaisesRegex(ValueError, "registers cosmology test: recombination_thermal_test"):
            self.check()

    def testSphOnlyRejectsLinkedFftw(self):
        self.configure(FMM=False)
        for library in ("-lfftw3f", "/usr/lib/libfftw3f_mpi.so", "/usr/lib/libfftw3f_threads.a"):
            with self.subTest(library=library):
                original = self.linkPath.read_text()
                self.linkPath.write_text(original + library + "\n")
                with self.assertRaisesRegex(ValueError, "gravity disabled but links FFTW"):
                    self.check()
                self.linkPath.write_text(original)

    def testCoreRejectsPhysicsLibrarySources(self):
        self.addCompilation("cosmictiger_core", "src/cosmology/background.cpp", {})
        with self.assertRaisesRegex(ValueError, "shared core compiles physics library source"):
            self.check()

    def testGravityRequiresSeparatePhysicsLibraries(self):
        for source in ("src/cosmology/boltzmann.cpp", "src/cosmology/fft.cpp", "src/analysis/power.cpp"):
            with self.subTest(source=source):
                self.configure()
                self.commands = [entry for entry in self.commands if not entry["file"].endswith(source)]
                self.addCompilation("cosmictiger_objects", source, self.definitions)
                with self.assertRaisesRegex(ValueError, "required library source missing: " + source):
                    self.check()

    def testSphOnlyRejectsHaloToolsAndSupportingSources(self):
        for source in ("src/tools/groups2halofunc.cpp", "src/tools/halo_catalog_export.cpp",
                       "src/tools/healpix2silo.cpp", "src/analysis/group_entry.cpp",
                       "src/analysis/compress.cpp", "src/analysis/healpix.cu", "src/math/fp16.cpp",
                       "tests/analysis_tools.cpp", "tests/halo_catalog_fixture.cpp",
                       "tests/compression.cpp", "tests/fp16.cpp", "tests/lightcone_cpu.cpp",
                       "tests/lightcone_cuda.cu", "tests/lightcone_epoch.cpp", "tests/rockstar_core.cpp",
                       "tests/rockstar_eigenpair.cpp", "tests/rockstar_properties.cpp"):
            with self.subTest(source=source):
                self.configure(FMM=False)
                self.cache["BUILD_TESTING"] = "ON"
                self.addCompilation("unused_halo", source, {})
                with self.assertRaisesRegex(ValueError, "gravity disabled but compiles gravity source: " + source):
                    self.check()

    def testEachGravitySolverRetainsHaloToolSources(self):
        for feature in ("FMM", "PMMM_PP", "PMMM_FMM"):
            with self.subTest(feature=feature):
                self.configure(**{"FMM": False, feature: True})
                self.cache["BUILD_TESTING"] = "ON"
                for source in ("src/tools/groups2halofunc.cpp", "src/tools/halo_catalog_export.cpp",
                               "src/tools/healpix2silo.cpp", "src/analysis/group_entry.cpp",
                               "src/analysis/compress.cpp", "src/math/fp16.cpp",
                               "tests/analysis_tools.cpp", "tests/halo_catalog_fixture.cpp", "tests/fp16.cpp"):
                    self.addCompilation("halo_tools", source, {})
                self.assertIn("Build features passed", self.check())

    def testPmmmPpRejectsTreeSourcesAndTests(self):
        for source in ("src/gravity/kick_workspace.cpp", "src/gravity/kick_executor.cpp",
                       "src/gravity/kick_interaction_counts.cpp", "src/gravity/source_access.cpp",
                       "src/gravity/sources.cpp", "tests/gravity/cartesian_kernels.cpp",
                       "tests/gravity/cartesian_kernels_cuda.cu", "tests/gravity/cartesian_l2l_cuda.cu",
                       "tests/gravity/fmm_acceptance.cpp", "tests/gravity/fmm_acceptance_cuda.cu",
                       "tests/gravity/fmm_geometry.cpp"):
            with self.subTest(source=source):
                self.configure(FMM=False, PMMM_PP=True)
                self.cache["BUILD_TESTING"] = "ON"
                self.addCompilation("unused_tree", source, {})
                with self.assertRaisesRegex(ValueError, "tree algorithms disabled but compiles tree source: " + source):
                    self.check()

    def testFmmRejectsParticleMeshSourcesAndTests(self):
        for source in ("src/gravity/pm/collective_dispatch.cpp", "tests/gravity/pm/fft_flops.cpp",
                       "src/particles/particle_mesh_sort.cpp", "tests/particle_mesh_sort.cpp",
                       "tests/mesh_domain.cpp", "tests/mesh_domains.cpp"):
            with self.subTest(source=source):
                self.configure()
                self.cache["BUILD_TESTING"] = "ON"
                self.addCompilation("unused_mesh", source, {})
                # Sources shared with PM/P3M report the broader mesh-solver diagnostic.
                with self.assertRaisesRegex(ValueError, "(PMMM disabled but compiles solver|"
                                            "mesh solvers disabled but compiles mesh) source: " +
                                            re.escape(source)):
                    self.check()

    def testFmmAllowsSharedParticleMeshCloud(self):
        self.cache["BUILD_TESTING"] = "ON"
        self.addCompilation("pm_cloud_test", "tests/gravity/pm/cloud.cpp", self.definitions)
        self.assertIn("Build features passed", self.check())

    def testConfiguredRuntimeMustNotMoveIntoSharedCore(self):
        self.commands = [entry for entry in self.commands if not entry["file"].endswith("src/core/hpx.cpp")]
        self.addCompilation("cosmictiger_core", "src/core/hpx.cpp", {})
        with self.assertRaisesRegex(ValueError, "required source missing: src/core/hpx.cpp"):
            self.check()

    def testCpuBuildRejectsCudaStreamImplementation(self):
        self.addCompilation("cosmictiger_core", "src/cuda/cuda_stream.cpp", {})
        with self.assertRaisesRegex(ValueError, "CUDA disabled but graph compiles src/cuda/cuda_stream.cpp"):
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
        with self.assertRaisesRegex(ValueError, "lacks equilibrium cooling source"):
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


class BuildConfigHeaderTest(unittest.TestCase):
    def compile(self, definitions):
        source = Path(__file__).resolve().parents[1]
        command = ["c++", "-std=c++20", "-fsyntax-only", "-x", "c++", "-I", str(source), "-"]
        command += [f"-D{name}={value}" for name, value in definitions.items()]
        return subprocess.run(command, input="#include <cosmictiger/core/build_config.hpp>\n",
                              text=True, capture_output=True)

    def definitions(self):
        return {"COSMICTIGER_ENABLE_" + name: int(name == "INTEGRATOR_TRADITIONAL")
                for name in ("SPH", "HOPKINS", "CHEMISTRY", "STARS", "STAR_FORMATION",
                             "INTEGRATOR_TRADITIONAL", "INTEGRATOR_HIERARCHICAL", "INTEGRATOR_GLOBAL")}

    def testExplicitCapabilitiesCompile(self):
        result = self.compile(self.definitions())
        self.assertEqual(result.returncode, 0, result.stderr)

    def testEveryMissingCapabilityFails(self):
        definitions = self.definitions()
        for missing in definitions:
            with self.subTest(missing=missing):
                result = self.compile({key: value for key, value in definitions.items() if key != missing})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Missing CosmicTiger build capabilities", result.stderr)

    def testImpossibleCapabilitiesFail(self):
        for values, diagnostic in (({"HOPKINS": 1}, "Hopkins capability requires SPH"),
                ({"CHEMISTRY": 1}, "chemistry capability requires SPH"),
                ({"STARS": 1, "STAR_FORMATION": 1}, "stars capability requires chemistry"),
                ({"STARS": 1}, "must match"),
                ({"INTEGRATOR_TRADITIONAL": 0}, "At least one time integrator")):
            with self.subTest(values=values):
                definitions = self.definitions() | {"COSMICTIGER_ENABLE_" + key: value for key, value in values.items()}
                result = self.compile(definitions)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)


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
                              'message(STATUS "hopkins=${COSMICTIGER_ENABLE_HOPKINS}")\n'
                              'message(STATUS "chemistry=${COSMICTIGER_ENABLE_CHEMISTRY}")\n'
                              'message(STATUS "effective_chemistry=${COSMICTIGER_WITH_CHEMISTRY}")\n'
                              'message(STATUS "effective_stars=${COSMICTIGER_WITH_STARS}")\n'
                              'message(STATUS "stars=${COSMICTIGER_ENABLE_STARS}")\n'
                              'message(STATUS "star_formation=${COSMICTIGER_ENABLE_STAR_FORMATION}")\n')
            return subprocess.run(["cmake", "-P", str(script)], text=True, capture_output=True)

    def testDefaultRange(self):
        result = self.configure({})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("range=5-7", result.stdout)
        self.assertIn("pmmm_pp=OFF", result.stdout)
        self.assertIn("pmmm_fmm=OFF", result.stdout)
        self.assertIn("default_gravity=fmm", result.stdout)
        self.assertIn("default_near=none", result.stdout)
        self.assertIn("hopkins=OFF", result.stdout)
        self.assertIn("chemistry=ON", result.stdout)
        self.assertIn("stars=OFF", result.stdout)
        self.assertIn("star_formation=OFF", result.stdout)

    def testHopkinsRequiresSphAndFixed64(self):
        for sph, hopkins in product(("OFF", "ON"), repeat=2):
            with self.subTest(sph=sph, hopkins=hopkins):
                result = self.configure({"COSMICTIGER_ENABLE_SPH": sph,
                                         "COSMICTIGER_ENABLE_HOPKINS": hopkins,
                                         "COSMICTIGER_POSITION_TYPE": "fixed64"})
                if hopkins == "ON" and sph == "OFF":
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("COSMICTIGER_ENABLE_HOPKINS=ON requires COSMICTIGER_ENABLE_SPH=ON", result.stderr)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("hopkins=" + hopkins, result.stdout)
        result = self.configure({"COSMICTIGER_ENABLE_HOPKINS": "ON",
                                 "COSMICTIGER_POSITION_TYPE": "fixed32"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("COSMICTIGER_ENABLE_HOPKINS=ON requires COSMICTIGER_POSITION_TYPE=fixed64", result.stderr)

    def testHopkinsControlsActualSourcesAndRetainsSharedSph(self):
        source = Path(__file__).resolve().parents[1]
        for (sph, hopkins), testing, cuda in product(
                (("OFF", "OFF"), ("ON", "OFF"), ("ON", "ON")), ("OFF", "ON"), ("OFF", "ON")):
            with self.subTest(sph=sph, hopkins=hopkins, testing=testing, cuda=cuda), tempfile.TemporaryDirectory() as directory:
                script = Path(directory) / "sources.cmake"
                script.write_text(
                    f'set(PROJECT_SOURCE_DIR "{source}")\n'
                    f'set(COSMICTIGER_ENABLE_SPH {sph} CACHE BOOL "")\n'
                    f'set(COSMICTIGER_ENABLE_HOPKINS {hopkins} CACHE BOOL "")\n'
                    'set(COSMICTIGER_POSITION_TYPE fixed64 CACHE STRING "")\n'
                    f'set(BUILD_TESTING {testing})\nset(COSMICTIGER_WITH_CUDA {cuda})\n'
                    f'include("{source / "cmake/features.cmake"}")\n'
                    f'include("{source / "cmake/variant_sources.cmake"}")\n'
                    'cosmictiger_feature_sources(selected)\n'
                    'foreach(path IN LISTS selected)\nmessage(STATUS "source=${path}")\nendforeach()\n')
                result = subprocess.run(["cmake", "-P", str(script)], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                for name in ("src/hydro/hopkins.cpp", "src/scenarios/sound_wave.cpp"):
                    self.assertEqual("source=" + name + "\n" in result.stdout, hopkins == "ON")
                for name in ("src/hydro/sph/sph_particles.cpp", "src/hydro/sph/sph_tree.cpp",
                             "src/hydro/sph/sph_hydro_cpu.cpp", "src/scenarios/gresho.cpp"):
                    self.assertEqual("source=" + name + "\n" in result.stdout, sph == "ON")

    def testStarsCapabilityCombinations(self):
        for sph, stars in product(("OFF", "ON"), repeat=2):
            with self.subTest(sph=sph, stars=stars):
                result = self.configure({"COSMICTIGER_ENABLE_SPH": sph,
                                         "COSMICTIGER_ENABLE_STARS": stars})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("stars=" + stars, result.stdout)
                self.assertIn("star_formation=" + stars, result.stdout)
                self.assertIn("effective_stars=" + ("ON" if sph == "ON" and stars == "ON" else "OFF"), result.stdout)

    def testFreshLegacyStarsOption(self):
        for value in ("OFF", "ON"):
            with self.subTest(value=value):
                result = self.configure({"COSMICTIGER_ENABLE_STAR_FORMATION": value})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("stars=" + value, result.stdout)
                self.assertIn("star_formation=" + value, result.stdout)

    def testStarsRequiresChemistryEvenWithoutSph(self):
        for sph in ("OFF", "ON"):
            with self.subTest(sph=sph):
                result = self.configure({"COSMICTIGER_ENABLE_SPH": sph,
                                         "COSMICTIGER_ENABLE_STARS": "ON",
                                         "COSMICTIGER_ENABLE_CHEMISTRY": "OFF"})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("COSMICTIGER_ENABLE_STARS=ON requires COSMICTIGER_ENABLE_CHEMISTRY=ON",
                              result.stderr)

    def testStarsWinsOverConflictingLegacyOption(self):
        for stars, legacy in (("OFF", "ON"), ("ON", "OFF")):
            with self.subTest(stars=stars, legacy=legacy):
                result = self.configure({"COSMICTIGER_ENABLE_STARS": stars,
                                         "COSMICTIGER_ENABLE_STAR_FORMATION": legacy})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("stars=" + stars, result.stdout)
                self.assertIn("star_formation=" + stars, result.stdout)

    def testStarsMigratedCacheCanBeDisabledAndReenabled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = Path(__file__).resolve().parents[1]
            (path / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.19)\nproject(stars_cache NONE)\n'
                f'include("{source / "cmake/features.cmake"}")\n'
                'message(STATUS "stars=${COSMICTIGER_ENABLE_STARS}")\n'
                'message(STATUS "star_formation=${COSMICTIGER_ENABLE_STAR_FORMATION}")\n')
            build = path / "build"
            build.mkdir()
            # Simulate the old cache before STARS existed, then migrate it.
            (build / "CMakeCache.txt").write_text("COSMICTIGER_ENABLE_STAR_FORMATION:BOOL=ON\n")
            for arguments, expected in (([], "ON"),
                    (["-DCOSMICTIGER_ENABLE_STARS=OFF"], "OFF"),
                    (["-DCOSMICTIGER_ENABLE_STAR_FORMATION=ON"], "OFF"),
                    (["-DCOSMICTIGER_ENABLE_STARS=ON"], "ON")):
                result = subprocess.run(["cmake", "-S", str(path), "-B", str(build), *arguments],
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("stars=" + expected, result.stdout)
                self.assertIn("star_formation=" + expected, result.stdout)

    def testStarsControlsActualSourceSelection(self):
        source = Path(__file__).resolve().parents[1]
        for sph, stars, testing, cuda in product(("OFF", "ON"), repeat=4):
            with self.subTest(sph=sph, stars=stars, testing=testing, cuda=cuda), tempfile.TemporaryDirectory() as directory:
                script = Path(directory) / "sources.cmake"
                script.write_text(
                    f'set(PROJECT_SOURCE_DIR "{source}")\n'
                    f'set(COSMICTIGER_ENABLE_SPH {sph} CACHE BOOL "")\n'
                    f'set(COSMICTIGER_ENABLE_STARS {stars} CACHE BOOL "")\n'
                    f'set(BUILD_TESTING {testing})\nset(COSMICTIGER_WITH_CUDA {cuda})\n'
                    f'include("{source / "cmake/features.cmake"}")\n'
                    f'include("{source / "cmake/variant_sources.cmake"}")\n'
                    'cosmictiger_feature_sources(selected)\n'
                    'foreach(path IN LISTS selected)\nmessage(STATUS "source=${path}")\nendforeach()\n')
                result = subprocess.run(["cmake", "-P", str(script)], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                enabled = sph == "ON" and stars == "ON"
                for name in ("src/hydro/sph/star_formation.cpp",):
                    self.assertEqual("source=" + name + "\n" in result.stdout, enabled)
                self.assertEqual("source=src/hydro/sph/star_formation_cuda.cu\n" in result.stdout,
                                 enabled and cuda == "ON")
                for name in ("tests/sph/star_formation.cpp", "tests/star_formation_particles.cpp"):
                    self.assertEqual("source=" + name + "\n" in result.stdout, enabled and testing == "ON")

    def testChemistryCapabilityCombinations(self):
        for sph, chemistry in product(("OFF", "ON"), repeat=2):
            with self.subTest(sph=sph, chemistry=chemistry):
                result = self.configure({"COSMICTIGER_ENABLE_SPH": sph,
                                         "COSMICTIGER_ENABLE_CHEMISTRY": chemistry})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("chemistry=" + chemistry, result.stdout)
                self.assertIn("effective_chemistry=" + ("ON" if sph == "ON" and chemistry == "ON" else "OFF"), result.stdout)

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

    def testGravityControlsActualSourceSelection(self):
        source = Path(__file__).resolve().parents[1]
        for solver, testing, cuda in product((None, "FMM", "PMMM_PP", "PMMM_FMM"),
                                              (False, True), (False, True)):
            with self.subTest(solver=solver, testing=testing, cuda=cuda), tempfile.TemporaryDirectory() as directory:
                configuration = {feature: feature in (solver, "SPH", "CHEMISTRY", "INTEGRATOR_TRADITIONAL")
                                 for feature in FEATURES}
                script = Path(directory) / "sources.cmake"
                script.write_text(
                    f'set(PROJECT_SOURCE_DIR "{source}")\n' +
                    "".join(f'set(COSMICTIGER_ENABLE_{feature} {"ON" if enabled else "OFF"} CACHE BOOL "")\n'
                            for feature, enabled in configuration.items()) +
                    f'set(BUILD_TESTING {"ON" if testing else "OFF"})\n'
                    f'set(COSMICTIGER_WITH_CUDA {"ON" if cuda else "OFF"})\n'
                    f'include("{source / "cmake/features.cmake"}")\n'
                    f'include("{source / "cmake/variant_sources.cmake"}")\n'
                    'cosmictiger_feature_sources(selected)\n'
                    'foreach(path IN LISTS selected)\nmessage(STATUS "source=${path}")\nendforeach()\n')
                result = subprocess.run(["cmake", "-P", str(script)], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                selected = {line.removeprefix("-- source=") for line in result.stdout.splitlines()
                            if line.startswith("-- source=")}
                self.assertEqual(sourceSelectionErrors(selected, configuration), [])
                self.assertIn("src/core/hpx.cpp", selected)
                self.assertNotIn("src/math/math.cpp", selected)
                for name in ("src/analysis/group_entry.cpp", "src/analysis/compress.cpp"):
                    self.assertEqual(name in selected, hasGravity(configuration), name)
                self.assertEqual("src/analysis/healpix.cu" in selected, hasGravity(configuration) and cuda)
                self.assertEqual("src/gravity/kick.cpp" in selected, hasTree(configuration))
                self.assertEqual("tests/analysis_backends.cpp" in selected, hasGravity(configuration) and testing)
                self.assertEqual("src/particles/particle_mesh_sort.cpp" in selected,
                                 solver in ("PMMM_PP", "PMMM_FMM"))

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
