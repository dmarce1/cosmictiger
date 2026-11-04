"""Exercise checkpoint corruption, reference coverage, and optimization-safe runners."""

import ast
import ctypes
from contextlib import redirect_stdout
import io
import importlib
import itertools
import json
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import camb_reference
import check_build_features
import checkpoint_format as checkpoint
import compare_native_recombination
import run_gadget4_cooling
import run_linear_growth
import run_star_formation_ids
import run_variational_sph


def checkpoint_fixture(*, hydro=True, cooling=False, stars=False, viscosity=None, balsara=False,
                       position_bytes=4, integer_bytes=4, global_step=False, pmmm=0,
                       ranks=1, converged=False, semiactive=True, proposed_rung=3):
    """Small writer-order fixture, with magic-like data inside a position array."""
    flags = 1 << 18
    flags |= int(hydro) | (int(cooling) << 11) | (int(stars) << 20)
    flags |= (int(position_bytes == 8) << 15) | (int(integer_bytes == 8) << 21)
    flags |= (int(global_step) << 19) | (int(bool(pmmm)) << 17)
    flags |= (int(viscosity is not None) << 14)
    flags |= int(hydro and balsara) << 13
    header = checkpoint.Header()
    header.magic, header.version, header.flags = 0x4354535048303031, 3, flags
    header.particles_per_species = 2
    header.physics[2] = 5.0 / 3.0
    output = bytearray(bytes(header))
    offsets = {}

    def put(code, *values):
        output.extend(struct.pack('=' + code, *values))

    put('5Q', 1, 2 if pmmm else 0, 1 if pmmm else 0, 5 if pmmm else 0,
        2 if global_step else 0)
    if pmmm:
        values = [pmmm, 8, 5, 1, 0, 0, 1, 0, 5, 0, 8 if pmmm >= 5 else 0]
        put(f'{ {1:7, 2:8, 3:10, 4:11, 5:11}[pmmm]}Q',
            *values[:{1:7, 2:8, 3:10, 4:11, 5:11}[pmmm]])
    if hydro:
        put('13d', *([1.] * 13))
        if viscosity is not None:
            put('7d', 0.1, 2., float(viscosity), 1., 2., 1., 2.)
    if cooling:
        offsets['cooling'] = len(output)
        put('6d', 6., .76, 5. / 3., 1e4, 0., 1.)
    if stars:
        offsets['star_model'] = len(output)
        put('11Q', 3, *([0] * 10))
    driver = checkpoint.Driver()
    driver.a, driver.tau, driver.iteration = 1., .1, 1
    output.extend(bytes(driver))
    if global_step:
        put('Qdd', 2, .1, .025 if hydro else 0.)
    integer = 'q' if integer_bytes == 8 else 'i'
    offsets['count'] = len(output)
    put(integer, 2)
    if hydro:
        put(integer, 1)
    for dimension in range(3):
        put('2Q' if position_bytes == 8 else '2I', 0x43545350 + dimension, 20 + dimension)
    put('6f', 1., 2., 3., 4., 5., 6.)
    put('2b', 0, 0)
    if hydro:
        put('2b', 0, 1)
        put(2 * integer, -1, 0)
        offsets['gas'] = len(output)
        version = 130000 + (110 if stars else 100 if cooling else 2)
        version += 0 if viscosity is None else 2 + 2 * viscosity
        version += int(balsara and viscosity is not None and viscosity > 0)
        offsets['gas_version'] = len(output) + 4
        put('7I', 0x43545350, version, integer_bytes, 4, 0, 3, 4)
        put(integer, 1)
        put('I', 0)  # initial-energy state, not yet converted to entropy
        put('d', 1.0)  # thermal scale factor
        put('2f', 100., .1)  # initial thermal array, then support radius array
        offsets['gas_mapping'] = len(output)
        put(integer, 1)
        put('f', .6)  # density; no persistent acceleration or entropy-rate arrays
        offsets['gas_flags'] = len(output)
        offsets['hydro_rung'] = len(output)
        encoded_rung = proposed_rung
        put('B', int(converged) | (int(semiactive) << 1) | (encoded_rung << 2))
        if viscosity is not None:
            offsets['viscosity_alpha'] = len(output)
            put('e', .25)
            if viscosity > 0:
                offsets['scaled_divergence'] = len(output)
                put('e', -.375)  # q = own conformal interval times canonical divergence
                if balsara:
                    put('B', 173)  # retained factor from the same reconstructed gradient
        if cooling:
            put('f', 1.)  # electron fraction
    if stars:
        offsets['metadata'] = len(output)
        put('2Q', 0x4354534650415254, 2)
        put('QQddd', 1, 0, -1., 0., 0.)
        offsets['second_identity'] = len(output)
        put('QQddd', 2, 3, -1., .1, .2)
    put('i' + integer, 0, 0)
    output.extend(bytes(ctypes.c_size_t(0)))
    for _ in range(ranks):
        output.extend(bytes(checkpoint.local_domain_type(integer_bytes)()))
    return bytes(output), offsets


class BuildCapabilityTests(unittest.TestCase):
    def test_rockstar_definitions_follow_requested_feature_and_gravity(self):
        self.assertIn('ROCKSTAR', check_build_features.FEATURES)
        self.assertIn('COSMICTIGER_ENABLE_ROCKSTAR', check_build_features.FEATURE_DEFINITIONS)
        for gravity, rockstar in itertools.product((False, True), repeat=2):
            with self.subTest(gravity=gravity, rockstar=rockstar):
                configuration = {feature: feature in ('SPH', 'INTEGRATOR_TRADITIONAL')
                                 for feature in check_build_features.FEATURES}
                configuration.update(FMM=gravity, ROCKSTAR=rockstar)
                definitions = check_build_features.configurationDefinitions(configuration, 32, (4, 6))
                self.assertEqual(definitions['COSMICTIGER_ENABLE_ROCKSTAR'], str(int(gravity and rockstar)))
                self.assertEqual(check_build_features.hasRockstar(configuration), gravity and rockstar)
                self.assertEqual(check_build_features.sourceSelectionErrors(
                    {'src/analysis/power.cpp', 'src/tools/healpix2silo.cpp'}, configuration),
                    [] if gravity else [
                        'gravity disabled but compiles cosmology source: src/analysis/power.cpp',
                        'gravity disabled but compiles gravity source: src/tools/healpix2silo.cpp'])


class CheckpointTests(unittest.TestCase):
    def test_layout_variants(self):
        for position_bytes, integer_bytes, global_step, state, viscosity, balsara in itertools.product(
                (4, 8), (4, 8), (False, True), ('hydro', 'cooling', 'stars'), (None, 0, 1, 2, 3),
                (False, True)):
            with self.subTest(position_bytes=position_bytes, integer_bytes=integer_bytes,
                              global_step=global_step, state=state, viscosity=viscosity, balsara=balsara):
                data, _ = checkpoint_fixture(position_bytes=position_bytes,
                    integer_bytes=integer_bytes, global_step=global_step,
                    cooling=state != 'hydro', stars=state == 'stars', viscosity=viscosity, balsara=balsara)
                result = checkpoint.decode_checkpoint(data)
                self.assertEqual((result.count, result.gas_count), (2, 1))
                self.assertEqual(result.positions[0][0], 0x43545350)
                self.assertEqual(result.gas_indices, (-1, 0))
                self.assertEqual(len(result.stars), 2 if state == 'stars' else 0)
                self.assertEqual(result.driver.global_timestep, .1 if global_step else 0.)
                self.assertEqual(result.driver.global_hydro_timestep, .025 if global_step else 0.)
                self.assertEqual(result.gas_flags, (14,))
                self.assertEqual(result.converged, (False,))
                self.assertEqual(result.semiactive, (True,))
                self.assertEqual(result.proposed_rungs, (3,))
                self.assertEqual(result.gas[0][3:6], (False, True, 3))
                self.assertEqual(result.bulk_viscosity_coefficients,
                                 [] if viscosity is None else [.25])
                self.assertEqual(result.divergence_histories,
                                     [-.375] if viscosity is not None and viscosity > 0 else [])
                self.assertEqual(result.balsara_factors,
                                 [173] if balsara and viscosity is not None and viscosity > 0 else [])

    def test_packed_gas_flags_preserve_all_valid_combinations(self):
        for converged, semiactive, proposed_rung in itertools.product(
                (False, True), (False, True), range(16)):
            with self.subTest(converged=converged, semiactive=semiactive,
                              proposed_rung=proposed_rung):
                data, offsets = checkpoint_fixture(viscosity=1, balsara=True,
                    converged=converged, semiactive=semiactive, proposed_rung=proposed_rung)
                result = checkpoint.decode_checkpoint(data)
                encoded_rung = proposed_rung
                expected = int(converged) | (int(semiactive) << 1) | (encoded_rung << 2)
                self.assertEqual(result.gas_flags, (expected,))
                self.assertEqual(result.converged, (converged,))
                self.assertEqual(result.semiactive, (semiactive,))
                self.assertEqual(result.proposed_rungs, (proposed_rung,))
                self.assertEqual(result.gas[0][3:6], (converged, semiactive, proposed_rung))
                self.assertEqual(offsets['viscosity_alpha'], offsets['gas_flags'] + 1)
                self.assertEqual(result.bulk_viscosity_coefficients, [.25])
                self.assertEqual(result.divergence_histories, [-.375])
                self.assertEqual(result.balsara_factors, [173])

    def test_packed_gas_flags_reject_unused_rung_codes(self):
        data, offsets = checkpoint_fixture()
        for encoded_rung, flags in itertools.product(range(16, 64), range(4)):
            altered = bytearray(data)
            struct.pack_into('=B', altered, offsets['gas_flags'], (encoded_rung << 2) | flags)
            with self.subTest(encoded_rung=encoded_rung, flags=flags), \
                    self.assertRaisesRegex(RuntimeError, 'hydro rung bound'):
                checkpoint.decode_checkpoint(altered)

    def test_pmmm_dark_matter_and_multiple_domains(self):
        for version, integer_bytes in itertools.product((1, 2, 3, 4, 5), (4, 8)):
            data, _ = checkpoint_fixture(hydro=False, pmmm=version,
                                          integer_bytes=integer_bytes, ranks=2)
            result = checkpoint.decode_checkpoint(data, ranks=2)
            self.assertEqual(result.pmmm[0], version)
            self.assertEqual(result.types, (0, 0))
            with self.assertRaisesRegex(RuntimeError, 'trailing'):
                checkpoint.decode_checkpoint(data)

    def test_reject_every_truncation(self):
        data, _ = checkpoint_fixture(cooling=True, stars=True, viscosity=2,
                                      integer_bytes=8, position_bytes=8, global_step=True)
        for end in range(len(data)):
            with self.subTest(end=end), self.assertRaises(RuntimeError):
                checkpoint.decode_checkpoint(data[:end])

    def test_reject_stale_model_versions_and_corruption(self):
        data, offsets = checkpoint_fixture(cooling=True, stars=True)
        mutations = [('cooling', 'd', 5., 'cooling checkpoint model'),
                     ('star_model', 'Q', 2, 'star-formation model'),
                     ('gas', 'I', 0, 'gas layout'),
                     ('gas_version', 'I', 90110, 'gas layout'),
                     ('gas_version', 'I', 100110, 'gas layout'),
                     ('gas_version', 'I', 110110, 'gas layout'),
                     ('gas_version', 'I', 120110, 'gas layout'),
                     ('count', 'i', 999, 'particle counts'),
                     ('gas_mapping', 'i', 0, 'gas mapping'),
                     ('hydro_rung', 'B', 64, 'hydro rung bound'),
                     ('metadata', 'Q', 0, 'star-particle metadata'),
                     ('second_identity', 'Q', 1, 'particle identities')]
        for name, code, value, message in mutations:
            altered = bytearray(data)
            struct.pack_into('=' + code, altered, offsets[name], value)
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, message):
                checkpoint.decode_checkpoint(altered)
        with self.assertRaisesRegex(RuntimeError, 'trailing'):
            checkpoint.decode_checkpoint(data + b'x')

    def test_compact_viscosity_corruption(self):
        data, offsets = checkpoint_fixture(viscosity=1)
        mutations = [
            ('viscosity_alpha', 'H', bits, 'viscosity coefficient')
            for bits in (0xbc00, 0x7c00, 0xfc00, 0x7e00)
        ]
        mutations += [
            ('scaled_divergence', 'H', bits, 'divergence history')
            for bits in (0x7c00, 0xfc00, 0x7e00)
        ]
        for name, code, value, message in mutations:
            altered = bytearray(data)
            struct.pack_into('=' + code, altered, offsets[name], value)
            with self.subTest(name=name, value=value), \
                    self.assertRaisesRegex(RuntimeError, message):
                checkpoint.decode_checkpoint(altered)

    def test_compact_history_preserves_half_values(self):
        data, offsets = checkpoint_fixture(viscosity=3, global_step=True)
        for bits in (0x0000, 0x8000, 0x0001, 0x8001, 0x0400, 0x8400, 0x7bff, 0xfbff):
            altered = bytearray(data)
            struct.pack_into('=H', altered, offsets['scaled_divergence'], bits)
            with self.subTest(bits=bits):
                result = checkpoint.decode_checkpoint(altered)
                scaled_divergence = result.divergence_histories[0]
                self.assertEqual(struct.pack('=e', scaled_divergence), struct.pack('=H', bits))

    def test_old_uv_checkpoint_inserts_checksum_and_is_rejected_early(self):
        data, offsets = checkpoint_fixture(cooling=True)
        old = run_gadget4_cooling.obsolete_uv_payload(data)
        self.assertEqual(len(old), len(data) + 8)
        self.assertEqual(struct.unpack_from('=d', old, offsets['cooling'])[0], 5.)
        with self.assertRaisesRegex(RuntimeError, 'cooling checkpoint model version'):
            checkpoint.decode_checkpoint(old)

    def test_reject_bad_layout_flag_before_reading_particles(self):
        data, _ = checkpoint_fixture()
        altered = bytearray(data)
        struct.pack_into('=I', altered, checkpoint.Header.flags.offset, 1 << 31)
        with self.assertRaisesRegex(RuntimeError, 'feature flags'):
            checkpoint.decode_checkpoint(altered)


class ReferenceCoverageTests(unittest.TestCase):
    def setUp(self):
        self.reference = [dict(n_eff=3.046, z=z, k_mpc=.01, delta_cdm=1.,
                               delta_baryon=1., delta_cb=1., x_e=1., T_m=1.) for z in (0., 10.)]

    def test_full_unique_grid_only(self):
        tolerances = {'n_eff': (1e-9, 1e-12), 'z': (1e-9, 1e-12)}
        pairs = camb_reference.matchReferenceRows(self.reference, self.reference[::-1], tolerances)
        self.assertEqual([row['z'] for _, row in pairs], [0., 10.])
        for candidate in ([], self.reference[:1], self.reference + self.reference[:1],
                          [self.reference[0]] * 2,
                          [dict(self.reference[0], z=math.nan), self.reference[1]]):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                camb_reference.matchReferenceRows(self.reference, candidate, tolerances)
        with self.assertRaises(ValueError):
            camb_reference.matchReferenceRows([self.reference[0]] * 2, self.reference, tolerances)

    def test_both_comparisons_reject_missing_and_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(reference=Path('reference.csv'), candidate=Path('candidate.csv'),
                                   output_dir=Path(directory), tolerance=1e-3, integer_redshifts=False)
            for compare in (camb_reference.compare, compare_native_recombination.compareHistories):
                for candidate in (self.reference[:1], [self.reference[0]] * 2):
                    with self.subTest(compare=compare.__module__, candidate=candidate):
                        with patch.object(camb_reference, 'readRows',
                                          side_effect=[self.reference, candidate]):
                            with self.assertRaises(ValueError):
                                compare(args)


class SourceIdentityTests(unittest.TestCase):
    def test_permutation_preserves_identity_source_and_swapped_ids_fail(self):
        for close_pack in (False, True):
            states = {}
            phases, side = (2 if close_pack else 1), 2
            for cell, phase, species in itertools.product(range(side ** 3), range(phases), range(2)):
                x, rem = divmod(cell, side * side)
                y, z = divmod(rem, side)
                offset = (.25 + phase * .5 if close_pack else 0.)
                offset += species * (.25 if close_pack else .5)
                states[2 * (phases * cell + phase) + species + 1] = {
                    'type': species, 'counter': species, 'velocity': (0., 0., 0.),
                    'position': tuple(((value + offset) / side) % 1. for value in (x, y, z))}
            run_star_formation_ids.check_lattice_identities(states, side, close_pack)
            reordered = dict(reversed(list(states.items())))
            run_star_formation_ids.compare_identities(states, reordered)
            altered = dict(states)
            altered[1], altered[3] = altered[3], altered[1]
            with self.assertRaisesRegex(RuntimeError, 'different lattice source'):
                run_star_formation_ids.check_lattice_identities(altered, side, close_pack)
            with self.assertRaisesRegex(RuntimeError, 'changed the position'):
                run_star_formation_ids.compare_identities(states, altered)


class VariationalCudaTests(unittest.TestCase):
    def run_mock_matrix(self, busy, cuda_status):
        def execute(spec, base, launcher):
            return {'name': spec.name, 'stem': spec.stem, 'backend': spec.backend,
                'localities': spec.localities, 'integrator': spec.integrator,
                'resolution': spec.resolution, 'overrides': spec.overrides, 'kind': spec.kind,
                'wall': 0., 'returnCode': 0, 'directory': str(base),
                'status': cuda_status if spec.backend == 'cuda' else 'ok',
                'message': 'mock CUDA result' if spec.backend == 'cuda' else '',
                'steps': 2, 'energyRows': 2, 'finalScale': .02, 'l1': .01,
                'finalEnergyError': 0., 'maxEnergyError': 0., 'maxDensity': 1.,
                'kinetic': [1., 1.], 'thermal': [1., 1.]}

        with tempfile.TemporaryDirectory() as directory:
            argv = ['run_variational_sph', '/fake/app', directory, '--quick', '--cuda', '--jobs=1']
            with patch.object(run_variational_sph.sys, 'argv', argv), \
                    patch.object(run_variational_sph, 'compiled_algorithms',
                                 return_value={'integrator': ['traditional']}), \
                    patch.object(run_variational_sph, 'gpuBusyWithOurJobs', return_value=busy), \
                    patch.object(run_variational_sph, 'execute', side_effect=execute), \
                    patch.object(run_variational_sph.subprocess, 'run',
                                 side_effect=AssertionError('test must not launch a process')), \
                    redirect_stdout(io.StringIO()):
                status = run_variational_sph.main()
            checks = json.loads((Path(directory) / 'results.json').read_text())['checks']
        return status, checks

    def test_cuda_skip_before_launch_and_after_backend_probe(self):
        for busy, gpu_status in (('CUDA explicitly disabled', 'ok'), (None, 'skipped')):
            with self.subTest(busy=busy, gpu_status=gpu_status):
                status, checks = self.run_mock_matrix(busy, gpu_status)
                self.assertEqual(status, 77)
                self.assertTrue(any(row['status'] == 'SKIP' and 'cuda' in row['name']
                                    for row in checks))
                self.assertFalse(any(row['status'] == 'FAIL' for row in checks))

    def test_cuda_failure_remains_failure(self):
        status, checks = self.run_mock_matrix(None, 'abort')
        self.assertEqual(status, 1)
        self.assertTrue(any(row['status'] == 'FAIL' and 'cuda' in row['name'] for row in checks))

    def test_only_explicit_missing_cuda_diagnostics_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = run_variational_sph.RunSpec('cuda_test', 'sod', Path('/fake/app'), backend='cuda')
            for message, expected in (
                    ('This executable was compiled without CUDA support', 'skipped'),
                    ('Selected CUDA backends require an available CUDA device', 'skipped'),
                    ('CUDA illegal memory access', 'abort')):
                result = run_variational_sph.analyzeRun(spec, Path(directory), 1, False, 0., message)
                self.assertEqual(result['status'], expected)
            result = run_variational_sph.analyzeRun(spec, Path(directory), None, True, 0.,
                'Selected CUDA backends require an available CUDA device')
            self.assertEqual(result['status'], 'timeout')


class OptimizedRunnerTests(unittest.TestCase):
    def test_application_runner_waits_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(run_linear_growth, 'application_arguments', return_value=[]), \
                    patch.object(run_linear_growth.subprocess, 'Popen') as popen:
                process = popen.return_value.__enter__.return_value
                process.wait.return_value = 7
                with self.assertRaisesRegex(RuntimeError, 'exited with 7'):
                    run_linear_growth.runDiagnostic(Path('/fake/executable'), Path(directory))
                process.wait.assert_called_once_with(timeout=180)

    def test_variational_mpi_launcher_argument_order(self):
        launcher = SimpleNamespace(mpi_exec='/mpi', mpi_numproc_flag='-np',
                                   mpi_preflag=['--before'], mpi_postflag=['--after'],
                                   hpx_argument=['--hpx:ini=hpx.parcel.mpi.enable=1'])
        for raw in (None, ['/app', '--raw']):
            spec = run_variational_sph.RunSpec('test', 'hydro_wave', Path('/app'),
                                               localities=2, rawCommand=raw)
            with patch.object(run_variational_sph, 'application_arguments', return_value=[]):
                command = run_variational_sph.buildCommand(spec, launcher)
            self.assertEqual(command[:6], ['/mpi', '-np', '2', '--before', '/app', '--after'])
            self.assertEqual(command[-2:], ['--hpx:localities=2', *launcher.hpx_argument])
            if raw:
                self.assertEqual(command[6], '--raw')

    def test_successful_cli_exit_guards_resolve_exception_types(self):
        # sys.exit(main()) raises SystemExit even on success. Its surrounding
        # except tuple is evaluated then, so stale module names can turn a
        # passing regression into a failing command after all checks complete.
        directory = Path(__file__).resolve().parent
        for name in ('run_checkpoint', 'run_gadget4_cooling', 'run_global_integrator',
                     'run_gravity_error', 'run_mpi_launcher', 'run_star_formation',
                     'run_star_formation_ids', 'run_linear_growth',
                     'camb_reference', 'compare_native_recombination'):
            with self.subTest(name=name):
                module = importlib.import_module(name)
                tree = ast.parse((directory / (name + '.py')).read_text())
                guards = [node for node in tree.body if isinstance(node, ast.If)
                          and isinstance(node.test, ast.Compare)
                          and isinstance(node.test.left, ast.Name)
                          and node.test.left.id == '__name__']
                self.assertEqual(len(guards), 1)
                namespace = dict(vars(module), __name__='__main__', main=lambda: 0)
                try:
                    exec(compile(ast.Module(body=guards, type_ignores=[]), name, 'exec'), namespace)
                except SystemExit as error:
                    self.assertIn(error.code, (None, 0))

    def test_regression_checks_are_not_python_assert_statements(self):
        directory = Path(__file__).resolve().parent
        for name in ('run_global_integrator.py', 'run_mpi_launcher.py', 'run_checkpoint.py',
                     'run_star_formation.py', 'run_gadget4_cooling.py', 'run_linear_growth.py'):
            tree = ast.parse((directory / name).read_text())
            with self.subTest(name=name):
                self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main()
