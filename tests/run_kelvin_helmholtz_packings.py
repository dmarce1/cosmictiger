"""Check production K-H placement, normalization, bulk uniformity and option errors."""
import argparse
import itertools
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

from checkpoint_format import decode_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('executable', type=Path)
    parser.add_argument('workdir', type=Path)
    parser.add_argument('dimension', type=int, choices=(2, 3))
    args = parser.parse_args()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    dimension = args.dimension
    half_cells = 8 if dimension == 2 else 6
    grid_side = 4 * half_cells
    spacing = 0.5 / half_cells
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    common = [str(args.executable.resolve()), '--hpx:threads=2', '--hpx:bind=none',
              '--run.scenario=kelvinHelmholtz', '--run.darkMatterParticleCount=0', '--gravity.enabled=false',
              '--hydro.enabled=true', '--hydro.cuda=false', '--drift.cuda=false',
              '--chemistry.enabled=false', '--starFormation.enabled=false',
              '--hydro.kernelFamily=wendland', '--hydro.kernelDegree=5', '--hydro.neighborNumber=32',
              '--hydro.viscosityCoefficientMin=0', '--hydro.viscosityCoefficientMax=0', '--hydro.vonNeumannRichtmyerCoefficient=0',
              '--hydro.conductionCoefficient=0',
              '--run.endTime=1e-7', '--run.numberOfOutputs=3', '--run.maxIterations=1',
              '--checkpoint.intervalSeconds=0', '--output.silo=false', '--output.directory=.',
              '--run.verbose=true']

    def offsets(sites):
        result = []
        for point in itertools.product((0, 1), repeat=dimension):
            centers = sum(point)
            site = 'C' if centers == dimension else 'V' if centers == 0 else 'E' if centers == 1 else 'F'
            if site in sites:
                result.append(point)
        return result

    def run(name, packing, count, extra=(), error=None):
        directory = root / name
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()
        result = subprocess.run(common + [f'--run.gasParticleCount={count}',
            f'--scenario.kelvinHelmholz.packing={packing}', *extra], cwd=directory,
            env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        (directory / 'run.log').write_text(result.stdout)
        if error is not None:
            assert result.returncode != 0 and error in result.stdout, (name, result.stdout[-3000:])
            assert not list(directory.glob('checkpoint.*/checkpoint.0.dat')), name
            return None, directory
        assert result.returncode == 0, (name, result.stdout[-4000:])
        checkpoint = max(directory.glob('checkpoint.*/checkpoint.0.dat'),
                         key=lambda path: int(path.parent.name.split('.')[1]))
        return decode_checkpoint(checkpoint.read_bytes()), directory

    pairs = (['CEV/CV', 'CEV/E', 'CV/C', 'CV/V', 'E/C', 'E/V'] if dimension == 2 else
             ['CFEV/FV', 'CFEV/CE', 'VF/CV', 'CE/CV', 'CV/V', 'CV/C', 'FE/E', 'FE/F'])
    for index, pair in enumerate(pairs):
        high, low = pair.split('/')
        low_count = len(offsets(low)) * 2**(dimension - 1) * half_cells**dimension
        count = 3 * low_count
        # CLI overrides a deliberately invalid INI value, and punctuation/case normalize.
        config = root / f'packing_{index}.ini'
        config.write_text('[scenario.kelvinHelmholz]\npacking = X/X\n'
                          '[scenario.cosmology]\ntwoLpt = false\nuseGlass = false\ngadget4Restart =\n'
                          '[scenario.plummer]\nradius = 0.02\n')
        entered = high.lower() + ('-' if index % 2 else ' : ') + low
        state, directory = run(f'valid_{index}', entered, count - 1,
            (f'--configuration={config}', f'--initial.closePack={str(index % 2 == 0).lower()}'))
        assert state.count == state.gas_count == count
        assert state.dimension_count == dimension
        mass = state.header.physics[1]
        assert math.isclose(low_count * mass / 0.5, 0.5, rel_tol=1e-14)
        assert math.isclose(2 * low_count * mass / 0.5, 1.0, rel_tol=1e-14)
        expected = set()
        for cell in itertools.product(range(2 * half_cells), repeat=dimension):
            for offset in offsets(low if cell[0] < half_cells else high):
                expected.add(tuple(2 * c + o for c, o in zip(cell, offset)))
        actual = set()
        bulk = [[], []]
        layer_counts = [0, 0]
        denominator = 2**(8 * state.position_bytes)
        for gas_index, particle in enumerate(state.gravity_indices):
            position = [axis[particle] / denominator for axis in state.positions]
            snapped = tuple(round(value * grid_side) % grid_side for value in position)
            assert snapped not in actual, (pair, 'duplicate site')
            actual.add(snapped)
            for value, coordinate in zip(position, snapped):
                difference = abs(value - coordinate / grid_side)
                assert min(difference, 1 - difference) < 1e-5 * spacing, (pair, position)
            layer = int(snapped[0] >= 2 * half_cells)
            layer_counts[layer] += 1
            streamwise_velocity = state.velocities[particle * dimension + 1]
            assert abs(streamwise_velocity - (1 if layer == 0 else -1)) < 1e-4
            density = state.densities[gas_index]
            assert math.isfinite(density) and density > 0
            pressure = state.entropies[gas_index] * density**state.header.physics[2]
            assert abs(pressure / 19.2 - 1) < 1e-4, (pair, pressure)
            distance_to_interface = min(position[0], 1 - position[0], abs(position[0] - 0.5))
            if distance_to_interface > 1.01 * state.smoothing_lengths[gas_index]:
                bulk[layer].append(density)
        assert actual == expected, (pair, 'missing or misplaced sites')
        assert layer_counts == [low_count, 2 * low_count]
        for values in bulk:
            assert values, (pair, 'no interior particles')
            assert max(values) - min(values) < 3e-5 * sum(values) / len(values), (pair, min(values), max(values))
        means = [sum(values) / len(values) for values in bulk]
        print(f'{dimension}D {pair}: {count} particles, exact mean ratio 2; '
              f'uniform bulk SPH densities {means[0]:.7g}, {means[1]:.7g}', flush=True)
        if index == 0:
            log = (directory / 'run.log').read_text()
            assert re.search(r'^scenario\.plummer\.radius\s*:\s*2\.0+e-02', log, re.M)
            # Re-read the selected layout and the extended Options serialization on restart.
            restarted = subprocess.run(common + [f'--run.gasParticleCount={count}',
                f'--scenario.kelvinHelmholz.packing={pair}', '--checkpoint.readIndex=1'],
                cwd=directory, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=120)
            (directory / 'restart.log').write_text(restarted.stdout)
            assert restarted.returncode == 0, restarted.stdout[-4000:]

    rejected = [('C/V', '2:1'), ('V/CV', '2:1'), ('CV', 'requires two site strings'),
                ('CV/V/C', 'requires two site strings'), ('X/V', 'Unknown cell lattice site')]
    rejected += ([('FV/CV', 'not applicable'), ('CE/C', 'nonuniform')] if dimension == 2 else
                 [('VE/CV', 'nonuniform'), ('CF/CV', 'nonuniform'), ('CFEV/C', '2:1')])
    for index, (pair, error) in enumerate(rejected):
        run(f'rejected_{index}', pair, 1000, error=error)
    print(f'{dimension}D K-H production packing and rejection checks passed', flush=True)


if __name__ == '__main__':
    main()
