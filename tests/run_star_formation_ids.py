"""Verify cosmological source identities survive a change from one to two ranks."""

import argparse
import json
import math
from pathlib import Path

from application_selection import application_arguments
from checkpoint_format import decode_checkpoint, require
from run_checkpoint import checkManifest
from run_star_formation import prepare, run_process


def read_identities(directory, ranks):
    checkpoint = directory / 'checkpoint.1'
    checkManifest(checkpoint, 1, endTime=1e-4, gravity=False, ranks=ranks)
    states = {}
    for rank in range(ranks):
        state = decode_checkpoint((checkpoint / f'checkpoint.{rank}.dat').read_bytes(), ranks=ranks)
        require(state.has_stars, 'identity regression lacks star-formation metadata')
        scale = 2 ** (8 * state.position_bytes)
        for index, metadata in enumerate(state.stars):
            identity, counter = metadata[:2]
            require(identity not in states, 'duplicate particle identity across checkpoint ranks')
            states[identity] = {'type': state.types[index], 'counter': counter,
                'position': tuple(state.positions[dim][index] / scale for dim in range(3)),
                'velocity': state.velocities[3 * index:3 * index + 3]}
    return states


def periodic_distance(left, right):
    return abs((left - right + .5) % 1. - .5)


def check_lattice_identities(states, side, close_pack):
    phases = 2 if close_pack else 1
    require(set(states) == set(range(1, 2 * side ** 3 * phases + 1)),
            'source identity range has gaps or duplicates')
    for identity, state in states.items():
        source, species = divmod(identity - 1, 2)
        cell, phase = divmod(source, phases)
        ix, remaining = divmod(cell, side * side)
        iy, iz = divmod(remaining, side)
        offset = (.25 + .5 * phase if close_pack else 0.)
        offset += species * (.25 if close_pack else .5)
        expected = tuple(((coordinate + offset) / side) % 1. for coordinate in (ix, iy, iz))
        require(state['type'] == species, f'identity {identity} is attached to the wrong species')
        require(all(periodic_distance(actual, initial) < 1e-6
                    for actual, initial in zip(state['position'], expected)),
                f'identity {identity} moved to a different lattice source: {state["position"]} != {expected}')
        require(all(math.isfinite(value) for value in state['velocity']),
                f'identity {identity} has a nonfinite velocity')


def compare_identities(serial, distributed):
    require(serial.keys() == distributed.keys(), 'MPI decomposition changed particle identities')
    for identity, first in serial.items():
        second = distributed[identity]
        require((first['type'], first['counter']) == (second['type'], second['counter']),
                f'MPI decomposition changed source metadata for identity {identity}')
        require(all(periodic_distance(a, b) < 1e-7
                    for a, b in zip(first['position'], second['position'])),
                f'MPI decomposition changed the position associated with identity {identity}')
        require(all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-7)
                    for a, b in zip(first['velocity'], second['velocity'])),
                f'MPI decomposition changed the velocity associated with identity {identity}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('workdir', type=Path)
    parser.add_argument('--mpi-exec', required=True)
    parser.add_argument('--mpi-numproc-flag', default='-n')
    parser.add_argument('--mpi-preflag', action='append', default=[])
    parser.add_argument('--mpi-postflag', action='append', default=[])
    parser.add_argument('--hpx-argument', action='append', default=[])
    args = parser.parse_args()
    executable, base = args.executable.resolve(), args.workdir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    selections = application_arguments(executable, hydro=True)
    common = [str(executable), '--hpx:threads=1', '--hpx:bind=none', *selections,
        '--run.scenario=cosmology', '--run.endTime=0.0001', '--run.numberOfOutputs=17',
        '--run.maxIterations=1', '--scenario.cosmology.useGlass=false',
        '--cosmology.sigma8=0', '--gravity.enabled=false', '--hydro.enabled=true',
        '--chemistry.enabled=true', '--starFormation.enabled=true',
        '--chemistry.initializationTemperature=10000', '--gravity.cuda=false',
        '--analysis.groups=false', '--analysis.views=false', '--analysis.tracers=false',
        '--analysis.power=false', '--lightcone.enabled=false', '--output.silo=false',
        '--output.directory=.', '--checkpoint.intervalSeconds=0']
    evidence = []
    for close_pack in (False, True):
        label = 'close_pack' if close_pack else 'lattice'
        # Keep eight cells per axis; close packing has two sites per cell.
        particle_count = 1024 if close_pack else 512
        states = []
        for ranks in (1, 2):
            directory = prepare(base, f'{label}_{ranks}')
            command = [*common, f'--initial.closePack={str(close_pack).lower()}',
                f'--run.darkMatterParticleCount={particle_count}',
                f'--run.gasParticleCount={particle_count}']
            if ranks == 2:
                command = [args.mpi_exec, args.mpi_numproc_flag, '2', *args.mpi_preflag,
                    command[0], *args.mpi_postflag, *command[1:], '--hpx:localities=2',
                    '--hpx:ini=hpx.parcel.mpi.enable=1', *args.hpx_argument]
            run_process(command, directory, f'{label}_{ranks}')
            states.append(read_identities(directory, ranks))
            check_lattice_identities(states[-1], 8, close_pack)
        compare_identities(*states)
        evidence.append({'close_pack': close_pack, 'particles': len(states[0]),
                         'source_association': True, 'same_ids_across_ranks': True})
        print(f'{label}: {len(states[0])} identities map to the same sources on one and two ranks')
    (base / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    main()
