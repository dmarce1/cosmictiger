"""Verify paired FMM measurements without running another workload."""
import csv
import math
from pathlib import Path


def rows(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def verify(root):
    for case in (1, 2):
        pairs = rows(root / f'fmm_measurement.{case}.pairs.csv')
        peers = rows(root / f'fmm_measurement.{case}.peers.csv')
        ranks = rows(root / f'fmm_measurement.{case}.ranks.csv')
        assert len(pairs) == 3 and len(peers) == len(ranks) == 12
        for repeat, pair in enumerate(pairs, 1):
            assert int(pair['repeat']) == repeat
            assert pair['first'] == ('reference' if repeat % 2 else 'export')
            assert int(pair['particles']) == (96 if case == 1 else 32 ** 3)
            assert int(pair['ranks']) == 2
            assert int(pair['total_worker_threads']) == 4
            assert pair['backend'] == 'cpu'
            for key in ('local_build_wall_s', 'exchange_wall_s', 'reference_force_wall_s',
                        'export_force_wall_s', 'cleanup_wall_s', 'pair_wall_s'):
                assert math.isfinite(float(pair[key])) and float(pair[key]) > 0
            shared = float(pair['local_build_wall_s']) + float(pair['cleanup_wall_s'])
            assert math.isclose(float(pair['reference_path_s']), shared + float(pair['reference_force_wall_s']), rel_tol=1e-8)
            assert math.isclose(float(pair['export_path_s']), shared + float(pair['exchange_wall_s']) + float(pair['export_force_wall_s']), rel_tol=1e-8)
            assert 0 <= float(pair['max_scaled_error']) <= 2e-5
            if case == 1:
                assert 0 <= float(pair['direct_acceleration_relative_l2']) <= .4 ** (int(pair['order']) - 1)
            for mode in ('reference', 'export'):
                p = [r for r in peers if int(r['repeat']) == repeat and r['mode'] == mode]
                r = [r for r in ranks if int(r['repeat']) == repeat and r['mode'] == mode]
                assert len(p) == len(r) == 2
                assert sum(int(x['fallbacks']) for x in r) == 0
                for a in p:
                    b = next(x for x in p if x['rank'] == a['peer'] and x['peer'] == a['rank'])
                    for sent, received in (('sent_bytes', 'received_bytes'), ('sent_messages', 'received_messages'),
                                           ('demand_sent_bytes', 'demand_received_bytes'),
                                           ('demand_sent_messages', 'demand_received_messages')):
                        assert a[sent] == b[received], (a, b, sent)
                    assert 0 <= int(a['used_nodes']) <= int(a['import_nodes'])
                    if mode == 'export':
                        assert int(a['sent_messages']) == int(a['received_messages']) == 1
                        assert int(a['export_nodes']) == int(b['import_nodes'])
                        assert int(a['export_particles']) == int(b['import_particles'])
                        assert 0 < int(a['covered_sources']) < int(pair['particles'])
                        assert int(a['demand_sent_messages']) == int(a['demand_received_messages']) == 0
                    else:
                        assert int(a['sent_messages']) == int(a['received_messages']) == 0
                        assert int(a['demand_sent_messages']) > 0
                for rank in r:
                    assert int(rank['owned_arena_bytes']) > 0
                    assert float(rank['traversal_s']) > 0


if __name__ == '__main__':
    import sys
    verify(Path(sys.argv[1]))
    print('FMM timing, accuracy, coverage and peer-direction invariants passed.')
