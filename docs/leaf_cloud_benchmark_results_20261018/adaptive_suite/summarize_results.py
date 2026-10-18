#!/usr/bin/env python3
"""Summarize a completed leaf/cloud comparison, after all timing processes finish.

Usage: summarize_results.py RESULTS_DIRECTORY [--pairs 5]
Writes summary.json and summary.md in RESULTS_DIRECTORY.
Requires both CPU and CUDA, before and after, and exactly the expected pairs.
Numerical differences are comparisons with the prior implementation, not errors
against a direct-summation or exact physical reference.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import median


COMPARATOR_PATH = Path(__file__).resolve().parent.parent / 'compare_force_csv.py'
spec = importlib.util.spec_from_file_location('force_comparator', COMPARATOR_PATH)
comparator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparator)


def summarize_timing(runs, expected_pairs):
    groups = defaultdict(dict)
    for run in runs:
        for entry in run['medians']:
            assert entry['backend'] == run['backend']
            key = (run['backend'], entry['fixture'], entry['potential'])
            label = (run['variant'], run['pair'])
            assert label not in groups[key]
            groups[key][label] = entry
    output = []
    for (backend, fixture, potential), records in sorted(groups.items()):
        assert len({record['position_hash'] for record in records.values()}) == 1
        assert len(records) == 2 * expected_pairs
        metrics = {}
        for field in ('kick_seconds', 'total_seconds', 'tree_seconds'):
            before = [records['before', pair][field] for pair in range(expected_pairs)]
            after = [records['after', pair][field] for pair in range(expected_pairs)]
            first, second = median(before), median(after)
            speedups = [old / new for old, new in zip(before, after)]
            paired_changes = [100 * (new / old - 1) for old, new in zip(before, after)]
            metrics[field] = {
                'before_process_medians_seconds': before,
                'after_process_medians_seconds': after,
                'before_median_seconds': first,
                'after_median_seconds': second,
                'time_change_percent': 100 * (second / first - 1),
                'speedup': first / second,
                'paired_speedup_median': median(speedups),
                'paired_speedup_range': [min(speedups), max(speedups)],
                'paired_time_change_percent_range': [min(paired_changes), max(paired_changes)],
            }
        output.append({
            'backend': backend,
            'fixture': fixture,
            'potential': potential,
            'pairs': expected_pairs,
            'samples_per_process': records['before', 0]['samples'],
            'particles': records['before', 0]['particles'],
            'position_hash': records['before', 0]['position_hash'],
            'metrics': metrics,
        })
    return output


def parse_leaf_histogram(path):
    rows = {}
    for line in path.read_text().splitlines():
        if line.startswith('STF_WALK_LEAF_HISTOGRAM,'):
            _, fixture, depth, particles, count = line.split(',')
            key = fixture, int(depth), int(particles)
            assert key not in rows, f'Duplicate histogram key in {path}'
            rows[key] = int(count)
    assert rows, f'Missing histogram in {path}'
    return rows


def summarize_leaves(result_directory, runs):
    first = None
    for run in runs:
        log = result_directory / f"{run['backend']}_pair{run['pair']}_{run['variant']}.log"
        histogram = parse_leaf_histogram(log)
        if first is None:
            first = histogram
        else:
            assert histogram == first, f'Histogram differs: {log}'
    output = {}
    for fixture in sorted({key[0] for key in first}):
        occupied = [(depth, size, count) for (name, depth, size), count in first.items()
                    if name == fixture and size > 0]
        depths = Counter()
        populations = Counter()
        for depth, size, count in occupied:
            depths[depth] += count
            populations[size] += count
        leaf_count = sum(count for _, _, count in occupied)
        small_count = sum(count for _, size, count in occupied if size < 38)
        output[fixture] = {
            'histograms_identical_across_processes': len(runs),
            'occupied_leaves': leaf_count,
            'particles': sum(size * count for _, size, count in occupied),
            'occupied_depth_range': [min(depths), max(depths)],
            'occupied_particle_count_range': [min(populations), max(populations)],
            'leaves_below_38_particles': small_count,
            'fraction_occupied_leaves_below_38': small_count / leaf_count,
            'particles_in_leaves_below_38': sum(size * count for _, size, count in occupied if size < 38),
            'occupied_leaves_by_depth': dict(sorted(depths.items())),
            'occupied_leaves_by_particle_count': dict(sorted(populations.items())),
            'empty_leaves_reported': sum(count for (name, _, size), count in first.items()
                                         if name == fixture and size == 0),
        }
    return output


def summarize_forces(runs):
    grouped = defaultdict(list)
    for run in runs:
        path = Path(run['force_csv'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == run['force_csv_sha256'], f'Force CSV checksum mismatch: {path}'
        grouped[run['backend'], run['variant']].append(run)
    for records in grouped.values():
        records.sort(key=lambda run: run['pair'])
    comparisons = []
    for backend in ('cpu', 'cuda'):
        first = grouped[backend, 'before'][0]
        second = grouped[backend, 'after'][0]
        result = comparator.compare(Path(first['force_csv']), Path(second['force_csv']))
        result.update({'kind': 'before_vs_after', 'backend': backend})
        comparisons.append(result)
    for variant in ('before', 'after'):
        first = grouped['cpu', variant][0]
        second = grouped['cuda', variant][0]
        result = comparator.compare(Path(first['force_csv']), Path(second['force_csv']))
        result.update({'kind': 'cpu_vs_cuda', 'variant': variant})
        comparisons.append(result)
    repeats = []
    for (backend, variant), records in sorted(grouped.items()):
        first = records[0]
        hash_groups = defaultdict(list)
        for record in records:
            hash_groups[record['force_csv_sha256']].append(record['pair'])
        differences = []
        seen = {first['force_csv_sha256']}
        for record in records[1:]:
            if record['force_csv_sha256'] not in seen:
                seen.add(record['force_csv_sha256'])
                result = comparator.compare(Path(first['force_csv']), Path(record['force_csv']))
                result['candidate_pair'] = record['pair']
                differences.append(result)
        repeats.append({
            'backend': backend, 'variant': variant,
            'processes': len(records),
            'byte_identical': len(hash_groups) == 1,
            'sha256_groups': dict(hash_groups),
            'differences_from_first_process': differences,
        })
    return {'comparisons': comparisons, 'repeat_consistency': repeats}


def format_markdown(result):
    lines = [
        'Timings are medians of five independent process medians, each containing 15 measured samples after one warmup.',
        'Tree + kick is the recorded median of each sample\'s combined tree and kick time. Negative time change means faster; speedup is before / after.',
        '',
        '| Backend | Fixture | Potential | Kick before (ms) | Kick after (ms) | Change | Speedup | Paired speedup range | Tree + kick before (ms) | Tree + kick after (ms) | Change |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in result['timings']:
        kick = row['metrics']['kick_seconds']
        total = row['metrics']['total_seconds']
        low, high = kick['paired_speedup_range']
        lines.append(f"| {row['backend']} | {row['fixture']} | {int(row['potential'])} | "
                     f"{1000*kick['before_median_seconds']:.3f} | {1000*kick['after_median_seconds']:.3f} | "
                     f"{kick['time_change_percent']:+.2f}% | {kick['speedup']:.4f}x | {low:.4f}–{high:.4f}x | "
                     f"{1000*total['before_median_seconds']:.3f} | {1000*total['after_median_seconds']:.3f} | "
                     f"{total['time_change_percent']:+.2f}% |")
    lines.extend(['', '| Fixture | Occupied leaves | Depth range | Particles/leaf range | Leaves with <38 particles |',
                  '|---|---:|---:|---:|---:|'])
    for name, row in result['leaf_histograms'].items():
        low_depth, high_depth = row['occupied_depth_range']
        low_size, high_size = row['occupied_particle_count_range']
        lines.append(f"| {name} | {row['occupied_leaves']} | {low_depth}–{high_depth} | {low_size}–{high_size} | "
                     f"{row['leaves_below_38_particles']} ({100*row['fraction_occupied_leaves_below_38']:.1f}%) |")
    lines.extend(['', 'Force comparisons use the final potential-enabled sample. Differences from the previous version are not errors against an exact-force reference.', '',
                  '| Comparison | Fixture | Acceleration relative L2 | Max acceleration vector difference | Max particle-relative acceleration difference | Potential relative L2 | Max potential difference |',
                  '|---|---|---:|---:|---:|---:|---:|'])
    for comparison in result['forces']['comparisons']:
        label = (f"{comparison['backend']} before/after" if comparison['kind'] == 'before_vs_after'
                 else f"{comparison['variant']} CPU/CUDA")
        for fixture, row in comparison['fixtures'].items():
            acceleration, potential = row['acceleration'], row['potential']
            lines.append(f"| {label} | {fixture} | {acceleration['relative_l2']:.6g} | "
                         f"{acceleration['max_absolute_norm']:.6g} | "
                         f"{acceleration['max_pointwise_relative_nonzero_reference']:.6g} | "
                         f"{potential['relative_l2']:.6g} | {potential['max_absolute_norm']:.6g} |")
    lines.append('')
    for row in result['forces']['repeat_consistency']:
        description = ('byte-identical' if row['byte_identical'] else
                       f"{len(row['sha256_groups'])} distinct CSV hashes; numerical repeat comparisons are in summary.json")
        lines.append(f"{row['backend']} {row['variant']}: {row['processes']} repeat exports are {description}.")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_directory', type=Path)
    parser.add_argument('--pairs', type=int, default=5)
    arguments = parser.parse_args()
    directory = arguments.results_directory.resolve()
    runs = json.loads((directory / 'runs.json').read_text())
    assert len(runs) == 4 * arguments.pairs, 'Timing protocol is incomplete'
    identities = {(run['backend'], run['variant'], run['pair']) for run in runs}
    expected = {(backend, variant, pair) for backend in ('cpu', 'cuda')
                for variant in ('before', 'after') for pair in range(arguments.pairs)}
    assert identities == expected
    assert all(run['exit_code'] == 0 for run in runs)
    result = {
        'runs_manifest': str(directory / 'runs.json'),
        'timing_method': 'Median of per-process medians; paired speedups use matching pair indices.',
        'timings': summarize_timing(runs, arguments.pairs),
        'leaf_histograms': summarize_leaves(directory, runs),
        'forces': summarize_forces(runs),
    }
    (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    markdown = format_markdown(result)
    (directory / 'summary.md').write_text(markdown)
    print(markdown, end='')


if __name__ == '__main__':
    main()
