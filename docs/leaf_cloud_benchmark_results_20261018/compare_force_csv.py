#!/usr/bin/env python3
"""Align STF walk exports by exact coordinates and report numerical differences.

Usage: compare_force_csv.py REFERENCE.csv CANDIDATE.csv [--output result.json]
Relative L2 is sqrt(sum |candidate-reference|^2 / sum |reference|^2).
Pointwise relative errors omit particles whose reference norm/value is zero;
those omitted particles and any nonzero differences among them are counted.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_rows(path):
    rows = {}
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = (row['fixture'], int(row['x_raw']), int(row['y_raw']), int(row['z_raw']))
            assert key not in rows, f'Duplicate coordinate key {key} in {path}'
            values = tuple(float(row[field]) for field in ('gx', 'gy', 'gz', 'phi'))
            assert all(math.isfinite(value) for value in values), f'Nonfinite value {key} in {path}'
            rows[key] = values
    return rows


def describe(reference, candidate, vector):
    indices = range(3) if vector else (3,)
    reference_squares = []
    difference_squares = []
    relative = []
    component_max = 0.0
    zero_reference = 0
    changed_zero_reference = 0
    for first, second in zip(reference, candidate):
        ref_squared = math.fsum(first[i] * first[i] for i in indices)
        delta_squared = math.fsum((second[i] - first[i]) ** 2 for i in indices)
        component_max = max(component_max, *(abs(second[i] - first[i]) for i in indices))
        reference_squares.append(ref_squared)
        difference_squares.append(delta_squared)
        if ref_squared > 0:
            relative.append(math.sqrt(delta_squared / ref_squared))
        else:
            zero_reference += 1
            changed_zero_reference += delta_squared != 0
    reference_sum = math.fsum(reference_squares)
    difference_sum = math.fsum(difference_squares)
    count = len(reference)
    return {
        'reference_rms': math.sqrt(reference_sum / count),
        'difference_rms': math.sqrt(difference_sum / count),
        'relative_l2': math.sqrt(difference_sum / reference_sum) if reference_sum else None,
        'max_absolute_norm': math.sqrt(max(difference_squares)),
        'max_absolute_component': component_max,
        'max_pointwise_relative_nonzero_reference': max(relative) if relative else None,
        'zero_reference_count': zero_reference,
        'changed_zero_reference_count': changed_zero_reference,
    }


def compare(reference_path, candidate_path):
    reference = read_rows(reference_path)
    candidate = read_rows(candidate_path)
    assert reference.keys() == candidate.keys(), 'Coordinate keys do not match'
    fixtures = {}
    for fixture in sorted({key[0] for key in reference}):
        keys = sorted(key for key in reference if key[0] == fixture)
        first = [reference[key] for key in keys]
        second = [candidate[key] for key in keys]
        fixtures[fixture] = {
            'particles': len(keys),
            'acceleration': describe(first, second, True),
            'potential': describe(first, second, False),
        }
    return {
        'reference': str(reference_path.resolve()),
        'candidate': str(candidate_path.resolve()),
        'reference_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        'candidate_sha256': hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        'fixtures': fixtures,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path)
    arguments = parser.parse_args()
    result = compare(arguments.reference, arguments.candidate)
    text = json.dumps(result, indent=2) + '\n'
    if arguments.output:
        arguments.output.write_text(text)
    print(text, end='')
