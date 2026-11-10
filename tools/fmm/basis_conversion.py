# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# Distributed under the Boost Software License, Version 1.0.
"""Exact Cartesian STF source / spherical local basis maps.

Derived in the repository's signs and normalization from regular solid
harmonics (Dehnen 2014, Eq. 2b, doi:10.1186/s40668-014-0001-7).
See docs/fmm_basis_conversion.md for the two distinct dual conventions.
All polynomial projection and verification take place during generation.
"""
from fractions import Fraction as Q
from functools import lru_cache, reduce
from math import gcd, lcm

from cartesian.translations import (
    cartesianMonomial, indexFactorial, reduceSourceIndex, tensorIndices,
)
from cartesian.types import indices
from .spherical import Dag, index


def add(*polynomials):
    result = {}
    for polynomial in polynomials:
        for power, value in polynomial.items():
            result[power] = result.get(power, Q(0)) + value
    return {power: value for power, value in result.items() if value}


def scale(polynomial, factor):
    return {power: value * factor for power, value in polynomial.items() if value * factor}


def multiply(left, right):
    result = {}
    for a, x in left.items():
        for b, y in right.items():
            power = tuple(i + j for i, j in zip(a, b))
            result[power] = result.get(power, Q(0)) + x * y
    return {power: value for power, value in result.items() if value}


@lru_cache(None)
def regularPolynomials(degree):
    """Exact real/imaginary coefficient dictionaries for U_nm."""
    x, y, z = [{tuple(int(i == axis) for i in range(3)): Q(1)} for axis in range(3)]
    r2 = add(multiply(x, x), multiply(y, y), multiply(z, z))
    result = {(0, 0): ({(0, 0, 0): Q(1)}, {})}
    for m in range(degree + 1):
        if m:
            real, imaginary = result[m - 1, m - 1]
            result[m, m] = (
                scale(add(multiply(x, real), scale(multiply(y, imaginary), -1)), Q(1, 2*m)),
                scale(add(multiply(x, imaginary), multiply(y, real)), Q(1, 2*m)),
            )
        if m < degree:
            result[m + 1, m] = tuple(multiply(z, p) for p in result[m, m])
        for n in range(m + 2, degree + 1):
            result[n, m] = tuple(scale(add(scale(multiply(z, a), 2*n - 1),
                scale(multiply(r2, b), -1)), Q(1, n*n - m*m))
                for a, b in zip(result[n - 1, m], result[n - 2, m]))
    return result


def cartesianIndex(alpha):
    x, y, z = alpha
    n = x + y + z
    return n*n + (x if z == 0 else n + 1 + x)


def sphericalPolynomial(n, signedOrder):
    return regularPolynomials(n)[n, abs(signedOrder)][int(signedOrder < 0)]


@lru_cache(None)
def sourceRows(degree):
    """S_nm = (-1)^n sum_alpha [x^alpha] U_nm C_alpha.

    Harmonicity allows raw moments to be replaced by STF moments; trace
    reduction eliminates every unavailable z-pair component. Verify the map
    by substituting the actual Cartesian P2M polynomials with exact rationals.
    """
    rows = {}
    for n in range(degree + 1):
        for m in range(-n, n + 1):
            polynomial = scale(sphericalPolynomial(n, m), (-1)**n)
            row = {}
            for alpha, value in polynomial.items():
                for stored, factor in reduceSourceIndex(alpha):
                    row[stored] = row.get(stored, Q(0)) + value * factor
            row = {a: value for a, value in row.items() if value}
            reconstructed = add(*(scale(dict(cartesianMonomial(a)), value)
                                  for a, value in row.items()))
            assert reconstructed == polynomial, (n, m, 'source polynomial identity')
            rows[index(n, m)] = {cartesianIndex(a): value for a, value in row.items()}
    return rows


@lru_cache(None)
def localRows(degree):
    """C_alpha = alpha! [x^alpha] (L_n0 U_n0 + 2 Re sum_m>0 L_nm conj(U_nm))."""
    rows = {}
    for n in range(degree + 1):
        for alpha in indices(n):
            row = {}
            for m in range(-n, n + 1):
                value = sphericalPolynomial(n, m).get(alpha, Q(0))
                value *= indexFactorial(alpha) * (1 if m == 0 else 2)
                if value:
                    row[index(n, m)] = value
            rows[cartesianIndex(alpha)] = row
        # Reconstruct the complete Taylor polynomial, including dependent
        # z-pair derivatives; a round trip of two maps would not prove this.
        for m in range(-n, n + 1):
            reconstructed = {}
            for full in tensorIndices(n):
                coefficient = sum(factor * rows[cartesianIndex(a)].get(index(n, m), Q(0))
                                  for a, factor in reduceSourceIndex(full))
                if coefficient:
                    reconstructed[full] = coefficient / indexFactorial(full)
            expected = scale(sphericalPolynomial(n, m), 1 if m == 0 else 2)
            assert reconstructed == expected, (n, m, 'local polynomial identity')
    return rows


def emit(name, order, rows, targetType, sourceType):
    graph = Dag()
    roots = []
    for _, row in sorted(rows.items()):
        values = list(row.values())
        common = Q(reduce(gcd, (abs(v.numerator) for v in values)),
                   reduce(lcm, (v.denominator for v in values))) if values else Q(1)
        if values and all(value < 0 for value in values):
            common = -common
        value = graph.zero
        for slot, coefficient in sorted(row.items()):
            value = graph.fma(graph.constant(coefficient / common),
                              graph.symbol(f'source[{slot}]'), value)
        roots.append(graph.scale(value, common))
    code, arithmetic, weighted = graph.emit(roots)
    conventional = weighted + arithmetic.get('fma', 0)
    constant = name[0].upper() + name[1:]
    lines = [f'template <> inline constexpr int k{constant}Flops<{order}> = {weighted};',
             f'template <> inline constexpr int k{constant}ConventionalFlops<{order}> = {conventional};',
             '/// Assign the equivalent coefficients in the destination basis without changing spatial scale.',
             'template <class Real>',
             f'CUDA_EXPORT void {name}({targetType}<Real, {order}>& target,',
             f'    {sourceType}<Real, {order}> const& source) {{', *code]
    for slot, root in zip(sorted(rows), roots):
        lines.append(f'    target[{slot}] = {graph.expression(root)};')
    lines += ['}', '']
    return '\n'.join(lines), dict(operator=name, fmm_order=order, arithmetic=arithmetic,
        weighted_flops=weighted, conventional_flops=conventional)


def generate(minLocalDegree=2, maxLocalDegree=8):
    lines = ['// Generated by tools/generate_fmm.py. Do not edit.', '#pragma once', '',
             '#include <cosmictiger/gravity/kernels/cartesian_tracefree_types.hpp>',
             '#include <cosmictiger/gravity/kernels/spherical_types.hpp>',
             '#include <cosmictiger/math/simd_math.hpp>', '',
             '// Derived same-center, same-units maps. Outputs are overwritten.',
             '// Sₙₘ = (-1)ⁿ Σα [r^α] Uₙₘ Cα (sources).',
             '// Cα = α! [r^α] Σₙ (Lₙ₀ Uₙ₀ + 2 Re Σₘ>₀ Lₙₘ conjugate(Uₙₘ)) (locals).',
             '// Symbol directory: C is a Cartesian STF moment or local derivative;',
             '// S is a packed spherical source; L is a spherical local; U is a regular harmonic.',
             '// See docs/fmm_basis_conversion.md for normalization and periodic contexts.',
             'namespace cosmictiger::gravity::basis_conversion {']
    names = ('sourceToSpherical', 'localToCartesian')
    for name in names:
        constant = name[0].upper() + name[1:]
        lines += [f'template <int order> inline constexpr int k{constant}Flops = 0;',
                  f'template <int order> inline constexpr int k{constant}ConventionalFlops = 0;']
    report = {'description': 'Exact sparse basis maps; overwrite outputs at the same center and units. '
              'No coordinate, potential or background scaling. Source dipoles always retained.',
              'cost_model': {'add': 1, 'subtract': 1, 'multiply': 1, 'fma': 1},
              'conventional_cost_model': {'add': 1, 'subtract': 1, 'multiply': 1, 'fma': 2},
              'kernels': []}
    for degree in range(minLocalDegree, maxLocalDegree + 1):
        order = degree + 1
        lines += [f'#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= {order}) && '
                  f'(!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= {order})']
        for name, rows, target, source in (
                ('sourceToSpherical', sourceRows(degree - 1), 'spherical::Multipole',
                 'cartesian::Multipole'),
                ('localToCartesian', localRows(degree), 'cartesian::Expansion',
                 'spherical::Expansion')):
            code, counts = emit(name, order, rows, target, source)
            lines.append(code)
            report['kernels'].append(counts)
        lines += ['#endif']
    lines += ['} // namespace cosmictiger::gravity::basis_conversion', '']
    return {'basis_conversion.hpp': '\n'.join(lines)}, report
