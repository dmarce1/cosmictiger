# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

"""Generate Cartesian symmetric tensors without imposing any trace identity.

Moments are raw physical-coordinate moments M_a=sum(m*x**a), and locals are
unscaled derivatives of the negative potential. Every contraction retains all
Cartesian multi-indices. The formulas are the multivariate binomial and Taylor
theorems, including the source-displacement sign (-1)**|b| in M2L/M2P.

The separate Plummer derivative emitter differentiates -(r*r+epsilon2)**(-1/2)
analytically. No radius-direction normalization is used, so epsilon2>0 permits
the origin. Arbitrary supplied derivative tensors allow future radial kernels.
"""

from collections import Counter
from fractions import Fraction
from itertools import product
from math import comb, factorial, prod


def indices(degree):
    return [(x, y, degree - x - y)
            for x in range(degree + 1) for y in range(degree - x + 1)]


def allIndices(degree):
    return [a for n in range(degree + 1) for a in indices(n)]


def index(a):
    x, y, z = a
    n = x + y + z
    return n * (n + 1) * (n + 2) // 6 + x * (n + 1) - x * (x - 1) // 2 + y


def fac(a):
    return prod(factorial(x) for x in a)


def addIndex(a, b):
    return tuple(x + y for x, y in zip(a, b))


def difference(a, b):
    return tuple(x - y for x, y in zip(a, b))


class Algebra:
    """Exact constants and expression CSE; emitted arithmetic is the FLOP model.

    FMA nodes are explicit, so their one-operation weight does not depend on
    compiler contraction flags. Coordinate powers and radial powers are shared.
    """
    def __init__(self):
        self.nodes = []
        self.cache = {}
        self.zero = self.constant(0)
        self.one = self.constant(1)

    def node(self, operation, *arguments):
        key = (operation,) + arguments
        if key not in self.cache:
            self.cache[key] = len(self.nodes)
            self.nodes.append(key)
        return self.cache[key]

    def constant(self, value):
        return self.node('constant', Fraction(value))

    def symbol(self, value):
        return self.node('symbol', value)

    def value(self, node):
        return self.nodes[node][1] if self.nodes[node][0] == 'constant' else None

    def mul(self, a, b):
        av, bv = self.value(a), self.value(b)
        if av is not None and bv is not None:
            return self.constant(av * bv)
        if av == 0 or bv == 0:
            return self.zero
        if av == 1:
            return b
        if bv == 1:
            return a
        return self.node('*', *sorted((a, b)))

    def scale(self, a, value):
        return self.mul(a, self.constant(value))

    def add(self, a, b):
        if a == self.zero:
            return b
        if b == self.zero:
            return a
        av, bv = self.value(a), self.value(b)
        if av is not None and bv is not None:
            return self.constant(av + bv)
        return self.node('+', *sorted((a, b)))

    def fma(self, a, b, c):
        if c == self.zero:
            return self.mul(a, b)
        if a == self.zero or b == self.zero:
            return c
        if a == self.one:
            return self.add(b, c)
        if b == self.one:
            return self.add(a, c)
        return self.node('fma', a, b, c)

    def monomials(self, degree):
        coordinates = [self.symbol(name) for name in ('dx', 'dy', 'dz')]
        powers = [[self.one] for _ in coordinates]
        for axis in range(3):
            for n in range(degree):
                powers[axis].append(self.mul(powers[axis][-1], coordinates[axis]))
        return {a: self.mul(self.mul(powers[0][a[0]], powers[1][a[1]]), powers[2][a[2]])
                for a in allIndices(degree)}

    def live(self, roots):
        done = set()
        def visit(node):
            if node in done:
                return
            done.add(node)
            if self.nodes[node][0] not in ('symbol', 'constant'):
                for child in self.nodes[node][1:]:
                    visit(child)
        for root in roots:
            visit(root)
        return done

    def expression(self, node):
        operation, *args = self.nodes[node]
        if operation == 'symbol':
            return args[0]
        if operation == 'constant':
            value = args[0]
            scalar = (f'{value.numerator}.0' if value.denominator == 1 else
                      f'({value.numerator}.0 / {value.denominator}.0)')
            return f'simdBroadcast<Real>({scalar})'
        return f't{node}'

    def emit(self, roots):
        live = self.live(roots)
        lines = []
        counts = Counter()
        for node in sorted(live):
            operation, *args = self.nodes[node]
            if operation in ('constant', 'symbol'):
                continue
            values = [self.expression(a) for a in args]
            if operation == 'fma':
                expression = f'simdFma({", ".join(values)})'
            elif operation == 'sqrt':
                expression = f'simdSqrt({values[0]})'
            else:
                expression = f'{values[0]} {operation} {values[1]}'
            lines.append(f'        Real const t{node} = {expression};')
            counts[operation] += 1
        return lines, dict(counts)


def finish(name, degree, arguments, algebra, result, operatorName, additive=True,
           centered=None):
    parameters = str(degree)
    if centered is not None:
        parameters += ', ' + str(centered).lower()
    lines = [f'template <> class {name}<{parameters}> {{', 'public:',
             '    template <typename Real, typename Target, typename Source = Target, typename Derivatives = Target>',
             f'    CUDA_EXPORT static void {"add" if additive else "evaluate"}({arguments}) {{']
    body, counts = algebra.emit(result.values())
    lines += body
    for target, root in result.items():
        lines.append(f'        {target} {"+=" if additive else "="} {algebra.expression(root)};')
    accumulations = len(result) if additive else 0
    conventional = sum((2 if operation == 'fma' else 1) * n
                       for operation, n in counts.items()) + accumulations
    weighted = sum((7 if operation == 'sqrt' else 1) * n
                   for operation, n in counts.items()) + accumulations
    lines += ['    }', f'    static constexpr int kFlops_ = {conventional};',
              f'    static constexpr int kWeightedFlops_ = {weighted};', '};', '']
    report = {'operator': operatorName, 'degree': degree, 'arithmetic': counts,
              'accumulations': accumulations, 'conventional_flops': conventional,
              'weighted_flops': weighted}
    if centered is not None:
        report['centered'] = centered
    return '\n'.join(lines), report


def p2m(degree):
    algebra = Algebra()
    powers = algebra.monomials(degree)
    mass = algebra.symbol('mass')
    result = {f'target[{index(a)}]': algebra.mul(mass, powers[a]) for a in allIndices(degree)}
    return finish('P2M', degree, 'Target& target, Real mass, Real dx, Real dy, Real dz',
                  algebra, result, 'P2M')


def m2m(degree):
    algebra = Algebra()
    powers = algebra.monomials(degree)
    source = {a: algebra.symbol(f'source[{index(a)}]') for a in allIndices(degree)}
    result = {}
    for a in allIndices(degree):
        total = algebra.zero
        for b in product(*(range(v + 1) for v in a)):
            coefficient = prod(comb(x, y) for x, y in zip(a, b))
            total = algebra.fma(algebra.scale(powers[difference(a, b)], coefficient), source[b], total)
        result[f'target[{index(a)}]'] = total
    return finish('M2M', degree,
                  'Target& target, Source const& source, Real dx, Real dy, Real dz',
                  algebra, result, 'M2M')


def m2l(degree, centered=False):
    algebra = Algebra()
    result = {}
    for a in allIndices(degree):
        total = algebra.zero
        for b in allIndices(min(degree - 1, degree - sum(a))):
            if centered and sum(b) == 1:
                continue
            source = algebra.symbol(f'source[{index(b)}]')
            derivative = algebra.symbol(f'derivatives[{index(addIndex(a, b))}]')
            total = algebra.fma(algebra.scale(source, Fraction((-1)**sum(b), fac(b))), derivative, total)
        result[f'target[{index(a)}]'] = total
    return finish('M2L', degree,
                  'Target& target, Source const& source, Derivatives const& derivatives',
                  algebra, result, 'M2L', centered=centered)


def l2l(degree):
    algebra = Algebra()
    powers = algebra.monomials(degree)
    result = {}
    for a in allIndices(degree):
        total = algebra.zero
        for b in allIndices(degree - sum(a)):
            source = algebra.symbol(f'source[{index(addIndex(a, b))}]')
            total = algebra.fma(algebra.scale(powers[b], Fraction(1, fac(b))), source, total)
        result[f'target[{index(a)}]'] = total
    return finish('L2L', degree,
                  'Target& target, Source const& source, Real dx, Real dy, Real dz',
                  algebra, result, 'L2L')


def l2p(degree):
    algebra = Algebra()
    powers = algebra.monomials(degree)
    result = {}
    for component in range(4):
        a = tuple(int(component == axis + 1) for axis in range(3))
        total = algebra.zero
        for b in allIndices(degree - sum(a)):
            source = algebra.symbol(f'source[{index(addIndex(a, b))}]')
            total = algebra.fma(algebra.scale(powers[b], Fraction(1 if component == 0 else -1, fac(b))),
                                source, total)
        result[f'target[{component}]'] = total
    return finish('L2P', degree,
                  'Target& target, Source const& source, Real dx, Real dy, Real dz',
                  algebra, result, 'L2P')


def p2l(degree):
    algebra = Algebra()
    mass = algebra.symbol('mass')
    result = {f'target[{index(a)}]': algebra.mul(mass, algebra.symbol(f'derivatives[{index(a)}]'))
              for a in allIndices(degree)}
    return finish('P2L', degree,
                  'Target& target, Real mass, Derivatives const& derivatives',
                  algebra, result, 'P2L')


def m2p(degree, centered=False):
    algebra = Algebra()
    result = {}
    for component in range(4):
        a = tuple(int(component == axis + 1) for axis in range(3))
        total = algebra.zero
        for b in allIndices(degree):
            if centered and sum(b) == 1:
                continue
            source = algebra.symbol(f'source[{index(b)}]')
            derivative = algebra.symbol(f'derivatives[{index(addIndex(a, b))}]')
            factor = Fraction((-1)**(sum(b) + int(component != 0)), fac(b))
            total = algebra.fma(algebra.scale(source, factor), derivative, total)
        result[f'target[{component}]'] = total
    return finish('M2P', degree,
                  'Target& target, Source const& source, Derivatives const& derivatives',
                  algebra, result, 'M2P', centered=centered)


def plummerDerivatives(degree):
    algebra = Algebra()
    powers = algebra.monomials(degree)
    coordinates = [algebra.symbol(name) for name in ('dx', 'dy', 'dz')]
    radiusSquared = algebra.symbol('epsilonSquared')
    for coordinate in coordinates:
        radiusSquared = algebra.fma(coordinate, coordinate, radiusSquared)
    inverseRadius = algebra.node('/', algebra.one, algebra.node('sqrt', radiusSquared))
    inverseRadiusSquared = algebra.mul(inverseRadius, inverseRadius)
    radial = [inverseRadius]
    for n in range(degree):
        radial.append(algebra.mul(radial[-1], inverseRadiusSquared))
    result = {}
    for a in allIndices(degree):
        total = algebra.zero
        n = sum(a)
        for k in product(*(range(v // 2 + 1) for v in a)):
            count = sum(k)
            b = tuple(x - 2 * y for x, y in zip(a, k))
            oddFactorial = prod(range(2 * (n - count) - 1, 0, -2))
            coefficient = Fraction((-1)**(n-count+1) * fac(a) * oddFactorial,
                                   2**count * fac(k) * fac(b))
            total = algebra.fma(algebra.scale(powers[b], coefficient), radial[n-count], total)
        result[f'target[{index(a)}]'] = total
    return finish('PlummerDerivatives', degree,
                  'Target& target, Real dx, Real dy, Real dz, Real epsilonSquared',
                  algebra, result, 'PlummerDerivatives', additive=False)


def radialDerivatives(degree):
    """Reconstruct full derivatives from direction and radius-weighted coefficients.

    radial.coefficient<j,t>() is r**(j-t) (1/r d/dr)**j g(r). The
    coefficient evaluator owns its arithmetic; this report counts reconstruction
    only. Keeping radius powers combined avoids tiny intermediates in scaled units.
    """
    algebra = Algebra()
    powers = algebra.monomials(degree)
    result = {}
    for a in allIndices(degree):
        total = algebra.zero
        n = sum(a)
        for k in product(*(range(v // 2 + 1) for v in a)):
            traces = sum(k)
            b = tuple(x - 2 * y for x, y in zip(a, k))
            coefficient = Fraction(fac(a), 2**traces * fac(k) * fac(b))
            radial = algebra.symbol(f'radial.template coefficient<{n-traces}, {traces}>()')
            total = algebra.fma(algebra.scale(powers[b], coefficient), radial, total)
        result[f'target[{index(a)}]'] = total
    code, counts = finish('RadialDerivatives', degree,
                         'Target& target, Real dx, Real dy, Real dz, Derivatives const& radial',
                         algebra, result, 'RadialDerivatives', additive=False)
    counts['description'] = 'Tensor reconstruction only; normalized direction and radial coefficients supplied by caller.'
    return code, counts


def generate(minLocalDegree=2, maxLocalDegree=8):
    if not 2 <= minLocalDegree <= maxLocalDegree <= 8:
        raise ValueError('CartesianSymmetric local degrees must be in [2, 8]')
    header = '''// Generated by tools/generate_fmm.py. Do not edit.
// Full symmetric Cartesian moments: no trace identity or harmonic assumption.
#pragma once

#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace cosmictiger::gravity::cartesian::detail {

// Sources and targets must not alias. All add methods accumulate into target.
// Real is explicit at each call, e.g. M2L<4, false>::add<double>(local, source, derivatives).
// Kernel template arguments are polynomial degrees, not the application's FMM order.
template <int degree> class P2M;
template <int degree> class M2M;
template <int degree, bool centered> class M2L;
template <int degree> class L2L;
template <int degree> class L2P;
template <int degree> class P2L;
template <int degree, bool centered> class M2P;
template <int degree> class PlummerDerivatives;
template <int degree> class RadialDerivatives;
'''
    pieces = [header]
    report = {'representation': 'CartesianSymmetric', 'min_local_degree': minLocalDegree,
              'max_local_degree': maxLocalDegree,
              'description': 'Raw physical moments; full negative-potential derivatives; source degree L-1, local degree L; M2L total degree <= L. Conventional FMA costs 2, other arithmetic 1; weighted FMA costs 1, sqrt 7, other arithmetic 1. Stores and signs folded into constants cost 0.',
              'kernels': []}
    for localDegree in range(minLocalDegree, maxLocalDegree + 1):
        sourceDegree = localDegree - 1
        order = localDegree + 1
        pieces.append(f'#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= {order}) && \\\n'
                      f'    (!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= {order})')
        for generator, degree in [(p2m, sourceDegree), (m2m, sourceDegree), (m2l, localDegree),
                                  (l2l, localDegree), (l2p, localDegree), (p2l, localDegree),
                                  (m2p, sourceDegree), (plummerDerivatives, localDegree),
                                  (radialDerivatives, localDegree)]:
            code, counts = generator(degree)
            counts['fmm_order'] = localDegree + 1
            pieces.append(code)
            report['kernels'].append(counts)
        for generator, degree in [(m2l, localDegree), (m2p, sourceDegree)]:
            code, counts = generator(degree, centered=True)
            counts['fmm_order'] = order
            pieces.append(code)
            report['kernels'].append(counts)
        pieces.append('#endif\n')
    pieces += ['} // namespace cosmictiger::gravity::cartesian::detail', '']
    return {'cartesian_kernels.hpp': '\n'.join(pieces)}, report
