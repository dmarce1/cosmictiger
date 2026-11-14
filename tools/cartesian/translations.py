# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

# Imported from OctotigerII; distributed under the Boost Software License, Version 1.0.
"""Generate compact Cartesian Cartesian M2M and L2L translations.

This module uses the caller's rational arithmetic DAG and C++ emitter. All
polynomial projection, translation, and trace reduction happen in Python;
the generated equal-order kernels contain only straight-line arithmetic.
"""
from fractions import Fraction as Q
from functools import lru_cache
from math import comb, factorial, prod


def tensorIndices(degree):
    """Full symmetric tensor indices, used only during generation."""
    return ((x, y, degree-x-y)
            for x in range(degree+1) for y in range(degree-x+1))


def indexFactorial(index):
    return prod(factorial(x) for x in index)


def oddFactorial(n):
    return prod(range(n, 0, -2))


@lru_cache(None)
def cartesianMonomial(alpha):
    """Exact coefficients of H_alpha(x) = Cartesian(x**n)_alpha.

    Pairing beta equal-coordinate indices produces |x|**(2*|beta|).
    Expanding that radius power gives an ordinary homogeneous harmonic
    polynomial, without factorial normalization of the tensor component.
    """
    n = sum(alpha)
    polynomial = {}
    for bx in range(alpha[0]//2+1):
        for by in range(alpha[1]//2+1):
            for bz in range(alpha[2]//2+1):
                beta = (bx, by, bz)
                k = sum(beta)
                free = tuple(a-2*b for a, b in zip(alpha, beta))
                coefficient = Q((-1)**k * indexFactorial(alpha)
                                * oddFactorial(2*n-2*k-1),
                                2**k * indexFactorial(beta) * indexFactorial(free)
                                * oddFactorial(2*n-1))
                for radius in tensorIndices(k):
                    powers = tuple(a+2*b for a, b in zip(free, radius))
                    value = coefficient * Q(factorial(k), indexFactorial(radius))
                    polynomial[powers] = polynomial.get(powers, Q(0)) + value
    return tuple(sorted((power, value) for power, value in polynomial.items() if value))


@lru_cache(None)
def reduceSourceIndex(index):
    """Linear combination of stored Cartesian components for a full index."""
    x, y, z = index
    if z < 2:
        return ((index, 1),)
    result = {}
    for shifted in ((x+2, y, z-2), (x, y+2, z-2)):
        for component, coefficient in reduceSourceIndex(shifted):
            result[component] = result.get(component, 0) - coefficient
    return tuple(sorted((a, c) for a, c in result.items() if c))


@lru_cache(None)
def m2mPolynomial(alpha):
    """Coefficients of the moment functional H_alpha(ratio*y + d).

    Each homogeneous part in y is harmonic, so its contraction with raw
    source moments equals its contraction with Cartesian source moments. The ratio
    power is the source rank and is applied once to each source component by
    the DAG builder below. Dictionary keys are (displacement power, source
    independent index).
    """
    result = {}
    for power, coefficient in cartesianMonomial(alpha):
        for bx in range(power[0]+1):
            for by in range(power[1]+1):
                for bz in range(power[2]+1):
                    beta = (bx, by, bz)
                    displacement = tuple(a-b for a, b in zip(power, beta))
                    value = coefficient * prod(comb(a, b) for a, b in zip(power, beta))
                    for source, factor in reduceSourceIndex(beta):
                        key = displacement, source
                        result[key] = result.get(key, Q(0)) + value*factor
    result = {key: value for key, value in result.items() if value}
    # Projection must act as the identity on an already-Cartesian source tensor.
    zeroShift = {source: value for (power, source), value in result.items()
                  if power == (0, 0, 0)}
    assert zeroShift == {alpha: Q(1)}
    return tuple(sorted(result.items()))


def coordinatePowers(g, value, order):
    result = [g.one]
    for _ in range(order):
        result.append(g.mul(result[-1], value))
    return result


def m2mDirect(order, algebra, quadrupoleTrace=True, dipoleFree=False):
    g = algebra.Dag()
    displacement = [g.symbol(name) for name in ('dx', 'dy', 'dz')]
    ratio = coordinatePowers(g, g.symbol('scaleRatio'), order)
    scaled = {a: g.mul(ratio[n], value) for n in range(order+1)
              for a, value in algebra.sourceComponents(g, 'Mi', n, quadrupoleTrace, dipoleFree).items()}
    result = {}
    for n in range(order+1):
        for alpha in algebra.indices(n):
            polynomial = {}
            for (power, source), coefficient in m2mPolynomial(alpha):
                polynomial.setdefault(power, []).append(g.scale(scaled[source], coefficient))
            coefficients = {power: g.add(*terms) for power, terms in polynomial.items()}
            result[algebra.index(alpha)] = algebra.horner(g, coefficients, displacement)
    if quadrupoleTrace and order >= 2:
        rawQuadrupole(g, result, displacement, ratio, order, algebra, dipoleFree)
    return g, result


@lru_cache(None)
def equivalentTransform(order):
    """Cartesian components to coefficients modulo tz**2 + tx**2 + ty**2.

    Both the raw tensor component and its formal monomial reduce with the
    same integer trace map. Their product gives this small, exact Gram
    matrix. Keeping this conversion separate avoids expanding every shifted
    Cartesian component into all source/displacement monomials.
    """
    result = {}
    for full in tensorIndices(order):
        reduced = reduceSourceIndex(full)
        for target, factor in reduced:
            for source, sourceFactor in reduced:
                key = target, source
                result[key] = result.get(key, Q(0)) + Q(factor*sourceFactor, indexFactorial(full))
    return tuple(sorted((key, value) for key, value in result.items() if value))


def m2mCompact(order, algebra, quadrupoleTrace=True, dipoleFree=False):
    """Factor a shift through compact harmonic-equivalent polynomials.

    Formal-variable multiplication occurs in the quotient tz**2=-tx**2-ty**2.
    Translation is multiplication by the dual basis B(d), the reduced
    coefficients of exp(d.t). The inverse basis transform uses derivatives
    of H_alpha at zero. The same-rank source term is an exact identity and
    bypasses both transforms.
    """
    g = algebra.Dag()
    displacement = [g.symbol(name) for name in ('dx', 'dy', 'dz')]
    ratio = coordinatePowers(g, g.symbol('scaleRatio'), order)
    basis = algebra.dualBasis(g, displacement, order)
    scaled = {a: g.mul(ratio[n], value) for n in range(order+1)
              for a, value in algebra.sourceComponents(g, 'Mi', n, quadrupoleTrace, dipoleFree).items()}
    equivalent = {}
    for n in range(order):
        transformed = {}
        for (target, source), coefficient in equivalentTransform(n):
            transformed.setdefault(target, []).append(g.scale(scaled[source], coefficient))
        equivalent.update({target: g.add(*terms) for target, terms in transformed.items()})
    result = {}
    for n in range(order+1):
        lower = {a: [] for a in algebra.indices(n)}
        for j in range(n):
            for alpha in algebra.indices(j):
                for beta in algebra.indices(n-j):
                    product = g.mul(equivalent[alpha], basis[beta])
                    full = tuple(a+b for a, b in zip(alpha, beta))
                    for target, coefficient in reduceSourceIndex(full):
                        lower[target].append(g.scale(product, coefficient))
        lower = {a: g.add(*terms) for a, terms in lower.items()}
        for alpha in algebra.indices(n):
            terms = [scaled[alpha]]
            for beta, coefficient in cartesianMonomial(alpha):
                if beta[2] < 2:
                    terms.append(g.scale(lower[beta], coefficient*indexFactorial(beta)))
            result[algebra.index(alpha)] = g.add(*terms)
    if quadrupoleTrace and order >= 2:
        rawQuadrupole(g, result, displacement, ratio, order, algebra, dipoleFree)
    return g, result


def rawQuadrupole(g, result, displacement, ratio, order, algebra, dipoleFree=False):
    """Replace the rank-two projection with the full translated raw moment."""
    mass = g.symbol('Mi[0]')
    dipoleComponents = algebra.sourceComponents(g, 'Mi', 1, False, dipoleFree)
    dipole = [dipoleComponents[tuple(int(i == axis) for i in range(3))]
              for axis in range(3)]
    for alpha in [*algebra.indices(2), (0, 0, 2)]:
        axes = [axis for axis, count in enumerate(alpha) for _ in range(count)]
        i, j = axes
        source = (g.symbol('Mi.zz') if alpha == (0, 0, 2) else
                  g.symbol(f'Mi[{algebra.index(alpha)}]'))
        target = (order + 1)**2 if alpha == (0, 0, 2) else algebra.index(alpha)
        result[target] = g.add(g.mul(ratio[2], source),
                              g.mul(ratio[1], g.add(g.mul(displacement[i], dipole[j]),
                                                   g.mul(displacement[j], dipole[i]))),
                              g.mul(mass, g.mul(displacement[i], displacement[j])))


def m2m(order, algebra, quadrupoleTrace=True, dipoleFree=False):
    candidates = [m2mDirect(order, algebra, quadrupoleTrace, dipoleFree),
                  m2mCompact(order, algebra, quadrupoleTrace, dipoleFree)]
    if dipoleFree:
        # Only sums centered at the parent COM are retained; rank one is zero.
        for _, result in candidates:
            for alpha in algebra.indices(1):
                result.pop(algebra.index(alpha))
    def cost(candidate):
        g, result = candidate
        normalized, signed = algebra.normalizeSigns(g, list(result.values()))
        plan = algebra.EmissionPlan(normalized, signed)
        live = normalized.live(root for root, _ in signed)
        return plan.statistics(live, range(len(signed)))['weighted_flops']
    return min(candidates, key=cost)


def l2l(order, algebra, quadrupoleTrace=True):
    g = algebra.Dag()
    displacement = [g.symbol(name) for name in ('dx', 'dy', 'dz')]
    ratio = coordinatePowers(g, g.symbol('scaleRatio'), order)
    basis = algebra.dualBasis(g, displacement, order)
    source = {a: g.symbol(f'Li[{algebra.index(a)}]')
              for n in range(order+1) for a in algebra.indices(n)}
    get = algebra.accessor(g, source)
    result = {}
    for n in range(order+1):
        for alpha in algebra.indices(n):
            terms = []
            for j in range(order-n+1):
                for beta in algebra.indices(j):
                    index = tuple(a+b for a, b in zip(alpha, beta))
                    terms.append(g.mul(get(index), basis[beta]))
            result[algebra.index(alpha)] = g.mul(ratio[n], g.add(*terms))
    if quadrupoleTrace and order >= 2:
        # Ranks above two are harmonic. The remaining quadratic is
        # lambda*dz**2/2, with lambda the true rank-two trace.
        trace = g.add(source[(2, 0, 0)], source[(0, 2, 0)], g.symbol('Li.zz'))
        traceZ = g.mul(trace, displacement[2])
        result[0] = g.add(result[0], g.scale(g.mul(traceZ, displacement[2]), Q(1, 2)))
        zIndex = algebra.index((0, 0, 1))
        result[zIndex] = g.add(result[zIndex], g.mul(ratio[1], traceZ))
        result[(order + 1)**2] = g.sub(g.mul(ratio[2], trace),
                                     g.add(result[algebra.index((2, 0, 0))],
                                           result[algebra.index((0, 2, 0))]))
    return g, result


def emit(operator, order, algebra, quadrupoleTrace=True, dipoleFree=False):
    if operator == 'M2M':
        g, result = m2m(order, algebra, quadrupoleTrace, dipoleFree)
        destination = 'Mo'
        source = 'Mi'
        data_type = 'Multipole'
        fmm_order = order + 2
    else:
        g, result = l2l(order, algebra, quadrupoleTrace)
        destination = 'Lo'
        source = 'Li'
        data_type = 'Expansion'
        fmm_order = order + 1
    g, signed = algebra.normalizeSigns(g, list(result.values()))
    result = dict(zip(result, signed))
    live = g.live(root for root, _ in result.values())
    plan = algebra.EmissionPlan(g, signed)
    statistics = plan.statistics(live, range(len(signed)))
    description = ('Accumulate translated source moments, applying the source-to-target scale ratio.'
                   if operator == 'M2M' else
                   'Accumulate translated local coefficients, applying the target-to-source scale ratio.')
    lines = ['/// ' + description, f'template <class Real> CUDA_EXPORT void {operator}(',
             f'    {data_type}<Real, {fmm_order}>& {destination},',
             f'    {data_type}<Real, {fmm_order}> const& {source},',
             '    Vector3<Real> const& distance, Real scaleRatio) {',
             '    Real const dx = distance[0];',
             '    Real const dy = distance[1];',
             '    Real const dz = distance[2];']
    lines.extend(g.emitConstants(live))
    lines.extend(plan.emit(live))
    for output, index in enumerate(result):
        target = f'{destination}.zz' if index == (order + 1)**2 else f'{destination}[{index}]'
        lines.extend(plan.accumulate(target, output))
    lines.extend(['}', ''])
    statistics.update({'operator': operator, 'order': order})
    return '\n'.join(lines), statistics


def generateTranslations(minOrder, maxOrder, algebra, quadrupoleTrace=True, dipoleFree=False):
    """Return C++ definitions and operation counts for the inclusive range.

    The caller emits these definitions inside the Cartesian namespace.
    No fixed maximum order is imposed by this module.
    """
    if minOrder < 0 or maxOrder < minOrder:
        raise ValueError('Translation orders must form a nonnegative inclusive range')
    definitions = []
    statistics = []
    for operator, order in [('M2M', minOrder), ('L2L', maxOrder)]:
        code, counts = emit(operator, order, algebra, quadrupoleTrace, dipoleFree)
        definitions.append(code)
        statistics.append(counts)
    return definitions, statistics
