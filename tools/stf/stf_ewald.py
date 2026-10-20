# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Emit the periodic Ewald derivatives alongside the STF gravity operators.

This module is a port of CosmicTiger's GPL-licensed fmm_kernel_codegen.cpp;
it does not derive from the Boost-licensed STF algebra. Ewald uses the legacy
rank-packed symmetric/traceless tensor layout, including its quadratic trace
component, rather than the STF generator's degree-packed harmonic layout.
"""

from math import factorial, pi, prod, sqrt


def symmetricIndex(alpha):
    x, y, z = alpha
    degree = x + y + z
    return degree * (degree + 1) * (degree + 2) // 6 + (y + z) * (y + z + 1) // 2 + z


def tracelessIndex(alpha, order):
    x, y, z = alpha
    return ((x + y) * (x + y + 1) // 2 + y
            + order * (order + 1) // 2 * (z == 1) + order * order * (z == 2))


def vectorFactorial(alpha):
    return prod(factorial(component) for component in alpha)


def doubleFactorial(value):
    return prod(range(value, 0, -2))


def closeToOne(value):
    return abs(1.0 - value) < 1.0e-20


def fullIndices(order):
    for x in range(order):
        for y in range(order - x):
            for z in range(order - x - y):
                yield x, y, z


def tracelessIndices(order):
    for x in range(order):
        for y in range(order - x):
            zmax = min(3, order) if x == 0 and y == 0 else min(order - x - y, 2)
            for z in range(zmax):
                yield x, y, z


def halfIndices(alpha):
    for x in range(alpha[0] // 2 + 1):
        for y in range(alpha[1] // 2 + 1):
            for z in range(alpha[2] // 2 + 1):
                yield x, y, z


def subtractTwice(alpha, beta):
    return tuple(a - 2 * b for a, b in zip(alpha, beta))


def computeDx(code, order):
    """Form Cartesian monomials in the legacy multiplication order."""
    code.append(f'T x[{(order + 1) * order * (order + 2) // 6}];')
    code.append('x[0] = simdBroadcast<T>(1);')
    for axis in range(3):
        alpha = tuple(int(component == axis) for component in range(3))
        code.append(f'x[{symmetricIndex(alpha)}] = dxrinv[{axis}];')
    flops = 0
    for degree in range(2, order):
        for x in range(degree + 1):
            for y in range(degree - x + 1):
                alpha = (x, y, degree - x - y)
                beta = list(alpha)
                betaDegree = degree
                minimumDegree = max(1, degree // 2)
                while betaDegree > minimumDegree:
                    for axis in range(3):
                        if betaDegree > minimumDegree and beta[axis] > 0:
                            beta[axis] -= 1
                            betaDegree -= 1
                gamma = tuple(a - b for a, b in zip(alpha, beta))
                comment = ' | '.join(' '.join(map(str, index)) for index in (alpha, beta, gamma))
                code.append(f'x[{symmetricIndex(alpha)}] = x[{symmetricIndex(gamma)}] '
                            f'* x[{symmetricIndex(beta)}]; // {comment}')
                flops += 1
    return flops


def appendInterleaved(code, assignments, operations):
    """Keep the existing independent-expression scheduling."""
    code.extend(assignments)
    midpoint = (len(operations) + 1) // 2
    for index in range(midpoint):
        code.append(operations[index])
        if index + midpoint < len(operations):
            code.append(operations[index + midpoint])


def weightedAssignment(target, source, factor, first):
    if closeToOne(factor):
        return f'{target} {"=" if first else "+="} {source};', int(not first)
    if closeToOne(-factor):
        return (f'{target} = -{source};' if first else f'{target} -= {source};'), 1
    if first:
        return f'{target} = simdBroadcast<T>({factor:.9e}) * {source};', 1
    return f'{target} = simdFma(simdBroadcast<T>({factor:.9e}), {source}, {target});', 2


def traceName(order, count, alpha):
    return f'drealOrder{order}Trace{count}Index{"".join(map(str, alpha))}'


def detraceFactor(alpha, beta):
    """Return one normalized detracing coefficient without integer overflow."""
    degree = sum(alpha)
    traceCount = sum(beta)
    reduced = subtractTwice(alpha, beta)
    numerator = ((-1 if traceCount % 2 else 1)
                 * doubleFactorial(2 * degree - 2 * traceCount - 1)
                 * vectorFactorial(alpha))
    denominator = 2**traceCount * vectorFactorial(beta) * vectorFactorial(reduced)
    # Preserve the original double-precision normalization and literal rounding,
    # but form the integer products exactly (rank eight exceeds signed int32).
    factor = numerator / denominator
    return factor * (1.0 / doubleFactorial(2 * degree - 1))


def computeDetrace(code, order):
    """Project the real-space derivatives, retaining the quadratic trace."""
    assignments = []
    operations = []
    flops = 0
    for traceCount in range(1, order // 2 + 1):
        for alpha in fullIndices(order - 2 * traceCount):
            degree = sum(alpha) + 2 * traceCount
            first = True
            for x in range(traceCount + 1):
                for y in range(traceCount - x + 1):
                    beta = (x, y, traceCount - x - y)
                    factor = factorial(traceCount) / vectorFactorial(beta)
                    sourceIndex = tuple(a + 2 * b for a, b in zip(alpha, beta))
                    target = traceName(degree, traceCount, alpha)
                    source = f'dreal[{symmetricIndex(sourceIndex)}]'
                    expression, count = weightedAssignment(target, source, factor, first)
                    if first:
                        assignments.append('T ' + expression)
                    else:
                        operations.append(expression)
                    flops += count
                    first = False
    appendInterleaved(code, assignments, operations)
    assignments = []
    operations = []
    for alpha in tracelessIndices(order):
        degree = sum(alpha)
        first = True
        for beta in halfIndices(alpha):
            traceCount = sum(beta)
            if degree == 2 and 2 in alpha and traceCount == 1:
                continue
            reduced = subtractTwice(alpha, beta)
            factor = detraceFactor(alpha, beta)
            source = (traceName(degree, traceCount, reduced) if traceCount
                      else f'dreal[{symmetricIndex(reduced)}]')
            target = f'd[{tracelessIndex(alpha, order)}]'
            expression, count = weightedAssignment(target, source, factor, first)
            (assignments if first else operations).append(expression)
            flops += count
            first = False
    appendInterleaved(code, assignments, operations)
    return flops


def computeDetraceEwald(code, order):
    operations = []
    flops = 0
    for alpha in fullIndices(order):
        degree = sum(alpha)
        for beta in halfIndices(alpha):
            traceCount = sum(beta)
            reduced = subtractTwice(alpha, beta)
            factor = (vectorFactorial(alpha)
                      / (2**traceCount * vectorFactorial(beta) * vectorFactorial(reduced)))
            targetIndex = symmetricIndex(alpha)
            target = f'dreal[{targetIndex}]'
            source = f'x[{symmetricIndex(reduced)}]'
            radial = f'dRinvPower{degree - traceCount}Order{traceCount}'
            if closeToOne(factor):
                expression = f'{target} = simdFma({source}, {radial}, {target});'
                flops += 2
            elif closeToOne(-factor):
                expression = f'{target} -= {source} * {radial};'
                flops += 2
            else:
                expression = (f'{target} = simdFma(simdBroadcast<T>({factor:.9e}), '
                              f'{source}*{radial}, {target});')
                flops += 3
            operations.append((targetIndex, expression))
    # The C++ generator used std::sort, leaving equal-destination accumulation
    # order implementation-dependent. Keep their traversal order deterministically.
    code.extend(expression for _, expression in sorted(operations, key=lambda entry: entry[0]))
    return flops


def radialDerivatives(code, order, origin):
    code.extend([
        'icnt++;',
        'T const r = simdSqrt(r2);',
        'T const n8r = simdBroadcast<T>(-8*SCALE_FACTOR_INV2) * r;',
        'T const rinv = simdSelect(r > simdBroadcast<T>(0), simdBroadcast<T>(1), '
        'simdBroadcast<T>(0)) / simdMax(r, simdBroadcast<T>(1.0e-20));',
    ])
    if origin:
        code.extend([
            'T exp0 = expnearzero(-simdBroadcast<T>(4*SCALE_FACTOR_INV2) * r2);',
            'T erf0 = erfnearzero(simdBroadcast<T>(2*SCALE_FACTOR_INV1) * r);',
            'T const expfactor = simdBroadcast<T>(2.256758334191025*SCALE_FACTOR_INV1) * exp0;',
        ])
    else:
        code.extend([
            'T exp0;',
            'T erfc0;',
            'erfcexp(simdBroadcast<T>(2.*SCALE_FACTOR_INV1) * r, &erfc0, &exp0);',
            'T const expfactor = fouroversqrtpi * simdBroadcast<T>(SCALE_FACTOR_INV1) * exp0;',
        ])
    code.extend([
        'T e0 = expfactor * rinv;',
        'T const rinv0 = simdBroadcast<T>(1);',
        'T const rinv1 = rinv;',
    ])
    flops = 0
    for power in range(2, (order + 1) // 2):
        left = power // 2
        right = power - left
        code.append(f'T const rinv{power} = rinv{left} * rinv{right};')
        flops += 1
    code.append('T const d0 = ' + ('erf0' if origin else '-erfc0') + ' * rinv;')
    for degree in range(1, order):
        code.append(f'T const d{degree} = simdFma(simdBroadcast<T>({-2 * degree + 1}) '
                    f'* d{degree - 1}, rinv, e0);')
        if degree != order - 1:
            code.append('e0 *= n8r;')
    for degree in range(order):
        for traceCount in range(degree + 1):
            if degree + traceCount < order:
                code.append(f'T const dRinvPower{degree}Order{traceCount} '
                            f'= d{degree} * rinv{traceCount};')
                flops += 1
    code.append('Array<T, NDIM> dxrinv;')
    for axis in range(3):
        code.append(f'dxrinv[{axis}] = dx[{axis}] * rinv;')
    flops += computeDx(code, order)
    flops += 37 + 7 * (order - 1) + order * (order + 1) // 2
    flops += computeDetraceEwald(code, order)
    return flops


def ewald(code, order):
    code.extend([
        f'// FMM order {order}.',
        'template<class T>',
        f'CUDA_EXPORT int ewaldCorrectionDerivatives(SymmetricTracelessTensor<T, {order}> &d, '
        'Array<T, NDIM> position) {',
        'position[0] *= simdBroadcast<T>(SCALE_FACTOR);',
        'position[1] *= simdBroadcast<T>(SCALE_FACTOR);',
        'position[2] *= simdBroadcast<T>(SCALE_FACTOR);',
        'FlopCounter<int> flops = 7;',
        'T r = simdSqrt(simdFma(position[0], position[0], simdFma(position[1], position[1], '
        'sqr(position[2]))));',
        f'T const fouroversqrtpi = simdBroadcast<T>({4.0 / sqrt(pi):.9e});',
        f'SymmetricTensor<T, {order}> dreal;',
        f'SymmetricTracelessTensor<T, {order}> dfour;',
        'dreal = 0.0f;',
        'dfour = 0.0f;',
        'd = 0.0f;',
        'auto const realsz = EwaldConst::nreal();',
        'auto const zeroMask = r > simdBroadcast<T>(0.0);',
        'int icnt = 0;',
        'auto const foursz = EwaldConst::nfour();',
        'for (int i = 0; i < foursz; i++) {',
        'auto const &h = EwaldConst::fourIndex(i);',
        f'auto const &d0Value = EwaldConst::fourExpansion<{order}>(i);',
        'T const hdotx = simdFma(simdBroadcast<T>(h[0]), position[0], '
        'simdFma(simdBroadcast<T>(h[1]), position[1], simdBroadcast<T>(h[2]) * position[2]));',
        'T cn, sn;',
        'sincos(simdBroadcast<T>(2.0 * M_PI * SCALE_FACTOR_INV1) * hdotx, &sn, &cn);',
    ])
    # Keep the Fourier sum first: overlapping its live accumulators with the
    # real-space work changes CUDA register allocation and table-load scheduling.
    isCosine = {tracelessIndex(alpha, order): sum(alpha) % 2 == 0
                for alpha in tracelessIndices(order)}
    for index in range(order * order + 1):
        trig = 'cn' if isCosine[index] else 'sn'
        code.append(f'dfour[{index}] = simdFma({trig}, simdBroadcast<T>(d0Value[{index}]), '
                    f'dfour[{index}]);')
    code.extend([
        '}',
        'Array<T, NDIM> dx;',
        'dx = position;',
        '{',
        'T r2 = simdFma(dx[0], dx[0], simdFma(dx[1], dx[1], sqr(dx[2])));',
    ])
    realFlops = radialDerivatives(code, order, True)
    code.extend([
        f'auto const drz = EwaldConst::d0<{order}>();',
        f'for (int i = 0; i < {(order + 2) * (order + 1) * order // 6}; i++) {{',
        'dreal[i] = simdSelect(zeroMask, dreal[i], -simdBroadcast<T>(drz[i]));',
        '}',
        '}',
        f'flops += icnt * {realFlops};',
        'for (int i = 0; i < realsz; i++) {',
        'auto const n = EwaldConst::realIndex(i);',
        'Array<T, NDIM> dx;',
        'for (int dim = 0; dim < NDIM; dim++) {',
        'dx[dim] = position[dim] - simdBroadcast<T>(SCALE_FACTOR) * n[dim];',
        '}',
        'T r2 = simdFma(dx[0], dx[0], simdFma(dx[1], dx[1], sqr(dx[2])));',
        'if (simdAnyOf(r2 < simdBroadcast<T>(SCALE_FACTOR2*EWALD_REAL_CUTOFF2))) {',
    ])
    radialDerivatives(code, order, False)
    code.extend(['}', '}'])
    detraceFlops = computeDetrace(code, order) + 16 + 3 * (order * order + 1)
    fourierFlops = 40 + 2 * (order * order + 1)
    code.extend([
        f'flops += {fourierFlops} * foursz + {detraceFlops + order * order + 1};',
        'd = d + dfour;',
        f'd[0] = simdBroadcast<T>({pi / 4.0:.9e} * SCALE_FACTOR_INV1) + d[0];',
        'return flops;',
        '}',
    ])


def generateEwald(minLocalDegree, maxLocalDegree):
    """Return Ewald overloads whose tensor order is local degree plus one."""
    code = [
        '// Generated by tools/generate_stf_m2l.py. Do not edit.',
        '#pragma once',
        '',
        '#include <cosmictiger/core/flops.hpp>',
        '#include <cosmictiger/fmm/ewald_indices.hpp>',
        '#include <cosmictiger/math/math.hpp>',
        '#include <cosmictiger/math/simd_math.hpp>',
        '#include <cosmictiger/math/tensor.hpp>',
        '',
        'namespace detail {',
    ]
    for localDegree in range(minLocalDegree, maxLocalDegree + 1):
        ewald(code, localDegree + 1)
    code.append('} // namespace detail')
    return '\n'.join(code) + '\n'
