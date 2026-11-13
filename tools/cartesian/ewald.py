# CosmicTiger; BSL-1.0.
"""Generate float-capable Cartesian periodic corrections and their M2L map.

Derived Ewald formulas for the negative unit-box potential, alpha=2:
 C(r) = erf(2r)/r - sum[n!=0] erfc(2|r+n|)/|r+n|
        - sum[h!=0] exp(-pi^2 h^2/4) cos(2pi h.r)/(pi h^2) + pi/4.
The nearest singular image is excluded. The constant Laplacian is -4pi;
its quadratic is carried separately from harmonic Cartesian coefficients.
Real images are paired before accumulation. Reciprocal coefficients are
computed at generation time. All runtime arithmetic uses the supplied Real.
"""
from fractions import Fraction as Q
from itertools import product
from math import exp, factorial, pi, prod


def half(alpha):
    return product(*(range(n // 2 + 1) for n in alpha))


def generate(minDegree, maxDegree, algebra):
    if minDegree != maxDegree:
        raise ValueError('Cartesian Ewald emission requires one build degree')
    p = minDegree
    order = p + 1
    indices = [a for n in range(p + 1) for a in algebra.indices(n)]
    literal = lambda x: f'{x:.18e}f'
    realImages = [n for n in product(range(-4, 5), repeat=3)
                  if next((v for v in n if v), 0) > 0
                  and sum(max(abs(v) - .5, 0)**2 for v in n) < 9]
    waves = [n for n in product(range(-4, 5), repeat=3)
             if next((v for v in n if v), 0) > 0 and sum(v*v for v in n) <= 16]
    table = ['#pragma once', '', '#include <cmath>', '#include <type_traits>', '',
             '#include <cosmictiger/math/simd_math.hpp>', '',
             'namespace cosmictiger::gravity::cartesian::detail {',
             '#ifdef __CUDA_ARCH__', '#define COSMICTIGER_EWALD_TABLE static __device__ __constant__',
             '#else', '#define COSMICTIGER_EWALD_TABLE inline constexpr', '#endif']
    table += [f'COSMICTIGER_EWALD_TABLE float kEwaldImages[{len(realImages)}][3] = {{']
    table += ['{' + ','.join(literal(v) for v in n) + '},' for n in realImages]
    table += ['};', f'COSMICTIGER_EWALD_TABLE float kEwaldWaves[{len(waves)}][{3+len(indices)}] = {{']
    for n in waves:
        n2 = sum(v*v for v in n)
        base = -2 * exp(-pi*pi*n2/4) / (pi*n2)
        values = list(n) + [base * (2*pi)**sum(a) * (-1)**(sum(a)//2) *
                           prod(v**k for v, k in zip(n, a)) for a in indices]
        table += ['{' + ','.join(literal(v) for v in values) + '},']
    table += ['};', '#undef COSMICTIGER_EWALD_TABLE']
    for name in ('exp', 'erfc', 'sin', 'cos'):
        table += ['template <class Real>', f'CUDA_EXPORT inline Real ewald{name.title()}(Real x) {{',
                  f'if constexpr (std::is_arithmetic_v<Real>) {{ return std::{name}(x); }}',
                  f'else {{ return Real([&](auto lane) {{ return std::{name}(x[lane]); }}); }}', '}']
    table += ['}']

    def derivatives():
        g = algebra.Dag()
        x = [g.symbol(f'x[{i}]') for i in range(3)]
        powers = [[g.one, v] for v in x]
        for axis in range(3):
            for n in range(2, p + 1):
                powers[axis].append(g.mul(powers[axis][-1], x[axis]))
        result = []
        for a in indices:
            terms = []
            for b in half(a):
                reduced = tuple(i-2*j for i, j in zip(a, b))
                factor = Q(prod(factorial(i) for i in a),
                           2**sum(b)*prod(factorial(i) for i in b)*prod(factorial(i) for i in reduced))
                term = g.symbol(f'q[{sum(a)-sum(b)}]')
                for axis in range(3):
                    term = g.mul(term, powers[axis][reduced[axis]])
                terms.append(g.scale(term, factor))
            result.append(g.add(*terms))
        g, signed = algebra.normalizeSigns(g, result)
        live = g.live(root for root, sign in signed)
        plan = algebra.EmissionPlan(g, signed)
        body = g.emitConstants(live) + plan.emit(live)
        for i in range(len(indices)):
            body += plan.accumulate(f'target[{i}]', i)
        return body, plan.statistics(live, range(len(indices)))

    radialBody, radialCount = derivatives()
    green = ['#pragma once', '', '#include <cosmictiger/math/vector3.hpp>', '#include <cosmictiger/gravity/fmm_order.hpp>',
             '#include <cosmictiger/gravity/kernels/cartesian_tracefree_types.hpp>',
             '#include <cosmictiger/gravity/kernels/cartesian_tracefree/ewald_common.hpp>', '',
             'namespace cosmictiger::gravity::cartesian {',
             'template <class Real> CUDA_EXPORT void accumulateEwaldRadial(',
             'Vector3<Real> const& x, Real const* q, Green<Real>& target) {']
    green += radialBody + ['}', 'template <class Real> CUDA_EXPORT void evaluateEwaldGreen(',
             f'Green<Real, {order}>& output, Vector3<Real> const& distance) {{',
             'using namespace math;', 'Real const one = broadcast<Real>(1);',
             'Vector3<Real> x = distance;', 'Real const r2 = x.dot(x);',
             'Real const y = broadcast<Real>(4) * r2;',
             'Real const exponential = detail::ewaldExp(-y);',
             'Real series = one;']
    for term in reversed(range(24)):
        green += [f'series = math::fma(series, y * broadcast<Real>({1/(p+1.5+term):.18e}), one);']
    green += [f'Real s = exponential * series * broadcast<Real>({1/(2*p+1):.18e});',
              f'Real q[{order}];', f'q[{p}] = broadcast<Real>({4/pi**.5*(-8)**p:.18e}) * s;']
    for n in range(p-1, -1, -1):
        green += [f's = math::fma(broadcast<Real>(2) * y, s, exponential) * broadcast<Real>({1/(2*n+1):.18e});',
                  f'q[{n}] = broadcast<Real>({4/pi**.5*(-8)**n:.18e}) * s;']
    green += ['output = broadcast<Real>(0);', 'accumulateEwaldRadial(x, q, output);',
              f'for (int image = 0; image < {len(realImages)}; ++image) {{',
              'Green<Real> pair{};', 'for (int sign = -1; sign <= 1; sign += 2) {',
              'for (int dim = 0; dim < 3; ++dim) {',
              'x[dim] = distance[dim] + broadcast<Real>(sign * detail::kEwaldImages[image][dim]);', '}',
              'Real const radiusSquared = x.dot(x);',
              'Real const inverseRadius = one / math::sqrt(radiusSquared);',
              'Real const inverseSquared = inverseRadius * inverseRadius;',
              'Real exponentialTerm = broadcast<Real>(2.256758334191025148e+00) * detail::ewaldExp(-broadcast<Real>(4) * radiusSquared);',
              'q[0] = -detail::ewaldErfc(broadcast<Real>(2) / inverseRadius) * inverseRadius;']
    for n in range(1, p+1):
        green += [f'q[{n}] = math::fma(broadcast<Real>({-(2*n-1)}), q[{n-1}], exponentialTerm) * inverseSquared;']
        if n < p:
            green += ['exponentialTerm *= broadcast<Real>(-8);']
    green += ['accumulateEwaldRadial(x, q, pair);', '}',
              f'for (int i = 0; i < {len(indices)}; ++i) {{ output[i] += pair[i]; }}', '}',
              f'for (int wave = 0; wave < {len(waves)}; ++wave) {{',
              'Real phase = broadcast<Real>(0);', 'for (int dim = 0; dim < 3; ++dim) {',
              'phase = math::fma(broadcast<Real>(detail::kEwaldWaves[wave][dim]), distance[dim], phase);', '}',
              f'phase *= broadcast<Real>({2*pi:.18e});',
              'Real const cosine = detail::ewaldCos(phase);', 'Real const sine = -detail::ewaldSin(phase);']
    for i, a in enumerate(indices):
        trig = 'sine' if sum(a) % 2 else 'cosine'
        green += [f'output[{i}] = math::fma(broadcast<Real>(detail::kEwaldWaves[wave][{i+3}]), {trig}, output[{i}]);']
    green += ['}', f'output[0] += broadcast<Real>({pi/4:.18e});']
    for i, a in enumerate(indices):
        green += [f'output[{i}] *= broadcast<Real>(kScaleFactorInverse1' + ' * kScaleFactorInverse1'*sum(a) + ');']
    green += ['}', '}']

    g = algebra.Dag()
    source = {a: g.symbol(f'M[{algebra.index(a)}]') for n in range(p) for a in algebra.indices(n)}
    deriv = {a: g.symbol(f'G[{algebra.index(a)}]') for a in indices}
    getM, getG = algebra.accessor(g, source), algebra.accessor(g, deriv)
    result = []
    for a in indices:
        terms = []
        for n in range(min(p-1, p-sum(a))+1):
            for nx in range(n+1):
                for ny in range(n-nx+1):
                    b = (nx, ny, n-nx-ny)
                    terms.append(g.scale(g.mul(getM(b), getG(algebra.plus(a,b))), Q((-1)**n, algebra.fac(b))))
        result.append(g.add(*terms))
    g, signed = algebra.normalizeSigns(g, result)
    live = g.live(root for root, sign in signed)
    plan = algebra.EmissionPlan(g, signed)
    counts = plan.statistics(live, range(len(indices)))
    m2l = ['#pragma once', '', '#include <cosmictiger/gravity/kernels/cartesian_tracefree_types.hpp>',
           '#include <cosmictiger/math/simd_math.hpp>', '',
           'namespace cosmictiger::gravity::cartesian {',
           f'inline constexpr int kEwaldM2LFlops = {counts["conventional_flops"]};',
           'template <class Real> CUDA_EXPORT void M2L(',
           f'Expansion<Real, {order}>& L, Multipole<Real, {order}> const& M, Green<Real, {order}> const& G) {{']
    m2l += g.emitConstants(live) + plan.emit(live)
    for i in range(len(indices)):
        m2l += plan.accumulate(f'L[{i}]', i)
    m2l += ['}', '}']
    # Fixed lattice loops execute the same source arithmetic on every call.
    central = 5+1+1+24*3+2+1+p*5
    real = 6+5+2+1+3+3+p*4-1
    fft = 6+1+2+2*len(indices)
    greenFlops = central + radialCount['conventional_flops'] + len(realImages)*(2*(real+radialCount['conventional_flops'])+len(indices)) + len(waves)*fft + 1 + len(indices)
    green.insert(-1, f'inline constexpr int kEwaldGreenFlops = {greenFlops};')
    return {'cartesian_tracefree/ewald_common.hpp': '\n'.join(table)+'\n',
            'cartesian_tracefree/ewald_green.hpp': '\n'.join(green)+'\n',
            'cartesian_tracefree/ewald_m2l.hpp': '\n'.join(m2l)+'\n'}, {
            'local_degree': p, 'real_pairs': len(realImages), 'fourier_pairs': len(waves),
            'm2l': counts, 'radial': radialCount, 'green_conventional_flops': greenFlops}
