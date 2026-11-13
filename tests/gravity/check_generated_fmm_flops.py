#!/usr/bin/env python3
# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
"""Independently execute generated kernels with a weighted counting scalar.

A temporary simd_math adapter replaces the arithmetic backend only. The actual
emitted C++ kernels are compiled unchanged, and every executed primitive charges
its cost. This checks live operations, destination accumulation, rotation setup,
and selected-kernel costs, independently of the Python expression DAGs.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile


COUNTING_HEADER = r'''#pragma once
#include <cmath>

class CountingReal {
public:
    using value_type = double;
    double value;
    inline static int operations = 0;
    inline static int squareRoots = 0;
    inline static int fusedOperations = 0;
    CountingReal(double x = 0) : value(x) {}
    CountingReal(CountingReal const& other) : value(other.value) {}
    CountingReal operator-() const { return CountingReal(-value); }
    CountingReal& operator+=(CountingReal const& rhs) {
        ++operations; value += rhs.value; return *this;
    }
    CountingReal& operator-=(CountingReal const& rhs) {
        ++operations; value -= rhs.value; return *this;
    }
    CountingReal& operator*=(CountingReal const& rhs) {
        ++operations; value *= rhs.value; return *this;
    }
};
inline CountingReal operator+(CountingReal a, CountingReal b) {
    ++CountingReal::operations; return CountingReal(a.value + b.value);
}
inline CountingReal operator-(CountingReal a, CountingReal b) {
    ++CountingReal::operations; return CountingReal(a.value - b.value);
}
inline CountingReal operator*(CountingReal a, CountingReal b) {
    ++CountingReal::operations; return CountingReal(a.value * b.value);
}
inline CountingReal operator/(CountingReal a, CountingReal b) {
    ++CountingReal::operations; return CountingReal(a.value / b.value);
}
inline bool operator==(CountingReal a, CountingReal b) { return a.value == b.value; }
inline bool operator<(CountingReal a, CountingReal b) { return a.value < b.value; }
inline bool operator>(CountingReal a, CountingReal b) { return a.value > b.value; }
template <typename Real> using SimdScalar = typename Real::value_type;
namespace math {
template <typename Real, typename Value>
Real broadcast(Value value) { return Real(value); }
template <typename Real>
Real fma(Real a, Real b, Real c) {
    ++Real::operations; ++Real::fusedOperations;
    return Real(std::fma(a.value, b.value, c.value));
}
struct Fma {
    template <typename Real>
    Real operator()(Real a, Real b, Real c) const { return math::fma(a, b, c); }
};
template <typename Real>
Real sqrt(Real a) {
    Real::operations += 7; ++Real::squareRoots; return Real(std::sqrt(a.value));
}
template <typename Real>
Real select(bool condition, Real yes, Real no) { return condition ? yes : no; }
template <typename Real>
Real max(Real a, Real b) { return a.value > b.value ? a : b; }
} // namespace math
template <typename Real, int lanes> class SimdVector;
'''


def cartesianFmmOrder(kernel):
    if 'expansion_order' in kernel:
        return kernel['expansion_order'] + 1
    offset = 2 if kernel['operator'] in ('P2M', 'M2M', 'M2P') else 1
    return kernel['order'] + offset


def selectReportOrders(report, minFmmOrder, maxFmmOrder):
    selected = deepcopy(report)

    def includes(order):
        return minFmmOrder <= order <= maxFmmOrder

    cartesian = selected['representations']['cartesianTracefree']
    for key in ('kernels', 'operators'):
        selected[key] = [row for row in selected[key] if includes(cartesianFmmOrder(row))]
        cartesian[key] = [row for row in cartesian[key] if includes(cartesianFmmOrder(row))]
    return selected


def appendAvailabilityChecks(lines):
    lines.extend([
        'template <class Real> concept HasCartesianM2L = requires(',
        '    cosmictiger::gravity::cartesian::Expansion<Real>& L,',
        '    cosmictiger::gravity::cartesian::Multipole<Real> const& M,',
        '    Vector3<Real> const& distance) {',
        '    cosmictiger::gravity::cartesian::M2L(M, distance, L);',
        '};',
        'static_assert(HasCartesianM2L<CountingReal>);',
        'template <class Real> concept HasCartesianM2Lewald = requires(',
        '    cosmictiger::gravity::cartesian::Expansion<Real>& L,',
        '    cosmictiger::gravity::cartesian::Multipole<Real> const& M,',
        '    cosmictiger::gravity::cartesian::Green<Real> const& G) {',
        '    cosmictiger::gravity::cartesian::M2Lewald(M, G, L);',
        '};',
        'static_assert(HasCartesianM2Lewald<CountingReal>);',
    ])


def appendCartesianChecks(lines, report, checks):
    cartesian = report['representations']['cartesianTracefree']
    operations = {
        'M2L': ('kM2LFlops', 'M2L(M, distance, L)', 11),
        'M2M': ('kM2MFlops', 'M2M(Mi, distance, ratio, Mo)', 0),
        'L2L': ('kL2LFlops', 'L2L(Li, distance, ratio, Lo)', 0),
        'P2M': ('kP2MFlops', 'P2M(mass, distance, targetM)', 0),
        'P2L': ('kP2LFlops', 'P2L(mass, distance, L)', 11),
        'M2P': ('kM2PFlops', 'M2P(M, distance, field)', 11),
        'L2P': ('kL2PFlops', 'L2P(L, distance, field)', 0),
    }
    for kernel in cartesian['kernels'] + cartesian['operators']:
        name = kernel.get('operator') or 'M2L'
        if name not in operations:
            continue
        degree = kernel.get('expansion_order', kernel.get('order'))
        order = degree + (2 if name in ('P2M','M2M','M2P') else 1)
        constant, call, geometry = operations[name]
        if name == 'M2L' and report['independent_m2l_ratios']:
            call = 'M2L(M, distance, ratio, ratio, L)'
        expected = kernel['weighted_flops'] + geometry
        expectedFma = kernel['arithmetic'].get('fma',0) + (2 if geometry else 0)
        label = f'Cartesian {name} degree {degree}'
        lines.extend([
            '    {',
            '        using namespace cosmictiger::gravity::cartesian;',
            f'        static_assert(cosmictiger::gravity::cartesian::{constant}<{degree}> == {kernel["conventional_flops"]});',
            f'        Multipole<Real> {"Mi{}, Mo{}" if name == "M2M" else "M{}, targetM{}"};',
            f'        Expansion<Real> {"Li{}, Lo{}" if name == "L2L" else "L{}, targetL{}"};',
            '        cosmictiger::gravity::Field<Real> field{};',
            '        Vector3<Real> const distance{Real(0.3), Real(-0.2), Real(0.4)};',
            '        Real const mass(0.7), ratio(0.6);',
            '        Real::operations = 0;',
            '        Real::squareRoots = 0;',
            '        Real::fusedOperations = 0;',
            f'        cosmictiger::gravity::cartesian::{call};',
            f'        if (Real::operations != {expected} || Real::fusedOperations != {expectedFma}) {{',
            f'            std::fprintf(stderr, "{label}: got %d ops/%d FMAs, expected {expected}/{expectedFma}\\n",',
            '                Real::operations, Real::fusedOperations);',
            '            return 6;',
            '        }',
            '    }',
        ])
        checks.append(label)


def appendEwaldChecks(lines, report, checks):
    ewald = report['representations']['cartesianEwald']
    kernel = ewald['m2l']
    expected = kernel['weighted_flops']
    expectedFma = kernel['arithmetic'].get('fma', 0)
    label = f'Cartesian M2Lewald degree {ewald["local_degree"]}'
    lines.extend([
        '    {',
        '        using namespace cosmictiger::gravity::cartesian;',
        f'        static_assert(kEwaldM2LFlops == {kernel["conventional_flops"]});',
        '        Multipole<Real> M{};',
        '        Green<Real> G{};',
        '        Expansion<Real> L{};',
        '        Real::operations = 0;',
        '        Real::squareRoots = 0;',
        '        Real::fusedOperations = 0;',
        '        M2Lewald(M, G, L);',
        f'        if (Real::operations != {expected} || Real::fusedOperations != {expectedFma}) {{',
        f'            std::fprintf(stderr, "{label}: got %d ops/%d FMAs, expected {expected}/{expectedFma}\\n",',
        '                Real::operations, Real::fusedOperations);',
        '            return 7;',
        '        }',
        '    }',
    ])
    checks.append(label)


def buildChecks(report):
    checks = []
    lines = ['#include <cstdio>', '']
    operators = ('p2m', 'p2l', 'm2m', 'm2l', 'm2p', 'l2l', 'l2p', 'ewald_m2l')
    for name in operators:
        lines.append(f'#include <cosmictiger/gravity/kernels/cartesian_tracefree/{name}.hpp>')
    for name in ('geometry',) + operators:
        lines.append(f'#include <tests/gravity/kernels/instrumentation/cartesian_tracefree_{name}.cpp>')
    appendAvailabilityChecks(lines)
    lines.extend(['int main() {', '    using Real = CountingReal;'])
    appendCartesianChecks(lines, report, checks)
    appendEwaldChecks(lines, report, checks)
    lines.extend([f'    std::printf("{len(checks)} executed weighted FLOP checks passed\\n");',
                  '    return 0;', '}'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated-dir', type=Path, required=True,
                        help='Directory containing operation_counts.json and generated headers')
    parser.add_argument('--generated-instrumentation-dir', type=Path,
                        help='Counting-test implementation directory (defaults match generate_fmm.py)')
    parser.add_argument('--compiler', default='c++')
    parser.add_argument('--min-fmm-order', type=int,
                        help='Smallest FMM order to compile and audit (default: generated minimum)')
    parser.add_argument('--max-fmm-order', type=int,
                        help='Largest FMM order to compile and audit (default: generated maximum)')
    args = parser.parse_args()
    projectRoot = Path(__file__).resolve().parents[2]
    generated = args.generated_dir.resolve()
    instrumentation = args.generated_instrumentation_dir
    if instrumentation is None:
        instrumentation = (projectRoot/'tests/gravity/kernels/instrumentation'
                           if generated == projectRoot/'cosmictiger/gravity/kernels'
                           else generated/'instrumentation')
    instrumentation = instrumentation.resolve()
    report = json.loads((generated/'operation_counts.json').read_text())
    generatedMinOrder = report['min_local_degree'] + 1
    generatedMaxOrder = report['max_local_degree'] + 1
    minFmmOrder = args.min_fmm_order if args.min_fmm_order is not None else generatedMinOrder
    maxFmmOrder = args.max_fmm_order if args.max_fmm_order is not None else generatedMaxOrder
    if not generatedMinOrder <= minFmmOrder <= maxFmmOrder <= generatedMaxOrder:
        parser.error(f'FMM orders must form an inclusive range within '
                     f'{generatedMinOrder}..{generatedMaxOrder}')
    report = selectReportOrders(report, minFmmOrder, maxFmmOrder)
    with tempfile.TemporaryDirectory(prefix='cosmictiger_fmm_counts_') as directory:
        root = Path(directory)
        shim = root/'cosmictiger/math/simd_math.hpp'
        shim.parent.mkdir(parents=True)
        shim.write_text(COUNTING_HEADER)
        vectorShim = root/'cosmictiger/math/vector3.hpp'
        vectorShim.write_text('''#pragma once
template <class Real> struct Vector3 {
    Real values[3];
    Vector3(Real x, Real y, Real z) : values{x, y, z} {}
    Real const& operator[](int i) const { return values[i]; }
    Real& operator[](int i) { return values[i]; }
};
''')
        fmmTypesShim = root/'cosmictiger/gravity/fmm_types.hpp'
        fmmTypesShim.parent.mkdir(parents=True)
        fmmTypesShim.write_text('''#pragma once
#include <cosmictiger/gravity/kernels/cartesian_tracefree_types.hpp>
namespace cosmictiger::gravity {
template <class Real> struct Field {
    Real values[4]{};
    Real& operator[](int i) { return values[i]; }
    Real const& operator[](int i) const { return values[i]; }
};
}
''')
        includeDir = root/'cosmictiger/gravity/kernels'
        includeDir.mkdir(parents=True)
        for base, destinationRoot, paths in (
                (generated, includeDir, generated.rglob('*.hpp')),
                (instrumentation, root/'tests/gravity/kernels/instrumentation',
                 instrumentation.glob('*.cpp'))):
            for path in paths:
                destination = destinationRoot/path.relative_to(base)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(path)
        source = root/'counts.cpp'
        source.write_text(buildChecks(report))
        executable = root/'counts'
        subprocess.run([args.compiler, '-std=c++20', '-O0', '-fno-fast-math',
                        '-ffp-contract=off', '-I'+str(root), '-I'+str(projectRoot),
                        f'-DORDER={minFmmOrder}', '-DCOSMICTIGER_FMM_COUNTING_ONLY',
                        str(source), '-o', str(executable)], check=True)
        subprocess.run([str(executable)], check=True)


if __name__ == '__main__':
    main()
