#!/usr/bin/env python3
# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
"""Independently execute generated kernels with a weighted counting scalar.

A temporary simd_math adapter replaces the arithmetic backend only. The actual
emitted C++ kernels are compiled unchanged, and every executed primitive charges
its cost. This checks live operations, destination accumulation, rotation setup,
and selected-kernel costs, independently of the Python expression DAGs. Candidate
reports retain all variants and are checked against the selected minimum.
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


def sphericalFmmOrder(kernel):
    offset = 2 if kernel['operator'] == 'P2m' else 1
    return kernel['degree'] + offset


def selectReportOrders(report, minFmmOrder, maxFmmOrder):
    selected = deepcopy(report)

    def includes(order):
        return minFmmOrder <= order <= maxFmmOrder

    cartesian = selected['representations']['cartesianTracefree']
    for key in ('kernels', 'operators'):
        selected[key] = [row for row in selected[key] if includes(cartesianFmmOrder(row))]
        cartesian[key] = [row for row in cartesian[key] if includes(cartesianFmmOrder(row))]
    spherical = selected['representations']['spherical']
    spherical['kernels'] = [row for row in spherical['kernels'] if includes(sphericalFmmOrder(row))]
    spherical['selections'] = [row for row in spherical['selections'] if includes(row['fmm_order'])]
    ewald = selected['representations']['sphericalEwald']
    ewald['orders'] = {order: counts for order, counts in ewald['orders'].items() if includes(int(order))}
    conversions = selected['representations']['basisConversion']
    conversions['kernels'] = [row for row in conversions['kernels'] if includes(row['fmm_order'])]
    return selected


def appendAvailabilityChecks(lines, report, minFmmOrder, maxFmmOrder):
    lines.extend([
        'template <typename Type> concept CompleteKernel = requires { sizeof(Type); };',
        'template <int order> concept HasCartesianM2L = requires(',
        '    cosmictiger::gravity::cartesian::tracefree::Expansion<CountingReal, order>& L,',
        '    cosmictiger::gravity::cartesian::tracefree::Multipole<CountingReal, order> const& M,',
        '    Vector3<CountingReal> const& distance) {',
        '    cosmictiger::gravity::cartesian::tracefree::detail::M2L(L, M, distance);',
        '};',
        'template <int order> concept HasSourceConversion = requires(',
        '    cosmictiger::gravity::spherical::Multipole<CountingReal, order>& S,',
        '    cosmictiger::gravity::cartesian::tracefree::Multipole<CountingReal, order> const& C) {',
        '    cosmictiger::gravity::basis_conversion::sourceToSpherical(S, C);',
        '};',
        'template <int order> concept HasLocalConversion = requires(',
        '    cosmictiger::gravity::cartesian::tracefree::Expansion<CountingReal, order>& C,',
        '    cosmictiger::gravity::spherical::Expansion<CountingReal, order> const& L) {',
        '    cosmictiger::gravity::basis_conversion::localToCartesian(C, L);',
        '};',
    ])
    for order in range(report['min_local_degree'] + 1, report['max_local_degree'] + 2):
        selected = str(minFmmOrder <= order <= maxFmmOrder).lower()
        degree = order - 1
        lines.append(f'static_assert(HasCartesianM2L<{order}> == {selected});')
        lines.append(f'static_assert(HasSourceConversion<{order}> == {selected});')
        lines.append(f'static_assert(HasLocalConversion<{order}> == {selected});')
        for kernel in (f'spherical::detail::P2M<{order - 2}>',
                       f'spherical::detail::Green<{degree}>',
                       f'spherical::detail::L2P<{degree}>',
                       f'spherical::EwaldGreen<{degree}>'):
            lines.append(f'static_assert(CompleteKernel<cosmictiger::gravity::{kernel}> == {selected});')


def appendCartesianChecks(lines, report, checks):
    cartesian = report['representations']['cartesianTracefree']
    operations = {
        'M2L': ('kM2LFlops', 'M2L(L, M, distance)', 11),
        'M2M': ('kM2MFlops', 'M2M(targetM, M, distance, ratio)', 0),
        'L2L': ('kL2LFlops', 'L2L(targetL, L, distance, ratio)', 0),
        'P2M': ('kP2MFlops', 'P2M(targetM, mass, distance)', 0),
        'P2L': ('kP2LFlops', 'P2L(L, mass, distance)', 11),
        'M2P': ('kM2PFlops', 'M2P(field, M, distance)', 11),
        'L2P': ('kL2PFlops', 'L2P(field, L, distance, ratio)', 0),
    }
    for kernel in cartesian['kernels'] + cartesian['operators']:
        name = kernel.get('operator') or 'M2L'
        if name not in operations:
            continue
        degree = kernel.get('expansion_order', kernel.get('order'))
        order = degree + (2 if name in ('P2M','M2M','M2P') else 1)
        constant, call, geometry = operations[name]
        if name == 'M2L' and report['independent_m2l_ratios']:
            call = 'M2L(L, M, distance, ratio, ratio)'
        expected = kernel['weighted_flops'] + geometry
        expectedFma = kernel['arithmetic'].get('fma',0) + (2 if geometry else 0)
        label = f'Cartesian {name} degree {degree}'
        lines.extend([
            '    {',
            '        using namespace cosmictiger::gravity::cartesian::tracefree;',
            f'        static_assert(cosmictiger::gravity::cartesian::tracefree::detail::{constant}<{degree}> == {kernel["conventional_flops"]});',
            f'        Multipole<Real, {order}> M{{}}, targetM{{}};',
            f'        Expansion<Real, {order}> L{{}}, targetL{{}};',
            '        cosmictiger::gravity::Field<Real> field{};',
            '        Vector3<Real> const distance{Real(0.3), Real(-0.2), Real(0.4)};',
            '        Real const mass(0.7), ratio(0.6);',
            '        Real::operations = 0;',
            '        Real::squareRoots = 0;',
            '        Real::fusedOperations = 0;',
            f'        cosmictiger::gravity::cartesian::tracefree::detail::{call};',
            f'        if (Real::operations != {expected} || Real::fusedOperations != {expectedFma}) {{',
            f'            std::fprintf(stderr, "{label}: got %d ops/%d FMAs, expected {expected}/{expectedFma}\\n",',
            '                Real::operations, Real::fusedOperations);',
            '            return 6;',
            '        }',
            '    }',
        ])
        checks.append(label)


def buildChecks(report, minFmmOrder, maxFmmOrder):
    checks = []
    lines = ['#include <array>', '#include <cstdio>', '',
             '#include <cosmictiger/gravity/kernels/basis_conversion.hpp>',
             '#include <cosmictiger/gravity/kernels/cartesian_tracefree/p2m.hpp>',
             '#include <cosmictiger/gravity/kernels/cartesian_tracefree/p2l.hpp>',
             '#include <cosmictiger/gravity/kernels/cartesian_tracefree/m2m.hpp>',
             '#include <cosmictiger/gravity/kernels/cartesian_tracefree/m2l.hpp>',
             '#include <cosmictiger/gravity/kernels/cartesian_tracefree/m2p.hpp>',
             '#include <cosmictiger/gravity/kernels/cartesian_tracefree/l2l.hpp>',
             '#include <cosmictiger/gravity/kernels/cartesian_tracefree/l2p.hpp>',
             '#include <cosmictiger/gravity/kernels/spherical_kernels.hpp>',
             '#include <cosmictiger/gravity/kernels/spherical_ewald_kernels.hpp>', '']
    appendAvailabilityChecks(lines, report, minFmmOrder, maxFmmOrder)
    lines.extend(['struct CountingRadial {',
                  '    template <int radial, int traces>',
                  '    CountingReal coefficient() const { return CountingReal(0.3); }',
                  '};'])
    lines.extend(['int specialCalls[4]{};',
                  'namespace cosmictiger::gravity::spherical::detail {'])
    for i, name in enumerate(('Exp', 'Erfc', 'Sin', 'Cos')):
        lines.extend(['template <>',
                      f'CountingReal ewald{name}(CountingReal const& value) {{',
                      f'    ++specialCalls[{i}]; return CountingReal(std::{name.lower()}(value.value));',
                      '}'])
    lines.extend(['}',
             'int main() {', '    using Real = CountingReal;',
             '    std::array<Real, 256> source{}, output{};',
             '    Real trace(0.2);',
             '    for (int i = 0; i < 256; ++i) {',
             '        source[i] = Real(0.25 / (i + 1));',
             '        output[i] = Real(0.125);', '    }'])
    appendCartesianChecks(lines, report, checks)
    representations = report['representations']
    for kernel in representations['basisConversion']['kernels']:
        name, order = kernel['operator'], kernel['fmm_order']
        constant = name[0].upper() + name[1:]
        sourceType, targetType = (
            ('cartesian::tracefree::Multipole', 'spherical::Multipole') if name == 'sourceToSpherical'
            else ('spherical::Expansion', 'cartesian::tracefree::Expansion'))
        expected = kernel['weighted_flops']
        fused = kernel['arithmetic'].get('fma', 0)
        if kernel['conventional_flops'] != expected + fused:
            raise ValueError(f'Incorrect conventional basis-conversion count: {kernel}')
        lines.extend(['    {', '        using namespace cosmictiger::gravity;',
            f'        {sourceType}<Real, {order}> source{{}};',
            f'        {targetType}<Real, {order}> target{{}};',
            f'        static_assert(basis_conversion::k{constant}Flops<{order}> == {expected});',
            f'        static_assert(basis_conversion::k{constant}ConventionalFlops<{order}> == {kernel["conventional_flops"]});',
            '        Real::operations = Real::fusedOperations = 0;',
            f'        basis_conversion::{name}(target, source);',
            f'        if (Real::operations != {expected} || Real::fusedOperations != {fused}) {{',
            f'            std::fprintf(stderr, "{name}<{order}> operation count mismatch\\n");',
            '            return 7;', '        }', '    }'])
        checks.append(f'{name}<{order}>')
    namespace = 'spherical'
    spherical = representations['spherical']
    selections = {}
    for selection in spherical['selections']:
        candidates = selection['candidate_flops']
        winner = min(range(len(candidates)), key=lambda i: candidates[i])
        if selection['selected_variant'] != winner or selection['weighted_flops'] != candidates[winner]:
            raise ValueError(f'Incorrect minimum selection: {selection}')
        key = selection['operator'], selection['degree']
        if key in selections:
            raise ValueError(f'Duplicate spherical selection: {selection}')
        variants = [kernel for kernel in spherical['kernels']
                    if (kernel['operator'], kernel['degree']) == key]
        if len(variants) != len(candidates) or any(
                kernel['variant'] != variant or kernel['weighted_flops'] != candidates[variant]
                for variant, kernel in enumerate(sorted(variants, key=lambda kernel: kernel['variant']))):
            raise ValueError(f'Candidate report disagrees with selection: {selection}')
        selections[key] = selection
    for kernel in spherical['kernels']:
        operation = kernel['operator']
        degree = kernel['degree']
        variant = kernel.get('variant')
        if variant is not None:
            selection = selections[operation, degree]
            if variant != selection['selected_variant']:
                continue
        name = {'P2m': 'P2M', 'L2p': 'L2P', 'Greens': 'Green'}[operation]
        parameters = str(degree)
        method = 'evaluate' if operation == 'Greens' else 'add'
        if operation == 'P2m':
            arguments = 'output, Real(0.7), Real(0.3), Real(-0.2), Real(0.4)'
        elif operation == 'Greens':
            arguments = 'output, Real(0.3), Real(-0.2), Real(0.4)'
        else:
            arguments = 'output, source, Real(0.3), Real(-0.2), Real(0.4)'
        fullName = f'cosmictiger::gravity::{namespace}::detail::{name}<{parameters}>'
        label = f'{namespace}::{name}<{parameters}>'
        expected = kernel['weighted_flops']
        lines.extend(['    {', f'        using Kernel = {fullName};',
                      f'        static_assert(Kernel::kFlops_ == {expected});',
                      '        Real::operations = 0;',
                      f'        Kernel::template {method}<Real>({arguments});',
                      f'        if (Real::operations != {expected}) {{',
                      f'            std::fprintf(stderr, "{label}: got %d, expected {expected}\\n", Real::operations);',
                      '            return 1;', '        }', '    }'])
        checks.append(label)
    for order, counts in representations['sphericalEwald']['orders'].items():
        degree = counts['local_degree']
        for name, method, arguments, expected in (
                ('EwaldGreen', 'evaluate',
                 'output, trace, Real(0.13), Real(-0.21), Real(0.31)',
                 counts['green_weighted_flops']),
                ('EwaldL2P', 'add',
                 'output, source, Real(0.23), Real(0.13), Real(-0.21), Real(0.31)',
                 None)):
            fullName = f'cosmictiger::gravity::spherical::{name}<{degree}>'
            lines.extend(['    {', f'        using Kernel = {fullName};',
                          '        Real::operations = 0;', '        Real::squareRoots = 0;',
                          '        for (int i = 0; i < 4; ++i) { specialCalls[i] = 0; }',
                          f'        Kernel::template {method}<Real>({arguments});',
                          '        if (Real::operations != Kernel::kFlops_) {',
                          f'            std::fprintf(stderr, "{name}<{degree}>: got %d, expected %d\\n", Real::operations, Kernel::kFlops_);',
                          '            return 1;', '        }'])
            if expected is not None:
                lines.append(f'        static_assert(Kernel::kFlops_ == {expected});')
            if name == 'EwaldGreen':
                for i, function in enumerate(('exp', 'erfc', 'sin', 'cos')):
                    lines.append(f'        if (specialCalls[{i}] != {counts["special_function_calls"][function]}) {{ return 2; }}')
                lines.append(f'        if (Real::squareRoots != {counts["sqrt_calls"]}) {{ return 3; }}')
            lines.append('    }')
            checks.append(fullName)
    lines.extend([f'    std::printf("{len(checks)} executed weighted FLOP checks and '
                  f'{len(representations["spherical"]["selections"])} minimum selections passed\\n");',
                  '    return 0;', '}'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated-dir', type=Path, required=True,
                        help='Directory containing operation_counts.json and generated headers')
    parser.add_argument('--compiler', default='c++')
    parser.add_argument('--min-fmm-order', type=int,
                        help='Smallest FMM order to compile and audit (default: generated minimum)')
    parser.add_argument('--max-fmm-order', type=int,
                        help='Largest FMM order to compile and audit (default: generated maximum)')
    args = parser.parse_args()
    projectRoot = Path(__file__).resolve().parents[2]
    generated = args.generated_dir.resolve()
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
        for path in generated.rglob('*.hpp'):
            name = path.relative_to(generated)
            destination = includeDir/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(path)
        source = root/'counts.cpp'
        source.write_text(buildChecks(report, minFmmOrder, maxFmmOrder))
        executable = root/'counts'
        subprocess.run([args.compiler, '-std=c++20', '-O0', '-fno-fast-math',
                        '-ffp-contract=off', '-I'+str(root), '-I'+str(projectRoot),
                        '-DORDER=4', f'-DFMM_MIN_ORDER={minFmmOrder}', f'-DFMM_MAX_ORDER={maxFmmOrder}',
                        str(source), '-o', str(executable)], check=True)
        subprocess.run([str(executable)], check=True)


if __name__ == '__main__':
    main()
