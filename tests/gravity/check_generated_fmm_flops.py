#!/usr/bin/env python3
# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
"""Independently execute generated kernels with a weighted counting scalar.

A temporary simd_math adapter replaces the arithmetic backend only. The actual
emitted C++ kernels are compiled unchanged, and every executed primitive charges
its cost. This checks live operations, destination accumulation, rotation setup,
and all candidate/winner metadata, independently of the Python expression DAGs.
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
template <typename Real, typename Value>
Real simdBroadcast(Value value) { return Real(value); }
template <typename Real>
Real simdFma(Real a, Real b, Real c) {
    ++Real::operations; ++Real::fusedOperations;
    return Real(std::fma(a.value, b.value, c.value));
}
template <typename Real>
Real simdSqrt(Real a) {
    Real::operations += 7; ++Real::squareRoots; return Real(std::sqrt(a.value));
}
template <typename Real>
Real simdSelect(bool condition, Real yes, Real no) { return condition ? yes : no; }
template <typename Real, int lanes> class SimdVector;
'''


def cartesianFmmOrder(kernel):
    if 'expansion_order' in kernel:
        return kernel['expansion_order'] + 1
    offset = 2 if kernel['operator'] in ('P2M', 'M2M', 'M2P') else 1
    return kernel['order'] + offset


def sphericalFmmOrder(kernel):
    offset = 2 if kernel['operator'] in ('P2m', 'M2m') else 1
    return kernel['degree'] + offset


def selectReportOrders(report, minFmmOrder, maxFmmOrder):
    selected = deepcopy(report)

    def includes(order):
        return minFmmOrder <= order <= maxFmmOrder

    cartesian = selected['representations']['cartesian']
    for key in ('kernels', 'operators'):
        selected[key] = [row for row in selected[key] if includes(cartesianFmmOrder(row))]
        cartesian[key] = [row for row in cartesian[key] if includes(cartesianFmmOrder(row))]
    spherical = selected['representations']['spherical']
    spherical['kernels'] = [row for row in spherical['kernels'] if includes(sphericalFmmOrder(row))]
    spherical['selections'] = [row for row in spherical['selections'] if includes(row['fmm_order'])]
    ewald = selected['representations']['sphericalEwald']
    ewald['orders'] = {order: counts for order, counts in ewald['orders'].items() if includes(int(order))}
    comparisons = selected['m2l_comparison']
    comparisons['orders'] = [row for row in comparisons['orders'] if includes(row['fmm_order'])]
    return selected


def appendAvailabilityChecks(lines, report, minFmmOrder, maxFmmOrder):
    lines.extend(['template <typename Type>',
                  'concept CompleteKernel = requires { sizeof(Type); };'])
    for order in range(report['min_local_degree'] + 1, report['max_local_degree'] + 2):
        localDegree = order - 1
        sourceDegree = order - 2
        expected = str(minFmmOrder <= order <= maxFmmOrder).lower()
        kernels = (
            f'cartesian::detail::M2lKernel<{localDegree}, false>',
            f'cartesian::detail::P2mKernel<{sourceDegree}>',
            f'cartesian::detail::L2lKernel<{localDegree}, true>',
            f'spherical::M2lKernel<{localDegree}>',
            f'spherical::P2mKernel<{sourceDegree}>',
            f'spherical::EwaldGreensKernel<{localDegree}>',
        )
        for kernel in kernels:
            lines.append(f'static_assert(CompleteKernel<cosmictiger::gravity::{kernel}> == {expected},')
            lines.append(f'    "FMM order {order} availability mismatch: {kernel}");')


def appendCartesianChecks(lines, report, checks):
    comparisons = {row['local_degree']: row for row in report['m2l_comparison']['orders']}
    for kernel in report['representations']['cartesian']['kernels']:
        if 'arithmetic_base' not in kernel:
            continue
        degree = kernel['expansion_order']
        centered = str(kernel['centered']).lower()
        expected = sum(kernel['arithmetic_base'].values()) + kernel['base_accumulations']
        row = comparisons[degree]
        field = 'cartesian_centered_weighted_flops' if kernel['centered'] else 'cartesian_weighted_flops'
        if row[field] != expected + 14 or row['cartesian_geometry_flops'] != 14:
            raise ValueError(f'Incorrect Cartesian M2L comparison: {row}')
        fullName = f'cosmictiger::gravity::cartesian::detail::M2lKernel<{degree}, {centered}>'
        for fullSource in (False, True):
            sourceDegree = degree if fullSource else degree - 1
            arithmetic = kernel['arithmetic_full' if fullSource else 'arithmetic_base']
            accumulations = kernel['base_accumulations']
            if fullSource:
                accumulations += kernel['full_extra_accumulations']
            expected = sum(arithmetic.values()) + accumulations
            fused = arithmetic.get('fma', 0)
            suffix = 'Full_' if fullSource else '_'
            label = f'Cartesian M2L degree {degree}, source={sourceDegree}, centered={centered}'
            lines.extend([
                '    {',
                f'        using Kernel = {fullName};',
                f'        static_assert(Kernel::kWeightedFlops{suffix} == {expected});',
                f'        static_assert(Kernel::kFlops{suffix} == {expected + fused});',
                '        Real x(0.3), y(-0.2), z(0.4);',
                '        Real::operations = 0;',
                '        Real::squareRoots = 0;',
                '        Real::fusedOperations = 0;',
                # Exact arithmetic of fmmInverseRadius, without HPX/app dependencies.
                # simdMax is expanded as its comparison/select equivalent here.
                '        Real const radius = simdSqrt(simdFma(x, x, simdFma(y, y, z * z)));',
                '        Real const inverseRadius = simdSelect(radius > Real(0), Real(1), Real(0)) /',
                '            simdSelect(radius < Real(1e-20), Real(1e-20), radius);',
                '        Real const ux = x * inverseRadius;',
                '        Real const uy = y * inverseRadius;',
                '        Real const uz = z * inverseRadius;',
                '        if (Real::operations != 14 || Real::squareRoots != 1 ||',
                '            Real::fusedOperations != 2) { return 4; }',
                f'        Kernel::template add<Real, decltype(output), decltype(source), {sourceDegree}>(',
                '            output, source, ux, uy, uz, inverseRadius, inverseRadius, inverseRadius);',
                f'        if (Real::operations != {expected + 14} || Real::fusedOperations != {fused + 2}) {{',
                f'            std::fprintf(stderr, "{label}: got %d ops/%d FMAs, expected {expected + 14}/{fused + 2}\\n",',
                '                Real::operations, Real::fusedOperations);',
                '            return 5;',
                '        }',
                '    }'])
            checks.append(label)

    kernels = report['representations']['cartesian']['kernels']
    operators = report['representations']['cartesian']['operators']
    for kernel in kernels + operators:
        if 'arithmetic' not in kernel:
            continue  # M2L is checked above; CUDA warp execution is not scalar.
        operation = kernel['operator']
        degree = kernel['order']
        name = {'P2M': 'P2mKernel', 'M2M': 'M2mKernel', 'P2L': 'P2lKernel',
                'M2P': 'M2pKernel', 'L2L': 'L2lKernel', 'L2P': 'L2pKernel',
                'EwaldM2L': 'EwaldM2lKernel'}[operation]
        parameters = str(degree)
        if operation in ('M2P', 'EwaldM2L'):
            parameters += ', ' + str(kernel['centered']).lower()
        if operation in ('L2L', 'L2P'):
            parameters += ', ' + str(kernel['with_trace']).lower()
        geometry = 'Real(0.3), Real(-0.2), Real(0.4)'
        arguments = {
            'P2M': 'output, Real(0.7), ' + geometry,
            'P2L': 'output, Real(0.7), ' + geometry + ', Real(0.8), Real(0.9)',
            'M2M': 'output, source, ' + geometry + ', Real(0.9)',
            'M2P': 'output, source, ' + geometry + ', Real(0.8), Real(0.9)',
            'L2L': 'output, source, ' + geometry + ', Real(0.9)',
            'L2P': 'output, source, ' + geometry + ', Real(0.9)',
            'EwaldM2L': 'output, source, derivatives',
        }[operation]
        expected = sum(kernel['arithmetic'].values()) + kernel['accumulations']
        fused = kernel['arithmetic'].get('fma', 0)
        label = f'Cartesian {name}<{parameters}>'
        lines.extend([
            '    {',
            f'        using Kernel = cosmictiger::gravity::cartesian::detail::{name}<{parameters}>;',
            f'        static_assert(Kernel::kWeightedFlops_ == {expected});',
            f'        static_assert(Kernel::kFlops_ == {expected + fused});',
            '        Real::operations = 0;',
            '        Real::fusedOperations = 0;',
            f'        Kernel::template add<Real>({arguments});',
            f'        if (Real::operations != {expected} || Real::fusedOperations != {fused}) {{',
            f'            std::fprintf(stderr, "{label}: got %d ops/%d FMAs, expected {expected}/{fused}\\n",',
            '                Real::operations, Real::fusedOperations);',
            '            return 6;',
            '        }',
            '    }'])
        checks.append(label)


def buildChecks(report, minFmmOrder, maxFmmOrder):
    checks = []
    lines = ['#include <array>', '#include <cstdio>', '',
             '#include <cosmictiger/gravity/generated_cartesian_m2l.hpp>',
             '#include <cosmictiger/gravity/generated_cartesian_operators.hpp>',
             '#include <cosmictiger/gravity/nitadori/generated_spherical.hpp>',
             '#include <cosmictiger/gravity/nitadori/generated_spherical_ewald.hpp>', '']
    appendAvailabilityChecks(lines, report, minFmmOrder, maxFmmOrder)
    lines.extend(['int specialCalls[4]{};',
                  'namespace cosmictiger::gravity::spherical::detail {'])
    for i, name in enumerate(('Exp', 'Erfc', 'Sin', 'Cos')):
        lines.extend(['template <>',
                      f'CountingReal ewald{name}(CountingReal const& value) {{',
                      f'    ++specialCalls[{i}]; return CountingReal(std::{name.lower()}(value.value));',
                      '}'])
    lines.extend(['}',
             'int main() {', '    using Real = CountingReal;',
             '    std::array<Real, 256> source{}, derivatives{}, output{};',
             '    Real trace(0.2);',
             '    for (int i = 0; i < 256; ++i) {',
             '        source[i] = Real(0.25 / (i + 1));',
             '        derivatives[i] = Real(0.5 / (i + 1));',
             '        output[i] = Real(0.125);', '    }'])
    appendCartesianChecks(lines, report, checks)
    representations = report['representations']
    namespace = 'spherical'
    for kernel in representations['spherical']['kernels']:
        operation = kernel['operator']
        degree = kernel['degree']
        variant = kernel.get('variant')
        name = operation + ('Variant' if variant is not None else 'Kernel')
        parameters = str(degree) + (f', {variant}' if variant is not None else '')
        method = 'evaluate' if operation == 'Greens' else 'add'
        if operation in ('P2m','P2l'):
            arguments = 'output, Real(0.7), Real(0.3), Real(-0.2), Real(0.4)'
        elif operation == 'Greens':
            arguments = 'output, Real(0.3), Real(-0.2), Real(0.4)'
        elif operation == 'M2lDerivatives':
            arguments = 'output, source, derivatives'
        else:
            arguments = 'output, source, Real(0.3), Real(-0.2), Real(0.4)'
        fullName = f'cosmictiger::gravity::{namespace}::{name}<{parameters}>'
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
    for selection in representations['spherical']['selections']:
        candidates = selection['candidate_flops']
        winner = min(range(len(candidates)), key=lambda i:candidates[i])
        if selection['selected_variant'] != winner or selection['weighted_flops'] != candidates[winner]:
            raise ValueError(f'Incorrect minimum selection: {selection}')
        fullName = f"cosmictiger::gravity::spherical::{selection['operator']}Kernel<{selection['degree']}>"
        lines.append(f'    static_assert({fullName}::kVariant_ == {winner});')
        lines.append(f'    static_assert({fullName}::kFlops_ == {candidates[winner]});')
    for order, counts in representations['sphericalEwald']['orders'].items():
        degree = counts['local_degree']
        for name, method, arguments, expected in (
                ('EwaldGreensKernel', 'evaluate',
                 'output, trace, Real(0.13), Real(-0.21), Real(0.31)',
                 counts['green_weighted_flops']),
                ('EwaldM2lKernel', 'add',
                 'output, trace, source, Real(0.23), derivatives, Real(0.41)',
                 counts['contraction_weighted_flops']),
                ('EwaldL2lKernel', 'add',
                 'output, trace, source, Real(0.23), Real(0.13), Real(-0.21), Real(0.31)',
                 None),
                ('EwaldL2pKernel', 'add',
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
            if name == 'EwaldGreensKernel':
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
        includeDir = root/'cosmictiger/gravity'
        includeDir.mkdir()
        for name in ('generated_cartesian_m2l.hpp', 'generated_cartesian_operators.hpp', 'nitadori/generated_spherical.hpp',
                     'nitadori/generated_spherical_ewald.hpp'):
            (includeDir/name).parent.mkdir(parents=True, exist_ok=True)
            (includeDir/name).symlink_to(generated/name)
        source = root/'counts.cpp'
        source.write_text(buildChecks(report, minFmmOrder, maxFmmOrder))
        executable = root/'counts'
        subprocess.run([args.compiler, '-std=c++20', '-O0', '-fno-fast-math',
                        '-ffp-contract=off', '-I'+str(root), '-I'+str(projectRoot),
                        f'-DFMM_MIN_ORDER={minFmmOrder}', f'-DFMM_MAX_ORDER={maxFmmOrder}',
                        str(source), '-o', str(executable)], check=True)
        subprocess.run([str(executable)], check=True)


if __name__ == '__main__':
    main()
