# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# Distributed under the GNU General Public License, version 2 or later.

"""Generate a cooperative, full-warp equal-scale local translation.

Rows contain independent Taylor contraction terms. Power-of-two contiguous
lane groups reduce one output each, followed by broadcasts of completed
coefficients. All rank selection, trace reduction and factorial algebra happen
here; CUDA executes a fixed schedule with no tensor-index traversal.
"""


def generateWarpL2l(order, algebra):
    if order > 6:
        raise ValueError('The CUDA warp schedule currently supports degrees through six')
    coefficientCount = (order + 1)**2
    outputs = []
    for rank in range(order):
        for alpha in algebra.indices(rank):
            terms = []
            for degree in range(1, order - rank + 1):
                for beta in algebra.indices(degree):
                    nx, ny, nz = algebra.plus(alpha, beta)
                    if nz < 2:
                        primary, secondary, kind = algebra.index((nx, ny, nz)), 0, 1
                    else:
                        assert nz == 2
                        primary = algebra.index((nx + 2, ny, 0))
                        secondary = algebra.index((nx, ny + 2, 0))
                        kind = 2
                    terms.append((algebra.index(beta), primary, secondary, kind))
            width = 1
            while width * 8 < len(terms):
                width *= 2
            if order - rank >= 2:
                width = max(2, width)
            outputs.append((algebra.index(alpha), alpha, width, terms))
    phases = []
    groups = []
    occupied = 0
    for output in outputs:
        if occupied + output[2] > 32:
            phases.append(groups)
            groups, occupied = [], 0
        groups.append((occupied, output))
        occupied += output[2]
    if groups:
        phases.append(groups)

    schedule = []
    phaseInfo = []
    assigned = set()
    for groups in phases:
        steps = max((len(output[3]) + output[2] - 1) // output[2] for _, output in groups)
        begin = len(schedule)
        for step in range(steps):
            row = [0] * 32
            for firstLane, (target, alpha, width, terms) in groups:
                for offset in range(width):
                    term = step * width + offset
                    if term < len(terms):
                        basis, primary, secondary, kind = terms[term]
                        key = target, term
                        assert key not in assigned, f'Duplicate CUDA L2L term: {key}'
                        assigned.add(key)
                    else:
                        basis, primary, secondary, kind = 0, 0, 0, 0
                    row[firstLane + offset] = (basis | primary << 6 | secondary << 12 |
                                                    kind << 18 | target << 20)
            schedule.append(row)
        phaseInfo.append((begin, steps, groups))
    expected = {(target, term) for target, _, _, terms in outputs for term in range(len(terms))}
    assert assigned == expected, f'CUDA L2L schedule misses terms: {expected - assigned}'
    assert len({target for _, groups in enumerate(phases) for _, (target, _, _, _) in groups}) == order**2

    g = algebra.Dag()
    displacement = [g.symbol(name) for name in ('dx', 'dy', 'dz')]
    basis = algebra.dualBasis(g, displacement, order)
    baseLive = g.live(value for alpha, value in basis.items() if sum(alpha) < order)
    fullLive = g.live(basis.values())
    code = ['#ifdef __CUDACC__',
            '// Read-only GLOBAL storage: row/lane loads coalesce instead of serializing',
            '// divergent constant-memory accesses. Internal linkage needs no device RDC.',
            f'static __device__ unsigned int const l2lWarpSchedule{order}[{len(schedule)}][32] = {{']
    code += ['    {' + ', '.join(f'{value}u' for value in row) + '},' for row in schedule]
    code += ['};', '', 'template <int order, bool withTrace> class L2lWarpKernel;',
             f'template <> class L2lWarpKernel<{order}, true> {{', 'public:']
    reductions = sum(max(output[2] for _, output in groups).bit_length() - 1 for groups in phases)
    fullFlops = sum(g.counts(fullLive).values()) * 32 + len(schedule) * 32 * 5 + reductions * 32 + len(outputs) + 6
    baseFlops = sum(g.counts(baseLive).values()) * 32 + len(schedule) * 32 * 5 + reductions * 32 + len(outputs) + 2
    code += ['    // Whole-warp emitted floating arithmetic; a fused operation counts twice.',
             '    // Includes replicated basis work and padded rows, excludes index/shuffle work.',
             f'    static constexpr int kFlops_ = {fullFlops};',
             f'    static constexpr int kFlopsNoPotential_ = {baseFlops};',
             '    template <typename Real, typename Expansion, typename Source>',
             '    __device__ static void translate(Expansion& target, Source const& source,',
             '        Real const dx, Real const dy, Real const dz, bool doPhi) {',
             '        // Contract: all 32 lanes enter with identical source/geometry; target',
             '        // is private per lane. Every lane receives the complete expansion.',
             '        int const lane = threadIdx.x & 31;',
             '        constexpr unsigned int kMask = 0xffffffffu;',
             '        // Two registers per lane hold the complete distributed Taylor basis.',
             '        // Shuffles avoid a large register-selection tree for dynamic indexing.',
             '        Real basisLow = simdBroadcast<Real>(0);',
             '        Real basisHigh = simdBroadcast<Real>(0);']
    code += g.emitConstants(fullLive, '        ')
    code += g.emit(baseLive, indent='        ')
    for alpha, root in basis.items():
        if sum(alpha) < order:
            index = algebra.index(alpha)
            owner = 'basisLow' if index < 32 else 'basisHigh'
            code += [f'        if (lane == {index & 31}) {{',
                     f'            {owner} = {g.expr(root)};', '        }']
    code.append('        if (doPhi) {')
    code += g.emit(fullLive, baseLive, '            ')
    for alpha, root in basis.items():
        if sum(alpha) == order:
            index = algebra.index(alpha)
            owner = 'basisLow' if index < 32 else 'basisHigh'
            code += [f'            if (lane == {index & 31}) {{',
                     f'                {owner} = {g.expr(root)};', '            }']
    code.append('        }')
    for phase, (begin, steps, groups) in enumerate(phaseInfo):
        activeLanes = groups[-1][0] + groups[-1][1][2]
        widthRuns = []
        for firstLane, output in groups:
            if widthRuns and widthRuns[-1][1] == output[2]:
                widthRuns[-1] = (firstLane + output[2], output[2])
            else:
                widthRuns.append((firstLane + output[2], output[2]))
        if activeLanes < 32:
            widthRuns.append((32, 1))
        widthExpression = str(widthRuns[-1][1])
        for end, width in reversed(widthRuns[:-1]):
            widthExpression = f'lane < {end} ? {width} : ({widthExpression})'
        code += ['        {', f'            // Fixed contraction phase {phase}: {steps} rows.',
                 f'            int const groupWidth = {widthExpression};',
                 '            Real partial = simdBroadcast<Real>(0);',
                 f'            unsigned int const firstEntry = __ldg(&l2lWarpSchedule{order}[{begin}][lane]);',
                 '            #pragma unroll',
                 f'            for (int row = {begin}; row < {begin + steps}; ++row) {{',
                 f'                unsigned int entry = __ldg(&l2lWarpSchedule{order}[row][lane]);']
        if phase == 0:
            width = groups[0][1][2]
            code += [f'                if (!doPhi && lane < {width}) {{',
                     '                    entry = 0;', '                }']
        code += ['                int const basisIndex = entry & 63u;',
                 '                int const primaryIndex = (entry >> 6) & 63u;',
                 '                int const secondaryIndex = (entry >> 12) & 63u;',
                 '                int const kind = (entry >> 18) & 3u;',
                 '                Real const sourceValue = source[primaryIndex] +',
                 '                    simdBroadcast<Real>(kind == 2) * source[secondaryIndex];',
                 '                Real const lowValue = __shfl_sync(kMask, basisLow, basisIndex & 31);',
                 '                Real const highValue = __shfl_sync(kMask, basisHigh, basisIndex & 31);',
                 '                Real const basisValue = basisIndex < 32 ? lowValue : highValue;',
                 '                Real const weight = simdBroadcast<Real>(kind == 0 ? 0 : (kind == 2 ? -1 : 1)) * basisValue;',
                 '                partial = simdFma(weight, sourceValue, partial);', '            }']
        largestWidth = max(output[2] for _, output in groups)
        for shift in (1 << bit for bit in reversed(range(largestWidth.bit_length() - 1))):
            code += [f'            Real const shuffled{shift} = __shfl_down_sync(kMask, partial, {shift});',
                     f'            partial += (lane & (groupWidth - 1)) + {shift} < groupWidth ? shuffled{shift} : simdBroadcast<Real>(0);']
        code += [f'            if ((lane & (groupWidth - 1)) == 0 && lane < {activeLanes}) {{',
                 '                partial += source[(firstEntry >> 20) & 63u];', '            }']
        for firstLane, (target, alpha, width, terms) in groups:
            if target == 0:
                code += [f'            if (doPhi && lane == {firstLane}) {{',
                         f'                partial = simdFma(simdBroadcast<Real>(0.5) * dz * dz, source[{coefficientCount}], partial);',
                         '            }']
            elif alpha == (0, 0, 1):
                code += [f'            if (lane == {firstLane}) {{',
                         f'                partial = simdFma(dz, source[{coefficientCount}], partial);', '            }']
        for firstLane, (target, alpha, width, terms) in groups:
            code.append(f'            target[{target}] = __shfl_sync(kMask, partial, {firstLane});')
        code.append('        }')
    for alpha in algebra.indices(order):
        index = algebra.index(alpha)
        code.append(f'        target[{index}] = source[{index}];')
    code += [f'        target[{coefficientCount}] = source[{coefficientCount}];',
             '        if (!doPhi) {',
             '            target[0] = source[0];',
             '        }',
             '        // The caller may overwrite shared source storage from lane zero on',
             '        // return. Finish EVERY lane source read before permitting that write.',
             '        __syncwarp(kMask);', '    }', '};', '#endif', '']
    statistics = {'operator': 'CUDA_L2L_WARP', 'order': order, 'with_trace': True,
                  'schedule_rows': len(schedule), 'phases': len(phases),
                  'shuffle_instructions_per_lane': reductions + len(outputs) + 2 * len(schedule),
                  'whole_warp_flops_with_potential': fullFlops,
                  'whole_warp_flops_without_potential': baseFlops}
    return '\n'.join(code), statistics
