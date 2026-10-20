# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# Distributed under the Boost Software License, Version 1.0.

"""Generate a cooperative, full-warp equal-scale local translation.

Rows contain independent Taylor contraction terms. Power-of-two contiguous
lane groups reduce one output each, followed by broadcasts of completed
coefficients. All rank selection, trace reduction and factorial algebra happen
here; CUDA executes a fixed schedule with no tensor-index traversal.

One invocation emits one fully specialized ``L2lWarpKernel<L, true>`` for the
requested local degree L in 2..11. The schedule table, the per-phase target
table, the number of basis registers, the phase code and every lane-group width
are compile-time constants of that degree; the emitted CUDA contains no runtime
loop over degrees and no dynamically sized array.

Schedule entries pack three 8-bit coefficient indices and a 2-bit kind:
``basis | primary << 8 | secondary << 16 | kind << 24``. Kind 0 is padding,
kind 1 adds ``source[primary] * B``, and kind 2 adds the trace-reduced
``-(source[primary] + source[secondary]) * B`` for a recovered ``nz == 2``
component. The output coefficient of each lane group is not packed into the
entry; a separate per-phase table maps every lane to its group's target.
"""

from collections import Counter

kMinDegree = 2
kMaxDegree = 11
kLanes = 32
kIndexBits = 8
kKindBits = 2


def generateWarpL2l(order, algebra):
    if order < kMinDegree or order > kMaxDegree:
        raise ValueError(f'The CUDA warp L2L schedule supports local degrees {kMinDegree} '
                         f'through {kMaxDegree}; degree {order} was requested')
    traceIndex = (order + 1)**2
    coefficientCount = traceIndex + 1
    indexLimit = 1 << kIndexBits
    assert coefficientCount <= indexLimit, 'Coefficient indices must fit the packed entry'
    basisWords = (traceIndex + kLanes - 1) // kLanes
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
                        assert nz == 2, f'Stored components carry nz < 2; got {alpha} + {beta}'
                        primary = algebra.index((nx + 2, ny, 0))
                        secondary = algebra.index((nx, ny + 2, 0))
                        kind = 2
                    terms.append((algebra.index(beta), primary, secondary, kind))
            width = 1
            while width * 8 < len(terms):
                width *= 2
            if order - rank >= 2:
                width = max(2, width)
            assert width <= kLanes, f'Output {alpha} needs a lane group wider than a warp'
            outputs.append((algebra.index(alpha), alpha, width, terms))
    phases = []
    groups = []
    occupied = 0
    for output in outputs:
        if occupied + output[2] > kLanes:
            phases.append(groups)
            groups, occupied = [], 0
        groups.append((occupied, output))
        occupied += output[2]
    if groups:
        phases.append(groups)
    for groups in phases:
        for firstLane, (target, alpha, width, terms) in groups:
            assert width & (width - 1) == 0, f'Lane group width {width} for {alpha} is not a power of two'
            assert firstLane + width <= kLanes, f'Lane group for {alpha} exceeds the warp'
            assert firstLane % width == 0, f'Lane group for {alpha} is not aligned to its width'
        activeLanes = groups[-1][0] + groups[-1][1][2]
        assert activeLanes <= kLanes, 'A phase activates more lanes than a warp holds'
    groupCounts = Counter(target for groups in phases for _, (target, _, _, _) in groups)
    expectedTargets = {algebra.index(alpha) for rank in range(order) for alpha in algebra.indices(rank)}
    assert set(groupCounts) == expectedTargets, 'Every output below the top rank needs a lane group'
    assert all(count == 1 for count in groupCounts.values()), 'An output owns more than one lane group'

    schedule = []
    targets = []
    phaseInfo = []
    assigned = set()
    for groups in phases:
        steps = max((len(output[3]) + output[2] - 1) // output[2] for _, output in groups)
        begin = len(schedule)
        laneTargets = [0] * kLanes
        for firstLane, (target, alpha, width, terms) in groups:
            for offset in range(width):
                laneTargets[firstLane + offset] = target
        targets.append(laneTargets)
        for step in range(steps):
            row = [0] * kLanes
            for firstLane, (target, alpha, width, terms) in groups:
                for offset in range(width):
                    term = step * width + offset
                    if term < len(terms):
                        basis, primary, secondary, kind = terms[term]
                        key = target, term
                        assert key not in assigned, f'Duplicate CUDA L2L term: {key}'
                        assigned.add(key)
                        for value in (basis, primary, secondary):
                            assert 0 <= value < indexLimit, f'Packed index {value} exceeds {kIndexBits} bits'
                        assert 1 <= kind < (1 << kKindBits), f'Packed kind {kind} exceeds {kKindBits} bits'
                        row[firstLane + offset] = (basis | primary << kIndexBits |
                                                   secondary << 2 * kIndexBits | kind << 3 * kIndexBits)
                    else:
                        assert row[firstLane + offset] == 0, 'Padded entries must stay kind 0'
            schedule.append(row)
        phaseInfo.append((begin, steps, groups))
    expected = {(target, term) for target, _, _, terms in outputs for term in range(len(terms))}
    assert assigned == expected, f'CUDA L2L schedule misses terms: {expected - assigned}'
    assert len({target for groups in phases for _, (target, _, _, _) in groups}) == order**2

    g = algebra.Dag()
    displacement = [g.symbol(name) for name in ('dx', 'dy', 'dz')]
    basis = algebra.dualBasis(g, displacement, order)
    baseLive = g.live(value for alpha, value in basis.items() if sum(alpha) < order)
    fullLive = g.live(basis.values())
    scheduleName = f'l2lWarpSchedule{order}'
    targetsName = f'l2lWarpTargets{order}'
    code = ['#ifdef __CUDACC__',
            '// Read-only GLOBAL storage: row/lane loads coalesce instead of serializing',
            '// divergent constant-memory accesses. Internal linkage needs no device RDC.',
            f'// Entry layout: basis | primary << {kIndexBits} | secondary << {2 * kIndexBits} | kind << {3 * kIndexBits}.',
            f'static __device__ unsigned int const {scheduleName}[{len(schedule)}][{kLanes}] = {{']
    code += ['    {' + ', '.join(f'{value}u' for value in row) + '},' for row in schedule]
    code += ['};',
             '// Per phase, the output coefficient reduced by each lane\'s group. Inactive',
             '// lanes hold zero and are excluded by the generated active-lane bounds.',
             f'static __device__ unsigned char const {targetsName}[{len(targets)}][{kLanes}] = {{']
    code += ['    {' + ', '.join(f'{value}u' for value in row) + '},' for row in targets]
    code += ['};', '', 'template <int order, bool withTrace> class L2lWarpKernel;',
             f'template <> class L2lWarpKernel<{order}, true> {{', 'public:']
    reductions = sum(max(output[2] for _, output in groups).bit_length() - 1 for groups in phases)
    fullFlops = sum(g.counts(fullLive).values()) * 32 + len(schedule) * 32 * 5 + reductions * 32 + len(outputs) + 6
    baseFlops = sum(g.counts(baseLive).values()) * 32 + len(schedule) * 32 * 5 + reductions * 32 + len(outputs) + 2
    code += ['    // Whole-warp emitted floating arithmetic; a fused operation counts twice.',
             '    // Includes replicated basis work and padded rows, excludes index/shuffle work.',
             f'    static constexpr int kFlops_ = {fullFlops};',
             f'    static constexpr int kFlopsNoPotential_ = {baseFlops};',
             f'    static constexpr int kDegree_ = {order};',
             f'    static constexpr int kTraceIndex_ = {traceIndex};',
             f'    static constexpr int kCoefficients_ = {coefficientCount};',
             f'    static constexpr int kBasisWords_ = {basisWords};',
             f'    static constexpr int kScheduleRows_ = {len(schedule)};',
             f'    static constexpr int kPhases_ = {len(phases)};',
             '    template <typename Real, typename Expansion, typename Source>',
             '    __device__ static void translate(Expansion& target, Source const& source,',
             '        Real const dx, Real const dy, Real const dz, bool doPhi) {',
             '        // Contract: all 32 lanes enter with identical source/geometry; target',
             '        // is private per lane. Every lane receives the complete expansion.',
             '        execute<true>(target, source, dx, dy, dz, doPhi);',
             '    }',
             '    template <typename Real, typename Source>',
             '    __device__ static void translateInto(Real* target, Source const& source,',
             '        Real const dx, Real const dy, Real const dz, bool doPhi) {',
             f'        // Contract: all 32 lanes enter with identical source/geometry; target',
             f'        // addresses {coefficientCount} values that never alias source. Group leaders store',
             '        // their coefficient once; strided lanes copy the unchanged ones. Every',
             '        // source read and target write is complete when this returns.',
             '        execute<false>(target, source, dx, dy, dz, doPhi);',
             '    }',
             '',
             'private:',
             '    template <bool kReplicate, typename Real, typename Target, typename Source>',
             '    __device__ static void execute(Target& target, Source const& source,',
             '        Real const dx, Real const dy, Real const dz, bool doPhi) {',
             '        int const lane = threadIdx.x & 31;',
             '        constexpr unsigned int kMask = 0xffffffffu;',
             f'        // {basisWords} register(s) per lane hold the complete distributed Taylor basis:',
             '        // basis index i lives in word i >> 5 on lane i & 31. Shuffles avoid a',
             '        // large register-selection tree for dynamic indexing.']
    code += [f'        Real basis{word} = simdBroadcast<Real>(0);' for word in range(basisWords)]
    code += g.emitConstants(fullLive, '        ')
    code += g.emit(baseLive, indent='        ')
    for alpha, root in basis.items():
        if sum(alpha) < order:
            index = algebra.index(alpha)
            code += [f'        if (lane == {index & 31}) {{',
                     f'            basis{index >> 5} = {g.expr(root)};', '        }']
    code.append('        if (doPhi) {')
    code += g.emit(fullLive, baseLive, '            ')
    for alpha, root in basis.items():
        if sum(alpha) == order:
            index = algebra.index(alpha)
            code += [f'            if (lane == {index & 31}) {{',
                     f'                basis{index >> 5} = {g.expr(root)};', '            }']
    code.append('        }')
    basisSelect = f'basisValue{basisWords - 1}'
    for word in reversed(range(basisWords - 1)):
        basisSelect = f'basisIndex < {(word + 1) * kLanes} ? basisValue{word} : ({basisSelect})'
    for phase, (begin, steps, groups) in enumerate(phaseInfo):
        activeLanes = groups[-1][0] + groups[-1][1][2]
        widthRuns = []
        for firstLane, output in groups:
            if widthRuns and widthRuns[-1][1] == output[2]:
                widthRuns[-1] = (firstLane + output[2], output[2])
            else:
                widthRuns.append((firstLane + output[2], output[2]))
        if activeLanes < kLanes:
            widthRuns.append((kLanes, 1))
        widthExpression = str(widthRuns[-1][1])
        for end, width in reversed(widthRuns[:-1]):
            widthExpression = f'lane < {end} ? {width} : ({widthExpression})'
        code += ['        {', f'            // Fixed contraction phase {phase}: {steps} rows, {len(groups)} outputs.',
                 f'            int const groupWidth = {widthExpression};',
                 f'            int const targetIndex = __ldg(&{targetsName}[{phase}][lane]);',
                 '            Real partial = simdBroadcast<Real>(0);',
                 '            #pragma unroll',
                 f'            for (int row = {begin}; row < {begin + steps}; ++row) {{',
                 f'                unsigned int entry = __ldg(&{scheduleName}[row][lane]);']
        if phase == 0:
            width = groups[0][1][2]
            assert groups[0][1][0] == 0, 'The potential group must open phase zero'
            code += [f'                if (!doPhi && lane < {width}) {{',
                     '                    entry = 0u;', '                }']
        code += [f'                int const basisIndex = entry & {indexLimit - 1}u;',
                 f'                int const primaryIndex = (entry >> {kIndexBits}) & {indexLimit - 1}u;',
                 f'                int const secondaryIndex = (entry >> {2 * kIndexBits}) & {indexLimit - 1}u;',
                 f'                int const kind = (entry >> {3 * kIndexBits}) & {(1 << kKindBits) - 1}u;',
                 '                Real const sourceValue = source[primaryIndex] +',
                 '                    simdBroadcast<Real>(kind == 2) * source[secondaryIndex];']
        code += [f'                Real const basisValue{word} = __shfl_sync(kMask, basis{word}, basisIndex & 31);'
                 for word in range(basisWords)]
        code += [f'                Real const basisValue = {basisSelect};',
                 '                Real const weight = simdBroadcast<Real>(kind == 0 ? 0 : (kind == 2 ? -1 : 1)) * basisValue;',
                 '                partial = simdFma(weight, sourceValue, partial);', '            }']
        largestWidth = max(output[2] for _, output in groups)
        for shift in (1 << bit for bit in reversed(range(largestWidth.bit_length() - 1))):
            code += [f'            Real const shuffled{shift} = __shfl_down_sync(kMask, partial, {shift});',
                     f'            partial += (lane & (groupWidth - 1)) + {shift} < groupWidth ? shuffled{shift} : simdBroadcast<Real>(0);']
        code += [f'            bool const leader = (lane & (groupWidth - 1)) == 0 && lane < {activeLanes};',
                 '            if (leader) {',
                 '                partial += source[targetIndex];', '            }']
        for firstLane, (target, alpha, width, terms) in groups:
            if target == 0:
                code += [f'            if (doPhi && lane == {firstLane}) {{',
                         f'                partial = simdFma(simdBroadcast<Real>(0.5) * dz * dz, source[{traceIndex}], partial);',
                         '            }']
            elif alpha == (0, 0, 1):
                code += [f'            if (lane == {firstLane}) {{',
                         f'                partial = simdFma(dz, source[{traceIndex}], partial);', '            }']
        code.append('            if constexpr (kReplicate) {')
        for firstLane, (target, alpha, width, terms) in groups:
            code.append(f'                target[{target}] = __shfl_sync(kMask, partial, {firstLane});')
        code += ['            } else {',
                 '                if (leader) {',
                 '                    target[targetIndex] = partial;',
                 '                }',
                 '            }',
                 '        }']
    topIndices = [algebra.index(alpha) for alpha in algebra.indices(order)]
    assert topIndices == list(range(order**2, traceIndex)), 'Top-rank coefficients must be contiguous'
    code.append('        if constexpr (kReplicate) {')
    for index in topIndices + [traceIndex]:
        code.append(f'            target[{index}] = source[{index}];')
    code += ['            if (!doPhi) {',
             '                target[0] = source[0];',
             '            }',
             '        } else {',
             '            // Top-rank coefficients and the trace copy through unchanged.',
             f'            for (int index = {order**2} + lane; index <= {traceIndex}; index += 32) {{',
             '                target[index] = source[index];',
             '            }',
             '            // Lane zero led the potential group, so this restore is same-thread.',
             '            if (!doPhi && lane == 0) {',
             '                target[0] = source[0];',
             '            }',
             '        }',
             '        // The caller may overwrite shared source storage from lane zero on',
             '        // return. Finish EVERY lane source read (and every target write)',
             '        // before permitting that write.',
             '        __syncwarp(kMask);', '    }', '};', '#endif', '']
    statistics = {'operator': 'CUDA_L2L_WARP', 'order': order, 'with_trace': True,
                  'coefficients': coefficientCount,
                  'schedule_rows': len(schedule), 'phases': len(phases),
                  'basis_words_per_lane': basisWords,
                  'shuffle_instructions_per_lane': reductions + len(outputs) + basisWords * len(schedule),
                  'shuffle_instructions_per_lane_into': reductions + basisWords * len(schedule),
                  'whole_warp_flops_with_potential': fullFlops,
                  'whole_warp_flops_without_potential': baseFlops}
    return '\n'.join(code), statistics
