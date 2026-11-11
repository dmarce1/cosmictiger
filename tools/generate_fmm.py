#!/usr/bin/env python3
# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

# Imported from OctotigerII; distributed under the Boost Software License, Version 1.0.
"""Generate Cartesian and Spherical FMM kernels.

Independent implementation of the harmonic-polynomial/partial-contraction
construction used by PKDGRAV3. No PKDGRAV source is copied. Source rank n and
local rank k satisfy n+k<=expansionOrder. The source cap is one less than
the local cap; the omitted highest source rank affects only potential.

Generation uses Python's standard library and clang-format. A rational expression DAG performs
constant folding, exact-expression CSE and dead-code elimination before C++
emission. No runtime polynomial algebra, factorials, loops or allocations occur
in the generated M2L kernels. Source moments are raw symmetric trace-free
Cartesian tensors; local coefficients are derivatives of the negative Newtonian potential.

Imported from dmarce1/experimental commit
7ba8f1829fdede7348a1418b7cb36c0e7a955302. Local adaptations provide scalar,
CPU SIMD and CUDA emission, point-to-moment/local-to-point operators, and
harmonic derivative contractions. See tools/cartesian/LICENSE.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction as Q
from functools import lru_cache
import json
import sys
import re
import subprocess
from math import factorial, prod, comb
from itertools import permutations
from pathlib import Path


# Regeneration should not leave Python cache artifacts in the source tree.
sys.dont_write_bytecode = True
projectRoot = Path(__file__).resolve().parents[1]
dimensionDefinition = re.search(r'^inline\s+constexpr\s+int\s+kDimensions\s*=\s*(\d+)\s*;\s*$',
                                (projectRoot / 'cosmictiger/core/defs.hpp').read_text(), re.MULTILINE)
if dimensionDefinition is None:
    raise RuntimeError('defs.hpp must define kDimensions as an inline constexpr int integer literal')
NDIM = int(dimensionDefinition.group(1))
outputDirectory = projectRoot / 'cosmictiger/gravity/kernels'
sourceDirectory = projectRoot / 'src/gravity/kernels'
instrumentationDirectory = projectRoot / 'tests/gravity/kernels/instrumentation'
reportPath = outputDirectory / 'operation_counts.json'


def indices(n):
    return [(x,n-x,0) for x in range(n+1)] + [(x,n-1-x,1) for x in range(n)]


def index(a):
    x,y,z=a; n=x+y+z
    return n*n+(x if z==0 else n+1+x)


def plus(a,b):
    return tuple(x+y for x,y in zip(a,b))


def fac(a):
    return prod(factorial(x) for x in a)


class Dag:
    def __init__(self):
        self.nodes=[]; self.cache={}
        self.zero=self.const(0); self.one=self.const(1)

    def node(self, op, *args):
        key=(op,)+args
        if key not in self.cache:
            self.cache[key]=len(self.nodes); self.nodes.append(key)
        return self.cache[key]

    def const(self,x): return self.node('constant',Q(x))
    def symbol(self,x): return self.node('symbol',x)
    def value(self,a): return self.nodes[a][1] if self.nodes[a][0]=='constant' else None

    def split(self,a):
        v=self.value(a)
        if v is not None: return v,self.one
        op,*args=self.nodes[a]
        if op=='*':
            for k in (0,1):
                v=self.value(args[k])
                if v is not None: return v,args[1-k]
        return Q(1),a

    def mul(self,a,b):
        # Fold a literal coefficient into another literal coefficient, but
        # preserve scaled intermediates otherwise: x*(15*C) can reuse 15*C.
        # Unconditionally extracting 15 here repeats that multiplication in
        # every later contraction that consumes C.
        ca,cb=self.value(a),self.value(b)
        if ca is not None and cb is not None: return self.const(ca*cb)
        if ca==0 or cb==0: return self.zero
        if ca==1: return b
        if cb==1: return a
        if ca is not None:
            v,c=self.split(b)
            if v!=1: return self.mul(self.const(ca*v),c)
        if cb is not None:
            v,c=self.split(a)
            if v!=1: return self.mul(self.const(cb*v),c)
        base=self.node('*',*sorted((a,b)))
        candidates=[base]
        # Reassociate only when the alternate inner product already exists.
        # Prefer fewer operations, then older shared subexpressions on ties.
        for x,y in ((a,b),(b,a)):
            if self.nodes[x][0]=='*':
                v,w=self.nodes[x][1:]
                for z,t in ((v,w),(w,v)):
                    pair=('*',)+tuple(sorted((y,z)))
                    if pair in self.cache:
                        candidates.append(self.node('*',*sorted((t,self.cache[pair]))))
        return min(candidates,key=lambda root:(sum(self.counts(self.live([root])).values()),sum(self.live([root])),root))

    def scale(self,a,v): return self.mul(self.const(v),a)

    def add(self,*terms):
        grouped={}
        for t in terms:
            c,a=self.split(t); grouped[a]=grouped.get(a,Q(0))+c
        terms=[(a,c) for a,c in sorted(grouped.items()) if c]
        if not terms: return self.zero
        common=Q(1)
        if all(c<0 for a,c in terms): common=-common
        terms=[(a,c/common) for a,c in terms]
        terms.sort(key=lambda ac: (ac[1]<0,ac[0]))
        def sumGroup(terms):
            if not terms: return self.zero
            out=self.scale(*terms[0])
            for a,c in terms[1:]: out=self.node('+',*sorted((out,self.scale(a,c))))
            return out
        positive=sumGroup([(a,c) for a,c in terms if c>0])
        negative=sumGroup([(a,-c) for a,c in terms if c<0])
        # Group a shared negative sum once, instead of subtracting its terms
        # separately in each output (e.g. the scalar radial combinations).
        out=positive if negative==self.zero else self.node('-',positive,negative)
        return self.scale(out,common)

    def sub(self,a,b): return self.add(a,self.scale(b,-1))

    def live(self, roots):
        done=set()
        def visit(a):
            if a in done: return
            done.add(a)
            if self.nodes[a][0] in ('+','-','*'):
                for b in self.nodes[a][1:]: visit(b)
        for a in roots: visit(a)
        return done

    def expr(self,a):
        node=self.nodes[a]
        if node[0]=='symbol': return node[1]
        if node[0]=='constant':
            if hasattr(self, 'constantNames') and a in self.constantNames:
                return f'math::broadcast<Real>({self.constantNames[a]})'
            q=node[1]
            if q.denominator==1:
                suffix='.0' if abs(q.numerator)>2**63-1 else ''
                return f'math::broadcast<Real>({q.numerator}{suffix})'
            return f'math::broadcast<Real>({q.numerator}.0 / {q.denominator}.0)'
        return f't{a}'

    def emitConstants(self,live,indent='\t\t'):
        """Name scalar coefficients before the generated runtime arithmetic.

        Retain the original literal type and rational evaluation so Real's
        conversions, including SIMD conversions, have unchanged semantics.
        """
        self.constantNames={}
        out=[]
        for a in sorted(live):
            if self.nodes[a][0]!='constant':
                continue
            expression=self.expr(a)[len('math::broadcast<Real>('):-1]
            name=f'k{len(self.constantNames)}'
            self.constantNames[a]=name
            out.append(f'{indent}constexpr auto {name} = {expression};')
        return out

    def emit(self,live,already=frozenset(),indent='\t\t'):
        out=[]
        for a in sorted(live-already):
            op,*args=self.nodes[a]
            if op in ('+','-','*'):
                out.append(f'{indent}Real const t{a} = {self.expr(args[0])} {op} {self.expr(args[1])};')
        return out

    def counts(self,live):
        counts=Counter(self.nodes[a][0] for a in live if self.nodes[a][0] in ('+','-','*'))
        return {op:counts[op] for op in ('*','+','-') if counts[op]}


class EmissionPlan:
    """Fuse single-use products after exact DAG construction and sign folding.

    A product shared by two consumers stays materialized. Products consumed
    only by an add/subtract become explicit math::fma calls; products consumed
    only by one output accumulate directly with math::fma. No reassociation is
    performed. This changes the intended rounding from two operations to one
    fused operation, without duplicating a shared multiplication.

    All signed roots, including conditional outputs, must be supplied together.
    The plan does not mutate the DAG. Inputs and outputs must not alias when
    output products are fused directly into destination accumulations.
    Cooperative CUDA basis emission uses the raw DAG emitter because its
    outputs belong to distinct lanes and branches.
    """
    def __init__(self, graph, signedRoots):
        self.graph = graph
        self.roots = list(signedRoots)
        live = graph.live(root for root, sign in self.roots)
        uses = Counter(root for root, sign in self.roots if root != graph.zero)
        for node in live:
            operation, *arguments = graph.nodes[node]
            if operation in ('+', '-', '*'):
                uses.update(arguments)
        self.fusions = {}
        self.suppressed = set()
        for node in sorted(live):
            operation, *arguments = graph.nodes[node]
            if operation not in ('+', '-'):
                continue
            # Either child saves one operation; deterministic left preference
            # preserves the original addition tree and constant factoring.
            for position, child in enumerate(arguments):
                if graph.nodes[child][0] == '*' and uses[child] == 1:
                    self.fusions[node] = position
                    self.suppressed.add(child)
                    break
        self.fusedOutputs = set()
        for output, (root, sign) in enumerate(self.roots):
            if graph.nodes[root][0] == '*' and uses[root] == 1:
                self.fusedOutputs.add(output)
                self.suppressed.add(root)
        self.outputCopies = {self.roots[output][0]: output for output in self.fusedOutputs}

    @staticmethod
    def signed(expression, sign):
        return expression if sign > 0 else f'(-{expression})'

    def emit(self, live, already=frozenset(), indent='\t\t'):
        graph = self.graph
        lines = []
        for node in sorted(live - already):
            if node in self.suppressed:
                continue
            operation, *arguments = graph.nodes[node]
            if operation not in ('+', '-', '*'):
                continue
            if node in self.fusions:
                position = self.fusions[node]
                product = arguments[position]
                left, right = graph.nodes[product][1:]
                other = arguments[1-position]
                productSign = -1 if operation == '-' and position == 1 else 1
                otherSign = -1 if operation == '-' and position == 0 else 1
                expression = ('math::fma(' + self.signed(graph.expr(left), productSign) + ', ' +
                              graph.expr(right) + ', ' + self.signed(graph.expr(other), otherSign) + ')')
            else:
                expression = f'{graph.expr(arguments[0])} {operation} {graph.expr(arguments[1])}'
            lines.append(f'{indent}Real const t{node} = {expression};')
        return lines

    def accumulate(self, target, output, indent='\t\t'):
        graph = self.graph
        root, sign = self.roots[output]
        if root == graph.zero:
            return []
        if output in self.fusedOutputs:
            left, right = graph.nodes[root][1:]
            return [f'{indent}{target} = math::fma({self.signed(graph.expr(left), sign)}, '
                    f'{graph.expr(right)}, {target});']
        operation = '+=' if sign > 0 else '-='
        return [f'{indent}{target} {operation} {graph.expr(root)};']

    def statistics(self, live, outputs):
        counts = Counter()
        for node in live - self.suppressed:
            operation = self.graph.nodes[node][0]
            if operation in ('+', '-', '*'):
                counts['fma' if node in self.fusions else operation] += 1
        accumulations = 0
        fusedAccumulations = 0
        for output in outputs:
            if self.roots[output][0] == self.graph.zero:
                continue
            if output in self.fusedOutputs:
                counts['fma'] += 1
                fusedAccumulations += 1
            else:
                accumulations += 1
        arithmetic = {operation: counts[operation] for operation in ('*', '+', '-', 'fma')
                      if counts[operation]}
        weighted = sum(arithmetic.values()) + accumulations
        return {'arithmetic': arithmetic, 'accumulations': accumulations,
                'fused_accumulations': fusedAccumulations,
                'weighted_flops': weighted,
                'conventional_flops': weighted + counts['fma']}


def accessor(g,components):
    @lru_cache(None)
    def get(a):
        x,y,z=a
        if z<2: return components[a]
        return g.scale(g.add(get((x+2,y,z-2)),get((x,y+2,z-2))),-1)
    return get


def dualBasis(g,u,degree):
    """Exact harmonic dual basis, factored by coordinate parity.

    Eliminating pairs of z indices from a Cartesian contraction gives this finite
    polynomial directly. The Euler recurrence is equivalent but repeats work.
    All terms share the parity of (a,b,c), leaving a polynomial in x²,y²,z².
    """
    squared=[g.mul(x,x) for x in u]
    B={}; used=set()
    for n in range(degree+1):
        for a,b,c in indices(n):
            polynomial={}
            for kx in range(a//2+1):
                for ky in range(b//2+1):
                    k=kx+ky
                    powers=(a-2*kx,b-2*ky,c+2*k)
                    coefficient=Q((-1)**k*comb(k,kx),fac(powers))
                    polynomial[tuple(v//2 for v in powers)]=g.const(coefficient)
            prefactor=g.one
            for dim,power in enumerate((a,b,c)):
                if power%2: prefactor=g.mul(prefactor,u[dim])
            B[a,b,c]=g.mul(prefactor,horner(g,polynomial,squared,used))
            used.update(g.live([B[a,b,c]]))
    return B


def radialPolynomial(n,a):
    """Exact unit-radius derivatives of |u+t|^(-2n-1), as angular monomials.

    The Kronecker-delta pairings leave a-2b free unit-vector factors. Keeping
    these monomials symbolic lets different source ranks share their angular
    factors before emission, recovering expressions such as PKDGRAV's T0.
    """
    k=sum(a); result={}
    for bx in range(a[0]//2+1):
        for by in range(a[1]//2+1):
            for bz in range(a[2]//2+1):
                b=(bx,by,bz); pairs=sum(b)
                free=tuple(x-2*y for x,y in zip(a,b))
                radial=prod(range(2*n+1,2*n+2*(k-pairs),2))
                coefficient=Q((-1)**(k-pairs)*radial*fac(a),fac(free)*fac(b)*2**pairs)
                result[free]=coefficient
    return result


def horner(g,p,u,used=None):
    """Try all three-variable Horner orders, sharing a global expression DAG."""
    def build(poly,axes):
        if not axes: return poly.get((0,0,0),g.zero)
        axis=axes[0]; groups={}
        for a,value in poly.items():
            b=list(a); degree=b[axis]; b[axis]=0
            groups.setdefault(degree,{})[tuple(b)]=value
        degree=max(groups,default=0)
        result=build(groups.get(degree,{}),axes[1:])
        for j in range(degree-1,-1,-1):
            result=g.add(g.mul(result,u[axis]),build(groups.get(j,{}),axes[1:]))
        return result
    candidates=[build(p,axes) for axes in permutations(range(NDIM))]
    # Select by additional work across outputs, not each output in isolation.
    # Unselected trial expressions are removed by the final liveness pass.
    if used is None:
        if not hasattr(g,'horner_used'): g.horner_used=set()
        used=g.horner_used
    root=min(candidates,key=lambda a:(sum(g.counts(g.live([a])-used).values()),a))
    used.update(g.live([root]))
    return root


def buildKernel(E,independentM2lRatios=False):
    g=Dag(); u=[g.mul(g.symbol(name), g.symbol('iR')) for name in ('dx','dy','dz')]
    invr=g.symbol('iR')
    eta=g.symbol('sourceRatio') if independentM2lRatios else invr
    v=g.symbol('targetRatio') if independentM2lRatios else invr
    B=dualBasis(g,u,E)
    weights=[g.one]
    for n in range(1,E+1): weights.append(g.scale(g.mul(weights[-1],eta),2*n-1))
    localWeights=[g.scale(invr,-1)]
    for k in range(1,E+1): localWeights.append(g.mul(localWeights[-1],v))
    contractions={}
    for n in range(E):
        K=E-n; J=min(n,K)
        M={a:g.symbol(f'M[{index(a)}]') for a in indices(n)}
        get=accessor(g,M)
        C={J:{a:g.mul(weights[n],g.add(*(g.mul(get(plus(a,b)),B[b]) for b in indices(n-J)))) for a in indices(J)}}
        for j in range(J-1,-1,-1):
            get=accessor(g,C[j+1])
            C[j]={a:g.scale(g.add(*(g.mul(u[i],get(plus(a,tuple(int(i==d) for d in range(NDIM)))))
                                      for i in range(NDIM))),Q(1,n-j)) for a in indices(j)}
        contractions[n]=C
    result={}
    for k in range(E+1):
        for a in indices(k):
            angular={}
            monopoleOnly = k == E
            for n,C in contractions.items():
                if n+k>E: continue
                for j,components in C.items():
                    if j>k: continue
                    for b,value in components.items():
                        if any(y>x for x,y in zip(a,b)): continue
                        factor=prod(comb(x,y) for x,y in zip(a,b))
                        derivative=tuple(x-y for x,y in zip(a,b))
                        for power,coefficient in radialPolynomial(n,derivative).items():
                            angular.setdefault(power,[]).append(g.scale(g.one if monopoleOnly else value,coefficient*factor))
            polynomial={power:g.add(*terms) for power,terms in angular.items()}
            if monopoleOnly:
                # Pull the mass into the shared radial weight. Derivatives of
                # 1/r have fixed coordinate parity, so factor odd powers out
                # and evaluate the remainder in squared coordinates.
                coefficient=g.mul(localWeights[k],g.symbol('M[0]'))
                for dim in range(NDIM):
                    if a[dim]%2: coefficient=g.mul(coefficient,u[dim])
                polynomial={tuple(v//2 for v in power):c for power,c in polynomial.items()}
                result[a]=g.mul(coefficient,horner(g,polynomial,[g.mul(x,x) for x in u]))
            else:
                result[a]=g.mul(localWeights[k],horner(g,polynomial,u))
    return g,result


def normalizeSigns(old,roots):
    """Carry signs on edges; absorb them in subtraction and accumulation.

    In particular, x*(-y) must not emit a separate negation followed by an
    addition when the consuming expression can simply subtract x*y.
    """
    new=type(old)()
    def signedSum(a,sa,b,sb):
        if a==new.zero:return b,sb
        if b==new.zero:return a,sa
        va,vb=new.value(a),new.value(b)
        if va is not None and vb is not None:
            val=sa*va+sb*vb
            return new.const(abs(val)),(-1 if val<0 else 1)
        if sa==sb:
            return new.node('+',*sorted((a,b))),sa
        if a==b:return new.zero,1
        # Give a difference and its negation one shared node plus an edge sign.
        if a>b:a,b,sa=b,a,-sa
        return new.node('-',a,b),sa
    @lru_cache(None)
    def visit(i):
        op,*args=old.nodes[i]
        if op=='constant':
            value=args[0]
            return new.const(abs(value)),(-1 if value<0 else 1)
        if op=='symbol':return new.symbol(args[0]),1
        a,sa=visit(args[0]);b,sb=visit(args[1])
        if op=='*':return new.mul(a,b),sa*sb
        if op=='-':sb=-sb
        assert op in ('+','-')
        return signedSum(a,sa,b,sb)
    result=[visit(i) for i in roots]
    return new,result


def emitKernel(E,independentM2lRatios=False):
    g,result=buildKernel(E,independentM2lRatios)
    g,signed=normalizeSigns(g,list(result.values()))
    result=dict(zip(result,signed))
    live=g.live(root for root,sign in result.values())
    plan=EmissionPlan(g,signed)
    stats=plan.statistics(live,range(len(result)))
    extra=', Real sourceRatio, Real targetRatio' if independentM2lRatios else ''
    out=['/// Accumulate Newtonian local coefficients from the source multipole and displacement.',
         f'template <class Real> CUDA_EXPORT void M2L(',
         f'    Expansion<Real, {E+1}>& L, Multipole<Real, {E+1}> const& M,',
         f'    Vector3<Real> const& distance{extra}) {{',
         '    Real const dx = distance[0];',
         '    Real const dy = distance[1];',
         '    Real const dz = distance[2];',
         '    Real const R = math::sqrt(math::fma(dx, dx, math::fma(dy, dy, dz * dz)));',
         '    Real const iR = select(R > broadcast<Real>(0), broadcast<Real>(1), broadcast<Real>(0)) /',
         '        max(R, broadcast<Real>(1.0e-20));',
         ]
    out+=g.emitConstants(live)
    out+=plan.emit(live)
    for output,a in enumerate(result):
        out+=plan.accumulate(f'L[{index(a)}]',output)
    out.append('}')
    if independentM2lRatios:
        out.extend(['/// Accumulate the multipole contribution using inverse-distance scale ratios.',
                    f'template <class Real> CUDA_EXPORT void M2L(',
                    f'    Expansion<Real, {E+1}>& L, Multipole<Real, {E+1}> const& M,',
                    '    Vector3<Real> const& distance) {',
                    '    Real const dx = distance[0];',
                    '    Real const dy = distance[1];',
                    '    Real const dz = distance[2];',
                    '    Real const R = math::sqrt(math::fma(dx, dx, math::fma(dy, dy, dz * dz)));',
                    '    Real const iR = math::select(R > math::broadcast<Real>(0),',
                    '        math::broadcast<Real>(1), math::broadcast<Real>(0)) /',
                    '        math::max(R, math::broadcast<Real>(1.0e-20));',
                    '    M2L(L, M, distance, iR, iR);',
                    '}'])
    out.append('')
    stats.update({'expansion_order':E,'source_order':E-1})
    return '\n'.join(out),stats


def replaceCartesianFields(cpp, sourceDegree, localDegree):
    from cartesian.types import component_name
    def field(number, degree):
        return component_name(next(a for rank in range(degree + 1)
                                   for a in indices(rank) if index(a) == number))
    # Emitted straight-line arithmetic names each coefficient directly.
    for variable, degree in (('Mi', sourceDegree), ('Mo', sourceDegree), ('M', sourceDegree),
                             ('Li', localDegree), ('Lo', localDegree), ('L', localDegree)):
        def replacement(match):
            number = int(match.group(1))
            name = field(number, degree)
            return variable + '.' + name
        cpp = re.sub(r'\b' + variable + r'\[(\d+)\]', replacement, cpp)
    return cpp


def generateCartesianTracefree(minLocalDegree=2, maxLocalDegree=8, independentM2lRatios=False):
    """Generate typed free functions in one header per Cartesian operator."""
    from cartesian.particles import generateParticles
    from cartesian.translations import generateTranslations
    from cartesian.cuda import generateWarpL2l
    from cartesian.types import generate_types
    operators = ('m2l', 'm2m', 'l2l', 'p2m', 'p2l', 'm2p', 'l2p')
    parts = {operator: [] for operator in operators}
    parts['l2l_warp'] = []
    statistics = []
    operatorStats = []
    flopNames = {'m2l':'kM2LFlops',
                 'm2m':'kM2MFlops', 'l2l':'kL2LFlops', 'p2m':'kP2MFlops',
                 'p2l':'kP2LFlops', 'm2p':'kM2PFlops', 'l2p':'kL2PFlops'}
    for localDegree in range(minLocalDegree, maxLocalDegree + 1):
        sourceDegree = localDegree - 1
        direct, directStats = emitKernel(localDegree, independentM2lRatios)
        statistics.append(directStats)
        translations, translationStats = generateTranslations(sourceDegree, localDegree, sys.modules[__name__])
        particles, particleStats = generateParticles(sourceDegree, localDegree, sys.modules[__name__])
        particles[1] += particles[1].replace(
            f'Expansion<Real, {localDegree+1}>& L',
            f'Derivatives<Real, {localDegree+1}>& L')
        warp, warpStats = generateWarpL2l(localDegree, sys.modules[__name__])
        warp = re.sub(r'\bsource\b', 'Li', warp)
        warp = re.sub(r'\btarget\b', 'Lo', warp)
        operatorStats += translationStats + particleStats + [warpStats]
        code = {'m2l': [(direct, directStats, 'm2l')],
                'm2m': [(translations[0], translationStats[0], 'm2m')],
                'l2l': [(translations[1], translationStats[1], 'l2l')],
                'p2m': [(particles[0], particleStats[0], 'p2m')],
                'p2l': [(particles[1], particleStats[1], 'p2l')],
                'm2p': [(particles[2], particleStats[2], 'm2p')],
                'l2p': [(particles[3], particleStats[3], 'l2p')]}
        for operator, variants in code.items():
            parts[operator].append(f'// FMM order {localDegree + 1}.')
            for body, counts, variant in variants:
                parts[operator].append(replaceCartesianFields(body, sourceDegree, localDegree))
                parts[operator].append(f'template <> inline constexpr int '
                                       f'{flopNames[variant]}<{localDegree if operator in ("m2l", "l2l", "p2l", "l2p") else sourceDegree}> = '
                                       f'{counts["conventional_flops"]};')
        parts['l2l_warp'].append(warp)
    outputs = {}
    for operator, definitions in parts.items():
        root = f'cartesian_tracefree/{operator}.hpp'
        if operator == 'l2l_warp':
            include = ('#include <cosmictiger/gravity/kernels/cartesian_tracefree_types.hpp>\n'
                       '#include <cosmictiger/math/vector3.hpp>')
            namespace = 'cosmictiger::gravity::cartesian'
            prelude = ('#ifdef __CUDACC__\n'
                       'template <int degree> inline constexpr int kL2LWarpFlops = 0;\n'
                       'template <int degree> inline constexpr int kL2LWarpNoPotentialFlops = 0;\n'
                       '#endif')
        else:
            include = ('#include <cosmictiger/gravity/fmm_types.hpp>\n'
                       '#include <cosmictiger/gravity/kernels/cartesian_tracefree/geometry.hpp>')
            namespace = 'cosmictiger::gravity::cartesian'
            names = [flopNames[operator]]
            prelude = '\n'.join(f'template <int degree> inline constexpr int {name} = 0;'
                                for name in names)
        outputs[root] = '\n'.join([
            '// Generated by tools/generate_fmm.py. Do not edit.',
            '#pragma once', '',
            '#include <cosmictiger/cuda/cuda_export.hpp>',
            '#include <cosmictiger/math/simd_math.hpp>', '',
            include, '',
            f'namespace {namespace} {{', prelude, *definitions,
            f'}} // namespace {namespace}', ''])
    outputs['cartesian_tracefree/geometry.hpp'] = '''// Generated by tools/generate_fmm.py. Do not edit.
#pragma once

#include <cosmictiger/math/simd_math.hpp>
#include <cosmictiger/math/vector3.hpp>

namespace cosmictiger::gravity::cartesian {
// R is the separation radius; iR is its inverse, zero at coincident centers.
template <class Real> CUDA_EXPORT Real inverseRadius(Vector3<Real> const& distance) {
    Real const dx = distance[0];
    Real const dy = distance[1];
    Real const dz = distance[2];
    Real const R = math::sqrt(math::fma(dx, dx, math::fma(dy, dy, dz * dz)));
    return math::select(R > math::broadcast<Real>(0), math::broadcast<Real>(1),
        math::broadcast<Real>(0)) / math::max(R, math::broadcast<Real>(1.0e-20));
}
} // namespace cosmictiger::gravity::cartesian
'''
    outputs['cartesian_tracefree_types.hpp'] = generate_types(maxLocalDegree, minLocalDegree)
    outputs['cartesian_tracefree_config.hpp'] = f'''// Generated by tools/generate_fmm.py. Do not edit.
#pragma once
namespace cosmictiger::gravity::cartesian {{
inline constexpr int kMinLocalDegree = {minLocalDegree};
inline constexpr int kMaxLocalDegree = {maxLocalDegree};
inline constexpr int kMinMultipoleDegree = {minLocalDegree - 1};
inline constexpr int kMaxMultipoleDegree = {maxLocalDegree - 1};
inline constexpr bool kIndependentM2lRatios = {str(independentM2lRatios).lower()};
}}
'''
    report = {'min_local_degree':minLocalDegree, 'max_local_degree':maxLocalDegree,
              'independent_m2l_ratios':independentM2lRatios,
              'description':'Static emitted arithmetic counts. Conventional FMA=2; weighted FMA=1. '
                            'Distance geometry is evaluated inside direct kernels and excluded from DAG counts.',
              'kernels':statistics, 'operators':operatorStats}
    outputs['operation_counts.json'] = json.dumps(report, indent=2) + '\n'
    return outputs


def generate(minLocalDegree=2, maxLocalDegree=8, independentM2lRatios=False):
    """Generate all representations at exactly the same requested order range.

    The near field's runtime representation is Cartesian tracefree; the spherical
    kernels serve the PMMM far field.
    Preserve the original report entries for existing arithmetic-count consumers;
    each new family declares its own complete cost convention in its report.
    """
    from fmm.spherical import generate as generateSpherical
    from fmm.spherical_ewald import generate as generateSphericalEwald
    from fmm.basis_conversion import generate as generateBasisConversion

    outputs = generateCartesianTracefree(minLocalDegree, maxLocalDegree, independentM2lRatios)
    report = json.loads(outputs.pop('operation_counts.json'))
    report['representations'] = {}
    for name, generator in (
            ('spherical', generateSpherical),
            ('sphericalEwald', generateSphericalEwald),
            ('basisConversion', generateBasisConversion)):
        generated, counts = generator(minLocalDegree, maxLocalDegree)
        overlap = outputs.keys() & generated.keys()
        if overlap:
            raise RuntimeError(f'Duplicate generated filenames: {sorted(overlap)}')
        outputs.update(generated)
        report['representations'][name] = counts
    report['representations']['cartesianTracefree'] = {
        'description': report['description'],
        'kernels': report['kernels'],
        'operators': report['operators'],
    }
    outputs['operation_counts.json'] = json.dumps(report, indent=2) + '\n'
    return outputs


def unnameUnusedGeneratedParameters(cpp):
    start=0
    while (match := re.search(r'CUDA_EXPORT (?:static )?void (?:add|P2M|L2P)\(', cpp[start:])) is not None:
        start += match.start()
        opening=cpp.find('{',start)
        if opening<0: raise RuntimeError('Generated add function has no body')
        depth=1; closing=opening+1
        while depth:
            depth+=(cpp[closing]=='{')-(cpp[closing]=='}')
            closing+=1
        signature=cpp[start:opening]
        body=cpp[opening:closing]
        body=re.sub(r'\(void\)\s*[A-Za-z][A-Za-z0-9_]*\s*;\s*','',body)
        candidates=set(re.findall(r'\bReal(?:\s+const)?\s+([A-Za-z][A-Za-z0-9_]*)\b',signature))
        for name in candidates:
            if not re.search(r'\b'+re.escape(name)+r'\b',body):
                signature,count=re.subn(r'\bReal(\s+const)?\s+'+re.escape(name)+r'\b',
                                         lambda match:'Real'+(match.group(1) or ''),signature)
                if count!=1: raise RuntimeError(f'Cannot unname {name} in generated function')
        cpp=cpp[:start]+signature+body+cpp[closing:]
        start=start+len(signature)+len(body)
    return cpp


SYMBOL_DIRECTORIES = {
    'cartesian_tracefree_types.hpp': """// Symbol directory (Cartesian trace-free coefficients):
// m: rank-zero coefficient; x, xy, xyz, etc.: stored tensor components.
// z-pair components are derived from trace-free symmetry.
""",
    'cartesian_tracefree/geometry.hpp': """// Symbol directory (Cartesian distance geometry):
// dx, dy, dz: target-minus-source displacement; R: its radius; iR: inverse radius.
""",
    'cartesian_tracefree/l2l_warp.hpp': """// Symbol directory (cooperative Cartesian L2L):
// Li: input local expansion; Lo: output local expansion.
// dx, dy, dz: center displacement; basis: distributed Taylor polynomial values.
""",
    'cartesian_tracefree/m2l.hpp': """// Symbol directory (Cartesian M2L):
// M: source multipole; L: destination local expansion.
// dx, dy, dz: target-minus-source center displacement; R: its radius; iR: 1/R.
// Stored tensor fields are named by Cartesian indices; zz and other z-pairs follow from trace-free symmetry.
""",
    'cartesian_tracefree/m2m.hpp': """// Symbol directory (Cartesian M2M):
// Mi: input multipole; Mo: output multipole; dx, dy, dz: center displacement.
// scaleRatio: source scale divided by target scale.
""",
    'cartesian_tracefree/l2l.hpp': """// Symbol directory (Cartesian L2L):
// Li: input local expansion; Lo: output local expansion; dx, dy, dz: center displacement.
// scaleRatio: target scale divided by source scale.
""",
    'cartesian_tracefree/p2m.hpp': """// Symbol directory (Cartesian P2M):
// M: destination multipole; mass: particle mass; dx, dy, dz: particle-center displacement.
""",
    'cartesian_tracefree/p2l.hpp': """// Symbol directory (Cartesian P2L):
// L: destination local expansion; dx, dy, dz: source-target displacement.
// R: source-target radius; iR: 1/R; mass: particle mass.
""",
    'cartesian_tracefree/m2p.hpp': """// Symbol directory (Cartesian M2P):
// M: source multipole; dx, dy, dz: source-target displacement.
// R: source-target radius; iR: 1/R; field: potential and negative gradient.
""",
    'cartesian_tracefree/l2p.hpp': """// Symbol directory (Cartesian L2P):
// L: source local expansion; dx, dy, dz: particle-center displacement.
// field: potential and negative gradient.
// Gradient uses the displacement coordinates; apply inverse length scale at the caller.
""",
    'spherical_kernels.hpp': """// Symbol directory (real spherical FMM operators):
// M: regular-harmonic source multipole; L: singular-harmonic local expansion.
// G: signed singular Green harmonics of -1/R; R: source-target distance.
// x, y, z: source offset for P2M, separation for Green, evaluation displacement for L2P.
// Index (n,m) denotes harmonic degree n and signed storage order m.
""",
    'spherical_ewald_kernels.hpp': """// Symbol directory (periodic spherical Ewald operators):
// L: target local expansion; G: periodic Green harmonics.
// x, y, z: minimum-image target-minus-source displacement in unit-box coordinates.
// sourceTrace, localTrace, greenTrace: independent radial/background terms.
// The primary free-space singularity is handled separately from the periodic correction.
""",
}


def useUnqualifiedMathInGeneratedFunctions(cpp):
    """Keep math helper imports local to each emitted function body."""
    signature = re.compile(r'(?:CUDA_EXPORT|__device__)[^;{}]*\([^;{}]*\)[^;{}]*\{')
    position = 0
    output = []
    while match := signature.search(cpp, position):
        opening = match.end() - 1
        depth = 1
        closing = opening + 1
        while depth:
            if cpp[closing] == '{':
                depth += 1
            elif cpp[closing] == '}':
                depth -= 1
            closing += 1
        output.append(cpp[position:opening + 1])
        body = cpp[opening + 1:closing - 1]
        imports = '\n\tusing namespace math;'
        if 'math::fma(' in body:
            imports += '\n\tFma const fma{};'
        output.append(imports + body.replace('math::', ''))
        output.append('}')
        position = closing
    output.append(cpp[position:])
    return ''.join(output)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check',action='store_true',help='fail if generated declarations or sources differ')
    ap.add_argument('--degree', type=int, default=6, choices=range(2, 9),
                    help='single potential expansion degree to emit (default: 6)')
    ap.add_argument('--output-dir',dest='outputDir',type=Path,default=outputDirectory,
                    help='directory receiving generated declarations and operation counts '
                         '(default: cosmictiger/gravity/kernels in the source tree)')
    ap.add_argument('--source-dir',dest='sourceDir',type=Path,
                    help='directory receiving generated implementations (default: src/gravity/kernels '
                         'in the source tree, or OUTPUT_DIR/sources for a custom --output-dir)')
    ap.add_argument('--instrumentation-dir',dest='instrumentationDir',type=Path,
                    help='directory receiving counting-test implementations (default: '
                         'tests/gravity/kernels/instrumentation in the source tree, or '
                         'OUTPUT_DIR/instrumentation for a custom --output-dir)')
    ap.add_argument('--independent-m2l-ratios',dest='independentM2lRatios',
                    action='store_true',help='emit independent Cartesian M2L source/target scale ratios')
    ap.add_argument('--no-format',dest='noFormat',action='store_true',
                    help='skip clang-format for temporary output; checked-in C++ uses formatting')
    args=ap.parse_args()
    if args.sourceDir is None:
        args.sourceDir = (sourceDirectory if args.outputDir.resolve() == outputDirectory
                          else args.outputDir / 'sources')
    if args.instrumentationDir is None:
        args.instrumentationDir = (instrumentationDirectory
                                   if args.outputDir.resolve() == outputDirectory
                                   else args.outputDir / 'instrumentation')
    from fmm.constants import hoistConstants
    from fmm.sources import executionSpace, singleOrderApi, splitSources

    outputs=generate(args.degree,args.degree,args.independentM2lRatios)
    outputs = {name: unnameUnusedGeneratedParameters(code) if name.endswith('.hpp') else code
               for name, code in outputs.items()}
    outputs = splitSources(singleOrderApi(outputs, args.degree), args.degree)
    notice = (projectRoot / 'tools/cartesian/generated_notice.txt').read_text()
    for name, cpp in list(outputs.items()):
        if not name.endswith(('.hpp', '.cpp', '.cu')):
            continue
        if name.endswith('.hpp'):
            cpp = hoistConstants(cpp)
        # The lightweight coefficient storage has no math calls and must also
        # compile before any header declares namespace math.
        if name not in ('cartesian_tracefree_types.hpp', 'spherical_types.hpp'):
            cpp = useUnqualifiedMathInGeneratedFunctions(cpp)
        directory = SYMBOL_DIRECTORIES.get(name, '')
        cpp = directory + ('\n' if directory else '') + cpp
        if name not in ('cartesian_tracefree_types.hpp', 'spherical_types.hpp'):
            cpp = re.sub(r'\bstruct (\w+(?:<[^{}]*>)?)\s*\{', r'class \1 {\npublic:', cpp)
            cpp = re.sub(r'\bstruct (\w+);', r'class \1;', cpp)
        cpp = executionSpace(cpp, name)
        if not args.noFormat:
            cpp = subprocess.run(['clang-format', '--style=file', '--assume-filename=' + str(outputDirectory / name)],
                                 input=cpp, text=True, capture_output=True, check=True).stdout
        outputs[name] = notice + '\n' + cpp

    failed=[]
    for name,text in outputs.items():
        if name.startswith('sources/'):
            path = args.sourceDir / name.removeprefix('sources/')
        elif name.startswith('instrumentation/'):
            path = args.instrumentationDir / name.removeprefix('instrumentation/')
        else:
            path = args.outputDir / name
        if args.check:
            if not path.exists() or path.read_text()!=text: failed.append(str(path))
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists() or path.read_text()!=text:
                path.write_text(text)
    if failed: ap.exit(1,'Stale generated files: '+', '.join(failed)+'\n')
    print('Generated files are current.' if args.check else
          f'Generated Cartesian and Spherical operators for potential degree {args.degree}.')


if __name__=='__main__': main()
