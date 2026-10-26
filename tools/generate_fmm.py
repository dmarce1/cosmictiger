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
local rank k satisfy n+k<=expansionOrder. The source cap can equal the local
cap, or be one less (the omitted highest source rank affects only potential).

Generation uses Python's standard library and clang-format. A rational expression DAG performs
constant folding, exact-expression CSE and dead-code elimination before C++
emission. No runtime polynomial algebra, factorials, loops or allocations occur
in the generated M2L kernels. Source moments are raw symmetric trace-free
Cartesian tensors; local coefficients are derivatives of the negative Newtonian potential.

Imported from dmarce1/experimental commit
7ba8f1829fdede7348a1418b7cb36c0e7a955302. Local adaptations provide scalar,
CPU SIMD and CUDA emission, point-to-moment/local-to-point operators, and
harmonic Ewald derivative contractions. See tools/cartesian/LICENSE. The periodic
Ewald derivative emitter in cartesian/ewald.py comes from CosmicTiger's former
C++ generator and is distributed under the Boost Software License, Version 1.0.
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
dimensionDefinition = re.search(r'^#define\s+NDIM\s+(\d+)\s*$',
                                (projectRoot / 'cosmictiger/core/defs.hpp').read_text(), re.MULTILINE)
if dimensionDefinition is None:
    raise RuntimeError('defs.hpp must define NDIM as an integer literal')
NDIM = int(dimensionDefinition.group(1))
outputDirectory = projectRoot / 'cosmictiger/gravity'
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
                return f'simdBroadcast<Real>({self.constantNames[a]})'
            q=node[1]
            if q.denominator==1:
                suffix='.0' if abs(q.numerator)>2**63-1 else ''
                return f'simdBroadcast<Real>({q.numerator}{suffix})'
            return f'simdBroadcast<Real>({q.numerator}.0 / {q.denominator}.0)'
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
            expression=self.expr(a)[len('simdBroadcast<Real>('):-1]
            name=f'kCoefficient{a}'
            self.constantNames[a]=name
            out.append(f'{indent}[[maybe_unused]] constexpr auto {name} = {expression};')
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
    only by an add/subtract become explicit simdFma calls; products consumed
    only by one output accumulate directly with simdFma. No reassociation is
    performed. This changes the intended rounding from two operations to one
    fused operation, without duplicating a shared multiplication.

    All signed roots, including conditional outputs, must be supplied together.
    The plan does not mutate the DAG, so its original liveness sets still split
    common and conditional emission correctly. In particular, a value needed
    by both a base output and a conditional output cannot disappear into either
    accumulation. Cooperative CUDA basis emission intentionally uses the raw
    DAG emitter because its outputs belong to distinct lanes and branches.
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
            if node in self.outputCopies:
                output = self.outputCopies[node]
                left, right = graph.nodes[node][1:]
                # Preserve the unfused DAG's operand reads before outputs
                # mutate; copies also retain the original conditional scope.
                lines.append(f'{indent}Real const fmaOutput{output}First = {graph.expr(left)};')
                lines.append(f'{indent}Real const fmaOutput{output}Second = {graph.expr(right)};')
                continue
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
                expression = ('simdFma(' + self.signed(graph.expr(left), productSign) + ', ' +
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
            return [f'{indent}{target} = simdFma({self.signed(f"fmaOutput{output}First", sign)}, '
                    f'fmaOutput{output}Second, {target});']
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


def buildKernel(E,centered):
    g=Dag(); u=[g.symbol('ux'),g.symbol('uy'),g.symbol('uz')]
    eta=g.symbol('sourceRatio'); v=g.symbol('targetRatio'); invr=g.symbol('inverseRadius')
    B=dualBasis(g,u,E)
    weights=[g.one]
    for n in range(1,E+1): weights.append(g.scale(g.mul(weights[-1],eta),2*n-1))
    localWeights=[g.scale(invr,-1)]
    for k in range(1,E+1): localWeights.append(g.mul(localWeights[-1],v))
    contractions={}
    top=g.zero
    for n in range(E+1):
        if centered and n==1: continue
        K=E-n; J=min(n,K)
        M={a:g.symbol(f'source[{index(a)}]') for a in indices(n)}
        get=accessor(g,M)
        C={J:{a:g.mul(weights[n],g.add(*(g.mul(get(plus(a,b)),B[b]) for b in indices(n-J)))) for a in indices(J)}}
        for j in range(J-1,-1,-1):
            get=accessor(g,C[j+1])
            C[j]={a:g.scale(g.add(*(g.mul(u[i],get(plus(a,tuple(int(i==d) for d in range(NDIM)))))
                                      for i in range(NDIM))),Q(1,n-j)) for a in indices(j)}
        if n==E and E>0:
            top=g.mul(localWeights[0],C[0][(0,0,0)])
        else:
            contractions[n]=C
    result={}
    for k in range(E+1):
        for a in indices(k):
            angular={}
            monopoleOnly = k == E or (centered and k == E-1)
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
                coefficient=g.mul(localWeights[k],g.symbol('source[0]'))
                for dim in range(NDIM):
                    if a[dim]%2: coefficient=g.mul(coefficient,u[dim])
                polynomial={tuple(v//2 for v in power):c for power,c in polynomial.items()}
                result[a]=g.mul(coefficient,horner(g,polynomial,[g.mul(x,x) for x in u]))
            else:
                result[a]=g.mul(localWeights[k],horner(g,polynomial,u))
    return g,result,top


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


def emitKernel(E,centered):
    g,result,top=buildKernel(E,centered)
    g,signed=normalizeSigns(g,list(result.values())+[top])
    result=dict(zip(result,signed[:-1])); top,topSign=signed[-1]
    baseLive=g.live(root for root,sign in result.values()); topLive=g.live([top])
    plan=EmissionPlan(g,signed)
    baseStats=plan.statistics(baseLive,range(len(result)))
    fullStats=plan.statistics(baseLive|topLive,range(len(signed)))
    out=[f'template <> class M2lKernel<{E}, {str(centered).lower()}> {{',
         'public:',
         f"\tstatic constexpr int kFlops_ = {baseStats['conventional_flops']};",
         f"\tstatic constexpr int kWeightedFlops_ = {baseStats['weighted_flops']};",
         f"\tstatic constexpr int kFlopsFull_ = {fullStats['conventional_flops']};",
         f"\tstatic constexpr int kWeightedFlopsFull_ = {fullStats['weighted_flops']};",
         '\ttemplate <typename Real, typename Expansion, typename Source, int sourceOrder>',
         '\tCUDA_EXPORT static void add(Expansion& local, Source const& source,',
         '\t\tReal ux, Real uy, Real uz, Real inverseRadius, Real sourceRatio, Real targetRatio) {']
    out+=g.emitConstants(baseLive|topLive)
    out.append('\t\t(void) ux; (void) uy; (void) uz; (void) sourceRatio; (void) targetRatio;')
    out+=plan.emit(baseLive)
    for output,a in enumerate(result):
        out+=plan.accumulate(f'local[{index(a)}]',output)
    if E>0 and top!=g.zero:
        out.append(f'\t\tif constexpr (sourceOrder == {E}) {{')
        out+=plan.emit(topLive,baseLive,'\t\t\t')
        out+=plan.accumulate('local[0]',len(result),'\t\t\t')
        out.append('\t\t}')
    out.extend(['\t}','};',''])
    stats={'expansion_order':E,'centered':centered,'base_source_order':max(0,E-1),
           'arithmetic_base':baseStats['arithmetic'],'arithmetic_full':fullStats['arithmetic'],
           'trace_extra_arithmetic':{},
           'base_accumulations':baseStats['accumulations'],
           'full_extra_accumulations':fullStats['accumulations']-baseStats['accumulations'],
           'base_fused_accumulations':baseStats['fused_accumulations'],
           'full_extra_fused_accumulations':fullStats['fused_accumulations']-baseStats['fused_accumulations'],
           'weighted_flops_base':baseStats['weighted_flops'],
           'weighted_flops_full':fullStats['weighted_flops'],
           'conventional_flops_base':baseStats['conventional_flops'],
           'conventional_flops_full':fullStats['conventional_flops'],
           'trace_accumulations':0}
    return '\n'.join(out),stats


def emitEwaldKernel(sourceOrder, expansionOrder, centered):
    """Contract raw Cartesian moments with externally computed harmonic derivatives.

    A constant Laplacian belongs to the root local expansion and is not a
    source moment. Only harmonic derivative components enter this operator.
    """
    g = Dag()
    source = {alpha: g.symbol(f'source[{index(alpha)}]')
              for degree in range(sourceOrder + 1) for alpha in indices(degree)}
    derivatives = {alpha: g.symbol(f'derivatives[{index(alpha)}]')
                   for degree in range(expansionOrder + 1) for alpha in indices(degree)}
    getSource = accessor(g, source)
    getDerivative = accessor(g, derivatives)
    result = {}
    for degree in range(expansionOrder + 1):
        for alpha in indices(degree):
            terms = []
            for rank in range(min(sourceOrder, expansionOrder - degree) + 1):
                if centered and rank == 1:
                    continue
                for nx in range(rank + 1):
                    for ny in range(rank - nx + 1):
                        beta = (nx, ny, rank - nx - ny)
                        terms.append(g.scale(g.mul(getSource(beta), getDerivative(plus(alpha, beta))),
                                             Q((-1)**rank, fac(beta))))
            result[alpha] = g.add(*terms)
    g, signed = normalizeSigns(g, list(result.values()))
    live = g.live(root for root, sign in signed)
    plan = EmissionPlan(g, signed)
    statistics = plan.statistics(live, range(len(signed)))
    code = [f'template <> class EwaldM2lKernel<{expansionOrder}, {str(centered).lower()}> {{',
            'public:',
            f"\tstatic constexpr int kFlops_ = {statistics['conventional_flops']};",
            f"\tstatic constexpr int kWeightedFlops_ = {statistics['weighted_flops']};",
            '\ttemplate <typename Real, typename Expansion, typename Source, typename Derivatives>',
            '\tCUDA_EXPORT static void add(Expansion& local, Source const& source,',
            '\t\tDerivatives const& derivatives) {']
    code += g.emitConstants(live)
    code += plan.emit(live)
    for output, alpha in enumerate(result):
        code += plan.accumulate(f'local[{index(alpha)}]', output)
    code += ['\t}', '};', '']
    statistics.update({'operator': 'EwaldM2L', 'source_order': sourceOrder,
                       'order': expansionOrder, 'centered': centered})
    return '\n'.join(code), statistics


def generateCartesian(minLocalDegree=2, maxLocalDegree=8):
    """Emit every (source degree p, local degree p + 1) pair in the inclusive range.

    Each degree becomes its own fully specialized kernel class, so all FMM orders
    coexist in one header set and the application selects an order at run time.
    """
    parts=['// Generated by tools/generate_fmm.py. Do not edit.',
           '// Distributed under the Boost Software License, Version 1.0.',
           '#pragma once', '', '#include <cosmictiger/cuda/cuda_export.hpp>',
           '#include <cosmictiger/math/simd_math.hpp>', '',
           'namespace cosmictiger::gravity::cartesian::detail {',
           'template <int expansionOrder, bool centered> class M2lKernel;',
           'template <int expansionOrder, bool centered> class EwaldM2lKernel;', '']
    stats=[]
    from cartesian.particles import generateParticles
    from cartesian.translations import generateTranslations
    from cartesian.cuda import generateWarpL2l
    from cartesian.ewald import generateEwald
    operators=['// Generated by tools/generate_fmm.py. Do not edit.',
        '// Distributed under the Boost Software License, Version 1.0.',
        '#pragma once', '', '#include <cosmictiger/cuda/cuda_export.hpp>',
        '#include <cosmictiger/math/simd_math.hpp>', '',
        'namespace cosmictiger::gravity::cartesian::detail {',
        'template <int order> class M2mKernel;',
        'template <int order, bool withTrace> class L2lKernel;',
        'template <int order> class P2lKernel;',
        'template <int order> class P2mKernel;',
        'template <int order, bool withTrace> class L2pKernel;',
        'template <int order, bool centered> class M2pKernel;', '']
    operatorStats=[]
    for maxOrder in range(minLocalDegree, maxLocalDegree + 1):
        minOrder = maxOrder - 1
        parts.append(f'// FMM order {maxOrder + 1}: source degree {minOrder}, local degree {maxOrder}.')
        for centered in (True,False):
            code,counts=emitKernel(maxOrder,centered); parts.append(code); stats.append(counts)
            code,counts=emitEwaldKernel(minOrder,maxOrder,centered); parts.append(code); stats.append(counts)
        particleParts,particleStats=generateParticles(minOrder,maxOrder,sys.modules[__name__])
        translationParts,translationStats=generateTranslations(minOrder,maxOrder,sys.modules[__name__])
        warpCode,warpStats=generateWarpL2l(maxOrder,sys.modules[__name__])
        operators.append(f'// FMM order {maxOrder + 1}: source degree {minOrder}, local degree {maxOrder}.')
        operators+=translationParts+particleParts+[warpCode]
        operatorStats+=translationStats+particleStats+[warpStats]
    parts.append('} // namespace cosmictiger::gravity::cartesian::detail\n')
    operators.append('} // namespace cosmictiger::gravity::cartesian::detail\n')
    config=f'''// Generated by tools/generate_fmm.py. Do not edit.
#pragma once
// Inclusive local-degree range; each local degree L pairs with source degree L - 1
// and FMM order L + 1.
#define COSMICTIGER_CARTESIAN_GENERATED_MIN_LOCAL_DEGREE {minLocalDegree}
#define COSMICTIGER_CARTESIAN_GENERATED_MAX_LOCAL_DEGREE {maxLocalDegree}
namespace cosmictiger::gravity::cartesian {{
inline constexpr int kMinLocalDegree = {minLocalDegree};
inline constexpr int kMaxLocalDegree = {maxLocalDegree};
inline constexpr int kMinMultipoleDegree = {minLocalDegree - 1};
inline constexpr int kMaxMultipoleDegree = {maxLocalDegree - 1};
}} // namespace cosmictiger::gravity::cartesian
'''
    report={'min_local_degree':minLocalDegree,'max_local_degree':maxLocalDegree,
            'description':'Static emitted arithmetic counts. +, - and * each count one. Explicit FMAs are listed separately: '
            'kFlops_ uses conventional FMA=2; kWeightedFlops_ uses FMA=1. Fused output accumulations are included in arithmetic, '
            'so only add the remaining listed standalone accumulations. Constant rational evaluation is compile time; '
            'geometry normalization/public validation and sqrt/division are excluded. No hardware-timing claim.',
            'kernels':stats,'operators':operatorStats}
    return {'generated_cartesian_m2l.hpp':'\n'.join(parts),
            'generated_cartesian_operators.hpp':'\n'.join(operators),
            'generated_config.hpp':config,
            'generated_cartesian_ewald.hpp':generateEwald(minLocalDegree,maxLocalDegree),
            'operation_counts.json':json.dumps(report,indent=2)+'\n'}


def m2lComparison(cartesianKernels, sphericalSelections):
    """Compare direct M2L at equal degrees, including displacement geometry.

    Cartesian normalization costs one multiply, two FMAs, one sqrt, one
    division, and three coordinate multiplies: 14 with FMA=1 and sqrt=7.
    Application unit conversion is outside both representation APIs.
    """
    geometryFlops = 14
    cartesian = {
        (kernel['expansion_order'], kernel['centered']):
            sum(kernel['arithmetic_base'].values()) + kernel['base_accumulations']
        for kernel in cartesianKernels if 'expansion_order' in kernel
    }
    rows = []
    for selection in sphericalSelections:
        if selection['operator'] != 'M2l':
            continue
        degree = selection['degree']
        rows.append({
            'fmm_order': degree + 1,
            'source_degree': degree - 1,
            'local_degree': degree,
            'cartesian_kernel_flops': cartesian[degree, False],
            'cartesian_geometry_flops': geometryFlops,
            'cartesian_weighted_flops': cartesian[degree, False] + geometryFlops,
            'cartesian_centered_weighted_flops': cartesian[degree, True] + geometryFlops,
            'spherical_weighted_flops': selection['weighted_flops'],
            'spherical_variant': ('none', 'azimuth', 'full')[selection['selected_variant']],
        })
    return {
        'description': 'Free-space M2L, including displacement geometry and additive output; '
            'source degree = FMM order - 2, local degree = FMM order - 1. '
            'Generic source dipoles are retained. The separate centered Cartesian column '
            'is valid only for zero source dipole. Application coordinate-unit conversion '
            'is excluded from both families. +, -, *, / and explicit FMA cost 1; sqrt costs 7. '
            'Both families emit explicit FMAs; no additional compiler contraction is assumed. '
            'These are emitted arithmetic counts, not measured instruction counts or timings.',
        'orders': rows,
    }


def generate(minLocalDegree=2, maxLocalDegree=8):
    """Generate all representations at exactly the same requested order range.

    The application's runtime representation remains Cartesian.
    The spherical kernels are independent generated APIs for later integration.
    Preserve the original report entries for existing arithmetic-count consumers;
    each new family declares its own complete cost convention in its report.
    """
    from fmm.spherical import generate as generateSpherical
    from fmm.spherical_ewald import generate as generateSphericalEwald

    outputs = generateCartesian(minLocalDegree, maxLocalDegree)
    report = json.loads(outputs.pop('operation_counts.json'))
    report['representations'] = {}
    for name, generator in (
            ('spherical', generateSpherical),
            ('sphericalEwald', generateSphericalEwald)):
        generated, counts = generator(minLocalDegree, maxLocalDegree)
        overlap = outputs.keys() & generated.keys()
        if overlap:
            raise RuntimeError(f'Duplicate generated filenames: {sorted(overlap)}')
        outputs.update(generated)
        report['representations'][name] = counts
    report['representations']['cartesian'] = {
        'description': report['description'],
        'kernels': report['kernels'],
        'operators': report['operators'],
    }
    report['m2l_comparison'] = m2lComparison(
        report['kernels'], report['representations']['spherical']['selections'])
    outputs['operation_counts.json'] = json.dumps(report, indent=2) + '\n'
    destinations = {
        'generated_cartesian_ewald.hpp': 'dehnen/generated_cartesian_ewald.hpp',
        'generated_spherical.hpp': 'nitadori/generated_spherical.hpp',
        'generated_spherical_ewald.hpp': 'nitadori/generated_spherical_ewald.hpp',
    }
    return {destinations.get(name, name): contents for name, contents in outputs.items()}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check',action='store_true',help='fail if the files in --output-dir differ')
    ap.add_argument('--min-order',dest='minLocalDegree',type=int,default=2,
                    help='smallest local degree to emit; FMM order is local degree + 1 (default: 2)')
    ap.add_argument('--max-order',dest='maxLocalDegree',type=int,default=8,
                    help='largest local degree to emit (default: 8)')
    ap.add_argument('--output-dir',dest='outputDir',type=Path,required=True,
                    help='directory receiving the generated headers and operation counts')
    ap.add_argument('--no-format',dest='noFormat',action='store_true',
                    help='skip clang-format (build-directory generation; compiled code is unchanged)')
    args=ap.parse_args()
    if args.minLocalDegree < 2 or args.maxLocalDegree < args.minLocalDegree:
        ap.error('local degrees must form an inclusive range starting at 2 or more')
    if args.maxLocalDegree > 8:
        ap.error('local degree must be at most 8 (FMM order 9), the supported production range')
    outputs=generate(args.minLocalDegree,args.maxLocalDegree)
    notice = (projectRoot / 'tools/cartesian/generated_notice.txt').read_text()
    for name, cpp in list(outputs.items()):
        if not name.endswith('.hpp'):
            continue
        cpp = re.sub(r'\bstruct (\w+(?:<[^{}]*>)?)\s*\{', r'class \1 {\npublic:', cpp)
        cpp = re.sub(r'\bstruct (\w+);', r'class \1;', cpp)
        if not args.noFormat:
            cpp = subprocess.run(['clang-format', '--style=file', '--assume-filename=' + str(outputDirectory / name)],
                                 input=cpp, text=True, capture_output=True, check=True).stdout
        outputs[name] = notice + '\n' + cpp

    failed=[]
    for name,text in outputs.items():
        path=args.outputDir/name
        if args.check:
            if not path.exists() or path.read_text()!=text: failed.append(str(path))
        else:
            path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)
    if failed: ap.exit(1,'Stale generated files: '+', '.join(failed)+'\n')
    print('Generated files are current.' if args.check else
          f'Generated Cartesian and Spherical operators '
          f'for local degrees {args.minLocalDegree}..{args.maxLocalDegree} '
          f'(available FMM orders {args.minLocalDegree + 1}..{args.maxLocalDegree + 1}; '
          'compiled runtime range is set by CMake).')


if __name__=='__main__': main()
