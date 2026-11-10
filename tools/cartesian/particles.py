# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

# Imported from OctotigerII; distributed under the Boost Software License, Version 1.0.
"""Generated Newtonian particle/local and multipole/particle operators.

Shares the M2L harmonic dual basis, contraction ladder, arithmetic DAG and
factoring passes. The M2P force includes every retained source rank.
"""
from fractions import Fraction as Q

from cartesian.translations import cartesianMonomial


def finish(A,g,outputs,signature,name,order,extra=None):
    g,signed=A.normalizeSigns(g,list(outputs.values()))
    live=g.live(root for root,sign in signed)
    plan=A.EmissionPlan(g,signed)
    stats=plan.statistics(live,range(len(signed)))
    descriptions = {
        'P2M': 'Accumulate the point mass into raw trace-free moments at the supplied displacement.',
        'P2L': 'Accumulate Newtonian local coefficients from a point mass at the supplied displacement.',
        'M2P': 'Accumulate potential and acceleration from all retained multipole ranks.',
        'L2P': 'Accumulate the local potential and negative gradient, scaling acceleration by inverseScale.',
    }
    code=['/// ' + descriptions[name], signature]
    code+=g.emitConstants(live)
    code+=plan.emit(live)
    for output,target in enumerate(outputs):
        code+=plan.accumulate(target,output)
    code+=['}','']
    stats.update({'operator':name,'order':order})
    if extra: stats.update(extra)
    return '\n'.join(code),stats


def p2l(A,P):
    g=A.Dag()
    invr=g.symbol('iR')
    u=[g.mul(g.symbol(x), invr) for x in ('dx','dy','dz')]
    weight=g.scale(g.mul(invr,g.symbol('mass')),-1)
    squared=[g.mul(x,x) for x in u]
    outputs={}
    for k in range(P+1):
        if k: weight=g.mul(weight,invr)
        for a in A.indices(k):
            coefficient=weight
            for axis in range(A.NDIM):
                if a[axis]%2: coefficient=g.mul(coefficient,u[axis])
            polynomial={tuple(v//2 for v in power):g.const(c)
                        for power,c in A.radialPolynomial(0,a).items()}
            outputs[f'L[{A.index(a)}]']=g.mul(coefficient,A.horner(g,polynomial,squared))
    signature=f'''template <class Real> CUDA_EXPORT void P2L(
    Expansion<Real, {P+1}>& L, Real mass, Vector3<Real> const& distance) {{
    Real const iR = inverseRadius(distance);
    Real const dx = distance[0];
    Real const dy = distance[1];
    Real const dz = distance[2];
'''
    return finish(A,g,outputs,signature,'P2L',P)


def m2p(A,P):
    g=A.Dag()
    invr=g.symbol('iR')
    u=[g.mul(g.symbol(x), invr) for x in ('dx','dy','dz')]
    B=A.dualBasis(g,u,P)
    weight=g.one; scalar=[]; radial=[]; vector=[[] for _ in range(A.NDIM)]
    for n in range(P+1):
        if n: weight=g.scale(g.mul(weight,invr),2*n-1)
        M={a:g.symbol(f'M[{A.index(a)}]') for a in A.indices(n)}
        if n==0:
            value=M[(0,0,0)]
        else:
            get=A.accessor(g,M)
            first=[]
            for axis in range(A.NDIM):
                a=tuple(int(d==axis) for d in range(A.NDIM))
                first.append(g.mul(weight,g.add(*(g.mul(get(A.plus(a,b)),B[b])
                                                  for b in A.indices(n-1)))))
                vector[axis].append(first[-1])
            value=g.scale(g.add(*(g.mul(u[d],first[d]) for d in range(A.NDIM))),Q(1,n))
        scalar.append(value)
        radial.append(g.scale(value,2*n+1))
    potential=g.scale(g.mul(invr,g.add(*scalar)),-1)
    r2inv=g.mul(invr,invr); radial=g.add(*radial)
    outputs={'field[0]':potential}
    for axis in range(A.NDIM):
        outputs[f'field[{axis+1}]']=g.mul(r2inv,g.sub(g.add(*vector[axis]),g.mul(u[axis],radial)))
    signature=f'''template <class Real> CUDA_EXPORT void M2P(
    gravity::Field<Real>& field, Multipole<Real, {P+2}> const& M,
    Vector3<Real> const& distance) {{
    Real const iR = inverseRadius(distance);
    Real const dx = distance[0];
    Real const dy = distance[1];
    Real const dz = distance[2];
'''
    return finish(A,g,outputs,signature,'M2P',P)


def p2m(A, order):
    """Project a point mass into raw Cartesian moments at normalized displacement."""
    g = A.Dag()
    displacement = [g.symbol(name) for name in ('dx', 'dy', 'dz')]
    mass = g.symbol('mass')
    outputs = {}
    for degree in range(order + 1):
        for alpha in A.indices(degree):
            polynomial = {power: g.const(coefficient)
                          for power, coefficient in cartesianMonomial(alpha)}
            outputs[f'M[{A.index(alpha)}]'] = g.mul(
                mass, A.horner(g, polynomial, displacement))
    signature = f'''template <class Real> CUDA_EXPORT void P2M(
    Multipole<Real, {order+2}>& M, Real mass, Vector3<Real> const& distance) {{
    Real const dx = distance[0];
    Real const dy = distance[1];
    Real const dz = distance[2];
'''
    return finish(A, g, outputs, signature, 'P2M', order)


def l2p(A, order):
    """Evaluate a local Taylor polynomial and its negative gradient."""
    g = A.Dag()
    displacement = [g.symbol(name) for name in ('dx', 'dy', 'dz')]
    inverseScale = g.symbol('inverseScale')
    basis = A.dualBasis(g, displacement, order)
    source = {alpha: g.symbol(f'L[{A.index(alpha)}]')
              for degree in range(order + 1) for alpha in A.indices(degree)}
    get = A.accessor(g, source)
    outputs = {}
    for component in range(A.NDIM + 1):
        alpha = tuple(int(component == axis + 1) for axis in range(A.NDIM))
        terms = [g.mul(get(A.plus(alpha, beta)), basis[beta])
                 for degree in range(order - sum(alpha) + 1)
                 for beta in A.indices(degree)]
        value = g.add(*terms)
        outputs[f'field[{component}]'] = value if component == 0 else g.scale(g.mul(inverseScale, value), -1)
    signature = f'''template <class Real> CUDA_EXPORT void L2P(
    gravity::Field<Real>& field, Expansion<Real, {order+1}> const& L,
    Vector3<Real> const& distance, Real inverseScale) {{
    Real const dx = distance[0];
    Real const dy = distance[1];
    Real const dz = distance[2];
'''
    return finish(A, g, outputs, signature, 'L2P', order)


def generateParticles(minOrder,maxOrder,algebra):
    code=[]
    stats=[]
    for operation, args in [(p2m, (minOrder,)), (p2l, (maxOrder,)),
                            (m2p, (minOrder,)), (l2p, (maxOrder,))]:
        text, counts = operation(algebra, *args)
        code.append(text)
        stats.append(counts)
    return code,stats
