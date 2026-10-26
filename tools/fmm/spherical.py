# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
"""Generate real spherical FMM arithmetic, independently from published equations.

Dehnen, arXiv:1405.2255, Eqs. (3), (49)-(54), (59)-(66), is the mathematical
source. The author's spherical_fmm implementation helped identify normalization
and finite-order pitfalls, but its C++ emitter is not copied here. Exact rational
constant construction, real conjugate symmetry, common-expression elimination,
and root-driven dead-code elimination specialize all three rotation schedules.
"""
from collections import Counter
from fractions import Fraction as Q
from functools import lru_cache
from math import factorial


class Dag:
    """Intern arithmetic and count exactly the expressions which are emitted.

    Negation, constants, copies, comparisons and selection are not arithmetic
    FLOPs. FMA/add/subtract/multiply/divide cost one and square root costs seven.
    Reciprocal square root is explicitly division by sqrt, hence costs eight.
    """
    def __init__(self):
        self.nodes = []
        self.cache = {}
        self.zero = self.constant(0)
        self.one = self.constant(1)

    def node(self, operation, *arguments):
        key = (operation, *arguments)
        if key not in self.cache:
            self.cache[key] = len(self.nodes)
            self.nodes.append(key)
        return self.cache[key]

    def constant(self, value):
        return self.node('constant', Q(value))

    def symbol(self, name):
        return self.node('symbol', name)

    def value(self, node):
        return self.nodes[node][1] if self.nodes[node][0] == 'constant' else None

    def negate(self, value):
        if self.value(value) is not None:
            return self.constant(-self.value(value))
        if self.nodes[value][0] == 'negate':
            return self.nodes[value][1]
        return self.node('negate', value)

    def multiply(self, left, right):
        a, b = self.value(left), self.value(right)
        if a is not None and b is not None:
            return self.constant(a*b)
        if a == 0 or b == 0:
            return self.zero
        if a == 1:
            return right
        if b == 1:
            return left
        if a == -1:
            return self.negate(right)
        if b == -1:
            return self.negate(left)
        if self.nodes[left][0] == 'negate':
            return self.negate(self.multiply(self.nodes[left][1], right))
        if self.nodes[right][0] == 'negate':
            return self.negate(self.multiply(left, self.nodes[right][1]))
        return self.node('multiply', *sorted((left, right)))

    def scale(self, value, factor):
        return self.multiply(value, self.constant(factor))

    def add(self, left, right):
        a, b = self.value(left), self.value(right)
        if a is not None and b is not None:
            return self.constant(a+b)
        if a == 0:
            return right
        if b == 0:
            return left
        if left == right:
            return self.scale(left, 2)
        if self.nodes[right][0] == 'negate':
            if self.nodes[right][1] == left:
                return self.zero
            return self.node('subtract', left, self.nodes[right][1])
        if self.nodes[left][0] == 'negate':
            if self.nodes[left][1] == right:
                return self.zero
            return self.node('subtract', right, self.nodes[left][1])
        return self.node('add', *sorted((left, right)))

    def subtract(self, left, right):
        return self.add(left, self.negate(right))

    def fma(self, left, right, third):
        a, b = self.value(left), self.value(right)
        if a is not None and b is not None:
            return self.add(self.constant(a*b), third)
        if a == 0 or b == 0:
            return third
        if a in (1,-1):
            return self.add(right if a == 1 else self.negate(right), third)
        if b in (1,-1):
            return self.add(left if b == 1 else self.negate(left), third)
        if third == self.zero:
            return self.multiply(left, right)
        return self.node('fma', *sorted((left, right)), third)

    def divide(self, left, right):
        if self.value(right) is not None:
            return self.scale(left, 1/self.value(right))
        return self.node('divide', left, right)

    def sqrt(self, value):
        return self.node('sqrt', value)

    def selectZero(self, condition, yes, no):
        return self.node('selectZero', condition, yes, no)

    def sumProducts(self, products):
        """Fold conjugate duplicates/signs before forming a chain of FMAs."""
        grouped = {}
        for coefficient, left, right in products:
            for which in range(2):
                value = left if which == 0 else right
                while self.nodes[value][0] == 'negate':
                    coefficient = -coefficient
                    value = self.nodes[value][1]
                if which == 0:
                    left = value
                else:
                    right = value
            if left == self.zero or right == self.zero:
                continue
            key = tuple(sorted((left, right)))
            grouped[key] = grouped.get(key, Q(0)) + coefficient
        value = self.zero
        for (left, right), coefficient in sorted(grouped.items()):
            if coefficient:
                value = self.fma(self.scale(left, coefficient), right, value)
        return value

    def live(self, roots):
        result = set()
        def visit(node):
            if node in result:
                return
            result.add(node)
            if self.nodes[node][0] not in ('constant', 'symbol'):
                for argument in self.nodes[node][1:]:
                    visit(argument)
        for root in roots:
            visit(root)
        return result

    def expression(self, node):
        operation, *arguments = self.nodes[node]
        if operation == 'constant':
            value = arguments[0]
            return f'simdBroadcast<Real>({value.numerator}.0 / {value.denominator}.0)'
        if operation == 'symbol':
            return arguments[0]
        if operation == 'negate':
            return f'(-{self.expression(arguments[0])})'
        return f't{node}'

    def emit(self, roots):
        live = self.live(roots)
        lines = []
        counts = Counter()
        for node in sorted(live):
            operation, *arguments = self.nodes[node]
            if operation in ('constant', 'symbol', 'negate'):
                continue
            values = [self.expression(argument) for argument in arguments]
            if operation in ('add','subtract','multiply','divide'):
                symbol = {'add':'+','subtract':'-','multiply':'*','divide':'/'}[operation]
                expression = f'{values[0]} {symbol} {values[1]}'
            elif operation == 'fma':
                expression = f'simdFma({", ".join(values)})'
            elif operation == 'sqrt':
                expression = f'simdSqrt({values[0]})'
            elif operation == 'selectZero':
                expression = f'simdSelect({values[0]} == simdBroadcast<Real>(0), {values[1]}, {values[2]})'
            else:
                raise ValueError(operation)
            lines.append(f'        Real const t{node} = {expression};')
            counts[operation] += 1
        weighted = sum(count*(7 if operation == 'sqrt' else 0 if operation == 'selectZero' else 1)
                       for operation, count in counts.items())
        return lines, dict(counts), weighted


def index(degree, order):
    return degree*(degree+1)+order


def components(degree):
    return [(n,m) for n in range(degree+1) for m in range(n+1)]


def symbols(g, name, degree):
    return {(n,m):(g.symbol(f'{name}[{index(n,m)}]'),
                   g.symbol(f'{name}[{index(n,-m)}]') if m else g.zero)
            for n,m in components(degree)}


def complexGet(g, values, n, m):
    if n < 0 or abs(m) > n:
        return g.zero, g.zero
    real, imaginary = values[n,abs(m)]
    if m < 0:
        return g.scale(real, (-1)**m), g.scale(imaginary, -(-1)**m)
    return real, imaginary


def complexMultiply(g, left, right):
    ar, ai = left
    br, bi = right
    return (g.fma(ar,br,g.negate(g.multiply(ai,bi))),
            g.fma(ar,bi,g.multiply(ai,br)))


def complexScale(g, value, coefficient):
    return tuple(g.scale(part, coefficient) for part in value)


def complexSumProducts(g, products):
    real, imaginary = [], []
    for coefficient, (ar, ai), (br, bi) in products:
        real.extend(((coefficient,ar,br),(-coefficient,ai,bi)))
        imaginary.extend(((coefficient,ar,bi),(coefficient,ai,br)))
    return g.sumProducts(real), g.sumProducts(imaginary)


def conjugate(g, value):
    return value[0], g.negate(value[1])


def regular(g, degree, x, y, z):
    """Normalized regular solid harmonics U, Dehnen (49),(53)."""
    radiusSquared = g.fma(x,x,g.fma(y,y,g.multiply(z,z)))
    result = {(0,0):(g.one,g.zero)}
    for m in range(degree+1):
        if m:
            result[m,m] = complexScale(g,complexMultiply(g,(x,y),result[m-1,m-1]),Q(1,2*m))
        if m < degree:
            result[m+1,m] = tuple(g.multiply(z,part) for part in result[m,m])
        for n in range(m+2,degree+1):
            zFactor = g.scale(z,Q(2*n-1,n*n-m*m))
            rFactor = g.scale(radiusSquared,Q(-1,n*n-m*m))
            result[n,m] = tuple(g.fma(zFactor,a,g.multiply(rFactor,b))
                                for a,b in zip(result[n-1,m],result[n-2,m]))
    return result


def singular(g, degree, x, y, z):
    """Negative-potential G_nm=(-1)^(n+1) Theta_nm."""
    radiusSquared = g.fma(x,x,g.fma(y,y,g.multiply(z,z)))
    inverseSquared = g.divide(g.one,radiusSquared)
    inverseRadius = g.divide(g.one,g.sqrt(radiusSquared))
    x,y,z = [g.negate(g.multiply(value,inverseSquared)) for value in (x,y,z)]
    result = {(0,0):(g.negate(inverseRadius),g.zero)}
    for m in range(degree+1):
        if m:
            result[m,m] = complexScale(g,complexMultiply(g,(x,y),result[m-1,m-1]),2*m-1)
        if m < degree:
            factor = g.scale(z,2*m+1)
            result[m+1,m] = tuple(g.multiply(factor,part) for part in result[m,m])
        for n in range(m+2,degree+1):
            zFactor = g.scale(z,2*n-1)
            rFactor = g.scale(inverseSquared,-((n-1)**2-m*m))
            result[n,m] = tuple(g.fma(zFactor,a,g.multiply(rFactor,b))
                                for a,b in zip(result[n-1,m],result[n-2,m]))
    return result


@lru_cache(None)
def swapCoefficient(n,m,l):
    """Exact dyadic B matrix of Dehnen Eq. (63)."""
    if n == 0:
        return Q(int(m == 0 and l == 0))
    if abs(l)>n or abs(m)>n:
        return Q(0)
    if m == 0:
        return (swapCoefficient(n-1,0,l-1)-swapCoefficient(n-1,0,l+1))/2
    previous = m-1 if m>0 else m+1
    return (swapCoefficient(n-1,previous,l-1)+swapCoefficient(n-1,previous,l+1)
            +(2 if m>0 else -2)*swapCoefficient(n-1,previous,l))/2


def swapAxes(g, values, degree, multipole):
    """Swap x and z using sparse real blocks, transpose for regular moments."""
    result = {}
    for n,m in components(degree):
        real, imaginary = [], []
        coefficient = lambda l: swapCoefficient(n,l,m) if multipole else swapCoefficient(n,m,l)
        for l in range(n+1):
            positive = coefficient(l)
            negative = (-1)**l*coefficient(-l) if l else Q(0)
            r, i = values[n,l]
            real.append((positive+negative,r,g.one))
            if m:
                imaginary.append((positive-negative,i,g.one))
        result[n,m] = g.sumProducts(real),g.sumProducts(imaginary)
    return result


def rotateZ(g, values, degree, cosine, sine):
    """Unit phases built recursively; zero orders are copied without work."""
    phase = [(g.one,g.zero)]
    for m in range(1,degree+1):
        phase.append(complexMultiply(g,phase[-1],(cosine,sine)))
    return {(n,m):complexMultiply(g,value,phase[m]) for (n,m),value in values.items()}


def geometry(g,x,y,z,full):
    """Axis-safe, lane-wise angles with explicit sqrt/division accounting."""
    rhoSquared = g.fma(x,x,g.multiply(y,y))
    safeSquared = g.selectZero(rhoSquared,g.one,rhoSquared)
    inverseRho = g.divide(g.one,g.sqrt(safeSquared))
    rho = g.multiply(rhoSquared,inverseRho)
    if full:
        radiusSquared = g.fma(z,z,rhoSquared)
        inverseRadius = g.divide(g.one,g.sqrt(radiusSquared))
        cosine1 = g.multiply(y,inverseRho)
        sine1 = g.selectZero(rhoSquared,g.one,g.multiply(x,inverseRho))
        cosine2 = g.selectZero(radiusSquared,g.one,g.multiply(z,inverseRadius))
        sine2 = g.negate(g.multiply(rho,inverseRadius))
        radius = g.multiply(radiusSquared,inverseRadius)
        return cosine1,sine1,cosine2,sine2,radius,inverseRadius
    cosine = g.selectZero(rhoSquared,g.one,g.multiply(x,inverseRho))
    sine = g.negate(g.multiply(y,inverseRho))
    return cosine,sine,rho


def forwardRotation(g,values,degree,multipole,angles,variant):
    if variant == 0:
        return values
    cosine,sine = angles[:2]
    values = rotateZ(g,values,degree,cosine,sine)
    if variant == 2:
        values = swapAxes(g,values,degree,multipole)
        values = rotateZ(g,values,degree,angles[2],angles[3])
        values = swapAxes(g,values,degree,multipole)
    return values


def backwardRotation(g,values,degree,multipole,angles,variant):
    if variant == 2:
        values = swapAxes(g,values,degree,multipole)
        values = rotateZ(g,values,degree,angles[2],g.negate(angles[3]))
        values = swapAxes(g,values,degree,multipole)
    return rotateZ(g,values,degree,angles[0],g.negate(angles[1])) if variant else values


def contraction(g,source,green,sourceDegree,targetDegree,totalDegree):
    result = {}
    for n,m in components(targetDegree):
        products = []
        for k in range(min(sourceDegree,totalDegree-n)+1):
            for l in range(-k,k+1):
                products.append((1,conjugate(g,complexGet(g,source,k,l)),complexGet(g,green,n+k,m+l)))
        result[n,m] = complexSumProducts(g,products)
    return result


def translateMultipole(g,source,harmonics,degree):
    result = {}
    for n,m in components(degree):
        products = []
        for k in range(n+1):
            for l in range(-k,k+1):
                products.append((1,complexGet(g,source,n-k,m-l),complexGet(g,harmonics,k,l)))
        result[n,m] = complexSumProducts(g,products)
    return result


def translateLocal(g,source,harmonics,degree,targetDegree):
    result = {}
    for n,m in components(targetDegree):
        products = []
        for k in range(degree-n+1):
            for l in range(-k,k+1):
                products.append((1,complexGet(g,source,n+k,m+l),conjugate(g,complexGet(g,harmonics,k,l))))
        result[n,m] = complexSumProducts(g,products)
    return result


def pack(values):
    outputs = {}
    for (n,m),(real,imaginary) in values.items():
        outputs[index(n,m)] = real
        if m:
            outputs[index(n,-m)] = imaginary
    return dict(sorted(outputs.items()))


def emitFunction(className,parameters,g,outputs,arguments,additive=True,operation=None,degree=None,variant=None):
    roots = []
    for slot,value in outputs.items():
        if additive:
            # Contract the final multiply directly into the caller's accumulator.
            if g.nodes[value][0] == 'multiply':
                left,right = g.nodes[value][1:]
                value = g.fma(left,right,g.symbol(f'output[{slot}]'))
            else:
                value = g.add(g.symbol(f'output[{slot}]'),value)
        roots.append(value)
    lines,counts,flops = g.emit(roots)
    method = 'add' if additive else 'evaluate'
    declaration = f'template <> class {className}<{parameters}> {{\npublic:\n'
    declaration += f'    static constexpr int kFlops_ = {flops};\n'
    declaration += '    template <typename Real, typename Output, typename Source = Output, typename Derivatives = Source>\n'
    declaration += f'    CUDA_EXPORT static void {method}(Output &output, {arguments}) {{\n'
    declaration += '\n'.join(lines)+'\n'
    for slot,root in zip(outputs,roots):
        declaration += f'        output[{slot}] = {g.expression(root)};\n'
    declaration += '    }\n};\n'
    record = dict(operator=operation or className,degree=degree,variant=variant,
                  arithmetic=counts,weighted_flops=flops)
    return declaration,record


def axialRegular(g,degree,distance):
    result = {(n,m):(g.zero,g.zero) for n,m in components(degree)}
    result[0,0] = g.one,g.zero
    for n in range(1,degree+1):
        result[n,0] = g.scale(g.multiply(result[n-1,0][0],distance),Q(1,n)),g.zero
    return result


def axialSingular(g,degree,inverseRadius):
    result = {(n,m):(g.zero,g.zero) for n,m in components(degree)}
    result[0,0] = g.negate(inverseRadius),g.zero
    for n in range(1,degree+1):
        result[n,0] = g.scale(g.multiply(result[n-1,0][0],inverseRadius),-n),g.zero
    return result


def operatorName(operation):
    return {'Greens':'Green', 'M2lDerivatives':'M2LDerivatives'}.get(operation,operation.upper())


def generateVariant(operation,degree,variant):
    g = Dag()
    x,y,z = [g.symbol(name) for name in ('x','y','z')]
    isM2l = operation in ('M2l','M2p')
    isMultipole = operation == 'M2m' or isM2l
    targetDegree = 1 if operation in ('M2p','L2p') else degree
    sourceDegree = degree-1 if isM2l else degree
    source = symbols(g,'source',sourceDegree)
    # M2M receives old-center minus new-center: moment coordinates are negated.
    if operation == 'M2m':
        x,y,z = [g.negate(value) for value in (x,y,z)]
    if variant == 2 and not isM2l:
        # Compose commuting z and planar translations. Align only the planar
        # displacement: one phase pair and two x/z swaps replace the four swaps
        # needed to align a general 3D displacement. Dehnen Appendix A.6.
        axial = axialRegular(g,degree,z)
        if operation == 'M2m':
            source = translateMultipole(g,source,axial,degree)
        else:
            source = translateLocal(g,source,axial,degree,degree)
        cosine,sine,rho = geometry(g,x,y,z,False)
        source = rotateZ(g,source,sourceDegree,cosine,sine)
        source = swapAxes(g,source,sourceDegree,isMultipole)
        axial = axialRegular(g,degree,rho)
        if operation == 'M2m':
            result = translateMultipole(g,source,axial,degree)
        else:
            result = translateLocal(g,source,axial,degree,targetDegree)
        result = swapAxes(g,result,targetDegree,isMultipole)
        result = rotateZ(g,result,targetDegree,cosine,g.negate(sine))
    else:
        angles = geometry(g,x,y,z,variant==2) if variant else None
        source = forwardRotation(g,source,sourceDegree,isMultipole,angles,variant)
        if variant == 1:
            x,y = angles[2],g.zero
        if isM2l:
            green = axialSingular(g,degree,angles[5]) if variant == 2 else singular(g,degree,x,y,z)
            result = contraction(g,source,green,sourceDegree,targetDegree,degree)
        else:
            harmonics = regular(g,degree,x,y,z)
            if operation == 'M2m':
                result = translateMultipole(g,source,harmonics,degree)
            else:
                result = translateLocal(g,source,harmonics,degree,targetDegree)
        result = backwardRotation(g,result,targetDegree,operation=='M2m',angles,variant)
    outputs = pack(result)
    if targetDegree == 1 and operation in ('M2p','L2p'):
        outputs = {0:outputs[0],1:g.negate(outputs[3]),2:g.negate(outputs[1]),3:g.negate(outputs[2])}
    arguments = 'Source const &source, Real x, Real y, Real z'
    return emitFunction(operatorName(operation),str(degree),g,outputs,arguments,
                        operation=operation,degree=degree,variant=variant)


def generateSimple(operation,degree):
    g = Dag()
    x,y,z = [g.symbol(name) for name in ('x','y','z')]
    if operation == 'P2m':
        values = regular(g,degree,*[g.negate(v) for v in (x,y,z)])
    elif operation in ('P2l','Greens'):
        values = singular(g,degree,x,y,z)
    elif operation == 'M2lDerivatives':
        values = contraction(g,symbols(g,'source',degree-1),symbols(g,'derivatives',degree),degree-1,degree,degree)
    else:
        raise ValueError(operation)
    outputs = pack(values)
    if operation in ('P2m','P2l'):
        mass = g.symbol('mass')
        outputs = {slot:g.multiply(value,mass) for slot,value in outputs.items()}
        arguments = 'Real mass, Real x, Real y, Real z'
    elif operation == 'Greens':
        arguments = 'Real x, Real y, Real z'
    else:
        arguments = 'Source const &source, Derivatives const &derivatives'
    return emitFunction(operatorName(operation),str(degree),g,outputs,arguments,
                        additive=operation!='Greens',operation=operation,degree=degree)


HEADER = '''// Generated by tools/generate_fmm.py. Do not edit.
#pragma once

#include <cosmictiger/gravity/fmm_kernels/spherical_types.hpp>
#include <cosmictiger/math/simd_math.hpp>

namespace cosmictiger::gravity::spherical::detail {

// Every operator adds to output. Output and source must not alias.
// Field ordering is potential, accelerationX, accelerationY, accelerationZ (-gradient).
// M2M displacement is old center minus new center. L2L/L2P displacement is
// new center/point minus old center. M2L displacement is target minus source.
'''


def generate(minLocalDegree=2,maxLocalDegree=8):
    parts = [HEADER]
    report = {'representation':'Spherical', 'cost_model':{'add':1,'subtract':1,'multiply':1,'divide':1,'fma':1,'sqrt':7},
              'description':'Exact live emitted arithmetic per lane, including geometry and output accumulation. Unary signs, copies, comparisons and selects cost zero. Division by square root costs eight. These weighted costs are not measured timings.',
              'kernels':[],'selections':[]}
    for operation in ('M2m','M2l','M2p','L2l','L2p','P2m','P2l','Greens','M2lDerivatives'):
        parts.append(f'template <int degree> class {operatorName(operation)};\n')
    for localDegree in range(minLocalDegree,maxLocalDegree+1):
        parts.append(f'#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= {localDegree + 1}) && '
                     f'(!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= {localDegree + 1})')
        for operation in ('P2m','P2l','Greens','M2lDerivatives'):
            degree = localDegree-1 if operation=='P2m' else localDegree
            code,stats = generateSimple(operation,degree)
            parts.append(code)
            report['kernels'].append(stats)
        for operation in ('M2m','M2l','M2p','L2l','L2p'):
            degree = localDegree-1 if operation=='M2m' else localDegree
            candidates = []
            implementations = []
            for variant in range(3):
                code,stats = generateVariant(operation,degree,variant)
                implementations.append(code)
                report['kernels'].append(stats)
                candidates.append(stats['weighted_flops'])
            selected = min(range(3),key=lambda variant:candidates[variant])
            description = ('no rotation', 'azimuthal rotation', 'axial alignment')[selected]
            parts.append(f'// Selected variant {selected}: {description} (minimum weighted arithmetic count).\n'
                         + implementations[selected])
            report['selections'].append(dict(operator=operation,fmm_order=localDegree+1,degree=degree,
                                             candidate_flops=candidates,selected_variant=selected,weighted_flops=candidates[selected]))
        parts.append('#endif')
    parts.append('} // namespace cosmictiger::gravity::spherical::detail\n')
    return {'spherical_kernels.hpp':'\n'.join(parts)},report
