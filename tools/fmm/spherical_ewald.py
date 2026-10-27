#!/usr/bin/env python3
# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# Distributed under the Boost Software License, Version 1.0.
"""Generate spherical periodic corrections with all irreducible trace terms.

Independent implementation of Nitadori, arXiv:1409.5981, Appendix B, in the
Dehnen/sFMM normalization. The central image is evaluated as a regular
polynomial times a positive lower-gamma series, never singular subtraction.
"""
from __future__ import annotations

from math import factorial, gamma, pi


NOTICE = """/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 */
"""
RADIUS = 6
SERIES_TERMS = 48


def index(n, m):
    return n * (n + 1) + m


def scalar(value):
    return f"simdBroadcast<Real>({float(value):.17g})"


def regular(degree):
    lines = ["output[0] = simdBroadcast<Real>(1);"]
    cost = 0
    fmas = 0
    for m in range(degree + 1):
        if m:
            if m == 1:
                lines += [f"output[{index(m,m)}] = x * {scalar(.5)};",
                          f"output[{index(m,-m)}] = y * {scalar(.5)};"]
                cost += 2
            else:
                a, b = index(m - 1, m - 1), index(m - 1, 1 - m)
                lines += [f"output[{index(m,m)}] = {scalar(1/(2*m))} * simdFma(x, output[{a}], -y * output[{b}]);",
                          f"output[{index(m,-m)}] = {scalar(1/(2*m))} * simdFma(y, output[{a}], x * output[{b}]);"]
                cost += 6
                fmas += 2
        if m < degree:
            for component in ([0] if m == 0 else [m, -m]):
                lines += [f"output[{index(m+1,component)}] = z * output[{index(m,component)}];"]
                cost += 1
        for n in range(m + 2, degree + 1):
            lines += [f"Real const zFactor{n}_{m} = {scalar(2*n-1)} * z;"]
            cost += 1
            for component in ([0] if m == 0 else [m, -m]):
                lines += [f"output[{index(n,component)}] = {scalar(1/(n*n-m*m))} * simdFma(zFactor{n}_{m}, output[{index(n-1,component)}], -radiusSquared * output[{index(n-2,component)}]);"]
                cost += 3
                fmas += 1
    return lines, cost, fmas


def contraction(degree):
    """Expand conj(M_kl) G_(n+k,m+l) using the real-storage symmetry."""
    from .spherical import generateSimple
    _, statistics = generateSimple('M2lDerivatives',degree)
    lines = [f"detail::M2LDerivatives<{degree}>::template add<Real>(local, moments, greens);"]
    cost = statistics['weighted_flops']
    conventional = statistics['conventional_flops']
    # Source radius moment first exists when source degree >=2.
    if degree >= 3:
        lines += [f"local[0] = simdFma(-{scalar(.5)} * greenTrace, sourceTrace, local[0]);"]
        cost += 2
        conventional += 3
    for component, factor in [(3, 2), (1, 2), (2, 1)]:
        operand = "-greenTrace" if factor == 1 else f"-{scalar(factor)} * greenTrace"
        lines += [f"local[{component}] = simdFma({operand}, moments[{component}], local[{component}]);"]
        cost += 1 + (factor != 1)
        conventional += 2 + (factor != 1)
    lines += ["localTrace = simdFma(greenTrace, moments[0], localTrace);"]
    cost += 1
    conventional += 2
    return lines, cost, conventional


def emitDegree(degree):
    count = (degree + 1)**2
    polynomial, polynomial_cost, polynomial_fmas = regular(degree)
    body = [f"template <> class EwaldGreen<{degree}> {{", "public:",
            f"static constexpr int kDegree_ = {degree};", f"static constexpr int kCount_ = {count};",
            f"static constexpr int kLatticeRadius_ = {RADIUS};",
            "template <typename Real>",
            "CUDA_EXPORT static void regular(Real* output, Real x, Real y, Real z, Real radiusSquared) {"]
    body += polynomial + ["}", "template <typename Real, typename Output>",
            "CUDA_EXPORT static void evaluateWorking(Output& output, Real& greenTrace, Real x, Real y, Real z) {",
            f"Real harmonics[{count}];", f"Real compensation[{count}];", f"Real imagePair[{count}];",
            "Real const radiusSquared = simdFma(x, x, simdFma(y, y, z * z));",
            "regular(harmonics, x, y, z, radiusSquared);",
            f"Real const argument = {scalar(4)} * radiusSquared;",
            "Real const exponential = detail::ewaldExp(-argument);"]
    central_cost = 4 + polynomial_cost
    for n in range(degree + 1):
        body += [f"Real series{n} = simdBroadcast<Real>(1);"]
        for term in reversed(range(SERIES_TERMS)):
            body += [f"series{n} = simdFma(series{n}, argument * {scalar(1/(n+1.5+term))}, simdBroadcast<Real>(1));"]
            central_cost += 2
        c = (-1)**n * 2**(2*n+1) / gamma(n+1.5)
        body += [f"Real const radial{n} = {scalar(c)} * exponential * series{n};"]
        central_cost += 2
        for m in range(-n, n+1):
            f = factorial(n-abs(m))*factorial(n+abs(m))
            operand = f"radial{n}" if f == 1 else f"{scalar(f)} * radial{n}"
            body += [f"output[{index(n,m)}] = {operand} * harmonics[{index(n,m)}];",
                     f"compensation[{index(n,m)}] = simdBroadcast<Real>(0);"]
            central_cost += 1 + (f != 1)
    body += [f"output[0] += {scalar(pi/4)};", f"greenTrace = {scalar(4*pi/3)};",
             f"for (int ix = -{RADIUS}; ix <= {RADIUS}; ++ix) {{",
             f"for (int iy = -{RADIUS}; iy <= {RADIUS}; ++iy) {{",
             f"for (int iz = -{RADIUS}; iz <= {RADIUS}; ++iz) {{",
             "int const latticeSquared = ix * ix + iy * iy + iz * iz;",
             f"if (latticeSquared == 0 || latticeSquared > {RADIUS*RADIUS} || ix < 0 || (ix == 0 && iy < 0) || (ix == 0 && iy == 0 && iz < 0)) {{ continue; }}",
             "for (int sign = -1; sign <= 1; sign += 2) {",
             "Real const imageX = x + simdBroadcast<Real>(sign * ix);",
             "Real const imageY = y + simdBroadcast<Real>(sign * iy);",
             "Real const imageZ = z + simdBroadcast<Real>(sign * iz);",
             "Real const imageSquared = simdFma(imageX, imageX, simdFma(imageY, imageY, imageZ * imageZ));",
             "Real const imageRadius = simdSqrt(imageSquared);",
             "Real const inverseRadius = simdBroadcast<Real>(1) / imageRadius;",
             "Real const inverseSquared = inverseRadius * inverseRadius;",
             "regular(harmonics, imageX, imageY, imageZ, imageSquared);",
             f"Real const imageArgument = {scalar(4)} * imageSquared;",
             "Real const imageExp = detail::ewaldExp(-imageArgument);",
             f"Real screening = detail::ewaldErfc({scalar(2)} * imageRadius);",
             f"Real gammaTerm = {scalar(4/pi**.5)} * imageRadius * imageExp;",
             "Real inversePower = inverseRadius;"]
    real_cost = 3 + 3 + 7 + 1 + 1 + polynomial_cost + 1 + 1 + 2
    for n in range(degree + 1):
        body += [f"Real const imageRadial{n} = {'-' if n%2==0 else ''}screening * inversePower;"]
        real_cost += 1
        for m in range(-n,n+1):
            slot = index(n,m)
            f = factorial(n-abs(m))*factorial(n+abs(m))
            operand = f"imageRadial{n}" if f == 1 else f"{scalar(f)} * imageRadial{n}"
            body += [f"Real const imageValue{slot} = {operand} * harmonics[{slot}];",
                     f"if (sign < 0) {{ imagePair[{slot}] = imageValue{slot}; }} else {{ imagePair[{slot}] += imageValue{slot}; }}"]
            real_cost += 1 + (f != 1)
        if n != degree:
            body += ["inversePower *= inverseSquared;", "screening += gammaTerm;",
                     f"gammaTerm *= imageArgument * {scalar(1/(n+1.5))};"]
            real_cost += 4
    body += ["}"]
    for slot in range(count):
        body += [f"detail::ewaldAccumulate(output[{slot}], compensation[{slot}], imagePair[{slot}]);"]
    body += ["Real const waveX = simdBroadcast<Real>(ix);", "Real const waveY = simdBroadcast<Real>(iy);", "Real const waveZ = simdBroadcast<Real>(iz);",
             "Real const waveSquared = simdBroadcast<Real>(latticeSquared);",
             "regular(harmonics, waveX, waveY, waveZ, waveSquared);",
             f"Real const phase = {scalar(2*pi)} * simdFma(waveX, x, simdFma(waveY, y, waveZ * z));",
             "Real const cosine = detail::ewaldCos(phase);", "Real const sine = detail::ewaldSin(phase);",
             f"Real const wavePotential = {scalar(-2/pi)} * detail::ewaldExp({scalar(-pi*pi/4)} * waveSquared) / waveSquared;"]
    wave_cost = polynomial_cost + 4 + 3
    df = 1
    for n in range(degree+1):
        if n:
            df *= 2*n-1
        phase = ["cosine", "-sine", "-cosine", "sine"][n%4]
        factor = (2*pi)**n / df
        body += [f"Real const waveRadial{n} = {scalar(factor)} * wavePotential * {phase};"]
        wave_cost += 2
        for m in range(-n,n+1):
            slot = index(n,m)
            f = factorial(n-abs(m))*factorial(n+abs(m))
            operand = f"waveRadial{n}" if f == 1 else f"{scalar(f)} * waveRadial{n}"
            body += [f"detail::ewaldAccumulate(output[{slot}], compensation[{slot}], {operand} * harmonics[{slot}]);"]
            wave_cost += 1 + (f != 1) + 4
    body += ["}", "}", "}", "}", "template <typename Real, typename Output>",
             "CUDA_EXPORT static void evaluate(Output& output, Real& greenTrace, Real x, Real y, Real z) {",
             "using Working = typename detail::EwaldWorkingType<Real>::Type;",
             f"Working temporary[{count}];", "Working trace;",
             "evaluateWorking<Working>(temporary, trace, detail::ewaldConvert<Working>(x), detail::ewaldConvert<Working>(y), detail::ewaldConvert<Working>(z));",
             f"for (int i = 0; i < {count}; ++i) {{ output[i] = detail::ewaldConvert<Real>(temporary[i]); }}",
             "greenTrace = detail::ewaldConvert<Real>(trace);", "}", "};"]
    direct, contraction_cost, contraction_conventional = contraction(degree)
    body += [f"template <> class EwaldM2L<{degree}> {{", "public:", f"static constexpr int kFlops_ = {contraction_cost};",
             f"static constexpr int kConventionalFlops_ = {contraction_conventional};",
             "template <typename Real, typename Local, typename Moments, typename Greens>",
             "CUDA_EXPORT static void add(Local& local, Real& localTrace, Moments const& moments, Real sourceTrace, Greens const& greens, Real greenTrace) {"]
    body += direct + ["}", "};"]
    pairs = sum(x*x+y*y+z*z <= RADIUS*RADIUS and (x>0 or x==0 and (y>0 or y==0 and z>0)) for x in range(-RADIUS,RADIUS+1) for y in range(-RADIUS,RADIUS+1) for z in range(-RADIUS,RADIUS+1))
    central_cost += 1
    # Preserve the historical arithmetic weights for kernel selection/reporting.
    # Runtime counters use FMA=2, sqrt=1 and one per elementary-function result.
    central_conventional = central_cost + polynomial_fmas + 2 + 1 + (degree + 1) * SERIES_TERMS
    real_conventional = real_cost + polynomial_fmas + 2 - 6 + 2
    wave_conventional = wave_cost + polynomial_fmas + 2 + 3
    report = {"local_degree":degree,"source_degree":degree-1,"real_image_count":2*pairs,"reciprocal_pair_count":pairs,"central_weighted_flops":central_cost,
              "real_pair_weighted_flops":2*real_cost+count*5,"reciprocal_pair_weighted_flops":wave_cost,
              "green_weighted_flops":central_cost+pairs*(2*real_cost+count*5+wave_cost),"contraction_weighted_flops":contraction_cost,
              "central_conventional_flops":central_conventional,
              "real_pair_conventional_flops":2*real_conventional+count*5,
              "reciprocal_pair_conventional_flops":wave_conventional,
              "green_conventional_flops":central_conventional+pairs*(2*real_conventional+count*5+wave_conventional),
              "contraction_conventional_flops":contraction_conventional,
              "special_function_calls":{"exp":1+3*pairs,"erfc":2*pairs,"sin":pairs,"cos":pairs},"sqrt_calls":2*pairs}
    body.insert(2, f"static constexpr int kFlops_ = {report['green_weighted_flops']};")
    body.insert(3, f"static constexpr int kConventionalFlops_ = {report['green_conventional_flops']};")
    return body,report


def generate(minLocalDegree=2,maxLocalDegree=8):
    if not 2 <= minLocalDegree <= maxLocalDegree <= 8:
        raise ValueError("Spherical Ewald is validated for local degrees 2 through 8")
    lines = [NOTICE,"#pragma once", "#include <cmath>", "#include <type_traits>", "", "#include <cosmictiger/cuda/cuda_export.hpp>",
             "#include <cosmictiger/gravity/fmm_kernels/spherical_kernels.hpp>", "#include <cosmictiger/math/simd_math.hpp>",
             "", "namespace cosmictiger::gravity::spherical {", "namespace detail {"]
    lines += ["template <typename Real> class EwaldWorkingType { public: using Type = Real; };",
              "template <> class EwaldWorkingType<float> { public: using Type = double; };",
              "#ifndef __CUDACC__",
              "template <int lanes> class EwaldWorkingType<SimdVector<float, lanes>> { public: using Type = SimdVector<double, lanes>; };",
              "#endif", "template <typename Destination, typename Source>",
              "CUDA_EXPORT inline Destination ewaldConvert(Source const& value) {",
              "if constexpr (std::is_same_v<Destination, Source>) { return value; }",
              "else if constexpr (std::is_arithmetic_v<Destination>) { return static_cast<Destination>(value); }",
              "else { return Destination([&](auto lane) { return static_cast<typename Destination::value_type>(value[lane]); }); }", "}"]
    for name in ["Exp","Erfc","Cos","Sin"]:
        function = name.lower()
        lines += ["template <typename Real>", f"CUDA_EXPORT inline Real ewald{name}(Real const& value) {{",
                  f"if constexpr (std::is_arithmetic_v<Real>) {{ return std::{function}(value); }}",
                  f"else {{ return Real([&](auto lane) {{ return std::{function}(value[lane]); }}); }}", "}"]
    lines += ["template <typename Real>", "CUDA_EXPORT inline void ewaldAccumulate(Real& sum, Real& correction, Real value) {",
              "Real const adjusted = value - correction;", "Real const next = sum + adjusted;", "correction = (next - sum) - adjusted;", "sum = next;", "}", "}",
              "template <int degree> class EwaldGreen;", "template <int degree> class EwaldM2L;"]
    counts = {}
    for degree in range(minLocalDegree,maxLocalDegree+1):
        code,report = emitDegree(degree)
        lines.append(f'#if (!defined(FMM_MIN_ORDER) || FMM_MIN_ORDER <= {degree + 1}) && '
                     f'(!defined(FMM_MAX_ORDER) || FMM_MAX_ORDER >= {degree + 1})')
        lines += code
        lines.append('#endif')
        counts[str(degree+1)] = report
    lines += ["template <int degree> class EwaldL2L {", "public:", "static constexpr int kFlops_ = detail::L2L<degree>::kFlops_ + 9;",
              "static constexpr int kConventionalFlops_ = detail::L2L<degree>::kConventionalFlops_ + 15;",
              "template <typename Real, typename Local, typename Source>",
              "CUDA_EXPORT static void add(Local& output, Real& outputTrace, Source const& source, Real sourceTrace, Real x, Real y, Real z) {",
              "detail::L2L<degree>::template add<Real>(output, source, x, y, z);",
              "Real const radiusSquared = simdFma(x, x, simdFma(y, y, z * z));",
              "output[0] = simdFma(-simdBroadcast<Real>(0.5) * sourceTrace, radiusSquared, output[0]);",
              "output[3] = simdFma(-sourceTrace, x, output[3]);",
              "output[1] = simdFma(-sourceTrace, y, output[1]);",
              "output[2] = simdFma(-sourceTrace, z, output[2]);",
              "outputTrace += sourceTrace;", "}", "};",
              "template <int degree> class EwaldL2P {", "public:", "static constexpr int kFlops_ = detail::L2P<degree>::kFlops_ + 8;",
              "static constexpr int kConventionalFlops_ = detail::L2P<degree>::kConventionalFlops_ + 14;",
              "template <typename Real, typename Field, typename Source>",
              "CUDA_EXPORT static void add(Field& field, Source const& source, Real sourceTrace, Real x, Real y, Real z) {",
              "detail::L2P<degree>::template add<Real>(field, source, x, y, z);",
              "Real const radiusSquared = simdFma(x, x, simdFma(y, y, z * z));",
              "field[0] = simdFma(-simdBroadcast<Real>(0.5) * sourceTrace, radiusSquared, field[0]);",
              "field[1] = simdFma(sourceTrace, x, field[1]);",
              "field[2] = simdFma(sourceTrace, y, field[2]);",
              "field[3] = simdFma(sourceTrace, z, field[3]);", "}", "};", "}", ""]
    return {"spherical_ewald_kernels.hpp":"\n".join(lines)}, {"cost_model":{"fma":1,"sqrt":7,"add":1,"multiply":1,"division":1,"special_functions":"reported separately"},"conventional_cost_model":{"fma":2,"sqrt":1,"add":1,"multiply":1,"division":1,"special_functions":1},"lattice_radius":RADIUS,"central_series_terms":SERIES_TERMS,"orders":counts}
