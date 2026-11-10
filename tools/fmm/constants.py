# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# Distributed under the Boost Software License, Version 1.0.
"""Give every generated gravity function a typed, sign-shared constant table."""
from fractions import Fraction
import re


NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
COEFFICIENT = rf'{NUMBER}(?:\s*/\s*{NUMBER})?'
FUNCTION = re.compile(r'(?:CUDA_EXPORT|__device__)[^;{}]*\([^;{}]*\)[^;{}]*\{')


def coefficient(expression):
    numerator, *denominator = expression.split('/')
    value = Fraction(numerator.strip())
    return value / Fraction(denominator[0].strip()) if denominator else value


def scientific(value):
    """Round an exact rational to 18 fractional digits, nearest with ties to even.

    Integer quotient/remainder arithmetic avoids intermediate decimal rounding,
    including values arbitrarily close to a rounding midpoint.
    """
    if not value:
        return '0.000000000000000000E+00'
    numerator, denominator = abs(value.numerator), value.denominator
    exponent = len(str(numerator)) - len(str(denominator))
    if (numerator < denominator * 10**exponent if exponent >= 0 else
            numerator * 10**(-exponent) < denominator):
        exponent -= 1
    shift = 18 - exponent
    if shift >= 0:
        numerator *= 10**shift
    else:
        denominator *= 10**(-shift)
    digits, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and digits % 2):
        digits += 1
    if digits == 10**19:
        digits //= 10
        exponent += 1
    mantissa = str(digits)
    sign = '-' if value < 0 else ''
    return f'{sign}{mantissa[0]}.{mantissa[1:]}E{exponent:+03d}'


def hoistConstants(cpp):
    """Hoist numerical broadcast arguments, leaving runtime indices untouched.

    SimdScalar<Real> is Real for scalar/CUDA instantiations and the lane type
    for SIMD. Constants therefore remain constexpr for both backends. Only
    literal coefficients (or literal rational divisions) are accepted here;
    broadcasts of runtime lattice coordinates and CUDA lane masks stay runtime.
    """
    position = 0
    output = []
    while match := FUNCTION.search(cpp, position):
        opening = match.end() - 1
        closing = opening + 1
        depth = 1
        while depth:
            depth += (cpp[closing] == '{') - (cpp[closing] == '}')
            closing += 1
        body = cpp[opening + 1:closing - 1]
        # Expand the older Cartesian DAG's names before assigning one table to
        # the complete function, including its geometry and output accumulations.
        previous = {}
        def removeDeclaration(match):
            previous[match[1]] = match[2]
            return ''
        body = re.sub(rf'^[ \t]*constexpr auto (k\d+) = ({COEFFICIENT});\n',
                      removeDeclaration, body, flags=re.MULTILINE)
        broadcast = re.compile(r'(?:math::)?broadcast<Real>\(([^()]*)\)')
        magnitudes = set()
        for argument in broadcast.finditer(body):
            expression = previous.get(argument[1], argument[1]).strip()
            if re.fullmatch(COEFFICIENT, expression):
                magnitudes.add(abs(coefficient(expression)))
        width = len(str(max(0, len(magnitudes) - 1)))
        names = {value: f'k{index:0{width}d}'
                 for index, value in enumerate(sorted(magnitudes))}
        def replace(match):
            expression = previous.get(match[1], match[1]).strip()
            if not re.fullmatch(COEFFICIENT, expression):
                return match[0]
            value = coefficient(expression)
            sign = '-' if value < 0 else ''
            return f'{sign}math::broadcast<Real>({names[abs(value)]})'
        body = broadcast.sub(replace, body)
        declarations = ''.join(
            f'\n    constexpr auto {name} = static_cast<Scalar>({scientific(value)});'
            for value, name in names.items())
        alias = '\n    using Scalar = SimdScalar<Real>;' if names else ''
        output.append(cpp[position:opening + 1] + alias + declarations + body + '}')
        position = closing
    output.append(cpp[position:])
    return ''.join(output)
