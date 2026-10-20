#!/usr/bin/env python3
"""Reproduce the single-precision erfcx polynomial (requires mpmath).

The transformation t=1/(1+x), z=2*t-1 maps x in [0,infinity) onto
z in (-1,1]. Interpolate erfcx(x)/t at 16 Chebyshev nodes. The production
kernel evaluates these coefficients in float with Estrin's scheme.
"""

import mpmath as mp

mp.mp.dps = 70


def transformed_erfc(z):
    t = (z + 1) / 2
    x = 1 / t - 1
    return mp.exp(x * x) * mp.erfc(x) / t


coefficients, error = mp.chebyfit(transformed_erfc, [-1, 1], 16, error=True)
for degree, coefficient in enumerate(reversed(coefficients)):
    print(f"c{degree} = {float(coefficient):.17g}")
print(f"Estimated interpolation error: {mp.nstr(error, 12)}")
