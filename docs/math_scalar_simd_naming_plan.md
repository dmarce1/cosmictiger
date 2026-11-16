# Scalar CUDA and CPU SIMD math helper naming

Recorded November 9, 2026. This is a naming note, not an implementation.

Proposed API: move the helpers shared by scalar CUDA kernels and CPU SIMD code
into the existing `math` namespace and remove the `simd` prefix. In particular,
rename `simdFma(a, b, c)` to `math::fma(a, b, c)`. The operation must retain
fused multiply-add rounding for scalar and packed values.

The same rule applies to the other shared helpers in
`cosmictiger/math/simd_math.hpp`:

| Current helper | Proposed spelling |
| --- | --- |
| `simdBroadcast<Real>(value)` | `math::broadcast<Real>(value)` |
| `simdSelect(mask, yes, no)` | `math::select(mask, yes, no)` |
| `simdMin(left, right)` | `math::min(left, right)` |
| `simdMax(left, right)` | `math::max(left, right)` |
| `simdSqrt(value)` | `math::sqrt(value)` |
| `simdFma(a, b, c)` | `math::fma(a, b, c)` |
| `simdAnyOf(mask)` | `math::anyOf(mask)` |
| `simdAllOf(mask)` | `math::allOf(mask)` |

This list covers the helpers with scalar CUDA behavior and CPU SIMD behavior.
The remaining `simd*` helpers in that header are CPU pack utilities and need
a separate naming decision. When implementing this, check overloads already in
`math` (notably `math::max(double, double)`) and the distinct
`special_functions::fma`/`sqrt` implementations before moving declarations.
Update generated FMM code through its generators, along with direct callers.
