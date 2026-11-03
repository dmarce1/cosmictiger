# Full symmetric Cartesian operators

The full symmetric family lives in `cosmictiger::gravity::cartesian`. Include
`cosmictiger/gravity/kernels/cartesian_operators.hpp` for the public API.
The existing STF family remains available in `cartesian::tracefree`, through
`cartesian_tracefree_operators.hpp`; production FMM continues using that family.
The two sets have distinct types and no implicit representation conversions.

## Types and normalization

`Multipole<Real, order>` stores every symmetric Cartesian moment through degree
`order - 2`, while `Expansion<Real, order>` and `Green<Real, order>` store every
component through degree `order - 1`. Orders 3 through 9 are supported. Through
degree `d`, the packed size is `(d + 1)(d + 2)(d + 3)/6`. Ranks are ordered by
total degree, then increasing x exponent, then increasing y exponent. The
`index(nx, ny, nz)` accessor and checked three-index access share this ordering.

Moments are raw moments `M[a] = sum_j mass[j] * offset[j]^a`, without factorial
normalization. Local and Green tensors contain ordinary derivatives of the
negative potential; the Newtonian unit-source potential is `-1/r`. Coordinates,
moments, and derivatives must use consistent units. Public operators do not
apply the production walk's normalized-coordinate scaling.

A Green tensor is distinct from an accumulated local expansion even though
both have the same size. No harmonic identity reconstructs omitted entries:
all traces, including repeated traces at higher rank, remain available.

## Operators

All seven public operators add into an initialized destination and return their
conventional arithmetic count. Template arguments for the scalar type and FMM
order are deduced from the typed arguments. ADL supports the same unqualified
operator spelling for full Cartesian, tracefree Cartesian, and spherical types.

| Operation | Arguments | Displacement |
| --- | --- | --- |
| `P2M` | `(target, mass, offset)` | Source particle minus moment center |
| `M2M` | `(target, source, displacement)` | Old moment center minus new center |
| `P2L` | `(target, mass, green, doPotential=true)` | Encoded in the supplied Green tensor |
| `M2L<centered=false>` | `(target, source, green, doPotential=true)` | Encoded in the supplied Green tensor |
| `M2P<centered=false>` | `(field, source, green, doPotential=true)` | Encoded in the supplied Green tensor |
| `L2L` | `(target, source, displacement, doPotential=true)` | New local center minus old center |
| `L2P` | `(field, source, displacement, doPotential=true)` | Evaluation point minus local center |

M2M and L2L require distinct source and destination objects. `doPotential=false`
preserves the destination's potential entry while still accumulating all other
entries. `gravity::Field<Real>` contains potential and three acceleration
components, with acceleration equal to minus the potential gradient.

For multi-indices `a` and `b`, M2L evaluates

```
L[a] += sum_b (-1)^|b| * M[b] * D[a+b] / b!
```

subject to `|b| <= order - 2` and `|a| + |b| <= order - 1`. M2P retains every
source rank through `order - 2` in both potential and acceleration. The explicit
`centered=true` specialization promises zero source dipoles; it omits rank one
only, retaining odd moments of rank three and above.

## Green helpers

`evaluateGreen(green, displacement)` overwrites the tensor with derivatives of
`-1/r`. The overload with a final `Real epsilonSquared` evaluates derivatives of
`-1/sqrt(r*r + epsilonSquared)`, a Plummer potential. Epsilon squared is constant
under differentiation. It must be nonnegative, and `r*r + epsilonSquared` must
be positive; positive softening permits coincident centers. These are mathematical
helpers and do not change the production softening choice.

P2L, M2L, and M2P also accept a displacement instead of a supplied Green tensor.
Their Newtonian overload ends with `bool doPotential=true`; the Plummer overload
accepts `Real epsilonSquared` before that optional bool. These coordinate
wrappers include derivative evaluation in their returned operation count. The
softening argument must match `Real`; the Newtonian potential flag accepts only
`bool`, preventing a mismatched softening scalar from silently becoming a flag.

A caller may fill Green with derivatives of any other sufficiently smooth
potential. The [Gaussian-screened radial kernel](gaussian_screening.md) provides
an `evaluateGreen` overload with an explicit screening parameter. The complete
FMM-PM runtime and long/short integrator remain subsequent work.

## Generation and validation

`tools/fmm/cartesian_symmetric.py` generates `cartesian_kernels.hpp` through the
shared `tools/generate_fmm.py` entry point. Exact rational algebra, common
subexpressions, shared powers, and explicit FMAs produce straight-line kernels.
Both Cartesian families honor the same `FMM_MIN_ORDER` and `FMM_MAX_ORDER` guards.

`kFlops_` counts FMA as two and square root as one conventional operation.
`kWeightedFlops_` counts FMA as one and square root as seven; other arithmetic
costs one. Reports include destination accumulation and state each convention.
Unary signs, copies, and constant evaluation carry no arithmetic cost. These
are source-operation counts, not measured timings. The report's `cartesian`
entry describes the full family and `cartesianTracefree` describes STF. Legacy
top-level STF count entries remain available.

`cartesian_symmetric` tests exercise independent raw-moment/binomial/Taylor
references, arbitrary nonharmonic locals, an exactly terminating quadratic
kernel, independently expanded Plummer derivatives, centered/generic source
moments, additive destinations, potential suppression, and float/double/SIMD.
CUDA tests exercise the same typed APIs in bounded device kernels. The independent
counting-scalar test executes generated arithmetic and checks both cost models;
`generated_fmm_current` checks reproducibility. Test availability is not evidence
that a particular CPU, CUDA, or distributed configuration has been executed.
