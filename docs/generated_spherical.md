# Generated spherical kernels

`tools/generate_fmm.py` generates `cosmictiger/gravity/kernels/spherical_kernels.hpp` for the same
inclusive local-degree range as the Cartesian representations. FMM order `q`
means source degree `q-2` and local degree `q-1`; the source headers contain
orders 3–9, with source degrees 1–7 and local degrees 2–8. CMake's `COSMICTIGER_EXPANSION_DEGREE` selects potential degrees, defaulting
to 4–6 (orders 5–7); use `2-8` for all generated degrees. `FMM_MIN_ORDER`/`FMM_MAX_ORDER` preprocessor guards
exclude other orders from compilation without removing them from the source.
This adds a generated API, not a runtime gravity-method selection.

## Normalization and storage

`cosmictiger::gravity::spherical::Multipole<Real, order>` and
`cosmictiger::gravity::spherical::Expansion<Real, order>` are distinct types.
Their template argument is the FMM order: multipoles store `(order-1)^2`
coefficients through degree `order-2`, while expansions store `order^2`
coefficients through degree `order-1`.
Slot `n(n+1)+m` stores the real part for `m>=0` and the imaginary
part of the positive-order coefficient for `m<0`. Complex negative orders obey
`C(n,-m)=(-1)^m conjugate(C(n,m))`. Do not confuse a negative storage index order
with the real or imaginary part of the corresponding negative complex order.

The regular solid harmonics are Dehnen's factorial-normalized `U_n^m`:

- `U_0^0=1`;
- `U_m^m=(x+i y) U_(m-1)^(m-1)/(2m)`;
- `U_(m+1)^m=z U_m^m`;
- `U_n^m=((2n-1)z U_(n-1)^m-r² U_(n-2)^m)/(n²-m²)`.

Source moments are `M_n^m=sum_a mass_a U_n^m(-a)`, where `a` is particle
position minus source center. The signed Green coefficients are
`G_n^m=(-1)^(n+1) Theta_n^m`, so `G_0^0=-1/r`, `G_1^0=z/r³`, and
`G_1^1=(x+i y)/r³`. The local contraction is
`L_n^m=sum_(k,l) conjugate(M_k^l) G_(n+k)^(m+l)` with
`k<=sourceDegree` and `n+k<=localDegree`. The potential is negative for a
positive Newtonian point mass.

The first local degree is the Cartesian gradient: `(L_1^1.real,
L_1^1.imag,L_1^0.real)`. Field APIs return `[potential,accelerationX,accelerationY,
accelerationZ]`, with acceleration equal to the negative gradient.

## API

The generated classes retained for PMMM are `P2M`, `Green`, and `L2P` in
`spherical::detail`. The `add` methods accumulate into initialized output;
`Green::evaluate` overwrites its output. They accept coefficient containers
supporting `operator[]`, including `Multipole`, `Expansion`, fixed C arrays,
and `std::array`. `Real` is explicit because the containers can differ. CUDA
scalar and CPU scalar/SIMD math share the emitter.

| Class | Method arguments after explicit `<Real>` |
| --- | --- |
| `P2M<S>` | `add(output,mass,x,y,z)` |
| `L2P<L>` | `add(field,source,x,y,z)` |
| `Green<L>` | `evaluate(output,x,y,z)` (overwrites) |

Here `S=L-1`. L2P uses evaluation point minus local center. P2M uses particle
position minus source center. Green uses target center minus source center and
requires nonzero separation. P2M and L2P accept zero displacement, either axis
sign, and heterogeneous SIMD lanes.

The PMMM far field uses the generated `Green` evaluator to add the primary
free-space interaction outside its near-field mask. It builds and caches the
periodic Ewald Green filter separately, then contracts source moments and that
filter in Fourier space. The periodic uniform-background quadratic is handled
outside the harmonic coefficient arrays, as described in
`generated_spherical_ewald.md`.

## Rotation choices and costs

The generator compares three rotation candidates for L2P:

- `v=0`: direct local evaluation, emitted as real arithmetic;
- `v=1`: azimuthal rotation places the displacement in the x–z plane, where
  harmonic imaginary components vanish;
- `v=2`: compose commuting axial and planar local evaluations, then align the
  planar part using one z phase pair and two x/z swaps.

Constant swap coefficients use the exact rational recurrence in Dehnen
Eq. (63), not floating-point matrix construction. Real conjugate symmetry creates
sparse real/imaginary blocks. Phase recurrences share powers across ranks.
Exact zeros and signs are folded, duplicate products are combined, common
expressions are interned, and a backwards traversal from the required outputs
omits unneeded rotation components and harmonics. The resulting arithmetic
uses explicit `simdFma` calls.

Only the L2P candidate with the smallest emitted weighted count is emitted.
Its preceding comment identifies
the chosen variant; `kFlops_` exposes its count. There are no public variant
classes or variant template arguments. Every candidate's count and the winning
variant appear in the operation report. Ties choose the lower variant number.
Counts include geometry setup, rotation, contraction and output accumulation.
Add/subtract/multiply/divide/FMA
cost 1; sqrt costs 7. Inverse square root is explicitly division by sqrt and
costs 8. Unary signs, constants, copies, comparisons and lane selection cost
zero. These are the user's weighted arithmetic costs, not hardware timings.
The operation report retains the candidates' costs for selection checks.

## Mathematical provenance

The implementation is an independent Python expression generator based on
Walter Dehnen, *A fast multipole method for stellar dynamics*,
[arXiv:1405.2255](https://arxiv.org/abs/1405.2255), Eqs. (3), (49)–(54) and
Appendix A.6, Eqs. (59)–(66). The manuscript is a 2014 publication whose
arXiv PDF title page carries the date March 26, 2018. The user's
[spherical_fmm](https://github.com/dmarce1/spherical_fmm) implementation was
consulted for normalization, the axial/planar composition optimization, and
previous finite-order rotation pitfalls. Its C++ generator is not copied.

## Validation

`tests/gravity/generated_spherical.cpp` uses independently written loop-based
complex harmonic recurrences. It checks P2M, Green, and L2P against those
recurrences, including additive output for the active evaluators. Local inputs
include arbitrary real and imaginary coefficients in every rank. Fixtures
include both axis signs, zero offsets, near-axis directions, and general
displacements. P2M errors are normalized by the norm of each rank so small
high-order moments remain meaningfully tested.

CMake splits this source with `GENERATED_SPHERICAL_TEST_DEGREE` to keep each
compilation bounded. It creates `generated_spherical_degree_<degree>` CTest tests
for each selected degree when PMMM support is enabled. Defining
`GENERATED_SPHERICAL_SKIP_SIMD` is available for a scalar-only diagnostic build.
