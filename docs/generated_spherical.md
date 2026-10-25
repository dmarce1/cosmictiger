# Generated spherical kernels

`tools/generate_fmm.py` generates `generated_spherical.hpp` for the same
inclusive local-degree range as the Cartesian representations. FMM order `q`
means source degree `q-2` and local degree `q-1`; the default orders are 3–9.
This adds a generated API, not a runtime gravity-method selection.

## Normalization and storage

`cosmictiger::gravity::spherical::Spherical<Real, degree>` stores `(degree+1)^2`
real numbers. Slot `n(n+1)+m` stores the real part for `m>=0` and the imaginary
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

The following generated classes have templated static methods. All `add`
methods accumulate into initialized output; input and output must not alias.
They accept any coefficient containers supporting `operator[]`, including
`Spherical`, fixed C arrays and `std::array`. `Real` is explicit because the
containers can differ. CUDA scalar and CPU scalar/SIMD math share the emitter.

| Class | Method arguments after explicit `<Real>` |
| --- | --- |
| `P2mKernel<S>` | `add(output,mass,x,y,z)` |
| `M2mKernel<S>` | `add(output,source,x,y,z)` |
| `M2lKernel<L>` | `add(output,source,x,y,z)` |
| `M2pKernel<L>` | `add(field,source,x,y,z)` |
| `L2lKernel<L>` | `add(output,source,x,y,z)` |
| `L2pKernel<L>` | `add(field,source,x,y,z)` |
| `P2lKernel<L>` | `add(output,mass,x,y,z)` |
| `GreensKernel<L>` | `evaluate(output,x,y,z)` (overwrites) |
| `M2lDerivativesKernel<L>` | `add(output,source,derivatives)` |

Here `S=L-1`. M2M displacement is the old source center minus the new source
center. L2L displacement is the new local center minus the old local center;
L2P uses evaluation point minus local center. P2M uses particle position minus
source center. M2L and Green functions use target center/point minus source
center. Free-space singular kernels require nonzero separation. Translations
accept zero displacement, either axis sign and heterogeneous SIMD lanes.

`M2lDerivativesKernel` contracts a general spherical Green array. A periodic
uniform-background quadratic needs an additional scalar trace; use the Ewald
API described in `generated_spherical_ewald.md` rather than dropping it.

## Rotation choices and costs

Each translation or evaluation operator is emitted as `XxxVariant<degree,v>`:

- `v=0`: direct complex-harmonic contraction, emitted as real arithmetic;
- `v=1`: azimuthal rotation places the displacement in the x–z plane, where
  harmonic imaginary components vanish;
- `v=2`: axial contraction. M2L/M2P use two z rotations and constant x/z swaps
  to align a general displacement with z. M2M/L2L/L2P compose commuting axial
  and planar translations, then align only the planar part using one z phase
  pair and two x/z swaps. This avoids two otherwise necessary swaps.

Constant swap coefficients use the exact rational recurrence in Dehnen
Eq. (63), not floating-point matrix construction. Regular/source transforms
use the transpose of the Green/local matrix. Real conjugate symmetry creates
sparse real/imaginary blocks. Phase recurrences share powers across ranks.
Exact zeros and signs are folded, duplicate products are combined, common
expressions are interned, and a backwards traversal from the required outputs
omits unneeded rotation components and harmonics. The resulting arithmetic
uses explicit `simdFma` calls.

The selected `XxxKernel<degree>` inherits the variant with the smallest emitted
weighted count. `kVariant_` exposes the choice; `kFlops_` exposes its count.
Every candidate's count and the winning variant appear in the operation
report. Ties choose the lower variant number. Counts include geometry setup,
rotation, contraction and output accumulation. Add/subtract/multiply/divide/FMA
cost 1; sqrt costs 7. Inverse square root is explicitly division by sqrt and
costs 8. Unary signs, constants, copies, comparisons and lane selection cost
zero. These are the user's weighted arithmetic costs, not hardware timings.
All three variants remain available for correctness comparison.

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

`tests/gravity/nitadori/generated_spherical.cpp` uses independently written loop-based complex
harmonic recurrences and translation/contraction sums, rather than the Python
DAG. It checks all three candidates, selected M2L dispatch, arbitrary Green
contraction, additive outputs, P2M/P2L normalization, and a three-particle direct
force comparison. Multipoles and locals include arbitrary real and imaginary
coefficients in every rank. Fixtures include both axis signs, zero translations,
near-axis directions and general displacements. P2M errors are normalized by
the norm of each rank so small high-order moments remain meaningfully tested.

The split order-3 through order-9 executables passed 181,468 comparisons with
GCC, strict arithmetic (`-fno-fast-math -ffp-contract=off`), scalar float/double
and heterogeneous four-lane double SIMD. Worst scaled discrepancy was
`3.876e-5` in the intentionally ill-conditioned float arbitrary-moment stress
cases; the float test budget is `4e-5`, and the double/SIMD budget is `2e-10`.
These are numerical and emitted-arithmetic checks, not runtime speed benchmarks.

CMake splits this source with `GENERATED_SPHERICAL_TEST_DEGREE` to keep each
compilation bounded. Run the `generated_spherical_order_3` through
`generated_spherical_order_9` CTest tests. Defining
`GENERATED_SPHERICAL_SKIP_SIMD` is available for a scalar-only diagnostic build.
