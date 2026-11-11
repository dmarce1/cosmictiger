# Cartesian/spherical basis conversions

The two maps needed at the tree/FFT boundaries are explicit conversion
operators on Cartesian multipoles and spherical expansions. Their generated
arithmetic lives directly in the cast definitions in
`cosmictiger/gravity/kernels/basis_conversion.hpp`. Including either coefficient
type header also makes its cast available:

```cpp
auto sphericalSource = static_cast<spherical::Multipole<Real>>(cartesianSource);
auto cartesianLocal = static_cast<cartesian::Expansion<Real>>(sphericalLocal);
```

Both return new coefficient objects, preserve every dipole component, and
require the same scalar type, FMM order, center and coordinate units on both
sides. The source degree is q-2; the local degree is q-1, for q=3..9. There is
no runtime allocation, factorial evaluation or dense matrix multiplication.
Float, double and CPU SIMD are supported; the casts are host/device. They do
not move centers, rescale coordinates or add background terms. Those operations
remain separately testable at the caller boundary. The reverse conversions
and implicit cross-basis conversions are not provided.

## Conventions

Let α=(a,b,c), |α|=a+b+c, α!=a!b!c!, and r^α=x^a y^b z^c.
The Cartesian source coefficients are raw symmetric trace-free moments:

\[
C_\alpha=\sum_j m_j\operatorname{STF}(\mathbf r_j^{\otimes n})_\alpha,
\qquad |\alpha|=n.
\]

The stored components have c=0 or c=1. Remaining components follow from
\(C_{a,b,c+2}=-C_{a+2,b,c}-C_{a,b+2,c}\).
Their slot is n²+a for c=0, or n²+n+1+a for c=1.

Regular solid harmonics follow the normalization of
[Dehnen (2014), Eq. 2b](https://link.springer.com/article/10.1186/s40668-014-0001-7):

\[
U_{nm}=\frac{(-1)^m r^n}{(n+m)!}P_n^m(z/r)e^{im\phi},
\quad U_{00}=1,\quad U_{10}=z,\quad U_{11}=\frac{x+iy}{2}.
\]

Here P includes the Condon–Shortley phase. The repository uses
\(S_{nm}=\sum_jm_j U_{nm}(-\mathbf r_j)\), hence an explicit (-1)^n
relative to moments formed at +r. Do not copy a source translation sign from
the paper without adapting this convention.
Slot n(n+1)+m contains the real part for m≥0; n(n+1)-m contains the imaginary
part of the positive-m coefficient for m>0. Negative slots are **not** the
negative-m complex harmonics.

## Derived source map

Expand \(U_{nm}=\sum_{|\alpha|=n}u_{nm\alpha}\mathbf r^\alpha\).
Harmonicity means its tensor is trace-free, so contraction with any trace
removed from a raw source moment vanishes. Therefore

\[
\boxed{S_{nm}=(-1)^n\sum_{|\alpha|=n}u_{nm\alpha}C_\alpha.}
\]

Reduce every z-pair on the right to the stored Cartesian components. Through
quadrupole order the result is

\[
\begin{aligned}
S_{00}&=C_0,& S_{10}&=-C_z,& S_{11}&=-\tfrac12(C_x+iC_y),\\
S_{20}&=-\tfrac34(C_{xx}+C_{yy}),&
S_{21}&=\tfrac12(C_{xz}+iC_{yz}),&
S_{22}&=\tfrac18(C_{xx}-C_{yy}+2iC_{xy}).
\end{aligned}
\]

For example, \(U_{20}=(2z^2-x^2-y^2)/4\) and
\(C_{zz}=-C_{xx}-C_{yy}\), giving the -3/4 above. The generator substitutes
the actual Cartesian STF point polynomials into every row and asserts exact
equality with \((-1)^n U_{nm}\), using rational arithmetic.

## Derived local map

The local coefficients instead represent the harmonic potential

\[
H(\mathbf r)=\sum_n\left[L_{n0}U_{n0}(\mathbf r)
+2\Re\sum_{m=1}^n L_{nm}\overline{U_{nm}(\mathbf r)}\right].
\]

Cartesian locals store derivatives, not raw polynomial coefficients:

\[
\boxed{D_\alpha=\partial^\alpha H(0)=\alpha!\,[\mathbf r^\alpha]H.}
\]

This is the dual evaluation map, not the inverse of the source map. In
particular, source signs and local factorials are different:

\[
\begin{aligned}
D_0&=L_{00},&D_x&=\Re L_{11},&D_y&=\Im L_{11},&D_z&=L_{10},\\
D_{xx}&=(-L_{20}+\Re L_{22})/2,&
D_{yy}&=(-L_{20}-\Re L_{22})/2,&D_{xy}&=\Im L_{22}/2,\\
D_{xz}&=\Re L_{21},&D_{yz}&=\Im L_{21}.
\end{aligned}
\]

The generator reconstructs the full Taylor polynomial from the compact
derivatives, including dependent z-pair components, and asserts its exact
equality to each spherical basis polynomial. This checks the maps separately;
an inverse/round-trip check alone would allow matching mistakes to cancel.

## Implicit periodic background

For the convention \(\Phi=H-\beta r^2/2\), define K=-3β and
\(H'=H+2\beta U_{20}\). Then

\[
\Phi(\mathbf r)=H'(\mathbf r)+Kz^2/2.
\]

`preparePeriodicLocal` adds 2β to L20 once. The local map then lowers Cartesian
Dxx and Dyy by β; the reconstructed harmonic Dzz rises by 2β. The implicit
Kz²/2 supplies the remaining z curvature, giving the required -β on all axes.
The known background is a context, not a per-local trace slot.

With the new local center displaced by d, the harmonic proxy transforms as

\[
H'_{new}(\mathbf x)=H'_{old}(\mathbf x+\mathbf d)+K d_zx_z+K d_z^2/2.
\]

Thus harmonic L2L is followed by K dz in Dz and K dz²/2 in D0. Evaluation
adds Kz²/2 to potential and -Kz to acceleration-z. These are the existing
spherical implicit-context formulas; the conversion tests also exercise their
Cartesian equivalents. Production Cartesian downward integration is Part 4.
Adding or subtracting a harmonic near local never increments K. The source
radial second moment/constant potential correction is separate; these maps
neither reconstruct it from STF moments nor replace its existing summary.

## Mesh/tree scaling

Let physical offset be r, tree offset y=κr (κ=`kScaleFactor`), and mesh offset
u=r/h. Write A and B for the two pure rank-n basis maps above. At the same center,

\[
S_n=(\kappa h)^{-n}A_n C_n^{tree},\qquad
D_n^{tree}=\kappa^{-1}(\kappa h)^{-n}B_n L_n^{mesh}.
\]

The extra κ⁻¹ makes tree evaluation give physical potential divided by κ;
physical acceleration is κ² times tree acceleration. Mesh-coordinate
acceleration is divided by h to become physical acceleration. With
\(\beta=4\pi M/(3V)\),

\[
\beta_{mesh}=\beta h^2,\qquad
\beta_{tree}=\frac{\beta_{mesh}}{\kappa^3h^2}=\frac{\beta}{\kappa^3}.
\]

The tree background coefficient is independent of mesh width. Centers must
be subtracted in native fixed-point coordinates before conversion to floating
point, especially across the periodic seam. For a mesh center that is not
exactly representable in fixed point, add the fixed-center minus geometric-center
residual after native subtraction. Source translation, radial summaries and
mesh-to-root local translation retain this residual, so arbitrary mesh sides
use the same geometric origin as the FFT and direct-pair paths.

## Cost and verification

The emitted arithmetic counts below charge FMA as one operation; conventional
two-FLOP FMA counts are separately recorded in `operation_counts.json` and
the header constants. Copies and sign flips cost zero under the existing
generator convention. These are not measured timings.

| FMM order q | Source degree | Local degree | Source map | Local map |
|---:|---:|---:|---:|---:|
| 3 | 1 | 2 | 2 | 5 |
| 4 | 2 | 3 | 9 | 19 |
| 5 | 3 | 4 | 23 | 41 |
| 6 | 4 | 5 | 44 | 74 |
| 7 | 5 | 6 | 76 | 121 |
| 8 | 6 | 7 | 121 | 184 |
| 9 | 7 | 8 | 181 | 262 |

`tests/gravity/basis_conversion.cpp` checks every stored basis component against
fields and translations; local potentials and analytic gradients use an
independent finite Legendre polynomial oracle. Asymmetric unequal-mass clouds
also compare converted STF P2M to the independent spherical polynomial.
Scalar float/double and SIMD float/double, fixed32/fixed64 periodic seams,
three mesh widths, nested background translations and harmonic near subtraction
are covered. Each configured order receives a CTest target. The counting-scalar
test executes both generated maps and verifies weighted/conventional counts.
`tests/gravity/basis_conversion_cuda.cu` provides compile-only device coverage.

Validation on November 9, 2026, in `/home/dmarce1/workspace/cosmictiger`:

- Exact rational identities and reproducible generation pass for q=3..9.
- Separate scalar/SIMD executables for all seven orders pass 267,610 checks;
  the largest scaled discrepancy across them is 5.37e-7 (float included).
- The counting-scalar audit passes 161 operator checks, including both maps
  at every order, plus the existing 35 spherical variant selections. The
  order-7-only audit also verifies excluded-order API availability.
- The Release `build/pmmm_only_cpu` targets `basis_conversion_order_7_test`
  and `fmm_api_test` build; all five focused CTests pass.
- NVCC `sm_86` compiles both maps in float/double through all seven orders.
  This is device compilation only; no GPU kernels were executed.
- No full application, MPI, sanitizer or whole-solve performance validation
  is claimed. Builds used at most two CPU cores; no GPU compute was used.

The new header also exposed a pre-existing include-order dependency: the
generator injected `using namespace math` into the lightweight Cartesian
coefficient storage, which has no math calls. Generation now omits that
injection for the storage header; coefficient layout and arithmetic are unchanged.

The maps are ready for the next steps: reuse completed double tree roots for
FFT deposition (Part 3), then convert completed FFT locals at the Cartesian
downward boundary (Part 4). This part changes neither runtime dataflow.
