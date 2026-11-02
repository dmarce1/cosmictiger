# Gaussian-screened radial kernels

The target solvers are **FMM-PM, PP-PM, and PM**. For FMM-PM and PP-PM, the
PM/FFT solve supplies the Gaussian-screened long-range force and FMM or direct
particle pairs supply the complementary short force. PM supplies mesh gravity
alone. There is no long-range FMM solver in this design. The shared smooth
radial component also serves the existing Ewald correction and independent
split checks. This implementation stage supplies the screened kernel and
Ewald reuse; mesh solvers and integration follow separately.

## Required solver and integration structure

All three PM variants have a mandatory, global outer PM interval. There is no
option to disable that outer structure. Its duration is selected from the PM
timestep limit and applicable physical/end-time limits, and need not be a
power of two. The existing constant/traditional/hierarchical integration
selection specifies the integration **inside** that interval. Inner block
steps are power-of-two subdivisions of the chosen outer duration. With pure
PM gravity, inner evolution is needed only when SPH or Hopkins is active.

The intended outer composition is a long-force half kick, inner evolution
including drift and short-force/hydro updates, and a long-force half kick at
the new synchronized endpoint. PM solves use all gravitating particles at
those synchronized boundaries. Short-force rungs do not trigger new FFTs.
Adjacent outer intervals may have different durations; their closing and
opening half kicks must use the corresponding durations. The interval-local
integer phase must remain separate from physical/cosmological elapsed time,
including checkpoints and restarts.

The PM acceleration criterion uses an effective mesh smoothing length, not
the particle's short-force softening. Mesh spacing and cloud order set this
scale for pure PM; Gaussian width also matters for split PM. The precise
criterion and cosmological factors must be matched to the existing gravity
kick convention. Short-range gravitational and hydro limits choose inner
subdivision levels. Reuse the PMMM particle-pair traversal where practical,
but replace the hard near-cell force partition with the Gaussian short kernel
and a controlled tail cutoff/search radius.

For unsoftened particle pairs the short-force multiplier, relative to the
Newtonian force, is

\[
\operatorname{erfc}(q)+\frac{2q}{\sqrt\pi}e^{-q^2},\qquad q=\alpha r.
\]

Within a physical softening region, use the physical softened force minus
the smooth long force. Multiplying an arbitrary softened force by the
unsoftened screening multiplier does not give the same split. The full
Ewald derivative order remains supported; no rank cap is introduced.

For future direct-PP acceleration, compare bounded-interval Chebyshev
approximations to `E(q)=erfc(q)` and to `H(x)=x*erfc(1/x)`, where `q=alpha*r`
and `x=1/q`. The latter gives potential `-alpha*H(x)` and inward force
`alpha^2*x^2*H'(x)`. Differentiate the same approximation used for potential,
and select its degree using force error over the actual interaction interval.
An exterior fit need not impose exact values at the origin; the softened
interior still needs the physical softened force minus the smooth long force.
These are approximation candidates, not active kernel backends. An origin
Taylor series alone is inadequate over the proposed full range `q<=5`.

## Shared radial API

The shared kernel uses inverse length `alpha = 1 / (2 * splitRadius)` and the
negative gravitational potential convention, without the factor `G * mass`:

\[
g_S(r)=-\operatorname{erfc}(\alpha r)/r,\qquad
g_L(r)=-\operatorname{erf}(\alpha r)/r.
\]

`GaussianScreening<Scalar>` is constructed from `alpha`; `fromSplitRadius(rs)`
provides the alternative length convention. The component and derivative order
are compile-time choices; the screening value may be supplied at runtime.
The header implementation also permits constant propagation for Ewald's fixed
screening. This does not make the separation-dependent special functions
compile-time operations.

The full symmetric `cartesian::evaluateGreen<GaussianComponent::Short/Long>`
overload accepts a displacement and screening object. It fills all Cartesian
derivatives, including traces, in the same coordinate units as that displacement.
Existing supplied-Green P2L, M2L, and M2P operators consume the result. The
short component requires nonzero separation; the long component is finite at
the origin. Neither component adds physical force softening or periodic images.

The reusable radial interface returns radius-weighted coefficients

\[
C_{j,t}=r^{j-t}\left(\frac1r\frac{d}{dr}\right)^j g(r),\qquad j+t\le p.
\]

`coefficient<j,t>()` and the unit displacement reconstruct derivatives of rank
`j+t`. Combining the radius factors before evaluating the coefficients avoids
underflow from intermediate raw radial derivatives in scaled coordinates. The
generated `RadialDerivatives` operation counts cover tensor reconstruction only;
the radial evaluator is separate.

The short recurrence stores `d_j = r^j A_j` directly and shares its physical
scaling across derivative orders; trace factors add the remaining inverse-radius
powers. The long component uses a Gaussian integral series near the origin and
a recurrence outside that region. This avoids subtracting nearly equal forces
to obtain the small long force. For example, with `q=alpha*r`,

\[
1-\left[\operatorname{erfc}(q)+\frac{2q}{\sqrt\pi}e^{-q^2}\right]
=\frac{4}{\sqrt\pi}\left(\frac{q^3}{3}-\frac{q^5}{5}
+\frac{q^7}{14}-\cdots\right).
\]

The squared separation must be finite and representable, and positive for the
short component. This is not an arbitrary-dynamic-range evaluator: a Gaussian
tail that has underflowed cannot be recovered by later dimensional rescaling.
Subnormal radial coefficients also lose relative accuracy, and Cartesian
multiplicities can amplify their absolute rounding error. A nonzero displacement
whose squared norm underflows to zero follows the long component's origin path.

Ewald retains its existing unit-box `alpha=2`, reciprocal tables, self correction,
background term, and Fourier-first accumulation order. Its internally scaled
coordinates require `alpha=2/kScaleFactor`. Other real-space images use the short
component. The primary-image correction uses the negative of the long component,
because the primary Newtonian potential is supplied separately. Exactly zero
separation continues to use the analytic self-correction table.

## Recovered particle-mesh cloud implementation

The recent ordinary PM solver is available in main's history. Commit
`9412ad483` added it on 2026-10-24; `71d4dd9eb` removed it on 2026-10-25.
The source immediately before removal is the appropriate recovery point:

```sh
git show '71d4dd9eb^:cosmictiger/gravity/particle_mesh_cloud.hpp'
git show '71d4dd9eb^:tests/particle_mesh_cloud.cpp'
git show '71d4dd9eb^:docs/particle_mesh.md'
```

The restored `cosmictiger/gravity/pm/cloud.hpp` contains centered cardinal B-spline weights: NGP (degree 0),
CIC (degree 1), TSC (degree 2), and PCS (degree 3). Three-dimensional assignment
is the product of the three one-dimensional weights. The tests explicitly
compare TSC and PCS against the one-dimensional SPH spline family. These are
Cartesian spline clouds, rather than a radial three-dimensional SPH density.

The exact fixed-point stencil selection in `cloudWeightsFromFixed` is preserved:
rounding a fixed64 position through double can otherwise choose the wrong cell
or half-cell. The restored tests cover partition/centroid, periodic wrap, boundary,
Nyquist derivative, and source-plus-sink assignment windows. The spectral
potential helper compensates both assignment passes with `1/W^2`; the Gaussian
long-range filter is an additional factor for the future FMM-PM solver.

Remote-tracking branches `origin/fmmpm` and `origin/treepm` also exist locally,
but their tips are from September 2022. The 2026 main-history implementation
above matches the requested SPH-equivalent cloud kernels and current naming.
This work restores the cloud kernels and tests, without reactivating the removed PM solver.

## Ewald kernel timing

On 2026-11-02, a paired CPU benchmark compared the preserved pre-extraction
Ewald kernel with the shared radial implementation. Each result is the median
of seven batches of 4096 full Ewald calls, pinned to one CPU, using GCC 15.
SIMD float uses eight lanes; timings below are microseconds per lane.

| Type | FMM order | Before | Shared kernel | Change |
| --- | ---: | ---: | ---: | ---: |
| double | 5 | 12.217 | 12.848 | +5.2% |
| SIMD float | 5 | 0.958 | 1.037 | +8.2% |
| double | 9 | 30.881 | 32.057 | +3.8% |
| SIMD float | 9 | 11.867 | 8.352 | -29.6% |

These are kernel timings, not full-walk or application measurements. The image
and Fourier sums and derivative ranks are unchanged. Source arithmetic does
not decrease at order 9, but the compiled SIMD body shrank from 90,712 to
74,679 bytes with fewer instructions and stack-reference sites. The exact
runtime cause of the order-dependent crossover has not been profiled.

## Validation

The `p3m` worktree passed the following focused checks on 2026-11-02:

- 90 CPU CTests: 29 screened-kernel tests, two strict/fast-math radial-range
  regressions, seven restored PM cloud tests, and existing Cartesian, Ewald,
  PMMM and operation-count regressions. Scalar float/double and CPU SIMD are
  checked through FMM orders 3–9 against independent long-double quadrature,
  analytic pair forces, scale transformations and short-field convergence.
- 22 actual CUDA CTests, with no skips: seven screened-kernel cases, seven
  full Cartesian cases, seven existing STF cases and cooperative L2L.
- Generator reproducibility for potential degrees 2–8, 273 executed weighted
  arithmetic checks, 35 spherical minimum selections and 80 build-checker
  self-tests.
- Syntax compilation of 26 moved/affected production C++ translation units,
  and object compilation of the production FMM order-5 CUDA instance,
  PMMM PP CUDA source and Ewald CUDA initializer.

After final formatting/include sorting, syntax checks also passed for all 26
production units and 17 standalone-test translation units. No arithmetic changed
during that formatting pass.

No full application link/run, multi-rank/MPI test or production simulation was
performed for this stage. The configuration disables hydrodynamics. The
Gaussian radial API and PP/FMM adapters are available, but the new scalar PM
solve, complete P3M traversals and mandatory outer PM interval remain future work.
