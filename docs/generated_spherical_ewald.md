# Generated spherical Ewald kernels

`tools/fmm/spherical_ewald.py` emits `generated_spherical_ewald.hpp` for the same
local degrees as the Cartesian and spherical generators (2 through 8,
FMM orders 3 through 9). The source retains every order; CMake's
`COSMICTIGER_FMM_MIN_ORDER`/`COSMICTIGER_FMM_MAX_ORDER` select the compiled
range, defaulting to 4 through 8, using `FMM_MIN_ORDER`/`FMM_MAX_ORDER`
preprocessor guards. This adds generator APIs only; it does not select a
new production gravity representation.

The formulas independently implement [Nitadori, PMMM, Appendix B, equations
34–42](https://arxiv.org/html/1409.5981v2), in the real spherical normalization of
[Dehnen, Appendix A](https://arxiv.org/pdf/1405.2255v1). The user's `spherical_fmm`
repository supplied useful normalization and validation comparisons; its
GPL-licensed C++ emitter was not copied into this Boost-licensed implementation.

## Representation and periodic meaning

Namespace `cosmictiger::gravity::spherical` uses slots
`index(n,m)=n*(n+1)+m`. Nonnegative slots hold real components; negative slots
hold the imaginary component at positive `|m|`. Complex negative orders obey
`C(n,-m)=(-1)^m*conj(C(n,m))`.

`EwaldGreensKernel<degree>::evaluate<Real>(greens, greenTrace, x, y, z)`
overwrites the harmonic coefficients of the **periodic correction** to negative
Newtonian potential in a unit cubic box, with splitting parameter alpha=2.
The displacement must be on the minimum-image branch, with each component in
[-0.5,0.5]. The central free-space singularity is removed. The origin is valid
and finite. Adding the ordinary free-space field supplies the complete
periodic interaction. A change of box length requires the appropriate
rank-dependent scaling before calling these unit-box APIs.

The routine generates the following work:

- The central correction uses regular harmonics multiplied by a positive
  lower-incomplete-gamma series. Combining the radial powers algebraically
  avoids `0*infinity` and cancellation of two singular terms.
- Noncentral real images use upper-incomplete-gamma screening factors whose
  normalized recurrence is generated through the requested degree.
- Reciprocal terms use regular harmonics of integer wave vectors, with exact
  degree-mod-four sine/cosine phases. Opposite wave vectors are paired.
- Opposite real images are paired before compensated accumulation. Both
  lattices use an integer sphere of radius six, 924 nonzero image vectors and
  462 reciprocal pairs. The test oracle instead uses radius seven.

For degrees at most eight and minimum-image displacements, the radius-six
cutoff is conservative: omitted real-image distances exceed
`6-sqrt(3)/2`, and omitted reciprocal terms have squared integer norm above 36.
The Gaussian factors decay at least as `exp(-4*r*r)` and
`exp(-pi*pi*k*k/4)`, respectively, times finite degree-eight polynomial factors.
This is a fixed supported-order cutoff, not a general arbitrary-order error
estimator. Extending the order range or displacement domain requires revisiting
it. Forty-eight positive-series stages cover `alpha^2*r^2 <= 3`; the ratio of
successive remaining terms is below `3/49.5`, including the lowest degree.
Floating-point cancellation, not the omitted series or lattice tail, sets the
observed error in the supported domain.

Float scalar and float SIMD inputs use **double working precision** for Ewald
harmonics and lattice accumulation, then round once into their output type.
This is necessary to preserve small, high-degree coefficients near symmetry
points. Double input stays double; no long double is used. CUDA scalar kernels
use the same policy. This is an accuracy choice, not a performance claim.

## Trace and moment contracts

`EwaldM2lKernel<degree>::add<Real>(local, localTrace, moments, sourceTrace,
                                greens, greenTrace)`
adds the general spherical contraction with source degree at most `degree-1`
and combined degree at most `degree`. It preserves pre-existing destination
values. Source moments use regular harmonics of **negative** source offset,
matching the other generated spherical kernels.

The radial source moment is maintained separately from harmonic storage. P2M
adds `mass*(x*x+y*y+z*z)` to it. For M2M with displacement `d = oldCenter-newCenter`,
the destination trace increment is
`sourceTrace - 4*d.x*M[3] - 4*d.y*M[1] - 2*d.z*M[2] + dot(d,d)*M[0]`.
This formula follows from expanding the squared source offset after translation;
the factors of two reflect the regular-harmonic dipole normalization. The base
harmonic P2M/M2M APIs cannot reconstruct this independent radial moment.

`sourceTrace` is the raw radial second moment `sum(mass*|offset|^2)`.
It contributes when source degree two is retained (local degree >=3).
`greenTrace` is `4*pi/3`; `localTrace` accumulates `greenTrace*sourceMass`.
The positive trace variable represents the physical polynomial
`-localTrace*|offset|^2/2`. Harmonic quadrupoles cannot encode that polynomial.
The generated contraction also includes its potential correction from
`sourceTrace` and its gradient correction from source dipoles, following
Nitadori's equations 38–42. This is one specific isotropic periodic trace;
it is not the full hierarchy of traces needed for Plummer softening.

`EwaldL2lKernel<degree>::add<Real>(output, outputTrace, source, sourceTrace,
                                x, y, z)`
adds the harmonic translation and transports this quadratic polynomial.
The offset is new center minus old center.
`EwaldL2pKernel<degree>::add<Real>(field, source, sourceTrace, x, y, z)`
evaluates it at a point offset. Fields are `[potential, accelerationX,
accelerationY, accelerationZ]`; acceleration is minus the potential gradient.
A constant quadratic trace does not change on translation, but it changes
both potential and acceleration.

Do not rotate an arbitrary periodic Green tensor into an axial free-space
contraction. Rotating the displacement does not rotate the periodic lattice
into an axisymmetric interaction.

## Counting and validation

The report separates central, per-real-pair, per-reciprocal-pair, total Green,
and general-contraction costs. Addition, subtraction, multiplication, division,
and FMA each cost one; sqrt costs seven. Sign changes, conversions, assignments,
integer lattice control, and comparisons do not count. Exp, erfc, sin, and cos
calls are listed separately. Pairing and compensated accumulation are included.
The corresponding complete per-lane counts appear in each kernel's `kFlops_`.
These are source-operation weights, not measured elapsed time or hardware
instruction counts; float promotion does not alter the counted expression DAG.

`tests/gravity/nitadori/generated_spherical_ewald.cpp` exercises every order in
the selected compiled range, float and double, origin and near-origin,
Cartesian axes, general positions, and box corners. It compares all coefficients to an independent complex
recurrence, Gaussian quadrature central term, explicit finite gamma sums, and
an unpaired, larger lattice. It also checks nonzero source dipoles and radial
traces, destination accumulation, trace-aware translations and point evaluation,
heterogeneous SIMD lanes, and exact executed-source arithmetic counts using an
instrumented scalar. CUDA compilation and execution are separate validation
layers; the host suite does not establish a CUDA runtime result.
