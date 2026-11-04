# SPH kernels and Monaghan artificial viscosity

The kernels and artificial viscosity follow the nonrelativistic SPH treatment in
J. J. Monaghan,
*Smoothed Particle Hydrodynamics*, **Annual Review of Astronomy and Astrophysics**
30 (1992), 543–574,
[doi:10.1146/annurev.aa.30.090192.002551](https://doi.org/10.1146/annurev.aa.30.090192.002551),
ADS bibcode `1992ARA&A..30..543M`
([open-access scan](https://articles.adsabs.harvard.edu/pdf/1992ARA%26A..30..543M)).
Equation and printed-page references below were checked against that scan.

The active fluid model uses density and entropy (with derived internal energy) with the
[variational pressure operator](variational_sph.md), including grad-H corrections.
It supports an ideal gas, primordial chemistry, and the SH03 star-formation
closure. Ordinary particle advection is used; the optional XSPH variant in
equation 3.19 is not enabled.

## Notation in the code

Equation implementations use descriptive camelCase identifiers. Adjacent comment blocks
retain Unicode source notation, equation numbers and citations, then give a directory
mapping each symbol to the implementation. Derived formulas and comoving conversions
are identified explicitly. The same convention applies to CPU, SIMD, and CUDA code.

| Symbol | Code name in the pair loops | Meaning |
| --- | --- | --- |
| α_a, α_b, α_ab | `bulkViscosityCoefficient`, `neighborBulkViscosityCoefficient`, `pairBulkViscosityCoefficient` | Linear artificial viscosity coefficients |
| β_ab | `pairVonNeumannRichtmyerCoefficient` | Quadratic artificial viscosity coefficient |
| ρ_a, P_a, u_a | `density`, `pressure`, `internalEnergy` | Sink thermodynamic state |
| H_a, H_b | `sinkKernelSmoothingLength`, `neighborSmoothingLength` | Compact support radii |
| r_ab, v_ab | `separation`, `velocityDifference` | Sink minus neighbor vectors |
| μ_ab, Π_ab | `viscosityVelocity`, `pairArtificialViscosity` | Monaghan velocity measure and pair viscosity |
| F_a, F_b, F_ab | `sinkGradientFactor`, `neighborGradientFactor`, `symmetricGradientFactor` | Radial kernel-gradient factors |
| ζ_a | `gradHCorrection` | Adaptive-support correction |
| Σ_b w, Σ_b q w′ | `kernelSum`, `kernelDerivativeMoment` | Dimensionless neighbor sums |

The `Pack` suffix denotes SIMD lane partial sums or values. Public INI/CLI keys
such as `hydro.viscosityAlpha` and `hydro.viscosityBeta` remain unchanged.

Particle storage `sphParticlesSmoothingLengths`, the `sphParticlesSmoothingLength()`
accessor, and the `sourceSmoothingLength`/`sinkSmoothingLength` working buffers
all hold the full compact-support radius \(H\). For the cubic spline \(H=2h\).
The names do not change the stored values or the kernel and viscosity length conversions.

## Kernel and density

The default spline in section 7, pp.554–555, is written here with the paper's
length denoted by \(h_M\):

\[
 W(r,h_M)=\frac{1}{\pi h_M^3}
 \begin{cases}
  1-\frac32q^2+\frac34q^3,&0\le q\le1,\\
  \frac14(2-q)^3,&1<q\le2,\\
  0,&q>2,
 \end{cases}
 \qquad q=r/h_M.
\]

The code stores the **support radius** \(H\), equal to \(2h_M\) for this default
cubic. Its dimensionless kernel has normalization \(8/\pi\), changes polynomial
at \(r/H=1/2\), and vanishes at \(r/H=1\). The common effective radius is
\(h=2\sigma\), where \(\sigma\) is the one-coordinate standard deviation of
the normalized kernel. It retains the existing `2 sigma` value without a
factor-of-two conversion. For the cubic, \(H=\sqrt{10/3}\,h\); thus the common
\(h\) and the paper's \(h_M\) are distinct.

The default `hydro.neighborNumber` is 42. The name cubic B-spline refers to this
radial M4 kernel; it is not a tensor-product bicubic interpolation.
`hydro.kernelFamily` and `hydro.kernelDegree` can instead select cardinal splines
of degrees 2–9, Wendland C0/C2/C4, or Ferrers F1–F4. Gravity has an independent
selection. See [kernel choices and length conventions](kernels.md) for the valid
degrees, conversion factors, and migration of explicit old settings. Production
gravity and SPH use three-dimensional tables; a separate arbitrary-dimensional
density-table generator is available as described there. Select SIMD CPU SPH with
`--hydro.cuda=false`, or CUDA SPH with `--hydro.cuda=true`. Both backends
implement support-radius solving, density, forces/heating, and neighbor timestep
propagation using the same kernel selections and length conventions. CPU workers
process separate tree leaves, with eight-lane SIMD packs over their neighbors.
All source fields are staged before a phase updates local sinks, including sources
fetched from other MPI ranks.

Particle drift independently uses the CPU when `drift.cuda=false`. Each available
initialization or analysis backend has its own CUDA selector. All selectors default
to `false`; the complete executable can run without a CUDA device.
Global, traditional, and hierarchical integration share the chronological
midpoint hydro operator. Gravity applies its own kicks. Each hydro event freezes
its source velocities, entropies, densities and viscosity coefficients before any
CPU/CUDA worker changes particles. The finer-rung endpoint owns each interacting
pair; its impulse is accumulated with equal and opposite signs, including the
reaction on an inactive neighbor. Event-local velocity and entropy increments are
returned to their owning MPI ranks and committed after the force pass. Timestep
selection retains only the scalar hydro rung bound (or the global raw timestep).
No per-gas acceleration or entropy-rate array survives an event. Enabled cooling
and star formation run at each gas particle's completed interval endpoint;
mesh gravity coupled to SPH is currently rejected.

The standalone `sph_cpu_test` target can be built with
`COSMICTIGER_WITH_CUDA=OFF` and does not call a CUDA device. The installed HPX
library may itself link the CUDA runtime.

Write the selected density kernel as \(W_H(r)=H^{-3}w(r/H)\). Density includes
the self contribution and uses the gather sum

\[
 \rho_i=\sum_j m_j W_{H_i}(r_{ij}).
\]

This gather choice is a deliberate practical variation: equation 6.3 on p.553
writes the adaptive density interpolant with the **source** length \(h_j\).
Section 6 permits summation instead of integrating the continuity equation, but
does not prescribe this gather solve or its numerical parameters.

Gas particles share one constant class mass. Other particle classes may have
different constant masses; there is no per-particle SPH mass field. The gas
smoothing solve holds

\[
 \frac{4\pi}{3}\frac{\rho_i H_i^3}{m}=N_{\rm target},
\]

so \(H_i\propto\rho_i^{-1/3}\), consistent with equation 6.2. The target is a
kernel-weighted count, including self, rather than an integer number of neighbors.
The self contribution imposes \(N_{\rm target}>(4\pi/3)w(0)\), which is
\(32/3\) for the default cubic. Startup checks use the selected kernel's floor,
including its single-precision representation. The solve respects the configured
radius bounds, supplied as `hydro.minEffectiveRadius`/`hydro.maxEffectiveRadius`
or `hydro.minSmoothingLength`/`hydro.maxSmoothingLength` and converted with the selected
hydro kernel. Explicit effective-radius settings take precedence with a warning
naming each ignored support-radius setting. It refreshes neighbors if the search
box is too small, and caps each local solve at 128 iterations. Its relative residual tolerance
is \(10^{-4}\); a bound-limited solution need not attain the requested target.
The periodic search uses one nearest image per pair, so the effective maximum
support is also capped at half the box width; a requested minimum above that limit
is rejected. The effective limits are printed at startup in initial interparticle
spacings. These numerical choices are implementation details, not prescriptions
in the paper.

## Pair force, viscosity, and thermal work

Let \(\mathbf r_{ij}=\mathbf r_i-\mathbf r_j\),
\(\mathbf v_{ij}=\mathbf v_i-\mathbf v_j\). The symmetric gradient is

\[
 \nabla_i\widetilde W_{ij}
 =\tfrac12[\nabla_i W_{H_i}(r_{ij})+\nabla_i W_{H_j}(r_{ij})].
\]

This is one of the two choices explicitly offered on p.554. A pair participates
when it lies inside either support radius. The zero-separation gradient is zero.

For `none`, `morrisAndMonaghan`, and `cullenAndDehnen`, the artificial-viscosity
contributions to the physical-coordinate pair rates are

\[
 \frac{d\mathbf v_i}{dt}
 =-\sum_j m_j \Pi_{ij}\nabla_i\widetilde W_{ij},\qquad
 \frac{du_i}{dt}
 =\frac12\sum_jm_j \Pi_{ij}\mathbf v_{ij}\cdot\nabla_i\widetilde W_{ij}.
\]

The [variational operator](variational_sph.md#constraint-and-equations) supplies
the pressure force and reversible thermal work separately.

These modes follow equation 4.1, the unnumbered definition of
\(\Pi_{ij}\) that follows it, and equation 4.2, pp.550–551:

\[
 \mu_{ij}=\frac{\bar\ell_{ij}\,\mathbf v_{ij}\cdot\mathbf r_{ij}}
 {r_{ij}^2+\epsilon\bar\ell_{ij}^2},\quad
 \Pi_{ij}=\frac{-\alpha\bar c_{ij}\mu_{ij}+\beta\mu_{ij}^2}{\bar\rho_{ij}}
 \quad\hbox{for }\mathbf v_{ij}\cdot\mathbf r_{ij}<0,
\]

and \(\Pi_{ij}=0\) otherwise. Arithmetic means are used for density and sound
speed. The viscosity length scale is
\(\bar\ell_{ij}=\sqrt{5/6}\,(h_i+h_j)/2\), using the effective radii
\(h_i=H_i/\texttt{smoothingLengthPerEffectiveRadius}\). The fixed factor makes
\(\bar\ell_{ij}=(H_i+H_j)/4\) for the cubic and preserves the previous
interpretation of the viscosity parameters. The same factor applies to every
selected family. The regularization remains \(\epsilon=0.01\). With variable
viscosity disabled the default \(\alpha=1\) and \(\beta=2\) recover the constant
coefficients of the paper. The default evolving coefficients are described below.

The optional Balsara (1995) shear limiter defaults on for constant α (`none`)
and Morris–Monaghan, and off for Cullen–Dehnen, Hu and Price. It uses equation (72) of
Springel, Pakmor, Zier & Reinecke (2021), *Simulating cosmic structure formation
with the GADGET-4 code*, MNRAS 506, 2871–2949,
[doi:10.1093/mnras/stab1855](https://doi.org/10.1093/mnras/stab1855)
([author PDF, section 5.3](https://wwwmpa.mpa-garching.mpg.de/gadget4/gadget4-code-paper.pdf)).
Equation (72) belongs to GADGET-4, not to the original Balsara paper.
An explicit `hydro.balsaraSwitch=true` remains supported for every model; for
CD, Hu or Price it prints a warning that their own shear limiter already supplies
shear suppression. This default is resolved after CLI/INI precedence.

The overlap concerns purpose, not algebraic identity.
[Cullen–Dehnen equations (13), (18)](https://arxiv.org/html/1006.1524#S3.SS3)
limit the shock indicator using the shear tensor and neighbor compression signs.
[Hu equations (17), (19)](https://arxiv.org/html/1402.1788#S2.SS3)
multiply the complete alpha update by a quadratic divergence/curl limiter;
[GADGET-4's force code](https://wwwmpa.mpa-garching.mpg.de/gadget4/doxygen/hydra_8cc_source.html#l00908)
therefore uses unit pair factors with `TIMEDEP_ART_VISC`.
[Price equations (46), (47)](https://arxiv.org/html/1702.03930#S2.SS2.SSS9)
already include a modified Balsara limiter in the shock indicator. An extra pair
factor further suppresses decaying alpha and, for Price, the independent quadratic
beta term retained by equations (40)–(41). It changes the published schemes'
forces; removing it is not a numerically neutral optimization. Constant α and
MM have no corresponding built-in shear limiter.

For constant α and Morris–Monaghan, enabling Balsara makes the density pass
accumulate the separate SPH gather estimates
\(\rho_i\,\nabla\!\cdot\mathbf v_i=-\sum_j m_j\mathbf v_{ij}\cdot\nabla_i W_{ij}(H_i)\)
and
\(\rho_i\,\nabla\!\times\mathbf v_i=\sum_j m_j\mathbf v_{ij}\times\nabla_i W_{ij}(H_i)\).
It computes \(f_i=|\nabla\!\cdot\mathbf v_i|/(|\nabla\!\cdot\mathbf v_i|+
|\nabla\!\times\mathbf v_i|)\), with \(f_i=0\) when both derivatives vanish,
and multiplies \(\Pi_{ij}\) by \((f_i+f_j)/2\) in both force and heating.
The density normalization cancels in the ratio. These direct divergence/curl
sums are omitted when Balsara is disabled, apart from the divergence required
for Morris–Monaghan's endpoint compression source.

When extra Balsara is explicitly enabled, Cullen–Dehnen, Hu and Price obtain both derivatives from the same
matrix-corrected velocity gradient \(\mathbf V\) already reconstructed for their
endpoint viscosity update. In canonical cosmological variables,

\[
 \mathbf V=\nabla_{\mathbf x}\mathbf w,\qquad
 G=\operatorname{tr}(\mathbf V),\qquad
 \omega^2=\sum_{k<l}(V_{kl}-V_{lk})^2,\qquad
 f=\frac{|G|}{|G|+\omega}.
\]

The common conversion from canonical to physical peculiar derivatives cancels
in the ratio. This is a matrix-corrected estimator of divergence and curl,
not the former uncorrected kernel-weighted gather estimator; their numerical
values need not agree on a disordered particle distribution. All derivatives
use peculiar velocity differences, with homogeneous Hubble expansion excluded
from the sensor.
For this shear correction, only \(f_i\) is retained between density and force
passes, encoded as one `uint8_t` per gas particle:

\[
 b_i=\operatorname{round}(255f_i),\qquad \widehat f_i=b_i/255.
\]

The endpoints zero and one are exact; the maximum absolute rounding error is
\(1/510\). Evaluation and force/source staging use float. Constant-α and
Morris–Monaghan runs recompute Balsara from the current midpoint density pass.
CD, Hu and Price compute it with \(\mathbf V\) at startup and completed own
endpoints, retaining the factor until the next such reconstruction. Their
midpoint passes perform neither separate divergence/curl sums nor a new
\(\mathbf V\) reconstruction. The derivative scratch is then discarded.
For CD, Hu and Price, the retained byte follows particle reordering, migration
and checkpoints when Balsara is enabled, preserving the sampled factor without
an extra gradient reconstruction. Constant-α and Morris–Monaghan factors remain
transient and are recomputed after restart or migration. Checkpoint metadata
records whether the limiter is enabled and rejects a restart with a different setting.
Use `--hydro.balsaraSwitch=false` to disable this shear correction.

The variational pressure terms and symmetric artificial viscosity conserve linear
and angular momentum and balance kinetic with internal energy. This algebraic
identity does not imply exact conservation by a finite-step asynchronous integrator.

## Morris–Monaghan variable viscosity

`hydro.viscosityEvolution=morrisAndMonaghan` enables the original additive switch of
Morris & Monaghan (1997), *A Switch to Reduce SPH Viscosity*, JCP 136, 41–50,
[doi:10.1006/jcph.1997.5690](https://doi.org/10.1006/jcph.1997.5690):

\[
 \frac{d\alpha_i}{dt}=-\frac{\alpha_i-\alpha^\ast}{\tau_i}+S_i
 \quad\text{(5)},\qquad
 \tau_i=\frac{h_{M,i}}{\mathcal C_1c_i}\quad\text{(6)},\qquad
 S_i=\max(-\nabla\!\cdot\mathbf v_i,0)\quad\text{(13)}.
\]

The defaults are \(\alpha^\ast=0.1\) and \(\mathcal C_1=0.2\), as used in their
calculations. There is no hard upper cap or \((\alpha_{\max}-\alpha)\) source
multiplier. The source uses equation (13) directly, without an additional
adiabatic-index scaling. The decay length is the same calibrated viscosity length
\(h_{M,i}=\sqrt{5/6}\,H_i/\texttt{smoothingLengthPerEffectiveRadius}\), equal to
\(H_i/2\) for the cubic spline. New particles start at \(\alpha^\ast\); this
smooth-flow initialization is an implementation choice, since the paper uses
problem-dependent initial values, including elevated α near the initial Sod interface.

Morris–Monaghan and Cullen–Dehnen pair forces and heating use
\(\bar\alpha_{ij}=(\alpha_i+\alpha_j)/2\), preserving pair symmetry.
Price mode uses the separate directed coefficients below.
The single quadratic coefficient `hydro.viscosityBeta` (default 2)
has a model-dependent meaning:

- With `morrisAndMonaghan`, `cullenAndDehnen`, and `hu_etal`, \(\beta_{ij}=\texttt{viscosityBeta}\,
  \bar\alpha_{ij}\), the convention of Morris & Monaghan's equation (3), so β decays
  with α.
- With `none` and `priceEtAl`, \(\beta_{ij}=\texttt{viscosityBeta}\) is constant: the
  fixed Monaghan coefficients, and Phantom's equation (41), in which β does not
  multiply α.

With `--hydro.viscosityEvolution=none` every pair uses the constant
`hydro.viscosityAlpha` and `hydro.viscosityBeta`; the disabled mode allocates no
per-particle variable-viscosity state.

At each particle's completed interval endpoint, the density pass computes the
instantaneous peculiar velocity divergence and advances α with a single backward
Euler step over that completed interval:

\[
 \alpha^{n+1}=\frac{\alpha^n+\Delta t(S+\alpha^\ast/\tau)}{1+\Delta t/\tau}.
\]

The source and decay time are frozen during this update. Divergence and curl are then discarded; neither their
history nor \(\nabla\cdot(\rho\mathbf v)\) is needed. Momentum divergence contains
an advected-density term and is not a Galilean-invariant compression indicator.
The derivative-of-divergence indicator described after equation (72) in GADGET-4
belongs to a later switch, not to Morris–Monaghan equation (13).

Only the scalar `bulkViscosityCoefficient` (α) survives sorting, domain migration,
and checkpoints; no rate or predictor is stored. It occupies two bytes in IEEE
binary16, while the update arithmetic uses float. The configured MM minimum, or
the selected gradient model's maximum, must lie in `[2^-24, 65504]`. Constant-alpha
viscosity has no per-particle alpha array and keeps its float range. The density
convergence flag ensures retries do not advance α twice. Updates occur only at
each particle's completed own interval endpoint, before gas rung changes and
source conversion. Subsequent midpoint force evaluations use the last endpoint's
α. Startup, semiactive measurements, output preparation and post-source bound
estimates do not advance α. This endpoint update gives the accepted first-order
viscosity timing. Silo writes a
dimensionless `alpha` gas field when viscosity evolution is enabled.
Morris–Monaghan retains and writes no divergence field; Cullen–Dehnen has
additional state described below.

In the cosmological variables below, the normalized density derivative sum is
\(D_i=\nabla_{\mathbf x}\cdot\mathbf w_i=a^2\theta_{{\rm pec},i}\), where
\(\theta_{\rm pec}=\nabla_{\mathbf r}\cdot\mathbf v_{\rm physical}-3H\). Thus

\[
 \frac{d\alpha_i}{d\tau}
 =-(\alpha_i-\alpha^\ast)\frac{\mathcal C_1c_{\rm stored,i}}{a h_{M,i}}
 +\frac{\max(-D_i,0)}{a}.
\]

This excludes homogeneous expansion from the compression sensor. The actual
viscous pair force still uses physical relative velocity including Hubble flow.

## Cullen–Dehnen variable viscosity

`--hydro.viscosityEvolution=cullenAndDehnen` selects Cullen & Dehnen (2010),
*Inviscid smoothed particle hydrodynamics*, MNRAS 408, 669–683,
[doi:10.1111/j.1365-2966.2010.17158.x](https://doi.org/10.1111/j.1365-2966.2010.17158.x)
([author paper](https://arxiv.org/pdf/1006.1524)). Their equations (13)–(16) define

\[
 A_i=\xi_i\max\!\left[-\frac{D(\nabla\cdot\mathbf v)_i}{Dt},0\right],\qquad
 \alpha_{{\rm loc},i}=\alpha_{\max}
 \frac{H_i^2 A_i}{(v_i^{\rm sig})^2+H_i^2 A_i},
\]
\[
 v_i^{\rm sig}=\max_{j:r_{ij}\le H_i}
 \left[\frac{c_i+c_j}{2}-\min(0,\mathbf v_{ij}\cdot\widehat{\mathbf r}_{ij})\right],
 \qquad \tau_i=\frac{H_i}{2\ell v_i^{\rm sig}}.
\]

Here the paper's smoothing length is the **full support radius** \(H_i\), unlike
the Morris–Monaghan cubic length \(H_i/2\). The CD floor and initial α are zero;
\(\ell=0.05\) follows the paper. `hydro.viscosityAlphaMax=2` is an implementation
choice. Equation (A15) raises α immediately to the target when needed and otherwise
decays it exponentially toward that target.

The limiter is equation (18), with the neighbor compression-sign measure from
(17):
\[
 R_i=\frac{\sum_jm_j\operatorname{sign}[(\nabla\cdot\mathbf v)_j]W_{ij}}{\rho_i},
 \qquad
 \xi_i=\frac{|2(1-R_i)^4\nabla\cdot\mathbf v_i|^2}
 {|2(1-R_i)^4\nabla\cdot\mathbf v_i|^2+\operatorname{tr}(\mathbf S_i\mathbf S_i^T)},
\]
where \(\mathbf S_i=\tfrac12(\mathbf V_i+\mathbf V_i^T)
-\tfrac13\operatorname{tr}(\mathbf V_i)\mathbf I\) and
\(\mathbf V_i=\nabla\mathbf v_i\). The matrix-corrected gradient estimates follow
(B2), (B4), and (B7), using mass/kernel-gradient weights so they do not read
densities being updated concurrently on other ranks. A symmetric pseudoinverse
handles planar or collinear neighborhoods on their resolved subspace.
The production shock slope uses the first alternative in Appendix B3:
the change of a particle's measured divergence between two sampling times.
Cosmological integration follows GADGET-4 by sampling the canonical gradient
\(G=\nabla_{\mathbf x}\cdot\mathbf w=a^2\theta_{\rm pec}\). With conformal
time \(d\tau=dt/a\), the locally derived conversion is

\[
 \mathcal D_n=\frac{1}{a_n^2}\frac{DG}{Dt}\simeq
 \frac{G_n-G_{\rm previous}}{a_n^3(\tau_n-\tau_{\rm previous})},
 \qquad
 a_n^4\mathcal D_n\simeq
 a_n\frac{G_n-G_{\rm previous}}{\tau_n-\tau_{\rm previous}}.
\]

The second expression is the slope passed to the viscosity model in stored
units. It includes the \(2H\theta_{\rm pec}\) term derived below; it is not
the finite difference of physical peculiar divergence.

Gradient history and its alpha update are committed at each particle's completed
interval endpoint, after density convergence and before gas rung changes.
The old live rung therefore gives the endpoint separation directly. For dyadic
intervals with base duration \(T\),

\[
 \Delta\tau=T\,2^{-r_{\rm live}}.
\]

Global stepping supplies the actual completed interval. The endpoint difference
estimates the slope centered on that interval's midpoint but is available only
at its end; its updated alpha is used by the following hydro kick. This accepted
viscosity timing is first order. Pressure and viscous pair-force evaluations
remain at their scheduled midpoints. Startup seeds the initial endpoint sample
without evolving alpha. Midpoint force solves, output preparation, semiactive
measurements and post-source timestep estimates do not commit gradient history.
Repeated density retries read unchanged committed history. No acceleration history
is needed.

All evolving-alpha models, including Morris–Monaghan, use this endpoint cadence;
only CD, Hu and Price require divergence history. When cooling or star formation
is active, the existing endpoint density/gradient pass supplies the update before
sources run. Without those sources, every evolving-alpha model performs an
additional endpoint pass. The post-source bound refresh advances neither alpha
nor history a second time.

CD uses the decoded canonical gradient for the immutable neighbor-sign snapshot
in equation (17), following the sweep order in Appendix A4. All three gradient
models share one committed scalar sample; no wrapper, second divergence or
previous sampled rung is retained. The `divv` output
key is unchanged and reports decoded canonical \(G\) at the last committed
endpoint, with inverse-code-time units and explicit convention metadata. The
previous sample's scale factor is not retained, so output does not divide that
sample by the current \(a^2\). The locally derived
dimensionless storage normalization is

\[
 q_n=\delta\tau_n G_n,\qquad
 \widehat G_n=\operatorname{float}(q_n)/\delta\tau_n.
\]

Here \(\delta\tau_n\) is the particle's own conformal hydro interval. The
`Float16Storage* sphScaledDivergence` array stores one two-byte value per gas
particle. The live dyadic rung recovers the normalization. Every dyadic rung
mutation rescales the value according to

\[
 q_{\rm new}=q_{\rm old}\,2^{r_{\rm old}-r_{\rm new}},
\]

preserving the decoded canonical gradient to storage accuracy. Global stepping
instead uses the previous completed global duration from shared driver state;
before the first completion it falls back to the base duration. Startup dyadic
samples use the current live rung. No previous-rung byte, per-particle timestamp,
or exponent is needed, and the derivative itself is transient.
The magnitude of \(q_n\) is not assumed bounded by one and is not clamped to it.
Values outside finite binary16 range are rejected rather than saturated. In
particular, the global startup base duration can exceed the subsequently accepted CFL
step, so startup normalization need not produce an order-one value.

The history occupies exactly two bytes per gas particle in memory, checkpoints
and migration, down from eight, in addition to the unchanged two-byte binary16 α.
Half precision can conceal small changes between samples;
this is a storage choice, not a property of the published switch. Target α, decay
rate, limiter and pending divergence are four float scratch fields (sixteen bytes),
allocated only for a history-updating density event and released after its
update. Every evolution scheme writes dimensionless `alpha`; selecting `none`
allocates neither α evolution state nor divergence state. Under CD, β follows α
as described above.

## Hu et al. variable viscosity

`--hydro.viscosityEvolution=hu_etal` (the default) selects Hu et al. (2014), *SPHGal*,
[equations (16)–(19)](https://arxiv.org/html/1402.1788#S2.SS3), implemented in
`viscosity/hu_etal.hpp`. The matrix-corrected peculiar gradients and GADGET-4
cosmological shock-slope conversion are shared with Cullen–Dehnen, but the target
is unlimited:

\[
 Q_i=\max\!\left(0,-\frac{D(\nabla\cdot\mathbf v)_i}{Dt}\right),\qquad
 \alpha_{{\rm tar},i}=\alpha_{\max}
 \frac{H_i^2Q_i}{H_i^2Q_i+c_i^2},\qquad
 \xi_i=\frac{(\nabla\cdot\mathbf v_i)^2}
 {(\nabla\cdot\mathbf v_i)^2+|\nabla\times\mathbf v_i|^2+10^{-4}(c_i/H_i)^2}.
\]

The entire update is multiplied by the current limiter, including decaying old
viscosity:

\[
 \alpha_i^{n+1}=\xi_i
 \begin{cases}
 \alpha_{{\rm tar},i},&\alpha_i^n\le\alpha_{{\rm tar},i},\\
 \alpha_{{\rm tar},i}+(\alpha_i^n-\alpha_{{\rm tar},i})e^{-\Delta t/\tau_i},
 &\alpha_i^n>\alpha_{{\rm tar},i}.
 \end{cases}
\]

`hydro.viscosityAlphaMaxHu=1` sets the unlimited maximum;
`hydro.viscosityDecayHu=0.05` gives \(\tau_i=H_i/(2\ell V_i)=10H_i/V_i\),
with the same maximum approaching-pair speed \(V_i\) as CD. Here \(H_i\) is the
full support radius. The switch starts at zero and imposes no extra alpha floor,
as in the equations above; Hu's separate 0.1 floor used in their test suite is
not imposed. The optional `hydro.balsaraSwitch` defaults off for Hu; explicitly
enabling it preserves the setting and warns about duplicate shear suppression.
Pair beta follows alpha as for CD; the existing Monaghan pair operator is retained.
Hu shares the gradient-state lifecycle, including `divv` output and migration,
but does not require a neighbor divergence-sign snapshot. Its strong limiter is
transient and reconstructed with the endpoint/startup velocity gradient. Optional
Balsara uses that same gradient and retains its factor for subsequent midpoint
forces. Checkpoints identify Hu separately and
reject a change of viscosity model, maximum, or decay setting on restart.

## Price et al. (Phantom) variable viscosity

`--hydro.viscosityEvolution=priceEtAl` selects the switch of Price et al. (2018),
*Phantom: A smoothed particle hydrodynamics and magnetohydrodynamics code for
astrophysics*, PASA 35, e031,
[doi:10.1017/pasa.2018.25](https://doi.org/10.1017/pasa.2018.25)
([arXiv:1702.03930](https://arxiv.org/abs/1702.03930)), section 2.2.9. It is their
variation on Cullen & Dehnen, implemented in `cosmictiger/sph/viscosity/price_etal.hpp`:

\[
 A_i=\xi_i\max\!\left[-\frac{d}{dt}(\nabla\cdot\mathbf v_i),0\right]\ \text{(46)},\qquad
 \xi_i=\frac{|\nabla\cdot\mathbf v_i|^2}{|\nabla\cdot\mathbf v_i|^2+|\nabla\times\mathbf v_i|^2}
 \ \text{(47)},
\]
\[
 \alpha_{{\rm loc},i}=\min\!\left(\frac{10\,h_i^2A_i}{c_{s,i}^2},\alpha_{\max}\right)
 \ \text{(48)},\qquad
 \tau_i=\frac{h_i}{\sigma_{\rm decay}c_{s,i}}.
\]

The differences from CD are: ξ is the Balsara-type ratio of equation (47) rather than
the sign-weighted \(\operatorname{tr}(\mathbf S\mathbf S^T)\) limiter, so no neighbor
divergence snapshot is gathered; the target uses the sound speed and the fixed factor
10, capped at `hydro.viscosityAlphaMaxPrice` (default 1, the paper's \(\alpha_{\max}\));
the decay time uses the sound speed with `hydro.viscosityDecayPrice`
(\(\sigma_{\rm decay}=0.1\)); and when \(\alpha_{{\rm loc}}\le\alpha\) the decay of
equation (49) is integrated with the backward Euler step of equation (50),
\(\alpha^{n+1}=(\alpha^n+\alpha_{\rm loc}\Delta t/\tau)/(1+\Delta t/\tau)\), otherwise α
is set to \(\alpha_{\rm loc}\) immediately. The paper's \(h_i\) is the smoothing length
with support \(2h_i\) for the cubic spline, so the code uses the same calibrated
viscosity length \(h_{M,i}\) as Morris–Monaghan. The velocity gradient and curl retain the CD appendix-B matrix reconstruction.
The shock slope uses the same completed-endpoint canonical-gradient difference and
GADGET-4 cosmological conversion as CD and Hu. All three models sample
\(G=a^2(\nabla\cdot\mathbf v_{\rm physical}-3H)\) and use peculiar gradients
in their limiters. Price therefore no longer needs a separate background-subtracted
sample alongside a total-divergence sample. This replaces the acceleration-gradient estimator in equations
(53)–(54), without changing the target, limiter or directed pair forces.
The initial α is zero, and the paper's quadratic coefficient is the constant
`hydro.viscosityBeta` of equation (41). Its scalar history shares CD's migration
and checkpoint lifecycle.

The pair force uses Phantom's equations (39)–(41). Write
\(w_{ab}=\mathbf v_{ab}\cdot\widehat{\mathbf r}_{ab}\); then

\[
 q^a_{ab}=-\tfrac12\rho_a(\alpha_a c_{s,a}+\beta|w_{ab}|)w_{ab}
 \quad(w_{ab}<0),\qquad q^a_{ab}=0\quad(w_{ab}\ge0),
\]
\[
 C_{ab}=\frac{m_bq^a_{ab}}{\rho_a^2\Omega_a},\qquad
 \left.\frac{d\mathbf v_a}{dt}\right|_{\rm AV}
 =-\sum_b(C_{ab}\nabla_aW_{ab}(h_a)+C_{ba}\nabla_aW_{ab}(h_b)),
\]
\[
 \left.\frac{du_a}{dt}\right|_{\rm AV}
 =\sum_b C_{ab}\mathbf v_{ab}\cdot\nabla_aW_{ab}(h_a).
\]
The last expression is the viscous heating in equation (42). Each side uses its
own density, sound speed, alpha, gradient and grad-H factor; there is no pair-averaged
alpha or symmetric-gradient approximation. With equal gas masses the existing
number-density constraint gives \(m_b/\Omega_a=m_b-\zeta_a\). The same directed
coefficients in force and heating preserve pair momentum and total energy, with
nonnegative shock heating when the grad-H factors are positive.

Stored variables use the physical relative velocity, represented as
\(\mathbf w_a-\mathbf w_b+(a\dot a)\mathbf x_{ab}\), for both dissipation and heating;
the conformal-time rates have the same extra \(1/a\) as the pressure terms.
The optional `hydro.balsaraSwitch` multiplies both directed coefficients by the
same symmetric factor. Set `hydro.balsaraSwitch=false` to reproduce the paper's
force without that extra limiter; Phantom's internal switch limiter still operates.
`hydro.viscosityEpsilon` has no effect on this force. The existing Courant formula
uses the maximum radial approach speed in place of regularized \(-\mu_{ab}\), and
the maximum individual alpha instead of its pair average. This retains CosmicTiger's
timestep calibration rather than replacing its integrator with Phantom's.
No artificial thermal conductivity is added by this viscosity option.

## Time integration and cosmological variables

All integrators use chronological midpoint SPH, with the split
`K_gravity(Δt/2) D(Δt/2) K_hydro(Δt) D(Δt/2) K_gravity(Δt/2)`.
There are no persistent hydro acceleration, gas gravity acceleration, or entropy-rate
arrays and no per-particle predictor clocks. Every event stages an immutable snapshot
of the current split state across all ranks before applying any impulse.

Each pair is evaluated once at the finer endpoint's midpoint; equal-rung pairs have
one owner selected by rank and gas index. Its identical impulse is accumulated with
opposite signs for both endpoints, including an inactive coarse neighbor. Directed
viscous heating is applied separately to each endpoint. Event-local increments are
exchanged before committing particle state, so internal momentum is conserved up to
accumulation and float-storage roundoff. External forces have no reciprocal gas
endpoint. The velocity-dependent viscosity update need only be first order in time.

Enabled cooling and star formation run once at each gas particle's own completed
interval endpoint, after the second drift and before the next gravity boundary.
They update canonical entropy using the endpoint density. Star conversion
preserves mass, position, velocity and old rung; ordinary gravity handles the
new collisionless particle. Sources use first-order splitting, and an all-gas
zero-kick estimate refreshes timestep bounds afterward without another viscosity
history update. See [star formation](star_formation.md) and
[equilibrium cooling](primordial_chemistry.md) for the source state contract.

Gas rungs change only at common all-particle synchronization boundaries. One
enforcement phase iterates immutable candidate-rung snapshots to a fixed point over
the union of both supports, including inactive neighbors, until every interacting
pair differs by at most one rung. It then opens the new intervals. New contacts
between boundaries need not satisfy the cap until the next common boundary; there
are no retroactive wakeups or rollback snapshots. Hierarchical gravity selects gas
rungs from total top-level gravity and the hydro bound, then freezes those rungs
through its correction ladder. The support radius is held fixed during drift;
active particles and selected inactive neighbors solve the neighbor constraint
before hydro. There is no separate radius predictor or divergence timestep bound.
The one-byte hydro rung proposal retains the latest force-based timestep bound
until that synchronization point. It is distinct from the live rung that defines
the already-open interval. Viscosity uses that live rung and rescales its scalar
history whenever the rung changes; it stores no additional rung byte.

The Courant-viscous limit uses

\[
 \Delta t_{\rm cv}=C_{\rm CFL}\min_i
 \frac{h_i}{c_i+0.6[\alpha c_i+\beta\max_j(-\mu_{ij},0)]}.
\]

The code interprets two apparent issues in the printed p.563 formulas:

- The printed force limit \(h/|\mathbf f|\) lacks a square root and has dimensions
  of time squared. The implemented acceleration limit scales as
  \(\sqrt{h/|\mathbf a|}\).
- The printed viscous limit uses \(\max\mu\), although equation 4.2 makes \(\mu\)
  negative for approaching particles. The implementation uses the nonnegative
  compression magnitude \(\max(-\mu,0)\).

The numerator uses the effective radius \(h_i=2\sigma_i\). The Courant coefficient
defaults to `0.22821773229381923`, converted from `0.25` by \(\sqrt{5/6}\)
to preserve the default cubic Courant-viscous limit. The hydro acceleration
coefficient is `gravity.timestepEta` times the fixed reference calibration
`0.7115965735877551`; the new default gravity eta is `0.20401599335954483`.
This preserves the former default cubic/Ferrers F4 hydro acceleration limit.
The additional fractional thermal-energy bound uses reversible pressure work
plus homogeneous expansion, with the fixed \(\sqrt{5/6}\) reference conversion.
Viscous shock heating is controlled by the approach-speed term in the CFL
criterion; including it in the fractional bound would force arbitrarily short
steps when cold gas first enters a shock. The explicit integrator retains its
cosmological expansion limits.
The gravity acceleration criterion uses its own effective radius, independently of
the selected hydro family. These reference conversions do not establish equal
timestep accuracy for every kernel shape.

The `star` test's constant-α schedule (zero during relaxation, 0.05 afterward)
sets `hydro.viscosityEvolution=none` and `viscosityBeta=2*viscosityAlpha`
for each stage, so the relaxation stage is inviscid. It relaxes its initial polytrope with a velocity damping term outside
the paper, \(d\mathbf w/d\tau \mathrel{-}= \lambda_{\rm d}\mathbf w\), where
\(\lambda_{\rm d}\) is the internal `hydro.damping` rate per code time unit. The
timestep criteria above do not include it, so the explicit kick is stable only while
\(\lambda_{\rm d}\Delta t\) stays below order one. The test therefore sets
\(\lambda_{\rm d}=1/t_{\rm dyn}\) with \(t_{\rm dyn}=(G\rho_0)^{-1/2}\); with the
cosmological code units \(t_{\rm dyn}\approx2.7\times10^5\) code times, so the former
\(\lambda_{\rm d}=1\) gave \(\lambda_{\rm d}\Delta t\approx3\times10^3\) and the rungs
diverged on the second step.

CosmicTiger evolves comoving positions \(\mathbf x\), conformal time
\(d\tau=dt/a\), stored velocity \(\mathbf w=a\mathbf v_{\rm pec}\), and stored
entropy \(A=P/\rho^\gamma\); the specific internal energy \(E=a^2u\) is derived. Density is comoving. See [entropy evolution](sph_entropy_evolution.md). The driver passes
\(a\,da/dt\) to the SPH `expansionRate` field; this differs from the driver's own physical
\(da/dt\) field. The scaled physical relative velocity used for viscosity is

\[
 a\,\Delta\mathbf v_{\rm physical}
 =\Delta\mathbf w+a\frac{da}{dt}\Delta\mathbf x.
\]

Peculiar velocity differences supply the pair pressure work. Physical relative
velocities supply viscosity and its heating, so homogeneous expansion does not
cause artificial viscosity. The diagnostic homogeneous energy rate is

\[
 \frac{dE}{d\tau}\bigg|_{\rm background}
 =(5-3\gamma)\frac{da}{dt}E,
\]

which vanishes for \(\gamma=5/3\). This term is not integrated into entropy;
the energy conversion already includes homogeneous adiabatic expansion. The pair-force and thermal-work prefactors contain
\(m/a\) after converting to these stored variables. These transformations are
repository requirements; the cited equations in the review are noncosmological.

## Cosmological viscosity conventions

Every viscosity sensor uses peculiar gradients: the Morris–Monaghan compression
source, Balsara factor, CD neighbor signs and shear limiter, and the Price/Hu
limiters and divergence histories. With physical Hubble rate \(H=\dot a/a\),

\[
 G=\nabla_{\mathbf x}\cdot\mathbf w,
 \qquad \theta_{\rm pec}=G/a^2,
 \qquad \theta_{\rm physical}=\theta_{\rm pec}+3H.
\]

This removes homogeneous expansion from shock detection without changing the
physical relative velocity used by the pair force, heating and compressive
signal speed. Those pair quantities still include \(a^2H\Delta\mathbf x\)
in stored velocity units. A pair must approach physically to receive viscosity.

The official GADGET-4 source uses the same canonical velocity
\(\mathbf w=a\mathbf v_{\rm pec}\)
([initialization, line 324](https://wwwmpa.mpa-garching.mpg.de/gadget4/doxygen/init_8cc_source.html#l00324)).
Its [density gradient](https://wwwmpa.mpa-garching.mpg.de/gadget4/doxygen/density_8cc_source.html#l00992)
uses peculiar velocity differences. Its
[Balsara and pair-force code](https://wwwmpa.mpa-garching.mpg.de/gadget4/doxygen/hydra_8cc_source.html#l00859)
likewise excludes Hubble expansion from the limiter but includes it in the pair
approach test. GADGET-4 disables the extra Balsara multiplier when its Hu-style
time-dependent viscosity is enabled; CosmicTiger retains the user's independent
`hydro.balsaraSwitch` option.

CosmicTiger follows GADGET-4's
[time-dependent viscosity update, lines 139–140](https://wwwmpa.mpa-garching.mpg.de/gadget4/doxygen/artificial__viscosity_8cc_source.html#l00139)
by differencing \(G\) in physical time before multiplying by the current
\(a^{-2}\), or equivalently by differencing in conformal time and multiplying
by current \(a^{-3}\). The continuum comparison below is locally derived:

\[
 \frac{1}{a^2}\frac{DG}{Dt}
 =\frac{D\theta_{\rm pec}}{Dt}+2H\theta_{\rm pec}.
\]

Thus the switch contains an additional \(2H\theta_{\rm pec}\) term relative to
differencing physical peculiar divergence. All three gradient models use this
same convention. The cited pages are the official published source documentation;
they are not a verification of the latest GitLab commit.

## Parameters and compatibility

Application options use `groupName.optionName` with camelCase names. The principal
hydrodynamic controls are:

| Option | Default | Meaning |
| --- | ---: | --- |
| `hydro.enabled` | `false` | Enable the SPH gas solver |
| `hydro.kernelFamily` | `spline` | Density family: `spline`, `wendland`, or `ferrers` |
| `hydro.kernelDegree` | `3` | Density polynomial degree; valid values depend on family |
| `hydro.gamma` | `5/3` | Ideal-gas adiabatic index |
| `hydro.viscosityAlpha` | `1` | Constant linear coefficient when variable viscosity is disabled |
| `hydro.viscosityBeta` | `2` | Quadratic coefficient: multiplies \(\bar\alpha_{ij}\) for `morrisAndMonaghan`, `cullenAndDehnen`, and `hu_etal`, constant for `none` and `priceEtAl` |
| `hydro.viscosityEvolution` | `hu_etal` | α evolution: `none`, `morrisAndMonaghan`, `cullenAndDehnen`, `priceEtAl`, or `hu_etal` |
| `hydro.viscosityAlphaMaxHu` | `1` | Hu maximum before the strong limiter |
| `hydro.viscosityDecayHu` | `0.05` | Hu decay coefficient in τ = H/(2 ell V) |
| `hydro.viscosityAlphaMin` | `0.1` | Morris–Monaghan positive α floor and initial value |
| `hydro.viscosityDecay` | `0.2` | Morris–Monaghan decay coefficient \(\mathcal C_1\) |
| `hydro.viscosityAlphaMax` | `2` | Cullen–Dehnen maximum α; the MM scheme has no upper cap |
| `hydro.viscosityDecayCD` | `0.05` | Cullen–Dehnen decay coefficient \(\ell\) |
| `hydro.viscosityAlphaMaxPrice` | `1` | Price et al. maximum α in equation (48) |
| `hydro.viscosityDecayPrice` | `0.1` | Price et al. decay coefficient \(\sigma_{\rm decay}\) |
| `hydro.viscosityEpsilon` | `0.01` | Denominator coefficient multiplying \(\bar\ell^2\) |
| `hydro.balsaraSwitch` | model-dependent | `true` for `none`/`morrisAndMonaghan`, `false` for CD/Hu/Price; explicit `true` for a gradient model is retained with a warning |
| `hydro.neighborNumber` | `42` | Kernel-weighted support-sphere count |
| `compute.sphBucketSize` | `0` (automatic) | SPH tree leaf limit; automatic sizing uses `floor(8/pi * hydro.neighborNumber)`, giving `106` at the default neighbor target |
| `hydro.minEffectiveRadius` | unset | Alternative minimum effective radius \(h=2\sigma\), in initial interparticle spacings |
| `hydro.minSmoothingLength` | `1e-6` | Minimum \(H\), in initial interparticle spacings |
| `hydro.maxEffectiveRadius` | unset | Alternative maximum effective radius \(h=2\sigma\), in initial interparticle spacings |
| `hydro.maxSmoothingLength` | `1000` | Requested maximum \(H\), in initial interparticle spacings; capped at half the box |
| `hydro.centralEffectiveRadius` | `0.0054772255750516604` | Imposed central-force effective radius in box units; uses the gravity kernel |
| `hydro.centralSmoothingLength` | derived | Alternative central-force support radius in box units; uses the gravity kernel |
| `hydro.centralSoftening` | legacy alias | Alias for `hydro.centralEffectiveRadius` with the same value and units |
| `hydro.cfl` | `0.22821773229381923` | Courant-viscous coefficient using the effective radius \(h=2\sigma\); also controls the calibrated fractional-energy bound |

For each radius pair, an explicit effective-radius key wins over the
support-radius key, including across CLI and INI inputs, with a warning naming
the ignored key. For the central force, precedence is `centralEffectiveRadius`,
then the `centralSoftening` alias, then `centralSmoothingLength`. The command line
still overrides INI values for the same key. See [kernel radii](kernels.md) for
the equivalent gravity options and conversion factors.

SPH and gravity use independent tree bucket sizes. `compute.bucketSize` controls
the gravity tree, while a positive `compute.sphBucketSize` fixes the SPH leaf
limit. Zero selects the historical SPH sizing heuristic above after CLI and INI
neighbor options have been resolved. Startup output and snapshot metadata report
the resolved positive size. The gravity compile-time bucket cap does not apply
to SPH, whose kernels iterate over each leaf's particle range.

Gas and dark matter share one fixed comoving gravity kernel radius.
SPH density, pressure, viscosity, and thermal work retain the gas-only hydro
kernel and variable smoothing radii. See
[gravitational softening](kernels.md#fixed-comoving-gravitational-softening) for
the gravity radius options and their independence from SPH smoothing.

Checkpoints use version 3, and the low byte of `reserved` is 2 for the
[variational operator](variational_sph.md). Legacy Monaghan checkpoints (code 1)
are rejected. The next two bytes encode independent gravity and hydro kernel
selections. Bit 24 identifies the compact gas layout after removal of stellar,
chemistry, and obsolete solver fields. Older SPH checkpoints lack this bit and
are rejected before particle data are loaded. The gas payload has a separate
version: base 130000 omits all three persistent force-rate arrays and stores
binary16 alpha plus, for gradient viscosity, one dimensionless binary16
canonical-divergence scalar. The history is serialized as `H` in two bytes,
matching memory and migration. Its normalization is recovered from the live rung
or shared completed global duration, with a global startup base fallback.
There is no previous-rung byte, sample timestamp, exponent or pending
diagnostic payload. The three former byte arrays for convergence, semiactivity
and proposed rung are replaced by one `SphParticleFlags` byte per gas particle:
bit 0 is convergence, bit 1 semiactivity, and bits 2–7 encode the proposed rung.
Codes 0–15 are current valid rungs. There is no unset sentinel; all other
codes, including 63, are rejected on load. The byte precedes viscosity data
after density and also migrates as one byte, saving two bytes per gas particle
in memory and these payloads. Explicit setters update the packed fields because
an individual bit cannot be returned as a mutable reference.

Enabled Balsara adds one retained byte for CD, Hu and Price
only; their migration payload carries the same byte. Constant-α and
Morris–Monaghan Balsara factors remain transient. The existing physics and
viscosity suffixes remain; earlier
gas payloads, including base 90000's signed-significand histories and the
intermediate base-100000 sampled-rung, base-110000 endpoint and base-120000
unpacked-flag representations,
are rejected.
See the
[pair-midpoint design](sph_pair_midpoint_plan.md) and
[SPH octree design](sph_octree_plan.md) for the retained tree and kick schedule.
Compatible dark-matter checkpoint layouts remain unchanged.
Historical kernel marker values retain their original meanings; the Ferrers F2
default changes the selected gravity marker. A change of kernel selection on
restart is rejected.
A separate parameter block also rejects changes to viscosity coefficients,
smoothing constraints, timestep
controls, gravity enablement, and external accelerations on SPH restart. These
checks still apply when migrating an old configuration to the new length and
timestep conventions; old SPH checkpoints are not automatically converted.
Remove `hydro.chemistry`, `hydro.conduction`, `hydro.diffusion`, `hydro.yReflect`,
and `gravity.sphSmoothing` from older INI files, including entries set to false.
The `run.scenario=polytropicStar` scenario is a self-gravitating gas sphere; it does
not use stellar particles or star formation.

## Validation scope

The [pair-midpoint design](sph_pair_midpoint_plan.md#validation) records successful
CPU/CUDA builds and focused base-130000 checks: 45 CPU, 27 CUDA kernel and five
Balsara checks; 21 Python tests in normal and optimized execution; fifteen
Balsara option cases; all ten selected CPU and both selected CUDA application
CTests; restart, reordering and two-rank migration fixtures. Sixteen preset
edits were statically audited. This is focused validation, not a full suite,
performance benchmark or scientific convergence claim. Older results remain
labeled by representation: signed-significand/base-90000 checks belong to commit
`1bc12b5f6`, and broader source results to `9ccd34a85` and base 80000.

The independent `sph_kernels` and `monaghan_sph` CTest targets retain the reference cubic checks:
float and double kernel normalization, spline joins, analytic versus
finite-difference derivatives, the paper's \(H=2h_M\) convention, viscosity signs
and coefficients, regularization, and invariance under length-unit changes and
velocity boosts. Pair tests combine variational pressure with Monaghan viscosity
and use unequal pressures,
densities, masses, and supports to check net force, torque, and total energy rate,
including a pair inside only one support. They also check separation, uniform
translation, and the distinction between peculiar and physical thermal work.

The `monaghan_sph_cuda` target invokes the production density, smoothing, and
hydrodynamics kernels. It checks lower, upper, and fixed support bounds; the
weighted-neighbor residual; periodic support intersection; a uniform lattice;
nonzero perturbations; timestep selection; unequal-support pair conservation;
and homogeneous expansion with a nonmonatomic gamma. Selected-kernel checks
exercise all 15 family/degree choices through the production SPH path.
The same target compares all four SIMD CPU passes directly with CUDA, including
both hydro phases, permuted source/sink mappings, periodic positions, inactive and
semiactive particles, support-search retries, and immutable source snapshots.

The `sph_cpu` regression exercises the production SIMD code with ordinary host
allocations. It checks independent analytic density/force/timestep references,
pair conservation, expansion, every kernel selection, SIMD tails, periodic
geometry, activity, support bounds/retries, and iterative neighbor rung changes.
On Linux it enables floating-point traps to catch invalid arithmetic in masked
lanes. `smoke_sod_cpu`, `smoke_sph_cpu`, and `smoke_sph_cpu_mpi` cover shock-tube
evolution, cosmological evolution with restart, and two-rank MPI evolution through
the application driver.

The separate `radial_kernel` and `radial_kernel_cuda` targets check the generated
CPU/SIMD/CUDA tables, moments, gravitational force/potential consistency, and
float behavior near zero and the support boundary; see [kernels.md](kernels.md).

Historical validation on 2026-10-16, before selectable kernels and the effective
radius conversion: the complete CUDA build and all 23 CTest checks passed,
including grouped CLI/INI checks, CPU/CUDA mixed-mass gravity, SPH checkpoint
compatibility, MPI migration and evolution, the coarse Sod problem, and a short
cosmological SPH evolution with restart. At the smoke test's coarse resolution,
Sod density relative L1 error was 0.11484; the largest sampled relative total-energy
drift was 8.24e-8. These are regression results, not a resolution-convergence study
or evidence of long-time accuracy. No performance claim follows from them.

## Source layout

Common particle storage, source staging, neighbor traversal, CPU/CUDA execution,
and timestep scheduling stay directly in `cosmictiger/sph/` and `src/sph/`.
The pressure formulation uses `variable_smoothing_lengths.hpp` and `volume.hpp`.
Artificial-viscosity policies live in `cosmictiger/sph/viscosity/`:

- `monaghan.hpp`: the baseline artificial-viscosity formula.
- `morris_and_monaghan.hpp`: compression/decay equation with a backward Euler alpha update.
- `hu_etal.hpp`: Hu strong-limiter diagnostics and alpha update.
- `cullen_and_dehnen.hpp`: gradient reconstruction, shock detection, alpha updates, and model diagnostics.
- `price_etal.hpp`: the Phantom shock detector and backward Euler alpha update, reusing the CD gradient reconstruction and state.
- `balsara.hpp`: the shear limiter, composable with every evolving-alpha model.

A two-byte bulk viscosity coefficient array stores alpha for all evolution
models. Cullen–Dehnen, Hu and Price share the two-byte scalar `sphScaledDivergence`
array, with no wrapper or previous-rung byte, and a
separate sixteen-byte temporary diagnostic entry. Optional Balsara storage is
one byte per gas particle; evaluation and staged sources use float. The combined
history, alpha and Balsara arrays therefore use five bytes per gas particle
instead of fourteen when all are enabled. Constant viscosity allocates neither
history nor evolving alpha. The temporary diagnostic array is allocated
after tree sorting, retained through density retries, and freed before source
conversion; it is absent from density-only and post-source bound refreshes.
Alpha and committed history participate in reordering, migration and checkpoints,
as does enabled Balsara for CD, Hu and Price. Constant-α and Morris–Monaghan
Balsara factors remain transient and are recomputed after restart or migration.
Their lifecycle remains in common particle code.
The shared density and hydro passes call these policies without duplicating
neighbor walks. Standalone policy tests use the same author-directory organization.
