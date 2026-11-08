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
equation 3.19 of Monaghan (1992) is not enabled.

## Notation in the code

Equation implementations use the source's Unicode symbols as identifiers. Adjacent
comment blocks retain the source notation, equation numbers and citations, then give a
directory with a short description of each symbol. A `Code names:` line spells any
symbol that is not a legal C++ identifier, such as `abs_r𞁑𞁓` for |r_ab|. Derived
formulas and comoving conversions are identified explicitly. The same convention
applies to CPU, SIMD, and CUDA code.

| Symbol | Code name in the pair loops | Meaning |
| --- | --- | --- |
| α_a, α_b, α_ab | `α𞁑`, `α𞁓`, `α𞁑𞁓` | Linear artificial viscosity coefficients |
| β_ab | `β𞁑𞁓` | Quadratic artificial viscosity coefficient |
| ρ_a, P_a, u_a | `ρ𞁑`, `P𞁑`, `u𞁑` | Sink thermodynamic state |
| H_a, H_b | `h𞁑`, `h𞁓` | Compact support radii |
| r_ab, v_ab | `r𞁑𞁓`, `v𞁑𞁓` | Sink minus neighbor vectors |
| μ_ab, Π_ab | `μ𞁑𞁓`, `Π𞁑𞁓` | Monaghan velocity measure and pair viscosity |
| F_a, F_b, F_ab | `F𞁑`, `F𞁓`, `F𞁑𞁓` | Radial kernel-gradient factors |
| ζ_a | `ζ𞁑` | Adaptive-support correction |
| Σ𞁓 w, Σ𞁓 q w′ | `ΣW`, `Σqwʹ` | Dimensionless neighbor sums |

The `Pack` suffix denotes SIMD lane partial sums or values. Public INI/CLI keys
such as `hydro.viscosityAlpha` and `hydro.viscosityBeta` remain unchanged.

Particle storage `sphParticlesSmoothingLengths`, the `sphParticlesSmoothingLength()`
accessor, and the `sourceSmoothingLength`/`sinkSmoothingLength` working buffers
all hold the full compact-support radius \(H\). For the cubic spline \(H=2h\).
The names do not change the stored values or the kernel and viscosity length conversions.

## Kernel and density

The default spline in section 7, pp.554–555, of Monaghan (1992) is written here
with that paper's length denoted by \(h_M\):

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
\(h\) and the \(h_M\) of Monaghan (1992) are distinct.

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
two-call explicit-midpoint RK2 hydro operator; it is the only positive-duration
hydro integration scheme. Gravity applies its own driver-scheduled kicks and
never invokes SPH from a gravity kick. Each of the two hydro stages freezes its
source velocities, entropies, densities and viscosity coefficients before any
CPU/CUDA worker changes particles. The finer-rung endpoint owns each interacting
pair; its impulse is accumulated with equal and opposite signs, including the
reaction on an inactive neighbor. Event-local velocity and entropy increments
are returned to their owning MPI ranks and summed before publishing the predictor
or the final state. A gas-indexed acceleration vector and entropy-rate scalar,
plus sparse baseline records for recipients, exist only within the event.
Timestep selection retains the scalar hydro rung bound (or the global raw
timestep); force-rate scratch is released before migration and checkpoints.
See [the complete time-integration contract](sph_rk2_time_integration.md) for the
equations and current validation limits. Enabled cooling
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

This gather choice is a deliberate practical variation: equation 6.3 on p.553 of
Monaghan (1992) writes the adaptive density interpolant with the **source**
length \(h_j\). Section 6 of Monaghan (1992) permits summation instead of
integrating the continuity equation, but does not prescribe this gather solve or
its numerical parameters.

Gas particles share one constant class mass. Other particle classes may have
different constant masses; there is no per-particle SPH mass field. The gas
smoothing solve holds

\[
 \frac{4\pi}{3}\frac{\rho_i H_i^3}{m}=N_{\rm target},
\]

so \(H_i\propto\rho_i^{-1/3}\), consistent with equation 6.2 of Monaghan (1992).
The target is a kernel-weighted count, including self, rather than an integer
number of neighbors.
The self contribution imposes \(N_{\rm target}>(4\pi/3)w(0)\), which is
\(32/3\) for the default cubic. Startup checks use the selected kernel's floor,
including its single-precision representation. The solve respects the configured
support-radius bounds `hydro.minSmoothingLength`/`hydro.maxSmoothingLength`.
It refreshes neighbors if the search
box is too small, and caps each local solve at 128 iterations. Its relative residual tolerance
is \(10^{-4}\); a bound-limited solution need not attain the requested target.
The periodic search uses one nearest image per pair, so the effective maximum
support is also capped at half the box width; a requested minimum above that limit
is rejected. The effective limits are printed at startup in initial interparticle
spacings. These numerical choices are implementation details, not prescriptions
in Monaghan (1992).

## Pair force, viscosity, and thermal work

Let \(\mathbf r_{ij}=\mathbf r_i-\mathbf r_j\),
\(\mathbf v_{ij}=\mathbf v_i-\mathbf v_j\). The symmetric gradient is

\[
 \nabla_i\widetilde W_{ij}
 =\tfrac12[\nabla_i W_{H_i}(r_{ij})+\nabla_i W_{H_j}(r_{ij})].
\]

This is one of the two choices explicitly offered on p.554 of Monaghan (1992). A
pair participates when it lies inside either support radius. The zero-separation
gradient is zero.

The artificial-viscosity contributions to the physical-coordinate pair rates are

\[
 \frac{d\mathbf v_i}{dt}
 =-\sum_j m_j \Pi_{ij}\nabla_i\widetilde W_{ij},\qquad
 \frac{du_i}{dt}
 =\frac12\sum_jm_j \Pi_{ij}\mathbf v_{ij}\cdot\nabla_i\widetilde W_{ij}.
\]

The [variational operator](variational_sph.md#constraint-and-equations) supplies
the pressure force and reversible thermal work separately.

These terms follow equation 4.1 of Monaghan (1992), the unnumbered definition of
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
coefficients of Monaghan (1992). The default evolving coefficients are described
below.

The Balsara (1995) shear limiter is on with Morris–Monaghan and off with
constant α (`none`); no separate option exists. It uses equation (72) of
Springel, Pakmor, Zier & Reinecke (2021), *Simulating cosmic structure formation
with the GADGET-4 code*, MNRAS 506, 2871–2949,
[doi:10.1093/mnras/stab1855](https://doi.org/10.1093/mnras/stab1855)
([author PDF, section 5.3](https://wwwmpa.mpa-garching.mpg.de/gadget4/gadget4-code-paper.pdf)).
Equation (72) belongs to GADGET-4, Springel et al. (2021), not to the original
Balsara (1995) paper.

Balsara makes the density pass accumulate the separate SPH gather estimates
\(\rho_i\,\nabla\!\cdot\mathbf v_i=-\sum_j m_j\mathbf v_{ij}\cdot\nabla_i W_{ij}(H_i)\)
and
\(\rho_i\,\nabla\!\times\mathbf v_i=\sum_j m_j\mathbf v_{ij}\times\nabla_i W_{ij}(H_i)\).
It computes \(f_i=|\nabla\!\cdot\mathbf v_i|/(|\nabla\!\cdot\mathbf v_i|+
|\nabla\!\times\mathbf v_i|)\), with \(f_i=0\) when both derivatives vanish,
and multiplies \(\Pi_{ij}\) by \((f_i+f_j)/2\) in both force and heating.
The density normalization cancels in the ratio. These direct divergence/curl
sums are omitted when constant viscosity disables Balsara.

All derivatives use peculiar velocity differences, with homogeneous Hubble
expansion excluded from the sensor.
For this shear correction, only \(f_i\) is retained between density and force
passes, encoded as one `uint8_t` per gas particle:

\[
 b_i=\operatorname{round}(255f_i),\qquad \widehat f_i=b_i/255.
\]

The endpoints zero and one are exact; the maximum absolute rounding error is
\(1/510\). Evaluation and force/source staging use float. Balsara is recomputed
from the baseline and predictor velocity states. The second
hydro evaluation uses the predictor's limiter. Factors are transient and are
recomputed after restart or migration. Checkpoint metadata records whether the
limiter is enabled and rejects a restart with a different setting.

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
multiplier. Morris & Monaghan (1997) use equation (13) directly, without an
additional adiabatic-index scaling. The decay length is the same calibrated
viscosity length \(h_{M,i}=\sqrt{5/6}\,H_i/\texttt{smoothingLengthPerEffectiveRadius}\),
equal to \(H_i/2\) for the cubic spline. New particles start at \(\alpha^\ast\);
this smooth-flow initialization is an implementation choice, since
Morris & Monaghan (1997) use problem-dependent initial values, including elevated
α near the initial Sod interface.

Morris–Monaghan pair forces and heating use
\(\bar\alpha_{ij}=(\alpha_i+\alpha_j)/2\), preserving pair symmetry.
The single quadratic coefficient `hydro.viscosityBeta` (default 2)
has a model-dependent meaning:

- With `morrisAndMonaghan`, \(\beta_{ij}=\texttt{viscosityBeta}\,
  \bar\alpha_{ij}\), the convention of equation (3) of Morris & Monaghan (1997), so
  β decays with α.
- With `none`, \(\beta_{ij}=\texttt{viscosityBeta}\) is constant: the fixed
  Monaghan coefficients.

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
The derivative-of-divergence indicator described after equation (72) of GADGET-4,
Springel et al. (2021), belongs to a later switch, not to equation (13) of
Morris & Monaghan (1997).

Only the scalar `bulkViscosityCoefficient` (α) survives sorting, domain migration,
and checkpoints; no rate or predictor is stored. It occupies two bytes in IEEE
binary16, while the update arithmetic uses float. The configured minimum must lie
in `[2^-24, 65504]`. Constant-alpha
viscosity has no per-particle alpha array and keeps its float range. The density
convergence flag ensures retries do not advance α twice. Morris–Monaghan updates occur only at
each particle's completed own interval endpoint, before gas rung changes and
source conversion. Subsequent midpoint force evaluations use the last endpoint's
α. Startup, semiactive measurements, output preparation and post-source bound
estimates do not advance α. This retained backward-Euler endpoint update is first
order; it limits the accuracy of the coupled variable-coefficient calculation
even though the velocity/entropy hydro operator uses RK2. Silo writes a
dimensionless `bulkViscosityCoefficient` gas field when viscosity evolution is enabled.
Morris–Monaghan retains and writes no divergence field.

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

## Optional artificial thermal conduction

`hydro.conductionCoefficient` is the dimensionless coefficient \(g\) in
Monaghan (1992), equation (4.8). It defaults to **0**, which disables conduction.
A finite positive value enables equations (4.5)/(4.8) of Monaghan (1992)
independently of the viscosity model, variable alpha, and Balsara limiter. For
example, `--hydro.conductionCoefficient=0.5` uses the approximate value discussed
on p.552 of Monaghan (1992). This models artificial thermal diffusion, rather
than a physical conductivity.

Using the symmetric kernel gradient and the same calibrated pair length
\(\bar h\) and regularization \(\eta^2=\epsilon\bar h^2\) as Monaghan viscosity,

\[
 q_i+q_j=g\bar h(\bar c+4|\mu_{ij}|),\qquad
 C_{ij}=-\frac{m_j(q_i+q_j)\,\mathbf r_{ij}\cdot\nabla_i\widetilde W_{ij}}
 {\bar\rho_{ij}(r_{ij}^2+\eta^2)}\ge0,\qquad
 \left(\frac{du_i}{dt}\right)_{\rm cond}=\sum_j C_{ij}(u_j-u_i).
\]

Here \(\mu_{ij}\) is the regularized Monaghan velocity measure. It vanishes
for separating pairs. The sound-speed
part remains active for stationary and separating pairs; constant positive
\(g\) therefore also diffuses smooth thermal gradients. No additional switch
restricts this option to shocks. The sign above follows the diffusion equation
(4.4) of Monaghan (1992) and its Gaussian example (4.6): with our gradient
convention, the printed leading minus in (4.5) multiplying \(u_i-u_j\) would
instead cause antidiffusion.

Derived conversion to stored variables \(E=a^2u\), \(\rho_c=a^3\rho\), and
conformal time \(dt=a\,d\tau\): evaluate \(C_{ij}\) using comoving lengths,
comoving densities, and the stored sound/relative velocities, then use
\(\Lambda_{ij}=C_{ij}/a\) and
\(\dot E_i=\sum_j\Lambda_{ij}(E_j-E_i)\). Each equal-mass pair adds identical
opposite energy increments to both endpoints in each hydro stage,
including inactive coarse neighbors. Each increment is converted separately:

\[
 \dot A_i\big|_{\rm cond}=(\gamma-1)a^{3\gamma-5}
 \rho_{c,i}^{1-\gamma}\dot E_i\big|_{\rm cond}.
\]

Consequently a hot particle's \(A\) can decrease while total thermodynamic
entropy increases. Conduction participates in the same two-stage RK2 update
as velocity and viscous entropy production, with no separate conduction pass
or persistent conduction-rate array. Each stage's immutable snapshot supplies
both sides of every exchange; the final increment is taken from the predictor
rates and applied to the original event baseline.

A derived explicit-diffusion timestep cap is applied in both global and rung
selection: \(\Delta\tau\le1/(2\sum_j\Lambda_{ij})\). It uses the sum of
nonnegative exchange coefficients even if the net energy rate is zero, and
provides a safety factor of two relative to the frozen-density forward-Euler
positivity bound. It supplements the existing hydrodynamic bounds.

## Time integration and cosmological variables

All integrators use chronological midpoint events. Each positive-duration hydro
event advances \(Y=(\mathbf w,A)\) with explicit-midpoint RK2:

\[
 k_1=F_r(Y^0),\qquad Y^*=Y^0+\frac{h_r}{2}k_1,\qquad
 k_2=F_r(Y^*),\qquad Y^+=Y^0+h_r k_2.
\]

This is the default and only hydro evolution scheme; no selector retains the
old one-call Euler kick. A zero-duration force/timestep estimate needs only one
evaluation and does not advance the state. Positions and scale factor stay at
the event's physical midpoint during both RK stages; the predictor is an
operator-stage state, not a future position or a separately advanced clock.

Each pair belongs to \(r=\max(r_i,r_j)\), with interval
\(h_r=T2^{-r}\). It is evaluated once per RK stage at that rung's events;
equal-rung pairs have one owner selected by rank and gas index. Both endpoints
receive the same impulse with opposite signs, including an inactive coarse
neighbor. Every pair in a stage reads the same immutable source state across
all ranks. Local and remote contributions are summed before publishing any
predictor or final state. The corrector restores each recipient's saved
baseline and adds the full second-stage increment; it does not add a full kick
on top of the predictor. Directed viscous heating and conductive entropy rates
are accumulated separately for the two endpoints.

For a coarse interval \(H\) and fine interval \(H/2\), the chronological
left-to-right composition is

\[
 D(H/4)\ K_f(H/2)\ D(H/4)\ K_c(H)\ D(H/4)\ K_f(H/2)\ D(H/4).
\]

Fine pairs run at \(H/4\) and \(3H/4\); coarse pairs run at \(H/2\).
Fine recoil reaches the coarse live state at each fine event and remains
present in later events. The driver orders coincident gravity operations as
closing gravity kick, hydro event, opening gravity kick. Gravity never calls
SPH. Its individual/global/hierarchical scheduling remains separate from the
hydro rung partition.

One acceleration vector and one entropy-rate scalar per gas particle are
event-local scratch, with sparse original-state records for reciprocal
recipients. They are released before sorting, migration, checkpoint or output;
there are no persistent force-rate arrays or per-particle predictor clocks.
Internal momentum is conserved up to accumulation and float-storage roundoff.
Finite-step total energy is not exactly conserved, and external forces have
no reciprocal gas endpoint.

With fixed rung/support topology and smooth, stage-consistent coefficients,
the RK2 event and palindromic drift/rung composition have second-order temporal
accuracy. The published alpha projections, Morris–Monaghan endpoint update,
adaptive rung/contact changes and source splitting have separate limitations;
the two-call hydro formula does not establish second-order accuracy for every
such coupled calculation. See the [full derivation and implementation contract](sph_rk2_time_integration.md),
including the alternative nested endpoint construction and focused validation
results.

Enabled cooling and star formation run once at each gas particle's own completed
interval endpoint, after the second drift and before the next gravity boundary.
They update canonical entropy using the endpoint density. Star conversion
preserves mass, position, velocity and old rung; ordinary gravity handles the
new collisionless particle. Sources use first-order splitting, and an all-gas
zero-kick estimate refreshes timestep bounds afterward without another viscosity
update. See [star formation](star_formation.md) and
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
the already-open interval. Viscosity uses that live rung; it stores no additional
rung byte.

The Courant-viscous limit uses

\[
 \Delta t_{\rm cv}=C_{\rm CFL}\min_i
 \frac{h_i}{c_i+0.6[\alpha c_i+\beta\max_j(-\mu_{ij},0)]}.
\]

The code interprets two apparent issues in the printed formulas of Monaghan (1992),
section 10.3, p. 563:

- The printed force limit \(h/|\mathbf f|\) lacks a square root and has dimensions
  of time squared. The implemented acceleration limit scales as
  \(\sqrt{h/|\mathbf a|}\).
- The printed viscous limit uses \(\max\mu\), although equation 4.2 of
  Monaghan (1992) makes \(\mu\) negative for approaching particles. The
  implementation uses the nonnegative compression magnitude \(\max(-\mu,0)\).

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
Monaghan (1992), \(d\mathbf w/d\tau \mathrel{-}= \lambda_{\rm d}\mathbf w\), where
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
repository requirements; the equations cited from Monaghan (1992) are
noncosmological.

## Cosmological viscosity conventions

Every viscosity sensor uses peculiar gradients: the Morris–Monaghan compression
source and the Balsara factor. With physical Hubble rate \(H=\dot a/a\),

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
approach test. The cited pages are the official published source documentation;
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
| `hydro.viscosityBeta` | `2` | Quadratic coefficient: multiplies \(\bar\alpha_{ij}\) for `morrisAndMonaghan`, constant for `none` |
| `hydro.viscosityEvolution` | `morrisAndMonaghan` | α evolution: `none` or `morrisAndMonaghan` |
| `hydro.viscosityAlphaMin` | `0.1` | Morris–Monaghan positive α floor and initial value |
| `hydro.viscosityDecay` | `0.2` | Morris–Monaghan decay coefficient \(\mathcal C_1\) |
| `hydro.conductionCoefficient` | `0` | Artificial thermal diffusion coefficient g; zero disables it |
| `hydro.viscosityEpsilon` | `0.01` | Denominator coefficient multiplying \(\bar\ell^2\) |
| `hydro.neighborNumber` | `42` | Kernel-weighted support-sphere count |
| `compute.sphBucketSize` | `0` (automatic) | SPH tree leaf limit; automatic sizing uses `floor(8/pi * hydro.neighborNumber)`, giving `106` at the default neighbor target |
| `hydro.minSmoothingLength` | `1e-6` | Minimum \(H\), in initial interparticle spacings |
| `hydro.maxSmoothingLength` | `1000` | Requested maximum \(H\), in initial interparticle spacings; capped at half the box |
| `hydro.centralEffectiveRadius` | `0.0054772255750516604` | Imposed central-force effective radius in box units; uses the gravity kernel |
| `hydro.centralSmoothingLength` | derived | Alternative central-force support radius in box units; uses the gravity kernel |
| `hydro.centralSoftening` | deprecated | Deprecated alias for `hydro.centralEffectiveRadius` with the same value and units; warns when set and is omitted from help and output metadata |
| `hydro.cfl` | `0.22821773229381923` | Courant-viscous coefficient using the effective radius \(h=2\sigma\); also controls the calibrated fractional-energy bound |

For the central-force radius pair, an explicit effective-radius key wins over the
support-radius key, including across CLI and INI inputs, with a warning naming
the ignored key. For the central force, precedence is `centralEffectiveRadius`,
then the deprecated `centralSoftening` alias, then `centralSmoothingLength`. The command line
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
are rejected before particle data are loaded. Bit 25 identifies gas state without
the retired viscosity models. Older SPH checkpoints lack it and are rejected
before their longer viscosity parameter block is read. The gas payload has a
separate version: base 140000 omits serialized force-rate arrays and stores
binary16 alpha for Morris–Monaghan, serialized as `H` in two bytes, matching
memory and migration. There is no previous-rung byte, sample timestamp, exponent
or pending diagnostic payload. The three former byte arrays for convergence, semiactivity
and proposed rung are replaced by one `SphParticleFlags` byte per gas particle:
bit 0 is convergence, bit 1 semiactivity, and bits 2–7 encode the proposed rung.
Codes 0–15 are current valid rungs. There is no unset sentinel; all other
codes, including 63, are rejected on load. The byte precedes viscosity data
after density and also migrates as one byte, saving two bytes per gas particle
in memory and these payloads. Explicit setters update the packed fields because
an individual bit cannot be returned as a mutable reference.

Enabled conduction adds feature flag bit 25 and one double coefficient after the
base SPH parameter block. Restarts require the same coefficient and enabled state.
With conduction disabled, the existing checkpoint layout is unchanged.

Balsara factors remain transient. The existing physics and viscosity suffixes
remain; earlier gas payloads, including every base below 140000, are rejected.
See the
[pair-midpoint design](sph_pair_midpoint_plan.md) and
[SPH octree design](sph_octree_plan.md) for the retained tree. The
[RK2 contract](sph_rk2_time_integration.md) supersedes the earlier one-call
hydro kick description.
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
The retired `hydro.viscosityAlphaMax`, `hydro.viscosityDecayCD`,
`hydro.viscosityAlphaMaxPrice`, `hydro.viscosityDecayPrice`,
`hydro.viscosityAlphaMaxHu` and `hydro.viscosityDecayHu` keys are also rejected.
The `run.scenario=polytropicStar` scenario is a self-gravitating gas sphere; it does
not use stellar particles or star formation.

## Validation scope

Two-call RK2 revision (November 4, 2026): Release CPU and CUDA applications
built and linked; all 46 focused CPU and 31 enabled CUDA SPH unit tests passed
(one pre-existing CUDA test disabled). All 14 focused application checks passed,
covering temporal refinement, CPU/CUDA gravity, two-rank reciprocal migration,
hierarchical CPU/MPI/CUDA evolution, moving self-gravitating gas, global
integration/restart, source timing, retained octrees and Sod. Independent
standalone CPU checks also passed. Tests demonstrate the smooth fixed-coefficient
hydro update's temporal order, immutable shared coarse sources and conservative
recoil. They do not establish second order for instantaneous switch projections
or first-order chemistry/star-formation splitting. See the
[validation table and numerical limits](sph_rk2_time_integration.md#8-implementation-and-validation).
The earlier results below retain their stated historical scope. No full
repository suite or performance-scaling study was run for this revision.

Artificial-conduction addition (November 4, 2026): the Release CPU application
build/link, 46 CPU SPH tests, and all 31 enabled CUDA SPH regressions passed
(one pre-existing CUDA test disabled). The new shared fixture checks the scalar
formula of Monaghan (1992), hot-to-cold transfer, unequal densities/supports,
cosmological scaling, unchanged pressure acceleration, uniform-temperature
cancellation, conductive timestep scaling, mixed-rung pair ownership, energy
conservation, positive thermal state, and increasing total thermodynamic entropy.
A separate CPU/CUDA parity case includes variable viscosity and SIMD tails. Focused CLI
checks cover the zero default, positive values, INI/CLI precedence and invalid
inputs. The existing restart fixture passes with g=0.3, rejects changed/disabled
conduction, and retains its other viscosity/pressure restart checks. Three-step
576-particle CPU Sod runs with g=0.5 completed under traditional and global
integration with finite positive entropy. This is focused validation; no full
repository suite, conduction convergence study, MPI conduction evolution, or
CUDA application evolution was run.


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
finite-difference derivatives, the \(H=2h_M\) convention of Monaghan (1992),
viscosity signs and coefficients, regularization, and invariance under length-unit
changes and velocity boosts. Pair tests combine variational pressure with Monaghan
viscosity and use unequal pressures,
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
- `balsara.hpp`: the shear limiter, always on for Morris–Monaghan and optional for constant α.

A two-byte bulk viscosity coefficient array stores Morris–Monaghan alpha.
Balsara storage is one byte per gas particle; evaluation and staged sources use
float. Morris–Monaghan therefore uses three bytes per gas particle.
Constant viscosity allocates no evolving alpha. RK2 also uses one event-local
entropy-rate scalar, plus sparse baseline records; these are not part of the
persistent coefficient storage above.
Alpha participates in reordering, migration and checkpoints.
Balsara factors remain transient and are recomputed after restart or migration.
Their lifecycle remains in common particle code.
The shared density and hydro passes call these policies without duplicating
neighbor walks. Standalone policy tests use the same author-directory organization.
