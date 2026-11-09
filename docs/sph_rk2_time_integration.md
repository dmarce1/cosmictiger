# Conservative SPH across time rungs: two-call RK2

**Status: implemented and focused validation passed, November 4, 2026.**
The default and only positive-duration SPH update is the chronological midpoint
construction below. It uses two hydro evaluations and one entropy-rate scalar per
gas particle, plus temporary affected-particle baseline and exchange records. No
event scratch enters migration or checkpoints.
SPH hydro is orchestrated by the driver, never called from a gravity kick.
The published Morris–Monaghan endpoint update is retained; its accuracy
limitation is explicit in section 7. Section 9 explains the alternative
nested endpoint construction and its additional state requirements.

The essential rule is that an interacting pair shares one integration clock:
the finer of its two particle rungs. At a pair event, both endpoints receive the
same momentum transfer with opposite signs, including an endpoint whose own
coarser interval has not closed. All pairs in one event read the same immutable
stage state. Their contributions are summed before any recipient is advanced.

## 1. Stored variables and the semidiscrete equations

Let physical time be \(t\), conformal time be \(\tau\), and the scale factor be
\(a(t)\). Production stores comoving position \(\mathbf x\), canonical peculiar
velocity \(\mathbf w\), comoving density \(\rho\), and entropy variable \(A\):

\[
dt=a\,d\tau,\qquad
\mathbf r=a\mathbf x,\qquad
\mathbf w=a\mathbf v_{\rm pec}=a^2\frac{d\mathbf x}{dt},\qquad
\frac{d\mathbf x}{d\tau}=\frac{\mathbf w}{a},
\]
\[
\rho=a^3\rho_{\rm phys},\qquad
A=\frac{P_{\rm phys}}{\rho_{\rm phys}^{\gamma}},\qquad
E=a^2u_{\rm phys}
  =A\,\frac{\rho^{\gamma-1}a^{5-3\gamma}}{\gamma-1}.
\tag{D1}
\]

Define \(\chi_i=E_i/A_i\), stored pressure
\(\mathcal P_i=(\gamma-1)\rho_iE_i=a^5P_{{\rm phys},i}\), and stored sound speed
\(c_i=\sqrt{\gamma(\gamma-1)E_i}=a c_{{\rm phys},i}\).
Equation (D1) is the local ideal-gas conversion implemented in
[`thermodynamics.hpp`](../cosmictiger/sph/thermodynamics.hpp), particularly
`sph_entropy::energyPerEntropy`, `storedEnergy`, and `fromStoredEnergy`.
For \(\gamma=5/3\), the explicit scale-factor power is zero. Adiabatic compression
and homogeneous expansion change the reconstructed \(E\), while leaving \(A\)
unchanged. No reversible pressure-work term is added to the entropy kick.

The following equations specialize to production's common gas mass \(m\).
The compact kernel support is denoted \(H_i\); time intervals below use
\(h_r\) or \(\Delta\), to keep the two lengths distinct. Set

\[
\mathbf x_{ij}=\mathbf x_i-\mathbf x_j,\quad
\mathbf w_{ij}=\mathbf w_i-\mathbf w_j,\quad
\mathbf g_i=\nabla_i W(|\mathbf x_{ij}|,H_i),\quad
\mathbf g_j=\nabla_i W(|\mathbf x_{ij}|,H_j),\quad
\overline{\mathbf g}_{ij}=\tfrac12(\mathbf g_i+\mathbf g_j).
\]

The density-based variational pressure force uses

\[
Q_{ij}=\frac{\mathcal P_i}{\rho_i^2}(m-\zeta_i),\qquad
\mathbf f^{P}_{ij}
 =-\frac1a\left(Q_{ij}\mathbf g_i+Q_{ji}\mathbf g_j\right),\qquad
\mathbf f^{P}_{ji}=-\mathbf f^{P}_{ij}.
\tag{D2}
\]

Here \(\mathbf f_{ij}\) is the contribution to \(d\mathbf w_i/d\tau\), so it
is acceleration in the stored variables, not momentum per unit time. The
grad-support correction is \(m-\zeta_i=m/\Omega_i\) for free equal-mass
supports and \(\zeta_i=0\) at a held support bound. This is the production
specialization of [Hopkins (2013), equations (12)–(13)](https://arxiv.org/abs/1206.5006)
and [Price (2012), equations (27), (30), and (35)](https://arxiv.org/abs/1012.1885).
The code's directed coefficients have already been divided by the recipient
mass; see [`variable_smoothing_lengths.hpp`](../cosmictiger/sph/variable_smoothing_lengths.hpp).

For the Monaghan viscosity branch, define the physical approach velocity in
stored units and its regularized scalar:

\[
\mathbf z_{ij}=\mathbf w_{ij}+a\dot a\,\mathbf x_{ij},\qquad
\mu_{ij}=\min\left(0,
 \frac{\bar\ell_{ij}\,\mathbf z_{ij}\cdot\mathbf x_{ij}}
 {|\mathbf x_{ij}|^2+\epsilon\bar\ell_{ij}^{\,2}}\right),
\]
\[
\Pi_{ij}=\frac{-\bar\alpha_{ij}\bar c_{ij}\mu_{ij}
                         +\beta_{ij}\mu_{ij}^{2}}{\bar\rho_{ij}},
\qquad
\mathbf f^{V}_{ij}=-\frac{m}{a}\Pi_{ij}\overline{\mathbf g}_{ij},
\]
\[
q^V_{ij}=q^V_{ji}
 =\frac{m}{2a}\Pi_{ij}\,
                  \mathbf z_{ij}\cdot\overline{\mathbf g}_{ij}.
\tag{D3}
\]

\(\bar\ell\) is the viscosity length obtained from the stored supports using
the configured kernel normalization, \(\alpha\) is the bulk viscosity
coefficient, and \(\beta\) is the von Neumann–Richtmyer coefficient. The optional
pair-symmetric Balsara factor multiplies both viscous force and heating.
These are the comoving conversions of [Monaghan (1992), section 4](https://articles.adsabs.harvard.edu/pdf/1992ARA%26A..30..543M),
implemented in [`monaghan.hpp`](../cosmictiger/sph/viscosity/monaghan.hpp).

For artificial conduction, the current equal-mass exchange has the form

\[
\kappa_{ij}=-\frac{m}{a}
 \frac{g\bar\ell_{ij}(\bar c_{ij}-4\mu_{ij})}
 {\bar\rho_{ij}(|\mathbf x_{ij}|^2+\epsilon\bar\ell_{ij}^{\,2})}
 \mathbf x_{ij}\cdot\overline{\mathbf g}_{ij}\ \geq 0,
\]
\[
q^C_{ij}=\kappa_{ij}(E_j-E_i),\qquad
q^C_{ji}=-q^C_{ij},\qquad
b_{ij}=\frac{q^V_{ij}+q^C_{ij}}{\chi_i}.
\tag{D4}
\]

The exact gradient/sign convention is documented alongside
[Monaghan (1992), equations (4.4)–(4.8)](https://articles.adsabs.harvard.edu/pdf/1992ARA%26A..30..543M)
in [`conduction.hpp`](../cosmictiger/sph/conduction.hpp).
The pair entropy increments generally are not opposite:
\(\chi_i b_{ij}+\chi_j b_{ji}=q^V_{ij}+q^V_{ji}\).
Conduction transfers equal energy; entropy storage converts each endpoint
using its own \(\chi\). Viscosity produces heat at both endpoints.

The complete semidiscrete system is therefore

\[
\mathbf x'_i=\mathbf w_i/a,\qquad
\mathbf w'_i=\mathbf F^G_i+\mathbf F^{\rm ext}_i
                 +\sum_j(\mathbf f^P_{ij}+\mathbf f^V_{ij}),\qquad
A'_i=\sum_j b_{ij}+S_i^{\rm other},
\tag{D5}
\]

where primes mean \(d/d\tau\). Other source terms, such as cooling, have their
own integration and conservation contracts; two hydro stages do not establish
their temporal accuracy.

## 2. Rung clocks belong to pairs as well as particles

For a base conformal interval \(\Delta\tau_0\), a particle on rung \(r_i\) has

\[
h_i=\Delta\tau_0\,2^{-r_i}.
\]

For an unordered interacting pair,

\[
r_{ij}=\max(r_i,r_j),\qquad h_{ij}=\min(h_i,h_j),\qquad
\tau_{ij,n}=\tau_0+(n+\tfrac12)h_{ij}.
\tag{D6}
\]

An event on exact rung \(r\) integrates only pairs with \(r_{ij}=r\), over
the duration \(h_r\). A finer neighbor therefore owns the whole fine/coarse
pair contribution. The same pair contributes nothing to the coarser event.
For equal rungs, the lexicographically smaller `(rank, gas index)` owns the
pair. These identities are valid only while the event prevents particle
migration and reordering. See [`pair_kick.hpp`](../cosmictiger/sph/pair_kick.hpp).

Define \(K_r\) as the vector field containing all such pair contributions,
plus each particle's nongravitational external acceleration on its own rung.
The vector field includes updates to *both* endpoints. Thus \(K_r\) is not
an operation restricted to writing particles whose own rung equals \(r\).

Let \(C\) have coarse interval \(\Delta\), and let its neighbors \(L,R\)
have fine interval \(\Delta/2\). Both \((L,C)\) and \((C,R)\) belong to
the fine operator. At \(\Delta/4\) and \(3\Delta/4\), \(C\) receives their
combined recoil and heat. At \(\Delta/2\), the coarse event contains only
coarse/coarse pairs and the coarse particle's own external source, if present.

```text
physical event time      0        Δ/4        Δ/2       3Δ/4        Δ
fine L                   |---------●----------|----------●---------|
coarse C                 |---------↕----------●----------↕---------|
fine R                   |---------●----------|----------●---------|
                                  Kf         Kc         Kf

● on a fine row: the fine particle's own hydro event
↕ on C: reciprocal updates from every participating fine/coarse pair
● on C at Δ/2: only the coarse-owned part of its hydro operator
```

Every particle drifts to the next event time, including coarse particles.
An inactive coarse particle is not held at an old position or velocity until
its own interval closes. At a later event, its live velocity includes all
earlier chronological pair impulses.

Rung selection uses the force, signal-speed, thermal and conduction limits
to obtain a positive candidate \(h_i^{\rm lim}\). In the current hydro kernel,
with effective kernel radius \(\ell_i^{\rm eff}\), Courant coefficient
\(C_{\rm CFL}\), calibrated acceleration coefficient \(\eta_h\), and
\(M_i=\max_j(-\mu_{ij})\), these have the forms

\[
h_i^{\rm CFL}=
 \frac{C_{\rm CFL}a\ell_i^{\rm eff}}
 {c_i+0.6(\alpha_i^{\rm bound}c_i+\beta_i^{\rm bound}M_i)},\qquad
h_i^{\rm acc}=\eta_h\sqrt{\frac{a\ell_i^{\rm eff}}{|\mathbf F_i^{\rm hydro+ext}|}},
\]
\[
h_i^{\rm thermal}=\frac{C_{\rm CFL}}{c_{\rm width}}
                         \frac{E_i}{|E_i^{\prime\,\rm reversible}|},\qquad
h_i^{\rm conduction}=\frac{1}{2\sum_j\kappa_{ij}},\qquad
h_i^{\rm lim}=\min(h^{\rm cap},h_i^{\rm CFL},h_i^{\rm acc},
                         h_i^{\rm thermal},h_i^{\rm conduction}).
\tag{D7a}
\]

A vanishing denominator supplies no restriction. \(c_{\rm width}\) is the
kernel viscosity-width conversion, and the two viscosity bounds use the
maximum coefficients encountered for that target. Gravity
has its separate timestep criterion; a hydro event does not fold a scheduled
gravity kick into the hydro acceleration limit. The dyadic proposal is

\[
r_i^{\rm hydro}=\max\left(0,\left\lceil
\log_2\frac{\Delta\tau_0}{h_i^{\rm lim}}\right\rceil\right).
\tag{D7}
\]

The existing diffusion bound is
\(h_i^{\rm lim}\sum_j\kappa_{ij}\leq1/2\); it uses the positive row sum
even when the net conductive rate vanishes. Additional floors and the
touching-neighbor rung-difference limit constrain the proposal. The live rung
and pair ownership stay fixed throughout an event. Commit new rungs only at
the scheduled synchronization boundary, after old intervals close and the
neighbor constraints reach their fixed point. No retroactive reassignment of
an already-applied impulse is permitted. A newly formed contact must be found
at the next applicable stage; the order proof below assumes a smooth contact
graph and does not establish accuracy across arbitrary wakeups or rung changes.

## 3. Chronological drift/kick composition

Let \(D(s)\) drift all particles over a physical conformal-time interval of
length \(s\), with velocity held during that drift:

\[
\mathbf x_i\leftarrow\mathbf x_i+
 \mathbf w_i\int_{\tau}^{\tau+s}\frac{d\sigma}{a(\sigma)}.
\tag{D8}
\]

At \(a=1\), this is \(\mathbf x_i\leftarrow\mathbf x_i+s\mathbf w_i\).
For a coarse interval \(\Delta\) and fine interval \(\Delta/2\), execute
the following sequence **from left to right in time**:

\[
\boxed{
D(\Delta/4)\;
K_f(\Delta/2)\;
D(\Delta/4)\;
K_c(\Delta)\;
D(\Delta/4)\;
K_f(\Delta/2)\;
D(\Delta/4)
}.
\tag{D9}
\]

There are three physical hydro events, at \(\Delta/4,\Delta/2,3\Delta/4\).
The first and last integrate the fine operator; the middle integrates the
coarse operator. Each \(K\) holds positions, scale factor, support and density
at its event geometry while advancing the momentum and entropy variables.
The two RK stages inside a \(K\) are stages of this fixed-geometry operator.
They do not drift the particles to another chronological event, and they do
not advance the global clock a second time.

More explicitly, for finest occupied rung \(R\), define an operator sequence
\(S_r(h_r)\) containing drift and all hydro operators on rungs \(r\) and finer:

\[
S_R(h_R)=D(h_R/2)K_R(h_R)D(h_R/2),
\]
\[
S_r(h_r)=S_{r+1}(h_r/2)K_r(h_r)S_{r+1}(h_r/2).
\tag{D9a}
\]

The written products again mean execute left to right. This recursively
generates increasing physical event times. There is no requirement to execute
a coarse event out of chronological order. A coarse \(K_r(h_r)\) has a large
*operator integration duration*, but it executes once at its physical center;
it does not move the clock backward to that interval's beginning or forward
to its endpoint. Its input is the state created by the preceding operators,
not a claim that every component equals the unsplit physical solution at the
start of a coarse interval.

Source-side density/support
information must be valid at the geometry of each event, including for every
coarse recoil recipient. An old coarse density cannot be treated as current
merely because its rung is inactive.

Explicitly time-dependent factors, including \(a\), must use their proper
event-centered values. Cosmological drifts require the appropriate integral
in (D8), or a second-order quadrature; using half of an asymmetric whole-step
drift factor for both halves is not generally sufficient.

### Coincident gravity boundaries

At a boundary that closes one gravitational interval, hosts a hydro event,
and opens the next gravitational interval, the required order is

\[
\boxed{G_{\rm close}\;\longrightarrow\;K_r\;
       \longrightarrow\;G_{\rm open}}.
\tag{D10}
\]

The close and open gravity durations belong to the completed and newly
selected intervals respectively, typically their half-kick weights. Hydro
reads the state after the closing gravity kick and before the opening kick.
Even if gravity acceleration depends only on position, viscous acceleration
depends on velocity, so moving \(G_{\rm open}\) through \(K_r\) changes its
input. Combining gravity close/open ahead of hydro introduces a leading
commutator error that a second hydro call alone cannot repair. Signed
hierarchical-gravity corrections remain gravity operations; they must not
trigger repeated or negative-duration viscous/conductive hydro events.
The driver must perform this ordering explicitly. The gravity kick may expose
its own closing and opening phases or instantaneous force diagnostics, but it
must never call SPH, dispatch an SPH continuation, or hide a hydro solve inside
a gravity particle loop.

## 4. Two hydro calls at one event

Write \(Y=(\{\mathbf w_i\},\{A_i\})\) and let \(F_r(Y)\) be the selected
operator's complete recipient rates at the event geometry. The implementation
uses explicit midpoint as the sole hydro time integrator:

\[
k_1=F_r(Y^0),\qquad
Y^*=Y^0+\frac{h_r}{2}k_1,
\]
\[
k_2=F_r(Y^*),\qquad
\boxed{Y^+=Y^0+h_r k_2}.
\tag{D11}
\]

These are [Goodman's Numerical Methods II notes, equations (10)–(11)](https://math.nyu.edu/~goodman/teaching/NumericalMethodsII2016/notes/lecture3.pdf),
applied to the event's hydro vector field. The one-call positive-duration
method has been removed from production; there is no runtime selector. A zero-duration
estimate changes no physical state and needs only the force/timestep evaluation
requested by its caller; that diagnostic operation is not an alternative
one-call time integrator.

For each stage \(s\), one owned pair supplies

\[
\delta\mathbf w_i^{(s)}=h_s\mathbf f_{ij}(Y^{(s)}),\qquad
\delta\mathbf w_j^{(s)}=-\delta\mathbf w_i^{(s)},
\]
\[
\delta A_i^{(s)}=h_s b_{ij}(Y^{(s)}),\qquad
\delta A_j^{(s)}=h_s b_{ji}(Y^{(s)}),
\tag{D12}
\]

where \(h_1=h_r/2\) constructs the predictor and \(h_2=h_r\) constructs
the final increment. Stage 2 recomputes pressure, sound speed, viscosity and
conduction using the predicted velocities and entropies, with each predicted
\(E_i\) reconstructed from \(A_i^*\). Supports and densities are unchanged
*within* this fixed-position operator, provided they were correctly prepared
at its start.

The final update starts from \(Y^0\). Adding a full second-stage increment to
\(Y^*\) would retain the provisional half-kick and implement a different,
inconsistent method. The predictor is disposable scratch, including the
predictor of an otherwise inactive coarse recipient.

### Two fine neighbors sharing one coarse recipient

Suppose the stage-one pair forces on coarse particle \(C\) are
\(\mathbf f_{CL}^0\) and \(\mathbf f_{CR}^0\). The predictor must be

\[
\mathbf w_C^*=\mathbf w_C^0+
 \frac{h_f}{2}(\mathbf f_{CL}^0+\mathbf f_{CR}^0),\qquad
A_C^*=A_C^0+\frac{h_f}{2}(b_{CL}^0+b_{CR}^0).
\tag{D13}
\]

Both pairs read \(C^0\) in stage one. Both read this same, fully assembled
\(C^*\) in stage two. Neither pair is allowed to read \(C\) after only its
other neighbor's update has arrived. Stage two finishes with

\[
\mathbf w_C^+=\mathbf w_C^0+
 h_f\left[\mathbf f_{CL}(Y^*)+\mathbf f_{CR}(Y^*)\right],\qquad
A_C^+=A_C^0+h_f[b_{CL}(Y^*)+b_{CR}(Y^*)].
\tag{D14}
\]

This is why independent per-pair predictor/correctors with immediate live
writes are insufficient. The vector field is the sum of incident pair rates;
the shared recipient's predicted state depends on that entire sum.

```text
                 immutable stage state Y⁰ on every rank
                       /                     \
                 pair L–C                 pair C–R
                       \                     /
                  sum all recipient scratch increments
                  finish reverse exchange on every rank
                                |
                 barrier → publish complete predictor Y*
                                |
                  invalidate previous source-stage caches
                       /                     \
                 pair L–C                 pair C–R
                       \                     /
                  sum all recipient scratch increments
                  finish reverse exchange on every rank
                                |
                    barrier → commit Y⁰ + h_f F_f(Y*)
```

The distributed contract is:

1. Freeze membership, ownership, geometry and all stage source fields. Save
   the baseline of every possible recipient, including remote coarse gas.
2. Stage all reads before computing contributions. Accumulate into scratch;
   CPU atomics/CUDA atomics may order sums differently but must not change
   the stage state read by another pair.
3. Finish every local compute and reverse recipient exchange before publishing
   any predictor to the next stage. A rank finishing its own target list is
   not evidence that all contributions to its recipients have arrived.
4. Reconstruct thermal state and invalidate source caches before staging the
   second pass. Stage-two reads must never mix baseline and predicted fields.
5. Finish stage two globally and commit its full increment against the saved
   baseline. Release scratch only after all dependent work completes.

Checkpointing, migration, sorting and restart are permitted at completed event
boundaries. They must not persist or expose the provisional predictor. A failed
event must restore its baseline before further work or terminate without
publishing a checkpoint as though the event completed.

### Storage: one entropy rate, plus event scratch

The added reusable rate storage is one entropy-rate field per gas particle: four
scalar-data bytes as a `float`, before alignment/container overhead. Its numerical
meaning and lifetime must be explicit. A selected-rung hydro rate and an applied
impulse have different meanings even when they have the same shape.

One algebraic possibility stores stage-one rates
\(k_1=(\mathbf a^{(1)},\dot A^{(1)})\), forms a provisional state, and uses

\[
Y^+=Y^*+h_r\left(k_2-\tfrac12 k_1\right)
     =Y^0+h_r k_2
\tag{D11a}
\]

in exact arithmetic. Stage two still must read a frozen predictor. It cannot
read live fields while this correction is being accumulated. If the predictor
has been rounded into float storage, reconstructing the baseline by subtracting
its half increment does not recover the original float exactly. Keeping the
baseline in existing event-local source buffers while publishing a separate
predicted source view avoids that issue, at a temporary-buffer cost that must
be reported. An implementation using (D11a) directly must bound and test its
extra roundoff and its rollback behavior.

The implementation saves the original velocity and entropy, plus the
first-stage proposed rung, in temporary records for affected particles. It
restores velocity and entropy exactly before the corrector, retains the tighter
rung proposal from both stages, then adds the stage-two increment. This avoids subtracting
a rounded predictor. Source snapshots and reciprocal-exchange buffers also occupy
temporary memory; the scalar rate field is not a bound on total event memory.

The selected-rung impulses are kept in reciprocal update records, so no rate
field need retain \(k_1\) for the corrector. The scalar field holds the stage
entropy rate. It is an event-local allocation, released after completion; no
persistent particle, migration or checkpoint fields are added.

## 5. What is conserved, and what second order means

For equal masses, every pair in each stage has

\[
m\delta\mathbf w_i+m\delta\mathbf w_j=0.
\tag{D15}
\]

The predictor and final RK2 state therefore preserve the internal hydro
contribution to total canonical momentum in exact arithmetic, independently
of rung differences and how many pairs share a recipient. The same property
holds for every chronological event. Central pair forces also have zero
internal torque about the consistent pair separation. Floating-point sums and
float particle commits introduce roundoff; external forces can change total
momentum. Unequal gas masses would require opposite *momentum* increments
divided by the two separate masses, rather than the current equal/opposite
velocity increments.

Conduction preserves the pair's stored thermal energy at a fixed geometry:
\(m\chi_i\delta A_i+m\chi_j\delta A_j=0\). Viscous heating compensates
viscous kinetic work at the semidiscrete level in the nonexpanding case.
The pressure/entropy formulation represents reversible thermal changes
through the position-dependent density and support relation. These facts do
not make explicit RK2 exactly energy conserving. In general,

\[
\tfrac12m|\mathbf w+\delta\mathbf w|^2
 -\tfrac12m|\mathbf w|^2
 =m\mathbf w\cdot\delta\mathbf w
  +\tfrac12m|\delta\mathbf w|^2,
\]

and a finite explicit heat update does not enforce an exact discrete
kinetic-plus-thermal identity. At \(a\ne1\), cosmological expansion also
changes the appropriate energy balance. The claim is pair momentum
conservation up to roundoff and second-order temporal error under the stated
conditions, not exact total-energy conservation, unconditional positivity,
time reversibility or symplecticity.

### Local derivation of order

Assume smooth autonomous \(F_r\), fixed rung assignment, and consistent
event data. Taylor expansion of the second stage in (D11) gives

\[
F_r(Y^*)=F_r(Y^0)+\frac{h_r}{2}F'_r(Y^0)F_r(Y^0)+O(h_r^2),
\]
\[
Y^+=Y^0+h_rF_r(Y^0)
       +\frac{h_r^2}{2}F'_r(Y^0)F_r(Y^0)+O(h_r^3).
\tag{D16}
\]

This matches the exact subflow through degree two. In the three-particle
example, the Jacobian product contains cross terms from both neighbors of
\(C\); the assembled predictor in (D13) is what retains those terms.
Using a stale coarse predictor, or a partially updated coarse state, changes
this expansion.

For exact autonomous drift and kick subflows, (D9) is a consistent palindromic
composition: total drift duration is \(\Delta\), total fine-kick duration is
\(\Delta\), and total coarse-kick duration is \(\Delta\). Its logarithm
has no quadratic term, so it agrees with the flow of \(D+K_f+K_c\) through
\(O(\Delta^2)\). Replacing each exact kick subflow by (D16) introduces only
\(O(h_r^3)\) local defects. With a fixed finite rung hierarchy, bounded smooth
derivatives, and stable evolution, the composed local defect is
\(O(\Delta^3)\) and the global time error is \(O(\Delta^2)\).

This proof is a local derivation of the proposed composition. Explicit
midpoint itself is not an exactly reversible subflow; matching its expansion
to second order is sufficient for this argument. In a time-dependent
background, symmetric event-time evaluation or an equivalent augmented-time
formulation must also retain the correct second-order time terms.

The proof does not cover a first-order support/source predictor, a delayed
viscosity coefficient, a nonsmooth switch projection, an inconsistent gravity
ordering, discontinuous rung reassignment, unresolved shocks, or finite
precision below the refinement-error floor. Such cases need their own
analysis and measurements.

## 6. Balsara limiter as stage data

The velocity-dependent Balsara limiter is stage data: its baseline value needs
the baseline velocity gradient, and its midpoint value needs the predictor
velocity gradient. Holding the baseline limiter fixed in the second force call
omits its velocity derivative from the RK2 expansion. Refreshing this limiter
does not itself advance the Morris–Monaghan coefficient.

## 7. Retained Morris–Monaghan update and its limitation

The selected policy is to preserve the published Morris–Monaghan relaxation
rather than replace it with an invented stage-consistent ODE. Its helper uses
backward Euler for the source/decay equation, once at each particle's completed
own interval endpoint, so both hydro stages read the coefficient from the last
endpoint.

Retaining this rule means the unqualified claim “the full adaptive-viscosity
system is second order for every smooth flow and shock” is unavailable.
Backward-Euler relaxation is first order even on a smooth decay branch. The
rigorous RK2 claim is for the smooth hydro evolution with fixed or suitably
stage-consistent coefficients, and the retained endpoint update remains a
documented limitation of the coupled calculation. It preserves the published
switch form and particle clocks; it does not extend the smooth, fixed-coefficient
order proof to the adaptive system.

## 8. Implementation and validation

The implementation is organized as follows:

- [`sph.cpp`](../src/sph/sph.cpp) freezes sources on all localities, evaluates
  and exchanges reciprocal updates, commits the predictor, then restores the
  original affected-particle values before the corrector. Zero-duration calls
  only estimate rates and timestep bounds. Both stage timestep proposals are retained.
- [`pair_kick.hpp`](../cosmictiger/sph/pair_kick.hpp) owns each unordered pair
  on its finer rung, with a rank/index tie break. CPU and CUDA kernels apply
  the same due-pair filter.
- [`driver.cpp`](../src/core/driver.cpp) and
  [`scenario_driver.cpp`](../src/core/scenario_driver.cpp) place coincident
  hydro between the gravity closing and opening actions. Deferred openings
  are sealed to base-particle identities before gas sorting.
- CPU/CUDA prehydro kernels rebuild the Balsara factor from the stage velocities
  before both hydro evaluations, and advance Morris–Monaghan alpha only at
  completed particle endpoints.

The persistent particle layout is unchanged. Completed-event checkpoints need
no RK stage data; a restarted run uses the new two-call method. There is no
legacy hydro integrator selector.

The production temporal test in
[`tests/sph/tree.cpp`](../tests/sph/tree.cpp), `verifySphTemporalOrder`, covers
analytic damping, a viscous mixed-rung pair, the pair with conduction, and a
three-particle chain with noncommuting coarse/coarse and coarse/fine operators.
The chain executes fine/coarse/fine composition at fixed geometry and compares
refined solutions. These cases isolate temporal integration with fixed viscosity
coefficients; they do not establish accuracy of the Morris–Monaghan endpoint update.

Kernel tests cover immutable shared-coarse reads under all 24 traversal orders.
Application gravity tests cover deferred signed openings, remote reciprocal
updates, and an empty root locality.
Existing application tests additionally exercise moving self-gravitating gas,
drift/kick refinement, retained trees, migration and completed-event restart.

Validation on November 4, 2026:

| Scope | Result |
| --- | --- |
| Release CPU and CUDA applications | Both built and linked |
| Focused CPU SPH unit suite | 46/46 passed |
| Focused CUDA SPH unit suite | 31/31 enabled tests passed; one pre-existing disabled test |
| Production temporal refinement (`sph_tree`) | Passed damping, viscosity, conduction and noncommuting mixed-rung chain cases |
| Gravity and reciprocal migration (`sph_gravity_cpu`, `sph_gravity_cuda`, `sph_migration_mpi`) | All passed, including complete-force sensors and exact state/order restoration |
| Hierarchical application (CPU, two-rank MPI, CUDA) | All passed |
| Moving self-gravitating gas (`sph_hierarchical_gravity`) | Passed; position errors 8.08e-8, 1.96e-8, 4.38e-9 over successive refinements |
| Global integration/restart and source timing (CPU and two-rank MPI) | All passed |
| Retained octree/migration application and Sod smoke | Both passed |
| SPH without gravity capability | Driver and scenario-driver objects compiled; whole application not rebuilt for this gate |

These are 14 distinct focused application checks, all passed on the CUDA-enabled
application using their specified CPU/CUDA backends. The standalone CPU build
also passed the temporal, source-timing and global integration/restart checks.
The viscous/conductive chain's last two refinements reduced velocity error by
4.66 and 4.50, and entropy error by 5.27 and 4.70, consistent with second order
before the float roundoff floor. The self-gravity refinement exercises moving
geometry and drift/kick composition separately from that fixed-geometry chain.

Restart checks require exact clocks and nonvelocity particle fields. Hydro
velocity comparisons permit four float epsilons times the maximum velocity
scale of that component; the observed discrepancy prompting this allowance
was at most 1.78e-15 in nearly zero transverse velocities. Nonfinite values,
larger perturbations and changed position bits are rejected. Collisionless
restart remains byte-exact. The full repository suite, performance scaling,
and a universal adaptive-switch convergence claim are outside this validation.

## 9. Alternative: nested endpoint predictors/correctors

Consider the following structure for a coarse interval
\(\Delta\): coarse predictor; two fine predictor/corrector steps of duration
\(\Delta/2\); coarse corrector. A precise second-order version is possible
using frozen slow forcing, as derived below. It remains an alternative to the
chronological midpoint split, not a claim about the implementation.

For a single smooth operator, an endpoint predictor/corrector can use

\[
Y^{p}=Y^0+\Delta F(Y^0),\qquad
Y^+=Y^0+\frac{\Delta}{2}\left[F(Y^0)+F(Y^{p})\right].
\tag{D20}
\]

Nesting that formula around different-rung pair exchanges needs additional
definitions. A coarse endpoint predictor at \(\tau+\Delta\) is a source
prediction for that time; it must not overwrite the coarse particle's live
chronological state while fine events still occur at earlier times. Every fine
stage needs a coarse source prediction at *its own* time, with a defined
interpolation/prediction rule for position, velocity, entropy and density.
Both endpoint rates must include consistent cross-rung coupling.

Let \(R_{C,f}\) be the sum of all reciprocal fine-owned increments actually
delivered to coarse particle \(C\) during its interval. A necessary final
accounting identity is

\[
Y_C^+=Y_C^0+I_{C,c}+R_{C,f},
\tag{D21}
\]

where \(I_{C,c}\) contains only the coarse-owned operator's contribution.
For a trapezoidal coarse operator this would be
\(I_{C,c}=\Delta[F_c^0+F_c^p]/2\), with its endpoint state still to be
defined consistently with the fine evolution. Replacing the live coarse state
with \(Y_C^0+I_{C,c}\) erases fine recoil and heat. Including fine-owned pair
rates in \(I_{C,c}\) as well as \(R_{C,f}\) double counts them. Likewise, if a
source predictor already contains an estimated fine contribution, adding the
actual fine contribution requires replacing that estimate, not retaining both.

### Frozen slow forcing: a complete local construction

Split the autonomous coupled system into fast and slow vector fields,
\(Y'=F(Y)+S(Y)\), with \(S\) containing the coarse-owned pair rates. At
\(Y^0\), evaluate \(S_0=S(Y^0)\). During the fine integration keep this
vector fixed and integrate

\[
Z'=F(Z)+S_0,\qquad Z(0)=Y^0
\tag{D22}
\]

over \(\Delta\), using two second-order substeps of duration \(\Delta/2\).
Then evaluate the slow field at the fine trial endpoint and apply only its
endpoint correction:

\[
\boxed{Y^+=Z_\Delta+
 \frac{\Delta}{2}\bigl[S(Z_\Delta)-S_0\bigr]}.
\tag{D23}
\]

Here “coarse predictor” means a held forcing supplied to every fine stage;
it does not mean publishing a future coarse position or velocity before the
fine evolution reaches that time. Equation (D23) corrects the accumulated
trial state in place, so every reciprocal fine contribution remains present.
The \(-S_0\) term replaces half of the already-integrated frozen slow forcing
with the final slow evaluation. Both evaluations of a slow pair must still
apply identical and opposite momentum transfers to its two endpoints.

For smooth fields, second-order integration of (D22) gives

\[
Z_\Delta=Y^0+\Delta(F_0+S_0)
 +\frac{\Delta^2}{2}F'_0(F_0+S_0)+O(\Delta^3),
\]
\[
\frac{\Delta}{2}[S(Z_\Delta)-S_0]
 =\frac{\Delta^2}{2}S'_0(F_0+S_0)+O(\Delta^3).
\]

Consequently

\[
Y^+=Y^0+\Delta(F_0+S_0)
 +\frac{\Delta^2}{2}(F'_0+S'_0)(F_0+S_0)+O(\Delta^3),
\tag{D24}
\]

which includes the fast/slow cross derivatives and agrees with the complete
system through degree two. This is a local derivation; the same smoothness,
stability, time-dependent-background and switch limitations stated earlier
apply. Every fine stage must use the held \(S_0\) plus the entire current fast
field, with immutable stage snapshots across neighbors and ranks.

With more than two levels, each recursive fine problem inherits the sum of
its ancestors' held forcings. A single per-particle acceleration vector plus
entropy-rate scalar can hold that *aggregate* forcing, but endpoint closure
also needs the corresponding ancestor's original \(S_0\) to form its own
subtraction in (D23). Overwriting the same two fields at each level loses this
information. An implementation needs a recursion stack with sparse initial-rate
records for the affected recipients, or an equivalent representation that
preserves the separate ancestor contributions. This is additional temporary
state beyond two universally reusable scalars/vector fields.

A checkpoint taken while ancestor intervals remain open must either serialize
that open stack, its held forcing and original-rate records, or be deferred
until all relevant intervals close. Restarting only positions, velocities,
entropy and the current aggregate forcing cannot reconstruct every pending
coarse correction. Thus the endpoint alternative has a valid local order
construction, together with concrete storage and restart obligations. The
chronological midpoint split in sections 2–5 avoids an open nested correction
stack across its completed event boundaries.
