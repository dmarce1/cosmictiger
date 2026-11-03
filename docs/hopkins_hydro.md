# Hopkins meshless hydrodynamics

Select `hydro.method=sph|mfm|mfv` in an INI file or on the command line.
`sph` remains the default and retains its existing numerical algorithms.
The MFM and MFV paths share the gas particle infrastructure and kernel options.
They are an optional build capability, disabled by default. Enable them with
`-DCOSMICTIGER_ENABLE_HOPKINS=ON`, `-DCOSMICTIGER_ENABLE_SPH=ON`, and
`-DCOSMICTIGER_POSITION_TYPE=fixed64`. CMake rejects Hopkins without SPH support
or Fixed64 positions. With Hopkins disabled, its driver and sound-wave scenario
are excluded from the build, and active MFM/MFV selections are rejected at startup.

The formulation follows Philip F. Hopkins, *A new class of accurate, mesh-free
hydrodynamic simulation methods*, MNRAS **450**, 53–110 (2015),
[doi:10.1093/mnras/stv195](https://doi.org/10.1093/mnras/stv195),
[arXiv:1409.7395](https://arxiv.org/html/1409.7395v2).
Equation numbers below refer to the arXiv version. This is CosmicTiger's
implementation of that formulation, not vendored GIZMO code or a complete
implementation of GIZMO's physics and numerical options.

## Discretization

For displacement `r_ij = x_j - x_i`, the local number density and volume are
`omega_i = sum_j W(r_ij, H_i)` and `V_i = 1 / omega_i`, including the self term.
The density is `rho_i = m_i / V_i`. Unlike SPH density summation, the volume
partition is independent of neighboring particle masses (equations 18–22).

The weighted moment matrix is
`E_i = sum_j (r_ij tensor r_ij) W(r_ij,H_i) / omega_i`.
Its inverse `B_i` gives the least-squares gradient
`grad f_i = B_i sum_j (f_j - f_i) r_ij W(r_ij,H_i) / omega_i`
(equations 24–26). With nonsingular geometry this reproduces linear fields.

The effective oriented face is
`A_ij = V_i B_i r_ij W(r_ij,H_i)/omega_i + V_j B_j r_ij W(r_ij,H_j)/omega_j`.
Pair fluxes update integrated mass, momentum, and total energy with opposite
signs, using `dQ_i/dt = -sum_j F_ij dot A_ij` (equations 30–35).

Slope-limited primitive states are reconstructed to the kernel-weighted
quadrature point and predicted by half a timestep before the Riemann solve.
The midpoint flux also predicts the effective face vector from the moving
volume/moment/support geometry and advances the face velocity with hydrodynamic
and external acceleration. If the predicted face is nonfinite or reverses
orientation, as can occur when a pair leaves the compact kernel support, that
pair uses a conservative first-order flux for the interval.
The MFV face follows the interpolated particle velocity; MFM additionally
moves the face at the Riemann zero-mass-flux speed, making its mass flux zero
(sections 2.4 and 2.6). The common Riemann flux supplies shock dissipation;
the SPH artificial-viscosity controls apply to the SPH method.

The slope-limiter parameters are fixed in the implementation, with no command-line
or INI controls: the global limiter uses beta bounds 1 and 2 and condition-number
scale 100, and the pair limiter uses `psi_1=0.5`, `psi_2=0.25` (Appendix B).
The current geometry acceptance bound is also 100, so accepted gradients use
the upper beta bound of 2.

The numerical primitives are in `cosmictiger/hydro/hopkins/hopkins.hpp`, with the shared
scalar/SIMD reconstruction and HLLC/HLL arithmetic in
`cosmictiger/hydro/hopkins/hopkins_simd.hpp`. HLLC uses the Davis extrema of the two
one-sided acoustic speeds; a rejected HLLC state falls back to HLL with those
same speeds and reconstructed inputs. For face vector \(\boldsymbol A\) and input frame
velocity \(\boldsymbol w\), define \(A=|\boldsymbol A|\),
\(\boldsymbol n=\boldsymbol A/A\),
\(\boldsymbol v_K=\boldsymbol v_K^{\rm lab}-\boldsymbol w\), and
\(u_K=\boldsymbol v_K\cdot\boldsymbol n\), for \(K=L,R\).
A zero-area face returns zero flux. All following states and fluxes are in
this moving frame until the final boost:

\[
 E_K=\frac{P_K}{\gamma-1}+\frac12\rho_K|\boldsymbol v_K|^2,\qquad
 U_K=(\rho_K,\rho_K\boldsymbol v_K,E_K),
\]
\[
 F_K=(\rho_Ku_K,\rho_Ku_K\boldsymbol v_K+P_K\boldsymbol n,(E_K+P_K)u_K).
\]

The acoustic speeds and Davis wave bounds are

\[
 S_L=\min(u_L-c_L,u_R-c_R),\qquad
 S_R=\max(u_L+c_L,u_R+c_R),\qquad
 c_K=\sqrt{\gamma P_K/\rho_K}.
\]

These are equation (48) in [Toro's HLLC lecture](https://www.prague-sum.com/download/2012/Toro_2-HLLC-RiemannSolver.pdf).
They require only arithmetic, two square roots, and lane-wise
minimum/maximum operations. There is no Roe average or iterative pressure solve.
The contact speed and star states use these unmodified signal speeds. Define
\(D_K=\rho_K(S_K-u_K)\) and \(D=D_L-D_R\). Then

\[
 S_* = \frac{P_R-P_L+D_Lu_L-D_Ru_R}{D},\qquad
 P_K^*=P_K+D_K(S_*-u_K),\qquad
 P_* = \frac12(P_L^*+P_R^*).
\]

The two star pressures agree algebraically; their average treats both sides
symmetrically under floating-point rounding. For each side,

\[
 \rho_K^*=\frac{D_K}{S_K-S_*},\qquad
 \boldsymbol m_K^*=\rho_K^*
     [\boldsymbol v_K+(S_*-u_K)\boldsymbol n],
\]
\[
 E_K^*=\frac{(S_K-u_K)E_K-P_Ku_K+P_*S_*}{S_K-S_*},\qquad
 U_K^*=(\rho_K^*,\boldsymbol m_K^*,E_K^*).
\]

For MFV, the flux uses the clipped *correction* speeds:

\[
 F_{\mathrm{HLLC}}=
 \begin{cases}
 F_L+\min(0,S_L)(U_L^*-U_L),&S_*\geq0,\\
 F_R+\max(0,S_R)(U_R^*-U_R),&S_*<0.
 \end{cases}
\]

Clipping the corrections combines the usual four wave regions into two contact
sides while retaining the physical upwind flux in a supersonic flow. The packed
kernel first checks whether all active lanes choose the same contact side; a
uniform pack takes that side directly, and a mixed pack selects its state with
a lane mask. Both then evaluate one physical flux and one correction. MFM
instead follows the contact-moving face and uses
\(F_{\rm MFM}=(0,P_*\boldsymbol n,P_*S_*)\), so its mass flux is exactly zero.

If the HLLC contact or either conserved star state is nonphysical, the affected
lane uses the single HLL mean and flux without changing the wave bounds:

\[
 U_H=\frac{S_RU_R-S_LU_L+F_L-F_R}{S_R-S_L},\qquad
 F_H=\frac{S_RF_L-S_LF_R+S_LS_R(U_R-U_L)}{S_R-S_L}.
\]

MFV selects \(F_L\) when \(S_L\geq0\), \(F_R\) when \(S_R\leq0\), and
\(F_H\) otherwise. MFM instead chooses the zero-mass-flux plane

\[
 s_H=\frac{F_{H,\rho}}{\rho_H},\qquad F_{\rm MFM,H}=F_H-s_HU_H.
\]

This conservative moving-face construction also appears in
[Lupi (2023), equations (15)-(18)](https://arxiv.org/html/2210.05682v2#S2.SS3.SSS2).
Its mass component is set to exact zero after forming \(s_H\); momentum and
energy retain the full HLL moving-face flux. Neither replacing \(F_H\) by the
EOS flux \(F(U_H)\) nor discarding only its mass component gives this result.
The positive HLL density places the moving plane within the fan because
\(s_H-S_L=\rho_L(u_L-S_L)/\rho_H>0\) and
\(S_R-s_H=\rho_R(S_R-u_R)/\rho_H>0\).

The HLL mean must have positive density and internal energy. Its normal
momentum flux can nevertheless be negative: equal states with opposing
normal velocities \((-v,+v)\) give \(s_H=0\) and
\(F_{H,m}\cdot\boldsymbol n=P-\rho cv\). This flux is not the EOS pressure
of the HLL mean and is not clamped. The fallback adds diffusion while retaining
opposite pair transfers and fixed particle masses in MFM. A physical HLL mean
alone does not establish positivity of the complete meshless timestep update.

For either method, writing the selected frame flux as
\((f_\rho,\boldsymbol f_m,f_E)\), the returned area-integrated laboratory flux is

\[
 \mathcal F_\rho=A f_\rho,\qquad
 \boldsymbol{\mathcal F}_m=A(\boldsymbol f_m+\boldsymbol w f_\rho),\qquad
 \mathcal F_E=A\left(f_E+\boldsymbol w\cdot\boldsymbol f_m
                     +\tfrac12|\boldsymbol w|^2 f_\rho\right).
\]

The boost always uses the original input frame \(\boldsymbol w\), including
after the HLL moving-plane subtraction. For accepted HLLC states, MFM returns
momentum flux \(AP_*\boldsymbol n\) and energy flux
\(AP_*(S_*+\boldsymbol w\cdot\boldsymbol n)\).
The reported signal speed is \(\max(|S_L|,|S_R|)\); the contact-speed field is
\(S_*\) for HLLC and \(s_H\) for HLL.

Inputs must be finite with \(\rho_K,P_K>0\) and \(\gamma>1\). Finite ordered
wave bounds and \(D<0\) are required before the contact calculation. HLLC is
accepted when \(S_L<S_*<S_R\) and \(P_*>0\), with finite star states,
\(\rho_K^*>0\), and \(E_K^*-|\boldsymbol m_K^*|^2/(2\rho_K^*)>0\) on both
sides. Otherwise that lane uses HLL. Invalid inputs, a nonphysical HLL mean,
nonfinite area norms, or nonfinite output fluxes still throw; diagnostics
identify the failing SIMD lane and production pair IDs. There are no wave-speed
retries, Einfeldt-speed substitutions, or exact-contact solves.
Ill-conditioned gradient matrices cause the actual neighbor search
to expand, up to the configured support bound, rather than introducing an SPH
pressure-force fallback. An invalid face-area predictor uses a conservative
first-order face flux.
A nonpositive mass at an own-step closure stops with a diagnostic. Already
published transfers cannot be retried on only one endpoint: there is no
one-sided first-order retry or nonconservative mass floor.

Thermal energy is recovered by subtracting the kinetic energy from the conserved
total energy, which loses accuracy through cancellation in cold, high-Mach gas.
MFM therefore carries the dual-energy formalism of Bryan et al. (1995),
Comput. Phys. Commun. 89, 149, with the entropic function
\(K=P\rho^{-\gamma}\) of each cell, in the code's comoving variables, as the
auxiliary variable. A fixed-mass cell exchanges no entropy, so \(K\) is constant
between closures except for the cosmological expansion source, which scales it
exactly as the closure scales the thermal energy, and the thermal sources, which
add the same specific energy to both estimates. At each own-step closure the
entropy estimate \(u_K=K\rho^{\gamma-1}/(\gamma-1)\), evaluated at the closing
kernel-partition volume, replaces the total-energy estimate when the latter is
nonpositive or below `hydro.dualEnergyPressureSwitch` (default `1e-3`) times the
cell's total energy (Bryan et al. equation 10). The energy this adds is booked in
the heating diagnostic like a thermal source, so the energy budget stays exact,
and the per-step report counts such cells as `switched=`. After every closure and
source of an instant, each closed cell's entropy follows its total-energy closure
when the cell's thermal energy density reaches `hydro.dualEnergyUpdateSwitch`
(default `0.1`) times the largest total energy density among the cell and its
neighbours (equation 11); otherwise it keeps its own adiabatic value. Shocks thus
heat the entropy while a cold cell beside hot gas keeps its cold state. Both
switches are ordinary `hydro` options with INI and command-line overrides.
Setting the pressure switch to zero restores the strict closure, which stops
with a diagnostic requesting a lower `hydro.cfl` when the thermal energy becomes
nonpositive. MFV exchanges mass and would have to advect the entropy with its
mass fluxes, so it keeps the strict total-energy closure. The entropy is stored
with the per-particle integration state, whose checkpoint layout version is 6.
This backend still does not reproduce the full high-Mach robustness machinery of
GIZMO.

## Implementation scope

The restored implementation uses double-precision hydrodynamic state and
arithmetic, and requires a build with `COSMICTIGER_POSITION_TYPE=fixed64`.
This requirement applies to MFM/MFV; the existing SPH and gravity-only methods
retain their supported position choices. The CPU meshless backend supports
`run.timeIntegrator=traditional` and `global` for both methods, and
`run.timeIntegrator=hierarchical` for fixed-mass MFM. MFV remains unavailable with
hierarchical gravity because changing source masses would require additional
handling in the held gravity interaction. The gravity solver is an independent
choice: `gravity.method.far=fmm` or `pmmm` can accompany the supported integrators.
A uniform global timestep is optional; choosing PMMM does not select it. MFM
accepts `chemistry.enabled` and `starFormation.enabled` under every supported
integrator (see [Chemistry and star formation](#chemistry-and-star-formation));
both must be disabled for MFV, and incompatible combinations are rejected during
option validation. Hierarchical MFM also
requires `lightcone.enabled=false` until crossings can be emitted from completed
hydro trajectory segments instead of tentative gravity projections.

`src/hydro/hopkins.cpp` is a CPU SIMD backend, including in a
CUDA-enabled build. Every HPX locality integrates its own gas cells against a
persistent hydro source tree of those cells, with read-only ghost copies of the
remote cells within stencil reach; rank 0 only sequences the phases of an event
and folds their results (see [Distributed event engine](#distributed-event-engine-november-2-2026)).
No locality holds the global gas state, and no neighbor list is stored.
`gravity.cuda` and `drift.cuda` independently select their CUDA-capable
components. Set `hydro.cuda=false` for this CPU meshless backend; selecting CUDA
for gravity or drift does not move the meshless flux calculation to the GPU.

The SIMD backend is `NativeSimdVector<double>` from
`cosmictiger/math/simd_backend.hpp`, exposed to Hopkins as `hopkins::SimdReal`.
It selects the compiler target's native double-precision width through standard
`<simd>` when available, or `<experimental/simd>` otherwise. The repository also
has `SimdDouble` (a fixed two-lane pack used by the tree) and `SimdDouble8` (eight
lanes); Hopkins uses the native-width type instead of fixing its width to either.
Hydrodynamic storage and arithmetic remain double precision.

Neighbor distance/kernel evaluations, support-radius residuals and derivatives,
geometry moments and their time derivatives, gradient contributions and extrema,
and timestep signal estimates are computed in packs. Pair reconstruction and
Riemann fluxes are also evaluated in packs. Fixed64 position differences are
formed before conversion to double. Reductions retain their scalar particle-ID
order, and the pair scatter retains its conservative equal-and-opposite updates.
Final partial packs use safe inactive inputs and store only active lanes.

SPH and MFM/MFV share the stable-ID geometry tree in
`cosmictiger/hydro/source_tree.hpp`, along with the selected kernel and gas
particle infrastructure. Hopkins retains the tree between events and refits its
bounds from gathered current positions and double support radii. Unchanged
membership requires no gas sort; adding or removing IDs rebuilds the topology.
Ownership changes and particle permutations leave the IDs intact. Initial
identity publication changes only the identifier fields, before the first
neighbor query, and checkpoint reload rebuilds the derived tree.

Both sides of a neighbor query carry padded support envelopes. The original
Fixed64 pair-distance test selects the final stencil, with reciprocal neighbors
and stable-ID reduction order. Every support solve or conditioning expansion
refreshes the tree before querying again, including neighbors whose larger
support reaches a smaller-support sink. This applies to fixed-mass MFM and
variable-mass MFV because spatial indexing is independent of mass.

The meshless support-radius solve uses the same kernel-number-density equation
with its own ordered scalar reduction, so sorting and MPI ownership do not
change the accumulation order. The meshless path computes its own volume,
density, gradients, Riemann fluxes, and conservative update. It does not call
the SPH support, density/viscosity, pressure/thermal-force, or predictor/corrector
operators.

With `run.timeIntegrator=traditional`, gas particles select individual
power-of-two timestep rungs from their local signal-speed, acceleration, and
expansion limits. A particle commits its conserved state and reconstructs its
primitive state and gradients only at its own endpoint or an explicit wake-up.
The persisted state consists of the conserved anchor `U0`, pending integrated
transfers `deltaU`, the anchor volume, gradients, source and trajectory anchors,
and explicit physical and integer beginning/end clocks. Primitive anchors are
recovered from `U0` and its own volume, never from `U0 + deltaU` while inactive.

Each pair opens a MUSCL–Hancock interval ending at the earlier of its endpoints'
scheduled boundaries. The beginning and predicted midpoint fluxes are retained.
At the pair endpoint, one integrated transfer is deposited with opposite signs
into both particles' pending ledgers, including an inactive coarse endpoint.
Coarse–coarse records remain untouched at intermediate fine events. They are
not discarded merely because the pair leaves the current neighbor graph.
Stable particle identifiers retain pair ownership through sorting and migration.

Inactive primitives, face velocities, geometry rates, and timestep signals use
one prediction from their retained own-step anchor. Positions drift to each
global event using the separately anchored mesh velocity and acceleration; a
fine drift neither commits `U0 + deltaU` nor rebuilds the coarse trajectory.
Supports and geometry are refreshed only for the active cells and their
first-ring halo (see [Retained distributed source records](#retained-distributed-source-records-november-2-2026)).
Individual rungs save primitive reconstructions, pair Riemann solves, and
conservative/source commits. `Hopkins tree` log rows separate sorted sources,
updated sources, refitted nodes, topology rebuilds, and visited candidates.
The same geometry tree serves hierarchical MFM, using retained trajectories to
obtain event-time positions even when the shared gravity coordinate is already
at a future endpoint. Pair-flux intervals and wake-up accounting retain the
same conservative elapsed-interval rule.

A newly restrictive limit or a nonphysical primitive prediction wakes an inactive
particle. Neighbor propagation bounds interacting timestep ratios by four.
Before a wake commits, each incident live pair is integrated exactly once over
its elapsed portion, using the retained linear flux representation; both signed
transfers are deposited and the unused future tail is retired. New intervals
open from the wake boundary. Under traditional integration, all gas accelerations
are refreshed by the gravity pass so newly woken gas has a fresh source endpoint;
source commits still occur only at each gas particle's own event. This costs more
gravity force evaluations than selecting only the current active gas sinks.

With `run.timeIntegrator=global`, all gas shares the raw global duration selected
from its signal-speed, acceleration, expansion, gravity, and endpoint bounds.
It is not rounded to a power of two. The same pair-open, pair-settle, and own-step
closure sequence uses physical times; open pair intervals persist across a
checkpoint until the next event. All supported modes have a terminal closure
that settles every elapsed pair, commits all gas, and opens no new interval.

Hierarchical MFM keeps an independent hydro rung in the conserved sidecar; the
shared particle rung continues to select gravity's active suffix. Before the
first signed gravity kick at an event, `hopkinsHierarchicalBegin` closes every
due hydro particle and every particle touched by the gravity ladder. It settles
all incident cached pair intervals first, including their equal and opposite
transfers to inactive neighbors. Each gravity impulse then acts on the closed
double-precision conserved state as `P += m*delta_v` and
`E += delta_v dot P_old + m*|delta_v|^2/2`. Self-gravity is excluded from the
continuous hydro source predictor, so that impulse is applied once.

Only after all closing and opening gravity adjustments have completed does
`hopkinsHierarchicalFinish` select hydro rungs, propagate hydro wake-ups, and
open new pair-flux intervals. A hydro wake leaves the gravity rung unchanged.
The global block clock covers the union of hydro and gravity events; empty
suffix levels preserve the gravity stack when hydro needs a finer event.
`Hopkins hierarchy begin` reports the gravity-active, hydro-due, and closed
counts separately; the usual `Hopkins step` row reports hydro rungs and wakes.
Full-synchronization diagnostics pair the top-ladder potential with kinetic
energy and momentum before opening gravity kicks, and with the closed thermal
state. The separate held kinetic and thermal sums feed cosmological expansion
work during the next drift; they do not overwrite the synchronized snapshot.

The MFM sidecar retains a Fixed64 position anchor, trajectory epoch, and drift
scale separately from the saved gravity endpoint. Exact-rung gravity drifts
record the endpoint instead of changing the gas coordinate immediately. After
hydro reconstruction, each raw gravity coordinate is projected to its saved
endpoint. A fine hydro wake reanchors the physical trajectory at the current
event and replaces that future projection. It may project beyond its own hydro
interval: this value is tentative and is refreshed at every intervening hydro
event before the coarse particle can enter a gravity tree again. The inactive
coarse prefix is absent from finer gravity trees. This permits hydro wake-ups
without changing a gravity interval that is already open. It does not eliminate
wake-ups or establish a new bound on nonlinear waves.

Persistent sorting, independent gravity scheduling and sparse MFM geometry work
are implemented; the retained-record design below replaced the earlier whole-gas
gather at every Begin/Finish event. The historical validation below predates
hierarchical MFM unless explicitly identified otherwise.

### Retained distributed source records (November 2, 2026)

MFM no longer collects every gas particle onto the coordinator at each event.
`src/hydro/hopkins.cpp` keeps two resident structures:

- Owner mirrors on every locality. For each local gas slot the owner stores the
  stable identifier and a bytewise key of every input the coordinator record is
  derived from: the padded-canonical `HopkinsParticleState`, the shared gravity
  rung, and the raw position, velocity, mass, internal energy, support and
  gravity acceleration where the state does not yet supply them. At each
  collection the owner recomputes the keys and reports only the records whose
  key changed, plus its complete identifier list when its membership changed
  (migration, first contact, or new gas). A live gravity force is never read
  while the kick runs: without `readForces` the previously mirrored force is
  reused, and the next collection that may read forces reports the change.
- A coordinator workspace that persists between events. Records sit in
  increasing identifier order, so slot order is reduction order and the
  summation order of every neighbor loop survives migration and restart. The
  workspace also holds the owner map, the hydro source tree and the per-cell
  geometry, face rates, primitives and gradients, together with per-event flags
  recording which cells were solved, built or touched. No neighbor list is
  stored: every consumer queries the tree for the stencil it needs, when it
  needs it.

The coordinator derives the time-dependent inputs of clean records itself:
under hierarchical gravity a sleeping record's position is its retained
trajectory `positionAt(t)` and its imposed acceleration follows from that
position. Those are the same formulas the owner evaluates for a changed record,
so no record is communicated merely because time advanced.

Sources in the tree carry trajectories. A sleeping hierarchical cell's source is
anchored at `positionEpoch` with velocity `meshVelocity/positionDriftScale`,
acceleration `meshAcceleration/positionDriftScale` and validity through its own
hydro endpoint (padded by the predictor's rounding tolerance). The tree's moving
bounds then enclose the retained predictor without any per-event refit of
sleeping sources. Static anchors (traditional and global integration, or cells
without a predictor) stay valid until their owner reports a changed record. The
topology is rebuilt only when the identifier set changes; an owner change alone
(migration) keeps the records and the tree, and a vanished identifier resets the
resident set and every owner mirror before recollecting.

The tree topology follows GADGET-4's practice for its neighbor tree, which is
rebuilt with every domain decomposition and otherwise only drifted in place:
the coordinator rebuilds every locality's topology once per event when at least
`hydro.treeRebuildActiveFraction` of the cells is active (default one third,
the paper's `N1 > N0/3` rule for merging the first time bin into the top one)
or when a domain rebound happened since the last event (`hopkinsDomainsRebound`,
called by the drivers after `domainsRebound`); every other event refits the
touched cells' leaves and ancestors. Membership changes still rebuild. The
decision uses the active set as first marked at the event (due, gravity-touched,
invalid-prediction and terminal cells), before the timestep scan wakes further
cells: in the hierarchical blast fixture the event at t = 0.00375 starts with one
due cell and wakes 381 more, and keeps the topology. The fully active events at
t = 0, 0.002 and 0.004 rebuild. Because
the stencil sets come from the exact Fixed64 pair test, the topology never
changes a result; it only bounds the envelope inflation of long-lived leaves
and the number of leaves a query visits (`Hopkins tree: candidates`).

Neighbor stencils are transient tree queries (`hopkinsNeighborQuery`). The
tree's broad phase accepts every source with `r <= max(radius, supportUpper_j)`,
so one point query from cell i with radius `h_i` also returns the cells whose
own support reaches i; the exact Fixed64 test `r < h_i || r < h_j` then yields
the symmetric stencil `{j : r_ij < max(h_i, h_j)}` directly, which is the set
the earlier gather and the earlier own/symmetric lists produced, in identifier
order. The support solve uses the one-sided test at 1.2 times the current
radius. For the symmetric test to be exact the tree's `supportUpper` must be
current for every cell whose support moved, so the tree is refit immediately
after each support solve and after each singular-moment expansion. Nothing is
kept between calls; the two consumers that make several passes over a cell
(face rates, gradients) query once and reuse the vector inside the call. The
environment variable `COSMICTIGER_HOPKINS_CHECK_NEIGHBORS` enables a brute-force
cross-check of every query against all resident records. Supports therefore
need no scatter/gather iteration: the constant kernel-number solve iterates on
the coordinator, and only the cells whose radius is still changing are requeried.

Each event refreshes the active set A (due cells, gravity-touched cells, invalid
predictions, terminal closures) and the first-ring halo N(A): both endpoints of
every incident pair flux need current supports, volume partitions, moment
matrices and face rates, and the active gradients need the halo primitives.
`hopkinsEnsureGeometry` solves the active supports, derives N(A) from the active
cells' stencils, solves the halo supports, refits the tree after every solve,
and builds geometry for A and N(A), repeating until no support or geometry
changes; it returns A ∪ N(A), which the kick reuses for the gradients and face
rates. From the first stencil query of an event on, no resident position may
change until the event closes. Woken cells that join A during the wake
loop extend the same flags incrementally. Cells outside A and N(A) keep their
own-step anchors, supports, predictors and cached pair intervals untouched;
only touched records (solved, closed, or credited by a settled pair) return to
their owners, and each owner refreshes the predicted public density, velocity
and internal-energy mirrors of its untouched initialized cells locally.

The wake logic is unchanged in form: every cell queries its symmetric stencil at
each event, so the required-rung scan over predicted neighbor states, the
invalid-prediction wake and the Appendix G rung-contrast propagation still
examine every sleeping cell. The scan runs in parallel over cells and applies
its results in identifier order; the Appendix G sweep keeps its sequential
semantics (a raise earlier in the sweep changes what a later cell raises) but
skips cells that cannot raise anybody and cells revisited with an unchanged
rung, and prefetches the due cells' stencils in parallel, a bounded chunk at a
time. Two inputs changed. A sleeping
cell's Courant radius comes from its own-event `anchorVolume` instead of a
fresh geometry solve (an active cell was just closed, so its anchor volume is
the fresh geometry), which keeps the scan independent of history and exactly
reproducible after a restart. Cells outside the halo keep the support from
their last refresh instead of being re-solved at every event. Conservative
search bounds prevent missed neighbors; they still do not validate a stale
prediction, and the wake checks remain the only guard against that.

Between a hierarchical `Begin` and its `Finish` the event stays open: geometry
solved before the gravity ladder remains valid because the hydro positions and
coordinator-owned supports do not change inside an event (unless `Begin` converted
cells to stars, after which `Finish` re-solves them; see
[Chemistry and star formation](#chemistry-and-star-formation)), and the records the
ladder modified (momentum, energy, gravity endpoint and rung of the active
cells) are reimported before reconstruction. Under traditional integration
`hopkinsPrepareGeometry` now receives the base timestep and the global flag so
that it can select the due cells while the gravity kick runs; the subsequent
`hopkinsKick` reimports the records whose forces changed without re-solving.

Restart and migration are unchanged in form (the merge with main's dual-energy
entropy field made the state layout version 7, 472 bytes, pair-cache version 2): checkpoint loading and gas teardown discard the resident
records and owner mirrors through `hopkinsResetSources`, the first event after
a restart recollects everything, and every later decision depends only on the
checkpointed state and the event time. The sound-wave diagnostic solves all
supports in a scratch copy of the workspace, so it no longer changes retained
supports as a side effect.

### Distributed event engine (November 2, 2026)

The workspace described above now exists once per locality and holds only that
locality's cells, in increasing identifier order, in slots `[0, ownedCount)`.
During an event a second slot range holds ghost copies of remote cells, also in
identifier order, indexed by a separate `ghostTree`; every stencil query visits
both trees, and stencils are sorted by identifier, so each reduction sums in the
order a single workspace would. Rank 0 runs an event as a sequence of phases
(`hopkinsRunPhase`), each broadcast to every locality and folded in rank order;
with one locality every phase is a direct call and nothing is communicated.

- Ghost exchange. Every leaf envelope of a locality's owned tree queries each
  peer with one radius `R = 1.5 * 1.2 * max(h_owned)`, raised to whatever an
  earlier support solve found it needed. A peer returns each owned cell `j` with
  `r(x_j, box) <= max(R, supportUpper_j)`, together with its derived state
  (geometry, face rates, support rate, primitive, gradient and the solved/built
  flags). The reply therefore holds every remote member of every symmetric
  stencil of the requester's cells, including members whose own support reaches
  them. Ghosts are replaced wholesale by each exchange, which follows every step
  that changes a support, a geometry, an activity flag or a public view.
- Supports and geometry. `hopkinsEnsureEverywhere` repeats the single-workspace
  steps in lockstep: solve the unsolved active cells, derive the halo, solve the
  unsolved halo cells, build geometry, with an exchange after each step that
  changed anything. A support solve whose search radius would pass `R` stops,
  resets the cell to its entry support and reports it incomplete; the
  coordinator widens the exchange and the cell is solved again from the start.
  Remote halo members are requested from their owners, which solve and build
  them as their own halo. Singular moment matrices expand in lockstep attempts,
  with an exchange between attempts.
- Pair intervals are held by the locality that opened them, the owner of the
  lower identifier at that time; they do not move when an endpoint migrates. At
  settlement each holder asks the peers for the activity of endpoints it does
  not own, integrates the interval and sends each endpoint's deposit to its
  current owner. Deposits are applied at the start of the next phase sorted by
  pair key, left before right, so every endpoint accumulates its shares in the
  order of one walk over a global pair map. Checkpoints write each locality's
  held intervals.
- Timesteps. Each locality scans its own cells. The Appendix G sweep runs on
  rank 0, seeded with every cell's rung and activity; the owners answer its
  stencil requests, and the final rungs return to them.
- Cells that migrate between a hierarchical `Begin` and its `Finish` carry their
  event state (support, solved/built/touched flags and derived geometry) to the
  new owner as a sidecar.
- MFV gravity masses project each held interval locally; rank 0 sums the
  shares in pair-key order and every locality installs the result.

The tree-rebuild decision is made by rank 0 from the global active count, and
every locality rebuilds its owned tree. The sound-wave diagnostic still needs
one locality.

Each event prints a `Hopkins work` row beside the existing `Hopkins step` and
`Hopkins tree` rows: `total`, `active`, `halo` (cells with geometry that are
not active), owner records `received` and `installed` with their byte counts
(`sizeof` of the serialized records plus local indices), tree `sources`
refitted, `supports` (cells solved) and `supportSums` (kernel-sum evaluations),
`geometry`, `rates` and `gradients` cells, stencil `queries` (every transient
tree query made by the support solve, halo derivation, geometry, face rates,
gradients, timestep scan, Appendix G sweep, entropy switch and pair opening),
tree `candidates` visited, and `pairs` opened.


Current `g4_hydro_tree` CPU validation includes a 16-event blast with 381 real
hydro wakes, unchanged mass, momentum error below 1.2e-18, energy error below
2.3e-16, and an exact restart. A self-gravitating wave approaches a refined
global reference: maximum position errors fall from 4.03e-4 through 1.62e-4 to
4.71e-5 under timestep refinement. Its initial synchronized energy agrees
exactly across those refinements and with the global reference. Separate short
FMM and PMMM-PP coupled runs pass exact restart checks; a two-rank FMM run does
as well. Traditional MFM/MFV coupling and restarts still pass. These small tests
do not establish production accuracy or a performance advantage.

The changed CUDA drift, SPH, and degree-four FMM objects compile, including an
`sm_86` device image. This is compilation coverage; GPU execution has not been
validated for the coupled hierarchy.

The validation results recorded later in this document predate this own-step
scheduler unless explicitly identified otherwise; they do not validate the new
wake, source, or overlap ordering.

Conserved accounting includes `U0 + deltaU`. Between coarse endpoints this is an
asynchronous ledger, not a claim that every cell's physical source terms are
synchronized to the fine clock. Before any gravity source construction, MFV mass
is projected to that force time from the anchor, completed pending transfers,
and the elapsed prefixes of all live pair records. This read-only projection
updates only the public mass mirror and never commits a hydro anchor. Terminal
diagnostics first close every remaining interval.

Mass, momentum, total energy, source histories, primitive/gradient anchors,
pair fluxes, and support radii persist in double precision. Traditional/global
coordinates reside in the shared Fixed64 particle array; hierarchical MFM also
retains its own Fixed64 trajectory anchor. Pair separations subtract the
periodic fixed-point integers before converting the local displacement to
double, preserving low coordinate bits that would be lost by first converting
both absolute positions. Drift adds a Fixed64-encoded double displacement to
the existing Fixed64 position. Rounded absolute coordinates in the derived
neighbor tree are used only for conservative candidate searches; exact pair
separations continue to use Fixed64 differences. Ordinary gas output fields
retain float projections, while the persistent source tree uses double support
bounds. The support solve retains its double radius through neighbor-search
refreshes and targets a relative kernel-count residual of `1e-13`. These changes
do not alter SPH's storage or numerical algorithms. Meshless checkpoint sidecars
use particle schema 6 and pair schema 2. Earlier asynchronous state layouts are
rejected rather than reinterpreted under the new event semantics.
The current CPU and CUDA drift interfaces retain double timestep and scale
arguments. CPU and CUDA coupled regression checks are described below. Ordinary particle output
fields remain limited by their float storage; conserved diagnostic reductions
and the sound-wave CSV diagnostic read the double meshless state.

The implemented CFL condition is
`Delta tau <= hydro.cfl * a * R_i / signal_i`, with
`R_i = (3 V_i / (4 pi))^(1/3)`, stored sound speed
`sqrt(gamma (gamma-1) a^2 u)`, and the maximum pair signal speed of equation 37.
This uses the volume-equivalent cell radius, not Hopkins's kernel-length
normalization in equation 36; the same numerical `hydro.cfl` is more conservative
here. The existing SPH CFL convention remains unchanged.
The dedicated one-dimensional `soundWave` scenario instead uses the paper's
`2 * supportRadius` Courant length with `hydro.cfl=0.2`.

The traditional/global force stage calls `hopkinsKick` after fresh gravity is available. It settles
completed or interrupted pair intervals, closes each due particle's source and
conservative interval, and opens new intervals after active reconstruction.
Source-independent support/geometry preparation has a separate entry point so
it can overlap gravity evaluation after gravity has completed its particle sorting.
Hydro tree refreshes update stable-ID geometry without reordering particle storage.
FMM gravity runs in an HPX future while the coordinator prepares hydro geometry;
the future is joined before importing fresh gas accelerations, closing source
terms, choosing timesteps, or reconstructing the next interval's fluxes. The
particle's stored `sourceAcceleration` remains its old force sample while the
gravity solver writes the new sample to the gas acceleration arrays. Neither
sample depends on enabling the optional analysis acceleration output, and no
stored potential is needed by this source integration.

For MFV, the gravity source mass is projected from the conserved anchor, pending
mass, and elapsed portions of live pair intervals before gravity builds its tree.
This projection does not commit a particle or alter its reconstruction anchor.
The PMMM and hydro force stages remain sequential in the current driver.
Distributed CUDA FMM also remains sequential: remote GPU source staging currently
resizes shared particle storage. Single-locality CUDA FMM and CPU FMM can overlap.

Gravity and external forces use an opening source half-step with the old mass
and old acceleration, the accumulated conservative hydro exchange, and a closing
half-step with the new mass and fresh acceleration. These operations execute
once at the particle's own closure. Damping uses its analytic exponential on
each half-step; expansion evolves thermal energy with the corresponding endpoint
coefficient. No later fine-event source correction modifies an inactive anchor.

The CPU/CUDA drift stage only moves positions along the anchored trajectory; it
performs no Riemann solve or conservative update. The predictor, pending ledger,
source moments, stable identifiers, and unexpired pair intervals persist so a
restart may occur between coarse-particle commits.

The shared geometry tree, gas storage, domain distribution, gravity, cosmological
units, scenario initialization, and output infrastructure are reused. MFV mass
is carried with each gas particle through redistribution and checkpointing;
restarts require the same hydrodynamic method. Cosmological conversions use
the existing stored `a v` velocity and `a^2 u` internal energy, with explicit
expansion source evolution. This implementation has not been validated for
large cosmological production simulations.

MFV also includes the gravitational work associated with mass moving between
cells, the effect discussed in Appendix H1. The implemented normalization is
defined explicitly here: for mass rate `F > 0` from cell `i` to cell `j` and
`r = x_j - x_i`, set `q = F r / 2`. The energy source rates at the endpoints
are `q dot g_i` and `q dot g_j`, with `g = -grad Phi`. Their sum is
`F r dot (g_i + g_j) / 2`, exactly `-F (Phi_j - Phi_i)` for a spatially linear
acceleration field. This cancels the potential-energy change of the transported
mass. These two work terms have the same orientation; they are not an
antisymmetric hydrodynamic energy exchange. MFM has zero mass flux and zero
such work.

The beginning and midpoint flux/separation predictions define a linear `q(t)`
on each pair interval. Settlement accumulates `I = integral q(t) dt` and
`J = integral (t - sourceTime) q(t) dt` in each endpoint's own pending source
history. At that endpoint's closure of duration `T`, the energy source includes
`I dot g_old + J dot (g_new - g_old) / T`. This integrates the endpoint-linear
force interpolation over the same completed mass transport, including intervals
shortened by a wake. Sorting, migration, and checkpoints preserve both moments.

With no damping, the own-interval momentum source is exactly
`T (m_old g_old + m_new g_new) / 2`, the endpoint-mass trapezoid in equation 100.
The source half-steps update conserved momentum and energy directly; without
sources, closure reduces to the single conserved addition `U0 + deltaU`.
Flux and mass-work predictions retain their own midpoint time centering.

This is the chosen work normalization, rather than a claim of
literal coefficient agreement with printed equation 101. That equation prints
`T/2` for each temporal mass-work endpoint term, while the spatially symmetric
work defined above uses `q = F r / 2`. The same printed factors occur in the
[published PDF, Appendix H1](https://academic.oup.com/mnras/article-pdf/450/1/53/18507840/stv195.pdf).
The implementation's normalization is checked against the independent potential
change, and its production integration helper is checked against analytic
time-dependent potential transport. SPH retains its original particle-state
interpretation and update.
The Appendix H1 mass-flux gravitational work is distinct from Appendix H2's
adaptive gravitational-softening correction. This integration retains the
existing fixed gravitational softening and does not implement the Appendix H2
adaptive-softening terms or claim full GIZMO gravity equivalence.

The ideal-gas equation of state uses `hydro.gamma`. These solvers do not add
GIZMO's MHD, feedback, particle splitting, or other optional physics.

### Chemistry and star formation

MFM couples the GADGET-4 equilibrium cooling of `docs/chemistry_cpu_support.md`
and the Springel & Hernquist star formation of `docs/star_formation.md` to its
conserved state; `hydro.gamma=5/3` is required by cooling as for SPH. Because MFM
exchanges no mass between cells, each cell's specific thermal energy maps
one-to-one onto its conserved total energy, and the per-particle cooling guess
and star-formation metadata stay with the cell. The sources are operator split
exactly as in the SPH step: when a cell's own-step interval closes in
`hopkinsKick`, after the pair transfers and the gravitational work of that
interval are folded into the anchor and before the new primitives, timestep
selection, reconstruction and pair fluxes, the combined source advances the cell
over the duration it actually closed. An early wake-up or the terminal closure
therefore integrates the source over the shortened interval rather than the
scheduled rung interval. The source consumes the cell's current density
(mass over the kernel-partition volume) and the float stored thermal energy, as
the SPH coupling does, and applies only the resulting change to the double
anchor, so storage rounding does not accumulate; the same specific-energy change
enters the cell's dual-energy entropy. The heating diagnostic and the
synchronized pressure integral of the cosmological energy budget follow the SPH
conventions.

Under traditional or global integration, a whole-cell conversion decided at a
closure keeps the cell's closing state for that kick: it opens no new pair interval (its old pairs were settled at the
closure), it still participates in its neighbours' volume partition and
gradients at that instant, and the scatter removes it from the gas set and
registers the new star for the opening gravity half kick that
`starFormationOpenNewStars` applies. The next kick rebuilds supports and
geometry without it. Per-step reports include the number of conversions as
`stars=`.

Under hierarchical integration, `hopkinsHierarchicalBegin` closes every hydro-due
cell and every cell whose gravity interval ends at the event. A cell on a finer
gravity rung than its hydro rung is therefore closed early at each of its gravity
events. The sources run right after these closures, before any signed gravity
impulse, over exactly the duration each cell closed, including those early
closures; the Finish re-closure of the same cells has zero duration and adds no
source. Cells first woken at Finish receive their sources there, as above. The
thermal change does not interact with the gravity impulses, which change only the
kinetic part of the conserved energy. A cell converted at Begin leaves the gas set
at Begin's install and enters the gravity ladder as a collisionless star; Finish
then re-solves supports and geometry for every needed cell, because Begin solved
them with the converted cell present. A converted cell's gravity coordinate is
reset to the star's ballistic drift from the closure to the end of its open gravity
interval, replacing the projected hydro trajectory. No newborn star receives an
opening kick from `starFormationOpenNewStars` under hierarchy: a star born at Begin
takes its kicks in the ladder, and a star born at a Finish wake keeps the gravity
interval its gas cell already opened. `Hopkins hierarchy begin` rows report Begin
conversions as `stars=`; `Hopkins step` rows report those of Finish wakes. The
validation record is in [hierarchical SPH](hierarchical_sph.md#validation-status).

MFV transfers mass between cells and would require advecting the cooling and
star-formation state with those fluxes; it still requires
`--chemistry.enabled=false --starFormation.enabled=false`.

For example, from the repository root run the existing Sod setup with either
meshless solver, substituting your build directory. Replace `traditional` with
`global` to select a raw global timestep:

```sh
path/to/build/cosmictiger --configuration=configurations/sod.ini --hydro.method=mfm \
  --gravity.method.far=fmm --gravity.expansionDegree=6 --run.timeIntegrator=traditional \
  --gravity.cuda=false --drift.cuda=false --hydro.cuda=false --hpx:threads=4
path/to/build/cosmictiger --configuration=configurations/sod.ini --hydro.method=mfv \
  --gravity.method.far=fmm --gravity.expansionDegree=6 --run.timeIntegrator=traditional \
  --gravity.cuda=false --drift.cuda=false --hydro.cuda=false --hpx:threads=4
```

## Validation

### Dual-energy entropy switch validation on 2026-11-02

Branch `claude/hopkins_entropy_switch`, CPU Release tree (g++-15, Hopkins + SPH +
cooling + star formation, Fixed64, FMM, traditional and global integrators).
`hopkins_test` passes 69 unit tests, including six new ones: entropic-function
inversion and adiabatic compression, the closure's expansion scaling against the
entropy scaling, a warm cell left conservative, a cold supersonic cell whose swamped
thermal energy is replaced from the entropy with the booked correction, a
nonpositive thermal energy rescued at both switch settings, and the update-switch
thresholds. The CTests `startup_options` (7.0 s), `hydro_methods` (23.9 s),
`hopkins_flops_application` (3.9 s), `hopkins_application` (147.8 s),
`hopkins_global_sources` (17.6 s), `hopkins_thermal_sources` (87.3 s),
`hopkins_fmm_traditional_cpu` (12.4 s), `hopkins_fmm_global_cpu` (13.6 s) and
`checkpoint_integrity` (94.4 s) passed serially; with the default switches no
fixture took a pressure switch, so their conservation and exact-restart checks are
unchanged. A 16^3 MFM Sod run over 40 steps took zero switches and conserved
energy exactly. The same run with the diagnostic setting
`--hydro.dualEnergyPressureSwitch=0.99`, which puts nearly every moving cell on
the entropy estimate, took 640 to 1920 switches per step, kept every state
positive, lost 0.55% of the total energy (the entropy formulation omits shock
heating, as expected) and differed from the conservative run by a mean relative
density of 2.6e-3 (maximum 2.6e-2) and a mean relative thermal energy of 2.0e-2
(maximum 0.25, at the shock). Not validated: the 128^3 `lcdm.ini` cosmology with
cooling and star formation that failed at z=6.86 was not rerun under the shared
6-core limit.

The numerical results dated 2026-10-21 and 2026-10-22 below are historical
measurements copied with the saved implementation. They are not fresh validation
of this integration with current main. The historical tables, plots, audit
scripts, and logs under `docs/hopkins_sound_wave_results_20261022/` were not
restored here; their saved source is the third parent of stash
`cef72fbeea6e33168c17002172f5918cb42a7df6`.

### Retained-record per-event work on 2026-11-02

Measured with `tests/run_hierarchical_mfm.py` in a fixed64 Hopkins CPU build
(Release, g++ 15, FMM, expansion degree 4, hierarchical/traditional/global
integrators) on the `mfm_retained_sources` branch. The `Hopkins work` rows of the
MFM blast (432 gas, 16 events, end time 0.004) give, per event:

| event kind | active | halo | records received | records installed | supports solved | kernel sums | geometry built | face rates | gradients | own-list queries | tree candidates | pairs opened |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full, t=0 (first contact) | 432 | 432 | 864 | 864 | 432 | 1728 | 432 | 432 | 432 | 2160 | 769824 | 10800 |
| full, t=0.002 (all gravity-touched) | 432 | 432 | 432 | 864 | 432 | 886 | 432 | 432 | 432 | 1318 | 449732 | 10800 |
| fine, one due cell (7 events) | 1 | 50 | 0 | 102 | 51 | 51 to 154 | 51 | 51 | 1 | 483 to 636 | 131423 to 185409 | 50 |
| fine, 51 due cells (6 events) | 51 | 229 | 0 | 560 | 280 | 280 to 793 | 280 | 280 | 51 | 712 to 1454 | 218480 to 474351 | 1972 |
| wake event, t=0.00375 | 382 | 50 | 0 | 491 | 432 | 1082 | 432 | 432 | 382 | 1930 | 632086 | 10288 |

Records are 592 bytes each including the local index (`sizeof(HopkinsParticle)`
plus `sizeof(ParticleInteger)`), so the one-cell events communicate 0 bytes to
the coordinator and 59976 bytes back; the first event communicates 511488 and
508032 bytes. The halo is the complete neighbor set of the active cells (about
50 per cell), so a one-cell event refreshes 51 cells and a 51-cell event 280.
Median fine/full ratios over the fourteen partial events: records installed
0.34, supports solved 0.38, kernel sums 0.17, geometry 0.38, face rates 0.38,
gradients 0.06, pairs opened 0.09, own-list queries 0.39, tree candidates 0.33.
The query and candidate ratios are the cost of the whole-gas wake scan: every
cell's own list is rebuilt once per event (432 queries) and the active and halo
cells are requeried after their supports move.

The previous implementation at the same tip (`44ca6eaf`) collected every
record at each gather and solved supports and geometry for all 432 cells in
both Begin and Finish. Its `Hopkins tree` rows show 2160 to 3456 source updates
per event, that is five to eight whole-gas gathers of 432 records (1.3 to 2.0 MB
each way per event), and 663552 to 1157328 tree candidates per event, against
131423 for the new one-cell event. These are counts of work items, not timings;
no production speedup is claimed, and the remaining whole-gas scan is the
own-list rebuild and required-rung check that keeps the wake logic intact.

In the self-gravitating wave fixture (512 gas, hierarchical, CFL 0.1 to 0.025)
every cell shares one hydro rung, so all 512 cells are active at every event and
the fine/full ratio is 1; its rows show 0 records received except at the two
events where the gravity ladder kicked all gas (512 records each way).

Regression results on the same branch: the blast keeps 16 events, 381 real
hydro wakes, unchanged mass, momentum error below 1.0e-18, energy error below
2.3e-16 and an exact restart; the wave refinement errors are 4.034e-4, 1.617e-4
and 4.699e-5 (bitwise identical to the pre-change run of this tip, whose
direct-pair gravity bound moved them from the 1.624e-4 and 4.712e-5 recorded
below), with identical initial synchronized energies. The FMM restart/split
matrix passes for MFM and MFV under the traditional, hierarchical and global
integrators, as do the individual-rung, real-wake and global-source fixtures
of `tests/run_hopkins.py`, the `hopkins_flops_application` CTest, all 117
Hopkins/SPH/drift/source-tree gtests, and the two SPH hierarchical CTests.
Multi-locality runs were not repeated for this change.

After merging `main` (`01107f96`: cooling and star formation coupling, HLL
fallback, matrix algebra, dual-energy entropy switch) into the retained-record
branch as `088543f9`, the same fixed64 CPU tree rebuilt and passed 133
Hopkins/SPH/drift/source-tree gtests, `hopkins_hierarchical_application` (16
events, 381 wakes, exact restart; wave refinement errors 4.059e-4, 1.635e-4,
5.085e-5 with the dual-energy switch now active), `hopkins_flops_application`,
both SPH hierarchical CTests, the FMM restart/split matrix for all three
integrators and the three `run_hopkins.py` fixtures. A second tree with
`COSMICTIGER_ENABLE_STARS=ON` and expansion degree 6 passed `hopkins_application`,
`hopkins_global_sources` and `hopkins_thermal_sources`, so gas-to-star
conversion drops resident records and rebuilds the tree topology as intended.
The checkpoint state layout is version 7 (472 bytes) after the merge.

### Distributed event engine on 2026-11-02

One locality against the unmodified `9009127f4` binaries: `run_hierarchical_mfm.py`
keeps all 260 checkpoints payload-identical, and `run_hopkins.py --matrix-only`
keeps the traditional (14) and global (18) FMM checkpoints payload-identical.
Two localities: `hopkins_localities_mpi` (`tests/run_hopkins_localities.py`) finds
the gravity-free blasts bitwise identical to one locality for the traditional,
global and hierarchical integrators, with about 39000 live pairs joining cells of
different localities. The two-locality FMM matrices match the `9009127f4`
coordinator's two-locality output per identifier on all checkpoints. Both builds
fail that matrix's restart check identically ("restart changed physical particle
state"), so the failure predates this engine. The targeted CPU CTest selection
(117 tests) and the stars-build Hopkins, star and chemistry selection (120) pass,
and the CUDA 13.1 build compiles.

### List-free stencil queries on 2026-11-02

The stored own and symmetric neighbor lists were removed on branch
`hopkins_distributed_sources` (base `9009127f4`, the merge of the retained-record
branch with main). Every consumer now makes a transient symmetric tree query,
the tree's `supportUpper` is refit after each support solve and singular-moment
expansion, the per-cell work of the support solve, geometry, face rates,
gradients, halo derivation, timestep scan, entropy switch and pair opening runs
under `hopkinsParallel`, and the Appendix G sweep keeps its sequential
semantics with chunked parallel stencil prefetch. The `SourceTree` counters are
relaxed atomics so concurrent queries are legal.

Validation against the `9009127f4` binaries built with the same options
(fixed64 Release trees, degree 4 without stars and degree 6 with stars):

- `tests/run_hierarchical_mfm.py` (blast, blast restart, three self-gravitating
  waves and the global reference): identical `report.json` (16 events, 381
  wakes, exact restart; wave errors 4.0593e-4, 1.6353e-4, 5.0852e-5). All 260
  checkpoints have byte-identical physical records, integration state, pair
  cache, clock, positions and rungs; the only differing bytes are one 8-byte
  prefix word of the checkpoint header (a timing field), which the manifest
  checksum reflects. The replayed blast prints identical `Hopkins step` and
  `Hopkins hierarchy` rows and energies with `--hpx:threads=1` and `=2`.
- `tests/run_hopkins.py --matrix-only` for the traditional (14 checkpoints) and
  global (18 checkpoints) integrators with FMM, MFM and MFV: payload-identical.
- `Hopkins work` rows change only in `queries`, `candidates`, `sources` and the
  `Hopkins tree` `updated`/`refitted` counts, because every consumer queries
  for itself and every solve refits the tree. For the blast: the full first
  event went from 2160 to 6049 queries (769824 to 1773294 candidates), the
  one-cell events from 483-636 to 594-697, the 51-cell events from 712-1454 to
  1681-2194, the wake event from 1930 to 5107, and the terminal event from 1558
  to 3718. These are counts of parallel work items; the serial whole-gas list
  rebuild and the per-event list storage (two index vectors per cell) are gone.

The transient queries only pay off once a tree query is cheap. The first
list-free build was no faster than the branch: each query re-evaluated every
visited node's moving envelope from its trajectory polynomials (interval
arithmetic with directed rounding) and every visited source's predicted
position, about 0.2 ms per query on 32³ cells. `SourceTree::prepare(time)` now
evaluates all node envelopes and source positions once per event
(`hopkinsOpenEvent`), in-place refits keep them current, and queries at that
time reuse them; the arithmetic is unchanged, so results stay bitwise identical.
Profile of `configurations/lcdm_present.ini` with `hydro.method=mfm`, every CUDA
backend off, `--hpx:threads=2`, ten steps, 32³ gas and dark matter (fixed64
Release, degree 6, stars and chemistry on; the pre-merge binary is the
`build/hopkins_sources_cpu` tree of main `01107f966`), summed exclusive
coordinator wall time of the `hydro.hopkins*` phases and the coordinator total:

| binary | `hydro.hopkins*` | coordinator total |
|---|---|---|
| main `01107f966` (whole-gas gather) | 204 s | 344 s |
| merge `9009127f4` (retained records, stored lists) | 313 s | 457 s |
| list-free, no envelope cache | 320 s | 519 s |
| list-free with prepared envelopes | 82 s | 181 s |

The merged branch was slower than main on this traditional-integrator case
because its support solve made its own-list queries serially (`hydro.hopkins.support`
3:34, `hydro.hopkins.lists` 1:04 of the 7:37 total). With prepared envelopes
the largest phases are the support solve (22 s), pair opening (14 s) and the halo
derivation inside `hydro.hopkins` (11 s); geometry, face rates, gradients, the
timestep scan and the closing entropy switch cost 5.7 to 6.4 s each. These are
two-thread numbers on a shared workstation, not production timings.

### Davis HLL fallback validation on 2026-11-02

The CPU application and unit binary were rebuilt in `build/hopkins_sources_cpu`
with GCC 15, Fixed64 positions, FMM expansion degree 6, and traditional/global
integrators. All 59 Hopkins unit tests passed. The fallback coverage includes
the captured failing pair's normal projections, extreme expansions, an
independent conservation-jump HLL reference, rejected star energy, supersonic
upwind selection, scalar/SIMD agreement, mixed fallback masks with NaN padding
and zero-area faces, pair reversal, Galilean covariance, and conservative
fixed-mass pair updates. Existing four-region HLLC and Sod checks still pass.

`hopkins_application` passed in 107.76 seconds, including MFM/MFV conservation,
individual-rung stepping, force coupling and exact restarts. `startup_options`
and `hopkins_flops_application` also passed. The original 65,536-particle
`release/lcdm_present.ini` failure was reproduced with CPU module overrides,
two workers, Silo disabled and redirected output. With the fallback, this same
setup completed two iterations successfully in 84.05 seconds. The original
failure occurred during initial spatial reconstruction of pair `(0,18704)`.

This is bounded CPU validation, not a full cosmological run, CUDA/MPI validation,
or a performance comparison. The `release/cosmictiger` binary was not rebuilt.

### Initial SIMD unit validation on 2026-11-02

This historical validation predates the HLL fallback described above.

The 49 tests in `tests/hopkins.cpp` pass in standalone GCC 15 C++20 builds at
`-O2`, with fast math and contraction disabled, on both the native four-double
target and the baseline SSE2 two-double target. A C++26 native build also passes;
the installed library still selects `<experimental/simd>` in that language
mode, so the standard `<simd>` branch is not validated here. The tests cover an independent
four-region HLLC reference, uniform and mixed contact masks, scalar/packed
agreement, moving-frame covariance, pair reversal, exact zero MFM mass flux,
every partial-pack length, zero-area faces, poisoned inactive inputs without
invalid/divide-by-zero exceptions, active-lane fatal diagnostics, packed
reconstruction and physical recovery, and FLOP accounting. The existing
independent Sod evolution remains positive, conservative, and convergent.
These unit checks do not establish application, MPI, GPU, or performance results
for the SIMD revision.

### Initial SIMD application validation on 2026-11-02

This historical validation predates the HLL fallback described above.

A fresh `build/hopkins_simd` CPU build uses GCC 15, C++20, `-O2 -march=native`,
Fixed64 positions, FMM expansion degree 6, and traditional/global integrators.
Hopkins retains `-fno-fast-math -ffp-contract=off`. The 49 unit CTests and
`hopkins_application`, `hopkins_global_sources`,
`hopkins_fmm_traditional_cpu`, `hopkins_fmm_global_cpu`, and
`hopkins_flops_application` pass. These cover MFM/MFV evolution, conservation,
individual-rung cache/wake-up behavior, exact physical restarts, source laws,
SPH default dispatch, and live FMM coupling. The repeated FLOP fixture initially
refused to overwrite its existing checkpoints; it passed after moving that
prior output aside and rerunning with a fresh output directory.

Both methods also completed the sound-wave campaign at `N=16,32,34,64,128`.
The non-multiple-of-four `N=34` supplements partial-pack unit coverage.
Density L1 errors decrease from `6.77544e-8` to `1.64421e-9` for MFM and
from `9.33296e-8` to `2.37154e-9` for MFV between `N=16` and `128`.
Artifacts are in `build/hopkins_simd/sound_wave_final/summary.json` and
`build/hopkins_simd/validation/` (logs, build/source hashes, and benchmark source).
No MPI, CUDA, PMMM, or long production run was performed for this revision.

A separate one-core GCC 15 `-O3 -march=native` kernel microbenchmark uses
16,384 varied physical faces, repeated 96 times, with uniform and mixed contact
masks for MFM/MFV. Five-run median paired speedups over the shared scalar solver
are 2.11–2.51 times including input gathering, and 2.31–2.84 times for prepacked
inputs. Checksums agree to the 12 printed significant digits. This measures
Riemann-kernel throughput, excluding production reconstruction, geometry,
scatter and HPX overhead; concurrent compilation and unpinned execution limit
timing precision. No end-to-end application speedup is established.

### Integration validation on 2026-11-01

Release builds with GCC 15, HPX, Fixed64 positions, potential expansion degree 6,
and all three gravity integrators compiled successfully with CUDA disabled and
with CUDA 13.1 targeting an RTX 3090. The following checks passed:

- Hopkins numerical primitives, Gresho initialization, and drift dispatch,
  including preservation of low Fixed64 bits and double drift intervals.
- Both MFM and MFV with FMM, PMMM-PP, and PMMM-FMM gravity under traditional and
  global stepping, on CPU and with CUDA gravity/drift. These short fixtures
  check live gravity, mass conservation, exact restarts, and clipped global
  endpoints for every combination.
- The serial wave, Sod, Gresho, and blast regression suite, including individual
  rungs, inactive predictor retention, pair-flux reuse, conservative updates,
  cross-method restart rejection, and default/explicit SPH parity.
- Two MPI localities for individual-rung MFM/MFV conservation and exact restart,
  and CPU gravity checks for imported MFV source masses and gas cache handling.
- Analytic uniform-force laws with unequal raw global timesteps; MFV restart
  with nonzero pending gravitational work; and independent analytic integration
  of cached mass-flux work in a time-dependent potential. The latter tests the
  production work-integration helper against potential-energy transport.
- CPU analysis pipelines with meshless mass and velocity overrides.

The run logs are under the untracked local directory
`build/hopkins_merge_20261101/`. These checks establish the tested integration
paths, not full-duration hydrodynamic or self-gravitating convergence, CUDA/MPI
scaling, or cosmological production accuracy. The stopped sound-wave amplitude
campaign was not resumed. Those historical runs executed coordinating-CPU hydro
sequentially with gravity; they did not exercise the newer preparation overlap.

For a Fixed64 build with `COSMICTIGER_ENABLE_HOPKINS=ON`, FMM and PMMM,
potential expansion degree 6, traditional
and global integration, SPH infrastructure, and testing enabled, build the application, `hopkins_test`,
and `gresho_test`, then run the primitive and end-to-end checks:

```sh
ctest --test-dir path/to/build --output-on-failure -L '^hopkins$' --parallel 2
```

The end-to-end script can also be run directly:

```sh
python3 tests/run_hopkins.py path/to/build/cosmictiger /tmp/hopkins_integration
```

For the individual-rung fixtures on two MPI localities, use an MPI-enabled build:

```sh
python3 tests/run_hopkins.py path/to/build/cosmictiger /tmp/hopkins_mpi \
  --mpi-exec=/path/to/mpiexec --localities=2 --async-only
```

`tests/hopkins.cpp` exercises geometric consistency, reconstruction, moving-face
fluxes, conservation, and shock/contact behavior. `tests/run_hopkins.py` runs
small three-dimensional wave, Sod, Gresho, and blast fixtures for both methods;
checks finite positive states and mass/momentum/energy conservation; verifies
MFV mass exchange and exact restart state; rejects cross-method restarts; and
compares explicit `sph` selection with the default. Uniform-force fixtures
check synchronized momentum and energy against analytic solutions; forced
restarts include nonzero pending MFV gravitational work. An additional blast
fixture requires distinct gas rungs and recorded reuse of pair fluxes while gas
is inactive, checks conservation at every fine-step checkpoint, and restarts
inside an unexpired coarse-pair interval. Restart comparison includes the
conservative/source state, inactive prediction anchors, and cached pair fluxes.
It checks that anchor ages stay within the gas-rung interval and that inactive
primitive/gradient anchors remain unchanged across intervening fine steps.
The optional MPI run reduces
all rank payloads and requires cached pairs crossing rank ownership. These short regression
fixtures do not establish convergence, full-duration vortex fidelity, or
production-scale MPI performance.

### Historical published sound-wave test on 2026-10-22

`soundWave` is a dedicated one-dimensional scenario for
[Hopkins (2015)](https://arxiv.org/html/1409.7395v2), section 4.2.1 and Figure 2.
It uses a periodic unit interval, mean
density and sound speed one, `gamma=5/3`, background pressure `3/5`, and
amplitude `A=1e-6`. The right-going linear reference is

```text
delta(x,t) = A sin(2 pi (x-t))
rho(x,t)   = 1 + delta(x,t)
vx(x,t)    = delta(x,t)
P(x,t)     = 3/5 + delta(x,t).
```

Each run ends after one crossing, at `t=1`. The sweep uses
`N=16,32,64,128,256,512,1024,2048`, CFL 0.2, cubic splines, HLLC and the ordinary
limiters. Every particle, including extrema, enters the unweighted density
`L1 = sum(abs(rho_i-rho_exact(x_i,t)))/N`. The Python runner recomputes the
analytic reference independently from the reported double coordinates and
checks the C++ reference columns. This is combined spatial/CFL refinement;
it does not isolate temporal order at a fixed particle count.

We initialize equal masses `m=1/N` and invert
`q=x+A*(1-cos(2*pi*x))/(2*pi)` at `q=(i+1/2)/N`. The historical mass/position
recipe is not specified in the paper, so this is a reconstruction of its
continuum initial condition, not a claim to have recovered the original IC
file. The later downloadable author IC has a different sound speed and crossing
time and was not substituted for the published setup.

The main configuration uses effective neighbor number four with the inspected
[author code's one-dimensional normalization](https://github.com/pfhopkins/gizmo-public/blob/78b66ffcfad6362f8aa0798f8a23a6c9ba5dd2ca/declarations/constants.h#L255-L257),
`N_ngb=2*h*sum(W)`, giving `h` approximately `2/N`. Equation 38 of the paper
prints the coefficient as one instead. A separate sensitivity configuration
uses `hydro.neighborNumber=8` in our convention, implementing the printed
equation with paper neighbor number four and `h` approximately `4/N`. The
historical discrepancy remains unresolved.

To run a new full main sequence with the restored implementation:

```sh
python3 tests/run_hopkins_sound_wave.py path/to/cosmictiger /tmp/sound_wave
```

The runner saves commands, resolved options, binary/configuration/runner hashes,
initial and final norms, adjacent-resolution orders, and conservation changes.
It refuses to mix changed executables or configurations within a campaign.
`--reprocess /tmp/sound_wave` recalculates diagnostics without running a solver.

The historical archive includes the failed original-storage campaigns. An
independent initial-only audit reproduced their density floor using Fixed32
coordinates and float separations. With ideal coordinates and double separations,
initial density error instead falls from `1.6342e-8` at N16 to `9.9887e-13` at N2048, following
`N^-2`. This establishes an initialization/representation error; it does not
by itself explain the later evolved error or validate the integrator. That
audit launched no simulation; its script is part of the historical archive
identified above, not this restored source tree.

The printed-equation run also exposed support iterations alternating between
adjacent float radii. A storage-bracket termination fix allowed that old-storage
sequence to complete; it still failed convergence. The subsequent precision
revision retains coordinates, drift increments, separations and support radii
in double, without changing the reconstruction, limiter or flux integrator.
For the current-convention N2048 run, `dt=1/8192`; the initial velocity amplitude
`1e-6` gives a displacement of at most `1.22e-10` per step. The Fixed32 spacing
is `2^-32=2.33e-10`. Computing a drift in double and then discarding its
sub-Fixed32 part at every step therefore loses relevant motion. The historical
persistent double coordinate preserved those increments. The restored backend
now uses authoritative Fixed64 coordinates, whose spacing is `2^-64`, instead;
the historical double-coordinate convergence results below do not validate
that changed representation.

Both historical double-precision sequences recovered approximately second-order
density L1 convergence. For the main normalization, the final errors are:

| N | MFM density L1 | MFV density L1 |
| ---: | ---: | ---: |
| 256 | 4.19896e-10 | 6.10911e-10 |
| 512 | 1.05712e-10 | 1.54517e-10 |
| 1024 | 2.61380e-11 | 3.82724e-11 |
| 2048 | 6.63130e-12 | 9.47034e-12 |

An ordinary least-squares fit of `log(L1)` against `log(N)` over these four
resolutions gives orders **1.99697 for MFM** and **2.00476 for MFV**. The
printed-equation normalization gives **1.97869** and **1.97608**, respectively.
The main-normalization density L2 orders are lower, 1.86119 and 1.87228;
the printed-equation sequence gives 1.98431 and 1.98089. The second-order claim
here refers specifically to the paper's density L1 resolution diagnostic,
with every particle included. No extrema were removed and no limiter or time
integration formula was changed to obtain these results.
An independent read of all 64 archived initial/final particle tables reproduced
every reported norm. Across the 32 runs, maximum absolute relative changes in
total mass and total energy were `2.22e-16` and `1.23e-16`, respectively.

On the final 2026-10-22 executable, all 48 focused tests passed (34 Hopkins,
five CPU drift and nine drift-dispatch tests). The complete CPU integration
suite and the individual-rung conservation/restart suite with two MPI localities
and one worker per locality also passed. Those historical checks included
default-versus-explicit-SPH equivalence and preservation of the SPH drift
rounding then in use. CUDA drift compiled successfully; GPU runtime accuracy
was not tested. The historical archive contains the commands, hashes, logs,
fit values, and initial/final particle tables. These results do not establish
that the restored implementation passes the current build or test suite.

Relevant papers citing Hopkins include:

- [Hopkins & Raives (2016)](https://academic.oup.com/mnras/article/455/1/51/983632),
  section 3.1/Figure 1, and
  [Hubber, Rosotti & Booth (2018), GANDALF](https://arxiv.org/html/1709.04488),
  section 6.1/Figure 4, report second-order wave convergence. These are
  resolution sequences, not isolated fixed-resolution time-refinement tests.
- [Alonso Asensio et al. (2023), PKDGRAV3](https://academic.oup.com/mnras/article/519/1/300/6847218),
  section 3.1.1/Figure 2, explicitly identify single-precision particle masses
  as preventing a `1e-6`-amplitude convergence test in their implementation.
  Their plot instead uses amplitude `1e-3`, gamma 1.4, 32 neighbors and disabled
  gradient limiting. Its fitted MFM/MFV orders are 1.84/1.89. Their mass-storage
  limitation differs from the coordinate/separation error demonstrated here.
- [Groth et al. (2023), OpenGadget3](https://arxiv.org/html/2301.03612v2),
  section 4.2.1/Figure 3, find between first- and second-order convergence for
  their different three-dimensional wave setup and attribute the reduction
  to limiting; they also discuss support-iteration-related scatter.
- [Pakmor et al. (2016), AREPO](https://arxiv.org/html/1503.00562v2), Figure 4,
  require both improved gradients and time integration to recover second-order
  vortex convergence. Their implemented time update combines primitive
  prediction and endpoint flux averaging. This is not evidence that GIZMO's
  published sound-wave test is first order.
- [Zier & Springel (2022)](https://arxiv.org/html/2205.07916v2), section 5.1/Figure 8,
  provide a fixed-resolution temporal-refinement test of their improved
  moving-mesh scheme, with approximately second-order errors.

### Historical individual-rung checks before the precision revision

These measurements were made on 2026-10-22 before the double-coordinate
revision. The fixed-resolution temporal probe below has not been repeated
with the final precision changes; the published sound-wave sequence above
does not independently establish its temporal convergence order.

The complete CPU integration suite passed with two worker threads, including
wave, Sod, Gresho, blast, forced evolution, method-mismatch rejection, and
explicit/default-SPH equivalence. The individual-rung suite also passed with
two MPI localities and two worker threads per locality. Restarts inside a
coarse-pair interval reproduced every physical particle field and persisted
solver field exactly; MPI comparison uses stable particle identifiers to allow
storage reordering and excludes native padding.

The 432-particle blast fixtures took eight fine steps to `t=0.0005`:

| Method | Minimum active gas | Maximum reused pair fluxes | Retained prediction anchors across steps |
| --- | ---: | ---: | ---: |
| MFM | 1 / 432 | 10750 / 10800 | 2867 |
| MFV | 15 / 432 | 10148 / 10800 | 1668 |

The tests verified that unchanged anchor clocks retain exactly unchanged
primitive/gradient values, and that inferred ages remain within the owning
gas-rung interval. Across the unforced CPU fixtures and MPI blast fixtures,
maximum relative conserved-energy drift was `3.27e-16`, momentum drift divided
by `sqrt(initial mass * initial energy)` was `2.01e-18`, and total mass was
unchanged. Public float fields passed an independent storage-precision check.
These conservation measurements compare each fine-step checkpoint with the
first checkpoint.

An independent smooth-wave probe used a regular periodic `N^3` lattice with
`m=1/N^3`, zero velocity, and
`u=(1+0.001*sin(2*pi*x))/(gamma-1)`, with `gamma=5/3`. These were separately
seeded checkpoints, rather than the built-in displaced `hydroWave` initial
condition. At `t=0.04`, uniform timesteps `0.005`, `0.0025`, and `0.00125`
for `N=8,16,32` gave the following velocity L2 errors relative to the analytic
linearized acoustic-plus-entropy solution, normalized by its velocity-mode RMS:

| Method | N8 | N16 | N32 |
| --- | ---: | ---: | ---: |
| MFM | 0.1806 | 0.04710 | 0.005752 |
| MFV | 0.1888 | 0.05172 | 0.008563 |

The analytic reference neglects terms quadratic in the perturbation amplitude.
These measurements show combined spatial and temporal refinement. A separate
fixed-N8 test with a 1% perturbation and successively halved timesteps showed
approximately first-order temporal self-convergence against the finest run.
Independent second-order time convergence is therefore not established by
these tests.

For the 432-cell blast at `t=0.0005`, halving the default CFL twice reduced
velocity RMS differences from a uniform-rung-7 reference from
`0.00823` to `0.00383` to `0.00169` for MFM and from
`0.01333` to `0.00624` to `0.00280` for MFV. All states remained positive and
MFM particle masses stayed fixed. The final robustness build repeated the
N8/N16 wave cases and the default-CFL/reference blast cases with exactly
unchanged decoded positions, masses, velocities, and internal energies. N32
and the two intermediate CFL runs were measured on the preceding build before
its support-edge safeguards; they were not repeated on the final build.

### Historical shared-rung checks on 2026-10-21

The measurements below used the earlier shared-rung integrator. They record
its validation history and do not establish accuracy of the individual-rung
implementation described above.

The CPU executable completed eight additional runs using two HPX threads:
`hydroWave` to `t=0.1` and `sod` to `t=0.03`, each with MFM/MFV and
`initial.particlesPerDimension=8,16`. All used 32 coarse intervals, the default
`gamma=5/3`, 42 neighbors, and `hydro.cfl=0.22821773229381923`, with gravity,
chemistry, and Silo disabled. All final masses, densities, and internal energies
were positive and finite. Across these cases, maximum relative mass and total
energy drift were `6.19e-8` and `6.91e-8`; maximum momentum drift divided by
`sqrt(initial mass * initial energy)` was `5.04e-9`. Initial energies were
computed from the scenario's stored float values, avoiding text-output rounding.

The existing Sod diagnostic compares 1000 sampled particles with its exact
solution. These are sampled relative L1 errors, not volume-integrated norms:

| Method | Density, N8 | Density, N16 | Pressure, N8 | Pressure, N16 |
| --- | ---: | ---: | ---: | ---: |
| MFM | 0.09225 | 0.03405 | 0.14028 | 0.05661 |
| MFV | 0.10682 | 0.04096 | 0.15392 | 0.05922 |

N8 and N16 initialize 576 and 4608 Sod gas particles, respectively. The errors
decrease with refinement, but these two sampled measurements do not establish
a convergence order. `hydroWave` has no exact-solution diagnostic and its
displacement scales with particle spacing; its runs establish stability and
conservation over this interval only.

To run the same Sod scenario with the current individual-rung implementation,
use the following and repeat with `mfv` and/or `particlesPerDimension=16`, using
a separate output directory for each case. This does not reproduce the
historical shared-rung integrator exactly:

```sh
path/to/build/cosmictiger --run.scenario=sod --hydro.method=mfm \
  --gravity.method.far=fmm --gravity.expansionDegree=6 --run.timeIntegrator=traditional \
  --initial.particlesPerDimension=8 --run.endTime=0.03 --run.numberOfSteps=32 \
  --gravity.cuda=false --drift.cuda=false --hydro.cuda=false --gravity.enabled=false --chemistry.enabled=false \
  --output.silo=false --output.directory=output/sod_mfm_n8 --hpx:threads=2
```

The printed `L1` is the sampled density error. `sod.txt` contains `x`, density,
exact density, pressure, and exact pressure; the pressure metric above is
`sum(abs(P - P_exact)) / sum(P_exact)`. Change `run.scenario` to `hydroWave`
and `run.endTime` to `0.1` for the wave checks.

Two-locality CPU checks also completed four wave steps with 512 gas particles
for each meshless method. Global checkpoint reductions conserved mass,
momentum, and energy to storage roundoff, with maximum relative energy drift
`1.06e-8`; MFM masses stayed fixed and MFV masses changed. The SPH and MFV
gravity/migration fixtures and a default-SPH distributed wave run passed.

Small cosmology runs with 512 gas and 512 dark-matter particles exercised both
methods with gravity and expansion enabled and chemistry disabled. With
`run.maxIterations=2`, the existing loop logged three steps and reached
`a` approximately `0.02032`, saving finite velocities and positive finite gas
mass, density, thermal energy, and support radii. These are execution checks,
not cosmological convergence or energy-conservation measurements. Affected
CUDA translation units were compile-checked; GPU execution was not tested.

### Own-step integration and independent backends: November 2, 2026

That revision used conserved anchors plus pending conservative increments,
retained gradients, anchored drift trajectories, and elapsed pair-flux intervals.
Its particle-state checkpoints used version 5; live pair records used version 2. Old
particle-state layouts are rejected. Fresh force samples and the own-step source
anchor remain separate, including when a particle wakes before its scheduled end.

Fresh validation used Fixed64 positions and FMM potential degree 6 (order 7):

- CPU-only and CUDA 13.1 application builds succeeded.
- The 101-case CPU selection covering Hopkins, SPH kernels, drift dispatch, and
  PMMM-PP/PMMM-FMM traditional/global coupling passed; all 19 CUDA SPH kernel
  tests passed, including deferred versus unsplit acceleration-source assembly.
- The full Hopkins application and global-source regressions passed, including
  conservative mass/momentum/energy accounting and exact restarts.
- Natural MFM/MFV blast fixtures completed two coarse steps with actual early
  wakes. MFM closed 381 particles early and truncated 10,222 pair intervals;
  MFV closed 8 early and truncated 248 pairs. The tests independently integrate
  every retired pair prefix and compare both endpoints' deposits. Three occupied
  bins span rungs 0 through 3; four simultaneously populated bins were not tested.
- Two-locality CPU individual-step/wake runs and gas migration passed, including
  exact restart immediately before wake-up.
- CPU Hopkins with CUDA FMM gravity and drift passed traditional/global coupling
  and exact-restart checks. CPU SPH with CUDA FMM gravity passed traditional and
  global smoke/restart checks. These include evidence of actual CUDA force launches.
- One-period MFM/MFV sound waves at 16, 32, and 64 particles completed. Density
  L1 error decreased from 6.67e-8 to 6.51e-9 for MFM and from 9.27e-8 to 9.05e-9
  for MFV. This is combined space/time refinement, not a temporal-order proof.

The module-specific CLI/INI migration passed 508 grouped-options cases and the
startup checks. CPU analysis in a CUDA-enabled executable also passed with a
runtime-call guard after fixing host allocation release before CUDA initialization.
Coordinator profiles and CUDA elapsed-work records do not establish a measured
overlap speedup; no such speedup is claimed.

Intermediate energy and output mass fields remain views of the asynchronous
conserved ledger; unfinished live-pair/source intervals are not reconstructed at
arbitrary observation times. In particular, an MFV lightcone mass is not an
independent reconstruction at its crossing time. Terminal synchronized diagnostics
are covered by the application tests. Large cosmological production evolution,
CUDA MPI overlap, and long-time gravitational-energy convergence remain unvalidated.
