# Retained SPH octree and midpoint hydro kicks

November 3, 2026. Branch `codex/sph_octree`.
This design supersedes the earlier shared gravity-leaf/SPH kick implementation.

## Authorized scope

Global, traditional, and hierarchical integration use the same adaptive SPH
neighbor octree and entropy formulation. Hydro uses Morris–Monaghan evolving
viscosity, or fixed viscosity, with one force evaluation per midpoint. The user
explicitly accepts first-order temporal accuracy from viscosity. Chemistry,
stars, PM/P3M, and acceleration-gradient viscosity switches are outside the new
midpoint implementation. Their existing traditional/global paths remain separate;
hierarchical SPH rejects unsupported switches rather than silently substituting.

## Operator schedule

Locally derived composition, written in execution order:

    K_gravity(h/2) D(h/2) K_hydro(h) D(h/2) K_gravity(h/2)

Pressure is evaluated at midpoint positions. A hydro event freezes its input
sources before any target publishes updated velocity or entropy. It applies one
full positive hydro kick and stores the proposed next hydro rung. Viscous heating
and viscosity evolution use the same physical interval; there is no RK2 stage.
The inviscid fixed-rung composition is second order. Viscosity and changing rung
assignments do not inherit that guarantee.

For a slow interval T and a fast interval T/2, let G_s contain all gravity terms
involving slow particles and G_f contain gravity internal to the fast set. H_s
and H_f update their respective target sets but read all required gas neighbors.
The symmetric fixed-rung reference schedule is:

    G_s(T/2) G_f(T/4)
    D(T/4) H_f(T/2) D(T/4)
    G_f(T/4) H_s(T) G_f(T/4)
    D(T/4) H_f(T/2) D(T/4)
    G_f(T/4) G_s(T/2)

The implementation combines coincident closing/opening gravity kicks before the
coarse hydro event. Inviscid pressure kicks commute with gravity; the ordering
error from velocity-dependent viscosity is within the accepted first-order
viscous treatment. This avoids saving gravity across hydro. The nested signed
gravity ladder itself remains gravitational; no hydro runs on its negative
correction calls.

A nonzero integer-clock boundary has one possible hydro midpoint:
`hydroRung = minRung(integerTime) - 1`. The finest current rung also receives its
hydro kick at the center of each driver interval. Targets are selected by exact
rung, while density/support preparation retains all necessary neighbors. Global
steps instead use one midpoint for all gas and an arbitrary positive interval.
An initial zero-duration force estimate seeds timestep bounds without a kick.

The driver advances all particles chronologically in half-intervals. This is a
conscious first implementation: there are no per-particle last-drift clocks,
future positions, back projections, or predictor records. It adds whole-array
drift work on fine hierarchical intervals. Lazy leaf/source drifts are a later
optimization and would require explicit drift-clock bookkeeping. Cosmological
half-drifts integrate their own intervals rather than halving a full-step factor.

GADGET-4 provides the gravity ladder (equation 50) and brackets an SPH step by
gravity half-kicks (§5.7.3); its inner SPH step is KDK. The midpoint DKD schedule
above and its integer-clock mapping are local derivations, not quoted equations.
Source: https://wwwmpa.mpa-garching.mpg.de/gadget4/gadget4-code-paper.pdf

## Tree retention and gravity rung merging

The spatial tree uses hybrid fanout near the bucket threshold (November 4, 2026):
for `B = compute.sphBucketSize`, counts above `4B` split into eight cells, counts
in `(2B, 4B]` into four, and counts in `(B, 2B]` into two. Counts at or below `B`
remain leaves. Partial splits bisect the longest one or two partition axes,
breaking equal-length ties in x/y/z order. Empty children remain absent; rank
routing stays binary.

Each SPH node retains its original spatial partition and an additional
`Fixed32Range driftBox` enclosing its moving members. Bounds expand using member
velocity extrema and the actual drift factor, rounding outward and conservatively
handling periodic wraps. Support changes refresh neighbor bounds. Small events
retain particle ordering; migration and other invalidations force reconstruction.

Both tree construction and refresh return `SphTreeCreateReturn::minVolumeRatio`
and `maxVolumeRatio`, the extrema of `volume(driftBox) / volume(box)` over
nonempty spatial nodes, including internal nodes. Rank-routing nodes are excluded, and an empty tree returns zero.
This measures conservative swept bounds since the last rebuild relative to the
original partition; periodic seam expansion contributes, while kernel-support
radii do not. A fresh tree can report less than one because tight particle bounds
need not fill their partition. Rebuilding resets the swept history. The driver
prints both values with its tree diagnostics.

`--compute.sphTreeRebuildFraction` defaults to one third of **all particles**.
Rebuild at `active/total >= fraction`, inclusive, without accumulating events.
The accepted interval is [0,1]; zero rebuilds every nonempty event. This remains
independent of `run.rungMergeFraction`. Hierarchical gravity promotes the lowest
rung when cumulative counts satisfy N_n <= 3 N_(n+1), including equal-cost ties.

## Storage and checkpoints

The 64-byte `SphHierarchicalState` and the inactive-position restore buffer are
removed. Midpoint gas has no separate persistent gravitational acceleration.
Existing hydro acceleration/entropy-rate arrays contain the hydro result only;
source arrays remain temporary immutable staging for each force pass.

The existing one-byte proposed-rung array now retains the last hydro timestep
constraint until the next hydro evaluation and is migrated/checkpointed. Gravity
combines this scalar constraint with its own timestep at eligible boundaries;
hydro does not change a rung halfway through an open interval. Global integration
adds one driver-level double for the latest raw hydro timestep bound. Global
adaptive gravity still needs temporary all-particle force storage between force
reduction and its opening kick; that storage is released after the kick.

The extra octree bound remains 24 bytes per node (normal-index node 200 → 224
bytes). The gas payload uses base version 60000, without a hierarchy offset.
Older gas payloads are intentionally rejected. Global driver extensions use
version 2 and serialize both pending global interval and hydro bound.

Checkpoints occur after the previous interval's midpoint hydro and final drift,
with every position at driver.tau and the current boundary gravity/hydro events
still pending. Restart rebuilds transient trees and source workspaces.

## Validation

Required checks include exact-rung target selection with inactive neighbors,
hydro-only acceleration and timestep estimates, actual-interval Morris–Monaghan
updates, direct CPU/CUDA gas gravity kicks, midpoint lattice/positive intervals,
retained bounds, physical convergence, and checkpoint/migration/restart coverage.
Current execution evidence is recorded separately in `sph_octree_validation.md`.
