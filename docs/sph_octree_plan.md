# Retained SPH octree and shared FMM leaf kick

November 3, 2026. Branch `codex/sph_octree`, baseline `dc0d3f0d9`.

## Authorized scope

Combine FMM gravity and SPH force evaluation at gravity leaves for global,
traditional, and hierarchical integration. All three use the same SPH formulation
and adaptive octree. Retain SPH octree membership between large active events.
Add a `Fixed32Range` that conservatively encloses drifting members of each node.
Rebuild and re-sort at an event when the active count is at least one third of
all particles, inclusive, rather than accumulating active counts across events.
Chemistry, stars, PM and P3M are outside this implementation scope.

## Design

1. Keep the original octant partition separate from moving membership bounds.
   Expand the latter using the minimum and maximum member drift velocities and
   the same displacement factor used by the particle drift. Handle periodic
   wrapping conservatively and round fixed-point bounds outwards. Update support
   bounds whenever smoothing lengths change. Inactive neighbors remain sources.
2. Reuse node membership and particle order on small active events. Rebuild at
   `active >= ceil(total/3)` and whenever topology is invalidated by migration,
   resizing or a gas permutation. Delay routine domain migration to rebuild
   events; gravity particle permutations preserve the gas-to-particle mapping.
3. Complete density, support, viscosity preparation and immutable source staging
   before leaf workers modify acceleration or velocity. Query the SPH octree once
   for a gravity-leaf gas batch, with both target and source kernel extents.
4. Consume the temporary gravity force directly when calculating total gas
   acceleration. Keep the total acceleration required for prediction and opening
   kicks, but release the separate per-gas gravity field on the combined path.
5. Close the old interval with its old duration. Reduce global timestep or run
   the traditional neighbor rung limiter before opening the new interval.
   Preserve initialization, closing-only diagnostics and exact checkpoint restart.
6. Keep CPU and CUDA behavior consistent. Any limitation must be explicit and
   tested; no silent bypass of selected force backends.

CPU gravity leaves consume their temporary field directly. CUDA gravity launches
retain a temporary field over their target interval, then run grouped SPH kernels
after the gravity stream completes. This removes persistent per-gas gravity
storage, but is not a single fused device kernel. Peak transient storage and
runtime gains have not been benchmarked. Standalone PM and source-update paths
retain their existing optional handoff arrays.

The gas checkpoint payload now has base version 50000, with a 1000 offset for
hierarchical predictor records and the existing physics/viscosity suffixes.
Earlier gas payloads are rejected; transient gravity storage is not serialized.
The hierarchy record is 64 bytes per gas particle in the validated build. The
extra node bound increases the normal-index SPH node from 200 to 224 bytes.

GADGET-4's documented approach similarly rebuilds its neighbor tree at large
active events and otherwise expands the nodes to contain their assigned gas:
https://wwwmpa.mpa-garching.mpg.de/gadget4/05_parameterfile/#domain-decomposition
The displacement-envelope formulas are derived locally from this code's drift.

## Hierarchical boundary

The user's clarified scope includes hierarchical integration with the same SPH
formulation and adaptive octree. Its gravity driver still evaluates nested active
source sets, applies signed gravitational correction kicks, and immediately
drifts a completed rung to its future endpoint. Hydrodynamics runs exactly once
at the top physical event, with positive closing and opening durations.

The hierarchy adds a gas-only predictor record: hydro acceleration, physical
predicted velocity, prediction time, stored-position time, the exact inverse
scale used by that drift, entropy time, and a hydro rung bound. The ordinary total
acceleration remains the force predictor; subtracting the stored hydro component
recovers the coarse gravitational predictor. There is no persistent separate
gravity-acceleration vector. These hierarchy-only records add memory; the
storage reduction of the global and traditional paths is not claimed here.

The implemented schedule is:

1. After the hierarchy's migration and gravity-tree construction, advance every
   gas velocity predictor to the event time. Project an inactive gas source from
   its stored future endpoint along its actual drift velocity, using the recorded
   cosmological drift factor. Its live endpoint is saved for restoration.
2. Run the common octree density and immutable force-source staging at that event
   time. The shared leaf kernel evaluates the same SPH force and entropy rates as
   global/traditional integration. Fine gravity events retain the coarse gravity
   contribution to the total acceleration; full-source events replace it.
3. Close each active gas hydro interval once using its new rates and elapsed
   positive duration. Apply the top gravity closing kick, synchronize the physical
   velocity predictor only at full-source gravity events, then apply the opening
   gravity kick. Signed finer gravity corrections never update entropy.
4. Merge hydro timestep proposals and the existing neighbor limiter, record the
   gas rung lower bound, and restore inactive source endpoints before continuing
   the gravity ladder. A gas particle opens its positive hydro half-step only when
   its final descending rung is known, immediately before its drift. Record the
   endpoint time and exact drift factor after that drift.
5. Migrate and checkpoint the complete predictor record. Source snapshots and
   source-time projections are transient and rebuilt after restart.

The shared neighbor limiter only adjusts active gas, as before this change. It
does not wake an inactive neighbor in the middle of its current interval. Adding
such wakeups would require closing the elapsed hydro interval and reconstructing
its stored future position and momentum before promoting its rung; changing the
rung alone is not sufficient. That separate formulation change is outside this
work. Chemistry, stars, PM and P3M remain outside the requested scope.

The separation of physical prediction from gravity's staggered momentum follows
GADGET-4's primary implementation. Its hierarchy stores a full gravitational
acceleration at the coarsest occupied gravity bin, and source prediction combines
that field with the hydro acceleration. See
[`kicks.cc`, source lines 357–363](https://wwwmpa.mpa-garching.mpg.de/gadget4/doxygen/kicks_8cc_source.html)
and [`predict.cc`, source lines 373–379](https://wwwmpa.mpa-garching.mpg.de/gadget4/doxygen/predict_8cc_source.html).
The retained neighbor tree alone does not remove this prediction requirement.

Validation must exercise multiple rungs, inactive sources stored at future
endpoints, positive entropy-update durations, and restart with pending hydro
half-kicks. The application regression checks the explicit clocks and accumulated
entropy, compares uniform-rung integration to traditional KDK, and resumes a
nonuniform blast checkpoint. Restart clocks, rungs and entropy remain exact;
small field-specific floating-point bounds cover changed neighbor summation order
when the transient octree is rebuilt. A self-gravitating polytrope checks
comparable gravity and hydro forces, timestep convergence, and bounded energy
and momentum error. The direct state fixture checks projection and
restoration against prescribed drift/predictor data. CPU/CUDA, MPI migration,
and timestep-convergence claims require their corresponding runtime results;
compilation or traditional-integrator results do not establish them.

## Validation

Completed checks and their limits are recorded in
[SPH octree validation](sph_octree_validation.md).

- Exact one-third boundary, retained topology, changed active counts and forced
  invalidation/rebuild.
- Moving/periodic cell bounds, mixed velocity signs, support growth and inactive
  large-support neighbors against exhaustive neighbor discovery.
- Leaf force results against the existing SPH reference, with mixed gas/DM
  buckets, leaf traversal ordering and nonuniform rungs.
- Global and traditional application runs, synchronized diagnostics, checkpoint
  restart, gravity-disabled SPH, CPU/CUDA parity and two-rank MPI where available.
- Build variants and focused regression suites; report actual validation and any
  unrun performance or long-run scientific comparisons.
