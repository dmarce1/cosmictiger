# Periodic particle mesh multipoles

PMMM is CosmicTiger's gravity solver. `COSMICTIGER_ENABLE_GRAVITY` (default
`ON`) compiles it in `./build.sh` or with CMake feature options; every
configuration builds one `cosmictiger` executable containing the selected
capabilities. The default configuration enables PMMM gravity and SPH, particle
chemistry, hierarchical gravity with midpoint SPH, and potential expansion degrees four
through six. A gravity-only configuration is:

```sh
cmake -S . -B build_pmmm -DCOSMICTIGER_ENABLE_SPH=OFF \
  -DCOSMICTIGER_EXPANSION_DEGREE=6
cmake --build build_pmmm --parallel 12
```

PMMM uses a periodic uniform mesh, spherical multipoles about cell centers and
FFT convolution for distant interactions. Neighboring cells use the adaptive FMM
tree walk, including direct particle-pair interactions at close leaves.
PMMM-FMM is the only gravity solver; `gravity.method.near` is retired.

The sole integration scheme is hierarchical short gravity with mandatory outer
long kicks and midpoint SPH, selected automatically.
`COSMICTIGER_EXPANSION_DEGREE` selects the single build-time potential degree
(default six). There is no runtime degree selector.

PMMM's forward, inverse, and cached Green-function FFTs run their parallel jobs
on the existing HPX worker pool through FFTW's threading callback. They share
the workers selected by `--hpx:threads` with the near-field tasks; FFTW does not
create a separate worker pool. Small transforms may expose fewer parallel jobs.
MPI still distributes the mesh between processes, with blocking MPI operations
kept on HPX's native I/O threads. The build still requires `fftw3f_threads` for
the callback API alongside `fftw3f_mpi` and `fftw3f`. Cosmology's independent
one-dimensional transforms remain single-threaded inside FFTW and are
parallelized by HPX. Their planning and destruction also run on native I/O
threads so waiting for the shared planner lock cannot block HPX workers.

```sh
./build_pmmm/cosmictiger \
  --gravity.meshSize=32 --gravity.openingAngle=0.5773502691896257 \
  --run.darkMatterParticleCount=32768 --run.gasParticleCount=32768 --hydro.enabled=false \
  --gravity.cuda=false --hpx:threads=12
```

The runtime settings can also be supplied in an INI file:

```ini
[run]
numberOfOutputs = 99 # includes both endpoint states

[gravity]
meshSize = 32
openingAngle = 0.5773502691896257

[hydro]
enabled = false
```

The former `gravity.method`, `gravity.pmmmNearMethod`, `gravity.fmmOrder`, and
`gravity.pmmmOrder` inputs are rejected on both the CLI and in INI files.
`--help` and the `build.*` startup fields report the enabled capabilities;
resolved option output reports the runtime selections.

`gravity.meshSize=n` requests `n` full-particle cells on each axis of the
normalized unit box, independently of the initial-condition particle count.
The default zero selects the smallest integer side with at most 32 particles
per cell on average, using the sum of the resolved dark-matter and gas counts,
after lattice rounding or external initial-condition loading. The side is raised
if needed to give every locality whole cells. Thus one million particles select
`32^3` cells (about 30.5 particles per cell). An explicit positive `n` overrides
this choice without rounding or prime-factor restriction; all three sides use
that same length, within indexing and memory limits.

`gravity.meshSize` also sets particle-domain cut granularity and is used by
every hierarchical stage, including empty stages. There is no separate coarse
mesh or active-fraction mesh scaling. The legacy `gravity.pmmmCoarseMeshSize`
CLI/INI key is accepted but ignored, so an old value cannot constrain a new
`gravity.meshSize` override. It is absent from resolved option output.

PMMM requires a power-of-two locality count, and the mesh must have at least
`2^ceil(log2(localities)/3)` cells per axis so every locality receives whole cells.

`gravity.openingAngle=theta` sets both the near FMM acceptance angle and the
PMMM near-free stencil; its default is `0.5773502691896257`, approximately
`sqrt(3)/3`, and its range is `(0,1]`. `gravity.pmmmNearWidth` is retired.
Assume mass fills each mesh cell of side `ell`, with enclosing radius
`r = sqrt(3)*ell/2`. The FMM cell-cell criterion accepts a far interaction only
when `d > (r_source+r_target)/theta`. Thus an integer cell offset `(i,j,k)` is
near exactly when

```
i*i + j*j + k*k <= 3/(theta*theta).
```

Equality stays near. This is a lattice sphere, approaching a spherical boundary
as theta decreases before periodic mesh saturation. Both Green subtraction and
near-cell enumeration use this same criterion. Smaller meshes visit each periodic
cell once, including the sink cell; even meshes use the positive half-box image
tie. Very small angles can include the entire mesh and do not require an integer
radius that could overflow. Small meshes can still have substantial periodic
finite-expansion error; accepting their size is not an accuracy guarantee.

The Cartesian FMM operators and adaptive tree acceptance evaluate the near-cell
region. Complete remote source-cell records feed local ghost trees; owned sink
cells run when their dependencies are ready. The spherical FFT result supplies
the distant interactions and periodic correction.

At an outer boundary, the endpoint kd-tree sort and owned-root construction
precede the long solve. Mesh moments come from those root multipoles. Solved mesh
locals are converted to Cartesian locals, translated down the same trees and
evaluated at their leaves. The immediately following opening short pass reuses
that forest explicitly, without another sort or owned-root build. No drift,
migration or particle permutation may intervene between the two passes.
Production has no direct spherical particle-to-mesh P2M or mesh-to-particle L2P
path. Their former wrappers survive only as independent test references.

`COSMICTIGER_EXPANSION_DEGREE` fixes the shared PMMM and near FMM potential degree.
The configured `gravity.openingAngle` stays fixed during cosmological evolution.
`compute.bucketSize` controls the near FMM tree bucket
size. Near FMM can reduce pair work in dense cells, but its additional
approximation and actual performance must be checked on the intended
distribution. With the near FMM capability enabled:

```sh
./build_pmmm/cosmictiger \
  --gravity.meshSize=32 --gravity.openingAngle=0.5773502691896257 \
  --compute.bucketSize=32 \
  --hydro.enabled=false --gravity.cuda=false
```

The compact gravity softening support must fit inside that near region:
the normalized support radius must not exceed the minimum box-to-box distance
among cells excluded from the stencil. For an excluded offset `(i,j,k)`, that
distance is `sqrt(max(abs(i)-1,0)^2 + max(abs(j)-1,0)^2 + max(abs(k)-1,0)^2)/N`.
The existing `gravity.smoothingLength` and `gravity.effectiveRadius` input
conventions still apply. The parser checks the exact finite-mesh minimum after
normalizing to box units; if every cell is near, only the existing half-box
support limit applies. The FMM tree walk retains the selected compact gravity kernel for
direct pairs, including distinct coincident particles, and exclude primary self.
PMMM accuracy is controlled by mesh size, expansion degree, and opening angle.
The opening angle controls both the near FMM and the mesh near/far split.

## Source organization

PMMM interfaces are in `cosmictiger/gravity/` and implementations in
`src/gravity/`. `solver.cpp` coordinates the solve; `fft.cpp` owns the
persistent transform/cache lifecycle; `mesh_exchange.cpp` transfers moments and
local expansions. `ghost_storage.cpp` owns the reusable source arena,
`ghost_exchange.cpp` handles whole-cell requests, and `cell_schedule.cpp` launches
ready cell work. `cell_tree.cpp` supplies exact mesh-cell geometry to the common
tree builder. `fmm_batch.cpp` adapts ready cells to the common traversal, and
`evaluation.cpp` combines and consumes the fields. `near_pp.cpp`/`.cu` provide the
alternative direct near field, and `pmmm_kernels.cpp` adapts the spherical operators.

Common Cartesian operators, tree construction, source views, and CPU/CUDA walks
remain directly in `gravity/`; the PMMM near FMM is their only user. Owned
particle arrays and sorting remain in `particles/`. See [source organization](source_layout.md) for
the complete ownership and generated-code layout.

## Expansion degrees and coefficient conventions

The PMMM far field and near FMM share a potential-degree convention and total-degree truncation.
`COSMICTIGER_EXPANSION_DEGREE=d` generates and compiles exactly one degree.
Supported degrees are two through
eight. For potential degree `d`, source multipoles have degree `S=d-1`, local
expansions have degree `L=d`, and a contraction retains only terms with source
degree plus local degree at most `d`. The internal historical order is
`q=d+1`, so the same limits are `S=q-2` and `L=q-1`.
This implementation does not retain the full
rectangular source/local contraction used by the experimental PMMM reference.

The spherical representation stores real and imaginary harmonic components
as real scalar fields. The channel counts are:

| Quantity | Scalar fields | At default potential degree 6 (`q=7`) |
| --- | ---: | ---: |
| Source multipoles | `d^2 = (q-1)^2` | 36 |
| Local expansions | `(d+1)^2 = q^2` | 49 |
| Cached harmonic Green filters | `(d+1)^2 = q^2` | 49 |

Let `h=1/n`. Cartesian source multipoles are translated from endpoint root
centers to mesh-cell centers and converted to spherical coefficients in mesh
units. A degree-`d` Green coefficient is the unit-box physical coefficient
multiplied by `h^d`. The inverse transform produces mesh locals, which are
converted to Cartesian tree-coordinate locals before tree descent.
`gravity.gm` multiplies the complete far-plus-near result once.

The contraction table expresses harmonic conjugation as signed real
coefficient products. Spatial Fourier coefficients are complex, but they
are multiplied without Fourier conjugation. Each inverse transform is
normalized by `1/n^3`; source moments already contain particle masses and
need no density or cell-volume multiplier.

## Periodic Green function and near field

The FFT Green filter contains the full periodic interaction outside the near
stencil and zero inside it, including the origin. The near FMM supplies both
the softened primary interaction and the regular Ewald correction for exactly
those source cells. Primary self interactions are excluded; finite periodic
self images remain.

The shared CPU/CUDA kick walk carries a second source checklist for the Ewald
correction. Generated Cartesian kernels use the same derivative, moment,
coordinate-scaling, and target-last conventions as the free-space kernels.
Near Ewald evaluation, M2L, local translation, and particle evaluation use float.
Particle pairs use ordinary minimum-image subtraction.

For minimum-image center displacement `d` in a unit box, the next image is at
`R2^2 = dot(d,d) + 1 - 2*max(abs(d))`. The regular correction's acceptance test
uses this distance with the sum of the source and sink radii. Across an
image-selection plane it also requires ordinary nearest-image separation,
so the primary and correction expansions cancel the same selected singular
image. Unresolved leaf pairs evaluate the regular correction directly.

The periodic uniform-background quadratic uses the same near/far mask.
An extra source FFT channel stores each cell's radial second moment, and an
extra Green/local channel carries the masked background coefficient. Their
convolutions supply the local curvature, dipole correction, and source-trace
potential term. A global background applied to all cells would double-count
the near contribution. Near Cartesian locals carry their constant Laplacian
separately; translation and particle evaluation include its quadratic, while
source nodes retain the radial second moment needed for the potential at
source degree two and above (`q >= 4`).

The potential follows the Ewald periodic gauge and retains finite periodic
self images. The sampled, near-subtracted kernel's Fourier zero mode is
retained.

## FFT distribution, cache, and particle ownership

Each cached mesh has an FFTW3-MPI communicator containing every locality.
All coefficient fields use interleaved batched real-to-complex or complex-to-real
transforms. Forward
plans produce transposed slabs, contraction operates directly on that layout,
and the inverse plan consumes it. There are no separate locality groups or
coefficient-group redistributions in this implementation.

The first solve at a mesh size creates three plans: source forward, Green-filter
forward, and local inverse. It also constructs and transforms the Green filters once.
Subsequent solves reuse those plans, buffers, and Green spectra. At the default
potential degree six, a normal solve performs 36 scalar-equivalent forward transforms and
49 inverse transforms through two batched execution calls; the 49 Green
transforms belong to initialization. This describes work, not measured speed.

Each mesh cache lasts until eviction, explicit PMMM destruction, or application
shutdown. Changing the requested mesh size, degree,
or opening angle requires explicit PMMM destruction.
The expansion degree remains fixed during an evolution.
Changing them between runs requires a new build only if the requested capability
or degree was not compiled.
Plans use `FFTW_ESTIMATE`; there is no persistent disk-wisdom cache. The
single-cell mesh uses serial batched identity transforms because FFTW-MPI
cannot create its plans; buffer layout and convolution are unchanged. MPI calls
run on native OS threads through HPX, use a communicator separate from HPX
parcel traffic, and require the application's existing `MPI_THREAD_MULTIPLE`
startup guarantee. FFTW planner operations share the process-wide planner
lock with other FFTW users.

PMMM uses the common recursive bisection for its particle domains, but
every cut lies on a mesh plane. Integer fixed-point comparisons preserve
these half-open planes even for non-dyadic mesh boundaries. The locality count
is a power of two; depth `d` of the bisection halves the locality range and cuts axis `d mod 3`.
Among the mesh planes that leave each child enough whole cells for the
cuts it still owes along that axis, the cut selects the plane whose particle
count comes closest to the balanced target, not the plane nearest the exact
balance coordinate. Ties prefer the plane nearest the middle of the parent.
Every stage uses the same mesh, so each cell is owned, with all its particles,
by exactly one locality. Balance is only as fine as these cells allow.
Gravity-free runs retain the
unrestricted continuous cuts. Particle migration and checkpoint reload preserve
the domains independently of the FFT slabs. Each solve sorts local particle
arrays in place into cell ranges, using parallel CPU partitioning and preserving
all attached particle fields. The hierarchical ladder keeps this ownership
exact for every active particle: a rung drifts only at its own level and is
migrated at that level before it next serves as a source.
The FFTW mesh retains its separate slab distribution, independent of particle
counts and particle-domain boundaries. Each cell's owner forms the cell's
moments about the mesh-cell center and sends them to that cell's FFTW slab
owner, which sums the packets before the FFT. After the inverse FFT, every
owner fetches the completed expansion for each of its occupied cells and
evaluates it on its own particles. Near-field sources never pass through the
slab owner: the owner of any cell is found from the particle-domain tree by the
cell center. Each locality asks the remote
owners of its neighbor cells for their particle counts, reserves ghost storage,
and then fetches each occupied remote cell once; each arrival permits its
source-tree construction. A source cell is copied once per recipient even when
several sink cells need it.

Ghost records occupy a separate source arena; owned particle arrays retain their
original size. The counts and storage are reserved before arrivals, so building
or filling one cell cannot invalidate a running walk's data. Each owner builds
its cell tree before serving the cell's final particle permutation. Rungs and
other source metadata are captured for the current stage before any sink kick
can change them. Every source cell, owned or ghost, contributes exactly one
source root, and periodic source copies retain the existing image selection.

All owned and ghost source trees are local, so near walks fetch no remote tree
nodes or particles. A cell's sink task waits only for its required neighborhood,
not for every incoming ghost cell. Ready sinks may run separately or in batches:
`gravity.pmmmBatchParticles` is a positive target sink count (default 4096).
A cell larger than the target runs whole in its own batch. At most the worker
count of transient batches run concurrently. Ghost records and cell trees are
cleared after the solve; the reusable ghost arena retains its allocated capacity.
Ghosts never enter migration, checkpoints, or owned particle counts. Neither mode permanently migrates particles into FFT slabs.

`PmmmDiagnostics::ghostParticles`, `ghostCells`, and `ghostBytes` describe the
last solve's remote source copies, remote source cells, and particle-record payload
bytes, summed across localities. Payload bytes exclude directory and transport
metadata. `nearRemoteTreeFetches` and `nearRemoteParticleFetches` count remote
fetches during the near walk; both must remain zero. Near-field preparation
communicates to obtain whole-cell ghost sources while ready cells can walk.
`nearBatches` and `peakBatchParticles` report batching, and
`completedWhileGhostsPending` reports sink cells completing while other incoming
cells remain outstanding. `storedAccelerationParticles` and
`storedPotentialParticles` expose retained solve-result arrays: both are zero
for the ordinary streamed kick path.

SPH couples to PMMM self-gravity on the hierarchical ladder (next section). Set
`hydro.enabled=false` explicitly for gravity-only cosmological PMMM runs, whose scenario
otherwise enables SPH when compiled.
Particle chemistry additionally requires `COSMICTIGER_ENABLE_CHEMISTRY=ON`
(the default). Setting it to `OFF` retains SPH without particle chemistry;
homogeneous cosmological recombination remains available.

Moment formation, Green construction, FFTs, spectral contraction, and local
expansion evaluation run on CPUs. `gravity.cuda=true` selects CUDA near-field
execution in a CUDA build and permits CUDA particle drift. For near FMM this
uses the existing FMM CPU/CUDA work distribution. Diagnostic fields named
`nearCpuAcceptedMultipoles` and `nearCpuDirectPairs` count CPU tree-walk work;
they do not claim to count GPU interactions.

The FMM walk groups whole sink cells toward `gravity.pmmmBatchParticles`.
CPU and CUDA use the existing FMM work distribution and reusable workspaces.

The local FMM path also compares the generated M2L arithmetic estimate with
direct PP work for geometrically accepted leaf pairs, choosing PP when its
estimate is smaller. Non-leaf acceptance is unchanged.
Batching follows the particle target without a separate 32-cell limit, so sparse
mesh cells can fill a useful batch.

The existing FMM near path accumulates potential and removes the softened
primary self term in single precision. That subtraction can lose precision
when the self potential is large compared with the remaining potential,
especially with a small softening support. This cancellation allowance is
separate from the opening-angle approximation and force convergence checks.

## Memory and timestep costs

FFTW fields use single precision; moment accumulation, spectral product sums,
and spherical evaluation use double precision. If FFTW reports complex
allocation counts `A_s` and `A_l` for the batched source and local transforms,
and a rank owns `N_local` real cells, the principal retained mesh storage is
approximately

```text
16*A_s + 24*A_l + 8*(q-1)^2*N_local bytes.
```

This includes two padded real arrays, source/local/Green complex arrays, and
the double moment array. For a balanced large mesh on `P` ranks, its leading
term is `[16*(q-1)^2 + 12*q^2]*n^3/P` bytes. Potential degree six (`q=7`) is about
1164 bytes per cell across the job, or roughly 2.3 GiB for a `128^3` mesh before dividing
among ranks. FFT padding, allocation imbalance, and plan storage add overhead.
These estimates exclude particle arrays, source copies, cell forests and metadata,
communication packets, output fields, and near-field CPU/CUDA workspaces.

The integration scheme is fixed: hierarchical short gravity, long kicks at outer
boundaries, and midpoint SPH. Gravity-free SPH uses the same midpoint scheme.

The hierarchical particle timestep ladder evolves gas and dark matter. Every ladder stage selects one
particle range as both its sources and sinks. Moment deposition, remote cell
packets, and near FMM all use that same selection; an inactive slow particle
does not enter a fast stage's force. Ascending/descending signed kicks and
per-rung drifts are retained. The gravity timestep criterion uses
the selected compact kernel's effective radius and the complete near-plus-far
acceleration.

Every outer and inner pass uses the configured full mesh, so frozen cell keys
have one meaning throughout an interval. Inner passes rebuild only the selected
source-and-sink cells, without depositing or transforming a mesh field.
`gravity.meshSize` determines particle-domain cut granularity.
`PmmmDiagnostics` reports `meshSize`, `activeParticles`, `totalParticles`,
and `evaluatedParticles`. Active particles are the selected sources and sinks;
inactive particles retain their velocities and rungs during a finer stage.

The hierarchical path calls `pmmmKick`: each completed cell's combined field is
consumed immediately for the kick and rung decision, without retaining
acceleration or potential arrays for all particles.
`analysis.saveAcceleration=true` and
`analysis.savePotential=true` independently request persistent particle
diagnostics; legacy `analysis.saveForce=true` requests both. These optional
fields are not checkpoint state. Calls to `pmmmSolve` alone collect full fields
and do not advance particles.

PMMM and adaptive FMM have different error and work controls even at the same
potential degree. Mesh refinement, opening angle, and degree convergence should be
checked on the intended particle distribution. Clustered cells can make the
direct near field expensive. No relative performance claim follows from the
coefficient counts or the FFT decomposition alone.

The `pmmm` regression compares the complete field against an independent
double-precision direct Ewald sum. A direct finite far-mesh expansion plus exact
near pairs separately checks the far truncation budget. The adaptive near FMM
uses different expansion centers, so it is checked against the complete-field
budget. Kernel tests independently check float point and M2L corrections,
including the source trace and local curvature. A `2e-5` numerical allowance
and a separate float self-potential cancellation allowance accompany the
fixture truncation limits below.

The off-center fixtures have the following potential and force truncation
budgets. These cover the unit-mass collisionless and mixed-mass SPH fixtures,
mesh sizes 5, 8, and 10, and several opening angles. They are regression limits
for these fixtures, not accuracy guarantees for arbitrary particle distributions.
Potential error is scaled by `max(1, abs(referencePotential))`; each force-component
error uses `max(1, norm(referenceAcceleration))`.

| Potential degree (`q-1`) | Potential budget | Force budget |
| ---: | ---: | ---: |
| 2 | `3e-2` | `6e-2` |
| 3 | `1.5e-3` | `2.5e-2` |
| 4 | `3.5e-4` | `7.5e-3` |
| 5 | `6e-5` | `2e-3` |
| 6 | `3e-5` | `5e-4` |
| 7 | `8e-6` | `1.25e-4` |
| 8 | `3e-6` | `5e-5` |

PMMM counts moment deposition, spherical far-field operators, spectral
contraction, FFTW plan execution, near interactions, and particle kicks.
Reported GFLOP/s divides these source-operation counts by the force-advancement
wall time, including communication, staging, and scheduling overhead. See
`flop_accounting.md` for the counting convention.

## Outer interval with frozen near membership

`gravity.pmmmOuterInterval = true` (PMMM-FMM,
the fixed hierarchical scheme) integrates the far field
on a mandatory outer interval and the near field on the
inner rung clock. One mesh transform runs per outer boundary; no inner pass
deposits, transforms or evaluates a far expansion.

### The split

PMMM's far/near split is geometric: a pair interacts directly when the two
particles' cell offset satisfies `i*i+j*j+k*k <= 3/(theta*theta)`, and through the mesh
otherwise. The outer interval freezes that assignment. When an interval opens,
every owned particle is sorted by position and its cell key is stored
(`particlesFrozenCell`); inside the interval the near passes group particles by
the stored key, wherever they have drifted, so

```text
H = T + V_far^P + V_near^P
```

is one fixed Hamiltonian split for the whole interval, with P the frozen
assignment. The pieces are smooth in the positions because membership does not
follow the particles, and the symmetric composition

```text
K_far(H/2) [ K_near(δ/2) D(δ) K_near(δ/2) ... ]  K_far(H/2)
```

keeps its second-order local error (Duncan, Levison & Lee 1998; Springel 2005,
eq. 33). Nothing limits how far a particle may move from its frozen cell; a
particle that left its cell keeps interacting directly with its frozen
neighbors and through the mesh with everything else. Near offsets are taken
from the image nearest the frozen cell center, which is exact as long as no
particle travels half a box inside one interval.

### Boundary equations

At a boundary with every particle at the synchronized position, with `S` the
near force and `L` the far force, `old` the frozen assignment being closed and
`new` the fresh assignment, `H` the outer durations and `δ` a particle's inner
durations:

```text
T_new   = L_new + S_new                      (one fresh far solve plus one near pass)
L_close = T_new - S_old                      (exact long complement of the old split)
Δp      = (δ_old/2) S_old + (H_old/2) L_close           closing
        + (H_new/2) L_new + (δ_new/2) S_new             opening
```

No far expansion is ever evaluated on the old assignment: the old long force
is the fresh total minus the old near force. A rung-zero particle (`δ_old =
H_old`) closes with `(H_old/2) T_new` exactly, independent of the assignment
change.

The hierarchical ladder needs no correction pass: its ascending levels close
the fast subsets on the old membership, the top level migrates, runs the far
solve, refreshes the membership and kicks every particle with `(H_old/2)(S_new +
L_new)` and `(H_new/2)(S_new + L_new)`, and the descending levels open the fast
subsets on the new membership. Only the boundary top level migrates; inside an
interval every particle stays with its owner, so each frozen cell keeps one
owner and `cellOwner` resolves from the frozen cell center.

### Interval selection

```text
h = min( η sqrt(a / (|L_new| N)) over particles, expansion bound, R )
n = max(1, floor(R / h))
H_new = R / n
R = time remaining to the next equally spaced output phase or run/stage end
```

with `N = gravity.meshSize`, so the length scale is one mesh cell (`1/N` in box
units) and `η = gravity.timestepEta`. The final interval is rounded upward to
an integer subdivision of the remaining output interval, as requested by
`pmSelectOuterInterval`; it may therefore exceed the raw proposal `h`.
`run.numberOfOutputs` includes both endpoints and sets `Nout - 1` equal intervals
in physical time for fixed scenarios or conformal time for cosmology. Every
enabled Silo phase is written; the old step count and Silo skip controls are retired.
Inner rungs come from the near force at the softening scale. The
potential energy at a boundary is the near potential of the boundary pass plus
the far potential of the far solve, counted once.

### State, restart and limits

Each particle carries a `uint64_t` frozen cell key (allocated only for this
mode) and, between the far solve and the boundary kick, the far acceleration in
`particlesLongAcceleration`. Checkpoints set header flag bit 23, write the keys
after the particle payload and the outer-interval extension (interval, start,
index) after the driver state; a restart inside an interval resumes the frozen
membership and a boundary checkpoint still holds the membership awaiting its
closing pass. `PmmmDiagnostics::transformCount` counts mesh transforms; the run
summary prints intervals, transforms and force passes.

The local FMM rebuilds the selected rung's cell trees at every short timestep.
Membership remains frozen, while each cell's bounds, centers, radii and moments
follow its current particles. Bounds can extend beyond the original cell and
across a periodic seam. Source and sink cell identities select periodic images;
the moving tree centers do not change near-neighbor membership. CPU and CUDA
walks use that same frame. No inner pass deposits moments on the mesh or runs an
FFT. The FMM near field uses this outer-interval scheme.

## SPH with PMMM self-gravity

Gas is kicked by the hierarchical (HOLD) ladder exactly as dark matter is: every
ladder stage selects one particle range as sources and sinks, and gas takes part in
the signed ascending and descending half-kicks and the long half-kicks of an outer
boundary (`src/gravity/pm_kick.cpp`). Two things differ, both because the SPH
midpoint integrator owns the gas rung and the hydro kick:

- Gas rungs change only at common boundaries. There the top level chooses each gas
  rung from the full gravity bound and the gas's hydro proposal,
  r = max(ceil(log₂(Δτ₀/Δτ_G)), r_old − 1, r_min, r_hydro), and the boundary rung
  balance closes it over touching neighbors. On all other levels the gas rung is held.
- The hydro kick of a rung-r interval is applied once, at its midpoint, for the whole
  interval (pair-once RK2). With dyadic rungs that midpoint is a boundary of every finer
  rung, so a coarse hydro event coincides with gravity half-kicks of finer gas. The
  hydro event must see the closing side only: gas opening half-kicks at that instant
  are collected (`gravityBeginDeferredGasOpening`) and applied right after it
  (`gravityFinishDeferredGasOpening`). An outer boundary never coincides with a hydro
  event, so no long half-kick is ever deferred.

Each SPH neighbor walk and pass is distributed with ghost leaves (see
`docs/sph_parent_tree_plan.md`). `tests/run_jeans_wave.py` (`ctest -R sph_jeans_wave`)
checks the coupling against the Jeans dispersion relation ω² = c²k² − 4πGρ₀ on 1, 2 and 4
localities; the measured frequency is within a few percent of ω_J at 20 particles per
wavelength, comparable to the SPH sound-speed error at that resolution, and the
localities agree to roundoff.

Gas also runs with the outer interval, in the cosmology and Plummer driver and in the
fixed-time scenario driver (`src/core/scenario_driver.cpp`), which follows the same
schedule: a boundary pass at every clock rollover, the outer interval as the rung base,
the exact interval clock and a terminal boundary. A boundary migrates every particle,
gas included (`domainsBegin(-1)`). A hydro rung
proposal is a rung number against the base step it was computed with, Δτ ≥ H 2^−r. The
first interval is chosen by the boundary pass of the first force step, after the initial
SPH timestep estimate, so that estimate takes the coarse interval as its base step;
every interval is at most that long, so those first rungs are never too coarse. At every
later boundary the proposals were made against the closing interval H_old, so a longer
opening interval H_new adds ceil(log₂(H_new/H_old)) rungs to them
(`hydroRungIntervalShift`).
`tests/run_sph_pmmm_interval.py` (`ctest -R sph_pmmm_interval_restart`) runs a small
cosmology with gas and dark matter and the Jeans wave, each on 1 and 2 localities,
restarts from a checkpoint inside an interval and requires the state after the next
boundary to match the uninterrupted run: bit for bit for the cosmology, within the
run-to-run roundoff for the Jeans wave, whose velocities are tiny. The Jeans frequency
with the interval agrees with the run without it to 0.03% of its error.

## Restart and regression checks

`tests/gravity/run_pmmm_interval.py` (`ctest -R pmmm_interval_cpu`) runs
the outer interval on the 512-particle Plummer fixture: one transform per
boundary plus the terminal pass and none inside, non-dyadic intervals selected
by the far acceleration bound, the interval clock, inner rungs from the near
force, frozen keys in every checkpoint, restart from inside an interval and
from a boundary, rejection of a restart with the mode switched off, and energy
and momentum against a split PMMM run with shorter outer intervals over the same time.

Checkpoints identify PMMM separately from FMM and record its requested mesh size,
domain mesh size (identical to the requested mesh), expansion degree (in the historical internal order
convention), opening angle, fixed FMM solver identity, and timestep compatibility parameters.
The executable must match the saved expansion degree and fixed integration
scheme; incompatible restarts are rejected. The saved near-field identity must
be FMM; PP checkpoints are retired.
Historical ordinary PM checkpoints are unsupported. Checkpoints do not store
FFTW plans or Green spectra: the first solve after restart reconstructs the
cache, then reuses it for later steps.
PMMM extension versions eight (constant angle) and nine (scheduled angle)
identify the complete periodic near/FMM split. The historical coarse-mesh slot
stores `gravity.meshSize`; the old near-width slot is reserved and zero.
Versions one through seven remain decodable but are rejected for restart:
their retained far fields use a different near/far split.
The saved angle, mesh sizes, expansion degree, FMM identity, and integration
scheme must match. Bucket size remains mutable rather than checkpoint identity.

The standalone `pmmm_kernels` tests check the numerical adapters and total
degree rule. Application tests compare complete fields against an independent
direct Ewald sum, exercise periodic boundaries,
and check PMMM-FMM cache reuse. With PMMM-FMM, a dense, unequal-mass fixture
proves that the CPU walk accepts multipoles and checks convergence when the near
opening angle is tightened. Distributed tests require ghost source exchange and zero remote
tree or particle fetches during the near walk. Repeated PMMM-FMM solves check owned counts, exact position bits, velocities, rungs, and saved
field sentinels after ghost cleanup. The suite includes valid split cell ownership,
empty localities, periodic seams, distinct coincident sources, exact cell-center containment near
the fixed-point upper boundary, arbitrary mesh sides, and CPU/CUDA agreement when
CUDA is requested. Streamed-stage tests compare signed kicks against an
independent Ewald sum over only the active sources, require unchanged inactive
sinks and no retained full-field result arrays, and exercise single-cell and
batched scheduling. A distributed observer gate holds one requested ghost
cell until an independent sink completes, checks that dependent sinks wait,
and compares the resulting fields with the ungated solve.
The evolution regression checks hierarchical clocks,
exactly one scheduled drift per active particle, current restarts, historical
PP-extension rejection and incompatible restarts. An endpoint regression checks
Cartesian tree descent against the direct spherical test oracle and verifies
that the immediately following short pass builds no additional owned roots.
The arbitrary-side force matrix checks `1, 2, 3, 5, 6, 7, 11, 12` against
an independent finite-expansion reference with the near tree walk. Source
translation checks cover orders 3–9 and both fixed-point widths; distributed
domain fixtures cover sides 5, 10 and 30, with checkpoint domain reload. Evolution and checkpoint regressions use a `30^3` full mesh.

A small CPU application regression can be selected explicitly:

```sh
./build_pmmm/cosmictiger --run.test=pmmm \
  --gravity.meshSize=8 \
  --run.darkMatterParticleCount=512 --run.gasParticleCount=512 --hydro.enabled=false \
  --gravity.cuda=false --hpx:threads=2 --hpx:bind=none
```

Test definitions describe coverage; successful CPU, CUDA, and MPI execution
must be established separately for the build being used.

The near-FMM cell forest reserves at most two local node slots per owned or
ghost source particle, plus domain joins and worker allocation-line slack. A
node slot costs `sizeof(TreeNode) + fmmPositionedMultipoleStride(order)` bytes;
the latter is the positioned multipole record padded to a 16-byte boundary.
Remote deserialized moments use a separate managed-memory chunk pool and do
not consume local node slots. The implementation checks the combined forest
capacity against the signed `TreeId` index range before allocation.

## Cosmological opening-angle schedule

`gravity.openingSchedule` accepts comma-separated `redshift:theta` pairs, for
example `49:0.41,20:0.51,2:0.61`. An empty value or `none` uses the constant
`gravity.openingAngle`. Scheduled angles must lie in `(0, 1]`; redshifts must
exceed `-1` and cannot repeat. Scheduling requires enabled gravity and the
`cosmology` scenario. All scheduled stencils must contain the softening support.

The initial angle comes from the entry nearest the original starting redshift
(ties choose the higher redshift). Expansion applies crossed thresholds in
order; contraction replays the same intervals backwards. A crossing inside a
PMMM interval takes effect at the next synchronized outer boundary, after the
old-membership closing ladder and before migration and the fresh far solve.
Multiple crossings before a boundary resolve to the angle at that boundary.
The domain decomposition is selected once to accommodate the whole schedule.
A changed angle invalidates retained FFT plans and Green spectra; the boundary
rebuilds them for the new near stencil.

Scheduled checkpoints store the normalized schedule, original start redshift,
and angle active in the frozen interval. Restart requires the same schedule
and restores that active angle before any force pass, even when the current
redshift has crossed a threshold since the interval opened. Constant-angle
checkpoints use version eight; scheduled checkpoints use version nine.
