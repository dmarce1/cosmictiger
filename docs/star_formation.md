# GADGET-4 multiphase star formation

This module follows standard `COOLING` + `STARFORMATION` in official GADGET-4
revision [`6fb393b5e25907f2ff06211f67d341c5e40e90d1`](https://gitlab.mpcdf.mpg.de/vrs/gadget4/-/commit/6fb393b5e25907f2ff06211f67d341c5e40e90d1).
The controlling sources are
[`sfr_eos.cc`](https://gitlab.mpcdf.mpg.de/vrs/gadget4/-/blob/6fb393b5e25907f2ff06211f67d341c5e40e90d1/src/cooling_sfr/sfr_eos.cc),
[`starformation.cc`](https://gitlab.mpcdf.mpg.de/vrs/gadget4/-/blob/6fb393b5e25907f2ff06211f67d341c5e40e90d1/src/cooling_sfr/starformation.cc)
and the main-loop call ordering in
[`run.cc`](https://gitlab.mpcdf.mpg.de/vrs/gadget4/-/blob/6fb393b5e25907f2ff06211f67d341c5e40e90d1/src/main/run.cc).

The underlying model is Springel & Hernquist (2003). The implemented closure
retains the gas particle's bulk thermal energy and relaxes it toward an
algebraic multiphase target. It does not persist separate cold/hot masses,
energies, or chemical histories. The former immediate energy/composition
projection, Anninos networks, metallicity enrichment, kinetic winds, and wind
decoupling options have been removed.

## Gas evolution

For eligible gas at physical density `rho`, the model calculates

```text
t_star = t_star_0 sqrt(rho_threshold / rho)
A = A_0 (rho / rho_threshold)^(-0.8)
u_hot = u_SN / (1 + A) + u_cold
u_eff = (1 - x) u_hot + x u_cold
```

The cloud fraction `x` comes from the hot-phase equilibrium cooling time,
evaluated at the bulk physical density. GADGET-4's atomic H/He equilibrium
cooling, fixed hydrogen mass fraction 0.76 and `gamma=5/3` are used throughout.
Cold-cloud energy uses the neutral mean molecular weight; supernova energy
and the `10^4 K` calibration energy use the fully ionized mean weight.
When deriving the physical threshold, the calibration hot energy is `u_SN/A_0`,
as in GADGET-4's initialization routine.

When current gas energy is below `u_eff`, it relaxes exponentially on
`t_relax = tₛₜₐᵣ (1-x)/(beta x (1+A))`. When it is above `u_eff`, ordinary
implicit cooling applies with `u_eff` as a floor. Gas whose resulting energy
exceeds `1.01 u_eff` has zero star-formation rate. The effective star-formation
timescale is bounded below by the selected interval after calculating the
cloud fraction. Ordinary gas below the threshold uses the same equilibrium
cooling module. The configured minimum physical energy is enforced.

Eligibility requires the physical density threshold and, in cosmological runs,
the comoving overdensity threshold. `criticalPhysicalDensity=0` requests the
GADGET-4 threshold calibration; a positive value directly specifies hydrogen
nuclei per cubic centimetre. `criticalOverDensity` multiplies the mean comoving
baryon density. It is not a multiple of the physical threshold.

## Timestep ordering and conversion

At the end of a selected SPH interval, GADGET-4's caller first samples star
creation using the previously stored star-formation rate. Surviving gas then
receives the full-interval cooling/multiphase update, which stores its new rate.
CosmicTiger follows that ordering. There are no opening source half steps,
accumulated half-step formation optical depths, wind clocks, or source-driven
outer-step retries.

For chronological midpoint SPH the source is applied after the second drift of
each particle's completed interval, before checkpointing or selecting its next
gravity interval. Dyadic integration selects old rungs at or above the endpoint's
closing rung; global integration supplies the actual completed step. A coincident
coarser hydro midpoint executes afterward in the next boundary pass. A source is
never called merely to initialize forces, refresh output, or reload a checkpoint.

The endpoint density solve supplies the thermal conversion. The backend consumes
temporary `E = a²u` and returns an updated energy; the bridge stages canonical
float entropy `A = (gamma-1) E rho_c^(1-gamma)` for the required `gamma=5/3`.
Every rank must prepare and validate successfully before any rank commits.
The source energy ledger uses the energy reconstructed from the staged entropy,
including its storage rounding. Only entropy and the electron estimate change
for surviving gas. The fixed-endpoint physical duration `a Δtau` retains the
existing first-order source splitting; it is not a second-order cooling scheme.

After sources, an all-gas zero-kick estimate refreshes hydro timestep proposals
and the global timestep limit, including neighbors whose sound-speed bounds
changed. It does not advance committed viscosity history. As with inviscid SPH,
gas rungs still change only at common all-particle boundaries: abrupt heating or
new contacts do not trigger immediate wakeups or rollback between boundaries.
This policy does not guarantee an immediate CFL response to new source heating.
All evolving alpha, including Morris–Monaghan, advances at each completed own
interval endpoint, before gas rung changes and before source conversion. CD, Hu
and Price also commit divergence history there; Morris–Monaghan stores no such
history. These updates reuse the endpoint density/gradient pass already needed
by cooling or star formation. Startup does not advance alpha. Ordinary midpoint
force evaluations use the last endpoint's alpha. For CD, Hu and Price, Balsara
uses the same reconstructed velocity gradient at startup and endpoints, and its
factor is retained until the next reconstruction. This replaces their old
uncorrected SPH divergence/curl estimate with the matrix-corrected estimate,
without an additional midpoint gradient pass. Constant-α and Morris–Monaghan
runs retain direct midpoint divergence/curl sums when Balsara is enabled.
The gradient-history endpoint difference is centered on the completed interval's midpoint but
available afterward; its alpha is used by the following hydro kick, giving the
accepted first-order viscosity timing. Midpoint pressure and viscous force
evaluations retain their existing schedule. The old live rung gives the completed
dyadic duration; every rung mutation rescales the saved scalar by
\(2^{r_{\rm old}-r_{\rm new}}\), so no previous-rung byte is needed. The four
pending viscosity diagnostics are temporary density-event scratch, released
before source conversion and not allocated by the post-source bound estimate.

For gas mass `m`, the event probability is `1-exp(-SFR dt/m)`. A successful event
converts the whole gas particle into a star; gas mass is not continuously
removed before the event. Position, velocity and particle mass are retained,
as is the old rung. The next ordinary gravity pass completes its old closing
kick and selects/opens its new collisionless interval; no saved gravitational
acceleration or additional newborn-star kick is used. All pair impulses involving
the converting particle's completed interval have already been delivered.
Conversion invalidates the SPH topology and caches, including on other ranks,
so the next walk rebuilds even below the ordinary activity threshold.
The star continues its gravitational evolution. Birth time is the scale
factor in cosmological runs and elapsed physical seconds in noncosmological
runs, matching GADGET-4. No kinetic wind kick is
part of the standard implementation.

CosmicTiger retains its particle-identity-based seeded random stream for
parallel execution. This is an integration convention, not a claim of identical
random draws or trajectories to GADGET-4. Thermodynamic and conversion state
must follow sorting, migration and restart together; old projection-model
checkpoints have a different state contract.

Generated cosmological IDs come from the initial source before displacement or
migration, independently of the number of localities. The source index is the
global lattice cell in x/y/z order, with a second index for each close-packed
phase, or the glass tile and particle index within `glass.bin`. Dark matter uses
`2*source+1` and its gas partner uses `2*source+2`. Fixed scenarios assign source-order
IDs on rank zero before distribution. Distributed initial conditions must supply
IDs rather than deriving them from rank offsets. This fixes the random stream
associated with each source particle; it does not guarantee identical floating
point trajectories or event times across decompositions.

## Configuration

Build with `COSMICTIGER_ENABLE_SPH=ON`,
`COSMICTIGER_ENABLE_CHEMISTRY=ON`, and `COSMICTIGER_ENABLE_STARS=ON`.
Choose traditional, hierarchical, or global integration with `hydro.method=sph`.
Hierarchical integration requires FMM gravity; traditional and global also support
disabled gravity. Mesh gravity with SPH remains deferred. A starting configuration is
[`configurations/star_formation/lcdm.ini`](../configurations/star_formation/lcdm.ini).

```ini
[hydro]
enabled = true
heatCapacityRatio = 1.6666666666666667

[chemistry]
enabled = true
initializationTemperature = 10000
minimumSpecificEnergy = 0

[starFormation]
enabled = true
beta = 0.1
evaporationFactor = 1000
starFormationTimeGyr = 2.1
coldTemperature = 1000
supernovaTemperature = 1e8
criticalPhysicalDensity = 0
criticalOverDensity = 57.7
seed = 42
```

`starFormationTimeGyr` is a physical timescale in Gyr. It corresponds to
GADGET-4's `MaxSfrTimescale` after converting that code's configured time unit;
a number printed in a GADGET-4 example is not automatically a number in Gyr.
All threshold/temperature/evaporation parameters retain the meanings of
`CritPhysDensity`, `CritOverDensity`, `TempClouds`, `TempSupernova`, and
`FactorEVP`. `beta` corresponds to `FactorSN`.

`chemistry.*` retains its historical group name but selects equilibrium cooling;
see [the cooling documentation](primordial_chemistry.md) for initialization,
physical energy floor and supported physics. External UV is disabled and no
radiation file is required. The background
cosmology's helium abundance remains independent of GADGET-4's fixed cooling
composition.

## Validation

Core equilibrium/calibration and relaxation fixtures, production CPU/CUDA
source fixtures, particle conversion/lifecycle fixtures, and application
restart tests cover different parts of the integration. Agreement with the
pinned equations does not establish identical SPH force discretization,
random streams, or galaxy statistics between codes. Report the actual builds,
backends and application runs used when assessing a change.

The endpoint regression checks mixed-rung source counts and durations, one
gradient-history update before sources, and no further history update or mechanical
kick from the post-source estimate.
The particle fixture compares a converted star with an identical existing star
through a nonzero ordinary gravity update for each compiled integrator.
Application restart tests include a controlled saved rate that guarantees
conversion, testing the final source event and subsequent evolution with an empty
gas population. Source payloads already carry particle ID, event counter, birth
time, saved specific formation rate and elapsed physical time; this integration
adds no persistent per-particle fields or force-rate arrays.

The SPH payload uses base version 140000 for compact viscosity storage and packed
gas flags. One `SphParticleFlags` byte follows density: convergence is bit 0,
semiactivity bit 1, and proposed rung bits 2–7. Rung codes 0–15 are valid; 63
decodes to `-1`, and the remaining codes are rejected. This replaces three byte
arrays in memory, checkpoints and migration, saving two bytes per gas particle.
Callers use explicit setters rather than mutable references to individual bits.
Evolving alpha remains two-byte binary16.

The Balsara factor is one byte, rounded to the nearest multiple of \(1/255\);
force/source staging uses float. It remains transient and is recomputed after
restart or migration. Balsara is always on for Morris–Monaghan and defaults on
for constant α.
Alpha and Balsara together use three bytes per gas particle. Constant viscosity
has no evolving alpha array. Source counters and elapsed source time retain their
existing meanings; no viscosity timestamp is added. Older SPH payload versions,
including every base below 140000, are rejected.

The source-history lifecycle fixtures passed on CPU, CUDA and two MPI ranks for
commit `1bc12b5f6`, the signed-significand/base-90000 representation, as recorded
in [the pair-midpoint design](sph_pair_midpoint_plan.md#validation). That is
historical validation. The same linked design records the later base-120000
endpoint-history/Balsara work and successful final packed-flag/base-130000
validation. The final CPU selection passed all ten CTests, including source
completion on CPU and MPI; both selected CUDA application tests also passed.
Focused kernel, restart, reordering and two-rank migration checks are detailed
there. These checks do not establish long-duration star-formation statistics,
performance or full-suite coverage.

The following results are historical validation of commit `9ccd34a85`, using
checkpoint base 80000. They do not validate subsequent compact-history
representations. On November 4, 2026, after that
compact-history update, source timing and conversion fixtures passed on CPU,
CUDA and two MPI ranks. The star-formation
application passed traditional, hierarchical and global evolution/restart cases
on each backend. Every controlled-conversion case converted all 512 gas
particles and restarted exactly with no gas remaining. CPU and CUDA cooling
evolution/restarts passed all three integrators. The tests also compare newborn
and existing stars through their next nonzero ordinary gravity update, and
verify source completion does not change committed viscosity state.

Evidence is in `/tmp/sph_compact_cpu_lifecycle.log`,
`/tmp/sph_compact_cpu_apps.log`, `/tmp/sph_compact_cuda_apps.log`, and
`/tmp/sph_compact_mpi_apps.log`, with structured results under the corresponding
build trees' `tests/star_formation_application*/evidence.json`. These are source
and scheduling regressions, not long-duration galaxy-statistics validation.
