# Scenario configurations

Select a physical simulation with `run.scenario`. Each scenario has its own
initializer under `src/scenarios/` and header under `cosmictiger/scenarios/`.
The evolution driver calls gravity, hydrodynamics, and cooling according to the enabled
physics. `run.test` is reserved for developer regressions and benchmarks.

Each INI below contains every application option except the top-level `help`
and `configuration` options, including commented alternatives for radius and
box-length inputs and an optional position-width assertion.
Leave `compute.positionType` commented
to use the width selected by `COSMICTIGER_POSITION_TYPE` when building. An explicit
position width must match the executable.
The files are independent: no other INI is loaded or inherited. Sections map to
CLI groups, and explicit CLI values override values in the selected file.
[default.ini](../default.ini) is a supplied cosmological configuration; it and
the cosmology scenario below enable GADGET-4 equilibrium cooling explicitly.
The parser defaults leave hydro, cooling and star formation disabled.

CMake `COSMICTIGER_ENABLE_*` options select which algorithms are compiled.
`gravity.method.far=fmm|pmmm`, `gravity.method.near=pp|fmm` (PMMM only), and
`run.timeIntegrator=traditional|hierarchical|global` select the run. A single
eligible compiled algorithm is automatic; multiple choices require CLI/INI
selection. Restart can recover omitted selections from the checkpoint.
The supplied configurations choose traditional integration and FMM when gravity
is enabled. Override those selections for a build containing different algorithms.

`COSMICTIGER_EXPANSION_DEGREE=4-6` compiles potential degrees 4, 5, and 6 by default.
`gravity.expansionDegree` chooses a compiled degree and defaults to the largest.
The implementation's historical order is one greater than potential degree.
Global FMM retains `gravity.openingAngle` and `gravity.openingSchedule`.
PMMM requires an explicit valid `gravity.meshSize`: a power of two at least
`2*gravity.pmmmNearWidth+1`; override the supplied zero with, for example,
`--gravity.meshSize=32 --gravity.method.far=pmmm --gravity.method.near=pp`.
`gravity.pmmmCoarseMeshSize` (default `meshSize/8`) is the coarsest hierarchical
stage mesh and the granularity of PMMM particle-domain cuts; PMMM runs need a
power-of-two locality count.

Traditional and global integration support SPH; `hydro.method` accepts only `sph`.
Hierarchical integration requires a gravity-capable build and advances collisionless
particles only; SPH runs reject it. Global integration advances all particles using the smallest
raw allowable timestep across all particles and localities, without power-of-two
rounding; final-time and scenario-stage boundaries can shorten that interval.
To run a collisionless cosmology, set
`--hydro.enabled=false --chemistry.enabled=false --starFormation.enabled=false`.

| Configuration | `run.scenario` | Enabled physics | Output directory |
| --- | --- | --- | --- |
| [plummer.ini](plummer.ini) | `plummer` | Gravity; dark matter only | `./output/plummer/` |
| [sod.ini](sod.ini) | `sod` | SPH | `./output/sod/` |
| [blast.ini](blast.ini) | `blast` | SPH | `./output/blast/` |
| [rayleigh_taylor.ini](rayleigh_taylor.ini) | `rayleighTaylor` | Gravity and SPH | `./output/rayleighTaylor/` |
| [kelvin_helmholtz.ini](kelvin_helmholtz.ini) | `kelvinHelmholtz` | SPH | `./output/kelvinHelmholtz/` |
| [disc.ini](disc.ini) | `disc` | Gravity and SPH | `./output/disc/` |
| [polytropic_star.ini](polytropic_star.ini) | `polytropicStar` | Gravity and SPH | `./output/polytropicStar/` |
| [hydro_wave.ini](hydro_wave.ini) | `hydroWave` | SPH | `./output/hydroWave/` |
| [gresho.ini](gresho.ini) | `gresho` | SPH, Gresho vortex | `./output/gresho/` |
| [cosmology.ini](cosmology.ini) | `cosmology` | Gravity, SPH, and GADGET-4 equilibrium cooling | `./output/cosmology/` |
| [glass.ini](glass.ini) | `glass` | Repulsive gravity with damping | `./output/glass/` |
| [lcdm_present.ini](lcdm_present.ini) | `cosmology` | Gravity, SPH, equilibrium cooling, and star formation; 99 coarse intervals | `./output/lcdm_present/` |
| [star_formation/lcdm.ini](star_formation/lcdm.ini) | `cosmology` | Gravity, SPH, equilibrium cooling, and star formation; CPU, 64 coarse intervals | `./output/star_formation/` |

For Gresho, `initial.closePack=true` uses a periodic face-centered cubic lattice
with four particles per conventional cell. `run.gasParticleCount` is rounded
up to the next `4*n^3` population. The false setting uses one particle per cell. The cosmological
initializer's existing two-particle lattice is body-centered cubic.

Run from a build directory containing the executable and any runtime inputs:

```sh
./cosmictiger --configuration=../../configurations/sod.ini --hpx:threads=12 --hpx:bind=none
```

From the top-level `release/` build directory, select the maintained present-day
preset directly:

```sh
./cosmictiger --configuration=../configurations/lcdm_present.ini --hpx:threads=12 --hpx:bind=none
```

Prefer referring to the maintained file instead of copying it into a build
directory. Local copies do not update when the program's options change.
The two star-forming LCDM presets use different cosmological parameters,
coarse-step counts, and output directories; both are intentional examples.
For older local configurations, remove the retired network, ion/molecular
fraction, chemistry subcycling, wind, and metal options. The current cooling
section contains `enabled`, `cuda`, `initializationTemperature`, and
`minimumSpecificEnergy`; see [cooling configuration](../docs/primordial_chemistry.md#configuration)
and [star formation](../docs/star_formation.md). The supplied initialization
temperature is 10000 K; zero preserves initial-condition thermal energy.

Each module selects CUDA independently. The default configuration uses CPU
backends; the GPU examples enable the relevant `gravity.cuda`, `hydro.cuda`,
`drift.cuda`, or analysis selectors explicitly. Set every enabled `.cuda` selector
to `false` for a CPU run. Gravity tuning is under `gravity.cuda.*`.
`sort.cuda`, `analysis.tracers.cuda`,
`analysis.slice.cuda`, and `analysis.views.cuda` must remain `false`.
HPX thread and locality options
stay on the command line. The examples use nominal resolution 32, except the
blast and disc profiles, which use 64. Override it with
`--run.darkMatterParticleCount=262144 --run.gasParticleCount=262144`. These are starting resolutions for
experimentation, not convergence results.

## Output

Every scenario configuration writes generated simulation products beneath its
`output.directory`, including particle Silo snapshots, text diagnostics,
checkpoints, and analysis products. Its lightcone directory is a subdirectory
of that location. Relative paths begin at the launch working directory; input
files such as existing glass loads retain their input-path meanings.

Restart with the same scenario configuration and `--checkpoint.readIndex=N`.
The selected output directory must contain `checkpoint.N`. Scenario checkpoints
record the scenario and requested duration; older checkpoints without this
metadata require their original executable to resume.

All examples enable full-box particle Silo output at every synchronized coarse
step. `output.silo=false` disables particle snapshots while retaining other
selected diagnostics. `analysis.views`, `analysis.power`, `analysis.groups`,
`analysis.tracers`, and `analysis.slice` select their respective products.

`output.siloInterval` counts synchronized coarse steps, not adaptive substeps or
wall-clock seconds. Finite-duration scenarios retain initial and final snapshots
even when the final cycle is not a multiple of the selected interval. Star
relaxation and subsequent evolution use successive snapshot cycle numbers.

The native output settings are `output.units=code` and
`output.frame=comoving`. CGS and cosmological unit views and proper-coordinate
frames are also available; see [units and frames](../docs/units.md). Output
bounds remain normalized box coordinates and must satisfy
`0 <= min < max <= 1` in each dimension.

## Durations and initial conditions

For finite-duration scenarios, `run.endTime=0` selects the initializer's default
duration. A positive value sets the total duration and scales the stages of a
multi-stage scenario proportionally. `run.numberOfSteps` sets the total coarse
interval count, divided proportionally between stages. Stage boundaries are
rounded to that common coarse clock so pending SPH half-steps use a consistent
timestep across transitions. `run.maxIterations`
bounds the adaptive evolution steps.

| Scenario | Default duration | Coarse intervals |
| --- | --- | --- |
| Sod | 0.09 | 20 |
| Blast | 0.005 | 16 |
| Rayleigh–Taylor | 10 | 256 |
| Kelvin–Helmholtz | 10 | 1024 |
| Disc | 16 | 1024 |
| Polytropic star | 10 then 100 dynamical times | 25 then 250 |
| Hydro wave | 0.5 | 64 |

The Plummer configuration initializes only collisionless dark matter and uses
the gravity driver; it replaces the former legacy Plummer hydro test. Glass
generation is a special single-locality relaxation and saves `glass.bin` under
its output directory. To use that glass for cosmological initial conditions,
copy the file to `./glass.bin` and set `initial.useGlass=true`. The exchange
format stores a native 32-bit side length followed by three `side^3` Q0.32 arrays,
including in fixed64 builds; the reader rejects inconsistent sizes. Glass
generation requires `initial.closePack=false`.

Rayleigh–Taylor retains periodic boundaries and its prescribed downward
acceleration, with self-gravity enabled. It is not a wall-bounded hydrostatic
benchmark. Its inherited transverse velocity seed is
`0.025 * [1 + cos(4 phi) + 0.5 cos(phi) + 0.5 cos(7 phi)] * sin(2 pi z)`,
where `phi = 2 pi (x - 0.5)`. This is the multiharmonic profile retained from
commit `d8e95728` and mirrored about the midplane in `5e926504`; the `0.1`
amplitude parameter is a scale, with peak velocity `0.075`, and does not
specify a nonnegative product envelope. The disc retains its prescribed central
potential alongside self-gravity. Its initial specific internal energy is `1e-4` in code units,
independent of resolution, replacing the legacy near-zero pressure. This keeps
the fractional thermal-change timestep bound usable as viscous heating begins,
while the thermal energy remains small compared with the orbital kinetic energy.
The star uses a Lane–Emden polytrope of index 1.5: the first stage uses damping
and zero artificial-viscosity alpha, while the second uses no damping and alpha
0.05. These stages select constant viscosity, with beta equal to twice alpha.
The star requires self-gravity. Its pressure profile has polytropic index 1.5,
while its internal energy uses the configured `hydro.gamma`. The blast deposits
unit total energy on the realized particle lattice and its reference solution
uses the configured `hydro.gamma`.

The cosmological configuration sets `run.darkMatterParticleCount` and
`run.gasParticleCount` independently. Each count rounds up to a complete
cubic lattice (or `2*n^3` for cosmological close packing). Class masses preserve
the total gas and dark-matter weights set by `omegaB` and `omegaC`.
It enables GADGET-4 equilibrium atomic cooling; see
[the cooling options](../docs/primordial_chemistry.md). A collisionless
cosmological run can disable both `hydro.enabled` and `chemistry.enabled`.

Cosmological runs require zero spatial curvature. The application rejects a
numeric `cosmology.omegaLambda` unless it closes the flat density budget with
matter and radiation; `auto` derives that value. The former closed, open, and
curved ΛCDM presets have been removed.

The [critical.ini](critical.ini) preset uses 32³ reference particles with a
reference mass of `3.58e9` solar masses, CPU execution, and 100 particle snapshots
at equal conformal intervals. Its derived comoving box side is about
6.58072 Mpc/h. It has `Ωm=1−Ωr`, `ΩΛ=Ωk=0`, and expands to `a=100`;
its physical expansion speed `da/dt` approaches zero at late times.

The `lcdm_present.ini` and `lcdm_ten_ages.ini` presets use `h=0.674`,
`Ωb=0.0224/h²`, and `Ωc=0.12/h²`. The critical preset increases cold matter to
close the flat density budget with zero Λ. These presets include photons and massless neutrinos with
`Ωr≈9.20939e-5`; the historical massive-neutrino density is not added to cold
matter. They use `sigma8=0.8607` with the native flat initial-condition solver.

The optional [lcdm_ten_ages.ini](lcdm_ten_ages.ini) continues the standard model
to ten times its present age, recalculated with the current radiation model.
Its present age is about 13.81146 Gyr and its final scale factor is about
1291.70. It uses the same reference mass, resolution, and snapshot count as the
critical preset, with a derived comoving box side of about 9.68717 Mpc/h.

These files contain complete options and describe their assumptions in their
headers. Their positive `run.endTime` values are **conformal code durations**,
not years, and are normalized to the reference mass and resolution in each
file to preserve the stated physical endpoint. To keep a fixed box while
changing resolution, remove or comment out
`units.massSolar` and set `units.boxLength` to the comoving side length in Mpc/h.
For example, `run.darkMatterParticleCount=2097152` and `units.boxLength=100` give a
100 Mpc/h box with 128³ collisionless particles when `initial.closePack=false`.
The reference mass is derived automatically from the resolution and cosmology.
Alternatively, keep `units.massSolar` in solar masses without `/h`; changing the
lattice from 32 to `N` at fixed box and cosmology then requires multiplying the
mass by `(32/N)^3`. Supplying both mass and box length is an error, even across
CLI and INI inputs. See [units and frames](../docs/units.md) for the reference
count and hydro mass conventions. Changing the box or cosmology also requires
recalculating the duration. See
[background evolution](../docs/cosmology.md) for stopping, signed Λ input,
accuracy controls, and restart behavior.

Gravity uses a fixed comoving support radius, with a Ferrers degree-4 density
kernel in these configurations. SPH has independently selected kernels and
variable smoothing radii; the examples use cubic splines and 42 weighted
neighbors. All SPH runs use the [variational pressure
operator](../docs/variational_sph.md), including chemistry and star-formation
examples and `default.ini`.
Every project INI lists all current application
options, with mutually exclusive alternatives commented where appropriate.
Cooling initialization uses `chemistry.initializationTemperature`; zero preserves
initial-condition energy. Set gravity radius through `gravity.effectiveRadius`,
`gravity.smoothingLength`, or the legacy `gravity.softening` alias. Alternative
radius representations remain commented to avoid conflicting inputs.

## Scenario initial-condition groups

Shared controls remain in `[initial]`: `cuda` and `closePack`. Scenario-specific
controls now use nested categories:

| Previous key | Current key |
| --- | --- |
| `initial.gadget4Restart` | `initial.cosmology.gadget4Restart` |
| `initial.useGlass` | `initial.cosmology.useGlass` |
| `initial.twoLpt` | `initial.cosmology.twoLpt` |
| `initial.plummerRadius` | `initial.plummer.radius` |

Update older local INIs/CLI commands to the new keys; the previous keys are no
longer registered. The same dot hierarchy appears in the Silo CLI metadata.
Scenarios without dedicated controls continue to use their fixed initial profiles.

### Kelvin–Helmholtz packing

```ini
[initial.kelvinHelmholz]
packing = CV/C
```

The category spelling is `kelvinHelmholz` as requested; the existing scenario
selector remains `run.scenario=kelvinHelmholtz`. The packing's first part is the
high-density region and its second is the low-density region. C, F, E and V mean
cell centers, face centers, edge centers and vertices. Letter order and case do
not matter. Any punctuation or whitespace separator is accepted, for example
`--initial.kelvinHelmholz.packing=CEV-cV` in 2D; quote whitespace or shell
metacharacters. Startup prints the normalized spelling.

The reusable `cell_lattice::CellLattice` class accepts every nonempty subset of
C/F/E/V. `appliesToDimension`, `sitesPerCell`, `planesPerCell`, `offsets`,
`isUniform` and `generate` expose its geometry. C/V apply in 1D–3D, E in 2D/3D,
and F in 3D. Shared boundary sites are emitted once, in half-open cells.
Uniformity means every site is related by a symmetry of the infinite periodic
point set, so all radial neighbor shells agree. Nonuniform subsets remain
available to the general class, but K–H rejects them.

K–H accepts any pair that is dimensionally applicable, uniform in each region,
and has exactly twice as many sites per cell on the high-density side:

| Dimension | Original surviving pairs |
| --- | --- |
| 2D | `CEV/CV`, `CEV/E`, `CV/C`, `CV/V`, `E/C`, `E/V` |
| 3D | `CFEV/FV`, `CFEV/CE`, `VF/CV`, `CE/CV`, `CV/V`, `CV/C` |

Equivalent letter permutations are accepted; so are other pairs satisfying the
same rules, such as `FE/E` and `FE/F` in 3D. `VE/CV` and `CF/CV` in 3D fail the
uniformity check. Unknown letters, missing or extra lattice parts, unsupported
dimensions, and incorrect high/low ratios are errors. `CV/C` is the default in
both 2D and 3D. `initial.closePack` does not select K–H packing.

Both regions use the same cell width. For n cells across a half-box and s sites
per low-density cell, the total population is `3*s*2^(D-1)*n^D`; requested counts
round upward to the next complete grid. Equal masses give exact cell-average
densities 0.5 and 1.0. The previous unequal-grid packing is removed. The random
perturbation retains positive modes strictly below both lattices' y-sampling
Nyquist limit. Uniform pressure still uses the solved SPH density. Interface
kernel overlap and kernel normalization can differ from the exact bulk means.
