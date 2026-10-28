# Scenario configurations

Select a physical simulation with `run.scenario`. Each scenario has its own
initializer under `src/scenarios/` and header under `cosmictiger/scenarios/`.
The evolution driver calls gravity, SPH, and chemistry according to the enabled
physics. `run.test` is reserved for developer regressions and benchmarks.

Each INI below contains every application option except the top-level `help`
and `configuration` options, including commented alternatives for radius inputs.
The files are independent: no other INI is loaded or inherited. Sections map to
CLI groups, and explicit CLI values override values in the selected file.
[default.ini](../default.ini) records the application defaults; the cosmology
scenario configuration below also enables primordial chemistry explicitly.

The gravity solver, PMMM near method, and expansion order are selected by
`./build.sh` or CMake when building the executable; the INIs contain runtime
controls only. Global FMM keeps `gravity.openingAngle` and `gravity.openingSchedule`
as runtime controls. A PMMM build requires an explicit valid `gravity.meshSize`,
and retains `gravity.pmmmNearWidth` at runtime. The supplied `meshSize = 0` is
unset; override it for PMMM, for example with `--gravity.meshSize=32`.
Gravity with SPH requires an FMM build. To use a collisionless cosmology with a
PMMM build, also set `--hydro.enabled=false --chemistry.enabled=false`.

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
| [cosmology.ini](cosmology.ini) | `cosmology` | Gravity, SPH, and primordial chemistry | `./output/cosmology/` |
| [glass.ini](glass.ini) | `glass` | Repulsive gravity with damping | `./output/glass/` |

Run from a build directory containing the executable and any runtime inputs:

```sh
./cosmictiger --configuration=../../configurations/sod.ini --hpx:threads=12 --hpx:bind=none
```

Use `--compute.cuda=false` for CPU execution. HPX thread and locality options
stay on the command line. The examples use nominal resolution 32, except the
blast and disc profiles, which use 64. Override it with
`--initial.particlesPerDimension=64`. These are starting resolutions for
experimentation, not convergence results.

## Output

Every scenario configuration writes generated simulation products beneath its
`output.directory`, including particle Silo snapshots, text diagnostics,
checkpoints, and analysis products. Its lightcone directory is a subdirectory
of that location. Relative paths begin at the launch working directory; input
files such as existing glass loads and chemistry radiation tables
retain their input-path meanings.

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
its output directory.

Rayleigh–Taylor retains periodic boundaries and its prescribed downward
acceleration, with self-gravity enabled. It is not a wall-bounded hydrostatic
benchmark. The disc retains its prescribed central potential alongside
self-gravity. Its initial specific internal energy is `1e-4` in code units,
independent of resolution, replacing the legacy near-zero pressure. This keeps
the fractional thermal-change timestep bound usable as viscous heating begins,
while the thermal energy remains small compared with the orbital kinetic energy.
The star uses a Lane–Emden polytrope of index 1.5: the first stage uses damping
and zero artificial-viscosity alpha, while the second uses no damping and alpha
0.05. The blast reference solution assumes gamma 5/3.

The cosmological configuration creates `N^3` gas and `N^3` dark-matter particles
for `initial.particlesPerDimension=N`. Class masses are fixed within each
species, with relative gas and dark-matter weights set by `omegaB` and `omegaC`.
It enables the `atomic6` primordial chemistry network; see
[primordial chemistry](../docs/primordial_chemistry.md). A collisionless
cosmological run can disable both `hydro.enabled` and `chemistry.enabled`.

Six collisionless cosmology presets use 32³ particles in a 100 Mpc/h box,
CPU execution, and 100 particle snapshots at equal conformal intervals:

| Preset | Background and endpoint |
| --- | --- |
| [lcdm_present.ini](lcdm_present.ini) | Standard flat ΛCDM, expanding to `a=1`. |
| [closed_no_dark_energy.ini](closed_no_dark_energy.ini) | Collapsing: `Ωm=2`, `ΩΛ=0`, through turnaround to contracting `a≈1e-6`. |
| [omega_k_negative.ini](omega_k_negative.ini) | `Ωk=-0.1`, positive spatial curvature, positive Λ; expanding to `a=10`. |
| [omega_k_positive.ini](omega_k_positive.ini) | `Ωk=+0.1`, negative spatial curvature, positive Λ; expanding to `a=10`. |
| [critical.ini](critical.ini) | Critical: `Ωm=1−Ωr`, `ΩΛ=Ωk=0`, expanding to `a=100`; physical `da/dt → 0`. |
| [subcritical_no_dark_energy.ini](subcritical_no_dark_energy.ini) | Subcritical: `Ωm≈0.313466`, `ΩΛ=0`, `Ωk≈0.686442`, expanding to `a=100`; physical `da/dt → H0 sqrt(Ωk) > 0`. |

`Ωk` has the opposite sign to spatial curvature: negative `Ωk` means a closed
geometry. The two mildly curved ΛCDM examples both keep expanding; positive
spatial curvature alone does not guarantee recollapse. In the critical and
subcritical no-dark-energy models, `H=adot/a` approaches zero in both cases,
but only the critical model's physical expansion speed `adot` approaches zero.

The standard, curved ΛCDM, and subcritical examples use `h=0.674`,
`Ωb=0.0224/h²`, and `Ωc=0.12/h²`. All include photons and massless neutrinos
with `Ωr≈9.20939e-5`; the historical massive-neutrino density is not added to
cold matter. The curved and collapsing presets use `sigma8=0` homogeneous
lattices because the native perturbed initial-condition solver requires
flatness. The flat standard and critical presets use `sigma8=0.8607`.

The optional [lcdm_ten_ages.ini](lcdm_ten_ages.ini) continues the standard model
to ten times its present age, recalculated with the current radiation model.
Its present age is about 13.81146 Gyr and its final scale factor is about
1291.70. This preserves the historical future-evolution example separately
from the six cosmology choices above.

These files contain complete options and describe their assumptions in their
headers. Their positive `run.endTime` values are **conformal code durations**,
not years. For a fixed box and cosmology, changing the lattice from 32 to `N`
requires multiplying `units.massSolar` by `(32/N)^3`; changing the box or the
cosmology also requires recalculating the duration. See
[background evolution](../docs/cosmology.md) for stopping, signed Λ input,
accuracy controls, and restart behavior.

Gravity uses a fixed comoving support radius, with a Ferrers degree-4 density
kernel in these configurations. SPH has independently selected kernels and
variable smoothing radii; the examples use cubic splines and 42 weighted
neighbors. Set gravity radius through `gravity.effectiveRadius`,
`gravity.supportRadius`, or the legacy `gravity.softening` alias. Alternative
radius representations remain commented to avoid conflicting inputs.
