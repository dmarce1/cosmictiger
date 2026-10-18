# Problem configurations

Each INI contains every application option except the top-level `help` and
`configuration` options. INI sections correspond to the CLI groups, and each
entry carries its help text as an inline comment. Values start from the current
CUDA-enabled source defaults, with the overrides described below. Edit any
entry directly; no other INI is loaded or inherited. The source defaults are
also available in [default.ini](../default.ini).

Run from a build directory containing the executable and runtime data:

```sh
./cosmictiger --configuration=../../configurations/kelvinHelmholz.ini --hpx:threads=12 --hpx:bind=none
```

Explicit command-line values override the file. For example, append
`--initial.particlesPerDimension=64` to change the resolution. The configurations
use nominal resolution 32, except the blast and disc profiles, which use 64.
These are starting resolutions for experimentation, not convergence claims.
Each problem configuration enables full-box particle Silo output in `./output`,
at every synchronized coarse step. `default.ini` sets `output.silo=false`.
Gravity uses the same fixed comoving Ferrers F2 kernel for gas, dark matter,
and stars. Set its radius with `gravity.effectiveRadius`, `gravity.supportRadius`,
or the legacy effective-radius alias `gravity.softening`. SPH uses a separate
variable smoothing radius and defaults to the cubic spline with 42 weighted
neighbors. All gas evolution requires CUDA. HPX thread/locality settings remain
on the command line and are not application INI options.

| Configuration | Selection | Problem-specific settings |
| --- | --- | --- |
| [kelvinHelmholz.ini](kelvinHelmholz.ini) | `run.test=helmholtz` | Counterflow shear layers, 32,600 equal-mass gas particles at resolution 32. |
| [sod.ini](sod.ini) | `run.test=sod` | Densities 1 and 0.125, pressures 1 and 0.1, initially at rest. |
| [sedovBlast.ini](sedovBlast.ini) | `run.test=blast` | Gaussian deposit of width 0.015, resolution 64, gamma 5/3. |
| [rayleighTaylor.ini](rayleighTaylor.ini) | `run.test=rt` | Chosen downward acceleration `hydro.yAcceleration=-0.5`; current periodic boundaries. |
| [orbitingDisc.ini](orbitingDisc.ini) | `run.test=disc` | Resolution 64, unit central force matching the initial circular velocity, softening 0.01. |
| [polytropicStar.ini](polytropicStar.ini) | `run.test=star` | Self-gravitating gas sphere, polytropic index 1.5, gamma 5/3. |
| [legacyPlummer.ini](legacyPlummer.ini) | `run.test=plummer` | Legacy hydro-driver test; see limitations below. |
| [plummer.ini](plummer.ini) | Empty test; `initial.plummer=true` | Dark-matter Plummer model through the normal driver. |
| [cosmologyDarkMatter.ini](cosmologyDarkMatter.ini) | Empty test; `hydro.enabled=false` | Cosmological dark matter with the source cosmology defaults. |
| [cosmologySph.ini](cosmologySph.ini) | Empty test; `hydro.enabled=true` | Cosmological gas and dark matter, resolution per class. |
| [createGlass.ini](createGlass.ini) | Empty test; `initial.createGlass=true` | Glass generation on one HPX locality; writes `glass.bin`. |

Particle masses are constant within each class and may differ between classes.
The standard cosmological SPH lattice retains `N^3` gas and `N^3` dark-matter
particles for `initial.particlesPerDimension=N`. Their code masses are
`omegaB/(omegaB+omegaC)` and `omegaC/(omegaB+omegaC)`; `units.massSolar` sets the
physical mass unit. Gravity radius inputs retain the initial reference spacing
of one class, not the spacing of the combined population. Hydro test fixtures
retain their existing class masses and thermodynamic normalization.

## Output and duration

Particle Silo output is configured in each INI. For a full-box view:

```ini
[output]
silo = true
siloInterval = 1
directory = ./output
units = code
frame = comoving
xMin = 0
xMax = 1
yMin = 0
yMax = 1
zMin = 0
zMax = 1
```

Bounds must be finite and satisfy `0 <= min < max <= 1` in each dimension.
Output is controlled only by CLI/INI options; explicit CLI values override the
INI. `output.silo=false` disables particle snapshots. `analysis.views`
independently controls HEALPix maps.

The `units=code` and `frame=comoving` defaults retain the native output layout.
To write CGS proper positions and density, use `units=cgs` and `frame=lab`.
For Mpc/h, solar masses/h, and km/s, use `units=cosmological`; either frame is
available. Setting both selectors to `both` writes all four converted views.
Converted velocities are always peculiar, excluding Hubble flow. Output bounds
still select the normalized comoving box, and fixed-point particle storage is
preserved. See [units and frames](../docs/units.md) for the full contract.

`output.siloInterval` is a positive number of synchronized coarse steps. Its
default of 1 preserves output at every coarse step, while 10 writes at cycles
0, 10, 20, and so on. It does not count adaptive substeps or wall-clock seconds.
For KH, one coarse step is `10/1024 = 0.009765625` simulation-time units.
Standalone hydro tests always retain their initial and final snapshots when
Silo output is enabled, even if the final cycle is not a multiple of the interval.
The cosmological driver applies the interval at its existing synchronized
output points. These settings do not alter the simulation timestep.

`output.directory` defaults to `./output`, relative to the working directory;
absolute paths are also supported. The directory is created automatically when
output is enabled. The hydro driver writes `view.0.<cycle>.silo` there, with the
original coarse cycle number in each name, so skipped cycles leave gaps.
The hydro driver also writes
`energy.dat` and `rho.dat`. The diagnostic text files append to existing files,
so use a separate working directory for each run. Cosmological runs also need
`power.init` and the build's `recfast` data/executable.

The standalone hydro initializers currently fix their end times and coarse
interval counts in the source. `run.maxIterations` and `run.numberOfSteps` do
not change those limits:

| Test | End time / duration | Coarse intervals |
| --- | --- | --- |
| Kelvin–Helmholtz | 10 | 1024 |
| Sod | 0.09 | 20 |
| Sedov blast | 0.005 | 16 |
| Rayleigh–Taylor | 10 | 256 |
| Orbiting disc | 16 | 1024 |
| Polytropic star | 10 then 100 dynamical times | 25 then 250 |
| Legacy Plummer | 100 dynamical times | 1024 |

## Existing initializer limitations

- The RT profile uses periodic boundaries and uniform acceleration. Reflecting
  y boundaries are unsupported. It is not a validated wall-bounded hydrostatic
  RT benchmark.
- The polytropic star is gas, independent of star formation and chemistry. Its
  legacy initializer overrides `hydro.viscosityAlpha` to 0 during its first
  stage and 0.05 during its second stage. Snapshot cycle numbering restarts
  in the second stage, so matching first-stage snapshot names are overwritten.
- The legacy Plummer selector enables SPH mode but creates only dark-matter
  particles, and its initial virial diagnostic divides zero by zero. Prefer
  `plummer.ini` for the normal dark-matter driver. That driver overrides
  `gravity.openingAngle` to 0.4.
- The Sedov reference solution assumes gamma 5/3; changing gamma also requires
  changing the reference if its error comparison is to remain meaningful.
- `hydroWaveTest()` exists in the source but has no selectable CLI entry, so it
  has no runnable configuration here. Developer regressions and timing tests
  are separate from these physical problem configurations.

Configuration coverage and parsing are checked without evolving these problems.
