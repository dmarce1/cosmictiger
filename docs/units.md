# Units and coordinate frames

CosmicTiger uses Boost.Units for dimensional checking in host-side physical
conversions. The solver keeps its native numerical representations, including
fixed-point particle positions and periodic position arithmetic. The position
type is a permanent precision requirement. Creating a physical coordinate for
output reads the fixed-point value directly into double precision; it does not
pass through `float` or replace the stored coordinate.

Unit system, coordinate frame, and evolution normalization are separate concepts.
CGS or cosmological units specify the numerical scale. Comoving or lab frame
specifies the scale-factor convention for coordinates and density. Native
velocity and internal energy have additional evolution normalizations, so a
native code-unit field is not automatically a physical comoving observable.

## Setting the mass or box length

Choose one input for the cosmological normalization:

- `units.massSolar`: reference particle mass in solar masses, without `/h`.
- `units.boxLength`: comoving box side length in Mpc/h.

The code derives the other quantity from the present-day mean matter density:
`L_comoving^3 = N_reference * massSolar / rho_m,0`, with consistent physical
units. Here `N_reference = max(run.darkMatterParticleCount, run.gasParticleCount)`
uses the resolved populations after rounding each requested count to its valid
layout. With both species present, each particle has mass
`m_s = f_s * N_reference * massSolar / N_s`, where `f_s` is the cosmic mass
fraction and `N_s` the species population. A sole populated species carries the
whole matter density. Adding an equally populated second species does not
double the total mass or box volume.

For a 128³ collisionless run in a 100 Mpc/h comoving box, use:

```ini
[run]
darkMatterParticleCount = 2097152
gasParticleCount = 0

[initial]
closePack = false

[units]
boxLength = 100
```

Remove or comment out any `massSolar` setting before using `boxLength`.
Explicitly supplying both is an error, including one in the INI and the other
on the command line. Command-line values still override the same key in the
INI. If neither is supplied, the default reference mass of
`3.58e9` solar masses is used.

`boxLength` is rejected for glass generation, which uses a fixed synthetic
length and Hubble parameter, and for `initial.cosmology.gadget4Restart` input, whose
header supplies the physical units.

When restarting an existing checkpoint, retain the same units selector and
value used to create it. Restart validation requires exact equality of the
physical unit scales; switching between mathematically equivalent mass and
length inputs can change rounding. Unequal cosmological species populations
add a flagged checkpoint extension that preserves both resolved initial counts
for restart validation.

At fixed `boxLength` and cosmology, changing the resolution automatically
changes the reference mass. The proper box length at redshift `z` is the
comoving length divided by `1+z`; in Mpc it is
`boxLength / (cosmology.hubble * (1+z))`.

## Particle output controls

```ini
[output]
silo = true
units = both
frame = both
```

`output.units` accepts `code`, `cgs`, `cosmological`, or `both`. The last choice
selects CGS and cosmological output together. `output.frame` accepts `comoving`,
`lab`, or `both`. The selectors are independent for converted output:

| Unit selector | Frame selector | Silo view directories |
| --- | --- | --- |
| `cgs` | `comoving` | `cgs/comoving` |
| `cgs` | `lab` | `cgs/lab` |
| `cosmological` | `comoving` | `cosmological/comoving` |
| `cosmological` | `lab` | `cosmological/lab` |
| `both` | `both` | All four directories above |

Either selector can use `both` while the other selects one value. Each view
contains gas and dark-matter point meshes with node-centered point variables,
including the gas kernel support radius `gas/kernelSupportRadius`. Positions and support radii
use the selected units and frame. Converted floating-point arrays use double
precision. The directories above are in the master Silo file, whose
multimeshes and multivariables reference locality files under the sibling
`.silo.data` directory. Each locality stores its views below `/locality_<rank>`;
keep the master and its `.silo.data` directory together when moving a snapshot.
Metadata records the selected units, frame, scale factor, Hubble parameter,
and the peculiar-velocity convention.

Each view contains separate `gas/`, `darkMatter/`, and (when present) `stars/`
directories. Each species has a point mesh named `mesh` and fields `xVelocity`,
`yVelocity`, `zVelocity` (only active axes), `timestepRung`, and `mass`. For example,
`gas/xVelocity` and `darkMatter/xVelocity` are distinct fields. Converted views
add their unit/frame prefix. `timestepRung` remains the committed integration rung.

Gas state also includes `density`, `entropyTracer`, `kernelSupportRadius`,
`pressure`, `temperature`, `meanMolecularWeight`, and `adiabaticIndex`.
The entropy tracer is A = P_phys/rho_phys^gamma, with the existing CGS conversion
`(lengthCm/timeSeconds)^2 * (massG/lengthCm^D)^(1-gamma)` and no extra frame factor.
The evolving viscosity coefficient is `bulkViscosityCoefficient`.
Density remains stored: support-radius bounds can break the unconstrained
neighbor-number relation, so reconstructing density from radius would be incorrect.

With chemistry enabled, gas also has `neutralHydrogenAbundance`,
`ionizedHydrogenAbundance`, `neutralHeliumAbundance`,
`singlyIonizedHeliumAbundance`, `doublyIonizedHeliumAbundance`, and
`electronAbundance`, all measured per hydrogen nucleus. The electron abundance
is the sum of the ion charges, and `meanMolecularWeight` is derived from it using
the cooling model's fixed hydrogen mass fraction X=0.76. Individual species stay
stored so subtraction does not erase small trace abundances. Chemistry temperature
stays stored because reconstructing it from the final abundances need not reproduce
the iterative temperature solve within floating-point roundoff.

Star formation adds `particleIdLow`, `particleIdHigh`, and `formationTime` to
particle species, plus gas `coldGasMassFraction`, `hotGasTemperature`, and
`specificStarFormationRate`. The two ID limbs retain exact 64-bit identities.
Tracer velocities use the same axis names and `timestepRung`. Light-cone point
exports use those velocity names plus `subgroupId` and `hostGroupId`; FFT dumps
use `realSpaceValue`, `fourierRealComponent`, and `fourierImaginaryComponent`.

Pressure and neutral temperature are derived expressions:

- Native pressure: `A * rho_c^gamma * a^(5-3*gamma)`, matching stored hydro pressure.
- Converted pressure: physical pressure in **both** frames. It is
  `entropyTracer * density^gamma` in the lab frame, with an additional
  `a^(-3*gamma)` in a comoving-density view.
- CGS pressure units are `erg/cm^D`; cosmological pressure units are
  `(km/s)^2*(Msun/h)/(Mpc/h)^3`.
- Existing cosmological factors retain their 3D convention; D1/D2 SPH uses a=1.
- Temperature is always in kelvin. With chemistry enabled, the existing
  equilibrium temperature remains a stored field. Without chemistry,
  `T = (gamma-1)*u_phys / (k_B*N_A*(1-3*Y/4))`, where Y is
  `cosmology.heliumFraction`; this is the existing neutral H/He convention.
  `meanMolecularWeight=1/(1-3*Y/4)` is then constant. This convention does not
  infer ionization from temperature.

Mass and constant thermodynamic parameters are `nodal_constant` expressions.
Pressure, neutral temperature, and chemistry electron abundance and molecular weight
are also [Silo derived-variable definitions](https://silo.readthedocs.io/latest/objects.html#dbputdefvars).
They contain fully qualified field paths in both master and locality files;
VisIt evaluates them pointwise on the corresponding particle mesh. Expression
names occupy the same logical subdirectories as stored fields. Each expression's
`<name>_unit` metadata records its units; it has no redundant particle array.
The mass expression remains in `mass_expression`. Other definitions use
`<name>_expression`. These are storage objects, not additional plot variables.
Standard VisIt `time`, `dtime`, and `cycle` names remain.

This replaces abbreviated field names (`vx`, `rho`, `A`, `r`, `alpha`, `divv`)
and the former `dark_matter/` directory. Existing saved VisIt sessions and scripts
need their paths updated.

The output cadence selects coarse output epochs; it does not synchronize the
individual-rung particle fields. Converted views record
`integration_state=stored_staggered`: their velocities and the gas entropy tracer are
converted from the stored integration stages, and gas density retains the last
SPH density evaluation. Positions are at the output epoch. These views should
not be treated as a newly reconstructed common-time state. Unit conversion
leaves the solver's staggered state and integration behavior intact.

The defaults are `output.units=code` and `output.frame=comoving`. They retain
raw evolution units, with the species directories described above. `code` cannot be
combined with `lab` or `both`: choose a converted unit system for those frames.
The selectors affect particle Silo output; they do not change the integration
scheme, the restart format, or the units of separate analysis products.

The `output.xMin` through `output.zMax` bounds always use the normalized comoving
box `[0, 1]`, regardless of output selection. For example, `xMin=0.25` selects
the same particles in all four views. The scale factor stretches the output
coordinates without changing this selection.

## Native state and CGS conversions

Let `L0=codeToCm`, `M0=codeToG`, and `T0=codeToS` be the run's base scales in
centimeters, grams, and seconds, and let `a` be the snapshot scale factor.
The comoving position is stored as normalized fixed point `q`. Stored velocity
is `w=a*v_pec/(L0/T0)`, stored specific internal energy is
`E=a^2*u/(L0/T0)^2`, and stored density is `rho_c` in units `M0/L0^3`.

| Quantity | CGS comoving view | CGS lab view |
| --- | --- | --- |
| Position | `L0*q` cm | `a*L0*q` cm |
| Mass | `M0*m` g | `M0*m` g |
| Peculiar velocity | `(L0/T0)*w/a` cm/s | `(L0/T0)*w/a` cm/s |
| Density | `(M0/L0^3)*rho_c` g/cm^3 | `(M0/L0^3)*rho_c/a^3` g/cm^3 |
| Specific internal energy | `(L0/T0)^2*E/a^2` erg/g | `(L0/T0)^2*E/a^2` erg/g |

Here lab means proper positions and proper volume density at the snapshot epoch.
Every converted velocity is the physical peculiar velocity relative to the
expanding background. There is no `H(a)*r` term. A stationary comoving particle
therefore has zero output velocity even as its proper position increases.
Specific internal energy is also physical in both converted frames; the
comoving selector does not restore the stored `a^2` factor. Native `code` output
instead uses the stored `w`, `rho_c`, and entropy-tracer normalization.

The scale factor `a` and the dimensionless Hubble parameter `h` are distinct
runtime values. `h=cosmology.hubble` sets astronomical unit scales; it is not
the snapshot expansion factor or a smoothing length. At `a=1` the comoving
and lab coordinates and densities coincide.

## Cosmological units

The cosmological system uses Mpc/h for length, solar masses/h for mass, km/s
for peculiar velocity, and `(km/s)^2` for specific internal energy. Write
`MpcCm` for one megaparsec in centimeters and `SolarMassG` for one solar mass
in grams. Convert the corresponding CGS result with:

| Quantity | Numerical conversion from CGS | Unit label |
| --- | --- | --- |
| Length | `lengthCm*h/MpcCm` | `Mpc/h` |
| Mass | `massG*h/SolarMassG` | `Msun/h` |
| Peculiar velocity | `velocityCmPerS/1e5` | `km/s` |
| Density | `densityGPerCm3*MpcCm^3/(SolarMassG*h^2)` | `(Msun/h)/(Mpc/h)^3` |
| Specific internal energy | `energyErgPerG/1e10` | `(km/s)^2` |

Apply these scale conversions to either column of the CGS table. The `h`
dependence in density follows from the mass unit divided by the cube of the
length unit; it must not be treated as a single factor of `h`.
The input `units.massSolar` uses solar masses without an implicit `/h`, while
`units.boxLength` uses comoving Mpc/h. Choosing cosmological output does not
reinterpret initial-condition or checkpoint parameters.

## Run settings in Silo

Every particle Silo file, including native code-unit output, records the public
application settings under `/cli`. Dots in CLI names become directories: for
example, `hydro.viscosityAlpha` is the single double-precision value
`/cli/hydro/viscosityAlpha`. Integer and boolean settings use single integer
values (booleans are 0 or 1); strings use null-terminated character arrays.
The constant viscosity coefficient is not repeated as a particle `bulkViscosityCoefficient` field.

These are the effective settings at the snapshot, including defaults,
configuration-file values, and runtime overrides or caps, expressed in the CLI's
units. Thus support radii and the gravity effective radius
(`/cli/gravity/effectiveRadius`) are recorded in interparticle spacings, and
`units.massSolar` is in solar masses. Deprecated aliases such as
`gravity.softening` and `hydro.centralSoftening` are not written. The removed keys
`hydro.minEffectiveRadius` and `hydro.maxEffectiveRadius` are no longer written either;
`/cli/hydro/minSmoothingLength` and `/cli/hydro/maxSmoothingLength` record the bounds. The scalar settings are shared
by all unit/frame views within the file.

`/cli/argc` and `/cli/argv/arg_0`, `arg_1`, and so on preserve the original process
argument list, including the executable and HPX flags, before HPX parses it.
Each argument is stored separately to preserve spaces and empty strings. These
arguments describe the invocation; the scalar settings describe the effective
run state, so runtime overrides can differ from the original input.

## Time and type boundaries

Cosmological integration uses conformal time, with `d tau = dt/a`, while
physical clocks use `dt`. For a differential interval at one epoch,
`dt_seconds = a*T0*d tau_code`. A finite physical elapsed time requires
integrating `a` over the conformal-time interval; multiplying a stored elapsed
conformal time by the final scale factor is generally incorrect.

Boost.Units quantities prevent incompatible dimensions from being combined.
Separate semantic wrappers distinguish quantities that share dimensions but
have different meanings, including comoving and proper coordinates, stored
and peculiar velocities, and conformal and physical time. Conversion requires
the explicit run scales and cosmological context. Native particle arrays,
CUDA payloads, and checkpoints retain their existing numerical representation;
only output and host conversion boundaries extract physical scalar values.

For a host calculation, include `cosmictiger/math/simulation_units.hpp` and wrap the
existing position without changing its storage:

```cpp
auto const scales = units::scalesFromOptions(getOptions());
units::CosmologicalContext const context(a, getOptions().cosmology.hubble);
units::ComovingPosition<Fixed32> const position(particlesPos(XDIM, i));
auto const proper = units::toProperLength(position, scales, context);
double const centimeters = proper.quantity().value();
double const megaparsecsOverH = units::toCosmologicalValue(proper, context);
```

`position.storage()` retains the original fixed-point value. Inverse position
conversions reject values outside the normalized box `[0, 1)` before constructing
the fixed-point type; periodic wrapping remains with the existing fixed-point
arithmetic. Raw GPU and SIMD interfaces retain their existing representations.
The CUDA compile regression checks the typed host boundary and fixed-point device
payload without requiring Boost quantity arithmetic inside kernels.

Output settings are distributed with `Options` to HPX localities. They do not
add fields to the on-disk checkpoint header or particle payload, and changing
the requested output views does not change the simulation's restart contract.

Silo particle meshes and fields carry standard float/double time and cycle metadata.
Each master and locality file also carries root `time` (float), `dtime` (double),
and `cycle` (integer), which VisIt reads for its database time slider. Frame
directories carry the same clock pair. Native noncosmological output uses elapsed code time; cosmological and converted
output uses physical age or elapsed physical time in Gyr. The root `time_unit`
and `time_convention` metadata identify the stored clock.
