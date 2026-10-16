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
contains materialized point meshes and fields, so its mesh geometry has the
selected units and frame. Converted floating-point arrays use double precision.
Metadata records the selected units, frame, scale factor, Hubble parameter,
and the peculiar-velocity convention. Mesh time and `time` metadata retain the
existing cosmological-age convention in Gyr (`time_unit=Gyr`,
`time_convention=cosmological_age`); this is not elapsed standalone hydro time.

The output cadence selects coarse output epochs; it does not synchronize the
individual-rung particle fields. Converted views record
`integration_state=stored_staggered`: their velocities and gas energies are
converted from the stored integration stages, and gas density retains the last
SPH density evaluation. Positions are at the output epoch. These views should
not be treated as a newly reconstructed common-time state. Unit conversion
leaves the solver's staggered state and integration behavior intact.

The defaults are `output.units=code` and `output.frame=comoving`. They retain
the existing root-level mesh names and raw evolution fields. `code` cannot be
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
instead preserves `w`, `rho_c`, and `E` for compatibility.

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
The input `units.massSolar` retains its existing meaning in solar masses,
without an implicit `/h`. Choosing cosmological output does not reinterpret
initial-condition or checkpoint parameters.

## Run settings in Silo

Every particle Silo file, including native code-unit output, records the public
application settings under `/cli`. Dots in CLI names become directories: for
example, `hydro.viscosityAlpha` is the single double-precision value
`/cli/hydro/viscosityAlpha`. Integer and boolean settings use single integer
values (booleans are 0 or 1); strings use null-terminated character arrays.
The constant viscosity coefficient is not repeated as a particle `alpha` field.

These are the effective settings at the snapshot, including defaults,
configuration-file values, and runtime overrides or caps, expressed in the CLI's
units. Thus support radii and gravity softening are recorded in interparticle
spacings, and `units.massSolar` is in solar masses. The scalar settings are shared
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

For a host calculation, include `cosmictiger/simulation_units.hpp` and wrap the
existing position without changing its storage:

```cpp
auto const scales = units::scalesFromOptions(getOptions());
units::CosmologicalContext const context(a, getOptions().hubble);
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
The separate legacy `lc2rockstar` reader consumes the old native `Options`
prefix and restores only the scalar fields it uses. The added output selectors
follow that prefix in memory. Legacy lightcone files remain tied to their
original compiler/library ABI; embedded historical string bytes are ignored.
