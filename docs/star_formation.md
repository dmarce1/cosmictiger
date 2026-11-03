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
`t_relax = t_star (1-x)/(beta x (1+A))`. When it is above `u_eff`, ordinary
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

With `hydro.method=mfm`, the same source runs when a meshless cell closes its
own-step interval, over the duration it actually closed, and a converted cell
leaves the gas set at that kick's scatter; see
[Chemistry and star formation](hopkins_hydro.md#chemistry-and-star-formation).

For gas mass `m`, the event probability is `1-exp(-SFR dt/m)`. A successful event
converts the whole gas particle into a star; gas mass is not continuously
removed before the event. Position, velocity and particle mass are retained,
and the star continues its gravitational evolution. Birth time is the scale
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
Choose traditional, hierarchical or global integration with `hydro.method=sph` or
`mfm` (MFM additionally needs `COSMICTIGER_ENABLE_HOPKINS=ON` and Fixed64 positions;
MFV is not supported). Hierarchical integration applies the source when each gas
particle's own interval closes, over that interval; see
[hierarchical SPH](hierarchical_sph.md#cooling-and-star-formation) and
[hierarchical MFM](hopkins_hydro.md#chemistry-and-star-formation). A starting configuration is
[`configurations/star_formation/lcdm.ini`](../configurations/star_formation/lcdm.ini).

```ini
[hydro]
enabled = true
gamma = 1.6666666666666667

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
