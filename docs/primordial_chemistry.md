# GADGET-4 equilibrium atomic cooling

The `chemistry.*` option group selects the standard GADGET-4 equilibrium H/He
cooling model with external UV disabled. Its name and build capability are retained for configuration
continuity. The former Anninos reaction networks, molecular species, manual ion
fractions, and chemistry subcycling controls have been removed.

The reference is official GADGET-4 revision
[`6fb393b5e25907f2ff06211f67d341c5e40e90d1`](https://gitlab.mpcdf.mpg.de/vrs/gadget4/-/commit/6fb393b5e25907f2ff06211f67d341c5e40e90d1),
especially [`cooling.cc`](https://gitlab.mpcdf.mpg.de/vrs/gadget4/-/blob/6fb393b5e25907f2ff06211f67d341c5e40e90d1/src/cooling_sfr/cooling.cc)
and [`cooling.h`](https://gitlab.mpcdf.mpg.de/vrs/gadget4/-/blob/6fb393b5e25907f2ff06211f67d341c5e40e90d1/src/cooling_sfr/cooling.h).
The physical cooling routines use CGS density, specific thermal energy and
seconds at the CosmicTiger boundary; stored hydrodynamic energy remains `a²u`.

## Physical state and algorithm

The gas has one thermal energy and one electron abundance estimate. At each
cooling evaluation, the H/He ion populations are solved in equilibrium at the
current temperature through collisional ionization equilibrium. The saved electron
abundance seeds that iteration; it is not a separately integrated ionization
history. Thermal evolution uses GADGET-4's bracketed implicit energy solve.

The cooling composition is fixed at hydrogen mass fraction `X=0.76`, helium
mass fraction `Y=0.24`, and `gamma=5/3`. The independent
`cosmology.heliumFraction` still controls the homogeneous cosmological
background/recombination calculation; it does not alter these cooling constants.

The rate table contains 2000 intervals over `log10(T/K)=1..9`. Its low- and
high-temperature branches follow the pinned implementation: the cold
abundances become neutral, and the hot branch becomes fully ionized with
free-free and cosmological Compton cooling. This replaces the former network's
hard `10^4 cm^-3` density rejection and `10^8 K` ceiling. The resulting model is
optically thin equilibrium atomic cooling; it does not supply molecular,
metal-line, shielding, or general high-density chemistry.

## Configuration

```ini
[hydro]
enabled = true
gamma = 1.6666666666666667

[chemistry]
enabled = true
initializationTemperature = 10000
minimumSpecificEnergy = 0
```

| Option | Meaning |
| --- | --- |
| `chemistry.enabled` | Enable GADGET-4 equilibrium atomic cooling. Requires SPH and `gamma=5/3`. |
| `chemistry.initializationTemperature` | GADGET-4 `InitGasTemp` in kelvin, default 10000. Zero preserves initial-condition thermal energy. |
| `chemistry.minimumSpecificEnergy` | GADGET-4 `MinEgySpec`, expressed here in physical erg/g. Default zero adds no energy floor. |

Temperatures and energy floors must be finite and nonnegative. Initialization
uses the GADGET-4 neutral/ionized mean-weight convention. Ordinary cooling
enforces the configured physical energy floor. A zero initialization temperature
requires a usable initial-condition energy; it is not a request for zero-energy
gas.

External UV photoionization and photoheating are not part of this model. The
TREECOOL reader, radiation-table configuration option and photo-rate branches
have been removed. No radiation file is needed. Atomic radiative cooling,
cosmological CMB Compton exchange and supernova heating remain active in their
respective paths. Older `chemistry.radiationTable` and `chemistry.radiationFile`
configuration keys are rejected.

## Integration and persistence

Cooling executes once after the completed selected SPH interval, matching the
pinned GADGET-4 main-loop ordering. It does not add opening or closing chemical
half steps. The local implicit solve consumes that interval; it does not change
an already selected outer SPH step or request an outer retry. Every time integrator
supplies the interval it actually closed.

The bridge prepares all updated gas states, checks the results and only then
commits them. The electron estimate follows gas particles through sorting,
migration and checkpoints. The single hydrodynamic thermal energy remains
authoritative. Checkpoint model/version metadata distinguishes this state from
the retired multi-species layout. Cooling fingerprint version 6 contains six
doubles and no radiation checksum; earlier cooling fingerprints are rejected.

Star-forming gas follows [the GADGET-4 multiphase prescription](star_formation.md)
using this same cooling kernel. The homogeneous recombination solver used by
the background and power-spectrum calculation remains separate and available
when particle cooling is disabled.

## Validation

`gadget4_cooling` exercises the core equilibrium and implicit-cooling routines.
`sph_chemistry_timesteps_cpu` exercises source integration and particle state;
`gadget4_cooling_integration_cpu` and `gadget4_cooling_integration_global_cpu`
exercise the application when their required capabilities are built. CUDA
coverage uses the production cooling implementation and strict floating-point
compilation. The existence of these targets is not a statement that a particular
checkout, device, MPI topology, or scientific resolution has passed them.
