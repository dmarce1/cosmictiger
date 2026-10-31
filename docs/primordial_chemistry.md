# Primordial thermochemistry

The opt-in primordial source operator evolves H, H+, He, He+, He++, electrons,
and, for `molecular9`, H−, H2 and H2+. CPU and CUDA dispatch the same reaction,
EOS and scalar source integration implementation. The runtime `chemistry.enabled`
option is disabled by default; cosmology scenario INIs enable it explicitly.
The prior design is retained as a historical record in
[the plan](primordial_chemistry_plan.md). The complete reaction catalogue,
source citations, fit domains and thermal approximations are in
[the reaction reference](primordial_reactions.md).

## Precision and units

`primordial_units.hpp` is the Boost.Units host boundary. Proper density, physical
specific energy and physical time are typed quantities. Multiplication by the
hydrogen mass and division by the hydrogen mass fraction and the typed electron
volt converts physical specific energy to eV per hydrogen nucleus. Only after
that dimensionally checked expression is evaluated is its value extracted as a
float. The inverse conversion is typed too. Both directions use the same exact-SI
electron-volt convention as the rates, represented by a Boost.Units energy
quantity, rather than the older rounded application constant. Cosmological conversions use the
existing `a^2 u` energy, comoving density and conformal-time interfaces.

The numerical payload stores eV/H, proper H nuclei/cm³, K, seconds, reaction
coefficients in cm³/s and radiation rates. The core is templated on one scalar
type: `StateT<Real>`, `EnvironmentT<Real>`, `SolverOptionsT<Real>` and
`advance<Real>`. Species, coefficients, EOS, energy and local time all use `Real`.
The application's aliases select `float`; callers may instantiate the same
implementation with `double`. There is no FloatFloat arithmetic, precision
promotion, compensated accumulation or signed-complement representation.
Boost.Units conversions and the application's global clock remain host boundaries.

All nine species are saved directly, in this order: H, H+, He, He+, He++,
electrons, H−, H2 and H2+. Each value is a particle count per hydrogen nucleus;
helium counts are therefore not normalized by helium, and molecule counts are
not doubled. The existing initialization options still specify ion fractions
and molecular nuclei fractions; `initializeState` converts them once into the
nine-species representation. `Atomic6` requires the last three counts to be zero.
Hydrogen, helium and charge constraints are checked with ordinary rounding
allowance. No species is encoded as a signed distance from one.

The standalone chemistry library is compiled without fast-math or FTZ. CPU
batch workers establish nearest rounding and gradual underflow with a scoped
guard and restore the caller's environment; CUDA uses matching strict flags.

## Running

Particle chemistry requires both `COSMICTIGER_ENABLE_SPH=ON` and
`COSMICTIGER_ENABLE_CHEMISTRY=ON`, which are the CMake defaults. Set
`-DCOSMICTIGER_ENABLE_CHEMISTRY=OFF` to build SPH without particle chemistry.
With SPH disabled, the chemistry setting is inactive. Homogeneous cosmological
recombination remains available regardless of these settings.
The variational pressure operator uses the composition-dependent pressure and
acoustic index supplied by particle chemistry.

For example, with an existing configured build:

```sh
./cosmictiger --hydro.enabled=true --chemistry.enabled=true \
  --chemistry.network=atomic6 \
  --chemistry.initialTemperature=1000 --chemistry.hydrogenIonFraction=0.0002 \
  --compute.cuda=false
```

Use `molecular9` for molecular channels and `compute.cuda=true` for CUDA dispatch
in a CUDA-enabled build. There is no silent backend fallback. CPU-only builds
need no CUDA toolkit; a CUDA-enabled executable can also operate without a CUDA
context when `compute.cuda=false`. Auxiliary feature limitations are listed in
[CPU support](chemistry_cpu_support.md).

The initial composition is explicit and is not automatically copied from RECFAST.
Initial fractional nuclei and temperature are configured in the `chemistry.*`
option group. Atomic6 requires its molecular fractions to be zero. Options and
run output record the selected model and subcycling controls.

With star formation disabled, chemistry follows each active gas particle's SPH
timestep. Each interval uses
`C(dt/2) K(dt/2) D(dt) K(dt/2) C(dt/2)` (CKDKC), where `C` advances chemistry,
`K` is the hydro/gravity kick and `D` drifts positions. At a particle's next
active boundary, the old hydro predictor is replaced before the closing chemistry
half step. That source step uses the saved interval that just completed; the
next opening chemistry half step uses the newly selected SPH rung, or the actual
selected interval in global-step mode. Inactive
particles do not advance chemistry during another particle's finer step.
Physical source duration uses the scale factor at the half-interval midpoint.
Density/composition-dependent pressure and frozen-composition sound speed enter
both CPU and CUDA hydro kernels, including the cosmological expansion term.
The cosmic energy diagnostic uses the pressure integral for variable gamma.

Hydro selects its next interval, including applicable neighbor limits, before
the opening source step. Chemistry advances once over that selected half-interval;
its scalar source solver subcycles internally. The updated EOS and hydro rates are
recomputed for the opening kick without changing the selected timestep or
repeating chemistry. Chemistry has no separate outer timestep option.
Check timestep refinement
for the intended problem: a successful local chemistry solve alone does not
bound operator-splitting error, and the symmetric ordering does not make the
first-order source solver second order.

With `starFormation.enabled=true`, the Springel–Hernquist source operator owns
the same opening/closing source clock and heating ledger. It invokes the shared
chemistry batch solver only for ordinary gas and decoupled wind intervals;
standalone chemistry steps are skipped and their pending clock fields remain
zero. Eligible dense gas instead receives the SH03 effective energy and a
mass-weighted neutral-cold plus atomic-CIE-hot composition. This projection
does not evolve separate non-equilibrium chemical histories in the unresolved
phases, and ordinary cooling is not applied to the effective mixture. Either
network may be combined with star formation; with `molecular9`, entering the
dense branch also discards the parcel's molecular fractions. See
[star formation](star_formation.md) for
threshold transitions, wind recoupling, radiation ownership, and projection
diagnostics.

The deprecated `chemistry.synchronizedRung` key is accepted and ignored so older
INI files still load. It can be removed; supplying it with chemistry enabled
prints a notice. It does not select a fixed timestep or restore compatibility
with old chemistry checkpoints.

## Solver and model limits

The source integrator follows Anninos, Zhang, Abel & Norman (1997),
[sections 3.1–3.2](https://arxiv.org/abs/astro-ph/9608041). Explicit thermal
subcycling precedes chemical subcycling over the supplied source interval.
For each chemical sweep the fixed order is H, H+, He, He+, He++, electrons,
algebraic equilibrium H− and H2+, then H2. The scalar update is
`n_new = (n_old + dt*C)/(1 + dt*D)`. Creation/destruction terms use earlier
species' newly updated values and the adopted reaction catalogue and rate fits.
The rate temperature is held at its post-thermal value during the chemical stage,
using the paper's advanced-temperature choice.
The fast-intermediate equilibrium assumptions are part of this solver; it no
longer evolves independent non-equilibrium histories for H− and H2+.

After the full chemical stage, the hydrogen and helium conservation corrections
change the dominant species, then charge conservation fixes electrons. Internal
subcycles retain the independently evolved electron value for timestep selection. All nine counts are retained through
storage, migration and restart, so neutral traces are not reconstructed from
rounded ion fractions at every source call.

`chemistry.fractionalChangeLimit` defaults to 0.1, the paper's ten-percent
subcycle bound on electron abundance and thermal energy. It is a fractional
change limit, not a local truncation-error estimate. Refining it does not by
itself remove the splitting error between chemistry and thermal evolution over
the supplied outer interval. `chemistry.electronFractionFloor` defaults to
1e-20 and uses that minimum electron scale only for timestep selection whenever
the electron abundance is below it; it never floors an abundance. `chemistry.maxSubsteps` bounds the combined number
of thermal and chemical substeps. An unsupported or failed source interval
returns the initial state and zero accepted time. The SPH bridge prepares all
ranks before committing results and does not commit a failed batch.

The old Newton/Jacobian, step-doubling and mixed-precision paths have been
removed. Their relative and absolute tolerance settings are replaced by the
subcycling controls above. CPU and CUDA run the same templated solver; there is
no separate reduced-matrix or mixed-network solver implementation.

The current density ceiling is 10^4 H nuclei/cm³. Both networks accept
10–10^8 K: the [cold extension](molecular9_cold.md) supplies the molecular
low-temperature cooling branches and the [hot extension](molecular9_hot.md)
keeps all nine species through dissociation, ionization and recombination,
listing which fits are held constant or extrapolated above their published
ranges. Unsupported states are errors, not floors or silent extrapolations.
This excludes a complete high-density collapse calculation and parts of very
cold molecular gas physics. There are no metals, dust, deuterium/HD, self-consistent radiative transfer,
shielding, three-body chemistry or state-resolved molecular populations. Molecular
rotational/vibrational thermodynamics uses an approximate fixed ortho/para model.
Unresolved thermal partition for some product-state reactions is explicitly
identified and bounded in the reaction reference. CMB Compton exchange is present;
CMB molecular-line pumping is not. These limitations matter near the CMB floor.

## Radiation input

Without a radiation file all imposed photo rates/heating are zero. CMB temperature
still scales with cosmological epoch for Compton exchange. A supplied table begins
with exactly:

```text
COSMICTIGER_PRIMORDIAL_RADIATION 1
```

Each subsequent data row contains redshift followed by fourteen nonnegative
scalars, in this order:

1. H photoionization, He photoionization, He+ photoionization, H− photodetachment;
2. H2 photodissociation, H2 photoionization, H2+ photodissociation, H2+ photoionization;
3. net heat deposited per H, He, He+, H−, H2 and H2+ absorber.

The first eight values have units s^-1 per absorber; the last six have units
 eV per absorber per second. Heating is the excess thermal deposition, so the
ionization/dissociation threshold must not be charged again. Rows have strictly
increasing redshift. Blank lines and whole lines beginning with `#` are allowed.
Interpolation is linear in redshift; extrapolation is rejected. The table is
an explicit externally supplied radiation model, not a built-in UV background.

## Persistence and diagnostics

Chemical fractions, source energy, EOS caches, predictor state, pending source
half-interval and opening heating migrate/sort with each gas particle. Opening
heating enters the accumulated source ledger when that particle closes its
interval. Checkpoints preserve this pending state for particles on different
rungs. The optional payload has a distinct physics flag and layout version;
old chemistry checkpoints with the common-rung layout are rejected.
Combined star-formation runs retain chemical fractions and source energy in
this payload, while their pending half-interval and heating live exclusively
in the star-formation state; standalone chemical step fields remain zero.
The direct nine-species payload uses physics version 3 and rejects older
chemistry checkpoints. Restart checks the physics version, network, solver settings,
initial composition, CMB/helium configuration, radiation file checksum and source
stage state. Chemistry-disabled checkpoints retain the previous payload.

Gas Silo output includes all nine direct species counts per H nucleus,
temperature, mean molecular weight and frozen gamma. Existing output metadata
still describes the stored staggered time stage; conversion to physical units
does not synchronize a snapshot.

## Regression entry points

`primordial_rates`, `primordial_solver`, `primordial_units`, `primordial_cuda`,
`primordial_integration_cpu`, `cpu_only_runtime`, `sph_cpu` and the existing CUDA
SPH kernel tests cover the relevant layers. `primordial_reference` instantiates
the same solver with double and checks independent analytic photoionization,
photodissociation, Compton and split photoheating limits under refinement.
Coefficient/EOS anchors and reaction conservation are tested separately.
`tests/run_primordial_integration.py --cuda` exercises full CUDA source/hydro
stepping and restart. Exact commands and observed outcomes belong in the run's
[validation record](anninos_validation.md); a configured test target is not
evidence that it passed.
