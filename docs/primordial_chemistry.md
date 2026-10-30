# Primordial thermochemistry

The opt-in primordial source operator evolves H, H+, He, He+, He++, electrons,
and, for `molecular9`, H−, H2 and H2+. CPU and CUDA dispatch the same reaction,
EOS and implicit integration implementation. Chemistry is disabled by default.
The implemented model and remaining research/optimization gates are distinguished
in [the plan](primordial_chemistry_plan.md). The complete reaction catalogue,
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

The device/CPU numerical payload contains documented floats: eV/H, proper H
nuclei/cm³, K, seconds, reaction coefficients in cm³/s, and specified radiation
rates. It never stores tiny CGS volume energy densities as floats. Boundary
conversion rejects nonfinite, overflowing and nonzero subnormal payload scales.
Host dimensional conversions use double, as do the existing global cosmological
clock and diagnostics; the chemistry device arithmetic uses float/FloatFloat.

Six independent stored fractions count Hp/H, Hep/He, Hepp/He, Hminus/H, 2H2/H,
and 2H2plus/H. For fraction c ≤ 1/2 the stored value is c; for c > 1/2 it is
−(1−c). The sign **bit** selects the branch, so negative zero represents one.
The minority is retained directly, including through checkpoint/migration; do
not decode the dominant value into a float and subtract it from one. To initialize
an extremely small minority, use `SignedAbundance::fromMinority` directly rather
than first constructing `1-minority` in ordinary floating arithmetic.

FloatFloat is used for constrained species closure, reaction sums, cancellation
sensitive rate polynomials, Newton residuals/linear algebra, trial states and
local clocks. Rate transcendental functions and EOS temperature are FP32. The
standalone chemistry library is compiled without fast-math or FTZ. CPU batch
workers establish nearest rounding and gradual underflow with a scoped guard and
restore the caller's floating-point environment; CUDA uses explicit strict flags.
Boost.Units does not enter device structures.

## Running

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
run output record the selected model and tolerances.

Chemistry follows each active gas particle's SPH timestep. Each interval uses
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
its implicit solver subcycles internally. The updated EOS and hydro rates are
recomputed for the opening kick without changing the selected timestep or
repeating chemistry. Chemistry has no separate outer timestep option.
Check timestep refinement
for the intended problem: a successful local chemistry solve alone does not
bound operator-splitting error, and the symmetric ordering does not make the
backward-Euler source solver second order.

The deprecated `chemistry.synchronizedRung` key is accepted and ignored so older
INI files still load. It can be removed; supplying it with chemistry enabled
prints a notice. It does not select a fixed timestep or restore compatibility
with old chemistry checkpoints.

## Solver and model limits

The current solver is adaptive backward Euler with step doubling, bounded Newton
iterations and positivity-aware line searches. The accepted fine solution consists
of two half steps. It is a first-order baseline, not SDIRK. Relative tolerance is a
**local** error control, not a promise of equally small global trajectory error.
The default is 1e-4 and requests below the tested floor 1e-5 are rejected.
Species absolute tolerances also matter for traces. Failure returns the last
accepted state/time and work counters. The SPH bridge prepares all ranks before
committing source results, and does not commit a partially failed batch.

The current density ceiling is 10^4 H nuclei/cm³. Atomic6 accepts 10–10^8 K;
molecular9 accepts 10–6000 K, with the [cold extension](molecular9_cold.md)
providing the previously missing low-temperature cooling branches. Unsupported
states are errors, not floors or silent extrapolations. This excludes a complete
high-density collapse calculation and parts of hot or very cold molecular gas
physics. There are no metals, dust, deuterium/HD, self-consistent radiative transfer,
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
Restart checks the reaction/EOS physics version, network, solver settings,
initial composition, CMB/helium configuration, radiation file checksum and source
stage state. Chemistry-disabled checkpoints retain the previous payload.

Gas Silo output includes species counts per H nucleus, six raw signed fractions,
temperature, mean molecular weight and frozen gamma. Existing output metadata
still describes the stored staggered time stage; conversion to physical units
does not synchronize a snapshot.

## Regression entry points

`primordial_rates`, `primordial_solver`, `primordial_units`, `primordial_cuda`,
`primordial_integration_cpu`, `cpu_only_runtime`, `sph_cpu` and the existing CUDA
SPH kernel tests cover the relevant layers. When NumPy/SciPy is available,
`primordial_radau_reference` compares the production RHS integration against an
independent Radau integrator. This tests the integrator, not the provenance of
rate fits; coefficient/EOS anchors and reaction conservation are tested separately.
`tests/run_primordial_integration.py --cuda` exercises full CUDA source/hydro
stepping and restart. Exact commands and observed outcomes belong in the run's
[validation record](primordial_validation.md); a configured test target is not
evidence that it passed.
