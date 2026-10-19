# Primordial chemistry implementation plan

Build a new local thermochemistry module with the same reaction network, equation
of state, and stiff integration algorithm on CPU and CUDA. Start with atomic H/He,
then add a complete molecular hydrogen network. Treat deuterium and HD as an
optional extension. Keep chemistry disabled by default until the atomic and
molecular release gates below pass.

RECFAST remains the homogeneous recombination history used by the linear
cosmology calculation. Particle chemistry evolves each gas particle from its own
density, energy, composition, and radiation environment. Neither its initial
composition nor its subsequent temperature should be inferred from a single
background electron fraction without a documented initialization model.

## Current implementation status (2026-10-18)

The first implementation now contains shared CPU/CUDA `atomic6` and
`molecular9` rate/EOS source, signed-complement float abundance storage, scaled
thermal energy, compensated working arithmetic, a bounded coupled backward-Euler
solver with step doubling, CPU/CUDA dispatch, and synchronized SPH integration.
The reaction catalogue is [primordial_reactions.md](primordial_reactions.md).
Build, runtime, restart, and parity evidence is recorded separately in
[primordial_chemistry.md](primordial_chemistry.md); the presence of code is not a
claim that every release gate below has passed.

The implemented bounds are **10–10⁸ K for atomic6**, **100–6000 K for molecular9**,
and **proper H-nucleus density at most 10⁴ cm⁻³** for both. Molecular9 contains
29 two-body H/He reactions, eight aggregate photon channels, five H₂ cooling
partners, H₂⁺ cooling, and approximate LTE population bridges. Molecular modes
use a frozen ortho:para ratio of 3:1. The dominant-process thermal closure leaves
some minor-channel product-state energy partition unresolved and exposes its
available binding-energy bound; it does not assume all exothermicity becomes
heat. Molecular CMB line pumping, shielding, UV tables, HD, metals, dust, and
high-density chemistry/opacity are not implemented.

The present solver is the **backward-Euler verification baseline**, with a
finite-difference Jacobian evaluated at FP32 coefficient resolution. An analytic
or generated Jacobian, L-stable second-order SDIRK, CPU SIMD, optimized GPU
scheduling, individual-rung chemistry, and a broad matched-library trajectory campaign
remain later work. Three independent Radau reference trajectories now test the
production RHS integrator and tolerance refinement; this does not replace
the full model-comparison campaign. Default relative tolerance
is `1e-4`; requests below `1e-5` are rejected by the current solver. These are
numerical controls, not claims of corresponding physical rate accuracy.

The sections below retain the staged design and outstanding release gates,
updated to the selected scaled-float implementation rather than the original
double-only proposal.

## Scientific scope

| Model | Species counted in the network | Intended use |
| --- | --- | --- |
| `atomic6` | H, H⁺, He, He⁺, He²⁺, e⁻ | Primordial atomic gas and optically thin IGM calculations with a specified radiation field |
| `molecular9` | Atomic species plus H⁻, H₂, H₂⁺ | Primordial gas with molecular formation and cooling, including minihalo applications within the validated density range |
| `deuterium12` | Molecular species plus D, D⁺, HD | Optional cooling of primordial gas where HD matters |

This hierarchy follows the useful separation in [Grackle's network
documentation](https://grackle.readthedocs.io/en/latest/Parameters.html). Electron
density is derived from charge neutrality rather than evolved as an independent
unconstrained abundance. A nine-species model must actually include the H₂⁺
channel; retaining only H⁻ and H₂ is a reduced model with a different scope.

The first release covers optically thin, low-density primordial gas. Its explicit
bounds are 10–10⁸ K for atomic6 and 100–6000 K for molecular9, with proper
hydrogen-nucleus density at most 10⁴ cm⁻³. Widening the molecular interval needs
new rate/cooling fits and a validation campaign; there is no automatic switch to
atomic chemistry when a molecular state leaves its domain. The low-density limit is a deliberate boundary,
consistent with the scope of [Abel et al. 1997](https://arxiv.org/abs/astro-ph/9608040).
The solver must report an unsupported state rather than silently extrapolating a
fit or imposing a temperature floor to hide an integration failure.

High-density collapse requires a separate extension: three-body H₂ formation and
its inverse reactions, chemical heating, line opacity, collision-induced emission
and continuum opacity as applicable. Metals, dust, star formation, feedback,
cosmic rays, and detailed ISM molecules are separate science modules. HD can be
important in previously ionized or shocked primordial gas, but is not required
for every primordial calculation; see [Johnson and Bromm
2006](https://arxiv.org/abs/astro-ph/0505304).

## Physics and rate provenance

Create a versioned reaction catalogue before implementing the solver. Each entry
must specify reactants, products, stoichiometry, coefficient units, temperature
and density domain, source equation or table, and the thermal energy transferred
to gas or radiation. Put the citation beside the implemented fit, including
piecewise branch boundaries. Document intentional updates relative to older
fits. Do not restore the removed chemistry implementation as the production
baseline.

For `atomic6`, include collisional ionization, radiative recombination,
dielectronic recombination of He⁺, collisional excitation cooling, ionization and
recombination cooling, free-free emission, and signed Compton energy exchange
with the CMB. Include photoionization and photoheating through an explicit
radiation input. Select a consistent case-A baseline for optically thin
recombination; case B requires an explicit radiative assumption and matched
energy terms. [Hui and Gnedin 1997](https://arxiv.org/abs/astro-ph/9612232) and the
[Grackle rate implementation](https://github.com/grackle-project/grackle/blob/main/src/clib/rate_functions.c)
are starting points for the rate audit. The release catalogue must pin exact
source revisions and original fit references rather than depend on a moving
upstream branch.

For `molecular9`, include radiative attachment, associative detachment, radiative
association through H₂⁺, charge transfer, mutual neutralization, recombination
and detachment, and the relevant collisional destruction channels. Add
photodetachment, photoionization, and Lyman–Werner photodissociation when radiation
is enabled. Evaluate H₂ cooling with the relevant H, H₂, He, electron, and proton
collision partners and the transition toward LTE. Define the ortho/para
assumption and include formation/dissociation energy consistently. [Glover and
Abel 2008](https://arxiv.org/abs/0803.1768) describes why collision partners and
rate choices affect the thermal evolution.

Every heating/cooling coefficient needs the correct number-density factors.
For example, the common-Gaunt-factor free-free term is proportional to
`ne * (nHp + nHep + 4*nHepp)`, as in [Katz, Weinberg and Hernquist 1996,
Table 1](https://arxiv.org/pdf/astro-ph/9509107). Resolve the conflicting published
reproductions of the old metastable-helium excitation fit before adopting it.
Use original level-population assumptions and an independently checked reference,
not the species name alone, to determine its density factor.

Default to no imposed ionizing background in isolated chemistry tests. Provide
a versioned table interface for H/He photoionization rates in s⁻¹ and heating
rates per absorber in eV s⁻¹. An explicit model such as [Haardt and Madau
2012](https://arxiv.org/abs/1105.2039) can then be selected without embedding a
hard-coded reionization history. Store table checksums and interpolation rules.
Reject unavailable redshift coverage unless a documented endpoint prescription
is explicitly selected.

A uniform UV background supports a particular optically thin model; it does not
predict patchy reionization. Molecular runs with radiation need an explicit
shielding choice and column-density estimator. Validate any local approximation
against its stated conditions; [Wolcott-Green, Haiman and Bryan
2011](https://arxiv.org/abs/1106.3523) provides a relevant H₂ self-shielding study.
Spatial radiation transport and dense self-shielded gas remain separate release
gates.

## State and energy contract

Store dimensionless abundances per hydrogen nucleus, with explicit elemental
hydrogen, helium, and optional deuterium budgets. Specify whether the hydrogen
budget includes the hydrogen bound in HD; it should, so the conservation relation
counts one H in HD and two H in each H₂ or H₂⁺. Derive electrons from ionic charges,
including the negative contribution of H⁻. Distinguish helium mass fraction from
He/H number ratio everywhere.

Evolve thermal energy in eV per H nucleus together with the independent
abundances. Store helium ion fractions per helium nucleus and hydrogen-bearing
species as nuclei fractions, including factors of two for H₂ and H₂⁺. Derive
neutrals and electrons through the compensated elemental/charge constraints.
Compute
temperature, pressure, heat capacity, mean molecular weight, and frozen-composition
adiabatic sound speed from one shared equation of state. For the atomic model,
the translational ideal-gas value γ = 5/3 is sufficient within this closure. The
molecular model requires a stated rotational/vibrational treatment and a
consistent temperature inversion. A dimensionful molecular number density must
never be used as a dimensionless fraction in a heat-capacity formula.

Keep the stored energy convention `E = a² u`, where `u` is physical thermal
specific energy. The current helpers in `cosmictiger/math/units.hpp` already encode
the boundary conversions:

- `rhoProper = rhoStored * units.massG / units.lengthCm³ / a³`;
- `uPhysical = E * (units.lengthCm / units.timeSeconds)² / a²`;
- `dtPhysical = units.timeSeconds * integral(a dTauCode)`.

Use `toProperDensity`, `toPhysicalSpecificEnergy`, `toStoredSpecificEnergy`, and
`toPhysicalInterval` at the host boundary. The last helper assumes fixed `a`;
use a time-centred scale factor or cosmological quadrature across a finite step.
Pass documented plain physical scalars to the device kernel. Convert physical
specific energy to `eV/H = uPhysical * mH / ((1-Y) * eV)` and proper density to
`nH = (1-Y) * rhoProper / mH` at the Boost.Units host boundary. Test this boundary
against the typed host conversions.

For a fixed-density source substep, convert `q = heating - cooling` in
erg cm⁻³ s⁻¹ to `du/dt = q/rhoProper`. Hydro owns compression, expansion and
mechanical work; chemistry owns the reaction and radiation terms. Define the
energy ledger for binding energies, emitted photons and absorbed photons so
ionization or molecular formation cannot be counted twice. Keep the CMB
temperature tied to the configured cosmology, with Compton heating below the CMB
temperature as well as cooling above it.

## Shared implementation and execution

Use small, fixed-size host/device functions for rates, the EOS, reaction
residuals, and accepted chemistry steps. Keep them independent of HPX, particle
arrays, launch code, and reductions. The physical source is shared between CPU
and CUDA; backend files schedule work and marshal state.

The implemented files are `primordial_types.hpp`, `primordial_rates.hpp`,
`primordial_eos.hpp`, `primordial_solver.hpp`, and `primordial_chemistry.hpp`,
with dispatch in `src/chemistry/primordial_chemistry.cpp` and `.cu`. The core stores
fractional nuclei as signed complements in float and thermal energy in eV/H.
Rate/transcendental evaluation and EOS temperature use FP32. Elemental/charge
closure, cancelling polynomial evaluation, source sums, Newton residuals,
corrections, and small dense linear algebra use `FloatFloat` where cancellation
requires it. Accepted storage is quantized only at step boundaries. A negative
zero abundance encoding is meaningful and must survive serialization.

This scaling addresses exponent range while preserving the directly stored
minority; increased mantissa precision alone would not fix poor units. Conversely,
scaling cannot eliminate cancellation or restore information lost by decoding
an upper-branch fraction to float and subtracting it from one. FloatFloat retains
the FP32 exponent range. Its use does not make FP32 rate functions arbitrarily
accurate, so solver tolerances and finite-difference perturbations must reflect
the coefficient resolution.

CPU dispatch parallelizes independent particles and CUDA starts with one particle
per thread. Measure register/local-memory pressure and stiff-step divergence
before changing GPU scheduling. CPU SIMD is deferred; future SIMD lanes must
retain independent error tests and acceptance. Independent binary64 references
belong in validation and do not require FP64 production kernels.

Production CPU support has a separate application gate. The original audit
found unconditional CUDA calls in `src/gravity/kick_workspace.cpp`, CUDA allocations in
`src/analysis/lightcone.cpp`, `__device__` in `cosmictiger/analysis/healpix.hpp`, and unconditional
`hpxInitCuda()` in `src/main.cpp`. The implementation addresses these dependencies;
a full CUDA-OFF build and no-GPU startup remain required evidence, as does CPU
selection in a CUDA-enabled executable without device initialization. Unsupported
optional features must fail explicitly. A portable chemistry test target alone
does not establish a GPU-independent application.

Use `compute.cuda` for production backend selection, consistent with existing
SPH dispatch. A failed CUDA chemistry step must return a status and diagnostics;
it must not silently run a different physical approximation or retry on the CPU.
Any optional future CPU retry mode must be explicit, counted, and reproducible.
Use strict floating-point compilation for the initial chemistry kernels: current
application-wide CUDA fast-math settings must not silently weaken finite-value
checks or sensitive rate evaluation. The scaled-FP32/FloatFloat model is the
selected baseline; fast-math changes still require a separate accuracy gate.

## Stiff integration and timestep coupling

Use a coupled implicit species-and-energy solve with adaptive substeps. The
current implementation is backward Euler with step doubling, a numerical
Jacobian, a pivoted dense FloatFloat solve, and bound-preserving damped Newton
updates. It is first order; symmetric hydro/source splitting alone cannot make
this solver second order. Keep the signed-complement branch fixed within each
Newton solve and recanonicalize accepted storage. The next solver milestone is
an L-stable second-order SDIRK method with an error estimator and an analytic or
generated Jacobian. Validate that Jacobian independently by finite differences
away from branch boundaries. Compare the accepted solver with
an independent high-accuracy stiff ODE integrator on the same physical model.

Evaluate every trial residual from an immutable accepted starting state. Use
independent abundances and elemental constraints to reduce redundant equations.
Reject a trial step on negative species, negative energy, failed charge closure,
unsupported rate input, or non-convergence; reduce the step and retry within
documented bounds. Zero abundances must be valid initial conditions. Do not
repair a failed step through arbitrary clipping or renormalization. Any permitted
roundoff-sized correction needs a defined threshold and a diagnostic count.

Scale the error norm by per-species absolute tolerances and relative tolerances;
trace intermediates and bulk species cannot share a single raw residual scale.
Return accepted physical time, maximum error, iteration/substep counts, and a
failure code. If the full requested interval cannot be completed, preserve the
last accepted state and stop or request a smaller outer step through a defined
driver path. Limit retries so one stiff particle cannot hang a GPU launch.

The present SPH update has predictor, replacement and final-kick phases in
`sphParticlesApplyUpdates`. Its phase 1 removes the earlier energy predictor;
chemistry energy must not be accidentally removed or applied twice there. Add
explicit source stages around the hydro update, initially at synchronized
timesteps. Use a symmetric split and verify temporal convergence rather than
assuming the split retains second order automatically.

Extend this to individual rungs only after documenting each particle's chemical
time, physical interval, and predictor/corrector ownership. Integrate each
interval once. Couple strong heating/cooling to the hydrodynamic timestep and
neighbor timestep propagation; internal chemistry subcycling alone cannot make a
large hydrodynamic step accurate. Refresh pressure and sound-speed caches after
accepted source updates. Keep all source fields stable while a CPU/GPU batch
reads them, following the existing SPH staging convention.

## Repository integration

| Existing component | Required change |
| --- | --- |
| `cosmictiger/core/options.hpp`, `src/core/options.cpp`, `cosmictiger/core/options_metadata.hpp` | Add a `chemistry` group for enablement, network, radiation model, tolerances and bounded solver work; validate incompatible selections and serialize the new fields |
| `cosmictiger/sph/sph_particles.hpp`, `src/sph/sph_particles.cpp` | Add optional abundance storage and integration-stage state; include it in allocation, free, swap, sorting, checkpoint visitation, and particle migration |
| `src/core/driver.cpp`, `src/sph/sph.cpp` | Add explicit chemistry scheduling, rung/timestep coupling, reduction of failures, and cache invalidation |
| `src/sph/sph_hydro_cpu.cpp`, `src/sph/sph_hydro.cu`, prehydro/rung code | Consume the shared EOS pressure and sound speed; replace constant-γ assumptions when molecular thermodynamics is selected |
| `cosmictiger/sph/sph_kernel.hpp` (`SphRunData`) and SPH remote caches | Stage required EOS/source fields consistently across CPU, CUDA and MPI localities |
| `src/cosmology/initialize.cpp` | Replace the fixed neutral 1000 K chemistry-enabled initialization with explicit abundances and thermal state from a named initial-condition model |
| Checkpoint code and output writers | Save species convention, network/rate/EOS/radiation versions, tables, units, particle state and split/rung timing; expose species, temperature and chemistry diagnostics |
| `CMakeLists.txt`, `tests/` | Add independent CPU tests, CUDA parity tests, integration/restart tests and controlled benchmarks |

Current SPH uses one configured `hydro.gamma` for pressure, sound speed and the
expansion term. Replacing that constant with a local value is insufficient by
itself: derive the expansion and thermal conversion from the new EOS and verify
the adiabatic limit. Chemistry-disabled runs must retain their present behavior.

Introduce a new checkpoint payload version when adding the species/state layout.
Preserve the existing manifest and integrity checks. Reject incompatible
network, EOS, radiation, and rate-set restarts with a useful message. Old
non-chemistry checkpoints may continue in their supported mode; enabling
chemistry from one requires an explicit initialization conversion, not fabricated
abundances during restart. Include staggered stage state, following the current
SPH checkpoint visitor. Persist signed-complement float encodings, including
negative zero, with the physics version and scaled energy convention. Do not
serialize decoded bulk fractions and thereby discard their minor complements.

## Milestones and release gates

1. **Define the model and interface.** Freeze the reaction/energy catalogue,
   units, abundance normalization, EOS, radiation assumptions and valid domain.
   Resolve ambiguous fits. Add citations at each rate and an independently
   reproducible table of values at temperatures spanning every branch.

2. **Deliver atomic CPU chemistry.** Implement the scaled scalar solver and
   CUDA-free tests. Verify analytic single-reaction cases, fixed-temperature
   ionization balance, cooling/heating trajectories, Compton equilibrium,
   `dt = 0` identity, vacuum handling, and rejection of unsupported input.
   Before claiming production CPU support, also pass a full
   `COSMICTIGER_WITH_CUDA=OFF` application build and startup with a small supported
   chemistry run on a host without a GPU. Verify that CPU selection in a
   CUDA-enabled executable makes no device initialization or allocation calls.

3. **Deliver atomic CUDA parity.** Instantiate the same model and solver on CUDA.
   Compare rate/RHS/Jacobian evaluations, accepted states and failure statuses
   across randomized and adversarial stiff inputs. Test empty/tail batches and
   recovery from rejected steps. Only then enable the atomic network in SPH.

4. **Integrate particle state and synchronized hydro.** Complete initialization,
   physical-unit conversions, source splitting, EOS fields, checkpoint/output
   and MPI migration. Test expanding homogeneous boxes, cooling shocks and
   uninterrupted-versus-restarted runs. Re-run chemistry-disabled SPH regressions.

5. **Deliver molecular CPU and CUDA chemistry.** Add all nine species, molecular
   EOS and cooling, then repeat the earlier gates. Validate no-radiation and
   specified-radiation cases separately, including photodetachment,
   photodissociation and shielding assumptions. Compare one-zone collapse and
   cooling-from-ionized-gas histories within the supported domain.

6. **Enable individual rungs and optimize.** Verify per-particle time ownership,
   neighbor response to thermal changes, MPI cache freshness and restarts at
   staggered phases. Benchmark scalar/parallel/SIMD CPU and CUDA throughput on
   identical states, including stiff outliers and data-transfer costs. Performance
   improvements cannot change the model or bypass error acceptance.

7. **Add HD or high-density physics when needed.** Extend the species catalogue,
   energy ledger, EOS, tests and checkpoint version as required. Admit each
   expanded density/radiation regime only after its own comparison campaign.

The implemented default relative tolerance is 10⁻⁴ with separate abundance and
energy absolute tolerances; the current numerical Jacobian path rejects requests
below 10⁻⁵. Establish trajectory convergence and failure behavior before claiming
a tighter supported limit. Exact integer reaction stoichiometry, compensated
nuclei/charge closure, and signed-minority storage have separate tests; none of
these alone proves integration accuracy. CPU/CUDA parity is assessed in the same
scaled norm and with matching acceptance/failure behavior. Bitwise equality is
not required, but discrepancies must remain below the validated integration
error budget.

Cross-check with a pinned [Grackle release](https://arxiv.org/abs/1610.09591),
matching reaction rates, recombination case, radiation tables, shielding,
initialization, temperature floor and energy convention. Where the physical
models differ, compare aligned subsets and explain the difference rather than
loosening a tolerance until the curves agree. Shared-source CPU/CUDA agreement
is necessary but cannot independently establish scientific correctness.

A release passes only when its supported-domain suite has nonnegative accepted
abundances and positive thermal energy, conserved nuclei/charge, converged energy
histories, correct unit and expansion scaling, reproducible restart/migration
behavior, and CPU/CUDA results within the agreed error budget. Compilation, local
ODE tests, GPU execution, MPI execution and full cosmological validation remain
distinct evidence.

## Decisions before each extension

Proceed first with atomic H/He, optically thin case-A recombination, no imposed
UV background, scaled FP32 with signed-complement storage and targeted
FloatFloat arithmetic, and synchronized source integration. The next
default is the complete nine-species network within the validated low-density
domain. Preserve the current chemistry-disabled mode for nonradiative work.

Choose the production UV/reionization model before an IGM science run. Choose
the target collapse density and shielding model before claiming minihalo or
protostellar coverage. Choose HD and its primordial D/H abundance when the
science case needs it. These decisions select physical models; they do not block
the shared CPU/CUDA solver and atomic-network work.
