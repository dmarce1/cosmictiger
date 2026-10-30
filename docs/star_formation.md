# Springel–Hernquist star formation

The optional star formation module implements the equilibrium, effective
equation-of-state form of Springel & Hernquist (2003),
[*Cosmological smoothed particle hydrodynamics simulations: a hybrid multiphase
model for star formation*](https://doi.org/10.1046/j.1365-8711.2003.06206.x).
Equation numbers below refer to the published paper. This implementation uses
one stochastic stellar generation per gas particle, \(N_g=1\).

Compile with `COSMICTIGER_ENABLE_SPH=ON`, `COSMICTIGER_ENABLE_CHEMISTRY=ON`, and
`COSMICTIGER_ENABLE_STARS=ON`; the latter defaults to `OFF`. The former
`COSMICTIGER_ENABLE_STAR_FORMATION` name initializes an unmigrated cache;
afterward `STARS` controls both names, including when explicitly set `OFF`.
At runtime set `hydro.enabled=true`, `starFormation.enabled=true`,
`chemistry.enabled=true`. The variational pressure operator uses the effective
pressure and acoustic index supplied by the
closure. The traditional and global time integrators are
supported. The gas adiabatic index must be \(\gamma=5/3\). Stars require
chemistry at configure time and runtime, with either `chemistry.network`.
Both networks cover 10–10⁸ K, so the hot gas released from this closure stays
inside the [molecular network's domain](molecular9_hot.md); ordinary
`molecular9` gas keeps all nine species through heating and cooling.

## Cooling assumptions and units

The dense-gas closure uses optically thin, primordial H/He collisional ionization
equilibrium (CIE), built from the existing Atomic6 reaction and thermal rate
primitives. The unresolved phases do not integrate separate non-equilibrium
chemical histories, and the closure does not impose that network's low-density
runtime limit. There are no
molecules, metal-line cooling, external UV background, or CMB exchange in this
adapter. Its atomic temperature domain is 10–10⁸ K; unsupported states fail
explicitly. Metallicity is an enrichment tracer and does not change cooling.

Ordinary gas and hydrodynamically decoupled winds instead advance the existing
non-equilibrium thermochemistry operator of the configured `chemistry.network`
(`atomic6` or `molecular9`). They use its configured
radiation table, cosmological CMB exchange, solver tolerances, and supported
density/temperature domain. They do not additionally call the CIE cooling
adapter. Thus imposed radiation affects ordinary gas but does not modify the
dense SH03 calibration or unresolved phase balance.

**This is a CIE realization of the SH03 model.** The paper's simulations
included an external UV background and different historical cooling fits.
Consequently the derived threshold and numerical curves need not reproduce
the paper's figures exactly. There is no supplied UV table or hard-coded
hydrogen-density threshold in this implementation.

The model evaluates proper density \(\rho\) in g cm⁻³, specific thermal
energy \(u\) in erg g⁻¹, pressure in erg cm⁻³, and physical time in seconds.
The adapter converts cosmological particle storage into these units before
calling the source model. Formation timestamps record elapsed physical seconds
since the simulation began, rather than cosmic age.

For hydrogen mass fraction \(X\), the neutral molecular weight is
\(\mu_n=[X+(1-X)/4]^{-1}\), and the fully ionized value is
\(\mu_i=[2X+3(1-X)/4]^{-1}\). Temperature parameters are converted using
\(u=k_B T/[(\gamma-1)\mu m_p]\): cold clouds and the nominal 10⁴ K warm
energy use \(\mu_n\); the supernova energy uses \(\mu_i\). The warm energy
is a reference for threshold calibration, not a floor imposed on ordinary
non-equilibrium gas. Ordinary gas can cool below that energy within the
configured network's domain. Pressure continuity at the calibrated threshold applies when the
ordinary gas has energy \(u_4\); a parcel arriving at another energy undergoes
a thermal projection whose change enters the source-energy ledger.

## Equilibrium closure

The cold-cloud fraction \(x=\rho_c/\rho\) and hot fraction \(1-x\) are
unresolved constituents of one SPH gas particle. They share its momentum;
they are not separate particle populations. The density-dependent formation
time and evaporation efficiency are equations (20)–(21):

\[
t_\star=t_0^\star(\rho/\rho_{\rm th})^{-1/2},\qquad
A=A_0(\rho/\rho_{\rm th})^{-4/5}.
\]

The hot energy is equation (11),
\(u_h=u_c+u_{\rm SN}/(1+A)\). With
\(Q=\beta u_{\rm SN}-(1-\beta)u_c\) and the CIE coefficient
\(\lambda(u_h)=\Lambda(\rho,u_h)/\rho^2\), equations (16)–(18) become

\[
y={t_\star\rho\lambda(u_h)\over Q},\qquad
x={2y\over1+2y+\sqrt{1+4y}}.
\]

The code retains the small hot fraction separately to avoid cancellation at
large \(y\). Equation (19) gives
\(u_{\rm eff}=(1-x)u_h+xu_c\) and
\(P_{\rm eff}=(\gamma-1)\rho u_{\rm eff}\). The SPH pressure uses this
effective closure in star-forming gas. Its signal speed uses the acoustic
derivative \(c_{\rm eff}^2=dP_{\rm eff}/d\rho\), evaluated by a local
logarithmic finite difference, and enters the hydrodynamic timestep.

The threshold is calibrated from continuity with the warm energy \(u_4\),
using equation (23) and the exact hot-energy expression:

\[
u_{h0}=u_c+{u_{\rm SN}\over1+A_0},\qquad
x_{\rm th}={u_{h0}-u_4\over u_{h0}-u_c},\qquad
\rho_{\rm th}={x_{\rm th}\over(1-x_{\rm th})^2}
{Q\over t_0^\star\lambda(u_{h0})}.
\]

Calibration requires \(u_{h0}>u_4>u_c\) and \(Q>0\). Changing cooling or
phase parameters recalculates \(\rho_{\rm th}\).
Eligibility uses this proper-density threshold alone. At sufficiently high
redshift the mean cosmic gas density can exceed it. This implementation does
not add the later GADGET overdensity requirement of 57.7.

## Chemical state and source ownership

When star formation is enabled, its source operator owns the opening and
closing half-intervals and the heating ledger for every active gas particle.
The driver does not also run standalone chemistry half-steps. Ordinary gas
uses the shared, nonmutating chemistry batch solver over the same physical
interval; the standalone `SphChemistryStep` clock and heating fields remain
zero. All local and remote preparations must succeed before chemical state,
energy, metallicity, wind clocks, or conversion events are committed.

Eligible dense gas receives \(u_{\rm eff}\) and a mass-weighted chemical
projection: cold gas is neutral and atomic, while hot gas has the CIE
composition at \(u_h\). Both phases use the same primordial hydrogen and
helium mass fractions. Each ionized fraction is the hot-phase fraction times
\(1-x\); molecular fractions are zero, so a `molecular9` parcel's H⁻, H₂ and
H₂⁺ are discarded on entry along with its atomic history. This preserves H/He
nuclei and charge closure. The stored chemical thermal energy is synchronized with the projected
SPH energy, but the hydrodynamic pressure and signal speed still come from the
SH03 effective equation of state. Ordinary chemical cooling is never applied
to this effective mixture.

Entering the dense branch replaces the parcel's previous chemical history
with that projection. Leaving it starts ordinary chemistry from the last
projected composition and current SPH energy. There is no independently
evolved hot/cold chemical state or hidden phase-history flag. A wind remains
ordinary gas until its decoupling clock expires; a half-step that straddles
expiry advances chemistry for the decoupled portion and applies the dense
closure only for any remaining eligible duration. Density-based recoupling
is evaluated at the source boundary.

Chemical output for dense particles describes the projected mixture. Its
single-state temperature is an effective diagnostic, not either phase's
physical temperature; `hot_temperature` reports the hot-phase temperature.
This is a combined ordinary-chemistry and equilibrium-SH03 model, not a
phase-resolved non-equilibrium multiphase chemistry calculation.

## Conversion, enrichment, and winds

The long-lived stellar mass rate is
\(\dot\rho_\star=(1-\beta)x\rho/t_\star\), equation (1). For an elapsed
source interval \(\Delta t\), equation (39) with \(N_g=1\) converts an
entire gas particle with probability
\(p_\star=1-\exp[-(1-\beta)x\Delta t/t_\star]\). This preserves the
fixed baryonic particle mass and avoids fractional-mass gravity sources.
Converted particles remain gravitational sources and sinks and cease SPH
interactions. The phase equations already include instantaneous recycling;
conversion must not subtract another factor \(\beta\) from particle mass.

Equation (40) adds
\(\Delta Z=y_\star(1-\beta)x\Delta t/t_\star\), where \(y_\star\) is
metal mass per mass locked into long-lived stars. Stellar particles retain
their inherited metallicity. Random draws use particle identity, an event
counter, the configured seed, and separate formation/wind streams so particle
sorting and worker assignment do not determine the realization. These state
fields must survive migration and checkpoint/restart.
Enrichment follows the linear expected mass increment while whole-particle
conversion samples the exponential probability. Their finite-step difference
vanishes with the source timestep.

Winds are optional. Equations (28) and (41) give

\[
p_w=1-\exp[-\eta(1-\beta)x\Delta t/t_\star],\qquad
v_w=\sqrt{{2\beta\chi u_{\rm SN}\over\eta(1-\beta)}}.
\]

Directions are isotropic. The kinetic wind energy is additional to the
unchanged thermal feedback channel, following section 4.1; \(\chi\) is
nonnegative and may exceed unity. Launched particles recouple to hydrodynamics
when either their elapsed decoupling time expires or their proper density
falls below the configured fraction of \(\rho_{\rm th}\). They remain
gravitationally active during this interval.
Formation is sampled first; surviving gas then samples a wind event. This
ordering has a finite-step bias relative to simultaneous competing events.
The selected gas timestep limits the combined formation and enabled-wind
hazard to 0.05; smaller timesteps reduce both this bias and the enrichment
discretization error.

Particle Silo output includes persistent identity, formation time,
metallicity, and remaining wind decoupling time. Gas additionally exposes
`cold_fraction`, `hot_temperature` in K, and `specific_sfr` in s⁻¹ (long-lived
stellar mass formed per gas mass per physical second). These describe the
unresolved equilibrium phases evaluated from the last stored SPH density,
converted to proper density at the output epoch. All three are zero for
ordinary gas and decoupled winds. They are independent of the selected output
unit system; snapshot metadata records these conventions.

## Runtime settings

All keys use the `starFormation.` prefix; in an INI file use a
`[starFormation]` section.

| Key | Default | Meaning |
| --- | ---: | --- |
| `enabled` | `false` | Enable the combined ordinary-chemistry and dense SH03 source |
| `beta` | `0.1` | Instantaneously recycled massive-star fraction |
| `evaporationFactor` | `1000` | \(A_0\) |
| `starFormationTimeGyr` | `2.1` | \(t_0^\star\), in Gyr |
| `coldTemperature` | `1000` | Cold temperature, K |
| `supernovaTemperature` | `1e8` | Supernova energy temperature, K |
| `seed` | `1` | Unsigned 64-bit stochastic seed |
| `metalYield` | `0.02` | \(y_\star\) |
| `winds` | `false` | Enable wind launching |
| `windMassLoading` | `2` | \(\eta\) |
| `windEnergyFraction` | `0.25` | \(\chi\) |
| `windDecouplingMyr` | `50` | Maximum physical decoupling interval, Myr |
| `windRecoupleDensityFraction` | `0.1` | Recoupling density relative to \(\rho_{\rm th}\) |
| `windDirection` | `isotropic` | Currently supported launch geometry |

A compact cosmological example is
[`configurations/star_formation/lcdm.ini`](../configurations/star_formation/lcdm.ini).
It uses CPU execution, the traditional integrator, and FMM gravity. Set solver
selectors to match the compiled capabilities when using a different build.

## Validation targets

`springel_and_hernquist_test` checks the pure model and CIE adapter independently
of HPX. `star_formation_cpu` exercises the particle integration fixture and
checks Silo phase diagnostics, species, and persistent identities in code,
CGS, and cosmological output frames. It also compares ordinary and decoupled
wind evolution with the shared chemistry batch solver, checks dense chemical
projection and threshold crossings, and verifies chemical/source rollback
after a failed preparation. The fixture covers EOS/timestep closure,
new-star gravity kicks, and stepping after all gas has converted.
`star_formation_particles_cpu` checks population persistence and compaction.
When HPX uses MPI, `star_formation_mpi` and `star_formation_particles_mpi`
repeat the source and persistence fixtures across two localities.
`star_formation_application` runs the fixtures with one and two workers, checks
cosmological evolution and exact particle/source-state restart with each
compiled FMM/PMMM solver and both eligible integrators, and rejects changed
model parameters or helium abundance on restart. `springel_and_hernquist_cuda_test`, when
CUDA is compiled, compares the pure scalar equations with the CPU.
`grouped_options` checks CLI/INI precedence, exact 64-bit seed metadata,
incompatible physics, and parameter rejection. `tests/check_build_features.py`
checks that both production source files disappear when the capability or SPH
is disabled; `tests/check_build_variants.py` tests those graph checks.

These targets are separate evidence from distributed evolution, CUDA runtime,
galaxy-scale star-formation-law convergence, and cosmological production runs.
A passing local model test alone does not validate those larger calculations.
