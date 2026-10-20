# Einstein Boltzmann solver audit and neutrino implementation plan

The interior synchronous-gauge equations mostly agree with Ma and Bertschinger. The current implementation nevertheless has errors in its multipole boundary, collision-stage timing, radiation parameter calculation, and spectrum bounds handling. Its timestep refinement also changes the unnormalized CDM power by percent-level amounts at some sampled wavenumbers. These should be resolved before adding massive neutrinos.

The checkout already evolves **massless neutrinos**, including their gravitational effects. `cosmology.nEff` defaults to 3.046, `computePowerSpectrum` passes `omegaNu` to the solver, and the `NLi` hierarchy evolves their density, velocity, quadrupole, and higher moments. **Massive neutrinos are absent.** No missing massive-neutrino physics is classified below as a defect in the existing massless model.

Audit date: 2026-10-20. Repository: `/home/dmarce1/workspace/cosmictiger`, branch `main`, HEAD `1efb52fb`, with pre-existing uncommitted changes. References below describe the working files inspected, including existing changes in options and initialization. This audit adds documentation and evidence only; it does not repair production code.

The primary mathematical reference is [Ma and Bertschinger, astro-ph/9506072](https://arxiv.org/pdf/astro-ph/9506072), especially equations 21, 42, 47–58, 63–69, and 96–97. Equation numbers refer to the supplied PDF.

## Variable conventions and equations that agree

Let \(x=\ln a\), \(\mathcal H=aH\), and \(v_s=\theta_s/k\). In the code:

| Code quantity | Meaning |
| --- | --- |
| `hdoti` | \(dh/dx\), rather than the conformal-time derivative |
| `etai`, `taui` | Synchronous metric \(\eta\), conformal time \(\tau\) |
| `eps` | \(k/\mathcal H\) |
| `thetabi`, `thetagami`, `thetanui` | \(v_b,v_\gamma,v_\nu\) |
| `F2i`, `N2i` | \(F_{\gamma2}=2\sigma_\gamma,\ F_{\nu2}=2\sigma_\nu\) |
| `sigmaT` | \(n_e\sigma_Tc/H\), the collision rate for evolution in \(x\) |
| `cs2` | Baryon sound speed squared divided by \(c^2\) |

With these conversions, the following parts are consistent:

| Implementation | Paper |
| --- | --- |
| `boltzmann.cpp:120–130`, metric evolution | 21a–c, combining the density constraint and trace equation |
| `boltzmann.cpp:131`, CDM continuity | 42 |
| `boltzmann.cpp:132,137,186–195`, baryon continuity, pressure and drag | 66 |
| `boltzmann.cpp:133,138,140–143,148–158`, photon and polarization transport | 63 |
| `boltzmann.cpp:135,139,144–147,155–156`, interior massless-neutrino evolution | 49 |
| `boltzmann.cpp:196–205`, polarization collision inverse | Collision terms in 63 |
| `zero_order.cpp:242–282`, thermal sound speed | The derivative definition in 68 |

The polarization inverse was also checked algebraically against its collision matrix. Its coefficients and signs are correct for frozen background coefficients. The defect is in how the full integrator evaluates and combines stages, not in that inverse.

The flat-universe metric equation implemented here can be written

\[
h_{xx}=-\left(2+\frac{d\ln H}{dx}\right)h_x
-\frac{8\pi Ga^2}{\mathcal H^2}(\delta\rho+3\delta P).
\]

This explains both the existing matter/radiation coefficients and the pressure source required when massive species are introduced. CDM has zero synchronous-gauge velocity by the coordinate choice; this does not imply zero simulation particle velocities.

## Confirmed defects

### Incorrect multipole boundary and timestep independent damping

**Priority P1.** `src/cosmology/boltzmann.cpp:160–162,222–237`.

For the actual last stored index \(\ell=\mathrm{LMAX}-1=1023\), substitution of the free-streaming closure gives

\[
\frac{dF_\ell}{dx}
=\frac{k}{\mathcal H}F_{\ell-1}
-\frac{\ell+1}{\tau\mathcal H}F_\ell
-S F_\ell,\qquad S=n_e\sigma_Tc/H.
\]

The photon and polarization equations use \(S\); the massless-neutrino equation omits it. The implemented streaming coefficient is smaller by `2*LMAX-1`, the photon/polarization closure damping has the same extra divisor, and the neutrino closure damping is absent. Moreover, the photon/polarization implicit denominator omits the timestep multiplying its damping rate.

An isolated test of the extracted production evolution body, with zero opacity and a nonzero last photon moment, returns approximately 0.130 of that moment after a step even as the step tends to zero. The update should instead tend to the identity. This proves an integration consistency error independently of any cosmological parameter choice.

Correct all three boundary equations and test spherical-Bessel free streaming, one-step consistency, and convergence with multipole depth. The error in an actual matter spectrum depends on whether boundary contamination reaches the relevant low moments; the fixture does not determine that error.

### Collision stages use the wrong times and stale background coefficients

**Priority P1.** `src/cosmology/boltzmann.cpp:108–117,170–185,241–248`.

The first explicit half-step leaves the shared `loga` at \(x_0+\Delta x/4\), its last derivative evaluation time. The following implicit calls add their stage offsets to that leftover value. Thus the opacity is evaluated at fractional times 0.542893 and 0.957107, instead of 0.292893 and 0.707107. The baryon and photon fractions and `hubble` remain at \(x_0+\Delta x/4\), even while `a` and opacity change.

For the controlled equation \(y_x=-e^x y\) over \(0\le x\le1\), this stage implementation gives relative errors of approximately −2.195%, −1.086%, and −0.540% at steps 0.05, 0.025, and 0.0125. Restoring the stage times gives −0.0561%, −0.0140%, and −0.00348%. The former converges at first order.

Make all stage times explicit from `loga0` and evaluate every background coefficient at its proper stage. Test time-dependent opacity and baryon loading, not only constant-coefficient relaxation.

### Configured radiation parameters do not determine the radiation density

**Priority P1.** `src/core/options.cpp:917–936`.

The total radiation density uses hardcoded `neff=3.086` and `thetaValue=1.0`. Only the subsequent division between photons and neutrinos uses the requested `cosmology.nEff`. The requested CMB temperature does not enter the total.

Relative to the configured defaults \(N_\mathrm{eff}=3.046,\ T_\gamma=2.7255\) K, total radiation is **1.2026% too large**. Setting `nEff=0` leaves a photon density **1.7121 times** its intended value. Changing neutrino abundance therefore changes the inferred photon density while leaving total radiation fixed.

Compute photons from the configured temperature first, then neutrino radiation from that photon density and the configured abundance. Add option-to-background regression coverage. The existing background test constructs physically consistent densities directly and bypasses this defect.

### Out of range power evaluation reads beyond the vector

**Priority P1.** `src/cosmology/initialize.cpp:100–119,221–222,447–474`.

`PowerSpectrumFunction::operator()` prints a range warning but still accesses all four interpolation entries. The internally computed table has a fixed wavenumber range, whereas the FFT's required range depends on box size and resolution.

AddressSanitizer reproduced a heap-buffer-overflow with the extracted production lookup, a 1024-element table over the production interval, and a query at \(k=50\ {\rm Mpc}^{-1}\). The fixture uses `std::vector`, matching the unchecked indexing inherited by the project's `Vector` in Release builds. Debug builds enable container bounds checks and abort instead. The default upper table bound is approximately \(17.013\ {\rm Mpc}^{-1}\). The same out-of-range path exists below the supported interval.

Derive required coverage from the Fourier grid, include interpolation padding, and reject unsupported requests before indexing. Validate input table size and positivity before logarithmic interpolation.

### Massless neutrino quadrupole initialization has a factor of four error

**Priority P2.** `src/cosmology/boltzmann.cpp:47–48,139,144–147`.

The initializer stores \(\sigma_\nu/2\), but the evolution interprets the slot as \(F_{\nu2}=2\sigma_\nu\). The leading multiplier must be `2.0`, not `0.5`, for the implemented state convention and the paper's growing-mode series.

This seeds a transient; its final spectrum effect was not measured separately. Test the initial density ratios, velocity ratios, quadrupole, metric constraints, and start-time convergence together.

### RECFAST and the perturbation solver use different neutrino backgrounds

**Priority P2.** `recfast/recfast.f:369–371`; `src/cosmology/initialize.cpp:66–70`.

The bundled RECFAST explicitly assumes three massless neutrinos. Its input wrapper does not pass `params.neff` or a general expansion history. Arbitrary `nEff`, including zero, consequently gives inconsistent recombination and perturbation backgrounds even after the option-density calculation is repaired.

Pass a consistent radiation model now and provide a shared expansion-history interface for massive neutrinos. Recompute recombination histories for comparison cosmologies.

### Nonflat requests silently obtain flat internal transfer functions

**Priority P2.** `src/core/options.cpp:937–942`; `cosmictiger/cosmology/zero_order.hpp:27–35`; `src/cosmology/initialize.cpp:66–70`.

The simulation options allow explicit `omegaLambda` and derived curvature, but `CosmicParameters` has neither field. The internal solver assumes \(\Omega_\Lambda=1-\Omega_m-\Omega_r\), and the recombination wrapper likewise constructs a flat background.

Initially reject nonflat requests when using this internal solver. Supporting them later requires curved-space perturbation treatment as well as a consistent background; adding an \(a^{-2}\) term alone is insufficient.

### Background range validation uses the wrong Boolean condition

**Priority P2.** `src/cosmology/boltzmann.cpp:73–74`.

The range test uses `amin < uni.amin && amax > uni.amax`, so either endpoint can be out of range without rejection. The interpolation functor then prints a diagnostic and extrapolates. Use the logical OR and validate finite positive \(k,a\), ordered integration bounds, and supported density domains.

For example, a zero-baryon case also needs an explicit supported path or rejection: the collision solve otherwise has zero denominators when both baryon density and opacity vanish.

## Algorithm and output choices

### Stiff operator splitting loses accuracy in tight coupling

The exact local collision inverse does not by itself make the split evolution accurate when collisions are much faster than a timestep. Explicit half-steps create photon–baryon slip and photon shear before the separate collision update removes them.

A frozen photon–baryon acoustic test with \(4\rho_\gamma/(3\rho_b)=1\), \(k=1\), and collision rate \(10^5\) gives roughly 3.0%, 1.5%, and 0.76% amplitude loss over one period at steps 0.0481, 0.0240, and 0.0120. The exact finite-collision solution has density amplitude 0.9999936 at that time. This is a controlled subsystem test, not a prediction of a cosmological spectrum error.

Use a tight-coupling expansion with a tested transition to the full hierarchy, or an IMEX method demonstrated to recover the stiff limiting equations accurately. Retain the small analytic collision blocks if useful.

### Stability limits are not accuracy tolerances

`boltzmann.cpp:90–103` selects a step from stability estimates and a fixed maximum of 0.05. There is no local error estimator, step rejection, or monitored Einstein constraint.

The extracted production solver and production background implementation were run to \(a=1\), with the existing fiducial RECFAST history, physically computed radiation densities, and fixed primordial normalization. Changing only the maximum step from 0.05 to 0.00625 produced these differences:

| \(k\), \({\rm Mpc}^{-1}\) | \(P_c(0.05)/P_c(0.00625)-1\) |
| --- | --- |
| 0.01 | −0.1551% |
| 0.1 | +1.1841% |
| 1 | +3.1188% |

Here \(P_c\) means the unnormalized \(\delta_c^2\). These numbers demonstrate step dependence; neither run is an independent truth solution. Subsequent `sigma8` normalization can change the residual amplitude differences. It cannot be assumed to eliminate differences that depend on wavenumber.

With neutrino gravitational weight set to zero, the same comparison at \(k=0.1\ {\rm Mpc}^{-1}\) changes \(\delta_c^2\) by 3.6932%. This fixture deliberately retains the fiducial electron history to isolate the perturbation integration; it is not a self-consistent photon-only recombination calculation. The timestep sensitivity is therefore not contingent on the presence of neutrino perturbations.

Introduce relative and absolute tolerances, component-aware norms, and constraint diagnostics. Vary start time and table resolution independently of timestep.

### Fixed large hierarchies spend work without guaranteeing convergence

Every mode evolves 1024 entries for each of photons, polarization, and massless neutrinos, even when a species is absent or its higher moments are negligible. Meanwhile, large \(k\tau\) can exceed the fixed truncation scale. A large constant cutoff does not compensate for an incorrect closure.

Make hierarchy depths configurable by species, validate them by refinement, and profile after accuracy repairs. Use controlled tight-coupling and late radiation approximations where justified. Do not multiply the present 1024-moment allocation by every massive-neutrino momentum node by default.

### Spectrum and initial condition contracts discard needed information

`boltzmann.cpp:279–300` computes matter weights but returns only \(\delta_c^2\) and \(\delta_b^2\). Despite the caller's `velK` name, the second output is baryon density power, not velocity power. The first is normalized to `sigma8` and supplies particle initial conditions.

If `sigma8` is intended to mean combined matter, it must be evaluated from the weighted **signed transfers before squaring**. For the existing model,

\[
\delta_{cb}=\frac{\Omega_c\delta_c+\Omega_b\delta_b}{\Omega_c+\Omega_b}.
\]

Alternatively, explicitly document and name a CDM-only normalization. Do not silently treat the two definitions as interchangeable.

`initialize.cpp:256–265` obtains earlier-time perturbations and velocities using approximate scale-independent matter-plus-Lambda growth from the \(z=0\) spectrum. This omits radiation and curvature from that rescaling and cannot represent massive-neutrino scale-dependent growth. Return signed density and appropriately defined velocity transfers at the requested redshift. Preserve relative baryon/CDM phases and specify the gauge conversion used for particle velocities.

The approximate radiation-era initialization is also a numerical choice: \(\tau=1/\mathcal H\) and the truncated growing-mode series require a start-time convergence test. The constraint-based initial `eta` is not evidence of a separate sign error.

### Additional maintenance findings

`boltzmann.cpp:277` waits for asynchronous tasks without consuming their futures. Exceptions stored by a worker can therefore be missed. Consume each result with `get()` and test failure propagation; the project's aborting `THROW_ERROR` path is not an exception-recovery mechanism.

The currently unused `ZeroOrderUniverse::redshiftToDensity` at `zero_order.cpp:71–79` multiplies the present matter fraction by the critical density at the requested epoch. A matter-density helper should return \(\Omega_{m,0}\rho_{\mathrm{crit},0}/a^3\). No active callers were found, so this is a dormant helper defect rather than an explanation of the measured transfer differences.

## Neutrino implementation sequence

1. **Establish a trustworthy massless baseline.** Repair the defects above, align the option and RECFAST backgrounds, and add direct solver tests. Make zero abundance actually remove neutrinos. Keep the initial supported geometry flat.

2. **Define the species and normalization API.** Represent masses, degeneracies, and neutrino temperatures explicitly, plus any remaining massless contribution. Define how massive species consume the early relativistic abundance so they are not double-counted. Keep CDM, baryons, photons, massless neutrinos, and massive neutrinos separate. Add primordial \(A_s\) and pivot scale, with explicit alternatives for \(\sigma_{8,cb}\) and \(\sigma_{8,m}\).

3. **Generalize the homogeneous background.** Integrate the Fermi–Dirac momentum distribution to obtain \(\rho_\nu(a)\) and \(P_\nu(a)\), using \(\epsilon(q,a)=\sqrt{q^2+a^2m^2}\). Tabulate these and the expansion rate consistently for recombination, perturbations, and simulation expansion. Test \(d\rho_\nu/d\ln a=-3(\rho_\nu+P_\nu)\), the relativistic limit, and the nonrelativistic limit. Massive neutrinos cannot remain in a constant `omegaR/a^4` term.

4. **Add a momentum-dependent perturbation hierarchy.** Use the synchronous-gauge system in equation 56 for \(\Psi_\ell(q)\), the stress integrals in 55, and initial conditions in 97. Convert conformal-time derivatives to \(d/d\ln a\) consistently. At every stage accumulate density, pressure, momentum, and shear from the quadrature. Generalize the Einstein sources to these sums; a massive species does not have \(\delta P=\delta\rho/3\). Use equation 58 as the starting closure and verify convergence in both momentum nodes and multipole depth. Keep the cheaper massless integrated hierarchy. Choose the starting epoch so the requested modes are superhorizon and the species are relativistic as assumed by the initial series; otherwise derive an appropriate alternative initialization.

5. **Replace the power-only interface.** Return signed \(T_c,T_b,T_\nu,T_{cb},T_m\), the relevant velocity transfers, epoch, gauge, units, and normalization. Compute \(T_m\) from background density weights at the requested epoch. Generate cold/baryon initial conditions directly at their starting redshift using common random phases. Independent reference comparisons must use matching transfer definitions; [CAMB documents its component and velocity conventions](https://camb.readthedocs.io/en/latest/transfer_variables.html).

6. **Specify neutrinos during subsequent simulation evolution separately.** The linear Einstein–Boltzmann solver does not require neutrino particles. A later simulation feature must choose homogeneous background only, linear response, or neutrino particles, and document that approximation. Do not assign massive-neutrino density to cold/baryon particle masses.

Suggested implementation units are a shared cosmology background, explicit species/state layout, pure perturbation right-hand side, collision/tight-coupling implementation, adaptive integrator, and transfer-output adapter. Preserve an external executable/file interface for independent CAMB reference runs.

## Acceptance tests

Use a proposed initial target of \(10^{-3}\) relative error in the delivered matter power over a declared \(k,z\) domain, with absolute tolerances for quantities crossing zero. This is a target, not an accuracy achieved by the present solver.

- Verify adiabatic initial series, matter-era growth, Einstein constraints, and photon–baryon momentum exchange.
- Verify free streaming against spherical-Bessel solutions and resolve errors by increasing multipole depth.
- Verify tight coupling, the transition out of it, timestep refinement, and recombination table refinement independently.
- Verify zero abundance and continuous recovery of the massless solution as masses tend to zero.
- Verify massive-species background conservation and convergence in momentum quadrature, hierarchy depth, and starting epoch.
- Compare signed component transfers and final power against CAMB or CLASS with identical cosmology, recombination assumptions, gauge, units, epochs, and normalization. Hold primordial amplitude fixed when measuring physical neutrino suppression.
- Check Fourier-grid coverage, signed-transfer combination, component mass accounting, and particle velocity conversion in the initial-condition pipeline.

## Evidence and validation boundary

Evidence is stored in `docs/boltzmann_audit_results_20261020/`. The scratch harnesses extract the production evolution functions while excluding HPX dispatch. Step-refinement variants change the maximum-step constant only. The production-background fixture uses the actual background and RECFAST reader, with the cached fiducial ionization table and correct physical radiation densities; it does not exercise the defective option-density calculation.

A fresh standalone compile of `tests/recombination_background.cpp`, `src/cosmology/zero_order.cpp`, and `src/cosmology/recfast.cpp` passed with the existing fiducial history. It gave \(T_{\rm gas}(z=50)=50.4544324\) K and final \(c_s^2/c^2=2.24391167\times10^{-15}\). This confirms the focused background regression, not the perturbation solver.

The audit includes algebra checks, extracted-source tests, and an ASan failure reproduction. It does not include a full application, CUDA, MPI, or independent CAMB/CLASS comparison. No QB login was attempted.
