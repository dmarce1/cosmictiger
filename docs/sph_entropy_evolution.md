# SPH entropy evolution

Implementation scope: November 3, 2026. The user requested changing the evolved
thermal variable from internal energy to the GADGET-style entropic function,
with minimal testing. Cooling and star formation, including their tests, are
temporarily disabled pending separate source conversions.

## Equations and state

The canonical thermal array is `sphParticlesEntropies`, storing
A = P_physical / rho_physical^gamma. Density remains comoving, rho_c = a^3 rho,
and the velocity convention remains w = a v_pec. Readers derive the former
scaled energy E = a^2 u on demand:

    E = A rho_c^(gamma-1) a^(5-3 gamma) / (gamma-1).

There is no persistent energy cache. Pressure, sound speed, output and thermal
energy diagnostics retain their former units. Transient force-staging buffers
still contain derived energies for the CPU/CUDA pair kernels.

Before the first converged density solve, the single thermal array holds the
initial E supplied by scenario initialization. The collective preparation before
the first force evaluation converts those values to A exactly once, using the
actual SPH density. Thereafter `sphParticlesInternalEnergy` is a value-returning
conversion, and explicit energy setters convert back to A. No energy reference
can bypass the conversion. A locality-wide initialization flag and scale factor
are checkpointed; all localities prepare them before any force staging begins.

Only artificial-viscosity heating Q_E contributes to the evolved rate:

    dA/dtau = (gamma-1) a^(3 gamma-5) rho_c^(1-gamma) Q_E.

Reversible pressure work and homogeneous expansion do not kick A. The force
kernels still calculate total dE/dtau as an optional diagnostic and reversible
work for the existing timestep bound. These are not evolved thermal state.
For gamma=5/3 the scale-factor exponent is exactly zero (also in float kernels).

## Integration and boundaries

1. Stage active predicted A = A_half + Adot_old dt_old/2 and velocity, deriving E
   with the current density and scale factor. Stored state is unchanged.
2. Evaluate forces and the viscous entropy rate on CPU or CUDA.
3. Close A and velocity with the fresh rates over dt_old/2.
4. Commit and limit the next rungs, then open with the reused rates over dt_new/2.

Inactive-neighbor velocity/entropy time prediction is unchanged by this task;
so are gravity coupling, rung selection and the existing viscosity switches.
Initialization, swapping, migration, and checkpointing carry the canonical value
and rate. Gas checkpoint versions are now 30000+, explicitly rejecting previous
energy-based payloads. Checkpoint readers expose derived internal energy while
retaining raw entropy records for exact restart comparisons.

Cooling/star-formation options fail explicitly, and source test registrations and
source subcases in shared Python suites are disabled. Their conversion physics
is deferred, not validated or silently approximated.

## Validation scope

Completed minimal checks:

- SPH-only CPU executable build and two focused entropy kernel tests passed.
- The SPH-only lifecycle fixture passed initialization, on-demand energy,
  prediction, closing/opening kicks, and checkpoint restoration.
- A two-iteration, 8^3-initial-grid Sod smoke run completed.
- All 16 Python checkpoint/parser regression tests passed.
- Full-feature migration, particle sorting, cosmology initialization, and deferred source files
  passed syntax checks; the changed CUDA hydro translation unit compiled.

No full suite, multi-locality evolution, convergence study, or GPU execution was
performed. War Thunder was ON: at most two compiler/runtime workers were used.
The checkpoint format intentionally rejects previous energy-based gas payloads.
