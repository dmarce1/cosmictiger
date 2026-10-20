# Molecular9 10 K validation, 2026-10-20

The implementation and its limits are described in
[Molecular9 down to 10 K](../molecular9_cold.md). Primary paper and data-table
citations are also beside the production formulas. The 80–100 K compatibility
interpolation is explicitly identified as our numerical prescription.

## Physics and CPU checks

The [core checks](core_checks/README.md) include independent binary64
calculations, primary CDS data, reproducible scripts, and the original warm
headers. All 12 tests in the broader focused campaign passed. A final rebuild
after an EOS optimization passed the four affected rate, solver, and native
recombination tests; see `final_cpu_tests.log`. Standard and fast-math native
recombination builds are tested separately.

- All 29 collisional rates and five molecular cooling coefficients remain
  finite and nonnegative on the sampled cold grid.
- 6,552 cold RHS/EOS states exercise 10–100 K, six compositions, four densities,
  and three CMB temperatures, including zero and nonzero H₂⁺.
- The cooling calculation agrees with independent evaluation of the published
  polynomial coefficients. Comparison with the separately tabulated upward
  rates has a maximum sampled difference of 3.1662%, within a 3.5% allowance
  for the polynomial fit.
- Tests check the value and first derivative at both matching boundaries,
  EOS inversion, independent partition sums, analytic Compton exchange,
  molecular cooling convergence, and subnormal H₂⁺ saturation.
- The warm comparison covers 19,920 states from 100 to 6000 K. Rates, fluxes,
  EOS, thermal ledger, and RHS are bitwise identical before and after the
  extension: 12,908,160 bytes with SHA-256
  `8f3b19e7be19c7c1616bb0fe343a985197f1786e3d619906139686f3b57cff58`.
  This is sampled preservation evidence, not an exhaustive proof over all
  floating-point inputs.

The cold solver convergence test uses an energy absolute tolerance of
10⁻¹⁰ eV per hydrogen nucleus so that relative-tolerance refinement is
meaningful. Production defaults and the source integration algorithm were
not changed.

## CUDA checks

`final_cuda_build.log` and `final_cuda_test_host.log` record the rebuilt
chemistry test and successful execution on the actual local GPU. This was
not a skip caused by unavailable sandbox device access.

The test compares cold CPU/CUDA rates, EOS, and RHS, including subnormal
molecular-ion cooling. It also follows an independently predictable heating
case across 100 K. Existing stiff-network, tail, trace, and failure cases
remain covered. Launches use at most five blocks of 64 threads. This is
chemistry kernel validation, not a full CUDA SPH simulation.

## Application checks and provenance

`application_checks/` contains dedicated CTest output, individual run logs,
and source/executable hashes captured before and after the final application
campaign. All three final application CTests passed in 214.16 seconds, with
all 13 captured source/binary hashes unchanged. The tests cover:

- Native atomic and molecular initial temperature/ionization, measured from
  all particles in an unevolved checkpoint against a separate native-history
  reference.
- Explicit CLI/INI overrides, manual defaults, invalid settings, and rejection
  below the supported temperature range.
- Production atomic, cold molecular, and legacy warm molecular evolution,
  with exact checkpoint restart and incompatible-network rejection.
- Molecular formation from initially zero default molecular abundances in
  the cold native case.

`manifest.json` identifies final source and executable hashes and records
build/test commands. Earlier application runs are kept in the explicitly
labeled `prior_before_final_eos_reuse/` directory; their archive-time executable
hash is not claimed as the hash of the binary they tested.

The separate [native initialization archive](../native_initialization_results_20261020/README.md)
records RECFAST/Fortran and runtime power-file removal, the native-only CAMB
interface regression, and earlier initialization/checkpoint checks.

These tests do not establish a molecular radiation model or remove the
existing fixed spin-mixture, density, and chemical-species limits. Molecular
CMB absorption, HD, metals/dust, shielding, and high-density collapse remain
outside this implementation. The 10 K boundary is a validity check, not a
temperature floor.
