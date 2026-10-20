# Native initialization and dependency removal, 2026-10-20

The runtime now uses native recombination and internally generated linear
power. Bundled RECFAST, its Fortran build requirement, the `power.init`
reader, and `initial.usePowerFile` were removed. Offline CAMB comparison tools
and frozen RECFAST reference data remain for validation.

With chemistry enabled, `chemistry.initialization = recombination` supplies
the initial gas temperature and H/He ion fractions at the requested starting
redshift. Each explicit scalar override takes precedence independently.
`chemistry.initialization = manual` retains the previous scalar defaults.
Restart restores the saved particle state. Molecular abundances retain their
existing explicit/default values.

This archive covers the removal and initialization phase, before the separate
10 K extension. `removal_hashes.json` identifies that earlier build;
`evolution_hashes_before_cold_extension.json` records unchanged SPH and
chemistry evolution files during that phase. Final application tests after
the cold extension are recorded in the
[cold validation archive](../molecular9_cold_results_20261020/README.md).

| Evidence | Result |
| --- | --- |
| `no_fortran_configure.log` | Fresh configure succeeded with a nonexistent Fortran compiler. |
| `build.log` | CPU application build succeeded. |
| `core_tests.log` | All 13 focused chemistry, recombination, background, and linear-power tests passed. |
| `ic_checkpoint_tests.log` | Both Boltzmann initialization modes, SPH CPU, and checkpoint integrity passed. |
| `camb_interface/` | Native-only reference executable agreed with CAMB at k = 0.01, 0.1, 1 Mpc⁻¹, z = 49 and 0, and N_eff = 0 and 3.046. |

Across those 12 sampled CAMB rows, the largest relative error was
9.070 × 10⁻⁵ for a density transfer and 2.052 × 10⁻⁴ for cold-dark-matter plus
baryon power. No fitted normalization was used. These are interface
regressions on a small grid, not a replacement for the broader
[earlier solver validation](../boltzmann_validation_results_20261020/).
The comparison JSON files preserve the original scratch paths; their CSV
inputs and outputs are copied alongside them here.

The earlier native/RECFAST validation is retained in
[native recombination results](../native_recombination_results_20261020/).
Current usage is documented in [native recombination](../native_recombination.md).
