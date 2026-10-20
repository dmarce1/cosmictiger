# Native homogeneous recombination validation, 2026-10-20

This archive records validation before removal of the bundled RECFAST executable
and table reader. Historical commands and hashes are preserved as evidence.
Current CTest comparisons use the saved RECFAST CSV files; the Boltzmann reference
driver now always uses the native history and no longer accepts a history-file
argument. See [the current model documentation](../native_recombination.md).

This directory records independent reference histories and validation of the
native homogeneous hydrogen/helium recombination calculation. Final results
use the production CMake targets `recombination_reference` and
`boltzmann_reference`, built from the actual native implementation. Run
metadata records executable and source hashes.

The history and matter comparisons are plotted in
[native_history_comparison.pdf](native_history_comparison.pdf) and
[native_matter_comparison.pdf](native_matter_comparison.pdf), with PNG copies
alongside them. `tests/plot_native_recombination.py` regenerates the figures
from the saved numerical results.

## Matched physical model

The two models have `N_eff=0` and `N_eff=3.046`, with massless neutrinos only.
Both use `h=0.6732`, `Omega_b=0.049389`, `Omega_c=0.26503`,
`T_CMB=2.7255 K`, and `Y_He=0.2454006`. The background is spatially flat.
The RECFAST input includes the native radiation density explicitly.
Reionization is disabled. Electron fractions mean `n_e/n_H`; gas
temperatures are in kelvin. Photon and matter temperatures are distinct
after thermal decoupling.

`tests/compare_native_recombination.py` generates the references, runs the
native test driver, and compares common redshift samples. Every numerical
process uses one CPU thread. CAMB is an isolated external Python dependency;
CosmicTiger does not link to it.

## Independent references

`recfast` contains 1,000 samples per model, at `z=9990, 9980, ..., 0`.
A temporary copy of the bundled Fortran source adds gas temperature to the
output and increases printed precision. Only two output statements change;
the equations, integration, and bundled source remain unchanged.
`output_instrumentation.patch` records the exact change and
`provenance.json` records source, executable, and compiler information.
The extra `x_H` and `x_He` output columns are RECFAST's ODE bookkeeping
variables; the early helium stages use a separate Saha electron fraction,
so `x_He` alone must not be interpreted as the physical He II abundance
there.

`camb` contains 2,550 samples per model: the RECFAST nodes, a denser grid
through hydrogen and helium recombination, and a logarithmic grid through
the residual and early ionized eras, up to `z=1e8`, plus every integer
redshift from 0 to 20. CAMB 1.6.5 uses its
RECFAST model with `Hswitch=1`, `Heswitch=6`, hydrogen fudge factor 1.125,
helium fudge factor 0.86, and accuracy boost 3. Full parameter files are
retained.

The standalone RECFAST and CAMB references are not numerically identical.
On their 1,000 common samples, the largest ionization differences are
0.139% in the helium epoch, 0.0474% in the hydrogen epoch, and 0.00109% in
the residual era. The largest gas-temperature difference is 0.00480%.
These are reference-to-reference differences, before comparison with the
native implementation; `recfast_vs_camb` contains the detailed results.

## Native ionization and temperature

The default native integration uses relative tolerance `1e-7` and maximum
log-scale-factor step `0.01`. The refined integration uses `1e-8` and
`0.005`. Both models and both precision settings completed all 2,550 samples.
The table gives maximum absolute relative differences at default settings.
Percentages apply separately to each quantity.

| N_eff | Reference | Samples | Electron fraction | Gas temperature |
| --- | --- | ---: | ---: | ---: |
| 0 | Standalone RECFAST | 1,000 | 0.06703% | 0.006622% |
| 3.046 | Standalone RECFAST | 1,000 | 0.06773% | 0.006632% |
| 0 | CAMB at integer redshifts | 2,151 | 0.12973% | 0.01142% |
| 3.046 | CAMB at integer redshifts | 2,151 | 0.12045% | 0.01143% |

The largest ionization differences occur during helium recombination.
Refining the native integration changes electron fractions by at most
`3.8230e-5` relative and temperature by `7.5795e-8` relative. The remaining
reference differences are larger than the measured numerical refinement.
`final_history` retains the full histories, epoch-specific comparisons,
refinement comparisons, and provenance. `final_summary.json` contains the
unrounded maxima.

### CAMB's low-redshift interpolation

The full dense CAMB comparison is retained in each
`camb_full_comparison` directory. It shows a 5.36% temperature difference
near `z=0.258925`, attributable to the reference interpolation. CAMB's
RECFAST implementation uses a fixed redshift spacing of one and interpolates
temperature with a cubic spline. Its natural endpoint condition distorts the
nearly quadratic, adiabatically cooling temperature between the final
integer nodes when reionization is disabled.
[CAMB 1.6.5 source](https://github.com/cmbant/CAMB/blob/1.6.5/fortran/recfast.f90#L358-L381)

At this fractional redshift for `N_eff=3.046`, native temperature is
`0.0344597 K`, CAMB returns `0.0364126 K`, and the adiabatic estimate from
CAMB's present temperature is `0.0344663 K`. Independently constructing a
natural cubic spline through CAMB's integer nodes reproduces the fractional
CAMB value to `4.51e-10` relative. The reproduction is saved in
`camb_low_z_interpolation/probe.json`. At the actual integer nodes, the
native temperature agrees within the bounds in the table above.

The full dense comparison deliberately keeps its failed 1% diagnostic flag;
the reference-node comparison and independent standalone RECFAST comparison
are the temperature accuracy checks. No reference data or discrepant rows
were discarded.

## Matter transfers and initial displacement velocities

The final native-background transfer campaigns cover 31 logarithmically
spaced modes from `1e-4` to `10/Mpc`, at redshifts 49 and 0, with both
`N_eff=0` and `N_eff=3.046`: 124 combinations per precision setting. The
default Boltzmann calculation uses multipole ceiling 64 and relative
tolerance `1e-6`; its refinement uses 128 and `1e-7`. Both use the default
native recombination history, so the Boltzmann convergence check is separate
from recombination convergence. No amplitude fitting is performed.

| N_eff | Boltzmann settings | CDM density | Baryon density | Combined density | Combined power |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Default | 0.01140% | 0.01140% | 0.01140% | 0.02278% |
| 0 | Refined | 0.001999% | 0.001994% | 0.001994% | 0.007861% |
| 3.046 | Default | 0.01971% | 0.01800% | 0.01946% | 0.03987% |
| 3.046 | Refined | 0.005142% | 0.004773% | 0.005091% | 0.01113% |

These are maximum absolute relative differences from CAMB. The independent
metric trace diagnostic stays below `8.967e-7` at default settings and
`1.577e-7` at refined settings.

Additional checks separate the numerical choices:

- At fixed Boltzmann tolerance `1e-7`, raising the multipole ceiling from
  64 to 128 changes matter transfers by at most `1.8300e-4` relative and
  combined power by `3.6565e-4`.
- At fixed ceiling 64, tightening Boltzmann tolerance from `1e-6` to
  `1e-7` changes transfers by at most `9.5153e-5` and power by `1.9030e-4`.
- Holding the refined Boltzmann settings fixed, refining only the native
  recombination calculation changes matter transfers by at most `4.8750e-8`
  and combined power by `4.1571e-8`. This last check uses
  `k=0.01, 0.1, 1, 10/Mpc`, both epochs, and both neutrino models.

`final_growth_probe` independently checks the initial displacement velocity
using CAMB's exact metric right-hand side:
`delta_cb' = -h'/2 - f_b (k/Hc) (theta_b/k)`. For `N_eff=3.046` at the four
modes above and both epochs, its maximum relative difference is
`1.9357e-4` at default settings and `5.0765e-5` at refined settings. The
logarithmic growth rate `delta_cb'/delta_cb` agrees within `4.0860e-6` and
`6.5995e-7`, respectively.

## Reproduction

From the repository root, with the isolated CAMB environment available:

```sh
python tests/compare_native_recombination.py recfast \
  --output-dir /tmp/native_recombination_validation/recfast

python tests/compare_native_recombination.py camb \
  --output-dir /tmp/native_recombination_validation/camb

python tests/compare_native_recombination.py native \
  --executable /path/to/recombination_reference --n-eff 3.046 \
  --max-log-step 0.01 --rtol 1e-7 \
  --redshifts /tmp/native_recombination_validation/camb/redshifts.txt \
  --output-dir /tmp/native_recombination_validation/native

python tests/compare_native_recombination.py compare \
  --reference /tmp/native_recombination_validation/camb/neff_3_046.csv \
  --candidate /tmp/native_recombination_validation/native/history.csv \
  --output-dir /tmp/native_recombination_validation/comparison \
  --integer-redshifts
```

Repeat the native run with `N_eff=0` and matching reference files. Numerical
refinement changes the integration tolerance and maximum log-scale step
independently of the reference calculation.

Matter-transfer validation uses `tests/camb_reference.py run
--native-recombination`. Its C++ driver calls the production
`createNativeZeroOrderUniverse` helper, using both the native electron
fraction and native gas temperature. The previous CAMB signed-density and
power reference grid is retained in
`../boltzmann_validation_results_20261020/camb_high_accuracy`.
The curvature sign, units, primordial spectrum, and absence of amplitude
fitting follow that directory's validation report.

The standard-library-only CI regression can be run without installing CAMB:

```sh
python tests/compare_native_recombination.py check-recfast \
  --native-executable /path/to/recombination_reference \
  --recfast-executable /path/to/recfast \
  --workdir /tmp/native_recfast_regression
```

It checks 1,000 electron fractions for each neutrino model with a 0.2%
relative bound. The independent full history campaign above additionally
checks gas temperature. The final CI run passed with a maximum electron
fraction difference of 0.06773%; results are retained in `final_recfast_ci`.

The final repository validation also passed 16 focused CTests and three
application tests. Native AddressSanitizer/UndefinedBehaviorSanitizer and
direct fast-math checks passed. Both first-order and second-order internal
initialization application runs succeeded with the RECFAST executable
absent. Eleven SPH chemistry files retained identical content hashes.
`application_checks/manifest.json` and its accompanying logs record this
application-level evidence.

These checks validate homogeneous recombination and linear matter transfers.
They do not establish local SPH chemistry accuracy, reionization modeling,
massive-neutrino behavior, or CMB-spectrum precision. The reference campaigns
used at most two concurrent CPU workers. Repository builds used up to four
workers, and all concurrent task work stayed within the shared 12-core
limit. No GPU or QB access was used.
