# External CAMB reference, 2026-10-20

These files compare the standalone CosmicTiger Einstein–Boltzmann transfer
driver with CAMB 1.6.5. CAMB runs in a separate Python process; CosmicTiger does
not link CAMB. `reference_environment.json` and `reference_requirements.txt`
record the reference environment. Every numerical process uses one CPU thread.

## Physical model and conventions

Both calculations use a flat cosmology with `h=0.6732`, `Omega_b=0.049389`,
`Omega_c=0.26503`, `T_CMB=2.7255 K`, and `Y_He=0.2454006`. Separate models use
`N_eff=0` and `N_eff=3.046`. Neutrinos are massless:
`mnu=0`, `num_massive_neutrinos=0`. Both use RECFAST, with reionization off;
CAMB's additional baryon temperature and ionization perturbations are off.
The saved CAMB parameter files contain the complete settings.

All wavenumbers are in inverse Mpc; all dimensional matter spectra are in
cubic Mpc. These are linear synchronous-gauge CDM and baryon perturbations.
The adiabatic initial condition has Ma–Bertschinger `eta -> +1` (`C=1/2`).
CAMB's native unit-curvature density outputs have the opposite sign; the
reference generator verifies `eta -> -1` at a superhorizon probe and negates
the density transfers. No amplitude fit or sigma-eight renormalization is
applied. The combined transfer is

```
delta_cb = (Omega_c delta_c + Omega_b delta_b) / (Omega_c + Omega_b)
P_cb(k) = 2 pi^2 A_s (k/k_pivot)^(n_s-1) delta_cb(k)^2 / k^3
```

Here `A_s=2.1e-9`, `n_s=0.966`, and `k_pivot=0.05/Mpc`. CAMB's independent
`delta_nonu` power interpolator supplies the reference `p_cb_mpc3`; the
direct-transfer reconstruction is retained as a cross-check.

## Reference precision

`camb_high_accuracy/camb_reference.csv` is the principal reference: 31
logarithmically spaced modes from `1e-4` to `10/Mpc`, at redshifts 49 and 0,
for both neutrino models (124 rows). The background calculation uses CAMB
accuracy boost 3 and the time-evolution hierarchy uses boost 5. Late radiation
truncation is disabled.

`camb_base` and `camb_convergence` check the reference through `1/Mpc` using
accuracy/hierarchy boosts 2/4 versus 3/5. `camb_high_k_check` and
`camb_high_k_convergence` perform the same check from `1` to `10/Mpc`.
The largest relative signed-density change is `2.30e-5`; the largest power
change is `1.72e-4`, including interpolation differences. In the principal
reference, direct-transfer and independent interpolated power agree within
`8.58e-5` relative.

`recfast_histories` contains histories generated with the corrected native
background parameters. Its provenance file records the source, executable,
and input/output hashes.

Directories beginning `pilot_`, `v3_`, or `v5_failed_` preserve intermediate
development results. They are not validation of the final solver.

## Native solver results

All four final campaigns completed. The default calculation uses maximum
multipole 64 and relative tolerance `1e-6`; the refined calculation uses
128 and `1e-7`. The entries below are maximum absolute relative differences
from CAMB, expressed as percentages, across all 31 modes and both epochs.

| N_eff | Settings | CDM density | Baryon density | Combined density | Combined power |
| --- | --- | ---: | ---: | ---: | ---: |
| 3.046 | Default | 0.01960% | 0.01785% | 0.01932% | 0.03958% |
| 3.046 | Refined | 0.005036% | 0.004633% | 0.004954% | 0.01085% |
| 0 | Default | 0.01140% | 0.01140% | 0.01140% | 0.02279% |
| 0 | Refined | 0.001998% | 0.001992% | 0.001993% | 0.008403% |

`final_summary.json` records the unrounded results. The default-to-refined
change is at most `1.4564e-4` relative for the individual matter transfers and
`2.8727e-4` for combined power. Separate checks isolate the two refinements:
at fixed tolerance `1e-7`, increasing the multipole ceiling from 64 to 128
changes matter transfers by at most `1.8279e-4` and power by `3.6553e-4`;
at fixed ceiling 64, tightening the tolerance from `1e-6` to `1e-7` changes
transfers by at most `9.5194e-5` and power by `1.9038e-4`.
The maximum normalized residual comparing the
independently integrated metric trace with the Hamiltonian constraint is
`8.967e-7` at default settings and `1.577e-7` at refined settings. The four
campaigns took approximately 8.8 seconds combined on this workstation,
including process startup; this is a focused test timing, not a simulation
performance benchmark. The final executable is the production CMake
`boltzmann_reference` target; run metadata records its SHA-256 and the hashes
of the inspected production sources and test driver.

Additional metric/velocity probes at `N_eff=3.046`, `k=0.01, 0.1, 1/Mpc`,
and both epochs are saved in `final_metric_velocity_probe.csv`. The largest
relative eta differences are `9.014e-5` and `1.280e-5` for default and refined
settings. The largest baryon-velocity differences are 1.22% and 0.395%, both
at `k=1/Mpc`, `z=0`. The matter-based error floor allows a larger relative
error in these small late-time velocities. The matter-transfer accuracy
above must not be applied to all state variables.

The CDM-plus-baryon displacement derivative used for initial velocities is
tested separately in `final_growth_probe`. The probe evaluates CAMB's exact
metric right-hand side using its symbolic API, then compares
`delta_cb' = -h'/2 - f_b (k/Hc) (theta_b/k)` at `k=0.01, 0.1, 1, 10/Mpc`
and both epochs. Its maximum relative differences are `1.9217e-4` at default
settings and `4.9410e-5` at refined settings. The growth rate
`f_cb = delta_cb'/delta_cb` differs by at most `4.0864e-6` and `6.6024e-7`,
respectively. Thus the larger relative error in the small baryon velocity
component does not cause a comparable error in the combined displacement
velocity.

## Reproduction

Create an isolated environment with the packages listed in
`reference_requirements.txt`. From the repository root:

```sh
python tests/camb_reference.py generate \
  --output-dir /tmp/camb_reference \
  --n-eff 3.046 0 --redshifts 49 0 \
  --kmin 1e-4 --kmax 10 --count 31 \
  --accuracy 3 --hierarchy-accuracy 5
```

Build the standalone `tests/boltzmann_reference.cpp` target, then run each
model against its matching RECFAST history. For example:

```sh
python tests/camb_reference.py run \
  --executable /path/to/boltzmann_reference \
  --history docs/boltzmann_validation_results_20261020/recfast_histories/neff_3.046/history.dat \
  --reference /tmp/camb_reference/camb_reference.csv \
  --n-eff 3.046 --max-k 10 --max-l 64 --rtol 1e-6 \
  --output-dir /tmp/native_reference

python tests/camb_reference.py compare \
  --reference /tmp/camb_reference/camb_reference.csv \
  --candidate /tmp/native_reference/transfers.csv \
  --output-dir /tmp/reference_comparison --tolerance 0.005
```

Repeat with `N_eff=0`, then with multipole ceiling 128 and relative tolerance
`1e-7` to check native numerical convergence. The `run` subcommand snapshots
the executable and history, retains per-mode raw output, and saves the
executable/history hashes and per-mode execution information. A failed run
has `"complete": false`; partial CSV output must not be treated as a complete
campaign.

To reproduce the displacement-derivative comparison, use the `probe-growth`
subcommand with the two native transfer files:

```sh
python tests/camb_reference.py probe-growth \
  --candidate /tmp/native_reference/transfers.csv /tmp/native_refined/transfers.csv \
  --native-parameters docs/boltzmann_validation_results_20261020/recfast_histories/neff_3.046/parameters.json \
  --output-dir /tmp/growth_comparison
```

This symbolic probe also requires SymPy and a Fortran compiler for CAMB's
temporary output callback. The recorded run used SymPy 1.12 and `gfortran`.

## Scope and sources

This validation targets signed matter density transfers and linear
CDM-plus-baryon power. It does not establish CMB spectra accuracy or the
accuracy of tiny late-time photon/neutrino multipoles. It does not test
massive neutrinos, nonlinear evolution, CUDA, MPI, or a full simulation run.

Conventions and configuration follow the official
[CAMB transfer documentation](https://camb.readthedocs.io/en/latest/transfer_variables.html),
[parameter documentation](https://camb.readthedocs.io/en/latest/model.html),
[evolution API documentation](https://camb.readthedocs.io/en/latest/results.html),
and [CAMB 1.6.5 source](https://github.com/cmbant/CAMB/tree/1.6.5).
