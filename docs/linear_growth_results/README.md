# Linear-growth measurements, 2026-10-17

The interpretation, equations, scope, and commands for rerunning the particle
experiments are in [the study report](../linear_growth.md). These files retain
the actual measurements and derived tables used by that report.

The source checkout was based on commit
`cebcb9eaed38a064fa6c14b6c5fb2f7c321bfc4d`, with uncommitted adaptive-softening
implementation and diagnostic changes. This is not a measurement of the clean
commit. `manifest.json` records source and artifact hashes at archival time.

## What ran

All cases use Ferrers F2 gravity. Adaptive cases use 24 weighted neighbors and
the conservation terms. Mixed cases contain equal particle counts with class
mass fractions `f_DM=0.8429197981`, `f_gas=0.1570802019`. Gas is pressure-free
throughout this diagnostic; the experiment does not exercise SPH pressure.

| Directory | Actual calculation |
| --- | --- |
| `adaptive_mixed_n8` | CUDA, two species, N=8 each, theta=0.1, eight wavevectors, central amplitudes 0.01 and 0.005 spacings; six particle evolutions |
| `adaptive_dm_n8` | CPU, one species, N=8, theta=0.1, five wavevectors, both central amplitudes |
| `fixed_mixed_n8` | CPU, two species, N=8 each, theta=0.1, three axis wavevectors, both central amplitudes; fixed support matches the adaptive mixed baseline |
| `adaptive_mixed_n16` | CUDA, two species, N=16 each, theta=0.1, axis modes 1 and 2, amplitude 0.005 spacing |
| `pilot_mixed`, `pilot_cuda` | CPU/CUDA comparison, two species, N=8, theta=0.2, axis mode 1, amplitude 0.01 spacing |
| `regression_cpu_n8` | Additional CPU comparison, two species, N=8, theta=0.1, axis mode 1, both amplitudes; the bounded CTest regression passed |
| `evolution_controls_n8` | CUDA particle evolutions: short target-matched mode with 16 RK4 steps and with half amplitude at 8 steps |

The six primary evolutions span `a=0.02` to `0.1` with eight logarithmic RK4
steps. Modes `(1,0,0)` and `(3,0,0)` each use a common continuum IC, a numerical
growing eigenmode, and a growing eigenmode whose amplitude matches continuum
total particle-center density at the final epoch. Their underlying continuum
initial displacement amplitude is `0.001` spacing. Every RK4 stage recomputes
production gravity. This validates the spatial response and linear growth
equation, not the production driver's timestep scheme.

## Files and units

- `linear_growth_response.csv` is the measured Jacobian with sine and cosine
  projections. It is normalized by `4*pi*G*rho_comoving`; support radii and
  `realized_amplitude` are in box-length units.
- `linear_growth_metadata.json` records the actual particle count, class
  masses, cosmology, kernel, backend, force tolerance, support, and baseline
  force residual for each run.
- `linear_growth_corrections.csv` contains the complete growing-branch
  eigenvector, all eigenvalues, density and momentum correction factors, and
  numerical-quality measures. The primary correction targets particle-center
  density at `a=0.1`; it is neither kernel-density deconvolution nor a 2LPT
  correction.
- `linear_growth_evolution.csv` contains actual particle projections in
  lattice-spacing units. `v` in this file means `du/dln(a)`. The simulation's
  stored momentum is `p=a^2*H*v`.
- `measured_evolution_requests.csv` preserves the exact initial states and
  step counts actually evolved. `evolution_summary.csv` and
  `evolution_comparison.csv` compare complete trajectories with an independent
  linear ODE using the measured matrix.
- `all_corrections.csv`, `response_quality.csv`, `backend_comparison.json`,
  and `grid_comparison.json` collect comparisons across cases.
- PNG and PDF figures are derived from these CSV files; the PDFs are suitable
  for export. The full complex backend-comparison norm includes both sine and
  cosine projections.

The analysis rejects incomplete runs or steps, missing or duplicate vector
components, nonfinite values, mismatched labels or scale-factor grids, and
initial states that do not match their requests. Deliberately truncated,
missing-run, duplicate-component, and wrong-initial-state traces were rejected.

## Recreate derived results without rerunning particles

From the repository root, use Python with NumPy, SciPy, and Matplotlib:

```sh
python3 tests/analyze_linear_growth.py \
  docs/linear_growth_results/adaptive_mixed_n8/linear_growth_response.csv \
  docs/linear_growth_results/adaptive_mixed_n8/linear_growth_metadata.json \
  /tmp/linear_growth_reanalysis --a-initial .02 --a-final .1 \
  --evolution-modes '1,0,0;3,0,0' --evolution-amplitude .001 --evolution-steps 8 \
  --evolution docs/linear_growth_results/adaptive_mixed_n8/linear_growth_evolution.csv \
  --evolution-requests docs/linear_growth_results/adaptive_mixed_n8/measured_evolution_requests.csv

python3 tests/analyze_linear_growth.py --self-check
python3 docs/linear_growth_results/reproduce_comparison.py
```

The self-checks compare growth against exact Einstein-de Sitter and regular
Meszaros solutions and the Ferrers Fourier transform against independent
radial quadrature. The comparison script regenerates its plot and summary
tables from archived per-case corrections.

Normal cosmological IC generation remains unchanged. No performance benchmark,
long cosmological run, hydrodynamic growth measurement, or automatic IC
correction was performed as part of this study.
