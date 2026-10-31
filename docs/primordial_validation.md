# Primordial chemistry validation — 2026-10-18

Validation-history note (2026-10-30): records below concerning signed-complement storage, FloatFloat and the coupled Newton solver describe the preceding implementation. The current solver uses direct nine-species storage and the templated Anninos algorithm; current evidence is recorded in [the Anninos validation](anninos_validation.md). See [the current solver](primordial_chemistry.md).

Repository: `/home/dmarce1/workspace/cosmictiger`, branch `main`, base commit
`f325926adb3489373c72ce121f3277b49df3c28f`. These results cover uncommitted
implementation work. The applied recovery stash
`5dbbfc0be00d05668e0d5a3fc5b3f518dec077c7` remains available.

## Implementation covered

- Shared atomic6/molecular9 rates, EOS and bounded coupled implicit integration.
- Signed-complement FP32 fractions, including negative zero; thermal energy in
  eV per H nucleus; targeted FloatFloat cancellation-sensitive arithmetic.
- Boost.Units density, energy and time boundaries, with an absolute electron-volt
  anchor and round trips through comoving/stored quantities.
- CPU batching and CUDA batching; CPU-only application build and CPU selection
  in a CUDA-enabled application without CUDA runtime calls.
- Composition-dependent pressure, sound speed and expansion work in CPU/CUDA
  SPH, synchronized source phases, particle migration, versioned checkpoints,
  radiation provenance and Silo fields.
- Earlier RECFAST and recombination-history corrections.

## Build configurations

Both configurations use Release builds, GCC/GFortran, the existing HPX installation
at `/home/dmarce1/local/hpx/Release`, and the same checked-out source.

```sh
cmake --build build/sph_cpu_validation --parallel 3
cmake --build build/release --parallel 3
```

`build/sph_cpu_validation` has `COSMICTIGER_WITH_CUDA=OFF`.
`build/release` has `COSMICTIGER_WITH_CUDA=ON` and CUDA architecture 86.
Both complete default builds passed, including the application, utilities and
registered test executables. CUDA runtime tests ran on an NVIDIA RTX 3090.
The CPU build does not compile CosmicTiger CUDA translation units. The supplied
HPX installation can still link CUDA libraries; guarded CPU execution establishes
that those libraries are not used for device work in the tested CPU paths.

## Reproducible checks

Run from the repository root:

```sh
ctest --test-dir build/sph_cpu_validation --output-on-failure --parallel 1 \
  -R '^(primordial_.*|cpu_only_runtime|recfast_.*|recombination_.*|sph_cpu)$'

ctest --test-dir build/release --output-on-failure --parallel 1 \
  -R '^(primordial_rates|primordial_solver|primordial_units|primordial_radau_reference|cpu_only_runtime|primordial_cuda|monaghan_sph_cuda)$'

python3 tests/run_primordial_integration.py build/release/cosmictiger power.init \
  build/release/tests/primordial_integration_cuda --cuda

CUDA_VISIBLE_DEVICES='' LD_PRELOAD="$PWD/build/release/libno_cuda_runtime.so" \
  ctest --test-dir build/release -R '^primordial_integration_cpu$' --output-on-failure
```

The CPU selection passed 14 tests. The CUDA selection passed 7 tests, including
actual device execution. These are focused selections, not a claim that every
unrelated repository CTest was run.

The four existing `float_float_*` regressions also passed in the CUDA build,
covering CPU and device implementations in standard and fast-math test modes.
The production chemistry library itself retains strict arithmetic flags.

Both networks passed two production iterations followed by restart from checkpoint
1 to checkpoint 2, with exact gas checkpoint payload agreement. This passed in
CPU-only, CUDA, and guarded CPU mode of the CUDA-enabled executable. The harness
also checks incompatible network/radiation rejection, unsupported molecular
temperature rejection, and requested chemistry Silo fields in all four physical
unit/frame combinations. The first CPU suite was followed by a separate Silo
readback; later integration runs include that check directly.

Two-locality MPI tests passed storage/migration/cache checks with chemistry enabled
and disabled. Two-locality production atomic6 and molecular9 runs also reproduced
gas checkpoint bytes on each rank after restart. Devices were hidden and CUDA
calls forbidden throughout these CPU MPI runs. MPI CUDA execution was not tested.

The new SPH pressure regression was also linked against the original hydro
implementation: it failed the composition-dependent pressure assertion, while
the changed implementation passes.

## Precision and scientific interpretation

The device disassembly of `libprimordial_chemistry.a` contains FP32 arithmetic and
no `DADD`, `DMUL`, `DFMA`, `DSETP`, `DSET` or `F64` instructions. This inspection
covers the chemistry kernel, not the whole simulation. Host unit conversion,
global cosmology and diagnostics still use double. No CPU/GPU speedup claim is
made by these correctness tests.

Independent SciPy Radau integration uses the production physical RHS with a
different integrator and Jacobian/error controller. It checks numerical integration;
it does not independently validate physical rate fits. Separate tests cover
coefficient anchors, integer stoichiometry, charge/nuclei closure, EOS derivatives,
signed minorities, invalid inputs, work limits and endpoint states.

The molecular-rich reference starts with H2 nuclei fraction 0.2, H+ fraction
1e-4, nH = 1 cm^-3 and 2000 K, then evolves for 1e12 seconds. Radau gives
1023.303 K and 0.137117190 eV/H, from 0.271709859 eV/H initially. Refining the
independent reference changes its endpoint by 3.01e-7 in the scaled state norm.

| Local tolerance | Maximum scaled state error | Relative thermal-energy error |
| --- | --- | --- |
| 1e-3 | 6.982e-3 | 3.818e-3 |
| 1e-4 | 2.058e-3 | 1.128e-3 |
| 1e-5 | 5.494e-4 | 3.008e-4 |

A final audit found and reproduced a false Newton-convergence case in a
quadratic trace-loss problem: the initial finite-difference floor made a large
residual look like a small correction, returning unchanged abundance with
`Success` and approximately 1000 times the correct abundance. Minority-scaled
probes and stricter termination now recover the roughly thousandfold depletion
to 0.64% global error at default local tolerance. Regressions cover direct and
complement encodings at 1e-13 and 1e-20 on CPU and CUDA. Exact hot/cold endpoints
remain covered. This is an integration regression, not an additional physical
reaction model.

The current solver is adaptive backward Euler with step doubling, a first-order
baseline. Local tolerance is not a bound on global trajectory error. The reaction
catalogue documents unresolved minor-channel product-state thermal partition.
This original validation covered atomic6 at 10–1e8 K and molecular9 at
100–6000 K. The later [10 K extension](molecular9_cold.md) and
[10⁸ K extension](molecular9_hot.md) have separate evidence.
Both stop above
1e4 H nuclei/cm³. HD/deuterium, shielding/transport, metals/dust, molecular CMB
line pumping and high-density collapse physics are outside the implementation.
Broad cooling-shock, collapse, matched-library and cosmological convergence
campaigns remain separate validation work.

The evidence archive includes before/after failure reproducers and initial
build/test logs for traceability. Final run logs are `ctest_cpu_final.log`,
`ctest_cuda_final.log`, `integration_cuda_final.log`,
`integration_cuda_build_cpu_final.log` and `mpi_production_final.log` under
`work/chemistry_implementation`. Complete default-build logs are
`build_cpu_last.log`/`build_cuda_last.log`; the final compile-time result-unit
strengthening is checked in `build_cpu_typed.log`/`build_cuda_typed.log`.
Its float arithmetic is unchanged, and the typed boundary regression was rerun.

See [usage and precision](primordial_chemistry.md),
[reaction sources and limits](primordial_reactions.md),
[CPU portability](chemistry_cpu_support.md) and
[staged design and remaining gates](primordial_chemistry_plan.md).
