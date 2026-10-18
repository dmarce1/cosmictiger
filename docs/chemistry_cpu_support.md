# Primordial chemistry: CPU and CUDA support

The primordial `atomic6` and `molecular9` networks have CPU and CUDA batch implementations. Cosmological SPH stages composition dependent pressure and sound speed into both hydrodynamic backends. `compute.cuda=false` selects the CPU solver, SIMD hydrodynamics, CPU gravity and CPU drift. It does not silently dispatch chemistry to a GPU.

`COSMICTIGER_WITH_CUDA=OFF` omits CosmicTiger's CUDA sources. A CUDA-enabled executable also supports `compute.cuda=false`: particle, tree, chemical state and staging allocations then use host memory, and startup skips device initialization. An externally supplied HPX installation can still link CUDA libraries; a linked library is distinct from using a device.

| Facility | CPU mode |
| --- | --- |
| Primordial rates, EOS and stiff source integration | Implemented |
| Cosmological SPH pressure, sound speed and expansion work | Implemented |
| Particle sorting, domain migration and checkpoints | Chemical state and EOS cache included when enabled |
| Legacy gas with `chemistry.enabled=false` | Existing ideal-gas path and SPH checkpoint payload retained |
| Lightcone group finding (`lightcone.enabled=true`) | Rejected; requires CUDA |
| Power-spectrum density accumulation (`analysis.power=true`) | Rejected when invoked; requires CUDA |
| Sphere-surface test and stellar sphere initialization | Rejected; their surface solver requires CUDA |

Use the existing dependency configuration, with `-DCOSMICTIGER_WITH_CUDA=OFF -DBUILD_TESTING=ON`, for a CPU build. For the existing validation build directory:

```sh
cmake --build build/sph_cpu_validation --parallel 3
ctest --test-dir build/sph_cpu_validation --output-on-failure \
  -R '^(sph_cpu|primordial_rates|primordial_solver|primordial_units|cpu_only_runtime)$'
```

A cosmological CPU run needs `--compute.cuda=false --hydro.enabled=true --chemistry.enabled=true`, the selected `--chemistry.network=atomic6` or `molecular9`, and `--lightcone.enabled=false --analysis.power=false`. Supply the usual initial-condition and cosmology configuration. Chemistry uses a common rung, selected with `chemistry.synchronizedRung`, and rejects a step when the hydrodynamic or gravity bound requires a finer rung. It is currently restricted to cosmological SPH initial conditions; the `sphGravity` regression is an explicit storage-test exception.

The Linux `cpu_only_runtime` regression hides all devices with `CUDA_VISIBLE_DEVICES=''` and preloads `no_cuda_runtime`. The guard terminates the process on device/context, allocation, transfer or launch calls. Compiler registration is allowed. It exercises driver checkpoint metadata, mixed-particle gravity, SPH allocation/sorting/checkpoint state, and chemical state storage. Run the same test against a CUDA-enabled build to check runtime CPU selection. A two-locality `--run.test=sphGravity --chemistry.enabled=true --compute.cuda=false --hpx:localities=2` invocation additionally exercises actual parcel migration and remote EOS/velocity cache reads.

Validation recorded during implementation on 2026-10-18: the complete CPU-only default build passed, including the application, utilities and all registered test executables; the rates, solver and unit-boundary regressions passed; the SIMD SPH suite passed with composition dependent force, sound-speed and expansion checks. Linking that new SPH suite against the original hydrodynamic implementation failed its composition-dependent pressure assertion, confirming that the test detects the missing coupling. Two-locality CPU migration/cache regressions then passed with chemistry both enabled and disabled. Both atomic6 and molecular9 also passed two production iterations and restart from checkpoint 1 to checkpoint 2, with identical gas payload bytes on each rank and validated manifest checksums. These MPI runs used two HPX threads per rank, hid all CUDA devices, and preloaded the CUDA-call guard. They are bounded integration checks, not convergence studies. The complete CUDA-enabled default build also passed. Its executable passed all three `cpu_only_runtime` guarded startup/storage/gravity cases with device visibility empty; full production atomic6/molecular9 stepping and exact restart also passed in that guarded CPU mode. Actual RTX 3090 source/hydro execution and restart passed separately. The [validation record](primordial_validation.md) distinguishes all build, CPU, GPU and MPI evidence.
