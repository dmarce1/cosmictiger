# Equilibrium cooling: CPU and CUDA support

GADGET-4 equilibrium H/He cooling has CPU and CUDA batch implementations.
`chemistry.cuda=false` selects CPU cooling. Hydrodynamics, gravity, drift, and
star formation select their backends independently with `hydro.cuda`,
`gravity.cuda`, `drift.cuda`, and `starFormation.cuda`. All default to `false`.
Hydrodynamics uses fixed gamma 5/3 for cooling gas and evolves its actual
thermal energy. The electron fraction is an equilibrium iteration guess.
See [the cooling model and options](primordial_chemistry.md) and
[the star-formation model](star_formation.md).

`COSMICTIGER_WITH_CUDA=OFF` omits CosmicTiger's CUDA sources. A CUDA-enabled
executable also supports all CUDA selectors set to `false`, using host allocations
and skipping device initialization. An external HPX installation may link CUDA
libraries without using a device.

Build cooling with `COSMICTIGER_ENABLE_SPH=ON` and
`COSMICTIGER_ENABLE_CHEMISTRY=ON`; add `COSMICTIGER_ENABLE_STARS=ON` for star
formation. Disabling particle cooling retains the independent homogeneous
cosmological recombination model. With SPH disabled, cooling is inactive.

A cosmological CPU run enables `hydro.enabled=true`, `chemistry.enabled=true`
and leaves every CUDA selector `false`. Cooling runs once for each completed selected SPH
interval. With star formation enabled, conversion uses the previously saved
SFR, followed by cooling and the new SFR. There are no opening chemistry
half-steps, outer source retries or chemistry timestep controls. State,
reordering, migration and checkpoints carry the electron guess and hydro
energy; star formation adds identity, event count, birth time, saved SFR and
elapsed physical time. Old network checkpoints are incompatible.

Registered focused checks include `gadget4_cooling`,
`sph_chemistry_timesteps_cpu`, `gadget4_cooling_integration_cpu` and, when the
global integrator is built, `gadget4_cooling_integration_global_cpu`. CPU/CUDA
SF comparisons use the shared production backend. Test registration alone is
not evidence that those checks have passed on a particular build.

The Linux `cpu_only_runtime` regression hides devices and preloads the
`no_cuda_runtime` guard to reject device/context, allocation, transfer and
launch calls. Run it against a CUDA-enabled build to check runtime CPU
selection. A two-locality `--run.test=sphGravity --chemistry.enabled=true
--chemistry.cuda=false --hydro.cuda=false --gravity.cuda=false --drift.cuda=false
--hpx:localities=2` invocation additionally exercises
parcel migration and remote EOS/velocity cache reads.

## Historical network validation

The following is retained evidence for the retired particle networks, not
validation of the GADGET-4 replacement. Its referenced tests and options may
require the historical checkout.

Validation recorded during implementation on 2026-10-18: the complete CPU-only default build passed, including the application, utilities and all registered test executables; the rates, solver and unit-boundary regressions passed; the SIMD SPH suite passed with composition dependent force, sound-speed and expansion checks. Linking that new SPH suite against the original hydrodynamic implementation failed its composition-dependent pressure assertion, confirming that the test detects the missing coupling. Two-locality CPU migration/cache regressions then passed with chemistry both enabled and disabled. Both atomic6 and molecular9 also passed two production iterations and restart from checkpoint 1 to checkpoint 2, with identical gas payload bytes on each rank and validated manifest checksums. These MPI runs used two HPX threads per rank, hid all CUDA devices, and preloaded the CUDA-call guard. They are bounded integration checks, not convergence studies. The complete CUDA-enabled default build also passed. Its executable passed all three `cpu_only_runtime` guarded startup/storage/gravity cases with device visibility empty; full production atomic6/molecular9 stepping and exact restart also passed in that guarded CPU mode. Actual RTX 3090 source/hydro execution and restart passed separately. The [validation record](primordial_validation.md) distinguishes all build, CPU, GPU and MPI evidence.
