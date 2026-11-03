# SPH and Hopkins MFM/MFV: what is shared and what could be

Scope: the working tree of `/home/dmarce1/workspace/cosmictiger` as of 2026-11-03. Several files have uncommitted edits (`hopkins.cpp`, `hopkins_state.hpp`, `sph_particles.{hpp,cpp}`, `domain.cpp`, `drift.cu`, `drift_cpu.cpp`), so line numbers refer to the working tree, not HEAD. I re-read the code wherever the inputs disagreed; those places are marked **Confirmed**. Opinions are marked *(design opinion)*.

## 1. Verdict

You can share the storage container, the particle bookkeeping and much of the leaf arithmetic today. Several parts are already one implementation:

- the gravity particle arrays and the gas reverse index;
- the float `SphThermodynamicState`, density and gravity-acceleration arrays;
- allocation, swap, remove and free;
- the migration record and the checkpoint wrapper;
- the `hydro_source::SourceTree` geometry tree;
- `kernelW`/`dkernelWDq`;
- the GADGET-4 cooling and SH03 star-formation batch backends, and `starFormationConvertGas`;
- one drift kernel per backend.

Your premise needs one correction. The only differences are *not* extra fields, precision, updating inactive particles, and mesh velocity versus fluid velocity. The code also differs in six other ways:

1. **Which copy is authoritative.** In global SPH the float arrays are the state. Hopkins keeps a double conserved anchor plus pending ledgers, and the float arrays are only mirrors (`hopkins.cpp:737-765, 787-803`). Hierarchical SPH sits in between, with a double `SourceState` built from the floats (`hierarchical.cpp:73-106`).
2. **The integration model.** SPH applies KDK half-kicks of rates (`sph_particles.cpp:172-243`). Hopkins closes conserved intervals per cell and keeps pair records that persist across events (`hopkins.cpp:241-273, 2803-2859`).
3. **Gather versus scatter.** SPH writes only the sink (`sph_hydro_cpu.cpp:251-267`). Hopkins deposits ±δ into both endpoints, inactive ones included (`hopkins.cpp:2658-2685`).
4. **Where hydro sits in the hierarchical ladder.** SPH runs once at the top of the ladder and opens per rung (`driver.cpp:1451-1454, 1542-1544`). MFM runs a Begin before the ladder and a Finish after it (`driver.cpp:1230, 1568`).
5. **Gravity coupling.** SPH defers the gas opening kick to hydro (`driver.cpp:1367`, `hierarchical.cpp:830-840`). MFM applies a double impulse inside all three gravity walkers (`kick_walk.hpp:571-586`, `kick_cuda_kernel.hpp:239-256`, `pm/kick.cpp:229-246`).
6. **Neighbour search.** There are three different searches (§2, neighbour rows).

Mesh velocity is a separate field already, and the shared drift consumes it. But today it is always set equal to the fluid velocity at the anchor (`hopkins.cpp:3899`). MFV is barred from hierarchical integration and from sources (`options.cpp:160-163, 173-176`).

## 2. Component table

Legend for **shared now?**: yes, partial or no. Rows marked **Contested** had conflicting inputs; the corrected reading is given.

### Storage

| Component | SPH today | Hopkins today | Shared now? | Shareable? | Blockers |
|---|---|---|---|---|---|
| Gravity particle arrays (pos, vel, rung, type, catIndex) | Accessed through `sphParticlesPosition/Velocity/Rung` → `particlesPos/Vel/Rung` (`sph_particles.hpp:164-174`). `particlesVel` is both the fluid and the drift velocity. | `particlesVel` is the float fluid mirror (`hopkins.cpp:763-765, 800-802`). The drift velocity is `meshVelocity`/`meshAcceleration` (`hopkins_state.hpp:62,75,103-105`). In hierarchical MFM, `particlesPos` of inactive gas holds a projection to `gravityEndTime` (`hopkins.cpp:748-754`). `particlesRung` is the gravity rung, and the hydro rung is `state.hydroRung` (`hopkins.cpp:620-623, 740-746`). | yes (storage) | needs a per-method semantics layer | **Contested** (original rating "already"). Pos, vel and rung *mean* different things per method and mode, so the storage is shared but the meaning is not. |
| Reverse index `sphParticlesParticleIndices` | Maintained incrementally by `particlesSwap`, but only under the hierarchical integrator (`particle_partition.cpp:70-80`). Otherwise rebuilt by scanning at the tree root (`distributed_tree.cpp:113-116`, `particles.cpp:1813-1832`). | Same map, same maintenance rules. Used for every pos/vel/rung access and for the SF state (`hopkins.cpp:600,628,770`). | yes | already | The repair rules depend on `run.timeIntegrator`, not on the method, and are spread over ~8 writers (`cell_tree.cpp:369-382`, `domain.cpp:714, 790-793`, `sph_particles.cpp:593,632,638-645`). |
| `SphThermodynamicState` {float u, float h} (`sph_particles.hpp:35-38`) | Authoritative u and h. | Read only while the state is uninitialized (`hopkins.cpp:588-594, 612-619`). Written at install: h always, u when phase≠0 (`hopkins.cpp:737, 762`), plus predicted u for non-installed initialized cells (`hopkins.cpp:800`). | yes (storage) | as storage and output seed only | **Contested** (original rating "already"). Ownership runs in opposite directions. Hopkins' thermal state is (m, p, E, s) in double (`hopkins_state.hpp:54-57`), and (u, h) cannot represent it. A precision template alone does not unify them. |
| `sphParticlesDensities` float | Authoritative in global SPH. In hierarchical SPH it mirrors the double `SourceState::density` (`hierarchical.cpp:390,737`). | Mirror written at install (`hopkins.cpp:738`) and for non-installed cells (`hopkins.cpp:799`). Hopkins never reads it for its own solve. | yes | storage only | There are three ownership models. Global SPH is the odd one out. The mirror carries load for output, scenarios and migration (`view.cpp:576`, `scenario_driver.cpp:55`, `domain.cpp:669,724`). |
| `sphParticlesGravityAccelerations` Array<float*,3> | Read by the force pass and by the hierarchical opening (`sph.cpp:598-600`, `hierarchical.cpp:649,836`). | Read only when `readForces` is set, otherwise taken from the owner-key mirror (`hopkins.cpp:683-687`). Ignored for the hierarchical hydro source (`hopkins.cpp:513-514`). Promoted to double `sourceAcceleration` (`hopkins.cpp:2787,2851`). | yes (storage) | write policy and read protocol differ | `refreshGasForces` writes forces for *all* gas only when method≠sph (`driver.cpp:740-742,913-914,969-970`). |
| `sphParticlesGravityTimesteps` float (transient) | Read as `sinkGravityTimestep` (`sph.cpp:602`, `hierarchical.cpp:640`). | The hydro timestep ignores it (no hits in `hopkins.cpp`). Read indirectly through `starFormationConvertGas` (`star_formation.cpp:259-260`, called from `hopkins.cpp:836`). The walker writes it for MFM gas (`kick_walk.hpp:544-556`). | partial | partial | Hierarchical MFM gets the same bound through the walker-owned gravity rung instead (`kick_walk.hpp:606-624`, `hopkins.cpp:620`). |
| `SourceState` (all double, `hierarchical_state.hpp:10-34`) | Hierarchical SPH only (`sph_particles.cpp:407-408`). | none | no | header/adapter only | It is derived from the floats, not authoritative. It carries option copies (`hierarchical_state.hpp:21-29`). Its trajectory is linear, not quadratic. Its id is assigned lazily (`hierarchical.cpp:76`). |
| `HopkinsParticleState` (424 B Fixed64, `hopkins_state.hpp:21-116`) | Freed for sph (`sph_particles.cpp:518-521`). | Authoritative. Mutated by hydro, by the gravity kicks (above) and by drift (`drift.cu:81-84`). | no | container plumbing only | Raw checkpoint dump, owner-key `memcmp` (`hopkins.cpp:538-551,565-578`), `offsetof` asserts and padding canonicalisation (`hopkins_state.hpp:118-136`). |
| `sphParticlesMasses` (float, MFV only) | Never allocated. `sphParticlesMass` returns `initial.gasMass` (`sph_particles.hpp:131-135`). | Projection of the double mass: `accountedMass` at install (`hopkins.cpp:757-759`) and `accountedMass+partial` before gravity (`hopkins.cpp:843-860`, `driver.cpp:860-866`). | accessor only | needs a per-method pre-gravity hook | **Contested** (original rating "already"). The MFM kick reads the double `state.mass` directly (`kick_walk.hpp:574-580`). The `"mfv"` string test is repeated at ≥7 sites. |
| SPH force arrays (accel, predicted accel, rates, grad-h, flags) | Allocated iff method==sph (`sph_particles.hpp:117-119`). | null | no | no | Hopkins keeps fluxes in pair records and ledgers instead. |
| Viscosity, C&D, Balsara arrays | Allocated by option (`sph_particles.cpp:415-421`). | not read | no | no (SPH physics) | Allocated and range-validated even for mfm when the option is set (`sph_particles.cpp:440-448`, `options.cpp:1598-1635`). |
| Chemistry `gadget4::State`, `Pair<float>` EOS, `StarFormationState` | Written in place by the commits (`sph_chemistry.cpp:254`, `star_formation.cpp:343-345`). | Copied into the workspace record and back at install (`hopkins.cpp:596-601,624-629,766-770`). The EOS array is never read; Hopkins hardcodes `kGammaMinusOne` (`hopkins.cpp:2943`). | yes (storage) | storage yes, glue partial | Hopkins sources run for MFM only (`hopkins.cpp:2897-2899`). |
| Migration record `GasMigrationState` (`domain.cpp:578-626`) | Fields gated by option. Hierarchical SPH extras at 620-624. | `HopkinsParticleState` when method≠sph (605-607), mass when mfv (602-604). Event state travels separately as `HopkinsSidecar`, broadcast to all peers (`hopkins.cpp:909-920, 3650-3665`). | yes | transport yes | The in-memory record always holds both anchors. The SPH force fields are serialised unconditionally even for Hopkins (592-601). |
| Checkpoint | One SPH block, `sphCheckpointVersion` (`sph_particles.cpp:1143-1157`). | Plus a `0x484f5053` v9 state block (`sph_particles.cpp:1206-1217,1255-1267`) and a pair cache `0x484f504341434845` v2, 176 B records (`hopkins.cpp:4478-4513`). | wrapper only | wrapper only | **Contested** (original rating "already"). There are three version authorities. +1000 is added for hierarchical runs even for mfm (`sph_particles.cpp:1156`). `docs/hopkins_hydro.md:217, 603-604` still say schema 6. |
| Allocation, teardown, remove | `sphParticlesResize/Remove/Free` (`sph_particles.cpp:406-670`). | Same functions. | yes | already (mechanics) | `sphParticlesFree` has no caller. `hopkinsOwnerKeys` is a sidecar that swap/remove do not maintain (`hopkins.cpp:553,777-782`). |

### Neighbour search

| Component | SPH today | Hopkins today | Shared now? | Shareable? | Blockers |
|---|---|---|---|---|---|
| Tree | Global: Fixed32Range leaf tree rebuilt every step. It permutes the gas arrays (`sph_tree.cpp:287-516, 359, 513`). Hierarchical: `Forest<SourceState,SphSourcePolicy>` over `SourceTree` (`hierarchical.cpp:44`, `source_forest.hpp:65`). | Bare `SourceTree` ×2, owned plus ghost (`hopkins.cpp:189,216`). | `SourceTree`: hierarchical SPH and Hopkins | tree yes; Forest layer no | `Forest::replaceActive` throws when it would overwrite an inactive source (`source_forest.hpp:108-110`). Hopkins refits inactive halo cells (`hopkins.cpp:1600,1630,1692`). |
| Candidate walk | Global: `sphTreeNeighbor` with stored leaf lists (`sph.cpp:173-334`, `sph_tree.cpp:97-116`). | Transient per-cell query (`hopkins.cpp:1349-1386`, comment 192-195). | no | partial | Different return types: TreeId ranges, SourceState copies, slot indices. |
| Exact pair test | float, r²<max(H_a²,H_b²) (`sph_hydro_cpu.cpp:108-113`, `sph_hydro.cu:198`). | double from a Fixed64 difference, `r<f·h_i ∨ r<h_j` (`hopkins.cpp:1309-1317, 360-370`). | no | predicate only | Precision, squared versus sqrt, the factor, and sign convention (r_a−r_b versus x_j−x_i). |
| Separation primitive | `distance`/`simdPositionDifference` (`fixed.hpp:337-340`, `sph_cpu.hpp:80-98`). | `doubleDistance` (`fixed.hpp:403-407`). | yes (primitive) | already | Only the gather wrappers differ. |
| Reduction order | Global: leaf order. Hierarchical: (locality, id). CUDA: depends on `blockDim`. | Neighbour lists sorted by identifier (`hopkins.cpp:1375-1382`). Deposits sorted by key (`hopkins.cpp:2657-2661`). | no | policy | Unifying changes bit patterns in one of the methods. |
| Multi-locality | Global: TreeId actions plus cache lines. Hierarchical: box broadcast to every locality per pass (`hierarchical.cpp:703-713`). | Ghost import from owned-leaf envelopes, with a radius retry (`hopkins.cpp:3454-3496, 1456-1462`). Deposits and sidecars are written back to owners. | serving-side `SourceTree` query only | read side partial | The Hopkins write-back side (deposits, halo marks, sidecars) has no SPH counterpart. |

### Kernels and support

| Component | SPH today | Hopkins today | Shared now? | Shareable? | Blockers |
|---|---|---|---|---|---|
| `kernelW`/`dkernelWDq` (`kernel.hpp:50-58`) | float / SimdFloat | double / HopkinsSimd, wrapped (`hopkins.cpp:451-477`). The 1D π/6 factor is valid only for spline 3. | yes | already | — |
| ΣW target 3N/4π | `sphTargetKernelSum` (`sph_kernel.hpp:123-125`), duplicated inline (`sph_prehydro_cpu.cpp:58`, `sph_cuda.hpp:138`). | `hopkinsNeighborTarget` (`hopkins.cpp:484-488`), adds a 1D branch. | no | yes | Hopkins reads `getOptions()` inside the helper, which rules out CUDA. Flop counting is done inside the helper. |
| Support solve | float Newton, tol 1e-4, 128 iterations (`sph_prehydro_cpu.cpp:34-120`), plus a CUDA copy (`sph_cuda.hpp:124-219`). | double, tol 1e-13, 256 iterations, `supportRadiusStep` (`hopkins.cpp:1420-1532`, `hopkins.hpp:422-457`). | no | step function only | The fallback direction differs: +0.2H against `copysign` (`sph_prehydro_cpu.cpp:103` vs `hopkins.hpp:445-446`). So does the next==H rule. The reach and ghost restart logic differs. Sharing would change results. |
| Density | m·ΣW/H³ with the self term included (`sph_prehydro_cpu.cpp:228-233`). | own mass × kernel number density; geometry plus moment matrix (`hopkins.cpp:1551-1595`). | no | no | MFV mass, 1D, retry on singular geometry. |
| grad-h / `sph_volume` | Ω, ζ (`volume.hpp:55-200`; the file is 223 lines). | Ḣ from its own sums (`hopkins.cpp:1866-1906`). | no | minimal | The bound semantics differ: relative 1e-4 against absolute 1e-13·target plus a sign test. |
| Central and external forces | `sphCompleteAcceleration` (`sph_kernel.hpp:86-117`), with explicit damping. | `hopkinsAcceleration` (`hopkins.cpp:504-532`), no damping (damping is an exp impulse elsewhere). | no | position force yes; damping no | The helper must be CUDA_EXPORT and must not call `getOptions`. |
| Flop accounting | `sph_flops::Counter` with single/double buckets. | `HopkinsFlopBatch`, all double (`hopkins.cpp:62-84`). | partial (per-kernel costs) | partial | The default bucket is opposite between them. Comparisons are counted in SPH but excluded in Hopkins. |

### Integrators (global and hierarchical)

| Component | SPH today | Hopkins today | Shared now? | Shareable? | Blockers |
|---|---|---|---|---|---|
| Driver dispatch | String compares on `hydro.method` (~80 sites in src/ and cosmictiger/). | Same. | skeleton only | interface | No HydroMethod type exists. |
| Global step | `sphStep1`+`sphStep2` (`driver.cpp:251-710`). The whole force pass runs inside the gravity overlap; `joinGravity` happens mid-step (481-505). | `hopkinsPrepareTree`/`Geometry` during the overlap, then `hopkinsKick` after the join (`driver.cpp:1024,1049,1109`). | no | interface (≈4 hooks) | The contract for what may run during the overlap differs. MFV has a pre-gravity mass hook. |
| Rung selection | `max(1,⌈log2⌉)`, coarsening previousRung−1, float (`sph_hydro_cpu.cpp:305-308`). | `max(minRung,⌈log2⌉)`, allowCoarsen only on the first selection, double (`hopkins.cpp:2264-2267`). | no | scalar helper | The floors differ (1 versus minRung). Hopkins wakes sleeping cells (`hopkins.cpp:2268-2273`). |
| Neighbour rung limiter | Pull, 1 rung, active-only (`sph_rungs_cpu.cpp:47-52`, `sph_kernel.hpp:119`). | Push, 2 rungs, wakes sleepers, identifier-order replay (`hopkins_wake_sweep.hpp:244-257`). | no | no | Gather against scatter, and different contrast. |
| Timestep criteria | Monaghan signal over `h_a`, u/|du/dt|, gravity bound (`sph_hydro_cpu.cpp:278-324`). Hierarchical adds a forest signal bound (`source_forest.hpp:174-224`). | Pairwise signal over R_cell, acceleration bound using `timestepEta` (`hopkins.cpp:2205-2244`). | no | expansion cap yes | `hydro.cfl` scales different lengths in the two methods. |
| Hierarchical hooks | `sphHierarchicalStep` at the top, `Open` per rung, `OpenGravity` (`driver.cpp:1451-1545`). | Begin before the ladder, Finish after it (`driver.cpp:1228-1233, 1566-1581`). | no | interface only | Ladder placement and rung ownership differ (`hopkins_hydro.hpp:30-31`). |
| Energy bookkeeping | Returns `HierarchicalSphResult` deltas (`hierarchical.cpp:936-947`). | Writes `EnergiesT&` in place (`hopkins.cpp:4225-4235, 4376-4391`). | `particlesSumEnergies` (branches internally, `particles.cpp:1906-1947`) | partial | tckin is updated incrementally (SPH) versus assigned outright (Hopkins) (`driver.cpp:1465` vs `hopkins.cpp:4381`). |
| Terminal closing pass | early exits (`driver.cpp:573-595`, `hierarchical.cpp:949-951`) | forces every cell active, settles all pairs (`hopkins.cpp:3171, 2741, 4363-4368`) | flag only | flag only | — |

### Drift

| Component | SPH today | Hopkins today | Shared now? | Shareable? | Blockers |
|---|---|---|---|---|---|
| Drift kernels (CPU, CUDA) | `vel[]·dt/a` (`drift_cpu.cpp:119-124`). | `meshState->driftVelocity(t_mid)` (`drift_cpu.cpp:111-124`, `drift.cu:75-113`). Hierarchical MFM skips the particle and stamps `gravityEndTime` (`drift_cpu.cpp:142-147,176-179`, `drift.cu:81-84`). | yes, one kernel per backend | already for global; no for hierarchical MFM | **Confirmed**: the skip is gated `method=="mfm"` only (`drift.cpp:57-58`). Hierarchical MFM positions are written by Hopkins `positionAt` (`hopkins.cpp:753`). The drift divides by `a`; `positionAt` divides by `positionDriftScale`. Under hierarchical MFM no gas lightcone entries are produced anywhere. |

### Gravity

| Component | SPH today | Hopkins today | Shared now? | Shareable? | Blockers |
|---|---|---|---|---|---|
| Gas mass to the tree | `initial.gasMass` (`particles.hpp:327-341`). | Same for MFM. MFV uses `sphParticlesMasses` (`particles.hpp:335-336`). | yes (read side) | yes, plus a per-method mass hook | MFV-only projection step (`hopkins.cpp:4138-4162`). |
| Kick, global path | Force stored, gas not kicked (`kick_walk.hpp:510-528`). | Same branch. | yes | already | `refreshGasForces` differs. |
| Kick, hierarchical path | Deferred top opening (`driver.cpp:1367`). Below the top level SPH gas *is* kicked in the walker by float fmaf (`kick_walk.hpp:589-600, 615-621`). | Double impulse on (m, p, E) in three backends, with the asserts `hierarchicalActive` and pending==0 (`kick_walk.hpp:571-586`). | no | per-particle core only | The MFM test differs per backend (`method=="mfm"` versus `hopkinsStates!=nullptr`, `kick_cuda_kernel.hpp:219`). |
| Gravity work | Implicit, from kinetic sums. | Explicit half-step work and MFV pair mass-flux work (`hopkins.hpp:190-272`, `hopkins.cpp:2680-2681, 2813-2823`). | no | no | — |

### Sources

| Component | SPH today | Hopkins today | Shared now? | Shareable? | Blockers |
|---|---|---|---|---|---|
| Cooling and SF batch backends | `sphChemistryAdvanceBatch`, `starFormationAdvanceBatch` | same calls (`hopkins.cpp:2949, 2976`) | yes | already | float-only inputs; one `gasMass` |
| Input gather and write-back | arrays, staged prepare/commit (`sph_chemistry.cpp:216-258`, `star_formation.cpp:306-349`) | E/m − kinetic → float; ρ = m/V; applies Δ to E plus entropy (`hopkins.cpp:2912-2984`) | no | adapter or policy | Thermal variable, transaction model, conversion timing. |
| Conversion to a star | `starFormationConvertGas` | same (`hopkins.cpp:836`), preceded by a ballistic re-anchor (`hopkins.cpp:816-835`) | yes | already | MFV mass would be lost (MFV is currently barred). |
| Interval to seconds | inline copy (`sph_chemistry.cpp:233-236`) plus helper (`sph_chemistry.cpp:146-156`) | helper (`hopkins.cpp:2973`) | partial | yes | The SF kernel works in seconds on the device (`star_formation_kernel.hpp:169`). |
| Pressure integral | `sphChemistryPressureIntegral` (`driver.cpp:596-607`) | same (`driver.cpp:1116-1128`) | yes | already | Global path only. |

### Options and wiring

| Component | SPH today | Hopkins today | Shared now? | Shareable? | Blockers |
|---|---|---|---|---|---|
| `hydro.*` options | ~11 SPH-only keys (viscosity, Balsara, `cuda`). | 3 meshless-only keys (dual-energy switches, `treeRebuildActiveFraction`). | mostly | already for kernel, bounds, gamma, cfl, N | The same `cfl` value means different things per method (`docs/hopkins_hydro.md:612-619`). |
| Build | `ENABLE_SPH`. | `ENABLE_HOPKINS` requires SPH and fixed64 (`cmake/features.cmake:33-39`). `hopkins.cpp` is compiled with `-fno-fast-math -ffp-contract=off` (`CMakeLists.txt:449-450`). | yes | already | FP contract for any shared translation unit. |
| CUDA | SPH hydro kernels. | Hydro is CPU-only (`options.cpp:110-113`). Device code touches Hopkins state in the kick and the drift. | no | — | — |

### Tests

| Component | SPH today | Hopkins today | Shared now? | Shareable? | Blockers |
|---|---|---|---|---|---|
| Checkpoint decoders | `checkpoint_format.decode_checkpoint` (194-240) | `run_hopkins.particle_rank_state` (27-219), which also decodes sph | partial (`read_prefix`) | yes | All-gas only, no groups or tracers. |
| Hierarchical drivers | `run_hierarchical_sph.py` | `run_hierarchical_mfm.py` | `run_hierarchical_sources.py` runs both (153, 176) | yes (adapter pattern) | Restart strictness differs: 2e-6 against exact. |
| Kernel unit tests | `tests/sph/cpu.cpp`, `cuda.cu` | `tests/hopkins.cpp` | no | leaf helpers | Precision and FP flags. |
| CTest matrix | backend loop | solver × integrator × backend (`CMakeLists.txt:2244-2280`) | partial (`hydro_source_tree`, `hydro_methods`) | partial | — |

## 3. Parallels (same job, different name)

| SPH | Hopkins |
|---|---|
| `SourceState.id = (rank<<48)+gas+1`, assigned lazily, 0 = unset (`hierarchical.cpp:76`) | `identifier = (rank<<48)\|counter`, assigned eagerly, `UINT64_MAX` = unset (`hopkins.cpp:869-897`) |
| `SourceState.position` double[3] | `positionAnchor` Vector3<Position> |
| `driftVelocity`, `driftScale` | `meshVelocity`, `positionDriftScale` |
| `hydroAcceleration` (signal only) | `meshAcceleration` (float, trajectory) |
| `epoch`, `validUntil` (snapped to the tick grid, `hierarchical.cpp:792-795`) | `sourceTime`, `endTime`, `endTick` (`hopkins.cpp:3873-3880`) |
| `rung` | `hydroRung` |
| `predictSource` (`hierarchical_sources.hpp:148-180`) | `hopkinsPredictedPrimitive` + `positionAt` (`hopkins.cpp:1974-1998`) |
| `liveSource` / `PackedRun` | `hopkinsOwnerRecord` / `hopkinsCollectDelta` (`hopkins.cpp:605-700`) |
| `sphHierarchicalResetSources` | `hopkinsResetSources` (both called from `sphParticlesFree`, `sph_particles.cpp:648-653`) |
| `Forest::replaceActive` | `hopkinsOpenSources` refit or reset (`hopkins.cpp:1766-1783`) |
| Boxes walk outer box (1+0.21)·h | `supportUpper` refit after the solve |
| `sphParticlesApplyUpdates` phases 0/1/2 | `hopkinsCloseParticle` + Schedule + Gradients |
| `limitSignalsLocal` + `kSphRunRungs` | `hopkinsTimestepPasses` + `WakeSweep` |
| `sphHierarchicalOpenGravity` (float v += g·dt/2) | `applyGravityImpulse` inside the walkers |
| `sphHierarchicalThermalEnergy` | `hopkinsHierarchicalDriftThermalEnergy` (`driver.cpp:2286-2298`) |
| `SphRunReturn` / `HierarchicalSphResult` | `HopkinsStepResult` + `EnergiesT&` |
| Remote cache lines `sphParticlesGlobalRead*` | `HopkinsGhostPayload` ghost exchange |
| Hierarchical migration of the completed-rung suffix (`domain.cpp:760-885`) | `HopkinsSidecar` for event state that migrates (both methods use the suffix path) |
| ΣW target (`sph_kernel.hpp:123`) | `hopkinsNeighborTarget` |
| `classifySupport` held bounds | `lowerHeld`/`upperHeld` with Ḣ=0 (`hopkins.cpp:1896-1906`) |
| `SphRunData.sourceAccelerationX` = neighbour hydro dv/dt | `state.sourceAcceleration` = gravity at `sourceTime` (**same name, different meaning**) |

## 4. The four integrator paths

| | SPH global/traditional | SPH hierarchical | Hopkins global/traditional | Hopkins hierarchical (MFM only) |
|---|---|---|---|---|
| Position drift | Shared kernel, `particlesVel·dt/a`, all particles (`driver.cpp:2337-2341`). | Shared kernel per exact rung after the hydro open (`driver.cpp:1542-1549`). | Shared kernel, `meshVelocity + (t−t0)·meshAcceleration` (`drift_cpu.cpp:111-124`). | The kernel skips the particle and stamps `gravityEndTime`. Hopkins writes `positionAt(gravityEndTime)` at install phase 1 (`hopkins.cpp:748-754`). |
| Inactive particles | Untouched. Drifted with their staggered velocity. No fresh force. | Predicted linearly from `SourceState`. One-sided updates on active sinks. | Predicted from U0, gradients and sourceAcceleration. Mirrors refreshed (`hopkins.cpp:787-803`). Receive pending deposits. Woken when the prediction is invalid. Forces refreshed for all gas. | Same as global, plus the gravity impulse requires a closed anchor (Begin closes gravity-touched cells, `hopkins.cpp:3675-3683`). |
| Neighbour search | `sphTreeCreate` + `sphTreeNeighbor`, stored lists, float. | Forest batch query broadcast to every locality, float kernels through `sphRunStaged`. | `SourceTree` ×2, per-cell point queries, double. | Same as global; sleeping cells become trajectory sources (`hopkins.cpp:1253-1266`). |
| Rung | One rung shared with gravity. `max(1,⌈log2⌉)`, prev−1, then the neighbour limiter loop. | Same kernel, plus the forest signal bound; writes `sphParticlesRung`. | Global: everything on rung 0. Traditional: `required`, allowCoarsen, then the Appendix G sweep. Written to `particlesRung` (`hopkins.cpp:740-741`). | `state.hydroRung` is separate from the gravity rung (`hopkins.cpp:746`). Extra suffix levels are pushed (`driver.cpp:1570-1580`). |
| Gravity on gas | Force folded into dv/dt. | Top opening deferred to hydro; float kick below the top level. | `sourceAcceleration` at closure, two half-steps (`hopkins.hpp:229-275`). | Double impulse in the walker. |

**What a unified path would need** *(design opinion)*:

1. One position owner per mode, and one trajectory form. Either both methods use a linear drift-segment anchor, or both use the quadratic mesh anchor. The drift scale should be `positionDriftScale` everywhere, not `a` in some places.
2. A per-method "inactive update" hook. For SPH it is a no-op. For Hopkins it covers mirror refresh, deposits and wake.
3. One search layer: `SourceTree` with a method policy (predict, exact test, padding). Ghost import goes on the read side. Hopkins keeps its deposit write-back.
4. One rung model: either two clocks (gravity and hydro) for both methods, or one clock for both.
5. A gravity "sink" interface: receive g; apply it to a velocity or to conserved state; report the kinetic change. It must cover four backends (CPU walk, CUDA, PM, PMMM) and be device-callable.
6. A HydroMethod interface in the driver with prepare, overlap(join), finish, hierarchicalBegin/Step/Finish, driftThermal and refreshGasForces.

## 5. Persistent state layout (struct-of-arrays)

Decisions taken as given: every persistent per-particle field is struct-of-arrays, with one array per scalar or per component. `HopkinsParticleState` is split into `MeshlessFiniteMassParticleState` and `MeshlessFiniteVolumeParticleState`.

### 5a. Field table

Categories:

- **CG** = common gas: allocated for every hydro method.
- **SPH** = SPH-only.
- **ML** = meshless-common: MFM and MFV.
- **MFV** = MFV-only.
- **MFM** = MFM-only. This adds a fifth category to the four requested, used for `entropy` only.

Sizes are per particle.

| Field | Bytes | Used by | Persistent? | Proposed array | Element type | Evidence and notes |
|---|---|---|---|---|---|---|
| internalEnergy (a²u) | 4 | sph authoritative; mirror for ML | P | CG | float | `sph_particles.hpp:36`; Hopkins writes it at `hopkins.cpp:762,800` |
| supportRadius (float) | 4 | sph; mirror for ML | P | CG | float | `hopkins.cpp:737` |
| density | 4 | sph; mirror for ML | P | CG | float | `sph_particles.hpp:61` |
| gravityAcceleration x,y,z | 12 | both | P | CG (already SoA) | float×3 | `sph_particles.hpp:53` |
| gravityTimestep | 4 | sph; read by SF for both | T | CG | float | not checkpointed or migrated (`sph_particles.cpp:359`) |
| particleIndex | 4/8 | both | P | CG (already SoA) | ParticleInteger | `sph_particles.hpp:50` |
| chemistry electronFraction | 4 | both, with chemistry | P | CG | float | `gadget4_cooling.hpp:34-43` |
| eos gammaMinusOne, gamma | 8 | sph reads; ML never reads | P | CG, or drop | float×2 | set to a constant when chemistry is on (`sph_particles.cpp:1087-1090`); Hopkins hardcodes `kGammaMinusOne` (`hopkins.cpp:2943`) |
| mfvMass projection | 4 | MFV | P | MFV | float | `sph_particles.hpp:62-63` |
| acceleration x,y,z | 12 | sph | P (staggered rungs) | SPH | float×3 | `sph_particles.hpp:40-42`; comment at `sph_particles.cpp:310` |
| predictedAcceleration x,y,z | 12 | sph | P | SPH (already SoA) | float×3 | — |
| predictedInternalEnergyRate, internalEnergyRate | 8 | sph | P | SPH | float×2 | — |
| convergence, semiactive flags | 2 | sph | T (but checkpointed) | SPH | char×2 | reset at `sph_particles.cpp:1483-1530` |
| gradHCorrection | 4 | sph | T traditional; P hierarchical | SPH | float | `sph_particles.hpp:67-71` |
| balsaraFactor | 4 | sph, by option | T traditional; P hierarchical | SPH | float | — |
| bulkViscosityCoefficient (alpha) | 4 | sph | P | SPH | float | `sph_particles.hpp` |
| C&D divergence | 4 | sph | P | SPH | float | `cullen_and_dehnen/state.hpp:13-18` |
| C&D pendingDivergence, targetAlpha, inverseDecayTime | 12 | sph | T | SPH, scratch | float×3 | — |
| SourceState fields (§5b) | 248 | sph hierarchical | P | SPH (hierarchical) | double / uint64 / int | `hierarchical_state.hpp:10-34` |
| identifier | 8 | ML | P | ML | uint64 | `hopkins_state.hpp:53` |
| mass | 8 | ML (fixed in MFM) | P | ML | double | still read in MFM (`hopkins.cpp:2867, 2926`) |
| totalEnergy | 8 | ML | P | ML | double | — |
| momentum x,y,z | 24 | ML | P | ML | double×3 | — |
| pendingEnergy, pendingMomentum x,y,z | 32 | ML | P | ML | double×4 | nonzero in MFM (pressure flux); accumulated on inactive cells |
| anchorVolume | 8 | ML | P | ML | double | — |
| meshVelocity x,y,z | 24 | ML | P | ML | double×3 | set only in the Gradients phase (`hopkins.cpp:3899`) |
| meshAcceleration x,y,z | 12 | ML | P | ML | float×3 | `hopkins.cpp:3911-3914` |
| sourceAcceleration x,y,z | 24 | ML | P | ML | double×3 | — |
| sourceTime, endTime, expansionCoefficient | 24 | ML | P | ML | double×3 | expansionCoefficient could be recomputed from sourceTime *(not verified)* |
| endTick | 8 | ML | P | ML | uint64 | — |
| gradients (ρ, P, v_x, v_y, v_z) × 3 | 120 | ML | P | ML | double×15 | Time prediction of an inactive cell uses only ∇·v and ∇P (`hopkins.hpp:691-711`). The full 15 slopes are needed for the spatial face reconstruction and the half-step Hancock predictor of an inactive endpoint in an active–inactive pair (`hopkins.cpp:2019-2025, 2484-2540`). |
| supportRadius (double) | 8 | ML | P | ML | double | the float CG copy is a rounded mirror (`hopkins.cpp:1808` comment) |
| positionAnchor x,y,z | 24 | ML, written only when hierarchical | P | ML | Position×3 | `hopkins.cpp:2778, 2857` |
| positionDriftScale, gravityEndTime | 16 | ML, hierarchical | P | ML | double×2 | gated by integrator, not method |
| hydroRung | 4 | ML, hierarchical | P | ML | int (char would do) | `hopkins.cpp:622,746` |
| initialized, predictorInitialized | 2 | ML | P | ML | bool/char×2 | — |
| hierarchicalActive | 1 | ML, hierarchical | per-ladder | ML | bool/char | see §5d |
| entropy | 8 | MFM consumers only | P | **MFM** (or ML; see open question) | double | written in both methods (`hopkins.cpp:2797, 2843`). **Confirmed**: EntropyViews runs for MFV unconditionally (`hopkins.cpp:3830-3831, 4315`), so moving entropy to MFM requires gating that phase. |
| pendingMass | 8 | MFV | P | MFV | double | **Confirmed**: identically zero in MFM, because `finiteMass = method=="mfm"` (`hopkins.cpp:2581-2582`) leaves `flux.mass = 0` (`hopkins_simd.hpp:441-447, 521-523`). The MFM kick asserts it is 0 (`kick_walk.hpp:573-576`). |
| pendingGravityIntegral x,y,z, pendingGravityWork x,y,z | 48 | MFV | P | MFV | double×6 | zero in MFM (`hopkins.hpp:195, 200-204`) |

Totals derived from the rows: ML ≈ 359 B per cell. MFV adds 56 B plus the 4 B float mass. MFM adds 8 B (entropy). Today every meshless cell pays 424 B.

### 5b. Decomposing the existing SPH AoS structs

- **`SphThermodynamicState`** becomes `internalEnergy[]` and `supportRadius[]`, both CG floats. Sites that copy the whole struct need gather and scatter: `sph_kernel.hpp:59` `sinkThermodynamicStates`, the prehydro writers (`sph_prehydro.cu:100,221`, `sph_prehydro_cpu.cpp:184,213`), the rungs readers (`sph_rungs.cu:105`, `sph_rungs_cpu.cpp:26`), `hierarchical.cpp:528,641,751`, the cache line `sph_particles.cpp:793-803`, and the checkpoint `sizeof` (`sph_particles.cpp:1170,1226`).
- **`SphAcceleration`** becomes `acceleration[3]` float arrays, the same shape as `sphParticlesPredictedAccelerations`. Writers: `sph_hydro.cu:302`, `sph_hydro_cpu.cpp:252`, `sph_particles.cpp:182-236,289-292`, `cullen_and_dehnen/particle_exchange.cpp:34`, `hierarchical.cpp:529,642,752`.
- **`sph_hierarchical::SourceState`** is decomposed into these groups:
  - id (uint64)
  - position[3]
  - driftVelocity[3]
  - synchronizedVelocity[3]
  - hydroAcceleration[3]
  - internalEnergy and internalEnergyRate
  - density and logDensityRate
  - physicalDivergence
  - supportRadius
  - gradHCorrection and gradHCorrectionRate
  - balsaraFactor
  - viscosityAlpha
  - rung
  - epoch, validUntil, driftScale

  The fields `minimumSupportRadius`, `maximumSupportRadius`, `pressureEnergyRatio` and `adiabaticIndex` copy option values. `liveSource` refreshes them every time (`hierarchical.cpp:79-80,96-97`), so they should be dropped from storage *(design opinion)*.

  Seven SourceState groups map one to one onto meshless fields: id↔identifier, position↔positionAnchor, driftVelocity↔meshVelocity, epoch↔sourceTime, validUntil↔endTime, driftScale↔positionDriftScale, rung↔hydroRung. A shared "trajectory anchor" array group is possible, but two things must be settled first: position type (double versus Fixed64) and kinematic order (linear versus quadratic) *(design opinion; see §7)*.
- **Viscosity alpha** is already the scalar `sphParticlesBulkViscosityCoefficients` array; Morris-Monaghan uses backward Euler without a stored rate or predictor.
- **`SphCullenDehnenState`** becomes divergence[] (persistent) plus pendingDivergence[], targetAlpha[] and inverseDecayTime[] (scratch). Migration and checkpoint already carry only the divergence (`domain.cpp:611-613`, `sph_particles.cpp:1188-1192`).
- **Chemistry**: `gadget4::State` becomes electronFraction[]. `Pair<float>` EOS becomes two arrays, or is dropped (see the table). The audit flags "EOS with stars but no cooling is migrated but not checkpointed", but that case cannot occur: starFormation requires chemistry (**Confirmed** `options.cpp:246-247`).

### 5c. What `MeshlessFiniteMassParticleState` and `MeshlessFiniteVolumeParticleState` would mean

*(design opinion)* Both should be **views**, not storage. Each is a device-copyable struct of base pointers plus an index, or an index into a `MeshlessArrays` struct of pointers. Each exposes:

- **Both views**: all ML fields from §5a, plus the CUDA_EXPORT methods `positionAt`, `driftVelocity`, `accountedEnergy` and `accountedMomentum` (today `hopkins_state.hpp:81-105`), rewritten to read the arrays.
- **MFM view**: adds `entropy`. `accountedMass()` returns `mass`; the pending mass is identically zero.
- **MFV view**: adds `pendingMass`, `pendingGravityIntegral`, `pendingGravityWork` and the float `mfvMass` projection. `accountedMass()` returns `mass + pendingMass`.

Keep the *workspace record* (`HopkinsParticle::state`, `hopkins.cpp:86-110`) as an AoS **value** struct. It is already a transient, HPX-serialised coordinator copy. Gather into it from the arrays in `hopkinsOwnerRecord` (`hopkins.cpp:605-641`), and scatter back in `hopkinsInstallSparse` (`hopkins.cpp:733-776`). That leaves most of the ~4.5k lines of `hopkins.cpp` unchanged.

The method branches inside the engine are runtime strings today (`hopkins.cpp:610,757,2581,2829,2897,4139,4204,4439`). A view type per method therefore needs either a template on the view, or two value-record types under a runtime variant. The gravity kick only ever needs the MFM view (it asserts that pending ledgers are empty), so `CudaKickData` would carry MFM pointers only.

### 5d. Hopkins fields that are really per-event scratch

The real per-event scratch, meaning geometry, primitives, rates and the solved/built/touched flags, already lives in `HopkinsWorkspace` (`hopkins.cpp:173-237`) and is not in persistent storage. In the persistent struct:

- **`hierarchicalActive`** lives for one ladder. It is set at BeginMark (`hopkins.cpp:3770`), cleared at install phase 1 (`hopkins.cpp:748`), and read by the kick asserts and `hopkinsMarkActive` (`hopkins.cpp:3171`). Checkpointing is refused while a ladder is pending (`hopkins.cpp:4479-4481`), so it need not be checkpointed. It must still survive migration during the ladder.
- **`gravityEndTime`** is written by drift during the ladder and read at phase 1, by star conversion and by `futureProjections` (`hopkins.cpp:753, 822, 3231`). Whether it is ever read before being rewritten after a checkpoint was not verified (§7).
- **`expansionCoefficient`** can probably be derived from `sourceTime` *(not verified)*.
- Everything else is genuinely persistent:
  - `pendingEnergy`/`pendingMomentum` accumulate on inactive cells between their own steps.
  - `gradients` and `anchorVolume` are needed to predict sleeping cells.
  - `meshVelocity`/`meshAcceleration` are needed by drift until the next event.

In MFM, `pendingMass` and the two gravity-work vectors are dead storage (56 B).

### 5e. Risks and every layout-dependent site

1. **Checkpoint.** Moving to SoA means a per-field write/read.
   - Gas header `sizeof` words: `sph_particles.cpp:1170-1171, 1225-1226`.
   - Hopkins block `{0x484f5053, 9, sizeof}` (`sph_particles.cpp:1206-1217, 1255-1267`) must become version 10 with a per-field layout. The pair cache v2 (`hopkins.cpp:4478-4513`) is unchanged unless `HopkinsPairRecord` changes.
   - `SourceState` is currently written raw, padding included, with no `sizeof` check (`sph_particles.cpp:351-353`).
   - `sphCheckpointVersion` (`sph_particles.cpp:1143-1157`).
   - Decoders: `tests/checkpoint_format.py:43-55, 218-253` and `tests/run_hopkins.py:23,67-80,136-158` hard-code the layouts.
   - `docs/hopkins_hydro.md:217, 603-604` is stale.
2. **Padding, asserts, owner key.** `hopkinsCanonicalizePadding` and the `offsetof` asserts (`hopkins_state.hpp:118-136`) disappear. `HopkinsOwnerKey` stores a byte image and compares with `memcmp` (`hopkins.cpp:538-551, 565-578`). Replace it with a packed shadow of the value record, or with a field-wise comparison. `tests/sph/gravity.cpp:470-484` does a byte-level memcpy of the state.
3. **Migration.** `GasMigrationState` and pack/unpack (`domain.cpp:578-626, 652-759`) need per-field gather/scatter, or the value record. Today the in-memory record always carries both the 424 B Hopkins anchor and the 248 B SPH anchor. Only serialisation is gated (`domain.cpp:602-624`).
4. **Permutation completeness.**
   - `sphParticlesSwap` (`sph_particles.cpp:358-368`) visits arrays through `visitSphCheckpointArrays` (`sph_particles.cpp:306-354`), plus three explicit swaps (GravityTimesteps, Balsara, GradH).
   - With ~70 scalar arrays, the risk of leaving one out goes up. `sphCheckpointTest` (`sph_particles.cpp:1274-1479`) only checks visited arrays.
   - `sphParticlesRemove` (`sph_particles.cpp:637-645`), `particle_mesh_sort.cpp:231-249`, and migration compaction (`domain.cpp:862-873, 945-965`) are all affected.
   - Recommendation *(design opinion)*: one registry of every array, holding pointer, element size, method gate and persistence flag, driving swap, resize, free, checkpoint and migration.
5. **Drift.** Change the `HopkinsParticleState*` parameter to an array view at these sites: `drift.cu:38,70-113,175-177,258-264`, `drift_cpu.cpp:24-40,80-82,95-124,142-147,176-179`, `drift_cpu.hpp:37`, `drift.cpp:63-69`. `positionAt`/`driftVelocity` must stay device-callable.
6. **Gravity kick.** Add per-field device pointers: `kick.hpp:32,45`, `kick_cuda_launch.hpp:522-524`, `kick_cuda_kernel.hpp:219-256`, `kick_walk.hpp:571-586`, `pm/kick.cpp:229-246`.
7. **Managed memory.** Every gas array comes from `allocateManagedMemory` (`sph_particles.cpp:395-404`; `cuda.cpp:49-58,110-112`), with no advise or prefetch. SoA helps here: today the GPU kick migrates whole 424 B records, about 9 per 4 KiB page. Per-field arrays would migrate only the touched fields *(design opinion; no profiling read)*.
8. **SPH kernels and staging.** `SphRunData` struct pointers (`sph_kernel.hpp:55-60`), `sph.cpp:588-616`, `staged_run.cpp:185-190`, and the hierarchical `PackedRun` (`hierarchical.cpp:528-531,579-586,641-647,751-760`).
9. **Other readers.** `view.cpp:572-573,580,584`, `particles.cpp:1908-1917`, `sound_wave.cpp:38-46`, and the cache lines at `sph_particles.cpp:49-76,788-803,1006-1072`.
10. **Tests.** `tests/hopkins.cpp:2127-2185`, `tests/drift_dispatch.cpp:32,63`, `tests/drift_cpu.cpp:247`, `tests/sph/cpu.cpp:135-136,231,239`, `tests/sph/cuda.cu:148-149,248,256,374-389`, `tests/particle_mesh_sort.cpp:127-139,200-206`, `tests/rayleigh_taylor.cpp:33-44`, `tests/units_output.cpp:174-379`.

## 6. Recommended order of work (smallest first)

1. **Small duplicate helpers.** Use `sphTargetKernelSum` in `sph_prehydro_cpu.cpp:58` and `sph_cuda.hpp:138`, with a dimension argument for `hopkins.cpp:484-488`. Use `sphChemistryPhysicalInterval` in `sph_chemistry.cpp:233-236`. Add one expansion-cap helper (`driver.cpp:306-323,756-775,895-901,961-966`; `hierarchical.cpp:881-885`; `hopkins.cpp:2183-2190`). Unblocks: less drift between copies.
2. **Fix the docs and comments.** `docs/hopkins_hydro.md:217,603-604` (version 9). The 16 INI files that say "hierarchical (gravity only)" (e.g. `configurations/blast.ini:12`). Unblocks: an accurate baseline.
3. **Method enum or traits.** Replace the ~80 `hydro.method` string compares (options.hpp/cpp, driver.cpp, kick_walk.hpp, kick.cu, gpu_staging.cpp, drift*, domain.cpp, sph_particles.*, hopkins.cpp). Unblocks: compile-time view types, removes string compares from hot paths (`particles.hpp:335`).
4. **A gas array registry.** One table that drives swap, resize, free, checkpoint and migration (`sph_particles.cpp:306-370,406-670`, `domain.cpp:578-759`). Unblocks: SoA without permutation omissions.
5. **SPH small structs to SoA.** `SphAcceleration` and `SphThermodynamicState`, plus the persistent/scratch split for viscosity and C&D. Bump the gas checkpoint version and update `tests/checkpoint_format.py`. Touches: §5e items 1, 4, 8 and 10. Unblocks: proves the registry and the checkpoint path.
6. **`HopkinsParticleState` storage to SoA,** behind a value record with gather/scatter at `hopkins.cpp:605-641,733-776`. Rewrite the owner key. Drift and kick get array views. Hopkins checkpoint goes to v10. Migration moves to per-field fields. Unblocks: precision templating per array, cheaper GPU paging.
7. **MFM/MFV split.** Move pendingMass and the gravity-work vectors into MFV arrays. Make the kick ABI MFM-only. Either gate EntropyViews for MFV (`hopkins.cpp:4315`) and move entropy to MFM, or keep entropy in ML. Branch hopkinsApplyDeposits and hopkinsCloseParticle (`hopkins.cpp:2671-2681, 2813-2850`) per view. Unblocks: 56 B per MFM cell.
8. **One gas identifier and trajectory anchor.** Assign the identifier at creation for every method. The SPH lazy id at `hierarchical.cpp:76` currently depends on slot. Decide whether SPH's `SourceState` trajectory groups share arrays with the ML anchor. Unblocks: a common Forest payload and id-keyed search for all four paths.
9. **Search unification.** Either give Hopkins a Forest policy, after a Forest refactor that allows inactive refits and thread-safe prediction (`source_forest.hpp:55-78,108-130`), or move hierarchical SPH onto bare `SourceTree`. Then retire `sphTreeNeighbor` for global SPH, keeping its particle-sort side effect. Unblocks: one neighbour layer.
10. **Driver HydroMethod interface** plus a gas-gravity sink policy across the four kick backends. Unblocks: one integrator skeleton.
11. **Test harness.** One checkpoint decoder, one process launcher, and a shared hierarchical driver built on the `run_hierarchical_sources.py` adapter pattern.

## 7. Open questions for you

1. **Entropy home.** Should `entropy` be MFM-only, which needs EntropyViews skipped for MFV (`hopkins.cpp:3830-3831, 4315`)? MFV keeps a strict closure per `docs/hopkins_hydro.md:215-216`.
2. **Hierarchical fields for MFV.** The hierarchical-only meshless fields (positionAnchor, positionDriftScale, gravityEndTime, hydroRung, hierarchicalActive) are gated by integrator. MFV is barred from hierarchical integration (`options.cpp:160-163`). Should they be allocated only for hierarchical MFM, or stay ML in case MFV gains the hierarchical integrator?
3. **Trajectory anchor.** Should SPH hierarchical and meshless share one trajectory-anchor array group? That needs a choice of position type (double versus Fixed64) and form: linear (`hierarchical_sources.hpp:155-157`) or quadratic (`hopkins_state.hpp:81-87`).
4. **Global SPH anchors.** Should global SPH gain persistent ids and anchors? Today it has neither (`sph_particles.cpp:407-408`).
5. **Rung model.** Two clocks (gravity and hydro) for both methods, or one? SPH writes the hydro rung into `particlesRung`; hierarchical MFM keeps `hydroRung` (`hopkins.cpp:740-746`).
6. **Mesh velocity.** It is always set to the fluid velocity (`hopkins.cpp:3899`). Is a distinct mesh velocity planned for MFV? That affects whether `meshVelocity` stays in ML.
7. **Authority of float arrays.** Should the CG float arrays (u, h, ρ) become non-authoritative mirrors for SPH too, matching hierarchical SPH and Hopkins? Or should SPH stay float-authoritative, with a precision template applied per array?
8. **`gravityEndTime` after checkpoint.** Is it read before being rewritten after a checkpoint, between ladders? If not, it can drop out of the checkpoint along with `hierarchicalActive`.
9. **EOS array.** Should the chemistry EOS array be dropped? Hopkins never reads it, and it is constant when chemistry is on (`sph_particles.cpp:1087-1090`).
10. **Reduction order.** Which order should a unified search keep: SPH (locality, id) or Hopkins' global id order? Which baselines pin the current bit patterns?
11. **MFV star mass.** If MFV ever gets sources, a converted star should carry its variable mass. Today `particlesMass` falls back to `initial.gasMass` (`particles.hpp:332-339`).
