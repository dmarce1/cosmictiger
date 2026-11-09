# SPH on the PMMM mesh: literature, core distinction, recommended design

## 1. What the literature does

All rows are verified from source or paper text unless marked (secondary).

| Code | Hydro spatial structure | Relation to PM mesh | hₘₐₓ rule | Rebuild / drift trigger | Ghost timing |
|---|---|---|---|---|---|
| SWIFT (Schaller+2024 §2.1, §9.1; `space_regrid.c`, `cell.h`) | Top-level grid, one octree per top cell, split at 400 particles | None: grid sized from global hₘₐₓ, "not related" to the gravity mesh (§9.1 footnote) | `SPH:h_max` = FLT_MAX; cell width ≥ 1.1·γ·hₘₐₓ instead | `γ·max(hₘₐₓᵢ,hₘₐₓⱼ) + dxᵢ + dxⱼ > dmin` forces a rebuild; membership frozen between rebuilds, no check on drift (`cell_drift.c`) | Whole top cells as proxies; owner re-sends `xv` after its drift, `rho` before the force loop, every active step (`engine_maketasks.c`) |
| GADGET-2 (Springel 2005 §3.1.2, §5.2) | Shared BH octree, per-node max h | Mesh is a pure Fourier device; separate decompositions | Minimum only (`MinGasHsmlFractional`) | Rebuild after 0.05 N force computations | Sink export per walk, two hypercube rounds per pass |
| GADGET-4 (Springel+2021 §5.7.1; `ngbtree.h`, `run.cc`) | Separate SPH neighbour tree, nodes carry position/velocity extrema, MaxHsml, MaxDtHsml, drifted lazily | Decoupled | No cap at all | Domain decomposition + tree rebuild only when active fraction > `ActivePartFracForNewDomainDecomp` (0.01); rank membership frozen between | On-demand fetch of foreign nodes/points during the walk |
| GIZMO (`forcetree_update.cc`, `density.cc`) | GADGET-3 tree, nodes kicked/drifted in place, hmax grown by div v | Decoupled | `MaxHsml` exists; FIRE sets 1e6; particles at the cap keep too few neighbours | `TreeDomainUpdateFrequency` 0.0005–0.05 | Sink export |
| PKDGRAV3 / Gasoline / ChaNGa (Potter+2017 §3.1; pkdgrav3 docs; `pkd.c` bndBall) | Local kd-trees per ORB domain, ball bounds from h·(1+0.1) | Decoupled | `dBallSizeLimit` default 1e30 | Membership moves only at base-step apex; `bTreeRepair` refits bounds between; ChaNGa re-bins on old splitters each substep | MDL software cache, pulled on demand |
| PHANTOM (Price+2018 §2.1.8; `neigh_kdtree.F90`) | One kd-tree rebuilt every derivative call, node hmax never lowered | n/a | None | Rebuild every step; `revtree` disabled for periodic boxes | Export active sink cells to leaf owners, partial sums returned |
| FDPS PM3 + list reuse (Iwasawa+2016 §3; Iwasawa+2020 §4–6; `tree_for_force.hpp`) | One global octree with PM cells pinned to a tree level, integer-index LET regions with `icut` | PM cell = tree cell | None; two-round exchange for symmetric cutoffs | Re-bin and `exchangeParticle` every step; reuse mode freezes lists ~10 steps with no built-in margin | Per step, or reused lists with refreshed data |
| Hydra (Couchman+1995 §2.1, §8) | Chaining mesh + refinements | h may not exceed the short-range search | Cap at base mesh only; otherwise escalate to coarser level | Per step | n/a |
| GOTPM (Dubinski+2004 §2) | One BH tree per cube of side = cutoff = 4 PM cells, 27-cube stencil | Cube ≫ PM cell | n/a | Trees rebuilt each step; slab ownership frozen 10 steps | Buffer of width Rₘₐₓ |
| PMMM (Nitadori 2014 §5.4) | "adaptive octree started from each cell" | Cell forest is the author's own proposal | n/a | Re-bin per evaluation | n/a |
| P3T / PeTar / GPLUM (Oshino+2011 §3; Wang+2020 eq. 24; Ishigaki+2021 eq. 22) | Neighbour lists frozen over the soft step | n/a | n/a | List radius = r_cut + 3σΔt (or 3|vᵢ|Δt, pairwise max) | Once per soft step |

Clues that matter for CosmicTiger:

- No production SPH code caps h at a fraction of a decomposition or PM cell; SWIFT does the inverse (cell adapts to hₘₐₓ), GADGET-4 and SWIFT default to no cap (`space_regrid.c`; GADGET-4 parameter docs). A cap is physically a density floor ρ_floor = m W(0, hₘₐₓ) (Dou+2024 eq. 4).
- Frozen membership with drift is standard, and completeness is kept either by exact refreshed bounds (GADGET-4 §5.7.1, PKDGRAV3 `bTreeRepair`) or by the pair inequality support + dxᵢ + dxⱼ ≤ cell width (SWIFT `cell.h`). Nobody uses membership geometry at search time as the guarantee.
- Symmetric max(hᵢ,hⱼ) pairs need the scatter side served: per-node max h (GADGET-2 §3.1.2) or a second exchange round (FDPS §3.2; Dave+1997 §3.3).
- Whole-cell proxies with the owner pushing `xv` then `rho` each active step (SWIFT `engine_maketasks.c`) is the one-way-push model; the PM step coinciding with the full rebuild and stray exchange (SWIFT `engine_step`) is exactly the PMMM outer boundary.
- Velocity skins (P3T, PeTar) or measured displacements (Verlet 1967, secondary via Nordlund notes) size the budget; SPH experience says a large skin pays poorly (Dominguez+2011, secondary via arXiv:2205.03074 §3.4.1).
- Typical TreePM/PMMM meshes have 2 to 4 spacings per cell (Ishiyama+2009 §2.3.1), so any cell-tied cap bites at realistic `meshSize`.
- Where the hydro cell must be larger than the gravity cell, codes coarsen the hydro structure (GOTPM 4 PM cells per cube; Hydra escalation), never the physics.

## 2. The core distinction

For gravity the frozen cell assignment is a choice of Hamiltonian split: H = T + V_far + V_near with the near/far partition fixed for the interval is a legitimate multi-rate splitting (Pelupessy+2012; Saha & Tremaine 1994; docs/pmmm.md:494-518), so a particle that leaves its cell is still integrated exactly for that split, and the only consequence is a controlled splitting error. SPH has no such freedom: the density, gradient and force sums are defined over every particle within max(hₐ,h_b) of the current position, and a missed pair is not a different splitting but a wrong operator (and a lost reciprocal kick in the pair-once RK2 at src/sph/sph.cpp:725-770). Therefore SPH membership may be frozen for ownership and tree topology, but the set of sources visible to a sink at each hydro event must be provably complete at that event, either by exact bounds or by a stencil inequality that accounts for drift, and ghost copies must carry current positions. The frozen gravity key is a fine place to hang the SPH forest; it is never the completeness certificate.

## 3. Recommended design

Synthesis: the comm_minimal protocol (one halo plan per outer interval, owner-driven one-way pushes) on a coarsened hydro mesh (from user_forest and hybrid_skin), with the four refutation fixes.

**Structure.** A hydro mesh of side n_h = largest power of two with n_h ≤ meshSize and N_gas^(1/3)/n_h ≥ 16 (16 mean spacings per hydro cell), clamped to n_h ≥ resolvedCoarseMeshSize (cosmictiger/gravity/mesh_size.hpp:45-55). Each hydro cell is a union of (meshSize/n_h)³ PMMM cells with exactly one owner because domain cuts lie on coarse planes (src/particles/domain.cpp:135-198); owner(key) = domainsFindRank(cell centre) as in src/gravity/mesh_exchange.cpp:27-34. Per locality: a gas-array directory {key, begin, end} from an O(N) counting sort, one `SphTreeNode` octree per occupied owned cell (existing splitter, src/sph/tree.cpp:539-616, box = cell box), one octree per received ghost cell over a persistent SPH ghost arena (lifecycle of `fmmGhostReserve/Clear/Free`, src/gravity/fmm/ghost_storage.cpp:29-89), and a fixed 27-entry source list per owned cell. No rank-routing nodes, no remote node lines, no gas cache lines on this path. Gravity-free runs keep today's global octree.

**Membership.** SPH key = particlesFrozenCell with the three indices shifted right by log₂(meshSize/n_h). Assigned at the boundary pass right after `pmmmKick(refreshMembership)` (src/core/driver.cpp:705-751), frozen for the interval, migration only at the boundary. A deep stray (dist(x, cell box) > c_h/4) triggers a neighbour-scoped re-sort by current cell before phase A of that event (no migration, gravity key untouched); this is a hard precondition of the completeness proof, not a heuristic (refutation 1).

**Communication budget** (512³ gas, 8 localities, L = 32 PMMM cells per locality side, meshSize 64 so n_h = 32, L_h = 16; gas fraction 0.5).

| Exchange | Payload | Bytes | When | Rounds |
|---|---|---|---|---|
| SPH halo plan | whole 27-stencil hydro cells, slab-filtered to particles within c_h/2 + s of the recipient's region | ~80 B/ghost; halo ≈ 20% of owned gas before filtering, ≈ 10% after | once per outer boundary (+ re-plan slice on deep strays) | 1 request/response (2 with foreign slices) |
| Phase A | x, v, h, A, rung for changed set {rung ≥ min(r+1, r_prev)} plus touched; x, v for every ghost of exported cells (eager replication, default) | 24–40 B/ghost | per hydro event | 1 one-way push per neighbour |
| Phase B | ρ, Ω, α, Balsara (Float16/uint8 packed) | ~24 B/active or semi-active ghost | per event | 1 one-way |
| Phases C, F | reverse `SphPairUpdate` keyed by arena index | 36 B/touched ghost | per event | 2 one-way |
| Phases D, E | predictors v*, A*; Balsara*, α* | 28 + 12 B/changed ghost | per event | 2 one-way |
| Gravity FMM ghosts (reference) | `ParticleGhost` 18 B | 2-cell halo = 42% of owned, all species | per near solve | request/response per cell |
| Gravity pp near (reference) | `PmmmNearParticle` 44 B, whole stencil cells per sink batch (src/gravity/evaluation.cpp:148-193) | ≈ 2.2 GB/locality/base step | per inner step | request/response per batch |
| One global octree (reference) | remote leaves in a 1.21h + leaf shell (~7% of owned) re-staged 5–6 passes/event at ~60 B, plus node lines and checklists | ≈ 21–25 B/owned particle/coarse event | per pass of every event | tens of dependent hops/event |

Per base step with seven rungs the SPH pushes total roughly 1.5 to 2× the pp near-field bytes with zero request/response traffic in steady state; the global octree moves fewer bytes only if its cache lines are shrunk from 10922 particles (src/core/options.cpp:1782) to leaf size, and it always costs an order of magnitude more rounds (refutation 4).

**Local work.** Per event O(N_local + G + cells): changed-set and stray scans, bounds refresh, 27-root stencil walk with leaves bounded by cell occupancy, eager replication O(G). Per plan: O(N) counting sort plus per-cell builds O(Σ p log(p/B)) with p the cell occupancy. Linear per locality up to the per-cell depth factor; rebuilding every event is allowed under your rule and is not needed.

**hₘₐₓ rule.** hₘₐₓ = c_h/2 − s, c_h = 1/n_h, with s the skin set at the boundary from the 99.9th-percentile gas drift over the coming interval, capped at c_h/4. In PMMM cell widths hₘₐₓ = (meshSize/n_h)/2 − s; in spacings hₘₐₓ = (N_gas^(1/3)/n_h)/2 − s ≥ 8d − s. Every stored h, active or not, is clamped to kSupportLimitRatio·hₘₐₓ in the plan's O(N) pass (refutation 1); solved particles get hmaxₐ = min(hₘₐₓ, c_h/2 − distₐ) through the existing hₘₐₓ parameter (src/sph/prehydro_cpu.cpp:62-66). Completeness is the pairwise statement: with dist ≤ c_h/4 for all particles at search time and h ≤ c_h/2 − s, distₐ + dist_b + max(hₐ,h_b) < c_h = the gap between non-adjacent cells, so every gather and scatter pair lies in the 27 local trees.

| 512³ gas, s → 0 | meshSize 64 | 128 | 256 |
|---|---|---|---|
| n_h | 32 | 32 | 32 |
| hₘₐₓ (PMMM cells) | 1 Δ | 2 Δ | 4 Δ |
| hₘₐₓ (spacings) | 8 d | 8 d | 8 d |
| N = 42 kept down to ρ/ρ̄ | 0.038 | 0.038 | 0.038 |
| density floor ρ/ρ̄ | 0.005 | 0.005 | 0.005 |
| uncommitted 0.5/meshSize (spacings) | 4 d | 2 d | 1 d |
| its N at mean density / floor | 42 / 0.04 | ~28 / 0.32 | ~4 / 2.5 |

**Ghost and stencil rule.** w_sph = 1 on the hydro mesh, fixed; no widening ever. Owners serve the exported slab of each cell once per plan; per event they push one message per neighbour per phase, possibly empty, so the receiver's barrier is a message count. Semi-active marking is done by the owner from its ghost copies of active sinks (interacting pairs are provably in adjacent cells).

**Drift and rebuild triggers.** Boundary: always (sort, build, plan, clamp h, reset s). Deep stray dist > c_h/4 anywhere on a locality: local re-sort before phase A, exported slabs re-snapshotted inside the phase-A message, missing stencil cells requested (one extra half-round, rare: c_h/4 = 4 spacings in one interval). Owned-tree re-split without membership change at the existing active-fraction rule (cosmictiger/sph/tree.hpp:208-231). Ghost trees retained with bounds refresh (owner's in-cell order is stable). Lazy drift of unchanged copies is an optimisation enabled only after bitwise comparison against eager replication.

**Timestep interleaving.** Boundary pass: `pmmmKick(frozenNearClose)` → `pmmmBoundaryMigrate` → `pmmmOuterBoundary` → `pmmmKick(refreshMembership)` → SPH plan. Inner step: `gravityBeginDeferredGasOpening(); pmmmKick(frozenNear, deferHydroOpening); gravitySealDeferredGasOpening();` → D(δ/2) → stray check → phases A–F around the density, stage-1, Balsara, stage-2 sequence → `gravityFinishDeferredGasOpening()` → D(δ/2). The gas branch of `particleMeshApplyField` (src/gravity/pm_kick.cpp:126-155) applies closing half-kicks directly and routes opening half-kicks into `gravityDeferredGasOpeningData()[particlesCatIndex(i)]` when deferring; `deferHydroOpening` is threaded through `pmmmIntervalParameters` (refutation 3). The gravity sort stays inside K_near, so gas-visible positions are immutable during the hydro stage: that is the "immutable source stage" of driver.cpp:792-794.

**Void behaviour.** N₀ = 42 down to ρ/ρ̄ ≈ 0.04, N ≈ 19 at 0.01, floor at 0.005 ρ̄, independent of `meshSize`. Truncation of strays binds only for void gas that drifted more than ~2 spacings past a face. Diagnostics per event: stray count, maximum truncation, bytes per phase.

**Reused code.** `neighbors`/`periodicIndex`/`cellOwner` (ghost_exchange.cpp:33-49, mesh_exchange.cpp:10-34), the `planGhosts`/`SourceCell` promise protocol (ghost_exchange.cpp:51-176, cell_schedule.cpp:19-104), the ghost arena lifecycle, `particleMeshSortImpl`'s partition pattern and `sphParticlesSort`, the SPH splitter, leaf bounds and refresh, the FindNeighbors walk and per-particle filters, `SphRunWorkspace::stage` with a ghost store, the bounded target and hₘₐₓ plumbing, `SphPairUpdate`/`sphOwnsPairKick`, the deferred-gas-opening buffers, `particlesFrozenCell`.

**Implementation order.** (1) Options: lift the SPH+PMMM rejection (options.cpp:380-383, 1717-1719), derive n_h and hₘₐₓ = 0.5/n_h − s replacing options.cpp:1668-1672, startup table, refuse if 0.8·hₘₐₓ < 2.15 d. (2) Gas gravity kick with deferred opening in `pm_kick.cpp` and the flag threading; one-locality test against the tree ladder. (3) Gas cell sort and directory; plan-time h clamp. (4) Owned forest build/refresh. (5) SPH ghost arena, plan with slab filter, ghost trees. (6) Local stencil search and staging; brute-force O(N²) pairwise completeness tests on 1 and 2 localities with injected drift. (7) Phases A–F with eager replication, reverse updates by arena index, message-count barriers; 1/2/8-locality conservation tests. (8) Deep-stray re-plan and foreign slices. (9) Changed-set pushes, lazy drift behind the eager oracle, byte counters versus pp traffic. (10) Star formation dead-in-place, restart, docs (docs/pmmm.md:585-588, docs/sph_octree_plan.md).

## 4. Alternatives and why not

**One global SPH octree (decoupled structures).** Correct and cheapest to build (two weeks), with hₘₐₓ = half the box, and the right first milestone for physics validation. Decisive flaw under the communication rule: every pass of every event clears the node cache (src/sph/sph.cpp:202-214) and re-stages the remote shell through 128 kB gas lines (sph.cpp:697), giving tens of dependent request/response hops per event; bytes are comparable, rounds are an order of magnitude worse, and without the outer interval the tree is rebuilt every step.

**user_forest (excursion-gated stencils).** The exact gate (m−1)Δ < hₘₐₓ + D_A + D_B is the best single idea in the set, but because other cells' D is not exchanged the bulk bound D = Δ/2 makes the stencil 125 cells at every event, copies are transient, an outlier all_gather is added per event, and widening to 343 cells is routine in infall; it also drops the reverse pair path the RK2 needs.

**SWIFT-style overlap-driven forest.** Right instinct (summary directory, box-driven selection, stray root, no cap), but it adds request/response rounds inside the smoothing-length iteration, widens reach with void h, re-fetches whole cells every event, and its boundary-only directory misses interior cells a deep sink can reach.

**hybrid_skin.** Good hydro mesh and startup table, but T_skin shortens the shared PMMM interval from gas velocities (a displacement limit you rejected), a single fast particle anywhere exhausts the global two-largest-displacement budget and widens every locality's halo by 2.74×, and the 125-cell halo is re-fetched twice per event.

## 5. Open questions for Dominic

1. Hydro cell size: 16 spacings (hₘₐₓ ≈ 8 d, N₀ kept to ρ/ρ̄ ≈ 0.04, halo ≈ 10–20% of owned gas) or 32 spacings (hₘₐₓ ≈ 16 d, floor 6e-4, double the halo)? Or expose it as `hydro.meshSize` rather than a derived constant?
2. Is a density floor at ~0.005 ρ̄ and N ≈ 19 at ρ/ρ̄ = 0.01 acceptable for your void science, or do you want the w_sph = 2 escape hatch kept as an option despite 2.7× bytes?
3. Do you accept the deep-stray re-sort at c_h/4 as a precondition (membership literally changes for those particles mid-interval), or do you prefer the literal frozen variant where strays' supports shrink by their excursion?
4. Eager replication of every ghost position per event (24 B/ghost, exact) as the permanent default, or lazy drift once validated?
5. Gravity-free runs: keep the global octree path indefinitely, or adopt a virtual hydro mesh so one SPH path remains?
6. Should the non-interval PMMM path (per-step plan) be supported, or should SPH + PMMM require `gravity.pmmmOuterInterval`?