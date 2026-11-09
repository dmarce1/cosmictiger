# Distributed SPH: parent tree of components over ghost-inclusive local octrees

## Status, November 9, 2026

| Milestone | State |
|---|---|
| Step 0 | Done: bounded target reverted, half-box error (`efa174cab`). |
| M1 parent tree | Done (`b6676d685`): `SphParentNode` components, topology and 1 to 4 locality tests. Its topology drives the collectives since M4. |
| M2 ghosts | Done (`b6676d685`); the only distributed path since M5, which retired `compute.sphGhosts`. Differs from 3.3 to 3.5: the export set is recomputed at every neighbor walk from a cut of at most 64 local tree nodes, instead of plan epochs with margins, so checks A and B and the extension round (M3) are not needed for correctness. |
| M3 | Subsumed by the per-walk sync. Margins could later skip syncs that would export nothing new. |
| M4 collectives | Done: every SPH collective forwards along `sphCollectiveChildren()` (the parent-tree fan-out, each rank reached once). The routing nodes are gone: each locality builds and refreshes its own octree at depth 0 in parallel, and a level-0 walk runs one walk per locality whose checklist is the local root plus the ghost leaves. The returned tree id is rank 0's local root. |
| M5 fetch path | Done: the remote node cache, the gas cache lines, the remote rung-candidate fetch and the roots broadcast are deleted. A remote node that is not a ghost leaf, remote gas read outside staging, and ghost fields older than the current refresh all throw. References: brute-force leaf lists (`sph_ghosts*`), owner-gathered records (`sph_octree_application_mpi`), and distributed runs against one locality (`sph_ghosts_application_mpi`), which exposed the stale gas map fixed in `529d66326`. `compute.particleCacheLineSize` stays: gravity and group finding use it. |
| M6 bytes | Changed-set pushes done: each pass refreshes only the field groups it stages (`sphRunGhostFields`), owners send only the particles whose bytes differ from what each importer holds, and ghost blocks survive a sync whose export is unchanged. Reading a group the current refresh did not bring throws. Field bytes fell from 94.3 to 23.9 MB (Jeans wave, 4 localities, 12 steps, 8000 gas) and from 41.9 to 11.2 MB (Sod, 4 localities); of what remains, densities are 38%, velocities 29% and positions 22%. Diffing leaf records saved 2% (every sync follows a bounds refresh) and was dropped. Local drifting of ghost copies is deferred: positions are now a fifth of the field bytes. |
| M7 gravity | Done on the hierarchical (HOLD) integrator only, at the user's direction; traditional and global reject SPH with gravity. Gate: `sph_jeans_wave`, `pmmm_evolution_sph_hierarchical_{cpu,mpi}`. |
| M8 | Partly done: CUDA staging with ghosts passes the CUDA tree's 124 SPH tests (traditional integrator; no hierarchical CUDA tree yet). Open: restart inside an outer interval with gas. |


November 8, 2026. The plan as written then; see the status table above for what is implemented. It turns the design Dominic
stated on November 8 into milestones, and follows the literature and design study in
`docs/sph_pmmm_coupling_research.md`. Line numbers refer to `main` at `095938065` plus the
uncommitted hₘₐₓ edits described under "Step 0".

Priority rule for every choice below (user, November 8): less inter-locality
communication comes first; extra local work is acceptable while it scales no worse than
linearly with the particles or mesh cells of a locality.

## 1. The design as stated

1. A parent tree whose leaves are exactly the locality domains of the PMMM decomposition.
   Every parent node is an HPX component and all connect to one root.
2. Each parent leaf, one per locality, is the root of that locality's SPH octree.
3. The parent is a hybrid octree/kd tree: 8-way while a node still holds 8 or more
   localities, kd while it holds 2 to 7.
4. The local particle set includes ghost zones, as PMMM's does, and the ghost region is
   part of the local octree. Local octrees overlap only as ghost over interior.
5. Parent cells track bound changes as the tree does now, and also the change in the
   boundary layer of every cell that has a ghost zone.
6. Fine timestepping is synchronized from the parent root, which lives on locality 0.
   Every other interior parent node lives on the locality of exactly one of its children.
7. The tree can then tell when a parent leaf needs particles outside its original ghost
   zone.

## 2. What the code had when this plan was written

This section and section 4 describe the code of November 8, 2026, before M1; the status
table at the top records what has changed since.

| Fact | Where |
|---|---|
| The SPH tree already has rank-routing nodes above the local roots. They follow the binary domain keys, `(key << 1) + child`, split ranks at the middle, and a node is created on `processRange.first`. | `src/sph/tree.cpp:400-424`, `:526-537` |
| Domain cuts cycle x, y, z with `depth % 3` and lie on coarse-mesh planes when gravity is on. | `src/particles/domain.cpp:300`, `:338`, `:435-441` |
| A local root spans the whole unit box, because retained gas may sit outside its rank's domain. | `src/sph/tree.cpp:485-493` |
| Leaf and node bounds (`innerBox`, `outerBox`, `driftBox`) are refreshed from current positions at every event; `sweptSphBounds` predicts drift. | `src/sph/tree.cpp:159`, `:791-899` |
| The neighbor walk is one distributed recursion from the global root, opening on `outer ∩ inner` in either direction. It clears the remote node cache at level 0. | `src/sph/sph.cpp:202-372` |
| Each pass copies ("stages") its sources into host arrays. Remote sources come through five gas cache-line readers plus the rung-candidate reader. | `src/sph/sph.cpp:814-916`, `cosmictiger/sph/particles.hpp:93-102`, `cosmictiger/sph/time_integration.hpp:59` |
| A gas cache line is 131072 bytes of positions, 10922 particles with 4-byte coordinates. | `src/core/options.cpp:1782` |
| Each pair is evaluated once by `sphOwnsPairKick`; the other endpoint's increment travels to its owner as `SphOwnedPairUpdate`, keyed by (rank, gas index). | `cosmictiger/sph/pair_kick.hpp:135-142`, `src/sph/sph.cpp:609-641` |
| The density solve loops: solve, refresh bounds, find neighbors again, until no support outgrew the 1.21 search multiplier. | `src/core/driver.cpp:371-480`, `cosmictiger/sph/sph.hpp:61` |
| The tree is rebuilt when a third of the particles are active, after migration, or when invalidated. | `src/core/driver.cpp:237-259` |
| Collectives fan out through `hpxChildren()`: 27 call sites in `src/sph`. | `sph.cpp` 10, `time_integration.cpp` 7, `particles.cpp` 6, `tree.cpp` 4 |
| PMMM ghosts are whole near-stencil cells in a separate arena, with position, type, rung and mass only. | `src/gravity/fmm/ghost_exchange.cpp:7-49`, `cosmictiger/gravity/fmm/ghost_storage.hpp:14-31` |
| SPH with gravity is rejected at startup, and the mesh kick throws for gas. | `src/core/options.cpp:380-383`, `:1717-1720`, `src/gravity/pm_kick.cpp:132` |
| There are no HPX components in the repository. HPX is 1.11.0 at `/home/dmarce1/local/hpx/Release`. | `git grep component_base` is empty |

Consequences for the design:

- Points 1, 2 and 6 describe a regrouping of what the routing nodes already do. The
  placement rule in point 6 is the existing one.
- Point 3 needs no change to the decomposition. Three consecutive binary levels always
  cover all three axes, and a node with 8 or more ranks always yields eight nonempty rank
  groups, so its eight grandchildren become its children. The children are not congruent
  octants, because each half chooses its own load-balanced plane; every child keeps its
  own box.
- Point 4 is the real change. It replaces the on-demand fetches of remote nodes and gas
  lines with copies that arrive by one-way pushes.

## 3. Design

### 3.1 Parent tree

One component type, `SphParentNode` (`cosmictiger/sph/parent_tree.hpp`,
`src/sph/parent_tree.cpp`), built from `hpx::components::component_base`, with actions
declared by `HPX_DEFINE_COMPONENT_ACTION` and created with `hpx::new_<SphParentNode>` on
`hpxLocalities()[processRange.first]`.

- **Topology.** A node covering n ranks has 8 children when n ≥ 8 (domain keys
  `(key << 3) + i`), 2 children when 2 ≤ n ≤ 7 (`(key << 1) + i`), and is a leaf when
  n = 1. The root covers `[0, hpxSize())` and so sits on locality 0.
- **Lifetime.** Built once per domain decomposition. `domainsRebound()` destroys and
  rebuilds it; an ordinary plan epoch only updates boxes.
- **State per node.** Domain box, rank range, child ids and rank ranges, and for each
  child the last reported bounds: current `innerBox`, `outerBox`, `driftBox`, the
  published interest boxes (3.3), the remaining margin (3.5), active count and ghost
  counts. A node's own values are the unions and sums of its children's.
- **Leaf.** A leaf is a thin handle. Particle data, the octree arena and the ghost store
  stay in the locality's existing static storage; the leaf's actions call the local
  functions.
- **Actions.** `plan` (build or rebuild local trees and publish interest), `refresh`
  (per-event bounds, margins and flags), `discover` (down-sweep that delivers each leaf
  its neighbor leaves and their boxes) and `run` (broadcast one SPH pass and reduce its
  `SphRunReturn`). All four are a broadcast down and a reduction up.

The parent tree carries control, bounds and discovery only. Particle data never passes
through it; ghosts travel directly between the two leaves concerned.

### 3.2 Local octree with ghost leaves

The local root has two parts: the owned octree over the gas arrays, unchanged, and a
ghost part.

- **Ghost store** (`cosmictiger/sph/ghost_store.hpp`): a structure-of-arrays arena with
  the fields the passes read (3.6), following the reserve/clear/free lifecycle of the
  gravity ghost arena.
- **Ghost leaves mirror owner leaves.** An owner exports whole leaves. The importer keeps
  each as an `SphTreeNode` record with the owner's rank and gas index range, pointing at a
  contiguous slice of the ghost store in the owner's order. A push is then a range copy
  with no index list, and `SphPairSourceId` still names (owner rank, owner gas index), so
  the pair-owner rule and the reverse pair updates work unchanged.
- **A small tree over the ghost leaves**, built by the importer from the leaf boxes, hangs
  under the local root beside the owned octree. The local walk starts at the local root
  and never leaves the locality.
- **Ghosts are sources only.** They are never targets, never solved and never kicked.
- **Ghost bounds are recomputed locally** by the same refresh code, from the ghost data
  the importer holds. No bounds are sent after the plan.

### 3.3 Interest boxes and the plan

At a plan epoch each leaf A publishes two **interest boxes** through the parent tree:

- inner⁺: the bounds of every particle A owns, swept over the coming interval with
  `sweptSphBounds`;
- outer⁺: the bounds of every owned support, active or not, at the 1.21 search
  multiplier, swept the same way.

`discover` then gives every leaf B the list of leaves A whose boxes meet its own, with
their boxes. The test is the walk's symmetric one, so the relation is symmetric: A is on
B's list exactly when B is on A's. It is not limited to the 26 adjacent domains, and it
uses the periodic intersection the walk already uses.

For each A on its list, B walks its own tree and **exports every leaf ℓ** with

    inner(ℓ) ∩ outer⁺(A) ≠ ∅   or   outer(ℓ) ∩ inner⁺(A) ≠ ∅.

Here outer(ℓ) bounds every support in the leaf, active or not, at the same multiplier.
The tree's `outerBox` covers active supports only, so the refresh gains this second
bound. B sends each exported leaf's node record and a full record of its particles. That
set is A's original ghost zone for the interval.

### 3.4 What happens at every fine event

| Step | Who | Messages | Dependent hops |
|---|---|---|---|
| 1. `refresh`: root broadcasts the event; each leaf refreshes its owned bounds, runs the two checks of 3.5 and reports boxes, margins, counts, flags | parent tree | one down and one up per tree edge | 2 × parent depth |
| 2. Extension, only if a flag is set (3.5) | parent tree, then the leaves concerned | interest boxes, then appended leaves | one more sweep and one push |
| 3. Before each pass: every leaf pushes the fields that pass reads to each leaf on its list | leaf to leaf | one message per neighbor per pass, possibly empty | 1 |
| 4. `run`: root broadcasts the pass; every leaf finds neighbors, stages and computes locally | parent tree | one down and one up per tree edge | 2 × parent depth |
| 5. Hydro passes only: reverse pair updates to owners, then the commit collective | leaf to leaf, as today | one message per neighbor | 1 |

Receivers count messages: a leaf proceeds when it has one message for the pass from every
leaf on its list. No pass issues a request and waits for a reply.

### 3.5 Completeness, and needing more than the original ghost zone

Claim: if a target leaf s on A and a source leaf ℓ on B satisfy today's opening test with
their current boxes, then ℓ is in A's ghost set.

Because outer(s) ⊆ outer(A) and inner(s) ⊆ inner(A), it is enough that

- **check A (local to A):** A's current boxes, taken over all owned particles and all
  owned supports, lie inside its published interest boxes, and
- **check B (local to B):** every leaf of B that passes the export test against A's
  published boxes, using B's current leaf boxes, is already exported.

Both checks run in step 1 and again whenever a density pass refreshes bounds. Each is a
box comparison or a walk of the locality's own tree against two boxes per neighbor, so
the work is linear in boundary leaves.

- If check B fails, B appends the missing leaves to its next push. No extra round.
- If check A fails, A publishes larger boxes. The parent tree reruns `discover` for A,
  which may add leaves to its list, and the leaves concerned append exports. This is the
  extension round, and it is the case point 7 of the design asks the tree to detect.

The ghost set is therefore grow-only within a plan epoch, strays and growing supports
included, and the neighbor leaf sets equal those of today's global walk. That equality is
the acceptance test of milestone 2.

"Margin" in 3.1 is the smallest distance, over the six faces, between a leaf's current
box and its published box. It is the tracked boundary-layer change of point 5. A margin
that keeps shrinking toward zero predicts an extension.

### 3.6 Ghost fields by pass

From `SphRunWorkspace::stage()`:

| Pass | Fields read from sources | Bytes per particle |
|---|---|---|
| Smoothing lengths | position | 12 |
| Density and gradients | internal energy, h, rung, velocity | 21 |
| Rungs | h, rung or rung candidate | 5 |
| Hydro force | internal energy, h, density, grad-h term, Balsara factor, α, rung, velocity | 37 |

Sizes assume 4-byte coordinates; fixed64 builds add 12 bytes to the position. Positions
are fixed within an event and are sent once per event. One midpoint event runs
the smoothing pass, density and gradients, force stage 1, the Balsara refresh and force
stage 2, about 130 bytes per ghost if every field is resent every pass.

### 3.7 Plan epochs

- **With PMMM:** the plan follows the boundary pass, after migration and the owned-tree
  rebuild. Membership is frozen for the interval, so owned gas may stray outside its
  domain; the boxes cover that. Owned trees are not rebuilt inside an interval.
- **Without gravity:** a plan epoch is every event at which the existing rule rebuilds
  the tree. Migration of the active rungs stays where it is.
- **A forced local rebuild** inside an epoch makes that owner re-plan its exports: its
  importers drop its ghost leaves and receive new ones in its next push.
- **Restart:** the ghost store is not checkpointed. A loaded run starts with a plan epoch.

### 3.8 hₘₐₓ

Completeness does not depend on hₘₐₓ, because the ghost set follows the supports. The
solve is the plain bounded Newton iteration again (commit `efa174cab`): the bounded
neighbor target was reverted, `hydro.minSmoothingLength` and `hydro.maxSmoothingLength`
are options as before, and a support that reaches half the periodic box and still lacks
neighbors stops the run (decision 1 in section 6).

## 4. Communication compared with the code of November 8

| | Today | This plan |
|---|---|---|
| Remote tree nodes | fetched on demand in cache lines during the walk, cache cleared every walk | none; ghost leaf records arrive with the plan |
| Remote particle data | fetched on demand, 10922 particles per line, caches cleared every pass | pushed by the owner, whole leaves, only for exported leaves |
| Dependent request and reply chains per pass | walk depth plus fetches | none in steady state |
| Collective depth | log₂ P through `hpxChildren()` and the routing nodes | ⌈log₈ P⌉ plus at most 3 kd levels |
| Reverse pair updates | one push per owner | unchanged |

Bytes, as estimates only, for the research report's example of 512³ gas on 8 localities
and about 1.5 million ghosts per locality: about 190 MB per event with every field resent
every pass (milestones 2 to 5), and 18 MB of positions plus about 120 bytes per changed
ghost after milestone 6. Milestone 6 measures these before choosing its second step.

## 5. Milestones

Every milestone builds and tests within the shared budget in `AGENTS.md`, currently 6
cores in total, so two-locality runs use 3 threads each.

**Step 0. Settle the working tree. Done November 8, 2026.** The hₘₐₓ-only edits were
discarded (saved as `handoff/hmax_only_uncommitted_2026_11_08.patch`) and the bounded
target reverted in `efa174cab`, built and tested on CPU and CUDA.

**M1. Component spike.** A minimal `SphParentNode` with one action, created on two
localities from locality 0, and a unit test `tests/sph/parent_tree.cpp`. Purpose: prove
component registration and remote creation in this build before anything depends on it.
Gate: the test passes on 1 and 2 localities.

**M2. Ghost store, plan and local walk, behind `compute.sphGhosts`.**
- Ghost store, export selection, mirrored leaves, the tree over ghost leaves.
- A `Ghost` branch in the six source readers; the neighbor walk restricted to the local
  root when the option is on.
- Plan and pushes as in 3.3 and 3.4, with every field resent every pass. Interest boxes
  travel by a plain gather to locality 0 and a broadcast for now.
- The fetch path stays as the reference.
- Gates: on 2 localities the set of (target leaf, source particle) pairs equals the fetch
  path's for every pass of a short run; a brute-force O(N²) neighbor check on a small
  case; momentum and energy conservation equal to the fetch path within round-off;
  `sph_only_mpi`, `sph_octree_application_mpi` and `smoke_sph_mpi` pass with the option
  on.

**M3. Checks and extension.** Checks A and B, grow-only appends, the extension round,
per-owner re-plan after a forced rebuild, margins in the event log. Gates: tests that
inject drift and support growth past the published boxes and still match the brute-force
neighbor sets; a test with a stray that crosses a whole neighbor domain.

**M4. Parent tree.** The full component tree with the 8-way and kd rule; `plan`,
`refresh`, `discover` and `run`; the 27 `hpxChildren()` collectives in `src/sph` routed
through it; rank-routing nodes removed from the `SphTreeNode` arena, so a local root is
depth 0. Gates: topology unit tests for 1, 2, 3, 7, 8, 9, 12 and 64 ranks against
`domainsRange`; every M2 and M3 gate again; identical results to M3 on 2 localities.

**M5. Remove the fetch path.** Delete the remote node cache, the gas cache lines and the
remote branches of the readers; make ghosts the only distributed path; stop reading
`compute.particleCacheLineSize` in `src/sph` (gravity and group finding keep it) and retire
`compute.sphGhosts` through the removed-options list in `tests/run_options.py`.

**M6. Fewer bytes.** First, changed-set pushes: an owner resends a field only for
particles a pass wrote. Then one of two ways to stop resending every position at every
event, chosen from measured byte counters: drift ghost copies locally and prove the
result bitwise against the resend path, or have each leaf send its active boxes to its
neighbors so that owners refresh only the leaves an event can touch.

**M7. Gravity.** Lift the two startup rejections; write the gas branch of
`particleMeshApplyField` with the deferred opening kick that `kick.hpp:223-228` already
declares; place the SPH plan after `pmmmBoundaryMigrate`; keep gas membership frozen
through the interval. Gates: one-locality energy test with gas and dark matter; the
PMMM interval tests with gas on 1 and 2 localities.

**M8. Remaining surfaces.** CUDA staging from the ghost store (kernels unchanged),
restart inside an interval, star formation and chemistry closing passes, flop accounting,
`docs/pmmm.md`, `docs/sph_octree_plan.md`, `docs/source_layout.md`.

Order rationale: ghosts come before the parent tree because the fetch path is the only
reference for validating them, and because routing the old distributed walk through
components would be work that M2 makes unnecessary.

## 6. Decisions needed from Dominic

1. **hₘₐₓ. Settled November 8, 2026, final:** no special treatment. The usual solve
   with its configured limits, and an error if a support reaches half the periodic box
   still lacking neighbors (`efa174cab`). The mesh tie, the bounded target and the
   removal of hₘᵢₙ are all dropped.
2. **Shape of the ghost zone.** This plan uses whole owner leaves selected by boxes,
   about one support plus one leaf deep. The literal PMMM analogue is whole near-stencil
   mesh cells, which the research estimated at 42% of owned particles for a 2-cell halo.
3. **Where ghosts live.** This plan keeps them in a separate store under the same local
   root. Interleaving them into the gas arrays would also give them base-particle slots,
   which the PMMM sort and the gas index map would then have to skip.
4. **kd levels.** Binary for 2 to 7 ranks, as the decomposition is. A 4-way level is
   possible for 4 to 7.
5. **Plan epoch without gravity.** The existing one-third-active rebuild rule, as in 3.7.
6. **Milestone 6's second step.** Local drift of ghosts or per-event active boxes.

## 7. Risks

- **Byte volume before M6.** Resending every field every pass is the simplest correct
  version and is heavy. It is a validation stage, not a production setting.
- **Interest boxes are one box pair per locality.** A single wide support near one face
  widens that face's whole layer. If the counters show it, publish a box pair per
  boundary subtree instead; the protocol does not change.
- **Summation order.** Staging order differs from the fetch path, so results match to
  round-off, not bitwise. The exact gate is the neighbor set, not the sums.
- **Semi-active sinks.** A target that is only a neighbor of a remote active particle
  must still see all its own neighbors. The export test covers it because outer⁺ spans
  every owned support, active or not.
- **Component registration.** No component exists here yet; M1 is there to find any
  build or startup problem first.
- **Concurrent work.** Another conversation may be editing `src/sph`. Check ownership and
  the handoff files before each milestone.

## 8. Not in this plan

CRKSPH and the meshless schemes, any change to the PMMM interval or its near field, a
displacement limit on the coarse interval, and per-cell SPH trees on a hydro mesh.
