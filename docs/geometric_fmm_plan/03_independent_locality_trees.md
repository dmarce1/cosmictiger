# 3. Independent locality trees and reference traversal

Created: 2026-11-14. Status: planned; not implemented by this task.

Recommended model: **GPT-6.1 Sol**. Reasoning effort: **High**.

## Working agreement

This is a plan for one future task, not evidence of implementation. Reading this file
alone does not authorize source edits or execution. Follow the user's requested mode:
return plans and pasted-code edits in chat; change repository files only when explicitly
asked to implement on disk. Run builds, tests, or benchmarks only when requested.
When checks are authorized, use the focused checks below and the current shared resource
budget; do not launch an unrelated full suite. Point out inconsistencies and continue
independent in-scope work. Do not proceed to the next step automatically.

Repository: `/home/dmarce1/workspace/cosmictiger`. Read applicable `AGENTS.md` and current
handoffs before editing. Preserve unrelated work and coordinate ownership. Model settings
below are recommendations; they do not authorize spawning helpers or switching models.
Report changed files, decisions, validation actually performed, and remaining limitations.
Update this step's completion note only when asked to maintain these plan files. Do not
commit or push unless requested.

## Agreed architecture

- Recursive spatial cuts determine particle ownership only; they need not follow FMM cells.
- Each locality builds its own kd-tree from the unit box, containing only its owned source
  particles and their multipole contributions. Only that owned tree supplies sinks.
- Bisect at the geometric midpoint using the repeating depth schedule
  `xyz | zyx | zxy | yxz | yzx | xzy`. Do not shrink boxes or move centers toward particles.
- Leaves measure maximum particle distance from the fixed center. Parents use the minimum
  of the geometric half-diagonal and the maximum child-center distance plus child radius.
- Each remote source locality has a separate imported, pruned kd-tree. Never graft ghosts
  into the owned tree. Identical geometric keys on different localities have different
  particle sets and partial multipoles; keep their contributions distinct.
- The source checklist starts with the owned root and remote source roots. Fine halos
  determine required detail, while distant contributions remain represented by coarse
  multipoles. There is no six-peer limit. Keep the existing Newtonian/Ewald conventions.

## Scope and prerequisites

Require the step 1 contract and step 2 local builder. Separate particle ownership from
FMM geometry and establish a correct multi-root reference path before pruning exports.
This step is implementation only when the user explicitly authorizes source edits.

## Work

1. Keep the existing recursive ownership decomposition and particle migration. Remove
   ownership-domain boxes from the geometry of the new FMM trees: each locality builds
   from the unit box at depth zero using its own source particles.
2. Separate collective arena preparation/reset from per-locality root construction.
   Calling root construction on every rank must not recursively allocate/reset every
   other rank's tree again. Publish a root descriptor and generation for each source rank,
   including explicit empty-source state.
3. Preserve source-locality identity separately from geometric cell identity. Existing
   TreeId uses locality and arena index; decide how stable geometric identities coexist
   with these runtime addresses. Never deduplicate cells solely by geometric key.
4. Start a local sink traversal with its owned root and a source checklist containing
   the owned root plus remote roots. Keep Newtonian and Ewald checklists consistent with
   existing periodic conventions. Only owned sinks accumulate forces and kick returns.
   No remote tree becomes a sink, and ghosts never enter owned particle ranges.
5. For this reference milestone, traverse complete remote source trees using the existing
   demand-fetch path where practical. This need not replicate complete remote trees in
   memory: the complete logical trees are available through fetching. Adapt the cache
   and source reader only enough to support independent roots and generations.
6. Audit assumptions equating single-rank ownership, local memory residence, localRoot,
   and sink eligibility. Preserve active-sink/all-required-source semantics and self-force
   exclusion. Keep partial multipoles separate; do not add a global aggregate alongside
   their constituent rank roots.
7. Include necessary CUDA data-model/staging compatibility in the design. Establish the
   CPU reference first; do not claim CUDA support until its separate checks are performed.
   Any temporarily unavailable execution path must be explicit, not silently incorrect.

## Focused validation, only if requested

Compare against direct or complete-source references using identical softened Newtonian
and periodic/Ewald conventions. Include 1, 2, and at least one non-power-of-two rank count,
unequal ownership cuts, different adaptive depths, identical geometric cells with disjoint
source particles, and an empty rank. Check global source mass/counts and exactly-once
owned sink updates. Compare numerical accuracy, not assumed bitwise equivalence across
changed summation orders.

## Completion criteria

The ownership map is unchanged, rank-local unit-box trees are independent, and the
multi-root reference walk accounts for every source owner exactly once. This reference
path will be the comparator for steps 4 and 5. Export pruning and communication optimization
remain out of scope.

## Source entry points

`src/particles/domain.cpp`, `cosmictiger/particles/domain.hpp`,
`src/gravity/global_fmm/distributed_tree.cpp`, `src/gravity/global_fmm/execution.cpp`,
`src/gravity/global_fmm/remote_cache.cpp`, `src/gravity/global_fmm/particle_exchange.cpp`,
`src/gravity/source_access.cpp`, `cosmictiger/gravity/tree.hpp`,
`cosmictiger/gravity/kick_walk.hpp`, `cosmictiger/gravity/source_access.hpp`,
`tests/gravity/global_fmm/fmm_tree.cpp`, `tests/gravity/global_fmm/fmm_walk.cpp`.

Next: `04_remote_source_tree_exports.md`.
