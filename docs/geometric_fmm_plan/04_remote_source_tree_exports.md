# 4. Destination-specific remote source-tree exports

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

Require the step 1 mixed-level export contract and step 3 complete-source reference.
Construct pruned source-tree descriptions for destinations. Transport integration belongs
to step 5; begin with an in-memory exporter/consumer so selection can be checked independently.
Do not merge these exports into destination-owned trees.

## Work

1. Build candidate regions from the destination ownership bounds rounded outward to
   level-cell boundaries and padded conservatively. Handle periodic wrapping explicitly.
   There is no six-peer cap, and rank adjacency alone does not determine required detail.
2. Use geometric Rmax(d), not a guessed actual radius, to predict coverage. For arbitrary
   source and target depths, the current Newtonian cell-cell sufficient separation is
   H(ds,dt)=max((Rmax(ds)+Rmax(dt))/theta, Rmax(ds)+Rmax(dt)+h_G).
   A rectangular halo can overinclude cells; it must not omit needed interactions.
3. Implement the contract's actual mixed-depth selection rule. Geometry predicts possible
   cells, not target topology. If target summaries are required, make their availability
   an explicit phase rather than assuming the source knows the destination's leaves.
   Preserve the periodic/Ewald coverage contract in addition to the Newtonian halo.
4. For each destination, construct a pruned version of this source locality's tree with
   unit-box root geometry and the necessary ancestor scaffold. Keep coarse multipoles
   for distant sources, children where refinement is needed, and particle payloads for
   unresolved true leaves. Near-field intersection selects detail, not whether a distant
   nonzero source owner contributes gravity at all.
5. Define explicit node kinds: internal, true particle leaf, and multipole-only terminal.
   Store counts/occupancy, topology, cloud radius, multipoles, source identity, and generation.
   Reconstruct geometry from a shared descriptor where useful, but do not omit occupancy
   or terminal meaning merely because bounds are predictable. Serialized pointers are not
   valid transport identifiers.
6. A pruned multipole terminal must support the receiver's permitted refinement/evaluation
   path. A coarse sink rejecting it is not permission to treat absent particles as present.
   Resolve it by guaranteed sink refinement, further source detail, or the reference
   fallback. Count fallback use so incomplete exports cannot masquerade as success.
7. Preserve complete source coverage per sink interaction: accept an ancestor OR descend
   into its covering descendants. Parent multipoles may coexist with children in storage,
   but their contributions must never both be accumulated for the same source/sink pair.
   Empty state must be explicit rather than inferred from missing payloads.
8. Start with export views built after owned multipoles are ready. Then permit the user's
   proposed build-time construction: select destination views during descent where enough
   information is known, fill radii/moments on the upward pass, and serialize only after
   completion. Compute each owned source multipole once and reuse it for all destinations.
   Do not rebuild the physical source tree independently for every destination.

## Focused validation, only if requested

Compare export-driven walks with the complete-source reference. Exercise coarse/fine
pairs in both directions, a receiver leaf coarser than an exported source terminal,
periodic seams/image ties, irregular ownership cuts, empty cells, and shared geometric
keys with distinct source masses. Audit coverage on small cases at particle identity
level so matching total mass cannot hide a compensating omission/duplication.

## Completion criteria

Each export is a complete remote source-tree representation for its declared destination
and generation under the specified walk rules. Any conservative extra detail is explicit.
Demonstrate zero missing-detail fallbacks for the targeted correctness cases before
claiming export sufficiency. Do not remove the reference reader yet.

## Source entry points

`cosmictiger/gravity/fmm_acceptance.hpp`, `cosmictiger/gravity/fmm_box.hpp`,
`cosmictiger/gravity/tree.hpp`, `src/gravity/tree_build.cpp`,
`src/gravity/global_fmm/remote_cache.cpp`, `src/gravity/global_fmm/particle_exchange.cpp`,
`cosmictiger/gravity/kick_walk.hpp`, `tests/gravity/global_fmm/fmm_walk.cpp`.

Next: `05_exchange_and_source_forest_walk.md`.
