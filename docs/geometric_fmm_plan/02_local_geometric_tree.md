# 2. Deterministic local geometry and cloud radii

Created: 2026-11-14. Status: planned; not implemented by this task.

Recommended model: **GPT-6.1 Terra**. Reasoning effort: **Medium**.

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

Read the completed decisions from `01_geometry_and_exchange_contract.md` in this directory.
This step implements local tree geometry only when explicitly authorized. Keep ownership
migration and remote transport outside this change. Stage the new local builder so the
old multi-rank domain-root assumptions are not silently combined with new unit-box geometry;
step 3 activates independent locality trees.

## Work

1. Introduce the depth-selected split-axis schedule and geometric side-length, center-offset,
   and radius-bound helpers. Prefer constexpr squared-radius data or supported constant
   square roots. Use the fixed schedule, not particle extents or an alternative tie rule.
2. Every new local root starts at [0,1)^3 with depth zero. Preserve input geometric boxes
   for all nodes, including empty nodes and leaves. Use the box midpoint for both stored
   center and every P2M/M2M expansion origin.
3. Partition owned particles at the prescribed midpoint. Continue nonempty subdivision
   until d >= dEwald and N <= bucketSize. Apply the contract's coincident-particle and
   representability terminal cases; no occupied-extent or equal-count fallback split.
   Limit each axis to min(32,P-1) cuts: depth 93 for Fixed32 positions and 96 for
   Fixed64 positions. Box boundaries remain exact Fixed32 ticks at every depth.
4. Measure leaf radii about their stored centers, using position arithmetic that preserves
   relevant native fixed-point differences. Aggregate nonempty child radii with
   min(Rmax(d), max_j(|c_j-c|+r_j)). Use double intermediates with outward rounding
   after integer conversion, products, sums and a direct `std::sqrt` call; preserve
   exactly zero leaf radii. Round conservatively when narrowing to float and retain
   strict floating-point compilation. No extended-precision arithmetic is required.
5. Preserve complete source moments, including dipoles, and existing source precision.
   Distinguish all owned sources required by gravity from the currently active sink subset.
6. Revisit local arena capacity/depth assumptions affected by sparse chains and overflow
   leaves. Do the necessary local support work only; do not redesign allocation generally.
   Remove obsolete particle-box/center selection and update affected flop accounting.

## Focused validation, only if requested

- Verify depth schedule, midpoint boxes and centers over a complete 18-cut cycle.
- Sparse/asymmetric particles must not shrink cells; measured radii must enclose particles.
- One particle, empty branches, coincident overfull particles, and points exactly on splits.
- Radius enclosure and recurrence checked with exact integer squared distances,
  including a radius just above a float boundary by one transverse Q64 tick.
- Deepest supported boxes remain distinct and exact; attempting another split fails.
- Single-locality force/multipole regression with nonzero dipoles about geometric centers.
- Cover both fixed32 and fixed64 when relevant builds are available and authorized.

These checks should distinguish behavioral correctness from simply repeating the helper
implementation. No MPI redesign, broad test suite, benchmark, or GPU runtime work here.

## Completion criteria

The deterministic local builder is usable with a unit-box root, preserves particle mass
and complete multipoles, terminates according to the explicit degeneracy policy, and
maintains conservative radii. State which checks actually ran. Leave multi-rank production
activation to step 3 rather than claiming the old domain-root path is compatible.

## Source entry points

`src/gravity/tree_build.cpp`, `src/gravity/tree_arena.cpp`,
`cosmictiger/gravity/tree_build.hpp`, `cosmictiger/gravity/tree_multipole.hpp`,
`cosmictiger/gravity/tree_source_moments.hpp`, `cosmictiger/gravity/fmm_box.hpp`,
`tests/gravity/global_fmm/fmm_tree.cpp`, `tests/gravity/tree_source_moments.cpp`.

Next: `03_independent_locality_trees.md`.
