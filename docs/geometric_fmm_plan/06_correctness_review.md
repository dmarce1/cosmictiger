# 6. Independent numerical and distributed correctness review

Created: 2026-11-14. Status: planned; not implemented by this task.

Recommended model: **GPT-6 Astra**. Reasoning effort: **High**.

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

Review the completed contract and implementation from steps 1 through 5. This step
is read-only by default. Give actionable findings with file/line references and concrete
counterexamples. Do not silently fix issues or expand into unrelated cleanup. Execute
focused checks only if requested; distinguish source reasoning from runtime evidence.

## Review questions

1. Is every particle owned exactly once? Does every rank root contain only its owned
   sources, while the active sink subset is handled correctly? Can a ghost ever enter
   an owned range, receive a kick, or be integrated?
2. Do identical geometric keys reconstruct identical boxes and centers on every rank?
   Are key width, split-plane ties, Fixed32 boxes, fixed32/fixed64 positions, maximum
   depth, and coincident-particle termination handled without nongeometric splitting?
3. Do actual cloud radii enclose every represented particle after rounding? Are empty
   children excluded, dipoles retained, and all translations centered consistently?
4. For each owned sink, does the source forest cover all owners without omissions or
   double counting? Same-key cells from different owners must both contribute. A stored
   ancestor and its descendants must not both be evaluated for the same source subset.
5. Does the halo proof cover unequal depths and the receiver's actual leaf/terminal
   behavior? Can a multipole-only terminal fail acceptance when neither side can legally
   refine? Are fallbacks visible rather than masking an insufficient export?
6. Do distant owners remain represented when fine halos do not intersect? Are candidate
   peers unbounded by the six face directions? Are occupancy/topology and empty exports
   explicit rather than guessed from geometry?
7. Do periodic images, nearest-image ties, softened near fields, and Ewald corrections
   have complete and nonduplicated coverage? Compare against the same physical operator,
   not a nonperiodic direct sum. Audit the established Newtonian/Ewald coupled paths.
8. Can stale ownership, changed theta/h_G, drifted sources, or reordered messages produce
   a complete-looking but inconsistent forest? Are source buffers live through completion
   and drained on errors? Does the reference cache obey the same generation rules?
9. Does CUDA understand imported source memory separately from ownership and sink locality?
   Do overflow leaves and changed topology remain legal on every supported execution path?
10. Is any claim of momentum conservation supported by the actual pair selection and force
    accumulation scheme, rather than merely by bidirectional communication?

## Evidence to request or inspect

Existing focused results from earlier steps; small source-identity coverage cases;
force/potential error versus an appropriate direct/complete reference; unequal and
non-power-of-two rank cases; empty/coincident cases; periodic seams; changed generations;
CPU and separately verified CUDA results. Expect floating-point summation differences
between decompositions; compare justified error bounds instead of requiring bitwise identity.

## Deliverable and completion criteria

A severity-ordered list of real findings, each with trigger, consequence, and suggested
minimal correction, plus a concise evidence/limitations summary. If no findings, say so
and state what was actually reviewed. Missing runtime evidence is a validation gap, not
proof of a numerical bug. Keep fixes in a separately authorized task or explicitly scoped
follow-up; do not launch step 7 automatically.

## Source entry points

The changed files from steps 1-5, particularly `cosmictiger/gravity/fmm_acceptance.hpp`,
`cosmictiger/gravity/kick_walk.hpp`, `src/gravity/tree_build.cpp`,
`src/gravity/global_fmm/`, `cosmictiger/gravity/tree.hpp`, and their focused tests.

Next, after any correctness fixes: `07_measurement_and_tuning.md`.
