# 5. Asynchronous exchange and imported source-forest traversal

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

Require the independent owned trees/reference path from step 3 and validated export
semantics from step 4. Integrate transport and imported source storage. This step does
not authorize new ownership decomposition or aggregation of different owners' multipoles.

## Work

1. Create separate imported tree storage for each source locality and generation. Rebuild
   local references from transport IDs; keep source ownership distinct from the locality
   where imported memory resides. Imported particle ranges address ghost source storage,
   never the owned sink arrays or original remote arena offsets without translation.
2. Use direct asynchronous peer exchanges with no fixed peer count. The lower locality ID
   may coordinate each unordered pair, as the user proposed. Prepare both directions from
   available export information and allow many pairs to progress concurrently. Do not
   serialize all higher-ID peers or require an avoidable dependent request/tree-search round.
3. Specify readiness and completion, including empty exports and failures. A dynamic list
   of near-detail peers is insufficient to infer complete distant-source coverage. Root
   descriptors/coarse exports must account for every required source owner, even when no
   detailed halo intersects that owner's region.
4. Associate descriptors and payloads with ownership generation, tree/source generation,
   multipole order/representation, and opening/softening settings affecting selection.
   Freeze each walk's source snapshot and reject stale or incomplete data. Drain pending
   work on errors before releasing referenced arenas. Rebuild plans when relevant state changes.
5. Launch the owned sink walk with the owned root plus imported remote roots on the source
   checklist. Maintain both Newtonian and Ewald source handling. Imported trees are source
   only; they have no kick return, time integration, or sink-side force storage.
6. Retain the step 3 reference reader as an explicitly counted development fallback for
   missing detail. Missing descendants of a multipole terminal cannot become a direct
   particle interaction. Completion claims must distinguish a fully supplied exchange
   from a walk rescued by demand fetching.
7. Stage the same separate source forest for CUDA without grafting it into the owned
   tree. Audit pointer/index remapping, ghost mass/position payloads, terminal kinds, and
   source-owner assumptions in GPU work batching. CPU success is not GPU validation.
8. Keep communication/data movement distinct from mutual force evaluation: a two-way
   exchange does not by itself make the two directional walks one momentum-conserving
   mutual interaction. Preserve established symmetric selection rules where applicable.

## Focused validation, only if requested

Compare exchanged forests to step 3's reference for 1/2/non-power-of-two localities,
uneven loads, an empty rank, delayed/out-of-order arrivals, and a rebuild after migration
or opening-angle change. Include missing/stale export diagnostics, exactly-once sink
completion, periodic Newtonian/Ewald agreement, and zero expected fallback requests.
Run focused CUDA parity/staging checks only if separately authorized and within the
current shared GPU budget. No QB access without an explicit task login budget.

## Completion criteria

Remote source trees remain distinct, all source owners are covered, generation readiness
is explicit, and only local owned sinks are updated. Report CPU, MPI, CUDA, and fallback
validation separately. Preserve the reference path through the independent review.

## Source entry points

`src/gravity/global_fmm/execution.cpp`, `src/gravity/global_fmm/remote_cache.cpp`,
`src/gravity/global_fmm/particle_exchange.cpp`, `src/gravity/global_fmm/gpu_staging.cpp`,
`src/gravity/source_access.cpp`, `cosmictiger/gravity/kick_walk.hpp`,
`cosmictiger/gravity/kick_cuda_kernel.hpp`, `cosmictiger/gravity/kick_workspace.hpp`,
`cosmictiger/gravity/source_access.hpp`, `tests/gravity/global_fmm/fmm_walk.cpp`.

Next: `06_correctness_review.md`.
