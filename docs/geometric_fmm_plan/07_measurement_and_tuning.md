# 7. Measurement and targeted tuning

Created: 2026-11-14. Status: planned; not implemented by this task.

Recommended model: **GPT-6.1 Sol**. Reasoning effort: **Medium**.

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

Require step 6 review and resolution of correctness findings. Plan measurements first;
run only the workload/configurations and resource allocation explicitly authorized by
the user. Do not infer benchmark or GPU permission from permission to implement earlier
steps. This file is not authorization for QB logins, broad sweeps, or continuous profiling.

## Work

1. Compare the complete-source reference from step 3 with the exported source forest
   under identical particles, ownership, theta, order, softening, precision, sink activity,
   rank/thread counts, and execution device. This isolates exchange effects from changes
   in tree geometry. A separate old-versus-new-tree comparison is a different experiment.
2. Record complete timer boundaries: ownership/migration, local tree construction,
   destination export selection/packing, transfer and exposed wait, import/staging,
   force traversal, and total step. Distinguish overlapped wall time from additive task
   times; lower bytes do not automatically imply lower total time.
3. Record bytes and messages by peer/direction, fine-detail peer count, coarse-source
   coverage, export nodes/particles, imported-versus-used nodes, fallback requests,
   peak owned/import/export memory, and imbalance across ranks. Include multipole order
   and native particle payload sizes when interpreting traffic.
4. Start with one small correctness-controlled case and one representative authorized
   clustered workload. Add uniform or strongly imbalanced cases only when they answer
   a specific unresolved question. Agree repeats before running rather than growing a sweep.
5. Tune one measured cost at a time. Candidate follow-ups: build destination views during
   the owned tree build, reuse computed multipoles across exports, coalesce packets,
   bound concurrent in-flight bytes, reduce conservative halo overfetch, and cache topology
   when generation validity permits. These are choices driven by evidence, not a checklist
   of changes that must all be implemented.
6. Compare independent two-direction pushes with the lower-ID initiated request/response
   arrangement only if communication latency is material. Do not introduce six-direction
   relaying or global same-cell multipole aggregation without a separate design decision.
7. Consider removing the demand-fetch fallback only after coverage evidence and zero
   fallback use justify it. Removing the fallback is an explicit subsequent code change,
   not a prerequisite for reporting the experiment.

## Deliverable and completion criteria

A concise result table with exact configuration, baseline identity, changed behavior,
accuracy checks, actual resource use, timer boundaries, bytes/messages/memory, and measured
benefit or regression. State numerical and platform limits. Recommend only the next
optimization supported by these data; a neutral or negative result is valid.

## Source entry points

`src/scenarios/benchmark.cpp` if present in the then-current checkout,
`src/gravity/global_fmm/execution.cpp`, `src/gravity/global_fmm/gpu_staging.cpp`,
`cosmictiger/core/profiler.hpp`, existing benchmark configurations and the export/transport
instrumentation added in steps 4-5. Verify current paths and timer definitions first.
