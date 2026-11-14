# Geometric FMM source-forest plan

Created: 2026-11-14. This task wrote planning documents only. No source changes,
builds, tests, benchmarks, commits, pushes, or remote logins were performed.

The files below are self-contained handoffs for separate chats. Each includes the
working agreement, agreed architecture, scope, prerequisites, source entry points,
and completion criteria. Execute steps sequentially; a file's existence does not mean
its work is complete. Check the preceding result in the actual checkout before starting.

## Steps and model settings

| Step | File | Recommended model | Reasoning |
|---|---|---|---|
| 1 | [Geometry and exchange contract](01_geometry_and_exchange_contract.md) | GPT-6 Astra | High |
| 2 | [Deterministic local geometry and cloud radii](02_local_geometric_tree.md) | GPT-6.1 Terra | Medium |
| 3 | [Independent locality trees and reference traversal](03_independent_locality_trees.md) | GPT-6.1 Sol | High |
| 4 | [Destination-specific remote source-tree exports](04_remote_source_tree_exports.md) | GPT-6.1 Sol | High |
| 5 | [Asynchronous exchange and imported source-forest traversal](05_exchange_and_source_forest_walk.md) | GPT-6.1 Sol | High |
| 6 | [Independent numerical and distributed correctness review](06_correctness_review.md) | GPT-6 Astra | High |
| 7 | [Measurement and targeted tuning](07_measurement_and_tuning.md) | GPT-6.1 Sol | Medium |

Astra handles the mathematical contract and independent correctness review; Terra
handles the scoped local geometry change; Sol handles distributed integration and
measurement. These are task-specific recommendations, not measured CosmicTiger model
comparisons. Start at the listed effort; use Extra high only for an unresolved hard
reasoning problem. No Max/Pro mode is assumed.

[Official model-selection guidance](https://developers.openai.com/api/docs/guides/model-selection)
and [reasoning-effort guidance](https://developers.openai.com/api/docs/guides/deployment-checklist)
support the general model/effort distinctions. Recommendations were checked on 2026-11-14;
confirm model availability in the model picker when starting a later chat.

## Agreed design

1. Recursive spatial cuts continue to determine particle ownership. They do not define
   the FMM cell boundaries.
2. Each locality builds an independent kd-tree rooted at the unit box, from its own
   source particles. Localities can have different occupied/adaptive structures.
3. Cells follow the midpoint schedule `xyz | zyx | zxy | yxz | yzx | xzy`, repeated by
   depth. Boxes are never shrunk to particles. Source and local-expansion centers are
   geometric cell centers; retain dipole moments.
4. The geometric upper radius is constexpr by depth. Actual leaf cloud radii are maximum
   particle distances; parent radii are capped child-cloud enclosures. This supersedes
   the earlier chat-only snippet that assigned every actual radius the half-diagonal.
5. Keep one separate pruned imported source tree per remote source owner. Matching local
   and remote cells have the same geometry but different particles and partial multipoles.
   Never merge these imports into the owned sink tree or deduplicate them by geometric key.
6. Only the owned tree supplies sinks. Start source traversal with its root plus remote
   source roots. Fine halos govern needed detail; distant sources still require coarse
   coverage. There is no six-remote cap.
7. Export selection can be prepared during the sender's own build when destination
   information is available, then populated from the upward pass. Reuse computed moments
   rather than independently building a physical source tree for each recipient.
8. Use asynchronous direct exchanges. A lower-ID initiator can coordinate an unordered
   pair without serializing pairs. Shared geometry does not eliminate occupancy/terminal
   metadata, mixed-depth handling, or periodic/Ewald coverage requirements.

## Suggested prompts for a new chat

For step 1, design only:

> Read `/home/dmarce1/workspace/cosmictiger/docs/geometric_fmm_plan/01_geometry_and_exchange_contract.md`.
> Work through this step and give me the contract in chat. Do not implement or run builds/tests.

For implementation, explicitly choose the file and execution scope, for example:

> Read `/home/dmarce1/workspace/cosmictiger/docs/geometric_fmm_plan/02_local_geometric_tree.md`.
> Implement only this step in the repository. Preserve unrelated work. Do not run builds,
> tests, or benchmarks yet. Report the changes and any unresolved issues.

If you also want the listed focused checks, say so and specify the shared resource budget.
For step 6, request a read-only review; for step 7, first request a measurement proposal,
then explicitly authorize the chosen runs. These prompts are examples, not current authorization.

## Dependency and validation sequence

Contract -> local geometry -> independent roots/reference traversal -> export selection
-> asynchronous imported source forest -> independent review -> measurement/tuning.

The reference traversal is established before pruning. Initially it may fetch complete
logical remote trees on demand. Later, exports are checked against it; fallback requests
are counted so an incomplete halo cannot hide behind successful demand fetching.
Preserve the existing Newtonian/Ewald operator when making those comparisons.

All proposed tests and measurements are future work. No performance improvement or
CPU/MPI/CUDA correctness result is claimed by this plan.
