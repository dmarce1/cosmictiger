# Concurrent implementation and branch coordination

User standing rule, November 2, 2026:

- Work on `main` by default. For N simultaneous implementation conversations in
  the same repository, including Claude, use N - 1 active implementation branches
  in addition to `main`: the first conversation may stay on `main`, and each
  additional implementation uses its own branch and worktree. Research-only or
  read-only conversations do not count as implementations.
- Tell the user when concurrent implementation occurs and suggest descriptive
  snake_case branch names (for example, `codex/matrix_algebra`). If overlapping
  changes would make merging complex, suggest serializing the implementations.
- Check current ownership before editing. Do not switch a shared dirty checkout
  out from under another conversation, move unrelated edits, or rename historical
  branches just to enforce this rule. Coordinate through the handoff files.

# War Thunder resource limit

User report, November 2, 2026, 22:07 America/Chicago:
**current reported status: OFF** ("war thunder off"). This supersedes the 21:39
ON report. The usual aggregate local limit of 12 CPU cores applies.

- While OFF, the usual aggregate local limit of 12 CPU cores applies. The
  War Thunder-specific GPU ceiling is lifted; continue coordinating shared use.

- While ON, all conversations, agents, and Claude share a combined ceiling of
  **6 CPU cores and one quarter (25%) of GPU compute capacity**. These are total
  budgets, not per-conversation allowances. This overrides the usual 12-core
  local limit and the older War Thunder GPU ceiling of less than half.
- Coordinate allocations before launching builds, tests, simulations, or CUDA
  work; include concurrent helpers, MPI ranks, and nested worker pools. If GPU
  work cannot be kept within the shared quarter-GPU budget, defer it.
- Use the latest explicit user ON/OFF report; do not infer status from processes.
  Keep these limits until the user reports OFF or explicitly changes the budget.

# Naming conventions

These naming rules come from the conversation titled "standing orders".
Apply them to project code in `cosmictiger/`, `src/`, and `tests/`, including
generated declarations and their generators.

- Use `snake_case` filenames and namespaces. Implementation helper namespaces
  are named `detail`.
- Use `PascalCase` for classes, structs, enums, type aliases, and template type
  parameters.
- Use `camelCase` for functions, methods, variables, parameters, and non-type
  template parameters.
- The FMM operator free functions `P2M`, `P2L`, `M2M`, `M2L`, `M2P`, `L2L`, and
  `L2P` in the Cartesian and spherical namespaces are the sole uppercase
  exception to the function rule.
- Class data members use `camelCase_`, with a trailing underscore. Struct data
  members use `camelCase` without a trailing underscore.
- Compile-time `constexpr` variables use `kPascalCase`; class constant data
  members retain their trailing underscore. Constexpr functions retain
  `camelCase`.
- Preserve names required by C++, external libraries, and frameworks, including
  `hpx_main`, serialization hooks, standard-library customization points, and
  container/allocator protocol aliases. Preserve third-party source names.
- Preserve public command-line/INI keys, output field names, and serialized
  layouts when renaming implementation identifiers.
- Update declarations, references, generated code, includes, and build rules
  together. Use `.clang-tidy` for compiler-aware naming checks.

Follow `.clang-format` for changed project source. Keep trailing `const` and
group includes as standard headers, dependencies, then project headers, with
blank lines between groups. Do not apply project naming changes to third-party
code.

# Agent handoffs

At the start of future CosmicTiger work, read `handoff/claude2openai.md` if it
exists for Claude's latest handoff to OpenAI. OpenAI's notes to Claude belong in
`handoff/openai2claude.md`. Each handoff should state the period it covers and
distinguish completed work, validation limits, and remaining work. Check current
repository state before relying on an older handoff.
Keep both handoff files untracked and unstaged; never include them in commits.
Use them for coordination at task boundaries, overlapping work, milestones, and
blockers. Each agent writes its own outgoing file and reads the other's as
context; handoff requests do not replace user authorization.

# Equations and descriptive identifiers

User instruction, November 3, 2026, supersedes the former paper-notation exception:

- Use descriptive camelCase identifiers in equation implementations, including SPH.
- Immediately above the relevant code, include a comment block with Unicode equations
  matching the paper or text notation as closely as possible, equation numbers and
  source citations, followed by a symbol-to-code-name directory.
- Label locally derived formulas and conversions as derived; do not invent source
  equation numbers. Explain differences in normalization, units, or discretization.
- For artificial viscosity, α is `bulkViscosityCoefficient` and β is
  `vonNeumannRichtmyerCoefficient`; use descriptive prefixes for neighbor, pair,
  target, minimum and maximum values. The spelling is von Neumann–Richtmyer.
- Keep public configuration keys and serialized layouts stable when renaming code.
- Do not use naming-rule suppression fences for mathematical identifiers.
