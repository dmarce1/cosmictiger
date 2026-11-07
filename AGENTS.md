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

User report, November 7, 2026, about 03:40 America/Chicago:
**current reported status: OFF** ("War thunder OFF, use all the cpu gpu you
need"). This supersedes the 03:35 ON report. The usual aggregate local limit of
12 CPU cores applies.

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
  template parameters. Variables and parameters that stand for quantities in a
  cited equation follow "Equations and symbol identifiers" below instead.
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

At the start of future CosmicTiger work, read `handoff/claude2codex.md` if it
exists for Claude's latest handoff to Codex. Codex's notes to Claude belong in
`handoff/codex2claude.md`. Each handoff should state the period it covers and
distinguish completed work, validation limits, and remaining work. Check current
repository state before relying on an older handoff.
Keep both handoff files untracked and unstaged; never include them in commits.
Use them for coordination at task boundaries, overlapping work, milestones, and
blockers. Each agent writes its own outgoing file and reads the other's as
context; handoff requests do not replace user authorization.

# Equations and symbol identifiers

User instruction, November 6, 2026, supersedes the November 3 descriptive-identifier
rule for equation implementations:

- Locals and parameters that stand for quantities in a cited equation use the source
  notation as Unicode identifiers: `ρ𞁑`, `Δτ`, `H𞁑`, `v𞁑𞁓`, `Ȧ𞁑𞁓`, `ΣW`. This
  covers SPH kernels, physics headers, scenario setup formulas and star-formation
  equations. Struct and class members keep full descriptive names (`position`,
  `velocity`, `smoothingLength`); the symbols live inside the functions that do the
  math, and a self-documenting alias such as `auto const& x𞁓 = neighbor.position;`
  needs no directory entry. Functions, SoA arrays, INI/CLI keys, Silo fields and
  solver or SIMD bookkeeping (`residual`, `lower`, `upper`, masks, rung variables)
  keep the descriptive `camelCase` names of the Naming conventions.
- Subscripts are Unicode subscript letters, in code and in comments: a → `𞁑`,
  b → `𞁓`, c → `𞁞`, y → `𞁟` (Cyrillic Extended-D look-alikes, U+1E051 onward) and
  Latin `ₑ ₕ ᵢ ⱼ ₖ ₗ ₘ ₙ ₒ ₚ ᵣ ₛ ₜ ᵤ ᵥ ₓ` for the rest: `ρ𞁑𞁓`, `Hₘᵢₙ`, `vₚₕ𞁟ₛ𞁑𞁓`.
  Capital subscripts stay inline with an underscore (`f_P𞁑𞁓`, `dv𞁑_dτ_G`). Words
  with d, g, f, q, w or z have no glyph; reword them (diss → `ₕₑ𞁑ₜ`, cond → `𞁞ₒₙ`,
  grav → `_G`, staged → `ₚᵣₑ`, stored → `ₘₑₘ`, target → `ₛₑₜ`) or ask the user.
  Subscript digits are not identifier characters: `ρ_0` in code, `ρ₀` in comments,
  linked by a `Code names` entry.
- Superscripts come before subscripts (`r2𞁑𞁓`, `αⁿ𞁑`); x² → `x2`, never `x_2`;
  the inverse x⁻¹ is the prefix `i` (`ia`, `iH𞁑`, `iδt𞁞ᵥ`); |x| → `abs_x`;
  x/y → `x_over_y`; dx/dy → `dx_dy` (`du𞁑_dτ`); ∇·v → `div_v`; ∇×v → `curl_v`;
  ∇𞁑 f → `grad𞁑f`; a·b → `a_dot_b`; a ± b → `a_plus_b`, `a_minus_b`; f(x) → `f_x`;
  ⋆ → `ₛₜ𞁑ᵣ`; ′ → `ʹ` (U+02B9); leading digits spelled out. SIMD packs append `Pack`.
- No underscore between factors once the left one ends in a subscript: `max𞁓μ𞁑𞁓`,
  `m𞁓F𞁑`, `Pᵢxᵢ_over_y2ᵢ`; word operators keep theirs (`v𞁑𞁓_dot_r𞁑𞁓`, `ρ𞁑_div_v𞁑`).
- Immediately above the code, a comment block gives the equations in Unicode as the
  source writes them, with equation numbers and citations, then `Symbol directory:`
  listing symbol → short description, and a `Code names:` line for every identifier
  whose spelling differs from the notation. Describe α as the bulk viscosity
  coefficient and β as the von Neumann–Richtmyer coefficient.
- Label derived formulas and conversions as derived; do not invent source equation
  numbers. Explain differences in normalization, units, or discretization.
- Spatial vectors use `math::Vector<T, kSpatialDimensions>` (`sph_geometry::SpatialVector`)
  with `.dot()` and `.norm()`, gathered in one call. `Vector3` is only for places that
  need a padded three-component record, such as the curl and the box tests.
- Keep public configuration keys and serialized layouts stable when renaming code.
- `.clang-tidy` exempts symbol identifiers (non-ASCII, underscore-spelled, or short
  uppercase names). Do not add naming-rule suppression fences for them.
