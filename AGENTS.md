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

# Paper-notation exception

The SPH computation code that transcribes Monaghan (1992), ARA&A 30, 543 names its
local variables with the paper's symbols instead of camelCase: `cosmictiger/sph/monaghan_sph.hpp`
and the pair, density, and support-radius loops in `src/sph/sph_hydro_cpu.cpp`,
`src/sph/sph_hydro.cu`, `src/sph/sph_prehydro_cpu.cpp`, `src/sph/sph_prehydro.cu`, and
`cosmictiger/sph/sph_cuda.hpp`. Those regions use the Unicode letters the compilers accept
(`ρ_a`, `P_b`, `Π_ab`, `μ_ab`, `h_ab`, `gradW̃_ab`, `δt_cv`, `ΣW`) and cite the equation
they implement; function parameters, data fields, and everything outside them keep the
descriptive names above. Fence such regions with
`NOLINTBEGIN/NOLINTEND(readability-identifier-naming)`. GCC 13 and CUDA 12.8 reject `∇`,
`∂`, and superscript digits in identifiers; write `gradW̃_ab`, `r_ab2`, and `η2` instead.
See `docs/monaghan_sph.md` for the symbol table.
