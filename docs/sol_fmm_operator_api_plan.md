# FMM type and operator API implementation plan for Sol

Implement a common, typed FMM API for Cartesian and spherical kernels, covering
application FMM orders 3 through 9. The user's final naming decision is to use
the existing `cosmictiger::gravity::cartesian` and
`cosmictiger::gravity::spherical` namespaces. Each namespace provides
`Multipole`, `Expansion`, and the uppercase free functions `P2M`, `P2L`, `M2M`,
`M2L`, `M2P`, `L2L`, and `L2P`. This supersedes the earlier proposal to prefix
the types with `Cartesian` or `Spherical`.

Both representations must return potential and acceleration consistently:
`acceleration = -gradient(potential)`. Use the same point-result component
order, `[potential, accelerationX, accelerationY, accelerationZ]`.

This is a planning handoff dated 2026-10-26. It does not implement the API or
establish new build or runtime validation results.

## Starting state and scope

The inspected checkout is `/home/dmarce1/workspace/cosmictiger`, branch `main`,
HEAD `ca6eb17e3c32f0c5acbedbb6f36bfc53ec202ff0`. It has extensive staged,
unstaged, and untracked work, including ongoing changes to the FMM generators,
CUDA annotations, CMake, and generated source headers. Reinspect that state
before implementation and preserve unrelated changes. Do not reset, clean the
shared checkout, or stage everything.

The source-header migration is already present in the working tree: generated
headers live under `cosmictiger/gravity/`; CMake regeneration is explicit; new
configurations default to orders 3–9. Preserve that behavior and any further
work completed in the main conversation. Do not resume or manage that
conversation's running commands from this handoff.

The scope is types, operator interfaces, generated declarations, call sites,
and tests. Preserve the numerical algorithms, expansion truncation, periodic
background treatment, force scaling, serialized layouts, tree geometry, and
interaction-selection policy. The production tree walk remains Cartesian;
this refactor does not add a runtime representation selector. Spherical PMMM
integration remains supported through its existing paths.

Read the applicable AGENTS.md files. Uppercase FMM function names are an
explicit user exception to the general camelCase rule. Add only these exact
names to the clang-tidy naming exception; preserve its existing exceptions.
Keep trailing const, braced loops, unqualified `size_t`, include groups, and
snake_case branch components. Local builds and tests share a maximum of 12
cores with other work. No benchmark or QB login is authorized by this note.

## Public types and order convention

Use the application FMM order as the integer template parameter everywhere in
the public API: `Multipole<Real, order>` and `Expansion<Real, order>`. For order
`q`, multipoles have degree `q - 2` and local expansions have degree `q - 1`.
Support `3 <= q <= 9` independently of a particular build's selected runtime
subset. Keep degree-based implementation helpers private.

| Current type or role | Public replacement |
| --- | --- |
| `FmmMultipole<Real, q>` and production Cartesian moments | `cartesian::Multipole<Real, q>` |
| `FmmExpansion<Real, q>` and production Cartesian locals | `cartesian::Expansion<Real, q>` |
| `spherical::Spherical<Real, q - 2>` used as moments | `spherical::Multipole<Real, q>` |
| `spherical::Spherical<Real, q - 1>` used as locals | `spherical::Expansion<Real, q>` |
| Cartesian or spherical Green coefficient arrays | `cartesian::Green<Real, q>` or `spherical::Green<Real, q>` |
| `Expansion2<Real>` and spherical four-component point results | Shared `cosmictiger::gravity::Field<Real>` |

`cartesian` and `spherical` above abbreviate the full namespaces under
`cosmictiger::gravity`. `Green` distinguishes precomputed kernel coefficients
from an accumulated local expansion; it does not change their mathematical
normalization. `Field` is representation-independent because potential and
acceleration are physical outputs, not expansion coefficients.

Make `Multipole<Real, order>` and `Expansion<Real, order>` distinct class
templates with `order` directly in their type identities. Merely renaming
aliases such as `using Multipole = Storage<Real, order - 2>` creates deduction
problems: C++ cannot infer `order` by solving `order - 2`. Two aliases of the
same underlying storage type also do not create distinct semantic types.
Composition or storage-only base classes are implementation choices, provided
they preserve the required layout and interfaces.

Preserve Cartesian coefficient ordering and storage sizes: `(q - 1)^2`
multipole coefficients and `q^2 + 1` local coefficients, including the existing
local trace slot. Preserve spherical harmonic coefficient ordering and the
existing separate periodic trace values. Retain required indexing, data
access, scalar assignment, copy, and serialization behavior. Assert size,
alignment, and the layout properties actually required by the transport and
CUDA code for every order.

Raw degree-indexed storage can remain in each representation's `detail`
namespace for generators, reference tests, and special contractions. Do not
introduce implicit Cartesian-to-spherical or multipole-to-expansion conversions.
Keep type-only headers independent of the large generated arithmetic headers.

## Operator interfaces and lookup

Declare the same function names in each representation's namespace. A typed
destination or source selects the correct function by normal overload
resolution and argument-dependent lookup. Qualified calls remain available.
Do not add another set of forwarding overloads in the parent namespace or
global namespace, and do not rely on `using namespace` directives.

The intended call style is:

```cpp
namespace cartesian = cosmictiger::gravity::cartesian;
namespace spherical = cosmictiger::gravity::spherical;

cartesian::Multipole<double, 7> cartesianMoments{};
spherical::Multipole<double, 7> sphericalMoments{};
cartesian::Expansion<double, 7> cartesianLocal{};
spherical::Expansion<double, 7> sphericalLocal{};
cosmictiger::gravity::Field<double> cartesianField{};
cosmictiger::gravity::Field<double> sphericalField{};

P2M(cartesianMoments, mass, particleOffset);
P2M(sphericalMoments, mass, particleOffset);
M2L(cartesianLocal, cartesianMoments, separation, doPotential);
M2L(sphericalLocal, sphericalMoments, separation, doPotential);
M2P(cartesianField, cartesianMoments, separation, doPotential);
M2P(sphericalField, sphericalMoments, separation, doPotential);
```

These are interface examples; `mass`, the offsets, and `doPotential` are caller
inputs. A representative declaration in each namespace is:

```cpp
template <class Real, int order>
CUDA_EXPORT int M2L(Expansion<Real, order> &target,
    Multipole<Real, order> const &source,
    Array<Real, NDIM> const &displacement, bool doPotential = true);
```

Use destination-first, additive operations consistently. A zero-initialized
destination requests a fresh result. `doPotential == false` leaves the
destination's potential coefficient unchanged while accumulating the requested
force or derivative components. Keep source and destination non-aliasing;
adapt existing assignments such as `expansion = l2L(expansion, ...)` through a
separate result before assignment. Do not accidentally translate and accumulate
an expansion into itself.

Handle the old wrappers' exceptions explicitly during migration. The current
`l2L` copies the source potential into its returned result when `doPhi` is
false; seed that value in the separate destination before an additive `L2L`
call where the caller relies on this behavior. The current `greensFunction`
overwrites its destination for unit source mass; its `P2L` replacement must
start from zero and pass mass one. Preserve ordinary additive `P2L` behavior
for other callers.

| Function | Destination | Source and additional inputs |
| --- | --- | --- |
| `P2M` | `Multipole` | Mass and particle offset from source center |
| `P2L` | `Expansion` | Mass and target-center-minus-particle displacement |
| `M2M` | `Multipole` | Same-order `Multipole`, old-center-minus-new-center displacement |
| `M2L` | `Expansion` | Same-order `Multipole`, target-minus-source displacement |
| `M2L` with precomputed coefficients | `Expansion` | Same-order `Multipole` and representation-matched `Green` |
| `M2P` | Shared `Field` | `Multipole`, target-point-minus-source-center displacement |
| `L2L` | `Expansion` | Same-order `Expansion`, new-center-minus-old-center displacement |
| `L2P` | Shared `Field` | `Expansion`, evaluation-point-minus-expansion-center displacement |

Give `P2M` and `P2L` an explicit typed destination: return type alone cannot
choose between Cartesian and spherical functions with otherwise identical
parameters. Reject mixed representations and mismatched orders at compile
time. Preserve centered-source optimizations as explicit internal or constrained
options; their zero-dipole precondition must not be inferred from a type rename.

Return the conventional per-call operation count where the application needs
it; callers may discard the return value. Keep weighted optimization costs
separate from conventional FLOP accounting. Small cost traits may replace
public `XxxKernel::kFlops_` references without introducing runtime dispatch.

## Units and the common point-result convention

Use potential and acceleration in both representations, with named accessors
on `Field` and the common `[potential, ax, ay, az]` coefficient order. There must
be no caller-visible gradient variant of this result.

The current raw Cartesian M2P and L2P kernels already produce acceleration.
The wrappers in `fmm_kernels.hpp` negate and rearrange it into `Expansion2`,
whose Cartesian first-degree entries are gradients. CPU and CUDA consumers
then subtract those gradients. Remove that intermediate gradient convention
and change the consuming subtraction to addition in the same migration.
Audit every `Expansion2` consumer, including SIMD reductions, force planes,
test references, and potential-disabled paths.

Preserve the application's existing unit conversion: normalized potential is
multiplied by `SCALE_FACTOR1`, and normalized acceleration by `SCALE_FACTOR2`
at the existing application boundary. Do not apply either scaling twice or
remove it when removing the sign flip. Check equivalence of the old gradient
path and the new acceleration path component by component.

For the common numerical API, use displacements in the same coordinates as
the input moments and expansions. Keep the application's displacement scaling
in a small explicit adapter at the boundary; both representations then receive
the same coordinate convention. Preserve PMMM's cell-coordinate conversions
and rank-dependent Green scaling separately. Cartesian and spherical
coefficients have different normalizations and must never be copied between
representations as if they were interchangeable.

## Periodic operations and CUDA specialization

Preserve spherical `sourceTrace`, `localTrace`, and `greenTrace` as explicit
sidecar arguments in periodic overloads. Rename the operations to `M2L`,
`L2L`, and `L2P` while retaining those additional arguments. Free-space and
periodic overloads must be distinguishable by their complete typed signatures.
This avoids changing the packed PMMM buffers merely to introduce new names.

The spherical source trace is the independent radial second moment, not a
quantity recoverable from harmonic coefficients. The positive local trace
represents `-localTrace * |offset|^2 / 2`; it is not numerically interchangeable
with the Cartesian local Laplacian slot. Preserve its existing contribution to
potential, acceleration, and translations, and the P2M/M2M sidecar updates.
Do not use the free-space axial-rotation shortcut for general periodic Green
contractions. Green evaluation remains an explicitly named helper, such as
`evaluateEwaldGreen`, separate from applying `M2L`.

Keep CUDA host/device annotations on the public functions and their types.
The cooperative Cartesian L2L path has different participation and aliasing
requirements from the scalar path. Preserve it as an internal implementation
or expose an explicitly tagged `L2L` overload; do not silently replace a
warp-cooperative call with a scalar overload or force a raw shared-memory
pointer to masquerade as a newly introduced owning class.

## Generation and source organization

Keep mathematical candidate selection inside the Python generator. Production
headers should expose only the selected operation, not `M2pVariant` and the
other public variant hierarchies. Preserve all candidate counts and selected
strategies in `operation_counts.json`. If numerical tests still compare all
variants, emit them into a test-only output directory or header. Preserve any
equivalent simplification already completed in the main conversation.

Use thin public functions over private generated bodies where necessary to
preserve the existing factoring, FMA structure, and normalization. Generated
classes or helper functions may remain private implementation details during
migration; ordinary application code must stop referring to `M2pKernel`,
`M2lKernel`, and their peers. Do not add runtime polymorphism or representation
switches to these calls.

Recommended organization, with paths relative to the repository root:

| Files | Work |
| --- | --- |
| New `cosmictiger/gravity/fmm_kernels/cartesian_types.hpp` and `spherical_types.hpp` | Representation-specific `Multipole`, `Expansion`, `Green`, and private storage; no generated arithmetic includes |
| `cosmictiger/gravity/fmm_types.hpp` | Shared `Field` and existing application transport structures; include the small type headers and migrate their views |
| `cosmictiger/gravity/fmm_kernels/cartesian_operators.hpp` and a new `spherical_operators.hpp` | Typed uppercase public functions; preserve a Cartesian-only include path so Cartesian translation units need not parse spherical arithmetic |
| `cosmictiger/gravity/fmm_kernels/cartesian_ewald_operators.hpp` | Typed Cartesian periodic overloads and Green adapters |
| `tools/generate_fmm.py`, `tools/cartesian/*.py`, `tools/fmm/spherical*.py` | Emit declarations against the new types and private implementation layout; preserve orders 3–9 and metadata |
| Generated headers under `cosmictiger/gravity/fmm_kernels/` | Regenerate from the updated generators; never patch emitted arithmetic by hand |
| `CMakeLists.txt`, `.clang-tidy`, `AGENTS.md` | Register headers and tests; document the narrow uppercase FMM naming exception; retain explicit-only regeneration |

Do not move the generated headers back into the build tree or register source
headers as CMake-generated outputs that clean targets can remove. The generator
CLI's `--min-order 2 --max-order 8` still denotes local degrees and produces
application FMM orders 3–9. Update documentation where it still says 3–8.

## Implementation sequence

1. Record the current dirty state and generator/header provenance. Establish
   the current numerical baseline before changing behavior; do not assume
   unfinished parent-session validation succeeded.
2. Introduce the distinct namespace-scoped classes and shared acceleration
   `Field`. Add focused deduction, layout, and rejected-call checks before
   converting all callers. Preserve the existing transport storage and
   serialization contract.
3. Add uppercase overloads for both representations with one documented
   additive, order, unit, and sign contract. Keep old wrappers temporarily
   only if needed to make the migration reviewable; do not leave two public
   APIs at completion.
4. Update the generators and emitted declarations. Keep selected arithmetic
   and operation-count provenance intact; isolate test-only candidates.
5. Migrate CPU and CUDA application calls, result consumers, Ewald paths,
   PMMM adapters, instantiation templates, and tests. Audit explicit template
   arguments that currently denote degree rather than application order.
6. Remove obsolete public aliases, lowercase operator wrappers, and public
   variant/kernel classes. Update docs and run the validation matrix below.

Important call-site groups include `tree_multipole.hpp`, `gravity_cpu.hpp`,
`gravity_cuda.hpp`, `kick_walk.hpp`, `kick_cuda_kernel.hpp`, `kick_cuda.hpp`,
`dehnen/periodic_cpu.hpp`, `dehnen/periodic_cuda.hpp`, CPU instance headers,
and `src/gravity/fmm_instance_*.in`. Check `src/gravity/nitadori/kernels.cpp`,
`src/main.cpp`, `tests/sph/gravity.cpp`, and FMM tests as well.

The `ExpansionBuffer`, `MultipoleBuffer`, and positioned multipole records use
layout-dependent views. Equal `sizeof` alone does not establish valid C++
object lifetime or aliasing. Review those views when introducing classes;
use explicit typed views or valid object construction where needed instead of
adding new unchecked casts. Preserve checkpoint bytes and CUDA staging strides.
PMMM's raw arrays should use typed non-owning views or internal adapters, not
reinterpret casts to newly invented owning objects.

## Validation and completion criteria

The following checks are implementation requirements, not results from this
planning conversation. Run focused correctness tests before any broader work.

| Area | Required evidence |
| --- | --- |
| API and deduction | Both namespaces compile qualified and unqualified calls for all seven operations at every order 3–9; `P2M` deduces through its destination; wrong representation/order combinations are rejected |
| Point fields | Both return potential and acceleration; a positive source on the positive x-axis accelerates a target at the origin in positive x; repeat for other axes and general positions, M2P and L2P |
| Numerical equivalence | Compare both representations to the existing independent references with float, double, and CPU SIMD; compare physical fields, not raw coefficients; account for existing finite-order truncation conventions |
| Accumulation and potential flags | Nonzero destination initialization, repeated additions, `doPotential` on/off, source immutability, and translation through separate destinations |
| Periodic behavior | Nonzero dipoles and radial traces, background quadratic terms, minimum-image limits, origin/near-origin Ewald evaluation, trace-aware L2L/L2P, and PMMM scaling |
| Layout and transport | Sizes/alignments/strides for orders 3–9, valid views, fixed32/fixed64 call paths, serialization/checkpoint compatibility |
| Generator and costs | `generate_fmm.py --check`, independent executed arithmetic checks, preserved candidate metadata, conventional versus weighted count separation |
| Build integration | Explicit regeneration works; ordinary builds do not rewrite headers; clean of a disposable validation build preserves source headers |
| CUDA | Compile both representations' public API, run scalar parity and Cartesian cooperative L2L checks, and preserve lane participation and shared-memory layout |
| Application | Focused tree/walk, Ewald, PMMM, and changed checkpoint tests; confirm the interaction policy and force results remain consistent |

Existing useful targets are `cartesian_kernels_test`,
`generated_spherical_order_3_test` through `generated_spherical_order_9_test`,
`generated_spherical_ewald_test`, `cartesian_ewald_cpu_test`, `fmm_flops_test`,
`pmmm_kernels_test`, `cartesian_kernels_cuda_test`, and
`cartesian_l2l_cuda_test`. The generator checks are `generated_fmm_current`
and `generated_fmm_flops`. Application CTests include `fmm_tree`,
`fmm_tree_fixed64`, and `fmm_walk_cpu`/`fmm_walk_cuda`; explicitly cover the
required order matrix rather than assuming a default-order invocation does so.
Add a small spherical CUDA API/parity check if the existing targets do not
exercise the new spherical overloads. Distinguish CPU, CUDA compile, CUDA
runtime, and MPI evidence in the completion report.

Completion means the new names and common point-field convention are used
throughout the supported call paths, both representations cover orders 3–9,
generated headers reproduce, and the validation above supports the refactor.
Report any remaining validation gaps explicitly. Deliver an owned-file diff
and concise results without committing or pushing unrelated work.

## Source references

- [Current Cartesian types and transport buffers](/home/dmarce1/workspace/cosmictiger/cosmictiger/gravity/fmm_types.hpp)
- [Current Cartesian wrappers and sign conversions](/home/dmarce1/workspace/cosmictiger/cosmictiger/gravity/fmm_kernels/cartesian_operators.hpp)
- [Order and degree definitions](/home/dmarce1/workspace/cosmictiger/cosmictiger/gravity/fmm_order.hpp)
- [Cartesian periodic contraction and background](/home/dmarce1/workspace/cosmictiger/cosmictiger/gravity/fmm_kernels/cartesian_ewald_operators.hpp)
- [Spherical generator and candidate selection](/home/dmarce1/workspace/cosmictiger/tools/fmm/spherical.py)
- [Spherical periodic generator](/home/dmarce1/workspace/cosmictiger/tools/fmm/spherical_ewald.py)
- [PMMM coefficient and field adapters](/home/dmarce1/workspace/cosmictiger/src/gravity/nitadori/kernels.cpp)
- [Spherical normalization and operator conventions](/home/dmarce1/workspace/cosmictiger/docs/generated_spherical.md)
- [Periodic trace conventions](/home/dmarce1/workspace/cosmictiger/docs/generated_spherical_ewald.md)
