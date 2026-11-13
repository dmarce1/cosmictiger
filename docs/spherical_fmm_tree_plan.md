# Spherical FMM kernels and Cartesian tree integration plan

> Historical record of the retired PMMM implementation. Its spherical kernels,
> basis conversions, and mesh/tree adapters have been removed. The descriptions
> and measurements below apply to the source version recorded here; see the
> [Cartesian global FMM documentation](generated_fmm.md) for the active solver.

Planning baseline: main at `61c1e0ea3057c6652923ecdddc9a05a757c1b729`,
November 9, 2026. This is an implementation instruction package. Parts 1-5 were
implemented in this checkout on November 9. Step 5 completed the integration
matrix and profiling and selected float L2L/L2P inside the local trees, with
double mesh/root boundaries.
Part 2's derivation, conventions, operation counts and validation are in
[historical basis-conversion derivation](https://github.com/dmarce1/cosmictiger/blob/38f59a72ab08e2355017728e60b338b352f1cd05/docs/fmm_basis_conversion.md).
Part 3's dataflow, precision/memory contracts and validation are in
[fmm_root_deposition.md](fmm_root_deposition.md).
Part 4's numerical boundary, selected schedule, memory bound and measurements
are in [fmm_far_locals.md](fmm_far_locals.md).
Part 5's current precision contract, regression matrix, complete-solve measurements
and explicit validation limits are in
[spherical_fmm_validation.md](spherical_fmm_validation.md).
Recheck HEAD, AGENTS.md, worktree ownership and handoffs before
starting a part. Preserve other conversations' changes.

The requested outcome is Cartesian local FMM trees with spherical FFT moments
and locals, generated through a consistent operator interface. Dipoles are
always present. Production local traces remain implicit from the known uniform
background; do not turn on a stored trace merely to implement the bridge.
Evaluate the arithmetic and performance of implicit-trace alternatives before
settling the local representation.

## Decisions and current behavior

1. Rewrite spherical kernels in the style of the current Cartesian generator:
   typed operators, explicit coefficient access, separate operator headers,
   generated straight-line arithmetic, reproducible generation and accurate
   operation counts. Preserve the spherical normalization and FFT layout.
2. The Newtonian near subtraction does not create a new trace hierarchy.
   Within each valid expansion region, the periodic Green function retains a
   constant Laplacian. Derivatives of that Laplacian vanish. One known scalar
   is sufficient; no extra FFT channel is needed.
3. Prefer compact locals with the background implicit. Compare the existing
   isotropic correction, known-trace reconstruction through one dependent
   component, and a quadratic retained at the mesh-cell origin. The efficiency
   evaluation below precedes the implementation parts.
4. Preserve the current potential convention, including its already existing
   scalar source correction. This does not require trace storage in every tree
   node. It is distinct from the known background Laplacian.
5. Reuse the existing per-cell kd-tree forest for FMM-near solves, including
   those with CUDA near evaluation. Deposit owned roots only, after shifting
   to the mesh center and changing coordinate units. Never deposit ghost roots.
   Keep current direct CPU deposition for PP-near and far-only solves, which
   do not build this forest. Do not introduce another tree for deposition.
6. Implement upward reuse separately from downward reuse. The latter can
   otherwise remove the current overlap between near evaluation and FFT.
7. Defer direct GPU cell P2M and compensated FP32 accumulation. The user chose
   root reuse for the present scope: the required CPU kd-tree already computes
   the moments, so a separate GPU particle sweep would duplicate that work.
   No GPU deposition prototype, backend selector or additional implementation
   part is required. Revisit only as a separate future optimization after
   profiling root reuse.

The source audit found these concrete integration points:

| Area | Current behavior and source |
| --- | --- |
| Spherical generation | `tools/fmm/spherical.py`, `tools/fmm/spherical_ewald.py`; the former public facade `cosmictiger/gravity/kernels/spherical_operators.hpp` was retired after this plan |
| Cartesian model | `tools/generate_fmm.py`, `tools/cartesian/`; emitted operators under `cosmictiger/gravity/kernels/cartesian_tracefree/` |
| Orders | `fmm_order.hpp`: order q has source degree q-2, local degree q-1, harmonic sizes (q-1)^2 and q^2; both paths use the same q |
| Convolution truncation | `src/gravity/pmmm_kernels.cpp:58`: source rank plus local rank is at most q-1; retain this, rather than adopting another paper's order convention |
| Current deposition | `src/gravity/mesh_exchange.cpp:155`: double spherical P2M and radial moment in a particle loop over each occupied cell |
| Ownership | `mesh_exchange.cpp:24`: production particle domains cut on coarse mesh planes, so each stage cell is wholly owned; slab ownership is separate |
| Tree timing | `src/gravity/solver.cpp:223`: deposition precedes starting the FMM near pipeline and building its trees |
| Tree centers | `src/gravity/fmm/cell_tree.cpp:112`, `src/gravity/tree_build.cpp:218`: constrained enclosing centers, not necessarily mesh centers or centers of mass |
| Precision | `cosmictiger/gravity/tree_multipole.hpp:45,78`: eight-lane float leaf P2M and horizontal reduction; double accumulation across packets; two-lane double parent M2M; every node narrows to float before its upward return |
| Current far evaluation | `src/gravity/fmm/cell_schedule.cpp:83`: near walk starts with zero local, overlaps FFT, then spherical far L2P runs per particle |
| Background | `src/gravity/fft.cpp:391`: global reduction of mass, three dipoles and one radial scalar; `mesh_exchange.cpp:133` corrects returned locals |
| Existing limits | `solver.cpp:209`: frozen outer-interval near work is PP-only; far-only boundary solves currently do not build the FMM forest |

The emitted Cartesian types in this baseline are compact, with no active trace
slot. The user identifies stored trace as an optional generator capability,
off by default. Follow that policy. Some comments in `fmm_order.hpp` still
describe an appended trace despite the actual q^2 count; reconcile comments and
the actual generator invocation during Part 1, without enabling trace storage.

### CPU kd-tree construction and precision

The existing recursive CPU kd-tree build already constructs the Cartesian
multipoles. `src/gravity/tree_build.cpp:114` partitions the particle range,
then starts its children. After both children finish, lines 370-373 translate
and combine their moments. A leaf computes P2M at lines 438-441, in a separate
particle scan within the same build after its bounds and radius are known.
Thus construction is integrated with tree sorting, although P2M is not fused
into the partition/swap loop. The proposed change reuses that computation;
it does not add a second tree build or a standalone tree-wide moment pass.

The current precision stages are:

| Stage | Arithmetic / storage |
| --- | --- |
| Leaf displacement and P2M | Eight-lane float SIMD |
| Leaf mass weighting and horizontal sum | Float, including the sum within each eight-particle packet |
| Accumulation across packets | Scalar double `mValue` |
| Child-to-parent M2M and two-child sum | Two-lane double SIMD, one lane per child |
| Upward child return and persistent near-field arena | Float |

`SimdDouble` is explicitly `SimdVector<double, 2>` in
`cosmictiger/math/simd_backend.hpp:68`; this is not an unspecified native width.
The leaf already has a double accumulator (`tree_multipole.hpp:83`). It narrows
at line 124, and each parent narrows at line 64, before returning a
`TreeCreateReturn` whose `MultipoleBuffer` holds float coefficients. Arena
insertion copies those already-rounded floats; it is not the first narrowing.

Git history confirms two earlier precision policies. In `9b46199bd`
(July 30, 2021), ordinary FMM used float SIMD P2M contributions, a double leaf
accumulator and two-lane double M2M (`src/tree.cpp:268-342`;
`tigerfmm/simd.hpp:102` defines the two-double `__m128d` pack). The mixed policy
also appears in `f8b75e22a` (July 27, 2024).

More directly relevant is the historical FMMPM mode at `a7b17873e`
(September 11, 2022): `cosmictiger/tree.hpp:171-175` returns
`pm_multipole<double>` upward, `src/tree.cpp:627-651` evaluates leaf P2M using
eight-lane logical double SIMD, and lines 555-567 use two-lane double M2M.
Lines 700-703 separately convert to float near-tree records. Ordinary FMM in
that same revision keeps the mixed leaf path. These are historical design
references, not instructions to restore the old coefficient types or kernels.

Use that FMMPM precision split as the preferred accuracy baseline: double
owned-source construction and upward moments, with separate float near-walk
records. Expose and preserve the existing double accumulation first, then
evaluate full-double leaf arithmetic and the current mixed leaf arithmetic as
separate candidates. This isolates the effect of removing intermediate node
narrowing from the effect of promoting leaf polynomial/reduction arithmetic.
Keep the mixed leaf path only if it preserves the mesh error envelope with a
useful measured cost benefit. The source conversion consumes the double root,
not a widened copy of its float near-tree record.

At the same hardware vector width, doubles have half as many lanes as floats;
this is an arithmetic-throughput consideration, not a measured 2x total-sort
penalty or an upper bound. M2M already performs the two-child arithmetic in
double. Preserving its output precision primarily changes return storage and
copies; partitioning, geometry and scheduling are unchanged. A double leaf
variant must choose an appropriate wider pack for particles rather than reuse
the two-lane, two-child M2M shape. Measure full build/deposition time, payload
memory and accuracy. The Step 0 FP32 local-kernel benchmark does not measure
these P2M/M2M precision choices.

## Background convention and efficiency evaluation

Use the code's negative Newtonian potential and acceleration = -gradient(phi).
For a periodic volume V, away from retained singular sources,

    Laplacian(G) = -4 pi / V.
    b = 4 pi / (3 V).
    beta = b M.
    phi(x) = H(x) - beta |x|^2 / 2.

Here H is harmonic and x is displacement from the local's reference center in
physical coordinates. For PMMM's cell coordinate u = x/h, beta becomes
beta_s = b M h^2. These equations are derived in the repository's sign and unit
conventions. Nitadori's Appendix B, equations 38-42, supplies the periodic
correction framework; its potential sign must not be copied without conversion:
[Particle mesh multipole method](https://arxiv.org/html/1409.5981v2#A2).

Subtracting selected primary Newtonian near terms does not alter this constant
Laplacian in each expansion region. The origin must use the regular Ewald
remainder. Keep periodic image contributions in near-mask entries.

### Three compact representations to compare

**Isotropic implicit correction.** Store H. At a shift d, add
-beta |d|^2/2 to the constant and -beta d to the gradient; keep beta implicit.
At evaluation, add -beta |x|^2/2 to potential and +beta x to acceleration.
With -beta/2 precomputed, this costs at most 13 conventional FLOPs per shift
and per point, or 6 for the force/gradient corrections alone. Reuse of squared
coordinates in the generated arithmetic can reduce the count. There is no
reason to add beta for each near M2L contribution: it belongs to the complete
far local once.

**Known-trace reconstruction through z.** Let K = -3 beta be the actual
Laplacian. Store the independent full local derivatives xx and yy, and recover

    actual zz = K - xx - yy.

All derivative tensors of rank >= 3 remain tracefree. Equivalently, store a
harmonic proxy H' with

    phi(x) = H'(x) + K z^2 / 2.

Relative to the ordinary harmonic local H, initialize Cartesian xx and yy by
subtracting beta from each. In the current spherical normalization,
U20 = (2 z^2 - x^2 - y^2)/4, so initialize the proxy by adding 2 beta to L20
(real slot 6). Do this once after forming the completed far expansion; keep
FFT coefficient packing and Green spectra in the existing harmonic convention.

Translation then requires only

    H'_new(x) = H'_old(x+d) + K d_z x_z + K d_z^2 / 2.

Evaluation adds K z^2/2 to potential and -K z to accelerationZ. With K/2
precomputed, either operation adds 5 conventional FLOPs, or 2 when only the
linear/force update is needed. Reusing z^2 can reduce the five to four. This
has the same coefficient count as a harmonic local. Explicit trace storage
does not reduce these operations; it adds a scalar to load/store or transport.

The proxy is a different semantic local convention. Use a clearly named type
or view/adapter so an ordinary harmonic L2P cannot silently evaluate it as the
physical potential. Its storage may reuse existing buffers. Source multipoles
remain STF; this change applies only to periodic local expansions. Preserve
the convention through rotations, translations and active-subtree selection.
If a rotation implementation acts on the proxy as a harmonic polynomial,
apply the residual correction in the original coordinate axes after the full
translation. Do not rotate the proxy and assume its residual axis changed for
free.

**Mesh-anchored quadratic.** Keep H and translate only harmonic locals. At
each particle add the quadratic relative to the original mesh-cell center,
not the current leaf center. This removes background corrections from L2L but
requires the cell-relative particle displacement at evaluation. It remains
correct with the already applied source constant and dipole corrections in H.
Preserve the mesh anchor through every subtree; apply the quadratic once after
near and far combination, before gravitational scaling. Measure the cost of
obtaining this second displacement, especially for fixed64, rather than
treating it as free.

Also compare the stronger mesh-anchored variant using the same H' proxy and
only K z_mesh^2/2 as the residual. This costs five correction FLOPs per point
and none per edge, plus anchored-coordinate work. If z_mesh is formed from
node-relative z plus a leaf-center offset, that adds one addition per point.
Against recentering, its standalone arithmetic saving is then 5E-N FLOPs,
where E is the number of L2L translations and N the number of points. Dense
leaves can therefore favor recentering. Test the coordinate precision before
substituting such an addition for the existing direct fixed-point displacement.
Construct this proxy at the mesh center before the harmonic mesh-to-root shift;
converting at the root while retaining a mesh-anchored residual would mix origins.
Its node gradient is only the proxy gradient until the anchored correction is
added. Do not apply recentering corrections as well in this variant.

For context, checked-in Cartesian conventional counts are:

| Local degree | Internal q | Harmonic L2L | Harmonic L2P | Five-FLOP correction / L2P |
| --- | --- | --- | --- | --- |
| 2 | 3 | 64 | 53 | 9.4% |
| 4 | 5 | 369 | 202 | 2.5% |
| 6 | 7 | 1242 | 487 | 1.0% |
| 8 | 9 | 3187 | 996 | 0.5% |

These are arithmetic ratios, not elapsed-time predictions. The current
spherical Ewald wrapper reports 15 extra conventional FLOPs for L2L and 14
for L2P, including its scalar multiplication and trace accumulation. With C
nonempty binary cell trees, L leaves and N particles, there are 2(L-C) tree
edges. A five-FLOP correction on these edges costs less than 10/(N/L) FLOPs
per particle. If mean occupancy is 64-128 that is below 0.16-0.08 FLOP
per particle; these occupancies are examples, not a guarantee from the default
128-particle leaf cap. Root recentering adds its own per-cell cost.
Active-subtree selection can execute a different set of shifts; count those
in the actual scheduling measurements.

### Isolated timing results and decision

The isolated benchmark used the checked-in generated Cartesian L2L/L2P bodies,
scalar FP32, GCC 15.2, `-O3 -march=native -ffp-contract=fast -fno-tree-vectorize`
on one pinned Ryzen 3900X CPU. Each case had three warmups and eleven samples
with shuffled node traversal. The focused results below use 256 locals and
eight points per local. Units are median L2L nanoseconds / 8 + median L2P
nanoseconds per point; this is an amortized kernel estimate, not a timed
combined tree traversal.

| Field | Local degree (q) | Implicit z trace | Stored z trace | Mesh-anchored z residual |
| --- | --- | --- | --- | --- |
| Potential and force | 4 (5) | 38.06 | 38.07 | 37.91 |
| Potential and force | 6 (7) | 137.06 | 137.46 | 136.94 |
| Potential and force | 8 (9) | 342.91 | 344.59 | 341.57 |
| Force only | 4 (5) | 22.08 | 21.99 | 22.10 |
| Force only | 6 (7) | 96.47 | 96.57 | 92.67 |
| Force only | 8 (9) | 265.05 | 266.87 | 256.99 |

Implicit and stored z variants use identical correction arithmetic. The stored
variant adds four bytes per FP32 local: 104/200/328 rather than 100/196/324
bytes at q=5/7/9. The anchored variant includes the extra addition forming
z_mesh from the node offset. Initial xx/yy adaptation costs two subtractions
per input local and is excluded from the timed sections for all z variants.

All 42 focused configurations passed the single-shift polynomial/force check;
the worst absolute discrepancy was 2.09e-7. Two initial 120-configuration sweeps,
also covering 32768 locals, one point per local and isotropic variants, passed
the same check. These verify the background algebra over the generated
arithmetic, not independent correctness of the harmonic kernels or multi-level
tree integration. Timing variability is recorded in the raw results; differences
of a few percent can reflect compiler scheduling and layout. No SIMD, CUDA,
MPI or whole-application timing was performed.

**Decision:** retain implicit trace as the production default. There is no
useful stored-trace advantage in this experiment. Use the compact z-trace
convention as the first recentered implementation candidate; integrate its
correction into the arithmetic DAG to share intermediate expressions. Keep
mesh-anchored z evaluation as a performance experiment for Part 4: its roughly
4%/3% force-only improvement at degrees 6/8 here is promising but not a
production speedup result. Select between them after actual displacement,
SIMD/CUDA and whole-solve overlap measurements. Do not add a public selector
or persistent trace field on the strength of this microbenchmark.

Reproducibility artifacts are preserved in the ignored
`build/spherical_fmm_trace_study/` directory: `README.md` records methodology
and the exact commands run from `/tmp/pmmm_background_bench`; `bench.cpp` and
`generated.hpp` are the standalone source; `RESULTS.md` includes timing ranges;
`results_focused.csv`, `results.csv` and `results_repeat.csv` retain raw timings.
The source is self-contained in that directory and can be recompiled there.

### Source radial scalar and the potential zero

The known local Laplacian is determined by M/V. The independent source scalar
Q = sum m |x - meshCellCenter|^2 is determined by the particle offsets. Current
PMMM already computes it and adds -b Q/2 to the potential for q >= 4. This is
constant over sinks for one solve, and has zero force, but changes with source
positions and the chosen mesh. It is not determined by M/V.

Retain the existing convention: ordinary/far energy includes sum m phi/2,
the cosmological energy residual uses that energy, and restart state retains
it. Dropping the term would change a time-dependent potential zero and needs
a separate energy-convention decision. No additional FFT or permanent node
trace is needed. Fuse Q accumulation into an existing owned-particle pass,
such as the post-tree-sort rung snapshot in `cell_schedule.cpp:183`, using
the same mesh-center coordinate convention as current deposition. Retain one
scalar per cell packet and the existing five-scalar global reduction. The
lowest-order q=3 convention intentionally omits this source-rank-two potential
term; preserve it during the refactor.

Unit contract: the existing packet `radialTrace` and reduction `background[4]`
store Q_s = sum m |u|^2 in mesh coordinates. The physical Q above is h^2 Q_s.
Keep Q_s in those fields and retain the existing h^2 multiplication at
consumption; do not deposit physical Q into the old packet and multiply by
h^2 a second time.

## Bridge contract

Generate separate maps for sources and locals. The local map is the dual
evaluation map, not the inverse source map. Use exact rational polynomial
algebra during generation; emit sparse straight-line operations at runtime.
Do not fit the transforms numerically at runtime.

For the same center and coordinate units, with C raw STF Cartesian source
moments and S the current complex notation for packed real spherical moments:

    S00 = C_m
    S10 = -C_z
    S11 = -(C_x + i C_y)/2
    S20 = -3(C_xx + C_yy)/4
    S21 = (C_xz + i C_yz)/2
    S22 = (C_xx - C_yy + 2 i C_xy)/8

Negative m slots store the imaginary part of the positive-m coefficient;
they are not independent negative-m complex coefficients. Keep the existing
slot n(n+1)+m. Retain every dipole component at every supported order, with
no option, COM specialization or assumption that it is zero.

For locals, derive the Cartesian derivatives by differentiating the spherical
evaluation polynomial. Low-rank checks include C_x=Re(L11), C_y=Im(L11),
C_z=L10, C_xx=(-L20+Re(L22))/2, C_yy=(-L20-Re(L22))/2,
C_xy=Im(L22)/2, C_xz=Re(L21), C_yz=Im(L21). Apply the selected periodic
local convention separately and explicitly.

Production coordinates have another normalization: y = kappa r for the tree,
kappa = 1000, while u = r/h for PMMM. The tree walk multiplies evaluated
potential by kappa and acceleration by kappa^2. Consequently, at a common center:

    source rank n: S_n = (kappa h)^(-n) * sourceBasisMap(C_n)
    local rank n:  C_n = kappa^(-1) (kappa h)^(-n) * localBasisMap(L_n)
    beta_cartesian = b M / kappa^3
    K_cartesian = -3 beta_cartesian

The extra kappa^(-1) in the local map is essential. It is not merely a change
of position units. Test it through the production potential and acceleration
rescaling, not only a coefficient round trip. Source shifts use old center
minus new center; local shifts use new center minus old center. Define origins
and units in both conversion APIs and test nonzero shifts across periodic seams.

## Implementation parts and model allocation

These are task-specific model recommendations, not measured model comparisons.
They use the current [OpenAI model guidance](https://developers.openai.com/api/docs/guides/model-selection)
and [reasoning-effort definitions](https://developers.openai.com/api/docs/guides/reasoning).
Use the model names available in the implementation chat. Escalate effort when
the mathematical or scheduling tests expose an unresolved issue.

| Part | Deliverable | Suggested model and effort | Depends on |
| --- | --- | --- | --- |
| 0 | Trace representation efficiency and numerical contract | GPT-6 Astra, xhigh | Source audit and isolated benchmark |
| 1 | Spherical generator and API rewrite | GPT-6.1 Sol, high | Part 0 contract |
| 2 | Cartesian/spherical conversion kernels | GPT-6 Astra, xhigh | Part 1 type/layout contract |
| 3 | Owned-root source deposition | GPT-6.1 Sol, high; xhigh for lifecycle debugging | Parts 1-2 |
| 4 | Cartesian downward far evaluation | GPT-6 Astra, xhigh | Parts 2-3 timing and correctness |
| 5 | Integrated regression, profiling and cleanup | GPT-6.1 Sol, high; independent Astra review | Parts 1-4 |

Parts 1 and 2 may overlap only after the coefficient/type contract is fixed
and file ownership is split. Parts 3 and 4 share lifecycle and scheduling
files and should run sequentially. Use main for the first active implementation
and a separate snake_case branch/worktree for each additional one, for example
`codex/spherical_kernels` and `codex/fmm_basis_conversion`. Planning/auditing
does not reserve those implementation files.

### Part 0 instructions

Freeze the sign, normalization, order, local-background and source-potential
contracts above. Compare scalar-stored, isotropic-implicit, known-trace-z and
mesh-anchored correction costs on identical generated kernels, with correctness
checks, shuffled inputs, warmup, repeated timings and a checksum. Record actual
compiler, flags, precision, working-set sizes and timer variation. Separate
kernel timing from predicted tree amortization. Select a compact default and
record what remains to be demonstrated on production SIMD/CUDA and whole solves.
Do not enable stored trace, add a dipole option, or change the potential gauge.

### Part 1 instructions

Rewrite `tools/fmm/spherical.py` and its Ewald companion to emit one header per
operator under `cosmictiger/gravity/kernels/spherical/`, with a lightweight type
header and facade analogous to Cartesian. Use typed uppercase P2M, P2L, M2M,
M2L, M2P, L2L and L2P. Expose explicit harmonic indexing/named accessors while
preserving FFT storage order. Reuse the established DAG/emission infrastructure
where practical; do not turn a layout cleanup into a new translation algorithm.
Preserve selected rotation variants and all generic dipole terms.

Implement the Part 0 compact background convention through explicit periodic
types/views or context, with no per-local scalar in the production layout.
The completed far local inherits the known background once. Additive harmonic
near terms do not increment it. Retain an explicit-trace reference only where
it is useful for verification or already supported; production defaults off.
Never apply a free-space axial rotation shortcut to an arbitrary periodic Green
tensor. Keep total-degree contraction, doPotential behavior, scalar/SIMD/CUDA
support and weighted/conventional FLOP reporting explicit.

Update generator outputs, facade users, CMake regeneration declarations and
operation-count checks together. Change generated files through the generator.
Acceptance: reproducible regeneration; all supported-order independent
spherical and Ewald tests; nonzero dipoles; near/origin Green behavior; nested
local translation with known background; CPU and CUDA compile coverage.

### Part 2 instructions

Generate Cartesian STF source-to-spherical and spherical-local-to-Cartesian
maps, using separate source and local degrees. Keep pure basis conversion,
recentring, physical scaling and background-convention adaptation independently
testable, even if a later fused entry point eliminates temporary arithmetic.
Provide scalar and SIMD/CUDA-compatible code with no runtime allocation,
factorials or dense matrix multiplication. Put the generator in a narrowly
owned new module such as `tools/fmm/basis_conversion.py`.

Acceptance: all q=3..9 and every stored basis component; asymmetric unequal-mass
clouds; nonzero dipoles; source conversion versus independent spherical P2M;
translation/conversion commutation; local evaluation and gradient equivalence;
known-background translation and near-addition composition; fixed32/fixed64
seams; production kappa/h scaling at more than one mesh width. Verify the
background coefficient in tree units is independent of h. Round-trip tests
alone are insufficient because two incorrect maps can cancel each other.

### Part 3 instructions

For the FMM-near backend, refactor ordering so completed owned roots can supply
mesh moments before the FFT. This source path also applies when CUDA executes
the near interactions; GPU availability does not change deposition selection.
Finish ghost count discovery, planGhosts and shared
arena reservation/reset before any owned or ghost tree construction. Prepare
source promises before any producer/consumer starts. Build each owned cell once, deposit its completed
root once, and reuse the same tree for near work. Ghost trees supply near work
only. Keep slab routing, bounded packets, failures and collective ordering.
Preserve post-sort particle mappings and SPH reverse links.

Reuse the existing CPU leaf P2M and bottom-up M2M during the same recursive sort.
Follow the historical FMMPM separation: double owned-source build results and
float persistent near-tree records. First expose the existing double leaf
accumulator before its narrowing and retain double moments in the transient
upward build payload. Feed those doubles into the already two-lane double M2M
and preserve the parent result in double. Narrow a separate copy only when storing each node
in the persistent near-field arena. Recenter, scale and convert the double
owned-cell root for deposition; do not widen its float arena copy instead.
Discard child payloads after parent reduction. This adds neither another tree
pass nor a persistent second tree, but its temporary/future storage must be
budgeted. Update affected build-return serialization and root-cache contracts;
avoid widening unrelated ghost/walk buffers merely for mesh deposition.

Compare three precision paths against the existing direct double mesh P2M:
the current float upward records, double upward records with current mixed
leaf P2M, and a full-double leaf/reference variant. The mixed leaf path already
accumulates packets in double, but its float horizontal sums and point moments
remain rounded. If these are limiting, first isolate widening the lanes before
horizontal reduction from changing displacement and P2M arithmetic. Require
double displacement, mass weighting, P2M and reduction only for the explicitly
full-double variant. Do not call a double accumulator around float P2M an
all-double construction. A double conversion cannot recover rounded bits.

Use full-double owned-source leaf arithmetic as the preferred accuracy baseline;
the current mixed leaf path is a measured optimization candidate. Choose the
minimal leaf precision that preserves the PMMM coefficient and field error
envelope, then measure whole tree-build plus deposition time. Two-lane
double parent arithmetic is already the baseline, so do not assign it a new
blanket 2x cost. Count return-payload traffic, future lifetimes and any leaf SIMD
width change separately. The alternatives are controlled experiments, not a
requirement for several permanent production modes. Retain the direct-deposition
oracle and report both arithmetic precision and storage precision in results.
Accumulate the existing source radial scalar in the already-required owned-cell
scan, in double about the mesh center; no permanent node trace is needed.
Publish owned-root deposition readiness only after its radial summary and
post-sort stage-rung snapshot are complete.

The intended dependency is owned-root completion -> deposition -> FFT. After
the shared arena has been prepared, ghost arrivals/builds and ready near batches
can overlap these stages. Near tasks may
wait for farReady to consume final fields; source producers and deposition must
not wait for those near consumers. Split pipeline start from any wait-for-all
operation that would create a cycle. Drain tasks on failure before freeing
arenas, promises or root metadata.

Keep the current direct double CPU deposition for PP and far-only solves.
Do not construct an FMM tree solely to support those paths in this part, and
do not add direct GPU deposition or compensated FP32 P2M. Preserve existing
capability gates and frozen-interval behavior, including no deposition or FFT
on frozen near-only steps. Acceptance: old double direct
cell moments versus root deposition, no missing/duplicate mass, dense and sparse
cells, unequal masses, empty localities, mixed rungs, delayed ghosts, failure
cleanup, one/two/four localities, and whole-solve phase timing.

### Part 4 instructions

Implemented November 9, 2026, initially with double descent. Step 5 retains the
dedicated schedule and uses float below the double root boundary. It reuses the near
batch's selected indices and original immutable owned tree. Whole-solve timings
and peak RSS were compared with a combined float-walk seed; the dedicated path
retains pre-FFT near progress with essentially equal measured time/memory.
See [fmm_far_locals.md](fmm_far_locals.md) for the small mesh-case difference and
validation limits. The instructions below remain the acceptance contract.

Convert one completed spherical far local per occupied sink cell. Adapt its
background convention, center and units, then use Cartesian L2L/L2P through
the local kd-tree. Keep complete far and harmonic near contributions distinct
until the background ownership is unambiguous. Apply gravity scaling and the
particle kick exactly once to their combined field.

Evaluate two schedules separately: a dedicated far-only downward pass after
FFT while near work retains its present overlap, and a single combined walk
seeded with the far local after FFT. The second saves a traversal but delays
the near walk. Choose using total solve time and memory, not operation counts.
The first must reuse or safely reserve active-sink selection; invoking the
generic batch walk twice can duplicate sink-only nodes and overrun arena
assumptions. Bound retained near fields; do not introduce an N-sized scratch
result merely to simplify synchronization.

Keep the PP/far-only direct evaluator as a supported fallback until separately
validated; this part does not add frozen-FMM support. Test mixed active leaves,
cell/root offsets, periodic images, doPotential=false, all relevant q,
near-plus-far versus the independent finite-expansion oracle, CPU/CUDA agreement,
delayed-ghost overlap and exception cleanup. Measure actual scheduling overlap.

### Part 5 instructions

Completed November 9, 2026. All-order/both-width helper checks, fixed64 q3-9
FMM/PP solves, fixed32 CPU/CUDA paths, true MPI, gravity-disabled capabilities,
evolution, SPH outer-interval energy and restart tests pass. Float tree descent
keeps the original error tolerances. Complete-solve medians are essentially
unchanged; no general speedup is established. The report records the low-order
fixture corrections, bounded GPU execution, raw results and untested configurations.
The following instructions describe the completed acceptance scope.

Run focused generator, kernel, conversion and PMMM tests first, then the
integration matrix affected by these changes. Retain the independent
finite-expansion oracle in `tests/gravity/pmmm.cpp:692` and the full Ewald oracle.
Do not loosen established tolerances to accept a new rounding path. Distinguish
existing failures from new ones using the same baseline and resource allocation.

Cover CPU and CUDA near paths, PP and FMM, gravity-disabled capability builds,
fixed32/fixed64, supported orders, unequal masses, clustered cells, sparse active
rungs, near widths, multiple mesh sizes, and true multi-process runs where
available. Preserve outer-interval far-only energy, restart and SPH index-map
tests. Report skipped/environment-blocked configurations explicitly.

Profile complete solves: owned tree construction, deposition, FFT, overlap,
downward pass, final combination, peak scratch, ghost bytes and compiler/code
size. Sweep occupancy and active fraction; report both forces and potential
requests. Compare source reuse alone, then source plus sink reuse. Avoid adding
public tuning knobs before measurements establish a need. Keep the old direct
path as a diagnostic oracle where useful, remove temporary production branches
only after the new path passes, and publish only a reviewed owned-file allowlist
if publication is subsequently authorized.

## Resource and delivery rules

Latest explicit user report in this chat: War Thunder ON. All local work shares
6 CPU cores and at most 25% GPU with other agents and Claude. Coordinate before
builds, MPI or GPU timing; defer GPU runs that cannot respect that ceiling.
The completed planning evaluation used one CPU core, no GPU and no QB logins;
that CPU allocation is now released.
No full application build or runtime validation is implied by an isolated
microbenchmark. Preserve exact validation levels in each handoff.

Each implementation part must leave a short handoff recording files owned,
the final numerical contract, commands actually run, failures or untested paths,
and what the next part may rely on. Follow current C++ naming, Unicode equation
identifiers and symbol directories, trailing const, include grouping and format
rules. No source implementation or publication is authorized by this plan alone.
