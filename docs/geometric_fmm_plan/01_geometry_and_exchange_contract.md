# 1. Geometry and exchange contract

Created: 2026-11-14. Status: step 1 design contract completed on 2026-11-14.
Solver implementation and runtime validation remain work for subsequent steps.

Recommended model: **GPT-6 Astra**. Reasoning effort: **High**.

## Working agreement

This file contains the step 1 design result, not evidence of solver implementation. Reading it
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

This is the mathematical and protocol design step. Produce a concrete contract in chat
(or a design document if explicitly requested); do not implement the solver. No prior
implementation is required. This contract is input to steps 2 through 5. Ordinary choices
should be resolved with a stated recommendation rather than a sequence of permission questions.

## Work

1. Define root depth zero and the exact 18-axis sequence
   `x y z z y x z x y y x z y z x x z y`. The split at depth d uses sequence[d % 18].
   A geometric cell identity must retain depth and sufficient path/coordinate bits;
   do not assume a single size_t can encode every supported depth. Source identity adds
   the owning locality; an exchange generation identifies the data snapshot.
2. Specify half-open particle assignment, including particles exactly on a split and the
   periodic unit-box seam. Bounds and centers must be reconstructed identically on peers.
   Account for both Position precision and the existing Fixed32 FmmBox representation.
   Define a terminal policy at the representable-depth limit. Recommended starting point:
   an overflow leaf when no exact geometric split is representable, provided leaf processing
   supports it. Coincident overfull particles can terminate after the Ewald requirement is
   met; never substitute a nongeometric split. Record any necessary leaf-size support work.
3. Let n_i(d) count prior cuts on axis i. Side lengths are L_i(d) = 2^(-n_i(d)) and
   Rmax(d) = sqrt(sum_i L_i(d)^2)/2. For d = 3q+s:
   - s=0: Rmax = 2^(-q) sqrt(3)/2.
   - s=1: Rmax = 2^(-q) 3/4.
   - s=2: Rmax = 2^(-q) sqrt(6)/4.
   The child-center offset is L_split(d)/4. Specify conservative rounding of stored
   radii and bounds; depth-only tables are geometric limits, not measured cloud radii.
4. For nonempty leaves, r = max_i |x_i-c|. For parents,
   r = min(Rmax(d), max_nonempty_children_j (|c_j-c|+r_j)). Empty clouds have r=0.
   This adapts Dehnen (2002), equation 9 and its corner cap, to prescribed geometric
   centers. Retain dipole moments: geometric centers are not generally centers of mass.
5. Use the ordinary leaf condition d >= dEwald AND N <= bucketSize; a full bucket
   needs no further subdivision. Empty branches need no forced subdivision. Translate the
   current Ewald condition to the first depth with both
   Rmax(d) < theta/[4(1+theta)] and Rmax(d) < 1/8-h_G/4.
   Handle parameters for which no supported depth satisfies it. The selected depth is
   runtime-dependent even though the geometric table is constexpr.
6. Derive conservative source/target separation for arbitrary depths ds and dt:
   H(ds,dt) = max((Rmax(ds)+Rmax(dt))/theta, Rmax(ds)+Rmax(dt)+h_G).
   For the current Newtonian cell-cell MAC, center distance > H is sufficient; equality
   stays unresolved. This is not by itself an Ewald/image-coverage proof.
7. Define how an ownership box is rounded outward to level-cell boundaries, periodically
   wrapped, and padded. Such covers include centers lying outside ownership bounds.
   Resolve mixed-depth traversal explicitly: a multipole exported as terminal must be
   usable by the receiver's actual sink refinement/evaluation rules. Do not apply the
   equal-level halo to all depth pairs without a proof.
8. Prefer shared geometric target covers for export decisions. If a tight export requires
   target topology not available from ownership geometry, specify a small target-summary
   exchange or a conservative extra-refinement rule. Do not make construction depend
   circularly on the remote export being constructed. Permit sender-side construction
   during its upward pass when the necessary destination information is already available.
9. Specify complete source coverage, self-exclusion, periodic image responsibilities,
   Newtonian/Ewald coupled handling, and the distinction between an internal source node,
   a true particle leaf, and a pruned multipole-only terminal. An absent child must never
   be interpreted as an empty physical cell merely because it was not transmitted.

## Deliverable and completion criteria

A compact contract with formulas, identity/terminal definitions, an explicit mixed-level
export/traversal rule, and worked examples: one shared cell on two owners, one coarse/fine
pair, one periodic seam, one empty rank, and coincident particles. Separate proved coverage
from assumptions and unresolved optimization. No performance claims are required.

## Source entry points and references

- `src/gravity/tree_build.cpp`, `src/gravity/global_fmm/distributed_tree.cpp`.
- `cosmictiger/gravity/fmm_acceptance.hpp`, `cosmictiger/gravity/fmm_box.hpp`.
- `cosmictiger/math/position.hpp`, `cosmictiger/core/tree_id.hpp`.
- [Dehnen 2002, equations 9-10](https://courses.physics.ucsd.edu/2017/Winter/physics141/Lectures/Lecture15/2002JCP...179...27D.pdf).

## Completed contract

The decisions below implement this design step. They take precedence over unresolved
choices in the work list above. Source inspection used main at
`b18af4112f6892ead43f5377fda115d244b80da9`. This is a three-dimensional, periodic
unit-box contract with the existing fixed gravity support and Cartesian moments.

### 1. Geometry, assignment, and identity

Root depth is zero. The split axis at depth d is entry d modulo 18 of
`x y z z y x z x y y x z y z x x z y`. Every consecutive aligned group of three
cuts uses each axis once. Let nᵢ(d) count cuts on axis i before depth d. A cell is
identified by `(depth, coordinate[3])`, where each integer kᵢ satisfies
0 <= kᵢ < 2ⁿⁱ. Its exact bounds and center are

    Bᵢ = [kᵢ 2⁻ⁿⁱ, (kᵢ + 1) 2⁻ⁿⁱ),   cᵢ = (2kᵢ + 1) 2⁻⁽ⁿⁱ⁺¹⁾.

On the selected axis, the children have coordinates 2kᵢ and 2kᵢ+1 and cut count
nᵢ+1; other coordinates are unchanged. Depth plus these coordinates determines
the binary path uniquely under this schedule. Use three unsigned 64-bit coordinates
and an unsigned 16-bit depth for the supported depths below, with explicit field serialization.
A single `size_t` path is insufficient. Existing domain-decomposition keys remain
separate; `TreeId(proc,index)` remains a storage handle, not a geometric identity.

Source identity is `(generation, owner, geometric cell)`. A generation includes a
run/session identifier and monotonically increasing build epoch, never reused while
messages or readers can survive. Node indices and source
particle indices are valid only within that owner's frozen generation. An imported
view also identifies its destination and destination-cover descriptor. A generation
freezes ownership, source ordering/positions/masses, topology, moments, effective
opening angle/support, and the protocol/geometry version. Migration, relevant drift,
source reordering, or changed acceptance parameters invalidates the affected views.
Do not mutate frozen source fields, reorder particles, or reuse arenas while readers
use that generation; ordinary updates of other owned sink fields remain possible.

Particles are canonical native Q0.P coordinates in [0,1), with P=32 or 64. Partition
by exact unsigned coordinates: below the split goes left; equality goes right.
Normalize the periodic seam before assignment: position 1 is position 0, belonging
only to the low-side cell. Tree cells themselves never wrap or shrink. Use unwrapped
native-coordinate differences for cloud radii, P2M, M2M and L2L; intercell force
displacements use the existing signed minimum-image convention, including its
half-box tie at -1/2. Never obtain assignment bits by converting Fixed64 to double.

### 2. Representability and terminal leaves

Geometric cells must have exact `Fixed32` box boundaries and exact centers in
`Position`. Therefore nᵢ <= min(32,P-1) on every axis. Center bits are
`(2kᵢ+1) << (P-nᵢ-1)`; construct them as integer bits. The root and the endpoint
at one need explicit extent/seam handling, not a shift by the integer width.

The supported CPU maximum is

    dMax = min(3 min(32,P-1), kMaxDepth-1),

giving 93 for Fixed32 positions and 96 for Fixed64 positions with
`kMaxDepth=128`. The deepest Fixed32-position cell has width two Fixed32 ticks,
so its midpoint remains representable. Fixed64 positions allow a cell one
Fixed32 tick wide. No geometric cell is finer than the Fixed32 box grid.
Advertise this inclusive limit in the generation descriptor. Current CUDA
stacks use `kCudaMaxDepth=64`; dispatch only subtrees whose required stack fits,
otherwise retain CPU processing. Their stack capacity is not permission to
change the geometric tree or truncate an export.

For exact dyadic bounds a,b, `FmmBox` stores the exact integers 2³²a and 2³²b
before reducing endpoints modulo 2³². Set `fullAxes` for an extent of 2³²;
equal endpoints without that bit mean a point. Compute these integers from the
cell descriptor. Adjacent deepest cells have distinct boxes; there is no
rounded box shared by finer geometric cells. Reconstruct centers and
subdivision from the cell descriptor, preserving the native Position bits.

For nonempty nodes, the ordinary terminal condition is exactly

    d >= dEwald AND N <= bucketSize.

N is the count of all owned gravity sources, not just active sinks; bucketSize must
be positive. N equal to bucketSize is an ordinary leaf once d >= dEwald. If this
condition fails, additional terminal cases, both requiring d >= dEwald, are:

- All particles have identical native positions: stop as a coincident overflow leaf.
- The next prescribed child exceeds dMax or cannot have exact bounds and center:
  stop as a depth/precision overflow leaf, recording which limit was reached.

Empty branches terminate immediately, with count and radius zero. An empty root is
explicit. Do not use a zero multipole mass as the occupancy test. Do not split at an
occupied extent or by particle count, drop particles, or displace a center to avoid
an overflow leaf. If dEwald cannot be reached, reject the configuration before a
force walk; early termination cannot restore the missing periodic invariant.

Overflow leaves require count-safe particle loops, source transport, cost arithmetic,
and allocations. CPU PP currently narrows/aggregates counts into `int`; CUDA has
tiled sinks but also finite list/stack/count capacities. These are support work for
steps 2 and 5, not established arbitrary-leaf support. Chunk oversized ranges without
creating new geometric leaves; until supported, diagnose an unsupported count/backend
before evaluation. Arena sizing must cover sparse single-child chains and empty ranks,
not just N/bucketSize; preserve stable addresses for readers.

### 3. Radii and conservative arithmetic

Derived from the side lengths Lᵢ(d)=2⁻ⁿⁱ, the geometric half-diagonal is

    R(d) = sqrt(sumᵢ Lᵢ(d)²)/2.
    R(3q)   = 2⁻q sqrt(3)/2
    R(3q+1) = 2⁻q 3/4
    R(3q+2) = 2⁻q sqrt(6)/4.

The child-center offset has length L_split(d)/4. The squared limits are dyadic and
can be tabulated constexpr; they describe cell geometry, not measured cloud radii.
For nonempty clouds, use

    r_leaf = max_particles p ||xₚ-c||,
    r_parent = min(R(d), max_nonempty children j (||cⱼ-c|| + rⱼ)).

This is the fixed-center adaptation of the child-enclosure and corner cap in
[Dehnen (2002), equation 9](https://courses.physics.ucsd.edu/2017/Winter/physics141/Lectures/Lecture15/2002JCP...179...27D.pdf#page=4).
Each candidate parent bound encloses every represented particle, so their minimum
also does. Dehnen's centers are centers of mass; ours are prescribed cell centers.
Retain dipoles and the current moment normalization, precision, and quadratic trace.
Build each owner's moments once from its own particles and reuse them in export views.

In stored arithmetic, let ρ(d) be an upper-rounded R(d). Evaluate leaf norms and
child-enclosure sums with bounded outward error, then narrow upward and cap by ρ(d).
Thus every stored r encloses the native positions and is <= ρ(d). Subtract native
integers before floating conversion; subtracting two rounded Fixed64 doubles loses
small separations. A final `nextafter` alone is not a proof for an arbitrarily
underestimated preceding norm. Use double intermediates with an upward step after
integer conversion and each nonnegative square/sum. Under strict IEEE arithmetic,
apply `nextafter(std::sqrt(sum), +infinity)` at the norm calculation, preserving
zero explicitly for an empty or zero-distance cloud, and narrow upward to float.
Parent bounds and Ewald thresholds retain their directed rounding as well.
Long double is unnecessary: Q64 squared differences are at least 2⁻¹²⁸ when
nonzero, within normal double range. Compile the consumers without fast-math or
implicit contraction. Validate enclosure using exact integer squared distances,
including a true radius just above a float boundary by a transverse Q64 tick.

All separation certificates compare a lower bound on distance against an upper
bound on the threshold; uncertainty and equality mean unresolved. Apply this rule
also to squared distances, cover padding, periodic image tests, and the runtime
inverse-theta conversion. Use the effective force-evaluation parameters, not a
different double/float approximation in the builder. Below, formulas state exact
real-arithmetic conditions; implementations must certify the rounded versions.

### 4. Required Ewald depth and periodic responsibility

For finite 0 < θ <= 1 and 0 < h_G < 1/2, choose the first supported depth satisfying

    R(d) < θ/[4(1+θ)]  AND  R(d) < 1/8-h_G/4.

This translates the current `treeCreate` box-radius checks. Select dEwald at runtime
from the geometric table. In finite precision use ρ(d), conservative parameter
bounds and sufficient arithmetic margin to certify the inequalities used below;
advance to the next depth if needed. Reject if no d <= dMax works. Changes in θ
(including an opening schedule) or h_G require rechecking tree depth and exports.

For minimum-image center displacement Δ, define

    m = maxᵢ |Δᵢ|,   D₁² = Δ·Δ,   D₂² = D₁² + 1 - 2m,
    u = r_source + r_target,       G = R(ds) + R(dt).

The current Ewald MAC requires D₂ > u/θ. Newtonian cell-cell acceptance requires
D₁ > u/θ and D₁ > u+h_G. D₂ is the second-nearest image distance; the Ewald kernel
already accounts for the periodic correction, background/zero-mode and normalization.
Transmit one source tree per owner, not 26 duplicate image trees. Preserve the
generated Ewald quadratic trace and the existing physical self/background convention.

Each source/sink interaction starts in `BothPending` (the current echecklist).
There are two resolution paths:

- Ewald accepted and a common nearest-image branch is certified: add Ewald once and
  continue as `NewtonianPending` (the current dchecklist).
- Otherwise, when both Ewald and Newtonian cell-cell MACs pass: evaluate both M2Ls
  at identical centers with the same displacement/image choice and consume the pair.

In other cases refine the source or sink without changing pending terms. Never resolve
an image-crossing pair through independently chosen Newtonian and Ewald expansions.
No particle-particle Ewald fallback is part of this contract.

**Descendant-center invariant.** The first path must certify the same image branch
for subsequent source/sink centers as well as particles. The prescribed radius
recurrence provides this: a leaf sphere contains its own center; if every child
sphere contains its occupied descendant centers, the child-enclosure bound contains
all those centers too. The geometric corner cap also encloses every descendant
center because it is inside the parent box. Taking their minimum preserves both
particle and occupied-descendant-center enclosure, including conservative rounding.
Thus the existing measured-radius certificate is sufficient:

    m + u < 1/2.

This is `fmmSinglePeriodicImage` with the stored enclosing radii. Preserve the
recurrence in steps 2 through 5: a tighter parent radius obtained by scanning only
particles would need a new descendant-center proof before supporting this handoff.
The stronger full-cell test m+G < 1/2 is an optional conservative certificate when
only geometry is available. Equality stays coupled. No image-branch change is
permitted during subsequent refinements of a NewtonianPending interaction.

**Leaf-pair termination proof.** Put R*=R(dEwald). For any nonempty leaf pair,
u <= G <= 2R*. Also D₂ >= 1/2, since
D₂² >= m²+1-2m = (1-m)² >= 1/4. The first depth inequality gives
2R*/θ < 1/2-2R*, so Ewald always passes at leaves. If the measured single-image
test fails, D₁ >= m >= 1/2-u >= 1/2-2R*, which is strictly greater than both
2R*/θ and 2R*+h_G. Thus the coupled path passes. Otherwise Ewald is resolved
separately and Newtonian can finish through existing softened PP/hybrid rules.
This proves termination in exact arithmetic; rounded predicates must preserve its
margins. Coincident/overflow leaves obey the same depth condition and proof.

### 5. Ownership covers and mixed-depth terminal certificate

Advertise destination ownership as a union of half-open boxes covering all its
owned particle positions for this generation, plus explicit empty-rank state.
Stale domain bounds after drift are insufficient: establish or enlarge the cover
before export selection. Split wrapped intervals at the seam. For a nonempty
unwrapped interval [aᵢ,bᵢ), its level-d cover is

    [floor(aᵢ/Lᵢ(d)) Lᵢ(d), ceil(bᵢ/Lᵢ(d)) Lᵢ(d)).

Take Cartesian products and the union over boxes; merge duplicates only within
this destination cover. Use the closure for distance bounds. Every occupied target
cell at depth d is contained in this cover, including its center even when that
center is outside the ownership box. Denote the cover C_d. Because the dyadic grids
are nested, C_(d+1) is contained in C_d.

For arbitrary source and target depths the sufficient Newtonian threshold is

    H(ds,dt) = max((R(ds)+R(dt))/θ, R(ds)+R(dt)+h_G).

This follows by replacing both actual cloud radii in the current cell-cell MAC by
their geometric upper bounds. It does not use particle-cell shortcuts. Equality
does not certify a terminal. H by itself says nothing about resolving periodic
pending terms; the periodic rules above are also required.

**Baseline, without target topology:** let d₀ be the destination's advertised
dEwald. Export a source node S as multipole-only only if

    dist_periodic(c_S, C_d₀) > H(ds,d₀).                         (terminal certificate)

Use upper-rounded ρ and H in the implemented test. To select possible near nodes
cheaply, pad every cover box by H in each coordinate, wrap the padded boxes, and
mark an axis full when its width reaches one. The periodic rectangular padding
contains the Euclidean H-neighborhood. Source centers outside that union are safe
terminal candidates; inside it, refine or use the exact conservative point-to-cover
distance to recover safe corner cases. Treat padded boundaries as unresolved.
Do not require the source center to lie inside its own ownership bounds.

**Mixed-level proof:** every actual destination leaf T has dt >= d₀ and
c_T in C_dt contained in C_d₀. R(dt) <= R(d₀), so the certificate implies

    D₁(c_S,c_T) > H(ds,d₀) >= H(ds,dt).

Both Newtonian cell-cell tests pass for that actual leaf. D₂ >= D₁, so Ewald also
passes if still pending; use either the certified separate path or the coupled path.
This covers a receiver leaf coarser or finer than S without assuming equal levels.
It also shows why testing all potential leaf depths separately is unnecessary for
this baseline. The coarse cover deliberately overexports detail near a destination;
no optimality or communication-volume claim follows.

At a coarser internal sink, test ordinary acceptance first. If a multipole-only
terminal cannot be consumed there, carry it unchanged, with its pending-term state,
to occupied sink children. This overrides source-opening/open-larger heuristics for
such terminals. Never request nonexistent source children or treat the terminal as
a particle leaf. At a true sink leaf, the certificate guarantees a cell-cell route.
If a runtime predicate still rejects it, flag/count a missing-detail or numerical
certificate failure and fetch the complete source subtree through the reference
path, retaining pending-term state without readding an accepted term, or stop if
that path is unavailable. Do not silently invent particles or omit
the interaction. Zero such failures is required before claiming export sufficiency.

The owned source tree remains complete. For other rejected pairs, refine available
internal sources or sinks; at true leaf pairs use the periodic resolution above
and then the ordinary Newtonian interactions. A true source particle leaf always
ships all its particles unless it satisfies the same multipole-only certificate.

### 6. Export records and completeness

An export header binds source owner, destination, generation, geometry version,
P, dMax, θ, h_G, moment order/layout, and the destination-cover descriptor/version.
Peers must agree on these values before consuming the view. Counts and offsets
are range-checked; serialize field contents, never arena pointers. Required node kinds:

| Kind | Meaning and required payload |
|---|---|
| Empty | Explicit zero source count; zero moments/radius; no children/particles. |
| Internal | Nonzero count, geometry, radius, moments, occupied-child mask and references to every occupied child. Empty child slots are explicit. |
| ParticleLeaf | True physical leaf, nonzero count, geometry, radius, moments and complete particle payload; ordinary/coincident/depth-limit reason recorded. |
| MultipoleTerminal | Nonzero count, geometry, radius, moments and terminal-certificate context; descendants/particles deliberately omitted, never declared physically empty. |

Particle payloads retain native positions and species labels, plus generation-qualified
particle identity `(owner, source index)`. Mass is derived from the configured
species masses; no per-particle mass array is stored or transferred. This is sufficient for audit/self identification
while that frozen ordering lives; persistent particle IDs may also be retained.
Imported particles are read-only sources, never sinks, migration candidates, or
members of the destination-owned moments.

Start export construction from each nonempty owner root. At a certified node emit
a MultipoleTerminal; otherwise emit all occupied children recursively, or complete
particles for a true leaf. Retain the ancestor scaffold. Do not discard a distant
source merely because it is outside a fine halo. An explicit empty source root
still completes the peer/generation protocol. An empty destination needs completion
metadata but no force payload. There is no six-peer or rank-adjacency restriction.

For each owner and each sink, the consumed source nodes/particle ranges form a
disjoint covering frontier of that owner's particles: consume a parent OR its
covering descendants, never both. Keep this invariant separately for Newtonian
and Ewald terms when they resolve at different levels. The union over owners is
complete because particle ownership is disjoint. Same geometric key on different
owners is not a duplicate; duplicate delivery of the same owner/generation record
must be idempotent or rejected. Missing expected records mean incomplete transport,
not empty cells. Walk only after the required generation view is complete.

For self-exclusion, preserve the inspected implementation's algebra: ordinary PP
includes the primary self term, and each owned sink initializes its potential with
`mass * selfPhi() / h_G` once to cancel it. Primary self acceleration is zero. Do not also
skip that PP term without adjusting the correction. Retain the periodic self-image
and background contributions. Distinct coincident particles remain interacting
sources; equal coordinates or equal geometric keys cannot identify self.

### 7. Construction phases and optional tightening

1. Freeze source ownership/state and distribute the destination cover descriptors,
   effective parameters and empty-rank flags. No remote multipoles are needed for this.
2. Build each owner's full unit-box tree. The baseline export predicate uses only
   destination geometry and source cell identity, so view selection can occur during
   descent. Fill node moments/radii/counts from the single owned upward pass.
3. Serialize completed destination views and exchange them asynchronously. A lower-ID
   pair initiator may coordinate readiness; both directed views still have distinct
   contents. All expected owners, including empty ones, must complete or report failure.
4. Validate headers and record completeness; traverse only owned sinks against the
   owned root plus every nonempty remote root, initially with both terms pending.
   Retire a generation only after consumers and any reference fetches finish.

Start with selection after the build for simplicity; moving it into descent/upward
work changes scheduling, not the contract. Never rebuild physical source moments
independently for each destination.

Optional tighter exports may replace C_d₀ by a summary frontier from the destination's
**owned** tree, with geometry and certified minimum leaf depth for each covered
region, or by an actual leaf summary. That requires a separate summary exchange
after owned topology is available, before export selection. Every advertised region
must still contain its possible leaf centers, and every possible leaf must be covered.
Do not infer a deeper minimum leaf depth just from particle density or an empty
received record. The baseline needs no target summary and has no circular dependency
on the remote source view being constructed.

### 8. Worked cases

- **One cell, two owners.** At depth 3, [0,1/2)³ has center (1/4,1/4,1/4).
  An ownership cut at x=0.3 can give owner A a mass-2 particle at x=1/8 and owner B
  a mass-3 particle at x=3/8, with y=z=1/4. Both trees have the same geometric cell
  with different source identities and partial moments. Its two contributions have
  total mass 5. It is an internal cell if dEwald>3, regardless of those small counts.
  Never deduplicate the two cells or add either imported moment to the owned sink tree.

- **Coarse/fine, including a coarser receiver leaf.** With θ=1/2 and h_G=0.01,
  dEwald=11: R(10)=0.09375 fails the 1/12 limit and R(11)=sqrt(6)/32 passes.
  Depths 12 and 18 have R=sqrt(3)/32 and sqrt(3)/128, so H=5sqrt(3)/64,
  approximately 0.1353. Centers (1/32,1/32,1/32) and (17/128,5/128,5/128)
  have distance sqrt(171)/128, approximately 0.1022. An equal-depth-18 threshold
  sqrt(3)/32, approximately 0.0541, would wrongly certify this mixed pair using
  only geometric limits. The correct certificate fails in either direction. In
  particular, a depth-12 receiver leaf cannot be forced to refine to rescue a bad
  depth-18 terminal: the sender must provide detail. Conversely, a certified terminal
  rejected by an internal receiver may be deferred until its actual leaf. Measured
  radii can allow earlier acceptance; failure of the geometric certificate alone
  does not prove failure of the actual MAC.

- **Periodic seam and outward cells.** Let the destination's x extent wrap as
  [63/64,1) union [0,1/64). At depth 12 the x cover is [15/16,1) union [0,1/16),
  whose cell centers 31/32 and 1/32 both lie outside the ownership extent. Keep them
  covered. Centers at 63/64 and 1/64 on opposite sides of the seam have minimum
  separation 1/32, not 31/32. Wrap padded covers and avoid duplicate source records;
  the Ewald operator supplies the images. A particle exactly at a midpoint goes
  right; a particle at the unit seam canonicalizes to zero and is owned once.

- **Empty rank.** It publishes count zero and an explicit empty root/completion,
  creates no sinks, and sends no nonzero moments. Peers do not wait for nonexistent
  particle records. A nonempty owner far from another rank still sends a coarse
  covering representation; spatial separation is not an empty-rank test.

- **Coincident particles.** Take N=32 and bucketSize=31 at (1/2,1/2,1/2), with the same
  θ and h_G as above. Equality at the first x,y,z midpoints selects right children;
  later cuts follow exact coordinate comparisons along one nonempty chain to
  depth 11. The count exceeds the bucket limit, so the cell terminates by the
  coincident exception. At that
  depth the fixed center is (17/32,17/32,9/16), giving r=sqrt(6)/32, not zero.
  Retain the nonzero dipole about that center, all 32 source particles, and their
  distinct identities. Cancel only each sink's own primary self contribution.
  Noncoincident particles unresolved at dMax similarly become overflow leaves;
  they are never divided by an artificial equality-plane/count split.

### 9. Completion, implementation obligations, and evidence

Step 1 is complete as a design artifact. Geometry reconstruction, cloud enclosure,
the mixed-depth terminal rule, complete-source coverage, descendant image consistency,
and leaf-pair periodic termination have arguments above. These are mathematical
guarantees conditional on the stated frozen ownership/coverage and conservative
arithmetic invariants; they are not numerical-test results or finite-order error bounds.

Source inspection found work that subsequent steps must implement: integer geometric
identity independent of domain keys; fixed centers/dipoles and the inclusive bucket limit;
native-precision/outward radius arithmetic preserving descendant-center enclosure;
overflow/count/arena support; explicit imported terminal kinds; sink deferral for terminals;
generation-qualified transport and counted reference fallback. Existing `leaf` alone
cannot represent the three nonempty node kinds. Existing CPU/CUDA depth and allocation
limits cannot be assumed compatible merely because geometry is mathematically defined.

Reviewed source entry points: [tree builder](../../src/gravity/tree_build.cpp),
[distributed builder and Ewald depth](../../src/gravity/global_fmm/distributed_tree.cpp),
[Newtonian acceptance](../../cosmictiger/gravity/fmm_acceptance.hpp),
[periodic traversal](../../cosmictiger/gravity/periodic_fmm.hpp),
[sink/source traversal](../../cosmictiger/gravity/kick_walk.hpp),
[fixed positions](../../cosmictiger/math/position.hpp),
[fixed arithmetic](../../cosmictiger/math/fixed.hpp),
[Fixed32 boxes](../../cosmictiger/gravity/fmm_box.hpp),
[node representation](../../cosmictiger/gravity/tree.hpp),
[arena sizing](../../src/gravity/tree_arena.cpp), and
[direct self convention](../../src/gravity/gravity.cpp).

Validation performed: source/reference inspection, document review and whitespace
checks only. No solver
files changed; no builds, unit/integration tests, benchmarks, GPU runs, or remote
logins were performed. Steps 2 through 7 remain unimplemented by this task. Tightened
target summaries, compression, overlap scheduling and performance tuning remain
optional optimizations after correctness checks; no speedup or zero-fallback result
is claimed here.

Next, only when requested: [02_local_geometric_tree.md](02_local_geometric_tree.md).
