# ROCKSTAR halo finding

For local snapshot browsing, filtering, spatial plots, host/subhalo inspection,
and CSV selections, see [Halo Explorer](halo_viewer.md).

The implementation follows the phase-space hierarchy, halo assignment, unbinding,
and property definitions in [Behroozi, Wechsler & Wu, arXiv:1110.4372v3](https://arxiv.org/html/1110.4372v3).
The paper is the algorithm reference; this code is an independent implementation,
with the simulation's gravity softening and the output conventions below.

## Algorithm and paper mapping

- Sections III.2–III.3: recursively normalize position and velocity dispersions,
  choose the nearest-neighbor linking-length quantile with `f=0.7`, and refine
  groups down to ten particles. Merge statistically indistinguishable seeds with
  the paper's squared significance threshold of 200. Assign particles with the
  seed halo's dynamical radius and velocity dispersion. Degenerate dispersion
  and tied linking distances terminate safely.
- Sections III.4–III.5.1: choose the center from the hierarchy level minimizing
  its Poisson position uncertainty. Build an internal hierarchy using the closest
  larger halo in phase space, before unbinding. Snapshot core-particle history
  can preserve the previous host when current exclusive counts exchange order.
  Catalogue host indices are subsequently determined by SO containment and are
  separate from the internal mass hierarchy. Core velocity uses particles within
  `0.1 Rvir`; if that aperture is empty, the nearest particle is the explicit
  discrete fallback. Bulk velocities include the particles in the measured system.
- Section III.5.2: measure spherical overdensities from cumulative radial counts
  about the halo center. All particles on an equal-radius shell are treated
  together. The outermost qualifying shell defines an integer mass, with
  `R = cbrt(3 N / (4 pi n_threshold))`; membership and radius therefore describe
  the same mean density. Host masses include internal descendant particles that
  actually lie inside the aperture.
- Section III.5.3 and Appendix B: freeze potentials before the single unbinding
  pass. Child particles rejected from their own halo can be reconsidered by a
  host. Major mergers at a ratio of at least 1:3 use the combined potential and
  bulk velocity. Require at least ten retained particles and a retained fraction
  of at least one half. Recompute reported energies from the final measured set.
- Section III.5.4: compute circular velocity from the enclosed count, fit an NFW
  profile using up to 50 radial bins with at least 15 particles per bin and a
  shell-integrated likelihood (the concentration `c = rmax / rs` is searched in
  `[1, e^16]` and reported only when the one-sigma profile-likelihood interval
  of `ln c` lies inside that window), and calculate the Klypin radius from
  `Vmax`, `Mvir`, and `Rvir`. Compute angular momentum, both spin parameters,
  offsets, and `T/|U|`. Shape tensors exclude substructure and use an orthogonal
  symmetric eigensolver. Snapshot descendants use the largest exclusive particle
  overlap.

## Internal gravity

Halo binding uses an isolated Barnes–Hut potential, not the periodic simulation
FMM potential. The tree applies Appendix B's monopole potential-error criterion
with a target relative error of 0.04 and opens cells whose source softening support
could overlap a sink. Direct interactions use the selected compact gravity kernel
and the fixed comoving gravity support. A particle's self term is excluded.

Positions are comoving code lengths and input velocities are peculiar velocities.
For equal-mass particles, `gravity.gm` is the code-unit gravitational constant times
one particle mass. The binding calculation uses

```
Phi_i = -gravity.gm / a * sum(j != i, softened_inverse_distance(i,j))
e_i   = 0.5 * |v_i - v_bulk|^2 + Phi_i
T     = 0.5 * sum_i |v_i - v_bulk|^2
U     = 0.5 * sum_i Phi_i
```

The potential source set and bulk velocity are those of the selected halo or
combined major-merger system. Reported catalogue energies are recomputed for the
bound particles inside the record's virial aperture. Physical angular momentum
uses `a * r_comoving cross (v - v_bulk)`. Circular velocity uses
`V_c^2 = gravity.gm * N(<r) / (a*r)`.

## Catalogue conventions

The existing `RockstarRecord` field order and two-byte half-field storage are
retained. Half fields now contain standard IEEE 754 binary16 bits, with signed
zero, subnormals, and round-to-nearest, ties-to-even conversion in the default
floating-point environment. Conversions use C++23 `std::float16_t` in a small
host-only library; the simulation and CUDA sources can remain C++20. The host
compiler and standard library must provide `<stdfloat>` and `std::float16_t`.
Every finite input outside `[-65504, 65504]` is rejected before conversion,
including values that could round back to the largest finite half. Catalogue
packing also rejects nonfinite properties and reports the offending field name.

Both snapshot and lightcone files have a companion `<filename>.format` sidecar:

```
format_version 2
float16 ieee754_binary16
record_size 128
record_count <number of records>
byte_order little
virial_density bryan_norman_flat
so_density_reference critical
```

The writer records the actual native record size and byte order (`little` or
`big`). The sidecar is published only after the data file closes successfully,
and an old sidecar is removed before rewriting a data file. Snapshot readers
must also require `rockstar_hosts.complete` to establish completion of all ranks.
These sidecars identify the half encoding; the remaining raw C++ record layout
is still native ABI dependent. Older files without the sidecar have an
unidentified encoding and must not be silently interpreted as IEEE binary16.
The former encoding used exponent zero and exponent 31 differently; any older
catalogue that needs preservation must be decoded with its original codec before
conversion, with the new finite-range checks applied.

Flat backgrounds with nonnegative Λ retain the Bryan–Norman flat fit, and open
backgrounds with zero Λ retain its open fit. Their `so_density_reference` is
`critical`: `m200`, `m500`, and `m2500` mean **M200c, M500c, and M2500c**.
The fit's published calibration covers matter fractions between 0.1 and 1;
using it at smaller fractions or with significant radiation remains an
extrapolation. See [Bryan & Norman, equation 6](https://arxiv.org/html/astro-ph/9710107v1).

For closed models, negative Λ, and other curved backgrounds with Λ, the complete
catalogue instead uses `virial_density 200_mean_matter` and
`so_density_reference mean_matter`. The legacy `mvir`/`rvir` slots then contain
the mass and radius at **200 times mean matter density**, and the other mass
fields mean **M200m, M500m, and M2500m**. The 500-density shape fields use the
same mean-matter reference. This is an explicit measurement convention, not a
prediction of virialization in a collapsing universe. It stays finite when
critical density vanishes at turnaround. Background-relative mass definitions
are described in [ROCKSTAR section III.5.2](https://arxiv.org/html/1110.4372v3#S3.SS5.SSS2).
The convention is chosen for the entire cosmology, so it does not change while
a halo crosses turnaround.

The exporter accepts older five-field sidecars as the historical
Bryan–Norman/critical-density convention. New catalogues carry both density
fields, and the viewer labels masses and radii from them; the binary record
layout and raw field names remain unchanged. Readers that require exactly the
old five metadata fields must be updated before reading new catalogues.

| Fields | Convention |
| --- | --- |
| `x,y,z` | Absolute halo peak position, comoving Mpc/h |
| `mvir,m200,m500,m2500` | Solar masses/h |
| Radii, position dispersions, position offsets, shape axes | Comoving kpc/h |
| Core/bulk velocity, velocity dispersions, `vcircMax` | km/s |
| `jx,jy,jz` | Physical angular momentum in (solar masses/h)(Mpc/h)(km/s) |
| `e` | (Solar masses/h)(km/s)^2 |
| `toW` | Nonnegative `T/|U|` |
| `lambda,lambdaB`, axis ratios | Dimensionless |
| `zValue` | Cosmological emission or snapshot redshift |

Writers output one inclusive record per halo. `id.sys` is one when internal
substructure contributes to the candidate system and zero for a leaf halo.
`pid=-1` denotes a distinct halo; other values index its containing host within
that FOF result. `id.fofId` identifies the outer FOF work unit and `id.haloId` the
halo within it. Snapshot persistent halo IDs are distinct from those packed record
indices.

The legacy `GroupEntry` output remains available: `com` is the measured center of
mass, its legacy inertia components are centered there, and halo radial profiles
are measured about the density peak. Its existing serialization is retained.
Snapshot output also contains:

- `groups.N/rockstar.RANK.dat`: raw `RockstarRecord` entries for that rank.
- `groups.N/rockstar_hosts.RANK.dat`: persistent halo ID and persistent host ID.
- `groups.N/rockstar_descendants.dat`: previous halo ID, descendant ID, and exclusive
  particle-overlap count; missing descendants use `NO_GROUP` and count zero.
- `groups.N/rockstar_hosts.complete`: completion manifest published after the rank
  outputs and descendant file succeed. Interrupted catalogues are not loaded as
  temporal history.

Lightcone output retains `lc.groups.EPOCH.RANK`. A FOF group is analyzed at its
center's emission scale factor. Absolute peak positions determine the individual
record redshifts. Particle velocities are converted at their emission epoch using
`tau_emission = tauMax - |x|` and a first-order scale-factor interpolation from the
drift reference epoch.

## Physical and numerical boundaries

Halo and lightcone analysis currently require equal-mass dark matter particles;
hydrodynamic runs are rejected by option validation. An SO search can only recover
mass represented in the outer FOF candidate list. The virial fit is calibrated for
flat matter–Lambda or open matter-only cosmologies; using the actual matter
fraction with radiation or other curvature/Lambda combinations is an extrapolation.

The paper's temporal correction requires an earlier catalogue. Fresh snapshots
and lightcone groups use spatial/phase-space information alone. Particle-overlap
descendants do not implement the separate Consistent Trees algorithm that repairs
disappearances and reappearances. Lightcone analysis treats each finite FOF group
at one representative epoch, rather than evolving all members to a common time.

NFW fits search the concentration `c = rmax / rs` in `[1, e^16]`. The lower edge
mirrors reference ROCKSTAR, whose `calc_scale_from_bins` (`nfw.c`) keeps `Rs`
strictly inside its binned radial range and reports the clamped value; treating
a scale radius outside the fitted profile as unresolved, and the `e^16` upper
edge, are this project's conventions, so a nonzero `rs` always satisfies
`rs <= rmax`. The Klypin limit
`c >= 2.1626` (Eqs. (5)-(6)) is not used as the lower edge because it expresses
the degeneracy of the `Vmax` inversion, not of the density profile; it would zero
the 33%/27%/13%/3% of true `c = 2/3/5/10` halos at 30 particles whose likelihood
optimum lies below it (fits the one-sigma test already rejects at that size) and
about 65% of true `c = 2` fits at 300 particles. A fit is
reported only when the total negative log-likelihood rises by at least `1/2` from
its optimum to both window edges, i.e. the one-sigma profile-likelihood interval
of `ln c` lies strictly inside the window (Wilks 1938); fits with fewer than two
resolved radial bins or degenerate shells return zero as before. Toward `c -> 0`
every NFW profile tends to `rho ~ 1/r` and the likelihood is flat in `ln c`, so a
30-44 particle halo (an exactly determined two-bin fit) or an unrelaxed shallow
profile could previously place its optimum at a tiny interior concentration,
numerically distinct from the window edge yet statistically identical to it; such
fits produced scale radii of up to `1e6 rmax` and aborted catalogue writes through
the binary16 range check.

Expect zero `rs` for many halos below roughly 100-150 particles with `c <~ 3`:
Poisson-sampled NFW halos are reported in about 55%/85%/96%/99% of cases for
`c = 3` at 30/60/100/150 particles, 34%/56%/72%/85%/98% for `c = 2` at
30/60/100/150/300, 79%/94%/98% for `c = 5` at 30/45/60, 96% for `c = 10` at 30,
and essentially always for `c >= 5` from 100 particles or `c >= 2` from 300; a
reported `rs` never exceeds about `0.43 rmax` at 30 particles. Halos with `c < 2`
need of order 1000 particles to exclude `c = 1` at one sigma: `c = 1.5` is
reported in about 53%/73%/98% of cases at 150/300/1000 particles. The surviving
low-concentration small-halo sample is biased high (median fitted-to-true `c`
about 1.75 for `c = 2` at 30 particles, 1.07 at 100; 1.33 for `c = 3` at 30). The
one-sigma test reports an exact `c = 1` halo as resolved about 10% of the time at
30 particles, rising to the asymptotic 16% (half of `P(chi^2_1 >= 1)`, Chernoff
1954) for well-sampled halos, and the record carries no significance field, so a
one-sigma `rs` is indistinguishable from a decisive one.

A zero `rs` means unresolved; consumers should fall back to `rkly`, which
reproduces reference ROCKSTAR's fallback (`nfw.c` skips the fit below 100
analysed particles and stores the Klypin radius instead; Behroozi, Wechsler & Wu
2013, Sec. III.5.4, note only that it is more robust below 100 particles; this
project fits from 30). The threshold `kMinBoundaryLogLikelihoodRise` in
`rockstar::nfwFit` is a quality knob; lowering it reports more boundary-biased
small-halo fits but cannot reintroduce scale radii outside the profile. The
golden-section search assumes a unimodal likelihood in `ln c`; a profile with a
compact central cusp can have a second optimum near `c ~ 1e4`, and the fit may
then report that local optimum.
Independently of the fit, the catalogue writer maps any `rs` outside `(0, rmax]`,
or nonfinite, to zero before the strict half-precision encoding, so encoding `rs`
can only fail if `rmax` already has. Catalogues written before this change may
contain `rs > rmax`; treat them as unresolved there. The Klypin estimate returns
zero for inconsistent or nonfinite inputs; `Vmax=Vvir` uses the limiting
concentration `2.162581587`. Half-precision record fields retain 11 significand
bits for normal values; their maximum finite magnitude is now 65504, and the
smallest positive subnormal is `2^-24`.

## Regression coverage

`fp16_standard` and `fp16_fast_math` check all 65,536 binary16 representations,
every finite adjacent-value rounding midpoint and its binary64 neighbors for
both signs, signed zeros, subnormals, special values, and strict overflow checks.
The second test compiles the caller with fast-math while the standard conversion
library retains IEEE compilation settings. The runtime `rockstar_cpu` test also
round-trips generated snapshot and lightcone records and synthetic half boundary
patterns through the shared catalogue writer, checking all fields and format
sidecars, along with field-specific overflow/nonfinite errors.

`rockstar_properties` checks SO membership/density consistency, equal-radius ties,
multiple crossings, distant dense pairs, cosmology limits, rotated and degenerate
shape tensors, NFW recovery from deterministic mass quantiles, the synthetic
two-bin likelihood plateau, the `[1, e^16]` concentration window and the one-sigma
resolution criterion (pinned from both sides, with exact `rho ~ 1/r` quantiles up
to 5000 particles, an upper-edge cusp and deterministic Poisson samples), and
Klypin inversion. Both the standard and the `-ffast-math` build of the test run
these cases.
`rockstar_eigenpair` exercises the public eigenpair wrapper. `lightcone_epoch`
checks emission-time direction, equivalent reference epochs, and velocity scaling.

The runtime `rockstar_cpu` regression checks unbinding order, SO properties at
multiple epochs, the core aperture, internal/output host separation, temporal host
preservation, major-merger binding, degenerate finder inputs, the writer-level
scale-radius guard and a synthetic plateau halo's record, record translation
and rotation properties, and lightcone epochs. `groupsCatalogue_cpu` checks snapshot
catalogue publication, particle tags, and temporal history. `groupsCatalogue_mpi`
uses two MPI localities with split FOF inputs, checks deferred remote membership
delivery and every per-rank output, swaps FOF analysis owners between snapshots,
and verifies replicated host history and exact particle-overlap descendants. This
integration fixture supplies known FOF memberships directly; it does not test the
spatial FOF search itself. `bhPotential_cpu`
compares tree potentials against direct pair sums. CUDA variants of the finder and
potential tests are separate tests; passing CPU regressions does not establish
CUDA or distributed runtime correctness. These synthetic regressions do not replace
a cosmological mass-function or clustering comparison against reference ROCKSTAR.
