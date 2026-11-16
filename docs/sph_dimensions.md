# Spatial dimensions for SPH

`hydro.dimensions` selects 1D, 2D, or 3D SPH at startup; the default is `3`.
The same executable supports all three choices. For example:

```sh
./release/cosmictiger --run.scenario=hydroWave --hydro.dimensions=2 \
  --run.gasParticleCount=4096 --run.darkMatterParticleCount=0 \
  --gravity.enabled=false --gravity.cuda=false \
  --chemistry.enabled=false --starFormation.enabled=false
```

Use `--hydro.dimensions=1` for 1D or `--hydro.dimensions=3` for 3D. The same
setting is accepted as `dimensions=2` in an INI `[hydro]` section. Values outside
1–3 are rejected. Dimensions are selected before particle allocation and remain
fixed during a run, including checkpoint restart.

The former `COSMICTIGER_DIMENSION_COUNT` CMake selector is retired. A cached
value of `3` is removed automatically; a different value produces migration
instructions. Remove it with `cmake -U COSMICTIGER_DIMENSION_COUNT` and pass
`--hydro.dimensions` when launching the executable. Physics capabilities may
still be selected at build time, but a build with gravity and chemistry can
run lower-dimensional SPH when those physics are disabled at runtime.

Lower-dimensional runs require `hydro.enabled=true`,
`run.darkMatterParticleCount=0`, `gravity.enabled=false`, `gravity.cuda=false`,
`chemistry.enabled=false`, and `starFormation.enabled=false`. They support
hierarchical integration with midpoint SPH. Hydro wave, Sod and blast initializers
are available in all dimensions; Kelvin–Helmholtz, Gresho and Rayleigh–Taylor
require at least two. Disc, self-gravitating star, Jeans wave, cosmology and other
gravity scenarios remain three-dimensional. Central gravity is rejected in lower
dimensions. The uniform y body force requires D >= 2.

Rayleigh–Taylor uses a fixed analytic acceleration toward the midplane, along
y in 2D and z in 3D. The existing `scenario.rayleighTaylor.zPlaneAcceleration`
key sets its magnitude in both dimensions. Its preset and scenario default
disable self-gravity, so a gravity-free build is sufficient. The mirrored
light/heavy layers, hydrostatic pressure and velocity perturbation use that
same vertical axis; particle counts round up to 24 n² in 2D and 96 n³ in 3D.
The existing glass input format is three-dimensional.

These are Cartesian calculations, not radial reductions of spherical or
cylindrical flow. Gamma remains the configured thermodynamic value. Cooling
and star formation require three-dimensional volumetric source physics.

## Geometry and storage

`spatialDimensions()` returns the selected runtime D. `kDimensions` is the fixed
XYZ capacity, three. Particle velocity vectors, transport records and workspace
vectors retain that capacity; only the active D position-component arrays are
allocated. Only active coordinates participate in hydrodynamics, kernel
normalization, lattice counts, and domain and SPH tree partitioning. Inactive
coordinates read as 0.5 and inactive velocities as zero.

The CPU SPH entry points dispatch once per leaf pass to source-local templates
for D=1, 2, or 3. CUDA dispatch selects the corresponding compiled kernel before
launch, and particle drift follows the same pattern. Shared template bodies keep
dimension-dependent loop bounds and scaling fixed within those kernels; common
control flow, storage and public interfaces remain shared. The runtime selection
also chooses density-kernel normalization and coefficients. The 3D path retains
the existing gravity-kernel table construction and arithmetic.

Silo velocity fields include only active components: `vx` in 1D, `vx`/`vy` in
2D, and `vx`/`vy`/`vz` in 3D, for both native and converted units.

Checkpoints encode D in physics-code bits 28–29 (zero means the historical
3D format). Bit 30 identifies compact lower-dimensional particle records.
Checkpoint serialization packs active components while in-memory records retain
XYZ capacity. Restart with a different `hydro.dimensions`, or an old
lower-dimensional XYZ-padded checkpoint, is rejected before particle payloads
load. Three-dimensional records and their layout tag are unchanged. Output
includes `dimension_count` and the compatibility field `build.dimensionCount`,
both reporting the selected runtime dimension. CGS density has units g/cm^D;
the entropy conversion uses that density scale. Cosmological-unit output is
unavailable below three dimensions.

For fixed32 positions, compact checkpoint position-plus-velocity payloads use
8D bytes per particle: 8 bytes in 1D, 16 in 2D, and 24 in 3D. For fixed64 the
payload uses 12D bytes. In memory, fixed32 position-plus-velocity storage uses
4D + 12 bytes per particle, and fixed64 uses 8D + 12 bytes: active position arrays
plus the XYZ velocity vector. Scalar particle attributes, transport records and
geometry/workspace overhead are additional.

## Equations

The full compact support radius remains H, with q = r/H. The following are
D-dimensional derivations of the existing variational formulation from
[Hopkins (2013), equations 8–13](https://arxiv.org/abs/1206.5006) and
[Price (2012)](https://arxiv.org/abs/1012.1885).

\[
W_D=H^{-D}w_D(q),\quad
S_{D-1}\int_0^1 w_D(q)q^{D-1}\,dq=1,\quad
V_D=S_{D-1}/D.
\]

\[
\nabla_i W_D=H^{-(D+1)}w_D'(q)\hat r_{ij},\quad
\partial_H W_D=-H^{-(D+1)}[D w_D+q w_D'].
\]

The support constraint is \(V_DH_i^Dn_i=N_{\rm target}\), so its dimensionless
target is \(\sum_j w_D=N_{\rm target}/V_D\), with \(V_D=2,\pi,4\pi/3\).
The dimensionless grad-H corrections are

\[
\Omega_i=-\frac{\sum_jq_{ij}w_D'}{D\sum_jw_D},\qquad
\zeta_i=\frac{D\sum_jm_jw_D+\sum_jm_jq_{ij}w_D'}{\sum_jq_{ij}w_D'}.
\]

Held supports retain zero zeta. With
\(Q_{ij}=(P_i m_i/\rho_i^2)(m_j-\zeta_i)\), pressure forces remain
\(m_i\dot v_i=-\sum_j[Q_{ij}\nabla_iW(H_i)+Q_{ji}\nabla_iW(H_j)]\).
Entropy is unchanged by reversible pressure work; viscosity and conduction
retain their compatible pair force/heating or thermal exchange forms.

The effective radius remains twice the one-coordinate standard deviation.
Its conversion from H depends on D and the kernel. The viscosity width
calibration recovers H/2 for the cubic spline in every dimension. Wendland
polynomial degrees are 1/4/7 in 1D and 2/5/8 in 2D/3D; spline and Ferrers
families retain their shapes and receive dimension-dependent normalization.

The traceless strain subtracts `divergence / D`. Vorticity is zero in 1D,
a scalar magnitude in 2D, and the usual vector magnitude in 3D.

## Cosmological convention

At the user's direction, cosmological scale-factor conversions retain their
existing 3D formulas, including `5 - 3*gamma`. Lower-dimensional SPH uses a=1;
this change does not introduce lower-dimensional cosmology.

## Validation targets

The dimensional mathematical tests exercise runtime selections of D=1/2/3 in
one executable: kernel normalizations, float tables, support derivatives, cubic
viscosity lengths, discrete variational forces against finite differences of
thermal energy, and exact affine-gradient reconstruction. Rayleigh–Taylor tests
exercise both D=2 and D=3 using parameterized cases in one executable. Application
checks must also cover dimensional command-line validation, lower-dimensional
scenario evolution, and checkpoint restart with matching and mismatched dimensions.

Runtime-selection validation (November 16, 2026):

- 82 focused CPU CTests passed, covering dimensional mathematics, SPH, drift,
  2D/3D Rayleigh–Taylor and Kelvin–Helmholtz applications, and gravity-disabled
  diagnostics. The allocation regression also rejects dimension changes while
  particle storage exists, including retained capacity with zero particles.
- Both lower-dimensional application suites passed using the same release
  executable with CPU backends: CLI/INI selection, invalid-input rejection,
  scenarios, Silo output, compact checkpoints, and restart. Restart positions
  and thermodynamics matched exactly; the largest velocity difference was
  `4.7e-10` in 2D.
- Two-locality CPU runs passed in 1D and 2D, including compact checkpoint
  decoding, particle counts, finite thermodynamics, and momentum checks.
  All 76 Python build-graph checks passed.
- The full CUDA release compiled and device-linked for SM86, and the CUDA
  regression translation unit compiled. GPU execution was deferred while
  War Thunder was ON. These are focused checks, without a full-suite,
  sanitizer, performance, or long-time convergence claim.

The historical results below describe the earlier compile-time implementation;
they do not by themselves validate the runtime-selection changes.

Focused implementation validation (November 5, 2026):

- D=1/2/3 mathematical tests and 33 CMake capability cases passed.
- 1D and 2D CPU scenario suites passed with all five viscosity modes and restart.
- The 2D CUDA scenario suite passed; the five 1D CUDA translation units compiled.
- A two-rank 2D CPU wave run and a rebuilt 3D CPU wave run passed.
- 51 existing 3D kernel/volume/viscosity checks and four FLOP checks passed.
- Checkpoints tagged with a different dimension were rejected explicitly.

Application runs used traditional integration and short smoke durations. These
checks do not establish long-time convergence, global-integrator runtime coverage,
1D GPU runtime coverage, or a full-suite/sanitizer result. The initial implementation used XYZ-padded storage; the subsequent compact-storage
validation is recorded separately below.

Compact-storage validation (same date): CPU scenario suites passed in 1D and 2D,
including Silo output, restart, and explicit rejection of the earlier padded
layout. Two-rank runs and native checkpoint decoding passed in both dimensions.
The 2D CUDA suite passed with both GPU hydro and GPU drift. All six relevant
1D CUDA translation units, including drift, compiled; 1D GPU runtime is untested.
The dimensional mathematical/configuration CTests passed. Measured checkpoint
reductions match the removed coordinate and velocity fields exactly: 2,048 bytes
for 128 1D particles and 8,192 bytes for 1,024 2D particles (fixed32).

Compared with the preceding padded-layout hydro-wave runs, active coordinates
and entropy matched exactly in both dimensions. The maximum velocity difference
was zero in 1D and approximately 7e-10 in 2D, within float accumulation roundoff.

The rebuilt 3D application passed a fresh wave run and loaded a checkpoint from
the preceding padded-storage implementation. All 13 focused 3D checkpoint-state,
CPU drift, and particle-selection tests passed. These are focused checks on the
combined working tree, not full-suite or sanitizer coverage.
