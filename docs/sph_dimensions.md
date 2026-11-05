# Spatial dimensions for SPH

`COSMICTIGER_DIMENSION_COUNT` is a CMake cache string, default `3`. Only `1`,
`2`, and `3` are accepted. Values below three require SPH and reject every
compiled gravity solver, even if gravity would be disabled at runtime.
For example:

```sh
cmake -S . -B build/sph_2d \
  -DCOSMICTIGER_DIMENSION_COUNT=2 \
  -DCOSMICTIGER_ENABLE_FMM=OFF \
  -DCOSMICTIGER_ENABLE_PMMM_PP=OFF \
  -DCOSMICTIGER_ENABLE_PMMM_FMM=OFF \
  -DCOSMICTIGER_ENABLE_PM=OFF \
  -DCOSMICTIGER_ENABLE_P3M_PP=OFF \
  -DCOSMICTIGER_ENABLE_P3M_FMM=OFF \
  -DCOSMICTIGER_ENABLE_CHEMISTRY=OFF \
  -DCOSMICTIGER_ENABLE_STARS=OFF
```

Lower-dimensional runs use `run.darkMatterParticleCount=0`. They support
traditional or global SPH integration. Hydro wave, Sod and blast initializers
are available in all dimensions; Kelvin–Helmholtz and Gresho require at least
two. Disc, self-gravitating star, and the existing z-directed Rayleigh–Taylor
initializer remain three-dimensional. Central gravity and the z-plane force
are rejected in lower dimensions. The uniform y body force requires D >= 2.
The existing glass input format is three-dimensional.

These are Cartesian calculations, not radial reductions of spherical or
cylindrical flow. Gamma remains the configured thermodynamic value. Cooling
and star formation require three-dimensional volumetric source physics.

## Geometry and storage

`kSpatialDimensions` is the configured D. `kDimensions` retains its historical
value three for padded XYZ storage, transport, and output records. Inactive
coordinates are initialized to 0.5 and inactive velocities to zero. Domain and
SPH tree partitioning use only active axes; hydro dot products use
`math::Vector<Real, kSpatialDimensions>`. Corrected velocity and acceleration
moments use `math::Matrix`, and their inverse uses `math::SymmetricMatrix`,
both of order D. Padded gradient outputs have zero inactive entries.

Checkpoints encode D in physics-code bits 28–29 (zero means the historical
3D format). A different-dimensional executable rejects the header before
loading particle payloads. Output includes `dimension_count` and
`build.dimensionCount` metadata. CGS density has units g/cm^D; the entropy
conversion uses that density scale. Cosmological-unit output is unavailable
below three dimensions.

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

`sph_dimension_1`, `sph_dimension_2`, and `sph_dimension_3` test all supported
kernel normalizations, float tables, support derivatives, cubic viscosity
lengths, discrete variational forces against finite differences of thermal
energy, and exact affine-gradient reconstruction. `check_sph_dimensions.py`
checks default/invalid dimensions and rejects each gravity solver independently.
Existing application fixtures assume 3D and remain registered in 3D builds.

Focused implementation validation (November 5, 2026):

- D=1/2/3 mathematical tests and 33 CMake capability cases passed.
- 1D and 2D CPU scenario suites passed with all five viscosity modes and restart.
- The 2D CUDA scenario suite passed; the five 1D CUDA translation units compiled.
- A two-rank 2D CPU wave run and a rebuilt 3D CPU wave run passed.
- 51 existing 3D kernel/volume/viscosity checks and four FLOP checks passed.
- Checkpoints tagged with a different dimension were rejected explicitly.

Application runs used traditional integration and short smoke durations. These
checks do not establish long-time convergence, global-integrator runtime coverage,
1D GPU runtime coverage, or a full-suite/sanitizer result. XYZ storage and transport
remain padded to three components.
