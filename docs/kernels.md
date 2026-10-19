# Smoothing and gravitational softening kernels

Gravity and SPH select independent spherical density kernels and radii. The
defaults are the radial cubic B-spline (M4) with 42 weighted neighbors for SPH,
and fixed comoving Ferrers F2 gravity for all particle classes, whose density
is proportional to `(1-q*q)^2`. The production three-dimensional table and
evaluators support the following 14 choices:

| `kernelFamily` | Allowed `kernelDegree` | Shape or interpretation |
| --- | --- | --- |
| `spline` | `3`, `4`, `5`, `6`, `7`, `8`, `9` | Centered cardinal B-spline of order `kernelDegree + 1`, radialized and normalized in 3D |
| `wendland` | `2`, `5`, `8` | The 3D Wendland C0, C2, and C4 kernels, respectively |
| `ferrers` | `2`, `4`, `6`, `8` | Ferrers F1 through F4: density proportional to `(1-q*q)^(kernelDegree/2)` |

The degree describes the density polynomial, not the gravitational potential or
force. It is the polynomial degree on each span for a spline. There is no
degree-nine Wendland or Ferrers member in these families: their next members
have degrees eleven and ten. Unsupported family/degree combinations are rejected.

For example, these INI sections choose a quintic spline for gravity and Wendland
C4 for SPH:

```ini
[gravity]
kernelFamily = spline
kernelDegree = 5
effectiveRadius = 0.01876988456953534
timestepEta = 0.20401599335954483

[hydro]
enabled = true
kernelFamily = wendland
kernelDegree = 8
neighborNumber = 120
cfl = 0.22821773229381923
```

The corresponding command-line keys are `--gravity.kernelFamily`,
`--gravity.kernelDegree`, `--hydro.kernelFamily`, and `--hydro.kernelDegree`.
The example neighbor target is a configuration example, not a recommended value
for all problems or a minimum required for Wendland kernels.

## Effective and support radii

The user-facing effective radius is

\[
R_{\mathrm{eff}}=2\sigma,\qquad
\sigma^2=\frac13\int r^2 W(\mathbf r)\,d^3r,\qquad
\int W(\mathbf r)\,d^3r=1.
\]

This preserves the existing `2 sigma` numerical length scale; the radius name
introduces no factor-of-two conversion. It is the resolution-based convention of
[Dehnen and Aly (2012), section 2.1](https://arxiv.org/html/1204.2471#S2.SS1).
Equal effective radii give the same leading long-wavelength smoothing term. They
do not equalize central potential depth, maximum force, particle noise, or all
higher moments. Gravity's effective radius is computed from its equivalent mass
density.

The compact support radius, \(R_{\mathrm{sup}}\), is the outer boundary of the
kernel. Internally the radial argument is `q = r/supportRadius` and the dimensional
density is `supportRadius^-3` times the tabulated shape. The table computes
`supportRadiusPerEffectiveRadius = supportRadius/effectiveRadius` from its
moments. Some 3D conversions are:

| Kernel | `supportRadius/effectiveRadius` |
| --- | ---: |
| Cubic spline | `sqrt(10/3) = 1.8257418584` |
| Wendland C2 | `sqrt(15)/2 = 1.9364916731` |
| Wendland C4 | `sqrt(39/8) = 2.2079402166` |
| Ferrers F2 | `3/2 = 1.5` |
| Ferrers F4 | `sqrt(13)/2 = 1.8027756377` |

The accepted radius options are:

| Purpose | Effective-radius key | Support-radius key | Units and kernel |
| --- | --- | --- | --- |
| Fixed comoving gravity | `gravity.effectiveRadius` | `gravity.supportRadius` | Initial reference spacings; gravity kernel |
| Imposed central force | `hydro.centralEffectiveRadius` | `hydro.centralSupportRadius` | Box units; gravity kernel |
| SPH minimum | `hydro.minEffectiveRadius` | `hydro.minSupportRadius` | Initial interparticle spacings; hydro kernel |
| SPH maximum | `hydro.maxEffectiveRadius` | `hydro.maxSupportRadius` | Initial interparticle spacings; hydro kernel |

Gravity's reference spacing is `numberOfParticles^-1/3` in box units, preserving
the existing radius-input convention. Cosmological SPH runs use
`numberOfParticles` particles per class, so this is the initial reference
spacing for one class, not the smaller mean spacing of the combined gas and
dark-matter population.

`gravity.softening` and `hydro.centralSoftening` remain aliases for the respective
effective radii, preserving their numeric values and units. Changing a kernel at
fixed effective radius changes its support radius using the factor above.

Both representations may be specified. An explicit effective-radius key takes
precedence over its legacy softening alias, and either takes precedence over a
support-radius key. Each ignored key produces a warning identifying that key.
This precedence applies across command-line and INI inputs; the command line
overrides an INI value for the same key. Defaults do not count as explicit
settings. To use support-radius input with a repository example INI, remove or
comment out its active softening alias or effective-radius setting.

SPH adapts and stores each particle's support radius because neighbor searches
must reach the support boundary. Its minimum and maximum bounds can use either
representation. The maximum support remains capped at half the box. Timesteps
use effective radius, computed as
`supportRadius / supportRadiusPerEffectiveRadius`; force and neighbor evaluation
use support radius. Checkpoint field order and stored radius units are unchanged.

The smoothing solve retains the kernel-weighted target

\[
N_{\mathrm{target}}=\frac{4\pi}{3}\frac{\rho R_{\mathrm{sup}}^3}{m}.
\]

This includes the self contribution. Consequently `hydro.neighborNumber` must
exceed the selected kernel's self floor, `(4*pi/3)*value(0)`, rather than one
universal threshold. Startup validation checks the floor in both double and
single precision. It is `32/3` for the default cubic. The target is not an integer
neighbor count. At equal effective radius, kernels with larger support generally
include more particles; at equal target, the achieved effective radius can differ.

## Density, gravity, and implementation

`cosmictiger/math/radial_kernel.hpp` generates fixed-size, allocation-free tables of
change points and Bernstein coefficients. `makeKernelTable` is `constexpr` and
can construct tables at compile time. The same generated density determines its
derivative, enclosed mass, gravitational potential, radius conversion, and central
potential. CPU scalar, CPU SIMD, and CUDA evaluators share this implementation.
The SPH solver itself continues to require CUDA; adding CPU evaluators is not a
CPU port of the complete density, neighbor, and hydrodynamics solver.

The separate `cosmictiger/math/radial_density_kernel.hpp` provides
`makeDensityKernelTable<dimension>(KernelSpec)` for any positive compile-time
dimension, with a distinct density-only table type. For example:

```cpp
constexpr auto kFourDimensional = radial_kernel::makeDensityKernelTable<4>(
    {radial_kernel::Family::Wendland, 9});
constexpr auto kTwentyDimensional = radial_kernel::makeDensityKernelTable<20>(
    {radial_kernel::Family::Spline, 9});
```

These tables contain the same coefficient/change-point representation and work
with the overloaded scalar, SIMD, and CUDA `value` and `derivative` evaluators.
The normalization is `supportRadius^-dimension`, and their effective radius uses
`sigma^2 = integral(r*r*W)/dimension`. Gravitational force and potential remain
explicitly three-dimensional and accept the production `KernelTable` only.

Spline and Ferrers shapes keep the degree choices listed above and acquire the
appropriate dimensional normalization. Wendland shapes change with dimension:
for nonnegative integer `k`, start with `(1-q)^ell`, where
`ell = floor(dimension/2) + k + 1`, and apply
`I[f](q) = integral_q^1 s*f(s) ds` exactly `k` times. Their degree is
`floor(dimension/2) + 1 + 3*k`. Only degrees at most nine are supported; thus,
for example, dimension one permits degrees 1/4/7, dimension four permits 3/6/9,
and dimensions above seventeen have no Wendland member within the degree cap.
Construction uses positive Bernstein coefficients and exact polynomial
integration rather than sampling the kernel.

`isValidDensity<dimension>(spec)` checks the dimension-dependent family/degree
combination. Invalid selections or a normalization not representable by the
double-precision table produce an empty table (`pieceCount == 0`). Very large
dimensions are also subject to compiler constexpr and storage limits; arbitrary
dimension does not promise arbitrary numerical range or resource use.

The gravitational conventions are

\[
M(q)=4\pi\int_0^q s^2 w(s)\,ds,\qquad
\mathbf a=-\frac{Gm}{R_{\mathrm{sup}}^3}\frac{M(q)}{q^3}\mathbf r,
\]

\[
\Phi=-\frac{Gm}{R_{\mathrm{sup}}}\left[\frac{M(q)}q+
4\pi\int_q^1 s\,w(s)\,ds\right].
\]

Outside the support radius, the interaction is exactly Newtonian. The evaluator returns a
positive potential magnitude; the gravitational potential carries the minus
sign shown above. The first span analytically removes the apparent divisions
by zero, giving finite central potential and force factor and zero acceleration
at zero separation. Integrated Bernstein coefficients avoid subtracting nearly
equal large powers near the support boundary.

## Fixed comoving gravitational softening

Gas and dark matter use the same selected gravity kernel and one fixed
comoving support radius. Set that radius with `gravity.effectiveRadius`,
`gravity.supportRadius`, or the legacy effective-radius alias `gravity.softening`.
The configured radius remains constant in comoving coordinates during evolution;
its physical length scales with the cosmological scale factor. Gravity does not
solve a neighbor constraint or use the gas smoothing radius.

SPH density, pressure, viscosity, and thermal work retain their variable,
gas-only smoothing radii and independently selected hydro kernel. Selecting the
same kernel family for gravity and SPH does not tie their lengths together.
The obsolete `gravity.sphSmoothing` option has been removed.

Particle masses are constant within each class and may differ between classes.
Cosmological initial conditions retain equal gas and dark-matter counts, with
code class masses `omegaB/omegaM` and `omegaC/omegaM`, respectively. Physical
masses are these values times the `units.massSolar` unit. SPH CUDA data carry one gas-class mass
scalar, with no mass value stored per SPH particle.

The hierarchical gravity path remains available for dark-matter-only runs.
Glass generation (`initial.createGlass=true`) uses its repulsive gravity and drag.

## Timesteps and migration from the old convention

The gravitational acceleration criterion uses the effective radius:

\[
\Delta t_{\mathrm{grav}}=C_{\mathrm{grav}}\sqrt{R_{\mathrm{eff,grav}}/|\mathbf a|},
\]

with the existing cosmological conversions applied in the implementation.
`gravity.timestepEta` is the prefactor `C_grav`. The Courant-viscous numerator
similarly uses the hydro effective radius. Equal effective radii do not guarantee
equal timestep error for different kernel shapes; accuracy and stability still require validation for
the selected problem and settings.

The earlier radius-unit conversion preserved the Ferrers F4 support and gravity
acceleration limit, and the cubic Courant-viscous limit, using these values:

| Setting | Previous default | Current default |
| --- | ---: | ---: |
| `gravity.softening` | `0.0125` in Plummer-equivalent epsilon | `0.01876988456953534` in `effectiveRadius = 2 sigma` |
| `gravity.timestepEta` | `0.25` | `0.20401599335954483` |
| `hydro.cfl` | `0.25` | `0.22821773229381923` |

These numeric values remain in use. The Ferrers F2 default changes the kernel
shape and does not preserve the former Ferrers F4 support at the same effective
radius.

For an old explicit Ferrers F4 configuration, convert softening by
`effectiveRadius = 1.5015907655628272 * epsilon`, gravity eta by
`eta_new = 0.8160639734381793 * eta_old`, and the cubic Courant coefficient by
`cfl_new = sqrt(5/6) * cfl_old`. Repository example configurations have been
converted. Existing user configuration files need explicit migration: a numeric
value supplied under an old option name is not automatically recognized as an
old convention. A change of kernel shape is a separate physical choice, not a
unit conversion.

Viscosity uses a fixed reference factor `sqrt(5/6)` times the mean SPH effective
radius. This equals Monaghan's `supportRadius/2` for the cubic and preserves the meaning of
the existing cubic `viscosityAlpha`, `viscosityBeta`, and regularization settings.
The hydro acceleration limit additionally retains a fixed reference calibration
`0.7115965735877551` multiplying gravity eta, preserving the historical
cubic/Ferrers F4 acceleration calibration through the radius-unit conversion.
These factors are independent of the selected kernel family.
The additional fractional thermal-energy timestep bound applies the same
fixed reference conversion to preserve their old default physical limits.

Kernel selections are encoded in the checkpoint physics marker. Restarting with
different selections is rejected. Checkpoints retain the version 3 format.
Historical kernel marker values retain their original meaning, so the Ferrers F2
default does not reinterpret a Ferrers F4 checkpoint. Existing SPH parameter checks still
apply, including timestep coefficients and the hydro neighbor target.

## Validation scope

The `run.test=force` direct reference supports fixed-softening, dark-matter-only
runs. It requires `hydro.enabled=false`; `run.test=sphGravity` exercises mixed
particle-class gravity with the common fixed radius.

The `radial_kernel` test covers all 14 kernels: unit mass and second moments,
independent polynomial references, derivatives, support joins, gravity against
radial quadrature and Poisson's equation, float behavior near the origin and
cutoff, and agreement with the former cubic density and Ferrers F4 force and
potential. On supported x86 builds it also checks native SIMD evaluation.
`radial_kernel_cuda` compares CUDA float and double evaluation against CPU results
at interior points, change points, the origin, and the cutoff. The existing
`monaghan_sph_cuda` tests exercise selected kernels through the SPH integration.
`radial_density_kernel` checks dimensional normalization and variance against
independent radial quadrature in dimensions 1–7 and 20, analytic Ferrers moments,
dimension-dependent Wendland shapes, and agreement with the production tables
in 3D. The CUDA parity target also exercises generic density tables in dimensions
1, 2, 3, 4, 7, and 20.

These are correctness checks, not kernel accuracy comparisons across physical
problems, convergence studies, or performance benchmarks.
