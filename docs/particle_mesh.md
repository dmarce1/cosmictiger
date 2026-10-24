# Particle-mesh gravity

Select the periodic particle-mesh (PM) solver with `gravity.method=pm` and give
the number of cells on each axis using `gravity.meshSize`. A value of 128 means
a `128 × 128 × 128` mesh in the normalized unit box. The mesh resolution is
independent of `initial.particlesPerDimension`. The default gravity method is
`fmm`; `gravity.meshSize=0` is the unset default and is rejected for PM.

```sh
./cosmictiger --gravity.method=pm --gravity.meshSize=128 --gravity.cloud=pcs \
  --initial.particlesPerDimension=64 --compute.cuda=false --hpx:threads=12
```

Equivalent INI settings are:

```ini
[gravity]
method = pm
meshSize = 128
cloud = pcs
```

The PM solve and particle cell sorting always run on CPUs. `compute.cuda`
continues to select the existing SPH and particle-drift backends: setting it to
`true` permits CPU PM together with CUDA SPH and CUDA drift. A build configured
with `COSMICTIGER_WITH_CUDA=OFF` supports the complete CPU path. Gravity still
includes both gas and dark matter, with their configured species masses.

## Clouds and force convention

`gravity.cloud` selects a centered one-dimensional cardinal B-spline, applied
as the tensor product of its Cartesian weights:

| Value | Name | Polynomial degree | Points per axis, `p` | Maximum cells per particle |
| --- | --- | ---: | ---: | ---: |
| `ngp` | Nearest grid point | 0 | 1 | 1 |
| `cic` | Cloud in cell | 1 | 2 | 8 |
| `tsc` | Triangular-shaped cloud | 2 | 3 | 27 |
| `pcs` | Piecewise cubic spline, default | 3 | 4 | 64 |

These Cartesian clouds share the B-spline family with the SPH spline kernels.
They use a product of three normalized one-dimensional weights; they are not
the radial three-dimensional SPH kernel evaluated at Cartesian distance. PM
currently exposes the four assignments above, independently of
`hydro.kernelFamily` and `hydro.kernelDegree`. The mesh size must be at least
`max(2, p)`. Odd sizes and sizes not divisible by the number of MPI ranks are
supported.

Each particle deposits its mass through its cloud. The force at that particle
is interpolated using the **same cloud weights**. Both operations matter: they
each contribute an assignment window. For mesh spacing `Δx=1/n` and integer
Fourier mode `m`, define

\[
  k = 2\pi m, \qquad
  W(k)=\prod_{d=1}^{3}
  \left[\frac{\sin(k_d\Delta x/2)}{k_d\Delta x/2}\right]^p.
\]

The zero-argument ratio is one. For nonzero modes, the solver applies

\[
  \widehat\Phi(k)=
  -\frac{4\pi\,\mathrm{gm}}{|k|^2 W(k)^2}\,
  \widehat\rho_{\mathrm{assigned}}(k),
  \qquad
  \widehat g_d(k)=-i k_d\widehat\Phi(k).
\]

Thus the Green function compensates both source and sink assignment with
`1/W²`. The mean-density mode has zero potential and force. On an even mesh,
the derivative multiplier for a Nyquist component is zero, preserving the
real-field symmetry of the discrete transform.

Assignment compensation corrects the principal Fourier window. It does not
remove aliased modes or recover information above the mesh resolution, and it
can amplify grid-scale noise. Increasing the cloud order suppresses aliases
more strongly before compensation but increases deposition and interpolation
work. Mesh refinement and convergence checks remain necessary. There is no
additional FMM softening convolution or short-range FMM correction in this
backend; this is pure periodic PM gravity.

The density array includes the cell-volume factor `n³`, and each inverse FFT is
normalized by `1/n³`. `gravity.gm` has the same meaning as in the FMM backend.
The returned acceleration uses the existing conformal-time kick convention;
the driver supplies the usual cosmological drift integral and physical energy
conversions.

## Distribution and sorting

Particles retain CosmicTiger's spatial ownership between localities. Before
each PM force evaluation, ordinary domain migration finishes, then each
locality groups its particle arrays in place by the flattened mesh-cell key.
The sort uses parallel partitions and parallel exchanges of disjoint ranges,
followed by independent recursive partitions. It moves all particle fields
together and repairs the gas/particle index mapping. It does not allocate a
second array of complete particle records; mesh arrays, tile workspaces, and
communication buffers still require additional memory.

FFTW3-MPI owns the distributed mesh slabs. Deposition accumulates local tiles
and sends their mesh contributions to the owning slabs. After the transforms,
the particle owners fetch the field tiles needed for sink interpolation. No
locality gathers the full particle population or full mesh for the solve.

MPI calls, including FFTW-MPI plan setup, execution, and destruction, run
through `hpx::run_as_os_thread`. A separate PM communicator isolates their
traffic, and process-local planner locking is shared with the existing serial
FFTW users. Collectives are coordinated in the same order on every rank.
The application requires actual `MPI_THREAD_MULTIPLE` support at startup.

Field requests are drained before interpolation consumes their results, including
when a later request cannot be launched. Collective dispatch failures terminate
the MPI job instead of attempting cleanup while other ranks may be blocked inside
a collective. A native abort worker is started before dispatch so this failure
path needs neither another allocation nor an available HPX I/O worker.

## Uniform adaptive timesteps

PM uses one common timestep for **every particle on every locality**. There is
no particle subcycling or hierarchical PM force evaluation. The common step
can adapt between evaluations and remains on the existing dyadic clock:

\[
  \Delta\tau = \frac{T}{N_{\mathrm{coarse}}}\,2^{-r},
\]

where `T` is the configured run duration and `N_coarse` is
`run.numberOfSteps`. All particles receive the same rung `r`; the current
ordinary-kick policy has a minimum rung of one. `run.minRung` can impose a
finer lower bound. Coarsening is limited to one rung at a time and can happen
only at a boundary compatible with the larger step. The terminal and progress
file show the common rung as both the minimum and maximum rung.

Gravity proposes its timestep using the cloud's effective radius, measured in
box units,

\[
  r_{\mathrm{eff}}=2\sigma=\frac{\sqrt{p/3}}{n}, \qquad
  \Delta\tau_g=\eta\sqrt{a\,r_{\mathrm{eff}}/|g|},
\]

with `eta = gravity.timestepEta`, together with the existing cosmological
expansion limit. This length is used by both dark-matter and gas gravity
timestep estimates; the FMM softening radius is not used for the PM estimate.
SPH contributes its existing hydrodynamic and thermal limits. The global
finest requested rung is chosen before opening either species' next
half-kick. The driver closes the previous dark-matter kick, performs the SPH
predictor/corrector when enabled, records synchronized energy diagnostics,
sets the common rung, opens the new kicks, and drifts every particle once.

With primordial chemistry enabled, `chemistry.synchronizedRung` retains its
fixed-step policy. A PM or hydro bound requiring a finer step causes an error
before the new opening kicks. Existing glass repulsion and drift damping are
retained, as are the existing restrictions on glass and SPH initializers.

## Potential energy and restarts

The mesh potential uses a zero-mean periodic gauge. Interpolating it at a
source particle includes that particle's own source-and-sink cloud response.
The reported particle potential subtracts this **discrete self potential**,
including its subcell dependence and assignment compensation. Potential
energy is then `sum(m_i * phi_i)/2`, with the driver's existing cosmological
scale conversion. The matched interpolation and odd force operator already
cancel a particle's self force to numerical precision. Potential values from
different gravity methods must be compared using consistent gauges and
self-interaction conventions.

PM checkpoints record a versioned extension identifying mesh size, cloud,
minimum rung, gravity timestep coefficient, and force convention. The backend
itself is part of the compatibility flags. Changing these settings on restart
is rejected. Particle data, SPH predictor state, and the shared clock are saved
normally; FFT plans, cell ranges, and mesh workspaces are reconstructed. FMM
checkpoint layouts retain their existing format.

The regression suite checks cloud normalization, sorting and field
preservation, direct reciprocal-space force/potential agreement, mass
conservation, periodic boundaries, self force, distributed execution,
uniform timesteps, exactly one drift, and uninterrupted versus restarted
evolution. The timestep regression changes the actual mesh gravity strength
and checks remote-locality timestep proposals, immediate refinement, the
integer-clock coarsening boundary, the one-rung growth limit, and both staged
half-kicks. CPU-only and CUDA-SPH/drift variants exercise the same CPU PM solve.
