# MFM preparation

The MFM preparation kernels live under `cosmictiger/mfm/` and `src/mfm/`.
They provide standalone CPU and CUDA entry points for smoothing lengths, density,
and inverse moment geometry. `hydro.hopkins=true` allocates the new persistent
geometry alongside the existing dual thermal storage. It still runs the SPH
force solver and its original preparation; selecting an MFM force solver is a
later step.

## Numerical state

All new numerical interfaces take `typename Real`. CPU and CUDA explicitly
instantiate `float` and `double` for dimensions 1, 2, and 3. Production gas storage
currently uses `float`; there is no new precision option. The standalone data
views also accept double density and support arrays, without staging through
SPH's float buffers. Positions retain the existing fixed-point representation.

The specialized kernel is the two-piece cubic spline (M4), with full compact
support `h`, twice the conventional spline smoothing length. It does not consult
the SPH kernel selector:

\[
W_D(r,h)=h^{-D}C_D
\begin{cases}
1-6q^2+6q^3,&0\leq q<1/2,\\
2(1-q)^3,&1/2\leq q<1,\\
0,&q\geq1,
\end{cases}
\qquad q=r/h,
\]

with `C_D = 4/3, 40/(7π), 8/π` for dimensions 1, 2, 3. Coefficients and
accumulation use the requested precision. The copied support solve retains the
existing SPH neighbor-number convention, including the unit-ball factor 2 in
1D, and the default relative tolerance `1e-4` in both precisions.

Following [Hopkins equations 6–7 and 13–14](https://arxiv.org/pdf/1409.7395),
the equal-mass density and normalized moments are

\[
S_i=\sum_j W_{ij},\quad \rho_i=m S_i,\quad
E_i=\frac{\sum_j W_{ij}\,\Delta x_{ij}\otimes\Delta x_{ij}}{S_i},
\quad B_i=E_i^{-1}.
\]

Self contributes to density and normalization but contributes zero to the moment
numerator. Periodic separations are formed before conversion to `Real`.
The kernels accumulate the dimensionless matrix `M=E/h²`, invert and check it,
then publish `B=M⁻¹/h²`. The default accepted infinity-norm condition bound is
`1e4`, configurable in `PrehydroParameters<Real>`. This is a local conditioning
policy, not the Frobenius-norm definition in Hopkins Appendix C.

## Calling the kernels

`mfm::RunData<Real, dimensions>` holds frozen source arrays, target and neighbor
tree views, and mutable sink arrays. Source and sink indices can differ; the
tree's source-to-sink ranges and gas-to-full-particle mapping must be supplied.
Sources remain immutable for the whole pass. Runtime dimensional dispatch happens
outside the kernels, so a 1D/2D matrix is never a zero-padded 3D matrix.

Call `smoothingLengthsCpu` per target-node entry, then refresh the staged source
supports as needed before `densityAndGeometryCpu`. CUDA's corresponding
`smoothingLengthsCuda` and `densityAndGeometryCuda` accept a `CudaStream&`, process
all target-node entries, wait for completion, and return merged diagnostics.
CUDA arrays must be device-accessible and the CUDA pool must be initialized for
neighbor-cache overflow.

A `needsAnotherPass` result requests expanded neighbor bounds and restaging.
The existing half-box support limit remains a hard error. Density preparation
also solves unconverged semiactive supports; active supports must first have
completed the smoothing pass. `initializeDensity` publishes the actual kernel
sum at the converged or bounded support on both backends.

Each attempted support or geometry update clears the previous inverse and its
validity byte. Singular, nonfinite, or ill-conditioned geometry remains invalid
and increments `geometryFailures`; it does not request an identical endless
support retry. Consumers must check validity and handle failure before using
`B`. Growing the neighbor set or falling back to a lower-order reconstruction is
the responsibility of a future MFM reconstruction driver. Callers must invalidate
or rebuild geometry when positions or supports change.

These kernels do not evolve thermal variables. The current production SPH
driver retains its existing density-dependent entropy/energy synchronization.
A future MFM driver must establish the thermal state after successful density
preparation and before reconstruction.

## Persistent layout

`mfm::ParticleStorage<Real, dimensions>` owns one array of
`math::SymmetricMatrix<Real, dimensions>` and a validity-byte array. A matrix has
exactly 1, 3, or 6 scalar entries in packed lower-triangle order. The production
gas lifecycle resizes, permutes, removes, and migrates both arrays together.
New entries are zero and invalid. No MFM arrays are allocated for ordinary SPH.

Hopkins checkpoints set flag 29 and append a versioned payload specifying the
dimension, scalar type/width, packing, component count, and particle count.
Ordinary SPH checkpoint bytes remain unchanged. Old Hopkins checkpoints without
the geometry payload are rejected. The Python checkpoint decoder exposes
`inverse_moments` and `geometry_valid`.

The CPU tests cover both precisions and all dimensions, independent density and
inverse residuals, periodic wrapping, active/semiactive selection, bounded and
free supports, retries, invalid geometry, and storage round trips. CUDA tests
exercise CPU parity and shared-neighbor overflow; a successful compile alone
does not establish GPU runtime correctness.
