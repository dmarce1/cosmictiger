# Source and header inventory

This inventory covers every tracked C/C++/CUDA source and header under `cosmictiger/`,
`src/`, `tests/`, and `benchmarks/` from documentation base `9b9e0eec6`, updated for global FMM restoration.
Build trees, third-party dependencies, and archived experiment sources under `docs/` are excluded.
Two CMake source templates and Python tooling are listed separately below. Descriptions summarize responsibility; they do not
imply that an optional backend is enabled or that every declared path runs in production.

**489 files:** 220 public/internal headers, 130 application sources/headers,
138 test sources/headers, and one benchmark.

## Application and library headers

| File | Contents |
| --- | --- |
| [cosmictiger/analysis/bh.hpp](../cosmictiger/analysis/bh.hpp) | Barnes–Hut tree and monopole acceptance for halo binding-energy calculations. |
| [cosmictiger/analysis/compress.hpp](../cosmictiger/analysis/compress.hpp) | Block compression of lightcone positions and velocities, with binary read/write support. |
| [cosmictiger/analysis/group_entry.hpp](../cosmictiger/analysis/group_entry.hpp) | Friends-of-friends catalogue records, including mass, kinematics, shape, and ancestry. |
| [cosmictiger/analysis/group_header.hpp](../cosmictiger/analysis/group_header.hpp) | Binary groups.dat header containing count, time, box length, and scale factor. |
| [cosmictiger/analysis/group_tree.hpp](../cosmictiger/analysis/group_tree.hpp) | Distributed spatial tree used to discover friends-of-friends groups. |
| [cosmictiger/analysis/groups.hpp](../cosmictiger/analysis/groups.hpp) | Distributed group membership, particle exchange, and catalogue orchestration. |
| [cosmictiger/analysis/groups_find.hpp](../cosmictiger/analysis/groups_find.hpp) | Asynchronous friends-of-friends linking through the group tree. |
| [cosmictiger/analysis/lightcone.hpp](../cosmictiger/analysis/lightcone.hpp) | Lightcone particle collection, sky partitioning, group finding, and output. |
| [cosmictiger/analysis/lightcone_entry.hpp](../cosmictiger/analysis/lightcone_entry.hpp) | Lightcone particle records and emission-time scale-factor conversion. |
| [cosmictiger/analysis/lightcone_map.hpp](../cosmictiger/analysis/lightcone_map.hpp) | Sparse HEALPix weight map with dense map output and sparse checkpoint storage. |
| [cosmictiger/analysis/power.hpp](../cosmictiger/analysis/power.hpp) | Particle density deposition and Fourier-space binning of the matter power spectrum. |
| [cosmictiger/analysis/power_cpu.hpp](../cosmictiger/analysis/power_cpu.hpp) | CPU particle-to-grid density deposition for folded power spectra. |
| [cosmictiger/analysis/power_cuda.hpp](../cosmictiger/analysis/power_cuda.hpp) | CUDA particle-to-grid density deposition for folded power spectra. |
| [cosmictiger/analysis/rockstar.hpp](../cosmictiger/analysis/rockstar.hpp) | Phase-space halo finding, subgroup hierarchy, and halo-particle processing. |
| [cosmictiger/analysis/rockstar_core.hpp](../cosmictiger/analysis/rockstar_core.hpp) | Shared geometric overlap, normalization, linking-length, and subgroup-label operations. |
| [cosmictiger/analysis/rockstar_properties.hpp](../cosmictiger/analysis/rockstar_properties.hpp) | Spherical-overdensity thresholds, NFW fitting, scale radii, and shape eigenpairs. |
| [cosmictiger/analysis/rockstar_record.hpp](../cosmictiger/analysis/rockstar_record.hpp) | Native halo catalogue record layout and packed halo identifiers. |
| [cosmictiger/chemistry/gadget4_cooling.hpp](../cosmictiger/chemistry/gadget4_cooling.hpp) | Equilibrium cooling tables, gas-state conversion, and thermal evolution using the GADGET-4 model. |
| [cosmictiger/chemistry/gadget4_cooling_impl.hpp](../cosmictiger/chemistry/gadget4_cooling_impl.hpp) | Host/device cooling-rate interpolation, equilibrium abundances, and implicit energy integration. |
| [cosmictiger/chemistry/recombination.hpp](../cosmictiger/chemistry/recombination.hpp) | Hydrogen/helium recombination and thermal-history integration for the cosmological background. |
| [cosmictiger/chemistry/sph.hpp](../cosmictiger/chemistry/sph.hpp) | Coupling of gas cooling and chemistry to SPH particles and their selected time intervals. |
| [cosmictiger/core/assert.hpp](../cosmictiger/core/assert.hpp) | Host/device assertion reporting and debug versus unconditional assertion macros. |
| [cosmictiger/core/build_config.hpp](../cosmictiger/core/build_config.hpp) | Required build-capability definitions and consistency checks for physics and integrators. |
| [cosmictiger/core/containers.hpp](../cosmictiger/core/containers.hpp) | Project vector, fixed-array, and pair containers with optional bounds checks and serialization. |
| [cosmictiger/core/defs.hpp](../cosmictiger/core/defs.hpp) | Shared dimensions, scalar/index types, feature switches, and low-level constants. |
| [cosmictiger/core/device_profile.hpp](../cosmictiger/core/device_profile.hpp) | Device profiling phase identifiers, counters, and per-block result records. |
| [cosmictiger/core/diagnostic_output.hpp](../cosmictiger/core/diagnostic_output.hpp) | Shared interfaces for simulation diagnostic output. |
| [cosmictiger/core/driver.hpp](../cosmictiger/core/driver.hpp) | Cosmological simulation evolution, force-step sequencing, output scheduling, and driver state. |
| [cosmictiger/core/file_handle.hpp](../cosmictiger/core/file_handle.hpp) | RAII ownership of FILE handles using a unique-pointer deleter. |
| [cosmictiger/core/fixed_capacity_vector.hpp](../cosmictiger/core/fixed_capacity_vector.hpp) | Small fixed-capacity host/device vector with explicit logical size. |
| [cosmictiger/core/flops.hpp](../cosmictiger/core/flops.hpp) | CPU/GPU floating-point operation counters, reductions, and reporting. |
| [cosmictiger/core/force_step.hpp](../cosmictiger/core/force_step.hpp) | Force-step aggregate results, merging, and serialization. |
| [cosmictiger/core/hpx.hpp](../cosmictiger/core/hpx.hpp) | Locality topology, HPX initialization, distributed helpers, and annotated execution policies. |
| [cosmictiger/core/interrupt_profile.hpp](../cosmictiger/core/interrupt_profile.hpp) | Signal-triggered profiling controls and reporting integration. |
| [cosmictiger/core/memused.hpp](../cosmictiger/core/memused.hpp) | Process memory-use measurement. |
| [cosmictiger/core/options.hpp](../cosmictiger/core/options.hpp) | Grouped runtime options, configuration parsing, defaults, and cross-option validation. |
| [cosmictiger/core/options_metadata.hpp](../cosmictiger/core/options_metadata.hpp) | Metadata for registering and describing grouped configuration options. |
| [cosmictiger/core/particle_count.hpp](../cosmictiger/core/particle_count.hpp) | Particle-count rounding for spatial lattices and Kelvin–Helmholtz initial conditions. |
| [cosmictiger/core/profiler.hpp](../cosmictiger/core/profiler.hpp) | Optional scoped timing/work profiling, phase transitions, counters, and reports. |
| [cosmictiger/core/profiling_runtime.hpp](../cosmictiger/core/profiling_runtime.hpp) | Runtime setup and shutdown of profiling facilities. |
| [cosmictiger/core/rung_merge.hpp](../cosmictiger/core/rung_merge.hpp) | Decision logic for merging particle timestep rungs at synchronization boundaries. |
| [cosmictiger/core/safe_io.hpp](../cosmictiger/core/safe_io.hpp) | Checked binary reads/writes and formatted runtime-error helpers. |
| [cosmictiger/core/semaphore.hpp](../cosmictiger/core/semaphore.hpp) | Counting semaphore implemented with atomic state and HPX suspension. |
| [cosmictiger/core/sph_midpoint.hpp](../cosmictiger/core/sph_midpoint.hpp) | Selection of the timestep rung for an SPH midpoint event. |
| [cosmictiger/core/stack_trace.hpp](../cosmictiger/core/stack_trace.hpp) | Stack-trace capture, fatal-signal reporting, and diagnostic output paths. |
| [cosmictiger/core/stack_vector.hpp](../cosmictiger/core/stack_vector.hpp) | Host/device traversal stack with variable-sized frames in contiguous storage. |
| [cosmictiger/core/startup.hpp](../cosmictiger/core/startup.hpp) | Startup-screen reporting interface. |
| [cosmictiger/core/terminal.hpp](../cosmictiger/core/terminal.hpp) | Terminal formatting for sections, durations, verbosity, and rung populations. |
| [cosmictiger/core/time.hpp](../cosmictiger/core/time.hpp) | Integer simulation-clock increments and synchronization-rung calculation. |
| [cosmictiger/core/timer.hpp](../cosmictiger/core/timer.hpp) | Accumulating elapsed-time stopwatch with start, stop, reset, and read operations. |
| [cosmictiger/core/tree_id.hpp](../cosmictiger/core/tree_id.hpp) | Distributed tree-node identifiers, comparisons, hashing, and serialization. |
| [cosmictiger/core/unordered_set_ts.hpp](../cosmictiger/core/unordered_set_ts.hpp) | Partitioned thread-safe unordered set and collection into a vector. |
| [cosmictiger/cosmology/background.hpp](../cosmictiger/cosmology/background.hpp) | Simulation background trajectory, conformal-time sampling, and shared background construction. |
| [cosmictiger/cosmology/background_plot.hpp](../cosmictiger/cosmology/background_plot.hpp) | Rendering and writing plots of cosmological background evolution. |
| [cosmictiger/cosmology/boltzmann.hpp](../cosmictiger/cosmology/boltzmann.hpp) | Linear Einstein–Boltzmann perturbation evolution and transfer-function integration. |
| [cosmictiger/cosmology/cosmology.hpp](../cosmictiger/cosmology/cosmology.hpp) | Simulation-facing cosmological clock, expansion, growth, and halo-fit interfaces. |
| [cosmictiger/cosmology/evolution.hpp](../cosmictiger/cosmology/evolution.hpp) | Expansion timestep limits, termination tests, and cosmological energy accounting. |
| [cosmictiger/cosmology/fft.hpp](../cosmictiger/cosmology/fft.hpp) | Distributed three-dimensional FFT storage, decomposition, transforms, and density-assignment filters. |
| [cosmictiger/cosmology/fftw_lock.hpp](../cosmictiger/cosmology/fftw_lock.hpp) | Process-wide locking of FFTW planner operations. |
| [cosmictiger/cosmology/fftw_threads.hpp](../cosmictiger/cosmology/fftw_threads.hpp) | Routing FFTW parallel loops through the application's threading runtime. |
| [cosmictiger/cosmology/friedmann.hpp](../cosmictiger/cosmology/friedmann.hpp) | Friedmann background integration, dense trajectory sampling, and interval quadrature. |
| [cosmictiger/cosmology/initialize.hpp](../cosmictiger/cosmology/initialize.hpp) | Linear-power normalization and Zel'dovich/second-order Lagrangian particle initialization. |
| [cosmictiger/cosmology/linear_power.hpp](../cosmictiger/cosmology/linear_power.hpp) | Logarithmic power-spectrum interpolation, top-hat variance, and transfer normalization. |
| [cosmictiger/cosmology/lpt.hpp](../cosmictiger/cosmology/lpt.hpp) | Reproducible Fourier modes, lattice shifts, and Lagrangian displacement/source helpers. |
| [cosmictiger/cosmology/native_recombination.hpp](../cosmictiger/cosmology/native_recombination.hpp) | Conversion of cosmology parameters into the native recombination/background model. |
| [cosmictiger/cosmology/recombination_thermal.hpp](../cosmictiger/cosmology/recombination_thermal.hpp) | Compton thermal coupling and stable gas-temperature advancement. |
| [cosmictiger/cosmology/zero_order.hpp](../cosmictiger/cosmology/zero_order.hpp) | Tabulated unperturbed universe, density fractions, and redshift/time/scale conversions. |
| [cosmictiger/cuda/cuda.hpp](../cosmictiger/cuda/cuda.hpp) | CUDA runtime initialization, device selection, memory allocation, and host fallback adapters. |
| [cosmictiger/cuda/cuda_check.hpp](../cosmictiger/cuda/cuda_check.hpp) | CUDA runtime-call error checking macros and helpers. |
| [cosmictiger/cuda/cuda_export.hpp](../cosmictiger/cuda/cuda_export.hpp) | Compiler-dependent host/device function qualifiers. |
| [cosmictiger/cuda/cuda_memory.hpp](../cosmictiger/cuda/cuda_memory.hpp) | Pooled host/device allocation, block synchronization, and memory-copy support. |
| [cosmictiger/cuda/cuda_memory_advice.hpp](../cosmictiger/cuda/cuda_memory_advice.hpp) | CUDA-version-compatible managed-memory advice and asynchronous prefetch wrappers. |
| [cosmictiger/cuda/cuda_reduce.hpp](../cosmictiger/cuda/cuda_reduce.hpp) | Warp/block reductions and prefix scans used by cooperative CUDA kernels. |
| [cosmictiger/cuda/cuda_stream.hpp](../cosmictiger/cuda/cuda_stream.hpp) | RAII CUDA streams with HPX-aware completion polling and asynchronous transfers. |
| [cosmictiger/cuda/cuda_unordered_map.hpp](../cosmictiger/cuda/cuda_unordered_map.hpp) | Integer-key hash map backed by device-accessible bucket vectors. |
| [cosmictiger/cuda/device_profiler.hpp](../cosmictiger/cuda/device_profiler.hpp) | Warp-cooperative device cycle timing and per-phase counter recording. |
| [cosmictiger/cuda/device_vector.hpp](../cosmictiger/cuda/device_vector.hpp) | Device-accessible owning vector with block-cooperative mutation and host lifecycle support. |
| [cosmictiger/cuda/kernel_error.hpp](../cosmictiger/cuda/kernel_error.hpp) | First-error capture in kernels and deferred host-side error handling. |
| [cosmictiger/gravity/build_config.hpp](../cosmictiger/gravity/build_config.hpp) | Gravity capability definitions and checks. |
| [cosmictiger/gravity/direct_reference.hpp](../cosmictiger/gravity/direct_reference.hpp) | Compensated direct-force reference summation and periodic correction lookup. |
| [cosmictiger/gravity/error.hpp](../cosmictiger/gravity/error.hpp) | Sampling and comparing production gravity against direct reference forces. |
| [cosmictiger/gravity/error_cuda.hpp](../cosmictiger/gravity/error_cuda.hpp) | CUDA evaluation of direct-reference forces for gravity-error samples. |
| [cosmictiger/gravity/fmm_acceptance.hpp](../cosmictiger/gravity/fmm_acceptance.hpp) | Opening-angle geometry and selection of accepted FMM interaction types. |
| [cosmictiger/gravity/fmm_cpu_instances.hpp](../cosmictiger/gravity/fmm_cpu_instances.hpp) | CPU FMM instantiation declarations for the build-selected order. |
| [cosmictiger/gravity/fmm_geometry.hpp](../cosmictiger/gravity/fmm_geometry.hpp) | Periodic fixed-position geometry and source/sink box-distance operations. |
| [cosmictiger/gravity/fmm_order.hpp](../cosmictiger/gravity/fmm_order.hpp) | Build-selected FMM order, coefficient counts, and storage sizes. |
| [cosmictiger/gravity/fmm_types.hpp](../cosmictiger/gravity/fmm_types.hpp) | Typed force, multipole, and local-expansion storage with transport buffers. |
| [cosmictiger/gravity/gravity.hpp](../cosmictiger/gravity/gravity.hpp) | Particle/cell force kernels, softening helpers, and gravity work counters. |
| [cosmictiger/gravity/gravity_cpu.hpp](../cosmictiger/gravity/gravity_cpu.hpp) | CPU/SIMD cell-cell, cell-particle, and particle-cell FMM interactions. |
| [cosmictiger/gravity/gravity_cuda.hpp](../cosmictiger/gravity/gravity_cuda.hpp) | Cooperative CUDA staging and direct particle/cell interaction kernels. |
| [cosmictiger/gravity/kernels/cartesian_tracefree/ewald_common.hpp](../cosmictiger/gravity/kernels/cartesian_tracefree/ewald_common.hpp) | Periodic Ewald image/wave tables and scalar/SIMD special-function helpers. |
| [cosmictiger/gravity/kernels/cartesian_tracefree/ewald_green.hpp](../cosmictiger/gravity/kernels/cartesian_tracefree/ewald_green.hpp) | Generated Cartesian derivatives of the regular periodic Ewald correction. |
| [cosmictiger/gravity/kernels/cartesian_tracefree/ewald_m2l.hpp](../cosmictiger/gravity/kernels/cartesian_tracefree/ewald_m2l.hpp) | Generated periodic correction multipole-to-local contraction. |
| [cosmictiger/gravity/kernels/cartesian_tracefree/geometry.hpp](../cosmictiger/gravity/kernels/cartesian_tracefree/geometry.hpp) | Inverse-radius geometry helpers for Cartesian FMM operators. |
| [cosmictiger/gravity/kernels/cartesian_tracefree/l2l.hpp](../cosmictiger/gravity/kernels/cartesian_tracefree/l2l.hpp) | Generated Cartesian trace-free kernels to translate a local expansion to a displaced center. |
| [cosmictiger/gravity/kernels/cartesian_tracefree/l2l_warp.hpp](../cosmictiger/gravity/kernels/cartesian_tracefree/l2l_warp.hpp) | Generated warp-cooperative CUDA local-expansion translation. |
| [cosmictiger/gravity/kernels/cartesian_tracefree/l2p.hpp](../cosmictiger/gravity/kernels/cartesian_tracefree/l2p.hpp) | Generated Cartesian trace-free kernels to evaluate local potential and acceleration at a particle. |
| [cosmictiger/gravity/kernels/cartesian_tracefree/m2l.hpp](../cosmictiger/gravity/kernels/cartesian_tracefree/m2l.hpp) | Generated Cartesian trace-free kernels to convert source multipoles into a target local expansion. |
| [cosmictiger/gravity/kernels/cartesian_tracefree/m2m.hpp](../cosmictiger/gravity/kernels/cartesian_tracefree/m2m.hpp) | Generated Cartesian trace-free kernels to translate and accumulate source multipoles. |
| [cosmictiger/gravity/kernels/cartesian_tracefree/m2p.hpp](../cosmictiger/gravity/kernels/cartesian_tracefree/m2p.hpp) | Generated Cartesian trace-free kernels to evaluate multipoles at a particle. |
| [cosmictiger/gravity/kernels/cartesian_tracefree/p2l.hpp](../cosmictiger/gravity/kernels/cartesian_tracefree/p2l.hpp) | Generated Cartesian trace-free kernels to accumulate a particle contribution into a local expansion. |
| [cosmictiger/gravity/kernels/cartesian_tracefree/p2m.hpp](../cosmictiger/gravity/kernels/cartesian_tracefree/p2m.hpp) | Generated Cartesian trace-free kernels to accumulate particle mass into source multipoles. |
| [cosmictiger/gravity/kernels/cartesian_tracefree_config.hpp](../cosmictiger/gravity/kernels/cartesian_tracefree_config.hpp) | Generated build-degree configuration for Cartesian trace-free expansions. |
| [cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp](../cosmictiger/gravity/kernels/cartesian_tracefree_operators.hpp) | Cartesian FMM operator includes and source-first declarations. |
| [cosmictiger/gravity/kernels/cartesian_tracefree_types.hpp](../cosmictiger/gravity/kernels/cartesian_tracefree_types.hpp) | Generated named coefficient fields and storage/access/serialization for Cartesian expansions. |
| [cosmictiger/gravity/kick.hpp](../cosmictiger/gravity/kick.hpp) | Gravity tree walks, pooled interaction lists, and deferred SPH gas-opening state. |
| [cosmictiger/gravity/kick_context.hpp](../cosmictiger/gravity/kick_context.hpp) | State supplied to gravity kick execution. |
| [cosmictiger/gravity/kick_cuda.hpp](../cosmictiger/gravity/kick_cuda.hpp) | CUDA kick control records and shared-memory list allocation. |
| [cosmictiger/gravity/kick_cuda_instances.hpp](../cosmictiger/gravity/kick_cuda_instances.hpp) | CUDA kick instantiation declarations for the build-selected order. |
| [cosmictiger/gravity/kick_cuda_kernel.hpp](../cosmictiger/gravity/kick_cuda_kernel.hpp) | CUDA gravity tree traversal, particle updates, and deferred gas-opening kernels. |
| [cosmictiger/gravity/kick_cuda_launch.hpp](../cosmictiger/gravity/kick_cuda_launch.hpp) | CUDA kick launch layout, occupancy/resource budgeting, storage, and profiling events. |
| [cosmictiger/gravity/kick_interaction_counts.hpp](../cosmictiger/gravity/kick_interaction_counts.hpp) | Records for gravity interaction counts. |
| [cosmictiger/gravity/kick_walk.hpp](../cosmictiger/gravity/kick_walk.hpp) | Host/device traversal of gravity source trees and accepted-interaction dispatch. |
| [cosmictiger/gravity/kick_workspace.hpp](../cosmictiger/gravity/kick_workspace.hpp) | Ownership and aggregation of gravity batch work, results, and device submissions. |
| [cosmictiger/gravity/reference_table.hpp](../cosmictiger/gravity/reference_table.hpp) | Periodic gravity correction table indexing, symmetry, interpolation, and construction. |
| [cosmictiger/gravity/softening.hpp](../cosmictiger/gravity/softening.hpp) | Conversion of configured softening length into the effective force-kernel radius. |
| [cosmictiger/gravity/source_access.hpp](../cosmictiger/gravity/source_access.hpp) | Local/remote gravity source-position access and reader registration. |
| [cosmictiger/gravity/sources.hpp](../cosmictiger/gravity/sources.hpp) | FMM source partitions, registered source views, and particle-mass access. |
| [cosmictiger/gravity/tree.hpp](../cosmictiger/gravity/tree.hpp) | Gravity-tree node records, source views, allocation, and traversal interfaces. |
| [cosmictiger/gravity/tree_build.hpp](../cosmictiger/gravity/tree_build.hpp) | Recursive construction of gravity trees and source multipoles. |
| [cosmictiger/gravity/tree_multipole.hpp](../cosmictiger/gravity/tree_multipole.hpp) | Particle-to-tree Cartesian multipole reduction and double-precision upward source construction. |
| [cosmictiger/gravity/tree_source_moments.hpp](../cosmictiger/gravity/tree_source_moments.hpp) | Single/double-precision source-moment storage and narrowing. |
| [cosmictiger/io/checkpoint.hpp](../cosmictiger/io/checkpoint.hpp) | Versioned checkpoint headers, manifests, compatibility checks, and distributed state save/restore. |
| [cosmictiger/io/output.hpp](../cosmictiger/io/output.hpp) | Particle, tracer, and slice output with checked file handling. |
| [cosmictiger/io/view.hpp](../cosmictiger/io/view.hpp) | Distributed Silo particle views, species fields, unit conversion, metadata, and visualization expressions. |
| [cosmictiger/math/cell_lattice.hpp](../cosmictiger/math/cell_lattice.hpp) | Periodic spatial lattice descriptions, offsets, plane counts, and occupied-site generation. |
| [cosmictiger/math/complex.hpp](../cosmictiger/math/complex.hpp) | Project complex-number type aliases and arithmetic integration. |
| [cosmictiger/math/constants.hpp](../cosmictiger/math/constants.hpp) | Shared mathematical and physical constants. |
| [cosmictiger/math/fixed.hpp](../cosmictiger/math/fixed.hpp) | Fixed-point arithmetic, raw-bit conversion, and periodic/unwrapped coordinate differences. |
| [cosmictiger/math/float16_storage.hpp](../cosmictiger/math/float16_storage.hpp) | Portable IEEE binary16 storage conversion without requiring CUDA runtime arithmetic. |
| [cosmictiger/math/fp16.hpp](../cosmictiger/math/fp16.hpp) | Half-precision type support, numeric limits, and conversion helpers. |
| [cosmictiger/math/interpolation.hpp](../cosmictiger/math/interpolation.hpp) | Interpolation-function construction and table evaluation. |
| [cosmictiger/math/kernel.hpp](../cosmictiger/math/kernel.hpp) | Selected hydrodynamic/gravitational radial kernels and host/device selection state. |
| [cosmictiger/math/math.hpp](../cosmictiger/math/math.hpp) | Small arithmetic, random-number, rounding, sinc, and atomic-add helpers. |
| [cosmictiger/math/matrix.hpp](../cosmictiger/math/matrix.hpp) | Fixed-size dense/symmetric/diagonal matrices, arithmetic, determinants, inversion, and linear solves. |
| [cosmictiger/math/matrix_eigen.hpp](../cosmictiger/math/matrix_eigen.hpp) | Jacobi eigensolver for symmetric matrices with convergence/status reporting. |
| [cosmictiger/math/position.hpp](../cosmictiger/math/position.hpp) | Configured fixed-point particle-position type and domain-coordinate conversion. |
| [cosmictiger/math/radial_density_kernel.hpp](../cosmictiger/math/radial_density_kernel.hpp) | Normalization and piecewise evaluation of radial SPH density kernels. |
| [cosmictiger/math/radial_kernel.hpp](../cosmictiger/math/radial_kernel.hpp) | Compile-time radial-kernel polynomial tables, derivatives, softened force, and potential. |
| [cosmictiger/math/range.hpp](../cosmictiger/math/range.hpp) | Multidimensional boxes, periodic intersections, indexing, and fixed-point distance tests. |
| [cosmictiger/math/simd_backend.hpp](../cosmictiger/math/simd_backend.hpp) | Selected SIMD backend types and configuration. |
| [cosmictiger/math/simd_math.hpp](../cosmictiger/math/simd_math.hpp) | Uniform scalar/SIMD broadcasts, masks, arithmetic, lane operations, reductions, and position differences. |
| [cosmictiger/math/simulation_units.hpp](../cosmictiger/math/simulation_units.hpp) | Construction of physical unit scales from simulation options. |
| [cosmictiger/math/units.hpp](../cosmictiger/math/units.hpp) | Typed CGS quantities and conversion between stored, comoving, proper, and cosmological values. |
| [cosmictiger/math/vector.hpp](../cosmictiger/math/vector.hpp) | Fixed-dimensional host/device vectors, arithmetic, norms, dot products, and mask selection. |
| [cosmictiger/math/vector3.hpp](../cosmictiger/math/vector3.hpp) | Three-component vector specialization with cross products and selection. |
| [cosmictiger/particles/domain.hpp](../cosmictiger/particles/domain.hpp) | Distributed particle-domain decomposition, load balancing, migration, and domain lookup. |
| [cosmictiger/particles/drift.hpp](../cosmictiger/particles/drift.hpp) | Particle drift dispatch, locality coordination, lightcone collection, and SPH bound updates. |
| [cosmictiger/particles/drift_cpu.hpp](../cosmictiger/particles/drift_cpu.hpp) | CPU drift kernel with periodic positions and lightcone-crossing output. |
| [cosmictiger/particles/particle_partition.hpp](../cosmictiger/particles/particle_partition.hpp) | Consistent swapping of particle fields and associated gas/star mappings. |
| [cosmictiger/particles/particle_selection.hpp](../cosmictiger/particles/particle_selection.hpp) | Evenly spaced particle sampling and tracer selection flags. |
| [cosmictiger/particles/particle_types.hpp](../cosmictiger/particles/particle_types.hpp) | Particle-species identifiers and shared type definitions. |
| [cosmictiger/particles/particles.hpp](../cosmictiger/particles/particles.hpp) | Particle structure-of-arrays storage, caches, accessors, initialization, persistence, and species changes. |
| [cosmictiger/scenarios/blast.hpp](../cosmictiger/scenarios/blast.hpp) | Hydrodynamic blast-wave initial conditions and completion diagnostics. |
| [cosmictiger/scenarios/cosmology.hpp](../cosmictiger/scenarios/cosmology.hpp) | Cosmological scenario initialization from the configured background and initial conditions. |
| [cosmictiger/scenarios/diagnostics.hpp](../cosmictiger/scenarios/diagnostics.hpp) | Shared particle sampling and scenario diagnostic records. |
| [cosmictiger/scenarios/disc.hpp](../cosmictiger/scenarios/disc.hpp) | Rotating-disc particle initial conditions. |
| [cosmictiger/scenarios/exact_sod.hpp](../cosmictiger/scenarios/exact_sod.hpp) | Exact one-dimensional Sod Riemann solution and pressure-root calculation. |
| [cosmictiger/scenarios/glass.hpp](../cosmictiger/scenarios/glass.hpp) | Glass particle relaxation setup and glass-file output. |
| [cosmictiger/scenarios/gresho.hpp](../cosmictiger/scenarios/gresho.hpp) | Gresho vortex initial density, pressure, and rotational velocity. |
| [cosmictiger/scenarios/hydro_wave.hpp](../cosmictiger/scenarios/hydro_wave.hpp) | Small-amplitude hydrodynamic wave initial conditions. |
| [cosmictiger/scenarios/jeans_wave.hpp](../cosmictiger/scenarios/jeans_wave.hpp) | Self-gravitating Jeans-wave setup and distributed Fourier-mode diagnostics. |
| [cosmictiger/scenarios/kelvin_helmholtz.hpp](../cosmictiger/scenarios/kelvin_helmholtz.hpp) | Shearing-layer Kelvin–Helmholtz initialization and thermodynamic state. |
| [cosmictiger/scenarios/kelvin_helmholtz_lattice.hpp](../cosmictiger/scenarios/kelvin_helmholtz_lattice.hpp) | Lattice packing and particle-count helpers for Kelvin–Helmholtz layers. |
| [cosmictiger/scenarios/plummer.hpp](../cosmictiger/scenarios/plummer.hpp) | Sampling of Plummer density profiles and equilibrium particle velocities. |
| [cosmictiger/scenarios/polytropic_star.hpp](../cosmictiger/scenarios/polytropic_star.hpp) | Lane–Emden integration, polytropic-star sampling, and final diagnostics. |
| [cosmictiger/scenarios/rayleigh_taylor.hpp](../cosmictiger/scenarios/rayleigh_taylor.hpp) | Stratified Rayleigh–Taylor initial conditions and perturbations. |
| [cosmictiger/scenarios/scenario.hpp](../cosmictiger/scenarios/scenario.hpp) | Scenario registration, dispatch, and collection of diagnostic particle samples. |
| [cosmictiger/scenarios/sedov.hpp](../cosmictiger/scenarios/sedov.hpp) | Self-similar Sedov blast-wave reference solution, quadrature, and root finding. |
| [cosmictiger/scenarios/sod.hpp](../cosmictiger/scenarios/sod.hpp) | Sod shock-tube initial conditions and comparison with the exact solution. |
| [cosmictiger/scenarios/sphere.hpp](../cosmictiger/scenarios/sphere.hpp) | Relaxation of particle positions onto an approximately uniform spherical shell. |
| [cosmictiger/sph/conduction.hpp](../cosmictiger/sph/conduction.hpp) | Pairwise artificial thermal-conduction coefficient and energy exchange. |
| [cosmictiger/sph/cpu.hpp](../cosmictiger/sph/cpu.hpp) | CPU entry points and per-node dispatch for SPH computational passes. |
| [cosmictiger/sph/cuda.hpp](../cosmictiger/sph/cuda.hpp) | CUDA entry points and launch coordination for SPH passes. |
| [cosmictiger/sph/flops.hpp](../cosmictiger/sph/flops.hpp) | Operation accounting for SPH interactions and kernel evaluation. |
| [cosmictiger/sph/geometry.hpp](../cosmictiger/sph/geometry.hpp) | Dimension-aware spatial vectors and periodic SPH particle geometry. |
| [cosmictiger/sph/ghosts.hpp](../cosmictiger/sph/ghosts.hpp) | Distributed gas ghost-leaf export, field refresh, staging validation, and statistics. |
| [cosmictiger/sph/neighbor_buffer.hpp](../cosmictiger/sph/neighbor_buffer.hpp) | Storage and access helpers for gathered SPH neighbor data. |
| [cosmictiger/sph/pair_kick.hpp](../cosmictiger/sph/pair_kick.hpp) | Pair-owned SPH impulses and staged midpoint update bookkeeping. |
| [cosmictiger/sph/parent_tree.hpp](../cosmictiger/sph/parent_tree.hpp) | HPX parent topology over locality domains and collective fan-out. |
| [cosmictiger/sph/particle_flags.hpp](../cosmictiger/sph/particle_flags.hpp) | Packed gas-particle activity and convergence flags. |
| [cosmictiger/sph/particles.hpp](../cosmictiger/sph/particles.hpp) | SPH structure-of-arrays storage, particle mappings, thermodynamics, and checkpoint support. |
| [cosmictiger/sph/run_data.hpp](../cosmictiger/sph/run_data.hpp) | SPH pass parameters, gathered source/sink data, and work/result records. |
| [cosmictiger/sph/sph.hpp](../cosmictiger/sph/sph.hpp) | SPH neighbor walks, staged workspaces, pass orchestration, and distributed pair updates. |
| [cosmictiger/sph/thermodynamics.hpp](../cosmictiger/sph/thermodynamics.hpp) | Conversions among SPH thermal energy, entropy, density, and pressure. |
| [cosmictiger/sph/time_integration.hpp](../cosmictiger/sph/time_integration.hpp) | SPH rung proposal/balancing and hierarchical midpoint event scheduling. |
| [cosmictiger/sph/tree.hpp](../cosmictiger/sph/tree.hpp) | Local SPH octrees, neighbor lists, bounds, activity summaries, and refresh lifecycle. |
| [cosmictiger/sph/variable_smoothing_lengths.hpp](../cosmictiger/sph/variable_smoothing_lengths.hpp) | Smoothing-length derivative corrections used by variable-resolution SPH. |
| [cosmictiger/sph/viscosity/balsara.hpp](../cosmictiger/sph/viscosity/balsara.hpp) | Shear limiter from the divergence and curl of the velocity field. |
| [cosmictiger/sph/viscosity/monaghan.hpp](../cosmictiger/sph/viscosity/monaghan.hpp) | Pairwise Monaghan artificial viscosity and associated thermal work. |
| [cosmictiger/sph/viscosity/morris_and_monaghan.hpp](../cosmictiger/sph/viscosity/morris_and_monaghan.hpp) | Time-dependent bulk-viscosity coefficient evolution and source/decay terms. |
| [cosmictiger/sph/volume.hpp](../cosmictiger/sph/volume.hpp) | Neighbor-volume equations, smoothing-length iteration, and convergence handling. |
| [cosmictiger/stars/springel_and_hernquist/cooling.hpp](../cosmictiger/stars/springel_and_hernquist/cooling.hpp) | Cooling interfaces used by the Springel–Hernquist multiphase model. |
| [cosmictiger/stars/springel_and_hernquist/equations.hpp](../cosmictiger/stars/springel_and_hernquist/equations.hpp) | Springel–Hernquist equilibrium, effective pressure, phase exchange, and star-formation equations. |
| [cosmictiger/stars/star_formation.hpp](../cosmictiger/stars/star_formation.hpp) | Preparation, stochastic gas-to-star conversion, staged evolution, and star-formation diagnostics. |
| [cosmictiger/stars/star_formation_cuda.hpp](../cosmictiger/stars/star_formation_cuda.hpp) | CUDA batch execution of star-formation and multiphase-gas updates. |
| [cosmictiger/stars/star_formation_kernel.hpp](../cosmictiger/stars/star_formation_kernel.hpp) | Host/device numerical kernels for the star-formation model. |
| [cosmictiger/stars/star_formation_state.hpp](../cosmictiger/stars/star_formation_state.hpp) | Persistent and staged gas/star model state and diagnostic records. |

## Application implementations

| File | Contents |
| --- | --- |
| [src/analysis/bh.cpp](../src/analysis/bh.cpp) | Barnes–Hut tree and monopole acceptance for halo binding-energy calculations. |
| [src/analysis/bh.cu](../src/analysis/bh.cu) | CUDA implementation: barnes–Hut tree and monopole acceptance for halo binding-energy calculations. |
| [src/analysis/compress.cpp](../src/analysis/compress.cpp) | Block compression of lightcone positions and velocities, with binary read/write support. |
| [src/analysis/group_entry.cpp](../src/analysis/group_entry.cpp) | Friends-of-friends catalogue records, including mass, kinematics, shape, and ancestry. |
| [src/analysis/group_tree.cpp](../src/analysis/group_tree.cpp) | Distributed spatial tree used to discover friends-of-friends groups. |
| [src/analysis/groups.cpp](../src/analysis/groups.cpp) | Distributed group membership, particle exchange, and catalogue orchestration. |
| [src/analysis/groups_find.cpp](../src/analysis/groups_find.cpp) | Asynchronous friends-of-friends linking through the group tree. |
| [src/analysis/healpix.cu](../src/analysis/healpix.cu) | CUDA accumulation of lightcone particles into HEALPix pixels. |
| [src/analysis/lightcone.cpp](../src/analysis/lightcone.cpp) | Lightcone particle collection, sky partitioning, group finding, and output. |
| [src/analysis/lightcone.cu](../src/analysis/lightcone.cu) | CUDA implementation: lightcone particle collection, sky partitioning, group finding, and output. |
| [src/analysis/lightcone_cpu.cpp](../src/analysis/lightcone_cpu.cpp) | CPU detection and interpolation of particle crossings of the observer's lightcone. |
| [src/analysis/power.cpp](../src/analysis/power.cpp) | Particle density deposition and Fourier-space binning of the matter power spectrum. |
| [src/analysis/power.cu](../src/analysis/power.cu) | CUDA implementation: particle density deposition and Fourier-space binning of the matter power spectrum. |
| [src/analysis/power_cpu.cpp](../src/analysis/power_cpu.cpp) | CPU particle-to-grid density deposition for folded power spectra. |
| [src/analysis/power_cuda.cu](../src/analysis/power_cuda.cu) | CUDA particle-to-grid density deposition for folded power spectra. |
| [src/analysis/rockstar.cpp](../src/analysis/rockstar.cpp) | Phase-space halo finding, subgroup hierarchy, and halo-particle processing. |
| [src/analysis/rockstar.cu](../src/analysis/rockstar.cu) | CUDA implementation: phase-space halo finding, subgroup hierarchy, and halo-particle processing. |
| [src/analysis/rockstar_catalogue.cpp](../src/analysis/rockstar_catalogue.cpp) | Snapshot halo catalogue construction, property aggregation, and output. |
| [src/chemistry/gadget4_cooling.cpp](../src/chemistry/gadget4_cooling.cpp) | Equilibrium cooling tables, gas-state conversion, and thermal evolution using the GADGET-4 model. |
| [src/chemistry/gadget4_cooling.cu](../src/chemistry/gadget4_cooling.cu) | CUDA implementation: equilibrium cooling tables, gas-state conversion, and thermal evolution using the GADGET-4 model. |
| [src/chemistry/recombination.cpp](../src/chemistry/recombination.cpp) | Hydrogen/helium recombination and thermal-history integration for the cosmological background. |
| [src/chemistry/sph.cpp](../src/chemistry/sph.cpp) | Coupling of gas cooling and chemistry to SPH particles and their selected time intervals. |
| [src/core/assert.cpp](../src/core/assert.cpp) | Host/device assertion reporting and debug versus unconditional assertion macros. |
| [src/core/assert.cu](../src/core/assert.cu) | CUDA implementation: host/device assertion reporting and debug versus unconditional assertion macros. |
| [src/core/driver.cpp](../src/core/driver.cpp) | Cosmological simulation evolution, force-step sequencing, output scheduling, and driver state. |
| [src/core/flops.cpp](../src/core/flops.cpp) | CPU/GPU floating-point operation counters, reductions, and reporting. |
| [src/core/flops.cu](../src/core/flops.cu) | CUDA implementation: cPU/GPU floating-point operation counters, reductions, and reporting. |
| [src/core/hpx.cpp](../src/core/hpx.cpp) | Locality topology, HPX initialization, distributed helpers, and annotated execution policies. |
| [src/core/interrupt_profile.cpp](../src/core/interrupt_profile.cpp) | Signal-triggered profiling controls and reporting integration. |
| [src/core/memused.cpp](../src/core/memused.cpp) | Process memory-use measurement. |
| [src/core/options.cpp](../src/core/options.cpp) | Grouped runtime options, configuration parsing, defaults, and cross-option validation. |
| [src/core/profiler.cpp](../src/core/profiler.cpp) | Optional scoped timing/work profiling, phase transitions, counters, and reports. |
| [src/core/profiling_runtime.cpp](../src/core/profiling_runtime.cpp) | Runtime setup and shutdown of profiling facilities. |
| [src/core/scenario_driver.cpp](../src/core/scenario_driver.cpp) | Fixed-time scenario evolution, diagnostics, checkpoints, and outer-interval scheduling. |
| [src/core/stack_trace.cpp](../src/core/stack_trace.cpp) | Stack-trace capture, fatal-signal reporting, and diagnostic output paths. |
| [src/cosmology/background.cpp](../src/cosmology/background.cpp) | Simulation background trajectory, conformal-time sampling, and shared background construction. |
| [src/cosmology/background_plot.cpp](../src/cosmology/background_plot.cpp) | Rendering and writing plots of cosmological background evolution. |
| [src/cosmology/boltzmann.cpp](../src/cosmology/boltzmann.cpp) | Linear Einstein–Boltzmann perturbation evolution and transfer-function integration. |
| [src/cosmology/cosmology.cpp](../src/cosmology/cosmology.cpp) | Simulation-facing cosmological clock, expansion, growth, and halo-fit interfaces. |
| [src/cosmology/evolution.cpp](../src/cosmology/evolution.cpp) | Expansion timestep limits, termination tests, and cosmological energy accounting. |
| [src/cosmology/fft.cpp](../src/cosmology/fft.cpp) | Distributed three-dimensional FFT storage, decomposition, transforms, and density-assignment filters. |
| [src/cosmology/fftw_threads.cpp](../src/cosmology/fftw_threads.cpp) | Routing FFTW parallel loops through the application's threading runtime. |
| [src/cosmology/friedmann.cpp](../src/cosmology/friedmann.cpp) | Friedmann background integration, dense trajectory sampling, and interval quadrature. |
| [src/cosmology/initialize.cpp](../src/cosmology/initialize.cpp) | Linear-power normalization and Zel'dovich/second-order Lagrangian particle initialization. |
| [src/cosmology/zero_order.cpp](../src/cosmology/zero_order.cpp) | Tabulated unperturbed universe, density fractions, and redshift/time/scale conversions. |
| [src/cuda/cuda.cpp](../src/cuda/cuda.cpp) | CUDA runtime initialization, device selection, memory allocation, and host fallback adapters. |
| [src/cuda/cuda_memory.cu](../src/cuda/cuda_memory.cu) | CUDA implementation: pooled host/device allocation, block synchronization, and memory-copy support. |
| [src/cuda/cuda_stream.cpp](../src/cuda/cuda_stream.cpp) | RAII CUDA streams with HPX-aware completion polling and asynchronous transfers. |
| [src/gravity/error.cpp](../src/gravity/error.cpp) | Sampling and comparing production gravity against direct reference forces. |
| [src/gravity/error_cuda.cu](../src/gravity/error_cuda.cu) | CUDA evaluation of direct-reference forces for gravity-error samples. |
| [src/gravity/fmm_order.cpp](../src/gravity/fmm_order.cpp) | Build-selected FMM order, coefficient counts, and storage sizes. |
| [src/gravity/gravity.cpp](../src/gravity/gravity.cpp) | Particle/cell force kernels, softening helpers, and gravity work counters. |
| [src/gravity/kick.cpp](../src/gravity/kick.cpp) | Gravity tree walks, pooled interaction lists, and deferred SPH gas-opening state. |
| [src/gravity/kick.cu](../src/gravity/kick.cu) | CUDA implementation: gravity tree walks, pooled interaction lists, and deferred SPH gas-opening state. |
| [src/gravity/kick_executor.cpp](../src/gravity/kick_executor.cpp) | Submission of prepared gravity kick batches to CUDA execution. |
| [src/gravity/kick_workspace.cpp](../src/gravity/kick_workspace.cpp) | Ownership and aggregation of gravity batch work, results, and device submissions. |
| [src/gravity/reference_table.cpp](../src/gravity/reference_table.cpp) | Periodic gravity correction table indexing, symmetry, interpolation, and construction. |
| [src/gravity/source_access.cpp](../src/gravity/source_access.cpp) | Local/remote gravity source-position access and reader registration. |
| [src/gravity/sources.cpp](../src/gravity/sources.cpp) | FMM source partitions, registered source views, and particle-mass access. |
| [src/gravity/tree_arena.cpp](../src/gravity/tree_arena.cpp) | Gravity-tree storage, multipole transport, node allocation, and remote-node bookkeeping. |
| [src/gravity/tree_build.cpp](../src/gravity/tree_build.cpp) | Recursive construction of gravity trees and source multipoles. |
| [src/io/checkpoint.cpp](../src/io/checkpoint.cpp) | Versioned checkpoint headers, manifests, compatibility checks, and distributed state save/restore. |
| [src/io/output.cpp](../src/io/output.cpp) | Particle, tracer, and slice output with checked file handling. |
| [src/io/view.cpp](../src/io/view.cpp) | Distributed Silo particle views, species fields, unit conversion, metadata, and visualization expressions. |
| [src/main.cpp](../src/main.cpp) | Program entry, MPI/FFTW/HPX initialization, capability checks, and simulation dispatch. |
| [src/math/fp16.cpp](../src/math/fp16.cpp) | Half-precision type support, numeric limits, and conversion helpers. |
| [src/math/kernel.cpp](../src/math/kernel.cpp) | Selected hydrodynamic/gravitational radial kernels and host/device selection state. |
| [src/math/kernel.cu](../src/math/kernel.cu) | CUDA implementation: selected hydrodynamic/gravitational radial kernels and host/device selection state. |
| [src/particles/domain.cpp](../src/particles/domain.cpp) | Distributed particle-domain decomposition, load balancing, migration, and domain lookup. |
| [src/particles/drift.cpp](../src/particles/drift.cpp) | Particle drift dispatch, locality coordination, lightcone collection, and SPH bound updates. |
| [src/particles/drift.cu](../src/particles/drift.cu) | CUDA implementation: particle drift dispatch, locality coordination, lightcone collection, and SPH bound updates. |
| [src/particles/drift_cpu.cpp](../src/particles/drift_cpu.cpp) | CPU drift kernel with periodic positions and lightcone-crossing output. |
| [src/particles/particle_partition.cpp](../src/particles/particle_partition.cpp) | Consistent swapping of particle fields and associated gas/star mappings. |
| [src/particles/particles.cpp](../src/particles/particles.cpp) | Particle structure-of-arrays storage, caches, accessors, initialization, persistence, and species changes. |
| [src/scenarios/blast.cpp](../src/scenarios/blast.cpp) | Hydrodynamic blast-wave initial conditions and completion diagnostics. |
| [src/scenarios/cosmology.cpp](../src/scenarios/cosmology.cpp) | Cosmological scenario initialization from the configured background and initial conditions. |
| [src/scenarios/disc.cpp](../src/scenarios/disc.cpp) | Rotating-disc particle initial conditions. |
| [src/scenarios/exact_sod.cpp](../src/scenarios/exact_sod.cpp) | Exact one-dimensional Sod Riemann solution and pressure-root calculation. |
| [src/scenarios/glass.cpp](../src/scenarios/glass.cpp) | Glass particle relaxation setup and glass-file output. |
| [src/scenarios/gresho.cpp](../src/scenarios/gresho.cpp) | Gresho vortex initial density, pressure, and rotational velocity. |
| [src/scenarios/hydro_wave.cpp](../src/scenarios/hydro_wave.cpp) | Small-amplitude hydrodynamic wave initial conditions. |
| [src/scenarios/jeans_wave.cpp](../src/scenarios/jeans_wave.cpp) | Self-gravitating Jeans-wave setup and distributed Fourier-mode diagnostics. |
| [src/scenarios/kelvin_helmholtz.cpp](../src/scenarios/kelvin_helmholtz.cpp) | Shearing-layer Kelvin–Helmholtz initialization and thermodynamic state. |
| [src/scenarios/plummer.cpp](../src/scenarios/plummer.cpp) | Sampling of Plummer density profiles and equilibrium particle velocities. |
| [src/scenarios/polytropic_star.cpp](../src/scenarios/polytropic_star.cpp) | Lane–Emden integration, polytropic-star sampling, and final diagnostics. |
| [src/scenarios/rayleigh_taylor.cpp](../src/scenarios/rayleigh_taylor.cpp) | Stratified Rayleigh–Taylor initial conditions and perturbations. |
| [src/scenarios/scenario.cpp](../src/scenarios/scenario.cpp) | Scenario registration, dispatch, and collection of diagnostic particle samples. |
| [src/scenarios/sedov.cpp](../src/scenarios/sedov.cpp) | Self-similar Sedov blast-wave reference solution, quadrature, and root finding. |
| [src/scenarios/sod.cpp](../src/scenarios/sod.cpp) | Sod shock-tube initial conditions and comparison with the exact solution. |
| [src/scenarios/sphere.cpp](../src/scenarios/sphere.cpp) | Relaxation of particle positions onto an approximately uniform spherical shell. |
| [src/scenarios/sphere.cu](../src/scenarios/sphere.cu) | CUDA implementation: relaxation of particle positions onto an approximately uniform spherical shell. |
| [src/sph/cpu.cpp](../src/sph/cpu.cpp) | CPU entry points and per-node dispatch for SPH computational passes. |
| [src/sph/cuda.cu](../src/sph/cuda.cu) | CUDA entry points and launch coordination for SPH passes. |
| [src/sph/ghosts.cpp](../src/sph/ghosts.cpp) | Distributed gas ghost-leaf export, field refresh, staging validation, and statistics. |
| [src/sph/hydro.cu](../src/sph/hydro.cu) | CUDA SPH pressure, viscosity, and thermal-force interactions. |
| [src/sph/hydro_cpu.cpp](../src/sph/hydro_cpu.cpp) | CPU/SIMD SPH pressure, viscosity, and thermal-force interactions. |
| [src/sph/parent_tree.cpp](../src/sph/parent_tree.cpp) | HPX parent topology over locality domains and collective fan-out. |
| [src/sph/particles.cpp](../src/sph/particles.cpp) | SPH structure-of-arrays storage, particle mappings, thermodynamics, and checkpoint support. |
| [src/sph/prehydro.cu](../src/sph/prehydro.cu) | CUDA smoothing-length solves, density estimates, and velocity/thermodynamic gradients. |
| [src/sph/prehydro_cpu.cpp](../src/sph/prehydro_cpu.cpp) | CPU smoothing-length solves, density estimates, and velocity/thermodynamic gradients. |
| [src/sph/rungs.cu](../src/sph/rungs.cu) | CUDA SPH timestep estimation and neighbor rung constraints. |
| [src/sph/rungs_cpu.cpp](../src/sph/rungs_cpu.cpp) | CPU SPH timestep estimation and neighbor rung constraints. |
| [src/sph/sph.cpp](../src/sph/sph.cpp) | SPH neighbor walks, staged workspaces, pass orchestration, and distributed pair updates. |
| [src/sph/time_integration.cpp](../src/sph/time_integration.cpp) | SPH rung proposal/balancing and hierarchical midpoint event scheduling. |
| [src/sph/tree.cpp](../src/sph/tree.cpp) | Local SPH octrees, neighbor lists, bounds, activity summaries, and refresh lifecycle. |
| [src/stars/star_formation.cpp](../src/stars/star_formation.cpp) | Preparation, stochastic gas-to-star conversion, staged evolution, and star-formation diagnostics. |
| [src/stars/star_formation_cuda.cu](../src/stars/star_formation_cuda.cu) | CUDA batch execution of star-formation and multiphase-gas updates. |
| [src/tools/groups2halofunc.cpp](../src/tools/groups2halofunc.cpp) | Standalone conversion of group catalogues into halo mass functions. |
| [src/tools/halo_catalog_export.cpp](../src/tools/halo_catalog_export.cpp) | Standalone validation and export of native snapshot halo catalogues to JSON/text. |
| [src/tools/healpix2silo.cpp](../src/tools/healpix2silo.cpp) | Standalone conversion of HEALPix maps into Silo visualization data. |

## Tests and test support

| File | Contents |
| --- | --- |
| [tests/analysis_backends.cpp](../tests/analysis_backends.cpp) | Application checks of analysis backends, lightcone groups/maps, and density deposition. |
| [tests/analysis_tools.cpp](../tests/analysis_tools.cpp) | Standalone catalogue, mass-function, and map conversion tool tests. |
| [tests/apex_profiler.cpp](../tests/apex_profiler.cpp) | HPX task, continuation, migration, and ordinary-function profiling regression executable. |
| [tests/application_test.cpp](../tests/application_test.cpp) | Adapter that runs application-level checks through the test framework. |
| [tests/application_test.hpp](../tests/application_test.hpp) | Adapter that runs application-level checks through the test framework. |
| [tests/bh.cpp](../tests/bh.cpp) | Barnes–Hut potentials against direct sums, including CUDA workspace reuse. |
| [tests/boltzmann.cpp](../tests/boltzmann.cpp) | Perturbation initial conditions, convergence, and invalid-input tests. |
| [tests/boltzmann_reference.cpp](../tests/boltzmann_reference.cpp) | Standalone Boltzmann reference-output executable. |
| [tests/cell_lattice.cpp](../tests/cell_lattice.cpp) | Lattice neighbor shells, packing pairs, and particle-count rounding tests. |
| [tests/compression.cpp](../tests/compression.cpp) | Round-trip and edge-case checks of compressed lightcone particles. |
| [tests/cosmology_evolution.cpp](../tests/cosmology_evolution.cpp) | Expansion timesteps, termination, and cosmological energy-accounting tests. |
| [tests/cosmology_flops.cpp](../tests/cosmology_flops.cpp) | Floating-point operation accounting for cosmological background helpers. |
| [tests/cuda_async.cpp](../tests/cuda_async.cpp) | CUDA stream completion, HPX progress, launch failure, and destructor behavior tests. |
| [tests/cuda_async.cu](../tests/cuda_async.cu) | CUDA stream completion, HPX progress, launch failure, and destructor behavior tests. |
| [tests/cuda_memory.cpp](../tests/cuda_memory.cpp) | Host-side allocation-pool concurrency, publication, reuse, and failure-injection tests. |
| [tests/cuda_memory_backend.cpp](../tests/cuda_memory_backend.cpp) | Host memory backend substitutes used by CUDA memory tests. |
| [tests/cuda_memory_cuda.cu](../tests/cuda_memory_cuda.cu) | On-device allocation-boundary, payload, and concurrent reuse tests. |
| [tests/cuda_memory_stubs/cosmictiger/core/containers.hpp](../tests/cuda_memory_stubs/cosmictiger/core/containers.hpp) | Minimal container substitutes for isolated CUDA memory tests. |
| [tests/cuda_memory_stubs/cosmictiger/cuda/cuda.hpp](../tests/cuda_memory_stubs/cosmictiger/cuda/cuda.hpp) | CUDA runtime declarations/substitutes for isolated allocation-pool tests. |
| [tests/cuda_memory_stubs/cosmictiger/math/math.hpp](../tests/cuda_memory_stubs/cosmictiger/math/math.hpp) | Minimal math substitutes for isolated CUDA memory tests. |
| [tests/cuda_test.hpp](../tests/cuda_test.hpp) | Shared fixture that checks CUDA device availability before running GPU tests. |
| [tests/device_profiler.cu](../tests/device_profiler.cu) | Device phase timing, counter correctness, and numerical-output preservation checks. |
| [tests/distributed_output.cpp](../tests/distributed_output.cpp) | Multi-locality Silo output, metadata, clock, and mass-expression checks. |
| [tests/drift_cpu.cpp](../tests/drift_cpu.cpp) | CPU drift positions, activity gates, fixed-point precision, and lightcone crossings. |
| [tests/drift_dispatch.cpp](../tests/drift_dispatch.cpp) | CPU/CUDA dispatch, empty ranges, lightcone gates, and SPH bound updates. |
| [tests/fftw_threads.cpp](../tests/fftw_threads.cpp) | FFTW worker callback and nested HPX parallelism tests. |
| [tests/fixed.cpp](../tests/fixed.cpp) | Fixed-point arithmetic, conversions, rounding, and floating-point exception regressions. |
| [tests/flop_counting_real.hpp](../tests/flop_counting_real.hpp) | Instrumented scalar type that counts arithmetic in numerical kernel tests. |
| [tests/flops.cpp](../tests/flops.cpp) | Host/distributed operation-counter tests and precision-category accounting. |
| [tests/flops_cuda.cu](../tests/flops_cuda.cu) | Device operation-counter accumulation and batching tests. |
| [tests/fp16.cpp](../tests/fp16.cpp) | Half-precision encodings, rounding boundaries, special values, overflow, and numeric limits. |
| [tests/friedmann.cpp](../tests/friedmann.cpp) | Friedmann trajectories, cosmological branches, dense output, and interval integration tests. |
| [tests/gadget4_cooling.cpp](../tests/gadget4_cooling.cpp) | Cooling equilibrium, temperature/energy conversion, and thermal evolution tests. |
| [tests/gravity/cartesian_kernel_test_helpers.hpp](../tests/gravity/cartesian_kernel_test_helpers.hpp) | Shared input records and all-order dispatch for Cartesian kernel tests. |
| [tests/gravity/cartesian_kernels.cpp](../tests/gravity/cartesian_kernels.cpp) | Cartesian moments/translations against polynomial references and SIMD parity checks. |
| [tests/gravity/cartesian_kernels_cuda.cu](../tests/gravity/cartesian_kernels_cuda.cu) | Device parity tests for generated Cartesian FMM kernels at supported orders. |
| [tests/gravity/cartesian_l2l_cuda.cu](../tests/gravity/cartesian_l2l_cuda.cu) | Warp-cooperative local-expansion translation against scalar/reference results. |
| [tests/gravity/cuda_workload_guard.cpp](../tests/gravity/cuda_workload_guard.cpp) | CUDA launch interceptor that enforces bounded test workloads and reports launches. |
| [tests/gravity/direct_reference.cpp](../tests/gravity/direct_reference.cpp) | Direct periodic gravity against an independent Ewald reference. |
| [tests/gravity/error_cuda.cpp](../tests/gravity/error_cuda.cpp) | CPU/CUDA parity for direct gravity-error reference evaluation. |
| [tests/gravity/fmm_acceptance.cpp](../tests/gravity/fmm_acceptance.cpp) | Scalar/SIMD opening criteria, reciprocal interactions, and hybrid-cost selection tests. |
| [tests/gravity/fmm_acceptance_cuda.cu](../tests/gravity/fmm_acceptance_cuda.cu) | Device parity for FMM acceptance, box geometry, and interaction selection. |
| [tests/gravity/fmm_api.cpp](../tests/gravity/fmm_api.cpp) | Typed FMM facade and point-field behavior tests. |
| [tests/gravity/fmm_geometry.cpp](../tests/gravity/fmm_geometry.cpp) | Exact periodic fixed-coordinate displacements, cell boundaries, and box-distance tests. |
| [tests/gravity/kick_device_profile.cu](../tests/gravity/kick_device_profile.cu) | CUDA kick profiling records and preservation of reference force results. |
| [tests/gravity/reference_table.cpp](../tests/gravity/reference_table.cpp) | Periodic correction-table symmetries and interpolation against reference values. |
| [tests/gravity/tree_source_moments.cpp](../tests/gravity/tree_source_moments.cpp) | Source-moment precision, serialization, and cloud-transport checks. |
| [tests/gresho.cpp](../tests/gresho.cpp) | Gresho vortex profile and initialization tests. |
| [tests/groups_catalogue.cpp](../tests/groups_catalogue.cpp) | Distributed group catalogues, baryon apertures, identities, and ancestry persistence. |
| [tests/halo_catalog_fixture.cpp](../tests/halo_catalog_fixture.cpp) | Writer for binary/text halo fixtures consumed by exporter tests. |
| [tests/lightcone_cpu.cpp](../tests/lightcone_cpu.cpp) | CPU lightcone friends-of-friends connectivity, remote activity, and deterministic-label tests. |
| [tests/lightcone_cuda.cu](../tests/lightcone_cuda.cu) | CUDA lightcone connectivity compared with expected group labels. |
| [tests/lightcone_epoch.cpp](../tests/lightcone_epoch.cpp) | Emission epoch, species inclusion, and lightcone physical-unit conversion tests. |
| [tests/linear_power.cpp](../tests/linear_power.cpp) | Power interpolation, variance normalization, and matter-transfer conversion tests. |
| [tests/lpt.cpp](../tests/lpt.cpp) | Fourier-mode generation, lattice shifts, and second-order displacement-source checks. |
| [tests/matrix.cpp](../tests/matrix.cpp) | Dense/symmetric/diagonal matrix arithmetic, inverses, determinants, and linear-system tests. |
| [tests/matrix_cuda.cu](../tests/matrix_cuda.cu) | Device matrix linear-system checks and CPU/CUDA parity. |
| [tests/matrix_eigen.cpp](../tests/matrix_eigen.cpp) | Symmetric eigenpair residuals, scaling, and convergence/status tests. |
| [tests/matrix_eigen_cuda.cu](../tests/matrix_eigen_cuda.cu) | Device parity for symmetric matrix eigensolvers. |
| [tests/mpi_thread_override.cpp](../tests/mpi_thread_override.cpp) | MPI interposition shim used to simulate reported thread-support levels. |
| [tests/native_recombination.cpp](../tests/native_recombination.cpp) | Native recombination physical limits, convergence, validation, and background adapter tests. |
| [tests/no_cuda_runtime.cpp](../tests/no_cuda_runtime.cpp) | Link-time guard against unintended CUDA runtime use in CPU-only configurations. |
| [tests/numerical_utilities.cpp](../tests/numerical_utilities.cpp) | Periodic positions, boxes, sphere intersections, and range arithmetic tests. |
| [tests/particle_count.cpp](../tests/particle_count.cpp) | Particle-count rounding and overflow/invalid-input tests. |
| [tests/particle_selection.cpp](../tests/particle_selection.cpp) | Evenly spaced tracer selection, including large particle counts. |
| [tests/phase_profiler.cpp](../tests/phase_profiler.cpp) | Host phase/work timing and device-work report validation. |
| [tests/power_cpu.cpp](../tests/power_cpu.cpp) | CPU folded density deposition, empty inputs, and uniform-lattice tests. |
| [tests/power_cuda.cu](../tests/power_cuda.cu) | CUDA density deposition against shared reference fixtures. |
| [tests/power_test_helpers.hpp](../tests/power_test_helpers.hpp) | Shared density-assignment fixtures, reference weights, and mass-conservation checks. |
| [tests/radial_density_kernel.cpp](../tests/radial_density_kernel.cpp) | SPH density-kernel values, normalization, derivatives, and dimension-dependent checks. |
| [tests/radial_kernel.cpp](../tests/radial_kernel.cpp) | Radial polynomial tables, softened force/potential, and legacy-kernel comparisons. |
| [tests/radial_kernel_cuda.cu](../tests/radial_kernel_cuda.cu) | Host/device parity for radial gravity and density kernels. |
| [tests/range_cuda_compile.cu](../tests/range_cuda_compile.cu) | CUDA compilation coverage for range operations. |
| [tests/rayleigh_taylor.cpp](../tests/rayleigh_taylor.cpp) | Stratified initialization and unresolved-lattice failure tests. |
| [tests/recombination_background.cpp](../tests/recombination_background.cpp) | Consistency checks of the recombination-derived background history. |
| [tests/recombination_reference.cpp](../tests/recombination_reference.cpp) | Standalone reference-output executable for recombination history. |
| [tests/recombination_thermal.cpp](../tests/recombination_thermal.cpp) | Compton coefficient and stable temperature-evolution tests. |
| [tests/rockstar.cpp](../tests/rockstar.cpp) | Halo unbinding, properties, hierarchy/history, catalogue publication, and binary round trips. |
| [tests/rockstar_core.cpp](../tests/rockstar_core.cpp) | Phase-space connectivity and geometric helper tests. |
| [tests/rockstar_properties.cpp](../tests/rockstar_properties.cpp) | Overdensity thresholds, shape eigenpairs, NFW fitting, and Klypin scale-radius tests. |
| [tests/rung_merge.cpp](../tests/rung_merge.cpp) | Timestep rung-merging decisions and aggregate ladder-cost tests. |
| [tests/safe_io.cpp](../tests/safe_io.cpp) | Short-read/write errors and formatted exception behavior tests. |
| [tests/scenarios/sedov.cpp](../tests/scenarios/sedov.cpp) | Sedov blast-wave reference solution tests. |
| [tests/scenarios/sphere_surface.cpp](../tests/scenarios/sphere_surface.cpp) | Spherical-shell relaxation, unit-radius, polyhedron, and spacing checks. |
| [tests/simd.cpp](../tests/simd.cpp) | SIMD arithmetic, conversions, masks, periodic subtraction, and partial-packet tests. |
| [tests/softening_flops.cpp](../tests/softening_flops.cpp) | Measured operation counts for gravitational softening. |
| [tests/sph/balsara/shear_limiter.cpp](../tests/sph/balsara/shear_limiter.cpp) | Balsara limiter behavior for compression, shear, and degenerate gradients. |
| [tests/sph/chemistry.cpp](../tests/sph/chemistry.cpp) | Gas cooling stage timing, restart, target uniqueness, and source-completion tests. |
| [tests/sph/conduction_cases.hpp](../tests/sph/conduction_cases.hpp) | Shared conduction-pair and dissipative-rate superposition tests. |
| [tests/sph/cpu.cpp](../tests/sph/cpu.cpp) | CPU SPH fixtures and tests for smoothing, density, gradients, forces, viscosity, and integration. |
| [tests/sph/cuda.cu](../tests/sph/cuda.cu) | CUDA SPH tests and CPU/device parity for smoothing, forces, timesteps, and staged updates. |
| [tests/sph/dimensions.cpp](../tests/sph/dimensions.cpp) | Lower-dimensional SPH kernel, state, and variational-force checks. |
| [tests/sph/flops.cpp](../tests/sph/flops.cpp) | Arithmetic accounting for SPH radial kernels. |
| [tests/sph/ghosts.cpp](../tests/sph/ghosts.cpp) | Distributed ghost neighbors, field refresh, ownership, staging, and copy consistency. |
| [tests/sph/global_timestep_cases.hpp](../tests/sph/global_timestep_cases.hpp) | Shared raw global-timestep selection test cases. |
| [tests/sph/kernels.cpp](../tests/sph/kernels.cpp) | Reference SPH kernel values and derivative helpers/tests. |
| [tests/sph/kernels.hpp](../tests/sph/kernels.hpp) | Reference SPH kernel values and derivative helpers/tests. |
| [tests/sph/midpoint_cases.hpp](../tests/sph/midpoint_cases.hpp) | Shared coarse/fine midpoint targeting and viscosity endpoint tests. |
| [tests/sph/monaghan/equations.cpp](../tests/sph/monaghan/equations.cpp) | Monaghan pair viscosity, conservation, and vector geometry tests. |
| [tests/sph/morris_and_monaghan/viscosity.cpp](../tests/sph/morris_and_monaghan/viscosity.cpp) | Time-dependent viscosity growth and decay tests. |
| [tests/sph/no_gravity.cpp](../tests/sph/no_gravity.cpp) | SPH-only application diagnostics without gravitational forces. |
| [tests/sph/pair_kick_cases.hpp](../tests/sph/pair_kick_cases.hpp) | Shared conservative pair-impulse and compensated-update tests. |
| [tests/sph/parent_tree.cpp](../tests/sph/parent_tree.cpp) | Locality-parent topology and distributed HPX tree lifecycle checks. |
| [tests/sph/rung_limiter_cases.hpp](../tests/sph/rung_limiter_cases.hpp) | Shared inactive-neighbor closure and support-boundary timestep tests. |
| [tests/sph/source_completion_cases.hpp](../tests/sph/source_completion_cases.hpp) | Shared deferred-source completion tests. |
| [tests/sph/source_timing.cpp](../tests/sph/source_timing.cpp) | Distributed scheduling and completion checks for gas source terms. |
| [tests/sph/star_formation.cpp](../tests/sph/star_formation.cpp) | Staged star-formation evolution, ledgers, restart, and distributed state tests. |
| [tests/sph/star_formation_cuda.cu](../tests/sph/star_formation_cuda.cu) | CPU/CUDA parity for multiphase gas and star-formation batches. |
| [tests/sph/test_helpers.hpp](../tests/sph/test_helpers.hpp) | Shared SPH assertion and approximate-comparison helpers. |
| [tests/sph/tree.cpp](../tests/sph/tree.cpp) | SPH tree metadata, neighbors, volume bounds, refresh/rebuild, timing, and distributed layout checks. |
| [tests/sph/volume.cpp](../tests/sph/volume.cpp) | Variational volume/smoothing solves, constraints, energy gradients, and kernel comparisons. |
| [tests/springel_and_hernquist.cpp](../tests/springel_and_hernquist.cpp) | Multiphase equilibrium, pressure, source evolution, and model validation tests. |
| [tests/springel_and_hernquist_cuda.cu](../tests/springel_and_hernquist_cuda.cu) | Device parity for Springel–Hernquist model evaluations. |
| [tests/stack_trace.cpp](../tests/stack_trace.cpp) | Signal handler lifecycle/restoration and controlled crash executable. |
| [tests/stack_trace_hpx.cpp](../tests/stack_trace_hpx.cpp) | Distributed HPX stack-trace inspection and controlled crash executable. |
| [tests/star_formation_particles.cpp](../tests/star_formation_particles.cpp) | Gas/star conversion, persistent identities, chemistry fields, and migration tests. |
| [tests/terminal.cpp](../tests/terminal.cpp) | Duration, verbosity, and rung-population formatting tests. |
| [tests/test.cpp](../tests/test.cpp) | Application-level test dispatch with FFT, domain, chemistry, and gravity checks. |
| [tests/test.hpp](../tests/test.hpp) | Application-level test dispatch with FFT, domain, chemistry, and gravity checks. |
| [tests/units.cpp](../tests/units.cpp) | Typed physical conversions, fixed storage, bounds, cosmological factors, and validation tests. |
| [tests/units_cuda_compile.cu](../tests/units_cuda_compile.cu) | Compilation coverage for host/device fixed-position unit conversions. |
| [tests/units_output.cpp](../tests/units_output.cpp) | Silo units, field expressions, command-line metadata, and output-clock checks. |
| [tests/vector.cpp](../tests/vector.cpp) | Vector dimension, arithmetic, scalar promotion, and selection tests. |

## Benchmarks

| File | Contents |
| --- | --- |
| [benchmarks/math/matrix.cpp](../benchmarks/math/matrix.cpp) | Standalone timing of matrix operations over multiple dimensions and scalar types. |

## CMake source templates

| File | Contents |
| --- | --- |
| [src/gravity/fmm_instance_cpu.cpp.in](../src/gravity/fmm_instance_cpu.cpp.in) | Per-order CPU FMM explicit-instantiation translation unit configured by CMake. |
| [src/gravity/fmm_instance_cuda.cu.in](../src/gravity/fmm_instance_cuda.cu.in) | Per-order CUDA kick explicit-instantiation translation unit configured by CMake. |

## Kernel generators and supporting tools

| File | Contents |
| --- | --- |
| [tools/generate_fmm.py](../tools/generate_fmm.py) | Main FMM generator: exact rational expression graphs, optimization, emission, and consistency checks. |
| [tools/cartesian/types.py](../tools/cartesian/types.py) | Emission of compact Cartesian coefficient types and scalar/SIMD storage. |
| [tools/cartesian/translations.py](../tools/cartesian/translations.py) | Cartesian multipole/local translations and their polynomial contractions. |
| [tools/cartesian/particles.py](../tools/cartesian/particles.py) | Cartesian particle-to-expansion and expansion-to-particle kernel generation. |
| [tools/cartesian/cuda.py](../tools/cartesian/cuda.py) | Warp-cooperative CUDA translation kernel generation. |
| [tools/halo_viewer/halo_viewer.py](../tools/halo_viewer/halo_viewer.py) | Local halo-catalogue viewer server and catalogue data access. |
| [scripts/run_mpi.py](../scripts/run_mpi.py) | MPI launch wrapper for configured application/test runs. |
| [scripts/plot_cosmology.py](../scripts/plot_cosmology.py) | Plotting utility for cosmological output. |

## Restored global FMM

| Source | Responsibility |
| --- | --- |
| [cosmictiger/gravity/global_fmm/distributed_tree.hpp](../cosmictiger/gravity/global_fmm/distributed_tree.hpp) | Distributed global particle tree construction. |
| [src/gravity/global_fmm/distributed_tree.cpp](../src/gravity/global_fmm/distributed_tree.cpp) | Distributed global particle tree construction. |
| [cosmictiger/gravity/global_fmm/execution.hpp](../cosmictiger/gravity/global_fmm/execution.hpp) | Distributed FMM task execution. |
| [src/gravity/global_fmm/execution.cpp](../src/gravity/global_fmm/execution.cpp) | Distributed FMM task execution. |
| [cosmictiger/gravity/global_fmm/remote_cache.hpp](../cosmictiger/gravity/global_fmm/remote_cache.hpp) | Remote node and multipole cache. |
| [src/gravity/global_fmm/remote_cache.cpp](../src/gravity/global_fmm/remote_cache.cpp) | Remote node and multipole cache. |
| [cosmictiger/gravity/global_fmm/particle_exchange.hpp](../cosmictiger/gravity/global_fmm/particle_exchange.hpp) | Remote particle source reads. |
| [src/gravity/global_fmm/particle_exchange.cpp](../src/gravity/global_fmm/particle_exchange.cpp) | Remote particle source reads. |
| [cosmictiger/gravity/global_fmm/gpu_staging.hpp](../cosmictiger/gravity/global_fmm/gpu_staging.hpp) | CUDA work and remote source staging. |
| [src/gravity/global_fmm/gpu_staging.cpp](../src/gravity/global_fmm/gpu_staging.cpp) | CUDA work and remote source staging. |
