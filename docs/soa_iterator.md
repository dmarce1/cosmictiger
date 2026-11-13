# SoA iteration and particle partitioning

`cosmictiger/soa_iterator.hpp` provides a C++20 random-access iterator over
independent arrays. Dereferencing produces a proxy row; swapping rows swaps every
field. `std::ranges::partition` can therefore partition on one field while keeping
the remaining fields aligned.

```cpp
auto first = soa::Iterator(position.data(), velocity.data(), identity.data());
static_assert(std::random_access_iterator<decltype(first)>);
static_assert(std::permutable<decltype(first)>);
auto tail = std::ranges::partition(first, first + position.size(),
    [split](auto const& row) {
        using std::get;
        return get<0>(row) < split;
    });
auto boundary = tail.begin() - first;
```

All fields must have the same row extent and independent, nonoverlapping storage;
passing the same field twice is invalid. The iterator does not own storage.
Reallocation invalidates iterators and proxies. End iterators are formed from the
same base iterator plus the row count. Empty ranges do not dereference storage.
Use C++20 ranges algorithms: the legacy iterator category is deliberately input,
since a row is a proxy rather than an actual reference to an owning tuple.

`iter_move` produces an owning tuple, and row assignment writes each field.
Move-only fields work with partition and row moves. A const-field iterator is
read-only. Field swaps use their normal C++ swap semantics; additional cross-array
maps or side effects require an adapter or the existing specialized swap routine.

For builds with `COSMICTIGER_ENABLE_SPH=OFF`, `particlesPartition` first scans the
already-correct prefix and suffix using only the position key. If any rearrangement
is needed, it selects a SoA iterator containing all position axes, velocity, rung,
and the enabled acceleration and potential fields. The split remains exact in the
stored fixed-point coordinate, with values equal to the split on the right.

Hydro/species metadata, star-formation state, local or atomic group labels, tracers,
and gravity-check sample indices retain the original scan with `particlesSwap`.
SPH-enabled builds retain the original implementation throughout. Optional storage
that is disabled is never dereferenced.

## Validation

Build the `soa_iterator_test` and `particle_partition_test` targets with
`BUILD_TESTING=ON`, then run:

```sh
ctest --test-dir build/soa_partition_tests \
    -R '^(soa_iterator|particle_partition)$' --output-on-failure
build/soa_partition_tests/tests/bin/particle_partition_test --benchmark 1048576 9
```

The tests check empty/singleton ranges, nonzero subranges, every small binary
arrangement, fixed-point split equality and adjacent ticks, all optional particle
fields, field identity, special floating-point representations, move-only values,
and iterator arithmetic. Checks remain active in release builds. Validation passed
with ASan/UBSan for the generic iterator and 3D/fixed32 SPH-off partition; release
checks also passed for 3D/fixed64 SPH off, 3D/fixed32 SPH on, and the supported
1D/fixed32 and 2D/fixed64 SPH-on/gravity-off configurations. Both new CTest targets
and the full SPH-off release build passed.

The optional benchmark compares the production entry point to the original nested
scan from main, preserving the baseline's external `particlesSwap` call. It restores
identical data outside the timed interval, alternates A/B order, excludes one
warm-up pair, and reports medians for single and recursive partitions down to 256
particles. Acceleration and potential storage are enabled.

## Measured tree-build performance

Measured on this workstation against the original partition in main commit
`21ef694fbebfa605e1b11b76ddaae8830f5eb784`, with GCC 15.2 and the checkout's release
optimization flags (`-O3`, `-march=native`, `-ffast-math -fno-finite-math-only`). Both
executables use identical application objects except the partition entry point.
The baseline scan is compiled separately from `particlesSwap`, preserving the
original call boundary; no LTO is enabled.

The comparison uses the existing z0, z2, and z20 inputs tiled 8³ times to
16,777,216 particles, bucket size 256, two HPX workers, one locality, and saved
acceleration/potential fields. Force solving is omitted in a shared temporary
benchmark object; the normal production `treeCreate` is timed. Diagnostic force
fields are initialized to zero in that fixture. No GPU is used. The run order is
main/SoA/SoA/main, with 11 builds per input per process. Initial-build columns are
medians of two samples; repeated-build columns are medians of 20 samples. All
node, leaf, and tree FLOP counts match.

| Input / theta | Main initial (s) | SoA initial (s) | Reduction | Main repeated (s) | SoA repeated (s) | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| z0 / 0.70 | 1.700512 | 1.395850 | 17.9% | 0.839204 | 0.780040 | 7.0% |
| z2 / 0.55 | 1.680986 | 1.370642 | 18.5% | 0.791350 | 0.745901 | 5.7% |
| z20 / 0.40 | 1.562807 | 1.309421 | 16.2% | 0.756830 | 0.717193 | 5.2% |

The one-thread partition benchmark (1,048,576 particles, nine measured repetitions)
reports production/original median ratios of 0.789/0.732 for random single/recursive
partitions, 0.748/0.717 for clustered data, and 0.728/0.443 for prepartitioned data.
These results establish partition and CPU tree-build improvements, not a measured
full CUDA force-solve speedup.

Local scripts, raw logs, preserved pre-edit source, and extracted samples are in
`build/soa_partition_validation/`. The executable's runtime log confirms two HPX
workers; its existing benchmark summary `threads` column reports a hardware
concurrency estimate of 12, not the configured worker count. The production
application also passed a CPU tree-and-force smoke run with 512 particles and two
repetitions.
