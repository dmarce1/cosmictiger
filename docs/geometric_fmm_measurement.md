# Geometric FMM measurement

Implemented 2026-11-14 over the uncommitted geometric-tree/source-forest work.
This measures exchange effects using the same geometric tree, not the legacy tree.
The production exchange and demand-fetch fallback are retained.

## Running a bounded comparison

`benchmark.compareSourceForest=true` enables paired complete-source and exported
walks. Existing benchmark behavior and `bench.N.dat` stay unchanged when false.
The other benchmark options still select input files, matching opening angles,
tiling, and repeats. Each pair uses one frozen source generation and all rung-zero
sinks, with unchanged velocities, ownership, positions, softening, order and device.
Demand caches are cleared before each walk. Reference traversal can run before any
export. The first path alternates by repeat, with no omitted warmup.

The runner below creates exactly a 96-particle deterministic case and a 32³-particle
case in two Gaussian halos, then runs three paired repeats on two local MPI ranks
with two HPX threads each. Obtain the applicable resource/workload authorization
before using it. It disables GPU backends and nested numerical worker pools.

```sh
python3 scripts/measure_geometric_fmm.py build/pmmm_only_cpu/cosmictiger /tmp/fmm_measurement_run
```

Use a fresh result directory. Inputs, the exact command and options, CPU/platform,
CMake cache, executable and source SHA-256 hashes, and the tracked patch are saved.
The source manifest includes new source files from the geometric implementation.
`tests/check_fmm_measurement.py RESULT_DIRECTORY` checks accuracy, timing arithmetic,
peer-direction symmetry, coverage, used/imported node counts and zero fallback use
without running additional work. The ordinary benchmark regression also exercises
the new mode on its small fixtures.

## Timer and counter interpretation

Three CSVs are written per input case, with one-based repeat numbers. The paired
configuration column `total_worker_threads` is HPX's aggregate across localities;
legacy `bench.N.dat` records threads local to one rank:


- `fmm_measurement.N.pairs.csv`: normalized configuration and coordinator wall
  intervals. Loading and completed ownership decomposition/migration are separate
  one-time setup intervals. The local-build interval dispatches and joins owned
  builds, without exchange. Exchange includes begin-import initialization, export
  selection/packing, dispatch, receive/install, owner joins and final readiness.
  Each force interval includes all-locality preflight, traversal, demand fetching
  or imported reads, and completed device work when selected. Shared cleanup
  retires trees, clears particle caches and ends measurement. Pair wall time
  additionally includes cache resets, metric collection and accuracy diagnostics.
- `fmm_measurement.N.ranks.csv`: local elapsed task intervals and memory estimates.
  Local build includes source-cover construction and owner registration. Export
  packing includes payload-size accounting; owner exchange includes packing and
  transfer completion. Exposed wait starts after the last destination dispatch and
  ends at the owner's joined transfers. Import includes validation, map insertion
  and size accounting; deserialization precedes this action body. GPU staging ends
  before device execution and is nested inside force traversal. These times can
  overlap within/across ranks; do not sum them to reconstruct wall time.
- `fmm_measurement.N.peers.csv`: source-owner/destination directions, application
  messages and serialized object bytes, exported/imported nodes and particles,
  coarse terminals, unique imported nodes read, and actual source counts represented
  by imported leaves/terminals. `coarse_sources` is the subset represented by
  multipole terminals; `fine_detail` identifies particle-bearing peers. Zero-count
  owners and destinations remain represented by the protocol. A complete-source
  demand miss counts its index request and returned cache line once, independent
  of cache hits or concurrent readers. Requester counters are mirrored when the
  coordinator collects metrics, giving both peer directions.

`reference_path_s` is shared build + reference force + shared cleanup;
`export_path_s` adds exchange to shared build + exported force + shared cleanup.
These are **reconstructed path costs on the same frozen tree**, not independently
measured evolution-step totals. They deliberately exclude diagnostics. The actual
joined pair wall is reported separately. Loading/domain work is identical shared
setup, reported separately rather than repeated or hidden in either force timer.

Payload bytes use HPX's preprocessing archive on the actual transmitted application
object. They include its archive/container representation and native position
precision, and exclude HPX action/parcel envelopes, acknowledgements, root-descriptor
broadcasts, readiness/preflight control, MPI framing, and measurement RPCs. They
are not NIC byte counters. The fallback counter retains the existing attempted-read
and materialization semantics; it is a zero-use guard, not a count of network
misses. Counts distinguish export pushes from demand requests
and responses. There is no claim that fewer payload bytes lower total wall time.

The owned-arena count includes reserved nodes, positioned multipoles and population
flags. Imported storage estimates map values, link pointers and buckets; export
vector storage records the largest constructed destination view's capacity. The
in-flight payload count is a conservative bound retaining every dispatched payload
until the owner's join. Demand-cache bytes count retained vector capacity, excluding
map/future overhead. Remote multipole bytes count allocated chunks for the shared
epoch, so a reference-first walk can contribute to a subsequent export snapshot.
Peak RSS is the process lifetime high-water mark in bytes, not a per-path reset.
These fields distinguish logical reservations, estimated container storage and
resident memory; they do not measure all transient builder, allocator, HPX or MPI
storage. Imported storage is sampled after a joined walk, when imports persist.
Reference rows zero export/import-only fields even in export-first pairs; epoch
multipole storage and peak RSS still describe the actual shared process.

Force/potential differences are checked component-wise against
`2e-5 * max(1, abs(first_path_value))`, with separate relative L2 norms. Small
untiled inputs of at most 256 particles also check acceleration against the
independent periodic direct reference using native-quantized positions, with
`max(1e-5, theta^(order-1))` relative L2 tolerance. `-1` means this bounded direct
check was not run. Every walk checks finite force/potential, unchanged velocities
and rung-zero activity. Exported walks must use zero demand-fetch fallbacks.

Measurement is opt-in. Timed paths include size accounting, lightweight use flags,
and counter locks, so they are instrumented costs. No accuracy or timing evidence
here establishes mixed-rung evolution, production scaling, different precision,
GPU performance, physical network failure behavior, or momentum conservation.
