# Halo Explorer

Halo Explorer imports completed CosmicTiger ROCKSTAR snapshots into a local
SQLite database and serves a read-only browser interface. Python 3.10 or later
and its standard library are sufficient; JavaScript, styles, and plots are
included locally, without a CDN or Python package installation.

## Build, import, and open

Build the native reader using the simulation's source and producing toolchain:

```sh
cmake --build build/release --target halo_catalog_export --parallel 6
```

Import one snapshot directory, including every rank's property and host files:

```sh
python3 tools/halo_viewer/halo_viewer.py import /path/to/groups.17 \
    --database /path/to/halos.sqlite \
    --exporter build/release/halo_catalog_export \
    --name "Run A, snapshot 17" --hubble 0.7
```

`--hubble` records the dimensionless `h` from the run configuration. It is
optional; unknown cosmology is left unspecified, and stored properties retain
their documented units. Repeating an import of the same files and `h` returns
the existing catalog. Multiple snapshots and runs can share one database.

Start the local browser service:

```sh
python3 tools/halo_viewer/halo_viewer.py serve --database /path/to/halos.sqlite
```

Open `http://127.0.0.1:8765`. Use `--port` to choose another port. The service
binds to loopback, accepts localhost requests, and exposes read-only queries.
Stop it with Ctrl-C. An empty database opens with import instructions.

## Explore a snapshot

- Choose the imported catalog. The overview reports its epoch and exact halo,
  distinct-halo, and subhalo counts.
- Apply ranges for the catalogue's main mass/radius, maximum circular velocity, or spin; search
  by exact halo ID; select distinct halos or subhalos. Expand **Spatial region**
  to bound a volume or slice using inclusive X/Y/Z coordinate ranges.
- Switch between XY, XZ, and YZ projections. Both axes use the same comoving
  distance per pixel. Click a point or select a table row with the mouse or
  keyboard to inspect the halo.
- Follow its containing host, ancestors, or direct subhalos. Following a link
  preserves the active filters; an inspector note identifies a halo outside the
  current selection.
- Export the complete matching selection as CSV, including all fields and exact
  decimal IDs. Pagination and spatial sampling never truncate an export. The
  selection metadata download includes units, filters, and source-file hashes.
- The URL preserves catalog, filters, sort, projection, page, and selected halo
  for use with the same database. IDs remain strings throughout the browser.

Table queries and plots use one filter implementation. Plots show a deterministic
sample of at most 20,000 points by default and explicitly report displayed and
matching counts. Bounds are computed from the entire selection. Coordinate
filters use an R*Tree only to narrow candidates and then check the stored
double-precision coordinates, including exact boundaries.

Halo positions are comoving Mpc/h; radii are comoving kpc/h; masses are solar
masses/h; velocities are km/s. The separate box length in `groups.dat` is Mpc,
without `h`. Density conventions are recorded per catalogue. For supported
Bryan-Norman backgrounds, `mvir` and `rvir` retain their virial meanings and
`m200`, `m500`, and `m2500` mean M200c, M500c, and M2500c, relative to critical
density. When the catalogue uses the 200-times-mean-matter fallback, those same
stored fields are displayed as `M200m_main`, `r200m`, M200m, M500m, and M2500m.
The 500-overdensity shape labels also use the matching reference. Filters,
table headings, inspector values, plot tooltips, and downloaded selection
metadata use these catalogue-specific definitions; CSV field names stay stable.
Records have inclusive halo masses, and `sys` describes internal substructure; distinct
versus subhalo classification uses the actual host link.

## Import validation and provenance

The native reader shares the simulation's `GroupHeader` and `RockstarRecord`
declarations. It requires `rockstar_hosts.complete`, `groups.dat`,
`rockstar_descendants.dat`, and all rank property/host files. Every property file
must have a `.format` sidecar identifying version 2, IEEE binary16, matching
native record size, byte order, and record count. Files from the earlier custom
half encoding are rejected. The surrounding raw records remain native ABI
dependent; use the producing toolchain rather than assuming cross-platform
compatibility from record size alone.

New sidecars also record `virial_density` (`bryan_norman_flat`,
`bryan_norman_open`, or `200_mean_matter`) and `so_density_reference` (`critical`
or `mean_matter`). Both fields must agree across ranks. Legacy sidecars without
them retain Bryan-Norman/critical-density labels; their flat/open distinction
is unknown. The viewer upgrades existing version-1 SQLite databases in place
to store these conventions, preserving their catalogs and halo records.

Record row `i` corresponds to host-sidecar row `i` within the same rank, as
guaranteed by the current writer. The reader checks this correspondence against
the FOF/local `pid` host index, rejects duplicate IDs, missing hosts, cycles,
nonfinite properties, truncated files, and count mismatches, and validates all
files before producing JSONL. Every property is written as a JSON real: the
shortest decimal that recovers the stored binary64, binary32, or binary16 value
exactly, so integral magnitudes such as a halo energy of -9.84e16 stay
parseable. Persistent snapshot IDs and packed local record indices remain
distinct.

The importer hashes its source files before and after decoding, inserts in one
transaction, and publishes a catalog only after successful validation. An import
failure preserves existing catalogs. The database records source paths, file
sizes/SHA-256 hashes, epoch, rank count, supplied `h`, and import time. Native
export validation currently holds the snapshot's records and identity indexes in
memory; very large catalogs need enough importer memory. Browser memory stays
bounded by table/point response limits.

The first milestone covers snapshots. Lightcone geometry, particle views,
measured density profiles, and descendant timelines are future work; the
descendant file is required for completion provenance but is not interpreted as
a temporal tree here.

## Validation

```sh
cmake --build build/release --target halo_catalog_export halo_catalog_fixture --parallel 6
ctest --test-dir build/release --output-on-failure -R '^halo_viewer$'
```

The integration test uses a native 64-halo/two-rank fixture, including nested
hosts, IDs larger than JavaScript's exact integer range, and IEEE half
subnormals. It independently decodes every half field with Python's standard
binary16 reader and exercises SQLite import, filters, exact spatial boundaries,
pagination, host navigation, complete CSV export, density-reference metadata,
legacy database migration, and rejected/corrupted imports.
