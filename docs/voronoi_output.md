# Voronoi gas output

`output.gasMesh=voronoi` selects a polyhedral Voronoi representation of the gas
particles in particle Silo snapshots. It is the default. Each gas particle is a
generating point; dark-matter particles remain a point mesh. The Voronoi cells
are a visualization of the current particle positions. The evolution method
remains SPH.

Use the option on the command line:

```sh
./cosmictiger --configuration=../../configurations/rayleigh_taylor.ini \
    --compute.cuda=false --output.gasMesh=voronoi --hpx:threads=12 --hpx:bind=none
```

Or set it in an INI file:

```ini
[output]
gasMesh = voronoi
```

`output.gasMesh=spheres` selects the existing CSG sphere representation, with
the SPH kernel support radius as each sphere's radius. The geometry selector
does not enable snapshots: `output.silo=true` is still required.

## Silo representation

Voronoi output uses a Silo UCD mesh named `gas` with a polyhedral zonelist.
The polygonal cell faces are stored explicitly, so VisIt can display their
edges and planar surfaces without discretizing overlapping CSG volumes.
Gas quantities are zone-centered values associated with the generating
particle. In particular, `rho` remains the SPH density; a visualization cell's
volume does not redefine density as particle mass divided by Voronoi volume.

The output follows the selected `output.units` and `output.frame`. Coordinates
are converted after the geometry is constructed, retaining the same cell
connectivity in the different unit and frame views. `kernel_support_radius`
remains an SPH diagnostic and does not set the size of a Voronoi cell.

## Periodic boundaries and output bounds

The tessellation uses the periodic unit cube in all three dimensions. Each
locality constructs complete cells for its owned generating points using
periodic images and the required remote neighbors. Output bounds select
intersections with the completed geometry, rather than selecting generating
points before tessellation. A cell can therefore contribute to a view even
when its generating point is outside that view.

Cells crossing a periodic boundary are wrapped and clipped into pieces within
the requested `output.xMin` through `output.zMax` box. A periodic cell can have
more than one output zone. These pieces retain their parent particle identity
and particle values; they do not represent additional particles. Full-cell and
clipped-piece volumes have different meanings, especially for partial views.

The following zone-centered diagnostics make that distinction explicit:

| Field | Meaning |
| --- | --- |
| `generator_id` | Snapshot-local generating-particle ID: locality in the upper 32 bits and local gas index in the lower 32 bits. |
| `image_offset_x`, `image_offset_y`, `image_offset_z` | Integer translation of the unwrapped parent cell before clipping. |
| `cell_volume` | Full parent-cell volume, repeated on every fragment of that cell. |
| `fragment_volume` | Volume of the displayed clipped fragment. |
| `hydro_mass` | Parent particle mass multiplied by `fragment_volume / cell_volume`. |

Volumes use the displayed length unit cubed. Intensive fields such as density,
specific energy, velocity, and chemical fractions repeat the parent value.
The fragment masses sum to the particle mass for a full periodic-box view;
a partial view includes only the corresponding geometric fraction. Summing
`cell_volume` over fragments double-counts split cells; use `fragment_volume`
for the displayed volume. The files record that topology can change at every
snapshot.

## Distributed files

Each Voronoi snapshot has a small master file and a sibling directory of
locality files:

```text
view.0.<cycle>.silo
view.0.<cycle>.silo.data/
    generation_<unique>/
        locality_0.silo
        locality_1.silo
        ...
```

Open the master file in VisIt. Its multimeshes and multivariables reference
the locality files. Keep the master and its `.silo.data` directory together
when moving or archiving a snapshot. Meshes within the locality files use
`/locality_<rank>/gas`, with corresponding directories for converted views.

Each write uses a fresh generation directory. The master is written under a
temporary name and atomically renamed only after all locality files are closed.
Repeating a snapshot cycle therefore preserves its previous published files
until the new master is ready. Older generations are retained.

Particle and polyhedral arrays are written by their owning locality. Only the
metadata needed to assemble the master file is gathered for final publication.
The optional `spheres` mode retains the legacy gathered single-file writer.

## Relation to future moving-mesh evolution

Cell construction runs on the CPU using the Delaunay dual. The current backend
is Geogram's serial BDEL implementation within each locality, with distributed
parallelism across localities. Its orientation and in-sphere predicates use
filtered exact arithmetic; coordinates and derived cell geometry are floating
point. The fetched dependency is pinned and retains its BSD license.

The geometry interface is separate from the Silo writer. It exposes cell
volumes and centroids, oriented polygonal faces, face areas and centroids,
neighbor identities, and explicit periodic image offsets. This allows the
geometry to be evaluated independently of visualization.

The IDs exported by this implementation identify particles within a snapshot.
They are not yet persistent identifiers across time steps and redistribution.
The current output is not a moving-mesh hydrodynamic method: conservative flux
updates, moving-face velocities, geometric conservation, persistent identities,
and time-integration validation remain separate work.
