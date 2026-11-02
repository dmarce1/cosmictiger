"""One bounded decoder for the current native checkpoint test format.

Layouts mirror checkpoint.cpp, particlesSave, sphParticlesSave and domainsSave.
Checkpoint files are native-endian/native-ABI; this helper reads files produced on
this host. Optional blocks are consumed in writer order, never found by magic
searches inside particle data. Unsupported model versions fail before offsets
can drift into a following block.
"""

import ctypes
import math
import struct
from types import SimpleNamespace


class Header(ctypes.Structure):
    _fields_ = [("magic", ctypes.c_uint64), ("version", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("physics", ctypes.c_double * 11),
                ("particles_per_species", ctypes.c_uint64), ("steps", ctypes.c_int32),
                ("reserved", ctypes.c_uint32)]


class Driver(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in
               ("a", "adot", "dummy", "tau", "tau_max", "energy0")]
    _fields_ += [("energies", ctypes.c_double * 10)]
    _fields_ += [(name, ctypes.c_int) for name in
                ("max_rung", "iteration", "min_rung", "step", "bucket_size")]
    _fields_ += [("processed", ctypes.c_size_t)]
    _fields_ += [(name, ctypes.c_double) for name in ("flops", "runtime", "years")]
    _fields_ += [("integer_time", ctypes.c_uint64)]


class Domain(ctypes.Structure):
    _fields_ = [("box", ctypes.c_double * 6), ("midpoint", ctypes.c_double),
                ("rank", ctypes.c_int)]


class DomainEntry(ctypes.Structure):
    _fields_ = [("key", ctypes.c_size_t), ("domain", Domain)]


class HierarchicalSource(ctypes.Structure):
    """Native sph_hierarchical::SourceState, including padding before epoch."""
    _fields_ = [("id", ctypes.c_uint64)]
    _fields_ += [(name, ctypes.c_double * 3) for name in
                 ("position", "drift_velocity", "synchronized_velocity", "hydro_acceleration")]
    _fields_ += [(name, ctypes.c_double) for name in
                 ("internal_energy", "internal_energy_rate", "density", "log_density_rate",
                  "physical_divergence", "support_radius", "minimum_support_radius",
                  "maximum_support_radius", "grad_h_correction", "grad_h_correction_rate",
                  "balsara_factor", "viscosity_alpha", "pressure_energy_ratio", "adiabatic_index")]
    _fields_ += [("rung", ctypes.c_int)]
    _fields_ += [(name, ctypes.c_double) for name in ("epoch", "valid_until", "drift_scale")]


def hierarchical_source_values(source):
    values = []
    for name, layout in source._fields_:
        value = getattr(source, name)
        values.extend(value if issubclass(layout, ctypes.Array) else (value,))
    return tuple(values)


def local_domain_type(integer_bytes):
    class LocalDomain(ctypes.Structure):
        _fields_ = [("box", ctypes.c_double * 6), ("process_range", ctypes.c_int * 2),
                    ("particle_range", (ctypes.c_int32 if integer_bytes == 4
                                        else ctypes.c_int64) * 2), ("depth", ctypes.c_int)]
    return LocalDomain


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class Reader:
    def __init__(self, data):
        self.data = data
        self.offset = 0

    def skip(self, size):
        require(0 <= size <= len(self.data) - self.offset,
                f"truncated checkpoint at byte {self.offset}: need {size} bytes")
        offset = self.offset
        self.offset += size
        return offset

    def take(self, code, count=1):
        require(count >= 0, "negative checkpoint array size")
        offset = self.skip(struct.calcsize("=" + code) * count)
        return struct.unpack_from(f"={count}{code}", self.data, offset)

    def native(self, layout):
        return layout.from_buffer_copy(self.data, self.skip(ctypes.sizeof(layout)))


def read_header(data):
    header = Reader(data).native(Header)
    require(header.magic == 0x4354535048303031 and header.version == 3,
            "unsupported checkpoint header")
    supported = sum(1 << bit for bit in (0, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21))
    require(not header.flags & (1 << 16), "retired particle-mesh checkpoint")
    require(not header.flags & ~supported, "unsupported checkpoint feature flags")
    return header


def read_prefix(data):
    reader = Reader(data)
    header = read_header(data)
    reader.skip(ctypes.sizeof(Header))
    flags = header.flags
    hydro, cooling, stars = (bool(flags & (1 << bit)) for bit in (0, 11, 20))
    require(not cooling or hydro, "cooling checkpoint requires SPH")
    require(not stars or cooling, "star formation checkpoint requires cooling")
    require(hydro or not flags & ((1 << 12) | (1 << 13) | (1 << 14)),
            "SPH parameter flags require SPH")
    result = SimpleNamespace(header=header, hydro=hydro, has_cooling=cooling, has_stars=stars,
                             position_bytes=8 if flags & (1 << 15) else 4,
                             integer_bytes=8 if flags & (1 << 21) else 4,
                             algorithms=(), pmmm=(), viscosity=(), cooling_parameters=(),
                             star_parameters=(), sph_parameters=())
    if flags & (1 << 18):
        result.algorithms = reader.take("Q", 5)
        version, far, near, degree, integrator = result.algorithms
        require(version == 1 and far in (0, 1, 2) and near in (0, 1, 2)
                and (degree == 0 if far == 0 else 3 <= degree <= 9)
                and (near != 0 if far == 2 else near == 0) and integrator in (0, 1, 2),
                "invalid checkpoint algorithm selection")
        require((integrator == 2) == bool(flags & (1 << 19)) and
                (integrator == 1) == bool(flags & (1 << 6)),
                "checkpoint algorithm and integrator flags disagree")
    if flags & (1 << 17):
        version = reader.take("Q")[0]
        require(version in (1, 2, 3, 4, 5), "unsupported PMMM checkpoint version")
        result.pmmm = (version, *reader.take("Q", {1: 6, 2: 7, 3: 9, 4: 10, 5: 10}[version]))
    viscosity_offset = 0
    if hydro:
        result.sph_parameters = reader.take("d", 13)
        if flags & (1 << 12):
            result.z_plane_acceleration = reader.take("d")[0]
        if flags & (1 << 14):
            result.viscosity = reader.take("d", 7)
            require(all(math.isfinite(value) for value in result.viscosity) and
                    result.viscosity[2] in (0.0, 1.0, 2.0),
                    "invalid checkpoint viscosity parameters")
            viscosity_offset = 2 + 2 * int(result.viscosity[2])
    if cooling:
        result.cooling_offset = reader.offset
        result.cooling_parameters = reader.take("d", 6)
        require(result.cooling_parameters[0] == 6.0,
                "unsupported cooling checkpoint model version")
        require(all(math.isfinite(value) for value in result.cooling_parameters),
                "nonfinite checkpoint cooling parameters")
    if stars:
        result.star_parameters = reader.take("Q", 11)
        require(result.star_parameters[0] == 3, "unsupported star-formation model version")
    result.driver_offset = reader.offset
    result.driver = reader.native(Driver)
    result.driver.integer_phase_offset = result.driver_offset + Driver.integer_time.offset
    result.driver.global_timestep = 0.0
    if flags & (1 << 19):
        result.driver.global_extension_offset = reader.offset
        version = reader.take("Q")[0]
        result.driver.global_timestep = reader.take("d")[0]
        require(version == 1 and math.isfinite(result.driver.global_timestep)
                and result.driver.global_timestep >= 0, "invalid checkpoint global timestep")
    result.particle_start = reader.offset
    result.has_hierarchical_sph = (hydro and bool(flags & (1 << 6)) and
                                   ((header.reserved >> 25) & 3) == 0)
    result.gas_version = ((90 if stars else 80 if cooling else 2) + viscosity_offset
                          + (1000 if hydro and flags & (1 << 6) else 0))
    result.reader = reader
    return result


def decode_checkpoint(data, ranks=1):
    result = read_prefix(data)
    reader = result.reader
    integer_code = "q" if result.integer_bytes == 8 else "i"
    count = result.count = reader.take(integer_code)[0]
    gas_count = result.gas_count = reader.take(integer_code)[0] if result.hydro else 0
    require(0 <= gas_count <= count <= result.header.particles_per_species * (2 if result.hydro else 1),
            "invalid checkpoint particle counts")
    result.positions = [reader.take("Q" if result.position_bytes == 8 else "I", count)
                        for _ in range(3)]
    result.velocities = reader.take("f", 3 * count)
    result.rungs = reader.take("b", count)
    if result.header.flags & (1 << 8):
        result.groups = reader.take("q", count)
    if result.header.flags & (1 << 9):
        result.tracers = reader.take("b", count)
    result.types = reader.take("b", count) if result.hydro else (0,) * count
    result.gas_indices = reader.take(integer_code, count) if result.hydro else ()
    result.cooling, result.stars = (), ()
    result.hierarchical_sources, result.hierarchical_bytes = (), b""
    result.gas = [() for _ in range(gas_count)]
    if result.hydro:
        result.gas_start = reader.offset
        gas_header = reader.take("I", 7)
        require(gas_header == (0x43545350, result.gas_version, result.integer_bytes, 8, 12, 3, 4),
                f"unsupported checkpoint gas layout: {gas_header}")
        require(reader.take(integer_code)[0] == gas_count, "checkpoint gas counts disagree")
        result.thermodynamic = reader.take("f", 2 * gas_count)
        accelerations = reader.take("f", 3 * gas_count)
        result.gravity_indices = reader.take(integer_code, gas_count)
        scalars = [reader.take("f", gas_count) for _ in range(3)]
        flags = [reader.take("B", gas_count) for _ in range(2)]
        vectors = [reader.take("f", gas_count) for _ in range(6)]
        if result.has_hierarchical_sph:
            anchor_start = reader.offset
            result.hierarchical_sources = tuple(
                hierarchical_source_values(reader.native(HierarchicalSource))
                for _ in range(gas_count))
            result.hierarchical_bytes = data[anchor_start:reader.offset]
        # Versions 2/4/6/8 and 80/82/84/86 have different fixed-state bases.
        delta = (result.gas_version - (90 if result.has_stars else 80 if result.has_cooling else 2)
                 - (1000 if result.has_hierarchical_sph else 0))
        width = {0: 0, 2: 2, 4: 3, 6: 3}[delta]
        viscosity = reader.take("f", width * gas_count)
        if result.has_cooling:
            result.cooling = tuple(reader.take("f", 3) for _ in range(gas_count))
        result.gas_end = reader.offset
        require(all(kind in (0, 1, 2) for kind in result.types) and
                result.types.count(1) == gas_count, "invalid checkpoint particle species")
        require(sorted(result.gravity_indices) == [i for i, kind in enumerate(result.types) if kind == 1],
                "missing or duplicate checkpoint gas mapping")
        for gas_index, particle in enumerate(result.gravity_indices):
            require(result.gas_indices[particle] == gas_index, "inconsistent checkpoint gas mapping")
            result.gas[gas_index] = (
                *result.thermodynamic[2 * gas_index:2 * gas_index + 2],
                *accelerations[3 * gas_index:3 * gas_index + 3],
                *(field[gas_index] for field in scalars),
                *(field[gas_index] for field in flags),
                *(field[gas_index] for field in vectors),
                *viscosity[width * gas_index:width * (gas_index + 1)],
                *(result.cooling[gas_index] if result.has_cooling else ()))
    if result.has_stars:
        require(reader.take("Q", 2) == (0x4354534650415254, 2),
                "unsupported checkpoint star-particle metadata")
        result.stars = tuple((*reader.take("Q", 2), *reader.take("d", 3)) for _ in range(count))
    if result.has_stars:
        identities = [state[0] for state in result.stars]
        require(all(identities) and len(set(identities)) == count,
                "duplicate or zero checkpoint particle identities")
        for kind, (_, _, formation, rate, elapsed) in zip(result.types, result.stars):
            require(all(math.isfinite(value) for value in (formation, rate, elapsed))
                    and rate >= 0 and elapsed >= 0 and (kind == 1 or rate == 0)
                    and ((formation >= 0) == (kind == 2)),
                    "invalid checkpoint star-particle state")
    result.particle_end = reader.offset
    rung_count = reader.take("i")[0]
    rung_begin = reader.take(integer_code)[0]
    require(0 <= rung_count <= 64 and 0 <= rung_begin <= count, "invalid checkpoint rung stack")
    rung_begins = reader.take(integer_code, rung_count)
    require(all(0 <= begin <= rung_begin for begin in rung_begins) and
            list(rung_begins) == sorted(rung_begins), "invalid nested checkpoint rung ranges")
    domain_count = reader.native(ctypes.c_size_t).value
    reader.skip(domain_count * ctypes.sizeof(DomainEntry))
    require(ranks > 0, "checkpoint locality count must be positive")
    reader.skip(ranks * ctypes.sizeof(local_domain_type(result.integer_bytes)))
    require(not result.header.flags & (1 << 7), "lightcone checkpoint decoding is unsupported")
    require(reader.offset == len(data), "trailing unparsed checkpoint data")
    require(all(math.isfinite(value) for value in result.velocities) and
            all(math.isfinite(value) for gas in result.gas for value in gas) and
            all(math.isfinite(value) for source in result.hierarchical_sources for value in source),
            "nonfinite checkpoint particle state")
    result.physical = data[result.particle_start:result.particle_end]
    return result
