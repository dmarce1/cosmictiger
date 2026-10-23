#!/usr/bin/env python3
"""Import completed CosmicTiger snapshots and serve a local halo catalog viewer.

The native exporter decodes the simulation's records. Everything else uses
Python's standard library; the browser assets work without a network connection.
"""

import argparse
from contextlib import closing
import csv
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tempfile
from urllib.parse import parse_qs, unquote, urlencode, urlsplit


HALF_FIELDS = (
    "zValue", "rmax", "rvir", "rvmax", "rs", "rkly", "sigX", "deltaX", "toW", "sigV",
    "sigVb", "vcircMax", "deltaV", "boa", "coa", "ax", "ay", "az", "boa500", "coa500",
    "ax500", "ay500", "az500", "lambda", "lambdaB", "vxc", "vyc", "vzc", "vxb", "vyb", "vzb",
)
FLOAT_FIELDS = ("x", "y", "z", "mvir", "e", "jx", "jy", "jz", "m200", "m500", "m2500", *HALF_FIELDS)
IDENTITY_FIELDS = ("halo_id", "host_id", "fof_id", "local_halo_id", "rank", "ordinal", "sys", "pid")
HALO_FIELDS = (*IDENTITY_FIELDS, *FLOAT_FIELDS)
FILTER_FIELDS = ("mvir", "rvir", "vcircMax", "lambda", "x", "y", "z")
SORT_FIELDS = {"halo_id", "rank", "mvir", "rvir", "vcircMax", "lambda", "x", "y", "z"}
UNITS = {name: "dimensionless" for name in HALF_FIELDS}
UNITS.update({name: "comoving Mpc/h" for name in ("x", "y", "z")})
UNITS.update({name: "solar masses/h" for name in ("mvir", "m200", "m500", "m2500")})
UNITS.update({name: "comoving kpc/h" for name in (
    "rmax", "rvir", "rvmax", "rs", "rkly", "sigX", "deltaX", "ax", "ay", "az", "ax500", "ay500", "az500")})
UNITS.update({name: "km/s" for name in (
    "sigV", "sigVb", "vcircMax", "deltaV", "vxc", "vyc", "vzc", "vxb", "vyb", "vzb")})
UNITS.update({name: "(solar masses/h)(Mpc/h)(km/s), physical" for name in ("jx", "jy", "jz")})
UNITS["e"] = "(solar masses/h)(km/s)^2"
UNITS["box_length_mpc"] = "Mpc"
STATIC_DIRECTORY = Path(__file__).resolve().parent / "static"
SCHEMA_VERSION = 2


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def open_database(path, readonly=False):
    path = Path(path).resolve()
    if readonly:
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def initialize_database(connection):
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version not in (0, 1, SCHEMA_VERSION):
        raise ValueError(f"Unsupported viewer database version {version}")
    if version == 0 and connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchone():
        raise ValueError("Database contains unrelated tables; choose a new database file")
    if version == SCHEMA_VERSION:
        return
    if version == 1:
        # Existing catalogs predate explicit density conventions. Their main
        # mass used Bryan-Norman and their SO masses used critical density.
        connection.executescript(f"""
            BEGIN IMMEDIATE;
            ALTER TABLE catalogs ADD COLUMN virial_density TEXT NOT NULL DEFAULT 'bryan_norman';
            ALTER TABLE catalogs ADD COLUMN so_density_reference TEXT NOT NULL DEFAULT 'critical';
            PRAGMA user_version={SCHEMA_VERSION};
            COMMIT;
        """)
        return
    scalar_columns = ",\n".join(f'"{name}" REAL NOT NULL' for name in FLOAT_FIELDS)
    connection.executescript(f"""
        CREATE TABLE catalogs (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, snapshot INTEGER NOT NULL,
            scale_factor REAL NOT NULL, time REAL NOT NULL, box_length_mpc REAL NOT NULL,
            hubble REAL, source_path TEXT NOT NULL, source_fingerprint TEXT NOT NULL UNIQUE,
            source_files TEXT NOT NULL, imported_at TEXT NOT NULL,
            halo_count INTEGER NOT NULL, rank_count INTEGER NOT NULL,
            virial_density TEXT NOT NULL, so_density_reference TEXT NOT NULL
        );
        CREATE TABLE halos (
            row_id INTEGER PRIMARY KEY, catalog_id INTEGER NOT NULL REFERENCES catalogs(id),
            sample_key INTEGER NOT NULL,
            halo_id TEXT NOT NULL, host_id TEXT, fof_id TEXT NOT NULL,
            local_halo_id INTEGER NOT NULL, rank INTEGER NOT NULL, ordinal INTEGER NOT NULL,
            sys INTEGER NOT NULL, pid INTEGER NOT NULL,
            {scalar_columns},
            UNIQUE(catalog_id, halo_id), UNIQUE(catalog_id, rank, ordinal),
            FOREIGN KEY(catalog_id, host_id) REFERENCES halos(catalog_id, halo_id)
                DEFERRABLE INITIALLY DEFERRED
        );
        CREATE VIRTUAL TABLE halo_positions USING rtree(id,x_min,x_max,y_min,y_max,z_min,z_max);
        CREATE TRIGGER halo_insert AFTER INSERT ON halos BEGIN
            INSERT INTO halo_positions VALUES(new.row_id,new.x,new.x,new.y,new.y,new.z,new.z);
        END;
        CREATE TRIGGER halo_delete AFTER DELETE ON halos BEGIN
            DELETE FROM halo_positions WHERE id=old.row_id;
        END;
        CREATE INDEX halo_hosts ON halos(catalog_id,host_id,mvir DESC,halo_id);
        CREATE INDEX halo_mass ON halos(catalog_id,mvir DESC,halo_id);
        CREATE INDEX halo_radius ON halos(catalog_id,rvir,halo_id);
        CREATE INDEX halo_vmax ON halos(catalog_id,vcircMax,halo_id);
        CREATE INDEX halo_spin ON halos(catalog_id,lambda,halo_id);
        CREATE INDEX halo_sample ON halos(catalog_id,sample_key,row_id);
        PRAGMA user_version={SCHEMA_VERSION};
    """)
    connection.execute("PRAGMA journal_mode=WAL")


def snapshot_sources(snapshot):
    manifest = (snapshot / "rockstar_hosts.complete").read_text().split()
    if len(manifest) != 3 or manifest[0] != "1":
        raise ValueError("Missing or invalid snapshot completion manifest")
    ranks, count = (int(value) for value in manifest[1:])
    if ranks <= 0 or ranks > 1000000 or count < 0:
        raise ValueError("Invalid rank or halo count in completion manifest")
    names = ["groups.dat", "rockstar_hosts.complete", "rockstar_descendants.dat"]
    for rank in range(ranks):
        names.extend((f"rockstar.{rank}.dat", f"rockstar.{rank}.dat.format", f"rockstar_hosts.{rank}.dat"))
    return names


def fingerprint_sources(snapshot, names):
    files = []
    digest = hashlib.sha256()
    for name in names:
        path = snapshot / name
        if not path.is_file():
            raise ValueError(f"Snapshot file is missing: {path}")
        checksum = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                checksum.update(chunk)
        entry = {"name": name, "bytes": path.stat().st_size, "sha256": checksum.hexdigest()}
        files.append(entry)
        digest.update(json.dumps(entry, sort_keys=True).encode())
    return digest.hexdigest(), files


def validate_halo(row):
    if row.get("type") != "halo":
        raise ValueError("Native exporter emitted an unexpected record")
    for key in ("halo_id", "fof_id"):
        if not isinstance(row.get(key), str) or not re.fullmatch(r"[0-9]+", row[key]):
            raise ValueError(f"Invalid textual identifier {key}")
    if row.get("host_id") is not None and not re.fullmatch(r"[0-9]+", row["host_id"]):
        raise ValueError("Invalid host identifier")
    for key in FLOAT_FIELDS:
        if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]):
            raise ValueError(f"Nonfinite or missing halo property {key}")
    for key in ("local_halo_id", "rank", "ordinal", "pid"):
        if type(row.get(key)) is not int:
            raise ValueError(f"Invalid integer halo property {key}")
    return [int(row[key]) if key == "sys" else row[key] for key in HALO_FIELDS]


def density_conventions(metadata):
    fields = ("virial_density", "so_density_reference")
    if any(key in metadata for key in fields) and not all(key in metadata for key in fields):
        raise ValueError("Catalog density conventions must specify both reference fields")
    virial = metadata.get("virial_density", "bryan_norman")
    reference = metadata.get("so_density_reference", "critical")
    valid = ((virial in ("bryan_norman", "bryan_norman_flat", "bryan_norman_open") and
              reference == "critical") or
             (virial == "200_mean_matter" and reference == "mean_matter"))
    if not valid:
        raise ValueError("Unsupported or inconsistent catalog density conventions")
    return virial, reference


def import_snapshot(snapshot, database, exporter, name=None, hubble=None):
    snapshot = Path(snapshot).resolve()
    if not snapshot.is_dir():
        raise ValueError(f"Snapshot directory does not exist: {snapshot}")
    if hubble is not None and (not math.isfinite(hubble) or hubble <= 0):
        raise ValueError("Hubble parameter h must be finite and positive")
    exporter = Path(exporter).resolve()
    if not exporter.is_file():
        raise ValueError(f"Native exporter not found: {exporter}; build target halo_catalog_export")
    names = snapshot_sources(snapshot)
    fingerprint, files = fingerprint_sources(snapshot, names)
    # User-supplied h is part of the import's scientific provenance.
    fingerprint = hashlib.sha256(f"{fingerprint}:{hubble!r}".encode()).hexdigest()
    with closing(open_database(database)) as connection:
        initialize_database(connection)
        connection.execute("BEGIN IMMEDIATE")
        process = None
        try:
            # Revalidate even idempotent imports, so damaged source files cannot
            # be mistaken for an already imported, complete snapshot.
            with tempfile.TemporaryFile(mode="w+") as errors:
                process = subprocess.Popen([str(exporter), str(snapshot)], stdout=subprocess.PIPE,
                                           stderr=errors, text=True, encoding="utf-8")
                metadata_line = process.stdout.readline()
                if not metadata_line:
                    process.wait()
                    errors.seek(0)
                    raise ValueError(errors.read().strip() or "Native exporter returned no catalog")
                metadata = json.loads(metadata_line)
                if metadata.get("type") != "catalog" or metadata.get("schema_version") != 1:
                    raise ValueError("Unsupported native exporter schema")
                virial_density, so_density_reference = density_conventions(metadata)
                existing = connection.execute("SELECT id FROM catalogs WHERE source_fingerprint=?",
                                              (fingerprint,)).fetchone()
                catalog_id = existing["id"] if existing else None
                if catalog_id is None:
                    cursor = connection.execute("""INSERT INTO catalogs
                        (name,snapshot,scale_factor,time,box_length_mpc,hubble,source_path,
                         source_fingerprint,source_files,imported_at,halo_count,rank_count,
                         virial_density,so_density_reference)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                            name or snapshot.name, metadata["snapshot"], metadata["scale_factor"],
                            metadata["time"], metadata["box_length_mpc"], hubble, str(snapshot),
                            fingerprint, json.dumps(files), datetime.now(timezone.utc).isoformat(),
                            metadata["halo_count"], metadata["rank_count"],
                            virial_density, so_density_reference))
                    catalog_id = cursor.lastrowid
                columns = ",".join(f'"{key}"' for key in ("catalog_id", "sample_key", *HALO_FIELDS))
                placeholders = ",".join("?" for _ in range(len(HALO_FIELDS) + 2))
                insert = f"INSERT INTO halos ({columns}) VALUES ({placeholders})"
                batch = []
                count = 0
                for line in process.stdout:
                    values = validate_halo(json.loads(line))
                    count += 1
                    if not existing:
                        # A stable order-independent sample avoids aliasing
                        # FOF record order (for example selecting only hosts).
                        sample_key = int.from_bytes(hashlib.blake2b(
                            values[0].encode("ascii"), digest_size=8,
                            person=b"HaloView").digest(), "big") & (2**63 - 1)
                        batch.append([catalog_id, sample_key, *values])
                        if len(batch) >= 1000:
                            connection.executemany(insert, batch)
                            batch.clear()
                if batch:
                    connection.executemany(insert, batch)
                if process.wait() != 0:
                    errors.seek(0)
                    raise ValueError(errors.read().strip() or "Native export failed")
                if count != metadata["halo_count"]:
                    raise ValueError("Native exported halo count does not match snapshot manifest")
                after, _ = fingerprint_sources(snapshot, names)
                after = hashlib.sha256(f"{after}:{hubble!r}".encode()).hexdigest()
                if fingerprint != after:
                    raise ValueError("Snapshot changed while importing; no catalog was published")
                connection.commit()
                return {"catalog_id": catalog_id, "halo_count": count, "existing": bool(existing),
                        "database": str(Path(database).resolve())}
        except Exception:
            connection.rollback()
            raise
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdout.close()


def catalog_dict(row):
    result = dict(row)
    result["redshift"] = 1.0 / result["scale_factor"] - 1.0
    virial, reference = density_conventions(result)
    result["virial_density"] = virial
    result["so_density_reference"] = reference
    suffix = "m" if reference == "mean_matter" else "c"
    fallback = virial == "200_mean_matter"
    masses = {field: f"M{overdensity}{suffix}" for field, overdensity in
              (("m200", 200), ("m500", 500), ("m2500", 2500))}
    masses["mvir"] = "M200m_main" if fallback else "Bryan-Norman virial"
    result["mass_definitions"] = masses
    result["radius_definitions"] = {"rvir": "r200m" if fallback else "Bryan-Norman virial"}
    labels = dict(masses, mvir="M200m_main" if fallback else "Virial mass",
                  rvir="r200m" if fallback else "Virial radius")
    for field, description in (("boa500", "Axis ratio b/a"), ("coa500", "Axis ratio c/a"),
                               ("ax500", "Shape axis X"), ("ay500", "Shape axis Y"),
                               ("az500", "Shape axis Z")):
        labels[field] = f"{description} (500{suffix})"
    result["field_labels"] = labels
    result["units"] = dict(UNITS)
    for field, definition in masses.items():
        result["units"][field] += f" ({definition})"
    result["units"]["rvir"] += f" ({result['radius_definitions']['rvir']})"
    result["source_files"] = json.loads(result["source_files"])
    return result


def halo_dict(row):
    result = {key: row[key] for key in HALO_FIELDS}
    result["sys"] = bool(result["sys"])
    return result


def catalog_by_id(connection, query):
    try:
        raw = query.get("catalog", "")
        if not re.fullmatch(r"[0-9]+", raw):
            raise ValueError
        identifier = int(raw)
        if identifier > 2**63 - 1:
            raise ValueError
    except ValueError:
        raise ApiError(400, "Choose a valid catalog") from None
    row = connection.execute("SELECT * FROM catalogs WHERE id=?", (identifier,)).fetchone()
    if row is None:
        raise ApiError(404, "Catalog not found")
    return row


def query_integer(query, key, default, minimum=0, maximum=2**31 - 1):
    value = query.get(key, str(default))
    if not re.fullmatch(r"[0-9]+", value):
        raise ApiError(400, f"{key} must be an integer between {minimum} and {maximum}")
    number = int(value)
    if number < minimum or number > maximum:
        raise ApiError(400, f"{key} must be between {minimum} and {maximum}")
    return number


def selection_query(query, catalog):
    allowed = {"catalog", "sort", "order", "limit", "offset", "halo_id", "kind"}
    allowed.update(f"{field}_{suffix}" for field in FILTER_FIELDS for suffix in ("min", "max"))
    if set(query) - allowed:
        raise ApiError(400, "Unknown filter: " + ", ".join(sorted(set(query) - allowed)))
    where = ["h.catalog_id=?"]
    parameters = [catalog["id"]]
    spatial = []
    for field in FILTER_FIELDS:
        bounds = {}
        for suffix, operator in (("min", ">="), ("max", "<=")):
            key = f"{field}_{suffix}"
            if key not in query or query[key] == "":
                continue
            try:
                value = float(query[key])
                if not math.isfinite(value):
                    raise ValueError
            except ValueError:
                raise ApiError(400, f"{key} must be a finite number") from None
            bounds[suffix] = value
            where.append(f'h."{field}" {operator} ?')
            parameters.append(value)
            if field in ("x", "y", "z"):
                # R*Tree is a float32 candidate index; h.field above always
                # applies the exact predicate to the stored double value.
                index_field = f"{field}_{'max' if suffix == 'min' else 'min'}"
                spatial.append((f"p.{index_field} {operator} ?", value))
        if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
            raise ApiError(400, f"{field} minimum must not exceed its maximum")
    kind = query.get("kind", "all")
    if kind == "distinct":
        where.append("h.host_id IS NULL")
    elif kind == "subhalo":
        where.append("h.host_id IS NOT NULL")
    elif kind != "all":
        raise ApiError(400, "kind must be all, distinct, or subhalo")
    if query.get("halo_id"):
        if not re.fullmatch(r"[0-9]{1,20}", query["halo_id"]):
            raise ApiError(400, "halo_id must be an exact decimal identifier")
        where.append("h.halo_id=?")
        parameters.append(str(int(query["halo_id"])))
    source = "halos h"
    if spatial:
        source += " JOIN halo_positions p ON p.id=h.row_id"
        for clause, value in spatial:
            where.append(clause)
            parameters.append(value)
    sort = query.get("sort", "mvir")
    order = query.get("order", "desc")
    if sort not in SORT_FIELDS:
        raise ApiError(400, "Unknown sort column")
    if order not in ("asc", "desc"):
        raise ApiError(400, "order must be asc or desc")
    ordering = (f'length(h.halo_id) {order}, h.halo_id {order}' if sort == "halo_id" else
                f'h."{sort}" {order}, length(h.halo_id) ASC, h.halo_id ASC')
    return source, " AND ".join(where), parameters, ordering


def send_json(handler, value, status=200, attachment=None):
    data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    if attachment:
        handler.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
    handler.end_headers()
    handler.wfile.write(data)


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "HaloExplorer/1"

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; "
                         "style-src 'self'; script-src 'self'; connect-src 'self'; "
                         "frame-ancestors 'none'; base-uri 'none'")
        super().end_headers()

    def log_message(self, format, *args):
        # No source paths or scientific selections in routine request logs.
        return

    def do_GET(self):
        try:
            host = self.headers.get("Host", "").split(":", 1)[0].lower()
            if host not in ("127.0.0.1", "localhost"):
                raise ApiError(403, "This viewer accepts localhost requests only")
            parts = urlsplit(self.path)
            values = parse_qs(parts.query, keep_blank_values=True, max_num_fields=64)
            if any(len(value) != 1 for value in values.values()):
                raise ApiError(400, "Duplicate query parameters are not allowed")
            query = {key: value[0] for key, value in values.items()}
            if parts.path.startswith("/api/"):
                with closing(open_database(self.server.database, readonly=True)) as connection:
                    # Count, rows, and bounds for each response share one SQLite
                    # read snapshot even if a CLI import commits concurrently.
                    connection.execute("BEGIN")
                    self.api(connection, parts.path, query)
            else:
                self.asset(parts.path)
        except ApiError as error:
            send_json(self, {"error": str(error)}, error.status)
        except (ValueError, OverflowError) as error:
            send_json(self, {"error": str(error)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except sqlite3.Error:
            send_json(self, {"error": "Database query failed"}, 500)

    def asset(self, path):
        files = {"/": ("index.html", "text/html"), "/index.html": ("index.html", "text/html"),
                 "/app.js": ("app.js", "text/javascript"), "/app.css": ("app.css", "text/css"),
                 "/static/app.js": ("app.js", "text/javascript"),
                 "/static/app.css": ("app.css", "text/css")}
        if path not in files:
            raise ApiError(404, "File not found")
        name, content_type = files[path]
        data = (STATIC_DIRECTORY / name).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def api(self, connection, path, query):
        if path == "/api/catalogs":
            rows = connection.execute("""SELECT c.*,
                (SELECT count(*) FROM halos h WHERE h.catalog_id=c.id AND h.host_id IS NULL) AS host_count,
                (SELECT count(*) FROM halos h WHERE h.catalog_id=c.id AND h.host_id IS NOT NULL) AS subhalo_count
                FROM catalogs c ORDER BY c.id DESC""").fetchall()
            send_json(self, {"catalogs": [catalog_dict(row) for row in rows]})
            return
        catalog = catalog_by_id(connection, query)
        if path.startswith("/api/halos/"):
            identifier = unquote(path[len("/api/halos/"):])
            row = connection.execute("SELECT * FROM halos WHERE catalog_id=? AND halo_id=?",
                                     (catalog["id"], identifier)).fetchone()
            if row is None:
                raise ApiError(404, "Halo not found")
            ancestors = []
            parent = row["host_id"]
            seen = {identifier}
            while parent is not None:
                if parent in seen:
                    raise ApiError(500, "Invalid cycle in imported host relationships")
                seen.add(parent)
                host = connection.execute("SELECT * FROM halos WHERE catalog_id=? AND halo_id=?",
                                          (catalog["id"], parent)).fetchone()
                if host is None:
                    raise ApiError(500, "Missing imported host halo")
                ancestors.append(halo_dict(host))
                parent = host["host_id"]
            children = connection.execute("""SELECT * FROM halos WHERE catalog_id=? AND host_id=?
                ORDER BY mvir DESC,halo_id""", (catalog["id"], identifier)).fetchall()
            send_json(self, {"halo": halo_dict(row), "host": ancestors[0] if ancestors else None,
                             "ancestors": ancestors, "subhalos": [halo_dict(child) for child in children],
                             "subhalo_count": len(children)})
            return
        if path not in ("/api/halos", "/api/points", "/api/export", "/api/export_metadata"):
            raise ApiError(404, "Endpoint not found")
        source, where, parameters, ordering = selection_query(query, catalog)
        offset = query_integer(query, "offset", 0)
        limit = query_integer(query, "limit", 20000 if path == "/api/points" else 50,
                              minimum=1, maximum=50000 if path == "/api/points" else 1000)
        if path == "/api/export_metadata":
            metadata = catalog_dict(catalog)
            send_json(self, {"catalog": metadata, "filters": query, "units": metadata["units"]},
                      attachment=f'halo_selection_{catalog["id"]}.json')
            return
        if path == "/api/export":
            self.export(connection, catalog, query, source, where, parameters, ordering)
            return
        if path == "/api/halos":
            total = connection.execute(f"SELECT count(*) FROM {source} WHERE {where}", parameters).fetchone()[0]
            rows = connection.execute(f"SELECT h.* FROM {source} WHERE {where} ORDER BY {ordering} LIMIT ? OFFSET ?",
                                      [*parameters, limit, offset]).fetchall()
            send_json(self, {"total": total, "items": [halo_dict(row) for row in rows],
                             "limit": limit, "offset": offset})
            return
        summary = connection.execute(f"""SELECT count(*) AS total,
            min(h.x),max(h.x),min(h.y),max(h.y),min(h.z),max(h.z)
            FROM {source} WHERE {where}""", parameters).fetchone()
        total = summary[0]
        rows = connection.execute(f"""SELECT h.halo_id,h.host_id,h.x,h.y,h.z,h.mvir,h.rvir
            FROM {source} WHERE {where} ORDER BY h.sample_key,h.row_id LIMIT ?""",
                                  [*parameters, limit]).fetchall()
        items = [{key: row[key] for key in ("halo_id", "host_id", "x", "y", "z", "mvir", "rvir")}
                 for row in rows]
        bounds = {axis: [summary[1 + 2 * index], summary[2 + 2 * index]]
                  for index, axis in enumerate(("x", "y", "z"))}
        send_json(self, {"total": total, "sampled": len(items) < total,
                         "items": items, "bounds": bounds})

    def export(self, connection, catalog, query, source, where, parameters, ordering):
        cursor = connection.execute(f"SELECT h.* FROM {source} WHERE {where} ORDER BY {ordering}", parameters)
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="halos_catalog_{catalog["id"]}.csv"')
        # Companion endpoint carries units, exact selection and source hashes.
        # It is also linked from the download response for script clients.
        metadata_query = urlencode({key: value for key, value in query.items()
                                    if key not in ("limit", "offset")})
        self.send_header("Link", f'</api/export_metadata?{metadata_query}>; rel="describedby"; type="application/json"')
        self.send_header("X-Halo-Units", "positions=comoving-Mpc/h; radii=comoving-kpc/h; masses=Msun/h; velocities=km/s")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        columns = ("catalog_id", "snapshot", *HALO_FIELDS)
        writer.writerow(columns)
        self.wfile.write(buffer.getvalue().encode("utf-8"))
        for row in cursor:
            buffer.seek(0)
            buffer.truncate(0)
            values = [catalog["id"], catalog["snapshot"], *(row[key] for key in HALO_FIELDS)]
            writer.writerow(values)
            self.wfile.write(buffer.getvalue().encode("utf-8"))


def default_exporter():
    root = Path(__file__).resolve().parents[2]
    choices = [root / "build/release/halo_catalog_export", root / "build/halo_catalog_export"]
    return next((path for path in choices if path.is_file()), choices[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import", help="Import a completed snapshot into SQLite")
    importer.add_argument("snapshot", type=Path)
    importer.add_argument("--database", type=Path, required=True)
    importer.add_argument("--exporter", type=Path, default=default_exporter())
    importer.add_argument("--name")
    importer.add_argument("--hubble", type=float, help="Dimensionless h, if known from the run configuration")
    server_parser = commands.add_parser("serve", help="Open the read-only local browser service")
    server_parser.add_argument("--database", type=Path, required=True)
    server_parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    try:
        if arguments.command == "import":
            print(json.dumps(import_snapshot(arguments.snapshot, arguments.database, arguments.exporter,
                                             arguments.name, arguments.hubble), allow_nan=False))
            return 0
        if arguments.port < 0 or arguments.port > 65535:
            raise ValueError("Port must be between 0 and 65535")
        with closing(open_database(arguments.database)) as connection:
            initialize_database(connection)
        server = ThreadingHTTPServer(("127.0.0.1", arguments.port), ViewerHandler)
        server.daemon_threads = True
        server.database = arguments.database.resolve()
        print(f"Halo Explorer: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    except (OSError, ValueError, sqlite3.Error, json.JSONDecodeError) as error:
        print(f"Halo Explorer: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
