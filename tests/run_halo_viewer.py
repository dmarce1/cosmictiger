#!/usr/bin/env python3
"""Exercise the native snapshot exporter, SQLite importer, and browser API.

Only Python's standard library is required. The native fixture deliberately
contains two rank files, nested substructure, IEEE half subnormals, and IDs
larger than JavaScript's exact integer range.
"""

import argparse
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import socket
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request


kHalfFields = (
    "zValue", "rmax", "rvir", "rvmax", "rs", "rkly", "sigX", "deltaX", "toW", "sigV",
    "sigVb", "vcircMax", "deltaV", "boa", "coa", "ax", "ay", "az", "boa500", "coa500",
    "ax500", "ay500", "az500", "lambda", "lambdaB", "vxc", "vyc", "vzc", "vxb", "vyb", "vzb",
)


def runCommand(arguments, expectedSuccess=True):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=60)
    if expectedSuccess and result.returncode != 0:
        raise AssertionError(
            f"Command failed ({result.returncode}): {arguments!r}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


class HaloViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="halo_viewer_test_")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.snapshot = cls.directory / "snapshot"
        cls.database = cls.directory / "catalog.sqlite"
        runCommand([str(arguments.fixture), str(cls.snapshot)])
        exported = runCommand([str(arguments.exporter), str(cls.snapshot)])
        cls.exported = [json.loads(line) for line in exported.stdout.splitlines() if line.strip()]
        cls.metadata = cls.exported[0]
        cls.expected = cls.exported[1:]
        if len(cls.expected) != 64:
            raise AssertionError(f"Expected 64 fixture halos, got {len(cls.expected)}")
        cls.importSnapshot(cls.snapshot)

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        cls.baseUrl = f"http://127.0.0.1:{port}"
        cls.serverLog = tempfile.TemporaryFile(mode="w+")
        cls.addClassCleanup(cls.serverLog.close)
        cls.server = subprocess.Popen(
            [sys.executable, str(arguments.viewer), "serve", "--database", str(cls.database),
             "--port", str(port)],
            stdout=cls.serverLog,
            stderr=subprocess.STDOUT,
        )
        cls.addClassCleanup(cls.stopServer)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if cls.server.poll() is not None:
                cls.serverLog.seek(0)
                raise AssertionError(f"Viewer exited during startup:\n{cls.serverLog.read()}")
            try:
                cls.catalogs = cls.requestJson("/api/catalogs")["catalogs"]
                break
            except (OSError, urllib.error.URLError):
                time.sleep(0.05)
        else:
            cls.serverLog.seek(0)
            raise AssertionError(f"Viewer did not start:\n{cls.serverLog.read()}")
        if len(cls.catalogs) != 1:
            raise AssertionError(f"Expected one imported catalog: {cls.catalogs!r}")
        cls.catalogId = cls.catalogs[0]["id"]
        cls.byId = {str(halo["halo_id"]): halo for halo in cls.expected}

    @classmethod
    def stopServer(cls):
        cls.server.terminate()
        try:
            cls.server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.server.kill()
            cls.server.wait(timeout=10)

    @classmethod
    def importSnapshot(cls, snapshot, expectedSuccess=True):
        return runCommand(
            [sys.executable, str(arguments.viewer), "import", str(snapshot),
             "--database", str(cls.database), "--exporter", str(arguments.exporter),
             "--name", "Deterministic fixture", "--hubble", "0.7"],
            expectedSuccess=expectedSuccess,
        )

    @classmethod
    def request(cls, path, query=None):
        if query:
            path += "?" + urllib.parse.urlencode(query)
        return urllib.request.urlopen(cls.baseUrl + path, timeout=10)

    @classmethod
    def requestJson(cls, path, query=None):
        with cls.request(path, query) as response:
            return json.load(response)

    def halos(self, **filters):
        return self.requestJson("/api/halos", {"catalog": self.catalogId, **filters})

    def points(self, **filters):
        return self.requestJson("/api/points", {"catalog": self.catalogId, **filters})

    def assertHttpError(self, status, path, query=None):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request(path, query)
        self.assertEqual(error.exception.code, status)
        error.exception.close()

    def allRows(self, **filters):
        rows = []
        offset = 0
        while True:
            page = self.halos(limit=13, offset=offset, **filters)
            rows.extend(page["items"])
            offset += len(page["items"])
            if offset >= page["total"]:
                break
            self.assertTrue(page["items"], "Pagination stopped before total")
        self.assertEqual(len(rows), page["total"])
        return rows

    def assertCatalogUnchanged(self, before):
        self.assertEqual(self.requestJson("/api/catalogs"), before)
        self.assertEqual(self.halos(limit=1)["total"], 64)

    def test_01_native_export_decodes_ieee_half_and_all_ranks(self):
        self.assertEqual(len(self.byId), 64)
        self.assertTrue(all(isinstance(row["halo_id"], str) for row in self.expected))
        self.assertTrue(all(int(row["halo_id"]) > 2**53 for row in self.expected))
        self.assertEqual(self.expected[0]["rs"], math.ldexp(1.0, -24))
        self.assertEqual(len({row["rank"] for row in self.expected}), 2)
        self.assertEqual(sum(row["host_id"] is None for row in self.expected), 8)
        self.assertTrue(any(
            row["host_id"] is not None and self.byId[row["host_id"]]["host_id"] is not None
            for row in self.expected
        ))
        for rank in (0, 1):
            data = (self.snapshot / f"rockstar.{rank}.dat").read_bytes()
            sidecar = dict(line.split() for line in
                           (self.snapshot / f"rockstar.{rank}.dat.format").read_text().splitlines())
            self.assertEqual(int(sidecar["record_size"]), 128)
            endian = "<" if sidecar["byte_order"] == "little" else ">"
            for row in (halo for halo in self.expected if halo["rank"] == rank):
                reference = struct.unpack_from(endian + "31e", data, row["ordinal"] * 128 + 66)
                for field, value in zip(kHalfFields, reference):
                    self.assertEqual(row[field], value, field)

    def test_02_import_metadata_and_exact_ids(self):
        catalog = self.requestJson("/api/catalogs")["catalogs"][0]
        self.assertEqual(catalog["name"], "Deterministic fixture")
        self.assertEqual(catalog["hubble"], 0.7)
        self.assertEqual(catalog["halo_count"], 64)
        self.assertEqual(catalog["snapshot"], 7)
        self.assertEqual(catalog["scale_factor"], 0.5)
        self.assertIn("units", catalog)
        page = self.halos(limit=50)
        self.assertEqual(page["total"], 64)
        self.assertEqual(len(page["items"]), 50)
        self.assertEqual(page["limit"], 50)
        self.assertEqual(page["offset"], 0)
        rows = self.allRows()
        self.assertEqual({row["halo_id"] for row in rows}, set(self.byId))
        for row in rows:
            expected = self.byId[row["halo_id"]]
            self.assertEqual(row["halo_id"], expected["halo_id"])
            for field in ("x", "y", "z", "mvir", "e", "jx", "jy", "jz", "m200", "m500", "m2500", *kHalfFields):
                self.assertEqual(row[field], expected[field], field)

    def test_03_import_is_idempotent(self):
        before = self.requestJson("/api/catalogs")
        self.importSnapshot(self.snapshot)
        self.assertCatalogUnchanged(before)

    def test_04_filters_and_stable_sorting(self):
        distinct = self.allRows(kind="distinct")
        subhalos = self.allRows(kind="subhalo")
        self.assertEqual(len(distinct), 8)
        self.assertEqual(len(subhalos), 56)
        self.assertTrue(all(row["host_id"] is None for row in distinct))
        self.assertTrue(all(row["host_id"] is not None for row in subhalos))
        for field in ("mvir", "rvir", "vcircMax", "lambda"):
            values = sorted({row[field] for row in self.expected})
            low = values[len(values) // 3]
            high = values[2 * len(values) // 3]
            rows = self.allRows(**{f"{field}_min": low, f"{field}_max": high})
            self.assertEqual(
                {row["halo_id"] for row in rows},
                {row["halo_id"] for row in self.expected if low <= row[field] <= high},
                field,
            )
        ordered = self.allRows(sort="mvir", order="desc")
        self.assertEqual([row["mvir"] for row in ordered],
                         sorted([row["mvir"] for row in self.expected], reverse=True))
        self.assertEqual([row["halo_id"] for row in ordered],
                         [row["halo_id"] for row in sorted(
                             self.expected, key=lambda row: (-row["mvir"], row["halo_id"]))])
        self.assertEqual([row["halo_id"] for row in ordered],
                         [row["halo_id"] for row in self.allRows(sort="mvir", order="desc")])
        exact = self.expected[-1]
        self.assertEqual(self.halos(halo_id=exact["halo_id"])["items"][0]["halo_id"], exact["halo_id"])

    def test_04b_mixed_width_ids_sort_numerically(self):
        specification = importlib.util.spec_from_file_location("halo_viewer_test_backend", arguments.viewer)
        backend = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(backend)
        identifiers = ["2", "10", "1", "100", "9007199254740993", "9007199254740992"]
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE TABLE halos (catalog_id INTEGER, halo_id TEXT, mvir REAL)")
            connection.executemany("INSERT INTO halos VALUES (1,?,1.0)",
                                   [(identifier,) for identifier in identifiers])
            # Exercise the actual query builder, without adding catalogs or
            # changing the deterministic native fixture's persistent IDs.
            for order in ("asc", "desc"):
                source, where, parameters, ordering = backend.selection_query(
                    {"sort": "halo_id", "order": order}, {"id": 1})
                rows = connection.execute(
                    f"SELECT h.halo_id FROM {source} WHERE {where} ORDER BY {ordering}",
                    parameters).fetchall()
                self.assertEqual([row[0] for row in rows],
                                 sorted(identifiers, key=int, reverse=order == "desc"))

    def test_05_spatial_bounds_are_exact_and_points_match_selection(self):
        target = self.expected[17]
        for axis in ("x", "y", "z"):
            filters = {f"{axis}_min": target[axis], f"{axis}_max": target[axis]}
            expected = {row["halo_id"] for row in self.expected if row[axis] == target[axis]}
            rows = self.allRows(**filters)
            self.assertEqual({row["halo_id"] for row in rows}, expected)
            plotted = self.points(limit=100, **filters)
            self.assertEqual(plotted["total"], len(expected))
            self.assertFalse(plotted["sampled"])
            self.assertEqual({row["halo_id"] for row in plotted["items"]}, expected)
        sampled = self.points(limit=10)
        self.assertEqual(sampled["total"], 64)
        self.assertTrue(sampled["sampled"])
        self.assertEqual(len(sampled["items"]), 10)
        self.assertIn("bounds", sampled)
        self.assertTrue(all(row["halo_id"] in self.byId for row in sampled["items"]))
        periodic = self.points(limit=8)
        self.assertEqual(periodic["total"], 64)
        self.assertTrue(periodic["sampled"])
        self.assertEqual(len(periodic["items"]), 8)
        self.assertEqual([row["halo_id"] for row in periodic["items"]],
                         [row["halo_id"] for row in self.points(limit=8)["items"]])
        # Every eighth source record is a distinct host in this fixture.
        # Sampling must not alias with that periodic catalog structure.
        self.assertTrue(any(row["host_id"] is None for row in periodic["items"]))
        self.assertTrue(any(row["host_id"] is not None for row in periodic["items"]))
        boundary = target["x"]
        self.assertEqual(
            {row["halo_id"] for row in self.allRows(x_min=math.nextafter(boundary, math.inf))},
            {row["halo_id"] for row in self.expected if row["x"] > boundary},
            "R*Tree candidates must be rechecked using stored double coordinates",
        )

    def test_06_host_subhalo_and_ancestor_links(self):
        for expected in self.expected:
            path = "/api/halos/" + urllib.parse.quote(expected["halo_id"], safe="")
            detail = self.requestJson(path, {"catalog": self.catalogId})
            self.assertEqual(detail["halo"]["halo_id"], expected["halo_id"])
            parent = expected["host_id"]
            self.assertEqual(None if detail["host"] is None else detail["host"]["halo_id"], parent)
            expectedAncestors = []
            while parent is not None:
                expectedAncestors.append(parent)
                parent = self.byId[parent]["host_id"]
            self.assertEqual([row["halo_id"] for row in detail["ancestors"]], expectedAncestors)
            children = {row["halo_id"] for row in self.expected if row["host_id"] == expected["halo_id"]}
            self.assertEqual({row["halo_id"] for row in detail["subhalos"]}, children)
            self.assertEqual(detail["subhalo_count"], len(children))

    def test_07_export_is_complete_and_matches_filters(self):
        selection = {"kind": "subhalo", "mvir_min": min(row["mvir"] for row in self.expected)}
        expectedIds = {row["halo_id"] for row in self.allRows(**selection)}
        with self.request("/api/export", {"catalog": self.catalogId, "limit": 2,
                                        "offset": 1, **selection}) as response:
            self.assertIn("text/csv", response.headers["Content-Type"])
            self.assertIn("attachment", response.headers["Content-Disposition"])
            exported = list(csv.DictReader(io.StringIO(response.read().decode("utf-8"))))
        self.assertEqual(len(exported), 56)
        self.assertEqual({row["halo_id"] for row in exported}, expectedIds)
        points = self.points(limit=100, **selection)
        self.assertEqual(points["total"], len(exported))
        self.assertEqual({row["halo_id"] for row in points["items"]}, expectedIds)

    def test_07b_empty_selection_is_exportable(self):
        selection = {"catalog": self.catalogId, "mvir_min": 1.0e30}
        self.assertEqual(self.halos(mvir_min=1.0e30)["total"], 0)
        points = self.points(mvir_min=1.0e30)
        self.assertEqual(points["total"], 0)
        self.assertFalse(points["sampled"])
        self.assertEqual(points["items"], [])
        with self.request("/api/export", selection) as response:
            reader = csv.DictReader(io.StringIO(response.read().decode("utf-8")))
            self.assertIn("halo_id", reader.fieldnames)
            self.assertEqual(list(reader), [])

    def test_08_invalid_queries_are_rejected(self):
        invalid = (
            {"sort": "mvir; DROP TABLE halos; --"}, {"order": "sideways"},
            {"limit": -1}, {"offset": -1}, {"kind": "satellites"},
            {"mvir_min": "NaN"}, {"x_max": "inf"},
            {"mvir_min": 3, "mvir_max": 2}, {"x_min": 1, "x_max": 0},
            {"rvir_min": "not_a_number"},
        )
        for query in invalid:
            with self.subTest(query=query):
                self.assertHttpError(400, "/api/halos", {"catalog": self.catalogId, **query})
        self.assertHttpError(404, "/api/halos", {"catalog": 999999999})
        self.assertHttpError(404, "/api/halos/nonexistent", {"catalog": self.catalogId})
        self.assertEqual(self.halos(limit=1)["total"], 64)

    def test_09_invalid_snapshots_do_not_modify_database(self):
        def truncate(directory):
            path = directory / "rockstar.0.dat"
            path.write_bytes(path.read_bytes()[:-1])

        def removeSidecar(directory):
            (directory / "rockstar.0.dat.format").unlink()

        def manifestCount(directory):
            path = directory / "rockstar_hosts.complete"
            fields = path.read_text().split()
            fields[-1] = str(int(fields[-1]) + 1)
            path.write_text(" ".join(fields) + "\n")

        def duplicateId(directory):
            path = directory / "rockstar.0.dat"
            records = bytearray(path.read_bytes())
            sidecar = dict(line.split() for line in (directory / "rockstar.0.dat.format").read_text().splitlines())
            recordSize = int(sidecar["record_size"])
            records[recordSize:recordSize + 8] = records[:8]
            path.write_bytes(records)

        def inconsistentHost(directory):
            path = directory / "rockstar_hosts.0.dat"
            lines = path.read_text().splitlines()
            fields = lines[1].split()
            fields[1] = "42"
            lines[1] = " ".join(fields)
            path.write_text("\n".join(lines) + "\n")

        for label, mutation in (
            ("truncated", truncate), ("missing_sidecar", removeSidecar),
            ("count_mismatch", manifestCount), ("duplicate_id", duplicateId),
            ("inconsistent_host", inconsistentHost),
        ):
            with self.subTest(mutation=label):
                destination = self.directory / label
                shutil.copytree(self.snapshot, destination)
                mutation(destination)
                exported = runCommand([str(arguments.exporter), str(destination)],
                                      expectedSuccess=False)
                self.assertNotEqual(exported.returncode, 0, exported.stdout)
                self.assertEqual(exported.stdout, "", "Exporter published a partial catalog")
                before = self.requestJson("/api/catalogs")
                result = self.importSnapshot(destination, expectedSuccess=False)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertCatalogUnchanged(before)

    def test_10_browser_assets_are_local_and_available(self):
        with self.request("/") as response:
            self.assertIn("text/html", response.headers["Content-Type"])
            html = response.read().decode("utf-8")
        self.assertTrue(html.strip())
        self.assertNotIn("cdn.", html)
        self.assertNotIn("https://", html)

    def test_11_failed_reimport_preserves_existing_catalog(self):
        path = self.snapshot / "rockstar.0.dat"
        original = path.read_bytes()
        before = self.requestJson("/api/catalogs")
        try:
            path.write_bytes(original[:-1])
            result = self.importSnapshot(self.snapshot, expectedSuccess=False)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertCatalogUnchanged(before)
            self.assertEqual({row["halo_id"] for row in self.allRows()}, set(self.byId))
        finally:
            path.write_bytes(original)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--viewer", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    arguments.exporter = arguments.exporter.resolve()
    arguments.fixture = arguments.fixture.resolve()
    arguments.viewer = arguments.viewer.resolve()
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
