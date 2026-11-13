#!/usr/bin/env python3
"""Print a Silo particle mesh with a redshift header and %16.10e columns.

Usage: python3 scripts/silo_to_text.py view.0.4.silo > particles.txt
Requires a C++17 compiler and the Silo development library (pkg-config: silo).
Coordinates retain the selected mesh's units; no rescaling is performed.
Redshift comes from scale_factor metadata or the matching cycle in time.txt.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


# Use the installed Silo headers rather than duplicating its C structure ABI.
READER = r'''
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <silo.h>

namespace detail {
double readRedshift(std::filesystem::path const& filename, std::string const& name) {
    std::unique_ptr<DBfile, decltype(&DBClose)> file(
        DBOpen(filename.c_str(), DB_UNKNOWN, DB_READ), DBClose);
    if (!file) {
        throw std::runtime_error("Cannot open " + filename.string());
    }
    auto directory = std::filesystem::path(name).parent_path();
    while (true) {
        auto const key = (directory / "scale_factor").string();
        if (DBInqVarExists(file.get(), key.c_str())) {
            double scaleFactor = 0.0;
            auto const type = DBGetVarType(file.get(), key.c_str());
            if (DBGetVarLength(file.get(), key.c_str()) != 1) {
                throw std::runtime_error("Expected scalar scale_factor metadata");
            }
            int status = -1;
            if (type == DB_DOUBLE) {
                status = DBReadVar(file.get(), key.c_str(), &scaleFactor);
            } else if (type == DB_FLOAT) {
                float value = 0.0f;
                status = DBReadVar(file.get(), key.c_str(), &value);
                scaleFactor = value;
            }
            if (status < 0 || !std::isfinite(scaleFactor) || scaleFactor <= 0.0 ||
                !std::isfinite(1.0 / scaleFactor)) {
                throw std::runtime_error("Invalid scale_factor metadata");
            }
            return 1.0 / scaleFactor - 1.0;
        }
        if (directory.empty() || directory == directory.root_path()) {
            break;
        }
        directory = directory.parent_path();
    }
    // Code-unit snapshots omit scale_factor. Their root cycle matches the
    // output step recorded in the run's time.txt, including locality leaves.
    int cycle = -1;
    if (!DBInqVarExists(file.get(), "/cycle") ||
        DBGetVarType(file.get(), "/cycle") != DB_INT ||
        DBGetVarLength(file.get(), "/cycle") != 1 ||
        DBReadVar(file.get(), "/cycle", &cycle) < 0) {
        throw std::runtime_error("No scale_factor or cycle metadata for redshift lookup");
    }
    auto runDirectory = filename.parent_path();
    for (auto parent = runDirectory; !parent.empty(); parent = parent.parent_path()) {
        if (parent.extension() == ".data") {
            runDirectory = parent.parent_path();
            break;
        }
        if (parent == parent.root_path()) {
            break;
        }
    }
    auto const timePath = runDirectory / "time.txt";
    std::ifstream times(timePath);
    std::string line;
    std::getline(times, line);
    if (line.find("redshift") != std::string::npos) {
        while (std::getline(times, line)) {
            std::istringstream row(line);
            int step;
            double realYears, conformalYears, scaleFactor, redshift;
            if (row >> step >> realYears >> conformalYears >> scaleFactor >> redshift &&
                step == cycle && std::isfinite(redshift)) {
                return redshift;
            }
        }
    }
    throw std::runtime_error("No redshift for cycle " + std::to_string(cycle) +
        " in " + timePath.string());
}

void writeMesh(std::filesystem::path const& filename, std::string const& name, int depth = 0) {
    if (depth > 16) {
        throw std::runtime_error("Multimesh reference nesting exceeds 16 levels");
    }
    std::unique_ptr<DBfile, decltype(&DBClose)> file(
        DBOpen(filename.c_str(), DB_UNKNOWN, DB_READ), DBClose);
    if (!file) {
        throw std::runtime_error("Cannot open " + filename.string());
    }
    auto const type = DBInqVarType(file.get(), name.c_str());
    if (type == DB_MULTIMESH) {
        std::unique_ptr<DBmultimesh, decltype(&DBFreeMultimesh)> mesh(
            DBGetMultimesh(file.get(), name.c_str()), DBFreeMultimesh);
        if (!mesh || (mesh->nblocks && !mesh->meshnames)) {
            throw std::runtime_error("Cannot read explicit multimesh references: " + name);
        }
        for (int i = 0; i < mesh->nblocks; ++i) {
            if (!mesh->meshnames[i]) {
                throw std::runtime_error("Missing multimesh block name");
            }
            std::string const reference = mesh->meshnames[i];
            if (reference == "EMPTY") {
                continue;
            }
            auto const separator = reference.find(':');
            auto nextFile = filename;
            auto nextMesh = reference;
            if (separator != std::string::npos) {
                nextFile = filename.parent_path() / reference.substr(0, separator);
                nextMesh = reference.substr(separator + 1);
            }
            if (nextMesh.empty()) {
                throw std::runtime_error("Empty mesh reference");
            }
            if (nextMesh.front() != '/') {
                auto const base = separator == std::string::npos ?
                    std::filesystem::path(name).parent_path() : std::filesystem::path("/");
                nextMesh = (base / nextMesh).string();
            }
            writeMesh(nextFile, nextMesh, depth + 1);
        }
        return;
    }
    if (type != DB_POINTMESH) {
        throw std::runtime_error("Not a point mesh or multimesh: " + name);
    }
    std::unique_ptr<DBpointmesh, decltype(&DBFreePointmesh)> mesh(
        DBGetPointmesh(file.get(), name.c_str()), DBFreePointmesh);
    if (!mesh || mesh->ndims != 3 ||
        (mesh->datatype != DB_FLOAT && mesh->datatype != DB_DOUBLE)) {
        throw std::runtime_error("Expected a three-dimensional float or double point mesh");
    }
    for (int i = 0; i < mesh->nels; ++i) {
        double xyz[3];
        for (int dim = 0; dim < 3; ++dim) {
            if (!mesh->coords[dim]) {
                throw std::runtime_error("Missing coordinate array");
            }
            xyz[dim] = mesh->datatype == DB_DOUBLE ?
                static_cast<double const*>(mesh->coords[dim])[i] :
                static_cast<float const*>(mesh->coords[dim])[i];
            if (!std::isfinite(xyz[dim])) {
                throw std::runtime_error("Non-finite particle coordinate");
            }
        }
        if (std::printf("%16.10e %16.10e %16.10e\n", xyz[0], xyz[1], xyz[2]) < 0) {
            throw std::runtime_error("Cannot write coordinates to stdout");
        }
    }
}
} // namespace detail

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("Expected filename and mesh path");
        }
        double const redshift = detail::readRedshift(argv[1], argv[2]);
        std::printf("# x, y, z; redshift = %.10e\n", redshift);
        detail::writeMesh(argv[1], argv[2]);
        if (std::fflush(stdout) != 0) {
            throw std::runtime_error("Cannot flush stdout");
        }
    } catch (std::exception const& error) {
        std::fprintf(stderr, "silo_to_text: %s\n", error.what());
        return 1;
    }
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="Silo master or particle leaf file")
    parser.add_argument(
        "--mesh", default="/darkMatter/mesh",
        help="Silo mesh path (default: /darkMatter/mesh)",
    )
    args = parser.parse_args()
    if not args.snapshot.is_file():
        parser.error(f"snapshot does not exist: {args.snapshot}")
    try:
        try:
            result = subprocess.run(
                ["pkg-config", "--cflags", "--libs", "silo"],
                check=True, capture_output=True, text=True,
            )
            flags = shlex.split(result.stdout)
        except (FileNotFoundError, subprocess.CalledProcessError):
            flags = ["-lsiloh5"]
        with tempfile.TemporaryDirectory(prefix="silo_to_text_") as temporary:
            source = Path(temporary) / "reader.cpp"
            binary = Path(temporary) / "reader"
            source.write_text(READER)
            subprocess.run(
                shlex.split(os.environ.get("CXX", "c++"))
                + ["-std=c++17", str(source), "-o", str(binary)] + flags,
                check=True, stdout=sys.stderr,
            )
            result = subprocess.run([str(binary), str(args.snapshot.resolve()), args.mesh])
            return result.returncode if result.returncode >= 0 else 128 - result.returncode
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"silo_to_text: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
