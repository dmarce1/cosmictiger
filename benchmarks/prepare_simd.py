#!/usr/bin/env python3
"""Prepare comparable SIMD source snapshots; never execute either application."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile


DEFAULT_BASELINE = "b50b0404f9dcbc074e8e1b55f7f4e30b8e0c44ea"
NEW_HEADERS = ("cosmictiger/math/simd_backend.hpp", "cosmictiger/math/simd_math.hpp")
HARNESS = "benchmarks/gravity.cpp"
CACHE_PATHS = (
    "HPX_DIR",
    "FFTW3F_INCLUDE_DIR",
    "FFTW3F_LIBRARY",
    "SILO_INCLUDE_DIR",
    "SILO_LIBRARY",
    "CHEALPIX_INCLUDE_DIR",
    "CHEALPIX_LIBRARY",
    "MPI_CXX_COMPILER",
    "MPI_CXX_HEADER_DIR",
    "MPI_mpi_LIBRARY",
    "MPI_mpi_cxx_LIBRARY",
)


def output(command, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cache(path, repository):
    cache = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cache[key.split(":", 1)[0]] = value
    if Path(cache.get("CMAKE_HOME_DIRECTORY", "")).resolve() != repository:
        raise ValueError(f"Dependency cache does not belong to this checkout: {path}")
    selected = {}
    prefixes = cache.get("CMAKE_PREFIX_PATH", "").split(";")
    if not prefixes or not all(prefix and Path(prefix).is_dir() for prefix in prefixes):
        raise ValueError("Dependency cache has missing or invalid CMAKE_PREFIX_PATH entries")
    selected["CMAKE_PREFIX_PATH"] = ";".join(prefixes)
    for key in CACHE_PATHS:
        value = cache.get(key)
        if not value or not Path(value).exists():
            raise ValueError(f"Dependency cache has a missing or invalid {key}")
        selected[key] = value
    mpi_includes = cache.get("MPI_CXX_COMPILER_INCLUDE_DIRS", "").split(";")
    if not mpi_includes or not all(path and Path(path).is_dir() for path in mpi_includes):
        raise ValueError("Dependency cache has missing or invalid MPI include directories")
    selected["MPI_CXX_COMPILER_INCLUDE_DIRS"] = ";".join(mpi_includes)
    mpi_libraries = cache.get("MPI_CXX_LIB_NAMES", "").split(";")
    if set(mpi_libraries) != {"mpi", "mpi_cxx"}:
        raise ValueError("Dependency cache must identify the validated mpi and mpi_cxx libraries")
    selected["MPI_CXX_LIB_NAMES"] = ";".join(mpi_libraries)
    cuda_compiler = cache.get("CMAKE_CUDA_COMPILER")
    if not cuda_compiler or not Path(cuda_compiler).is_file():
        raise ValueError("Dependency cache has a missing or invalid CUDA compiler")
    architectures = cache.get("CMAKE_CUDA_ARCHITECTURES", "")
    if not architectures or not all(arch.isdecimal() for arch in architectures.split(";")):
        raise ValueError("Dependency cache must specify explicit numeric CUDA architectures")
    selected["CMAKE_CUDA_COMPILER"] = cuda_compiler
    selected["CMAKE_CUDA_ARCHITECTURES"] = architectures
    selected["CMAKE_CUDA_FLAGS"] = cache.get("CMAKE_CUDA_FLAGS", "")
    selected["CMAKE_CUDA_FLAGS_RELEASE"] = cache.get("CMAKE_CUDA_FLAGS_RELEASE", "-O3 -DNDEBUG")
    return selected


def archive_source(repository, commit, destination):
    with tempfile.TemporaryFile() as archive:
        subprocess.run(["git", "archive", "--format=tar", commit], cwd=repository,
                       stdout=archive, check=True)
        archive.seek(0)
        with tarfile.open(fileobj=archive) as source:
            source.extractall(destination, filter="data")


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=False)


def snapshot_hashes(directory):
    hashes = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            hashes[relative] = {"symlink": target,
                                "sha256": hashlib.sha256(target.encode()).hexdigest()}
        elif path.is_file():
            hashes[relative] = sha256(path)
    return hashes


def freeze_source(directory):
    for path in directory.rglob("*"):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)
    directory.chmod(directory.stat().st_mode & ~0o222)


def instrument_source(directory, harness_bytes):
    harness_path = directory / HARNESS
    harness_path.parent.mkdir(parents=True, exist_ok=True)
    harness_path.write_bytes(harness_bytes)
    anchor = "void test(std::string test) {"
    # Historical snapshots may keep the dispatcher among application sources.
    candidates = [directory / "tests/test.cpp", *(directory / "src").rglob("test.cpp")]
    dispatchers = [path for path in candidates if path.is_file() and anchor in path.read_text()]
    if len(dispatchers) != 1:
        raise ValueError(f"Expected one test dispatcher in {directory}, found {len(dispatchers)}")
    dispatcher = dispatchers[0]
    original = dispatcher.read_text()
    if original.count(anchor) != 1 or 'test == "simdBenchmark"' in original:
        raise ValueError(f"Unexpected test dispatcher in {dispatcher}")
    replacement = (
        "void simdBenchmark();\n\n" + anchor + "\n"
        '    if (test == "simdBenchmark") {\n'
        "        simdBenchmark();\n"
        "        return;\n"
        "    }"
    )
    dispatcher.write_text(original.replace(anchor, replacement))
    cmake = directory / "CMakeLists.txt"
    with cmake.open("a") as stream:
        stream.write(
            "\n# Local benchmark instrumentation, identical in both snapshots.\n"
            'target_sources(cosmictiger PRIVATE "${PROJECT_SOURCE_DIR}/benchmarks/gravity.cpp")\n'
            "set_property(TARGET cosmictiger PROPERTY CXX_STANDARD 20)\n"
        )


def write_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_logged(command, log):
    print(shlex.join(command), flush=True)
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def compile_provenance(variant):
    source = Path(variant["source_dir"])
    build = Path(variant["build_dir"])
    commands_path = build / "compile_commands.json"
    entries = json.loads(commands_path.read_text())
    commands = {}
    for entry in entries:
        command = entry.get("arguments") or shlex.split(entry["command"])
        if not any("CMakeFiles/cosmictiger.dir/" in token for token in command):
            continue
        filename = Path(entry["file"])
        if not filename.is_absolute():
            filename = Path(entry["directory"]) / filename
        relative = filename.relative_to(source).as_posix()
        commands[relative] = [token.replace(str(source), "<SOURCE>")
                              .replace(str(build), "<BUILD>") for token in command]
    if HARNESS not in commands or "src/gravity/gravity.cpp" not in commands:
        raise ValueError("Compile database is missing benchmark or production gravity commands")
    variant["compile_commands_sha256"] = sha256(commands_path)
    variant["normalized_compile_commands"] = commands
    return commands


def build_variants(manifest, manifest_path):
    for name, variant in manifest["variants"].items():
        try:
            variant["build_status"] = "configuring"
            write_manifest(manifest_path, manifest)
            run_logged(variant["configure_command"], manifest_path.parent / f"{name}_configure.log")
            compile_provenance(variant)
            variant["build_status"] = "configured"
        except Exception:
            variant["build_status"] = "failed"
            write_manifest(manifest_path, manifest)
            raise
        write_manifest(manifest_path, manifest)
    old = manifest["variants"]["old"]["normalized_compile_commands"]
    new = manifest["variants"]["new"]["normalized_compile_commands"]
    manifest["identical_compile_commands"] = old == new
    write_manifest(manifest_path, manifest)
    if old != new:
        differing = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
        raise ValueError(f"Compiler arguments differ between snapshots: {differing}")
    for name, variant in manifest["variants"].items():
        try:
            variant["build_status"] = "building"
            write_manifest(manifest_path, manifest)
            run_logged(variant["build_command"], manifest_path.parent / f"{name}_build.log")
            variant["executable_sha256"] = sha256(Path(variant["executable"]))
            variant["build_status"] = "built"
            if snapshot_hashes(Path(variant["source_dir"])) != variant["source_sha256"]:
                raise ValueError(f"Source snapshot changed during the {name} build")
        except Exception:
            variant["build_status"] = "failed"
            write_manifest(manifest_path, manifest)
            raise
        write_manifest(manifest_path, manifest)


def main():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE,
                        help="Explicit pre-migration Git ref")
    parser.add_argument("--output", type=Path, default=repository / "build/simd_benchmark")
    parser.add_argument("--cache", type=Path, default=repository / "build/sph_validation/CMakeCache.txt",
                        help="Read dependency paths only; never reuse its build configuration")
    parser.add_argument("--compiler", default="/usr/bin/g++-13")
    parser.add_argument("--jobs", type=int, choices=range(1, 13), default=8)
    parser.add_argument("--build", action="store_true",
                        help="Also configure and compile both snapshots; never run them")
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise ValueError(f"Refusing to overwrite an existing preparation: {destination}")
    harness = repository / HARNESS
    required = [harness, *(repository / name for name in NEW_HEADERS)]
    for path in required:
        if not path.is_file():
            raise ValueError(f"Required benchmark source is missing: {path}")
    harness_bytes = harness.read_bytes()
    compiler = Path(args.compiler).resolve()
    compiler_version = output([str(compiler), "-dumpfullversion", "-dumpversion"])
    if compiler_version.split(".")[0] != "13":
        raise ValueError(f"This comparison requires GNU 13, got {compiler_version}")
    compiler_banner = output([str(compiler), "--version"])
    if "Free Software Foundation" not in compiler_banner:
        raise ValueError("This comparison requires the GNU compiler")
    cmake = shutil.which("cmake")
    if not cmake:
        raise ValueError("cmake is unavailable")
    dependency_paths = read_cache(args.cache.resolve(), repository)
    baseline_commit = output(["git", "rev-parse", "--verify", args.baseline + "^{commit}"], repository)
    working_head = output(["git", "rev-parse", "HEAD"], repository)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repository).decode().split("\0")
    tracked = sorted(set(name for name in tracked if name))
    tracked_set = set(tracked)
    included_untracked = [name for name in (*NEW_HEADERS, HARNESS) if name not in tracked_set]
    common_args = [
        "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CXX_COMPILER={compiler}",
        "-DCMAKE_C_COMPILER=/usr/bin/gcc-13", "-DCMAKE_Fortran_COMPILER=/usr/bin/gfortran-13",
        f"-DCMAKE_CUDA_HOST_COMPILER={compiler}",
        "-DCMAKE_CXX_STANDARD=20", "-DCMAKE_CXX_FLAGS=", "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG",
        "-DCMAKE_C_FLAGS=", "-DCMAKE_Fortran_FLAGS=", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DCOSMICTIGER_WITH_CUDA=ON", "-DCOSMICTIGER_WITH_ORDER=7",
        "-DCOSMICTIGER_WITH_LONG_LONG_PART_INT=OFF", "-DBUILD_TESTING=OFF",
        *[f"-D{key}={value}" for key, value in dependency_paths.items()],
    ]
    for executable in ("/usr/bin/gcc-13", "/usr/bin/gfortran-13"):
        if not Path(executable).is_file():
            raise ValueError(f"Required companion compiler is missing: {executable}")
    if not shutil.which("ninja"):
        raise ValueError("Ninja is unavailable")
    destination.mkdir(parents=True)
    diff_path = destination / "working_tree.patch"
    diff_path.write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repository))
    manifest = {
        "schema_version": 1, "baseline_ref": args.baseline, "baseline_commit": baseline_commit,
        "working_head": working_head, "repository": str(repository),
        "included_untracked": included_untracked,
        "tracked_diff": {"path": str(diff_path), "sha256": sha256(diff_path)},
        "harness": {"path": HARNESS, "sha256": hashlib.sha256(harness_bytes).hexdigest()},
        "preparation_script_sha256": sha256(Path(__file__)),
        "dependency_cache": {"path": str(args.cache.resolve()), "sha256": sha256(args.cache)},
        "toolchain": {"compiler": str(compiler), "version": compiler_version,
                      "compiler_banner": compiler_banner, "cmake": cmake,
                      "cmake_version": output([cmake, "--version"]),
                      "common_cmake_args": common_args,
                      "expected_cxx_flags": ["-O3", "-DNDEBUG", "-ffast-math", "-march=native", "-std=gnu++20"]},
        "identical_compile_commands": None, "variants": {},
        "benchmarks_executed": False,
        "cpu_kernels_only": True, "cuda_startup_required": True,
    }
    manifest_path = destination / "manifest.json"
    write_manifest(manifest_path, manifest)
    for name in ("old", "new"):
        source = destination / name / "source"
        build = destination / name / "build"
        source.mkdir(parents=True)
        if name == "old":
            archive_source(repository, baseline_commit, source)
        else:
            for relative in tracked:
                path = repository / relative
                if path.is_file() or path.is_symlink():
                    copy_file(path, source / relative)
            for relative in NEW_HEADERS:
                copy_file(repository / relative, source / relative)
        instrument_source(source, harness_bytes)
        manifest["variants"][name] = {
            "source_dir": str(source), "build_dir": str(build),
            "executable": str(build / "cosmictiger"), "executable_sha256": None,
            "source_sha256": snapshot_hashes(source), "build_status": "prepared",
            "configure_command": [cmake, "-S", str(source), "-B", str(build), *common_args],
            "build_command": [cmake, "--build", str(build), "--target", "cosmictiger", "--parallel", str(args.jobs)],
        }
        freeze_source(source)
        write_manifest(manifest_path, manifest)
    print(f"Prepared immutable source snapshots and commands: {manifest_path}")
    if args.build:
        build_variants(manifest, manifest_path)
        print("Both applications built; neither was executed.")
    else:
        print("No configuration, compilation, application, CTest, or benchmark was executed.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"SIMD preparation failed: {error}", file=sys.stderr)
        sys.exit(1)
