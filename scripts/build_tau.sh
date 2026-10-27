#!/usr/bin/env bash
# Build a private, reproducible TAU runtime for the HPX APEX listener.
# No application instrumentation compiler or second APEX library is needed.
set -euo pipefail

tauVersion=2.35.2
tauRoot=${TAU_BUILD_ROOT:-"$HOME/work/tau2"}
tauPrefix=${TAU_PREFIX:-"$HOME/local/tau"}
tauDownloads=${TAU_DOWNLOAD_DIR:-"$tauRoot/downloads"}
tauSource="$tauRoot/tau-$tauVersion"
tauJobs=${TAU_BUILD_JOBS:-6}
tauMaxJobs=$(( $(nproc) / 2 ))
(( tauMaxJobs >= 1 )) || tauMaxJobs=1
(( tauMaxJobs <= 6 )) || tauMaxJobs=6
if [[ ! $tauJobs =~ ^[1-9][0-9]*$ ]]; then
    printf 'TAU_BUILD_JOBS must be a positive integer.\n' >&2
    exit 2
fi
(( tauJobs <= tauMaxJobs )) || tauJobs=$tauMaxJobs
export TAU_BUILD_JOBS=$tauJobs
# Configure launches nested make processes for its dependencies as well.
export MAKEFLAGS="-j$tauJobs"

for command in curl sha256sum tar python3 make mpicc mpicxx mpifort; do
    command -v "$command" >/dev/null || {
        printf 'Missing required command: %s\n' "$command" >&2
        exit 1
    }
done
mkdir -p "$tauRoot" "$tauDownloads" "$tauPrefix"

download() {
    local name=$1 checksum=$2 url=$3
    local archive="$tauDownloads/$name"
    if [[ ! -f $archive ]]; then
        curl --fail --location --silent --show-error "$url" -o "$archive.part"
        mv "$archive.part" "$archive"
    fi
    printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check --status || {
        printf 'Checksum mismatch: %s\n' "$archive" >&2
        exit 1
    }
}

download "tau-$tauVersion.tar.gz" \
    e3dfe11fc71198c2e43d9ad6b238c00711a30a071f48a7ba52b4a81bea694659 \
    "https://www.cs.uoregon.edu/research/tau/tau_releases/tau-$tauVersion.tar.gz"
download binutils-2.40.tar.gz \
    d7f82c4047decf43a6f769ac32456a92ddb6932409a585c633cdd4e9df23d956 \
    https://www.cs.uoregon.edu/research/paracomp/tau/tauprofile/dist/binutils-2.40.tar.gz
download libunwind-1.8.1.tar.gz \
    ddf0e32dd5fafe5283198d37e4bf9decf7ba1770b6e7e006c33e6df79e6a6157 \
    https://www.cs.uoregon.edu/research/paracomp/tau/tauprofile/dist/libunwind-1.8.1.tar.gz

if [[ ! -d $tauSource ]]; then
    tar -xzf "$tauDownloads/tau-$tauVersion.tar.gz" -C "$tauRoot"
    printf '%s\n' "$tauVersion" > "$tauSource/.cosmictiger_source_version"
elif [[ ! -f $tauSource/.cosmictiger_source_version ]] || \
        [[ $(cat "$tauSource/.cosmictiger_source_version") != "$tauVersion" ]]; then
    printf 'Refusing to modify an unrecognized TAU source directory: %s\n' "$tauSource" >&2
    exit 1
fi

# TAU uses these archives before trying its older HTTP download locations.
mkdir -p "$tauSource/external_dependencies"
cp "$tauDownloads/binutils-2.40.tar.gz" "$tauDownloads/libunwind-1.8.1.tar.gz" \
    "$tauSource/external_dependencies/"

# Upstream hard-codes eight BFD build jobs, independently of MAKEFLAGS.
# Limit that sub-build too, preserving the workstation's shared CPU budget.
python3 - "$tauSource/configure" "$tauSource/src/Profile/TauBfd.cpp" \
    "$tauSource/tools/src/tau_exec" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
source = path.read_text()
updated = source
original = 'make -j8 > tau_build.log 2>&1'
patched = 'make -j"${TAU_BUILD_JOBS:-6}" > tau_build.log 2>&1'
if patched not in source:
    if source.count(original) != 1:
        raise SystemExit('Unexpected TAU BFD build command; refusing a broad patch')
    updated = updated.replace(original, patched)
# These pinned C dependencies predate GCC 15's C23 default. libunwind's
# malloc-interposition test uses an old unspecified-argument prototype.
updated = updated.replace('CFLAGS=-fPIC CXXFLAGS=-fPIC',
                          "CFLAGS='-fPIC -std=gnu17' CXXFLAGS=-fPIC")
if updated != source:
    path.write_text(updated)

# Linux maps describe the executable segment, which can start at a nonzero
# file offset. Include that offset before converting from file to ELF address;
# otherwise PIE executables and shared libraries can appear as UNRESOLVED.
path = pathlib.Path(sys.argv[2])
source = path.read_text()
original = '''#else
    addr0 = probeAddr;
    addr1 = probeAddr - unit->addressMaps[matchingIdx]->start;
    addr2 = addr1 + module->textOffset;
#endif'''
offset_only = original.replace('addr1 + module->textOffset',
                              'addr1 + unit->addressMaps[matchingIdx]->offset + module->textOffset')
patched = '''#else
    addr0 = probeAddr - unit->addressMaps[matchingIdx]->start
      + unit->addressMaps[matchingIdx]->offset + module->textOffset;
    addr1 = probeAddr;
    addr2 = 0;
#endif'''
if patched not in source:
    old = original if source.count(original) == 1 else offset_only
    if source.count(old) != 1:
        raise SystemExit('Unexpected TAU BFD address calculation; refusing a broad patch')
    path.write_text(source.replace(old, patched))

# Opt in to replacing tau_exec's final waiting shell with the application.
# MPI supervisors then retain a direct parent-child relationship. Dry runs,
# PTTS, Python, Julia, and other specialized launch modes remain unchanged.
path = pathlib.Path(sys.argv[3])
source = path.read_text()
# Upstream's extra fi was masked by exit before Bash parsed the final line.
source = source.replace('    $dryrun export LD_PRELOAD=$TAUEX_LD_PRELOAD\n  fi\n  fi\n',
                        '    $dryrun export LD_PRELOAD=$TAUEX_LD_PRELOAD\n  fi\n')
original = '''  else
    $dryrun "$@"
    retval=$?
  fi'''
patched = '''  else
    if [ -z "$dryrun" ] && [ "${COSMICTIGER_TAU_EXEC_REPLACE:-0}" = 1 ]; then
      exec "$@"
    fi
    $dryrun "$@"
    retval=$?
  fi'''
if patched not in source:
    if source.count(original) != 1:
        raise SystemExit('Unexpected tau_exec launch command; refusing a broad patch')
    source = source.replace(original, patched)
if source != path.read_text():
    path.write_text(source)
PY

cd "$tauSource"
printf 'Building TAU %s with %s jobs into %s\n' "$tauVersion" "$tauJobs" "$tauPrefix"
./configure "-prefix=$tauPrefix" -cc=mpicc -c++=mpicxx -fortran=mpifort \
    -mpi -pthread -bfd=download -unwind=download
make -j"$tauJobs" install

tauArch=$(./utils/archfind)
test -x "$tauPrefix/$tauArch/bin/tau_exec"
test -x "$tauPrefix/$tauArch/bin/pprof"
printf '\nTAU runtime: %s/%s/bin/tau_exec\n' "$tauPrefix" "$tauArch"
printf 'Use: APEX_TAU=1 %s/%s/bin/tau_exec -T mpi,pthread -ebs <executable> [arguments]\n' \
    "$tauPrefix" "$tauArch"
