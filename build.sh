#!/usr/bin/env bash
# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

set -u

showUsage() {
    printf 'Usage: %s\n' "${0##*/}"
    printf 'Run the interactive CosmicTiger configure and build menu in a terminal.\n'
}

case "${1:-}" in
    --help|-h)
        showUsage
        exit 0
        ;;
    '')
        if (( $# != 0 )); then
            showUsage >&2
            exit 2
        fi
        ;;
    *)
        showUsage >&2
        exit 2
        ;;
esac

if [[ ! -t 0 || ! -t 1 ]]; then
    printf 'An interactive terminal is required. Run ./build.sh in a terminal.\n' >&2
    exit 2
fi

rootDir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || exit 1
cd -- "$rootDir" || exit 1

defaultDependencyPrefixes=
defaultCudaArchitectures=
readBuildDefaults() {
    local cacheFile=$1
    local cacheKey cacheValue prefix
    local cachedSourceDir= cachedPrefixPath= cachedArchitectures=
    local -a cachedPrefixes
    [[ -f $cacheFile ]] || return 0
    # Read cache data without evaluating it as shell code.
    while IFS='=' read -r cacheKey cacheValue; do
        case "$cacheKey" in
            CMAKE_HOME_DIRECTORY:*) cachedSourceDir=$cacheValue ;;
            CMAKE_PREFIX_PATH:*) cachedPrefixPath=$cacheValue ;;
            CMAKE_CUDA_ARCHITECTURES:*) cachedArchitectures=$cacheValue ;;
        esac
    done < "$cacheFile"
    [[ $cachedSourceDir == "$rootDir" ]] || return 0
    if [[ -z $defaultDependencyPrefixes && -n $cachedPrefixPath ]]; then
        IFS=';' read -r -a cachedPrefixes <<< "$cachedPrefixPath"
        for prefix in "${cachedPrefixes[@]}"; do
            [[ -d $prefix ]] || continue
            # The HPX prefix is selected separately for the chosen build mode.
            if [[ -n ${HOME:-} && $prefix == "$HOME/local/hpx/"* ]]; then
                continue
            fi
            defaultDependencyPrefixes+="${defaultDependencyPrefixes:+;}$prefix"
        done
    fi
    if [[ -z $defaultCudaArchitectures && -n $cachedArchitectures &&
        $cachedArchitectures != native ]]; then
        defaultCudaArchitectures=$cachedArchitectures
    fi
}

# Prefer the default build, then reuse another build belonging to this checkout.
for cacheFile in "$rootDir/build/release/CMakeCache.txt" "$rootDir"/build/*/CMakeCache.txt; do
    readBuildDefaults "$cacheFile"
    if [[ -n $defaultDependencyPrefixes && -n $defaultCudaArchitectures ]]; then
        break
    fi
done
defaultCudaArchitectures=${CUDAARCHS:-${defaultCudaArchitectures:-native}}

coreCount=
if command -v nproc >/dev/null 2>&1; then
    # OpenMP thread settings can make nproc report 1 even when more CPUs are available.
    coreCount=$(unset OMP_NUM_THREADS OMP_THREAD_LIMIT; nproc)
fi
if [[ ! $coreCount =~ ^[1-9][0-9]*$ ]] && command -v getconf >/dev/null 2>&1; then
    coreCount=$(getconf _NPROCESSORS_ONLN)
fi
if [[ ! $coreCount =~ ^[1-9][0-9]*$ ]]; then
    coreCount=1
fi

# Build and test stages run sequentially and share half of the available CPUs.
maxJobs=$((coreCount / 2))
if (( maxJobs < 1 )); then
    maxJobs=1
fi
# MPI integration tests use two localities with two HPX threads each.
maxTestJobs=$((maxJobs / 4))
if (( maxTestJobs < 1 )); then
    maxTestJobs=1
fi

buildMode=release
enableCuda=ON
testSuite=OFF
buildJobs=4
testJobs=2
if (( buildJobs > maxJobs )); then
    buildJobs=$maxJobs
fi
if (( testJobs > maxTestJobs )); then
    testJobs=$maxTestJobs
fi
hpxPrefix=
# CMake uses colon-separated environment paths and semicolon-separated cache paths.
dependencyPrefixes=${CMAKE_PREFIX_PATH-$defaultDependencyPrefixes}
dependencyPrefixes=${dependencyPrefixes//:/;}
cudaArchitectures=
logFile=
promptValue=

trap 'printf "\nInterrupted. Log: %s\n" "${logFile:-none}" >&2; exit 130' INT
trap 'printf "\nTerminated. Log: %s\n" "${logFile:-none}" >&2; exit 143' TERM

readInput() {
    printf '%s' "$1"
    if ! IFS= read -r promptValue; then
        printf '\nInput closed; no build started.\n'
        exit 0
    fi
}

chooseMode() {
    while true; do
        printf '\nBuild mode: 1) Release  2) RelWithDebInfo  3) Debug  4) Profile\n'
        readInput "Choice [${buildMode}]: "
        case "${promptValue,,}" in
            '') return ;;
            1|release) buildMode=release; return ;;
            2|relwithdebinfo) buildMode=relwithdebinfo; return ;;
            3|debug) buildMode=debug; return ;;
            4|profile) buildMode=profile; return ;;
        esac
        printf 'Choose 1 through 4, or press Enter to keep the current mode.\n'
    done
}

chooseToggle() {
    local label=$1
    local currentValue=$2
    while true; do
        readInput "$label [${currentValue}] (y/n, Enter keeps): "
        case "${promptValue,,}" in
            '') promptValue=$currentValue; return ;;
            y|yes|on) promptValue=ON; return ;;
            n|no|off) promptValue=OFF; return ;;
        esac
        printf 'Enter y or n.\n'
    done
}

chooseNumber() {
    local label=$1
    local currentValue=$2
    local minimum=$3
    local maximum=$4
    local numericValue
    while true; do
        readInput "$label [${currentValue}] (${minimum}-${maximum}, Enter keeps): "
        if [[ -z $promptValue ]]; then
            promptValue=$currentValue
            return
        fi
        if [[ $promptValue =~ ^[0-9]+$ && ${#promptValue} -le 6 ]]; then
            numericValue=$((10#$promptValue))
            if (( numericValue >= minimum && numericValue <= maximum )); then
                promptValue=$numericValue
                return
            fi
        fi
        printf 'Enter a whole number from %s to %s.\n' "$minimum" "$maximum"
    done
}

chooseText() {
    local label=$1
    local currentValue=$2
    readInput "$label [${currentValue:-automatic default}] (Enter keeps, - resets): "
    if [[ $promptValue == '-' ]]; then
        promptValue=
    elif [[ -z $promptValue ]]; then
        promptValue=$currentValue
    fi
}

showMainMenu() {
    printf '\nCosmicTiger builder\n'
    printf '  1) Build mode: %s\n' "$buildMode"
    printf '  2) CUDA: %s\n' "$enableCuda"
    printf '  3) Build and run test suite: %s\n' "$testSuite"
    printf '  4) Parallel build jobs: %s\n' "$buildJobs"
    printf '  5) Parallel test jobs: %s\n' "$testJobs"
    printf '  6) HPX installation prefix: %s\n' "${hpxPrefix:-automatic by build mode}"
    printf '  7) CUDA architectures: %s\n' "${cudaArchitectures:-$defaultCudaArchitectures}"
    printf '  8) Additional dependency prefixes: %s\n' "${dependencyPrefixes:-system default}"
    printf '  Enter or b) Configure and build    q) Quit\n'
    printf 'Build jobs use at most half the available CPUs; MPI tests allow four CPUs each.\n'
}

dialogChoice() {
    whiptail --clear --backtitle 'CosmicTiger builder' "$@" 3>&1 1>&2 2>&3
}

chooseDialogNumber() {
    local label=$1
    local currentValue=$2
    local maximum=$3
    local numericValue
    while true; do
        if ! promptValue=$(dialogChoice --title "$label" \
            --inputbox "Enter a whole number of jobs from 1 to $maximum." \
            10 68 "$currentValue"); then
            promptValue=$currentValue
            return
        fi
        if [[ -z $promptValue ]]; then
            promptValue=$currentValue
            return
        fi
        if [[ $promptValue =~ ^[0-9]+$ && ${#promptValue} -le 6 ]]; then
            numericValue=$((10#$promptValue))
            if (( numericValue >= 1 && numericValue <= maximum )); then
                promptValue=$numericValue
                return
            fi
        fi
        whiptail --clear --backtitle 'CosmicTiger builder' \
            --title 'Invalid job count' \
            --msgbox "Enter a whole number from 1 to $maximum." 8 58
    done
}

chooseDialogText() {
    local label=$1
    local currentValue=$2
    local description=$3
    if promptValue=$(dialogChoice --title "$label" \
        --inputbox "$description" 10 72 "$currentValue"); then
        if [[ $promptValue == '-' ]]; then
            promptValue=
        fi
    else
        promptValue=$currentValue
    fi
}

if command -v whiptail >/dev/null 2>&1 && [[ ${TERM:-dumb} != dumb ]]; then
    while true; do
        if ! menuChoice=$(dialogChoice --title 'Build configuration' \
            --cancel-button 'Quit' --default-item 1 --notags \
            --menu 'Use arrow keys to choose a setting. Builds use up to half the available CPUs.' \
            21 78 10 \
            1 "Build mode: $buildMode" \
            2 "CUDA: $enableCuda" \
            3 "Build and run test suite: $testSuite" \
            4 "Parallel build jobs: $buildJobs" \
            5 "Parallel test jobs: $testJobs" \
            6 "HPX installation prefix: ${hpxPrefix:-automatic by build mode}" \
            7 "CUDA architectures: ${cudaArchitectures:-$defaultCudaArchitectures}" \
            8 "Additional dependency prefixes: ${dependencyPrefixes:-system default}" \
            b 'Configure and build' \
            q 'Quit'); then
            printf 'No build started.\n'
            exit 0
        fi
        case "$menuChoice" in
            b) break ;;
            q) printf 'No build started.\n'; exit 0 ;;
            1)
                if promptValue=$(dialogChoice --title 'Build mode' \
                    --default-item "$buildMode" \
                    --menu 'Select a build mode.' 14 64 4 \
                    release 'Optimized, without debug symbols' \
                    relwithdebinfo 'Optimized, with debug symbols' \
                    debug 'Debug checks and symbols' \
                    profile 'Profiling with gprof'); then
                    buildMode=$promptValue
                fi
                ;;
            2|3)
                if [[ $menuChoice == 2 ]]; then
                    dialogLabel='CUDA'
                    dialogCurrent=$enableCuda
                else
                    dialogLabel='Build and run test suite'
                    dialogCurrent=$testSuite
                fi
                if promptValue=$(dialogChoice --title "$dialogLabel" \
                    --default-item "$dialogCurrent" \
                    --menu "Select $dialogLabel." 12 58 2 \
                    ON 'Enabled' OFF 'Disabled'); then
                    if [[ $menuChoice == 2 ]]; then
                        enableCuda=$promptValue
                    else
                        testSuite=$promptValue
                    fi
                fi
                ;;
            4)
                chooseDialogNumber 'Parallel build jobs' "$buildJobs" "$maxJobs"
                buildJobs=$promptValue
                ;;
            5)
                chooseDialogNumber 'Parallel test jobs' "$testJobs" "$maxTestJobs"
                testJobs=$promptValue
                ;;
            6)
                chooseDialogText 'HPX installation prefix' "$hpxPrefix" \
                    'Enter an HPX prefix, or leave blank to use ~/local/hpx/<build type> if installed.'
                hpxPrefix=$promptValue
                ;;
            7)
                chooseDialogText 'CUDA architectures' "$cudaArchitectures" \
                    "Enter architectures (e.g. 86), native, or OFF for the compiler default. Blank uses $defaultCudaArchitectures."
                cudaArchitectures=$promptValue
                ;;
            8)
                chooseDialogText 'Additional dependency prefixes' "$dependencyPrefixes" \
                    'Enter semicolon-separated dependency prefixes. Use - to clear them.'
                dependencyPrefixes=$promptValue
                ;;
        esac
    done
else
    while true; do
        showMainMenu
        readInput 'Choice [build]: '
        case "${promptValue,,}" in
            ''|b|build) break ;;
            q|quit) printf 'No build started.\n'; exit 0 ;;
            1) chooseMode ;;
            2) chooseToggle 'Enable CUDA' "$enableCuda"; enableCuda=$promptValue ;;
            3) chooseToggle 'Build and run test suite' "$testSuite"; testSuite=$promptValue ;;
            4) chooseNumber 'Parallel build jobs' "$buildJobs" 1 "$maxJobs"; buildJobs=$promptValue ;;
            5) chooseNumber 'Parallel test jobs' "$testJobs" 1 "$maxTestJobs"; testJobs=$promptValue ;;
            6) chooseText 'HPX installation prefix' "$hpxPrefix"; hpxPrefix=$promptValue ;;
            7) chooseText 'CUDA architectures' "$cudaArchitectures"; cudaArchitectures=$promptValue ;;
            8) chooseText 'Additional dependency prefixes' "$dependencyPrefixes"; dependencyPrefixes=$promptValue ;;
            *) printf 'Unknown option.\n' ;;
        esac
    done
fi

for requiredCommand in cmake ninja gfortran tee mktemp; do
    if ! command -v "$requiredCommand" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$requiredCommand" >&2
        exit 1
    fi
done
if [[ $testSuite == ON ]] && ! command -v ctest >/dev/null 2>&1; then
    printf 'Missing required command: ctest\n' >&2
    exit 1
fi

buildDir="$rootDir/build/$buildMode"
if ! mkdir -p -- "$buildDir"; then
    printf 'Cannot create build directory: %s\n' "$buildDir" >&2
    exit 1
fi
logFile=$(mktemp "$buildDir/builder.XXXXXX.log") || exit 1

case "$buildMode" in
    release) buildType=Release; hpxBuildType=Release ;;
    relwithdebinfo) buildType=RelWithDebInfo; hpxBuildType=RelWithDebInfo ;;
    debug) buildType=Debug; hpxBuildType=Debug ;;
    profile) buildType=Profile; hpxBuildType=Release ;;
esac
if [[ -z $hpxPrefix && -z ${HPX_DIR:-} && -n ${HOME:-} &&
    -d $HOME/local/hpx/$hpxBuildType ]]; then
    hpxPrefix="$HOME/local/hpx/$hpxBuildType"
fi
if [[ $hpxPrefix == '~/'* && -n ${HOME:-} ]]; then
    hpxPrefix="${HOME}/${hpxPrefix:2}"
fi
prefixPath=$dependencyPrefixes
if [[ -n $hpxPrefix ]]; then
    prefixPath="$hpxPrefix${prefixPath:+;$prefixPath}"
fi

configureCommand=(cmake -S "$rootDir" -B "$buildDir" -G Ninja
    -UHPX_DIR '-UMPI_*'
    "-DCMAKE_BUILD_TYPE=$buildType"
    "-DCOSMICTIGER_WITH_CUDA=$enableCuda"
    "-DBUILD_TESTING=$testSuite"
    "-DCMAKE_PREFIX_PATH=$prefixPath")
# Help FindMPI locate mpi.h when Open MPI wrapper output contains diagnostics.
if command -v mpicxx >/dev/null 2>&1; then
    mpiIncludeText=$(mpicxx --showme:incdirs 2>/dev/null) || mpiIncludeText=
    read -r -a mpiIncludeDirectories <<< "$mpiIncludeText"
    for includeDir in "${mpiIncludeDirectories[@]}"; do
        if [[ -f $includeDir/mpi.h ]]; then
            configureCommand+=("-DMPI_CXX_HEADER_DIR=$includeDir")
            break
        fi
    done
fi
if [[ $buildMode == profile ]]; then
    configureCommand+=(
        "-DCMAKE_C_FLAGS_PROFILE=-O2 -g -pg -DNDEBUG"
        "-DCMAKE_CXX_FLAGS_PROFILE=-O2 -g -pg -DNDEBUG"
        "-DCMAKE_Fortran_FLAGS_PROFILE=-O2 -g -pg -DNDEBUG"
        "-DCMAKE_CUDA_FLAGS_PROFILE=-O2 -g -DNDEBUG -Xcompiler=-pg"
        "-DCMAKE_EXE_LINKER_FLAGS_PROFILE=-pg")
fi
if [[ $enableCuda == ON ]]; then
    configureCommand+=("-DCMAKE_CUDA_ARCHITECTURES=${cudaArchitectures:-$defaultCudaArchitectures}")
fi
# Initialization calls ./recfast, so build it alongside the main executable.
buildCommand=(cmake --build "$buildDir" --target cosmictiger recfast --parallel "$buildJobs")
testBuildCommand=(cmake --build "$buildDir" --target all --parallel "$buildJobs")
testCommand=(ctest --test-dir "$buildDir" --output-on-failure --parallel "$testJobs")

runStage() {
    local stageName=$1
    local commandText
    local -a pipeStatuses
    shift
    printf -v commandText '%q ' "$@"
    printf '\n%s: %s\n' "$stageName" "$commandText"
    printf '\n%s: %s\n' "$stageName" "$commandText" >> "$logFile" || return 1
    "$@" 2>&1 | tee -a "$logFile"
    pipeStatuses=("${PIPESTATUS[@]}")
    if (( pipeStatuses[1] != 0 )); then
        printf '%s could not write the log: %s\n' "$stageName" "$logFile" >&2
        return 1
    fi
    if (( pipeStatuses[0] != 0 )); then
        printf '%s failed (exit %s). Log: %s\n' "$stageName" "${pipeStatuses[0]}" "$logFile" >&2
        return "${pipeStatuses[0]}"
    fi
    printf '%s succeeded.\n' "$stageName"
}

printf '\nBuild directory: %s\nLog: %s\n' "$buildDir" "$logFile"
runStage Configure "${configureCommand[@]}" || exit $?
runStage 'Build cosmictiger and recfast' "${buildCommand[@]}" || exit $?
if [[ $testSuite == ON ]]; then
    runStage 'Build test suite' "${testBuildCommand[@]}" || exit $?
    runStage 'Run test suite' "${testCommand[@]}" || exit $?
fi
printf '\nCosmicTiger build succeeded: %s\nLog: %s\n' "$buildDir" "$logFile"
if [[ $testSuite == ON ]]; then
    printf 'Test suite passed.\n'
else
    printf 'Test suite was not requested.\n'
fi
