#!/usr/bin/env bash
# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

set -u

showUsage() {
    printf 'Usage: %s [-j [jobs|max]]\n' "${0##*/}"
    printf 'Run the interactive CosmicTiger configure and build menu in a terminal.\n'
    printf 'The menu selects compiled capabilities for cosmictiger.\n'
    printf 'SPH requires traditional or global integration; gravity may be disabled.\n'
    printf 'Particle chemistry is independently selectable and requires SPH.\n'
    printf 'Stars are selectable with COSMICTIGER_ENABLE_STARS and require chemistry and SPH.\n'
    printf 'Position storage is selected for the whole build directory; fixed32 is the default.\n'
    printf 'Use -j with no argument or -j max to use half the available CPUs.\n'
}

requestedBuildJobs=
if (( $# > 0 )); then
    case "$1" in
        --help|-h)
            showUsage
            exit 0
            ;;
        -j)
            shift
            if (( $# > 0 )) && [[ $1 =~ ^[0-9]+$ || $1 == max ]]; then
                requestedBuildJobs=$1
                shift
            else
                requestedBuildJobs=max
            fi
            ;;
        *) showUsage >&2; exit 2 ;;
    esac
fi
if (( $# != 0 )); then
    showUsage >&2
    exit 2
fi

if [[ ! -t 0 || ! -t 1 ]]; then
    printf 'An interactive terminal is required. Run ./build.sh in a terminal.\n' >&2
    exit 2
fi

rootDir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || exit 1
cd -- "$rootDir" || exit 1

defaultCudaArchitectures=
readBuildDefaults() {
    local cacheFile=$1
    local cacheKey cacheValue
    local cachedSourceDir= cachedArchitectures=
    [[ -f $cacheFile ]] || return 0
    # Read cache data without evaluating it as shell code.
    while IFS='=' read -r cacheKey cacheValue; do
        case "$cacheKey" in
            CMAKE_HOME_DIRECTORY:*) cachedSourceDir=$cacheValue ;;
            CMAKE_CUDA_ARCHITECTURES:*) cachedArchitectures=$cacheValue ;;
        esac
    done < "$cacheFile"
    [[ $cachedSourceDir == "$rootDir" ]] || return 0
    if [[ -z $defaultCudaArchitectures && -n $cachedArchitectures &&
        $cachedArchitectures != native ]]; then
        defaultCudaArchitectures=$cachedArchitectures
    fi
}

# Prefer the default build, then reuse another build belonging to this checkout.
for cacheFile in "$rootDir/build/release_cuda/CMakeCache.txt" "$rootDir"/build/*/CMakeCache.txt; do
    readBuildDefaults "$cacheFile"
    if [[ -n $defaultCudaArchitectures ]]; then
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

# Keep build and test concurrency within half the available CPUs.
maxJobs=$((coreCount / 2))
(( maxJobs > 0 )) || maxJobs=1
# MPI integration tests use two localities with two HPX threads each.
maxTestJobs=$((coreCount / 8))
if (( maxTestJobs < 1 )); then
    maxTestJobs=1
fi

buildMode=release
enableFmm=ON
enablePmmmPp=OFF
enablePmmmFmm=OFF
expansionDegree=4-6
enableTraditional=ON
enableHierarchical=OFF
enableGlobal=OFF
positionType=fixed32
enableCuda=ON
enableSph=ON
enableChemistry=ON
enableStars=OFF
testSuite=OFF
buildJobs=4
testJobs=2
if (( buildJobs > maxJobs )); then
    buildJobs=$maxJobs
fi
if (( testJobs > maxTestJobs )); then
    testJobs=$maxTestJobs
fi
if [[ -n $requestedBuildJobs ]]; then
    if [[ $requestedBuildJobs == max ]]; then
        buildJobs=$maxJobs
    elif [[ $requestedBuildJobs =~ ^[0-9]+$ ]] && [[ $requestedBuildJobs =~ [1-9] ]]; then
        buildJobs=$requestedBuildJobs
    else
        printf 'Build job count must be a positive integer or max.\n' >&2
        exit 2
    fi
fi
hpxPrefix=
# CMake uses colon-separated environment paths and semicolon-separated cache paths.
dependencyPrefixes=${CMAKE_PREFIX_PATH-}
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

chooseGravityFeatures() {
    chooseToggle 'Compile global FMM' "$enableFmm"; enableFmm=$promptValue
    chooseToggle 'Compile PMMM with PP near field' "$enablePmmmPp"; enablePmmmPp=$promptValue
    chooseToggle 'Compile PMMM with FMM near field' "$enablePmmmFmm"; enablePmmmFmm=$promptValue
}

chooseExpansionDegree() {
    while true; do
        readInput "Potential expansion degrees [${expansionDegree}] (2-8, scalar or inclusive range): "
        [[ -z $promptValue ]] && return
        if [[ $promptValue =~ ^([2-8])(-([2-8]))?$ ]] &&
            [[ -z ${BASH_REMATCH[3]} || ${BASH_REMATCH[1]} -le ${BASH_REMATCH[3]} ]]; then
            expansionDegree=$promptValue
            return
        fi
        printf 'Enter a degree or an ordered inclusive range, for example 6 or 4-6.\n'
    done
}

chooseIntegrators() {
    chooseToggle 'Compile traditional nested integration' "$enableTraditional"; enableTraditional=$promptValue
    chooseToggle 'Compile hierarchical integration' "$enableHierarchical"; enableHierarchical=$promptValue
    chooseToggle 'Compile global adaptive integration' "$enableGlobal"; enableGlobal=$promptValue
}

choosePositionType() {
    while true; do
        printf '\nPosition storage: 1) fixed32 (default)  2) fixed64\n'
        readInput "Choice [${positionType}]: "
        case "${promptValue,,}" in
            '') return ;;
            1|fixed32) positionType=fixed32; return ;;
            2|fixed64) positionType=fixed64; return ;;
        esac
        printf 'Choose 1 or 2, or press Enter to keep the current position storage.\n'
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
    while true; do
        readInput "$label [${currentValue}] (positive whole number, Enter keeps): "
        if [[ -z $promptValue ]]; then
            promptValue=$currentValue
            return
        fi
        if [[ $promptValue =~ ^[0-9]+$ && $promptValue =~ [1-9] ]]; then
            return
        fi
        printf 'Enter a positive whole number.\n'
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
    printf '  9) Gravity: FMM=%s PMMM PP=%s PMMM FMM=%s\n' "$enableFmm" "$enablePmmmPp" "$enablePmmmFmm"
    printf ' 10) Potential expansion degrees: %s\n' "$expansionDegree"
    printf ' 11) Integrators: traditional=%s hierarchical=%s global=%s\n' "$enableTraditional" "$enableHierarchical" "$enableGlobal"
    printf ' 12) Build SPH support: %s\n' "$enableSph"
    printf ' 13) Position storage: %s\n' "$positionType"
    printf ' 14) Build particle chemistry (requires SPH): %s\n' "$enableChemistry"
    printf ' 15) Build stars (requires chemistry and SPH): %s\n' "$enableStars"
    printf '  Enter or b) Configure and build    q) Quit\n'
    printf 'Build jobs are capped at half the available CPUs; MPI tests use four CPUs each.\n'
}

validateFeatures() {
    if [[ $enableFmm != ON && $enablePmmmPp != ON && $enablePmmmFmm != ON && $enableSph != ON ]]; then
        printf 'Enable at least one physics component.\n' >&2; return 1
    fi
    if [[ $enableTraditional != ON && $enableHierarchical != ON && $enableGlobal != ON ]]; then
        printf 'Enable at least one integrator.\n' >&2; return 1
    fi
    if [[ $enableSph == ON && $enableTraditional != ON && $enableGlobal != ON ]]; then
        printf 'SPH requires traditional or global integration.\n' >&2; return 1
    fi
    if [[ $enableStars == ON && $enableChemistry != ON ]]; then
        printf 'Stars require particle chemistry. Enable chemistry (option 14) or disable stars (option 15).\n' >&2; return 1
    fi
    if (( buildJobs > maxJobs )); then
        printf 'Build concurrency is limited to %s jobs.\n' "$maxJobs" >&2; return 1
    fi
    if (( testJobs > maxTestJobs )); then
        printf 'Test concurrency is limited to %s jobs (four CPUs per MPI test).\n' "$maxTestJobs" >&2; return 1
    fi
}

while true; do
    showMainMenu
    readInput 'Choice [build]: '
    case "${promptValue,,}" in
        ''|b|build) if validateFeatures; then break; fi ;;
        q|quit) printf 'No build started.\n'; exit 0 ;;
        1) chooseMode ;;
        2) chooseToggle 'Enable CUDA' "$enableCuda"; enableCuda=$promptValue ;;
        3) chooseToggle 'Build and run test suite' "$testSuite"; testSuite=$promptValue ;;
        4) chooseNumber 'Parallel build jobs' "$buildJobs"; buildJobs=$promptValue ;;
        5) chooseNumber 'Parallel test jobs' "$testJobs"; testJobs=$promptValue ;;
        6) chooseText 'HPX installation prefix' "$hpxPrefix"; hpxPrefix=$promptValue ;;
        7) chooseText 'CUDA architectures' "$cudaArchitectures"; cudaArchitectures=$promptValue ;;
        8) chooseText 'Additional dependency prefixes' "$dependencyPrefixes"; dependencyPrefixes=$promptValue ;;
        9) chooseGravityFeatures ;;
        10) chooseExpansionDegree ;;
        11) chooseIntegrators ;;
        12) chooseToggle 'Build SPH support' "$enableSph"; enableSph=$promptValue ;;
        13) choosePositionType ;;
        14) chooseToggle 'Build particle chemistry (requires SPH)' "$enableChemistry"; enableChemistry=$promptValue ;;
        15) chooseToggle 'Build stars (requires chemistry and SPH)' "$enableStars"; enableStars=$promptValue ;;
        *) printf 'Unknown option.\n' ;;
    esac
done

for requiredCommand in cmake make tee mktemp; do
    if ! command -v "$requiredCommand" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$requiredCommand" >&2
        exit 1
    fi
done
if [[ $testSuite == ON ]] && ! command -v ctest >/dev/null 2>&1; then
    printf 'Missing required command: ctest\n' >&2
    exit 1
fi

if [[ $enableCuda == ON ]]; then platformVariant=cuda; else platformVariant=cpu; fi
buildTarget=cosmictiger
buildDir="$rootDir/build/${buildMode}_${platformVariant}"
if ! mkdir -p -- "$buildDir"; then
    printf 'Cannot create build directory: %s\n' "$buildDir" >&2
    exit 1
fi
logFile=$(mktemp "$buildDir/builder.XXXXXX.log") || exit 1

case "$buildMode" in
    release) buildType=Release; hpxBuildType=Release; requireApex=OFF ;;
    relwithdebinfo) buildType=RelWithDebInfo; hpxBuildType=RelWithDebInfo; requireApex=ON ;;
    debug) buildType=Debug; hpxBuildType=Debug; requireApex=ON ;;
    profile) buildType=Profile; hpxBuildType=RelWithDebInfo; requireApex=ON ;;
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

# Rediscover dependency paths so old cached /tmp installs do not hide system packages.
configureCommand=(cmake -S "$rootDir" -B "$buildDir" -G 'Unix Makefiles'
    -UHPX_DIR '-UMPI_*'
    -UCOSMICTIGER_TIME_INTEGRATOR -UCOSMICTIGER_GRAVITY_METHOD
    -UCOSMICTIGER_PMMM_NEAR_METHOD -UCOSMICTIGER_WITH_ORDER
    -UCOSMICTIGER_ENABLE_PMMM -UCOSMICTIGER_ENABLE_PMMM_NEAR_PP
    -UCOSMICTIGER_ENABLE_PMMM_NEAR_FMM
    -UCOSMICTIGER_FMM_MIN_ORDER -UCOSMICTIGER_FMM_MAX_ORDER -UCOSMICTIGER_WITH_SPH
    '-U*GSL*' '-UCHEALPIX_*' '-UHEALPIX_CXX_*'
    "-DCMAKE_BUILD_TYPE=$buildType"
    "-DCOSMICTIGER_WITH_CUDA=$enableCuda"
    "-DCOSMICTIGER_ENABLE_SPH=$enableSph"
    "-DCOSMICTIGER_ENABLE_CHEMISTRY=$enableChemistry"
    "-DCOSMICTIGER_ENABLE_STARS=$enableStars"
    "-DCOSMICTIGER_ENABLE_FMM=$enableFmm"
    "-DCOSMICTIGER_ENABLE_PMMM_PP=$enablePmmmPp"
    "-DCOSMICTIGER_ENABLE_PMMM_FMM=$enablePmmmFmm"
    "-DCOSMICTIGER_EXPANSION_DEGREE=$expansionDegree"
    "-DCOSMICTIGER_ENABLE_INTEGRATOR_TRADITIONAL=$enableTraditional"
    "-DCOSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL=$enableHierarchical"
    "-DCOSMICTIGER_ENABLE_INTEGRATOR_GLOBAL=$enableGlobal"
    "-DCOSMICTIGER_POSITION_TYPE=$positionType"
    "-DCOSMICTIGER_REQUIRE_APEX=$requireApex"
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
        "-DCMAKE_CUDA_FLAGS_PROFILE=-O2 -g -DNDEBUG -Xcompiler=-pg"
        "-DCMAKE_EXE_LINKER_FLAGS_PROFILE=-pg")
fi
if [[ $enableCuda == ON ]]; then
    configureCommand+=("-DCMAKE_CUDA_ARCHITECTURES=${cudaArchitectures:-$defaultCudaArchitectures}")
fi
buildCommand=(cmake --build "$buildDir" --target "$buildTarget" --parallel "$buildJobs")
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

printf '\nGravity: FMM=%s PMMM PP=%s PMMM FMM=%s; potential degrees=%s\n' "$enableFmm" "$enablePmmmPp" "$enablePmmmFmm" "$expansionDegree"
printf 'Integrators: traditional=%s hierarchical=%s global=%s\n' "$enableTraditional" "$enableHierarchical" "$enableGlobal"
printf 'SPH support: %s\n' "$enableSph"
printf 'Particle chemistry: %s (requires SPH)\n' "$enableChemistry"
printf 'Stars: %s (requires chemistry and SPH)\n' "$enableStars"
printf 'Position storage: %s\n' "$positionType"
printf 'Build target: %s\n' "$buildTarget"
printf 'Build directory: %s\nLog: %s\n' "$buildDir" "$logFile"
runStage Configure "${configureCommand[@]}" || exit $?
runStage "Build $buildTarget" "${buildCommand[@]}" || exit $?
if [[ $testSuite == ON ]]; then
    runStage 'Build test suite' "${testBuildCommand[@]}" || exit $?
    runStage 'Run test suite' "${testCommand[@]}" || exit $?
fi
printf '\nCosmicTiger build succeeded: %s/%s\nLog: %s\n' "$buildDir" "$buildTarget" "$logFile"
if [[ $testSuite == ON ]]; then
    printf 'Test suite passed.\n'
else
    printf 'Test suite was not requested.\n'
fi
