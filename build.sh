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
    printf 'The menu includes SPH support, enabled by default; disable it to omit SPH code and tests.\n'
    printf 'Use -j with no argument or -j max to use all available CPUs.\n'
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
for cacheFile in "$rootDir/build/release/CMakeCache.txt" "$rootDir"/build/*/CMakeCache.txt; do
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

# `-j` may explicitly select any positive job count; max means all CPUs.
maxJobs=$coreCount
# MPI integration tests use two localities with two HPX threads each.
maxTestJobs=$((coreCount / 8))
if (( maxTestJobs < 1 )); then
    maxTestJobs=1
fi

buildMode=release
gravityVariant=fmm
fmmOrder=7
fmmOrderExplicit=false
timeIntegrator=leapfrog
enableCuda=ON
enableSph=ON
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
        buildJobs=$coreCount
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

setGravityVariant() {
    gravityVariant=$1
    if [[ $fmmOrderExplicit == false ]]; then
        if [[ $gravityVariant == fmm ]]; then
            fmmOrder=7
        else
            fmmOrder=8
        fi
    fi
}

chooseGravityVariant() {
    while true; do
        printf '\nGravity solver: 1) FMM  2) PMMM-FMM  3) PMMM-PP\n'
        readInput "Choice [${gravityVariant}]: "
        case "${promptValue,,}" in
            '') return ;;
            1|fmm) setGravityVariant fmm; return ;;
            2|pmmm_fmm|pmmm-fmm) setGravityVariant pmmm_fmm; return ;;
            3|pmmm_pp|pmmm-pp) setGravityVariant pmmm_pp; return ;;
        esac
        printf 'Choose 1 through 3, or press Enter to keep the current solver.\n'
    done
}

chooseFmmOrder() {
    while true; do
        readInput "FMM kernel order [${fmmOrder}] (3 through 9, Enter keeps): "
        case "$promptValue" in
            '') return ;;
            [3-9]) fmmOrder=$promptValue; fmmOrderExplicit=true; return ;;
        esac
        printf 'Enter an integer from 3 through 9.\n'
    done
}

chooseTimeIntegrator() {
    while true; do
        printf '\nTime integrator: 1) Leapfrog (KDK)  2) Hierarchical (dark matter only)\n'
        readInput "Choice [${timeIntegrator}]: "
        case "${promptValue,,}" in
            '') return ;;
            1|leapfrog) timeIntegrator=leapfrog; return ;;
            2|hierarchical) timeIntegrator=hierarchical; return ;;
        esac
        printf 'Choose 1 or 2, or press Enter to keep the current integrator.\n'
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
    printf '  9) Gravity solver: %s\n' "$gravityVariant"
    printf ' 10) FMM kernel order: %s\n' "$fmmOrder"
    printf ' 11) Time integrator: %s\n' "$timeIntegrator"
    printf ' 12) Build SPH support: %s\n' "$enableSph"
    printf '  Enter or b) Configure and build    q) Quit\n'
    printf 'Build jobs accept any positive count; MPI tests use four CPUs each.\n'
}

dialogChoice() {
    whiptail --clear --backtitle 'CosmicTiger builder' "$@" 3>&1 1>&2 2>&3
}

chooseDialogNumber() {
    local label=$1
    local currentValue=$2
    while true; do
        if ! promptValue=$(dialogChoice --title "$label" \
            --inputbox 'Enter a positive whole number of jobs.' \
            10 68 "$currentValue"); then
            promptValue=$currentValue
            return
        fi
        if [[ -z $promptValue ]]; then
            promptValue=$currentValue
            return
        fi
        if [[ $promptValue =~ ^[0-9]+$ && $promptValue =~ [1-9] ]]; then
            return
        fi
        whiptail --clear --backtitle 'CosmicTiger builder' \
            --title 'Invalid job count' \
            --msgbox 'Enter a positive whole number.' 8 58
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
            --menu 'Use arrow keys to choose a setting. Build jobs accept any positive count.' \
            24 78 13 \
            1 "Build mode: $buildMode" \
            2 "CUDA: $enableCuda" \
            3 "Build and run test suite: $testSuite" \
            4 "Parallel build jobs: $buildJobs" \
            5 "Parallel test jobs: $testJobs" \
            6 "HPX installation prefix: ${hpxPrefix:-automatic by build mode}" \
            7 "CUDA architectures: ${cudaArchitectures:-$defaultCudaArchitectures}" \
            8 "Additional dependency prefixes: ${dependencyPrefixes:-system default}" \
            9 "Gravity solver: $gravityVariant" \
            10 "FMM kernel order: $fmmOrder" \
            11 "Time integrator: $timeIntegrator" \
            12 "Build SPH support: $enableSph" \
            b 'Configure and build' \
            q 'Quit'); then
            printf 'No build started.\n'
            exit 0
        fi
        case "$menuChoice" in
            9)
                if promptValue=$(dialogChoice --title 'Gravity solver' \
                    --default-item "$gravityVariant" \
                    --menu 'Select the solver compiled into this executable.' 13 68 3 \
                    fmm 'Global FMM' \
                    pmmm_fmm 'PMMM with FMM near interactions' \
                    pmmm_pp 'PMMM with direct particle-pair near interactions'); then
                    setGravityVariant "$promptValue"
                fi
                ;;
            10)
                if promptValue=$(dialogChoice --title 'FMM kernel order' \
                    --default-item "$fmmOrder" \
                    --menu 'Select the only FMM order compiled into this executable.' 17 68 7 \
                    3 'Multipole degree 1' 4 'Multipole degree 2' \
                    5 'Multipole degree 3' 6 'Multipole degree 4' \
                    7 'Multipole degree 5' 8 'Multipole degree 6' \
                    9 'Multipole degree 7'); then
                    fmmOrder=$promptValue
                    fmmOrderExplicit=true
                fi
                ;;
            11)
                if promptValue=$(dialogChoice --title 'Time integrator' \
                    --default-item "$timeIntegrator" \
                    --menu 'Select the time integrator compiled into this executable.' 12 72 2 \
                    leapfrog 'Ordinary block-timestep kick-drift-kick (default)' \
                    hierarchical 'Hierarchical gravity (dark matter only; no SPH)'); then
                    timeIntegrator=$promptValue
                fi
                ;;
            b) break ;;
            q) printf 'No build started.\n'; exit 0 ;;
            1)
                if promptValue=$(dialogChoice --title 'Build mode' \
                    --default-item "$buildMode" \
                    --menu 'Select a build mode.' 14 64 4 \
                    release 'Optimized, without debug symbols' \
                    relwithdebinfo 'Optimized, debug symbols and APEX tasks' \
                    debug 'Debug checks, symbols and APEX tasks' \
                    profile 'APEX tasks and host gprof instrumentation'); then
                    buildMode=$promptValue
                fi
                ;;
            2|3|12)
                if [[ $menuChoice == 2 ]]; then
                    dialogLabel='CUDA'
                    dialogCurrent=$enableCuda
                elif [[ $menuChoice == 3 ]]; then
                    dialogLabel='Build and run test suite'
                    dialogCurrent=$testSuite
                else
                    dialogLabel='Build SPH support'
                    dialogCurrent=$enableSph
                fi
                if promptValue=$(dialogChoice --title "$dialogLabel" \
                    --default-item "$dialogCurrent" \
                    --menu "Select $dialogLabel." 12 58 2 \
                    ON 'Enabled' OFF 'Disabled'); then
                    if [[ $menuChoice == 2 ]]; then
                        enableCuda=$promptValue
                    elif [[ $menuChoice == 3 ]]; then
                        testSuite=$promptValue
                    else
                        enableSph=$promptValue
                    fi
                fi
                ;;
            4)
                chooseDialogNumber 'Parallel build jobs' "$buildJobs"
                buildJobs=$promptValue
                ;;
            5)
                chooseDialogNumber 'Parallel test jobs' "$testJobs"
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
            4) chooseNumber 'Parallel build jobs' "$buildJobs"; buildJobs=$promptValue ;;
            5) chooseNumber 'Parallel test jobs' "$testJobs"; testJobs=$promptValue ;;
            6) chooseText 'HPX installation prefix' "$hpxPrefix"; hpxPrefix=$promptValue ;;
            7) chooseText 'CUDA architectures' "$cudaArchitectures"; cudaArchitectures=$promptValue ;;
            8) chooseText 'Additional dependency prefixes' "$dependencyPrefixes"; dependencyPrefixes=$promptValue ;;
            9) chooseGravityVariant ;;
            10) chooseFmmOrder ;;
            11) chooseTimeIntegrator ;;
            12) chooseToggle 'Build SPH support' "$enableSph"; enableSph=$promptValue ;;
            *) printf 'Unknown option.\n' ;;
        esac
    done
fi

for requiredCommand in cmake ninja tee mktemp; do
    if ! command -v "$requiredCommand" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$requiredCommand" >&2
        exit 1
    fi
done
if [[ $testSuite == ON ]] && ! command -v ctest >/dev/null 2>&1; then
    printf 'Missing required command: ctest\n' >&2
    exit 1
fi

case "$gravityVariant" in
    fmm) gravityMethod=fmm; pmmmNearMethod=pp ;;
    pmmm_fmm) gravityMethod=pmmm; pmmmNearMethod=fmm ;;
    pmmm_pp) gravityMethod=pmmm; pmmmNearMethod=pp ;;
esac
buildDir="$rootDir/build/${buildMode}_${gravityVariant}_order${fmmOrder}_${timeIntegrator}"
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
configureCommand=(cmake -S "$rootDir" -B "$buildDir" -G Ninja
    -UHPX_DIR '-UMPI_*'
    '-U*GSL*' '-UCHEALPIX_*' '-UHEALPIX_CXX_*'
    "-DCMAKE_BUILD_TYPE=$buildType"
    "-DCOSMICTIGER_WITH_CUDA=$enableCuda"
    "-DCOSMICTIGER_WITH_SPH=$enableSph"
    "-DCOSMICTIGER_GRAVITY_METHOD=$gravityMethod"
    "-DCOSMICTIGER_PMMM_NEAR_METHOD=$pmmmNearMethod"
    "-DCOSMICTIGER_WITH_ORDER=$fmmOrder"
    "-DCOSMICTIGER_TIME_INTEGRATOR=$timeIntegrator"
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
buildCommand=(cmake --build "$buildDir" --target cosmictiger --parallel "$buildJobs")
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

printf '\nGravity solver: %s; FMM kernel order: %s\n' "$gravityVariant" "$fmmOrder"
printf 'Time integrator: %s\n' "$timeIntegrator"
printf 'SPH support: %s\n' "$enableSph"
printf 'Build directory: %s\nLog: %s\n' "$buildDir" "$logFile"
runStage Configure "${configureCommand[@]}" || exit $?
runStage 'Build cosmictiger' "${buildCommand[@]}" || exit $?
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
