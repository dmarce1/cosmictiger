# Historical SIMD comparison scripts

These scripts supported the October 18, 2026 migration from custom SIMD
wrappers to standard-library SIMD (`91fccebc`). They compare against baseline
`b50b0404f9dcbc074e8e1b55f7f4e30b8e0c44ea` and are retained as source records,
with `.py.txt` extensions, rather than current build or benchmark commands.

They were retired for review finding `build_cmake_scripts#12`. Their dependency
cache, Open MPI `mpi_cxx` requirement, dispatcher source anchor, single-order
CMake option, and `BUILD_TESTING=OFF` execution path do not match the current
tree. Moving the files also invalidates their original relative-path assumptions.
Do not use these scripts with a current checkout. To reproduce the historical
experiment, inspect its original revision and recorded toolchain first.

For current SIMD correctness coverage, configure a build with `BUILD_TESTING=ON` using the
[repository build instructions](../../../README.md), then build the
`simd_auto_test`, `simd_baseline_test`, and `simd_native_test` targets and run
the corresponding `simd_` CTests. These are correctness tests; they do not
replace the old paired performance experiment. There is no maintained
replacement for that old/new snapshot benchmark pipeline.
