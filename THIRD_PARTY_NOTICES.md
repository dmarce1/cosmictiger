# Third-party code and dependencies

CosmicTiger's original code is licensed under the Boost Software License 1.0
in [LICENSE](LICENSE) and [LICENSE_1_0.txt](LICENSE_1_0.txt). This grant applies
to CosmicTiger's contributions, not to third-party material whose terms are
listed below or retained in individual files. Existing copyright notices
remain in place. Historical snapshots and patches under `docs/` retain the
notices of the versions they record.

The current executable includes and links GPL components. Relicensing
CosmicTiger's contributions does not remove those components' obligations or
make the combined executable distributable under Boost terms alone.

## Bundled and adapted source

| Material | Location | Attribution and retained terms |
| --- | --- | --- |
| Fatal-signal stack tracer | `src/core/stack_trace.cpp`, `cosmictiger/core/stack_trace.hpp`, `cmake/FindLibbacktrace.cmake` | Adapted from OctoTiger `stacktrace_fix`, commit `a0720ee02feebf776b797b2fdc74edf6bcc5bd83`; copyright 2026 AUTHORS, Boost Software License 1.0. |
| HEALPix CUDA adaptation | `src/analysis/healpix.cu` | Copyright 1997–2010 Krzysztof M. Gorski, Eric Hivon, Benjamin D. Wandelt, Anthony J. Banday, Matthias Bartelmann, Reza Ansari and Kenneth M. Ganga. GPL version 2 or later; original notice preserved. Compiled in CUDA builds. |
| GADGET-4 snapshot header declaration | `GadgetIoHeader` in `cosmictiger/particles/particles.hpp` | The declaration is explicitly marked as copied from GADGET-4. Upstream `src/io/snap_io.h` credits Volker Springel and contributing authors (2014–2020). GADGET-4 is GPL-licensed; the original imported revision and precise license version were not recorded here. The copied declaration is excluded from the project's Boost grant. |
| RECFAST effective-atom recombination physics | `src/chemistry/recombination.cpp` | RECFAST 1.5.2, Douglas Scott; University of British Columbia, copyright 1999–2010. The file retains UBC's permissive use/copy/modify/distribute grant and warranty disclaimer verbatim. CosmicTiger's integration and interpolation are original. |
| STF generator import | `tools/generate_stf_m2l.py`, `tools/stf/` | Imported from OctotigerII / `dmarce1/experimental`, commit `7ba8f1829fdede7348a1418b7cb36c0e7a955302`; Boost Software License 1.0, retained in `tools/stf/LICENSE`. The CosmicTiger Ewald emitter and generated-notice template now use Boost terms as well. |
| Sedov reference solution | `src/sph/sedov.cpp` | Existing copyright 2019 AUTHORS and Boost Software License 1.0 notice preserved. The source also attributes its Fortran origin to the [Cococubed Sedov implementation](http://cococubed.asu.edu/research_pages/sedov.shtml) and records the f2c translation. The historical AUTHORS identity and original Fortran revision are not pinned in this checkout. |
| Exact Sod reference solution | `src/sph/exact_sod.cpp`, `cosmictiger/sph/exact_sod.hpp` | Existing copyright 2019 AUTHORS and Boost Software License 1.0 notice preserved. The implementation attributes the original [ITAM `sod.f`](http://www.itam.nsc.ru/OLD2/flowlib/SRC/sod.f); its revision is not pinned here. |
| Formatter configuration | `.clang-format` | Copyright 2016 Thomas Heller and Hartmut Kaiser; Boost Software License 1.0. |
| Linux memory-usage snippet | `parseLine` / `cpuMemUse` in `src/core/memused.cpp` | Adapted from Lanzelot's [Stack Overflow answer](https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process), first posted September 15, 2008; modified for CosmicTiger's resident-memory reporting. Stack Overflow content uses CC BY-SA (2.5 for the original 2008 contribution; later revisions can carry later versions). The imported revision is not recorded, so the original link is retained and no Boost grant is asserted for this snippet. See [Stack Overflow's licensing policy](https://stackoverflow.com/help/licensing). |
| CUDA double-precision `atomicAdd` fallback | `cosmictiger/cuda/cuda.hpp` | Explicitly attributed to NVIDIA's CUDA Programming Guide. The original documentation version and separate snippet license were not recorded; the attribution remains and the snippet is excluded from the project's Boost grant. |
| cuFFT error helper | `cosmictiger/cuda/cuda.hpp` | Attributed to the [NVIDIA developer forum discussion](https://forums.developer.nvidia.com/t/cufft-error-handling/29231). The original author/revision and reuse license are not recorded here; no Boost grant is asserted for this snippet. |
| Floating-point `atomicMax` helper | `cosmictiger/cuda/cuda.hpp` | Attributed to [this Local Coder page](https://localcoder.org/how-do-i-use-atomicmax-on-floating-point-values-in-cuda). The original author/revision and reuse license are not recorded here; no Boost grant is asserted for this snippet. |

[licenses/GPL-2.0.txt](licenses/GPL-2.0.txt) supplies the GPL version 2 text for
the retained HEALPix material. The per-file “or later” permissions remain intact.
The provenance gaps identified above require the original license evidence or
replacement implementations before claiming every embedded snippet is covered
by a known permissive license.

The former top-level `recfast/` implementation is no longer a bundled build
dependency. The native C++ adaptation still needs its UBC attribution. Archived
RECFAST numerical reference results under `docs/` support regression tests;
those data and their recorded provenance are not an active Fortran dependency.

## Libraries provided by the build environment

These are located by `CMakeLists.txt`; their full source is not bundled here.
Installed versions and build options determine the complete transitive
dependency list, including libraries used by HPX and Silo.

| Dependency | Use | License / source of terms |
| --- | --- | --- |
| HPX | Parallel runtime | [Boost Software License 1.0](https://github.com/TheHPXProject/hpx/blob/master/LICENSE_1_0.txt). |
| Boost | Program options, units, stack traces and support headers | [Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt); retain any package-specific notices. |
| libbacktrace | Stack-trace symbol and source-line resolution | The installed `backtrace.h` supplies its permissive redistribution terms; retain the selected package's copyright, conditions and disclaimer. |
| GSL and GSL CBLAS | Random generation in initialization and particle operations | [GNU GPL](https://www.gnu.org/software/gsl/); this is an active dependency. |
| FFTW3 single precision, serial and MPI | FFTs in `src/cosmology/fft.cpp` and MPI FFT support | [GNU GPL](https://www.fftw.org/faq/section1.html#isfftwfree), unless separately licensed; this is an active dependency. |
| MPI implementation | HPX communication and FFTW3-MPI support | Supplied by the selected MPI implementation; retain its package-specific license and notices. |
| HEALPix C and C++ libraries | Sphere pixelization and analysis | [GNU GPL](https://healpix.sourceforge.io/downloads.php). |
| Silo | Scientific visualization output | [Silo's BSD-style copyright and license](https://github.com/LLNL/Silo/blob/main/COPYRIGHT); underlying storage libraries have their own terms. |
| CUDA toolkit/runtime, when enabled | GPU execution | NVIDIA's toolkit/component licenses; CUDA's presence does not change third-party source grants. |
| C/C++ runtimes, `libquadmath`, `libdl`, threads | Language, extended precision, dynamic loading and threading support | Platform/compiler package licenses and applicable runtime exceptions. |

Python, CMake and the C++/CUDA compilers are build/test tools, not bundled
application libraries. Optional validation scripts can also use independently
installed scientific packages such as NumPy, SciPy and CAMB; those packages
retain their own licenses.
