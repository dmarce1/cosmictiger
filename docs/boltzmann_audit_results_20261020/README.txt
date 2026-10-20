Einstein-Boltzmann numerical audit scratch evidence (2026-10-20)

Production files were read only. Each process used one CPU core; at most two
compiler/run processes ran concurrently. No HPX, MPI, CUDA or QB validation.

build_actual.py extracts the actual init/kernel/evolve source from boltzmann.cpp
and actual fraction formulas; minimal synthetic ZeroOrderUniverse removes HPX.
actual_terminal.cpp uses the unchanged evolution source and initializes only
F_(LMAX-1)=1 with sigmaT=0 and k=1e-9. actual_terminal_results.txt shows that
shrinking a one-step log(a) interval to 1e-6 still drops that moment to .12998,
while a consistent finite-rate discretization must tend to 1. The intercepted
opacity evaluation times are c=.54289322,.95710678 relative to the interval.

reproduce.py separately tests the implemented SDIRK algebra/time offsets on
scalar y'=-exp(x)y and on the frozen-coefficient photon-baryon acoustic subsystem.
It is a simplified mathematical regression, not a production cosmology test.
synthetic_results.txt is the recorded output. NumPy and SciPy are used.

build_background.py extracts exact boltzmann init/evolution and zero_order.cpp
functions (dropping only heavy include/logging dependencies), includes production
headers and links src/cosmology/recfast.cpp. The only evolution edit in scratch
variants is max dlog(a)=.05 -> the indicated smaller cap. Build, e.g.:

g++ -O2 -std=c++20 -I/home/dmarce1/workspace/cosmictiger \
  /tmp/eb_integrator_audit/background_5e-2.cpp \
  /home/dmarce1/workspace/cosmictiger/src/cosmology/recfast.cpp \
  -o /tmp/eb_integrator_audit/background_5e-2

/tmp/eb_integrator_audit/background_5e-2 \
  /home/dmarce1/workspace/cosmictiger/build/sph_cpu_validation/tests/recfast_history/fiducial/history.dat

Fiducial RECFAST history SHA256:
b0e50cf8e0de6d2f17eb6a124628976d3684cb18a16f98dce91d7aa7e4b73936

Parameters equal tests/recombination_background.cpp: omegaB=.049389,
omegaC=.26503, nEff=3.046, Y=.2454006, Tcmb=2.7255K, h=.6732.
Photon density is computed directly from blackbody density and omegaNu from
(7/8)(4/11)^(4/3)*nEff*omegaGamma. Tilt=.966, amplitude=1. Evolve a=uni.amin
through a=1. k=.01,.1,1 Mpc^-1. Source hashes in source_sha256.txt.

Comparing cap=.05 to cap=.00625 yields relative delta_c differences
-0.0775564%, +0.590289%, +1.54744% and squared-amplitude differences
-0.155053%, +1.18406%, +3.11883%. This is a timestep sensitivity experiment,
not an external reference, converged solution, or sigma8-normalized P(k) error.
The cap variants retain every other identified solver defect.

photon_only_*.cpp derives from the same harness by setting omegaNu=0,nEff=0
and testing k=.1. The RECFAST xe history is deliberately held fixed to isolate
the solver; this is not a self-consistent photon-only recombination history.
The .05 to .00625 differences are +1.82986% in delta_c and +3.69321% squared.

Persistent copies of the evidence are stored beside this README. Run
python3 docs/boltzmann_audit_results_20261020/build_actual.py
and build_background.py to regenerate the scratch C++ sources. The generators
resolve the repository root from their location and create /tmp/eb_integrator_audit.
The synthetic reproduce.py requires NumPy and SciPy.

power_bounds.cpp extracts the production PowerSpectrumFunction lookup. Its
Vector alias is std::vector, matching the unchecked indexing inherited by the
project Vector in Release builds (NDEBUG). In Debug, CHECK_BOUNDS makes the
same invalid query abort instead. Reproduce its expected ASan failure with:
c++ -std=c++20 -O1 -g -DNDEBUG -fsanitize=address -fno-omit-frame-pointer \
  docs/boltzmann_audit_results_20261020/power_bounds.cpp -o /tmp/eb_power_bounds
ASAN_OPTIONS=detect_leaks=0 /tmp/eb_power_bounds

background_regression.log records a fresh standalone build of
 tests/recombination_background.cpp src/cosmology/zero_order.cpp
 src/cosmology/recfast.cpp with -std=c++20 -O2 -I. and the fiducial history.
manifest.json records the audited working-file hashes and validation scope.
