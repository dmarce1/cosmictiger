#!/usr/bin/env bash
set -euo pipefail
cd /home/dmarce1/workspace/cosmictiger
# Run one compilation/timing at a time; coordinate with other local work.
nvcc -std=c++20 -O3 --use_fast_math -DNDEBUG -DORDER=7 -DUSE_CUDA -I. -arch=sm_86 --expt-relaxed-constexpr -rdc=true --compiler-options=-O3,-ffast-math,-march=native '-DCOSMICTIGER_STF_BASELINE_HEADER="/tmp/cosmictiger_stf_baseline/fmm_kernels.hpp"' benchmarks/stf_l2l_cuda.cu -o /tmp/cosmictiger_stf_l2l_bench/legacy_warp
/tmp/cosmictiger_stf_l2l_bench/legacy_warp 32768 20 7 > /tmp/cosmictiger_stf_l2l_bench/legacy_warp.csv 2> /tmp/cosmictiger_stf_l2l_bench/legacy_warp.stderr
nvcc -std=c++20 -O3 --use_fast_math -DNDEBUG -DORDER=7 -DUSE_CUDA -I. -arch=sm_86 --expt-relaxed-constexpr -rdc=true --compiler-options=-O3,-ffast-math,-march=native -DCOSMICTIGER_STF_L2L_LANE0 benchmarks/stf_l2l_cuda.cu -o /tmp/cosmictiger_stf_l2l_bench/stf_lane0
/tmp/cosmictiger_stf_l2l_bench/stf_lane0 32768 20 7 > /tmp/cosmictiger_stf_l2l_bench/stf_lane0.csv 2> /tmp/cosmictiger_stf_l2l_bench/stf_lane0.stderr
nvcc -std=c++20 -O3 --use_fast_math -DNDEBUG -DORDER=7 -DUSE_CUDA -I. -arch=sm_86 --expt-relaxed-constexpr -rdc=true --compiler-options=-O3,-ffast-math,-march=native benchmarks/stf_l2l_cuda.cu -o /tmp/cosmictiger_stf_l2l_bench/stf_warp
/tmp/cosmictiger_stf_l2l_bench/stf_warp 32768 20 7 > /tmp/cosmictiger_stf_l2l_bench/stf_warp.csv 2> /tmp/cosmictiger_stf_l2l_bench/stf_warp.stderr
/usr/local/cuda/bin/compute-sanitizer --tool synccheck --error-exitcode 1 /tmp/cosmictiger_stf_l2l_bench/stf_warp --check
