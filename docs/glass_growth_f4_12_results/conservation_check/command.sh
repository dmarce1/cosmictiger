#!/bin/bash
set -euo pipefail
/usr/bin/c++ -DADAPTIVE_SOFTENING_TEST_SIMD -O3 -DNDEBUG -std=gnu++20 -march=native -I/home/dmarce1/workspace/cosmictiger /tmp/cosmictiger_f4_conservation_check/adaptive_softening_f4.cpp -o /tmp/cosmictiger_f4_conservation_check/adaptive_softening_f4
/tmp/cosmictiger_f4_conservation_check/adaptive_softening_f4
