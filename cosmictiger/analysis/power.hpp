/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#ifndef POWER_HPP_
#define POWER_HPP_

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/math/range.hpp>

// Binned power spectrum: mean power, bin wavenumber and its standard error.
struct PowerSpectrumT {
    Vector<float> p;
    Vector<float> k;
    Vector<float> perr;
};

// Deposits the particles on the FFT grid with fold m and bins the spectrum.
PowerSpectrumT powerSpectrumCompute(int m);
Vector<float> accumulateDensityCpu(int fold, int gridSize, Range<int64_t> box);
Vector<float> accumulateDensityCuda(int m, int ndim, Range<int64_t> intbox);

#endif /* POWER_HPP_ */
