/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
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
