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

#ifndef FFTS_HPP_
#define FFTS_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/complex.hpp>

void fft3d_init(int64_t N, float init_const = 0.0);
void fft3d_execute();
void fft3d_inv_execute();
vector<float> fft3d_read_real(const range<int64_t>&);
vector<cmplx> fft3d_read_complex(const range<int64_t>&);
range<int64_t> fft3d_complex_range();
void fft3d_accumulate_real(const range<int64_t>&, const vector<float>&);
void fft3d_accumulate_complex(const range<int64_t>&, const vector<cmplx>&);
void fft3d_destroy();
void fft3d_force_real();
range<int64_t> fft3d_real_range();
vector<cmplx>& fft3d_complex_vector();

struct power_spectrum_t {
	vector<float> P;
	vector<float> k;
	vector<float> Perr;
};

power_spectrum_t fft3d_power_spectrum();
void fft3d2silo(bool real);


#endif /* FFT_HPP_ */
