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

#ifndef FFT_HPP_
#define FFT_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/complex.hpp>

void fft3d_vect_init(int64_t N, int  vect_size, double init_const = 0.0);
void fft3d_vect_execute();
void fft3d_vect_inv_execute();
vector<vector<double>> fft3d_vect_read_real(const range<int64_t>&);
vector<vector<complex<double>>> fft3d_vect_read_complex(const range<int64_t>&);
range<int64_t> fft3d_vect_complex_range();
void fft3d_vect_accumulate_real(const range<int64_t>&, const vector<vector<double>>&);
void fft3d_vect_accumulate_complex(const range<int64_t>&, const vector<vector<complex<double>>>&);
void fft3d_vect_destroy();
void fft3d_vect_force_real();
range<int64_t> fft3d_vect_real_range();
vector<vector<complex<double>>>& fft3d_vect_complex_vector();

struct power_spectrum_vect_t {
	vector<double> P;
	vector<double> k;
	vector<double> Perr;
};

power_spectrum_vect_t fft3d_vect_power_spectrum();
void fft3d_vect2silo(bool real);


#endif /* FFT_HPP_ */
