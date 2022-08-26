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

void fft3d_init(int64_t N, float init_const = 0.0);
void fft3d_execute();
void fft3d_inv_execute();
vector<float> fft3d_read_real(const range<int64_t>&);
vector<cmplx> fft3d_read_complex(const range<int64_t>&);
void fft3d_accumulate_real(const range<int64_t>&, const vector<float>&);
void fft3d_accumulate_complex(const range<int64_t>&, const vector<cmplx>&);
void fft3d_destroy();
void fft3d_force_real();
range<int64_t> fft3d_complex_range();
range<int64_t> fft3d_real_range();
vector<cmplx>& fft3d_complex_vector();

struct power_spectrum_t {
	vector<float> P;
	vector<float> k;
	vector<float> Perr;
};

power_spectrum_t fft3d_power_spectrum();
void fft3d2silo(bool real);
/*
 inline CUDA_EXPORT float cloud_weight(float x) {
 x = fabs(x);
 if (x > 2.0f) {
 return 0.0f;
 } else if (x < 1.0f) {
 return (4.0f - 6.0f * sqr(x) + 3.0f * x * sqr(x)) / 6.0f;
 } else {
 return (8.0f - 12.0 * x + 6.0f * sqr(x) - x * sqr(x)) / 6.0f;
 }
 }
 */

inline CUDA_EXPORT float cloud_weight(float x) {
	x = fabs(x);
	const float x2 = sqr(x);
	const float x3 = x * x2;
	const float x4 = x2 * x2;
	const float x5 = x3 * x2;
	if (x > 3.0f) {
		return 0.0f;
	} else if (x < 1.0f) {
		return (33.0f - 30.0 * x2 + 15.0f * x4 - 5.0f * x5) / 60.0f;
	} else if (x < 2.0f) {
		return (51.0 + 75.0 * x - 210 * x2 + 150 * x3 - 45 * x4 + 5 * x5) / 120.0;
	} else {
		return (243.0 - 405.0 * x + 270 * x2 - 90 * x3 + 15 * x4 - x5) / 120.0;
	}
}

inline double cloud_filter(double kh) {
	const double s = sinc(0.5 * kh);
	return 1.0 / (sqr(s * sqr(s)));
}
#define CLOUD_MIN -2
#define CLOUD_MAX 3

/*
 inline CUDA_EXPORT float cloud_weight(float  x) {
 if (x < -1.0f || x > 2.0f) {
 return 0.0f;
 } else if (x < 0.0f) {
 return 0.5f * sqr(1.0f + x);
 } else if (x < 1.0f) {
 return 0.5f + x - sqr(x);
 } else {
 return 0.5f * sqr(x - 2.0f);
 }
 }

 inline double cloud_filter(double kh) {
 const double s = sinc(0.5 * kh);
 return 1.0 / (sqr(s) * s);
 }
 #define CLOUD_MIN -1
 #define CLOUD_MAX 1

 */

#endif /* FFT_HPP_ */
