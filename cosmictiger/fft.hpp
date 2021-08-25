/*
 * fft.hpp
 *
 *  Created on: Jun 23, 2021
 *      Author: dmarce1
 */

#ifndef FFT_HPP_
#define FFT_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/complex.hpp>

void fft3d_init(int64_t N);
void fft3d_execute();
void fft3d_inv_execute();
vector<double> fft3d_read_real(const range<int64_t>&);
vector<cmplx> fft3d_read_complex(const range<int64_t>&);
void fft3d_accumulate_real(const range<int64_t>&, const vector<double>&);
void fft3d_accumulate_complex(const range<int64_t>&, const vector<cmplx>&);
void fft3d_destroy();
void fft3d_force_real();
range<int64_t> fft3d_complex_range();
range<int64_t> fft3d_real_range();
vector<cmplx>& fft3d_complex_vector();
vector<double> fft3d_power_spectrum();
void fft3d2silo(bool real);

#endif /* FFT_HPP_ */
