/*
 * fft.hpp
 *
 *  Created on: Jun 23, 2021
 *      Author: dmarce1
 */

#ifndef FFT_HPP_
#define FFT_HPP_

#include <tigerfmm/defs.hpp>
#include <tigerfmm/range.hpp>
#include <tigerfmm/complex.hpp>

void fft3d_init(int N);
void fft3d_execute();
void fft3d_inv_execute();
vector<double> fft3d_read_real(const range<int>&);
vector<cmplx> fft3d_read_complex(const range<int>&);
void fft3d_accumulate_real(const range<int>&, const vector<double>&);
void fft3d_accumulate_complex(const range<int>&, const vector<cmplx>&);
void fft3d_destroy();
void fft3d_force_real();
range<int> fft3d_complex_range();
range<int> fft3d_real_range();
vector<cmplx>& fft3d_complex_vector();

#endif /* FFT_HPP_ */
