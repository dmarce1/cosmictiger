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

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/fft_vector.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/kernels.hpp>

#include <fftw3.h>
#include <silo.h>

#define MAX_BOX_SIZE (8*1024*1024)

static int64_t N;
static int vsize;
static vector<range<int64_t>> real_boxes;
static array<vector<range<int64_t>>, NDIM> cmplx_boxes;
static range<int64_t> real_mybox;
static array<range<int64_t>, NDIM> cmplx_mybox;
static vector<std::shared_ptr<spinlock_type>> mutexes;

static vector<vector<double>> R;
static vector<vector<complex<double>>> Y;
static vector<vector<complex<double>>> Y1;

static void find_boxes(range<int64_t> box, vector<range<int64_t>>& boxes, int begin, int end);
static void split_box(range<int64_t> box, vector<range<int64_t>>& real_boxes);
static void vect_transpose(int, int);
static void vect_shift(bool);
static void vect_update();
static void fft3d_vect_phase1();
static void fft3d_vect_phase2(int, bool);
static void fft3d_vect_phase3();
static vector<vector<complex<double>>> vect_transpose_read(const range<int64_t>&, int dim1, int dim2);
static vector<vector<complex<double>>> vect_shift_read(const range<int64_t>&, bool);

HPX_PLAIN_ACTION (fft3d_vect_accumulate_real);
HPX_PLAIN_ACTION (fft3d_vect_accumulate_complex);
HPX_PLAIN_ACTION (fft3d_vect_init);
HPX_PLAIN_ACTION (fft3d_vect_execute);
HPX_PLAIN_ACTION (fft3d_vect_phase1);
HPX_PLAIN_ACTION (fft3d_vect_phase2);
HPX_PLAIN_ACTION (fft3d_vect_phase3);
HPX_PLAIN_ACTION (fft3d_vect_read_real);
HPX_PLAIN_ACTION (fft3d_vect_read_complex);
HPX_PLAIN_ACTION (fft3d_vect_destroy);
HPX_PLAIN_ACTION (vect_transpose_read);
HPX_PLAIN_ACTION (vect_transpose);
HPX_PLAIN_ACTION (vect_shift);
HPX_PLAIN_ACTION (vect_shift_read);
HPX_PLAIN_ACTION (vect_update);

#include <fenv.h>


void fft3d_vect_execute() {
//	PRINT("FFT z\n");
	fft3d_vect_phase1();
//	PRINT("Transpose y-z\n");
	vect_transpose(1, 2);
//	PRINT("FFT y\n");
	fft3d_vect_phase2(1, false);
//	PRINT("Shifting\n");
	vect_shift(false);
//	PRINT("FFT x\n");
	fft3d_vect_phase2(0, false);
//	PRINT("Transpose z-x\n");
	vect_transpose(2, 0);
	vect_update();
//	PRINT("done\n");

}

void fft3d_vect_inv_execute() {
//	PRINT("Transpose z-x\n");
	vect_transpose(0, 2);
//	PRINT("inv FFT x\n");
	fft3d_vect_phase2(0, true);
//	PRINT("Shifting\n");
	vect_shift(true);
//	PRINT("inv FFT y\n");
	fft3d_vect_phase2(1, true);
//	PRINT("Transpose y-z\n");
	vect_transpose(2, 1);
//	PRINT("inv FFT z\n");
	fft3d_vect_phase3();
//	PRINT("done\n");

}

range<int64_t> fft3d_vect_complex_range() {
	return cmplx_mybox[ZDIM];
}

range<int64_t> fft3d_vect_real_range() {
	return real_mybox;
}

vector<vector<complex<double>>>& fft3d_vect_complex_vector() {
	return Y;
}

vector<vector<double>> fft3d_vect_read_real(const range<int64_t>& this_box) {
	vector<hpx::future<void>> futs;
	vector<vector<double>> data(vsize, vector<double>(this_box.volume()));
	if (real_mybox.contains(this_box)) {
		array<int64_t, NDIM> i;
		for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
			for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
				for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
					const int k = this_box.index(i);
					const int j = real_mybox.index(i);
					for (int l = 0; l < vsize; l++) {
						data[l][k] = R[l][j];
					}
				}
			}
		}
	} else {
		for (int ri = 0; ri < hpx_size(); ri++) {
			array<int64_t, NDIM> si;
			for (si[0] = -N; si[0] <= +N; si[0] += N) {
				for (si[1] = -N; si[1] <= +N; si[1] += N) {
					for (si[2] = -N; si[2] <= +N; si[2] += N) {
						const auto shifted_box = this_box.shift(si);
						const auto inter = real_boxes[ri].intersection(shifted_box);
						if (inter.volume()) {
							vector < range < int64_t >> inters;
							split_box(inter, inters);
							for (auto this_inter : inters) {
								auto fut = hpx::async<fft3d_vect_read_real_action>(hpx_localities()[ri], this_inter);
								futs.push_back(fut.then([si,this_box,this_inter,&data](hpx::future<vector<vector<double>>> fut) {
									auto this_data = fut.get();
									array<int64_t,NDIM> i;
									for (i[0] = this_inter.begin[0]; i[0] != this_inter.end[0]; i[0]++) {
										for (i[1] = this_inter.begin[1]; i[1] != this_inter.end[1]; i[1]++) {
											for (i[2] = this_inter.begin[2]; i[2] != this_inter.end[2]; i[2]++) {
												auto j = i;
												for( int dim = 0; dim < NDIM; dim++) {
													j[dim] -= si[dim];
												}
												const int ll =this_box.index(j);
												const int kk =this_inter.index(i);
												for( int l=0;l<vsize;l++) {
													data[l][ll] = this_data[l][kk];
												}
											}
										}
									}
								}));
							}
						}
					}
				}
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	return std::move(data);
}

vector<vector<complex<double>>> fft3d_vect_read_complex(const range<int64_t>& this_box) {
	vector<hpx::future<void>> futs;
	vector<vector<complex<double>>> data(vsize,vector<complex<double>>(this_box.volume()));
	const auto mybox = cmplx_mybox[ZDIM];
	if (mybox.contains(this_box)) {
		array<int64_t, NDIM> i;
		for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
			for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
				for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
					const int j = this_box.index(i);
					const int k = mybox.index(i);
					for( int l = 0; l < vsize; l++) {
						data[l][j] = Y[l][k];
					}
				}
			}
		}
	} else {
		for (int ri = 0; ri < hpx_size(); ri++) {
			array<int64_t, NDIM> si;
			for (si[0] = -N; si[0] <= +N; si[0] += N) {
				for (si[1] = -N; si[1] <= +N; si[1] += N) {
					for (si[2] = -N; si[2] <= +N; si[2] += N) {
						const auto shifted_box = this_box.shift(si);
						const auto inter = cmplx_boxes[ZDIM][ri].intersection(shifted_box);
						if (inter.volume()) {
							vector < range < int64_t >> inters;
							split_box(inter, inters);
							for (auto this_inter : inters) {
								auto fut = hpx::async<fft3d_vect_read_complex_action>(hpx_localities()[ri], this_inter);
								futs.push_back(fut.then([si,this_box,this_inter,&data](hpx::future<vector<vector<complex<double>>>> fut) {
													auto this_data = fut.get();
													array<int64_t,NDIM> i;
													for (i[0] = this_inter.begin[0]; i[0] != this_inter.end[0]; i[0]++) {
														for (i[1] = this_inter.begin[1]; i[1] != this_inter.end[1]; i[1]++) {
															for (i[2] = this_inter.begin[2]; i[2] != this_inter.end[2]; i[2]++) {
																auto j = i;
																for( int dim = 0; dim < NDIM; dim++) {
																	j[dim] -= si[dim];
																}
																const int kk = this_box.index(j);
																const int ll =this_inter.index(i);
																for(int k=0;k<vsize;k++) {
																	data[k][kk] = this_data[k][ll];
																}
															}
														}
													}
												}));
							}
						}
					}
				}
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	return std::move(data);
}

void fft3d_vect_accumulate_real(const range<int64_t>& this_box, const vector<vector<double>>& data) {
	vector<hpx::future<void>> futs;
	if (real_mybox.contains(this_box)) {
		array<int64_t, NDIM> i;
		for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
			for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
				const auto& box = real_mybox;
				const int64_t mtxindex = (i[0] - box.begin[0]) * (box.end[1] - box.begin[1]) + (i[1] - box.begin[1]);
				std::lock_guard<spinlock_type> lock(*mutexes[mtxindex]);
				for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
					const int jj = box.index(i);
					const int kk = this_box.index(i);
					for (int l = 0; l < vsize; l++) {
						R[l][jj] += data[l][kk];
					}
				}
			}
		}
	} else {
		for (int bi = 0; bi < real_boxes.size(); bi++) {
			array<int64_t, NDIM> si;
			for (si[0] = -N; si[0] <= +N; si[0] += N) {
				for (si[1] = -N; si[1] <= +N; si[1] += N) {
					for (si[2] = -N; si[2] <= +N; si[2] += N) {
						const auto shifted_box = this_box.shift(si);
						const auto inter = real_boxes[bi].intersection(shifted_box);
						if (!inter.empty()) {
							vector < range < int64_t >> inters;
							split_box(inter, inters);
							for (auto this_inter : inters) {
								vector<vector<double>> this_data;
								this_data.resize(vsize, vector<double>(this_inter.volume()));
								array<int64_t, NDIM> i;
								for (i[0] = this_inter.begin[0]; i[0] != this_inter.end[0]; i[0]++) {
									for (i[1] = this_inter.begin[1]; i[1] != this_inter.end[1]; i[1]++) {
										for (i[2] = this_inter.begin[2]; i[2] != this_inter.end[2]; i[2]++) {
											const int64_t k = this_inter.index(i);
											auto j = i;
											for (int dim = 0; dim < NDIM; dim++) {
												j[dim] -= si[dim];
											}
											const int64_t l = this_box.index(j);
											ASSERT(k < this_data.size());
											ASSERT(l < data.size());
											for (int p = 0; p < vsize; p++) {
												this_data[p][k] = data[p][l];
											}
										}
									}
								}
								auto fut = hpx::async<fft3d_vect_accumulate_real_action>(hpx_localities()[bi], this_inter, std::move(this_data));
								fut.get();
//								futs.push_back(std::move(fut));
							}
						}
					}
				}
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d_vect_accumulate_complex(const range<int64_t>& this_box, const vector<vector<complex<double>>>& data) {
	vector<hpx::future<void>> futs;
	const auto& box = cmplx_mybox[ZDIM];
	if (box.contains(this_box)) {
		array<int64_t, NDIM> i;
		for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
			for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
				const int64_t mtxindex = (i[0] - box.begin[0]) * (box.end[1] - box.begin[1]) + (i[1] - box.begin[1]);
				std::lock_guard<spinlock_type> lock(*mutexes[mtxindex]);
				for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
					const int ll = box.index(i);
					const int kk = this_box.index(i);
					for( int p =0;p<vsize;p++) {
						Y[p][ll] += data[p][kk];
					}
				}
			}
		}
	} else {
		for (int64_t bi = 0; bi < cmplx_boxes[ZDIM].size(); bi++) {
			array<int64_t, NDIM> si;
			for (si[0] = -N; si[0] <= +N; si[0] += N) {
				for (si[1] = -N; si[1] <= +N; si[1] += N) {
					for (si[2] = -N; si[2] <= +N; si[2] += N) {
						const auto shifted_box = this_box.shift(si);
						const auto inter = cmplx_boxes[ZDIM][bi].intersection(shifted_box);
						if (!inter.empty()) {
							vector < range < int64_t >> inters;
							split_box(inter, inters);
							for (auto this_inter : inters) {
								vector<vector<complex<double>>> this_data;
								this_data.resize(vsize, vector<complex<double>>(this_inter.volume()));
								array<int64_t, NDIM> i;
								for (i[0] = this_inter.begin[0]; i[0] != this_inter.end[0]; i[0]++) {
									for (i[1] = this_inter.begin[1]; i[1] != this_inter.end[1]; i[1]++) {
										for (i[2] = this_inter.begin[2]; i[2] != this_inter.end[2]; i[2]++) {
											auto j = i;
											for (int dim = 0; dim < NDIM; dim++) {
												j[dim] -= si[dim];
											}
											const int cc = this_inter.index(i);
											const int dd = this_box.index(j);
											for(int ll = 0; ll< vsize;ll++) {
												this_data[ll][cc] = data[ll][dd];
											}
										}
									}
								}
								auto fut = hpx::async<fft3d_vect_accumulate_complex_action>(hpx_localities()[bi], this_inter, std::move(this_data));
								futs.push_back(std::move(fut));
							}
						}
					}
				}
			}
		}
	}

	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d_vect_init(int64_t N_, int vsize0, double init_const) {
	fft3d_vect_destroy();
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<fft3d_vect_init_action>(c, N_, vsize0,init_const));
	}
	N = N_;
	vsize = vsize0;
	range<int64_t> box;
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = 0;
		box.end[dim] = N;
	}
	real_boxes.resize(hpx_size());
	find_boxes(box, real_boxes, 0, hpx_size());
	real_mybox = real_boxes[hpx_rank()];
	R.resize(0);
	R.resize(vsize, vector<double>(real_mybox.volume(), init_const));
	for (int dim = 0; dim < NDIM; dim++) {
		for (int dim1 = 0; dim1 < NDIM; dim1++) {
			box.begin[dim1] = 0;
			box.end[dim1] = N;
		}
		box.end[dim] = N / 2 + 1;
		cmplx_boxes[dim].resize(hpx_size());
		find_boxes(box, cmplx_boxes[dim], 0, hpx_size());
		cmplx_mybox[dim] = cmplx_boxes[dim][hpx_rank()];
	}
	Y.resize(0);
	Y.resize(vsize, vector<complex<double>>(cmplx_mybox[ZDIM].volume(), complex<double>(0.0, 0.0)));
	mutexes.resize(N * N);
	for (int64_t i = 0; i < N * N; i++) {
		mutexes[i] = std::make_shared<spinlock_type>();
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d_vect_destroy() {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<fft3d_vect_destroy_action>(c));
	}
	R = decltype(R)();
	Y = decltype(Y)();
	Y1 = decltype(Y1)();

	hpx::wait_all(futs.begin(), futs.end());

}

static void vect_update() {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<vect_update_action>(c));
	}
	Y = std::move(Y1);
	hpx::wait_all(futs.begin(), futs.end());
}

static void fft3d_vect_phase1() {
	static mutex_type mtx;
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<fft3d_vect_phase1_action>(c));
	}
	array<int64_t, NDIM> i;
	Y.resize(vsize, vector<complex<double>>(cmplx_mybox[ZDIM].volume()));
	for (int vi = 0; vi < vsize; vi++) {
		futs.push_back(hpx::async([](int vi) {
			array<int64_t,NDIM> i;
			for (i[0] = real_mybox.begin[0]; i[0] != real_mybox.end[0]; i[0]++) {
				fftw_plan p;
				fftw_complex out[N / 2 + 1];
				double in[N];
				std::unique_lock<mutex_type> lock(mtx);
				p = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE );
				lock.unlock();
				for (i[1] = real_mybox.begin[1]; i[1] != real_mybox.end[1]; i[1]++) {
					for (i[2] = 0; i[2] != N; i[2]++) {
						in[i[2]] = R[vi][real_mybox.index(i)];

					}
					fftw_execute(p);
					for (i[2] = 0; i[2] < N / 2 + 1; i[2]++) {
						const int64_t l = cmplx_mybox[ZDIM].index(i);
						Y[vi][l].real() = out[i[2]][0];
						Y[vi][l].imag() = out[i[2]][1];

					}
				}
				lock.lock();
				fftw_destroy_plan(p);
				lock.unlock();
			}
		}, vi));
	}
	hpx::wait_all(futs.begin(), futs.end());
	R = decltype(R)();
}

static void fft3d_vect_phase2(int dim, bool inv) {
	static mutex_type mtx;
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<fft3d_vect_phase2_action>(c, dim, inv));
	}
	Y = std::move(Y1);
	const double norm = inv ? 1.0f / N : 1.0;
	for (int vi = 0; vi < vsize; vi++) {
		futs.push_back(hpx::async([dim,inv,norm](int vi) {
			array<int64_t,NDIM> i;
			for (i[0] = cmplx_mybox[dim].begin[0]; i[0] != cmplx_mybox[dim].end[0]; i[0]++) {
				fftw_complex out[N];
				fftw_complex in[N];
				fftw_plan p;
				std::unique_lock<mutex_type> lock(mtx);
				p = fftw_plan_dft_1d(N, in, out, inv ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);
				lock.unlock();
				for (i[1] = cmplx_mybox[dim].begin[1]; i[1] != cmplx_mybox[dim].end[1]; i[1]++) {
					ASSERT(cmplx_mybox[dim].begin[2]==0);
					ASSERT(cmplx_mybox[dim].end[2]==N);
					for (i[2] = 0; i[2] < N; i[2]++) {
						const auto l = cmplx_mybox[dim].index(i);
						ASSERT(l >= 0 );
						ASSERT( l < Y.size());
						in[i[2]][0] = Y[vi][l].real();
						in[i[2]][1] = Y[vi][l].imag();
					}
					fftw_execute(p);
					for (i[2] = 0; i[2] < N; i[2]++) {
						const int64_t l = cmplx_mybox[dim].index(i);
						Y[vi][l].real() = out[i[2]][0] * norm;
						Y[vi][l].imag() = out[i[2]][1] * norm;
					}
				}
				lock.lock();
				fftw_destroy_plan(p);
				lock.unlock();
			}
		}, vi));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

static void fft3d_vect_phase3() {
	static mutex_type mtx;
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<fft3d_vect_phase3_action>(c));
	}
	array<int64_t, NDIM> i;
	Y = std::move(Y1);
	R.resize(0);
	R.resize(vsize, vector<double>(real_mybox.volume()));
	const double Ninv = 1.0 / N;
	for (int vi = 0; vi < vsize; vi++) {
		futs.push_back(hpx::async([Ninv](int vi) {
			array<int64_t,NDIM> i;
			for (i[0] = cmplx_mybox[ZDIM].begin[0]; i[0] != cmplx_mybox[ZDIM].end[0]; i[0]++) {
				fftw_plan p;
				fftw_complex in[N / 2 + 1];
				double out[N];
				std::unique_lock<mutex_type> lock(mtx);
				p = fftw_plan_dft_c2r_1d(N, in, out, FFTW_ESTIMATE );
				lock.unlock();
				for (i[1] = cmplx_mybox[ZDIM].begin[1]; i[1] !=cmplx_mybox[ZDIM].end[1]; i[1]++) {
					for (i[2] = 0; i[2] < N / 2 + 1; i[2]++) {
						const int64_t l = cmplx_mybox[ZDIM].index(i);
						in[i[2]][0] = Y[vi][l].real();
						in[i[2]][1] = Y[vi][l].imag();
					}
					fftw_execute(p);
					for (i[2] = 0; i[2] != N; i[2]++) {
						R[vi][real_mybox.index(i)] = out[i[2]] * Ninv;

					}
				}
				lock.lock();
				fftw_destroy_plan(p);
				lock.unlock();
			}
		}, vi));
	}
	hpx::wait_all(futs.begin(), futs.end());
	Y = decltype(Y)();
}

static void find_boxes(range<int64_t> box, vector<range<int64_t>>& boxes, int begin, int end) {
	if (end - begin == 1) {
		boxes[begin] = box;
	} else {
		const int xdim = (box.end[0] - box.begin[0] > 1) ? 0 : 1;
		auto left = box;
		auto right = box;
		const int mid = (begin + end) / 2;
		const double w = double(mid - begin) / double(end - begin);
		left.end[xdim] = right.begin[xdim] = (((1.0 - w) * box.begin[xdim] + w * box.end[xdim]) + 0.5);
		find_boxes(left, boxes, begin, mid);
		find_boxes(right, boxes, mid, end);
	}
}

static void vect_transpose(int dim1, int dim2) {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<vect_transpose_action>(c, dim1, dim2));
	}
	range<int64_t> tbox = cmplx_mybox[dim1].transpose(dim1, dim2);
	Y1.resize(cmplx_mybox[dim1].volume());
	for (int64_t bi = 0; bi < cmplx_boxes[dim2].size(); bi++) {
		const auto tinter = cmplx_boxes[dim2][bi].intersection(tbox);
		vector<double> data;
		if (!tinter.empty()) {
			vector < range < int64_t >> tinters;
			split_box(tinter, tinters);
			for (auto this_tinter : tinters) {
				auto inter = this_tinter.transpose(dim1, dim2);
				auto fut = hpx::async<vect_transpose_read_action>(hpx_localities()[bi], this_tinter, dim1, dim2);
				futs.push_back(fut.then([inter,dim1](hpx::future<vector<vector<complex<double>>>> fut) {
					auto data = fut.get();
					array<int64_t,NDIM> i;
					for (i[0] = inter.begin[0]; i[0] != inter.end[0]; i[0]++) {
						for (i[1] = inter.begin[1]; i[1] != inter.end[1]; i[1]++) {
							for (i[2] = inter.begin[2]; i[2] != inter.end[2]; i[2]++) {
								const int k = inter.index(i);
								const int64_t l = cmplx_mybox[dim1].index(i);
								ASSERT(k < data.size());
								ASSERT(l < Y1.size());
								for( int vi=0;vi<vsize;vi++) {
									Y1[vi][l] = data[vi][k];
								}
							}
						}
					}
				}));
			}
		}
	}

	hpx::wait_all(futs.begin(), futs.end());

}

static vector<vector<complex<double>>> vect_transpose_read(const range<int64_t>& this_box, int dim1, int dim2) {
	vector<vector<complex<double>>> data(vsize,vector<complex<double>>(this_box.volume()));
	ASSERT(cmplx_mybox[dim2].contains(this_box));
	auto tbox = this_box.transpose(dim1, dim2);
	array<int64_t, NDIM> i;
	for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
		for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
			for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
				auto j = i;
				std::swap(j[dim1], j[dim2]);
				const int jj =tbox.index(j);
				const int ii=cmplx_mybox[dim2].index(i);
				for(int ll =0;ll<vsize;ll++) {
					data[ll][jj] = Y[ll][ii];
				}
			}
		}
	}
	return std::move(data);
}

static void vect_shift(bool inv) {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<vect_shift_action>(c, inv));
	}
	const int dim2 = inv ? YDIM : XDIM;
	const int dim1 = inv ? XDIM : YDIM;
	range<int64_t> tbox = inv ? cmplx_mybox[dim2].shift_up() : cmplx_mybox[dim2].shift_down();
	Y1.resize(0);
	Y1.resize(vsize, vector<complex<double>>(cmplx_mybox[dim2].volume()));
	for (int64_t bi = 0; bi < cmplx_boxes[dim1].size(); bi++) {
		const auto tinter = cmplx_boxes[dim1][bi].intersection(tbox);
		vector<double> data;
		if (!tinter.empty()) {
			vector < range < int64_t >> tinters;
			split_box(tinter, tinters);
			for (auto this_tinter : tinters) {
				auto inter = inv ? this_tinter.shift_down() : this_tinter.shift_up();
				auto fut = hpx::async<vect_shift_read_action>(hpx_localities()[bi], this_tinter, inv);
				futs.push_back(fut.then([inter,dim2](hpx::future<vector<vector<complex<double>>>> fut) {
					auto data = fut.get();
					array<int64_t,NDIM> i;
					for (i[0] = inter.begin[0]; i[0] != inter.end[0]; i[0]++) {
						for (i[1] = inter.begin[1]; i[1] != inter.end[1]; i[1]++) {
							for (i[2] = inter.begin[2]; i[2] != inter.end[2]; i[2]++) {
								const int64_t k = inter.index(i);
								const int64_t l = cmplx_mybox[dim2].index(i);
								ASSERT(k < data.size());
								ASSERT(l < Y1.size());
								for(int vi = 0; vi < vsize;vi++) {
									Y1[vi][l] = data[vi][k];
								}
							}
						}
					}
				}));
			}
		}
	}

	hpx::wait_all(futs.begin(), futs.end());

}

static vector<vector<complex<double>>> vect_shift_read(const range<int64_t>& this_box, bool inv) {
	vector<vector<complex<double>>> data(vsize,vector<complex<double>>(this_box.volume()));
	auto tbox = inv ? this_box.shift_down() : this_box.shift_up();
	array<int64_t, NDIM> i;
	const int dim = inv ? XDIM : YDIM;
	for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
		for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
			for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
				auto j = inv ? shift_down(i) : shift_up(i);
				const int jj =tbox.index(j);
				const int ii =cmplx_mybox[dim].index(i);
				for(int vi = 0;vi <vsize;vi++) {
					data[vi][jj] = Y[vi][ii];
				}
			}
		}
	}
	return std::move(data);
}

static void split_box(range<int64_t> box, vector<range<int64_t>>& real_boxes) {
	if (box.volume() < MAX_BOX_SIZE) {
		real_boxes.push_back(box);
	} else {
		auto children = box.split();
		split_box(children.first, real_boxes);
		split_box(children.second, real_boxes);
	}
}
