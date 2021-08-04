#include <tigerfmm/cuda.hpp>
#include <tigerfmm/fft.hpp>
#include <tigerfmm/hpx.hpp>

#include <fftw3.h>

#define MAX_BOX_SIZE (8*1024*1024)

static int N;
static vector<range<int>> real_boxes;
static array<vector<range<int>>, NDIM> cmplx_boxes;
static range<int> real_mybox;
static array<range<int>, NDIM> cmplx_mybox;
static vector<std::shared_ptr<spinlock_type>> mutexes;

static vector<float> R;
static vector<cmplx> Y;
static vector<cmplx> ym;
static vector<cmplx> Y1;

static void find_boxes(range<int> box, vector<range<int>>& boxes, int begin, int end);
static void split_box(range<int> box, vector<range<int>>& real_boxes);
static void transpose(int, int);
static void shift(bool);
static void update();
static void fft3d_phase1();
static void fft3d_phase2(int, bool);
static void fft3d_phase3();
static void finish_force_real();
static vector<cmplx> transpose_read(const range<int>&, int dim1, int dim2);
static vector<cmplx> shift_read(const range<int>&, bool);

HPX_PLAIN_ACTION(fft3d_accumulate_real);
HPX_PLAIN_ACTION(fft3d_accumulate_complex);
HPX_PLAIN_ACTION(fft3d_init);
HPX_PLAIN_ACTION(fft3d_execute);
HPX_PLAIN_ACTION(fft3d_phase1);
HPX_PLAIN_ACTION(fft3d_phase2);
HPX_PLAIN_ACTION(fft3d_phase3);
HPX_PLAIN_ACTION(fft3d_read_real);
HPX_PLAIN_ACTION(fft3d_read_complex);
HPX_PLAIN_ACTION(fft3d_force_real);
HPX_PLAIN_ACTION(fft3d_destroy);
HPX_PLAIN_ACTION(transpose_read);
HPX_PLAIN_ACTION(transpose);
HPX_PLAIN_ACTION(shift);
HPX_PLAIN_ACTION(shift_read);
HPX_PLAIN_ACTION(update);
HPX_PLAIN_ACTION(finish_force_real);

void fft3d_execute() {
	PRINT("FFT z\n");
	fft3d_phase1();
	PRINT("Transpose y-z\n");
	transpose(1, 2);
	PRINT("FFT y\n");
	fft3d_phase2(1, false);
	PRINT("Shifting\n");
	shift(false);
	PRINT("FFT x\n");
	fft3d_phase2(0, false);
	PRINT("Transpose z-x\n");
	transpose(2, 0);
	update();
	PRINT("done\n");

}

void fft3d_inv_execute() {
	PRINT("Transpose z-x\n");
	transpose(0, 2);
	PRINT("inv FFT x\n");
	fft3d_phase2(0, true);
	PRINT("Shifting\n");
	shift(true);
	PRINT("inv FFT y\n");
	fft3d_phase2(1, true);
	PRINT("Transpose y-z\n");
	transpose(2, 1);
	PRINT("inv FFT z\n");
	fft3d_phase3();
	PRINT("done\n");

}

void fft3d_force_real() {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<fft3d_force_real_action>(c));
	}

	const auto& box = cmplx_mybox[ZDIM];
	array<int, NDIM> i;
	range<int> mirror_box = box;
	range<int> slim_box = box;
	slim_box.end[ZDIM] = 1;
	ym.resize(slim_box.volume());
	mirror_box.end[ZDIM] = 1;
	for (int dim = 0; dim < NDIM - 1; dim++) {
		mirror_box.begin[dim] = N - box.end[dim] + 1;
		mirror_box.end[dim] = N - box.begin[dim] + 1;
	}
	const auto data = fft3d_read_complex(mirror_box);
	for (i[0] = mirror_box.begin[0]; i[0] != mirror_box.end[0]; i[0]++) {
		for (i[1] = mirror_box.begin[1]; i[1] != mirror_box.end[1]; i[1]++) {
			for (i[2] = mirror_box.begin[2]; i[2] != mirror_box.end[2]; i[2]++) {
				auto j = i;
				j[0] = (N - j[0]) % N;
				j[1] = (N - j[1]) % N;
				const auto y1 = Y[box.index(j)];
				const auto y2 = data[mirror_box.index(i)];
				const auto y = (y1 + y2.conj()) * 0.5;
				ym[slim_box.index(j)] = y;
			}
		}
	}

	hpx::wait_all(futs.begin(), futs.end());

	if (hpx_rank() == 0) {
		finish_force_real();
	}
}

range<int> fft3d_complex_range() {
	return cmplx_mybox[ZDIM];
}

range<int> fft3d_real_range() {
	return real_mybox;
}

vector<cmplx>& fft3d_complex_vector() {
	return Y;
}

void finish_force_real() {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<finish_force_real_action>(c));
	}
	const auto& box = cmplx_mybox[ZDIM];
	array<int, NDIM> i;
	range<int> slim_box = box;
	slim_box.end[ZDIM] = 1;
	for (i[0] = slim_box.begin[0]; i[0] != slim_box.end[0]; i[0]++) {
		for (i[1] = slim_box.begin[1]; i[1] != slim_box.end[1]; i[1]++) {
			for (i[2] = slim_box.begin[2]; i[2] != slim_box.end[2]; i[2]++) {
				Y[box.index(i)] = ym[slim_box.index(i)];
			}
		}
	}

	hpx::wait_all(futs.begin(), futs.end());
}

vector<float> fft3d_read_real(const range<int>& this_box) {
	vector<hpx::future<void>> futs;
	vector<float> data(this_box.volume());
	if (real_mybox.contains(this_box)) {
		array<int, NDIM> i;
		for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
			for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
				for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
					data[this_box.index(i)] = R[real_mybox.index(i)];
				}
			}
		}
	} else {
		for (int ri = 0; ri < hpx_size(); ri++) {
			array<int, NDIM> si;
			for (si[0] = -N; si[0] <= +N; si[0] += N) {
				for (si[1] = -N; si[1] <= +N; si[1] += N) {
					for (si[2] = -N; si[2] <= +N; si[2] += N) {
						const auto shifted_box = this_box.shift(si);
						const auto inter = real_boxes[ri].intersection(shifted_box);
						if (inter.volume()) {
							vector<range<int>> inters;
							split_box(inter, inters);
							for (auto this_inter : inters) {
								auto fut = hpx::async<fft3d_read_real_action>(hpx_localities()[ri], this_inter);
								futs.push_back(fut.then([si,this_box,this_inter,&data](hpx::future<vector<float>> fut) {
									auto this_data = fut.get();
									array<int, NDIM> i;
									for (i[0] = this_inter.begin[0]; i[0] != this_inter.end[0]; i[0]++) {
										for (i[1] = this_inter.begin[1]; i[1] != this_inter.end[1]; i[1]++) {
											for (i[2] = this_inter.begin[2]; i[2] != this_inter.end[2]; i[2]++) {
												auto j = i;
												for( int dim = 0; dim < NDIM; dim++) {
													j[dim] -= si[dim];
												}
												data[this_box.index(j)] = this_data[this_inter.index(i)];
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

vector<cmplx> fft3d_read_complex(const range<int>& this_box) {
	vector<hpx::future<void>> futs;
	vector<cmplx> data(this_box.volume());
	const auto mybox = cmplx_mybox[ZDIM];
	if (mybox.contains(this_box)) {
		array<int, NDIM> i;
		for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
			for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
				for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
					data[this_box.index(i)] = Y[mybox.index(i)];
				}
			}
		}
	} else {
		for (int ri = 0; ri < hpx_size(); ri++) {
			array<int, NDIM> si;
			for (si[0] = -N; si[0] <= +N; si[0] += N) {
				for (si[1] = -N; si[1] <= +N; si[1] += N) {
					for (si[2] = -N; si[2] <= +N; si[2] += N) {
						const auto shifted_box = this_box.shift(si);
						const auto inter = cmplx_boxes[ZDIM][ri].intersection(shifted_box);
						if (inter.volume()) {
							vector<range<int>> inters;
							split_box(inter, inters);
							for (auto this_inter : inters) {
								auto fut = hpx::async<fft3d_read_complex_action>(hpx_localities()[ri], this_inter);
								futs.push_back(fut.then([si,this_box,this_inter,&data](hpx::future<vector<cmplx>> fut) {
									auto this_data = fut.get();
									array<int, NDIM> i;
									for (i[0] = this_inter.begin[0]; i[0] != this_inter.end[0]; i[0]++) {
										for (i[1] = this_inter.begin[1]; i[1] != this_inter.end[1]; i[1]++) {
											for (i[2] = this_inter.begin[2]; i[2] != this_inter.end[2]; i[2]++) {
												auto j = i;
												for( int dim = 0; dim < NDIM; dim++) {
													j[dim] -= si[dim];
												}
												data[this_box.index(j)] = this_data[this_inter.index(i)];
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

void fft3d_accumulate_real(const range<int>& this_box, const vector<float>& data) {
	vector<hpx::future<void>> futs;
	if (real_mybox.contains(this_box)) {
		array<int, NDIM> i;
		for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
			for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
				const auto& box = real_mybox;
				const int mtxindex = (i[0] - box.begin[0]) * (box.end[1] - box.begin[1]) + (i[1] - box.begin[1]);
				std::lock_guard<spinlock_type> lock(*mutexes[mtxindex]);
				for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
					R[box.index(i)] += data[this_box.index(i)];
				}
			}
		}
	} else {
		for (int bi = 0; bi < real_boxes.size(); bi++) {
			array<int, NDIM> si;
			for (si[0] = -N; si[0] <= +N; si[0] += N) {
				for (si[1] = -N; si[1] <= +N; si[1] += N) {
					for (si[2] = -N; si[2] <= +N; si[2] += N) {
						const auto shifted_box = this_box.shift(si);
						const auto inter = real_boxes[bi].intersection(shifted_box);
						if (!inter.empty()) {
							vector<range<int>> inters;
							split_box(inter, inters);
							for (auto this_inter : inters) {
								vector<float> this_data;
								this_data.resize(this_inter.volume());
								array<int, NDIM> i;
								for (i[0] = this_inter.begin[0]; i[0] != this_inter.end[0]; i[0]++) {
									for (i[1] = this_inter.begin[1]; i[1] != this_inter.end[1]; i[1]++) {
										for (i[2] = this_inter.begin[2]; i[2] != this_inter.end[2]; i[2]++) {
											const int k = this_inter.index(i);
											auto j = i;
											for (int dim = 0; dim < NDIM; dim++) {
												j[dim] -= si[dim];
											}
											const int l = this_box.index(j);
											assert(k < this_data.size());
											assert(l < data.size());
											this_data[k] = data[l];
										}
									}
								}
								auto fut = hpx::async<fft3d_accumulate_real_action>(hpx_localities()[bi], this_inter, std::move(this_data));
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

void fft3d_accumulate_complex(const range<int>& this_box, const vector<cmplx>& data) {
	vector<hpx::future<void>> futs;
	const auto& box = cmplx_mybox[ZDIM];
	if (box.contains(this_box)) {
		array<int, NDIM> i;
		for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
			for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
				const int mtxindex = (i[0] - box.begin[0]) * (box.end[1] - box.begin[1]) + (i[1] - box.begin[1]);
				std::lock_guard<spinlock_type> lock(*mutexes[mtxindex]);
				for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
					Y[box.index(i)] += data[this_box.index(i)];
				}
			}
		}
	} else {
		for (int bi = 0; bi < cmplx_boxes[ZDIM].size(); bi++) {
			array<int, NDIM> si;
			for (si[0] = -N; si[0] <= +N; si[0] += N) {
				for (si[1] = -N; si[1] <= +N; si[1] += N) {
					for (si[2] = -N; si[2] <= +N; si[2] += N) {
						const auto shifted_box = this_box.shift(si);
						const auto inter = cmplx_boxes[ZDIM][bi].intersection(shifted_box);
						if (!inter.empty()) {
							vector<range<int>> inters;
							split_box(inter, inters);
							for (auto this_inter : inters) {
								vector<cmplx> this_data;
								this_data.resize(this_inter.volume());
								array<int, NDIM> i;
								for (i[0] = this_inter.begin[0]; i[0] != this_inter.end[0]; i[0]++) {
									for (i[1] = this_inter.begin[1]; i[1] != this_inter.end[1]; i[1]++) {
										for (i[2] = this_inter.begin[2]; i[2] != this_inter.end[2]; i[2]++) {
											auto j = i;
											for (int dim = 0; dim < NDIM; dim++) {
												j[dim] -= si[dim];
											}
											this_data[this_inter.index(i)] = data[this_box.index(j)];
										}
									}
								}
								auto fut = hpx::async<fft3d_accumulate_complex_action>(hpx_localities()[bi], this_inter, std::move(this_data));
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

void fft3d_init(int N_) {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<fft3d_init_action>(c, N_));
	}
	N = N_;
	range<int> box;
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = 0;
		box.end[dim] = N;
	}
	real_boxes.resize(hpx_size());
	find_boxes(box, real_boxes, 0, hpx_size());
	real_mybox = real_boxes[hpx_rank()];
	R.resize(real_mybox.volume(), 0.0);
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
	Y.resize(cmplx_mybox[ZDIM].volume(), cmplx(0.0, 0.0));
	mutexes.resize(N * N);
	for (int i = 0; i < N * N; i++) {
		mutexes[i] = std::make_shared<spinlock_type>();
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void fft3d_destroy() {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<fft3d_destroy_action>(c));
	}
	R = decltype(R)();
	Y = decltype(Y)();
	Y1 = decltype(Y1)();
	ym = decltype(ym)();

	hpx::wait_all(futs.begin(), futs.end());

}

static void update() {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<update_action>(c));
	}
	Y = std::move(Y1);
	hpx::wait_all(futs.begin(), futs.end());
}

static void fft3d_phase1() {
	static mutex_type mtx;
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<fft3d_phase1_action>(c));
	}
	array<int, NDIM> i;
	Y.resize(cmplx_mybox[ZDIM].volume());
	for (i[0] = real_mybox.begin[0]; i[0] != real_mybox.end[0]; i[0]++) {
		futs.push_back(hpx::async([](array<int,NDIM> j) {
			fftwf_plan p;
			fftwf_complex out[N / 2 + 1];
			float in[N];
			std::unique_lock<mutex_type> lock(mtx);
			p = fftwf_plan_dft_r2c_1d(N, in, out, 0);
			lock.unlock();
			auto i = j;
			for (i[1] = real_mybox.begin[1]; i[1] != real_mybox.end[1]; i[1]++) {
				for (i[2] = 0; i[2] != N; i[2]++) {
					in[i[2]] = R[real_mybox.index(i)];
				}
				fftwf_execute(p);
				for (i[2] = 0; i[2] < N / 2 + 1; i[2]++) {
					const int l = cmplx_mybox[ZDIM].index(i);
					Y[l].real() = out[i[2]][0];
					Y[l].imag() = out[i[2]][1];
				}
			}
			lock.lock();
			fftwf_destroy_plan(p);
			lock.unlock();
		}, i));
	}
	hpx::wait_all(futs.begin(), futs.end());
	R = decltype(R)();
}

static void fft3d_phase2(int dim, bool inv) {
	static mutex_type mtx;
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<fft3d_phase2_action>(c, dim, inv));
	}
	Y = std::move(Y1);
	array<int, NDIM> i;
	const float norm = inv ? 1.0f / N : 1.0;
	for (i[0] = cmplx_mybox[dim].begin[0]; i[0] != cmplx_mybox[dim].end[0]; i[0]++) {
		futs.push_back(hpx::async([dim,inv,norm](array<int,NDIM> j) {
			fftwf_complex out[N];
			fftwf_complex in[N];
			fftwf_plan p;
			std::unique_lock<mutex_type> lock(mtx);
			p = fftwf_plan_dft_1d(N, in, out, inv ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);
			lock.unlock();
			auto i = j;
			for (i[1] = cmplx_mybox[dim].begin[1]; i[1] != cmplx_mybox[dim].end[1]; i[1]++) {
				assert(cmplx_mybox[dim].begin[2]==0);
				assert(cmplx_mybox[dim].end[2]==N);
				for (i[2] = 0; i[2] < N; i[2]++) {
					const auto l = cmplx_mybox[dim].index(i);
					assert(l >= 0 );
					assert( l < Y.size());
					in[i[2]][0] = Y[l].real();
					in[i[2]][1] = Y[l].imag();
				}
				fftwf_execute(p);
				for (i[2] = 0; i[2] < N; i[2]++) {
					const int l = cmplx_mybox[dim].index(i);
					Y[l].real() = out[i[2]][0] * norm;
					Y[l].imag() = out[i[2]][1] * norm;
				}
			}
			lock.lock();
			fftwf_destroy_plan(p);
			lock.unlock();
		}, i));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

static void fft3d_phase3() {
	static mutex_type mtx;
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<fft3d_phase3_action>(c));
	}
	array<int, NDIM> i;
	Y = std::move(Y1);
	R.resize(real_mybox.volume());
	const float Ninv = 1.0 / N;
	for (i[0] = cmplx_mybox[ZDIM].begin[0]; i[0] != cmplx_mybox[ZDIM].end[0]; i[0]++) {
		futs.push_back(hpx::async([Ninv](array<int,NDIM> j) {
			fftwf_plan p;
			fftwf_complex in[N / 2 + 1];
			float out[N];
			std::unique_lock<mutex_type> lock(mtx);
			p = fftwf_plan_dft_c2r_1d(N, in, out, 0);
			lock.unlock();
			auto i = j;
			for (i[1] = cmplx_mybox[ZDIM].begin[1]; i[1] !=cmplx_mybox[ZDIM].end[1]; i[1]++) {
				for (i[2] = 0; i[2] < N / 2 + 1; i[2]++) {
					const int l = cmplx_mybox[ZDIM].index(i);
					in[i[2]][0] = Y[l].real();
					in[i[2]][1] = Y[l].imag();
				}
				fftwf_execute(p);
				for (i[2] = 0; i[2] != N; i[2]++) {
					R[real_mybox.index(i)] = out[i[2]] * Ninv;
				}
			}
			lock.lock();
			fftwf_destroy_plan(p);
			lock.unlock();
		}, i));
	}
	hpx::wait_all(futs.begin(), futs.end());
	Y = decltype(Y)();
}

static void find_boxes(range<int> box, vector<range<int>>& boxes, int begin, int end) {
	if (end - begin == 1) {
		boxes[begin] = box;
	} else {
		const int xdim = (box.end[0] - box.begin[0] > 1) ? 0 : 1;
		auto left = box;
		auto right = box;
		const int mid = (begin + end) / 2;
		const float w = float(mid - begin) / float(end - begin);
		left.end[xdim] = right.begin[xdim] = (((1.0 - w) * box.begin[xdim] + w * box.end[xdim]) + 0.5);
		find_boxes(left, boxes, begin, mid);
		find_boxes(right, boxes, mid, end);
	}
}

static void transpose(int dim1, int dim2) {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<transpose_action>(c, dim1, dim2));
	}
	range<int> tbox = cmplx_mybox[dim1].transpose(dim1, dim2);
	Y1.resize(cmplx_mybox[dim1].volume());
	for (int bi = 0; bi < cmplx_boxes[dim2].size(); bi++) {
		const auto tinter = cmplx_boxes[dim2][bi].intersection(tbox);
		vector<float> data;
		if (!tinter.empty()) {
			vector<range<int>> tinters;
			split_box(tinter, tinters);
			for (auto this_tinter : tinters) {
				auto inter = this_tinter.transpose(dim1, dim2);
				auto fut = hpx::async<transpose_read_action>(hpx_localities()[bi], this_tinter, dim1, dim2);
				futs.push_back(fut.then([inter,dim1](hpx::future<vector<cmplx>> fut) {
					auto data = fut.get();
					array<int, NDIM> i;
					for (i[0] = inter.begin[0]; i[0] != inter.end[0]; i[0]++) {
						for (i[1] = inter.begin[1]; i[1] != inter.end[1]; i[1]++) {
							for (i[2] = inter.begin[2]; i[2] != inter.end[2]; i[2]++) {
								const int k = inter.index(i);
								const int l = cmplx_mybox[dim1].index(i);
								assert(k < data.size());
								assert(l < Y1.size());
								Y1[l] = data[k];
							}
						}
					}
				}));
			}
		}
	}

	hpx::wait_all(futs.begin(), futs.end());

}

static vector<cmplx> transpose_read(const range<int>& this_box, int dim1, int dim2) {
	vector<cmplx> data(this_box.volume());
	assert(cmplx_mybox[dim2].contains(this_box));
	auto tbox = this_box.transpose(dim1, dim2);
	array<int, NDIM> i;
	for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
		for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
			for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
				auto j = i;
				std::swap(j[dim1], j[dim2]);
				data[tbox.index(j)] = Y[cmplx_mybox[dim2].index(i)];
			}
		}
	}
	return std::move(data);
}

static void shift(bool inv) {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<shift_action>(c, inv));
	}
	const int dim2 = inv ? YDIM : XDIM;
	const int dim1 = inv ? XDIM : YDIM;
	range<int> tbox = inv ? cmplx_mybox[dim2].shift_up() : cmplx_mybox[dim2].shift_down();
	Y1.resize(cmplx_mybox[dim2].volume());
	for (int bi = 0; bi < cmplx_boxes[dim1].size(); bi++) {
		const auto tinter = cmplx_boxes[dim1][bi].intersection(tbox);
		vector<float> data;
		if (!tinter.empty()) {
			vector<range<int>> tinters;
			split_box(tinter, tinters);
			for (auto this_tinter : tinters) {
				auto inter = inv ? this_tinter.shift_down() : this_tinter.shift_up();
				auto fut = hpx::async<shift_read_action>(hpx_localities()[bi], this_tinter, inv);
				futs.push_back(fut.then([inter,dim2](hpx::future<vector<cmplx>> fut) {
					auto data = fut.get();
					array<int, NDIM> i;
					for (i[0] = inter.begin[0]; i[0] != inter.end[0]; i[0]++) {
						for (i[1] = inter.begin[1]; i[1] != inter.end[1]; i[1]++) {
							for (i[2] = inter.begin[2]; i[2] != inter.end[2]; i[2]++) {
								const int k = inter.index(i);
								const int l = cmplx_mybox[dim2].index(i);
								assert(k < data.size());
								assert(l < Y1.size());
								Y1[l] = data[k];
							}
						}
					}
				}));
			}
		}
	}

	hpx::wait_all(futs.begin(), futs.end());

}

static vector<cmplx> shift_read(const range<int>& this_box, bool inv) {
	vector<cmplx> data(this_box.volume());
	auto tbox = inv ? this_box.shift_down() : this_box.shift_up();
	array<int, NDIM> i;
	const int dim = inv ? XDIM : YDIM;
	for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
		for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
			for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
				auto j = inv ? shift_down(i) : shift_up(i);
				data[tbox.index(j)] = Y[cmplx_mybox[dim].index(i)];
			}
		}
	}
	return std::move(data);
}

static void split_box(range<int> box, vector<range<int>>& real_boxes) {
	if (box.volume() < MAX_BOX_SIZE) {
		real_boxes.push_back(box);
	} else {
		auto children = box.split();
		split_box(children.first, real_boxes);
		split_box(children.second, real_boxes);
	}
}
