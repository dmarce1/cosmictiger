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
#include <cosmictiger/fft.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/constants.hpp>

#include <fftw3.h>
#include <silo.h>

#define MAX_BOX_SIZE (8*1024*1024)

static int64_t N;
static vector<range<int64_t>> real_boxes;
static array<vector<range<int64_t>>, NDIM> cmplx_boxes;
static range<int64_t> real_mybox;
static array<range<int64_t>, NDIM> cmplx_mybox;
static vector<std::shared_ptr<spinlock_type>> mutexes;

static vector<float> R;
static vector<cmplx> Y;
static vector<cmplx> ym;
static vector<cmplx> Y1;

static void find_boxes(range<int64_t> box, vector<range<int64_t>>& boxes, int begin, int end);
static void split_box(range<int64_t> box, vector<range<int64_t>>& real_boxes);
static void transpose(int, int);
static void shift(bool);
static void update();
static void fft3d_phase1();
static void fft3d_phase2(int, bool);
static void fft3d_phase3();
static void finish_force_real();
static vector<cmplx> transpose_read(const range<int64_t>&, int dim1, int dim2);
static vector<cmplx> shift_read(const range<int64_t>&, bool);
struct pspecret {
	vector<float> k;
	vector<float> p;
	vector<double> c;
	vector<float> pdev;
};
static pspecret power_spectrum_compute(const vector<float>& kbins);
static pspecret power_spectrum_compute2(pspecret, const vector<float>& kbins);

HPX_PLAIN_ACTION (fft3d_accumulate_real);
HPX_PLAIN_ACTION (fft3d_accumulate_complex);
HPX_PLAIN_ACTION (fft3d_init);
HPX_PLAIN_ACTION (fft3d_execute);
HPX_PLAIN_ACTION (fft3d_phase1);
HPX_PLAIN_ACTION (fft3d_phase2);
HPX_PLAIN_ACTION (fft3d_phase3);
HPX_PLAIN_ACTION (fft3d_read_real);
HPX_PLAIN_ACTION (fft3d_read_complex);
HPX_PLAIN_ACTION (fft3d_force_real);
HPX_PLAIN_ACTION (fft3d_destroy);
HPX_PLAIN_ACTION (transpose_read);
HPX_PLAIN_ACTION (transpose);
HPX_PLAIN_ACTION (shift);
HPX_PLAIN_ACTION (shift_read);
HPX_PLAIN_ACTION (update);
HPX_PLAIN_ACTION (finish_force_real);
HPX_PLAIN_ACTION (power_spectrum_compute);
HPX_PLAIN_ACTION (power_spectrum_compute2);

void fft3d2silo(bool real) {
	if (real) {
		DBfile *db = DBCreateReal("real.silo", DB_CLOBBER, DB_LOCAL, "", DB_PDB);
		const char* coordnames[] = { "x", "y", "z" };
		vector<float> x;
		vector<float> y;
		vector<float> z;
		auto box = real_mybox;
		for (int i = 0; i < N; i++) {
			x.push_back(i);
		}
		for (int i = 0; i < N; i++) {
			y.push_back(i);
		}
		for (int i = 0; i < N; i++) {
			z.push_back(i);
		}
		float* coords[NDIM] = { x.data(), y.data(), z.data() };
		int dims[NDIM] = { (int) N, (int) N, (int) N };
		DBPutQuadmesh(db, "real_mesh", coordnames, coords, dims, NDIM, DB_FLOAT, DB_COLLINEAR, NULL);
		DBPutQuadvar1(db, "real_data", "real_mesh", R.data(), dims, NDIM, NULL, 0, DB_FLOAT, DB_NODECENT, NULL);
		DBClose(db);
	} else {
		DBfile *db = DBCreateReal("complex.silo", DB_CLOBBER, DB_LOCAL, "", DB_PDB);
		const char* coordnames[] = { "z", "y", "x" };
		vector<float> x;
		vector<float> y;
		vector<float> z;
		auto box = cmplx_mybox[ZDIM];
		for (int i = 0; i < N; i++) {
			x.push_back(i);
		}
		for (int i = 0; i < N; i++) {
			y.push_back(i);
		}
		for (int i = 0; i < N / 2 + 1; i++) {
			z.push_back(i);
		}
		float* coords[NDIM] = { z.data(), y.data(), x.data() };
		int dims[NDIM] = { (int) N / 2 + 1, (int) N, (int) N };
		DBPutQuadmesh(db, "complex_mesh", coordnames, coords, dims, NDIM, DB_FLOAT, DB_COLLINEAR, NULL);
		vector<float> i;
		vector<float> r;
		for (int j = 0; j < box.volume(); j++) {
			i.push_back(Y[j].imag());
			r.push_back(Y[j].real());
		}
		DBPutQuadvar1(db, "complex_real", "complex_mesh", r.data(), dims, NDIM, NULL, 0, DB_FLOAT, DB_NODECENT, NULL);
		DBPutQuadvar1(db, "complex_imag", "complex_mesh", i.data(), dims, NDIM, NULL, 0, DB_FLOAT, DB_NODECENT, NULL);
		DBClose(db);
	}

}

void fft3d_execute() {
//	PRINT("FFT z\n");
	fft3d_phase1();
//	PRINT("Transpose y-z\n");
	transpose(1, 2);
//	PRINT("FFT y\n");
	fft3d_phase2(1, false);
//	PRINT("Shifting\n");
	shift(false);
//	PRINT("FFT x\n");
	fft3d_phase2(0, false);
//	PRINT("Transpose z-x\n");
	transpose(2, 0);
	update();
//	PRINT("done\n");

}

void fft3d_inv_execute() {
//	PRINT("Transpose z-x\n");
	transpose(0, 2);
//	PRINT("inv FFT x\n");
	fft3d_phase2(0, true);
//	PRINT("Shifting\n");
	shift(true);
//	PRINT("inv FFT y\n");
	fft3d_phase2(1, true);
//	PRINT("Transpose y-z\n");
	transpose(2, 1);
//	PRINT("inv FFT z\n");
	fft3d_phase3();
//	PRINT("done\n");

}

power_spectrum_t fft3d_power_spectrum() {
	vector<float> kbins;
	const double box_size = get_options().code_to_cm / constants::mpc_to_cm;
	const int Nfour = get_options().Nfour;
	const double Ns = std::max(Nfour / 8.0, 32.0);
	const double kmin = 0.999999;
	const double kmax = Nfour / 2.0;
	const double logkmin = log(kmin);
	const double logkmax = log(kmax);
	const double dlogk = (logkmax - logkmin) / Ns;
	for (int i = 0; i <= Ns; i++) {
		double logk = logkmin + dlogk * i;
		kbins.push_back(exp(logk));
	}
	auto pr = power_spectrum_compute(kbins);
	const auto& count = pr.c;
	auto& power = pr.p;
	auto& ks = pr.k;
	power_spectrum_t ret;
	ret.P.resize(kbins.size() - 1);
	ret.k.resize(kbins.size() - 1);
	ret.Perr.resize(kbins.size() - 1, 0.0);
	for (int i = 0; i < power.size(); i++) {
		if (count[i] > 0) {
			power[i] /= count[i];
			ks[i] /= count[i];
		}
		ks[i] = sqrt(kbins[i] * kbins[i + 1]);
		ks[i] *= 2.0 * M_PI / box_size;
		ret.P[i] = power[i];
		ret.k[i] = ks[i];
	}
	pr = power_spectrum_compute2(pr, kbins);
	for (int i = 0; i < power.size(); i++) {
		if (pr.c[i] > 0) {
			if (count[i] > 1) {
				ret.Perr[i] = sqrt(pr.pdev[i]) / sqrt(count[i] - 1) / sqrt(count[i]);
			} else {
				ret.P[i] = 0.0;
				ret.Perr[i] = 0.0;
			}
		}
	}

	return ret;
}

int search_kbins(vector<float> kbins, double k, int binmin, int binmax) {
	if (k < 0.0 || k >= kbins.back()) {
		return -1;
	} else {
		if (binmax - binmin == 1) {
			ALWAYS_ASSERT(k >= kbins[binmin]);
			ALWAYS_ASSERT(k < kbins[binmin + 1]);
			return binmin;
		} else {
			int binmid = (binmax + binmin) / 2;
			if (k >= kbins[binmid]) {
				return search_kbins(kbins, k, binmid, binmax);
			} else {
				return search_kbins(kbins, k, binmin, binmid);
			}
		}
	}
	return -1;
}
;

static pspecret power_spectrum_compute(const vector<float>& kbins) {
	vector<hpx::future<pspecret>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<power_spectrum_compute_action>(c, kbins));
	}
	const int64_t N = get_options().parts_dim;
	const float box_size = get_options().code_to_cm / constants::mpc_to_cm;
	const auto& box = cmplx_mybox[ZDIM];
	array<int64_t, NDIM> I;

	const int nbins = kbins.size() - 1;
	vector<float> power(nbins, 0.0);
	vector<double> count(nbins, 0);
	vector<float> ktot(nbins, 0.0);
	for (I[0] = box.begin[0]; I[0] != box.end[0]; I[0]++) {
		futs.push_back(hpx::async([N,nbins,&kbins,box]( array<int64_t, NDIM> I) {
			const int64_t i = I[0] < N / 2 ? I[0] : I[0] - N;
			const double sx = cloud_filter(2.0 * M_PI * i / N);
			pspecret ret;
			vector<float> power(nbins, 0.0);
			vector<double> count(nbins, 0);
			vector<float> ktot(nbins, 0.0);
			for (I[1] = box.begin[1]; I[1] != box.end[1]; I[1]++) {
				const int64_t j = I[1] < N / 2 ? I[1] : I[1] - N;
				const double sy = cloud_filter(2.0 * M_PI * j / N);
				for (I[2] = box.begin[2]; I[2] != box.end[2]; I[2]++) {
					const int64_t k = I[2] < N / 2 ? I[2] : I[2] - N;
					const double sz = cloud_filter(2.0 * M_PI * k / N);
					const double c0 = sx * sy * sz;
					Y[box.index(I)] *= c0;
					const double iii = std::sqrt(i * i + j * j + k * k);
					if( iii > 0 ) {
						const int bin = search_kbins(kbins, iii, 0, nbins);
						if (bin >= 0 ) {
							const float pwr = Y[box.index(I)].norm();
							count[bin] += 1;
							power[bin] += pwr;
							ktot[bin] += iii;
						}
					}
				}
			}
			ret.p = std::move(power);
			ret.c = std::move(count);
			ret.k = std::move(ktot);
			return ret;
		}, I));
	}
	for (auto& f : futs) {
		const auto pr = f.get();
		const auto& this_count = pr.c;
		const auto& this_power = pr.p;
		const auto& this_ktot = pr.k;
		for (int i = 0; i < this_count.size(); i++) {
			count[i] += this_count[i];
			power[i] += this_power[i];
			ktot[i] += this_ktot[i];
		}
	}
	pspecret rc;
	rc.p = std::move(power);
	rc.c = std::move(count);
	rc.k = std::move(ktot);
	return rc;
}

static pspecret power_spectrum_compute2(pspecret rc, const vector<float>& kbins) {
	vector<hpx::future<pspecret>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<power_spectrum_compute2_action>(c, rc, kbins));
	}
	const int64_t N = get_options().parts_dim;
	const float box_size = get_options().code_to_cm / constants::mpc_to_cm;
	const auto& box = cmplx_mybox[ZDIM];
	array<int64_t, NDIM> I;

	const int nbins = kbins.size() - 1;
	vector<float> devs(nbins, 0.0);
	for (I[0] = box.begin[0]; I[0] != box.end[0]; I[0]++) {
		futs.push_back(hpx::async([N,nbins,&kbins,box,rc]( array<int64_t, NDIM> I) {
			const int64_t i = I[0] < N / 2 ? I[0] : I[0] - N;
			const double sx = sinc(M_PI * i / N);
			pspecret ret;
			vector<float> pdev(nbins, 0.0);
			for (I[1] = box.begin[1]; I[1] != box.end[1]; I[1]++) {
				const int64_t j = I[1] < N / 2 ? I[1] : I[1] - N;
				for (I[2] = box.begin[2]; I[2] != box.end[2]; I[2]++) {
					const int64_t k = I[2] < N / 2 ? I[2] : I[2] - N;
					const double iii = std::sqrt(i * i + j * j + k * k);
					if( iii > 0 ) {
						const int bin = search_kbins(kbins, iii, 0, nbins);
						if (bin >= 0 ) {
							const float pwr = Y[box.index(I)].norm();
							pdev[bin] += sqr(pwr - rc.p[bin]);
						}
					}
				}
			}
			ret.pdev = std::move(pdev);
			return ret;
		}, I));
	}
	for (auto& f : futs) {
		const auto pr = f.get();
		const auto& this_pdev = pr.pdev;
		for (int i = 0; i < nbins; i++) {
			devs[i] += this_pdev[i];
		}
	}
	rc.pdev = std::move(devs);
	return rc;
}

void fft3d_force_real() {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<fft3d_force_real_action>(c));
	}
	const auto& box = cmplx_mybox[ZDIM];
	array<int64_t, NDIM> i;
	range<int64_t> mirror_box = box;
	range<int64_t> slim_box = box;
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

range<int64_t> fft3d_complex_range() {
	return cmplx_mybox[ZDIM];
}

range<int64_t> fft3d_real_range() {
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
	array<int64_t, NDIM> i;
	range<int64_t> slim_box = box;
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

vector<float> fft3d_read_real(const range<int64_t>& this_box) {
	vector<hpx::future<void>> futs;
	vector<float> data(this_box.volume());
	if (real_mybox.contains(this_box)) {
		array<int64_t, NDIM> i;
		for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
			for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
				for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
					data[this_box.index(i)] = R[real_mybox.index(i)];
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
								auto fut = hpx::async<fft3d_read_real_action>(hpx_localities()[ri], this_inter);
								futs.push_back(fut.then([si,this_box,this_inter,&data](hpx::future<vector<float>> fut) {
									auto this_data = fut.get();
									array<int64_t,NDIM> i;
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

vector<cmplx> fft3d_read_complex(const range<int64_t>& this_box) {
	vector<hpx::future<void>> futs;
	vector<cmplx> data(this_box.volume());
	const auto mybox = cmplx_mybox[ZDIM];
	if (mybox.contains(this_box)) {
		array<int64_t, NDIM> i;
		for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
			for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
				for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
					data[this_box.index(i)] = Y[mybox.index(i)];
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
								auto fut = hpx::async<fft3d_read_complex_action>(hpx_localities()[ri], this_inter);
								futs.push_back(fut.then([si,this_box,this_inter,&data](hpx::future<vector<cmplx>> fut) {
									auto this_data = fut.get();
									array<int64_t,NDIM> i;
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

void fft3d_accumulate_real(const range<int64_t>& this_box, const vector<float>& data) {
	vector<hpx::future<void>> futs;
	if (real_mybox.contains(this_box)) {
		array<int64_t, NDIM> i;
		for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
			for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
				const auto& box = real_mybox;
				const int64_t mtxindex = (i[0] - box.begin[0]) * (box.end[1] - box.begin[1]) + (i[1] - box.begin[1]);
				std::lock_guard<spinlock_type> lock(*mutexes[mtxindex]);
				for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
					R[box.index(i)] += data[this_box.index(i)];
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
								vector<float> this_data;
								this_data.resize(this_inter.volume());
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

void fft3d_accumulate_complex(const range<int64_t>& this_box, const vector<cmplx>& data) {
	vector<hpx::future<void>> futs;
	const auto& box = cmplx_mybox[ZDIM];
	if (box.contains(this_box)) {
		array<int64_t, NDIM> i;
		for (i[0] = this_box.begin[0]; i[0] != this_box.end[0]; i[0]++) {
			for (i[1] = this_box.begin[1]; i[1] != this_box.end[1]; i[1]++) {
				const int64_t mtxindex = (i[0] - box.begin[0]) * (box.end[1] - box.begin[1]) + (i[1] - box.begin[1]);
				std::lock_guard<spinlock_type> lock(*mutexes[mtxindex]);
				for (i[2] = this_box.begin[2]; i[2] != this_box.end[2]; i[2]++) {
					Y[box.index(i)] += data[this_box.index(i)];
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
								vector<cmplx> this_data;
								this_data.resize(this_inter.volume());
								array<int64_t, NDIM> i;
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

void fft3d_init(int64_t N_, float init_const) {
	fft3d_destroy();
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<fft3d_init_action>(c, N_, init_const));
	}
	N = N_;
	range<int64_t> box;
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = 0;
		box.end[dim] = N;
	}
	real_boxes.resize(hpx_size());
	find_boxes(box, real_boxes, 0, hpx_size());
	real_mybox = real_boxes[hpx_rank()];
	R.resize(real_mybox.volume(), init_const);
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
	for (int64_t i = 0; i < N * N; i++) {
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
	array<int64_t, NDIM> i;
	Y.resize(cmplx_mybox[ZDIM].volume());
	for (i[0] = real_mybox.begin[0]; i[0] != real_mybox.end[0]; i[0]++) {
		futs.push_back(hpx::async([](array<int64_t,NDIM> j) {
			fftwf_plan p;
			fftwf_complex out[N / 2 + 1];
			float in[N];
			std::unique_lock<mutex_type> lock(mtx);
			p = fftwf_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE );
			lock.unlock();
			auto i = j;
			for (i[1] = real_mybox.begin[1]; i[1] != real_mybox.end[1]; i[1]++) {
				for (i[2] = 0; i[2] != N; i[2]++) {
					in[i[2]] = R[real_mybox.index(i)];
				}
				fftwf_execute(p);
				for (i[2] = 0; i[2] < N / 2 + 1; i[2]++) {
					const int64_t l = cmplx_mybox[ZDIM].index(i);
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
	array<int64_t, NDIM> i;
	const float norm = inv ? 1.0f / N : 1.0;
	for (i[0] = cmplx_mybox[dim].begin[0]; i[0] != cmplx_mybox[dim].end[0]; i[0]++) {
		futs.push_back(hpx::async([dim,inv,norm](array<int64_t,NDIM> j) {
			fftwf_complex out[N];
			fftwf_complex in[N];
			fftwf_plan p;
			std::unique_lock<mutex_type> lock(mtx);
			p = fftwf_plan_dft_1d(N, in, out, inv ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);
			lock.unlock();
			auto i = j;
			for (i[1] = cmplx_mybox[dim].begin[1]; i[1] != cmplx_mybox[dim].end[1]; i[1]++) {
				ASSERT(cmplx_mybox[dim].begin[2]==0);
				ASSERT(cmplx_mybox[dim].end[2]==N);
				for (i[2] = 0; i[2] < N; i[2]++) {
					const auto l = cmplx_mybox[dim].index(i);
					ASSERT(l >= 0 );
					ASSERT( l < Y.size());
					in[i[2]][0] = Y[l].real();
					in[i[2]][1] = Y[l].imag();
				}
				fftwf_execute(p);
				for (i[2] = 0; i[2] < N; i[2]++) {
					const int64_t l = cmplx_mybox[dim].index(i);
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
	array<int64_t, NDIM> i;
	Y = std::move(Y1);
	R.resize(real_mybox.volume());
	const float Ninv = 1.0 / N;
	for (i[0] = cmplx_mybox[ZDIM].begin[0]; i[0] != cmplx_mybox[ZDIM].end[0]; i[0]++) {
		futs.push_back(hpx::async([Ninv](array<int64_t,NDIM> j) {
			fftwf_plan p;
			fftwf_complex in[N / 2 + 1];
			float out[N];
			std::unique_lock<mutex_type> lock(mtx);
			p = fftwf_plan_dft_c2r_1d(N, in, out, FFTW_ESTIMATE );
			lock.unlock();
			auto i = j;
			for (i[1] = cmplx_mybox[ZDIM].begin[1]; i[1] !=cmplx_mybox[ZDIM].end[1]; i[1]++) {
				for (i[2] = 0; i[2] < N / 2 + 1; i[2]++) {
					const int64_t l = cmplx_mybox[ZDIM].index(i);
					in[i[2]][0] = Y[l].real();
					in[i[2]][1] = Y[l].imag();
					/*					if( i[2] == 0 || i[2] == N / 2) {
					 const float value = Y[l].imag() / (Y[l].abs()+1e-20);
					 if(std::abs(value) > 5.0e-3) {
					 PRINT( "!!!! %i %i %i %e %e\n", i[0], i[1], i[2], Y[l].real(), Y[l].imag());
					 }
					 }*/
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

static void find_boxes(range<int64_t> box, vector<range<int64_t>>& boxes, int begin, int end) {
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
	range<int64_t> tbox = cmplx_mybox[dim1].transpose(dim1, dim2);
	Y1.resize(cmplx_mybox[dim1].volume());
	for (int64_t bi = 0; bi < cmplx_boxes[dim2].size(); bi++) {
		const auto tinter = cmplx_boxes[dim2][bi].intersection(tbox);
		vector<float> data;
		if (!tinter.empty()) {
			vector < range < int64_t >> tinters;
			split_box(tinter, tinters);
			for (auto this_tinter : tinters) {
				auto inter = this_tinter.transpose(dim1, dim2);
				auto fut = hpx::async<transpose_read_action>(hpx_localities()[bi], this_tinter, dim1, dim2);
				futs.push_back(fut.then([inter,dim1](hpx::future<vector<cmplx>> fut) {
					auto data = fut.get();
					array<int64_t,NDIM> i;
					for (i[0] = inter.begin[0]; i[0] != inter.end[0]; i[0]++) {
						for (i[1] = inter.begin[1]; i[1] != inter.end[1]; i[1]++) {
							for (i[2] = inter.begin[2]; i[2] != inter.end[2]; i[2]++) {
								const int k = inter.index(i);
								const int64_t l = cmplx_mybox[dim1].index(i);
								ASSERT(k < data.size());
								ASSERT(l < Y1.size());
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

static vector<cmplx> transpose_read(const range<int64_t>& this_box, int dim1, int dim2) {
	vector<cmplx> data(this_box.volume());
	ASSERT(cmplx_mybox[dim2].contains(this_box));
	auto tbox = this_box.transpose(dim1, dim2);
	array<int64_t, NDIM> i;
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
	range<int64_t> tbox = inv ? cmplx_mybox[dim2].shift_up() : cmplx_mybox[dim2].shift_down();
	Y1.resize(cmplx_mybox[dim2].volume());
	for (int64_t bi = 0; bi < cmplx_boxes[dim1].size(); bi++) {
		const auto tinter = cmplx_boxes[dim1][bi].intersection(tbox);
		vector<float> data;
		if (!tinter.empty()) {
			vector < range < int64_t >> tinters;
			split_box(tinter, tinters);
			for (auto this_tinter : tinters) {
				auto inter = inv ? this_tinter.shift_down() : this_tinter.shift_up();
				auto fut = hpx::async<shift_read_action>(hpx_localities()[bi], this_tinter, inv);
				futs.push_back(fut.then([inter,dim2](hpx::future<vector<cmplx>> fut) {
					auto data = fut.get();
					array<int64_t,NDIM> i;
					for (i[0] = inter.begin[0]; i[0] != inter.end[0]; i[0]++) {
						for (i[1] = inter.begin[1]; i[1] != inter.end[1]; i[1]++) {
							for (i[2] = inter.begin[2]; i[2] != inter.end[2]; i[2]++) {
								const int64_t k = inter.index(i);
								const int64_t l = cmplx_mybox[dim2].index(i);
								ASSERT(k < data.size());
								ASSERT(l < Y1.size());
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

static vector<cmplx> shift_read(const range<int64_t>& this_box, bool inv) {
	vector<cmplx> data(this_box.volume());
	auto tbox = inv ? this_box.shift_down() : this_box.shift_up();
	array<int64_t, NDIM> i;
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

static void split_box(range<int64_t> box, vector<range<int64_t>>& real_boxes) {
	if (box.volume() < MAX_BOX_SIZE) {
		real_boxes.push_back(box);
	} else {
		auto children = box.split();
		split_box(children.first, real_boxes);
		split_box(children.second, real_boxes);
	}
}
