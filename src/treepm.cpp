#include <cosmictiger/defs.hpp>

#include <cosmictiger/treepm.hpp>
#include <cosmictiger/domain.hpp>
#include <cosmictiger/fft_dbl.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/power.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/kernels.hpp>

static vector<complex<float>> Y0;
static device_vector<pair<part_int>> chain_mesh;
static vector<int> tree_roots;
static size_t local_part_count;

#ifdef FMMPM
tensor_trless_sym<vector<complex<double>>, PM_ORDER> D_k;
tensor_trless_sym<vector<double>, (PM_ORDER - 1)> M_x;
tensor_trless_sym<vector<complex<double>>, (PM_ORDER - 1)> M_k;
tensor_trless_sym<vector<double>, PM_ORDER> L_x;
#endif

#define FILTER_N 65536
#define FILTER_MAX (M_PI * 64.0)

std::function<double(double)> treepm_init_filter() {
	constexpr double M = 1025;
	const double dk = FILTER_MAX / (FILTER_N - 1.0);
	const auto I1 = [](double r, double k) {
		return green_rho(r) * 4.0 * M_PI * sqr(r) * sinc(r * k);
	};
	const auto I2 = [](double r, double k) {
		if( k==0.0 ) {
			return 0.0;
		} else {
			return green_rho(r) * 4.0 * M_PI * sqr(r) * (cos(r*k) - sinc(r*k)) / k;
		}
	};
	vector<double> filter1;
	vector<double> filter2;
	for (int n = 0; n <= FILTER_N; n++) {
		double k = n * dk;
		double i1 = 0.0;
		double i2 = 0.0;
		double r = 0.0;
		double dr = 1.0 / (M - 1);
		i1 += (1.0 / 3.0) * I1(0.0, k) * dr;
		i2 += (1.0 / 3.0) * I2(0.0, k) * dr;
		for (int m = 1; m < M - 1; m += 2) {
			r = m * dr;
			i1 += (4.0 / 3.0) * I1(r, k) * dr;
			i2 += (4.0 / 3.0) * I2(r, k) * dr;
		}
		for (int m = 2; m < M - 1; m += 2) {
			r = m * dr;
			i1 += (2.0 / 3.0) * I1(r, k) * dr;
			i2 += (2.0 / 3.0) * I2(r, k) * dr;
		}
		filter1.push_back(i1);
		filter2.push_back(i2);
	}
	const auto func = [filter1, filter2, dk](double k) {
		if( k >= FILTER_MAX) {
			return 0.0;
		} else {
			int i = k / dk;
			const double y1 = filter1[i];
			const double y2 = filter1[i+1];
			double k1 = filter2[i];
			double k2 = filter2[i+1];
			const double a = k1 * dk - (y2 - y1);
			const double b = -k2 * dk + (y2 - y1);
			const double t = k /dk - i;
			return (1.0-t)*y1+t*y2+t*(1-t)*((1-t)*a+t*b);
		}
	};
	FILE* fp = fopen("green.txt", "wt");
	for (double k = 0.0; k < FILTER_MAX; k += 0.01) {
		fprintf(fp, "%e %e\n", k, func(k));
	}
	fclose(fp);

	return func;
}

static range<int64_t> chain_box;

void treepm_compute_gravity(int Nres, bool do_phi);
void treepm_create_chainmesh(int Nres);
void treepm_cleanup();
void treepm_save_fourier();
void treepm_compute_density(int N);
void treepm_filter_fourier(int dim, int Nres);
void treepm_free_fourier();
void treepm_save_field(int dim, int Nres);
void treepm_create_chainmesh(int Nres);
void treepm_green_init(int Nres);
void treepm_multi_init(int n, int Nres);
void treepm_expansion_init(int n, int m, int l, int Nres);
void treepm_green_save(int Nres);
void treepm_multi_save(int n, int Nres);
void treepm_expansion_save(int n, int m, int l, int Nres);
size_t treepm_particles_count(range<int64_t> box);
void treepm_exchange_and_sort(int Nres);
vector<array<vector<fixed32>, NDIM>> treepm_particles_get(range<int64_t> box);
kick_return treepm_short_range(kick_params, int);
static range<int64_t> double_box2int_box(range<double> box, int Nres);
range<int64_t> treepm_get_fourier_box(int Nres);
void treepm_long_range(int Nres, size_t nparts, bool do_phi);

HPX_PLAIN_ACTION (treepm_expansion_init);
HPX_PLAIN_ACTION (treepm_expansion_save);
HPX_PLAIN_ACTION (treepm_multi_init);
HPX_PLAIN_ACTION (treepm_multi_save);
HPX_PLAIN_ACTION (treepm_green_init);
HPX_PLAIN_ACTION (treepm_green_save);
HPX_PLAIN_ACTION (treepm_particles_get);
HPX_PLAIN_ACTION (treepm_particles_count);
HPX_PLAIN_ACTION (treepm_save_field);
HPX_PLAIN_ACTION (treepm_free_fourier);
HPX_PLAIN_ACTION (treepm_save_fourier);
HPX_PLAIN_ACTION (treepm_filter_fourier);
HPX_PLAIN_ACTION (treepm_compute_density);
HPX_PLAIN_ACTION (treepm_create_chainmesh);
HPX_PLAIN_ACTION (treepm_exchange_and_sort);
HPX_PLAIN_ACTION (treepm_cleanup);
HPX_PLAIN_ACTION (treepm_short_range);

kick_return treepm_kick(kick_params params) {
	kick_return kr;
	const auto& opts = get_options();
	int i;
	for (i = 2; pow(i, NDIM) / hpx_size() < 32 * 32 * 32; i += 2) {
	}
	const int Nres = i;
	const double rs = opts.p3m_rs / Nres;
	params.theta = get_options().theta;
#ifdef TREEPM
	params.phi0 = green_phi0(nparts, rs);
#endif
	timer tm;
	PRINT("Doing chainmesh\n");
	tm.start();
	treepm_create_chainmesh(Nres);
	tm.stop();
	PRINT("took %e s\n", tm.read());

	PRINT("Doing exchange and sort\n");
	tm.reset();
	tm.start();
	treepm_exchange_and_sort(Nres);
	tm.stop();
	PRINT("took %e s\n", tm.read());

	PRINT("Doing long range\n");
	tm.reset();
	tm.start();
	treepm_long_range(Nres, get_options().nparts, params.do_phi);
	tm.stop();
	PRINT("took %e s\n", tm.read());

	PRINT("Doing short range\n");
	tm.reset();
	tm.start();
	kr = treepm_short_range(params, Nres);
	timer tm2;
	tm2.start();
	treepm_cleanup();
	tm2.stop();
	PRINT("-----> %e\n", tm2.read());
	tm.stop();
	PRINT("took %e s\n", tm.read());
	return kr;
}

kick_return treepm_short_range(kick_params params, int Nres) {
	vector<hpx::future<kick_return>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_short_range_action>(c, params, Nres));
	}
	timer tm;
	tm.start();
#ifdef TREEPM
	params.rs = get_options().p3m_rs / Nres;
#endif
	const int Nbnd = get_options().p3m_chainnbnd;
	auto box = treepm_get_fourier_box(Nres);
	vector<kick_workitem> works;
	works.reserve(box.volume());
//	PRINT("Nbnd = %i  Nres = %i chain_box = %i %i %i %i %i %i\n", Nbnd, Nres, box.begin[XDIM], box.end[XDIM], box.begin[YDIM], box.end[YDIM], box.begin[ZDIM],
//			box.end[ZDIM]);

	int cell_count = 0;
	array<int64_t, NDIM> I, J;
	for (J[XDIM] = -Nbnd; J[XDIM] <= Nbnd; J[XDIM]++) {
		for (J[YDIM] = -Nbnd; J[YDIM] <= +Nbnd; J[YDIM]++) {
			for (J[ZDIM] = -Nbnd; J[ZDIM] <= +Nbnd; J[ZDIM]++) {
				auto K = J;
				for (int dim = 0; dim < NDIM; dim++) {
					K[dim] = std::max(std::abs(K[dim]) - 0.999999, 0.0);
				}
				if (sqr(K[XDIM], K[YDIM], K[ZDIM]) < sqr(Nbnd)) {
					cell_count++;
				}
			}
		}
	}
	PRINT("%i cells per workitem\n", cell_count);
	vector<hpx::future<void>> futs;
	mutex_type mutex;
	vector<hpx::future<void>> futs2;
	for (I[XDIM] = box.begin[XDIM]; I[XDIM] < box.end[XDIM]; I[XDIM]++) {
		futs2.push_back(hpx::async([cell_count,Nbnd,&mutex,box,Nres,&works](array<int64_t, NDIM> I) {
			for (I[YDIM] = box.begin[YDIM]; I[YDIM] < box.end[YDIM]; I[YDIM]++) {
				for (I[ZDIM] = box.begin[ZDIM]; I[ZDIM] < box.end[ZDIM]; I[ZDIM]++) {
					kick_workitem work;
					work.dchecklist.reserve(cell_count);
					array<int64_t, NDIM> J;
					for (J[XDIM] = I[XDIM] - Nbnd; J[XDIM] <= I[XDIM] + Nbnd; J[XDIM]++) {
						for (J[YDIM] = I[YDIM] - Nbnd; J[YDIM] <= I[YDIM] + Nbnd; J[YDIM]++) {
							for (J[ZDIM] = I[ZDIM] - Nbnd; J[ZDIM] <= I[ZDIM] + Nbnd; J[ZDIM]++) {
								auto K = J - I;
								for( int dim =0; dim< NDIM; dim++) {
									K[dim] = std::max(std::abs(K[dim])-0.999999, 0.0);
								}
								if( sqr(K[XDIM], K[YDIM], K[ZDIM]) < sqr(Nbnd) ) {
									tree_id id;
									id.index = tree_roots[chain_box.index(J)];
									if( id.index >= 0 ) {
										work.dchecklist.push_back(id);
									}
								}
							}
						}
					}
					const auto jjj = box.index(I);
					for( int n = 0; n < PM_EXPANSION_SIZE; n++) {
						work.L[n] = L_x[n][jjj];
					}
					for (int dim = 0; dim < NDIM; dim++) {
						work.pos[dim] = (I[dim] + 0.5) / Nres;
					}
					work.self.index = tree_roots[chain_box.index(I)];
					std::lock_guard<mutex_type> lock(mutex);
					if( work.self.index >= 0 ) {
						works.push_back(work);
					}
				}
			}
		}, I));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	tm.stop();
	PRINT("----> %e\n", tm.read());
	auto rc = cuda_execute_kicks(params, &particles_pos(XDIM, 0), &particles_pos(YDIM, 0), &particles_pos(ZDIM, 0), tree_data(), works);
	for (auto& kr : futs1) {
		rc += kr.get();
	}
	return rc;
}

void treepm_cleanup() {
	vector<hpx::future<void>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_cleanup_action>(c));
	}
	particles_resize(local_part_count);
#ifdef TREEPM
	treepm_free_fields();
#endif
	hpx::wait_all(futs1.begin(), futs1.end());

}
void treepm_exchange_and_sort(int Nres) {
	vector<hpx::future<void>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_exchange_and_sort_action>(c, Nres));
	}
	const auto Nbnd = get_options().p3m_chainnbnd;
	const auto& localities = hpx_localities();
	auto my_intrbox = domains_find_my_box();
	const auto my_rbox = my_intrbox.pad((double) Nbnd / Nres);
	vector<pair<int, range<int64_t>>> boxes;
	vector < range < int64_t >> shifts;
	vector<range<double>> rboxes(1, my_rbox);
	tree_roots.resize(chain_box.volume());
	tree_allocate_nodes();
	for (int dim = 0; dim < NDIM; dim++) {
		vector<range<double>> tmp;
		for (int i = 0; i < rboxes.size(); i++) {
			if (rboxes[i].begin[dim] < 0.0 && rboxes[i].end[dim] > 1.0) {
				auto box1 = rboxes[i];
				auto box2 = rboxes[i];
				auto box3 = rboxes[i];
				box1.end[dim] = box2.begin[dim] = 0.0;
				box2.end[dim] = box3.begin[dim] = 1.0;
				tmp.push_back(box1);
				tmp.push_back(box2);
				tmp.push_back(box3);
			} else if (rboxes[i].begin[dim] < 0.0) {
				auto box1 = rboxes[i];
				auto box2 = rboxes[i];
				box1.end[dim] = box2.begin[dim] = 0.0;
				tmp.push_back(box1);
				tmp.push_back(box2);
			} else if (rboxes[i].end[dim] > 1.0) {
				auto box1 = rboxes[i];
				auto box2 = rboxes[i];
				box1.end[dim] = box2.begin[dim] = 1.0;
				tmp.push_back(box1);
				tmp.push_back(box2);
			} else {
				tmp.push_back(rboxes[i]);
			}
		}
		rboxes = std::move(tmp);
	}
	for (int i = 0; i < rboxes.size(); i++) {
		auto these_rboxes = domains_find_intersecting_boxes(rboxes[i]);
		for (int i = 0; i < these_rboxes.size(); i++) {
			if (!my_intrbox.contains(these_rboxes[i].second)) {
				pair<int, range<int64_t>> entry;
				entry.first = these_rboxes[i].first;
				entry.second = double_box2int_box(these_rboxes[i].second, Nres);
				range < int64_t > shift = entry.second;
				for (int dim = 0; dim < NDIM; dim++) {
					shift.begin[dim] = (shift.begin[dim] + Nres) % Nres;
					shift.end[dim] = (shift.end[dim] - 1 + Nres) % Nres + 1;
					ALWAYS_ASSERT(shift.end[dim] > shift.begin[dim]);
				}
				boxes.push_back(entry);
				shifts.push_back(shift);
			}
		}
	}
	vector < hpx::future < size_t >> count_futs;
	for (int i = 0; i < boxes.size(); i++) {
		count_futs.push_back(hpx::async<treepm_particles_count_action>(localities[boxes[i].first], shifts[i]));
	}
	vector<part_int> pbnds(count_futs.size() + 1);
	pbnds[0] = 0;
	for (int i = 0; i < count_futs.size(); i++) {
		pbnds[i + 1] = count_futs[i].get() + pbnds[i];
	}
	local_part_count = particles_size();
	particles_resize(particles_size() + pbnds.back());
	const auto int_box = treepm_get_fourier_box(Nres);
	array<int64_t, NDIM> I;
	for (int n = 0; n < PM_MULTIPOLE_SIZE; n++) {
		M_x[n].resize(int_box.volume());
	}
	for (I[XDIM] = int_box.begin[XDIM]; I[XDIM] < int_box.end[XDIM]; I[XDIM]++) {
		futs1.push_back(hpx::async([int_box,Nres](array<int64_t, NDIM> I) {
			for (I[YDIM] = int_box.begin[YDIM]; I[YDIM] < int_box.end[YDIM]; I[YDIM]++) {
				for (I[ZDIM] = int_box.begin[ZDIM]; I[ZDIM] < int_box.end[ZDIM]; I[ZDIM]++) {
					auto j = chain_box.index(I);
					range<double> rbox;
					for (int dim = 0; dim < NDIM; dim++) {
						rbox.begin[dim] = (double) I[dim] / Nres;
						rbox.end[dim] = (double) (I[dim] + 1) / Nres;
					}
					if( chain_mesh[j].second - chain_mesh[j].first > 0 ) {
						auto rc = tree_create( tree_create_params(), 0, pair<int, int>(hpx_rank(),hpx_rank()+1), chain_mesh[j], rbox, 0, false);
						const auto ini = int_box.index(I);
						for( int l = 0; l < PM_MULTIPOLE_SIZE; l++) {
							M_x[l][ini] = rc.multi[l];
						}
						tree_roots[j] = rc.id.index;
					} else {
						tree_roots[j] = -1;
					}
				}
			}
		}, I));
	}
	for (int i = 0; i < boxes.size(); i++) {
		auto fut = hpx::async<treepm_particles_get_action>(localities[boxes[i].first], shifts[i]);
		auto vfut = fut.then([i,&boxes,&pbnds](hpx::future<vector<array<vector<fixed32>, NDIM>>> fut) {
			auto parts = fut.get();
			const auto box = boxes[i].second;
			part_int start = pbnds[i] + local_part_count;
			array<int64_t,NDIM> I;
			for (I[XDIM] = box.begin[XDIM]; I[XDIM] < box.end[XDIM]; I[XDIM]++) {
				for (I[YDIM] = box.begin[YDIM]; I[YDIM] < box.end[YDIM]; I[YDIM]++) {
					for (I[ZDIM] = box.begin[ZDIM]; I[ZDIM] < box.end[ZDIM]; I[ZDIM]++) {
						const int index = box.index(I);
						const auto& pos = parts[index];
						for( int dim = 0; dim < NDIM; dim++) {
							memcpy(&particles_pos(dim,start), pos[dim].data(), sizeof(fixed32)*pos[dim].size());
						}
						const auto ci = chain_box.index(I);
						chain_mesh[ci].first = start;
						start += pos[XDIM].size();
						chain_mesh[ci].second = start;
					}
				}
			}
		});
		vfut = vfut.then([i,&boxes,Nres](hpx::future<void> f) {
			f.get();
			array<int64_t,NDIM> I;
			vector<hpx::future<void>> futs;
			const auto box = boxes[i].second;
			for (I[XDIM] = box.begin[XDIM]; I[XDIM] < box.end[XDIM]; I[XDIM]++) {
				futs.push_back(hpx::async([box,Nres](array<int64_t, NDIM> I) {
									for (I[YDIM] = box.begin[YDIM]; I[YDIM] < box.end[YDIM]; I[YDIM]++) {
										for (I[ZDIM] = box.begin[ZDIM]; I[ZDIM] < box.end[ZDIM]; I[ZDIM]++) {
											auto j = chain_box.index(I);
											range<double> rbox;
											for (int dim = 0; dim < NDIM; dim++) {
												rbox.begin[dim] = (double) I[dim] / Nres;
												rbox.end[dim] = (double) (I[dim] + 1) / Nres;
												if( rbox.begin[dim] < 0.0 ) {
													rbox.begin[dim] += 1.0;
													rbox.end[dim] += 1.0;
												} else if( rbox.end[dim] > 1.0 ) {
													rbox.begin[dim] -= 1.0;
													rbox.end[dim] -= 1.0;
												}
											}
											if( chain_mesh[j].second - chain_mesh[j].first > 0 ) {
												auto rc = tree_create( tree_create_params(), 0, pair<int, int>(hpx_rank(),hpx_rank()+1), chain_mesh[j], rbox, 0, false);
												tree_roots[j] = rc.id.index;
											} else {
												tree_roots[j] = -1;
											}
										}
									}
								}, I));
			}
			return hpx::when_all(futs.begin(), futs.end());
		});
		futs1.push_back(std::move(vfut));
	}
	hpx::wait_all(futs1.begin(), futs1.end());
}

size_t treepm_particles_count(range<int64_t> box) {
	array<int64_t, NDIM> I;
	size_t count = 0;
	for (I[XDIM] = box.begin[XDIM]; I[XDIM] < box.end[XDIM]; I[XDIM]++) {
		for (I[YDIM] = box.begin[YDIM]; I[YDIM] < box.end[YDIM]; I[YDIM]++) {
			for (I[ZDIM] = box.begin[ZDIM]; I[ZDIM] < box.end[ZDIM]; I[ZDIM]++) {
				const auto rng = chain_mesh[chain_box.index(I)];
				count += rng.second - rng.first;
			}
		}
	}
	return count;
}

vector<array<vector<fixed32>, NDIM>> treepm_particles_get(range<int64_t> box) {
	array<int64_t, NDIM> I;
	size_t count = 0;
	vector<array<vector<fixed32>, NDIM>> res;
	res.resize(box.volume());
	for (I[XDIM] = box.begin[XDIM]; I[XDIM] < box.end[XDIM]; I[XDIM]++) {
		for (I[YDIM] = box.begin[YDIM]; I[YDIM] < box.end[YDIM]; I[YDIM]++) {
			for (I[ZDIM] = box.begin[ZDIM]; I[ZDIM] < box.end[ZDIM]; I[ZDIM]++) {
				const auto rng = chain_mesh[chain_box.index(I)];
				const auto size = rng.second - rng.first;
				auto& entry = res[box.index(I)];
				for (int dim = 0; dim < NDIM; dim++) {
					entry[dim].resize(size);
					memcpy(entry[dim].data(), &particles_pos(dim, rng.first), sizeof(fixed32) * size);
				}
			}
		}
	}
	return res;
}

static range<int64_t> double_box2int_box(range<double> rbox, int Nres) {
	range<int64_t> box;
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = lround(rbox.begin[dim] * Nres);
		box.end[dim] = lround(rbox.end[dim] * Nres);
	}
	return box;
}

range<int64_t> treepm_get_fourier_box(int Nres) {
	ALWAYS_ASSERT(Nres >= get_options().p3m_Nmin);
	auto rbox = domains_find_my_box();
	return double_box2int_box(rbox, Nres);
}

void treepm_sort_particles(int Nres, range<int64_t> box, pair<part_int> part_range) {
	const auto boxvol = box.volume();
	if (boxvol == 0) {
		ALWAYS_ASSERT(false);
		return;
	} else if (boxvol == 1) {
		const auto index = chain_box.index(box.begin);
		chain_mesh[index] = part_range;
	} else {
		const int xdim = box.longest_dim();
		const int imid = (box.begin[xdim] + box.end[xdim]) / 2;
		part_int lo = part_range.first;
		part_int hi = part_range.second;
		while (lo < hi) {
			if (particles_pos(xdim, lo).to_double() * Nres >= imid) {
				while (lo != hi) {
					hi--;
					if (particles_pos(xdim, hi).to_double() * Nres < imid) {
						particles_swap(lo, hi);
						break;
					}
				}
			}
			lo++;
		}
		auto left_rng = part_range;
		auto right_rng = part_range;
		auto left_box = box;
		auto right_box = box;
		left_rng.second = right_rng.first = hi;
		left_box.end[xdim] = right_box.begin[xdim] = imid;
		static std::atomic<int> nthreads = 0;
		hpx::future<void> fut;
		bool thread = nthreads++ < 4 * hpx_hardware_concurrency() - 1;
		thread = thread && (left_rng.second - left_rng.first) > 1024;
		if (thread) {
			fut = hpx::async([Nres, left_box, left_rng]() {
				treepm_sort_particles(Nres, left_box, left_rng);
				nthreads--;
			});
		} else {
			nthreads--;
			treepm_sort_particles(Nres, left_box, left_rng);
			fut = hpx::make_ready_future();
		}
		treepm_sort_particles(Nres, right_box, right_rng);
		fut.get();
	}
}

void treepm_long_range(int Nres, size_t nparts, bool do_phi) {
#ifdef TREEPM
	fft3d_init(Nres, -(double) nparts / (sqr(Nres) * Nres));
	treepm_compute_density(Nres);
	fft3d_execute();
	treepm_save_fourier();
	fft3d_destroy();
	const int dim_max = do_phi ? NDIM + 1 : NDIM;
	for (int dim = 0; dim < dim_max; dim++) {
		fft3d_init(Nres);
		treepm_filter_fourier(dim, Nres);
		fft3d_inv_execute();
		treepm_save_field(dim, Nres);
		fft3d_destroy();
	}
	treepm_free_fourier();
#else
	static bool green_init = false;
	const int Nbnd = get_options().p3m_chainnbnd;
	fft3d_dbl_init(Nres);
	if (!green_init) {
		timer tm;
		tm.start();
		PRINT("Initializing greens function\n");
		green_init = true;
		const auto rsz = treepm_get_fourier_box(Nres).volume();
		const auto csz = fft3d_dbl_complex_range().volume();
		for (int i = 0; i < PM_EXPANSION_SIZE; i++) {
			D_k[i].resize(csz);
			L_x[i].resize(rsz);
		}
		for (int i = 0; i < PM_MULTIPOLE_SIZE; i++) {
			M_k[i].resize(csz);
		}
		treepm_green_init(Nres);
		fft3d_dbl_execute();
		treepm_green_save(Nres);
		fft3d_dbl_destroy();
		fft3d_dbl_init(Nres);
		tm.stop();
		PRINT("Done in %e\n", tm.read());
	}
	for (int n = 0; n < PM_MULTIPOLE_SIZE; n++) {
		fft3d_dbl_init(Nres);
		treepm_multi_init(n, Nres);
		fft3d_dbl_execute();
		treepm_multi_save(n, Nres);
		fft3d_dbl_destroy();
	}
	for (int n = 0; n < PM_ORDER; n++) {
		for (int m = 0; m < PM_ORDER - n; m++) {
			for (int l = 0; l < PM_ORDER - n - m; l++) {
				if (l >= 2 && !(l == 2 && n == 0 && m == 0)) {
					continue;
				}
				fft3d_dbl_init(Nres);
				treepm_expansion_init(n, m, l, Nres);
				fft3d_dbl_inv_execute();
				treepm_expansion_save(n, m, l, Nres);
				fft3d_dbl_destroy();
			}
		}
	}

#endif
}

void treepm_green_init(int Nres) {
	vector<hpx::future<void>> futs1, futs2;
	const int nbnd = get_options().p3m_chainnbnd;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_green_init_action>(c, Nres));
	}
	const auto box = fft3d_dbl_real_range();
	vector<double> R(box.volume());
	array<int64_t, NDIM> I;
	for (I[XDIM] = box.begin[XDIM]; I[XDIM] < box.end[XDIM]; I[XDIM]++) {
		futs2.push_back(hpx::async([nbnd,box,Nres,&R](array<int64_t,NDIM> I) {
			array<long double, NDIM > x;
			array<int64_t, NDIM> J;
			for (I[YDIM] = box.begin[YDIM]; I[YDIM] < box.end[YDIM]; I[YDIM]++) {
				for (I[ZDIM] = box.begin[ZDIM]; I[ZDIM] < box.end[ZDIM]; I[ZDIM]++) {
					for (int dim = 0; dim < NDIM; dim++) {
						J[dim] = (I[dim] < Nres / 2 ? I[dim] : I[dim] - Nres);
						x[dim] = (double) J[dim] / Nres;
					}
					long double phi_ewald = high_precision_ewald(x);
					long double phi_direct = -1.0l / sqrtl(sqr(x[XDIM],x[YDIM],x[YDIM]));
					long double phi = phi_ewald;
					int dist2 = 0;
					for( int dim = 0; dim < NDIM; dim++) {
						dist2 += sqr(std::min(std::abs(J[dim]-0.999999),0.0));
					}
					if( dist2 >= sqr(nbnd)) {
						phi += phi_direct;
					}
					const auto index = box.index(I);
					R[index] = phi;
				}
			}}, I));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	fft3d_dbl_accumulate_real(box, R);

	hpx::wait_all(futs1.begin(), futs1.end());
}

void treepm_green_save(int Nres) {
	vector<hpx::future<void>> futs1, futs2;
	const int nbnd = get_options().p3m_chainnbnd;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_green_save_action>(c, Nres));
	}
	const auto box = fft3d_dbl_complex_range();
	D_k[0] = fft3d_dbl_read_complex(box);
	array<int64_t, NDIM> I;
	for (I[XDIM] = box.begin[XDIM]; I[XDIM] < box.end[XDIM]; I[XDIM]++) {
		futs1.push_back(hpx::async([box,Nres](array<int64_t,NDIM> I) {
			const auto* nodes = tree_data();
			for (I[YDIM] = box.begin[YDIM]; I[YDIM] < box.end[YDIM]; I[YDIM]++) {
				for (I[ZDIM] = box.begin[ZDIM]; I[ZDIM] < box.end[ZDIM]; I[ZDIM]++) {
					array<complex<double>, NDIM> K;
					for (int dim = 0; dim < NDIM; dim++) {
						K[dim] = complex<double>(0.0,2.0 * M_PI * (I[dim] < Nres / 2 ? I[dim] : I[dim] - Nres));
					}
					tensor_sym<complex<double>, PM_ORDER> kk = vector_to_sym_tensor<complex<double>,PM_ORDER>(K);
					const auto kk_trless = kk.detraceD();
					const auto index = box.index(I);
					for( int n = 1; n < PM_EXPANSION_SIZE; n++) {
						D_k[n][index] = D_k[0][index] * kk_trless[n];
					}

				}
			}}, I));
	}
	hpx::wait_all(futs1.begin(), futs1.end());
}

void treepm_multi_init(int n, int Nres) {
	vector<hpx::future<void>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_multi_init_action>(c, n, Nres));
	}
	const auto box = treepm_get_fourier_box(Nres);
	fft3d_dbl_accumulate_real(box, M_x[n]);

	hpx::wait_all(futs1.begin(), futs1.end());
}

void treepm_multi_save(int n, int Nres) {
	vector<hpx::future<void>> futs1, futs2;
	const int nbnd = get_options().p3m_chainnbnd;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_multi_save_action>(c, n, Nres));
	}
	const auto box = fft3d_dbl_complex_range();
	M_k[n] = fft3d_dbl_read_complex(box);
	hpx::wait_all(futs1.begin(), futs1.end());
}

complex<double> M_k_sym(int n, int m, int l, int i) {
	if (l > 1) {
		if (l == 2 && n == 0 && m == 0) {
			return M_k[trless_index(n, m, l, PM_ORDER - 1)][i];
		} else {
			return -M_k_sym(n + 2, m, l - 2, i) - M_k_sym(n, m + 2, l - 2, i);
		}
	} else {
		return M_k[trless_index(n, m, l, PM_ORDER - 1)][i];
	}
}

complex<double> D_k_sym(int n, int m, int l, int i) {
	if (l > 1) {
		if (l == 2 && n == 0 && m == 0) {
			return D_k[trless_index(n, m, l, PM_ORDER)][i];
		} else {
			return -D_k_sym(n + 2, m, l - 2, i) - D_k_sym(n, m + 2, l - 2, i);
		}
	} else {
		return D_k[trless_index(n, m, l, PM_ORDER)][i];
	}
}

void treepm_expansion_init(int nx, int ny, int nz, int Nres) {
	vector<hpx::future<void>> futs1;
	vector<hpx::future<void>> futs2;
	const int nbnd = get_options().p3m_chainnbnd;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_expansion_init_action>(c, nx, ny, nz, Nres));
	}
	const auto box = fft3d_dbl_complex_range();
	vector<complex<double>> Y(box.volume());
	array<int64_t, NDIM> I;
	array<int, NDIM> n;
	array<int, NDIM> m;
	n[XDIM] = nx;
	n[YDIM] = ny;
	n[ZDIM] = nz;
	const int n0 = n[0] + n[1] + n[2];
	const int q0 = intmin(PM_ORDER - n0, (PM_ORDER - 1));
	for (auto& y : Y) {
		y = complex<double>(0.0, 0.0);
	}
	for (m[0] = 0; m[0] < q0; m[0]++) {
		for (m[1] = 0; m[1] < q0 - m[0]; m[1]++) {
			for (m[2] = 0; m[2] < q0 - m[0] - m[1]; m[2]++) {
				const double coeff = 1.0 / vfactorial(m);
				const int nthreads = hpx_hardware_concurrency();
				futs2.resize(0);
				const int li = trless_index(n[0], n[1], n[2], PM_ORDER);
				for (int proc = 0; proc < nthreads; proc++) {
					futs2.push_back(hpx::async([proc,nthreads,coeff,&Y,m,n,li]() {
						const auto b = (size_t) proc * Y.size() / nthreads;
						const auto e = (size_t) (proc + 1) * Y.size() / nthreads;
						for( size_t i = b; i < e; i++) {
							Y[i] += M_k_sym(m[0], m[1],m[2], i) * D_k_sym(m[0]+n[0], m[1]+n[1],m[2]+n[2], i) * coeff;
						}
					}));

				}
				hpx::wait_all(futs2.begin(), futs2.end());
			}
		}
	}
	fft3d_dbl_accumulate_complex(box, Y);
	hpx::wait_all(futs1.begin(), futs1.end());

}

void treepm_expansion_save(int n, int m, int l, int Nres) {
	vector<hpx::future<void>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_expansion_save_action>(c, n, m, l, Nres));
	}
	const auto box = treepm_get_fourier_box(Nres);
	auto L = fft3d_dbl_read_real(box);
	L_x(n, m, l) = L;
	hpx::wait_all(futs1.begin(), futs1.end());
}

void treepm_create_chainmesh(int Nres) {
	vector<hpx::future<void>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_create_chainmesh_action>(c, Nres));
	}
	const int Nbnd = get_options().p3m_chainnbnd;
	chain_box = treepm_get_fourier_box(Nres);
	chain_box = chain_box.pad(Nbnd);
//	PRINT("Nres = %i chain_box = %i %i %i %i %i %i\n", Nres, chain_box.begin[XDIM], chain_box.end[XDIM], chain_box.begin[YDIM], chain_box.end[YDIM],
//			chain_box.begin[ZDIM], chain_box.end[ZDIM]);
	chain_mesh.resize(chain_box.volume());
	treepm_sort_particles(Nres, treepm_get_fourier_box(Nres), particles_current_range());
	hpx::wait_all(futs1.begin(), futs1.end());
}

void treepm_free_fourier() {
	vector<hpx::future<void>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_free_fourier_action>(c));
	}
	Y0 = decltype(Y0)();
	hpx::wait_all(futs1.begin(), futs1.end());
}

void treepm_save_fourier() {
	vector<hpx::future<void>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_save_fourier_action>(c));
	}
	Y0 = fft3d_read_complex(fft3d_complex_range());
	hpx::wait_all(futs1.begin(), futs1.end());
}

void treepm_save_field(int dim, int Nres) {
#ifdef TREEPM
	vector<hpx::future<void>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_save_field_action>(c, dim, Nres));
	}
	auto box = treepm_get_fourier_box(Nres);
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] += CLOUD_MIN;
		box.end[dim] += CLOUD_MAX;
	}
	treepm_set_field(dim, fft3d_dbl_read_real(box));
	hpx::wait_all(futs1.begin(), futs1.end());
#endif
}

void treepm_filter_fourier(int dim, int Nres) {
	vector<hpx::future<void>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_filter_fourier_action>(c, dim, Nres));
	}
	const double rs = get_options().p3m_rs / Nres;
	const double hsoft = get_options().hsoft;
	array<int64_t, NDIM> I;
	const auto i = complex<float>(0, 1);
	auto box = fft3d_complex_range();
	const double h = 1.0 / Nres;
	decltype(Y0) Y(Y0.size());
	vector<hpx::future<void>> futs2;
	const static auto gfilter = treepm_init_filter();
	for (I[XDIM] = box.begin[XDIM]; I[XDIM] < box.end[XDIM]; I[XDIM]++) {
		futs2.push_back(hpx::async([hsoft,box,h,rs,i,dim,&Y,Nres](array<int64_t,NDIM> I) {
			array<complex<float>, NDIM + 1> k;
			k[NDIM] = i;
			for (I[YDIM] = box.begin[YDIM]; I[YDIM] < box.end[YDIM]; I[YDIM]++) {
				for (I[ZDIM] = box.begin[ZDIM]; I[ZDIM] < box.end[ZDIM]; I[ZDIM]++) {
					double k2 = 0.0;
					double coeff = sqr(Nres)*Nres;
					for (int dim = 0; dim < NDIM; dim++) {
						k[dim] = complex<float>(2.0 * M_PI * (I[dim] < Nres / 2 ? I[dim] : I[dim] - Nres), 0.0);
						coeff *= sqr(cloud_filter(k[dim].real() * h));
						k2 += sqr(k[dim].real());
					}
					coeff *= gfilter(sqrt(k2)*rs);
					coeff *= gfilter(sqrt(k2)*hsoft);
					const auto index = box.index(I);
					if (k2 > 0.0) {
						Y[index] = Y0[index] * i * coeff * k[dim] * (4.0 * M_PI)/ k2;
					} else {
						Y[index] = complex<float>(0.f, 0.f);
					}
				}
			}}, I));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	fft3d_accumulate_complex(box, Y);
	hpx::wait_all(futs1.begin(), futs1.end());
}

void treepm_compute_density(int N) {
#ifdef TREEPM
	vector<hpx::future<void>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_compute_density_action>(c, N));
	}
	treepm_allocate_fields(N);
	range<int64_t> rho_box = treepm_get_fourier_box(N);
	const auto int_box = rho_box;
	for (int dim = 0; dim < NDIM; dim++) {
		rho_box.begin[dim] += CLOUD_MIN;
		rho_box.end[dim] += CLOUD_MAX;
	}
	auto rho = treepm_compute_density_local(N, chain_mesh, int_box, chain_box, rho_box);
	vector<double> rho0(rho.begin(), rho.end());
	fft3d_dbl_accumulate_real(rho_box, rho0);
	hpx::wait_all(futs1.begin(), futs1.end());
#endif
}

