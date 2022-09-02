#include <cosmictiger/defs.hpp>
#ifdef TREEPM

#include <cosmictiger/treepm.hpp>
#include <cosmictiger/domain.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/power.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/timer.hpp>

static vector<cmplx> Y0;
static device_vector<pair<part_int>> chain_mesh;
static vector<int> tree_roots;
static size_t local_part_count;

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
size_t treepm_particles_count(range<int64_t> box);
void treepm_exchange_and_sort(int Nres);
vector<array<vector<fixed32>, NDIM>> treepm_particles_get(range<int64_t> box);
kick_return treepm_short_range(kick_params, int);
static range<int64_t> double_box2int_box(range<double> box, int Nres);
range<int64_t> treepm_get_fourier_box(int Nres);
void treepm_long_range(int Nres, size_t nparts, bool do_phi);

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
	const auto& opts = get_options();
	const auto nparts = particles_active_count();
	int i;
	for (i = 2; nparts * pow(i * opts.p3m_Nmin, -NDIM) > opts.p3m_chainres; i++) {
	}
	i--;
	const int Nres = i * opts.p3m_Nmin;
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
	treepm_long_range(Nres, nparts, params.do_phi);
	tm.stop();
	PRINT("took %e s\n", tm.read());

	PRINT("Doing short range\n");
	tm.reset();
	tm.start();
	auto kr = treepm_short_range(params, Nres);
	treepm_cleanup();
	PRINT("took %e s\n", tm.read());

	return kr;
}

kick_return treepm_short_range(kick_params params, int Nres) {
	vector<hpx::future<kick_return>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_short_range_action>(c, params, Nres));
	}
	params.rs = get_options().p3m_rs / Nres;
	const int Nbnd = get_options().p3m_chainnbnd;
	vector<kick_workitem> works;
	auto box = treepm_get_fourier_box(Nres);
//	PRINT("Nbnd = %i  Nres = %i chain_box = %i %i %i %i %i %i\n", Nbnd, Nres, box.begin[XDIM], box.end[XDIM], box.begin[YDIM], box.end[YDIM], box.begin[ZDIM],
//			box.end[ZDIM]);

	vector<hpx::future<void>> futs;
	mutex_type mutex;
	array<int64_t, NDIM> I;
	vector<hpx::future<void>> futs2;
	for (I[XDIM] = box.begin[XDIM]; I[XDIM] < box.end[XDIM]; I[XDIM]++) {
		futs2.push_back(hpx::async([Nbnd,&mutex,box,Nres,&works](array<int64_t, NDIM> I) {
			for (I[YDIM] = box.begin[YDIM]; I[YDIM] < box.end[YDIM]; I[YDIM]++) {
				for (I[ZDIM] = box.begin[ZDIM]; I[ZDIM] < box.end[ZDIM]; I[ZDIM]++) {
					kick_workitem work;
					array<int64_t, NDIM> J;
					for (J[XDIM] = I[XDIM] - Nbnd; J[XDIM] <= I[XDIM] + Nbnd; J[XDIM]++) {
						for (J[YDIM] = I[YDIM] - Nbnd; J[YDIM] <= I[YDIM] + Nbnd; J[YDIM]++) {
							for (J[ZDIM] = I[ZDIM] - Nbnd; J[ZDIM] <= I[ZDIM] + Nbnd; J[ZDIM]++) {
								auto K = J - I;
								if( sqr(K[XDIM], K[YDIM], K[ZDIM]) <= sqr(Nbnd) ) {
									tree_id id;
									id.index = tree_roots[chain_box.index(J)];
									if( id.index >= 0 ) {
										work.dchecklist.push_back(id);
									}
								}
							}
						}
					}
					work.L = 0.f;
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
	treepm_free_fields();
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
	for (I[XDIM] = int_box.begin[XDIM]; I[XDIM] < int_box.end[XDIM]; I[XDIM]++) {
		for (I[YDIM] = int_box.begin[YDIM]; I[YDIM] < int_box.end[YDIM]; I[YDIM]++) {
			futs1.push_back(hpx::async([int_box,Nres](array<int64_t, NDIM> I) {
				for (I[ZDIM] = int_box.begin[ZDIM]; I[ZDIM] < int_box.end[ZDIM]; I[ZDIM]++) {
					auto j = chain_box.index(I);
					range<double> rbox;
					for (int dim = 0; dim < NDIM; dim++) {
						rbox.begin[dim] = (double) I[dim] / Nres;
						rbox.end[dim] = (double) (I[dim] + 1) / Nres;
					}
					if( chain_mesh[j].second - chain_mesh[j].first > 0 ) {
						//		PRINT( "%li %li\n",chain_mesh[j].first, chain_mesh[j].second);
					auto rc = tree_create( tree_create_params(), 0, pair<int, int>(hpx_rank(),hpx_rank()+1), chain_mesh[j], rbox, 0, false);
					tree_roots[j] = rc.id.index;
				} else {
					tree_roots[j] = -1;
				}
			}
		}, I));
		}
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

void treepm_long_range(int Nres, size_t nparts, bool do_phi) {

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
		if (nthreads++ < hpx_hardware_concurrency() - 1) {
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
	vector<hpx::future<void>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_save_field_action>(c, dim, Nres));
	}
	auto box = treepm_get_fourier_box(Nres);
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] += CLOUD_MIN;
		box.end[dim] += CLOUD_MAX;
	}
	treepm_set_field(dim, fft3d_read_real(box));
	hpx::wait_all(futs1.begin(), futs1.end());
}

void treepm_filter_fourier(int dim, int Nres) {
	vector<hpx::future<void>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_filter_fourier_action>(c, dim, Nres));
	}
	const float rs = get_options().p3m_rs / Nres;
	array<int64_t, NDIM> I;
	const auto i = cmplx(0, 1);
	auto box = fft3d_complex_range();
	const float h = 1.0 / Nres;
	const float rs2 = sqr(rs);
	decltype(Y0) Y(Y0.size());
	vector<hpx::future<void>> futs2;
	for (I[XDIM] = box.begin[XDIM]; I[XDIM] < box.end[XDIM]; I[XDIM]++) {
		futs2.push_back(hpx::async([box,h,rs2,i,dim,&Y,Nres](array<int64_t,NDIM> I) {
			array<cmplx, NDIM + 1> k;
			k[NDIM] = i;
			for (I[YDIM] = box.begin[YDIM]; I[YDIM] < box.end[YDIM]; I[YDIM]++) {
				for (I[ZDIM] = box.begin[ZDIM]; I[ZDIM] < box.end[ZDIM]; I[ZDIM]++) {
					float k2 = 0.0;
					float filter = sqr(Nres)*Nres;
					for (int dim = 0; dim < NDIM; dim++) {
						k[dim] = cmplx(2.0 * M_PI * (I[dim] < Nres / 2 ? I[dim] : I[dim] - Nres), 0.0);
						filter *= sqr(cloud_filter(k[dim].real() * h));
						k2 += sqr(k[dim].real());
					}
					//		filter *= exp(-k2 * rs2);
				const auto index = box.index(I);
				if (k2 > 0.0) {
					Y[index] = Y0[index] * i * filter * k[dim] * (4.0 * M_PI)/ k2;
				} else {
					Y[index] = cmplx(0.f, 0.f);
				}
			}
		}}, I));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	fft3d_accumulate_complex(box, Y);
	hpx::wait_all(futs1.begin(), futs1.end());
}

void treepm_compute_density(int N) {
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
	vector<float> rho0(rho.begin(), rho.end());
	fft3d_accumulate_real(rho_box, rho0);
	hpx::wait_all(futs1.begin(), futs1.end());
}

#endif
