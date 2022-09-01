#include <cosmictiger/defs.hpp>
#ifdef TREEPM

#include <cosmictiger/domain.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/power.hpp>

static vector<float> field[NDIM + 1];
static vector<cmplx> Y0;
static vector<cmplx> Y;
static range<int64_t> chain_box;
static vector<pair<part_int>> chain_mesh;
static size_t local_part_count;

void treepm_save_fourier();
void treepm_compute_density(int N);
void treepm_filter_fourier(int dim, int Nres);
void treepm_restore_fourier();
void treepm_free_fourier();
void treepm_save_field(int dim, int Nres);
void treepm_create_chainmesh(int Nres);
size_t treepm_particles_count(range<int64_t> box);
void treepm_exchange_boundaries(int Nres);
vector<array<vector<fixed32>, NDIM>> treepm_particles_get(range<int64_t> box);
static range<int64_t> double_box2int_box(range<double> box, int Nres);

HPX_PLAIN_ACTION (treepm_particles_get);
HPX_PLAIN_ACTION (treepm_particles_count);
HPX_PLAIN_ACTION (treepm_save_field);
HPX_PLAIN_ACTION (treepm_free_fourier);
HPX_PLAIN_ACTION (treepm_save_fourier);
HPX_PLAIN_ACTION (treepm_restore_fourier);
HPX_PLAIN_ACTION (treepm_filter_fourier);
HPX_PLAIN_ACTION (treepm_compute_density);
HPX_PLAIN_ACTION (treepm_create_chainmesh);
HPX_PLAIN_ACTION (treepm_exchange_boundaries);

void treepm_exchange_boundaries(int Nres) {
	vector<hpx::future<void>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_exchange_boundaries_action>(c, Nres));
	}
	const auto Nbnd = get_options().p3m_chainnbnd;
	const auto& localities = hpx_localities();
	auto my_rbox = domains_find_my_box();
	my_rbox.pad(1.0 / Nbnd - 0.000001);
	vector<pair<int, range<int64_t>>> boxes;
	vector < range < int64_t >> shifts;
	vector<range<double>> rboxes(1, my_rbox);
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
			}
		}
		rboxes = std::move(tmp);
	}
	for (int i = 0; i < rboxes.size(); i++) {
		auto these_rboxes = domains_find_intersecting_boxes(rboxes[i]);
		for (int i = 0; i < these_rboxes.size(); i++) {
			if (these_rboxes[i].second != my_rbox) {
				pair<int, range<int64_t>> entry;
				entry.first = these_rboxes[i].first;
				entry.second = double_box2int_box(these_rboxes[i].second, Nres);
				range < int64_t > shift = entry.second;
				for (int dim = 0; dim < NDIM; dim++) {
					shift.begin[dim] = (shift.begin[dim] + Nres) % Nres;
					shift.end[dim] = (shift.end[dim] + Nres) % Nres;
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
	particles_resize(pbnds.back());

	for (int i = 0; i < boxes.size(); i++) {
		auto fut = hpx::async<treepm_particles_get_action>(localities[boxes[i].first], shifts[i]);
		futs1.push_back(fut.then([i,&boxes,&pbnds](hpx::future<vector<array<vector<fixed32>, NDIM>>> fut) {
			auto parts = fut.get();
			const auto box = boxes[i].second;
			part_int start = pbnds[i];
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
		}));
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

void treepm_compute_gravity(int Nres, bool do_phi) {
	fft3d_init(Nres);
	treepm_compute_density(Nres);
	fft3d_execute();
	treepm_save_fourier();
	fft3d_destroy();
	const int dim_max = do_phi ? NDIM + 1 : NDIM;
	for (int dim = 0; dim < dim_max; dim++) {
		fft3d_init(Nres);
		treepm_restore_fourier();
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
	chain_box.pad(Nbnd);
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
	Y = decltype(Y)();
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

void treepm_restore_fourier() {
	vector<hpx::future<void>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_restore_fourier_action>(c));
	}
	Y = Y0;
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
	field[dim] = fft3d_read_real(box);
	hpx::wait_all(futs1.begin(), futs1.end());
}

void treepm_filter_fourier(int dim, int Nres) {
	vector<hpx::future<void>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_filter_fourier_action>(c, dim, Nres));
	}
	array<int64_t, NDIM> I;
	array<cmplx, NDIM + 1> k;
	const auto i = cmplx(0, 1);
	k[NDIM] = i;
	auto box = fft3d_complex_range();
	const float h = 1.0 / Nres;
	for (I[XDIM] = box.begin[XDIM]; I[XDIM] < box.end[XDIM]; I[XDIM]++) {
		for (I[YDIM] = box.begin[YDIM]; I[YDIM] < box.end[YDIM]; I[YDIM]++) {
			for (I[ZDIM] = box.begin[ZDIM]; I[ZDIM] < box.end[ZDIM]; I[ZDIM]++) {
				float k2 = 0.0;
				float filter = 1.0;
				for (int dim = 0; dim < NDIM; dim++) {
					k[dim] = cmplx(2.0 * M_PI * I[dim], 0.0);
					filter *= cloud_filter(k[dim].real() * h);
					k2 += sqr(k[dim].real());
				}
				const auto index = box.index(I);
				if (k2 > 0.0) {
					Y[index] = Y0[index] * i * filter * k[dim] / k2;
				} else {
					Y[index] = cmplx(0.f, 0.f);
				}
			}
		}
	}
	fft3d_accumulate_complex(box, Y);
	hpx::wait_all(futs1.begin(), futs1.end());
}

void treepm_compute_density(int N) {
	vector<float> rho0;
	vector<hpx::future<void>> futs1;
	for (auto c : hpx_children()) {
		futs1.push_back(hpx::async<treepm_compute_density_action>(c, N));
	}
	range<int64_t> intbox = treepm_get_fourier_box(N);
	for (int dim = 0; dim < NDIM; dim++) {
		intbox.begin[dim] += CLOUD_MIN;
		intbox.end[dim] += CLOUD_MAX;
	}
	auto rho = accumulate_density_cuda(1, N, intbox);
	fft3d_accumulate_real(intbox, rho);
	hpx::wait_all(futs1.begin(), futs1.end());
}

#endif
