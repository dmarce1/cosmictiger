#include <cosmictiger/defs.hpp>
#ifdef TREEPM

#include <cosmictiger/domain.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/power.hpp>

static vector<float> field[NDIM + 1];
static vector<cmplx> Y0;
static vector<cmplx> Y;

void treepm_save_fourier();
void treepm_compute_density(int N);
void treepm_filter_fourier(int dim, int Nres);
void treepm_restore_fourier();
void treepm_save_field(int dim, int Nres);

HPX_PLAIN_ACTION (treepm_save_field);
HPX_PLAIN_ACTION (treepm_save_fourier);
HPX_PLAIN_ACTION (treepm_restore_fourier);
HPX_PLAIN_ACTION (treepm_filter_fourier);
HPX_PLAIN_ACTION (treepm_compute_density);

range<int64_t> treepm_get_fourier_box(int Nres) {
	ALWAYS_ASSERT(Nres >= get_options().p3m_Nmin);
	auto rbox = domains_find_my_box();
	range<int64_t> box;
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = rbox.begin[dim] * Nres + 0.5;
		box.end[dim] = rbox.end[dim] * Nres + 0.5;
	}
	return box;
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

void treepm_compute_gravity(bool do_phi) {
	const size_t nparts = particles_active_count();
	const double ndim = pow(nparts, 1.0 / NDIM);
	const int Nres = std::max((int) (2.0 * ceil(ndim / 2.0)), get_options().p3m_Nmin);
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
