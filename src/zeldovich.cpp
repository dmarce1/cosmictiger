#include <cosmictiger/zeldovich.hpp>
#include <cosmictiger/particle_server.hpp>
#include <cosmictiger/hpx.hpp>

HPX_PLAIN_ACTION(_2lpt);
HPX_PLAIN_ACTION(_2lpt_correction1);
HPX_PLAIN_ACTION(_2lpt_correction2);
HPX_PLAIN_ACTION(_2lpt_init);
HPX_PLAIN_ACTION(_2lpt_destroy);
HPX_PLAIN_ACTION(_2lpt_phase);
HPX_PLAIN_ACTION(phi1_to_particles);
HPX_PLAIN_ACTION(phi2_to_particles);

static vector<cmplx> delta2;
static std::vector<float> delta2_part;

void _2lpt_init(int N) {
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<_2lpt_init_action>(hpx_localities()[i], N));
		}
	}
	int xbegin = hpx_rank() * N / hpx_size();
	int xend = (hpx_rank() + 1) * N / hpx_size();
	int xspan = xend - xbegin;
	delta2.resize(xspan * N * N);
	for (int i = 0; i < xspan * N * N; i++) {
		delta2[i] = cmplx(0.0, 0.0);
	}
	delta2_part.resize(xspan * N * N, 0.0);
	hpx::wait_all(futs.begin(), futs.end());

}

void _2lpt_destroy() {
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<_2lpt_destroy_action>(hpx_localities()[i]));
		}
	}
	delta2_part = decltype(delta2_part)();
	delta2 = decltype(delta2)();
	hpx::wait_all(futs.begin(), futs.end());

}

void _2lpt_phase(int N, int phase) {
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<_2lpt_phase_action>(hpx_localities()[i], N, phase));
		}
	}
	int xbegin = hpx_rank() * N / hpx_size();
	int xend = (hpx_rank() + 1) * N / hpx_size();
	int xspan = xend - xbegin;
	if (phase > 2 * NDIM) {
		auto this_delta2_part = fourier3d_read_real(xbegin, xend, 0, N, 0, N);
		for (int i = 0; i < xspan * N * N; i++) {
			delta2[i].real() -= this_delta2_part[i] * this_delta2_part[i];
		}
	} else {
		if (phase % 2 == 0) {
			delta2_part = fourier3d_read_real(xbegin, xend, 0, N, 0, N);
		} else {
			auto delta2_this_part = fourier3d_read_real(xbegin, xend, 0, N, 0, N);
			for (int i = 0; i < xspan * N * N; i++) {
				delta2[i].real() += delta2_part[i] * delta2_this_part[i];
			}
		}
	}
	hpx::wait_all(futs.begin(), futs.end());

}

static double constrain_range(double& r) {
	while (r >= 1.0) {
		r -= 1.;
	}
	while (r < 0.) {
		r += 1.;
	}
	return r;
}

float phi1_to_particles(int N, float box_size, float D1, float prefactor, int dim) {
	std::vector<hpx::future<float>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<phi1_to_particles_action>(hpx_localities()[i], N, box_size, D1, prefactor, dim));
		}
	}
	double dmax = 0.0f;
	particle_server pserv;
	particle_set& parts = pserv.get_particle_set();
	int xb = hpx_rank() * N / hpx_size();
	int xe = (hpx_rank() + 1) * N / hpx_size();
	int xspan = xe - xb;
	part_int count = xspan * N * N;
	if (dim == 0) {
		parts.resize(count);
		for (int xi = xb; xi < xe; xi++) {
			const float x = (float(xi) + 0.5f) / float(N);
			for (int yi = 0; yi < N; yi++) {
				const float y = (float(yi) + 0.5f) / float(N);
				for (int zi = 0; zi < N; zi++) {
					const float z = (float(zi) + 0.5f) / float(N);
					part_int index = N * N * (xi - xb) + N * yi + zi;
					parts.pos(0, index) = x;
					parts.pos(1, index) = y;
					parts.pos(2, index) = z;
				}
			}
		}
	}
	auto phi1 = fourier3d_read_real(xb, xe, 0, N, 0, N);
	const float factor = -D1 / box_size;
	for (int xi = xb; xi < xe; xi++) {
		const float x = (float(xi) + 0.5f) / float(N);
		for (int yi = 0; yi < N; yi++) {
			const float y = (float(yi) + 0.5f) / float(N);
			for (int zi = 0; zi < N; zi++) {
				const float z = (float(zi) + 0.5f) / float(N);
				part_int i0 = N * N * (xi - xb) + N * yi + zi;
				const double dx = phi1[i0] * factor;
				double x = parts.pos(dim, i0).to_double() + dx;
				constrain_range(x);
				parts.pos(dim, i0) = x;
				parts.vel(dim, i0) = prefactor * dx;
				dmax = std::max(dmax, dx);
			}
		}
	}
	dmax *= N;
	for (auto& f : futs) {
		dmax = std::max(f.get(), (float) dmax);
	}
	return dmax;
}


float phi2_to_particles(int N, float box_size, float D2, float prefactor, int dim) {
	std::vector<hpx::future<float>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<phi2_to_particles_action>(hpx_localities()[i], N, box_size, D2, prefactor, dim));
		}
	}
	double dmax = 0.0f;
	particle_server pserv;
	particle_set& parts = pserv.get_particle_set();
	int xb = hpx_rank() * N / hpx_size();
	int xe = (hpx_rank() + 1) * N / hpx_size();
	int xspan = xe - xb;
	part_int count = xspan * N * N;
	auto phi2 = fourier3d_read_real(xb, xe, 0, N, 0, N);
	const float factor = D2 / box_size;
	for (int xi = xb; xi < xe; xi++) {
		const float x = (float(xi) + 0.5f) / float(N);
		for (int yi = 0; yi < N; yi++) {
			const float y = (float(yi) + 0.5f) / float(N);
			for (int zi = 0; zi < N; zi++) {
				const float z = (float(zi) + 0.5f) / float(N);
				part_int i0 = N * N * (xi - xb) + N * yi + zi;
				const double dx = phi2[i0] * factor;
				double x = parts.pos(dim, i0).to_double() + dx;
				constrain_range(x);
				parts.pos(dim, i0) = x;
				parts.vel(dim, i0) += prefactor * dx;
				dmax = std::max(dmax, dx);
			}
		}
	}
	dmax *= N;
	for (auto& f : futs) {
		dmax = std::max(f.get(), (float) dmax);
	}
	return dmax;
}

void _2lpt(const interp_functor<float> den_k, int N, float box_size, int dim1, int dim2, int seed) {
	if (hpx_rank() == 0) {
		fourier3d_initialize(N);
	}
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<_2lpt_action>(hpx_localities()[i], den_k, N, box_size, dim1, dim2, seed));
		}
	}
	vector<cmplx> Y;
	int xbegin = hpx_rank() * N / hpx_size();
	int xend = (hpx_rank() + 1) * N / hpx_size();
	int xspan = xend - xbegin;
	Y.resize(xspan * N * N);
	const float factor = std::pow(box_size, -1.5);
	execute_2lpt_kernel(Y.data(), xbegin, xend, den_k, N, box_size, dim1, dim2);
	fourier3d_accumulate(xbegin, xend, 0, N, 0, N, std::move(Y));
	hpx::wait_all(futs.begin(), futs.end());
	if (hpx_rank() == 0) {
		fourier3d_mirror();
		fourier3d_execute();
	}
}
void _2lpt_correction_f2delta2(int N);
HPX_PLAIN_ACTION(_2lpt_correction_f2delta2);

void _2lpt_correction_f2delta2(int N) {
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<_2lpt_correction_f2delta2_action>(hpx_localities()[i], N));
		}
	}
	int xbegin = hpx_rank() * N / hpx_size();
	int xend = (hpx_rank() + 1) * N / hpx_size();
	delta2 = fourier3d_read(xbegin, xend, 0, N, 0, N);
	hpx::wait_all(futs.begin(), futs.end());
}

void _2lpt_correction1(int N, float box_size) {
	if (hpx_rank() == 0) {
		fourier3d_initialize(N);
	}
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<_2lpt_correction1_action>(hpx_localities()[i], N, box_size));
		}
	}
	int xbegin = hpx_rank() * N / hpx_size();
	int xend = (hpx_rank() + 1) * N / hpx_size();
	fourier3d_accumulate(xbegin, xend, 0, N, 0, N, std::move(delta2));
	hpx::wait_all(futs.begin(), futs.end());
	if (hpx_rank() == 0) {
		fourier3d_inv_execute();
	}
}

void _2lpt_correction2(int N, float box_size, int dim) {
	int xbegin = hpx_rank() * N / hpx_size();
	int xend = (hpx_rank() + 1) * N / hpx_size();
	if (hpx_rank() == 0) {
		if (dim == 0) {
			_2lpt_correction_f2delta2(N);
		}
		fourier3d_initialize(N);
	}
	std::vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<_2lpt_correction2_action>(hpx_localities()[i], N, box_size, dim));
		}
	}
	auto Y = delta2;
	execute_2lpt_correction_kernel(Y.data(), xbegin, xend, N, box_size, dim);
	fourier3d_accumulate(xbegin, xend, 0, N, 0, N, std::move(Y));
	hpx::wait_all(futs.begin(), futs.end());
	if (hpx_rank() == 0) {
		fourier3d_execute();
	}
}
