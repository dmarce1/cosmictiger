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

#include <cosmictiger/constants.hpp>
#include <cosmictiger/cosmology.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/initialize.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/zero_order.hpp>
#include <cosmictiger/boltzmann.hpp>

#include <gsl/gsl_rng.h>

#define RECFAST_N 1000
#define RECFAST_Z0 9990
#define RECFAST_Z1 0
#define RECFAST_DZ 10

std::function<double(double)> run_recfast(cosmic_params params) {
	std::function<double(double)> func;
	std::string filename = "recfast.in." + std::to_string(hpx_rank());
	FILE* fp = fopen(filename.c_str(), "wb");
	if (fp == NULL) {
		THROW_ERROR("Unable to write to %s\n", filename.c_str());
	}
	filename = "recfast.out." + std::to_string(hpx_rank());
	FILE* fp1 = fopen(filename.c_str(), "rb");
	if (fp1) {
		fclose(fp1);
		std::string command = "rm " + filename + "\n";
		if (system(command.c_str()) != 0) {
			THROW_ERROR("Unable to execute %s\n", command.c_str());
		}
	}
	fprintf(fp, "%s\n", filename.c_str());
	fprintf(fp, "%f %f %f\n", params.omega_b, params.omega_c, 1.0 - params.omega_b - params.omega_c);
	fprintf(fp, "%f %f %f\n", 100 * params.hubble, params.Theta * 2.73, params.Y);
	fprintf(fp, "1\n");
	fprintf(fp, "6\n");
	fclose(fp);
	std::string cmd = "cat recfast.in." + std::to_string(hpx_rank()) + " | ./recfast 1> /dev/null 2> /dev/null";
	if (system(cmd.c_str()) != 0) {
		THROW_ERROR("Unable to run %s\n", cmd.c_str());
	}
	fp = fopen(filename.c_str(), "rb");
	char d1[2];
	char d2[4];
	if (fscanf(fp, " %s %s\n", d1, d2) == 0) {
		THROW_ERROR("unable to read %s\n", filename.c_str());
	}
	std::vector<double> xe;
	for (int i = 0; i < RECFAST_N; i++) {
		float z;
		float this_xe;
		if (fscanf(fp, "%f %f\n", &z, &this_xe) == 0) {
			THROW_ERROR("unable to read %s\n", filename.c_str());
		}
		xe.push_back(this_xe);
	}
	std::vector<double> tmp;
	for (int i = 0; i < RECFAST_N; i++) {
		tmp.push_back(xe.back());
		xe.pop_back();
	}
	xe = std::move(tmp);
	fclose(fp);
	auto inter_func = [xe](double a) {
		const double z = 1.0 / a - 1.0;
		const int i1 = std::min(std::max((int)(z / RECFAST_DZ),1),RECFAST_N-2);
		if( i1 == RECFAST_N - 2 ) {
			return (double) xe.back();
		} else {
			const int i0 = i1 - 1;
			const int i2 = i1 + 1;
			const int i3 = i1 + 2;
			const double t = z / RECFAST_DZ - i1;
			const double y0 = xe[i0];
			const double y1 = xe[i1];
			const double y2 = xe[i2];
			const double y3 = xe[i3];
			const double ct = t * (1.0 - t);
			const double d = -0.5 * ct * ((1.0 - t) * y0 + t * y3);
			const double b = (1.0 - t + ct * (1.0 - 1.5 * t)) * y1;
			const double c = (t + ct * (1.5 * t - 0.5)) * y2;
			return d + b + c;
		}
	};
	return std::move(inter_func);
}

struct power_spectrum_function {
	vector<float> P;
	float logkmin;
	float logkmax;
	float dlogk;
	float operator()(float k) const {
		float logk = std::log(k);
		const float k0 = (logk - logkmin) / dlogk;
		int i0 = int(k0) - 1;
		if (i0 < 0) {
			PRINT("power.init does not have sufficient range min %e max %e tried %e\n", exp(logkmin), exp(logkmax), k);
		}
		if (i0 > P.size() - 4) {
			PRINT("power.init does not have sufficient range min %e max %e tried %e\n", exp(logkmin), exp(logkmax), k);
		}
		int i1 = i0 + 1;
		int i2 = i0 + 2;
		int i3 = i0 + 3;
		float x = k0 - i1;
		float y0 = std::log(P[i0]);
		float y1 = std::log(P[i1]);
		float y2 = std::log(P[i2]);
		float y3 = std::log(P[i3]);
		float k1 = (y2 - y0) * 0.5;
		float k2 = (y3 - y1) * 0.5;
		float a = y1;
		float b = k1;
		float c = -2 * k1 - k2 - 3 * y1 + 3 * y2;
		float d = k1 + k2 + 2 * y1 - 2 * y2;
		return std::exp(a + b * x + c * x * x + d * x * x * x);
	}
	float sigma8_integrand(float x) const {
		float R = 8 / get_options().hubble;
		const float c0 = float(9) / (2. * float(M_PI) * float(M_PI)) / powf(R, 6);
		float k = std::exp(x);
		float P = (*this)(k);
		float tmp = (std::sin(k * R) - k * R * std::cos(k * R));
		return c0 * P * tmp * tmp * std::pow(k, -3);
	}
	float normalize() {
		const int N = 8 * 1024;
		float sum = 0.0;
		float logkmax = this->logkmax - 2.5 * dlogk;
		float logkmin = this->logkmin + 1.5 * dlogk;
		float dlogk = (logkmax - logkmin) / N;
		sum = (1.0 / 3.0) * (sigma8_integrand(logkmax) + sigma8_integrand(logkmin)) * dlogk;
		for (int i = 1; i < N; i += 2) {
			float logk = logkmin + i * dlogk;
			sum += (4.0 / 3.0) * sigma8_integrand(logk) * dlogk;
		}
		for (int i = 2; i < N; i += 2) {
			float logk = logkmin + i * dlogk;
			sum += (2.0 / 3.0) * sigma8_integrand(logk) * dlogk;
		}
		sum = sqr(get_options().sigma8) / sum;
		for (int i = 0; i < P.size(); i++) {
			P[i] *= sum;
		}
		if (hpx_rank() == 0) {
			PRINT("Normalization = %e\n", sum);
		}
		return sum;
	}

};

static void zeldovich_begin(int dim1, int dim2);
static float zeldovich_end(int dim, bool, float, float, float);
static void twolpt_init();
static void twolpt_destroy();
static void twolpt(int, int);
static void twolpt_phase(int phase);
static void twolpt_correction1();
static void twolpt_f2delta2_inv();
static void twolpt_correction2(int dim);
static power_spectrum_function read_power_spectrum();

HPX_PLAIN_ACTION (zeldovich_begin);
HPX_PLAIN_ACTION (zeldovich_end);
HPX_PLAIN_ACTION (twolpt_init);
HPX_PLAIN_ACTION (twolpt_phase);
HPX_PLAIN_ACTION (twolpt_correction1);
HPX_PLAIN_ACTION (twolpt_f2delta2_inv);
HPX_PLAIN_ACTION (twolpt_correction2);
HPX_PLAIN_ACTION (twolpt);
HPX_PLAIN_ACTION (twolpt_destroy);

static vector<float> delta2;
static vector<cmplx> delta2_inv;
static vector<float> delta2_part;

power_spectrum_function compute_power_spectrum() {
	power_spectrum_function func;
	zero_order_universe zeroverse;
	double* result_ptr;
	interp_functor<double> m_k;
	interp_functor<double> vel_k;
	int Nk = 1024;
	vector<cos_state> states(Nk);

	auto& uni = zeroverse;
	cosmic_params params;
	params.omega_b = get_options().omega_b;
	params.omega_c = get_options().omega_c;
	params.omega_gam = get_options().omega_gam;
	params.omega_nu = get_options().omega_nu;
	params.Y = get_options().Y0;
	params.Neff = get_options().Neff;
	params.Theta = get_options().Theta;
	params.hubble = get_options().hubble;
	PRINT("Computing zero order universe...");
	fflush (stdout);
	auto fxe = run_recfast(params);
	create_zero_order_universe(&uni, fxe, 1.1, params);
	PRINT("Done.\n");
	const auto ns = get_options().ns;
	fflush(stdout);
	double kmin;
	double kmax;
	kmin = 1e-5 * params.hubble;
	kmax = 25.271 * params.hubble;
	einstein_boltzmann_interpolation_function(&m_k, &vel_k, states.data(), &zeroverse, kmin, kmax, 1.0, Nk, zeroverse.amin, 1.0, false, ns);

	func.logkmin = log(kmin);
	func.logkmax = log(kmax);
	func.dlogk = (func.logkmax - func.logkmin) / Nk;
	for (int i = 0; i < Nk; i++) {
		double k = exp(func.logkmin + (double) i * func.dlogk);
		func.P.push_back(m_k(k));
	}
	const auto norm = func.normalize();
	if (hpx_rank() == 0) {
		const int M = 2 * Nk;
		const double logkmin = log(kmin);
		const double logkmax = log(kmax);
		const double dlogk = (logkmax - logkmin) / M;
		FILE* fp = fopen("power.dat", "wt");
		const double lh = params.hubble;
		const double lh3 = lh * lh * lh;
		for (int i = 1; i < M - 2; i++) {
			double k = exp(logkmin + (double) i * dlogk);
			fprintf(fp, "%e %e %e\n", k / lh, norm * m_k(k) * lh3, norm * vel_k(k) * lh3);
		}
		fclose(fp);
	}
	return func;
}

void initialize_glass() {
	if (hpx_size() > 1) {
		PRINT("Glass can only run on one processor!\n");
		abort();
	}
	const int nthreads = hpx_hardware_concurrency();
	const part_int part_dim = get_options().parts_dim;
	const size_t nparts = sqr(part_dim) * part_dim;
	vector<hpx::future<void>> futs;
	particles_resize(nparts);
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads,nparts]() {
			gsl_rng * rndgen = gsl_rng_alloc(gsl_rng_taus);
			gsl_rng_set(rndgen, proc+42);
			const part_int b = (size_t) proc * nparts / nthreads;
			const part_int e = (size_t) (proc+1) * nparts / nthreads;
			for( part_int i = b; i < e; i++) {
				const double x = gsl_rng_uniform_pos(rndgen);
				const double y = gsl_rng_uniform_pos(rndgen);
				const double z = gsl_rng_uniform_pos(rndgen);
				particles_pos(XDIM,i) = x;
				particles_pos(YDIM,i) = y;
				particles_pos(ZDIM,i) = z;
				particles_vel(XDIM,i) = 0;
				particles_vel(YDIM,i) = 0;
				particles_vel(ZDIM,i) = 0;
				particles_rung(i) = 0;
			}
			gsl_rng_free(rndgen);
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

void initialize(double z0) {
	const int64_t N = get_options().parts_dim;
	const float omega_m = get_options().omega_m;
	const float a0 = 1.0 / (z0 + 1.0);
	const float D1 = cosmos_growth_factor(omega_m, a0) / cosmos_growth_factor(omega_m, 1.0);
	const float Om = omega_m / (omega_m + (a0 * a0 * a0) * (1.0 - omega_m));
	const float f1 = std::pow(Om, 5.f / 9.f);
	const float H0 = constants::H0 * get_options().code_to_s * get_options().hubble;
	const float H = H0 * std::sqrt(omega_m / (a0 * a0 * a0) + 1.0 - omega_m);
	float prefac1 = f1 * H * a0 * a0;
	const float D2 = -3.f * sqr(D1) / 7.f;
	const float f2 = 2.f * std::pow(Om, 6.f / 11.f);
	float prefac2 = f2 * H * a0 * a0;
	PRINT("D1 = %e\n", D1);
	PRINT("D2 = %e\n", D2);
	PRINT("prefac1 = %e\n", prefac1 / a0);
	PRINT("prefac2 = %e\n", prefac2 / a0);
	for (int dim = 0; dim < NDIM; dim++) {
		fft3d_init(N);
		zeldovich_begin(dim, NDIM);
//		fft3d_force_real();
		fft3d_inv_execute();
		float mask = get_options().twolpt ? 0.0 : 1.0;
		float dxmax = zeldovich_end(dim, dim == 0, D1, prefac1, mask);
		fft3d_destroy();
		PRINT("%e\n", dxmax);
	}
	if (get_options().twolpt) {
		twolpt_init();

		PRINT("2LPT phase 1\n");
		twolpt(1, 1);
		twolpt_phase(0);
		fft3d_destroy();

		PRINT("2LPT phase 2\n");
		twolpt(2, 2);
		twolpt_phase(1);
		fft3d_destroy();

		PRINT("2LPT phase 3\n");
		twolpt(0, 0);
		twolpt_phase(2);
		fft3d_destroy();

		PRINT("2LPT phase 4\n");
		twolpt(2, 2);
		twolpt_phase(3);
		fft3d_destroy();

		PRINT("2LPT phase 5\n");
		twolpt(0, 0);
		twolpt_phase(4);
		fft3d_destroy();

		PRINT("2LPT phase 6\n");
		twolpt(1, 1);
		twolpt_phase(5);
		fft3d_destroy();

		PRINT("2LPT phase 7\n");
		twolpt(0, 1);
		twolpt_phase(6);
		fft3d_destroy();

		PRINT("2LPT phase 8\n");
		twolpt(0, 2);
		twolpt_phase(7);
		fft3d_destroy();

		PRINT("2LPT phase 9\n");
		twolpt(1, 2);
		twolpt_phase(8);
		fft3d_destroy();

		twolpt_correction1();
		twolpt_f2delta2_inv();
		fft3d_destroy();

		for (int dim = 0; dim < NDIM; dim++) {
			PRINT("Computing 2LPT correction to %c positions and velocities\n", 'x' + dim);
			twolpt_correction2(dim);
			float dxmax = zeldovich_end(dim, false, -D2, prefac2, 1.0);
			fft3d_destroy();
			PRINT("%e\n", dxmax);
		}
		twolpt_destroy();
	}
}

void twolpt_f2delta2_inv() {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<twolpt_f2delta2_inv_action>(HPX_PRIORITY_HI, c));
	}
	const auto box = fft3d_complex_range();
	delta2_inv = fft3d_read_complex(box);
	hpx::wait_all(futs.begin(), futs.end());
}

void twolpt_correction1() {
	const int N = get_options().parts_dim;
	if (hpx_rank() == 0) {
		fft3d_init(N);
	}
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<twolpt_correction1_action>(HPX_PRIORITY_HI, c));
	}
	const auto box = fft3d_real_range();
	fft3d_accumulate_real(box, delta2);
//	fft3d2silo(true);
	hpx::wait_all(futs.begin(), futs.end());
	if (hpx_rank() == 0) {
		fft3d_execute();
	}
}

void twolpt_correction2(int dim) {
	const int N = get_options().parts_dim;
	if (hpx_rank() == 0) {
		fft3d_init(N);
	}
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<twolpt_correction2_action>(HPX_PRIORITY_HI, c, dim));
	}
	const float box_size = get_options().code_to_cm / constants::mpc_to_cm;
	const float factor = std::pow(box_size, -1.5) * N * N * N;
	const auto box = fft3d_complex_range();
	auto Y = delta2_inv;
	array<int64_t, NDIM> I;
	vector<hpx::future<void>> futs2;
	for (I[0] = box.begin[0]; I[0] != box.end[0]; I[0]++) {
		futs2.push_back(hpx::async([N,box,box_size,&Y,dim,factor](array<int64_t,NDIM> I) {
			const int i = (I[0] < N / 2 ? I[0] : I[0] - N);
			const float kx = 2.f * (float) M_PI / box_size * float(i);
			for (I[1] = box.begin[1]; I[1] != box.end[1]; I[1]++) {
				const int j = (I[1] < N / 2 ? I[1] : I[1] - N);
				const float ky = 2.f * (float) M_PI / box_size * float(j);
				for (I[2] = box.begin[2]; I[2] != box.end[2]; I[2]++) {
					const int k = (I[2] < N / 2 ? I[2] : I[2] - N);
					const int i2 = sqr(i, j, k);
					const int64_t index = box.index(I);
					if (i2 > 0 && i2 < N * N / 4) {
						float kz = 2.f * (float) M_PI / box_size * float(k);
						float k2 = kx * kx + ky * ky + kz * kz;
						const float K[NDIM] = {kx, ky, kz};
						Y[index] = -cmplx(0, 1) * K[dim] * Y[index] / k2;
					} else {
						Y[index] = cmplx(0, 0);
					}
				}
			}
		}, I));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	fft3d_accumulate_complex(box, Y);
	hpx::wait_all(futs.begin(), futs.end());
	if (hpx_rank() == 0) {
//		fft3d2silo(false);
//		std::string cmd = std::string("cp complex.silo complex.") + std::to_string(dim) + ".silo\n";
//		system(cmd.c_str());
		fft3d_inv_execute();
	}
}

void twolpt_generate(int dim1, int dim2) {
	const int64_t N = get_options().parts_dim;
	const float box_size = get_options().code_to_cm / constants::mpc_to_cm;
	const float factor = std::pow(box_size, -1.5) * N * N * N;
	vector<cmplx> Y;
	static auto power = get_options().use_power_file ? read_power_spectrum() : compute_power_spectrum();
	const auto box = fft3d_complex_range();
	Y.resize(box.volume());
	array<int64_t, NDIM> I;
	constexpr int rand_init_iters = 8;
	vector<hpx::future<void>> futs2;
	for (I[0] = box.begin[0]; I[0] != box.end[0]; I[0]++) {
		futs2.push_back(hpx::async([N,box,box_size,&Y,dim1,dim2,factor](array<int64_t,NDIM> I) {
			const int i = (I[0] < N / 2 ? I[0] : I[0] - N);
			const float kx = 2.f * (float) M_PI / box_size * float(i);
			for (I[1] = box.begin[1]; I[1] != box.end[1]; I[1]++) {
				int seed = (I[0] * N + I[1])*1234 + 42;
				gsl_rng * rndgen = gsl_rng_alloc(gsl_rng_taus);
				gsl_rng_set(rndgen, seed);
				for( int l = 0; l < rand_init_iters; l++) {
					gsl_rng_uniform_pos(rndgen);
				}
				const int j = (I[1] < N / 2 ? I[1] : I[1] - N);
				const float ky = 2.f * (float) M_PI / box_size * float(j);
				for (I[2] = box.begin[2]; I[2] != box.end[2]; I[2]++) {
					const int64_t index = box.index(I);
					if( I[2] > 0 ) {
						const int k = (I[2] < N / 2 ? I[2] : I[2] - N);
						const int i2 = sqr(i, j, k);
						if (i2 > 0 && i2 < N * N / 4) {
							const float kz = 2.f * (float) M_PI / box_size * float(k);
							const float k2 = kx * kx + ky * ky + kz * kz;
							const float K0 = sqrtf(kx * kx + ky * ky + kz * kz);
							const float x = gsl_rng_uniform_pos(rndgen);
							const float y = gsl_rng_uniform_pos(rndgen);
							const cmplx K[NDIM + 1] = { {0,-kx}, {0,-ky}, {0,-kz}, {1,0}};
							const auto rand_normal = expc(cmplx(0, 1) * 2.f * float(M_PI) * y) * sqrtf(-logf(fabsf(x)));
							Y[index] = rand_normal * sqrtf(power(K0)) * factor * K[dim1] * K[dim2] / k2;
						} else {
							Y[index] = cmplx(0.f, 0.f);
						}
					}
				}
				gsl_rng_free(rndgen);
				I[2] = 0;
				const int64_t index = box.index(I);
				const int J0 = I[0] > N / 2 ? N - I[0] : I[0];
				const int J1 = I[1] > N / 2 ? N - I[1] : I[1];
				seed = (J0 * N + J1)*1234 + 42;
				rndgen = gsl_rng_alloc(gsl_rng_taus);
				gsl_rng_set(rndgen, seed);
				for( int l = 0; l < rand_init_iters; l++) {
					gsl_rng_uniform_pos(rndgen);
				}
				const int k = (I[2] < N / 2 ? I[2] : I[2] - N);
				const int i2 = sqr(i, j, k);
				float sgn = 1.0;
				if( dim1 != NDIM) {
					sgn *= -1.0;
				}
				if( dim2 != NDIM) {
					sgn *= -1.0;
				}
				if (i2 > 0 && i2 < N * N / 4) {
					const float kz = 2.f * (float) M_PI / box_size * float(k);
					const float k2 = kx * kx + ky * ky + kz * kz;
					const float K0 = sqrtf(kx * kx + ky * ky + kz * kz);
					const float x = gsl_rng_uniform_pos(rndgen);
					const float y = gsl_rng_uniform_pos(rndgen);
					const cmplx K[NDIM + 1] = { {0,-kx}, {0,-ky}, {0,-kz}, {1,0}};
					const auto rand_normal = expc(cmplx(0, 1) * 2.f * float(M_PI) * y) * sqrtf(-logf(fabsf(x)));
					Y[index] = rand_normal * sqrtf(power(K0)) * factor * K[dim1] * K[dim2] / k2;
					if( I[0] > N / 2 ) {
						Y[index] = Y[index].conj() * sgn;
					} else if( I[0] == 0 ) {
						if( I[1] > N / 2 ) {
							Y[index] = Y[index].conj() * sgn;
						}
					}
				} else {
					Y[index] = cmplx(0.f, 0.f);
				}
				gsl_rng_free(rndgen);

			}
		}, I));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	fft3d_accumulate_complex(box, Y);
}

void twolpt(int dim1, int dim2) {
	const int N = get_options().parts_dim;
	if (hpx_rank() == 0) {
		fft3d_init(N);
	}
	vector<hpx::future<void>> futs;
	if (hpx_rank() == 0) {
		for (int i = 1; i < hpx_size(); i++) {
			futs.push_back(hpx::async<twolpt_action>(hpx_localities()[i], dim1, dim2));
		}
	}
	vector<cmplx> Y;
	const auto box = fft3d_real_range();
	Y.resize(box.volume());
	twolpt_generate(dim1, dim2);
	fft3d_accumulate_complex(box, Y);
	hpx::wait_all(futs.begin(), futs.end());
	if (hpx_rank() == 0) {
//		fft3d_force_real();
		fft3d_inv_execute();
	}
}

void twolpt_phase(int phase) {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<twolpt_phase_action>(HPX_PRIORITY_HI, c, phase));
	}
	const auto box = fft3d_real_range();
	const auto vol = box.volume();
	if (phase > 2 * NDIM) {
		auto this_delta2_part = fft3d_read_real(box);
		for (int i = 0; i < vol; i++) {
			delta2[i] -= this_delta2_part[i] * this_delta2_part[i];
		}
	} else {
		if (phase % 2 == 0) {
			delta2_part = fft3d_read_real(box);
		} else {
			auto delta2_this_part = fft3d_read_real(box);
			for (int i = 0; i < vol; i++) {
				delta2[i] += delta2_part[i] * delta2_this_part[i];
			}
		}
	}

	hpx::wait_all(futs.begin(), futs.end());
}

void twolpt_init() {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<twolpt_init_action>(HPX_PRIORITY_HI, c));
	}
	const auto box = fft3d_real_range();
	delta2.resize(box.volume(), 0.0);
	hpx::wait_all(futs.begin(), futs.end());
}

void twolpt_destroy() {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<twolpt_destroy_action>(HPX_PRIORITY_HI, c));
	}
	delta2 = decltype(delta2)();
	delta2_inv = decltype(delta2_inv)();
	delta2_part = decltype(delta2_part)();
	hpx::wait_all(futs.begin(), futs.end());
}

static void zeldovich_begin(int dim1, int dim2) {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<zeldovich_begin_action>(HPX_PRIORITY_HI, c, dim1, dim2));
	}
	twolpt_generate(dim1, dim2);
	hpx::wait_all(futs.begin(), futs.end());

}

static range<int64_t> find_my_box(range<int64_t> box, int begin, int end, int depth) {
	if (end - begin == 1) {
		return box;
	} else {
		const int xdim = depth % NDIM;
		const int mid = (begin + end) / 2;
		const float w = float(mid - begin) / float(end - begin);
		if (hpx_rank() < mid) {
			auto left = box;
			left.end[xdim] = (((1.0 - w) * box.begin[xdim] + w * box.end[xdim]) + 0.5);
			return find_my_box(left, begin, mid, depth + 1);
		} else {
			auto right = box;
			right.begin[xdim] = (((1.0 - w) * box.begin[xdim] + w * box.end[xdim]) + 0.5);
			return find_my_box(right, mid, end, depth + 1);
		}
	}
}

static range<int64_t> find_my_box() {
	range<int64_t> box;
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = 0;
		box.end[dim] = get_options().parts_dim;
	}
	return find_my_box(box, 0, hpx_size(), 0);
}

static float zeldovich_end(int dim, bool init_parts, float D1, float prefac1, float mask) {
	const bool sph = get_options().sph;
	const bool chem = get_options().chem;
	const float Y0 = get_options().Y0;
	float dxmax = 0.0;
	spinlock_type mutex;
	const int64_t N = get_options().parts_dim;
	const float box_size = get_options().code_to_cm / constants::mpc_to_cm;
	const auto box = find_my_box();
	vector<hpx::future<float>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<zeldovich_end_action>(HPX_PRIORITY_HI, c, dim, init_parts, D1, prefac1, mask));
	}
	const auto Y = fft3d_read_real(box);
	array<int64_t, NDIM> I;
	const float Ninv = 1.0 / N;
	const float box_size_inv = 1.0 / box_size;
	vector<hpx::future<void>> local_futs;
	float entropy;
	if (init_parts) {
		particles_resize(box.volume());
		if (sph) {
			sph_particles_resize(box.volume());

			zero_order_universe uni;
			cosmic_params params;
			params.omega_b = get_options().omega_b;
			params.omega_c = get_options().omega_c;
			params.omega_gam = get_options().omega_gam;
			params.omega_nu = get_options().omega_nu;
			params.Y = get_options().Y0;
			params.Neff = get_options().Neff;
			params.Theta = get_options().Theta;
			params.hubble = get_options().hubble;
			auto fxe = run_recfast(params);
			create_zero_order_universe(&uni, fxe, 1.0, params);
			const float a1 = 1.0 / (get_options().z0 + 1.0);
			const double Kphys = uni.K(a1);
			const double c2g = get_options().code_to_g;
			const double c2s = get_options().code_to_s;
			const double c2cm = get_options().code_to_cm;
			const double c2e = c2g / (c2s * c2s) / c2cm;
			const double c2den = c2g / (c2cm * c2cm * c2cm);
			const double c2ent = c2e / std::pow(c2den, 5.0 / 3.0);
			entropy = Kphys / c2ent;
			PRINT("Entropy conversion factor = %e\n", c2ent);
			PRINT("Initial entropy in code units = %e\n", entropy);
		}
		const float h3 = get_options().neighbor_number / (4.0 / 3.0 * M_PI) / std::pow(get_options().parts_dim, 3);
		const float h = std::pow(h3, 1.0 / 3.0);
		for (I[0] = box.begin[0]; I[0] != box.end[0]; I[0]++) {
			local_futs.push_back(hpx::async([box,Ninv,sph,entropy,h,chem,Y0](array<int64_t,NDIM> I) {
				const float omega_m = get_options().omega_m;
				const float omega_b = get_options().omega_b;
				const float omega = omega_m + omega_b;
				const float dm_shift = sph ? 0.5 * omega_b / get_options().parts_dim / omega: 0.0;
				const float gas_shift = -0.5 * omega_m / get_options().parts_dim / omega;
				for (I[1] = box.begin[1]; I[1] != box.end[1]; I[1]++) {
					for (I[2] = box.begin[2]; I[2] != box.end[2]; I[2]++) {
						const int64_t index = box.index(I);
						for (int dim1 = 0; dim1 < NDIM; dim1++) {
							float x = (I[dim1] + 0.5) * Ninv;
							particles_pos(dim1, index) = x + dm_shift;
							particles_vel(dim1, index) = 0.0;
							if( sph ) {
								sph_particles_pos(dim1, index) = x + gas_shift;
								sph_particles_vel(dim1, index) = 0.0;
							}
						}
						particles_rung(index) = 0;
						if( sph ) {
							sph_particles_rung(index) = 0;
#ifdef SPH_TOTAL_ENERGY
					sph_particles_ent(index) = 0.0;
#else
					sph_particles_ent(index) = entropy;
#endif
					sph_particles_smooth_len(index) = h;
					if( chem ) {
						sph_particles_He0(index) = Y0;
						sph_particles_Z(index) = 1.0e-30f;
						sph_particles_Hp(index) = 1.0e-30f;
						sph_particles_Hn(index) = 1.0e-30f;
						sph_particles_H2(index) = 1.0e-30f;
						sph_particles_Hep(index) = 1.0e-30f;
						sph_particles_Hepp(index) = 1.0e-30f;
					}
				}
			}
		}
	}, I));
		}
		hpx::wait_all(local_futs.begin(), local_futs.end());
		local_futs.resize(0);
	}
	for (I[0] = box.begin[0]; I[0] != box.end[0]; I[0]++) {
		futs.push_back(hpx::async([sph, box, D1, prefac1, mask, dim, &Y, box_size_inv, N](array<int64_t,NDIM> I) {
			float this_dxmax = 0.0;
			for (I[1] = box.begin[1]; I[1] != box.end[1]; I[1]++) {
				for (I[2] = box.begin[2]; I[2] != box.end[2]; I[2]++) {
					const int64_t index = box.index(I);
					float x = particles_pos(dim, index).to_float();
					const float dx = -D1 * Y[index] * box_size_inv;
					if (std::abs(dx * N) > this_dxmax) {
						this_dxmax = std::abs(dx * N);
					}
					x += dx;
					if (x >= 1.0) {
						x -= 1.0;
					} else if (x < 0.0) {
						x += 1.0;
					}
					particles_pos(dim, index) = x;
					particles_vel(dim, index) += prefac1 * dx;
					if( sph ) {
						float x = sph_particles_pos(dim, index).to_float();
						const float dx = -D1 * Y[index] * box_size_inv;
						if (std::abs(dx * N) > this_dxmax) {
							this_dxmax = std::abs(dx * N);
						}
						x += dx;
						if (x >= 1.0) {
							x -= 1.0;
						} else if (x < 0.0) {
							x += 1.0;
						}
						sph_particles_pos(dim, index) = x;
						sph_particles_vel(dim, index) += prefac1 * dx;
#ifdef SPH_TOTAL_ENERGY
				sph_particles_ent(index) += mask*get_options().sph_mass * sqr(sph_particles_vel(dim, index))*0.5f;
#endif
			}
		}
	}
	return this_dxmax;
}, I));
	}
	for (auto& f : futs) {
		const auto tmp = f.get();
		dxmax = std::max(dxmax, tmp);
	}
	return dxmax;
}

static power_spectrum_function read_power_spectrum() {
	power_spectrum_function func;
	const float h = get_options().hubble;
	FILE* fp = fopen("power.init", "rt");
	vector<std::pair<float, float>> data;
	if (!fp) {
		PRINT("Fatal - unable to open power.init\n");
		abort();
	}
	float kmax = 0.0;
	float kmin = std::numeric_limits<float>::max();
	while (!feof(fp)) {
		float k;
		float p;
		if (fscanf(fp, "%f %f\n", &k, &p) != 0) {
			k *= h;
			p /= (h * h * h);
			func.P.push_back(p);
			kmax = std::max(kmax, (float) k);
			kmin = std::min(kmin, (float) k);
		}
	}
	func.logkmin = std::log(kmin);
	func.logkmax = std::log(kmax);
	func.dlogk = (func.logkmax - func.logkmin) / (func.P.size() - 1);
	fclose(fp);
	func.normalize();
	return func;
}
