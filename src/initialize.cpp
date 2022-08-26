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
static float zeldovich_end(int dim, bool, float, float);
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
static vector<fixed32> X0[NDIM];

static void init_X0();

power_spectrum_function compute_power_spectrum() {
	power_spectrum_function func;
	zero_order_universe zeroverse;
	double* result_ptr;
	cosmic_params params;
	interp_functor<double> m_k;
	interp_functor<double> vel_k;
	int Nk = 1024;
	vector<cos_state> states(Nk);

	auto& uni = zeroverse;
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

void initialize(double z0) {
	init_X0();
	const int64_t N = get_options().Nfour;
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
		float dxmax = zeldovich_end(dim, dim == 0, D1, prefac1);
		PRINT("dxmax = %e\n", dxmax);
		fft3d_destroy();
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
//		fft3d2silo(false);
//		system( "cp complex.silo complex.3.silo\n");
		fft3d_destroy();

		for (int dim = 0; dim < NDIM; dim++) {
			PRINT("Computing 2LPT correction to %c positions and velocities\n", 'x' + dim);
			twolpt_correction2(dim);
			float dxmax = zeldovich_end(dim, false, -D2, prefac2);
			fft3d_destroy();
			PRINT("%e\n", dxmax);
		}
		twolpt_destroy();
	}
	for (int dim = 0; dim < NDIM; dim++) {
		X0[dim] = vector<fixed32>();
	}
}

void twolpt_f2delta2_inv() {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<twolpt_f2delta2_inv_action>(c));
	}
	const auto box = fft3d_complex_range();
	delta2_inv = fft3d_read_complex(box);
	hpx::wait_all(futs.begin(), futs.end());
}

void twolpt_correction1() {
	const int N = get_options().Nfour;
	if (hpx_rank() == 0) {
		fft3d_init(N);
	}
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<twolpt_correction1_action>(c));
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
	const int N = get_options().Nfour;
	if (hpx_rank() == 0) {
		fft3d_init(N);
	}
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<twolpt_correction2_action>(c, dim));
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
	const int64_t N = get_options().Nfour;
	const float box_size = get_options().code_to_cm / constants::mpc_to_cm;
	const float factor = std::pow(box_size, -1.5) * N * N * N;
	vector<cmplx> Y;
	static auto power = get_options().use_power_file ? read_power_spectrum() : compute_power_spectrum();
	const auto box = fft3d_complex_range();
	Y.resize(box.volume());
	array<int64_t, NDIM> I;
	constexpr int rand_init_iters = 8;
	vector<hpx::future<void>> futs2;
//	const auto filter = get_options().use_glass || get_options().close_pack;
	const bool filter = true;
	for (I[0] = box.begin[0]; I[0] != box.end[0]; I[0]++) {
		futs2.push_back(hpx::async([N,box,box_size,&Y,dim1,dim2,factor,filter](array<int64_t,NDIM> I) {
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
						const float kz = 2.f * (float) M_PI / box_size * float(k);
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
						if( filter ) {
							Y[index] *= cloud_filter( kx * box_size / N);
							Y[index] *= cloud_filter( ky * box_size / N);
							Y[index] *= cloud_filter( kz * box_size / N);
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
				const float kz = 2.f * (float) M_PI / box_size * float(k);
				if (i2 > 0 && i2 < N * N / 4) {
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
				if( filter ) {
					Y[index] *= cloud_filter( kx * box_size / N);
					Y[index] *= cloud_filter( ky * box_size / N);
					Y[index] *= cloud_filter( kz * box_size / N);
				}
				gsl_rng_free(rndgen);

			}
		}, I));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	fft3d_accumulate_complex(box, Y);
}

void twolpt(int dim1, int dim2) {
	const int N = get_options().Nfour;
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
		futs.push_back(hpx::async<twolpt_phase_action>(c, phase));
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
		futs.push_back(hpx::async<twolpt_init_action>(c));
	}
	const auto box = fft3d_real_range();
	delta2.resize(box.volume(), 0.0);
	hpx::wait_all(futs.begin(), futs.end());
}

void twolpt_destroy() {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<twolpt_destroy_action>(c));
	}
	delta2 = decltype(delta2)();
	delta2_inv = decltype(delta2_inv)();
	delta2_part = decltype(delta2_part)();
	hpx::wait_all(futs.begin(), futs.end());
}

static void zeldovich_begin(int dim1, int dim2) {
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<zeldovich_begin_action>(c, dim1, dim2));
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
		box.end[dim] = get_options().Nfour;
	}
	return find_my_box(box, 0, hpx_size(), 0);
}

static void init_X0() {
	vector<hpx::future<void>> futs;
	const double Ninv = 1.0 / get_options().parts_dim;
	const auto ibox = find_my_box();
	const auto glass = get_options().use_glass;
	const auto close_pack = get_options().close_pack;
	if (!glass) {
		auto box = find_my_box();
		array<int64_t, NDIM> I;
		part_int index = 0;
		size_t total = box.volume();
		if (close_pack) {
			total *= 2;
		}
		for (int dim = 0; dim < NDIM; dim++) {
			X0[dim].resize(total);
		}
		for (I[0] = box.begin[0]; I[0] != box.end[0]; I[0]++) {
			for (I[1] = box.begin[1]; I[1] != box.end[1]; I[1]++) {
				for (I[2] = box.begin[2]; I[2] != box.end[2]; I[2]++) {
					if (close_pack) {
						for (int dim1 = 0; dim1 < NDIM; dim1++) {
							double x = I[dim1] * Ninv;
							X0[dim1][index] = x + 0.25;
							X0[dim1][index + 1] = x + 0.5 * Ninv;
						}
						index += 2;
					} else {
						for (int dim1 = 0; dim1 < NDIM; dim1++) {
							double x = I[dim1] * Ninv;
							X0[dim1][index] = x;// + dim1 * 0.123;
						}
						index++;
					}
				}
			}
		}
	} else {
		range<double> box;
		FILE* fp = fopen("glass.bin", "rb");
		if (!fp) {
			THROW_ERROR("Unable to open glass.bin\n");
		}
		int glass_parts_dim;
		int parts_dim = get_options().parts_dim;
		for (int dim = 0; dim < NDIM; dim++) {
			box.begin[dim] = (double) ibox.begin[dim] / parts_dim;
			box.end[dim] = (double) ibox.end[dim] / parts_dim;
		}
		FREAD(&glass_parts_dim, sizeof(int), 1, fp);
		if (parts_dim % glass_parts_dim != 0) {
			THROW_ERROR("parts_dim (=%i) must be a multiple of the glass dim (%i)\n", parts_dim, glass_parts_dim);
		}
		int Nf = parts_dim / glass_parts_dim;
		size_t glass_nparts = sqr(glass_parts_dim) * glass_parts_dim;
		static vector<fixed32> x(glass_nparts);
		static vector<fixed32> y(glass_nparts);
		static vector<fixed32> z(glass_nparts);
		FREAD(x.data(), sizeof(fixed32), glass_nparts, fp);
		FREAD(y.data(), sizeof(fixed32), glass_nparts, fp);
		FREAD(z.data(), sizeof(fixed32), glass_nparts, fp);
		fclose(fp);
		const int jb = box.begin[XDIM] * Nf;
		const int lb = box.begin[ZDIM] * Nf;
		const int kb = box.begin[YDIM] * Nf;
		const int je = std::min((int) box.end[XDIM] * Nf, Nf - 1) + 1;
		const int ke = std::min((int) box.end[YDIM] * Nf, Nf - 1) + 1;
		const int le = std::min((int) box.end[ZDIM] * Nf, Nf - 1) + 1;
		mutex_type mutex;
		vector<part_int> counts(je - jb);
		for (int j = jb; j < je; j++) {
			futs.push_back(hpx::async([j,lb,le,jb,kb,ke,glass_nparts,&mutex,box,Nf,&counts]() {
				part_int this_count = 0;
				for (int k = kb; k < ke; k++) {
					for (int l = lb; l < le; l++) {
						for (size_t i = 0; i < glass_nparts; i++) {
							double x0 = x[i].to_double();
							double y0 = y[i].to_double();
							double z0 = z[i].to_double();
							const double X = (j + x0) / Nf;
							if (X >= box.begin[XDIM] && X < box.end[XDIM]) {
								const double Y = (k + y0) / Nf;
								if (Y >= box.begin[YDIM] && Y < box.end[YDIM]) {
									const double Z = (l + z0) / Nf;
									if (Z >= box.begin[ZDIM] && Z < box.end[ZDIM]) {
										this_count++;
									}
								}
							}
						}
					}
				}
				counts[j - jb] = this_count;
			}));
		}
		hpx::wait_all(futs.begin(), futs.end());
		futs.resize(0);
		size_t total = 0;
		for (int i = 0; i < counts.size(); i++) {
			const auto this_count = counts[i];
			counts[i] = total;
			total += this_count;
		}
		for (int dim = 0; dim < NDIM; dim++) {
			X0[dim].resize(total);
		}
		for (int j = jb; j < je; j++) {
			const part_int index_begin = counts[j - jb];
			futs.push_back(hpx::async([j,lb,le,kb,ke,glass_nparts,&mutex,box,Nf, index_begin]() {
				part_int iii = index_begin;
				for (int k = kb; k < ke; k++) {
					for (int l = lb; l < le; l++) {
						for (size_t i = 0; i < glass_nparts; i++) {
							double x0 = x[i].to_double();
							double y0 = y[i].to_double();
							double z0 = z[i].to_double();
							const double X = (j + x0) / Nf;
							if (X >= box.begin[XDIM] && X < box.end[XDIM]) {
								const double Y = (k + y0) / Nf;
								if (Y >= box.begin[YDIM] && Y < box.end[YDIM]) {
									const double Z = (l + z0) / Nf;
									if (Z >= box.begin[ZDIM] && Z < box.end[ZDIM]) {
										X0[XDIM][iii] = X;
										X0[YDIM][iii] = Y;
										X0[ZDIM][iii] = Z;
										iii++;
									}
								}
							}
						}
					}
				}
			}));
		}
	}
	hpx::wait_all(futs.begin(), futs.end());
	particles_resize(X0[XDIM].size());
	for (part_int i = 0; i < X0[XDIM].size(); i++) {
		particles_pos(XDIM, i) = X0[XDIM][i];
		particles_pos(YDIM, i) = X0[YDIM][i];
		particles_pos(ZDIM, i) = X0[ZDIM][i];
		particles_vel(XDIM, i) = 0.0;
		particles_vel(YDIM, i) = 0.0;
		particles_vel(ZDIM, i) = 0.0;
		particles_rung(i) = 0;
	}
}

static float zeldovich_end(int dim, bool init_parts, float D1, float prefac1) {
//PRINT( "enter zeldovich_end\n");
	float dxmax = 0.0;
	spinlock_type mutex;
	const int64_t N = get_options().Nfour;
	const double box_size = get_options().code_to_cm / constants::mpc_to_cm;
	auto box = find_my_box();
	const bool glass = get_options().use_glass;
	for (int dim = 0; dim < NDIM; dim++) {
		box.end[dim] += CLOUD_MAX;
		box.begin[dim] += CLOUD_MIN;
	}
	vector<hpx::future<float>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<zeldovich_end_action>(c, dim, init_parts, D1, prefac1));
	}
	static vector<float> Y;
	Y = fft3d_read_real(box);
	array<int64_t, NDIM> I;
	const float Ninv = 1.0 / N;
	const int nthreads = hpx_hardware_concurrency();
	const double box_size_inv = 1.0 / box_size;
	for (part_int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads, N,box, box_size_inv, dim, D1, prefac1]() {
			const size_t b = (size_t) proc * particles_size() / nthreads;
			const size_t e = (size_t) (proc + 1) * particles_size() / nthreads;
			double dxmax = 0.0;
			for( part_int i = b; i < e; i++) {
				const double x0 = X0[XDIM][i].to_double();
				const double y0 = X0[YDIM][i].to_double();
				const double z0 = X0[ZDIM][i].to_double();
				const int i1 = x0 * N;
				const int j1 = y0 * N;
				const int k1 = z0 * N;
				double t[NDIM] = {x0*N - i1, y0*N - j1, z0*N-k1};
				double dx = 0.0;
				for( int j = CLOUD_MIN; j <= CLOUD_MAX; j++) {
					double wtx = cloud_weight(-j+t[XDIM]);
					for( int k = CLOUD_MIN; k <= CLOUD_MAX; k++) {
						double wty = cloud_weight(-k+t[YDIM]);
						for( int l = CLOUD_MIN; l <= CLOUD_MAX; l++) {
							double wtz = cloud_weight(-l+t[ZDIM]);
							double wt = wtx * wty * wtz;
							dx += Y[box.index(i1+j,j1+k,k1+l)] * wt;
						}
					}
				}
				dx *= -D1 * box_size_inv;
				double x = particles_pos(dim, i).to_double();
				x += dx;
				if (x >= 1.0) {
					x -= 1.0;
				} else if (x < 0.0) {
					x += 1.0;
				}
				particles_pos(dim,i) = x;
				particles_vel(dim, i) += prefac1 * dx;
				const double Ndim = pow(get_options().nparts, 1.0 / NDIM);
				dxmax = std::max(std::abs(dxmax), dx * Ndim);
			}
			return (float) dxmax;
		}));
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
