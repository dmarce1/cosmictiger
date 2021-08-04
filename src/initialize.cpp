#include <tigerfmm/constants.hpp>
#include <tigerfmm/fft.hpp>
#include <tigerfmm/hpx.hpp>
#include <tigerfmm/initialize.hpp>
#include <tigerfmm/options.hpp>
#include <tigerfmm/math.hpp>
#include <tigerfmm/particles.hpp>

#include <gsl/gsl_rng.h>

struct power_spectrum_function {
	vector<float> P;
	float logkmin;
	float logkmax;
	float dlogk;
	float operator()(float k) const {
		float logk = std::log(k);
		const float k0 = (logk - logkmin) / dlogk;
		int i0 = int(k0) - 1;
		i0 = std::max(0, i0);
		i0 = std::min((int) (P.size() - 4), i0);
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
	void normalize() {
		const int N = 8 * 1024;
		double sum = 0.0;
		double dlogk = (logkmax - logkmin) / N;
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
	}

};

static void zeldovich_begin(int dim);
static float zeldovich_end(int dim);
static power_spectrum_function read_power_spectrum();

HPX_PLAIN_ACTION(zeldovich_begin);
HPX_PLAIN_ACTION(zeldovich_end);

double growth_factor(double omega_m, float a) {
	const double omega_l = 1.f - omega_m;
	const double a3 = a * sqr(a);
	const double deninv = 1.f / (omega_m + a3 * omega_l);
	const double Om = omega_m * deninv;
	const double Ol = a3 * omega_l * deninv;
	return a * 2.5 * Om / (pow(Om, 4.f / 7.f) - Ol + (1.f + 0.5f * Om) * (1.f + 0.014285714f * Ol));
}

inline cmplx expc(cmplx z) {
	float x, y;
	float t = std::exp(z.real());
	sincosf(z.imag(), &y, &x);
	x *= t;
	y *= t;
	return cmplx(x, y);
}

static void zeldovich_begin(int dim) {
	const int N = get_options().parts_dim;
	const float box_size = get_options().code_to_cm / constants::mpc_to_cm;
	vector<cmplx> Y;
	vector<hpx::future<void>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<zeldovich_begin_action>(c, dim));
	}
	auto power = read_power_spectrum();
	const auto box = fft3d_complex_range();
	Y.resize(box.volume());
	array<int, NDIM> I;
	const float factor = std::pow(box_size, -1.5) * N * N * N;
	vector<hpx::future<void>> futs2;
	for (I[0] = box.begin[0]; I[0] != box.end[0]; I[0]++) {
		futs2.push_back(hpx::async([N,box,box_size,&Y,dim,&power,factor](array<int,NDIM> I) {
			const int seed = I[0] * 1234 + 42;
			gsl_rng * rndgen = gsl_rng_alloc(gsl_rng_taus);
			gsl_rng_set(rndgen, seed);
			const int i = I[0] < N / 2 ? I[0] : I[0] - N;
			const float kx = 2.f * (float) M_PI / box_size * float(i);
			for (I[1] = box.begin[1]; I[1] != box.end[1]; I[1]++) {
				const int j = I[1] < N / 2 ? I[1] : I[1] - N;
				const float ky = 2.f * (float) M_PI / box_size * float(j);
				for (I[2] = box.begin[2]; I[2] != box.end[2]; I[2]++) {
					const int k = I[2] < N / 2 ? I[2] : I[2] - N;
					const int i2 = sqr(i, j, k);
					const int index = box.index(I);
					if (i2 > 0 && i2 < N * N / 4) {
						const float kz = 2.f * (float) M_PI / box_size * float(k);
						const float k2 = kx * kx + ky * ky + kz * kz;
						const float k = sqrtf(kx * kx + ky * ky + kz * kz);
						const float x = gsl_rng_uniform(rndgen);
						const float y = gsl_rng_uniform(rndgen);
						const float K[NDIM] = {kx, ky, kz};
						const auto rand_normal = expc(cmplx(0, 1) * 2.f * float(M_PI) * y) * std::sqrt(-std::log(std::abs(x)));
						Y[index] = rand_normal * std::sqrt(power(k)) * factor * K[dim] / k2;
					} else {
						Y[index] = cmplx(0.f, 0.f);
					}
				}
			}
			gsl_rng_free(rndgen);
		}, I));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	fft3d_accumulate_complex(box, std::move(Y));
	hpx::wait_all(futs.begin(), futs.end());

}

static float zeldovich_end(int dim) {
	float dxmax = 0.0;
	spinlock_type mutex;
	const int N = get_options().parts_dim;
	const float box_size = get_options().code_to_cm / constants::mpc_to_cm;
	const float omega_m = get_options().omega_m;
	const float a0 = 1.0 / (get_options().z0 + 1.0);
	const auto box = fft3d_real_range();
	vector<hpx::future<float>> futs;
	for (auto c : hpx_children()) {
		futs.push_back(hpx::async<zeldovich_end_action>(c, dim));
	}
	const auto Y = fft3d_read_real(box);
	const float D1 = growth_factor(omega_m, a0) / growth_factor(omega_m, 1.0);
	const double Om = omega_m / (omega_m + (a0 * a0 * a0) * (1.0 - omega_m));
	const double f1 = std::pow(Om, 5.f / 9.f);
	const double H0 = constants::H0 * get_options().code_to_s * get_options().hubble;
	const double H = H0 * std::sqrt(omega_m / (a0 * a0 * a0) + 1.0 - omega_m);
	double prefac1 = f1 * H * a0 * a0;

	if (dim == 0) {
		particles_resize(box.volume());
	}
	array<int, NDIM> I;
	vector<hpx::future<void>> futs1;
	for (I[0] = box.begin[0]; I[0] != box.end[0]; I[0]++) {
		futs1.push_back(hpx::async([box,box_size,D1,prefac1,dim,&Y,N,&mutex,&dxmax](array<int,NDIM> I) {
			const float Ninv = 1.0 / N;
			const float box_size_inv = 1.0 / box_size;
			float this_dxmax = 0.0;
			for (I[1] = box.begin[1]; I[1] != box.end[1]; I[1]++) {
				for (I[2] = box.begin[2]; I[2] != box.end[2]; I[2]++) {
					float x = (I[dim] + 0.5) * Ninv;
					const int index = box.index(I);
					const float dx = -D1 * Y[index] * box_size_inv;
					this_dxmax = std::max(this_dxmax, std::abs(dx * N));
					x += dx;
					if (x >= 1.0) {
						x -= 1.0;
					} else if (x < 0.0) {
						x += 1.0;
					}
					particles_pos(dim, index) = x;
					particles_vel(dim, index) = prefac1 * dx;
					particles_rung(index) = 0;
				}
			}
			std::lock_guard<spinlock_type> lock(mutex);
			dxmax = std::max(dxmax, this_dxmax);
		}, I));
	}
	hpx::wait_all(futs1.begin(), futs1.end());
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
			kmax = std::max(kmax, k);
			kmin = std::min(kmin, k);
		}
	}
	func.logkmin = std::log(kmin);
	func.logkmax = std::log(kmax);
	func.dlogk = (func.logkmax - func.logkmin) / (func.P.size() - 1);
	fclose(fp);
	func.normalize();
	return func;
}

void initialize() {
	const int N = get_options().parts_dim;
	for (int dim = 0; dim < NDIM; dim++) {
		fft3d_init(N);
		zeldovich_begin(dim);
		fft3d_force_real();
		fft3d_inv_execute();
		float dxmax = zeldovich_end(dim);
		fft3d_destroy();
		PRINT("%e\n", dxmax);

	}
}
