#include <cosmictiger/containers.hpp>
#include <cosmictiger/options.hpp>

#define RECFAST_N 1000
#define RECFAST_Z0 9990
#define RECFAST_Z1 0
#define RECFAST_DZ 10

static double omega_m;
static double omega_c;
static double omega_b;
static double omega_r;
static double omega_nu;
static double omega_gam;
static double little_h;
static double H0 = 100.0;
static double T0;
static double Yp;

std::function<double(double)> run_recfast() {
	std::function<double(double)> func;
	vector<std::pair<double, double>> res;
	FILE* fp = fopen("recfast.out", "rb");
	if (fp != NULL) {
		fclose(fp);
		if (system("rm recfast.out") != 0) {
			THROW_ERROR("Unable to erase recfast.out\n");
		}
	}
	fp = fopen("recfast.in", "wb");
	if (fp == NULL) {
		THROW_ERROR("Unable to write to recfast.in\n");
	}
	fprintf(fp, "recfast.out\n");
	fprintf(fp, "%f %f %f\n", omega_b, omega_c, 1.0 - omega_m);
	fprintf(fp, "%f %f %f\n", H0 * little_h, T0, Yp);
	fprintf(fp, "1\n");
	fprintf(fp, "6\n");
	fclose(fp);
	if (system("cat recfast.in | ./recfast 1> /dev/null 2> /dev/null") != 0) {
		THROW_ERROR("Unable to run RECFAST\n");
	}
	fp = fopen("recfast.out", "rb");
	char d1[2];
	char d2[4];
	if (fscanf(fp, " %s %s\n", d1, d2) == 0) {
		THROW_ERROR("unable to read recfast.out\n");
	}
	vector<float> xe;
	for (int i = 0; i < RECFAST_N; i++) {
		float z;
		float this_xe;
		if (fscanf(fp, "%f %f\n", &z, &this_xe) == 0) {
			THROW_ERROR("unable to read recfast.out\n");
		}
		xe.push_back(this_xe);
	}
	vector<float> tmp;
	for (int i = 0; i < RECFAST_N; i++) {
		tmp.push_back(xe.back());
		xe.pop_back();
	}
	xe = std::move(tmp);
	fclose(fp);
	auto inter_func = [xe](double loga) {
		const double a = exp(loga);
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
	for (double loga = log(1.0e-6); loga < 0.0; loga += 0.01) {
		const double z = 1.0 / exp(loga) - 1.0;
		PRINT("%e %e\n", z, inter_func(loga));
	}
	return std::move(inter_func);
}

void eb_compute() {
	omega_b = get_options().omega_b;
	omega_c = get_options().omega_c;
	omega_gam = get_options().omega_gam;
	omega_nu = get_options().omega_nu;
	omega_m = omega_c + omega_b;
	omega_r = omega_gam + omega_nu;
	little_h = get_options().hubble;
	T0 = 2.73 * get_options().Theta;
	Yp = get_options().Yp;
	run_recfast();
}
