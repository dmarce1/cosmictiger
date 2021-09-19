#include <cosmictiger/containers.hpp>
#include <cosmictiger/cosmology.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/math.hpp>

#define RECFAST_N 1000
#define RECFAST_Z0 9990
#define RECFAST_Z1 0
#define RECFAST_DZ 10

#define LMAX 32

#define hdoti 0
#define etai 1
#define taui 2
#define deltaci 3
#define deltabi 4
#define thetabi 5
#define FLi 6
#define GLi (6+LMAX)
#define NLi (6+2*LMAX)

#define deltagami (FLi+0)
#define thetagami (FLi+1)
#define F2i (FLi+2)
#define deltanui (NLi+0)
#define thetanui (NLi+1)
#define N2i (NLi+2)
#define G0i (GLi+0)
#define G1i (GLi+1)
#define G2i (GLi+2)

#define NFIELD (6+(3*LMAX))

using cos_state = array<double,NFIELD>;
static double omega_m;
static double omega_c;
static double omega_b;
static double omega_r;
static double omega_nu;
static double omega_gam;
static double little_h;
static double H0 = 100.0;
static double H0cgs;
static double T0;
static double Yp;

double cosmos_hubble(double a) {
	const auto H = constants::H0 * little_h;
	const auto omega_m = get_options().omega_m;
	const auto omega_r = get_options().omega_r;
	const auto omega_lambda = 1.0 - omega_m - omega_r;
	return H * std::sqrt(omega_r / (a * a * a * a) + omega_m / (a * a * a) + omega_lambda);
}

template<class T>
struct interp_functor {
	vector<T> values;
	T amin;
	T amax;
	T minloga;
	T maxloga;
	int N;
	T dloga;
	T operator()(T a) const {
		T loga = logf(a);
		if (loga < minloga || loga > maxloga) {
			PRINT("Error in interpolation_function out of range %e %e %e\n", a, amin, amax);
		}
		int i1 = int((loga - minloga) / (dloga));
		if (i1 > N - 3) {
			i1 = N - 3;
		} else if (i1 < 1) {
			i1 = 1;
		}
		int i0 = i1 - 1;
		int i2 = i1 + 1;
		int i3 = i2 + 1;
		const T c0 = values[i1];
		const T c1 = -values[i0] / (T) 3.0 - (T) 0.5 * values[i1] + values[i2] - values[i3] / (T) 6.0;
		const T c2 = (T) 0.5 * values[i0] - values[i1] + (T) 0.5 * values[i2];
		const T c3 = -values[i0] / (T) 6.0 + (T) 0.5 * values[i1] - (T) 0.5 * values[i2] + values[i3] / (T) 6.0;
		T x = (loga - i1 * dloga - minloga) / dloga;
		return c0 + c1 * x + c2 * x * x + c3 * x * x * x;
	}
};

template<class T>
inline void build_interpolation_function(interp_functor<T>* f, const vector<T>& values, T amin, T amax) {
	T minloga = log(amin);
	T maxloga = log(amax);
	int N = values.size();
	T dloga = (maxloga - minloga) / (N - 1);
	interp_functor<T> functor;
	functor.values = std::move(values);
	functor.maxloga = maxloga;
	functor.minloga = minloga;
	functor.dloga = dloga;
	functor.amin = amin;
	functor.amax = amax;
	functor.N = N;
	*f = functor;
}

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
		//	PRINT("%e %e\n", z, inter_func(loga));
	}
	return std::move(inter_func);
}

double rho_baryon(double a) {
	return (3.0 * omega_b * constants::H0 * constants::H0 * little_h * little_h) / (8.0 * M_PI * constants::G * a * a * a);
}

std::pair<std::function<double(double)>, std::function<double(double)>> zero_order_universe(double amin, double amax,
		const std::function<double(double)> &xefunc) {
	using namespace constants;
	double minloga = log(amin);
	double maxloga = log(amax);
	double loga = minloga;
	constexpr int N = 10000;
	vector<double> sound_speed2(N + 1);
	vector<double> thomson(N + 1);
	double dloga = (maxloga - minloga) / N;
	double Trad = T0 / amin;
	double Tgas = Trad;
	double t = 0.0;
	double rho_b = rho_baryon(amin);
	double nnuc = rho_b / constants::mh;
	double nHe = Yp * nnuc / 4;
	double nH = (1.0 - Yp) * nnuc;
	double ne = nH + 2 * nHe;
	double last_a = amin;
	double a, P1, P2, rho1, rho2, hubble, cs2;
	hubble = cosmos_hubble(a);
	double sigmaT = c * constants::sigma_T * ne / hubble;
	thomson[0] = sigmaT;
	double P = constants::kb * (nH + nHe + ne) * Tgas;
	P1 = P2 = P;
	rho1 = rho2 = rho_b;
	for (int i = 1; i <= N; i++) {
		loga = minloga + i * dloga;
		a = exp(loga);
//		PRINT("%e %e %e %e %e %e\n", a, nH, nHp, nHe, nHep, nHepp);
		P2 = P1;
		P1 = P;
		rho2 = rho1;
		rho1 = rho_b;
		hubble = cosmos_hubble(a);
		nH /= rho_b;
		nHe /= rho_b;
		rho_b = rho_baryon(a);
		nH *= rho_b;
		nHe *= rho_b;
		double ne = xefunc(loga) * nH;
		double Trad = T0 / a;
		double dt = dloga / hubble;
		const double gamma = 1.0 - 1.0 / sqrt(2.0);
		const double mu = (nH + 4 * nHe) * constants::mh / (nH + nHe + ne);
		double sigmaC = mu / constants::me * constants::c * (8.0 / 3.0) * omega_gam / (a * omega_m) * constants::sigma_T * ne / hubble;
		const double dTgasdT1 = ((Tgas + gamma * dloga * sigmaC * Trad) / (1 + gamma * dloga * (2 + sigmaC)) - Tgas) / (gamma * dloga);
		const double T1 = Tgas + (1 - 2 * gamma) * dTgasdT1 * dloga;
		const double dTgasdT2 = ((T1 + gamma * dloga * sigmaC * Trad) / (1 + gamma * dloga * (2 + sigmaC)) - T1) / (gamma * dloga);
		Tgas += 0.5 * (dTgasdT1 + dTgasdT2) * dloga;
		double n = nH + nHe;
		P = constants::kb * (n + ne) * Tgas;
		sigmaT = constants::c * sigma_T * ne / hubble;
		t += dt;
		if (i == 1) {
			cs2 = (P - P1) / (rho_b - rho1);
		} else {
			cs2 = (P - P2) / (rho_b - rho2);
		}
		sound_speed2[i - 1] = cs2 / sqr(constants::c);
		thomson[i] = sigmaT;
		//	PRINT( "%e %e %e\n", a, Tgas, Trad);
	}
	cs2 = (P - P1) / (rho_b - rho1);
	sound_speed2[N - 1] = cs2 / c;
	interp_functor<double> fcs2, fthom;
	build_interpolation_function(&fcs2, sound_speed2, amin, amax);
	build_interpolation_function(&fthom, thomson, amin, amax);
	auto f1 = [fcs2](double a) {
		return fcs2(a);
	};
	auto f2 = [fthom](double a) {
		return fthom(a);
	};
	return std::make_pair(f1, f2);
}

void einstein_boltzmann_init(cos_state* uptr, double k, double normalization, double a, double ns) {
	cos_state& U = *uptr;
	double Oc, Ob, Ogam, Onu;
	double Or = omega_r / (omega_r + a * omega_m + (a * a * a * a) * ((float) 1.0 - omega_m - omega_r));
	double Om = omega_m / (omega_r / a + omega_m + (a * a * a) * ((float) 1.0 - omega_m - omega_r));
	Ogam = omega_gam * Or / omega_r;
	Onu = omega_nu * Or / omega_r;
	Ob = omega_b * Om / omega_m;
	Oc = omega_c * Om / omega_m;
	double hubble = 1e5/3e10 * little_h * H0 * sqrt(omega_r / (a * a * a * a) + omega_m / (a * a * a) + (1 - omega_r - omega_m));
	double eps = k / (a * hubble);
	double C = normalization * pow(eps, -1.5f) * powf(eps, (ns - 1.f) * 0.5f);
	double Rnu = Onu / Or;
	U[taui] = (double) 1.0 / (a * hubble);
	U[deltanui] = U[deltagami] = -(double) 2.0 / (double) 3.0 * C * eps * eps;
	U[deltabi] = U[deltaci] = (double) 3.0 / (double) 4.0 * U[deltagami];
	U[thetabi] = U[thetagami] = -C / (double) 18.0 * eps * eps * eps;
	U[thetanui] = ((double) 23 + (double) 4 * Rnu) / ((double) 15 + (double) 4 * Rnu) * U[thetagami];
	U[N2i] = (double) 0.5 * ((double) 4.0 * C) / ((double) 3.0 * ((double) 15 + (double) 4 * Rnu)) * eps * eps;
	U[hdoti] = (double) (double) 2.0 * C * eps * eps;
	U[G0i] = U[G1i] = U[G2i] = U[F2i] = (double) 0.0;
	for (int l = 3; l < LMAX; l++) {
		U[FLi + l] = (double) 0.0;
		U[NLi + l] = (double) 0.0;
		U[GLi + l] = (double) 0.0;
	}
	U[etai] = ((double) 0.5 * U[hdoti] - ((double) 1.5 * (Oc * U[deltaci] + Ob * U[deltabi]) + (double) 1.5 * (Ogam * U[deltagami] + Onu * U[deltanui])))
			/ (eps * eps);
}

void einstein_boltzmann(cos_state* uptr, std::function<double(double)> fcs2, std::function<double(double)> fsigma_T, double k, double amin, double amax) {
	cos_state& U = *uptr;
	cos_state U0;
	double loga = log(amin);
	double logamax = log(amax);
	while (loga < logamax) {
		double Oc, Ob, Ogam, Onu;
		double a = exp(loga);
		double Or = omega_r / (omega_r + a * omega_m + (a * a * a * a) * ((float) 1.0 - omega_m - omega_r));
		double Om = omega_m / (omega_r / a + omega_m + (a * a * a) * ((float) 1.0 - omega_m - omega_r));
		Ogam = omega_gam * Or / omega_r;
		Onu = omega_nu * Or / omega_r;
		Ob = omega_b * Om / omega_m;
		Oc = omega_c * Om / omega_m;
		double hubble = 1e5/3e10 * little_h * H0 * sqrt(omega_r / (a * a * a * a) + omega_m / (a * a * a) + (1 - omega_r - omega_m));
		double eps = k / (a * hubble);
		double cs2 = fcs2(a);
		double lambda_i = 0.0;
		lambda_i = std::max(lambda_i, sqrt(((double) LMAX + (double) 1.0) / ((double) LMAX + (double) 3.0)) * eps);
		lambda_i = std::max(lambda_i, sqrt((double) 3.0 * pow(eps, 4) + (double) 8.0 * eps * eps * Or) / sqrt((double) 5) / eps);
		double lambda_r = (eps + sqrt(eps * eps + (double) 4.0 * cs2 * pow(eps, (double) 4))) / ((double) 2.0 * eps);
		double dloga_i = (double) 2.0 * (double) 1.73 / lambda_i;
		double dloga_r = (double) 2.0 * (double) 2.51 / lambda_r;
		double dloga = std::min(std::min((double) 5e-2, std::min((double) 0.9 * dloga_i, (double) 0.1 * dloga_r)), logamax - loga);
		double loga0 = loga;
		const auto compute_explicit =
				[&](int step) {
					U0 = U;
					cos_state dudt;
					constexpr double beta[3] = {1, 0.25, (2.0 / 3.0)};
					constexpr double tm[3] = {0, 1, 0.5};
					for (int i = 0; i < 3; i++) {
						loga = loga0 + (double) 0.5 * (tm[i] + step) * dloga;
						a = exp(loga);
						double Or = omega_r / (omega_r + a * omega_m + (a * a * a * a) * ((float) 1.0 - omega_m - omega_r));
						double Om = omega_m / (omega_r / a + omega_m + (a * a * a) * ((float) 1.0 - omega_m - omega_r));
						Ogam = omega_gam * Or / omega_r;
						Onu = omega_nu * Or / omega_r;
						Ob = omega_b * Om / omega_m;
						Oc = omega_c * Om / omega_m;
						hubble = 1e5/3e10 * little_h * H0 * sqrt(omega_r / (a * a * a * a) + omega_m / (a * a * a) + (1 - omega_r - omega_m));
						eps = k / (a * hubble);
						Or = Ogam + Onu;
						cs2 = fcs2(a);
						dudt[taui] = (double) 1.0 / (a * hubble);
						dudt[etai] = ((double) 1.5 * ((Ob * U[thetabi]) + ((double) 4.0 / (double) 3.0) * (Ogam * U[thetagami] + Onu * U[thetanui])) / eps);
						double factor = ((a * omega_m) + (double) 4 * a * a * a * a * ((double) 1 - omega_m - omega_r))
						/ ((double) 2 * a * omega_m + (double) 2 * omega_r + (double) 2 * a * a * a * a * ((double) 1 - omega_m - omega_r));
						dudt[hdoti] =
						(-factor * U[hdoti] - ((double) 3.0 * (Oc * U[deltaci] + Ob * U[deltabi]) + (double) 6.0 * (Ogam * U[deltagami] + Onu * U[deltanui])));
						dudt[deltaci] = -(double) 0.5 * U[hdoti];
						dudt[deltabi] = -eps * U[thetabi] - (double) 0.5 * U[hdoti];
						dudt[deltagami] = -(double) 4.0 / (double) 3.0 * eps * U[thetagami] - ((double) 2.0 / (double) 3.0) * U[hdoti];
						dudt[deltanui] = -(double) 4.0 / (double) 3.0 * eps * U[thetanui] - ((double) 2.0 / (double) 3.0) * U[hdoti];
						dudt[thetabi] = -U[thetabi] + cs2 * eps * U[deltabi];
						dudt[thetagami] = eps * ((double) 0.25 * U[deltagami] - (double) 0.5 * U[F2i]);
						dudt[thetanui] = eps * ((double) 0.25 * U[deltanui] - (double) 0.5 * U[N2i]);
						dudt[F2i] = ((double) 8.0 / (double) 15.0) * eps * U[thetagami] + ((double) 4.0 / (double) 15.0) * U[hdoti] + ((double) 8.0 / (double) 5.0) * dudt[etai]
						- ((double) 3.0 / (double) 5.0) * eps * U[FLi + 3];
						dudt[N2i] = ((double) 8.0 / (double) 15.0) * eps * U[thetanui] + ((double) 4.0 / (double) 15.0) * U[hdoti] + ((double) 8.0 / (double) 5.0) * dudt[etai]
						- ((double) 3.0 / (double) 5.0) * eps * U[NLi + 3];
						dudt[GLi + 0] = -eps * U[GLi + 1];
						dudt[GLi + 1] = eps / (double) (3) * (U[GLi + 0] - (double) 2 * U[GLi + 2]);
						dudt[GLi + 2] = eps / (double) (5) * ((double) 2 * U[GLi + 1] - (double) 3 * U[GLi + 3]);
						for (int l = 3; l < LMAX - 1; l++) {
							dudt[FLi + l] = eps / (double) (2 * l + 1) * ((double) l * U[FLi - 1 + l] - (double) (l + 1) * U[FLi + 1 + l]);
							dudt[NLi + l] = eps / (double) (2 * l + 1) * ((double) l * U[NLi - 1 + l] - (double) (l + 1) * U[NLi + 1 + l]);
							dudt[GLi + l] = eps / (double) (2 * l + 1) * ((double) l * U[GLi - 1 + l] - (double) (l + 1) * U[GLi + 1 + l]);
						}
						dudt[FLi + LMAX - 1] = (eps * U[FLi + LMAX - 2]) / (double) (2 * LMAX - 1);
						dudt[NLi + LMAX - 1] = (eps * U[NLi + LMAX - 2]) / (double) (2 * LMAX - 1);
						dudt[GLi + LMAX - 1] = (eps * U[GLi + LMAX - 2]) / (double) (2 * LMAX - 1);
						for (int f = 0; f < NFIELD; f++) {
							U[f] = ((double) 1 - beta[i]) * U0[f] + beta[i] * (U[f] + dudt[f] * dloga * (double) 0.5);
						}
					}
				};

		auto compute_implicit_dudt =
				[&](double loga, double dloga) {
					a = exp(loga);
					double thetab = U[thetabi];
					double thetagam = U[thetagami];
					double F2 = U[F2i];
					double G0 = U[G0i];
					double G1 = U[G1i];
					double G2 = U[G2i];
					double thetab0 = thetab;
					double thetagam0 = thetagam;
					double F20 = F2;
					double G00 = G0;
					double G10 = G1;
					double G20 = G2;
					double sigma = fsigma_T(a);

					thetab = -((-(double) 3 * Ob * thetab0 - (double) 3 * dloga * Ob * sigma * thetab0 - (double) 4 * dloga * Ogam * sigma * thetagam0)
							/ ((double) 3 * Ob + (double) 3 * dloga * Ob * sigma + (double) 4 * dloga * Ogam * sigma));
					thetagam = -((-(double) 3 * dloga * Ob * sigma * thetab0 - (double) 3 * Ob * thetagam0 - (double) 4 * dloga * Ogam * sigma * thetagam0)
							/ ((double) 3 * Ob + (double) 3 * dloga * (double) Ob * sigma + (double) 4 * dloga * Ogam * sigma));
					F2 = -((-(double) 10 * F20 - (double) 4 * dloga * F20 * sigma - dloga * G00 * sigma - dloga * G20 * sigma)
							/ (((double) 1 + dloga * sigma) * ((double) 10 + (double) 3 * dloga * sigma)));
					G0 = -((-(double) 10 * G00 - (double) 5 * dloga * F20 * sigma - (double) 8 * dloga * G00 * sigma - (double) 5 * dloga * G20 * sigma)
							/ (((double) 1 + dloga * sigma) * ((double) 10 + (double) 3 * dloga * sigma)));
					G1 = G10 / ((double) 1 + dloga * sigma);
					G2 = -((-(double) 10 * G20 - dloga * F20 * sigma - dloga * G00 * sigma - (double) 4 * dloga * G20 * sigma)
							/ (((double) 1 + dloga * sigma) * ((double) 10 + (double) 3 * dloga * sigma)));
					array<double, NFIELD> dudt;
					for (int f = 0; f < NFIELD; f++) {
						dudt[f] = (double) 0.0;
					}
					dudt[thetabi] = (thetab - thetab0) / dloga;
					dudt[thetagami] = (thetagam - thetagam0) / dloga;
					dudt[F2i] = (F2 - F20) / dloga;
					dudt[G0i] = (G0 - G00) / dloga;
					dudt[G1i] = (G1 - G10) / dloga;
					dudt[G2i] = (G2 - G20) / dloga;
					for (int l = 3; l < LMAX - 1; l++) {
						dudt[GLi + l] = U[GLi + l] * ((double) 1 / ((double) 1 + dloga * sigma) - (double) 1) / dloga;
						dudt[FLi + l] = U[FLi + l] * ((double) 1 / ((double) 1 + dloga * sigma) - (double) 1) / dloga;
					}
					dudt[GLi + LMAX - 1] = U[GLi + LMAX - 1]
					* ((double) 1 / ((double) 1 + (sigma + (double) LMAX / (U[taui] * a * hubble) / ((double) 2 * (double) LMAX - (double) 1))) - (double) 1) / dloga;
					dudt[FLi + LMAX - 1] = U[FLi + LMAX - 1]
					* ((double) 1 / ((double) 1 + (sigma + (double) LMAX / (U[taui] * a * hubble) / ((double) 2 * (double) LMAX - (double) 1))) - (double) 1) / dloga;
					return dudt;
				};

		compute_explicit(0);
		double gamma = (double) 1.0 - (double) 1.0 / sqrt((double) 2);

		auto dudt1 = compute_implicit_dudt(loga + gamma * dloga, gamma * dloga);
		for (int f = 0; f < NFIELD; f++) {
			U[f] += dudt1[f] * ((double) 1.0 - (double) 2.0 * gamma) * dloga;
		}
		auto dudt2 = compute_implicit_dudt(loga + ((double) 1.0 - gamma) * dloga, gamma * dloga);
		for (int f = 0; f < NFIELD; f++) {
			U[f] += (dudt1[f] * ((double) -0.5 + (double) 2.0 * gamma) + dudt2[f] * (double) 0.5) * dloga;
		}

		compute_explicit(1);

		loga = loga0 + dloga;
	}
}
void eb_compute() {
	omega_b = get_options().omega_b;
	omega_c = get_options().omega_c;
	omega_gam = get_options().omega_gam;
	omega_nu = get_options().omega_nu;
	omega_m = omega_c + omega_b;
	omega_r = omega_gam + omega_nu;
	H0cgs = H0 * 100000.0 / constants::mpc_to_cm;
	little_h = get_options().hubble;
	T0 = 2.73 * get_options().Theta;
	Yp = get_options().Yp;
	const auto xe = run_recfast();
	double amin = 1.0e-6;
	auto rc = zero_order_universe(amin, 1.0, xe);
	amin *= 10.0;
	int Nk = 1024;
	double kmin = 0.10000E-03;
	double kmax = 0.25271E+02;
	kmin *= little_h;
	kmax *= little_h;
	double logkmin = log(kmin);
	double logkmax = log(kmax);
	double dlogk = (logkmax - logkmin) / (Nk - 1);
	cos_state U;
	for (double logk = logkmin; logk <= logkmax; logk += dlogk) {
		const double k = exp(logk);
		einstein_boltzmann_init(&U, k, 1.0, amin, 1.0);
		einstein_boltzmann(&U, rc.first, rc.second, exp(logk), amin, 1.0);
		PRINT("%e %e\n", k / little_h, sqr((omega_c * U[deltaci] + omega_b * U[deltabi]) / omega_m) * sqr(little_h) * little_h);
	}

}
