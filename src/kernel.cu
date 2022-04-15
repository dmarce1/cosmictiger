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

#define __KERNEL_CU__
#include <cosmictiger/options.hpp>
#include <cosmictiger/kernel.hpp>

#include <cosmictiger/safe_io.hpp>

void kernel_set_type(double n) {
	const auto w = [n](double r) {
		return pow(sinc(M_PI*r),n);
	};
	constexpr int N = 100000;
	double sum = 0.0;
	const double dr = 1.0 / N;
	for (int i = N - 1; i >= 1; i--) {
		double r = double(i) / N;
		sum += r * r * w(r) * 4.0 * M_PI * dr;
	}
	kernel_norm = 1.0 / sum;
	PRINT("KERNEL NORM = %e\n", kernel_norm);
	kernel_index = n;

	CUDA_CHECK(cudaMallocManaged(&pot_series, sizeof(float) * NTAYLOR));
	CUDA_CHECK(cudaMallocManaged(&f_series, sizeof(float) * NTAYLOR));
#define List(...) {__VA_ARGS__}
#define Gamma tgamma
#define Power pow
#define Sqrt sqrt
#define Pi M_PI
	n = 4.745;
	double coeffs[16] = List((Sqrt(Pi)*Gamma(1 + n/2.))/Gamma((1 + n)/2.),
		   -0.125*(n*Power(Pi,2.5)*Gamma(1 + n/2.))/Gamma((1 + n)/2.),
		   ((-2*n + 3*Power(n,2))*Power(Pi,4.5)*Gamma(1 + n/2.))/(384.*Gamma((1 + n)/2.)),
		   (Sqrt(Pi)*(-0.00002170138888888889*(n*Power(Pi,6)) -
		        ((-1 + n)*n*Power(Pi,6))/3072. - ((-2 + n)*(-1 + n)*n*Power(Pi,6))/3072.)*
		      Gamma(1 + n/2.))/Gamma((1 + n)/2.),
		   (Sqrt(Pi)*((n*Power(Pi,8))/1.032192e7 + ((-1 + n)*n*Power(Pi,8))/163840. +
		        ((-2 + n)*(-1 + n)*n*Power(Pi,8))/49152. +
		        ((-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,8))/98304.)*Gamma(1 + n/2.))/
		    Gamma((1 + n)/2.),(Sqrt(Pi)*(-2.691144455467372e-10*(n*Power(Pi,10)) -
		        (17*(-1 + n)*n*Power(Pi,10))/2.4772608e8 -
		        (7*(-2 + n)*(-1 + n)*n*Power(Pi,10))/1.179648e7 -
		        ((-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,10))/1.179648e6 -
		        ((-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,10))/3.93216e6)*Gamma(1 + n/2.))
		     /Gamma((1 + n)/2.),(Sqrt(Pi)*((n*Power(Pi,12))/1.9619905536e12 +
		        (31*(-1 + n)*n*Power(Pi,12))/5.94542592e10 +
		        ((-2 + n)*(-1 + n)*n*Power(Pi,12))/9.289728e7 +
		        (19*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,12))/5.6623104e8 +
		        ((-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,12))/3.7748736e7 +
		        ((-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,12))/1.8874368e8)*
		      Gamma(1 + n/2.))/Gamma((1 + n)/2.),
		   (Sqrt(Pi)*(-7.001187498614334e-16*(n*Power(Pi,14)) -
		        ((-1 + n)*n*Power(Pi,14))/3.4879832064e11 -
		        (13*(-2 + n)*(-1 + n)*n*Power(Pi,14))/9.512681472e10 -
		        ((-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,14))/1.189085184e9 -
		        ((-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,14))/7.5497472e8 -
		        ((-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,14))/1.50994944e9 -
		        ((-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,14))/
		         1.056964608e10)*Gamma(1 + n/2.))/Gamma((1 + n)/2.),
		   (Sqrt(Pi)*((n*Power(Pi,16))/1.371195958099968e18 +
		        (5461*(-1 + n)*n*Power(Pi,16))/4.57065319366656e17 +
		        (31*(-2 + n)*(-1 + n)*n*Power(Pi,16))/2.39175991296e13 +
		        (457*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,16))/3.04405807104e13 +
		        (43*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,16))/1.01468602368e12 +
		        (29*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,16))/
		         7.247757312e11 + ((-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*
		           Power(Pi,16))/7.247757312e10 +
		        ((-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,16))/
		         6.7645734912e11)*Gamma(1 + n/2.))/Gamma((1 + n)/2.),
		   (Sqrt(Pi)*(-5.958254611429682e-22*(n*Power(Pi,18)) -
		        (257*(-1 + n)*n*Power(Pi,18))/6.581740598879846e18 -
		        (63047*(-2 + n)*(-1 + n)*n*Power(Pi,18))/6.581740598879846e18 -
		        (491*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,18))/2.41089399226368e15 -
		        (713*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,18))/7.305739370496e14 -
		        (569*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,18))/
		         3.652869685248e14 - (17*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*
		           n*Power(Pi,18))/1.73946175488e13 -
		        ((-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,18))/
		         4.05874409472e12 - ((-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*
		           (-2 + n)*(-1 + n)*n*Power(Pi,18))/4.870492913664e13)*Gamma(1 + n/2.))/
		    Gamma((1 + n)/2.),(Sqrt(Pi)*((n*Power(Pi,20))/2.5510826561258285e24 +
		        (73*(-1 + n)*n*Power(Pi,20))/7.10410096387031e20 +
		        (1069*(-2 + n)*(-1 + n)*n*Power(Pi,20))/1.8804973139656704e19 +
		        (164573*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,20))/7.521989255862682e19 +
		        (317*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,20))/1.83687161315328e16 +
		        (367*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,20))/
		         8.349416423424e15 + (131*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*
		           n*Power(Pi,20))/2.9222957481984e15 +
		        (13*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*
		           Power(Pi,20))/6.493990551552e14 +
		        ((-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*
		           Power(Pi,20))/2.5975962206208e14 +
		        ((-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*
		           (-1 + n)*n*Power(Pi,20))/3.8963943309312e15)*Gamma(1 + n/2.))/
		    Gamma((1 + n)/2.),(Sqrt(Pi)*(-2.1211603623510776e-28*(n*Power(Pi,22)) -
		        (1271*(-1 + n)*n*Power(Pi,22))/5.714425149721856e24 -
		        ((-2 + n)*(-1 + n)*n*Power(Pi,22))/3.6084322356166656e18 -
		        (269*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,22))/1.4041046610943672e19 -
		        (7141*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,22))/
		         2.9252180439465984e19 - (14797*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*
		           Power(Pi,22))/1.5429721550487552e19 -
		        (2101*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,22))/
		         1.402701959135232e18 - (173*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*
		           (-2 + n)*(-1 + n)*n*Power(Pi,22))/1.636485618991104e17 -
		        (11*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*
		           Power(Pi,22))/3.11711546474496e16 -
		        ((-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*
		           (-1 + n)*n*Power(Pi,22))/1.870269278846976e16 -
		        ((-10 + n)*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*
		           (-2 + n)*(-1 + n)*n*Power(Pi,22))/3.428827011219456e17)*Gamma(1 + n/2.))/
		    Gamma((1 + n)/2.),(Sqrt(Pi)*((n*Power(Pi,24))/1.0409396852733332e31 +
		        (60787*(-1 + n)*n*Power(Pi,24))/1.50860823952657e29 +
		        (309979*(-2 + n)*(-1 + n)*n*Power(Pi,24))/2.7429240718664908e26 +
		        (64027*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,24))/4.5829976138120146e23 +
		        (1277*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,24))/
		         4.493134915501975e20 + (37859*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*
		           Power(Pi,24))/2.2465674577509876e21 +
		        ((-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,24))/
		         2.571620258414592e16 + (19*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*
		           (-2 + n)*(-1 + n)*n*Power(Pi,24))/4.6548924273524736e17 +
		        (53*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*
		           Power(Pi,24))/2.513641910770336e18 +
		        (7*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*
		           (-1 + n)*n*Power(Pi,24))/1.2824703626379264e18 +
		        ((-10 + n)*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*
		           (-2 + n)*(-1 + n)*n*Power(Pi,24))/1.4962154230775808e18 +
		        ((-11 + n)*(-10 + n)*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*
		           (-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,24))/3.2916739307706778e19)*
		      Gamma(1 + n/2.))/Gamma((1 + n)/2.),
		   (Sqrt(Pi)*(-3.6948863613975007e-35*(n*Power(Pi,26)) -
		        (241*(-1 + n)*n*Power(Pi,26))/3.887729916987239e29 -
		        (259459*(-2 + n)*(-1 + n)*n*Power(Pi,26))/6.631245008908e28 -
		        (1352291*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,26))/1.567385183923709e27 -
		        (121391*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,26))/
		         4.3647596322019187e24 - (15749*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*
		           Power(Pi,26))/6.418764165002822e22 -
		        (367139*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,26))/
		         4.493134915501975e23 - (59*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*
		           (-2 + n)*(-1 + n)*n*Power(Pi,26))/4.800357815707238e19 -
		        (43*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*
		           Power(Pi,26))/4.654892427352474e19 -
		        (61*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*
		           (-1 + n)*n*Power(Pi,26))/1.6757612738468905e20 -
		        ((-10 + n)*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*
		           (-2 + n)*(-1 + n)*n*Power(Pi,26))/1.3299692649578496e19 -
		        ((-11 + n)*(-10 + n)*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*
		           (-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,26))/1.3166695723082711e20 -
		        ((-12 + n)*(-11 + n)*(-10 + n)*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*
		           (-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,26))/3.423340888001505e21)*
		      Gamma(1 + n/2.))/Gamma((1 + n)/2.),
		   (Sqrt(Pi)*((n*Power(Pi,28))/8.184284181493056e37 +
		        (22369621*(-1 + n)*n*Power(Pi,28))/2.7280947271643517e37 +
		        (4156343*(-2 + n)*(-1 + n)*n*Power(Pi,28))/3.568936063794285e32 +
		        (3974590523*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,28))/8.689583459673043e32 +
		        (244879627*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,28))/
		         1.0532828435967325e30 + (39888853*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*
		           n*Power(Pi,28))/1.3199033127778602e28 +
		        (277471*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,28))/
		         1.9410342834968533e25 + (2723363*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*
		           (-2 + n)*(-1 + n)*n*Power(Pi,28))/9.058159989651982e25 +
		        (4201*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*
		           Power(Pi,28))/1.3272029288867373e23 +
		        (3229*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*
		           (-1 + n)*n*Power(Pi,28))/1.8098221757546417e23 +
		        (167*(-10 + n)*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*
		           (-2 + n)*(-1 + n)*n*Power(Pi,28))/3.016370292924403e22 +
		        (59*(-11 + n)*(-10 + n)*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*
		           (-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,28))/6.320013947079701e22 +
		        ((-12 + n)*(-11 + n)*(-10 + n)*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*
		           (-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,28))/1.2640027894159403e22 +
		        ((-13 + n)*(-12 + n)*(-11 + n)*(-10 + n)*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*
		           (-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,28))/
		         3.8341417945616855e23)*Gamma(1 + n/2.))/Gamma((1 + n)/2.),
		   (Sqrt(Pi)*(-3.511074584737332e-42*(n*Power(Pi,30)) -
		        (617093*(-1 + n)*n*Power(Pi,30))/6.5474273451944445e38 -
		        (19720755713*(-2 + n)*(-1 + n)*n*Power(Pi,30))/6.5474273451944445e38 -
		        (66437323*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,30))/3.155690835354947e33 -
		        (29232262193*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,30))/
		         1.7379166919346086e34 - (20221788413*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*
		           (-1 + n)*n*Power(Pi,30))/6.319697061580395e32 -
		        (112228607*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*
		           Power(Pi,30))/5.279613251111441e29 -
		        (1113323*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*
		           Power(Pi,30))/1.8116319979303964e27 -
		        (19209583*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*
		           (-1 + n)*n*Power(Pi,30))/2.1739583975164757e28 -
		        (4679*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*
		           (-1 + n)*n*Power(Pi,30))/6.825615062846077e24 -
		        (2173*(-10 + n)*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*
		           (-2 + n)*(-1 + n)*n*Power(Pi,30))/7.239288703018567e24 -
		        (599*(-11 + n)*(-10 + n)*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*
		           (-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,30))/7.963217573320424e24 -
		        ((-12 + n)*(-11 + n)*(-10 + n)*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*
		           (-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,30))/9.480020920619552e22 -
		        ((-13 + n)*(-12 + n)*(-11 + n)*(-10 + n)*(-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*
		           (-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(Pi,30))/
		         1.314562900992578e24 - ((-14 + n)*(-13 + n)*(-12 + n)*(-11 + n)*(-10 + n)*
		           (-9 + n)*(-8 + n)*(-7 + n)*(-6 + n)*(-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*
		           (-1 + n)*n*Power(Pi,30))/4.6009701534740225e25)*Gamma(1 + n/2.))/
		    Gamma((1 + n)/2.));
	pot_series[0] = 1.f;
	for (int m = 0; m < NTAYLOR; m++) {
		pot_series[0] += 4 * M_PI * coeffs[m] / (2 + 2 * m) / (3 + 2 * m);
	}
	for (int m = 1; m < NTAYLOR; m++) {
		pot_series[m] = -4 * M_PI * coeffs[m - 1] / (2 * m * (2 * m + 1));
	}
	for (int m = 0; m < NTAYLOR - 1; m++) {
		PRINT("%e\n", coeffs[m]);
		f_series[m] = -(2 * m + 2) * pot_series[m + 1];
	}
	f_series[NTAYLOR-1] = 0.f;
	FILE* fp = fopen("soften.txt", "wt");
	for (int m = 0; m < 100; m++) {
		double r = (double) m / 100.0;
		double pot = kernelPot(r);
		double fqinv = kernelFqinv(r);
		fprintf(fp, "%e %e %e\n", r, pot, fqinv);
	}
	fclose(fp);
	abort();
}

void kernel_output() {
}

double kernel_stddev(std::function<double(double)> W) {
	double sum = 0.f;
	int N = 10000;
	double dq = 1.0 / N;
	for (int i = 0; i < N; i++) {
		double q = (i + 0.5) * dq;
		sum += sqr(sqr(q)) * W(q) * dq;
	}
	sum *= 4.0 * M_PI;
	return sqrt(sum);
}

void kernel_adjust_options(options& opts) {

	constexpr double bspline_width = 4.676649e-01;
	constexpr double bspline_n = 60;
	double sum = 0.0;
	constexpr int N = 1024;
	for (int i = 1; i < N; i++) {
		double q = (double) i / N;
		sum += sqr(q) * 4.0 * M_PI / N * sqr(q) * kernelW(q);
	}
	double width = sqrt(sum);
	PRINT("kernel width = %e\n", width);
	const double n = pow(width / bspline_width, -3) * bspline_n;
	const double cfl = opts.cfl * width / bspline_width;
	PRINT("Setting neighbor number to %e\n", n);
	PRINT("Adjusting CFL to %e\n", cfl);

	opts.neighbor_number = n;
	opts.cfl = cfl;

	opts.sph_bucket_size = 8.0 / M_PI * opts.neighbor_number;

}
