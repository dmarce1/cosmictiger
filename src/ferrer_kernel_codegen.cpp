#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <array>

#include <functional>

#include <cosmictiger/defs.hpp>

static int ntab = 0;

void indent() {
	ntab++;
}

void deindent() {
	ntab--;
}

//#define BSPLINE

template<class ...Args>
void tprint(const char* str, Args&&...args) {
	for (int i = 0; i < ntab; i++) {
		printf("\t");
	}
	printf(str, std::forward<Args>(args)...);
}

void tprint(const char* str) {
	for (int i = 0; i < ntab; i++) {
		printf("\t");
	}
	printf("%s", str);
}

double pochhammer(double a, int n) {
	if (n == 0) {
		return 1.0;
	} else {
		return (a + n - 1) * pochhammer(a, n - 1);
	}
}

double factorial(int n) {
	double a = 1.0;
	for (int i = 1; i <= n; i++) {
		a *= i;
	}
	return a;
}

using polynomial = std::vector<double>;

polynomial poly_add(const polynomial& A, const polynomial& B) {
	polynomial C;
	C.resize(std::max(A.size(), B.size()));
	const int N = std::min(A.size(), B.size());
	for (int i = 0; i < N; i++) {
		C[i] = A[i] + B[i];
	}
	if (A.size() > B.size()) {
		for (int i = N; i < A.size(); i++) {
			C[i] = A[i];
		}
	} else {
		for (int i = N; i < B.size(); i++) {
			C[i] = B[i];
		}
	}
	return C;
}

void poly_shrink_to_fit(polynomial& A) {
	if (A.size()) {
		while (A.back() == 0.0) {
			A.pop_back();
			if (!A.size()) {
				break;
			}
		}
	}
}
void poly_shrink_to_fits(std::vector<polynomial>& A) {
	for (int n = 0; n < A.size(); n++) {
		if (A[n].size()) {
			while (A[n].back() == 0.0 && A[n].size()) {
				A[n].pop_back();
			}
		}
	}
}

polynomial poly_mult(const polynomial& A, const polynomial& B) {
	polynomial C;
	C.resize(A.size() * B.size(), 0.0);
	const int N = std::min(A.size(), B.size());
	for (int i = 0; i < A.size(); i++) {
		for (int j = 0; j < B.size(); j++) {
			C[i + j] += A[i] * B[j];
		}
	}
	poly_shrink_to_fit(C);
	return C;
}

polynomial poly_flip(const polynomial& A) {
	auto B = A;
	for (int i = 1; i < A.size(); i += 2) {
		B[i] = -B[i];
	}
	return B;
}

polynomial poly_shift(const polynomial& A, double x) {
	if (x != 0.0) {
		polynomial B(A.size(), 0.0);
		std::vector<double> co(1, 1.0);
		for (int n = 0; n < A.size(); n++) {
			for (int m = 0; m < co.size(); m++) {
				B[m] += pow(-x, (co.size() - 1) - m) * co[m] * A[n];
			}
			for (int i = co.size() - 1; i >= 1; i--) {
				co[i] = co[i] + co[i - 1];
			}
			co.push_back(1.0);
		}
		return B;
	} else {
		return A;
	}
}

polynomial poly_normalize(const polynomial& A) {
	double norm = 0.0;
	for (int i = 0; i < A.size(); i++) {
		norm += A[i] * 4.0 * M_PI / (i + 3);
	}
	polynomial C(A.size());
	for (int i = 0; i < A.size(); i++) {
		C[i] = A[i] / norm;
	}
	return C;
}

polynomial poly_rho2pot(const polynomial& rho) {
	polynomial pot(rho.size() + 2);
	pot[0] = 1.0;
	pot[1] = 0.0;
	for (int i = 0; i < rho.size(); i++) {
		pot[i + 2] = -4.0 * M_PI * rho[i] / ((i + 3) * (i + 2));
		pot[0] += 4.0 * M_PI * rho[i] / ((i + 3) * (i + 2));
	}
	poly_shrink_to_fit(pot);
	return pot;
}

polynomial poly_rho2force(const polynomial& rho) {
	polynomial force(rho.size());
	for (int i = 0; i < rho.size(); i++) {
		force[i] = 4.0 * M_PI * rho[i] / (i + 3);
	}
	poly_shrink_to_fit(force);
	return force;
}

std::string poly_to_string(const polynomial& A) {
	std::string str = std::to_string(A[0]);
	if (A[1] != 0.0) {
		str += A[1] > 0.0 ? " + " : " - ";
		str += std::to_string(std::abs(A[1])) + " * x";
	}
	for (int i = 2; i < A.size(); i++) {
		if (A[i] != 0.0) {
			str += A[i] > 0.0 ? " + " : " - ";
			str += std::to_string(std::abs(A[i])) + " * x**" + std::to_string(i);
		}
	}
	return str;
}

polynomial poly_filter(int m, double kmax) {
	polynomial A;
	double err;
	int n = 0;
	do {
		double num = pochhammer(1.5 + 0.5 * m, n);
		num *= pow(-0.25, n);
		num *= 4.0 * M_PI;
		double den = pochhammer(1.5, n);
		den *= pochhammer(2.5 + 0.5 * m, n);
		den *= factorial(n);
		den *= 3 + m;
		double coeff = num / den;
		err = pow(kmax, 2 * n) * std::abs(coeff);
		n++;
		A.push_back(coeff);
		A.push_back(0.0);
	} while (err > 1e-7);
	A.pop_back();
	return A;
}

polynomial poly_filter(polynomial rho, double kmax) {
	polynomial C;
	for (int i = 0; i < rho.size(); i++) {
		auto A = poly_filter(i, kmax);
		for (int j = 0; j < A.size(); j++) {
			A[j] *= rho[i];
		}
		C = poly_add(A, C);
	}
	return C;
}

bool poly_zero(const polynomial& A) {
	for (int i = 0; i < A.size(); i++) {
		if (A[i] != 0.0) {
			return false;
		}
	}
	return true;
}

bool polys_zero(const std::vector<polynomial>& A) {
	for (int n = 0; n < A.size(); n++) {
		for (int i = 0; i < A[n].size(); i++) {
			if (A[n][i] != 0.0) {
				return false;
			}
		}
	}
	return true;
}

struct conditional {
	std::pair<double, double> x;
	polynomial f;
};

struct potential_type {
	conditional r;
	double i;
};

#ifdef BSPLINE

struct expansion_type {
	std::vector<std::vector<polynomial>> r;
	std::vector<std::vector<polynomial>> i;
};

expansion_type poly_pot2expansion(const std::vector<potential_type>& pot, int order) {
	expansion_type expansion;
	std::vector<polynomial> r(pot.size());
	std::vector<polynomial> i(pot.size());
	for (int j = 0; j < pot.size(); j++) {
		r[j] = pot[j].r.f;
		i[j].resize(2, 0.0);
		i[j][1] = pot[j].i;
	}
	expansion.i.push_back(i);
	expansion.r.push_back(r);
	for (int j = 0; j < order; j++) {
		double r0 = 0.0;
		bool jump = false;
		for (int ci = 0; ci < pot.size(); ci++) {
			if (r[ci].size()) {
				for (int n = 0; n < (int) r[ci].size() - 1; n++) {
					r[ci][n] = r[ci][n + 1] * (n + 1);
				}
				r[ci].pop_back();
				if (r[ci].size()) {
					r0 = r[ci][0];
					jump = true;
					for (int n = 0; n < (int) r[ci].size() - 1; n++) {
						r[ci][n] = r[ci][n + 1];
					}
					if (r[ci].size()) {
						r[ci].pop_back();
					}
				}
			}
			i.resize(i[ci].size() + 1);
			for (int n = i[ci].size() - 1; n >= 2; n--) {
				i[ci][n] = i[ci][n - 1] / -(n - 1);
			}
			i[ci][1] = i[ci][0] = 0.0;
			i[ci].resize(i[ci].size() + 1);
			for (int n = i[ci].size() - 1; n >= 1; n--) {
				i[ci][n] = i[ci][n - 1];
			}
			if (jump) {
				i[ci][1] = r0;
			}
		}
		expansion.r.push_back(r);
		expansion.i.push_back(i);
	}
	if (polys_zero(expansion.r.back())) {
		expansion.r.pop_back();
	}
	if (polys_zero(expansion.i.back())) {
		expansion.i.pop_back();
	}
	for (int i = 0; i < expansion.r.size(); i++) {
		poly_shrink_to_fits(expansion.r[i]);
	}
	for (int i = 0; i < expansion.i.size(); i++) {
		poly_shrink_to_fits(expansion.i[i]);
	}
	return expansion;
}

#else

struct expansion_type {
	std::vector<polynomial> r;
	std::vector<polynomial> i;
};

expansion_type poly_pot2expansion(const polynomial& pot, int order) {
	expansion_type expansion;
	expansion.r.push_back(pot);
	auto r = pot;
	decltype(r) i(2, 0.0);
	expansion.i.push_back(i);
	for (int j = 0; j < order; j++) {
		double r0 = 0.0;
		bool jump = false;
		if (r.size()) {
			for (int n = 0; n < (int) r.size() - 1; n++) {
				r[n] = r[n + 1] * (n + 1);
			}
			r.pop_back();
			if (r.size()) {
				r0 = r[0];
				jump = true;
				for (int n = 0; n < (int) r.size() - 1; n++) {
					r[n] = r[n + 1];
				}
				if (r.size()) {
					r.pop_back();
				}
			}
		}
		i.resize(i.size() + 1);
		for (int n = i.size() - 1; n >= 2; n--) {
			i[n] = i[n - 1] / -(n - 1);
		}
		i[1] = i[0] = 0.0;
		i.resize(i.size() + 1);
		for (int n = i.size() - 1; n >= 1; n--) {
			i[n] = i[n - 1];
		}
		if (jump) {
			i[1] = r0;
		}
		expansion.r.push_back(r);
		expansion.i.push_back(i);
	}
	if (poly_zero(expansion.r.back())) {
		expansion.r.pop_back();
	}
	if (poly_zero(expansion.i.back())) {
		expansion.i.pop_back();
	}
	for (int i = 0; i < expansion.r.size(); i++) {
		poly_shrink_to_fit(expansion.r[i]);
	}
	for (int i = 0; i < expansion.i.size(); i++) {
		poly_shrink_to_fit(expansion.i[i]);
	}
	return expansion;
}

#endif

bool poly_even_only(polynomial A) {
	for (int i = 1; i < A.size(); i += 2) {
		if (A[i] != 0.0) {
			return false;
		}
	}
	return true;
}

bool close2zero(double number) {
	return std::abs(number) < 1e-10;
}

void poly_print_fma_instructions(polynomial A, const char* varname = "q", const char* type = "float") {
	if (poly_even_only(A)) {
		tprint("y = %s(%.8e);\n", type, A.back());
		while (A.size() > 1) {
			A.pop_back();
			A.pop_back();
			if (!A.size()) {
				break;
			}
			if (A.back() != 0.0 && !close2zero(A.back())) {
				tprint("y = fma( y, %s2, %s(%.8e) );\n", varname, type, A.back());
			} else {
				tprint("y *= q2;\n");
			}
		}
	} else {
		tprint("y = %s(%.8e);\n", type, A.back());
		while (A.size() > 1) {
			A.pop_back();
			if (A.back() != 0.0 && !close2zero(A.back())) {
				tprint("y = fma( y, %s, %s(%.8e) );\n", varname, type, A.back());
			} else {
				tprint("y *= %s;\n", varname);
			}
		}
	}
}

double poly_eval(polynomial A, double x) {
	if (A.size()) {
		double sum = A.back();
		while (A.size() > 1) {
			A.pop_back();
			sum = std::fma(sum, x, A.back());
		}
		return sum;
	} else {
		return 0.0;
	}
}

double sqr(double a) {
	return a * a;
}

inline double green_a(double k) {
	if (k > 0.5) {
		return 945 * (5 * k * (-21 + 2 * sqr(k)) * cos(k) + (105 - 45 * sqr(k) + sqr(sqr(k))) * sin(k)) / (k * sqr(sqr(sqr(k))));
	} else {
		return 1.0 - 1.0 / 22.0 * sqr(k) + 1.0 / 1144.0 * sqr(sqr(k)) - (1.0 / 102960.0) * sqr(sqr(k)) * sqr(k);
	}
}

polynomial poly_pow(const polynomial& A, int n) {
	polynomial B(1, 1.0);
	for (int i = 0; i < n; i++) {
		B = poly_mult(B, A);
	}
	return B;
}

polynomial poly_neg(const polynomial& A) {
	polynomial B(A.size());
	for (int i = 0; i < A.size(); i++) {
		B[i] = -A[i];
	}
	return B;
}

void conditionals_compress(std::vector<conditional>& B) {
	int i = 0;
	while (i < B.size()) {
		int j = i + 1;
		while (j < B.size()) {
			if (B[j].x == B[i].x) {
				B[i].f = poly_add(B[i].f, B[j].f);
				B[j] = B.back();
				B.pop_back();
			} else {
				j++;
			}
		}
		i++;
	}
}

std::vector<conditional> next_cloud(const std::vector<conditional>& A) {
	std::vector<conditional> B;
	for (int i = 0; i < A.size(); i++) {
		double shift = A[i].x.first;
		const auto Ashift = poly_shift(A[i].f, -shift);
		polynomial Aleft(Ashift.size() + 1, 0.0), Aright(Ashift.size() + 1, 0.0);
		for (int n = 0; n < Ashift.size(); n++) {
			const auto co = Ashift[n] / (1.0 + n);
			Aleft[n + 1] += co;
		}
		polynomial basis(2);
		basis[0] = -1.0;
		basis[1] = 1.0;
		const auto one = polynomial(1, 1.0);
		for (int n = 0; n < Ashift.size(); n++) {
			auto term = poly_add(one, poly_neg(poly_pow(basis, n + 1)));
			Aright = poly_add(Aright, poly_mult(term, polynomial(1, Ashift[n] / (n + 1))));
		}
		Aleft = poly_shift(Aleft, shift);
		Aright = poly_shift(Aright, shift);
		conditional cl, cr;
		cl.x.first = A[i].x.first;
		cl.x.second = A[i].x.second;
		cl.f = Aleft;
		cr.x.first = A[i].x.first + 1;
		cr.x.second = A[i].x.second + 1;
		cr.f = Aright;
		B.push_back(cl);
		B.push_back(cr);
	}
	conditionals_compress(B);
	double minx = 100000;
	double maxx = -100000;
	for (int i = 0; i < B.size(); i++) {
		minx = std::min(minx, B[i].x.first);
		maxx = std::max(maxx, B[i].x.second);
	}
	double shift = 0.5 * (minx + maxx);
//	printf( "shifting %e\n", shift);
	for (int i = 0; i < B.size(); i++) {

		B[i].f = poly_shift(B[i].f, -shift);
		B[i].x.first -= shift;
		B[i].x.second -= shift;
	}
	return B;
}

void conditionals_print(std::vector<conditional> A) {
	for (int i = 0; i < A.size(); i++) {
		auto poly = poly_to_string(A[i].f);
		printf("%s if x is between %e and %e\n", poly.c_str(), A[i].x.first, A[i].x.second);
	}
}

void print_conditionals_fma(std::vector<conditional> cloud) {
	std::sort(cloud.begin(), cloud.end(), [](conditional a,conditional b) {
		return a.x.second < b.x.second;
	});
	double xmin = cloud[0].x.first;
	double xmax = cloud.back().x.second;
	tprint("y = 0.0f;\n");
	for (int i = 0; i < cloud.size(); i++) {
		tprint("%s( x < float(%.8e) ) {\n", i == 0 ? "if" : "} else if", cloud[i].x.second);
		indent();
		poly_print_fma_instructions(cloud[i].f, "x");
		deindent();
		if (i == cloud.size() - 1) {
			tprint("}\n");
		}
	}
}

#ifdef BSPLINE
void print_green_direct(polynomial rho, std::vector<potential_type> pot, std::vector<potential_type> force, expansion_type expansion, polynomial filter, double kmax) {
#else
void print_green_direct(polynomial rho, polynomial pot, polynomial force, expansion_type expansion, polynomial filter, double kmax) {
#endif

#ifndef BSPLINE
	tprint("#pragma once\n");
	tprint("\n");
	tprint("CUDA_EXPORT inline void green_direct(float& phi, float& f, float r, float r2, float rinv, float rsinv, float rsinv2) {\n");
	indent();
	tprint("const float q = r * rsinv;\n");
	tprint("float q2 = sqr(q);\n");
	tprint("f = phi = 0.f;\n");
	tprint("if (q2 < 1.f) {\n");
	indent();
	tprint("phi = rinv;\n");
	tprint("f = rinv * sqr(rinv);\n");
	tprint("float y;\n");
	poly_print_fma_instructions(pot);
	tprint("phi -= y * rsinv;\n");
	poly_print_fma_instructions(force);
	tprint("f -= y * sqr(rsinv) * rsinv;\n");
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");
	tprint("\n");
#endif

	const int L = ORDER;
	tprint("\n");
	tprint("CUDA_EXPORT inline array<float, %i> green_kernel(float r, float rsinv, float rsinv2, bool do_phi) {\n", L);
	indent();
	tprint("array<float, %i> d;\n", L);
	tprint("float q0 = 1.f;\n");
	tprint("float q1 = r * rsinv;\n");
	tprint("float& q = q1;\n");
	tprint("float qinv = 1.f / q1;\n");
	tprint("float q2 = sqr(q1);\n");
	for (int i = 3; i < L; i++) {
		tprint("float q%i = q1 * q%i;\n", i, i - 1);
	}
	tprint("float rsinv1 = rsinv;\n");
	for (int i = 3; i <= L; i++) {
		tprint("float rsinv%i = rsinv * rsinv%i;\n", i, i - 1);
	}
	tprint("float rinv1 = 1.f / r;\n");
	for (int i = 2; i <= L; i++) {
		tprint("float rinv%i = rinv1 * rinv%i;\n", i, i - 1);
	}
	tprint("if (q2 < 1.f) {\n");
	indent();
	tprint("float y;\n");
	tprint("float z;\n");
	int n = -1;
	for (int i = 0; i < L; i++) {
		if (i == 0) {
			tprint("if( do_phi ) {\n");
			indent();
		}

		if (expansion.r.size() > i) {
			if (expansion.r[i].size()) {
				poly_print_fma_instructions(expansion.r[i]);
			} else {
				tprint("y = 0.f;\n");
			}
		}
		if (expansion.i.size() > i) {
			if (expansion.i[i].size()) {
				tprint("z = y;\n");
				poly_print_fma_instructions(expansion.i[i], "qinv");
				tprint("y += z;\n");
			}
		}
		tprint("y *= q%i * rsinv%i;\n", i, i + 1);
		tprint("d[%i] = fmaf( float(%i), rinv%i, y);\n", i, n, i + 1);
		n = -n * (2 * i + 1);
		if (i == 0) {
			deindent();
			tprint("}\n");
		}
	}
	deindent();
	tprint("} else {\n");
	indent();
	for (int i = 0; i < L; i++) {
		tprint("d[%i] = 0.f;\n", i);
	}
	deindent();
	tprint("}\n");
	tprint(" return d;\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	/*
	 tprint("\n");
	 tprint("CUDA_EXPORT inline double green_filter(double k) {\n", L);
	 indent();
	 tprint("if(k > double(%.8e)) {\n", kmax);
	 indent();
	 tprint("return 0.f;\n");
	 deindent();
	 tprint("}\n");
	 tprint("double y;\n");
	 tprint("double q2 = sqr(k);\n");
	 poly_print_fma_instructions(filter, "q", "double");
	 tprint("return y;\n");
	 deindent();
	 tprint("}\n");
	 tprint("\n");*/

	tprint("\n");
	tprint("CUDA_EXPORT inline float green_phi0(float nparts, float rs) {\n", L);
	indent();
	tprint("return float(%.8e) * sqr(rs) * (nparts - 1) +  float(%.8e) / rs;\n", -4.0 * M_PI * filter[2], pot[0]);
	deindent();
	tprint("}\n");
	tprint("\n");

	tprint("\n");
	tprint("CUDA_EXPORT inline float green_rho(float q) {\n", L);
	indent();
	tprint("float y;\n");
	tprint("float q2 = sqr(q);\n");
	poly_print_fma_instructions(rho, "q");
	tprint("return y;\n");
	deindent();
	tprint("}\n");
	tprint("\n");

	FILE* fp = fopen("filter.txt", "wt");
	for (double k = 0.01; k < kmax; k += .1) {
		auto a = green_a(k);
		auto n = poly_eval(filter, k);
		fprintf(fp, "%e %e %e\n", k, a, n);
	}
	fclose(fp);

	tprint("\n");
	tprint("CUDA_EXPORT inline void gsoft(float& f, float& phi, float q2, float hinv, float h2inv, float h3inv, bool do_phi) {\n");
	indent();
	tprint("q2 *= h2inv;\n");
	tprint("f = phi = 0.f;\n");
	tprint("if (q2 < 1.f) {\n");
	indent();
	tprint("float y;\n");
	tprint("if( do_phi ) {\n");
	indent();
	poly_print_fma_instructions(pot);
	tprint("phi = y * hinv;\n");
	deindent();
	tprint("}\n");
	poly_print_fma_instructions(force);
	tprint("f = y * h3inv;\n");
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("\n");

	tprint("CUDA_EXPORT inline float self_phi() {\n");
	indent();
	tprint("float f, phi;\n");
	tprint("gsoft(f, phi, 0.0f, 1.f, 1.f, 1.f, true);\n");
	tprint("return phi;\n");
	deindent();
	tprint("}\n");
	tprint("\n");

}

polynomial lane_emden(double n) {
	std::vector<double> a, b;
	a.push_back(1);
	b.push_back(1);
	int k = 1;
	while (std::abs(b.back()) > 1e-15) {
		double sum = 0.0;
		double ak = -b[k - 1] / (2 * k * (2 * k + 1));
		a.push_back(ak);
		for (int j = 0; j <= k - 1; j++) {
			sum += (n * (k - j) - j) * (a[k - j] * b[j]);
		}
		double bk = sum / k;
		printf("%e\n", bk);
		b.push_back(bk);
		k++;
	}
	polynomial A;
	for (int i = 0; i < b.size(); i++) {
		A.push_back(b[i]);
		if (i < b.size() - 1) {
			A.push_back(0.0);
		}
	}
	return A;
}

//#define LUCY

#define CLOUD_ORDER 5
#define STENCIL_ORDER 4

std::vector<conditional> conditionals_remove_negatives(std::vector<conditional>& A) {
	std::vector<conditional> B;
	for (int i = 0; i < A.size(); i++) {
		if (A[i].x.second > 0.0) {
			conditional b = A[i];
			b.x.first = std::max(0.0, b.x.first);
			B.push_back(b);
		}
	}
	return B;
}

polynomial poly_integrate(const polynomial& A) {
	polynomial B(A.size() + 1);
	B[0] = 0.0;
	for (int i = 1; i < B.size(); i++) {
		B[i] = A[i - 1] / i;
	}
	return B;
}

polynomial poly_integrate_a2x(const polynomial& A, double a) {
	auto B = poly_integrate(A);
	B[0] -= poly_eval(B, a);
	return B;
}

const double control_point(int i) {
	return i;

}

polynomial poly_scale(const polynomial& A, double a) {
	auto B = A;
	for (int n = 0; n < A.size(); n++) {
		B[n] *= pow(1.0 / a, n);
	}
	return B;
}

polynomial Wendland(int l, int k) {
	polynomial W;
	if (k == 0) {
		polynomial a(2);
		a[0] = 1.0;
		a[1] = -1.0;
		W = poly_pow(a, l);
	} else {
		polynomial a(2);
		a[0] = 0.0;
		a[1] = 1.0;
		auto b = Wendland(l, k - 1);
		auto I = poly_mult(a, b);
		I = poly_integrate_a2x(I, 1.0);
		W = poly_neg(I);
	}
	W = poly_normalize(W);
	return W;

}

std::vector<conditional> Bspline_produce(int i, int k) {
	std::vector<conditional> B;
	if (k == 0) {
		conditional A;
		A.x.first = i;
		A.x.second = i + 1.0;
		A.f.resize(1, 1.0);
		return std::vector < conditional > (1, A);
	} else {
		auto left = Bspline_produce(i, k - 1);
		auto right = Bspline_produce(i + 1, k - 1);
		polynomial wleft(2), wright(2);
		const double ti = control_point(i);
		const double tip1 = control_point(i + 1);
		const double tipk = control_point(i + k);
		const double tipkp1 = control_point(i + k + 1);
		wleft[0] = -ti / (tipk - ti);
		wleft[1] = 1.0 / (tipk - ti);
		wright[0] = tipkp1 / (tipkp1 - tip1);
		wright[1] = -1.0 / (tipkp1 - tip1);
		for (int i = 0; i < left.size(); i++) {
			left[i].f = poly_mult(left[i].f, wleft);
		}
		for (int i = 0; i < right.size(); i++) {
			right[i].f = poly_mult(right[i].f, wright);
		}
		B.insert(B.begin(), left.begin(), left.end());
		B.insert(B.begin(), right.begin(), right.end());
		conditionals_compress(B);
		return B;
	}
}

double conditionals_eval(const std::vector<conditional>& A, double x) {
	for (int i = 0; i < A.size(); i++) {
		if (x >= A[i].x.first && x < A[i].x.second) {
			return poly_eval(A[i].f, x);
		}
	}
	return 0.0;
}

conditional conditional_shift(const conditional& A, double shift) {
	conditional B;
	B.x.first = A.x.first + shift;
	B.x.second = A.x.second + shift;
	B.f = poly_shift(A.f, shift);
	return B;
}

conditional conditional_scale(const conditional& A, double a) {
	conditional B;
	B.f = poly_scale(A.f, a);
	B.x.first = A.x.first * a;
	B.x.second = A.x.second * a;
	return B;
}

void conditionals_sort(std::vector<conditional>& A) {
	std::sort(A.begin(), A.end(), [](conditional a, conditional b) {
		return a.x.first < b.x.first;
	});
}

std::vector<conditional> conditionals_to_mass_function(std::vector<conditional> A) {
	conditionals_sort(A);
	std::vector<conditional> mass(A.size());
	double M0 = 0.0;
	polynomial fourpir2(3, 0.0);
	fourpir2[2] = 4.0 * M_PI;
	for (int i = 0; i < A.size(); i++) {
		auto M = poly_integrate(poly_mult(A[i].f, fourpir2));
		M[0] -= poly_eval(M, A[i].x.first);
		M[0] += M0;
		fflush (stdout);
		M0 = poly_eval(M, A[i].x.second);
		mass[i].f = M;
		mass[i].x = A[i].x;
	}
	return mass;
}

std::vector<conditional> Bspline(int k) {
	auto B = Bspline_produce(0, k);
	const double shift = -(k + 1) / 2.0;
	for (int i = 0; i < B.size(); i++) {
		B[i] = conditional_scale(conditional_shift(B[i], shift), 1.0 / -shift);
	}
	B = conditionals_remove_negatives(B);
	auto M = conditionals_to_mass_function(B);
	auto mass = poly_eval(M.back().f, 1.0);
	auto norm = 1.0 / mass;
	for (int i = 0; i < B.size(); i++) {
		B[i].f = poly_mult(polynomial(1, norm), B[i].f);
	}
	return B;
}

double evaluate_potential(const potential_type& pot, double x) {
	return poly_eval(pot.r.f, x) + pot.i / x;
}

double potentials_eval(const std::vector<potential_type>& A, double x) {
	for (int i = 0; i < A.size(); i++) {
		if (x >= A[i].r.x.first && x < A[i].r.x.second) {
			return evaluate_potential(A[i], x);
		}
	}
	return 0.0;
}

polynomial poly_derivative(const polynomial& A) {
	polynomial B;
	if (A.size()) {
		B.resize(A.size() - 1);
		for (int n = 1; n < A.size(); n++) {
			B[n - 1] = A[n] * n;
		}
	}
	return B;
}

std::vector<potential_type> conditionals_to_potential(const std::vector<conditional>& A) {
	std::vector<potential_type> F, pot;
	auto M = conditionals_to_mass_function(A);
	F.resize(A.size());
	pot.resize(A.size());
	for (int n = 0; n < A.size(); n++) {
		F[n].r.f.resize(M[n].f.size() - 2);
		for (int m = 0; m < F[n].r.f.size(); m++) {
			F[n].r.f[m] = M[n].f[m + 2];
		}
		F[n].r.x = M[n].x;
		F[n].i = -M[n].f[0];
	}
	double pot0 = 1.0;
	for (int n = A.size() - 1; n >= 0; n--) {
		auto phi = poly_neg(poly_integrate_a2x(F[n].r.f, F[n].r.x.second));
		pot[n].r.x = F[n].r.x;
		pot[n].r.f = phi;
		pot[n].r.f[0] += F[n].i / F[n].r.x.second;
		pot[n].r.f[0] += pot0;
		pot[n].i = -F[n].i;
		pot0 = evaluate_potential(pot[n], F[n].r.x.first);
	}
	return pot;
}

//#define WENDLAND

template<int M>
class matrix: public std::array<std::array<double, M>, M> {
public:
	matrix<M - 1> cofactor_matrix(int i, int j) {
		const auto& A = *this;
		matrix<M - 1> B;
		for (int n = 0; n < M - 1; n++) {
			const int n0 = n < i ? n : n + 1;
			for (int m = 0; m < M - 1; m++) {
				const int m0 = m < j ? m : m + 1;
				B[n][m] = A[n0][m0];
			}
		}
		return B;
	}
	void print() {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < M; j++) {
				printf("%e ", (*this)[i][j]);
			}
			printf("\n");
		}
	}
	double determinant() {
		double det = 0.0;
		double sgn = 1.0;
		for (int n = 0; n < M; n++) {
			const auto a = (*this)[0][n] * sgn * cofactor_matrix(0, n).determinant();
			det += a;
			sgn *= -1.0;
		}
		return det;
	}
	matrix transpose() {
		const auto& A = *this;
		matrix B;
		for (int n = 0; n < M; n++) {
			for (int m = 0; m < M; m++) {
				B[n][m] = A[m][n];
			}
		}
		return B;
	}
	matrix operator*(const double a) {
		const auto& A = *this;
		matrix B;
		for (int n = 0; n < M; n++) {
			for (int m = 0; m < M; m++) {
				B[n][m] = a * A[n][m];
			}
		}
		return B;
	}
	matrix inverse() {
		const auto& A = *this;
		matrix B;
		for (int n = 0; n < M; n++) {
			for (int m = 0; m < M; m++) {
				const double sgn = (n + m) % 2 == 0 ? 1.0 : -1.0;
				B[n][m] = sgn * cofactor_matrix(n, m).determinant();
			}
		}
		B = B.transpose();
		const auto det = determinant();
		B = B * (1.0 / det);
		return B;
	}

};

template<>
class matrix<1> : public std::array<std::array<double, 1>, 1> {
public:
	double determinant() {
		return (*this)[0][0];
	}

};

template<int N>
matrix<2 * N> spline_coefficients() {
	matrix<2 * N> A;
	for (int n = 0; n < 2 * N; n++) {
		for (int m = 0; m < 2 * N; m++) {
			A[n][m] = 0.0;
		}
	}
	for (int n = 0; n < N; n++) {
		A[n][n] = factorial(n);
	}
	for (int n = 0; n < 2 * N; n++) {
		A[N][n] = 1.0;
		for (int m = N + 1; m < 2 * N; m++) {
			A[m][n] = A[m - 1][n] * (n - (m - N - 1));
		}
	}
	return A.inverse();
}

double integrate(const std::function<double(double)>& func, double a, double b, int N = 16384) {
	N = 2 * (N / 2) + 1;
	double dx = (b - a) / (N - 1);
	double sum = (1.0 / 3.0) * (func(a) + func(b)) * dx;
	for (int i = 1; i < N - 1; i += 2) {
		sum += 4.0 / 3.0 * func(a + i * dx) * dx;
	}
	for (int i = 2; i < N - 1; i += 2) {
		sum += 2.0 / 3.0 * func(a + i * dx) * dx;
	}
	return sum;
}

template<int N>
std::function<double(double)> func_from_data(std::array<std::vector<double>, N> I, double dx, double xmin, double xmax) {
	auto coeff = spline_coefficients<N>();
	const double dxinv = 1.0 / dx;
	auto func = [I,dxinv,dx,coeff,xmin,xmax](double x) {
		if( x < xmin || x >xmax) {
			printf( "out of range interpolation\n");
			abort();
		}
		const int i = std::min((int) (x * dxinv), (int)I[0].size()-2);
		polynomial co(2*N, 0.0);
		for( int j = 0; j < 2 * N; j++) {
			for( int m = 0; m < N; m++) {
				co[j] += coeff[j][m] * I[m][i] * pow(dx,m);
			}
			for( int m = 0; m < N; m++) {
				co[j] += coeff[j][m+N] * I[m][i+1] * pow(dx,m);
			}
		}
		return poly_eval(co, x * dxinv - i);
	};
	return func;
}

template<int N>
std::function<double(double)> construct_interpolation(const std::array<std::function<double(double)>, N> f, double xmin, double xmax, int M) {

	const double dx = (xmax - xmin) / (M - 1);
	std::array<std::vector<double>, N> I;
	for (int n = 0; n < N; n++) {
		I[n].resize(M, 0.0);
	}
	for (int i = 0; i < M; i++) {
		const double x = i * dx;
		for (int n = 0; n < N; n++) {
			I[n][i] = f[n](x);
		}
	}
	const double dxinv = 1.0 / dx;
	auto func = func_from_data<N>(I, dx, xmin, xmax);
	int NCHECK = 1024;
	double abs_err = 0.0;
	double rel_err = 0.0;
	for (int i = 0; i < NCHECK; i++) {
		double x = (double) i / (NCHECK - 1);
		double a = func(x);
		double b = f[0](x);
		abs_err += sqr(a - b);
		rel_err += sqr(1.0 - b / a);
	}
//	printf("Error = %e %e\n", sqrt(abs_err / NCHECK), sqrt(rel_err / NCHECK));
	return func;
}

std::function<double(double)> compute_mass_function(std::function<double(double)> rho, std::function<double(double)> drho, std::function<double(double)> d2rho,
		int ninterp) {
	std::array<std::vector<double>, 4> I;
	double dr = 1.0 / (ninterp - 1);
	for (int i = 0; i < 4; i++) {
		I[i].resize(ninterp);
	}
	for (int i = 0; i < ninterp; i++) {
		const double r = i * dr;
		std::function<double(double)> integrand = [rho](double r ) {return 4.0 * M_PI * r * r * rho(r);};
		I[0][i] = integrate(integrand, 0.0, r);
		I[1][i] = integrand(r);
		I[2][i] = 4.0 * M_PI * r * (2.0 * rho(r) + r * drho(r));
		I[3][i] = 4.0 * M_PI * (2.0 * rho(r) + 4.0 * r * drho(r) + r * r * d2rho(r));
	}
	return func_from_data<4>(I, dr, 0.0, 1.0);
}

struct green_type {
	int n;
	std::vector<double> a;

};

green_type green_divr(const green_type& A) {
	green_type B = A;
	B.n--;
	return B;
}

void green_print(green_type A) {
	printf("%e Mr / r^%i ", A.a[0], -A.n);
	if (A.a.size() > 1) {
		printf(" + ");
	}
	for (int i = 1; i < A.a.size(); i++) {
		printf("d%idrho%i / r^%i ", i - 1, i - 1, -(3 + A.n + i - 1));
		if (i != A.a.size() - 1) {
			printf(" + ");
		}
	}
	printf("\n");
}

green_type green_deriv(const green_type& A) {
	green_type B;
	B.a.resize(A.a.size() + 1, 0.0);
	B.a[0] += A.a[0] * double(A.n);
	B.a[1] += A.a[0] * 4.0 * M_PI;
	for (int i = 1; i < A.a.size(); i++) {
		B.a[i + 1] += A.a[i];
		B.a[i] += A.a[i] * (A.n + 3 + i - 1);
	}
	B.n = A.n - 1;
	return B;
}

std::function<double(double)> green_function(green_type A, std::function<double(double)> Mr, std::vector<std::function<double(double)>> drho) {
	if (A.a.size() > drho.size() + 1) {
		printf("Error in green_function\n");
		abort();
	}
	std::function<double(double)> func = [A,Mr,drho](double r) {
		double sum;
		if( r == 0.0 ) {
			r = 1e-7;
			sum = Mr(fabs(r)) * A.a[0];
			for( int i = 0; i < A.a.size()-1; i++) {
				sum += 0.5*drho[i](-r) * A.a[i+1] * pow(r,3+i);
				sum += 0.5*drho[i](r) * A.a[i+1] * pow(r,3+i);
			}
			sum *= pow(r, A.n);
		} else {
			sum = Mr(fabs(r)) * A.a[0];
			for( int i = 0; i < A.a.size()-1; i++) {
				sum += drho[i](r) * A.a[i+1] * pow(r,3+i);
			}
			sum *= pow(r, A.n);
		}
		return sum;
	};
	return func;
}

template<int N>
void output_interpolation_code(const char* name, const std::array<std::vector<double>, N>& I, int ninterp, double dx) {
	static auto coeff = spline_coefficients<N>();
	tprint("\n");
	tprint("CUDA_EXPORT inline double %s(float x) {\n", name);
	indent();
	tprint("if( x >= 1.0 ) {\n", 1.0 / dx);
	indent();
	tprint("return 0.0;\n");
	deindent();
	tprint("}\n");
	ninterp--;
	std::array<std::vector<double>, 2 * N> J;
	for (int m = 0; m < 2 * N; m++) {
		J[m].resize(ninterp, 0.0);
	}
	for (int n = 0; n < ninterp; n++) {
		for (int j = 0; j < 2 * N; j++) {
			for (int m = 0; m < N; m++) {
				J[j][n] += coeff[j][m] * I[m][n] * pow(dx, m);
			}
			for (int m = 0; m < N; m++) {
				J[j][n] += coeff[j][m + N] * I[m][n + 1] * pow(dx, m);
			}
		}
	}
	tprint("constexpr float co[%i][%i] = {", 2 * N, ninterp);
	for (int m = 0; m < 2 * N; m++) {
		printf("{");
		for (int n = 0; n < ninterp; n++) {
			printf("%.8e", J[m][n]);
			if (n != ninterp - 1) {
				printf(",");
			}
		}
		printf("}");
		if (m != 2 * N - 1) {
			printf(", ");
		}
	}
	printf("};\n");
	tprint("float x0 = x * float(%.8e);\n", 1.0 / dx);
	tprint("int i = x * float(%.8e);\n", 1.0 / dx);
	tprint("float t = x0 - i;\n", 1.0 / dx, 1.0 / dx);
	tprint("float y = co[%i][i];\n", 2 * N - 1);
	for (int i = 2 * N - 2; i >= 0; i--) {
		tprint("y = fmaf( y, t, co[%i][i] );\n", i);
	}
	tprint("return y;\n");
	deindent();
	tprint("}\n");
	tprint("\n");
}

void create_potentials_code(const std::function<double(int, double)> drho, int ninterp) {
	tprint("#pragma once\n");
	tprint("\n");
	double dr = 1.0 / (ninterp - 1);
	std::array<std::vector<double>, 4> I;
	for (int i = 0; i < 4; i++) {
		I[i].resize(ninterp);
	}
	double pot0;
	for (int i = 0; i < ninterp; i++) {
		const double r = i * dr;
		std::function<double(double)> integrand = [drho](double r ) {return 4.0 * M_PI * r * r * drho(0,r);};
		I[0][i] = integrate(integrand, 0.0, r);
		I[1][i] = integrand(r);
		I[2][i] = 4.0 * M_PI * r * (2.0 * drho(0, r) + r * drho(1, r));
		I[3][i] = 4.0 * M_PI * (2.0 * drho(0, r) + 4.0 * r * drho(1, r) + r * r * drho(2, r));
	}
	auto Mr = func_from_data<4>(I, dr, 0.0, 1.0);
	std::vector<std::function<double(double)>> drho_table(3 + ORDER);
	for (int i = 0; i < 3 + ORDER; i++) {
		drho_table[i] = [drho,i](double r) {
			return drho(i,r);
		};
	}
	green_type green0, green1, green2, green3;
	green0.n = -2;
	green0.a = polynomial(1, 1.0);
	green1 = green_deriv(green0);
	green2 = green_deriv(green1);
	auto dydx1 = green_function(green0, Mr, drho_table);
	auto dydx2 = green_function(green1, Mr, drho_table);
	auto dydx3 = green_function(green2, Mr, drho_table);
	for (int i = 0; i < ninterp; i++) {
		const double r = i * dr;
		std::function<double(double)> integrand = [Mr](double r ) {return r == 0.0 ? 0.0 : Mr(r)/(r*r);};
		I[0][i] = 1.0 - integrate(integrand, 1.0, r);
		if (i == 0) {
			pot0 = I[0][0];
		}
		I[1][i] = -dydx1(r);
		I[2][i] = -dydx2(r);
		I[3][i] = -dydx3(r);
	}
	output_interpolation_code<4>("potential_kernel", I, ninterp, dr);
	output_interpolation_code<4>("green_kernel_0", I, ninterp, dr);

	green0.n = -3;
	green0.a = polynomial(1, 1.0);
	green1 = green_deriv(green0);
	green2 = green_deriv(green1);
	green3 = green_deriv(green2);
	auto y0 = green_function(green0, Mr, drho_table);
	dydx1 = green_function(green1, Mr, drho_table);
	dydx2 = green_function(green2, Mr, drho_table);
	dydx3 = green_function(green3, Mr, drho_table);
	for (int i = 0; i < ninterp; i++) {
		const double r = i * dr;
		I[0][i] = y0(r);
		I[1][i] = dydx1(r);
		I[2][i] = dydx2(r);
		I[3][i] = dydx3(r);
	}
	output_interpolation_code<4>("force_kernel", I, ninterp, dr);

	for (int order = 1; order < ORDER; order++) {
		green0.n = -3;
		green0.a = polynomial(1, 1.0);
		for (int i = 1; i < order; i++) {
			green0 = green_deriv(green0);
			green0 = green_divr(green0);
		}
		green0.n += 2 * order;
		green1 = green_deriv(green0);
		green2 = green_deriv(green1);
		green3 = green_deriv(green2);
		auto y0 = green_function(green0, Mr, drho_table);
		dydx1 = green_function(green1, Mr, drho_table);
		dydx2 = green_function(green2, Mr, drho_table);
		dydx3 = green_function(green3, Mr, drho_table);
		for (int i = 0; i < ninterp; i++) {
			const double r = i * dr;
			I[0][i] = y0(r);
			I[1][i] = dydx1(r);
			I[2][i] = dydx2(r);
			I[3][i] = dydx3(r);
		}
		std::string name = "green_kernel_" + std::to_string(order);
		output_interpolation_code<4>(name.c_str(), I, ninterp, dr);
	}
	tprint("\n");
	tprint("CUDA_EXPORT inline array<float, %i> green_kernel(float r, float rsinv, float rsinv2) {\n", ORDER);
	indent();
	tprint("array<float, %i> d;\n", ORDER);
	tprint("float q = r * rsinv;\n");
	tprint("float qinv0 = 1.0f;\n");
	tprint("float qinv1 = 1.0f / q;\n");
	tprint("float rsinv1 = rsinv1;\n");
	tprint("float rinv1 = 1.f / r;\n");
	tprint("float y;\n");
	for (int i = 2; i < ORDER; i++) {
		tprint("float qinv%i = qinv%i * qinv%i;\n", i, i / 2, i - i / 2);
		tprint("float rinv%i = rinv%i * rinv%i;\n", i, i / 2, i - i / 2);
		if (i != 2) {
			tprint("float rsinv%i = rsinv%i * rsinv%i;\n", i, i / 2, i - i / 2);
		}
	}
	tprint("float rsinv%i = rsinv%i * rsinv%i;\n", ORDER, ORDER / 2, ORDER - ORDER / 2);
	tprint("float rinv%i = rinv%i * rinv%i;\n", ORDER, ORDER / 2, ORDER - ORDER / 2);
	int n = 1;
	for (int i = 0; i < ORDER; i++) {
		tprint("y = green_kernel_%i( q ) * qinv%i * rsinv%i;\n", i, i, i + 1);
		tprint("d[%i] = fmaf( float(%i), rinv%i, y);\n", i, n, i + 1);
		n *= (2 * i - 1);
	}
	tprint("return d;\n");
	deindent();
	tprint("}\n");
	tprint("\n");

	tprint("\n");
	tprint("CUDA_EXPORT inline void green_direct(float& phi, float& f, float r, float r2, float rinv, float rsinv, float rsinv2) {\n");
	indent();
	tprint("const float q = r * rsinv;\n");
	tprint("phi = rinv;\n");
	tprint("f = rinv * sqr(rinv);\n");
	tprint("float y = potential_kernel(q);\n");
	tprint("phi -= y * rsinv;\n");
	tprint("y = force_kernel(q);\n");
	tprint("f -= y * sqr(rsinv) * rsinv;\n");
	deindent();
	tprint("}\n");
	tprint("\n");

	for (int i = 0; i < ninterp; i++) {
		const double r = i * dr;
		I[0][i] = drho(0, r);
		I[1][i] = drho(1, r);
		I[2][i] = drho(2, r);
		I[3][i] = drho(3, r);
	}
	output_interpolation_code<4>("green_rho", I, ninterp, dr);

	tprint("\n");
	tprint("CUDA_EXPORT inline float green_phi0(float nparts, float rs) {\n");
	indent();
	tprint("return float(%.8e) * sqr(rs) * (nparts - 1) +  float(%.8e) / rs;\n", 0.5 * integrate([drho](double r) {
		return 16.0*M_PI*M_PI*drho(0,r) * r*r*r*r/3.0;}, 0.0, 1.0), pot0);
	deindent();
	tprint("}\n");
	tprint("\n");

}

std::function<double(double)> compute_potential_function(std::function<double(double)> rho, std::function<double(double)> drho,
		std::function<double(double)> d2rho, int ninterp) {
	double dr = 1.0 / (ninterp - 1);
	std::array<std::vector<double>, 4> I;
	for (int i = 0; i < 4; i++) {
		I[i].resize(ninterp);
	}
	for (int i = 0; i < ninterp; i++) {
		const double r = i * dr;
		std::function<double(double)> integrand = [rho](double r ) {return 4.0 * M_PI * r * r * rho(r);};
		I[0][i] = integrate(integrand, 0.0, r);
		I[1][i] = integrand(r);
		I[2][i] = 4.0 * M_PI * r * (2.0 * rho(r) + r * drho(r));
		I[3][i] = 4.0 * M_PI * (2.0 * rho(r) + 4.0 * r * drho(r) + r * r * d2rho(r));
	}
	auto dM0 = I[3][0] / 6.0;
	const auto d2Mrdr20 = 4.0 * M_PI * rho(0.0) - 2.0 * dM0;
	auto Mr = func_from_data<4>(I, dr, 0.0, 1.0);

	std::vector<std::function<double(double)>> drho_table;
	drho_table.push_back(rho);
	drho_table.push_back(drho);
	drho_table.push_back(d2rho);
	green_type green;
	green.n = -2;
	green.a.resize(1, 1.0);
	auto dphidr = green_function(green, Mr, drho_table);
	green = green_deriv(green);
	auto dphi2dr2 = green_function(green, Mr, drho_table);
	green = green_deriv(green);
	const double dr0 = 1.0e-6 * dr;
	auto dphi3dr3 = green_function(green, Mr, drho_table);
	for (int i = 0; i < ninterp; i++) {
		const double r = i * dr;
		std::function<double(double)> integrand = [Mr](double r ) {return r == 0.0 ? 0.0 : Mr(r)/(r*r);};
		std::function<double(double)> integrand2 = [Mr,rho](double r ) {return -2.0 * Mr(r) / (r * r * r) + 4.0 * M_PI * rho(r);};
		std::function<double(double)> integrand3 =
				[Mr,rho,drho](double r ) {return 6.0 * Mr(r) / (r * r * r * r) - 8.0 * M_PI * rho(r) / r + 4.0 * M_PI * drho(r);};
		I[0][i] = -1.0 + integrate(integrand, 1.0, r);
		I[1][i] = r == 0.0 ? 0.5 * (dphidr(dr0) + dphidr(-dr0)) : dphidr(r);
		I[2][i] = r == 0.0 ? 0.5 * (dphi2dr2(dr0) + dphi2dr2(-dr0)) : dphi2dr2(r);
		I[3][i] = r == 0.0 ? 0.5 * (dphi3dr3(dr0) + dphi3dr3(-dr0)) : dphi3dr3(r);
	}
	output_interpolation_code<4>("test", I, ninterp, dr);
	return func_from_data<4>(I, dr, 0.0, 1.0);
}

polynomial poly_truncate(const polynomial& A) {
	polynomial B(A.size());
	for (int i = 0; i < A.size(); i++) {
		if (fabs(A[i]) >= 1.19e-7 * abs(A[0])) {
			B[i] = A[i];
		} else {
			B[i] = 0.0;
		}
	}
	poly_shrink_to_fit(B);
	return B;
}

template<int N>
std::vector<double> create_stencil() {
	std::vector<double> stencil;
	constexpr int M = 2 * N + 1;
	matrix<M> A;
	for (int n = 0; n < M; n++) {
		double x = n - N;
		double x0 = 1.0;
		for (int m = 0; m < M; m++) {
			A[n][m] = x0;
			x0 *= x;
		}
	}
	A = A.inverse();
	for (int i = 0; i < M; i++) {
		stencil.push_back(A[1][i]);
	}
	return stencil;
}

int main() {

	auto stencil = create_stencil<STENCIL_ORDER>();
	polynomial basis(3);
	basis[0] = 1.0;
	basis[1] = 0.0;
	basis[2] = -1.0;
	auto rho = basis;
	for (int i = 0; i < 3; i++) {
		rho = poly_mult(rho, basis);
	}
	rho = poly_normalize(rho);

#define Power(a,b) pow(a,b)
#define Pi M_PI

//	double coeffs[9] = { 1, -0.16666666666666666 * Power(Pi, 2), Power(Pi,4) / 120., -0.0001984126984126984 * Power(Pi, 6), Power(Pi,8) / 362880.,
//			-2.505210838544172e-8 * Power(Pi, 10), Power(Pi,12) / 6.2270208e9, -7.647163731819816e-13 * Power(Pi, 14), Power(Pi,16) / 3.55687428096e14 };

//	rho.resize(18);
//	for (int i = 0; i < 9; i++) {
//		rho[2 * i] = coeffs[i];
//		rho[2 * i + 1] = 0.0;
//	}
//	auto rho0 = rho;
//	rho = poly_mult(rho, rho0);
//	rho = poly_mult(rho, rho0);
//	rho = poly_normalize(rho);
//	rho = poly_truncate(rho);
	/*auto drhodx = [rho](int n, double r ) {
	 polynomial rho0 = rho;
	 for( int i = 0; i < n; i++) {
	 rho0 = poly_derivative(rho0);
	 }
	 return poly_eval(rho0, r);
	 };
	 create_potentials_code(drhodx, 9);*/

	polynomial Y(1);
	Y[0] = 1.0;
	conditional ngp;
	ngp.f = Y;
	ngp.x.first = -.5;
	ngp.x.second = .5;
	std::vector<conditional> cloud(1, ngp);
	for (int i = 0; i < CLOUD_ORDER; i++) {
		cloud = next_cloud(cloud);
	}
	cloud = conditionals_remove_negatives(cloud);
#ifndef BSPLINE
#ifdef LUCY
	auto rho = basis1;
	for (int i = 0; i < p; i++) {
		rho = poly_mult(rho, basis2);
	}
	rho = poly_normalize(rho);
#else
#ifdef WENDLAND
	auto rho = Wendland(4,2);
#else
#endif
#endif
	auto pot = poly_rho2pot(rho);
	pot = poly_truncate(pot);
	for (double r = 0.0; r < 1.0; r += 0.01) {
		//	printf("%e %e %e %e\n", r, -poly_eval(pot, r), pot0(r), (-poly_eval(pot, r) - pot0(r)) / poly_eval(pot, r));
	}
	auto force = poly_rho2force(rho);
	force = poly_truncate(force);
	auto expansion = poly_pot2expansion(pot, ORDER);
	for (int i = 0; i < ORDER; i++) {
		expansion.r[i] = poly_truncate(expansion.r[i]);
	}

	auto kmax = 8.0 * M_PI;
	auto filter = poly_filter(rho, kmax);

	print_green_direct(rho, pot, force, expansion, filter, kmax);
#else
	auto rho = Bspline(ORDER - 3);
	auto pot = conditionals_to_potential(rho);
	auto force = pot;
	for (int i = 0; i < pot.size(); i++) {
		force[i].r.f = poly_neg(poly_derivative(pot[i].r.f));
		force[i].i = pot[i].i;
	}
#endif

	tprint("\n");
	tprint("#define CLOUD_MIN -%i\n", CLOUD_ORDER / 2);
	tprint("#define CLOUD_MAX %i\n", CLOUD_ORDER / 2 + 1);
	tprint("\n");

	tprint("inline CUDA_EXPORT float cloud_weight(float x) {\n");
	indent();
	tprint("float y;\n");
	tprint("x = abs(x);\n");
	tprint("const float x2 = x * x;\n");
	print_conditionals_fma(cloud);
	tprint("return y;\n");
	deindent();
	tprint("}\n");
	tprint("\n");

	tprint("inline CUDA_EXPORT float cloud_filter(float kh) {\n");
	indent();
	tprint("const double s = sinc(0.5 * kh);\n");
	tprint("return pow(s, -%i);\n", CLOUD_ORDER + 1);
	deindent();
	tprint("}\n");

	tprint("\n");
	/*	tprint("inline CUDA_EXPORT float apply_stencil(const array<float, %i>& y ) {\n", 2 * STENCIL_ORDER + 1);
	 indent();
	 tprint("float w = 0.0f;\n", stencil[0]);
	 for (int i = 0; i <= STENCIL_ORDER; i++) {
	 int j = i;
	 if (stencil[j] != 0.0) {
	 tprint("w = fmaf( float(%.8e), y[%i], w);\n", stencil[j], j);
	 }
	 j = 2 * STENCIL_ORDER - i;
	 if (stencil[j] != 0.0) {
	 tprint("w = fmaf( float(%.8e), y[%i], w);\n", stencil[j], j);
	 }
	 }
	 tprint("return w;\n");
	 deindent();
	 tprint("}\n");
	 tprint("\n");
	 tprint("#define STENCIL_ORDER %i\n", STENCIL_ORDER);*/

}