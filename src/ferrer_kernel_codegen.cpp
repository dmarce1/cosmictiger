#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

#include <cosmictiger/defs.hpp>

static int ntab = 0;

void indent() {
	ntab++;
}

void deindent() {
	ntab--;
}

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
		while (A.back() == 0.0 && A.size()) {
			A.pop_back();
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

bool poly_even_only(polynomial A) {
	for (int i = 1; i < A.size(); i += 2) {
		if (A[i] != 0.0) {
			return false;
		}
	}
	return true;
}

void poly_print_fma_instructions(polynomial A, const char* varname = "q", const char* type = "float") {
	if (poly_even_only(A)) {
		tprint("y = %s(%.8e);\n", type, A.back());
		while (A.size() > 1) {
			A.pop_back();
			A.pop_back();
			if (A.back() != 0.0) {
				tprint("y = fma( y, %s2, %s(%.8e) );\n", varname, type, A.back());
			} else {
				tprint("y *= q2;\n");
			}
		}
	} else {
		tprint("y = %s(%.8e);\n", type, A.back());
		while (A.size() > 1) {
			A.pop_back();
			if (A.back() != 0.0) {
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

struct conditional {
	std::pair<double, double> x;
	polynomial f;
};

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

void print_green_direct(polynomial rho, polynomial pot, polynomial force, expansion_type expansion, polynomial filter, double kmax) {
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

	const int L = ORDER;
	tprint("\n");
	tprint("CUDA_EXPORT inline array<float, %i> green_kernel(float r, float rsinv, float rsinv2) {\n", L);
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

}

//#define LUCY

#define CLOUD_ORDER 4

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

void conditionals_sort(std::vector<conditional>& A) {
	std::sort(A.begin(), A.end(), [](conditional a, conditional b) {
		return a.x.first < b.x.first;
	});
}

struct potential_type {
	conditional r;
	double i;
};

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
		fflush(stdout);
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

std::vector<conditional> conditionals_to_force(const std::vector<conditional>& A) {
}

int main() {

	auto B = conditionals_to_mass_function(Bspline(2));
	for (double x = 0.0; x < 1.0; x += 0.01) {
		printf("%e %e\n", x, conditionals_eval(B, x));
	}

	return 0;
	polynomial Y(1);
	Y[0] = 1.0;
	Y[1] = 0.0;
	conditional ngp;
	ngp.f = Y;
	ngp.x.first = -.5;
	ngp.x.second = .5;
	std::vector<conditional> cloud(1, ngp);
	for (int i = 0; i < CLOUD_ORDER; i++) {
		cloud = next_cloud(cloud);
	}
	cloud = conditionals_remove_negatives(cloud);
#ifdef LUCY
	auto rho = basis1;
	for (int i = 0; i < p; i++) {
		rho = poly_mult(rho, basis2);
	}
#else
	polynomial basis(3);
	basis[0] = 1.0;
	basis[1] = 0.0;
	basis[2] = -1.0;
	auto rho = basis;
	for (int i = 0; i < ORDER - 3; i++) {
		rho = poly_mult(rho, basis);
	}
#endif
	rho = poly_normalize(rho);
	auto pot = poly_rho2pot(rho);
	auto force = poly_rho2force(rho);
	auto expansion = poly_pot2expansion(pot, ORDER);

	auto kmax = 8.0 * M_PI;
	auto filter = poly_filter(rho, kmax);

	print_green_direct(rho, pot, force, expansion, filter, kmax);

	tprint("\n");
	tprint("#define CLOUD_MIN -%i\n", CLOUD_ORDER / 2);
	tprint("#define CLOUD_MAX %i\n", CLOUD_ORDER / 2 + 1);
	tprint("\n");

	tprint("inline CUDA_EXPORT float cloud_weight(float x) {\n");
	indent();
	tprint("float y;\n");
	tprint("x = abs(x);\n");
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

}
