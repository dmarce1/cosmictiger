#include <vector>
#include <cmath>
#include <string>

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
		str += std::to_string(std::abs(A[1])) + "*x";
	}
	for (int i = 2; i < A.size(); i++) {
		if (A[i] != 0.0) {
			str += A[i] > 0.0 ? " + " : " - ";
			str += std::to_string(std::abs(A[i])) + "*x^" + std::to_string(i);
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
	double sum = A.back();
	while (A.size() > 1) {
		A.pop_back();
		sum = std::fma(sum, x, A.back());
	}
	return sum;
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

void print_green_direct(polynomial pot, polynomial force, expansion_type expansion, polynomial filter, double kmax) {
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
	tprint("\n");

	tprint("\n");
	tprint("CUDA_EXPORT inline float green_phi0(float nparts, float rs) {\n", L);
	indent();
	tprint("return float(%.8e) * sqr(rs) * (nparts - 1) +  float(%.8e) / rs;\n", -4.0 * M_PI * filter[2], pot[0]);
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

int main() {
	int p = ORDER - 1;
	polynomial basis1(2);
	polynomial basis2(2);
	basis1[0] = 1.0;
	basis1[1] = p;
	basis2[0] = 1.0;
	basis2[1] = -1.0;

	polynomial basis(3);
	basis[0] = 1.0;
	basis[1] = 0.0;
	basis[2] = -1.0;

	auto rho = basis1;
	for (int i = 0; i < p; i++) {
		rho = poly_mult(rho, basis2);
	}
	rho = poly_normalize(rho);
	auto pot = poly_rho2pot(rho);
	auto force = poly_rho2force(rho);
	auto expansion = poly_pot2expansion(pot, ORDER);

	auto kmax = 8.0 * M_PI;
	auto filter = poly_filter(rho, kmax);

	print_green_direct(pot, force, expansion, filter, kmax);

}
