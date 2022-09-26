#include <stdio.h>
#include <cmath>
#include <utility>
#include <algorithm>
#include <vector>
#include <climits>

static int ntab = 0;

#define MBITS 23

double factorial(int n) {
	return n == 0 ? 1.0 : n * factorial(n - 1);
}

double Ylm_xy0(int l, int m, double z) {
	if (m > l) {
		return 0.0;
	} else if (l == 0) {
		return 1.0;
	} else if (l == m) {
		return sqrt(1.0 - z * z) * Ylm_xy0(l - 1, l - 1, z) / (2 * l);
	} else {
		return ((2 * l - 1) * z * Ylm_xy0(l - 1, m, z) - Ylm_xy0(l - 2, m, z)) / (l * l - m * m);
	}
}

struct Ylm_max {
	double operator()(int l, int m) {
		int N = 64;
		double next = 0.0;
		int ix = 0;
		int iy = N;
		for (int iz = 0; iz <= N; iz++) {
			next = std::max(next, Ylm_xy0(l, m, (double) iz / N));
		}
		return next;
	}
};

struct Ylm_max_array {
	std::vector<std::vector<double>> a;
	Ylm_max_array(int P) :
			a((P + 1), std::vector<double>((P + 1))) {
		for (int i = 0; i <= P; i++)
			for (int j = 0; j <= i; j++) {
				a[i][j] = 1.001 * Ylm_max()(i, j);
			}
	}
	const double operator()(int l, int m) const {
		return a[l][m];
	}

};

double Brot(int n, int m, int l) {
	if (n == 0 && m == 0 && l == 0) {
		return 1.0;
	} else if (abs(l) > n) {
		return 0.0;
	} else if (m == 0) {
		return 0.5 * (Brot(n - 1, m, l - 1) - Brot(n - 1, m, l + 1));
	} else if (m > 0) {
		return 0.5 * (Brot(n - 1, m - 1, l - 1) + Brot(n - 1, m - 1, l + 1) + 2.0 * Brot(n - 1, m - 1, l));
	} else {
		return 0.5 * (Brot(n - 1, m + 1, l - 1) + Brot(n - 1, m + 1, l + 1) - 2.0 * Brot(n - 1, m + 1, l));
	}
}

bool close21(double a) {
	return std::abs(1.0 - a) < 1.0e-20;
}
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

int index(int l, int m) {
	return l * (l + 1) + m;
}

double nonepow(int m) {
	return m % 2 == 0 ? double(1) : double(-1);
}
int z_rot(int P, const char* name, bool noevenhi) {
	//noevenhi = false;
	int flops = 0;
	tprint("\n");
//	tprint("template<class T>\n");
	tprint("{\n");
//	tprint(" inline void %s( array<T,%i>& %s, T cosphi, T sinphi ) {\n", fname, (P + 1) * (P + 1), name);
	indent();
	tprint("T tmp;\n");
	tprint("T Rx = cosphi;\n");
	tprint("T Ry = sinphi;\n");
	int mmin = 1;
	bool initR = true;
	for (int m = 1; m <= P; m++) {
		for (int l = m; l <= P; l++) {
			if (noevenhi && l == P) {
				if ((((P + l) / 2) % 2 == 1) ? m % 2 == 0 : m % 2 == 1) {
					continue;
				}
			}
			if (!initR) {
				tprint("tmp = Rx;\n");
				tprint("Rx = Rx * cosphi - Ry * sinphi;\n");
				tprint("Ry = tmp * sinphi + Ry * cosphi;\n");
				flops += 6;
				initR = true;
			}

			if (noevenhi && (l >= P - 1 && (m % 2 != ((P + l) / 2) % 2))) {
				tprint("%s[%i] = %s[%i] * Ry;\n", name, index(l, -m), name, index(l, m));
				tprint("%s[%i] *= Rx;\n", name, index(l, m));
				flops += 2;
			} else {
				tprint("tmp = %s[%i];\n", name, index(l, m));
				tprint("%s[%i] = %s[%i] * Rx - %s[%i] * Ry;\n", name, index(l, m), name, index(l, m), name, index(l, -m));
				tprint("%s[%i] = tmp * Ry + %s[%i] * Rx;\n", name, index(l, -m), name, index(l, -m));
				flops += 6;
			}

		}
		initR = false;
	}
	deindent();
	tprint("}\n");
	return flops;
}

int m2l(int P, const char* mname, const char* lname) {
	int flops = 0;
	tprint("{\n");
	indent();
	tprint("T c0[%i];\n", P + 1);
	tprint("c0[0] = rinv;\n");
	for (int n = 1; n <= P; n++) {
		tprint("c0[%i] = rinv * c0[%i];\n", n, n - 1);
		flops++;
	}
	for (int n = 2; n <= P; n++) {
		tprint("c0[%i] *= T(%.16e);\n", n, factorial(n));
		flops++;
	}
	for (int n = 0; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			int k = m;
			const int maxk = std::min(P - n, P - 1);
			if (k <= maxk) {
				tprint("%s[%i] = %s%s[%i] * c0[%i];\n", lname, index(n, m), m % 2 == 0 ? "" : "-", mname, index(k, m), n + k);
				flops += 1 + (m % 2 == 1);
				if (m != 0) {
					tprint("%s[%i] = %s%s[%i] * c0[%i];\n", lname, index(n, -m), m % 2 == 0 ? "" : "-", mname, index(k, -m), n + k);
					flops += 1 + (m % 2 == 1);
				}
				for (int k = m + 1; k <= maxk; k++) {
					tprint("%s[%i] %s= %s[%i] * c0[%i];\n", lname, index(n, m), m % 2 == 0 ? "+" : "-", mname, index(k, m), n + k);
					if (m != 0) {
						tprint("%s[%i] %s= %s[%i] * c0[%i];\n", lname, index(n, -m), m % 2 == 0 ? "+" : "-", mname, index(k, -m), n + k);
						flops += 2;
					}
					flops += 2;
				}
			} else {
				tprint("%s[%i] = T(0);\n", lname, index(n, m));
				if (m != 0) {
					tprint("%s[%i] = T(0);\n", lname, index(n, -m));
				}
			}
		}
	}
	deindent();
	tprint("}\n");
	return flops;

}

int xz_swap(int P, const char* name, bool inv, bool m_restrict, bool l_restrict, bool noevenhi) {
	//noevenhi = false;
	tprint("\n");
//	tprint("template<class T>\n");
//	tprint(" inline void %s( array<T,%i>& %s ) {\n", fname, (P + 1) * (P + 1), name);
	tprint("{\n");
	indent();
	tprint("array<T, %i> A;\n", 2 * P + 1);
	tprint("T tmp;\n");
	int flops = 0;
	auto brot = [inv](int n, int m, int l) {
		if( inv ) {
			return Brot(n,m,l);
		} else {
			return Brot(n,l,m);
		}
	};
	for (int n = 1; n <= P; n++) {
		int lmax = n;
		if (l_restrict && lmax > (P + 1) - n) {
			lmax = (P + 1) - n;
		}
		for (int m = -lmax; m <= lmax; m++) {
			tprint("A[%i] = %s[%i];\n", m + P, name, index(n, m));
		}
		std::vector < std::vector<std::pair<float, int>>>ops(2 * n + 1);
		int mmax = n;
		if (m_restrict && mmax > (P + 1) - n) {
			mmax = (P + 1) - n;
		}
		int mmin = 0;
		int stride = 1;
		if (noevenhi && n == P + 1) {
			mmin = (((P + 1 + n) / 2) % 2 == 1 ? 1 : 0);
			stride = 2;
		}
		for (int m = 0; m <= mmax; m += stride) {
			for (int l = 0; l <= lmax; l++) {
				double r = l == 0 ? brot(n, m, 0) : brot(n, m, l) + nonepow(l) * brot(n, m, -l);
				double i = l == 0 ? 0.0 : brot(n, m, l) - nonepow(l) * brot(n, m, -l);
				if (r != 0.0) {
					ops[n + m].push_back(std::make_pair(r, P + l));
				}
				if (i != 0.0 && m != 0) {
					ops[n - m].push_back(std::make_pair(i, P - l));
				}
			}
		}
		for (int m = 0; m < 2 * n + 1; m++) {
			std::sort(ops[m].begin(), ops[m].end(), [](std::pair<float,int> a, std::pair<float,int> b) {
				return a.first < b.first;
			});
		}
		for (int m = 0; m < 2 * n + 1; m++) {
			for (int l = 0; l < ops[m].size(); l++) {
				int len = 1;
				while (len + l < ops[m].size()) {
					if (ops[m][len + l].first == ops[m][l].first && !close21(ops[m][l].first)) {
						len++;
					} else {
						break;
					}
				}
				if (len == 1) {
					if (close21(ops[m][l].first)) {
						tprint("%s[%i] %s= A[%i];\n", name, index(n, m - n), l == 0 ? "" : "+", ops[m][l].second);
						flops += 1 - (l == 0);
					} else {
						tprint("%s[%i] %s= T(%.16e) * A[%i];\n", name, index(n, m - n), l == 0 ? "" : "+", ops[m][l].first, ops[m][l].second);
						flops += 2 - (l == 0);
					}
				} else {
					tprint("tmp = A[%i];\n", ops[m][l].second);
					for (int p = 1; p < len; p++) {
						tprint("tmp += A[%i];\n", ops[m][l + p].second);
						flops++;
					}
					tprint("%s[%i] %s= T(%.16e) * tmp;\n", name, index(n, m - n), l == 0 ? "" : "+", ops[m][l].first);
					flops += 2 - (l == 0);
				}
				l += len - 1;
			}

		}
	}
	deindent();
	tprint("}\n");
	return flops;
}

int main() {
	tprint("#pragma once\n");
	tprint("\n");
	tprint("#include <cosmictiger/containers.hpp>\n");
	tprint("#include <cosmictiger/cuda.hpp>\n");
	tprint("\n");
	for (int P = 1; P <= 12; P++) {
		int flops = 0;

		tprint("\n");
		tprint("template<class T>\n");
		tprint(" array<T, %i> spherical_M2L(array<T, %i> M, T x, T y, T z) {\n", (P + 1) * (P + 1), P * P);
		indent();
		tprint("array<T,%i> L;\n", (P + 1) * (P + 1));
		tprint("const T R2 = (x * x + y * y);\n");
		flops += 3;
		tprint("const T R = sqrt(R2);\n");
		flops += 4;
		tprint("const T Rinv = T(1) / (R + T(1e-30));\n");
		flops += 5;
		tprint("const T r = sqrt(z * z + R2);\n");
		flops += 6;
		tprint("const T rinv = T(1) / r;\n");
		flops += 4;
		tprint("T cosphi0;\n");
		tprint("T cosphi;\n");
		tprint("T sinphi0;\n");
		tprint("T sinphi;\n");

		tprint("cosphi = y * Rinv;\n");
		flops++;
		tprint("sinphi = x * Rinv;\n");
		flops++;
		tprint("const auto multi_rot = [&M,&cosphi,&sinphi]()\n");
		flops += 2*z_rot(P - 1, "M", false);
		tprint(";\n");
		tprint("multi_rot();\n");


//	flops += xz_swap(P - 1, "M", false, false, false, false);
		flops += xz_swap(P - 1, "M", false, false, false, false);

		tprint("cosphi0 = cosphi;\n");
		tprint("sinphi0 = sinphi;\n");
		tprint("cosphi = z * rinv;\n");
		flops++;
		tprint("sinphi = -R * rinv;\n");
		flops += 2;
		tprint("multi_rot();\n");
		flops += xz_swap(P - 1, "M", false, true, false, false);
		flops += m2l(P, "M", "L");
		flops += xz_swap(P, "L", true, false, true, false);

		tprint("sinphi = -sinphi;\n");
		flops += 1;
		flops += z_rot(P, "L", true);
		//	flops += z_rot(P, "L", true);
		//	flops += xz_swap(P, "L", true, false, false, true);
		flops += xz_swap(P, "L", true, false, false, true);
		tprint("cosphi = cosphi0;\n");
		tprint("sinphi = -sinphi0;\n");
		flops += 1;
		flops += z_rot(P, "L", false);
		tprint("return L;\n");
		tprint("\n");
		tprint("//FLOPS = %i\n", flops);
		deindent();
		tprint("}");
		tprint("\n");
		fprintf(stderr, "%i %i\n", P, flops);
	}
	return 0;
}
