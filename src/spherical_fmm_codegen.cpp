#include <stdio.h>
#include <cmath>
#include <utility>
#include <algorithm>
#include <vector>

static int ntab = 0;

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

constexpr double nonepow(int m) {
	return m % 2 == 0 ? double(1) : double(-1);
}

int z_rot(int P, const char* name, bool noevenhi) {
	int flops = 0;
	tprint("{\n");
	indent();
	tprint("T tmp;\n");
	int mmin = 1;
	for (int m = 1; m <= P; m++) {
		for (int l = m; l <= P; l++) {
			if (noevenhi && l == P) {
				if ((((P + l) / 2) % 2 == 1) ? m % 2 == 1 : m % 2 == 0) {
					continue;
				}
			}
			if (noevenhi && (l >= P - 1 && (m % 2 != ((P + l) / 2) % 2))) {
				tprint("%s[%i] = %s[%i] * Ry + %s[%i] * Rx;\n", name, index(l, m), name, index(l, -m), name, index(l, -m));
				tprint("%s[%i] *= Rx;\n", name, index(l, m), name, index(l, m));
				flops += 4;
			} else {
				tprint("tmp = %s[%i];\n", name, index(l, m));
				tprint("%s[%i] = %s[%i] * Rx - %s[%i] * Ry;\n", name, index(l, m), name, index(l, m), name, index(l, -m));
				tprint("%s[%i] = tmp * Ry + %s[%i] * Rx;\n", name, index(l, -m), name, index(l, -m));
				flops += 6;
			}

		}
		if (m != P) {
			tprint("tmp = Rx;\n");
			tprint("Rx = Rx * cosphi - Ry * sinphi;\n");
			tprint("Ry = tmp * sinphi + Ry * cosphi;\n");
			flops += 6;
		}
	}
	deindent();
	tprint("}\n");
	return flops;
}

double factorial(int n) {
	return n == 0 ? 1.0 : n * factorial(n - 1);
}

int m2l(int P, const char* mname, const char* lname) {
	int flops = 0;
	tprint("{\n");
	indent();
	tprint("T coeff[P+1];\n");
	tprint("coeff[0] = rinv;\n");
	for (int n = 1; n <= P; n++) {
		tprint("coeff[%i] = rinv * coeff[%i];\n", n, n - 1);
		flops++;
	}
	for (int n = 1; n <= P; n++) {
		tprint("coeff[%i] *= T(%.16e);\n", n, n - 1, factorial(n));
		flops++;
	}
	for (int n = 0; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			int k = m;
			tprint("%s[%i] = %s%s[%i] * coeff[%i];\n", lname, index(n, m), m % 2 == 0 ? "" : "-", mname, index(k, m), n + k);
			flops += 1 + (m % 2 == 1);
			if (n != 0) {
				tprint("%s[%i] = %s%s[%i] * coeff[%i];\n", lname, index(n, -m), m % 2 == 0 ? "" : "-", mname, index(k, -m), n + k);
				flops += 1 + (m % 2 == 1);
			}
			for (int k = m + 1; k <= std::min(P - n, P - 1); k++) {
				tprint("%s[%i] %s= %s[%i] * coeff[%i];\n", lname, index(n, m), m % 2 == 0 ? "+" : "-", mname, index(k, m), n + k);
				if (n != 0) {
					tprint("%s[%i] %s= %s[%i] * coeff[%i];\n", lname, index(n, -m), m % 2 == 0 ? "+" : "-", mname, index(k, -m), n + k);
					flops += 2;
				}
				flops += 2;
			}
		}
	}
	deindent();
	tprint("}\n");
	return flops;

}

int xz_swap(int P, const char* name, bool inv, bool m_restrict, bool l_restrict, bool noevenhi) {
	tprint("{\n");
	indent();
	tprint("array<T, %i> A;\n", 2 * P + 1);
	tprint("T tmp;\n");
	int flops = 0;
	auto brot = [inv](int n, int m, int l) {
		if( inv ) {
			return Brot(n,l,m);
		} else {
			return Brot(n,m,l);
		}
	};
	for (int n = 1; n <= P; n++) {
		int lmax = n;
		if (l_restrict && lmax > (P) - n) {
			lmax = (P) - n;
		}
		for (int m = -lmax; m <= lmax; m++) {
			tprint("A[%i] = %s[%i];\n", m + P, name, m + P);
		}
		std::vector < std::vector<std::pair<float, int>>>ops(2 * n + 1);
		int mmax = n;
		if (m_restrict && mmax > (P + 1) - n) {
			mmax = (P + 1) - n;
		}
		int mmin = 0;
		int stride = 1;
		if (noevenhi && n == P) {
			mmin = (((P + n) / 2) % 2 == 1 ? 1 : 0);
			stride = 2;
		}
		for (int m = 0; m <= mmax; m += stride) {
			for (int l = 0; l <= lmax; l++) {
				double r = l == 0 ? brot(n, m, 0) : brot(n, m, l) + nonepow(l) * brot(n, m, -l);
				double i = l == 0 ? 0.0 : brot(n, m, l) - nonepow(l) * brot(n, m, -l);
				if (r != 0.0) {
					ops[n + m].push_back(std::make_pair(r, P + l));
				}
				if (i != 0.0 && m > 0) {
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
						tprint("%s[%i] %s= A[%i];\n", name, index(n, n - m), l == 0 ? "" : "+", ops[m][l].second);
						flops += 1 - (l == 0);
					} else {
						tprint("%s[%i] %s= T(%.16e) * A[%i];\n", name, index(n, n - m), l == 0 ? "" : "+", ops[m][l].first, ops[m][l].second);
						flops += 2 - (l == 0);
					}
				} else {
					tprint("tmp = A[%i];\n", ops[m][l].second);
					for (int p = 1; p < len; p++) {
						tprint("tmp += A[%i];\n", ops[m][l + p].second);
						flops++;
					}
					tprint("%s[%i] %s= T(%.16e) * tmp;\n", name, index(n, n - m), l == 0 ? "" : "+", ops[m][l].first);
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
	int P = 7;
	int flops = 0;
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT array<T, %i> spherical_M2L(array<T, %i> M, T x, T y, T z) {\n");
	indent();
	tprint("const T R2 = (x * x + y * y);\n");
	flops += 3;
	tprint("const T R = sqrt(x * x + y * y);\n");
	flops += 7;
	tprint("const T Rinv = T(1) / R;\n");
	flops += 4;
	tprint("const T r = sqrt(z * z + R2);\n");
	flops += 6;
	tprint("const T rinv = T(1) / r;\n");
	flops += 4;
	tprint("T Rx0;\n");
	tprint("T Ry0;\n");
	tprint("T Rx;\n");
	tprint("T Ry;\n");

	tprint("Rx = y * Rinv;\n");
	flops++;
	tprint("Ry = x * Rinv;\n");
	flops++;
	flops += z_rot(P - 1, "M", false);

	flops += xz_swap(P - 1, "M", false, false, false, false);

	tprint("Rx0 = Rx;\n");
	tprint("Ry0 = Ry;\n");
	tprint("Rx = z * rinv;\n");
	flops++;
	tprint("Ry = -R * rinv;\n");
	flops += 2;
	flops += z_rot(P - 1, "M", false);

	flops += xz_swap(P - 1, "M", false, true, false, false);
	flops += m2l(P, "M", "L");

	flops += xz_swap(P, "L", true, false, true, false);

	tprint("Ry = -Ry;\n");
	flops += 1;
	flops += z_rot(P, "L", true);
	flops += xz_swap(P, "L", true, false, false, true);
	tprint("Rx = Rx0;\n");
	tprint("Ry = -Ry0;\n");
	flops += 1;
	flops += z_rot(P, "L", false);
	tprint("//FLOPS = %i\n", flops);
	deindent();
	tprint("}");
	return 0;
}
