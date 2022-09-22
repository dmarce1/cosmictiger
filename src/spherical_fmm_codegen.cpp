#include <stdio.h>
#include <cmath>
#include <utility>

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

void do_rotation(int P) {
	tprint("\n");
	tprint("template<class T>");
	tprint("CUDA_EXPORT void spherical_harmonic_rotate(array<T, %i>& O, T x, T y, T z, T R, T Rinv, T r) {\n", (P + 1) * (P + 1));
	indent();
	tprint("const auto A = O;\n");
	int flops = 0;
	for (int n = 1; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			tprint("O[%i] = T(0);\n", index(n, m));
			if (m != 0) {
				tprint("O[%i] = T(0);\n", index(n, -m));
			}
			for (int l = 0; l <= n; l++) {
				double r = l == 0 ? Brot(n, m, 0) : Brot(n, m, l) + nonepow(l) * Brot(n, m, -l);
				double i = l == 0 ? 0.0 : Brot(n, m, l) - nonepow(l) * Brot(n, m, -l);
				//	r *= (1<<n);
				//		i *= (1<<n);
				if (r != 0.0) {
					if (r == 1.0) {
						tprint("O[%i] += A[%i];\n", index(n, m), index(n, l));
						flops++;
					} else {
						tprint("O[%i] += T(%.16e) * A[%i];\n", index(n, m), r, index(n, l));
						flops += 2;
					}
				}
				if (m != 0) {
					if (i != 0.0) {
						if (i == 1.0) {
							tprint("O[%i] += A[%i];\n", index(n, -m), index(n, -l));
							flops++;
						} else {
							tprint("O[%i] += T(%.16e) * A[%i];\n", index(n, -m), i, index(n, -l));
							flops += 2;
						}
					}
				}
			}
		}
	}
	tprint( "// FLOPS = %i\n", flops);
	deindent();
	tprint("}\n");
	tprint("\n");
}

int main() {
	int P = 7;
	do_rotation(P);
	return 0;
}
