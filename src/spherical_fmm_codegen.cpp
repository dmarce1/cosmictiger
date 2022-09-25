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
		int N = 1024;
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
			a((P+1),std::vector<double>((P+1))) {
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

template<int P>
int multi_bits(int n = 1) {
	if (n == P + 1) {
		return 0;
	} else {
		return (2 * n + 1) * (MBITS - n + 1) + multi_bits<P>(n + 1);
	}
}

int decompress(int P) {
	int nextbit = CHAR_BIT;
	int byte = -1;
	const auto read_bits = [&nextbit,&byte](int count) {
//		int res = 0;
			if (nextbit == CHAR_BIT) {
				nextbit = 0;
				byte++;
			}
			if (count <= CHAR_BIT - nextbit) {
//			res = bits[byte] >> nextbit;
				tprint( "i = bits[%i] >> %i;\n", byte, nextbit);
//	res &= (1 << count) - 1;
				tprint( "i &= %i;\n", (1<<count)-1);
				nextbit += count;
//			return res;
			} else {
				int n = 0;
//			res = bits[byte] >> nextbit;
				tprint( "i = bits[%i] >> %i;\n", byte, nextbit);
				n += CHAR_BIT - nextbit;
				nextbit = 0;
				byte++;
				while (count >= CHAR_BIT + n) {
//				res |= ((int) bits[byte]) << n;
					tprint( "i |= ((int) bits[%i]) << %i;\n", byte, n);
					byte++;
					n += CHAR_BIT;
				}
				if (count - n > 0) {
					//		res |= (((int) bits[byte]) & ((1 << (count - n)) - 1)) << (n);
					tprint( "i |= (((int) bits[%i]) & %i;\n", byte, ((1 << (count - n)) - 1) << (n));
					nextbit = count - n;
				}
				//	return res;
			}
		};

	int flops = 0;
	tprint( "template<class T, int P>\n");
	tprint( "struct compressed_multipole {\n");
	indent();
	tprint( "array<T,%i>& bits;\n", (P+1)*(P+1));
	tprint( "T scale;\n");
	tprint( "T mass;\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint(" array<T,%i> spherical_multipole_decompress(const array<T,%i>& bits) {\n");
	indent();
	tprint("array<T,%i> M;\n");

	const Ylm_max_array norms(P);
	tprint( "T rpow = T(1);\n");
//	spherical_expansion<T, P> O;
//	O[0].real() = mass;
//	O[0].imag() = 0.f;
	tprint( "int i;\n");
	tprint( "int s;\n");
	tprint( "T v;\n");
	tprint( "M[0] = mass;\n");
	for (int n = 1; n <= P; n++) {
		for (int m = -n; m <= n; m++) {
			read_bits(MBITS - n + 1);
//			int sgn = i & 1;
			tprint( "s = i & 1;\n");
//			i >>= 1;
			tprint( "i >>= 1;\n");
			float value = (sgn ? -1.0f : 1.0f) * (float) i / (float) (1 << (MBITS - n));
			value *= rpow * mass * norms(n, abs(m));
			if (m >= 0) {
				O[index(n, m)].real() = value;
			} else {
				O[index(n, -m)].imag() = value;
			}
		}
		rpow *= r;
	}
	return O;


	deindent();
	tprint("}\n");
	tprint("\n");
	return flops;
}

int z_rot(const char* fname, int P, const char* name, bool noevenhi) {
	//noevenhi = false;
	int flops = 0;
	tprint("\n");
	tprint("template<class T>\n");
	tprint(" inline void %s( array<T,%i>& %s, T cosphi, T sinphi ) {\n", fname, (P + 1) * (P + 1), name);
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

int xz_swap(const char* fname, int P, const char* name, bool inv, bool m_restrict, bool l_restrict, bool noevenhi) {
	//noevenhi = false;
	tprint("\n");
	tprint("template<class T>\n");
	tprint(" inline void %s( array<T,%i>& %s ) {\n", fname, (P + 1) * (P + 1), name);
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
	for (int P = 1; P <= 12; P++) {
		int flops = 0;
		tprint("#pragma once\n");
		tprint("#\n");
		tprint("#include <cosmictiger/containers.hpp>\n");
		tprint("#include <cosmictiger/cuda.hpp>\n");
		tprint("\n");
		flops += 2 * z_rot("spherical_rotate_z_multipole", P - 1, "M", false);
		flops += z_rot("spherical_rotate_z_expansion_abridged", P, "L", true);
		flops += z_rot("spherical_rotate_z_expansion_full", P, "L", false);
		flops += xz_swap("spherical_swap_zx_multipole_full", P - 1, "M", false, false, false, false);
		flops += xz_swap("spherical_swap_zx_multipole_abridged", P - 1, "M", false, true, false, false);
		flops += xz_swap("spherical_swap_zx_expansion_abridged1", P, "L", true, false, true, false);
		flops += xz_swap("spherical_swap_zx_expansion_abridged2", P, "L", true, false, false, true);

		tprint("\n");
		tprint("template<class T>\n");
		tprint(" array<T, %i> spherical_M2L(array<T, %i> M, T x, T y, T z) {\n", (P + 1) * (P + 1), P * P);
		indent();
		tprint("array<T,%i> L;\n", (P + 1) * (P + 1));
		tprint("const T R2 = (x * x + y * y);\n");
		flops += 3;
		tprint("const T R = sqrt(R2);\n");
		flops += 4;
		tprint("const T Rinv = T(1) / (R + T(1-e30));\n");
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
		tprint("spherical_rotate_z_multipole(M, cosphi, sinphi);\n");

//	flops += xz_swap(P - 1, "M", false, false, false, false);
		tprint("spherical_swap_zx_multipole_full(M);\n");

		tprint("cosphi0 = cosphi;\n");
		tprint("sinphi0 = sinphi;\n");
		tprint("cosphi = z * rinv;\n");
		flops++;
		tprint("sinphi = -R * rinv;\n");
		flops += 2;
		tprint("spherical_rotate_z_multipole(M, cosphi, sinphi);\n");

//	flops += xz_swap(P - 1, "M", false, true, false, false);
//	flops += xz_swap(P - 1, "M", false, false, false, false);
		tprint("spherical_swap_zx_multipole_abridged(M);\n");

		flops += m2l(P, "M", "L");

//	flops += xz_swap(P, "L", true, false, true, false);
		tprint("spherical_swap_zx_expansion_abridged1(L);\n");

		tprint("sinphi = -sinphi;\n");
		flops += 1;
		tprint("spherical_rotate_z_expansion_abridged(L, cosphi, sinphi);\n");
		//	flops += z_rot(P, "L", true);
		//	flops += xz_swap(P, "L", true, false, false, true);
		tprint("spherical_swap_zx_expansion_abridged2(L);\n");
		tprint("cosphi = cosphi0;\n");
		tprint("sinphi = -sinphi0;\n");
		flops += 1;
		tprint("spherical_rotate_z_expansion_full(L, cosphi, sinphi);\n");
		tprint("return L;\n");
		tprint("//FLOPS = %i\n", flops);
		deindent();
		tprint("}");
		tprint("\n");
		fprintf(stderr, "%i %i\n", P, flops);
	}
	return 0;
}
