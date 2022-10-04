#include <stdio.h>
#include <cmath>
#include <utility>
#include <algorithm>
#include <vector>
#include <climits>
#include <unordered_map>
#include <cosmictiger/complex.hpp>
static int ntab = 0;
static int tprint_on = true;

static bool nophi = false;
static bool fmaops = true;

CUDA_EXPORT constexpr int cindex(int l, int m) {
	return l * (l + 1) / 2 + m;
}

template<class T>
T nonepow(int m) {
	return m % 2 == 0 ? double(1) : double(-1);
}

template<class T, int P>
struct spherical_expansion: public std::array<complex<T>, (P + 1) * (P + 2) / 2> {
	CUDA_EXPORT
	inline complex<T> operator()(int n, int m) const {
		if (m >= 0) {
			return (*this)[cindex(n, m)];
		} else {
			return (*this)[cindex(n, -m)].conj() * nonepow<T>(m);
		}
	}
	void print() const {
		for (int l = 0; l <= P; l++) {
			for (int m = 0; m <= l; m++) {
				printf("%e + i %e  ", (*this)(l, m).real(), (*this)(l, m).imag());
			}
			printf("\n");
		}
	}
};

double factorial(int n) {
	return n == 0 ? 1.0 : n * factorial(n - 1);
}

struct hash {
	size_t operator()(std::array<int, 3> i) const {
		return i[0] * 12345 + i[1] * 42 + i[2];
	}
};

double Brot(int n, int m, int l) {
	static std::unordered_map<std::array<int, 3>, double, hash> values;
	std::array<int, 3> key;
	key[0] = n;
	key[1] = m;
	key[2] = l;
	if (values.find(key) != values.end()) {
		return values[key];
	} else {
		double v;
		if (n == 0 && m == 0 && l == 0) {
			v = 1.0;
		} else if (abs(l) > n) {
			v = 0.0;
		} else if (m == 0) {
			v = 0.5 * (Brot(n - 1, m, l - 1) - Brot(n - 1, m, l + 1));
		} else if (m > 0) {
			v = 0.5 * (Brot(n - 1, m - 1, l - 1) + Brot(n - 1, m - 1, l + 1) + 2.0 * Brot(n - 1, m - 1, l));
		} else {
			v = 0.5 * (Brot(n - 1, m + 1, l - 1) + Brot(n - 1, m + 1, l + 1) - 2.0 * Brot(n - 1, m + 1, l));
		}
		values[key] = v;
		return v;
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
	if (tprint_on) {
		for (int i = 0; i < ntab; i++) {
			printf("\t");
		}
		printf(str, std::forward<Args>(args)...);
	}
}

void tprint(const char* str) {
	if (tprint_on) {
		for (int i = 0; i < ntab; i++) {
			printf("\t");
		}
		printf("%s", str);
	}
}

void set_tprint(bool c) {
	tprint_on = c;
}

int index(int l, int m) {
	return l * (l + 1) + m;
}

int greens_body(int P) {
	int flops = 0;
	tprint("const T r2 = x * x + y * y + z * z;\n");
	flops += 5;
	tprint("const T r2inv = T(1) / r2;\n");
	flops += 4;
	tprint("O[0] = SQRT(r2inv);\n");
	tprint("O[%i] = T(0);\n", (P + 1) * (P + 1));
	flops += 4;
	tprint("x *= r2inv;\n");
	tprint("y *= r2inv;\n");
	tprint("z *= r2inv;\n");
	flops += 3;
	tprint("T ax;\n");
	tprint("T ay;\n");
	for (int m = 0; m <= P; m++) {
		if (m == 1) {
			tprint("O[%i] = x * O[0];\n", index(m, m));
			tprint("O[%i] = y * O[0];\n", index(m, -m));
			flops += 2;
		} else if (m > 0) {
			tprint("ax = O[%i] * T(%i);\n", index(m - 1, m - 1), 2 * m - 1);
			tprint("ay = O[%i] * T(%i);\n", index(m - 1, -(m - 1)), 2 * m - 1);
			tprint("O[%i] = x * ax - y * ay;\n", index(m, m));
			tprint("O[%i] = FMA(y, ax, x * ay);\n", index(m, -m));
			flops += 8 - fmaops;
		}
		if (m + 1 <= P) {
			tprint("O[%i] = T(%i) * z * O[%i];\n", index(m + 1, m), 2 * m + 1, index(m, m));
			flops += 2;
			if (m != 0) {
				tprint("O[%i] = T(%i) * z * O[%i];\n", index(m + 1, -m), 2 * m + 1, index(m, -m));
				flops += 2;
			}
		}
		for (int n = m + 2; n <= P; n++) {
			if (m != 0) {
				tprint("ax = T(%i) * z;\n", 2 * n - 1);
				tprint("ay = T(-%i) * r2inv;\n", (n - 1) * (n - 1) - m * m);
				tprint("O[%i] = (ax * O[%i] + ay * O[%i]);\n", index(n, m), index(n - 1, m), index(n - 2, m));
				tprint("O[%i] = FMA(ax, O[%i], ay * O[%i]);\n", index(n, -m), index(n - 1, -m), index(n - 2, -m));
				flops += 8 - fmaops;
			} else {
				if ((n - 1) * (n - 1) - m * m == 1) {
					tprint("O[%i] = (T(%i) * z * O[%i] - r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), index(n - 2, m));
					flops += 4;
				} else {
					tprint("O[%i] = (T(%i) * z * O[%i] - T(%i) * r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), (n - 1) * (n - 1) - m * m,
							index(n - 2, m));
					flops += 5;
				}

			}
		}
	}
	return flops;
}

int m2l_cp(int P) {
	int flops = 0;
	tprint("\n");
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void P2L%s( expansion_type<T,%i>& L, T x, T y, T z ) {\n", nophi ? "_nopot" : "", P);
	indent();
	tprint("expansion_type<T, %i> O;\n", P);
	greens_body(P);
	for (int n = nophi; n < (P + 1) * (P + 1); n++) {
		tprint("L[%i] += O[%i];\n", n, n);
		flops++;
	}
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int greens(int P) {
	int flops = 0;
	tprint("\n");
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void greens(expansion_type<T,%i>& O, T x, T y, T z ) {\n", P);
	indent();
	greens_body(P);
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int greens_xz(int P) {
	int flops = 0;
	tprint("\n");
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void greens_xz(expansion_xz_type<T,%i>& O, T x, T z, T r2inv ) {\n", P);
	indent();
	tprint("O[0] = SQRT(r2inv);\n");
	tprint("O[%i] = T(0);\n", (P + 1) * (P + 1));
	flops += 4;
	tprint("x *= r2inv;\n");
	flops += 1;
	tprint("z *= r2inv;\n");
	flops += 1;
	tprint("T ax;\n");
	tprint("T ay;\n");
	const auto index = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	for (int m = 0; m <= P; m++) {
		if (m == 1) {
			tprint("O[%i] = x * O[0];\n", index(m, m));
			flops += 1;
		} else if (m > 0) {
			tprint("ax = O[%i] * T(%i);\n", index(m - 1, m - 1), 2 * m - 1);
			tprint("O[%i] = x * ax;\n", index(m, m));
			flops += 2;
		}
		if (m + 1 <= P) {
			tprint("O[%i] = T(%i) * z * O[%i];\n", index(m + 1, m), 2 * m + 1, index(m, m));
			flops += 2;
		}
		for (int n = m + 2; n <= P; n++) {
			if (m != 0) {
				tprint("ax = T(%i) * z;\n", 2 * n - 1);
				tprint("ay = T(-%i) * r2inv;\n", (n - 1) * (n - 1) - m * m);
				tprint("O[%i] = FMA(ax, O[%i], ay * O[%i]);\n", index(n, m), index(n - 1, m), index(n - 2, m));
				flops += 5 - fmaops;
			} else {
				if ((n - 1) * (n - 1) - m * m == 1) {
					tprint("O[%i] = (T(%i) * z * O[%i] - r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), index(n - 2, m));
					flops += 4;
				} else {
					tprint("O[%i] = (T(%i) * z * O[%i] - T(%i) * r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), (n - 1) * (n - 1) - m * m,
							index(n - 2, m));
					flops += 5;
				}
			}
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int z_rot(int P, const char* name, bool noevenhi, bool exclude) {
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
				tprint("Ry = FMA(tmp, sinphi, Ry * cosphi);\n");
				flops += 6 - fmaops;
				initR = true;
			}
			if (exclude && l == P && m % 2 == 1) {
				tprint("%s[%i] = -%s[%i] * Ry;\n", name, index(l, m), name, index(l, -m));
				tprint("%s[%i] *= Rx;\n", name, index(l, -m));
				flops += 3;
			} else if (exclude && l == P && m % 2 == 0) {
				tprint("%s[%i] = %s[%i] * Ry;\n", name, index(l, -m), name, index(l, m));
				tprint("%s[%i] *= Rx;\n", name, index(l, m));
				flops += 2;
			} else {
				if (noevenhi && ((l >= P - 1 && m % 2 == P % 2))) {
					tprint("%s[%i] = %s[%i] * Ry;\n", name, index(l, -m), name, index(l, m));
					tprint("%s[%i] *= Rx;\n", name, index(l, m));
					flops += 2;
				} else {
					tprint("tmp = %s[%i];\n", name, index(l, m));
					tprint("%s[%i] = %s[%i] * Rx - %s[%i] * Ry;\n", name, index(l, m), name, index(l, m), name, index(l, -m));
					tprint("%s[%i] = FMA(tmp, Ry, %s[%i] * Rx);\n", name, index(l, -m), name, index(l, -m));
					flops += 6 - fmaops;
				}
			}

		}
		initR = false;
	}
	deindent();
	tprint("}\n");
	return flops;
}

int m2l(int P, int Q, const char* mname, const char* lname) {
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
	tprint("T Lr;\n");
	tprint("T Li;\n");
//	tprint("for( int n = 0; n < %i; n++) {", Q == 1 ? 4 : (Q + 1) * (Q + 1) + 1);
//	indent();
//	tprint("L[n] = T(0);\n");
//	deindent();
//	tprint("}\n");
	for (int n = nophi; n <= Q; n++) {
		for (int m = 0; m <= n; m++) {
			if (m % 2 == 0) {
				bool pfirst = true;
				bool nfirst = true;
				const int maxk = std::min(P - n, P - 1);
				bool looped = true;
				for (int k = m; k <= maxk; k++) {
					looped = true;
					if (pfirst) {
						pfirst = false;
						tprint("%s[%i] = %s[%i] * c0[%i];\n", lname, index(n, m), mname, index(k, m), n + k);
						flops += 1;
					} else {
						tprint("%s[%i] = FMA(%s[%i], c0[%i], %s[%i]);\n", lname, index(n, m), mname, index(k, m), n + k, lname, index(n, m));
						flops += 2 - fmaops;
					}
					if (m != 0) {
						if (nfirst) {
							nfirst = false;
							tprint("%s[%i] = %s[%i] * c0[%i];\n", lname, index(n, -m), mname, index(k, -m), n + k);
							flops += 1;
						} else {
							tprint("%s[%i] = FMA(%s[%i], c0[%i], %s[%i]);\n", lname, index(n, -m), mname, index(k, -m), n + k, lname, index(n, -m));
							flops += 2 - fmaops;
						}
					}
				}
			} else {
				bool pfirst = true;
				bool nfirst = true;
				const int maxk = std::min(P - n, P - 1);
				bool looped = true;
				for (int k = m; k <= maxk; k++) {
					looped = true;
					if (pfirst) {
						pfirst = false;
						tprint("Lr = %s[%i] * c0[%i];\n", mname, index(k, m), n + k);
						flops += 1;
					} else {
						tprint("Lr = FMA(%s[%i], c0[%i], Lr);\n", mname, index(k, m), n + k);
						flops += 2 - fmaops;
					}
					if (m != 0) {
						if (nfirst) {
							nfirst = false;
							tprint("Li = %s[%i] * c0[%i];\n", mname, index(k, -m), n + k);
							flops += 1;
						} else {
							tprint("Li = FMA(%s[%i], c0[%i], Li);\n", mname, index(k, -m), n + k);
							flops += 2 - fmaops;
						}
					}
				}
				if (!pfirst) {
					tprint("%s[%i] = -Lr;\n", lname, index(n, m));
					flops++;
				}
				if (!nfirst) {
					tprint("%s[%i] = -Li;\n", lname, index(n, -m));
					flops++;
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
		if (l_restrict && lmax > (P) - n) {
			lmax = (P) - n;
		}
		for (int m = -lmax; m <= lmax; m++) {
			if (noevenhi && P == n && P % 2 != abs(m) % 2) {
			} else {
				tprint("A[%i] = %s[%i];\n", m + P, name, index(n, m));
			}
		}
		std::vector < std::vector<std::pair<float, int>>>ops(2 * n + 1);
		int mmax = n;
		if (m_restrict && mmax > (P) - n) {
			mmax = (P + 1) - n;
		}
		int mmin = 0;
		int stride = 1;
		for (int m = 0; m <= mmax; m += stride) {
			for (int l = 0; l <= lmax; l++) {
				if (noevenhi && P == n && P % 2 != abs(l) % 2) {
					continue;
				}
				double r = l == 0 ? brot(n, m, 0) : brot(n, m, l) + nonepow<double>(l) * brot(n, m, -l);
				double i = l == 0 ? 0.0 : brot(n, m, l) - nonepow<double>(l) * brot(n, m, -l);
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
			if (ops[m].size() == 0 && !m_restrict && !l_restrict) {
				//			fprintf(stderr, " %i %i %i %i %i\n", m_restrict, l_restrict, noevenhi, n, m - n);
				//			tprint("%s[%i] = T(0);\n", name, index(n, m - n));
			}
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

int m2l_rot1(int P, int Q) {
	int flops = 0;
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void M2%s%s(expansion_type<T, %i>& L0, multipole_type<T, %i> M, T x, T y, T z) {\n", Q == 1 ? "P" : "L", nophi ? "_nopot" : "", Q, P);
	indent();
	const auto tpo = tprint_on;
	set_tprint(false);
	flops += greens_xz(P);
	set_tprint(tpo);
	tprint("expansion_type<T, %i> L;\n", Q);
	tprint("const T R2 = (x * x + y * y);\n");
	flops += 3;
	tprint("const T R = SQRT(R2);\n");
	flops += 4;
	tprint("const T Rzero = T(R<T(1e-37));\n");
	tprint("const T Rinv = T(1) / (R + Rzero);\n");
	flops += 5;
	tprint("const T r2inv = T(1) / (z * z + R2);\n");
	flops += 6;
	tprint("T cosphi = x * Rinv + Rzero;\n");
	flops++;
	tprint("T sinphi = -y * Rinv;\n");
	flops += 2;
	flops += z_rot(P - 1, "M", false, false);
	tprint("expansion_xz_type<T,%i> O;\n", P);
	tprint("greens_xz(O, R, z, r2inv);\n");
	const auto oindex = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	for (int n = nophi; n <= Q; n++) {
		for (int m = 0; m <= n; m++) {
			bool nfirst = true;
			bool pfirst = true;
			const int kmax = std::min(P - n, P - 1);
			for (int k = 0; k <= kmax; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					bool mreal = false;
					int gxsgn = 1;
					int mxsgn = 1;
					int mysgn = 1;
					char* gxstr = nullptr;
					char* mxstr = nullptr;
					char* mystr = nullptr;
					if (m + l > 0) {
						asprintf(&gxstr, "O[%i]", oindex(n + k, m + l));
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							asprintf(&gxstr, "O[%i]", oindex(n + k, -m - l));
						} else {
							asprintf(&gxstr, "O[%i]", oindex(n + k, -m - l));
							gxsgn = -1;
						}
					} else {
						asprintf(&gxstr, "O[%i]", oindex(n + k, 0));
					}
					if (l > 0) {
						asprintf(&mxstr, "M[%i]", index(k, l));
						asprintf(&mystr, "M[%i]", index(k, -l));
						mysgn = -1;
					} else if (l < 0) {
						if (l % 2 == 0) {
							asprintf(&mxstr, "M[%i]", index(k, -l));
							asprintf(&mystr, "M[%i]", index(k, l));
						} else {
							asprintf(&mxstr, "M[%i]", index(k, -l));
							asprintf(&mystr, "M[%i]", index(k, l));
							mxsgn = -1;
							mysgn = -1;
						}
					} else {
						mreal = true;
						asprintf(&mxstr, "M[%i]", index(k, 0));
					}
					const auto csgn = [](int i) {
						return i > 0 ? '+' : '-';
					};
					const auto csgn2 = [](int i) {
						return i > 0 ? '+' : ' ';
					};
					if (!mreal) {
						if (pfirst) {
							tprint("L[%i] = %c%s * %s;\n", index(n, m), csgn2(mxsgn * gxsgn), mxstr, gxstr);
							pfirst = false;
							flops += 1 + (mxsgn * gxsgn == 1);
						} else {
							tprint("L[%i] %c= %s * %s;\n", index(n, m), csgn(mxsgn * gxsgn), mxstr, gxstr);
							flops += 2;
						}
						if (m > 0) {
							if (nfirst) {
								tprint("L[%i] = %c%s * %s;\n", index(n, -m), csgn2(mysgn * gxsgn), mystr, gxstr);
								nfirst = false;
								flops += 1 + (mysgn * gxsgn == 1);
							} else {
								tprint("L[%i] %c= %s * %s;\n", index(n, -m), csgn(mysgn * gxsgn), mystr, gxstr);
								flops += 2;
							}
						}
					} else {
						if (pfirst) {
							tprint("L[%i] = %c%s * %s;\n", index(n, m), csgn2(mxsgn * gxsgn), mxstr, gxstr);
							pfirst = false;
							flops += 1 + (mxsgn * gxsgn == 1);
						} else {
							tprint("L[%i] %c= %s * %s;\n", index(n, m), csgn(mxsgn * gxsgn), mxstr, gxstr);
							flops += 2;
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
		}
	}
	tprint("sinphi = -sinphi;\n");
	flops++;
	flops += z_rot(Q, "L", false, false);
	flops++;
	tprint("for( int n = 0; n < %i; n++) {", (Q + 1) * (Q + 1));
	indent();
	tprint("L0[n] += L[n];\n");
	deindent();
	tprint("}\n");
	flops += (Q + 1) * (Q + 1);

	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int m2l_ewald(int P) {
	int flops = 0;
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void M2L_ewald%s(expansion_type<T, %i>& L, const multipole_type<T, %i>& M,T x, T y, T z) {\n", nophi ? "_nopot" : "", P, P);
	indent();
	tprint("expansion_type<T, %i> G;\n", P);
	tprint("greens_ewald(G, x, y, z);\n");
	tprint("M2L%s(L,M,G);\n", nophi ? "_nopot" : "");
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int m2lg(int P, int Q) {
	int flops = 0;
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void M2L%s(expansion_type<T, %i>& L, const multipole_type<T, %i>& M, const expansion_type<T, %i>& O) {\n", nophi ? "_nopot" : "", Q, P,
			P);
	indent();
	for (int n = nophi; n <= Q; n++) {
		for (int m = 0; m <= n; m++) {
			const int kmax = std::min(P - n, P - 1);
			for (int k = 0; k <= kmax; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					bool greal = false;
					bool mreal = false;
					int gxsgn = 1;
					int gysgn = 1;
					int mxsgn = 1;
					int mysgn = 1;
					char* gxstr = nullptr;
					char* gystr = nullptr;
					char* mxstr = nullptr;
					char* mystr = nullptr;
					if (m + l > 0) {
						asprintf(&gxstr, "O[%i]", index(n + k, m + l));
						asprintf(&gystr, "O[%i]", index(n + k, -m - l));
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							asprintf(&gxstr, "O[%i]", index(n + k, -m - l));
							asprintf(&gystr, "O[%i]", index(n + k, m + l));
							gysgn = -1;
						} else {
							asprintf(&gxstr, "O[%i]", index(n + k, -m - l));
							asprintf(&gystr, "O[%i]", index(n + k, m + l));
							gxsgn = -1;
						}
					} else {
						greal = true;
						asprintf(&gxstr, "O[%i]", index(n + k, 0));
					}
					if (l > 0) {
						asprintf(&mxstr, "M[%i]", index(k, l));
						asprintf(&mystr, "M[%i]", index(k, -l));
						mysgn = -1;
					} else if (l < 0) {
						if (l % 2 == 0) {
							asprintf(&mxstr, "M[%i]", index(k, -l));
							asprintf(&mystr, "M[%i]", index(k, l));
						} else {
							asprintf(&mxstr, "M[%i]", index(k, -l));
							asprintf(&mystr, "M[%i]", index(k, l));
							mxsgn = -1;
							mysgn = -1;
						}
					} else {
						mreal = true;
						asprintf(&mxstr, "M[%i]", index(k, 0));
					}
					const auto csgn = [](int i) {
						return i > 0 ? '+' : '-';
					};
					if (!mreal) {
						if (!greal) {
							tprint("L[%i] %c= %s * %s;\n", index(n, m), csgn(mxsgn * gxsgn), mxstr, gxstr);
							tprint("L[%i] %c= %s * %s;\n", index(n, m), csgn(-mysgn * gysgn), mystr, gystr);
							flops += 4;
							if (m > 0) {
								tprint("L[%i] %c= %s * %s;\n", index(n, -m), csgn(mysgn * gxsgn), mystr, gxstr);
								tprint("L[%i] %c= %s * %s;\n", index(n, -m), csgn(mxsgn * gysgn), mxstr, gystr);
								flops += 4;
							}
						} else {
							tprint("L[%i] %c= %s * %s;\n", index(n, m), csgn(mxsgn * gxsgn), mxstr, gxstr);
							flops += 2;
							if (m > 0) {
								tprint("L[%i] %c= %s * %s;\n", index(n, -m), csgn(mysgn * gxsgn), mystr, gxstr);
								flops += 2;
							}
						}
					} else {
						if (!greal) {
							tprint("L[%i] %c= %s * %s;\n", index(n, m), csgn(mxsgn * gxsgn), mxstr, gxstr);
							flops += 2;
							if (m > 0) {
								tprint("L[%i] %c= %s * %s;\n", index(n, -m), csgn(mxsgn * gysgn), mxstr, gystr);
								flops += 2;
							}
						} else {
							tprint("L[%i] %c= %s * %s;\n", index(n, m), csgn(mxsgn * gxsgn), mxstr, gxstr);
							flops += 2;
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (gystr) {
						free(gystr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
		}
	}
	if (!nophi && P > 2) {
		tprint("L[%i] -= T(0.5) * O[%i] * M[%i];\n", index(0, 0), (P + 1) * (P + 1), P * P);
		flops += 3;
	}
	if (P > 1) {
		tprint("L[%i] -= T(2) * O[%i] * M[%i];\n", index(1, -1), (P + 1) * (P + 1), index(1, -1));
		tprint("L[%i] -= O[%i] * M[%i];\n", index(1, +0), (P + 1) * (P + 1), index(1, +0));
		tprint("L[%i] -= T(2) * O[%i] * M[%i];\n", index(1, +1), (P + 1) * (P + 1), index(1, +1));
		tprint("L[%i] -= T(0.5) * O[%i] * M[%i];\n", (P + 1) * (P + 1), (P + 1) * (P + 1), index(0, 0));
	}
	flops += 11;
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int m2l_norot(int P, int Q) {
	int flops = 0;
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void M2%s%s(expansion_type<T, %i>& L, const multipole_type<T, %i>& M, T x, T y, T z) {\n", Q == 1 ? "P" : "L", nophi ? "_nopot" : "", Q,
			P);
	indent();
	const auto c = tprint_on;
	set_tprint(false);
	flops += greens(P);
	set_tprint(c);
	tprint("expansion_type<T,%i> O;\n", P);
	tprint("greens(O, x, y, z);\n");
	for (int n = nophi; n <= Q; n++) {
		for (int m = 0; m <= n; m++) {
			const int kmax = std::min(P - n, P - 1);
			for (int k = 0; k <= kmax; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					bool greal = false;
					bool mreal = false;
					int gxsgn = 1;
					int gysgn = 1;
					int mxsgn = 1;
					int mysgn = 1;
					char* gxstr = nullptr;
					char* gystr = nullptr;
					char* mxstr = nullptr;
					char* mystr = nullptr;
					if (m + l > 0) {
						asprintf(&gxstr, "O[%i]", index(n + k, m + l));
						asprintf(&gystr, "O[%i]", index(n + k, -m - l));
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							asprintf(&gxstr, "O[%i]", index(n + k, -m - l));
							asprintf(&gystr, "O[%i]", index(n + k, m + l));
							gysgn = -1;
						} else {
							asprintf(&gxstr, "O[%i]", index(n + k, -m - l));
							asprintf(&gystr, "O[%i]", index(n + k, m + l));
							gxsgn = -1;
						}
					} else {
						greal = true;
						asprintf(&gxstr, "O[%i]", index(n + k, 0));
					}
					if (l > 0) {
						asprintf(&mxstr, "M[%i]", index(k, l));
						asprintf(&mystr, "M[%i]", index(k, -l));
						mysgn = -1;
					} else if (l < 0) {
						if (l % 2 == 0) {
							asprintf(&mxstr, "M[%i]", index(k, -l));
							asprintf(&mystr, "M[%i]", index(k, l));
						} else {
							asprintf(&mxstr, "M[%i]", index(k, -l));
							asprintf(&mystr, "M[%i]", index(k, l));
							mxsgn = -1;
							mysgn = -1;
						}
					} else {
						mreal = true;
						asprintf(&mxstr, "M[%i]", index(k, 0));
					}
					const auto csgn = [](int i) {
						return i > 0 ? '+' : '-';
					};
					if (!mreal) {
						if (!greal) {
							tprint("L[%i] %c= %s * %s;\n", index(n, m), csgn(mxsgn * gxsgn), mxstr, gxstr);
							tprint("L[%i] %c= %s * %s;\n", index(n, m), csgn(-mysgn * gysgn), mystr, gystr);
							flops += 4;
							if (m > 0) {
								tprint("L[%i] %c= %s * %s;\n", index(n, -m), csgn(mysgn * gxsgn), mystr, gxstr);
								tprint("L[%i] %c= %s * %s;\n", index(n, -m), csgn(mxsgn * gysgn), mxstr, gystr);
								flops += 4;
							}
						} else {
							tprint("L[%i] %c= %s * %s;\n", index(n, m), csgn(mxsgn * gxsgn), mxstr, gxstr);
							flops += 2;
							if (m > 0) {
								tprint("L[%i] %c= %s * %s;\n", index(n, -m), csgn(mysgn * gxsgn), mystr, gxstr);
								flops += 2;
							}
						}
					} else {
						if (!greal) {
							tprint("L[%i] %c= %s * %s;\n", index(n, m), csgn(mxsgn * gxsgn), mxstr, gxstr);
							flops += 2;
							if (m > 0) {
								tprint("L[%i] %c= %s * %s;\n", index(n, -m), csgn(mxsgn * gysgn), mxstr, gystr);
								flops += 2;
							}
						} else {
							tprint("L[%i] %c= %s * %s;\n", index(n, m), csgn(mxsgn * gxsgn), mxstr, gxstr);
							flops += 2;
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (gystr) {
						free(gystr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

std::vector<complex<double> > spherical_singular_harmonic(int P, double x, double y, double z) {
	const double r2 = x * x + y * y + z * z;
	const double r2inv = double(1) / r2;
	complex<double> R = complex<double>(x, y);
	std::vector<complex<double>> O((P + 1) * (P + 1));
	O[cindex(0, 0)] = complex<double>(sqrt(r2inv), double(0));
	R *= r2inv;
	z *= r2inv;
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
			O[cindex(m, m)] = O[cindex(m - 1, m - 1)] * R * double(2 * m - 1);
		}
		if (m + 1 <= P) {
			O[cindex(m + 1, m)] = double(2 * m + 1) * z * O[cindex(m, m)];
		}
		for (int n = m + 2; n <= P; n++) {
			O[cindex(n, m)] = (double(2 * n - 1) * z * O[cindex(n - 1, m)] - double((n - 1) * (n - 1) - m * m) * r2inv * O[cindex(n - 2, m)]);
		}
	}
	return O;
}

bool close2zero(double a) {
	return abs(a) < 1e-10;
}

int ewald_greens(int P) {
	int flops = 0;
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void greens_ewald%s(expansion_type<T, %i>& G, T x0, T y0, T z0) {\n", nophi ? "_nopot" : "", P, P);
	indent();
	const auto c = tprint_on;

	//set_tprint(false);
	//flops += greens(P);
	//set_tprint(c);
	tprint("for (int i = 0; i < %i; i++) {\n", (P + 1) * (P + 1));
	indent();
	tprint("G[i] = T(0);\n");
	deindent();
	tprint("}\n");

	constexpr double alpha = 2.0;
	tprint("expansion_type<T, %i> Gr;\n", P);
	int cnt = 0;
	for (int ix = -3; ix <= 3; ix++) {
		for (int iy = -3; iy <= 3; iy++) {
			for (int iz = -3; iz <= 3; iz++) {
				if (ix * ix + iy * iy + iz * iz > 3.1 * 3.1) {
					continue;
				}
				cnt++;
			}
		}
	}
	flops *= cnt;
	set_tprint(false);
	flops += cnt * greens(P);
	set_tprint(c);

	bool first = true;
	int rcnt = cnt - 1;
	tprint("const T r2 = x0 * x0 + y0 * y0 + z0 * z0;\n");
	flops += 5;
	tprint("const T r = SQRT(r2);\n");
	flops += 4;
	tprint("greens(Gr, x0, y0, z0);\n");
	tprint("T gamma1 = T(%.16e) * ERFC(T(%.16e) * r);\n", sqrt(M_PI), alpha);
	flops += 10;
	tprint("const T xfac = T(%.16e) * r2;\n", alpha * alpha);
	flops += 1;
	tprint("const T exp0 = EXP(-xfac);\n");
	flops += 9;
	tprint("T xpow = T(%.16e) * r;\n", alpha);
	flops++;
	if (nophi) {
		tprint("T gamma0inv = T(%.16e);\n", 1.0 / sqrt(M_PI));
	} else {
		tprint("T gamma0inv = T(%.16e);\n", -2.0 / sqrt(M_PI));
	}
	tprint("T gamma;\n");
	tprint("const T sw = r2 > T(0);\n");
	flops += 1;
	if (nophi) {
		tprint("gamma1 = T(%.16e) * gamma1 + xpow * exp0;\n", 0.5);
		flops += 3;
		if (P != 1) {
			tprint("xpow *= xfac;\n");
			flops += rcnt;
		}
	}
	for (int l = nophi; l <= P; l++) {
		tprint("gamma = gamma1 * gamma0inv;\n");
		flops += rcnt;
		for (int m = -l; m <= l; m++) {
			tprint("G[%i] -= sw*(gamma - T(%.1e)) * Gr[%i];\n", index(l, m), nonepow<double>(l), index(l, m));
			flops += 4;
		}
		if (l == 0) {
			tprint("G[%i] += (T(1) - sw)*T(%.16e);\n", index(0, 0), (2) * alpha / sqrt(M_PI));
			flops += 3;
		}
		tprint("gamma0inv *= T(1) / T(%.16e);\n", -(l + 0.5));
		if (l != P) {
			tprint("gamma1 = T(%.16e) * gamma1 + xpow * exp0;\n", l + 0.5);
			flops += 3;
			if (l != P - 1) {
				tprint("xpow *= xfac;\n");
				flops += rcnt;
			}
		}
	}

	tprint("for (int ix = -3; ix <= 3; ix++) {\n");
	indent();
	tprint("const T& x = x0 - T(ix);\n");
	flops += cnt;
	tprint("for (int iy = -3; iy <= 3; iy++) {\n");
	indent();
	tprint("const T& y = y0 - T(iy);\n");
	flops += cnt;
	tprint("for (int iz = -3; iz <= 3; iz++) {\n");
	indent();
	tprint("const T& z = z0 - T(iz);\n");
	flops += cnt;
	tprint("const T i2 = (ix * ix + iy * iy + iz * iz);\n");
	tprint("if (i2 > 3.46 * 3.46 && i2 != 0.0) {\n");
	indent();
	tprint("continue;\n");
	deindent();
	tprint("}\n");
	tprint("const T r2 = x * x + y * y + z * z;\n");
	flops += rcnt * 5;
	tprint("const T r = SQRT(r2);\n");
	flops += rcnt * 4;
	tprint("greens(Gr, x, y, z);\n");
	tprint("T gamma1 = T(%.16e) * ERFC(T(%.16e) * r);\n", sqrt(M_PI), alpha);
	flops += rcnt * 10;
	tprint("const T xfac = T(%.16e) * r2;\n", alpha * alpha);
	flops += rcnt * 1;
	tprint("const T exp0 = EXP(-xfac);\n");
	flops += rcnt * 9;
	tprint("T xpow = T(%.16e) * r;\n", alpha);
	flops++;
	if (nophi) {
		tprint("T gamma0inv = T(%.16e);\n", 1.0 / sqrt(M_PI));
	} else {
		tprint("T gamma0inv = T(%.16e);\n", -2.0 / sqrt(M_PI));
	}
	tprint("T gamma;\n");
	if (nophi) {
		tprint("gamma1 = T(%.16e) * gamma1 + xpow * exp0;\n", 0.5);
		flops += rcnt * 3;
		if (P != 1) {
			tprint("xpow *= xfac;\n");
			flops += rcnt;
		}
	}
	for (int l = nophi; l <= P; l++) {
		tprint("gamma = gamma1 * gamma0inv;\n");
		flops += rcnt;
		for (int m = -l; m <= l; m++) {
			tprint("G[%i] -= gamma * Gr[%i];\n", index(l, m), index(l, m));
			flops += rcnt * 2;
		}
		tprint("gamma0inv *= T(1) / T(%.16e);\n", -(l + 0.5));
		if (l != P) {
			tprint("gamma1 = T(%.16e) * gamma1 + xpow * exp0;\n", l + 0.5);
			flops += rcnt * 3;
			if (l != P - 1) {
				tprint("xpow *= xfac;\n");
				flops += rcnt;
			}
		}
	}
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");
	tprint("T cosphi, sinphi, hdotx, phi;\n");

	for (int hx = -2; hx <= 2; hx++) {
		for (int hy = -2; hy <= 2; hy++) {
			for (int hz = -2; hz <= 2; hz++) {
				const int h2 = hx * hx + hy * hy + hz * hz;
				if (h2 <= 8 && h2 > 0) {
					const double h = sqrt(h2);
					auto G0 = spherical_singular_harmonic(P, (double) hx, (double) hy, (double) hz);
					bool init = false;
					if (hx) {
						if (hx == 1) {
							tprint("hdotx %s= x0;\n", init ? "+" : "");
							flops += init;
							init = true;
						} else if (hx == -1) {
							if (init) {
								tprint("hdotx -= x0;\n");
								flops++;
							} else {
								tprint("hdotx = -x0;\n");
								flops++;
							}
							init = true;
						} else {
							tprint("hdotx %s= T(%i) * x0;\n", init ? "+" : "", hx);
							flops += 1 + init;
							init = true;
						}
					}
					if (hy) {
						if (hy == 1) {
							tprint("hdotx %s= y0;\n", init ? "+" : "");
							flops += init;
							init = true;
						} else if (hy == -1) {
							if (init) {
								tprint("hdotx -= y0;\n");
								flops++;
							} else {
								flops++;
								tprint("hdotx = -y0;\n");
							}
							init = true;
						} else {
							flops++;
							tprint("hdotx %s= T(%i) * y0;\n", init ? "+" : "", hy);
							flops += 1 + init;
							init = true;
						}
					}
					if (hz) {
						if (hz == 1) {
							tprint("hdotx %s= z0;\n", init ? "+" : "");
							flops += init;
							init = true;
						} else if (hz == -1) {
							if (init) {
								flops++;
								tprint("hdotx -= z0;\n");
							} else {
								flops++;
								tprint("hdotx = -z0;\n");
							}
							init = true;
						} else {
							flops++;
							tprint("hdotx %s= T(%i) * z0;\n", init ? "+" : "", hz);
							flops += 1 + init;
							init = true;
						}
					}
					tprint("phi = T(%.16e) * hdotx;\n", 2.0 * M_PI);
					flops++;
					tprint("SINCOS(phi, &sinphi, &cosphi);\n");
					flops += 8;
					double gamma0inv = 1.0f / sqrt(M_PI);
					double hpow = 1.f / h;
					double pipow = 1.f / sqrt(M_PI);
					if (nophi) {
						gamma0inv /= 0 + 0.5f;
						hpow *= h * h;
						pipow *= M_PI;
					}
					for (int l = nophi; l <= P; l++) {
						for (int m = 0; m <= l; m++) {
							double c0 = gamma0inv * hpow * pipow * exp(-h * h * double(M_PI * M_PI) / (alpha * alpha));
							std::string ax = "cosphi";
							std::string ay = "sinphi";
							int xsgn = 1;
							int ysgn = 1;
							if (l % 4 == 3) {
								std::swap(ax, ay);
								ysgn = -1;
							} else if (l % 4 == 2) {
								ysgn = xsgn = -1;
							} else if (l % 4 == 1) {
								std::swap(ax, ay);
								xsgn = -1;
							}
							if (!close2zero(G0[cindex(l, m)].real())) {
								tprint("G[%i] += T(%.16e) * %s;\n", index(l, m), -xsgn * c0 * G0[cindex(l, m)].real(), ax.c_str());
								flops += 2;
								if (m != 0) {
									tprint("G[%i] += T(%.16e) * %s;\n", index(l, -m), -ysgn * c0 * G0[cindex(l, m)].real(), ay.c_str());
									flops += 2;
								}
							}
							if (!close2zero(G0[cindex(l, m)].imag())) {
								flops += 2;
								tprint("G[%i] += T(%.16e) * %s;\n", index(l, m), ysgn * c0 * G0[cindex(l, m)].imag(), ay.c_str());
								if (m != 0) {
									flops += 2;
									tprint("G[%i] += T(%.16e) * %s;\n", index(l, -m), -xsgn * c0 * G0[cindex(l, m)].imag(), ax.c_str());
								}
							}
						}
						gamma0inv /= l + 0.5f;
						hpow *= h * h;
						pipow *= M_PI;
					}
				}
			}
		}
	}
	tprint("G[%i] = T(%.16e);\n", (P + 1) * (P + 1), (4.0 * M_PI / 3.0));
	if (!nophi) {
		tprint("G[%i] += T(%.16e);\n", index(0, 0), M_PI / (alpha * alpha));
		flops++;
	}

	deindent();
	tprint("/* flops = %i */\n", flops);
	tprint("}");
	tprint("\n");
	return flops;
}

int m2l_rot2(int P, int Q) {
	int flops = 0;
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void M2%s%s(expansion_type<T, %i>& L0, multipole_type<T, %i> M, T x, T y, T z) {\n", Q == 1 ? "P" : "L", nophi ? "_norot" : "", Q, P);
	indent();
	tprint("expansion_type<T, %i> L;\n", Q);

	/*	tprint("for( int n = 0; n < %i; n++) {", Q == 1 ? 4 : (Q + 1) * (Q + 1) + 1);
	 indent();
	 tprint("L[n] = T(0);\n");
	 deindent();
	 tprint("}\n");*/

	tprint("const T R2 = (x * x + y * y);\n");
	flops += 3;
	tprint("const T R = SQRT(R2);\n");
	flops += 4;
	tprint("const T Rzero = T(R<T(1e-37));\n");
	tprint("const T Rinv = T(1) / (R + Rzero);\n");
	flops += 5;
	tprint("const T r = SQRT(z * z + R2);\n");
	tprint("const T rzero = T(r<T(1e-37));\n");
	flops += 6;
	tprint("const T rinv = T(1) / (r + rzero);\n");
	flops += 5;
	tprint("T cosphi0;\n");
	tprint("T cosphi;\n");
	tprint("T sinphi0;\n");
	tprint("T sinphi;\n");

	tprint("cosphi = y * Rinv;\n");
	flops++;
	tprint("sinphi = x * Rinv + Rzero;\n");
	flops += 2;
	tprint("const auto multi_rot = [&M,&cosphi,&sinphi]()\n");
	flops += 2 * z_rot(P - 1, "M", false, false);
	tprint(";\n");
	tprint("multi_rot();\n");

	flops += xz_swap(P - 1, "M", false, false, false, false);

	tprint("cosphi0 = cosphi;\n");
	tprint("sinphi0 = sinphi;\n");
	tprint("cosphi = z * rinv + rzero;\n");
	flops += 2;
	tprint("sinphi = -R * rinv;\n");
	flops += 2;
	tprint("multi_rot();\n");
	flops += xz_swap(P - 1, "M", false, true, false, false);
	flops += m2l(P, Q, "M", "L");
	flops += xz_swap(Q, "L", true, false, true, false);

	tprint("sinphi = -sinphi;\n");
	flops += 1;
	flops += z_rot(Q, "L", true, false);
	//	flops += z_rot(P, "L", true);
	//	flops += xz_swap(P, "L", true, false, false, true);
	flops += xz_swap(Q, "L", true, false, false, true);
	tprint("cosphi = cosphi0;\n");
	tprint("sinphi = -sinphi0;\n");
	flops += 1;
	flops += z_rot(Q, "L", false, true);
	tprint("for( int n = 0; n < %i; n++) {", (Q + 1) * (Q + 1));
	indent();
	tprint("L0[n] += L[n];\n");
	deindent();
	tprint("}\n");
	flops += (Q + 1) * (Q + 1);
	tprint("\n");
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int regular_harmonic(int P) {
	int flops = 0;
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void regular_harmonic(expansion_type<T, %i>& Y, T x, T y, T z) {\n", P);
	indent();
	tprint("const T r2 = x * x + y * y + z * z;\n");
	flops += 5;
	tprint("T ax;\n");
	tprint("T ay;\n");
	tprint("Y[0] = T(1);\n");
	tprint("Y[%i] = r2;\n", (P + 1) * (P + 1));
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
//	Y[index(m, m)] = Y[index(m - 1, m - 1)] * R / T(2 * m);
			if (m - 1 > 0) {
				tprint("ax = Y[%i] * T(%.16e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("ay = Y[%i] * T(%.16e);\n", index(m - 1, -(m - 1)), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax - y * ay;\n", index(m, m));
				tprint("Y[%i] = y * ax + x * ay;\n", index(m, -m));
				flops += 6;
			} else {
				tprint("ax = Y[%i] * T(%.16e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax;\n", index(m, m));
				tprint("Y[%i] = y * ax;\n", index(m, -m));
				flops += 3;
			}
		}
		if (m + 1 <= P) {
//			Y[index(m + 1, m)] = z * Y[index(m, m)];
			if (m == 0) {
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
				flops += 1;
			} else {
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, -m), index(m, -m));
				flops += 2;
			}
		}
		for (int n = m + 2; n <= P; n++) {
			const double inv = double(1) / (double(n * n) - double(m * m));
//			Y[index(n, m)] = inv * (T(2 * n - 1) * z * Y[index(n - 1, m)] - r2 * Y[index(n - 2, m)]);
			tprint("ax =  T(%.16e) * z;\n", inv * double(2 * n - 1));
			tprint("ay =  T(%.16e) * r2;\n", -(double) inv);
			tprint("Y[%i] = ax * Y[%i] + ay * Y[%i];\n", index(n, m), index(n - 1, m), -(double) inv, index(n - 2, m));
			flops += 5;
			if (m != 0) {
				tprint("Y[%i] = ax * Y[%i] + ay * Y[%i];\n", index(n, -m), index(n - 1, -m), -(double) inv, index(n - 2, -m));
				flops += 3;
			}
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int regular_harmonic_xz(int P) {
	int flops = 0;
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void regular_harmonic_xz(expansion_xz_type<T, %i>& Y, T x, T z) {\n", P);
	indent();
	tprint("const T r2 = x * x + z * z;\n");
	flops += 5;
	tprint("T ax;\n");
	tprint("T ay;\n");
	tprint("Y[0] = T(1);\n");
	tprint("Y[%i] = r2;\n", (P + 2) * (P + 1) / 2);
	const auto index = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
//	Y[index(m, m)] = Y[index(m - 1, m - 1)] * R / T(2 * m);
			if (m - 1 > 0) {
				tprint("ax = Y[%i] * T(%.16e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax;\n", index(m, m));
				flops += 2;
			} else {
				tprint("ax = Y[%i] * T(%.16e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax;\n", index(m, m));
				flops += 2;
			}
		}
		if (m + 1 <= P) {
//			Y[index(m + 1, m)] = z * Y[index(m, m)];
			if (m == 0) {
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
				flops += 1;
			} else {
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
				flops += 1;
			}
		}
		for (int n = m + 2; n <= P; n++) {
			const double inv = double(1) / (double(n * n) - double(m * m));
//			Y[index(n, m)] = inv * (T(2 * n - 1) * z * Y[index(n - 1, m)] - r2 * Y[index(n - 2, m)]);
			tprint("ax =  T(%.16e) * z;\n", inv * double(2 * n - 1));
			tprint("ay =  T(%.16e) * r2;\n", -(double) inv);
			tprint("Y[%i] = ax * Y[%i] + ay * Y[%i];\n", index(n, m), index(n - 1, m), -(double) inv, index(n - 2, m));
			flops += 5;
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int M2M_norot(int P) {
	int flops = 0;
	const auto c = tprint_on;
	set_tprint(false);
	flops += regular_harmonic(P);
	set_tprint(c);
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void M2M%s(multipole_type<T, %i>& M, T x, T y, T z) {\n", nophi ? "_norot" : "", P + 1);
	indent();
//const auto Y = spherical_regular_harmonic<T, P>(-x, -y, -z);
	tprint("expansion_type<T, %i> Y;\n", P);
	tprint("regular_harmonic(Y, -x, -y, -z);\n");
	flops += 3;
	if (P > 1 && !nophi) {
		tprint("M[%i] -= T(4) * x * M[%i];\n", (P + 1) * (P + 1), index(1, 1));
		tprint("M[%i] -= T(4) * y * M[%i];\n", (P + 1) * (P + 1), index(1, -1));
		tprint("M[%i] -= T(2) * z * M[%i];\n", (P + 1) * (P + 1), index(1, 0));
		tprint("M[%i] += x * x * M[%i];\n", (P + 1) * (P + 1), index(0, 0));
		tprint("M[%i] += y * y * M[%i];\n", (P + 1) * (P + 1), index(0, 0));
		tprint("M[%i] += z * z * M[%i];\n", (P + 1) * (P + 1), index(0, 0));
		flops += 18;
	}
	for (int n = P; n >= 0; n--) {
		for (int m = 0; m <= n; m++) {
			for (int k = 1; k <= n; k++) {
				const int lmin = std::max(-k, m - n + k);
				const int lmax = std::min(k, m + n - k);
				for (int l = -k; l <= k; l++) {
					char* mxstr = nullptr;
					char* mystr = nullptr;
					char* gxstr = nullptr;
					char* gystr = nullptr;
					int mxsgn = 1;
					int mysgn = 1;
					int gxsgn = 1;
					int gysgn = 1;
					if (abs(m - l) > n - k) {
						continue;
					}
					if (-abs(m - l) < k - n) {
						continue;
					}
					if (m - l > 0) {
						asprintf(&mxstr, "M[%i]", index(n - k, abs(m - l)));
						asprintf(&mystr, "M[%i]", index(n - k, -abs(m - l)));
					} else if (m - l < 0) {
						if (abs(m - l) % 2 == 0) {
							asprintf(&mxstr, "M[%i]", index(n - k, abs(m - l)));
							asprintf(&mystr, "M[%i]", index(n - k, -abs(m - l)));
							mysgn = -1;
						} else {
							asprintf(&mxstr, "M[%i]", index(n - k, abs(m - l)));
							asprintf(&mystr, "M[%i]", index(n - k, -abs(m - l)));
							mxsgn = -1;
						}
					} else {
						asprintf(&mxstr, "M[%i]", index(n - k, 0));
					}
					if (l > 0) {
						asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
						asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
					} else if (l < 0) {
						if (abs(l) % 2 == 0) {
							asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
							asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
							gysgn = -1;
						} else {
							asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
							asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
							gxsgn = -1;
						}
					} else {
						asprintf(&gxstr, "Y[%i]", index(k, 0));
					}
					const auto csgn = [](int i) {
						return i > 0 ? '+' : '-';
					};
					tprint("M[%i] %c= %s * %s;\n", index(n, m), csgn(mxsgn * gxsgn), mxstr, gxstr);
					flops += 2;
					if (gystr && mystr) {
						tprint("M[%i] %c= %s * %s;\n", index(n, m), csgn(-mysgn * gysgn), mystr, gystr);
						flops += 2;
					}
					if (m > 0) {
						if (gystr) {
							tprint("M[%i] %c= %s * %s;\n", index(n, -m), csgn(mxsgn * gysgn), mxstr, gystr);
							flops += 2;
						}
						if (mystr) {
							tprint("M[%i] %c= %s * %s;\n", index(n, -m), csgn(mysgn * gxsgn), mystr, gxstr);
							flops += 2;
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (gystr) {
						free(gystr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int M2M_rot1(int P) {
	int flops = 0;
	const auto c = tprint_on;
	set_tprint(false);
	flops += regular_harmonic_xz(P);
	set_tprint(c);

	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void M2M%s(multipole_type<T, %i>& M, T x, T y, T z) {\n", nophi ? "_norot" : "", P + 1);
	indent();

	tprint("const T R2 = (x * x + y * y);\n");
	flops += 3;
	tprint("const T R = SQRT(R2);\n");
	flops += 4;
	tprint("const T Rzero = (R<T(1e-37));\n");
	flops++;
	tprint("const T Rinv = T(1) / (R + Rzero);\n");
	flops += 5;
	tprint("const T r2inv = T(1) / (z * z + R2);\n");
	flops += 6;
	tprint("T cosphi = x * Rinv + Rzero;\n");
	flops += 2;
	tprint("T sinphi = -y * Rinv;\n");
	flops += 2;
	flops += z_rot(P, "M", false, false);
	if (P > 1 && !nophi) {
		tprint("M[%i] -= T(4)*R * M[%i];\n", (P + 1) * (P + 1), index(1, 1));
		tprint("M[%i] -= T(2)*z * M[%i];\n", (P + 1) * (P + 1), index(1, 0));
		tprint("M[%i] += R * R * M[%i];\n", (P + 1) * (P + 1), index(0, 0));
		tprint("M[%i] += z * z * M[%i];\n", (P + 1) * (P + 1), index(0, 0));
		flops += 12;
	}
	const auto yindex = [](int l, int m) {
		return l*(l+1)/2+m;
	};

	tprint("expansion_xz_type<T, %i> Y;\n", P);
	tprint("regular_harmonic_xz(Y, -R, -z);\n");
	flops += 2;
	for (int n = P; n >= 0; n--) {
		for (int m = 0; m <= n; m++) {
			for (int k = 1; k <= n; k++) {
				const int lmin = std::max(-k, m - n + k);
				const int lmax = std::min(k, m + n - k);
				for (int l = -k; l <= k; l++) {
					char* mxstr = nullptr;
					char* mystr = nullptr;
					char* gxstr = nullptr;
					int mxsgn = 1;
					int mysgn = 1;
					int gxsgn = 1;
					if (abs(m - l) > n - k) {
						continue;
					}
					if (-abs(m - l) < k - n) {
						continue;
					}
					if (m - l > 0) {
						asprintf(&mxstr, "M[%i]", index(n - k, abs(m - l)));
						asprintf(&mystr, "M[%i]", index(n - k, -abs(m - l)));
					} else if (m - l < 0) {
						if (abs(m - l) % 2 == 0) {
							asprintf(&mxstr, "M[%i]", index(n - k, abs(m - l)));
							asprintf(&mystr, "M[%i]", index(n - k, -abs(m - l)));
							mysgn = -1;
						} else {
							asprintf(&mxstr, "M[%i]", index(n - k, abs(m - l)));
							asprintf(&mystr, "M[%i]", index(n - k, -abs(m - l)));
							mxsgn = -1;
						}
					} else {
						asprintf(&mxstr, "M[%i]", index(n - k, 0));
					}
					asprintf(&gxstr, "Y[%i]", yindex(k, abs(l)));
					if (l < 0 && abs(l) % 2 != 0) {
						gxsgn = -1;
					}
					//	M[index(n, m)] += Y(k, l) * M(n - k, m - l);
					const auto csgn = [](int i) {
						return i > 0 ? '+' : '-';
					};
					tprint("M[%i] %c= %s * %s;\n", index(n, m), csgn(mxsgn * gxsgn), mxstr, gxstr);
					flops += 2;
					if (m > 0) {
						if (mystr) {
							tprint("M[%i] %c= %s * %s;\n", index(n, -m), csgn(mysgn * gxsgn), mystr, gxstr);
							flops += 2;
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
		}

	}
	tprint("sinphi = -sinphi;\n");
	flops++;
	flops += z_rot(P, "M", false, false);
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int M2M_rot2(int P) {
	int flops = 0;
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void M2M%s(multipole_type<T, %i>& M, T x, T y, T z) {\n", nophi ? "_nopot" : "", P + 1);
	indent();
	//const auto Y = spherical_regular_harmonic<T, P>(-x, -y, -z);

	tprint("const T R2 = (x * x + y * y);\n");
	flops += 3;
	tprint("const T R = SQRT(R2);\n");
	flops += 4;
	tprint("const T r = SQRT(R2 + z * z);\n");
	flops += 6;
	tprint("const T Rzero = T(R<T(1e-37));");
	tprint("const T rzero = T(r<T(1e-37));");
	flops += 2;
	tprint("const T Rinv = T(1) / (R + Rzero);\n");
	flops += 5;
	tprint("const T rinv = T(1) / (r + rzero);\n");
	flops += 5;
	tprint("T cosphi = y * Rinv;\n");
	flops++;
	tprint("T sinphi = x * Rinv + Rzero;\n");
	flops += 2;
	flops += z_rot(P, "M", false, false);
	flops += xz_swap(P, "M", false, false, false, false);
	tprint("T cosphi0 = cosphi;\n");
	tprint("T sinphi0 = sinphi;\n");
	tprint("cosphi = z * rinv + rzero;\n");
	flops++;
	tprint("sinphi = -R * rinv;\n");
	flops += z_rot(P, "M", false, false);
	flops += xz_swap(P, "M", false, false, false, false);
	if (P > 1 && !nophi) {
		tprint("M[%i] -= T(2) * r * M[%i];\n", (P + 1) * (P + 1), index(1, 0));
		tprint("M[%i] += r * r * M[%i];\n", (P + 1) * (P + 1), index(0, 0));
		flops += 6;
	}
	tprint("T c0[%i];\n", P + 1);
	tprint("c0[0] = T(1);\n");
	for (int n = 1; n <= P; n++) {
		tprint("c0[%i] = -r * c0[%i];\n", n, n - 1);
		flops += 2;
	}
	for (int n = 2; n <= P; n++) {
		tprint("c0[%i] *= T(%.16e);\n", n, 1.0 / factorial(n));
		flops += 1;
	}
	for (int n = P; n >= 0; n--) {
		for (int m = 0; m <= n; m++) {
			for (int k = 1; k <= n; k++) {
				if (abs(m) > n - k) {
					continue;
				}
				if (-abs(m) < k - n) {
					continue;
				}
				tprint("M[%i] += M[%i] * c0[%i];\n", index(n, m), index(n - k, m), k);
				flops += 2;
				if (m > 0) {
					tprint("M[%i] += M[%i] * c0[%i];\n", index(n, -m), index(n - k, -m), k);
					flops += 2;
				}
			}

		}
	}
	flops += xz_swap(P, "M", false, false, false, false);
	tprint("sinphi = -sinphi;\n");
	flops += z_rot(P, "M", false, false);
	flops += xz_swap(P, "M", false, false, false, false);
	tprint("cosphi = cosphi0;\n");
	tprint("sinphi = -sinphi0;\n");
	flops += 1;
	flops += z_rot(P, "M", false, false);
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int L2L_norot(int P) {
	int flops = 0;
	const auto c = tprint_on;
	set_tprint(false);
	flops += regular_harmonic(P);
	set_tprint(c);
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void L2L%s(expansion_type<T, %i>& L, T x, T y, T z) {\n", nophi ? "_norot" : "", P);
	indent();
	//const auto Y = spherical_regular_harmonic<T, P>(-x, -y, -z);
	tprint("expansion_type<T, %i> Y;\n", P);
	tprint("regular_harmonic(Y, -x, -y, -z);\n");
	flops += 3;
	for (int n = nophi; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			for (int k = 1; k <= P - n; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					char* mxstr = nullptr;
					char* mystr = nullptr;
					char* gxstr = nullptr;
					char* gystr = nullptr;
					int mxsgn = 1;
					int mysgn = 1;
					int gxsgn = 1;
					int gysgn = 1;
					if (m + l > 0) {
						asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
						asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mysgn = -1;
						} else {
							asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mxsgn = -1;
						}
					} else {
						asprintf(&mxstr, "L[%i]", index(n + k, 0));
					}
					if (l > 0) {
						asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
						asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
						gysgn = -1;
					} else if (l < 0) {
						if (abs(l) % 2 == 0) {
							asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
							asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
						} else {
							asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
							asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
							gxsgn = -1;
							gysgn = -1;
						}
					} else {
						asprintf(&gxstr, "Y[%i]", index(k, 0));
					}
					const auto csgn = [](int i) {
						return i > 0 ? '+' : '-';
					};
					tprint("L[%i] %c= %s * %s;\n", index(n, m), csgn(mxsgn * gxsgn), mxstr, gxstr);
					flops += 2;
					if (gystr && mystr) {
						tprint("L[%i] %c= %s * %s;\n", index(n, m), csgn(-mysgn * gysgn), mystr, gystr);
						flops += 2;
					}
					if (m > 0) {
						if (gystr) {
							tprint("L[%i] %c= %s * %s;\n", index(n, -m), csgn(mxsgn * gysgn), mxstr, gystr);
							flops += 2;
						}
						if (mystr) {
							tprint("L[%i] %c= %s * %s;\n", index(n, -m), csgn(mysgn * gxsgn), mystr, gxstr);
							flops += 2;
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (gystr) {
						free(gystr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
		}
	}
	if (P > 1) {
		tprint("L[%i] -= T(2) * x * L[%i];\n", index(1, 1), (P + 1) * (P + 1));
		tprint("L[%i] -= T(2) * y * L[%i];\n", index(1, -1), (P + 1) * (P + 1));
		tprint("L[%i] -= T(2) * z * L[%i];\n", index(1, 0), (P + 1) * (P + 1));
		flops += 9;
		if (!nophi) {
			tprint("L[%i] += x * x * L[%i];\n", index(0, 0), (P + 1) * (P + 1));
			tprint("L[%i] += y * y * L[%i];\n", index(0, 0), (P + 1) * (P + 1));
			tprint("L[%i] += z * z * L[%i];\n", index(0, 0), (P + 1) * (P + 1));
			flops += 9;
		}
	}

	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int L2L_rot1(int P) {
	int flops = 0;
	const auto c = tprint_on;
	set_tprint(false);
	flops += regular_harmonic_xz(P);
	set_tprint(c);

	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void L2L%s(expansion_type<T, %i>& L, T x, T y, T z) {\n", nophi ? "_nopot" : "", P);
	indent();

	tprint("const T R2 = (x * x + y * y);\n");
	flops += 3;
	tprint("const T R = SQRT(R2);\n");
	flops += 4;
	tprint("const T Rzero = (R<T(1e-37));\n");
	flops++;
	tprint("const T Rinv = T(1) / (R + Rzero);\n");
	flops += 5;
	tprint("T cosphi = x * Rinv + Rzero;\n");
	flops += 2;
	tprint("T sinphi = -y * Rinv;\n");
	flops += z_rot(P, "L", false, false);
	const auto yindex = [](int l, int m) {
		return l*(l+1)/2+m;
	};

	tprint("expansion_xz_type<T, %i> Y;\n", P);
	tprint("regular_harmonic_xz(Y, -R, -z);\n");
	flops += 2;
	for (int n = 0; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			for (int k = 1; k <= P - n; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					char* mxstr = nullptr;
					char* mystr = nullptr;
					char* gxstr = nullptr;
					int mxsgn = 1;
					int mysgn = 1;
					int gxsgn = 1;
					if (m + l > 0) {
						asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
						asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mysgn = -1;
						} else {
							asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mxsgn = -1;
						}
					} else {
						asprintf(&mxstr, "L[%i]", index(n + k, 0));
					}
					asprintf(&gxstr, "Y[%i]", yindex(k, abs(l)));
					if (l < 0 && abs(l) % 2 != 0) {
						gxsgn = -1;
					}
					//	L[index(n, m)] += Y(k, l) * M(n + k, m + l);
					const auto csgn = [](int i) {
						return i > 0 ? '+' : '-';
					};
					tprint("L[%i] %c= %s * %s;\n", index(n, m), csgn(mxsgn * gxsgn), mxstr, gxstr);
					flops += 2;
					if (m > 0) {
						if (mystr) {
							tprint("L[%i] %c= %s * %s;\n", index(n, -m), csgn(mysgn * gxsgn), mystr, gxstr);
							flops += 2;
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
		}
	}
	tprint("sinphi = -sinphi;\n");
	flops++;
	flops += z_rot(P, "L", false, false);
	flops++;
	if (P > 1) {
		tprint("L[%i] -= T(2) * x * L[%i];\n", index(1, 1), (P + 1) * (P + 1));
		tprint("L[%i] -= T(2) * y * L[%i];\n", index(1, -1), (P + 1) * (P + 1));
		tprint("L[%i] -= T(2) * z * L[%i];\n", index(1, 0), (P + 1) * (P + 1));
		flops += 9;
		if (!nophi) {
			tprint("L[%i] += x * x * L[%i];\n", index(0, 0), (P + 1) * (P + 1));
			tprint("L[%i] += y * y * L[%i];\n", index(0, 0), (P + 1) * (P + 1));
			tprint("L[%i] += z * z * L[%i];\n", index(0, 0), (P + 1) * (P + 1));
			flops += 9;
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int L2L_rot2(int P) {
	int flops = 0;
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void L2L%s(expansion_type<T, %i>& L, T x, T y, T z) {\n", nophi ? "_nopot" : "", P);
	indent();
	//const auto Y = spherical_regular_harmonic<T, P>(-x, -y, -z);

	tprint("const T R2 = (x * x + y * y);\n");
	flops += 3;
	tprint("const T R = SQRT(R2);\n");
	flops += 4;
	tprint("const T r = SQRT(R2 + z * z);\n");
	flops += 6;
	tprint("const T Rzero = T(R<T(1e-37));");
	tprint("const T rzero = T(r<T(1e-37));");
	flops += 2;
	tprint("const T Rinv = T(1) / (R + Rzero);\n");
	flops += 5;
	tprint("const T rinv = T(1) / (r + rzero);\n");
	flops += 5;
	tprint("T cosphi = y * Rinv;\n");
	flops++;
	tprint("T sinphi = x * Rinv + Rzero;\n");
	flops += 1;
	flops += 1;
	flops += z_rot(P, "L", false, false);
	flops += xz_swap(P, "L", true, false, false, false);
	tprint("T cosphi0 = cosphi;\n");
	tprint("T sinphi0 = sinphi;\n");
	tprint("cosphi = z * rinv + rzero;\n");
	flops++;
	tprint("sinphi = -R * rinv;\n");
	flops += z_rot(P, "L", false, false);
	flops += xz_swap(P, "L", true, false, false, false);
	tprint("T c0[%i];\n", P + 1);
	tprint("c0[0] = T(1);\n");
	for (int n = 1; n <= P; n++) {
		tprint("c0[%i] = -r * c0[%i];\n", n, n - 1);
		flops += 2;
	}
	for (int n = 2; n <= P; n++) {
		tprint("c0[%i] *= T(%.16e);\n", n, 1.0 / factorial(n));
		flops += 1;
	}
	for (int n = nophi; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			for (int k = 1; k <= P - n; k++) {
				if (abs(m) > n + k) {
					continue;
				}
				if (-abs(m) < -(k + n)) {
					continue;
				}
				tprint("L[%i] += L[%i] * c0[%i];\n", index(n, m), index(n + k, m), k);
				flops += 2;
				if (m > 0) {
					tprint("L[%i] += L[%i] * c0[%i];\n", index(n, -m), index(n + k, -m), k);
					flops += 2;
				}
			}

		}
	}
	flops += xz_swap(P, "L", true, false, false, false);
	tprint("sinphi = -sinphi;\n");
	flops += z_rot(P, "L", false, false);
	flops += xz_swap(P, "L", true, false, false, false);
	tprint("cosphi = cosphi0;\n");
	tprint("sinphi = -sinphi0;\n");
	flops += 1;
	flops += z_rot(P, "L", false, false);
	if (P > 1) {
		tprint("L[%i] -= T(2) * x * L[%i];\n", index(1, 1), (P + 1) * (P + 1));
		tprint("L[%i] -= T(2) * y * L[%i];\n", index(1, -1), (P + 1) * (P + 1));
		tprint("L[%i] -= T(2) * z * L[%i];\n", index(1, 0), (P + 1) * (P + 1));
		flops += 9;
		if (!nophi) {
			tprint("L[%i] += x * x * L[%i];\n", index(0, 0), (P + 1) * (P + 1));
			tprint("L[%i] += y * y * L[%i];\n", index(0, 0), (P + 1) * (P + 1));
			tprint("L[%i] += z * z * L[%i];\n", index(0, 0), (P + 1) * (P + 1));
			flops += 9;
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int L2P(int P) {
	int flops = 0;
	const auto c = tprint_on;
	set_tprint(false);
	flops += regular_harmonic(P);
	set_tprint(c);
	tprint("\n");
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT expansion_type<T, %i> L2P%s(expansion_type<T, %i>& L, T x, T y, T z) {\n", 1, nophi ? "_norot" : "", P);
	indent();
	//const auto Y = spherical_regular_harmonic<T, P>(-x, -y, -z);
	tprint("expansion_type<T, %i> Y;\n", P);
	tprint("expansion_type<T,1> L1;\n");
	if (!nophi) {
		tprint("L1[0] = L1[1] = L1[2] = L1[3] = T(0);\n");
	} else {
		tprint("L1[1] = L1[2] = L1[3] = T(0);\n");
	}
	tprint("regular_harmonic(Y, -x, -y, -z);\n");
	flops += 3;
	for (int n = nophi; n <= 1; n++) {
		for (int m = 0; m <= n; m++) {
			bool first = true;
			for (int k = 0; k <= P - n; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					char* mxstr = nullptr;
					char* mystr = nullptr;
					char* gxstr = nullptr;
					char* gystr = nullptr;
					int mxsgn = 1;
					int mysgn = 1;
					int gxsgn = 1;
					int gysgn = 1;
					if (m + l > 0) {
						asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
						asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mysgn = -1;
						} else {
							asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mxsgn = -1;
						}
					} else {
						asprintf(&mxstr, "L[%i]", index(n + k, 0));
					}
					if (l > 0) {
						asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
						asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
						gysgn = -1;
					} else if (l < 0) {
						if (abs(l) % 2 == 0) {
							asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
							asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
						} else {
							asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
							asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
							gxsgn = -1;
							gysgn = -1;
						}
					} else {
						asprintf(&gxstr, "Y[%i]", index(k, 0));
					}
					const auto csgn = [](int i) {
						return i > 0 ? '+' : '-';
					};
					if (first) {
						tprint("L1[%i] = %s%s * %s;\n", index(n, m), (mxsgn * gxsgn) > 0 ? "" : "-", mxstr, gxstr);
						first = false;
						flops += ((mxsgn * gxsgn) == -1);
					} else {
						tprint("L1[%i] %c= %s * %s;\n", index(n, m), csgn(mxsgn * gxsgn), mxstr, gxstr);
						flops += 2;
					}
					if (gystr && mystr) {
						tprint("L1[%i] %c= %s * %s;\n", index(n, m), csgn(-mysgn * gysgn), mystr, gystr);
						flops += 2;
					}
					if (m > 0) {
						if (gystr) {
							tprint("L1[%i] %c= %s * %s;\n", index(n, -m), csgn(mxsgn * gysgn), mxstr, gystr);
							flops += 2;
						}
						if (mystr) {
							tprint("L1[%i] %c= %s * %s;\n", index(n, -m), csgn(mysgn * gxsgn), mystr, gxstr);
							flops += 2;
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (gystr) {
						free(gystr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
		}
	}
	if (P >= 1) {
		tprint("L1[%i] -= T(2) * x * L[%i];\n", index(1, 1), (P + 1) * (P + 1));
		tprint("L1[%i] -= T(2) * y * L[%i];\n", index(1, -1), (P + 1) * (P + 1));
		tprint("L1[%i] -= T(2) * z * L[%i];\n", index(1, 0), (P + 1) * (P + 1));
		flops += 9;
		if (!nophi) {
			tprint("L1[%i] += x * x * L[%i];\n", index(0, 0), (P + 1) * (P + 1));
			tprint("L1[%i] += y * y * L[%i];\n", index(0, 0), (P + 1) * (P + 1));
			tprint("L1[%i] += z * z * L[%i];\n", index(0, 0), (P + 1) * (P + 1));
			flops += 9;
		}
	}
	tprint("return L1;");

	deindent();
	tprint("}\n");
	tprint("\n");

	return flops;
}
int P2M(int P) {
	int flops = 0;
	tprint("\n");
	const auto c = tprint_on;
	set_tprint(false);
	flops += regular_harmonic(P);
	set_tprint(c);
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT void P2M%s( multipole_type<T,%i>& M, T x, T y, T z ) {\n", nophi ? "_nopot" : "", P + 1);
	indent();
	tprint("regular_harmonic(M, -x, -y, -z);\n");
	if (!nophi) {
		tprint("M[%i] = x * x + y * y + z * z;\n", (P + 1) * (P + 1));
	}
	flops += 5;
	deindent();
	tprint("}\n");
	tprint("\n");
	return flops + 3;
}

int main() {
	tprint("#pragma once\n");
	tprint("\n");
	tprint("#include <cosmictiger/containers.hpp>\n");
	tprint("#include <cosmictiger/cuda.hpp>\n");
	tprint("\n");
	tprint("CUDA_EXPORT inline float SQRT(float x) {\n");
	indent();
	tprint("return sqrtf(x);\n");
	deindent();
	tprint("}\n");
	tprint("CUDA_EXPORT inline double SQRT(double x) {\n");
	indent();
	tprint("return sqrt(x);\n");
	deindent();
	tprint("}\n");
	tprint("CUDA_EXPORT inline float EXP(float x) {\n");
	indent();
	tprint("return expf(x);\n");
	deindent();
	tprint("}\n");
	tprint("CUDA_EXPORT inline double EXP(double x) {\n");
	indent();
	tprint("return exp(x);\n");
	deindent();
	tprint("}\n");
	tprint("CUDA_EXPORT inline float ERFC(float x) {\n");
	indent();
	tprint("return erfcf(x);\n");
	deindent();
	tprint("}\n");
	tprint("CUDA_EXPORT inline double ERFC(double x) {\n");
	indent();
	tprint("return ERFC(x);\n");
	deindent();
	tprint("}\n");
	tprint("CUDA_EXPORT inline void SINCOS(float x, float * s, float* c) {\n");
	indent();
	tprint("sincosf(x, s, c);\n");
	deindent();
	tprint("}\n");
	tprint("CUDA_EXPORT inline void SINCOS(double x, double* s, double* c) {\n");
	indent();
	tprint("sincos(x, s, c);\n");
	deindent();
	tprint("}\n");
	tprint("CUDA_EXPORT inline float FMA(float a, float b, float c) {\n");
	indent();
	tprint("return fmaf(a, b, c);\n");
	deindent();
	tprint("}\n");
	tprint("CUDA_EXPORT inline double FMA(double a, double b, double c) {\n");
	indent();
	tprint("return fma(a, b, c);\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("template<class T, int P>\n");
	tprint("using multipole_type = array<T, (P > 2) ? P*P+1:P*P>;\n");
	tprint("\n");
	tprint("template<class T, int P>\n");
	tprint("using expansion_type = array<T, (P > 1) ? (P+1)*(P+1)+1:(P+1)*(P+1)>;\n");
	tprint("\n");
	tprint("template<class T, int P>\n");
	tprint("using expansion_xz_type = array<T, (P > 1) ? (P+1)*(P+2)/2+1:(P+1)*(P+2)/2>;\n");
	tprint("\n");
	constexpr int pmax = 10;
	for (int b = 0; b < 1; b++) {
		nophi = b != 0;
		std::vector<int> pc_flops(pmax + 1);
		std::vector<int> cp_flops(pmax + 1);
		std::vector<int> cc_flops(pmax + 1);
		std::vector<int> m2m_flops(pmax + 1);
		std::vector<int> l2l_flops(pmax + 1);
		std::vector<int> l2p_flops(pmax + 1);
		std::vector<int> p2m_flops(pmax + 1);
		std::vector<int> pc_rot(pmax + 1);
		std::vector<int> cc_rot(pmax + 1);
		std::vector<int> m2m_rot(pmax + 1);
		std::vector<int> l2l_rot(pmax + 1);
		set_tprint(false);
		fprintf(stderr, "%2s %5s %5s %2s %5s %5s %2s %5s %5s %5s %5s %2s %5s %5s %2s %5s %5s %5s %5s %8s %8s %8s\n", "p", "CC", "eff", "-r", "PC", "eff", "-r",
				"CP", "eff", "M2M", "eff", "-r", "L2L", "eff", "-r", "P2M", "eff", "L2P", "eff", "CC_ewald", "green", "m2l");
		for (int P = 3; P <= pmax; P++) {
			auto r0 = m2l_norot(P, P);
			auto r1 = m2l_rot1(P, P);
			auto r2 = m2l_rot2(P, P);
			if (r0 <= r1 && r0 <= r2) {
				cc_flops[P] = r0;
				cc_rot[P] = 0;
			} else if (r1 <= r0 && r1 <= r2) {
				cc_flops[P] = r1;
				cc_rot[P] = 1;
			} else {
				cc_flops[P] = r2;
				cc_rot[P] = 2;
			}
			r0 = m2l_norot(P, 1);
			r1 = m2l_rot1(P, 1);
			r2 = m2l_rot2(P, 1);
			if (r0 <= r1 && r0 <= r2) {
				pc_flops[P] = r0;
				pc_rot[P] = 0;
			} else if (r1 <= r0 && r1 <= r2) {
				pc_flops[P] = r1;
				pc_rot[P] = 1;
			} else {
				pc_flops[P] = r2;
				pc_rot[P] = 2;
			}
			cp_flops[P] = m2l_cp(P);
			r0 = M2M_norot(P - 1);
			r1 = M2M_rot1(P - 1);
			r2 = M2M_rot2(P - 1);
			if (r0 <= r1 && r0 <= r2) {
				m2m_flops[P] = r0;
				m2m_rot[P] = 0;
			} else if (r1 <= r0 && r1 <= r2) {
				m2m_flops[P] = r1;
				m2m_rot[P] = 1;
			} else {
				m2m_flops[P] = r2;
				m2m_rot[P] = 2;
			}
			r0 = L2L_norot(P);
			r1 = L2L_rot1(P);
			r2 = L2L_rot2(P);
			if (r0 <= r1 && r0 <= r2) {
				l2l_flops[P] = r0;
				l2l_rot[P] = 0;
			} else if (r1 <= r0 && r1 <= r2) {
				l2l_flops[P] = r1;
				l2l_rot[P] = 1;
			} else {
				l2l_flops[P] = r2;
				l2l_rot[P] = 2;
			}
			l2p_flops[P] = L2P(P);
			p2m_flops[P] = P2M(P - 1);
			int eflopsg = ewald_greens(P);
			int eflopsm = m2lg(P, P);

			fprintf(stderr, "%2i %5i %5.2f %2i %5i %5.2f %2i %5i %5.2f %5i %5.2f %2i %5i %5.2f %2i %5i %5.2f %5i %5.2f %8i %8i %8i\n", P, cc_flops[P],
					cc_flops[P] / pow(P + 1, 3), cc_rot[P], pc_flops[P], pc_flops[P] / pow(P + 1, 2), pc_rot[P], cp_flops[P], cp_flops[P] / pow(P + 1, 2),
					m2m_flops[P], m2m_flops[P] / pow(P + 1, 3), m2m_rot[P], l2l_flops[P], l2l_flops[P] / pow(P + 1, 3), l2l_rot[P], p2m_flops[P],
					p2m_flops[P] / pow(P + 1, 2), l2p_flops[P], l2p_flops[P] / pow(P + 1, 2), eflopsg + eflopsm, eflopsg, eflopsm);
		}
		set_tprint(true);
		if (b == 0)
			regular_harmonic(2);
		if (b == 0)
			regular_harmonic_xz(2);
		for (int P = 3; P <= pmax; P++) {
			if (b == 0)
				greens(P);
			if (b == 0)
				greens_xz(P);
			switch (cc_rot[P]) {
			case 0:
				m2l_norot(P, P);
				break;
			case 1:
				m2l_rot1(P, P);
				break;
			case 2:
				m2l_rot2(P, P);
				break;
			};
			if (P > 1) {
				switch (pc_rot[P]) {
				case 0:
					m2l_norot(P, 1);
					break;
				case 1:
					m2l_rot1(P, 1);
					break;
				case 2:
					m2l_rot2(P, 1);
					break;
				};
			}
			m2l_cp(P);
			if (b == 0)
				regular_harmonic(P);
			if (b == 0)
				regular_harmonic_xz(P);
			switch (m2m_rot[P]) {
			case 0:
				M2M_norot(P - 1);
				break;
			case 1:
				M2M_rot1(P - 1);
				break;
			case 2:
				M2M_rot2(P - 1);
				break;
			};
			switch (l2l_rot[P]) {
			case 0:
				L2L_norot(P);
				break;
			case 1:
				L2L_rot1(P);
				break;
			case 2:
				L2L_rot2(P);
				break;
			};
			L2P(P);
			P2M(P - 1);
			ewald_greens(P);
			m2lg(P, P);
			m2l_ewald(P);
		}
	}
	return 0;
}
