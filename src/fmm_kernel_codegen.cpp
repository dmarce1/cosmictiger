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

#define CODE_GEN_CPP
#include <cosmictiger/defs.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/tensor.hpp>

#include <algorithm>
#include <cmath>

static int ntab = 0;

void indent() {
	ntab++;
}

void deindent() {
	ntab--;
}

bool close21(double a) {
	return std::abs(1.0 - a) < 1.0e-20;
}

int sym_index(int l, int m, int n) {
	return (l + m + n) * (l + m + n + 1) * ((l + m + n) + 2) / 6 + (m + n) * ((m + n) + 1) / 2 + n;
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

#ifndef TREEPM

int compute_dx(int P, const char* name = "X", bool trless = false, bool dble = false) {
	array<int, NDIM> n;
	tprint("%s x[%i];\n", dble ? "T" : "T", trless ? P * P + 1 : (P + 1) * P * (P + 2) / 6);
	auto index = trless ? std::function<int(int, int, int)>([P](int x, int y, int z) {
				return trless_index(x,y,z,P);
			}) :std::function<int(int, int, int)>(sym_index);
	tprint("x[%i] = %s(1);\n", index(0, 0, 0), dble ? "T" : "T");
	tprint("x[%i] = %s[0];\n", index(1, 0, 0), name);
	tprint("x[%i] = %s[1];\n", index(0, 1, 0), name);
	tprint("x[%i] = %s[2];\n", index(0, 0, 1), name);
	int flops = 0;

	for (int n0 = 2; n0 < P; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[0] - n[1];
				if (trless && ((n[2] > 2) || (n[2] > 1 && !(n[0] == 0 && n[1] == 0)))) {
					continue;
				}
				array<int, NDIM> j = n;
				int j0 = n0;
				const int jmin = std::max(1, n0 / 2);
				while (j0 > jmin) {
					for (int dim = 0; dim < NDIM && j0 > jmin; dim++) {
						if (j[dim] > 0) {
							j[dim]--;
							j0--;
						}
					}
				}
				array<int, NDIM> k;
				k = n - j;
				if (trless && (((j[2] > 2) || (j[2] > 1 && !(j[0] == 0 && j[1] == 0))))) {
					PRINT("!!!!!!!!!!!\n");
					abort();
				}
				if (trless && (((k[2] > 2) || (k[2] > 1 && !(k[0] == 0 && k[1] == 0))))) {
					PRINT("!!!!!!!!!!!\n");
					abort();
				}
				tprint("x[%i] = x[%i] * x[%i]; // %i %i %i | %i %i %i | %i %i %i\n", index(n[0], n[1], n[2]), index(k[0], k[1], k[2]), index(j[0], j[1], j[2]),
						n[0], n[1], n[2], j[0], j[1], j[2], k[0], k[1], k[2]);
				flops++;
			}
		}
	}
	return flops;
}

int compute_dx_tensor(int P, const char* name = "X") {
	array<int, NDIM> n;
	tprint("tensor_sym<T,%i> dx;\n", P);
	tprint("dx[0] = T(1);\n");
	tprint("dx[%i] = %s[0];\n", sym_index(1, 0, 0), name);
	tprint("dx[%i] = %s[1];\n", sym_index(0, 1, 0), name);
	tprint("dx[%i] = %s[2];\n", sym_index(0, 0, 1), name);
	int flops = 0;
	for (int n0 = 2; n0 < P; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[0] - n[1];
				array<int, NDIM> j = n;
				int j0 = n0;
				const int jmin = std::max(1, n0 / 2);
				while (j0 > jmin) {
					for (int dim = 0; dim < NDIM && j0 > jmin; dim++) {
						if (j[dim] > 0) {
							j[dim]--;
							j0--;
						}
					}
				}
				array<int, NDIM> k;
				k = n - j;
				tprint("dx[%i]= dx[%i] * dx[%i];\n", sym_index(n[0], n[1], n[2]), sym_index(k[0], k[1], k[2]), sym_index(j[0], j[1], j[2]));
				flops++;
			}
		}
	}
	return flops;
}

int acc(std::string a, array<int, NDIM> n, std::string c, array<int, NDIM> j) {
	tprint("%s%i%i%i += %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], c.c_str(), j[0], j[1], j[2]);
	return 1;
}

int dec(std::string a, array<int, NDIM> n, std::string c, array<int, NDIM> j) {
	tprint("%s%i%i%i -= %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], c.c_str(), j[0], j[1], j[2]);
	return 1;
}

int eqp(std::string a, array<int, NDIM> n, std::string c, array<int, NDIM> j) {
	tprint("%s%i%i%i = %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], c.c_str(), j[0], j[1], j[2]);
	return 0;
}

int eqn(std::string a, array<int, NDIM> n, std::string c, array<int, NDIM> j) {
	tprint("%s%i%i%i = -%s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], c.c_str(), j[0], j[1], j[2]);
	return 1;
}

int mul(std::string a, array<int, NDIM> n, double b, std::string c, array<int, NDIM> j) {
	tprint("%s%i%i%i = T(%.9e) * %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], b, c.c_str(), j[0], j[1], j[2]);
	return 1;
}

int fma(std::string a, array<int, NDIM> n, double b, std::string c, array<int, NDIM> j) {
	tprint("%s%i%i%i = fmaf(T(%.9e), %s%i%i%i, %s%i%i%i);\n", a.c_str(), n[0], n[1], n[2], b, c.c_str(), j[0], j[1], j[2], a.c_str(), n[0], n[1], n[2]);
	return 2;
}

template<int P>
int compute_detrace(std::string iname, std::string oname, char type = 'f') {
	array<int, NDIM> m;
	array<int, NDIM> k;
	array<int, NDIM> n;

	struct entry_type {
		array<int, NDIM> src;
		array<int, NDIM> dest;
		double factor;
	};
	vector<entry_type> entries;
	int flops = 0;

	vector<std::string> asn;
	vector<std::string> op;
	for (int m0 = 1; m0 <= P / 2; m0++) {
		array<int, NDIM> j;
		const int Q = P - 2 * m0;
		for (j[0] = 0; j[0] < Q; j[0]++) {
			for (j[1] = 0; j[1] < Q - j[0]; j[1]++) {
				for (j[2] = 0; j[2] < Q - j[0] - j[1]; j[2]++) {
					const int j0 = j[0] + j[1] + j[2];
					int n0 = j0 + 2 * m0;
					bool first = true;
					array<int, NDIM> k;
					for (k[0] = 0; k[0] <= m0; k[0]++) {
						for (k[1] = 0; k[1] <= m0 - k[0]; k[1]++) {
							k[2] = m0 - k[0] - k[1];
							const double num = factorial(m0);
							const double den = vfactorial(k);
							const double factor = num / den;
							const auto p = j + k * 2;
							char* str;
							if (first) {
								if (close21(factor)) {
									ASPRINTF(&str, "T %s_%i_%i_%i%i%i = %s[%i];\n", iname.c_str(), n0, m0, j[0], j[1], j[2], iname.c_str(), sym_index(p[0], p[1], p[2]));
								} else if (close21(-factor)) {
									ASPRINTF(&str, "T %s_%i_%i_%i%i%i = -%s[%i];\n", iname.c_str(), n0, m0, j[0], j[1], j[2], iname.c_str(),
											sym_index(p[0], p[1], p[2]));
									flops++;
								} else {
									ASPRINTF(&str, "T %s_%i_%i_%i%i%i = T(%.9e) * %s[%i];\n", iname.c_str(), n0, m0, j[0], j[1], j[2], factor, iname.c_str(),
											sym_index(p[0], p[1], p[2]));
									flops++;
								}
								asn.push_back(str);
								first = false;
								free(str);
							} else {
								if (close21(factor)) {
									ASPRINTF(&str, "%s_%i_%i_%i%i%i += %s[%i];\n", iname.c_str(), n0, m0, j[0], j[1], j[2], iname.c_str(), sym_index(p[0], p[1], p[2]));
									flops++;
								} else if (close21(-factor)) {
									ASPRINTF(&str, "%s_%i_%i_%i%i%i -= %s[%i];\n", iname.c_str(), n0, m0, j[0], j[1], j[2], iname.c_str(), sym_index(p[0], p[1], p[2]));
									flops++;
								} else {
									ASPRINTF(&str, "%s_%i_%i_%i%i%i = fmaf(T(%.9e), %s[%i], %s_%i_%i_%i%i%i);\n", iname.c_str(), n0, m0, j[0], j[1], j[2], factor,
											iname.c_str(), sym_index(p[0], p[1], p[2]), iname.c_str(), n0, m0, j[0], j[1], j[2]);
									flops += 2;
								}
								op.push_back(str);
								free(str);
							}
						}
					}
				}
			}
		}
	}
	int maxop = (op.size() + 1) / 2;
	for (int i = 0; i < asn.size(); i++) {
		tprint("%s", asn[i].c_str());
	}
	for (int i = 0; i < maxop; i++) {
		tprint("%s", op[i].c_str());
		if (i + maxop < op.size()) {
			tprint("%s", op[i + maxop].c_str());
		}
	}
	op.resize(0);
	asn.resize(0);
	for (n[0] = 0; n[0] < P; n[0]++) {
		for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
			const int nzmax = (n[0] == 0 && n[1] == 0) ? intmin(3, P) : intmin(P - n[0] - n[1], 2);
			for (n[2] = 0; n[2] < nzmax; n[2]++) {
				const int n0 = n[0] + n[1] + n[2];
				bool first = true;
				for (m[0] = 0; m[0] <= n[0] / 2; m[0]++) {
					for (m[1] = 0; m[1] <= n[1] / 2; m[1]++) {
						for (m[2] = 0; m[2] <= n[2] / 2; m[2]++) {
							const int m0 = m[0] + m[1] + m[2];
							if (type == 'd' && ((n0 == 2 && (n[0] == 2 || n[1] == 2 || n[2] == 2)) && m0 == 1)) {
								continue;
							}
							double num = double(n1pow(m0) * dfactorial(2 * n0 - 2 * m0 - 1) * vfactorial(n));
							double den = double((1 << m0) * vfactorial(m) * vfactorial(n - (m) * 2));
							double factor = num / den;
							if (type == 'd') {
								factor *= 1.0 / dfactorial(2 * n0 - 1);
							}
							const auto p = n - m * 2;
							char* str;
							if (first) {
								if (m0 > 0) {
									if (close21(factor)) {
										ASPRINTF(&str, "%s[%i] = %s_%i_%i_%i%i%i;\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), iname.c_str(), n0, m0, p[0], p[1],
												p[2]);
									} else if (close21(-factor)) {
										ASPRINTF(&str, "%s[%i] = -%s_%i_%i_%i%i%i;\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), iname.c_str(), n0, m0, p[0],
												p[1], p[2]);
										flops++;
									} else {
										ASPRINTF(&str, "%s[%i] = T(%.9e) * %s_%i_%i_%i%i%i;\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), factor, iname.c_str(),
												n0, m0, p[0], p[1], p[2]);
										flops++;
									}
								} else {
									if (close21(factor)) {
										ASPRINTF(&str, "%s[%i] = %s[%i];\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), iname.c_str(),
												sym_index(p[0], p[1], p[2]));
									} else if (close21(-factor)) {
										ASPRINTF(&str, "%s[%i] = -%s[%i];\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), iname.c_str(),
												sym_index(p[0], p[1], p[2]));
										flops++;
									} else {
										ASPRINTF(&str, "%s[%i] = T(%.9e) * %s[%i];\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), factor, iname.c_str(),
												sym_index(p[0], p[1], p[2]));;
										flops++;
									}
								}
								asn.push_back(str);
								free(str);
								first = false;
							} else {
								if (close21(factor)) {
									if (m0 > 0) {
										ASPRINTF(&str, "%s[%i] += %s_%i_%i_%i%i%i;\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), iname.c_str(), n0, m0, p[0],
												p[1], p[2]);
										flops++;
									} else {
										ASPRINTF(&str, "%s[%i] += %s[%i];\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), iname.c_str(),
												sym_index(p[0], p[1], p[2]));
										flops++;
									}
								} else if (close21(-factor)) {
									if (m0 > 0) {
										ASPRINTF(&str, "%s[%i] -= %s_%i_%i_%i%i%i;\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), iname.c_str(), n0, m0, p[0],
												p[1], p[2]);
										flops++;
									} else {
										ASPRINTF(&str, "%s[%i] -= %s[%i];\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), iname.c_str(),
												sym_index(p[0], p[1], p[2]));
										flops++;
									}
								} else {
									if (m0 > 0) {
										ASPRINTF(&str, "%s[%i] = fmaf(T(%.9e), %s_%i_%i_%i%i%i, %s[%i]);\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), factor,
												iname.c_str(), n0, m0, p[0], p[1], p[2], oname.c_str(), trless_index(n[0], n[1], n[2], P));
										flops += 2;
									} else {
										ASPRINTF(&str, "%s[%i] = fmaf(T(%.9e), %s[%i], %s[%i]);\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), factor,
												iname.c_str(), sym_index(p[0], p[1], p[2]), oname.c_str(), trless_index(n[0], n[1], n[2], P));
										flops += 2;
									}
								}
								op.push_back(str);
								free(str);
							}
						}
					}
				}
			}
		}
	}
	maxop = (op.size() + 1) / 2;
	for (int i = 0; i < asn.size(); i++) {
		tprint("%s", asn[i].c_str());
	}
	for (int i = 0; i < maxop; i++) {
		tprint("%s", op[i].c_str());
		if (i + maxop < op.size()) {
			tprint("%s", op[i + maxop].c_str());
		}
	}

	return flops;
}

template<int P>
int compute_detraceD(std::string iname, std::string oname, char type = 'f') {
	array<int, NDIM> m;
	array<int, NDIM> k;
	array<int, NDIM> n;

	struct entry_type {
		array<int, NDIM> src;
		array<int, NDIM> dest;
		double factor;
	};
	vector<entry_type> entries;
	int flops = 0;
	vector<std::string> asn;
	vector<std::string> op;
	op.resize(0);
	asn.resize(0);
	for (n[0] = 0; n[0] < P; n[0]++) {
		for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
			const int nzmax = (n[0] == 0 && n[1] == 0) ? intmin(3, P) : intmin(P - n[0] - n[1], 2);
			for (n[2] = 0; n[2] < nzmax; n[2]++) {
				const int n0 = n[0] + n[1] + n[2];
				bool first = true;
				for (m[0] = 0; m[0] <= n[0] / 2; m[0]++) {
					for (m[1] = 0; m[1] <= n[1] / 2; m[1]++) {
						for (m[2] = 0; m[2] <= n[2] / 2; m[2]++) {
							const int m0 = m[0] + m[1] + m[2];
							double num = double(n1pow(m0) * dfactorial(2 * n0 - 2 * m0 - 1) * vfactorial(n));
							double den = double((1 << m0) * vfactorial(m) * vfactorial(n - (m) * 2));
							double factor = num / den;
							const auto p = n - m * 2;
							char* str;
							if (first) {
								if (close21(factor)) {
									ASPRINTF(&str, "%s[%i] = %s[%i];\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), iname.c_str(),
											trless_index(p[0], p[1], p[2], P));
								} else if (close21(-factor)) {
									ASPRINTF(&str, "%s[%i] = -%s[%i];\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), iname.c_str(),
											trless_index(p[0], p[1], p[2], P));
									flops++;
								} else {
									ASPRINTF(&str, "%s[%i] = T(%.9e) * %s[%i];\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), factor, iname.c_str(),
											trless_index(p[0], p[1], p[2], P));
									flops++;
								}
								asn.push_back(str);
								free(str);
								first = false;
							} else {
								if (close21(factor)) {
									ASPRINTF(&str, "%s[%i] += %s[%i];\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), iname.c_str(),
											trless_index(p[0], p[1], p[2], P));
									flops += 1;
								} else if (close21(-factor)) {
									ASPRINTF(&str, "%s[%i] -= %s[%i];\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), iname.c_str(),
											trless_index(p[0], p[1], p[2], P));
									flops += 1;
								} else {
									ASPRINTF(&str, "%s[%i] = fmaf(T(%.9e), %s[%i], %s[%i]);\n", oname.c_str(), trless_index(n[0], n[1], n[2], P), factor, iname.c_str(),
											trless_index(p[0], p[1], p[2], P), oname.c_str(), trless_index(n[0], n[1], n[2], P));
									flops += 1;
								}
								op.push_back(str);
								free(str);
							}
						}
					}
				}
			}
		}
	}
	int maxop = (op.size() + 1) / 2;
	for (int i = 0; i < asn.size(); i++) {
		tprint("%s", asn[i].c_str());
	}
	for (int i = 0; i < maxop; i++) {
		tprint("%s", op[i].c_str());
		if (i + maxop < op.size()) {
			tprint("%s", op[i + maxop].c_str());
		}
	}

	return flops;
}

template<int P>
int compute_detrace_ewald(std::string iname, std::string oname) {

	array<int, NDIM> m;
	array<int, NDIM> k;
	array<int, NDIM> n;

	struct entry_type {
		array<int, NDIM> src;
		array<int, NDIM> dest;
		double factor;
	};
	vector<entry_type> entries;
	int flops = 0;

	vector<std::string> asn;
	vector<std::string> op;
	op.resize(0);
	asn.resize(0);
	for (n[0] = 0; n[0] < P; n[0]++) {
		for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
			for (n[2] = 0; n[2] < P - n[0] - n[1]; n[2]++) {
				const int n0 = n[0] + n[1] + n[2];
				bool first = true;
				for (m[0] = 0; m[0] <= n[0] / 2; m[0]++) {
					for (m[1] = 0; m[1] <= n[1] / 2; m[1]++) {
						for (m[2] = 0; m[2] <= n[2] / 2; m[2]++) {
							const int m0 = m[0] + m[1] + m[2];
							double num = double(vfactorial(n));
							double den = double((1 << m0) * vfactorial(m) * vfactorial(n - (m) * 2));
							double factor = num / den;
							const auto p = n - m * 2;
							char* str;
							if (close21(factor)) {
								ASPRINTF(&str, "%s[%i] = fmaf(%s[%i], Drinvpow_%i_%i, %s[%i]);\n", oname.c_str(), sym_index(n[0], n[1], n[2]), iname.c_str(),
										sym_index(p[0], p[1], p[2]), n0 - m0, m0, oname.c_str(), sym_index(n[0], n[1], n[2]));
								flops += 2;
							} else if (close21(-factor)) {
								ASPRINTF(&str, "%s[%i] -= %s[%i] * Drinvpow_%i_%i;\n", oname.c_str(), sym_index(n[0], n[1], n[2]), iname.c_str(),
										sym_index(p[0], p[1], p[2]), n0 - m0, m0);
								flops += 2;
							} else {
								ASPRINTF(&str, "%s[%i] = fmaf(T(%.9e), %s[%i]*Drinvpow_%i_%i, %s[%i]);\n", oname.c_str(), sym_index(n[0], n[1], n[2]), factor,
										iname.c_str(), sym_index(p[0], p[1], p[2]), n0 - m0, m0, oname.c_str(), sym_index(n[0], n[1], n[2]));

								flops += 3;
							}
							op.push_back(str);
							free(str);
						}
					}
				}
			}
		}
	}
	int maxop = op.size();
	for (int i = 0; i < maxop; i++) {
		tprint("%s", op[i].c_str());
	}

	return flops;
}

#define P ORDER
template<int Q>
void const_ref_compute(int sign, tensor_trless_sym<int, Q>& counts, tensor_trless_sym<int, Q>& signs, array<int, NDIM> n) {
	int flops = 0;
	if (n[2] >= 2 && !(n[0] == 0 && n[1] == 0 && n[2] == 2)) {
		auto n1 = n;
		auto n2 = n;
		n1[2] -= 2;
		n2[2] -= 2;
		n1[0] += 2;
		n2[1] += 2;
		const_ref_compute(-sign, counts, signs, n1);
		const_ref_compute(-sign, counts, signs, n2);
	} else {
		counts(n)++;signs
		(n) = sign;
	}
}

template<int Q>
int print_const_ref(std::string name, std::string& cmd, const tensor_trless_sym<int, Q>& counts, const tensor_trless_sym<int, Q>& signs, array<int, NDIM> n) {
	int flops = 0;
	array<int, NDIM> k;
	array<int, NDIM> last_index;
	for (k[0] = 0; k[0] < Q; k[0]++) {
		for (k[1] = 0; k[1] < Q - k[0]; k[1]++) {
			for (k[2] = 0; k[2] < Q - k[0] - k[1]; k[2]++) {
				if (k[2] < 2 || (k[0] == 0 && k[1] == 0 && k[2] == 2)) {
					if (counts(k)) {
						last_index = k;
					}
				}
			}
		}
	}
	cmd += signs(last_index) == 1 ? "+" : "-";
	if (signs(last_index) != 1) {
		flops++;
	}
	int opened = 0;
	bool fma = false;
	for (int l = 0; l < Q * Q + 1; l++) {
		if (counts[l]) {
			if (opened && !fma) {
				cmd += "+";
			} else if (fma) {
				cmd += ",";
			}
			flops++;
			opened++;
			if (counts[l] == 1) {
				cmd += "(";
				cmd += name;
				cmd += "[";
				cmd += std::to_string(l);
				cmd += "]";
			} else {
				flops++;
				if (n[0] != last_index[0] || n[1] != last_index[1] || n[2] != last_index[2]) {
					fma = true;
				} else {
					fma = false;
				}
				if (fma) {
					cmd += "fmaf(";
				} else {
					cmd += "(";
				}
				cmd += "T(";
				cmd += std::to_string(counts[l]);
				if (fma) {
					cmd += "),";
				} else {
					cmd += ")*";
				}
				cmd += name;
				cmd += "[";
				cmd += std::to_string(l);
				cmd += "]";
			}
		}
	}
	for (int l = 0; l < opened; l++) {
		cmd += ")";
	}
	return flops;
}

template<int Q>
int const_reference_trless(std::string name) {
	array<int, NDIM> n;
	int flops = 0;
	for (n[0] = 0; n[0] < Q; n[0]++) {
		for (n[1] = 0; n[1] < Q - n[0]; n[1]++) {
			for (n[2] = 0; n[2] < Q - n[0] - n[1]; n[2]++) {
				if (!(n[2] >= 2 && !(n[0] == 0 && n[1] == 0 && n[2] == 2))) {
					tprint("const T %s%i%i%i = ", name.c_str(), n[0], n[1], n[2]);
				} else {
					tprint("const T %s%i%i%i = ", name.c_str(), n[0], n[1], n[2]);
				}
				tensor_trless_sym<int, Q> counts;
				tensor_trless_sym<int, Q> signs;
				counts = 0;
				signs = 1;
				const_ref_compute(+1, counts, signs, n);
				std::string cmd;
				flops += print_const_ref(name, cmd, counts, signs, n);
				if (cmd[0] == '+') {
					flops--;
					cmd[0] = ' ';
				}
				printf("%s;\n", cmd.c_str());
			}
		}
	}
	return flops;
}

template<int Q>
int const_reference_trless_tensor(std::string name, std::string oname) {
	array<int, NDIM> n;
	int flops = 0;
	for (n[0] = 0; n[0] < Q; n[0]++) {
		for (n[1] = 0; n[1] < Q - n[0]; n[1]++) {
			for (n[2] = 0; n[2] < Q - n[0] - n[1]; n[2]++) {
				tprint("%s[%i] = ", oname.c_str(), sym_index(n[0], n[1], n[2]));
				tensor_trless_sym<int, Q> counts;
				tensor_trless_sym<int, Q> signs;
				counts = 0;
				signs = 1;
				const_ref_compute(+1, counts, signs, n);
				std::string cmd;
				flops += print_const_ref(name, cmd, counts, signs, n);
				if (cmd[0] == '+') {
					flops--;
					cmd[0] = ' ';
				}
				printf("%s;\n", cmd.c_str());
			}
		}
	}
	return flops;
}

void reference_trless(std::string name, int Q) {
	for (int l = 0; l < Q; l++) {
		for (int m = 0; m < Q - l; m++) {
			for (int n = 0; n < Q - l - m; n++) {
				if (n > 1 && !(l == 0 && m == 0 && n == 2)) {
					continue;
				}
				const int index = trless_index(l, m, n, Q);
				tprint("T& %s%i%i%i = %s[%i];\n", name.c_str(), l, m, n, name.c_str(), index);
			}
		}
	}
}

void reference_sym(std::string name, int Q) {
	for (int l = 0; l < Q; l++) {
		for (int m = 0; m < Q - l; m++) {
			for (int n = 0; n < Q - l - m; n++) {
				const int index = sym_index(l, m, n);
				tprint("const T %s%i%i%i = %s[%i];\n", name.c_str(), l, m, n, name.c_str(), index);
			}
		}
	}
}

template<int Q>
void do_expansion(bool two) {
	int flops = 0;
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT\n");
	tprint("#ifdef __CUDACC__\n");
	tprint("__noinline__\n");
	tprint("#endif\n");
	if (two) {
		tprint("tensor_trless_sym<T, %i> L2P(const tensor_trless_sym<T, %i>& La, array<T,NDIM> X, bool do_phi) {\n", Q, P);
	} else {
		tprint("tensor_trless_sym<T, %i> L2L(const tensor_trless_sym<T, %i>& La, array<T,NDIM> X, bool do_phi) {\n", Q, P);

	}
	indent();
	tprint("X[0] *= T(SCALE_FACTOR);\n");
	tprint("X[1] *= T(SCALE_FACTOR);\n");
	tprint("X[2] *= T(SCALE_FACTOR);\n");
	tprint("tensor_trless_sym<T, %i> Lb;\n", Q);
	flops += 3 + compute_dx(P);
	array<int, NDIM> n;
	array<int, NDIM> k;
	int phi_flops = 0;
	flops += const_reference_trless<P>("La");
	if (two) {
		tprint("Lb(0,0,0) = La(0,0,0);\n");
		tprint("Lb(1,0,0) = La(1,0,0);\n");
		tprint("Lb(0,1,0) = La(0,1,0);\n");
		tprint("Lb(0,0,1) = La(0,0,1);\n");
	} else {
		tprint("Lb = La;\n");
	}
	struct entry {
		int Ldest;
		array<int, NDIM> p;
		array<int, NDIM> k;
		double factor;
	};
	vector<vector<entry>> entries(two ? 4 : Q * Q + 1);
	for (int n0 = 0; n0 < Q; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[1] - n[0];
				if (n[2] <= 1 || (n[0] == 0 && n[1] == 0 && n[2] == 2)) {
					const int n0 = n[0] + n[1] + n[2];
					for (k[0] = 0; k[0] < P - n0; k[0]++) {
						for (k[1] = 0; k[1] < P - n0 - k[0]; k[1]++) {
							for (k[2] = 0; k[2] < P - n0 - k[0] - k[1]; k[2]++) {
								const auto factor = double(1) / double(vfactorial(k));
								const auto p = n + k;
								const int p0 = p[0] + p[1] + p[2];
								if (n != p) {
									entry e;
									e.Ldest = trless_index(n[0], n[1], n[2], Q);
									e.factor = factor;
									e.k = k;
									e.p = p;
									entries[e.Ldest].push_back(e);
								}
							}
						}
					}
				}
			}
		}
	}
	for (auto& e : entries) {
		std::sort(e.begin(), e.end(), [](entry a, entry b) {
					return( a.factor < b.factor );
				});
	}
	double last_factor = 1.0;
	tprint("if( do_phi ) {\n");
	indent();
	for (int j = 0; j < entries[0].size(); j++) {
		const auto factor = entries[0][j].factor;
		const auto p = entries[0][j].p;
		const auto k = entries[0][j].k;
		const auto index = entries[0][j].Ldest;
		if (!close21(last_factor / factor)) {
			tprint("Lb[0] *= T(%.9e);\n", last_factor / factor);
			last_factor = factor;
			phi_flops++;
		}
		tprint("Lb[%i] = fmaf( x[%i], La%i%i%i, Lb[%i]);\n", index, sym_index(k[0], k[1], k[2]), p[0], p[1], p[2], index);
		phi_flops += 2;
	}
	if (!close21(last_factor)) {
		tprint("Lb[0] *= T(%.9e);\n", last_factor);
	}
	deindent();
	phi_flops++;
	tprint("}\n");
	int total_size = 0;
	for (int i = 1; i < entries.size(); i++) {
		total_size += entries[i].size();
	}
	int mid;
	int half_size = 0;
	for (int i = 1; i < entries.size(); i++) {
		if (half_size >= total_size / 2) {
			mid = i;
			break;
		}
		half_size += entries[i].size();
	}
	char* str;
	vector<std::string> cmds1, cmds2;
	for (int i = 1; i < entries.size(); i++) {
		last_factor = 1.0;
		auto& cmds = i >= mid ? cmds2 : cmds1;
		if (entries[i].size()) {
			const auto index = entries[i][0].Ldest;
			for (int j = 0; j < entries[i].size(); j++) {
				const auto factor = entries[i][j].factor;
				const auto p = entries[i][j].p;
				const auto k = entries[i][j].k;
				if (!close21(factor / last_factor)) {
					ASPRINTF(&str, "Lb[%i] *= T(%.9e);\n", index, last_factor / factor);
					cmds.push_back(str);
					free(str);
					flops++;
					last_factor = factor;
				}
				ASPRINTF(&str, "Lb[%i] = fmaf( x[%i], La%i%i%i, Lb[%i]);\n", index, sym_index(k[0], k[1], k[2]), p[0], p[1], p[2], index);
				cmds.push_back(str);
				free(str);
				flops += 2;
			}
			if (!close21(last_factor)) {
				ASPRINTF(&str, "Lb[%i] *= T(%.9e);\n", index, last_factor);
				cmds.push_back(str);
				free(str);
			}
		}
	}

	int i = 0;
	int j = 0;
	while (i < cmds1.size()) {
		tprint("%s", cmds1[i].c_str());
		i++;
	}
	while (j < cmds2.size()) {
		tprint("%s", cmds2[j].c_str());
		j++;
	}
	tprint("return Lb;\n");
	printf("/* FLOPS = %i + do_phi * %i*/\n", flops, phi_flops);
	deindent();
	tprint("}\n");
}

template<int Q>
void apply_scale_factorM(const char* name) {
	array<int, NDIM> n;
	for (int n0 = 0; n0 < Q; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[1] - n[0];
				if (n[2] <= 1 || (n[0] == 0 && n[1] == 0 && n[2] == 2)) {
					tprint("%s[%i] *= SCALE_FACTOR_INV%i;\n", name, trless_index(n[0], n[1], n[2], Q), n0);
				}
			}
		}
	}
}

template<int Q>
void apply_scale_factorL(const char* name) {
	array<int, NDIM> n;
	for (int n0 = 0; n0 < Q; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[1] - n[0];
				if (n[2] <= 1 || (n[0] == 0 && n[1] == 0 && n[2] == 2)) {
					tprint("%s[%i] *= SCALE_FACTOR_INV%i;\n", name, trless_index(n[0], n[1], n[2], Q), n0 + 1);
				}
			}
		}
	}
}

template<int Q>
void do_expansion_cuda() {
	int flops = 0;
	struct entry {
		int Lsource;
		int Ldest;
		int xsource;
		float factor;
	};
	array<int, NDIM> n;
	array<int, NDIM> k;
	vector<entry> entries;
	vector<entry> phi_entries;
	for (int n0 = 0; n0 < Q; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[1] - n[0];
				if (n[2] <= 1 || (n[0] == 0 && n[1] == 0 && n[2] == 2)) {
					const int n0 = n[0] + n[1] + n[2];
					for (k[0] = 0; k[0] < P - n0; k[0]++) {
						for (k[1] = 0; k[1] < P - n0 - k[0]; k[1]++) {
							for (k[2] = 0; k[2] < P - n0 - k[0] - k[1]; k[2]++) {
								const auto factor = double(1) / double(vfactorial(k));
								const auto p = n + k;
								const int p0 = p[0] + p[1] + p[2];
								if (n != p) {
									entry e;
									e.Ldest = trless_index(n[0], n[1], n[2], P);
									e.xsource = sym_index(k[0], k[1], k[2]);
									e.Lsource = sym_index(p[0], p[1], p[2]);
									e.factor = factor;
									if (n0 == 0) {
										phi_entries.push_back(e);
									} else {
										entries.push_back(e);
									}
								}
							}
						}
					}
				}
			}
		}
	}
	std::sort(entries.begin(), entries.end(), [](entry a, entry b) {
				if( a.Ldest < b.Ldest) {
					return true;
				} else if( a.Ldest > b.Ldest) {
					return false;
				} else {
					return a.Lsource < b.Lsource;
				}
			});
	vector<entry> entries1, entries2;
	for (int i = 0; i < entries.size(); i++) {
		//	if (i < (entries.size() + 1) / 2) {
		entries1.push_back(entries[i]);
		//	} else {
		//		entries2.push_back(entries[i]);
		//	}
	}

	tprint("static __constant__ char Ldest1[%i] = { ", entries1.size());
	for (int i = 0; i < entries1.size(); i++) {
		printf("%i", entries1[i].Ldest);
		if (i != entries1.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");
	tprint("static __constant__ float factor1[%i] = { ", entries1.size());
	for (int i = 0; i < entries1.size(); i++) {
		printf("float(%.9e)", entries1[i].factor);
		if (i != entries1.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");
	tprint("static __constant__ char xsrc1[%i] = { ", entries1.size());
	for (int i = 0; i < entries1.size(); i++) {
		printf("%i", entries1[i].xsource);
		if (i != entries1.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");
	tprint("static __constant__ char Lsrc1[%i] = { ", entries1.size());
	for (int i = 0; i < entries1.size(); i++) {
		printf("%i", entries1[i].Lsource);
		if (i != entries1.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");

	tprint("static __constant__ float phi_factor[%i] = { ", phi_entries.size());
	for (int i = 0; i < phi_entries.size(); i++) {
		printf("float(%.9e)", phi_entries[i].factor);
		if (i != phi_entries.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");
	tprint("static __constant__ char phi_Lsrc[%i] = { ", phi_entries.size());
	for (int i = 0; i < phi_entries.size(); i++) {
		printf("%i", phi_entries[i].Lsource);
		if (i != phi_entries.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");

	tprint("#ifdef __CUDACC__\n");
	tprint("template<class T>\n");
	tprint("__device__\n");
	tprint("tensor_trless_sym<T, %i> L2L_cuda(const tensor_trless_sym<T, %i>& La, array<T,NDIM> X, bool do_phi) {\n", Q, P);

	indent();

	tprint("const int tid = threadIdx.x;\n");
	tprint("X[0] *= T(SCALE_FACTOR);\n");
	tprint("X[1] *= T(SCALE_FACTOR);\n");
	tprint("X[2] *= T(SCALE_FACTOR);\n");
	tprint("tensor_trless_sym<T, %i> Lb;\n", Q);
	tprint("tensor_sym<T, %i> Lc;\n", Q);
	tprint("for( int i = 0; i < EXPANSION_SIZE; i ++ ) {\n");
	indent();
	tprint("Lb[i] = 0.0f;\n");
	deindent();
	tprint("}\n");
	tprint("for( int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE ) {\n");
	indent();
	tprint("Lb[i] = La[i];\n");
	deindent();
	tprint("}\n");
	flops += compute_dx_tensor(P);
	flops += const_reference_trless_tensor<P>("La", "Lc");
	flops += 4 * entries.size();
	flops += 3;
	tprint("for( int i = tid; i < %i; i+=WARP_SIZE) {\n", entries1.size());
	indent();
	tprint("Lb[Ldest1[i]] = fmaf(factor1[i] * dx[xsrc1[i]], Lc[Lsrc1[i]], Lb[Ldest1[i]]);\n");
	deindent();
	tprint("}\n");
	tprint("if( do_phi ) {\n");
	indent();
	tprint("for( int i = tid; i < %i; i+=WARP_SIZE) {\n", phi_entries.size());
	indent();
	tprint("Lb[0] = fmaf(phi_factor[i] * dx[phi_Lsrc[i]], Lc[phi_Lsrc[i]], Lb[0]);\n");
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");

	tprint("for (int P = warpSize / 2; P >= 1; P /= 2) {\n");
	indent();
	tprint("for (int i = 0; i < EXPANSION_SIZE; i++) {\n");
	indent();
	tprint("Lb[i] += __shfl_xor_sync(0xffffffff, Lb[i], P);\n");
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");

	tprint("return Lb;\n");
	printf("/* FLOPS = %i + do_phi * %i*/\n", flops, (int) (4 * phi_entries.size()));
	deindent();
	tprint("}\n");
	tprint("#endif\n");
}

void ewald(int direct_flops) {
	tprint("#include <cosmictiger/flops.hpp>\n");
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT int ewald_greens_function(tensor_trless_sym<T,%i> &D, array<T, NDIM> X) {\n", P);
	indent();
	tprint("X[0] *= T(SCALE_FACTOR);\n");
	tprint("X[1] *= T(SCALE_FACTOR);\n");
	tprint("X[2] *= T(SCALE_FACTOR);\n");
	tprint("ewald_const econst;\n");
	tprint("flop_counter<int> flops = %i;\n", 7);
	tprint("T r = sqrt(fmaf(X[0], X[0], fmaf(X[1], X[1], sqr(X[2]))));\n"); // 6
	tprint("const T fouroversqrtpi = T(%.9e);\n", 4.0 / sqrt(M_PI));
	tprint("tensor_sym<T, %i> Dreal;\n", P);
	tprint("tensor_trless_sym<T,%i> Dfour;\n", P);
	tprint("Dreal = 0.0f;\n");
	tprint("Dfour = 0.0f;\n");
	tprint("D = 0.0f;\n");
	tprint("const auto realsz = econst.nreal();\n");
	tprint("const T zero_mask = r > T(0.0);\n");// 1
	int these_flops = 0;

	tprint("int icnt = 0;\n");

	tprint("array<T, NDIM> dx;\n");
	tprint("dx = X;\n");
	tprint("{\n");// 1
	tprint("T r2 = fmaf(dx[0], dx[0], fmaf(dx[1], dx[1], sqr(dx[2])));\n");// 5
	indent();
	tprint("icnt++;\n");
	tprint("const T r = sqrt(r2);\n");// 1
	tprint("const T n8r = T(-8*SCALE_FACTOR_INV2) * r;\n");// 1
	tprint("const T rinv = (r > T(0)) / max(r, 1.0e-20);\n");// 2
	tprint("T exp0 = expnearzero( -T(4*SCALE_FACTOR_INV2) * r2 );\n");
	tprint("T erf0 = erfnearzero(T(2*SCALE_FACTOR_INV1) * r);\n");
	tprint("const T expfactor = T(2.256758334191025*SCALE_FACTOR_INV1) * exp0;\n");// 1
	tprint("T e0 = expfactor * rinv;\n");// 1
	tprint("const T rinv0 = T(1);\n");// 2
	tprint("const T rinv1 = rinv;\n");// 2
	for (int l = 2; l < (P + 1) / 2; l++) {
		const int i = l / 2;
		const int j = l - i;
		these_flops++;
		tprint("const T rinv%i = rinv%i * rinv%i;\n", l, i, j);                      // (P-2)
	}
	tprint("const T d0 = erf0 * rinv;\n");                                       // 2
	for (int l = 1; l < P; l++) {
		tprint("const T d%i = fmaf(T(%i) * d%i, rinv, e0);\n", l, -2 * l + 1, l - 1);            // (P-1)*4
		if (l != P - 1) {
			tprint("e0 *= n8r;\n");                                                // (P-1)
		}
	}
	for (int l = 0; l < P; l++) {
		for (int m = 0; m <= l; m++) {
			if (l + m < P) {
				tprint("const T Drinvpow_%i_%i = d%i * rinv%i;\n", l, m, l, m);
				these_flops++;
			}
		}
	}
	tprint("array<T,NDIM> dxrinv;\n");
	tprint("dxrinv[0] = dx[0] * rinv;\n");
	tprint("dxrinv[1] = dx[1] * rinv;\n");
	tprint("dxrinv[2] = dx[2] * rinv;\n");
	these_flops += compute_dx(P, "dxrinv", false, true);
	these_flops += 37 + 7 * (P - 1) + P * (P + 1) / 2;

	these_flops += compute_detrace_ewald<P>("x", "Dreal");
	tprint("const auto Drz = econst.D0();\n");
	tprint("for( int i = 0; i < %i; i++) {\n", (P + 2) * (P + 1) * P / 6);
	indent();
	tprint("Dreal[i] *= zero_mask;\n");
	tprint("Dreal[i] -= (T(1) - zero_mask) * Drz[i];\n");
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");
	tprint("flops += icnt * %i;\n", these_flops);

	tprint("for (int i = 0; i < realsz; i++) {\n");
	indent();
	tprint("const auto n = econst.real_index(i);\n");
	tprint("array<T, NDIM> dx;\n");
	tprint("for (int dim = 0; dim < NDIM; dim++) {\n");
	indent();
	tprint("dx[dim] = X[dim] - T(SCALE_FACTOR) * n[dim];\n");                                // 3
	deindent();
	tprint("}\n");
	tprint("T r2 = fmaf(dx[0], dx[0], fmaf(dx[1], dx[1], sqr(dx[2])));\n");// 5
	tprint("if (anytrue(r2 < T(SCALE_FACTOR2*EWALD_REAL_CUTOFF2))) {\n");// 1
	indent();
	tprint("icnt++;\n");// 1
	tprint("const T r = sqrt(r2);\n");// 1
	tprint("const T n8r = T(-8 * SCALE_FACTOR_INV2) * r;\n");// 1
	tprint("const T rinv = (r > T(0)) / max(r, 1.0e-20);\n");// 2
	tprint("T exp0;\n");
	tprint("T erfc0;\n");
	tprint("erfcexp(T(2.*SCALE_FACTOR_INV1) * r, &erfc0, &exp0);\n");// 20
	tprint("const T expfactor = fouroversqrtpi  * T(SCALE_FACTOR_INV1) * exp0;\n");// 1
	tprint("T e0 = expfactor * rinv;\n");// 1
	tprint("const T rinv0 = T(1);\n");// 2
	tprint("const T rinv1 = rinv;\n");// 2
	for (int l = 2; l < (P + 1) / 2; l++) {
		const int i = l / 2;
		const int j = l - i;
		these_flops++;
		tprint("const T rinv%i = rinv%i * rinv%i;\n", l, i, j);                      // (P-2)
	}
	tprint("const T d0 = -erfc0 * rinv;\n");                                       // 2
	for (int l = 1; l < P; l++) {
		tprint("const T d%i = fmaf(T(%i) * d%i, rinv, e0);\n", l, -2 * l + 1, l - 1);            // (P-1)*4
		if (l != P - 1) {
			tprint("e0 *= n8r;\n");                                                // (P-1)
		}
	}
	for (int l = 0; l < P; l++) {
		for (int m = 0; m <= l; m++) {
			if (l + m < P) {
				tprint("const T Drinvpow_%i_%i = d%i * rinv%i;\n", l, m, l, m);
				these_flops++;
			}
		}
	}
	tprint("array<T,NDIM> dxrinv;\n");
	tprint("dxrinv[0] = dx[0] * rinv;\n");
	tprint("dxrinv[1] = dx[1] * rinv;\n");
	tprint("dxrinv[2] = dx[2] * rinv;\n");
	these_flops += compute_dx(P, "dxrinv");
	array<int, NDIM> m;
	array<int, NDIM> k;
	array<int, NDIM> n;
	these_flops += 37 + 7 * (P - 1) + P * (P + 1) / 2;

	these_flops += compute_detrace_ewald<P>("x", "Dreal");

	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");

	tprint("const auto foursz = econst.nfour();\n");
	tprint("for (int i = 0; i < foursz; i++) {\n");
	indent();
	tprint("const auto &h = econst.four_index(i);\n");
	tprint("const auto& D0 = econst.four_expansion(i);\n");
	tprint("const T hdotx = fmaf(h[0], X[0], fmaf(h[1], X[1], h[2] * X[2]));\n"); // 5
	tprint("T cn, sn;\n");
	tprint("sincos(T(2.0 * M_PI * SCALE_FACTOR_INV1) * hdotx, &sn, &cn);\n");// 35
	these_flops = 40;
	bool iscos[P * P + 1];
	for (k[0] = 0; k[0] < P; k[0]++) {
		for (k[1] = 0; k[1] < P - k[0]; k[1]++) {
			const int zmax = (k[0] == 0 && k[1] == 0) ? intmin(3, P) : intmin(P - k[0] - k[1], 2);
			for (k[2] = 0; k[2] < zmax; k[2]++) {
				const int k0 = k[0] + k[1] + k[2];
				iscos[trless_index(k[0], k[1], k[2], P)] = (k0 % 2 == 0);
			}
		}
	}
	int maxi = (P * P + 1);
	for (int i = 0; i < maxi; i++) {
		int j = i + maxi;
		tprint("Dfour[%i] = fmaf(%cn, D0[%i], Dfour[%i]);\n", i, iscos[i] ? 'c' : 's', i, i);
		these_flops += 2;

	}

	deindent();
	tprint("}\n");
//	reference_sym("Dreal", P);
//	reference_trless("D", P);
	int those_flops = compute_detrace<P>("Dreal", "D", 'd');
	those_flops += 16 + 3 * (P * P + 1);
	tprint("flops += %i * foursz + %i;\n", these_flops, those_flops + P * P + 1);
	tprint("D = D + Dfour;\n");// P*P+1
	tprint("D[0] = T(%.9e * SCALE_FACTOR_INV1) + D[0]; \n", M_PI / 4.0);// 1
	tprint("return flops;\n");
	deindent();
	tprint("}\n");

}

int main() {

	fprintf(stderr, "Generating FMM kernels for Pmax = %i\n", P - 1);

	int flops = 0;

	tprint("/*\n"
			"CosmicTiger - A cosmological N-Body code\n"
			"Copyright (C) 2021  Dominic C. Marcello\n"
			"\n"
			"This program is free software; you can redistribute it and/or\n"
			"modify it under the terms of the GNU General Public License\n"
			"as published by the Free Software Foundation; either version 2\n"
			"of the License, or (at your option) any later version.\n"
			"\n"
			"This program is distributed in the hope that it will be useful,\n"
			"but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
			"MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
			"GNU General Public License for more details.\n"
			"\n"
			"You should have received a copy of the GNU General Public License\n"
			"along with this program; if not, write to the Free Software\n"
			"Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.\n"
			"*/\n");

	tprint("#pragma once\n");
	tprint("#include <cosmictiger/tensor.hpp>\n");
	tprint("#include <cosmictiger/cuda.hpp>\n");
	tprint("#include <cosmictiger/ewald_indices.hpp>\n");
	tprint("#include <cosmictiger/math.hpp>\n");
	tprint("#include <cosmictiger/float2double.hpp>\n");
	tprint("template<class T>\n");
	tprint("using expansion = tensor_trless_sym<T,%i>;\n", P);
	tprint("template<class T>\n");
	tprint("using expansion2 = tensor_trless_sym<T,%i>;\n", 2);
	tprint("template<class T>\n");
	tprint("using multipole = tensor_trless_sym<T,%i>;\n", P - 1);
	tprint("#define EXPANSION_SIZE %i\n", P * P + 1);
	tprint("#define MULTIPOLE_SIZE %i\n", (P - 1) * (P - 1) + 1);
	const auto maxval = 1000.0;
	tprint("#define SCALE_FACTOR %ef\n", maxval);
	for (int n = 0; n <= P; n++) {
		tprint("#define SCALE_FACTOR%i %ef\n", n, pow(maxval, n));
		tprint("#define SCALE_FACTOR_INV%i %ef\n", n, 1.0 / pow(maxval, n));
	}

	tprint("\n\ntemplate<class T>\n");
	tprint("CUDA_EXPORT\n");
	tprint("inline int greens_function(tensor_trless_sym<T, %i>& D, array<T, NDIM> X, bool scale = true) {\n", P);
	flops = 0;
	indent();
	tprint("if( scale ) {\n");
	indent();
	tprint("X[0] *= T(SCALE_FACTOR);\n");
	tprint("X[1] *= T(SCALE_FACTOR);\n");
	tprint("X[2] *= T(SCALE_FACTOR);\n");
	deindent();
	tprint("}\n");
	tprint("auto r2 = sqr(X[0], X[1], X[2]);\n");
	tprint("r2 = sqr(X[0], X[1], X[2]);\n");
	tprint("const T r = sqrt(r2);\n");
	tprint("const T rinv1 = -(r > T(0)) / max(r, T(1e-20));\n");
	for (int i = 1; i < P; i++) {
		const int j = i / 2;
		const int k = i - j;
		tprint("const T rinv%i = -rinv%i * rinv%i;\n", i + 1, j + 1, k);
	}
	tprint("X[0] *= rinv1;\n");
	tprint("X[1] *= rinv1;\n");
	tprint("X[2] *= rinv1;\n");
	flops += 12;
	flops += compute_dx(P, "X", true);
//	reference_trless("D", P);
	flops += compute_detraceD<P>("x", "D");
	flops += 11 + (P - 1) * 2;
	array<int, NDIM> k;
	for (k[0] = 0; k[0] < P; k[0]++) {
		for (k[1] = 0; k[1] < P - k[0]; k[1]++) {
			const int zmax = (k[0] == 0 && k[1] == 0) ? intmin(3, P) : intmin(P - k[0] - k[1], 2);
			for (k[2] = 0; k[2] < zmax; k[2]++) {
				const int k0 = k[0] + k[1] + k[2];
				tprint("D[%i] *= rinv%i;\n", trless_index(k[0], k[1], k[2], P), k0 + 1);
				flops++;
			}
		}
	}
	tprint("return %i + scale * NDIM;\n", flops);
	deindent();
	tprint("}\n");

	ewald(flops);

	array<int, NDIM> n;
	array<int, NDIM> m;
	for (int Pmax = 2; Pmax <= P; Pmax += P - 2) {
		flops = 0;
		tprint("\n\ntemplate<class T>\n");
		tprint("CUDA_EXPORT\n");
		tprint("inline int M2L(tensor_trless_sym<T, %i>& L, const tensor_trless_sym<T, %i>& M, const tensor_trless_sym<T, %i>& D, bool do_phi) {\n", Pmax,
				P - 1, P);
		indent();
		int phi_flops = 0;
		flops += const_reference_trless<P - 1>("M");
		flops += const_reference_trless<P>("D");
		n[0] = n[1] = n[2] = 0;
		const int n0 = 0;
		const int q0 = intmin(P - n0, P - 1);
		struct entry_type {
			array<int, NDIM> n;
			array<int, NDIM> m;
			double coeff;
		};
		vector<vector<entry_type>> entries;
		entries.resize(Pmax <= 2 ? Pmax * Pmax : Pmax * Pmax + 1);
		for (n[0] = 0; n[0] < Pmax; n[0]++) {
			for (n[1] = 0; n[1] < Pmax - n[0]; n[1]++) {
				const int nzmax = (n[0] == 0 && n[1] == 0) ? intmin(3, Pmax) : intmin(Pmax - n[0] - n[1], 2);
				for (n[2] = 0; n[2] < nzmax; n[2]++) {
					const int n0 = n[0] + n[1] + n[2];
					const int q0 = intmin(P - n0, P - 1);
					for (m[0] = 0; m[0] < q0; m[0]++) {
						for (m[1] = 0; m[1] < q0 - m[0]; m[1]++) {
							for (m[2] = 0; m[2] < q0 - m[0] - m[1]; m[2]++) {
								const double coeff = 1.0 / vfactorial(m);
								entry_type e;
								e.coeff = coeff;
								e.n = n;
								e.m = m;
								const int index = trless_index(n[0], n[1], n[2], Pmax);
								entries[index].push_back(e);
							}
						}
					}
				}
			}
		}
		const auto sort_func = [Pmax](entry_type a, entry_type b) {
			if( a.coeff < b.coeff ) {
				return true;
			} else {
				return false;
			}
		};
		for (auto& e : entries) {
			std::sort(e.begin(), e.end(), sort_func);
		}
		for (int i = 0; i < 1; i++) {
			double last_coeff = 1.0;
			const auto n = entries[i][0].n;
			const int nindex = trless_index(n[0], n[1], n[2], Pmax);
			auto& fl = phi_flops;
			tprint("if( do_phi ) {\n");
			indent();
			for (int j = 0; j < entries[i].size(); j++) {
				const auto coeff = entries[i][j].coeff;
				const auto m = entries[i][j].m;
				if (!close21(last_coeff / coeff)) {
					double factor = last_coeff / coeff;
					tprint("L[%i] *= T(%.9e);\n", nindex, factor);
					last_coeff = coeff;
					fl++;
				}
				tprint("L[%i] = fmaf(M%i%i%i, D%i%i%i, L[%i]);\n", nindex, m[0], m[1], m[2], n[0] + m[0], n[1] + m[1], n[2] + m[2],
						trless_index(n[0], n[1], n[2], Pmax));
				fl += 2;
			}
			if (!close21(last_coeff)) {
				fl++;
				tprint("L[%i] *= T(%.9e);\n", nindex, last_coeff);
			}
			if (nindex == 0) {
				deindent();
				tprint("}\n");
			}
		}
		int total_size = 0;
		for (int i = 1; i < entries.size(); i++) {
			total_size += entries[i].size();
		}
		int half_size = 0;
		int mid;
		for (int i = 1; i < entries.size(); i++) {
			half_size += entries[i].size();
			if (half_size >= total_size / 2) {
				mid = i;
				break;
			}
		}
		vector<std::string> cmds1, cmds2;
		for (int i = 1; i < entries.size(); i++) {
			double last_coeff = 1.0;
			const auto n = entries[i][0].n;
			const int nindex = trless_index(n[0], n[1], n[2], Pmax);
			auto& fl = flops;
			auto& cmds = i >= mid ? cmds2 : cmds1;
			char* str;
			for (int j = 0; j < entries[i].size(); j++) {
				const auto coeff = entries[i][j].coeff;
				const auto m = entries[i][j].m;
				if (!close21(last_coeff / coeff)) {
					double factor = last_coeff / coeff;
					ASPRINTF(&str, "L[%i] *= T(%.9e);\n", nindex, factor);
					cmds.push_back(str);
					free(str);
					last_coeff = coeff;
					fl++;
				}
				ASPRINTF(&str, "L[%i] = fmaf(M%i%i%i, D%i%i%i, L[%i]);\n", nindex, m[0], m[1], m[2], n[0] + m[0], n[1] + m[1], n[2] + m[2],
						trless_index(n[0], n[1], n[2], Pmax));
				cmds.push_back(str);
				free(str);
				fl += 2;
			}
			if (!close21(last_coeff)) {
				fl++;
				ASPRINTF(&str, "L[%i] *= T(%.9e);\n", nindex, last_coeff);
				cmds.push_back(str);
			}
		}
		int i = 0;
		int j = 0;
		while (i < cmds1.size() || j < cmds2.size()) {
			if (i < cmds1.size()) {
				tprint("%s", cmds1[i].c_str());
				i++;
			}
			if (j < cmds2.size()) {
				tprint("%s", cmds2[j].c_str());
				j++;
			}
		}
		tprint("return %i + do_phi * %i;\n", flops, phi_flops);
		deindent();
		tprint("}\n");
	}

	tprint("\n\ntemplate<class T>\n");
	tprint("CUDA_EXPORT\n");
	tprint("tensor_trless_sym<T, %i> P2M(array<T, NDIM>& X) {\n", P - 1);
	flops = 0;
	indent();
	tprint("tensor_trless_sym<T, %i> M;\n", P - 1);
	tprint("X[0] *= -T(SCALE_FACTOR);\n");
	tprint("X[1] *= -T(SCALE_FACTOR);\n");
	tprint("X[2] *= -T(SCALE_FACTOR);\n");
//	reference_trless("M", P - 1);
	flops += 6;
	flops += compute_dx(P - 1);
	flops += compute_detrace<P - 1>("x", "M", 'd');
	tprint("return M;\n");
	printf("/* FLOPS = %i*/\n", flops);
	deindent();
	tprint("}\n");

	tprint("\n\ntemplate<class T>\n");
	tprint("CUDA_EXPORT\n");
	tprint("tensor_trless_sym<T, %i> M2M(const tensor_trless_sym<T,%i>& Ma, array<T, NDIM>& X) {\n",
			P - 1, P - 1);
	flops = 0;
	indent();
	tprint("tensor_sym<T, %i> Mb;\n", P - 1);
	tprint("tensor_trless_sym<T, %i> Mc;\n", P - 1);
	tprint("X[0] *= -T(SCALE_FACTOR);\n");
	tprint("X[1] *= -T(SCALE_FACTOR);\n");
	tprint("X[2] *= -T(SCALE_FACTOR);\n");
	flops += 6;
	flops += const_reference_trless<P - 1>("Ma");
//	reference_sym("Mb", P - 1);
//	reference_trless("Mc", P - 1);
	flops += compute_dx(P - 1);

	for (int i = 0; i < (P - 1) * P * (P + 1) / 6; i++) {
		for (n[0] = 0; n[0] < P - 1; n[0]++) {
			for (n[1] = 0; n[1] < P - n[0] - 1; n[1]++) {
				for (n[2] = 0; n[2] < P - n[0] - n[1] - 1; n[2]++) {
					if (i == sym_index(n[0], n[1], n[2])) {
						tprint("Mb[%i] = Ma%i%i%i;\n", i, n[0], n[1], n[2]);
					}
				}
			}
		}
	}
	struct mentry {
		array<int, NDIM> n;
		array<int, NDIM> k;
		double factor;
	};
	vector<vector<mentry>> mentries((P - 1) * P * (P + 1) / 6);
	for (int n0 = 0; n0 <= P - 2; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[0] - n[1];
				for (k[0] = 0; k[0] <= intmin(n0, n[0]); k[0]++) {
					for (k[1] = 0; k[1] <= intmin(n0 - k[0], n[1]); k[1]++) {
						for (k[2] = 0; k[2] <= intmin(n0 - k[0] - k[1], n[2]); k[2]++) {
							const auto factor = (vfactorial(n)) / double(vfactorial(k) * vfactorial(n - k));
							if (n != k) {
								mentry e;
								e.n = n;
								e.k = k;
								e.factor = factor;
								const int index = sym_index(n[0], n[1], n[2]);
								mentries[index].push_back(e);
							}
						}
					}
				}
			}
		}
	}
	for (auto& m : mentries) {
		std::sort(m.begin(), m.end(), [](mentry a, mentry b) {
					return a.factor > b.factor;
				});
	}
	int total_size = 0;
	for (int i = 0; i < mentries.size(); i++) {
		total_size += mentries[i].size();
	}
	int half_size = 0;
	int mid;
	for (int i = 0; i < mentries.size(); i++) {
		if (half_size >= total_size / 2) {
			mid = i;
			break;
		}
		half_size += mentries[i].size();
	}
	vector<std::string> cmds1, cmds2;
	char* str;
	for (int i = 0; i < mentries.size(); i++) {
		auto& cmds = i < mid ? cmds1 : cmds2;
		double last_factor = 1.0;
		if (mentries[i].size()) {
			const auto n = mentries[i][0].n;
			const int nindex = sym_index(n[0], n[1], n[2]);
			for (int j = 0; j < mentries[i].size(); j++) {
				const auto k = mentries[i][j].k;
				const auto factor = mentries[i][j].factor;
				if (!close21(last_factor / factor)) {
					ASPRINTF(&str, "Mb[%i] *= T(%.9e);\n", nindex, last_factor / factor);
					cmds.push_back(str);
					flops++;
					free(str);
					last_factor = factor;
				}
				ASPRINTF(&str, "Mb[%i] = fmaf( x[%i], Ma%i%i%i, Mb[%i]);\n", nindex, sym_index(n[0] - k[0], n[1] - k[1], n[2] - k[2]), k[0], k[1], k[2],
						sym_index(n[0], n[1], n[2]));
				cmds.push_back(str);
				free(str);
				flops += 2;
			}
			if (!close21(last_factor)) {
				ASPRINTF(&str, "Mb[%i] *= T(%.9e);\n", nindex, last_factor);
				cmds.push_back(str);
				flops++;
				free(str);
			}
		}
	}
	int i = 0;
	int j = 0;
	while (i < cmds1.size()) {
		tprint("%s", cmds1[i].c_str());
		i++;
	}
	while (j < cmds2.size()) {
		tprint("%s", cmds2[j].c_str());
		j++;
	}

	flops += compute_detrace<P - 1>("Mb", "Mc", 'd');

	tprint("return Mc;\n");
	printf("/* FLOPS = %i*/\n", flops);
	deindent();
	tprint("}\n");

	do_expansion<P>(false);

	do_expansion<2>(true);

#ifdef USE_CUDA
	do_expansion_cuda<P>();
#endif

	tprint("template<class T>\n");
	tprint("CUDA_EXPORT int apply_scale_factor_inv(tensor_trless_sym<T,%i> &L) {\n", P);
	indent();
	apply_scale_factorL<P>("L");
	tprint("return %i;\n", P * P + 1);
	deindent();
	tprint("}\n");

	tprint("template<class T>\n");
	tprint("CUDA_EXPORT int apply_scale_factor(tensor_trless_sym<T,%i> &M) {\n", P - 1);
	indent();
	apply_scale_factorM<P - 1>("M");
	tprint("return %i;\n", (P - 1) * (P - 1) + 1);

	deindent();
	tprint("}\n");

}

#else

#define P ORDER
template<int Q>
int compute_dx(const char* name = "X") {
	array<int, NDIM> n;
	tprint("%s x[%i];\n", "T", (Q + 1) * Q * (Q + 2) / 6);
	auto index = std::function<int(int, int, int)>(sym_index);
	tprint("x[%i] = %s(1);\n", index(0, 0, 0), "T");
	tprint("x[%i] = %s[0];\n", index(1, 0, 0), name);
	tprint("x[%i] = %s[1];\n", index(0, 1, 0), name);
	tprint("x[%i] = %s[2];\n", index(0, 0, 1), name);
	int flops = 0;

	for (int n0 = 2; n0 < Q; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[0] - n[1];
				array<int, NDIM> j = n;
				int j0 = n0;
				const int jmin = std::max(1, n0 / 2);
				while (j0 > jmin) {
					for (int dim = 0; dim < NDIM && j0 > jmin; dim++) {
						if (j[dim] > 0) {
							j[dim]--;
							j0--;
						}
					}
				}
				array<int, NDIM> k;
				k = n - j;
				tprint("x[%i] = x[%i] * x[%i];\n", index(n[0], n[1], n[2]), index(k[0], k[1], k[2]), index(j[0], j[1], j[2]));
				flops++;
			}
		}
	}
	return flops;
}

int compute_detrace_ewald(std::string iname, std::string oname) {

	array<int, NDIM> m;
	array<int, NDIM> k;
	array<int, NDIM> n;

	struct entry_type {
		array<int, NDIM> src;
		array<int, NDIM> dest;
		double factor;
	};
	vector<entry_type> entries;
	int flops = 0;

	vector<std::string> asn;
	vector<std::string> op;
	op.resize(0);
	asn.resize(0);
	for (n[0] = 0; n[0] < P; n[0]++) {
		for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
			for (n[2] = 0; n[2] < P - n[0] - n[1]; n[2]++) {
				const int n0 = n[0] + n[1] + n[2];
				bool first = true;
				for (m[0] = 0; m[0] <= n[0] / 2; m[0]++) {
					for (m[1] = 0; m[1] <= n[1] / 2; m[1]++) {
						for (m[2] = 0; m[2] <= n[2] / 2; m[2]++) {
							const int m0 = m[0] + m[1] + m[2];
							double num = double(vfactorial(n));
							double den = double((1 << m0) * vfactorial(m) * vfactorial(n - (m) * 2));
							double factor = num / den;
							const auto p = n - m * 2;
							char* str;
							if (close21(factor)) {
								ASPRINTF(&str, "%s[%i] = fmaf(%s[%i], Drinvpow_%i_%i, %s[%i]);\n", oname.c_str(), sym_index(n[0], n[1], n[2]), iname.c_str(),
										sym_index(p[0], p[1], p[2]), n0 - m0, m0, oname.c_str(), sym_index(n[0], n[1], n[2]));
								flops += 2;
							} else if (close21(-factor)) {
								ASPRINTF(&str, "%s[%i] -= %s[%i] * Drinvpow_%i_%i;\n", oname.c_str(), sym_index(n[0], n[1], n[2]), iname.c_str(),
										sym_index(p[0], p[1], p[2]), n0 - m0, m0);
								flops += 2;
							} else {
								ASPRINTF(&str, "%s[%i] = fmaf(T(%.9e), %s[%i]*Drinvpow_%i_%i, %s[%i]);\n", oname.c_str(), sym_index(n[0], n[1], n[2]), factor,
										iname.c_str(), sym_index(p[0], p[1], p[2]), n0 - m0, m0, oname.c_str(), sym_index(n[0], n[1], n[2]));

								flops += 3;
							}
							op.push_back(str);
							free(str);
						}
					}
				}
			}
		}
	}
//	std::sort(op.begin(), op.end(), [](std::string a, std::string b) {
//				return atoi(a.c_str()+6) < atoi(b.c_str()+6);
//			});
	int maxop = op.size();
	for (int i = 0; i < maxop; i++) {
		tprint("%s", op[i].c_str());
	}

	return flops;
}

int compute_dx_tensor(const char* name = "X") {
	array<int, NDIM> n;
	tprint("tensor_sym<T,%i> dx;\n", P);
	tprint("dx[0] = T(1);\n");
	tprint("dx[%i] = %s[0];\n", sym_index(1, 0, 0), name);
	tprint("dx[%i] = %s[1];\n", sym_index(0, 1, 0), name);
	tprint("dx[%i] = %s[2];\n", sym_index(0, 0, 1), name);
	int flops = 0;
	for (int n0 = 2; n0 < P; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[0] - n[1];
				array<int, NDIM> j = n;
				int j0 = n0;
				const int jmin = std::max(1, n0 / 2);
				while (j0 > jmin) {
					for (int dim = 0; dim < NDIM && j0 > jmin; dim++) {
						if (j[dim] > 0) {
							j[dim]--;
							j0--;
						}
					}
				}
				array<int, NDIM> k;
				k = n - j;
				tprint("dx[%i]= dx[%i] * dx[%i];\n", sym_index(n[0], n[1], n[2]), sym_index(k[0], k[1], k[2]), sym_index(j[0], j[1], j[2]));
				flops++;
			}
		}
	}
	return flops;
}

template<int Q>
void do_expansion_cuda() {
	int flops = 0;
	struct entry {
		int Lsource;
		int Ldest;
		int xsource;
		float factor;
	};
	array<int, NDIM> n;
	array<int, NDIM> k;
	vector<entry> entries;
	vector<entry> phi_entries;
	for (int n0 = 0; n0 < Q; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[1] - n[0];
				const int n0 = n[0] + n[1] + n[2];
				for (k[0] = 0; k[0] < P - n0; k[0]++) {
					for (k[1] = 0; k[1] < P - n0 - k[0]; k[1]++) {
						for (k[2] = 0; k[2] < P - n0 - k[0] - k[1]; k[2]++) {
							const auto factor = double(1) / double(vfactorial(k));
							const auto p = n + k;
							const int p0 = p[0] + p[1] + p[2];
							if (n != p) {
								entry e;
								e.Ldest = sym_index(n[0], n[1], n[2]);
								e.xsource = sym_index(k[0], k[1], k[2]);
								e.Lsource = sym_index(p[0], p[1], p[2]);
								e.factor = factor;
								if (n0 == 0) {
									phi_entries.push_back(e);
								} else {
									entries.push_back(e);
								}
							}
						}
					}
				}
			}
		}
	}
	std::sort(entries.begin(), entries.end(), [](entry a, entry b) {
		if( a.Ldest < b.Ldest) {
			return true;
		} else if( a.Ldest > b.Ldest) {
			return false;
		} else {
			return a.Lsource < b.Lsource;
		}
	});
	vector<entry> entries1, entries2;
	for (int i = 0; i < entries.size(); i++) {
		entries1.push_back(entries[i]);
	}

	tprint("static __device__ char Ldest1[%i] = { ", entries1.size());
	for (int i = 0; i < entries1.size(); i++) {
		printf("%i", entries1[i].Ldest);
		if (i != entries1.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");
	tprint("static __constant__ float factor1[%i] = { ", entries1.size());
	for (int i = 0; i < entries1.size(); i++) {
		printf("float(%.9e)", entries1[i].factor);
		if (i != entries1.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");
	tprint("static __constant__ char xsrc1[%i] = { ", entries1.size());
	for (int i = 0; i < entries1.size(); i++) {
		printf("%i", entries1[i].xsource);
		if (i != entries1.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");
	tprint("static __constant__ char Lsrc1[%i] = { ", entries1.size());
	for (int i = 0; i < entries1.size(); i++) {
		printf("%i", entries1[i].Lsource);
		if (i != entries1.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");

	tprint("static __constant__ float phi_factor[%i] = { ", phi_entries.size());
	for (int i = 0; i < phi_entries.size(); i++) {
		printf("float(%.9e)", phi_entries[i].factor);
		if (i != phi_entries.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");
	tprint("static __constant__ char phi_Lsrc[%i] = { ", phi_entries.size());
	for (int i = 0; i < phi_entries.size(); i++) {
		printf("%i", phi_entries[i].Lsource);
		if (i != phi_entries.size() - 1) {
			printf(",");
		}
	}
	tprint("};\n");

	tprint("#ifdef __CUDACC__\n");
	tprint("template<class T>\n");
	tprint("__device__\n");
	tprint("tensor_sym<T, %i> L2L_cuda(const tensor_sym<T, %i>& La, array<T,NDIM> X, bool do_phi) {\n", Q, P);

	indent();

	tprint("const int tid = threadIdx.x;\n");
	/*	tprint("X[0] *= T(SCALE_FACTOR);\n");
	 tprint("X[1] *= T(SCALE_FACTOR);\n");
	 tprint("X[2] *= T(SCALE_FACTOR);\n");*/
	tprint("tensor_sym<T, %i> Lb;\n", Q);
	tprint("for( int i = 0; i < EXPANSION_SIZE; i ++ ) {\n");
	indent();
	tprint("Lb[i] = 0.0f;\n");
	deindent();
	tprint("}\n");
	tprint("for( int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE ) {\n");
	indent();
	tprint("Lb[i] = La[i];\n");
	deindent();
	tprint("}\n");
	flops += compute_dx_tensor();
	flops += 3;
	tprint("for( int i = tid; i < %i; i+=WARP_SIZE) {\n", entries1.size()));
	indent();
	tprint("Lb[Ldest1[i]] = fmaf(factor1[i] * dx[xsrc1[i]], La[Lsrc1[i]], Lb[Ldest1[i]]);\n");
	deindent();
	tprint("}\n");
	tprint("if( do_phi ) {\n");
	indent();
	tprint("for( int i = tid; i < %i; i+=WARP_SIZE) {\n", phi_entries.size());
	indent();
	tprint("Lb[0] = fmaf(phi_factor[i] * dx[phi_Lsrc[i]], La[phi_Lsrc[i]], Lb[0]);\n");
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");

	tprint("for (int P = warpSize / 2; P >= 1; P /= 2) {\n");
	indent();
	tprint("for (int i = 0; i < EXPANSION_SIZE; i++) {\n");
	indent();
	tprint("Lb[i] += __shfl_xor_sync(0xffffffff, Lb[i], P);\n");
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");

	tprint("return Lb;\n");
	printf("/* FLOPS = %i + do_phi * %i*/\n", flops, (int) (4 * phi_entries.size()));
	deindent();
	tprint("}\n");
	tprint("#endif\n");
}

template<int Q>
void do_expansion(bool two) {
	int flops = 0;
	tprint("\n");
	tprint("template<class T>\n");
	tprint("CUDA_EXPORT\n");
	tprint("#ifdef __CUDACC__\n");
	tprint("__noinline__\n");
	tprint("#endif\n");
	if (two) {
		tprint("tensor_sym<T, %i> L2P(const tensor_sym<T, %i>& La, array<T,NDIM> X, bool do_phi) {\n", Q, P);
	} else {
		tprint("tensor_sym<T, %i> L2L(const tensor_sym<T, %i>& La, array<T,NDIM> X, bool do_phi) {\n", Q, P);

	}
	indent();
	/*tprint("X[0] *= T(SCALE_FACTOR);\n");
	 tprint("X[1] *= T(SCALE_FACTOR);\n");
	 tprint("X[2] *= T(SCALE_FACTOR);\n");*/
	tprint("tensor_sym<T, %i> Lb;\n", Q);
	flops += 3 + compute_dx<P>();
	array<int, NDIM> n;
	array<int, NDIM> k;
	int phi_flops = 0;
	if (two) {
		tprint("Lb[%i] = La[%i];\n", sym_index(0, 0, 0), sym_index(0, 0, 0));
		tprint("Lb[%i] = La[%i];\n", sym_index(1, 0, 0), sym_index(1, 0, 0));
		tprint("Lb[%i] = La[%i];\n", sym_index(0, 1, 0), sym_index(0, 1, 0));
		tprint("Lb[%i] = La[%i];\n", sym_index(0, 0, 1), sym_index(0, 0, 1));
	} else {
		tprint("Lb = La;\n");
	}
	struct entry {
		int Ldest;
		array<int, NDIM> p;
		array<int, NDIM> k;
		double factor;
	};
	vector<vector<entry>> entries(two ? 4 : (Q + 2) * (Q + 1) * Q / 6);
	for (int n0 = 0; n0 < Q; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[1] - n[0];
				const int n0 = n[0] + n[1] + n[2];
				for (k[0] = 0; k[0] < P - n0; k[0]++) {
					for (k[1] = 0; k[1] < P - n0 - k[0]; k[1]++) {
						for (k[2] = 0; k[2] < P - n0 - k[0] - k[1]; k[2]++) {
							const auto factor = double(1) / double(vfactorial(k));
							const auto p = n + k;
							const int p0 = p[0] + p[1] + p[2];
							if (n != p) {
								entry e;
								e.Ldest = sym_index(n[0], n[1], n[2]);
								e.factor = factor;
								e.k = k;
								e.p = p;
								entries[e.Ldest].push_back(e);
							}
						}
					}
				}
			}
		}
	}
	for (auto& e : entries) {
		std::sort(e.begin(), e.end(), [](entry a, entry b) {
			return( a.factor < b.factor );
		});
	}
	double last_factor = 1.0;
	tprint("if( do_phi ) {\n");
	indent();
	for (int j = 0; j < entries[0].size(); j++) {
		const auto factor = entries[0][j].factor;
		const auto p = entries[0][j].p;
		const auto k = entries[0][j].k;
		const auto index = entries[0][j].Ldest;
		if (!close21(last_factor / factor)) {
			tprint("Lb[0] *= T(%.9e);\n", last_factor / factor);
			last_factor = factor;
			phi_flops++;
		}
		tprint("Lb[%i] = fmaf( x[%i], La[%i], Lb[%i]);\n", index, sym_index(k[0], k[1], k[2]), sym_index(p[0], p[1], p[2]), index);
		phi_flops += 2;
	}
	if (!close21(last_factor)) {
		tprint("Lb[0] *= T(%.9e);\n", last_factor);
	}
	deindent();
	phi_flops++;
	tprint("}\n");
	int total_size = 0;
	for (int i = 1; i < entries.size(); i++) {
		total_size += entries[i].size();
	}
	int mid;
	int half_size = 0;
	for (int i = 1; i < entries.size(); i++) {
		if (half_size >= total_size / 2) {
			mid = i;
			break;
		}
		half_size += entries[i].size();
	}
	char* str;
	vector<std::string> cmds1, cmds2;
	for (int i = 1; i < entries.size(); i++) {
		last_factor = 1.0;
		auto& cmds = i >= mid ? cmds2 : cmds1;
		if (entries[i].size()) {
			const auto index = entries[i][0].Ldest;
			for (int j = 0; j < entries[i].size(); j++) {
				const auto factor = entries[i][j].factor;
				const auto p = entries[i][j].p;
				const auto k = entries[i][j].k;
				if (!close21(factor / last_factor)) {
					ASPRINTF(&str, "Lb[%i] *= T(%.9e);\n", index, last_factor / factor);
					cmds.push_back(str);
					free(str);
					flops++;
					last_factor = factor;
				}
				ASPRINTF(&str, "Lb[%i] = fmaf( x[%i], La[%i], Lb[%i]);\n", index, sym_index(k[0], k[1], k[2]), sym_index(p[0], p[1], p[2]), index);
				cmds.push_back(str);
				free(str);
				flops += 2;
			}
			if (!close21(last_factor)) {
				ASPRINTF(&str, "Lb[%i] *= T(%.9e);\n", index, last_factor);
				cmds.push_back(str);
				free(str);
			}
		}
	}

	int i = 0;
	int j = 0;
	while (i < cmds1.size()) {
		tprint("%s", cmds1[i].c_str());
		i++;
	}
	while (j < cmds2.size()) {
		tprint("%s", cmds2[j].c_str());
		j++;
	}
	tprint("return Lb;\n");
	printf("/* FLOPS = %i + do_phi * %i*/\n", flops, phi_flops);
	deindent();
	tprint("}\n");
}

int main() {

	fprintf(stderr, "Generating FMM kernels for Pmax = %i\n", P - 1);

	int flops = 0;

	tprint("/*\n"
			"CosmicTiger - A cosmological N-Body code\n"
			"Copyright (C) 2021  Dominic C. Marcello\n"
			"\n"
			"This program is free software; you can redistribute it and/or\n"
			"modify it under the terms of the GNU General Public License\n"
			"as published by the Free Software Foundation; either version 2\n"
			"of the License, or (at your option) any later version.\n"
			"\n"
			"This program is distributed in the hope that it will be useful,\n"
			"but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
			"MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
			"GNU General Public License for more details.\n"
			"\n"
			"You should have received a copy of the GNU General Public License\n"
			"along with this program; if not, write to the Free Software\n"
			"Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.\n"
			"*/\n");

	tprint("#pragma once\n");
	tprint("#include <cosmictiger/tensor.hpp>\n");
	tprint("#include <cosmictiger/cuda.hpp>\n");
	tprint("#include <cosmictiger/ewald_indices.hpp>\n");
	tprint("#include <cosmictiger/math.hpp>\n");
	tprint("#include <cosmictiger/float2double.hpp>\n");
	tprint("template<class T>\n");
	tprint("using expansion = tensor_sym<T,%i>;\n", P);
	tprint("template<class T>\n");
	tprint("using expansion2 = tensor_sym<T,%i>;\n", 2);
	tprint("template<class T>\n");
	tprint("using multipole = tensor_sym<T,%i>;\n", P - 1);
	tprint("#define EXPANSION_SIZE %i\n", (P + 2) * (P + 1) * P / 6);
	tprint("#define MULTIPOLE_SIZE %i\n", (P + 1) * P * (P - 1) / 6);
	const auto maxval = 1000.0;
	/*	tprint("#define SCALE_FACTOR %ef\n", maxval);
	 for (int n = 0; n <= P; n++) {
	 tprint("#define SCALE_FACTOR%i %ef\n", n, pow(maxval, n));
	 tprint("#define SCALE_FACTOR_INV%i %ef\n", n, 1.0 / pow(maxval, n));
	 }*/

	tprint("\n\ntemplate<class T>\n");
	tprint("CUDA_EXPORT\n");
	tprint("inline int greens_function(tensor_sym<T, %i>& D, array<T, NDIM> dx, float rsinv, float rsinv2) {\n", P);
	flops = 0;
	indent();
//	tprint("if( scale ) {\n");
//	indent();
//	tprint("X[0] *= T(SCALE_FACTOR);\n");
//	tprint("X[1] *= T(SCALE_FACTOR);\n");
//	tprint("X[2] *= T(SCALE_FACTOR);\n");
//	deindent();
//	tprint("}\n");

	tprint("D = T(0);\n"); // 5
	tprint("T r2 = fmaf(dx[0], dx[0], fmaf(dx[1], dx[1], sqr(dx[2])));\n"); // 5
	tprint("const T r = sqrt(r2);\n"); // 1
	tprint("const T n8r = T(-0.5) * r * rsinv2;\n"); // 1
	tprint("const T rinv = 1.f / r;\n"); // 2
	tprint("T exp0 = expf( -T(0.25) * rsinv2 * r2 );\n");
	tprint("T erf0 = erff( T(0.5) * rsinv * r);\n");
	tprint("const T expfactor = T(1.0/%.8e) * rsinv * exp0;\n", sqrt(M_PI)); // 1
	tprint("T e0 = expfactor * rinv;\n"); // 1
	tprint("const T rinv0 = T(1);\n"); // 2
	tprint("const T rinv1 = rinv;\n"); // 2
	for (int l = 2; l < (P + 1) / 2; l++) {
		const int i = l / 2;
		const int j = l - i;
		tprint("const T rinv%i = rinv%i * rinv%i;\n", l, i, j);                      // (P-2)
	}
	tprint("const T d0 = erf0 * rinv;\n");                                       // 2
	for (int l = 1; l < P; l++) {
		tprint("const T d%i = fmaf(T(%i) * d%i, rinv, e0);\n", l, -2 * l + 1, l - 1);            // (P-1)*4
		if (l != P - 1) {
			tprint("e0 *= n8r;\n");                                                // (P-1)
		}
	}
	for (int l = 0; l < P; l++) {
		for (int m = 0; m <= l; m++) {
			if (l + m < P) {
				tprint("const T Drinvpow_%i_%i = d%i * rinv%i;\n", l, m, l, m);
			}
		}
	}
	tprint("array<T,NDIM> dxrinv;\n");
	tprint("dxrinv[0] = dx[0] * rinv;\n");
	tprint("dxrinv[1] = dx[1] * rinv;\n");
	tprint("dxrinv[2] = dx[2] * rinv;\n");
	compute_dx<P>("dxrinv");
	compute_detrace_ewald("x", "D");
	tprint("D[0] += T(%.9e) * rsinv2; \n", 0.25 * M_PI);                                                // 1
	tprint("return 0; \n", M_PI);                                                // 1
	deindent();
	tprint("}\n");

	array<int, NDIM> n;
	array<int, NDIM> m;
	for (int Pmax = 2; Pmax <= P; Pmax += P - 2) {
		flops = 0;
		tprint("\n\ntemplate<class T>\n");
		tprint("CUDA_EXPORT\n");
		tprint("inline int M2L(tensor_sym<T, %i>& L, const tensor_sym<T, %i>& M, const tensor_sym<T, %i>& D, bool do_phi) {\n", Pmax,
		P - 1, P);
		indent();
		int phi_flops = 0;
		n[0] = n[1] = n[2] = 0;
		const int n0 = 0;
		const int q0 = intmin(P - n0, P - 1);
		struct entry_type {
			array<int, NDIM> n;
			array<int, NDIM> m;
			double coeff;
		};
		vector<vector<entry_type>> entries;
		entries.resize(Pmax * (Pmax + 1) * (Pmax + 2) / 6);
		for (n[0] = 0; n[0] < Pmax; n[0]++) {
			for (n[1] = 0; n[1] < Pmax - n[0]; n[1]++) {
				for (n[2] = 0; n[2] < Pmax - n[0] - n[1]; n[2]++) {
					const int n0 = n[0] + n[1] + n[2];
					const int q0 = intmin(P - n0, P - 1);
					for (m[0] = 0; m[0] < q0; m[0]++) {
						for (m[1] = 0; m[1] < q0 - m[0]; m[1]++) {
							for (m[2] = 0; m[2] < q0 - m[0] - m[1]; m[2]++) {
								const double coeff = 1.0 / vfactorial(m);
								entry_type e;
								e.coeff = coeff;
								e.n = n;
								e.m = m;
								const int index = sym_index(n[0], n[1], n[2]);
								entries[index].push_back(e);
							}
						}
					}
				}
			}
		}
		const auto sort_func = [Pmax](entry_type a, entry_type b) {
			if( a.coeff < b.coeff ) {
				return true;
			} else {
				return false;
			}
		};
		for (auto& e : entries) {
			std::sort(e.begin(), e.end(), sort_func);
		}
		for (int i = 0; i < 1; i++) {
			double last_coeff = 1.0;
			const auto n = entries[i][0].n;
			const int nindex = trless_index(n[0], n[1], n[2], Pmax);
			auto& fl = phi_flops;
			tprint("if( do_phi ) {\n");
			indent();
			for (int j = 0; j < entries[i].size(); j++) {
				const auto coeff = entries[i][j].coeff;
				const auto m = entries[i][j].m;
				if (!close21(last_coeff / coeff)) {
					double factor = last_coeff / coeff;
					tprint("L[%i] *= T(%.9e);\n", nindex, factor);
					last_coeff = coeff;
					fl++;
				}
				tprint("L[%i] = fmaf(M[%i], D[%i], L[%i]);\n", nindex, sym_index(m[0], m[1], m[2]), sym_index(n[0] + m[0], n[1] + m[1], n[2] + m[2]),
						sym_index(n[0], n[1], n[2]));
				fl += 2;
			}
			if (!close21(last_coeff)) {
				fl++;
				tprint("L[%i] *= T(%.9e);\n", nindex, last_coeff);
			}
			if (nindex == 0) {
				deindent();
				tprint("}\n");
			}
		}
		int total_size = 0;
		for (int i = 1; i < entries.size(); i++) {
			total_size += entries[i].size();
		}
		int half_size = 0;
		int mid;
		for (int i = 1; i < entries.size(); i++) {
			half_size += entries[i].size();
			if (half_size >= total_size / 2) {
				mid = i;
				break;
			}
		}
		vector<std::string> cmds1, cmds2;
		for (int i = 1; i < entries.size(); i++) {
			double last_coeff = 1.0;
			const auto n = entries[i][0].n;
			const int nindex = sym_index(n[0], n[1], n[2]);
			auto& fl = flops;
			auto& cmds = i >= mid ? cmds2 : cmds1;
			char* str;
			for (int j = 0; j < entries[i].size(); j++) {
				const auto coeff = entries[i][j].coeff;
				const auto m = entries[i][j].m;
				if (!close21(last_coeff / coeff)) {
					double factor = last_coeff / coeff;
					ASPRINTF(&str, "L[%i] *= T(%.9e);\n", nindex, factor);
					cmds.push_back(str);
					free(str);
					last_coeff = coeff;
					fl++;
				}
				ASPRINTF(&str, "L[%i] = fmaf(M[%i], D[%i], L[%i]);\n", nindex, sym_index(m[0], m[1], m[2]), sym_index(n[0] + m[0], n[1] + m[1], n[2] + m[2]),
						sym_index(n[0], n[1], n[2]));
				cmds.push_back(str);
				free(str);
				fl += 2;
			}
			if (!close21(last_coeff)) {
				fl++;
				ASPRINTF(&str, "L[%i] *= T(%.9e);\n", nindex, last_coeff);
				cmds.push_back(str);
			}
		}
		int i = 0;
		int j = 0;
		while (i < cmds1.size() || j < cmds2.size()) {
			if (i < cmds1.size()) {
				tprint("%s", cmds1[i].c_str());
				i++;
			}
			if (j < cmds2.size()) {
				tprint("%s", cmds2[j].c_str());
				j++;
			}
		}
		tprint("return %i + do_phi * %i;\n", flops, phi_flops);
		deindent();
		tprint("}\n");
	}

	tprint("\n\ntemplate<class T>\n");
	tprint("CUDA_EXPORT\n");
	tprint("tensor_sym<T, %i> P2M(array<T, NDIM>& X) {\n", P - 1);
	flops = 0;
	indent();
	tprint("tensor_sym<T, %i> M;\n", P - 1);
	/*	tprint("X[0] *= -T(SCALE_FACTOR);\n");
	 tprint("X[1] *= -T(SCALE_FACTOR);\n");
	 tprint("X[2] *= -T(SCALE_FACTOR);\n");*/
	tprint("X[0] *= -T(1);\n");
	tprint("X[1] *= -T(1);\n");
	tprint("X[2] *= -T(1);\n");
	flops += 6;
	auto index = std::function<int(int, int, int)>(sym_index);
	tprint("M[%i] = %s(1);\n", index(0, 0, 0), "T");
	tprint("M[%i] = %s[0];\n", index(1, 0, 0), "X");
	tprint("M[%i] = %s[1];\n", index(0, 1, 0), "X");
	tprint("M[%i] = %s[2];\n", index(0, 0, 1), "X");
	flops = 0;

	for (int n0 = 2; n0 < P; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[0] - n[1];
				array<int, NDIM> j = n;
				int j0 = n0;
				const int jmin = std::max(1, n0 / 2);
				while (j0 > jmin) {
					for (int dim = 0; dim < NDIM && j0 > jmin; dim++) {
						if (j[dim] > 0) {
							j[dim]--;
							j0--;
						}
					}
				}
				array<int, NDIM> k;
				k = n - j;
				tprint("M[%i] = M[%i] * M[%i];\n", index(n[0], n[1], n[2]), index(k[0], k[1], k[2]), index(j[0], j[1], j[2]));
				flops++;
			}
		}
	}
	tprint("return M;\n");
	deindent();
	tprint("}\n");
	tprint("\n\ntemplate<class T>\n");
	tprint("CUDA_EXPORT\n");
	tprint("tensor_sym<T, %i> M2M(const tensor_sym<T,%i>& Ma, array<T, NDIM>& X) {\n",
	P - 1, P - 1);
	flops = 0;
	indent();
	tprint("auto Mb = Ma;\n", P - 1);
	//tprint("X[0] *= -T(SCALE_FACTOR);\n");
//	tprint("X[1] *= -T(SCALE_FACTOR);\n");
//	tprint("X[2] *= -T(SCALE_FACTOR);\n");
	tprint("X[0] *= -T(1);\n");
	tprint("X[1] *= -T(1);\n");
	tprint("X[2] *= -T(1);\n");
	flops += 6;
	//	reference_sym("Mb", P - 1);
	//	reference_trless("Mc", P - 1);
	compute_dx<P - 1>();
	struct mentry {
		array<int, NDIM> n;
		array<int, NDIM> k;
		double factor;
	};
	array<int, NDIM> k;
	vector<vector<mentry>> mentries((P - 1) * P * (P + 1) / 6);
	for (int n0 = 0; n0 <= P - 2; n0++) {
		for (n[0] = 0; n[0] <= n0; n[0]++) {
			for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
				n[2] = n0 - n[0] - n[1];
				for (k[0] = 0; k[0] <= intmin(n0, n[0]); k[0]++) {
					for (k[1] = 0; k[1] <= intmin(n0 - k[0], n[1]); k[1]++) {
						for (k[2] = 0; k[2] <= intmin(n0 - k[0] - k[1], n[2]); k[2]++) {
							const auto factor = (vfactorial(n)) / double(vfactorial(k) * vfactorial(n - k));
							if (n != k) {
								mentry e;
								e.n = n;
								e.k = k;
								e.factor = factor;
								const int index = sym_index(n[0], n[1], n[2]);
								mentries[index].push_back(e);
							}
						}
					}
				}
			}
		}
	}
	for (auto& m : mentries) {
		std::sort(m.begin(), m.end(), [](mentry a, mentry b) {
			return a.factor > b.factor;
		});
	}
	int total_size = 0;
	for (int i = 0; i < mentries.size(); i++) {
		total_size += mentries[i].size();
	}
	int half_size = 0;
	int mid;
	for (int i = 0; i < mentries.size(); i++) {
		if (half_size >= total_size / 2) {
			mid = i;
			break;
		}
		half_size += mentries[i].size();
	}
	vector<std::string> cmds1, cmds2;
	char* str;
	for (int i = 0; i < mentries.size(); i++) {
		auto& cmds = i < mid ? cmds1 : cmds2;
		double last_factor = 1.0;
		if (mentries[i].size()) {
			const auto n = mentries[i][0].n;
			const int nindex = sym_index(n[0], n[1], n[2]);
			for (int j = 0; j < mentries[i].size(); j++) {
				const auto k = mentries[i][j].k;
				const auto factor = mentries[i][j].factor;
				if (!close21(last_factor / factor)) {
					ASPRINTF(&str, "Mb[%i] *= T(%.9e);\n", nindex, last_factor / factor);
					cmds.push_back(str);
					flops++;
					free(str);
					last_factor = factor;
				}
				ASPRINTF(&str, "Mb[%i] = fmaf( x[%i], Ma[%i], Mb[%i]);\n", nindex, sym_index(n[0] - k[0], n[1] - k[1], n[2] - k[2]), sym_index(k[0], k[1], k[2]),
						sym_index(n[0], n[1], n[2]));
				cmds.push_back(str);
				free(str);
				flops += 2;
			}
			if (!close21(last_factor)) {
				ASPRINTF(&str, "Mb[%i] *= T(%.9e);\n", nindex, last_factor);
				cmds.push_back(str);
				flops++;
				free(str);
			}
		}
	}
	int i = 0;
	int j = 0;
	while (i < cmds1.size()) {
		tprint("%s", cmds1[i].c_str());
		i++;
	}
	while (j < cmds2.size()) {
		tprint("%s", cmds2[j].c_str());
		j++;
	}

	tprint("return Mb;\n");
	printf("\n", flops);
	deindent();
	tprint("}\n");
	do_expansion< P >(false);

	do_expansion<2>(true);

#ifdef USE_CUDA
	do_expansion_cuda<P>();
#endif
	/*

	 tprint("template<class T>\n");
	 tprint("CUDA_EXPORT int apply_scale_factor_inv(tensor_trless_sym<T,%i> &L) {\n", P);
	 indent();
	 apply_scale_factorL < P > ("L");
	 tprint("return %i;\n", P * P + 1);
	 deindent();
	 tprint("}\n");

	 tprint("template<class T>\n");
	 tprint("CUDA_EXPORT int apply_scale_factor(tensor_trless_sym<T,%i> &M) {\n", P - 1);
	 indent();
	 apply_scale_factorM < P - 1 > ("M");
	 tprint("return %i;\n", (P - 1) * (P - 1) + 1);

	 deindent();
	 tprint("}\n");
	 */
}

#endif
