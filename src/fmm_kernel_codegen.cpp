/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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
#include <algorithm>
#include <cmath>
#include <functional>

#include <cosmictiger/containers.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/tensor.hpp>

static int ntab = 0;

void indent() {
    ntab++;
}

void deindent() {
    ntab--;
}

int symIndex(int l, int m, int n) {
    return (l + m + n) * (l + m + n + 1) * ((l + m + n) + 2) / 6 + (m + n) * ((m + n) + 1) / 2 + n;
}

template <class... Args>
void tprint(char const* str, Args&&... args) {
    for (int i = 0; i < ntab; i++) {
        printf("\t");
    }
    printf(str, std::forward<Args>(args)...);
}

void tprint(char const* str) {
    for (int i = 0; i < ntab; i++) {
        printf("\t");
    }
    printf("%s", str);
}

int computeDx(int pValue2, char const* name = "position", bool trless = false, bool dble = false) {
    Array<int, NDIM> n;
    tprint("%s x[%i];\n", dble ? "T" : "T",
        trless ? pValue2 * pValue2 + 1 : (pValue2 + 1) * pValue2 * (pValue2 + 2) / 6);
    auto index = trless ? std::function<int(int, int, int)>([pValue2](int x, int y, int z) {
        return trlessIndex(x, y, z, pValue2);
    }) :
                          std::function<int(int, int, int)>(symIndex);
    tprint("x[%i] = %s(1);\n", index(0, 0, 0), dble ? "T" : "T");
    tprint("x[%i] = %s[0];\n", index(1, 0, 0), name);
    tprint("x[%i] = %s[1];\n", index(0, 1, 0), name);
    tprint("x[%i] = %s[2];\n", index(0, 0, 1), name);
    int flops = 0;

    for (int n0 = 2; n0 < pValue2; n0++) {
        for (n[0] = 0; n[0] <= n0; n[0]++) {
            for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
                n[2] = n0 - n[0] - n[1];
                if (trless && ((n[2] > 2) || (n[2] > 1 && !(n[0] == 0 && n[1] == 0)))) {
                    continue;
                }
                Array<int, NDIM> j = n;
                int j0 = n0;
                int const jmin = std::max(1, n0 / 2);
                while (j0 > jmin) {
                    for (int dim = 0; dim < NDIM && j0 > jmin; dim++) {
                        if (j[dim] > 0) {
                            j[dim]--;
                            j0--;
                        }
                    }
                }
                Array<int, NDIM> k;
                k = n - j;
                if (trless && (((j[2] > 2) || (j[2] > 1 && !(j[0] == 0 && j[1] == 0))))) {
                    PRINT("!!!!!!!!!!!\n");
                    abort();
                }
                if (trless && (((k[2] > 2) || (k[2] > 1 && !(k[0] == 0 && k[1] == 0))))) {
                    PRINT("!!!!!!!!!!!\n");
                    abort();
                }
                tprint("x[%i] = x[%i] * x[%i]; // %i %i %i | %i %i %i | %i %i %i\n",
                    index(n[0], n[1], n[2]), index(k[0], k[1], k[2]), index(j[0], j[1], j[2]), n[0],
                    n[1], n[2], j[0], j[1], j[2], k[0], k[1], k[2]);
                flops++;
            }
        }
    }
    return flops;
}

int computeDxTensor(int pValue2, char const* name = "position") {
    Array<int, NDIM> n;
    tprint("SymmetricTensor<T,%i> dx;\n", pValue2);
    tprint("dx[0] = T(1);\n");
    tprint("dx[%i] = %s[0];\n", symIndex(1, 0, 0), name);
    tprint("dx[%i] = %s[1];\n", symIndex(0, 1, 0), name);
    tprint("dx[%i] = %s[2];\n", symIndex(0, 0, 1), name);
    int flops = 0;
    for (int n0 = 2; n0 < pValue2; n0++) {
        for (n[0] = 0; n[0] <= n0; n[0]++) {
            for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
                n[2] = n0 - n[0] - n[1];
                Array<int, NDIM> j = n;
                int j0 = n0;
                int const jmin = std::max(1, n0 / 2);
                while (j0 > jmin) {
                    for (int dim = 0; dim < NDIM && j0 > jmin; dim++) {
                        if (j[dim] > 0) {
                            j[dim]--;
                            j0--;
                        }
                    }
                }
                Array<int, NDIM> k;
                k = n - j;
                tprint("dx[%i]= dx[%i] * dx[%i];\n", symIndex(n[0], n[1], n[2]),
                    symIndex(k[0], k[1], k[2]), symIndex(j[0], j[1], j[2]));
                flops++;
            }
        }
    }
    return flops;
}

int acc(std::string a, Array<int, NDIM> n, std::string c, Array<int, NDIM> j) {
    tprint("%s%i%i%i += %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], c.c_str(), j[0], j[1], j[2]);
    return 1;
}

int dec(std::string a, Array<int, NDIM> n, std::string c, Array<int, NDIM> j) {
    tprint("%s%i%i%i -= %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], c.c_str(), j[0], j[1], j[2]);
    return 1;
}

int eqp(std::string a, Array<int, NDIM> n, std::string c, Array<int, NDIM> j) {
    tprint("%s%i%i%i = %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], c.c_str(), j[0], j[1], j[2]);
    return 0;
}

int eqn(std::string a, Array<int, NDIM> n, std::string c, Array<int, NDIM> j) {
    tprint("%s%i%i%i = -%s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], c.c_str(), j[0], j[1], j[2]);
    return 1;
}

int mul(std::string a, Array<int, NDIM> n, double b, std::string c, Array<int, NDIM> j) {
    tprint("%s%i%i%i = T(%.9e) * %s%i%i%i;\n", a.c_str(), n[0], n[1], n[2], b, c.c_str(), j[0],
        j[1], j[2]);
    return 1;
}

int fma(std::string a, Array<int, NDIM> n, double b, std::string c, Array<int, NDIM> j) {
    tprint("%s%i%i%i = fmaf(T(%.9e), %s%i%i%i, %s%i%i%i);\n", a.c_str(), n[0], n[1], n[2], b,
        c.c_str(), j[0], j[1], j[2], a.c_str(), n[0], n[1], n[2]);
    return 2;
}

bool close21(double a) {
    return std::abs(1.0 - a) < 1.0e-20;
}
template <int pValue>
int computeDetrace(std::string iname, std::string oname, char type = 'f') {
    Array<int, NDIM> m;
    Array<int, NDIM> k;
    Array<int, NDIM> n;

    struct EntryType {
        Array<int, NDIM> src;
        Array<int, NDIM> dest;
        double factor;
    };
    Vector<EntryType> entries;
    int flops = 0;

    Vector<std::string> asn;
    Vector<std::string> op;
    for (int m0 = 1; m0 <= pValue / 2; m0++) {
        Array<int, NDIM> j;
        int const q = pValue - 2 * m0;
        for (j[0] = 0; j[0] < q; j[0]++) {
            for (j[1] = 0; j[1] < q - j[0]; j[1]++) {
                for (j[2] = 0; j[2] < q - j[0] - j[1]; j[2]++) {
                    int const j0 = j[0] + j[1] + j[2];
                    int n0 = j0 + 2 * m0;
                    bool first = true;
                    Array<int, NDIM> k;
                    for (k[0] = 0; k[0] <= m0; k[0]++) {
                        for (k[1] = 0; k[1] <= m0 - k[0]; k[1]++) {
                            k[2] = m0 - k[0] - k[1];
                            double const num = factorial(m0);
                            double const den = vfactorial(k);
                            double const factor = num / den;
                            auto const p = j + k * 2;
                            char* str;
                            if (first) {
                                if (close21(factor)) {
                                    ASPRINTF(&str, "T %sOrder%iTrace%iIndex%i%i%i = %s[%i];\n",
                                        iname.c_str(), n0, m0, j[0], j[1], j[2], iname.c_str(),
                                        symIndex(p[0], p[1], p[2]));
                                } else if (close21(-factor)) {
                                    ASPRINTF(&str, "T %sOrder%iTrace%iIndex%i%i%i = -%s[%i];\n",
                                        iname.c_str(), n0, m0, j[0], j[1], j[2], iname.c_str(),
                                        symIndex(p[0], p[1], p[2]));
                                    flops++;
                                } else {
                                    ASPRINTF(&str,
                                        "T %sOrder%iTrace%iIndex%i%i%i = T(%.9e) * %s[%i];\n",
                                        iname.c_str(), n0, m0, j[0], j[1], j[2], factor,
                                        iname.c_str(), symIndex(p[0], p[1], p[2]));
                                    flops++;
                                }
                                asn.push_back(str);
                                first = false;
                                free(str);
                            } else {
                                if (close21(factor)) {
                                    ASPRINTF(&str, "%sOrder%iTrace%iIndex%i%i%i += %s[%i];\n",
                                        iname.c_str(), n0, m0, j[0], j[1], j[2], iname.c_str(),
                                        symIndex(p[0], p[1], p[2]));
                                    flops++;
                                } else if (close21(-factor)) {
                                    ASPRINTF(&str, "%sOrder%iTrace%iIndex%i%i%i -= %s[%i];\n",
                                        iname.c_str(), n0, m0, j[0], j[1], j[2], iname.c_str(),
                                        symIndex(p[0], p[1], p[2]));
                                    flops++;
                                } else {
                                    ASPRINTF(&str,
                                        "%sOrder%iTrace%iIndex%i%i%i = fmaf(T(%.9e), %s[%i], "
                                        "%sOrder%iTrace%iIndex%i%i%i);\n",
                                        iname.c_str(), n0, m0, j[0], j[1], j[2], factor,
                                        iname.c_str(), symIndex(p[0], p[1], p[2]), iname.c_str(),
                                        n0, m0, j[0], j[1], j[2]);
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
    for (n[0] = 0; n[0] < pValue; n[0]++) {
        for (n[1] = 0; n[1] < pValue - n[0]; n[1]++) {
            int const nzmax =
                (n[0] == 0 && n[1] == 0) ? intmin(3, pValue) : intmin(pValue - n[0] - n[1], 2);
            for (n[2] = 0; n[2] < nzmax; n[2]++) {
                int const n0 = n[0] + n[1] + n[2];
                bool first = true;
                for (m[0] = 0; m[0] <= n[0] / 2; m[0]++) {
                    for (m[1] = 0; m[1] <= n[1] / 2; m[1]++) {
                        for (m[2] = 0; m[2] <= n[2] / 2; m[2]++) {
                            int const m0 = m[0] + m[1] + m[2];
                            if (type == 'd' &&
                                ((n0 == 2 && (n[0] == 2 || n[1] == 2 || n[2] == 2)) && m0 == 1)) {
                                continue;
                            }
                            double num =
                                double(n1pow(m0) * dfactorial(2 * n0 - 2 * m0 - 1) * vfactorial(n));
                            double den =
                                double((1 << m0) * vfactorial(m) * vfactorial(n - (m) * 2));
                            double factor = num / den;
                            if (type == 'd') {
                                factor *= 1.0 / dfactorial(2 * n0 - 1);
                            }
                            auto const p = n - m * 2;
                            char* str;
                            if (first) {
                                if (m0 > 0) {
                                    if (close21(factor)) {
                                        ASPRINTF(&str, "%s[%i] = %sOrder%iTrace%iIndex%i%i%i;\n",
                                            oname.c_str(), trlessIndex(n[0], n[1], n[2], pValue),
                                            iname.c_str(), n0, m0, p[0], p[1], p[2]);
                                    } else if (close21(-factor)) {
                                        ASPRINTF(&str, "%s[%i] = -%sOrder%iTrace%iIndex%i%i%i;\n",
                                            oname.c_str(), trlessIndex(n[0], n[1], n[2], pValue),
                                            iname.c_str(), n0, m0, p[0], p[1], p[2]);
                                        flops++;
                                    } else {
                                        ASPRINTF(&str,
                                            "%s[%i] = T(%.9e) * %sOrder%iTrace%iIndex%i%i%i;\n",
                                            oname.c_str(), trlessIndex(n[0], n[1], n[2], pValue),
                                            factor, iname.c_str(), n0, m0, p[0], p[1], p[2]);
                                        flops++;
                                    }
                                } else {
                                    if (close21(factor)) {
                                        ASPRINTF(&str, "%s[%i] = %s[%i];\n", oname.c_str(),
                                            trlessIndex(n[0], n[1], n[2], pValue), iname.c_str(),
                                            symIndex(p[0], p[1], p[2]));
                                    } else if (close21(-factor)) {
                                        ASPRINTF(&str, "%s[%i] = -%s[%i];\n", oname.c_str(),
                                            trlessIndex(n[0], n[1], n[2], pValue), iname.c_str(),
                                            symIndex(p[0], p[1], p[2]));
                                        flops++;
                                    } else {
                                        ASPRINTF(&str, "%s[%i] = T(%.9e) * %s[%i];\n",
                                            oname.c_str(), trlessIndex(n[0], n[1], n[2], pValue),
                                            factor, iname.c_str(), symIndex(p[0], p[1], p[2]));
                                        ;
                                        flops++;
                                    }
                                }
                                asn.push_back(str);
                                free(str);
                                first = false;
                            } else {
                                if (close21(factor)) {
                                    if (m0 > 0) {
                                        ASPRINTF(&str, "%s[%i] += %sOrder%iTrace%iIndex%i%i%i;\n",
                                            oname.c_str(), trlessIndex(n[0], n[1], n[2], pValue),
                                            iname.c_str(), n0, m0, p[0], p[1], p[2]);
                                        flops++;
                                    } else {
                                        ASPRINTF(&str, "%s[%i] += %s[%i];\n", oname.c_str(),
                                            trlessIndex(n[0], n[1], n[2], pValue), iname.c_str(),
                                            symIndex(p[0], p[1], p[2]));
                                        flops++;
                                    }
                                } else if (close21(-factor)) {
                                    if (m0 > 0) {
                                        ASPRINTF(&str, "%s[%i] -= %sOrder%iTrace%iIndex%i%i%i;\n",
                                            oname.c_str(), trlessIndex(n[0], n[1], n[2], pValue),
                                            iname.c_str(), n0, m0, p[0], p[1], p[2]);
                                        flops++;
                                    } else {
                                        ASPRINTF(&str, "%s[%i] -= %s[%i];\n", oname.c_str(),
                                            trlessIndex(n[0], n[1], n[2], pValue), iname.c_str(),
                                            symIndex(p[0], p[1], p[2]));
                                        flops++;
                                    }
                                } else {
                                    if (m0 > 0) {
                                        ASPRINTF(&str,
                                            "%s[%i] = fmaf(T(%.9e), %sOrder%iTrace%iIndex%i%i%i, "
                                            "%s[%i]);\n",
                                            oname.c_str(), trlessIndex(n[0], n[1], n[2], pValue),
                                            factor, iname.c_str(), n0, m0, p[0], p[1], p[2],
                                            oname.c_str(), trlessIndex(n[0], n[1], n[2], pValue));
                                        flops += 2;
                                    } else {
                                        ASPRINTF(&str, "%s[%i] = fmaf(T(%.9e), %s[%i], %s[%i]);\n",
                                            oname.c_str(), trlessIndex(n[0], n[1], n[2], pValue),
                                            factor, iname.c_str(), symIndex(p[0], p[1], p[2]),
                                            oname.c_str(), trlessIndex(n[0], n[1], n[2], pValue));
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

template <int pValue>
int computeDetraceD(std::string iname, std::string oname, char type = 'f') {
    Array<int, NDIM> m;
    Array<int, NDIM> k;
    Array<int, NDIM> n;

    struct EntryType {
        Array<int, NDIM> src;
        Array<int, NDIM> dest;
        double factor;
    };
    Vector<EntryType> entries;
    int flops = 0;
    Vector<std::string> asn;
    Vector<std::string> op;
    op.resize(0);
    asn.resize(0);
    for (n[0] = 0; n[0] < pValue; n[0]++) {
        for (n[1] = 0; n[1] < pValue - n[0]; n[1]++) {
            int const nzmax =
                (n[0] == 0 && n[1] == 0) ? intmin(3, pValue) : intmin(pValue - n[0] - n[1], 2);
            for (n[2] = 0; n[2] < nzmax; n[2]++) {
                int const n0 = n[0] + n[1] + n[2];
                bool first = true;
                for (m[0] = 0; m[0] <= n[0] / 2; m[0]++) {
                    for (m[1] = 0; m[1] <= n[1] / 2; m[1]++) {
                        for (m[2] = 0; m[2] <= n[2] / 2; m[2]++) {
                            int const m0 = m[0] + m[1] + m[2];
                            double num =
                                double(n1pow(m0) * dfactorial(2 * n0 - 2 * m0 - 1) * vfactorial(n));
                            double den =
                                double((1 << m0) * vfactorial(m) * vfactorial(n - (m) * 2));
                            double factor = num / den;
                            auto const p = n - m * 2;
                            char* str;
                            if (first) {
                                if (close21(factor)) {
                                    ASPRINTF(&str, "%s[%i] = %s[%i];\n", oname.c_str(),
                                        trlessIndex(n[0], n[1], n[2], pValue), iname.c_str(),
                                        trlessIndex(p[0], p[1], p[2], pValue));
                                } else if (close21(-factor)) {
                                    ASPRINTF(&str, "%s[%i] = -%s[%i];\n", oname.c_str(),
                                        trlessIndex(n[0], n[1], n[2], pValue), iname.c_str(),
                                        trlessIndex(p[0], p[1], p[2], pValue));
                                    flops++;
                                } else {
                                    ASPRINTF(&str, "%s[%i] = T(%.9e) * %s[%i];\n", oname.c_str(),
                                        trlessIndex(n[0], n[1], n[2], pValue), factor,
                                        iname.c_str(), trlessIndex(p[0], p[1], p[2], pValue));
                                    flops++;
                                }
                                asn.push_back(str);
                                free(str);
                                first = false;
                            } else {
                                if (close21(factor)) {
                                    ASPRINTF(&str, "%s[%i] += %s[%i];\n", oname.c_str(),
                                        trlessIndex(n[0], n[1], n[2], pValue), iname.c_str(),
                                        trlessIndex(p[0], p[1], p[2], pValue));
                                    flops += 1;
                                } else if (close21(-factor)) {
                                    ASPRINTF(&str, "%s[%i] -= %s[%i];\n", oname.c_str(),
                                        trlessIndex(n[0], n[1], n[2], pValue), iname.c_str(),
                                        trlessIndex(p[0], p[1], p[2], pValue));
                                    flops += 1;
                                } else {
                                    ASPRINTF(&str, "%s[%i] = fmaf(T(%.9e), %s[%i], %s[%i]);\n",
                                        oname.c_str(), trlessIndex(n[0], n[1], n[2], pValue),
                                        factor, iname.c_str(),
                                        trlessIndex(p[0], p[1], p[2], pValue), oname.c_str(),
                                        trlessIndex(n[0], n[1], n[2], pValue));
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

template <int pValue>
int computeDetraceEwald(std::string iname, std::string oname) {
    Array<int, NDIM> m;
    Array<int, NDIM> k;
    Array<int, NDIM> n;

    struct EntryType {
        Array<int, NDIM> src;
        Array<int, NDIM> dest;
        double factor;
    };
    Vector<EntryType> entries;
    int flops = 0;

    Vector<std::string> asn;
    Vector<std::string> op;
    op.resize(0);
    asn.resize(0);
    for (n[0] = 0; n[0] < pValue; n[0]++) {
        for (n[1] = 0; n[1] < pValue - n[0]; n[1]++) {
            for (n[2] = 0; n[2] < pValue - n[0] - n[1]; n[2]++) {
                int const n0 = n[0] + n[1] + n[2];
                bool first = true;
                for (m[0] = 0; m[0] <= n[0] / 2; m[0]++) {
                    for (m[1] = 0; m[1] <= n[1] / 2; m[1]++) {
                        for (m[2] = 0; m[2] <= n[2] / 2; m[2]++) {
                            int const m0 = m[0] + m[1] + m[2];
                            double num = double(vfactorial(n));
                            double den =
                                double((1 << m0) * vfactorial(m) * vfactorial(n - (m) * 2));
                            double factor = num / den;
                            auto const p = n - m * 2;
                            char* str;
                            if (close21(factor)) {
                                ASPRINTF(&str,
                                    "%s[%i] = fmaf(%s[%i], dRinvPower%iOrder%i, %s[%i]);\n",
                                    oname.c_str(), symIndex(n[0], n[1], n[2]), iname.c_str(),
                                    symIndex(p[0], p[1], p[2]), n0 - m0, m0, oname.c_str(),
                                    symIndex(n[0], n[1], n[2]));
                                flops += 2;
                            } else if (close21(-factor)) {
                                ASPRINTF(&str, "%s[%i] -= %s[%i] * dRinvPower%iOrder%i;\n",
                                    oname.c_str(), symIndex(n[0], n[1], n[2]), iname.c_str(),
                                    symIndex(p[0], p[1], p[2]), n0 - m0, m0);
                                flops += 2;
                            } else {
                                ASPRINTF(&str,
                                    "%s[%i] = fmaf(T(%.9e), %s[%i]*dRinvPower%iOrder%i, %s[%i]);\n",
                                    oname.c_str(), symIndex(n[0], n[1], n[2]), factor,
                                    iname.c_str(), symIndex(p[0], p[1], p[2]), n0 - m0, m0,
                                    oname.c_str(), symIndex(n[0], n[1], n[2]));

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
    std::sort(op.begin(), op.end(),
        [](std::string a, std::string b) { return atoi(a.c_str() + 6) < atoi(b.c_str() + 6); });
    int maxop = op.size();
    for (int i = 0; i < maxop; i++) {
        tprint("%s", op[i].c_str());
    }

    return flops;
}

#define P ORDER
template <int q>
void constRefCompute(int sign, SymmetricTracelessTensor<int, q>& counts,
    SymmetricTracelessTensor<int, q>& signs, Array<int, NDIM> n) {
    int flops = 0;
    if (n[2] >= 2 && !(n[0] == 0 && n[1] == 0 && n[2] == 2)) {
        auto n1 = n;
        auto n2 = n;
        n1[2] -= 2;
        n2[2] -= 2;
        n1[0] += 2;
        n2[1] += 2;
        constRefCompute(-sign, counts, signs, n1);
        constRefCompute(-sign, counts, signs, n2);
    } else {
        counts(n)++;
        signs(n) = sign;
    }
}

template <int q>
int printConstRef(std::string name, std::string& cmd,
    SymmetricTracelessTensor<int, q> const& counts, SymmetricTracelessTensor<int, q> const& signs,
    Array<int, NDIM> n) {
    int flops = 0;
    Array<int, NDIM> k;
    Array<int, NDIM> lastIndex;
    for (k[0] = 0; k[0] < q; k[0]++) {
        for (k[1] = 0; k[1] < q - k[0]; k[1]++) {
            for (k[2] = 0; k[2] < q - k[0] - k[1]; k[2]++) {
                if (k[2] < 2 || (k[0] == 0 && k[1] == 0 && k[2] == 2)) {
                    if (counts(k)) {
                        lastIndex = k;
                    }
                }
            }
        }
    }
    cmd += signs(lastIndex) == 1 ? "+" : "-";
    if (signs(lastIndex) != 1) {
        flops++;
    }
    int opened = 0;
    bool fma = false;
    for (int l = 0; l < q * q + 1; l++) {
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
                if (n[0] != lastIndex[0] || n[1] != lastIndex[1] || n[2] != lastIndex[2]) {
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

template <int q>
int constReferenceTrless(std::string name) {
    Array<int, NDIM> n;
    int flops = 0;
    for (n[0] = 0; n[0] < q; n[0]++) {
        for (n[1] = 0; n[1] < q - n[0]; n[1]++) {
            for (n[2] = 0; n[2] < q - n[0] - n[1]; n[2]++) {
                if (!(n[2] >= 2 && !(n[0] == 0 && n[1] == 0 && n[2] == 2))) {
                    tprint("T const %s%i%i%i = ", name.c_str(), n[0], n[1], n[2]);
                } else {
                    tprint("T const %s%i%i%i = ", name.c_str(), n[0], n[1], n[2]);
                }
                SymmetricTracelessTensor<int, q> counts;
                SymmetricTracelessTensor<int, q> signs;
                counts = 0;
                signs = 1;
                constRefCompute(+1, counts, signs, n);
                std::string cmd;
                flops += printConstRef(name, cmd, counts, signs, n);
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

template <int q>
int constReferenceTrlessTensor(std::string name, std::string oname) {
    Array<int, NDIM> n;
    int flops = 0;
    for (n[0] = 0; n[0] < q; n[0]++) {
        for (n[1] = 0; n[1] < q - n[0]; n[1]++) {
            for (n[2] = 0; n[2] < q - n[0] - n[1]; n[2]++) {
                tprint("%s[%i] = ", oname.c_str(), symIndex(n[0], n[1], n[2]));
                SymmetricTracelessTensor<int, q> counts;
                SymmetricTracelessTensor<int, q> signs;
                counts = 0;
                signs = 1;
                constRefCompute(+1, counts, signs, n);
                std::string cmd;
                flops += printConstRef(name, cmd, counts, signs, n);
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

void referenceTrless(std::string name, int q) {
    for (int l = 0; l < q; l++) {
        for (int m = 0; m < q - l; m++) {
            for (int n = 0; n < q - l - m; n++) {
                if (n > 1 && !(l == 0 && m == 0 && n == 2)) {
                    continue;
                }
                int const index = trlessIndex(l, m, n, q);
                tprint("T& %s%i%i%i = %s[%i];\n", name.c_str(), l, m, n, name.c_str(), index);
            }
        }
    }
}

void referenceSym(std::string name, int q) {
    for (int l = 0; l < q; l++) {
        for (int m = 0; m < q - l; m++) {
            for (int n = 0; n < q - l - m; n++) {
                int const index = symIndex(l, m, n);
                tprint("T const %s%i%i%i = %s[%i];\n", name.c_str(), l, m, n, name.c_str(), index);
            }
        }
    }
}

template <int q>
void doExpansion(bool two) {
    int flops = 0;
    tprint("template<class T>\n");
    tprint("CUDA_EXPORT\n");
    tprint("#ifdef __CUDACC__\n");
    tprint("__noinline__\n");
    tprint("#endif\n");
    if (two) {
        tprint("SymmetricTracelessTensor<T, %i> l2P(SymmetricTracelessTensor<T, %i> const& la, "
               "Array<T,NDIM> "
               "position, bool doPhi) {\n",
            q, P);
    } else {
        tprint("SymmetricTracelessTensor<T, %i> l2L(SymmetricTracelessTensor<T, %i> const& la, "
               "Array<T,NDIM> "
               "position, bool doPhi) {\n",
            q, P);
    }
    indent();
    tprint("position[0] *= T(SCALE_FACTOR);\n");
    tprint("position[1] *= T(SCALE_FACTOR);\n");
    tprint("position[2] *= T(SCALE_FACTOR);\n");
    tprint("SymmetricTracelessTensor<T, %i> lb;\n", q);
    flops += 3 + computeDx(P);
    Array<int, NDIM> n;
    Array<int, NDIM> k;
    int phiFlops = 0;
    flops += constReferenceTrless<P>("la");
    if (two) {
        tprint("lb(0,0,0) = la(0,0,0);\n");
        tprint("lb(1,0,0) = la(1,0,0);\n");
        tprint("lb(0,1,0) = la(0,1,0);\n");
        tprint("lb(0,0,1) = la(0,0,1);\n");
    } else {
        tprint("lb = la;\n");
    }
    struct Entry {
        int ldest;
        Array<int, NDIM> p;
        Array<int, NDIM> k;
        double factor;
    };
    Vector<Vector<Entry>> entries(two ? 4 : q * q + 1);
    for (int n0 = 0; n0 < q; n0++) {
        for (n[0] = 0; n[0] <= n0; n[0]++) {
            for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
                n[2] = n0 - n[1] - n[0];
                if (n[2] <= 1 || (n[0] == 0 && n[1] == 0 && n[2] == 2)) {
                    int const n0 = n[0] + n[1] + n[2];
                    for (k[0] = 0; k[0] < P - n0; k[0]++) {
                        for (k[1] = 0; k[1] < P - n0 - k[0]; k[1]++) {
                            for (k[2] = 0; k[2] < P - n0 - k[0] - k[1]; k[2]++) {
                                auto const factor = double(1) / double(vfactorial(k));
                                auto const p = n + k;
                                int const p0 = p[0] + p[1] + p[2];
                                if (n != p) {
                                    Entry e;
                                    e.ldest = trlessIndex(n[0], n[1], n[2], q);
                                    e.factor = factor;
                                    e.k = k;
                                    e.p = p;
                                    entries[e.ldest].push_back(e);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (auto& e : entries) {
        std::sort(e.begin(), e.end(), [](Entry a, Entry b) { return (a.factor < b.factor); });
    }
    double lastFactor = 1.0;
    tprint("if( doPhi ) {\n");
    indent();
    for (int j = 0; j < entries[0].size(); j++) {
        auto const factor = entries[0][j].factor;
        auto const p = entries[0][j].p;
        auto const k = entries[0][j].k;
        auto const index = entries[0][j].ldest;
        if (!close21(lastFactor / factor)) {
            tprint("lb[0] *= T(%.9e);\n", lastFactor / factor);
            lastFactor = factor;
            phiFlops++;
        }
        tprint("lb[%i] = fmaf( x[%i], la%i%i%i, lb[%i]);\n", index, symIndex(k[0], k[1], k[2]),
            p[0], p[1], p[2], index);
        phiFlops += 2;
    }
    if (!close21(lastFactor)) {
        tprint("lb[0] *= T(%.9e);\n", lastFactor);
    }
    deindent();
    phiFlops++;
    tprint("}\n");
    int totalSize = 0;
    for (int i = 1; i < entries.size(); i++) {
        totalSize += entries[i].size();
    }
    int mid;
    int halfSize = 0;
    for (int i = 1; i < entries.size(); i++) {
        if (halfSize >= totalSize / 2) {
            mid = i;
            break;
        }
        halfSize += entries[i].size();
    }
    char* str;
    Vector<std::string> cmds1, cmds2;
    for (int i = 1; i < entries.size(); i++) {
        lastFactor = 1.0;
        auto& cmds = i >= mid ? cmds2 : cmds1;
        if (entries[i].size()) {
            auto const index = entries[i][0].ldest;
            for (int j = 0; j < entries[i].size(); j++) {
                auto const factor = entries[i][j].factor;
                auto const p = entries[i][j].p;
                auto const k = entries[i][j].k;
                if (!close21(factor / lastFactor)) {
                    ASPRINTF(&str, "lb[%i] *= T(%.9e);\n", index, lastFactor / factor);
                    cmds.push_back(str);
                    free(str);
                    flops++;
                    lastFactor = factor;
                }
                ASPRINTF(&str, "lb[%i] = fmaf( x[%i], la%i%i%i, lb[%i]);\n", index,
                    symIndex(k[0], k[1], k[2]), p[0], p[1], p[2], index);
                cmds.push_back(str);
                free(str);
                flops += 2;
            }
            if (!close21(lastFactor)) {
                ASPRINTF(&str, "lb[%i] *= T(%.9e);\n", index, lastFactor);
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
    tprint("return lb;\n");
    printf("/* FLOPS = %i + doPhi * %i*/\n", flops, phiFlops);
    deindent();
    tprint("}\n");
}

template <int q>
void applyScaleFactorM(char const* name) {
    Array<int, NDIM> n;
    for (int n0 = 0; n0 < q; n0++) {
        for (n[0] = 0; n[0] <= n0; n[0]++) {
            for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
                n[2] = n0 - n[1] - n[0];
                if (n[2] <= 1 || (n[0] == 0 && n[1] == 0 && n[2] == 2)) {
                    tprint("%s[%i] *= SCALE_FACTOR_INV%i;\n", name,
                        trlessIndex(n[0], n[1], n[2], q), n0);
                }
            }
        }
    }
}

template <int q>
void applyScaleFactorL(char const* name) {
    Array<int, NDIM> n;
    for (int n0 = 0; n0 < q; n0++) {
        for (n[0] = 0; n[0] <= n0; n[0]++) {
            for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
                n[2] = n0 - n[1] - n[0];
                if (n[2] <= 1 || (n[0] == 0 && n[1] == 0 && n[2] == 2)) {
                    tprint("%s[%i] *= SCALE_FACTOR_INV%i;\n", name,
                        trlessIndex(n[0], n[1], n[2], q), n0 + 1);
                }
            }
        }
    }
}

template <int q>
void doExpansionCuda() {
    int flops = 0;
    struct Entry {
        int lsource;
        int ldest;
        int xsource;
        float factor;
    };
    Array<int, NDIM> n;
    Array<int, NDIM> k;
    Vector<Entry> entries;
    Vector<Entry> phiEntries;
    for (int n0 = 0; n0 < q; n0++) {
        for (n[0] = 0; n[0] <= n0; n[0]++) {
            for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
                n[2] = n0 - n[1] - n[0];
                if (n[2] <= 1 || (n[0] == 0 && n[1] == 0 && n[2] == 2)) {
                    int const n0 = n[0] + n[1] + n[2];
                    for (k[0] = 0; k[0] < P - n0; k[0]++) {
                        for (k[1] = 0; k[1] < P - n0 - k[0]; k[1]++) {
                            for (k[2] = 0; k[2] < P - n0 - k[0] - k[1]; k[2]++) {
                                auto const factor = double(1) / double(vfactorial(k));
                                auto const p = n + k;
                                int const p0 = p[0] + p[1] + p[2];
                                if (n != p) {
                                    Entry e;
                                    e.ldest = trlessIndex(n[0], n[1], n[2], P);
                                    e.xsource = symIndex(k[0], k[1], k[2]);
                                    e.lsource = symIndex(p[0], p[1], p[2]);
                                    e.factor = factor;
                                    if (n0 == 0) {
                                        phiEntries.push_back(e);
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
    std::sort(entries.begin(), entries.end(), [](Entry a, Entry b) {
        if (a.ldest < b.ldest) {
            return true;
        } else if (a.ldest > b.ldest) {
            return false;
        } else {
            return a.lsource < b.lsource;
        }
    });
    Vector<Entry> entries1, entries2;
    for (int i = 0; i < entries.size(); i++) {
        //	if (i < (entries.size() + 1) / 2) {
        entries1.push_back(entries[i]);
        //	} else {
        //		entries2.push_back(entries[i]);
        //	}
    }

    tprint("static __device__ char ldest1[%i] = { ", entries1.size());
    for (int i = 0; i < entries1.size(); i++) {
        printf("%i", entries1[i].ldest);
        if (i != entries1.size() - 1) {
            printf(",");
        }
    }
    tprint("};\n");
    tprint("inline __constant__ float factor1[%i] = { ", entries1.size());
    for (int i = 0; i < entries1.size(); i++) {
        printf("float(%.9e)", entries1[i].factor);
        if (i != entries1.size() - 1) {
            printf(",");
        }
    }
    tprint("};\n");
    tprint("inline __constant__ char xsrc1[%i] = { ", entries1.size());
    for (int i = 0; i < entries1.size(); i++) {
        printf("%i", entries1[i].xsource);
        if (i != entries1.size() - 1) {
            printf(",");
        }
    }
    tprint("};\n");
    tprint("inline __constant__ char lsrc1[%i] = { ", entries1.size());
    for (int i = 0; i < entries1.size(); i++) {
        printf("%i", entries1[i].lsource);
        if (i != entries1.size() - 1) {
            printf(",");
        }
    }
    tprint("};\n");

    tprint("inline __constant__ float phiFactor[%i] = { ", phiEntries.size());
    for (int i = 0; i < phiEntries.size(); i++) {
        printf("float(%.9e)", phiEntries[i].factor);
        if (i != phiEntries.size() - 1) {
            printf(",");
        }
    }
    tprint("};\n");
    tprint("inline __constant__ char phiLsrc[%i] = { ", phiEntries.size());
    for (int i = 0; i < phiEntries.size(); i++) {
        printf("%i", phiEntries[i].lsource);
        if (i != phiEntries.size() - 1) {
            printf(",");
        }
    }
    tprint("};\n");

    tprint("#ifdef __CUDACC__\n");
    tprint("template<class T>\n");
    tprint("__device__\n");
    tprint("SymmetricTracelessTensor<T, %i> l2LCuda(SymmetricTracelessTensor<T, %i> const& la, "
           "Array<T,NDIM> "
           "position, bool doPhi) {\n",
        q, P);

    indent();

    tprint("int const tid = threadIdx.x;\n");
    tprint("position[0] *= T(SCALE_FACTOR);\n");
    tprint("position[1] *= T(SCALE_FACTOR);\n");
    tprint("position[2] *= T(SCALE_FACTOR);\n");
    tprint("SymmetricTracelessTensor<T, %i> lb;\n", q);
    tprint("SymmetricTensor<T, %i> lc;\n", q);
    tprint("for( int i = 0; i < EXPANSION_SIZE; i ++ ) {\n");
    indent();
    tprint("lb[i] = 0.0f;\n");
    deindent();
    tprint("}\n");
    tprint("for( int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE ) {\n");
    indent();
    tprint("lb[i] = la[i];\n");
    deindent();
    tprint("}\n");
    flops += computeDxTensor(P);
    flops += constReferenceTrlessTensor<P>("la", "lc");
    flops += 4 * entries.size();
    flops += 3;
    tprint("for( int i = tid; i < %i; i+=WARP_SIZE) {\n",
        entries1.size() - 1 + (entries1.size() == entries2.size() ? 1 : 0));
    indent();
    tprint("lb[ldest1[i]] = fmaf(factor1[i] * dx[xsrc1[i]], lc[lsrc1[i]], lb[ldest1[i]]);\n");
    deindent();
    tprint("}\n");
    tprint("if( doPhi ) {\n");
    indent();
    tprint("for( int i = tid; i < %i; i+=WARP_SIZE) {\n", phiEntries.size());
    indent();
    tprint("lb[0] = fmaf(phiFactor[i] * dx[phiLsrc[i]], lc[phiLsrc[i]], lb[0]);\n");
    deindent();
    tprint("}\n");
    deindent();
    tprint("}\n");

    tprint("for (int p = warpSize / 2; p >= 1; p /= 2) {\n");
    indent();
    tprint("for (int i = 0; i < EXPANSION_SIZE; i++) {\n");
    indent();
    tprint("lb[i] += __shfl_xor_sync(0xffffffff, lb[i], p);\n");
    deindent();
    tprint("}\n");
    deindent();
    tprint("}\n");

    tprint("return lb;\n");
    printf("/* FLOPS = %i + doPhi * %i*/\n", flops, (int) (4 * phiEntries.size()));
    deindent();
    tprint("}\n");
    tprint("#endif\n");
}

void ewald(int directFlops) {
    tprint("template<class T>\n");
    tprint("CUDA_EXPORT int ewaldGreensFunction(SymmetricTracelessTensor<T,%i> &d, Array<T, NDIM> "
           "position) "
           "{\n",
        P);
    indent();
    tprint("position[0] *= T(SCALE_FACTOR);\n");
    tprint("position[1] *= T(SCALE_FACTOR);\n");
    tprint("position[2] *= T(SCALE_FACTOR);\n");
    tprint("EwaldConst econst;\n");
    tprint("FlopCounter<int> flops = %i;\n", 7);
    tprint("T r = sqrt(fmaf(position[0], position[0], fmaf(position[1], position[1], "
           "sqr(position[2]))));\n");    // 6
    tprint("T const fouroversqrtpi = T(%.9e);\n", 4.0 / sqrt(M_PI));
    tprint("SymmetricTensor<T, %i> dreal;\n", P);
    tprint("SymmetricTracelessTensor<T,%i> dfour;\n", P);
    tprint("dreal = 0.0f;\n");
    tprint("dfour = 0.0f;\n");
    tprint("d = 0.0f;\n");
    tprint("auto const realsz = econst.nreal();\n");
    tprint("T const zeroMask = r > T(0.0);\n");    // 1
    int theseFlops = 0;

    tprint("int icnt = 0;\n");

    tprint("Array<T, NDIM> dx;\n");
    tprint("dx = position;\n");
    tprint("{\n");                                                             // 1
    tprint("T r2 = fmaf(dx[0], dx[0], fmaf(dx[1], dx[1], sqr(dx[2])));\n");    // 5
    indent();
    tprint("icnt++;\n");
    tprint("T const r = sqrt(r2);\n");                           // 1
    tprint("T const n8r = T(-8*SCALE_FACTOR_INV2) * r;\n");      // 1
    tprint("T const rinv = (r > T(0)) / max(r, 1.0e-20);\n");    // 2
    tprint("T exp0 = expnearzero( -T(4*SCALE_FACTOR_INV2) * r2 );\n");
    tprint("T erf0 = erfnearzero(T(2*SCALE_FACTOR_INV1) * r);\n");
    tprint("T const expfactor = T(2.256758334191025*SCALE_FACTOR_INV1) * exp0;\n");    // 1
    tprint("T e0 = expfactor * rinv;\n");                                              // 1
    tprint("T const rinv0 = T(1);\n");                                                 // 2
    tprint("T const rinv1 = rinv;\n");                                                 // 2
    for (int l = 2; l < (P + 1) / 2; l++) {
        int const i = l / 2;
        int const j = l - i;
        theseFlops++;
        tprint("T const rinv%i = rinv%i * rinv%i;\n", l, i, j);    // (P-2)
    }
    tprint("T const d0 = erf0 * rinv;\n");    // 2
    for (int l = 1; l < P; l++) {
        tprint("T const d%i = fmaf(T(%i) * d%i, rinv, e0);\n", l, -2 * l + 1, l - 1);    // (P-1)*4
        if (l != P - 1) {
            tprint("e0 *= n8r;\n");    // (P-1)
        }
    }
    for (int l = 0; l < P; l++) {
        for (int m = 0; m <= l; m++) {
            if (l + m < P) {
                tprint("T const dRinvPower%iOrder%i = d%i * rinv%i;\n", l, m, l, m);
                theseFlops++;
            }
        }
    }
    tprint("Array<T,NDIM> dxrinv;\n");
    tprint("dxrinv[0] = dx[0] * rinv;\n");
    tprint("dxrinv[1] = dx[1] * rinv;\n");
    tprint("dxrinv[2] = dx[2] * rinv;\n");
    theseFlops += computeDx(P, "dxrinv", false, true);
    theseFlops += 37 + 7 * (P - 1) + P * (P + 1) / 2;

    theseFlops += computeDetraceEwald<P>("x", "dreal");
    tprint("auto const drz = econst.d0();\n");
    tprint("for( int i = 0; i < %i; i++) {\n", (P + 2) * (P + 1) * P / 6);
    indent();
    tprint("dreal[i] *= zeroMask;\n");
    tprint("dreal[i] -= (T(1) - zeroMask) * drz[i];\n");
    deindent();
    tprint("}\n");
    deindent();
    tprint("}\n");
    tprint("flops += icnt * %i;\n", theseFlops);

    tprint("for (int i = 0; i < realsz; i++) {\n");
    indent();
    tprint("auto const n = econst.realIndex(i);\n");
    tprint("Array<T, NDIM> dx;\n");
    tprint("for (int dim = 0; dim < NDIM; dim++) {\n");
    indent();
    tprint("dx[dim] = position[dim] - T(SCALE_FACTOR) * n[dim];\n");    // 3
    deindent();
    tprint("}\n");
    tprint("T r2 = fmaf(dx[0], dx[0], fmaf(dx[1], dx[1], sqr(dx[2])));\n");    // 5
    tprint("if (anytrue(r2 < T(SCALE_FACTOR2*EWALD_REAL_CUTOFF2))) {\n");      // 1
    indent();
    tprint("icnt++;\n");                                         // 1
    tprint("T const r = sqrt(r2);\n");                           // 1
    tprint("T const n8r = T(-8 * SCALE_FACTOR_INV2) * r;\n");    // 1
    tprint("T const rinv = (r > T(0)) / max(r, 1.0e-20);\n");    // 2
    tprint("T exp0;\n");
    tprint("T erfc0;\n");
    tprint("erfcexp(T(2.*SCALE_FACTOR_INV1) * r, &erfc0, &exp0);\n");                  // 20
    tprint("T const expfactor = fouroversqrtpi  * T(SCALE_FACTOR_INV1) * exp0;\n");    // 1
    tprint("T e0 = expfactor * rinv;\n");                                              // 1
    tprint("T const rinv0 = T(1);\n");                                                 // 2
    tprint("T const rinv1 = rinv;\n");                                                 // 2
    for (int l = 2; l < (P + 1) / 2; l++) {
        int const i = l / 2;
        int const j = l - i;
        theseFlops++;
        tprint("T const rinv%i = rinv%i * rinv%i;\n", l, i, j);    // (P-2)
    }
    tprint("T const d0 = -erfc0 * rinv;\n");    // 2
    for (int l = 1; l < P; l++) {
        tprint("T const d%i = fmaf(T(%i) * d%i, rinv, e0);\n", l, -2 * l + 1, l - 1);    // (P-1)*4
        if (l != P - 1) {
            tprint("e0 *= n8r;\n");    // (P-1)
        }
    }
    for (int l = 0; l < P; l++) {
        for (int m = 0; m <= l; m++) {
            if (l + m < P) {
                tprint("T const dRinvPower%iOrder%i = d%i * rinv%i;\n", l, m, l, m);
                theseFlops++;
            }
        }
    }
    tprint("Array<T,NDIM> dxrinv;\n");
    tprint("dxrinv[0] = dx[0] * rinv;\n");
    tprint("dxrinv[1] = dx[1] * rinv;\n");
    tprint("dxrinv[2] = dx[2] * rinv;\n");
    theseFlops += computeDx(P, "dxrinv");
    Array<int, NDIM> m;
    Array<int, NDIM> k;
    Array<int, NDIM> n;
    theseFlops += 37 + 7 * (P - 1) + P * (P + 1) / 2;

    theseFlops += computeDetraceEwald<P>("x", "dreal");

    deindent();
    tprint("}\n");
    deindent();
    tprint("}\n");

    tprint("auto const foursz = econst.nfour();\n");
    tprint("for (int i = 0; i < foursz; i++) {\n");
    indent();
    tprint("auto const &h = econst.fourIndex(i);\n");
    tprint("auto const& d0Value = econst.fourExpansion(i);\n");
    tprint("T const hdotx = fmaf(h[0], position[0], fmaf(h[1], position[1], h[2] * "
           "position[2]));\n");    // 5
    tprint("T cn, sn;\n");
    tprint("sincos(T(2.0 * M_PI * SCALE_FACTOR_INV1) * hdotx, &sn, &cn);\n");    // 35
    theseFlops = 40;
    bool iscos[P * P + 1];
    for (k[0] = 0; k[0] < P; k[0]++) {
        for (k[1] = 0; k[1] < P - k[0]; k[1]++) {
            int const zmax = (k[0] == 0 && k[1] == 0) ? intmin(3, P) : intmin(P - k[0] - k[1], 2);
            for (k[2] = 0; k[2] < zmax; k[2]++) {
                int const k0 = k[0] + k[1] + k[2];
                iscos[trlessIndex(k[0], k[1], k[2], P)] = (k0 % 2 == 0);
            }
        }
    }
    int maxi = (P * P + 1);
    for (int i = 0; i < maxi; i++) {
        int j = i + maxi;
        tprint("dfour[%i] = fmaf(%cn, d0Value[%i], dfour[%i]);\n", i, iscos[i] ? 'c' : 's', i, i);
        theseFlops += 2;
    }

    deindent();
    tprint("}\n");
    //	reference_sym("Dreal", P);
    //	reference_trless("D", P);
    int thoseFlops = computeDetrace<P>("dreal", "d", 'd');
    thoseFlops += 16 + 3 * (P * P + 1);
    tprint("flops += %i * foursz + %i;\n", theseFlops, thoseFlops + P * P + 1);
    tprint("d = d + dfour;\n");                                             // P*P+1
    tprint("d[0] = T(%.9e * SCALE_FACTOR_INV1) + d[0]; \n", M_PI / 4.0);    // 1
    tprint("return flops;\n");
    deindent();
    tprint("}\n");
}

int main() {
    fprintf(stderr, "Generating FMM kernels for Pmax = %i\n", P - 1);

    int flops = 0;

    tprint("/*\n"
           "CosmicTiger - A cosmological N-Body code\n"
           "Copyright (C) 2026  Dominic C. Marcello\n"
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

    tprint("#pragma once\n\n");
    tprint("#include <cosmictiger/cuda.hpp>\n");
    tprint("#include <cosmictiger/ewald_indices.hpp>\n");
    tprint("#include <cosmictiger/float2double.hpp>\n");
    tprint("#include <cosmictiger/flops.hpp>\n");
    tprint("#include <cosmictiger/math.hpp>\n");
    tprint("#include <cosmictiger/tensor.hpp>\n\n");
    tprint("template<class T>\n");
    tprint("using Expansion = SymmetricTracelessTensor<T,%i>;\n", P);
    tprint("template<class T>\n");
    tprint("using Expansion2 = SymmetricTracelessTensor<T,%i>;\n", 2);
    tprint("template<class T>\n");
    tprint("using Multipole = SymmetricTracelessTensor<T,%i>;\n", P - 1);
    tprint("#define EXPANSION_SIZE %i\n", P * P + 1);
    tprint("#define MULTIPOLE_SIZE %i\n", (P - 1) * (P - 1) + 1);
    auto const maxval = 1000.0;
    tprint("#define SCALE_FACTOR %ef\n", maxval);
    for (int n = 0; n <= P; n++) {
        tprint("#define SCALE_FACTOR%i %ef\n", n, pow(maxval, n));
        tprint("#define SCALE_FACTOR_INV%i %ef\n", n, 1.0 / pow(maxval, n));
    }

    tprint("\n\ntemplate<class T>\n");
    tprint("CUDA_EXPORT\n");
    tprint("inline int greensFunction(SymmetricTracelessTensor<T, %i>& d, Array<T, NDIM> position, "
           "bool "
           "scale = true) {\n",
        P);
    flops = 0;
    indent();
    tprint("if( scale ) {\n");
    indent();
    tprint("position[0] *= T(SCALE_FACTOR);\n");
    tprint("position[1] *= T(SCALE_FACTOR);\n");
    tprint("position[2] *= T(SCALE_FACTOR);\n");
    deindent();
    tprint("}\n");
    tprint("auto r2 = sqr(position[0], position[1], position[2]);\n");
    tprint("r2 = sqr(position[0], position[1], position[2]);\n");
    tprint("T const r = sqrt(r2);\n");
    tprint("T const rinv1 = -(r > T(0)) / max(r, T(1e-20));\n");
    for (int i = 1; i < P; i++) {
        int const j = i / 2;
        int const k = i - j;
        tprint("T const rinv%i = -rinv%i * rinv%i;\n", i + 1, j + 1, k);
    }
    tprint("position[0] *= rinv1;\n");
    tprint("position[1] *= rinv1;\n");
    tprint("position[2] *= rinv1;\n");
    flops += 12;
    flops += computeDx(P, "position", true);
    //	reference_trless("D", P);
    flops += computeDetraceD<P>("x", "d");
    flops += 11 + (P - 1) * 2;
    Array<int, NDIM> k;
    for (k[0] = 0; k[0] < P; k[0]++) {
        for (k[1] = 0; k[1] < P - k[0]; k[1]++) {
            int const zmax = (k[0] == 0 && k[1] == 0) ? intmin(3, P) : intmin(P - k[0] - k[1], 2);
            for (k[2] = 0; k[2] < zmax; k[2]++) {
                int const k0 = k[0] + k[1] + k[2];
                tprint("d[%i] *= rinv%i;\n", trlessIndex(k[0], k[1], k[2], P), k0 + 1);
                flops++;
            }
        }
    }
    tprint("return %i + scale * NDIM;\n", flops);
    deindent();
    tprint("}\n");

    ewald(flops);

    Array<int, NDIM> n;
    Array<int, NDIM> m;
    for (int pmax = 2; pmax <= P; pmax += P - 2) {
        flops = 0;
        tprint("\n\ntemplate<class T>\n");
        tprint("CUDA_EXPORT\n");
        tprint("inline int m2L(SymmetricTracelessTensor<T, %i>& l, SymmetricTracelessTensor<T, %i> "
               "const& m, "
               "SymmetricTracelessTensor<T, %i> const& d, bool doPhi) {\n",
            pmax, P - 1, P);
        indent();
        int phiFlops = 0;
        flops += constReferenceTrless<P - 1>("m");
        flops += constReferenceTrless<P>("d");
        n[0] = n[1] = n[2] = 0;
        int const n0 = 0;
        int const q0 = intmin(P - n0, P - 1);
        struct EntryType {
            Array<int, NDIM> n;
            Array<int, NDIM> m;
            double coeff;
        };
        Vector<Vector<EntryType>> entries;
        entries.resize(pmax <= 2 ? pmax * pmax : pmax * pmax + 1);
        for (n[0] = 0; n[0] < pmax; n[0]++) {
            for (n[1] = 0; n[1] < pmax - n[0]; n[1]++) {
                int const nzmax =
                    (n[0] == 0 && n[1] == 0) ? intmin(3, pmax) : intmin(pmax - n[0] - n[1], 2);
                for (n[2] = 0; n[2] < nzmax; n[2]++) {
                    int const n0 = n[0] + n[1] + n[2];
                    int const q0 = intmin(P - n0, P - 1);
                    for (m[0] = 0; m[0] < q0; m[0]++) {
                        for (m[1] = 0; m[1] < q0 - m[0]; m[1]++) {
                            for (m[2] = 0; m[2] < q0 - m[0] - m[1]; m[2]++) {
                                double const coeff = 1.0 / vfactorial(m);
                                EntryType e;
                                e.coeff = coeff;
                                e.n = n;
                                e.m = m;
                                int const index = trlessIndex(n[0], n[1], n[2], pmax);
                                entries[index].push_back(e);
                            }
                        }
                    }
                }
            }
        }
        auto const sortFunc = [pmax](EntryType a, EntryType b) {
            if (a.coeff < b.coeff) {
                return true;
            } else {
                return false;
            }
        };
        for (auto& e : entries) {
            std::sort(e.begin(), e.end(), sortFunc);
        }
        for (int i = 0; i < 1; i++) {
            double lastCoeff = 1.0;
            auto const n = entries[i][0].n;
            int const nindex = trlessIndex(n[0], n[1], n[2], pmax);
            auto& fl = phiFlops;
            tprint("if( doPhi ) {\n");
            indent();
            for (int j = 0; j < entries[i].size(); j++) {
                auto const coeff = entries[i][j].coeff;
                auto const m = entries[i][j].m;
                if (!close21(lastCoeff / coeff)) {
                    double factor = lastCoeff / coeff;
                    tprint("l[%i] *= T(%.9e);\n", nindex, factor);
                    lastCoeff = coeff;
                    fl++;
                }
                tprint("l[%i] = fmaf(m%i%i%i, d%i%i%i, l[%i]);\n", nindex, m[0], m[1], m[2],
                    n[0] + m[0], n[1] + m[1], n[2] + m[2], trlessIndex(n[0], n[1], n[2], pmax));
                fl += 2;
            }
            if (!close21(lastCoeff)) {
                fl++;
                tprint("l[%i] *= T(%.9e);\n", nindex, lastCoeff);
            }
            if (nindex == 0) {
                deindent();
                tprint("}\n");
            }
        }
        int totalSize = 0;
        for (int i = 1; i < entries.size(); i++) {
            totalSize += entries[i].size();
        }
        int halfSize = 0;
        int mid;
        for (int i = 1; i < entries.size(); i++) {
            halfSize += entries[i].size();
            if (halfSize >= totalSize / 2) {
                mid = i;
                break;
            }
        }
        Vector<std::string> cmds1, cmds2;
        for (int i = 1; i < entries.size(); i++) {
            double lastCoeff = 1.0;
            auto const n = entries[i][0].n;
            int const nindex = trlessIndex(n[0], n[1], n[2], pmax);
            auto& fl = flops;
            auto& cmds = i >= mid ? cmds2 : cmds1;
            char* str;
            for (int j = 0; j < entries[i].size(); j++) {
                auto const coeff = entries[i][j].coeff;
                auto const m = entries[i][j].m;
                if (!close21(lastCoeff / coeff)) {
                    double factor = lastCoeff / coeff;
                    ASPRINTF(&str, "l[%i] *= T(%.9e);\n", nindex, factor);
                    cmds.push_back(str);
                    free(str);
                    lastCoeff = coeff;
                    fl++;
                }
                ASPRINTF(&str, "l[%i] = fmaf(m%i%i%i, d%i%i%i, l[%i]);\n", nindex, m[0], m[1], m[2],
                    n[0] + m[0], n[1] + m[1], n[2] + m[2], trlessIndex(n[0], n[1], n[2], pmax));
                cmds.push_back(str);
                free(str);
                fl += 2;
            }
            if (!close21(lastCoeff)) {
                fl++;
                ASPRINTF(&str, "l[%i] *= T(%.9e);\n", nindex, lastCoeff);
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
        tprint("return %i + doPhi * %i;\n", flops, phiFlops);
        deindent();
        tprint("}\n");
    }

    tprint("\n\ntemplate<class T>\n");
    tprint("CUDA_EXPORT\n");
    tprint("SymmetricTracelessTensor<T, %i> p2M(Array<T, NDIM>& position) {\n", P - 1);
    flops = 0;
    indent();
    tprint("SymmetricTracelessTensor<T, %i> m;\n", P - 1);
    tprint("position[0] *= -T(SCALE_FACTOR);\n");
    tprint("position[1] *= -T(SCALE_FACTOR);\n");
    tprint("position[2] *= -T(SCALE_FACTOR);\n");
    //	reference_trless("M", P - 1);
    flops += 6;
    flops += computeDx(P - 1);
    flops += computeDetrace<P - 1>("x", "m", 'd');
    tprint("return m;\n");
    printf("/* FLOPS = %i*/\n", flops);
    deindent();
    tprint("}\n");

    tprint("\n\ntemplate<class T>\n");
    tprint("CUDA_EXPORT\n");
    tprint("SymmetricTracelessTensor<T, %i> m2M(SymmetricTracelessTensor<T,%i> const& ma, Array<T, "
           "NDIM>& position) {\n",
        P - 1, P - 1);
    flops = 0;
    indent();
    tprint("SymmetricTensor<T, %i> mb;\n", P - 1);
    tprint("SymmetricTracelessTensor<T, %i> mc;\n", P - 1);
    tprint("position[0] *= -T(SCALE_FACTOR);\n");
    tprint("position[1] *= -T(SCALE_FACTOR);\n");
    tprint("position[2] *= -T(SCALE_FACTOR);\n");
    flops += 6;
    flops += constReferenceTrless<P - 1>("ma");
    //	reference_sym("Mb", P - 1);
    //	reference_trless("Mc", P - 1);
    flops += computeDx(P - 1);

    for (int i = 0; i < (P - 1) * P * (P + 1) / 6; i++) {
        for (n[0] = 0; n[0] < P - 1; n[0]++) {
            for (n[1] = 0; n[1] < P - n[0] - 1; n[1]++) {
                for (n[2] = 0; n[2] < P - n[0] - n[1] - 1; n[2]++) {
                    if (i == symIndex(n[0], n[1], n[2])) {
                        tprint("mb[%i] = ma%i%i%i;\n", i, n[0], n[1], n[2]);
                    }
                }
            }
        }
    }
    struct Mentry {
        Array<int, NDIM> n;
        Array<int, NDIM> k;
        double factor;
    };
    Vector<Vector<Mentry>> mentries((P - 1) * P * (P + 1) / 6);
    for (int n0 = 0; n0 <= P - 2; n0++) {
        for (n[0] = 0; n[0] <= n0; n[0]++) {
            for (n[1] = 0; n[1] <= n0 - n[0]; n[1]++) {
                n[2] = n0 - n[0] - n[1];
                for (k[0] = 0; k[0] <= intmin(n0, n[0]); k[0]++) {
                    for (k[1] = 0; k[1] <= intmin(n0 - k[0], n[1]); k[1]++) {
                        for (k[2] = 0; k[2] <= intmin(n0 - k[0] - k[1], n[2]); k[2]++) {
                            auto const factor =
                                (vfactorial(n)) / double(vfactorial(k) * vfactorial(n - k));
                            if (n != k) {
                                Mentry e;
                                e.n = n;
                                e.k = k;
                                e.factor = factor;
                                int const index = symIndex(n[0], n[1], n[2]);
                                mentries[index].push_back(e);
                            }
                        }
                    }
                }
            }
        }
    }
    for (auto& m : mentries) {
        std::sort(m.begin(), m.end(), [](Mentry a, Mentry b) { return a.factor > b.factor; });
    }
    int totalSize = 0;
    for (int i = 0; i < mentries.size(); i++) {
        totalSize += mentries[i].size();
    }
    int halfSize = 0;
    int mid;
    for (int i = 0; i < mentries.size(); i++) {
        if (halfSize >= totalSize / 2) {
            mid = i;
            break;
        }
        halfSize += mentries[i].size();
    }
    Vector<std::string> cmds1, cmds2;
    char* str;
    for (int i = 0; i < mentries.size(); i++) {
        auto& cmds = i < mid ? cmds1 : cmds2;
        double lastFactor = 1.0;
        if (mentries[i].size()) {
            auto const n = mentries[i][0].n;
            int const nindex = symIndex(n[0], n[1], n[2]);
            for (int j = 0; j < mentries[i].size(); j++) {
                auto const k = mentries[i][j].k;
                auto const factor = mentries[i][j].factor;
                if (!close21(lastFactor / factor)) {
                    ASPRINTF(&str, "mb[%i] *= T(%.9e);\n", nindex, lastFactor / factor);
                    cmds.push_back(str);
                    flops++;
                    free(str);
                    lastFactor = factor;
                }
                ASPRINTF(&str, "mb[%i] = fmaf( x[%i], ma%i%i%i, mb[%i]);\n", nindex,
                    symIndex(n[0] - k[0], n[1] - k[1], n[2] - k[2]), k[0], k[1], k[2],
                    symIndex(n[0], n[1], n[2]));
                cmds.push_back(str);
                free(str);
                flops += 2;
            }
            if (!close21(lastFactor)) {
                ASPRINTF(&str, "mb[%i] *= T(%.9e);\n", nindex, lastFactor);
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

    flops += computeDetrace<P - 1>("mb", "mc", 'd');

    tprint("return mc;\n");
    printf("/* FLOPS = %i*/\n", flops);
    deindent();
    tprint("}\n");

    doExpansion<P>(false);

    doExpansion<2>(true);

#ifdef USE_CUDA
    doExpansionCuda<P>();
#endif

    tprint("template<class T>\n");
    tprint("CUDA_EXPORT int applyScaleFactorInv(SymmetricTracelessTensor<T,%i> &l) {\n", P);
    indent();
    applyScaleFactorL<P>("l");
    tprint("return %i;\n", P * P + 1);
    deindent();
    tprint("}\n");

    tprint("template<class T>\n");
    tprint("CUDA_EXPORT int applyScaleFactor(SymmetricTracelessTensor<T,%i> &m) {\n", P - 1);
    indent();
    applyScaleFactorM<P - 1>("m");
    tprint("return %i;\n", (P - 1) * (P - 1) + 1);

    deindent();
    tprint("}\n");
}
