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

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/math/tensor.hpp>

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
    tprint("x[%i] = simdBroadcast<T>(1);\n", index(0, 0, 0));
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
                                        "T %sOrder%iTrace%iIndex%i%i%i = simdBroadcast<T>(%.9e) * "
                                        "%s[%i];\n",
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
                                        "%sOrder%iTrace%iIndex%i%i%i = "
                                        "simdFma(simdBroadcast<T>(%.9e), %s[%i], "
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
                                            "%s[%i] = simdBroadcast<T>(%.9e) * "
                                            "%sOrder%iTrace%iIndex%i%i%i;\n",
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
                                        ASPRINTF(&str,
                                            "%s[%i] = simdBroadcast<T>(%.9e) * %s[%i];\n",
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
                                            "%s[%i] = simdFma(simdBroadcast<T>(%.9e), "
                                            "%sOrder%iTrace%iIndex%i%i%i, "
                                            "%s[%i]);\n",
                                            oname.c_str(), trlessIndex(n[0], n[1], n[2], pValue),
                                            factor, iname.c_str(), n0, m0, p[0], p[1], p[2],
                                            oname.c_str(), trlessIndex(n[0], n[1], n[2], pValue));
                                        flops += 2;
                                    } else {
                                        ASPRINTF(&str,
                                            "%s[%i] = simdFma(simdBroadcast<T>(%.9e), %s[%i], "
                                            "%s[%i]);\n",
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
                                    "%s[%i] = simdFma(%s[%i], dRinvPower%iOrder%i, %s[%i]);\n",
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
                                    "%s[%i] = simdFma(simdBroadcast<T>(%.9e), "
                                    "%s[%i]*dRinvPower%iOrder%i, %s[%i]);\n",
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
                    cmd += "simdFma(";
                } else {
                    cmd += "(";
                }
                cmd += "simdBroadcast<T>(";
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

// Emit the Ewald correction derivatives for one FMM order P (tensor ranks 0..P-1).
template <int P>
void ewald() {
    tprint("// FMM order %i.\n", P);
    tprint("template<class T>\n");
    tprint("CUDA_EXPORT int ewaldCorrectionDerivatives(SymmetricTracelessTensor<T,%i> &d, Array<T, "
           "NDIM> "
           "position) "
           "{\n",
        P);
    indent();
    tprint("position[0] *= simdBroadcast<T>(SCALE_FACTOR);\n");
    tprint("position[1] *= simdBroadcast<T>(SCALE_FACTOR);\n");
    tprint("position[2] *= simdBroadcast<T>(SCALE_FACTOR);\n");
    tprint("FlopCounter<int> flops = %i;\n", 7);
    tprint("T r = simdSqrt(simdFma(position[0], position[0], simdFma(position[1], position[1], "
           "sqr(position[2]))));\n");    // 6
    tprint("T const fouroversqrtpi = simdBroadcast<T>(%.9e);\n", 4.0 / sqrt(M_PI));
    tprint("SymmetricTensor<T, %i> dreal;\n", P);
    tprint("SymmetricTracelessTensor<T,%i> dfour;\n", P);
    tprint("dreal = 0.0f;\n");
    tprint("dfour = 0.0f;\n");
    tprint("d = 0.0f;\n");
    tprint("auto const realsz = EwaldConst::nreal();\n");
    tprint("auto const zeroMask = r > simdBroadcast<T>(0.0);\n");    // 1
    int theseFlops = 0;

    tprint("int icnt = 0;\n");

    tprint("Array<T, NDIM> dx;\n");
    tprint("dx = position;\n");
    tprint("{\n");                                                                   // 1
    tprint("T r2 = simdFma(dx[0], dx[0], simdFma(dx[1], dx[1], sqr(dx[2])));\n");    // 5
    indent();
    tprint("icnt++;\n");
    tprint("T const r = simdSqrt(r2);\n");                                    // 1
    tprint("T const n8r = simdBroadcast<T>(-8*SCALE_FACTOR_INV2) * r;\n");    // 1
    tprint("T const rinv = simdSelect(r > simdBroadcast<T>(0), simdBroadcast<T>(1), "
           "simdBroadcast<T>(0)) / simdMax(r, simdBroadcast<T>(1.0e-20));\n");    // 2
    tprint("T exp0 = expnearzero( -simdBroadcast<T>(4*SCALE_FACTOR_INV2) * r2 );\n");
    tprint("T erf0 = erfnearzero(simdBroadcast<T>(2*SCALE_FACTOR_INV1) * r);\n");
    tprint(
        "T const expfactor = simdBroadcast<T>(2.256758334191025*SCALE_FACTOR_INV1) * exp0;\n");    // 1
    tprint("T e0 = expfactor * rinv;\n");                // 1
    tprint("T const rinv0 = simdBroadcast<T>(1);\n");    // 2
    tprint("T const rinv1 = rinv;\n");                   // 2
    for (int l = 2; l < (P + 1) / 2; l++) {
        int const i = l / 2;
        int const j = l - i;
        theseFlops++;
        tprint("T const rinv%i = rinv%i * rinv%i;\n", l, i, j);    // (P-2)
    }
    tprint("T const d0 = erf0 * rinv;\n");    // 2
    for (int l = 1; l < P; l++) {
        tprint("T const d%i = simdFma(simdBroadcast<T>(%i) * d%i, rinv, e0);\n", l, -2 * l + 1,
            l - 1);    // (P-1)*4
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
    tprint("auto const drz = EwaldConst::d0<%i>();\n", P);
    tprint("for( int i = 0; i < %i; i++) {\n", (P + 2) * (P + 1) * P / 6);
    indent();
    tprint("dreal[i] = simdSelect(zeroMask, dreal[i], -simdBroadcast<T>(drz[i]));\n");
    deindent();
    tprint("}\n");
    deindent();
    tprint("}\n");
    tprint("flops += icnt * %i;\n", theseFlops);

    tprint("for (int i = 0; i < realsz; i++) {\n");
    indent();
    tprint("auto const n = EwaldConst::realIndex(i);\n");
    tprint("Array<T, NDIM> dx;\n");
    tprint("for (int dim = 0; dim < NDIM; dim++) {\n");
    indent();
    tprint("dx[dim] = position[dim] - simdBroadcast<T>(SCALE_FACTOR) * n[dim];\n");    // 3
    deindent();
    tprint("}\n");
    tprint("T r2 = simdFma(dx[0], dx[0], simdFma(dx[1], dx[1], sqr(dx[2])));\n");             // 5
    tprint("if (simdAnyOf(r2 < simdBroadcast<T>(SCALE_FACTOR2*EWALD_REAL_CUTOFF2))) {\n");    // 1
    indent();
    tprint("icnt++;\n");                                                        // 1
    tprint("T const r = simdSqrt(r2);\n");                                      // 1
    tprint("T const n8r = simdBroadcast<T>(-8 * SCALE_FACTOR_INV2) * r;\n");    // 1
    tprint("T const rinv = simdSelect(r > simdBroadcast<T>(0), simdBroadcast<T>(1), "
           "simdBroadcast<T>(0)) / simdMax(r, simdBroadcast<T>(1.0e-20));\n");    // 2
    tprint("T exp0;\n");
    tprint("T erfc0;\n");
    tprint("erfcexp(simdBroadcast<T>(2.*SCALE_FACTOR_INV1) * r, &erfc0, &exp0);\n");    // 20
    tprint(
        "T const expfactor = fouroversqrtpi  * simdBroadcast<T>(SCALE_FACTOR_INV1) * exp0;\n");    // 1
    tprint("T e0 = expfactor * rinv;\n");                // 1
    tprint("T const rinv0 = simdBroadcast<T>(1);\n");    // 2
    tprint("T const rinv1 = rinv;\n");                   // 2
    for (int l = 2; l < (P + 1) / 2; l++) {
        int const i = l / 2;
        int const j = l - i;
        theseFlops++;
        tprint("T const rinv%i = rinv%i * rinv%i;\n", l, i, j);    // (P-2)
    }
    tprint("T const d0 = -erfc0 * rinv;\n");    // 2
    for (int l = 1; l < P; l++) {
        tprint("T const d%i = simdFma(simdBroadcast<T>(%i) * d%i, rinv, e0);\n", l, -2 * l + 1,
            l - 1);    // (P-1)*4
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

    tprint("auto const foursz = EwaldConst::nfour();\n");
    tprint("for (int i = 0; i < foursz; i++) {\n");
    indent();
    tprint("auto const &h = EwaldConst::fourIndex(i);\n");
    tprint("auto const& d0Value = EwaldConst::fourExpansion<%i>(i);\n", P);
    tprint("T const hdotx = simdFma(simdBroadcast<T>(h[0]), position[0], "
           "simdFma(simdBroadcast<T>(h[1]), position[1], simdBroadcast<T>(h[2]) * "
           "position[2]));\n");    // 5
    tprint("T cn, sn;\n");
    tprint(
        "sincos(simdBroadcast<T>(2.0 * M_PI * SCALE_FACTOR_INV1) * hdotx, &sn, &cn);\n");    // 35
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
        tprint("dfour[%i] = simdFma(%cn, simdBroadcast<T>(d0Value[%i]), dfour[%i]);\n", i,
            iscos[i] ? 'c' : 's', i, i);
        theseFlops += 2;
    }

    deindent();
    tprint("}\n");
    //	reference_sym("Dreal", P);
    //	reference_trless("D", P);
    int thoseFlops = computeDetrace<P>("dreal", "d", 'd');
    thoseFlops += 16 + 3 * (P * P + 1);
    tprint("flops += %i * foursz + %i;\n", theseFlops, thoseFlops + P * P + 1);
    tprint("d = d + dfour;\n");                                                            // P*P+1
    tprint("d[0] = simdBroadcast<T>(%.9e * SCALE_FACTOR_INV1) + d[0]; \n", M_PI / 4.0);    // 1
    tprint("return flops;\n");
    deindent();
    tprint("}\n");
}

int main() {
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

    tprint(
        "// Generated Ewald correction; STF translations come from tools/generate_stf_m2l.py.\n");
    tprint("#pragma once\n\n");
    tprint("#include <cosmictiger/fmm/ewald_indices.hpp>\n");
    tprint("#include <cosmictiger/fmm/fmm_scale.hpp>\n");
    tprint("#include <cosmictiger/core/flops.hpp>\n");
    tprint("#include <cosmictiger/math/math.hpp>\n");
    tprint("#include <cosmictiger/math/simd_math.hpp>\n");
    tprint("#include <cosmictiger/math/tensor.hpp>\n\n");
    tprint("namespace detail {\n");
    // One overload per supported FMM order; the caller's tensor order selects it.
    ewald<3>();
    ewald<4>();
    ewald<5>();
    ewald<6>();
    ewald<7>();
    ewald<8>();
    ewald<9>();
    tprint("} // namespace detail\n");
    return 0;
}
