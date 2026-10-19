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

#pragma once

#include <cosmictiger/core/assert.hpp>
#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>

inline double factorial(double n) {
    ASSERT(n >= 0);
    if (n == 0) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

inline int intmin(int a, int b) {
    return a < b ? a : b;
}

inline int dfactorial(int n) {
    ASSERT(n >= -1);
    if (n >= -1 && n <= 1) {
        return 1;
    } else {
        return n * dfactorial(n - 2);
    }
}

template <int nValue>
inline int vfactorial(Array<int, nValue> const& n) {
    return factorial(n[0]) * factorial(n[1]) * factorial(n[2]);
}

inline int n1pow(int n) {
    return (n & 1) ? -1 : 1;
}

template <class T, int pValue>
class SymmetricTracelessTensor : public Array<T, pValue * pValue + 1> {
private:
public:
    static constexpr int kN_ = pValue * pValue + 1;

    CUDA_EXPORT
    inline SymmetricTracelessTensor& operator=(T const& other) {
        for (int i = 0; i < kN_; i++) {
            (*this)[i] = other;
        }
        return *this;
    }

    CUDA_EXPORT
    inline SymmetricTracelessTensor operator+(SymmetricTracelessTensor const& other) const {
        SymmetricTracelessTensor<T, pValue> result;
        for (int i = 0; i < kN_; i++) {
            result[i] = other[i] + (*this)[i];
        }
        return result;
    }

    inline SymmetricTracelessTensor operator-(SymmetricTracelessTensor const& other) const {
        SymmetricTracelessTensor<T, pValue> result;
        for (int i = 0; i < kN_; i++) {
            result[i] = other[i] - (*this)[i];
        }
        return result;
    }

    CUDA_EXPORT
    inline T& operator()(int l, int m, int n) {
        l += m;
        ASSERT(l >= 0);
        ASSERT(m >= 0);
        ASSERT(n >= 0);
        ASSERT(l < pValue);
        ASSERT(m <= l);
        ASSERT(n <= 1 || (n == 2 && l == 0 && m == 0));
        return (*this)[l * (l + 1) / 2 + m + (pValue * (pValue + 1) / 2) * (n == 1) +
            (kN_ - 1) * (n == 2)];
    }

    CUDA_EXPORT
    inline T operator()(int l, int m, int n) const {
        if (n > 1) {
            if (l == 0 && m == 0 && n == 2) {
                return (*this)[kN_ - 1];
            } else {
                return -((*this)(l + 2, m, n - 2) + (*this)(l, m + 2, n - 2));
            }
        } else {
            l += m;
            ASSERT(l >= 0);
            ASSERT(m >= 0);
            ASSERT(n >= 0);
            ASSERT(l < pValue);
            ASSERT(m <= l);
            ASSERT(n <= 1);
            return (*this)[l * (l + 1) / 2 + m + (pValue * (pValue + 1) / 2) * n];
        }
    }

    T operator()(Array<int, NDIM> const& n) const {
        return (*this)(n[0], n[1], n[2]);
    }

    T& operator()(Array<int, NDIM> const& n) {
        return (*this)(n[0], n[1], n[2]);
    }
};

inline Vector<int> indicesBegin(int pValue2) {
    Vector<int> v;
    v.reserve(pValue2);
    for (int i = 0; i < pValue2; i++) {
        v.push_back(0);
    }
    return v;
}

inline bool indicesInc(Vector<int>& i) {
    if (i.size() == 0) {
        return false;
    }
    int j = 0;
    while (i[j] == NDIM - 1) {
        i[j] = 0;
        j++;
        if (j == i.size()) {
            i[0] = -1;
            return false;
        }
    }
    i[j]++;
    return true;
}

inline Vector<int> indicesEnd(int pValue2) {
    Vector<int> v;
    v.reserve(pValue2);
    v.push_back(-1);
    for (int i = 1; i < pValue2; i++) {
        v.push_back(0);
    }
    return v;
}

inline Array<int, NDIM> indicesToSym(Vector<int> const& indices) {
    Array<int, NDIM> j;
    j[0] = j[1] = j[2] = 0;
    for (int i = 0; i < indices.size(); i++) {
        j[indices[i]]++;
    }
    return j;
}

inline Vector<int> symToIndices(Array<int, NDIM> const& i) {
    Vector<int> indices;
    indices.reserve(i[0] + i[1] + i[2]);
    for (int dim = 0; dim < NDIM; dim++) {
        for (int j = 0; j < i[dim]; j++) {
            indices.push_back(dim);
        }
    }
    return indices;
}

template <class T, int pValue>
class SymmetricTensor : public Array<T, (pValue * (pValue + 1) * (pValue + 2)) / 6> {
private:
public:
    static constexpr int kN_ = (pValue * (pValue + 1) * (pValue + 2)) / 6;

    CUDA_EXPORT
    inline SymmetricTensor& operator=(T const& other) {
        for (int i = 0; i < kN_; i++) {
            (*this)[i] = other;
        }
        return *this;
    }

    inline SymmetricTensor operator+(SymmetricTensor const& other) const {
        SymmetricTensor<T, pValue> result;
        for (int i = 0; i < kN_; i++) {
            result[i] = other[i] + (*this)[i];
        }
        return result;
    }

    inline T operator()(int l, int m, int n) const {
        m += n;
        l += m;
        ASSERT(l >= 0);
        ASSERT(m >= 0);
        ASSERT(n >= 0);
        ASSERT(l < pValue);
        ASSERT(m <= l);
        ASSERT(n <= m);
        return (*this)[l * (l + 1) * (l + 2) / 6 + m * (m + 1) / 2 + n];
    }

    inline T& operator()(int l, int m, int n) {
        m += n;
        l += m;
        ASSERT(l >= 0);
        ASSERT(m >= 0);
        ASSERT(n >= 0);
        ASSERT(l < pValue);
        ASSERT(m <= l);
        ASSERT(n <= m);
        return (*this)[l * (l + 1) * (l + 2) / 6 + m * (m + 1) / 2 + n];
    }

    T operator()(Array<int, NDIM> const& n) const {
        return (*this)(n[0], n[1], n[2]);
    }

    T& operator()(Array<int, NDIM> const& n) {
        return (*this)(n[0], n[1], n[2]);
    }

    inline SymmetricTracelessTensor<T, pValue> detraceD() const {
        SymmetricTracelessTensor<T, pValue> aValue;
        SymmetricTensor<T, pValue> const& bValue = *this;
        Array<int, NDIM> m;
        Array<int, NDIM> k;
        Array<int, NDIM> n;
        for (n[0] = 0; n[0] < pValue; n[0]++) {
            for (n[1] = 0; n[1] < pValue - n[0]; n[1]++) {
                int const nzmax =
                    (n[0] == 0 && n[1] == 0) ? intmin(3, pValue) : intmin(pValue - n[0] - n[1], 2);
                for (n[2] = 0; n[2] < nzmax; n[2]++) {
                    aValue(n) = T(0);
                    int const n0 = n[0] + n[1] + n[2];
                    for (m[0] = 0; m[0] <= n[0] / 2; m[0]++) {
                        for (m[1] = 0; m[1] <= n[1] / 2; m[1]++) {
                            for (m[2] = 0; m[2] <= n[2] / 2; m[2]++) {
                                int const m0 = m[0] + m[1] + m[2];
                                T num =
                                    T(n1pow(m0) * dfactorial(2 * n0 - 2 * m0 - 1) * vfactorial(n));
                                T den = T((1 << m0) * vfactorial(m) * vfactorial(n - (m) * 2));
                                T const fnm = num / den;
                                if ((n0 == 2 && (n[0] == 2 || n[1] == 2 || n[2] == 2)) && m0 == 1) {
                                    continue;
                                }
                                for (k[0] = 0; k[0] <= m0; k[0]++) {
                                    for (k[1] = 0; k[1] <= m0 - k[0]; k[1]++) {
                                        k[2] = m0 - k[0] - k[1];
                                        auto const p = n - (m) * 2 + (k) * 2;
                                        num = factorial(m0);
                                        den = vfactorial(k);
                                        T const number = fnm * num / den;
                                        aValue(n) += number * bValue(p);
                                    }
                                }
                            }
                        }
                    }
                    aValue(n) /= T(dfactorial(2 * n0 - 1));
                }
            }
        }
        return aValue;
    }
};

template <class T, int pValue>

SymmetricTensor<T, pValue> vectorToSymTensor(Array<T, NDIM> const& vec) {
    SymmetricTensor<T, pValue> xValue;
    Array<int, NDIM> n;
    T x = T(1);
    for (n[0] = 0; n[0] < pValue; n[0]++) {
        T y = T(1);
        for (n[1] = 0; n[1] < pValue - n[0]; n[1]++) {
            T z = T(1);
            for (n[2] = 0; n[2] < pValue - n[0] - n[1]; n[2]++) {
                xValue(n) = x * y * z;
                z *= vec[2];
            }
            y *= vec[1];
        }
        x *= vec[0];
    }
    return xValue;
}

inline int trlessIndex(int l, int m, int n, int q) {
    return (l + m) * ((l + m) + 1) / 2 + (m) + (q * (q + 1) / 2) * (n == 1) + (q * q) * (n == 2);
}
