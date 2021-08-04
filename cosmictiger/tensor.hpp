#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/containers.hpp>

inline
int factorial(int n) {
	assert(n >= 0);
	if (n == 0) {
		return 1;
	} else {
		return n * factorial(n - 1);
	}
}

inline int intmin(int a, int b) {
	return a < b ? a : b;
}

inline
int dfactorial(int n) {
	assert(n >= -1);
	if (n >= -1 && n <= 1) {
		return 1;
	} else {
		return n * dfactorial(n - 2);
	}
}

template<int N>
inline
int vfactorial(const array<int, N>& n) {
	return factorial(n[0]) * factorial(n[1]) * factorial(n[2]);
}

inline
int n1pow(int n) {
	return (n & 1) ? -1 : 1;
}

template<class T, int P>
class tensor_trless_sym: public array<T, P * P + 1> {

private:

public:

	static constexpr int N = P * P + 1;

	CUDA_EXPORT
	inline tensor_trless_sym& operator=(const T& other) {
		for (int i = 0; i < N; i++) {
			(*this)[i] = other;
		}
		return *this;
	}

	CUDA_EXPORT
	inline tensor_trless_sym operator+(const tensor_trless_sym& other) const {
		tensor_trless_sym<T, P> result;
		for (int i = 0; i < N; i++) {
			result[i] = other[i] + (*this)[i];
		}
		return result;
	}

	inline tensor_trless_sym operator-(const tensor_trless_sym& other) const {
		tensor_trless_sym<T, P> result;
		for (int i = 0; i < N; i++) {
			result[i] = other[i] - (*this)[i];
		}
		return result;
	}

	CUDA_EXPORT
	inline T& operator()(int l, int m, int n) {
		l += m;
		assert(l >= 0);
		assert(m >= 0);
		assert(n >= 0);
		assert(l < P);
		assert(m <= l);
		assert(n <= 1 || (n == 2 && l == 0 && m == 0));
		return (*this)[l * (l + 1) / 2 + m + (P * (P + 1) / 2) * (n == 1) + (N - 1) * (n == 2)];
	}

	CUDA_EXPORT
	inline T operator()(int l, int m, int n) const {
		if (n > 1) {
			if (l == 0 && m == 0 && n == 2) {
				return (*this)[N - 1];
			} else {
				return -((*this)(l + 2, m, n - 2) + (*this)(l, m + 2, n - 2));
			}
		} else {
			l += m;
			assert(l >= 0);
			assert(m >= 0);
			assert(n >= 0);
			assert(l < P);
			assert(m <= l);
			assert(n <= 1);
			return (*this)[l * (l + 1) / 2 + m + (P * (P + 1) / 2) * n];
		}
	}

	T operator()(const array<int, NDIM>& n) const {
		return (*this)(n[0], n[1], n[2]);
	}

	T& operator()(const array<int, NDIM>& n) {
		return (*this)(n[0], n[1], n[2]);
	}

};

inline vector<int> indices_begin(int P) {
	vector<int> v;
	v.reserve(P);
	for (int i = 0; i < P; i++) {
		v.push_back(0);
	}
	return v;
}

inline bool indices_inc(vector<int>& i) {
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

inline vector<int> indices_end(int P) {
	vector<int> v;
	v.reserve(P);
	v.push_back(-1);
	for (int i = 1; i < P; i++) {
		v.push_back(0);
	}
	return v;

}

inline array<int, NDIM> indices_to_sym(const vector<int>& indices) {
	array<int, NDIM> j;
	j[0] = j[1] = j[2] = 0;
	for (int i = 0; i < indices.size(); i++) {
		j[indices[i]]++;
	}
	return j;
}

inline vector<int> sym_to_indices(const array<int, NDIM>& i) {
	vector<int> indices;
	indices.reserve(i[0] + i[1] + i[2]);
	for (int dim = 0; dim < NDIM; dim++) {
		for (int j = 0; j < i[dim]; j++) {
			indices.push_back(dim);
		}
	}
	return indices;
}

template<class T, int P>
class tensor_sym: public array<T, (P * (P + 1) * (P + 2)) / 6> {

private:

public:

	static constexpr int N = (P * (P + 1) * (P + 2)) / 6;

	CUDA_EXPORT
	inline tensor_sym& operator=(const T& other) {
		for (int i = 0; i < N; i++) {
			(*this)[i] = other;
		}
		return *this;
	}

	inline tensor_sym operator+(const tensor_sym& other) const {
		tensor_sym<T, P> result;
		for (int i = 0; i < N; i++) {
			result[i] = other[i] + (*this)[i];
		}
		return result;
	}

	inline T operator()(int l, int m, int n) const {
		m += n;
		l += m;
		assert(l >= 0);
		assert(m >= 0);
		assert(n >= 0);
		assert(l < P);
		assert(m <= l);
		assert(n <= m);
		return (*this)[l * (l + 1) * (l + 2) / 6 + m * (m + 1) / 2 + n];
	}

	inline T& operator()(int l, int m, int n) {
		m += n;
		l += m;
		assert(l >= 0);
		assert(m >= 0);
		assert(n >= 0);
		assert(l < P);
		assert(m <= l);
		assert(n <= m);
		return (*this)[l * (l + 1) * (l + 2) / 6 + m * (m + 1) / 2 + n];
	}

	T operator()(const array<int, NDIM>& n) const {
		return (*this)(n[0], n[1], n[2]);
	}

	T& operator()(const array<int, NDIM>& n) {
		return (*this)(n[0], n[1], n[2]);
	}

	inline tensor_trless_sym<T, P> detraceD() const {
		tensor_trless_sym<T, P> A;
		const tensor_sym<T, P>& B = *this;
		array<int, NDIM> m;
		array<int, NDIM> k;
		array<int, NDIM> n;
		for (n[0] = 0; n[0] < P; n[0]++) {
			for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
				const int nzmax = (n[0] == 0 && n[1] == 0) ? intmin(3, P) : intmin(P - n[0] - n[1], 2);
				for (n[2] = 0; n[2] < nzmax; n[2]++) {
					A(n) = T(0);
					const int n0 = n[0] + n[1] + n[2];
					for (m[0] = 0; m[0] <= n[0] / 2; m[0]++) {
						for (m[1] = 0; m[1] <= n[1] / 2; m[1]++) {
							for (m[2] = 0; m[2] <= n[2] / 2; m[2]++) {
								const int m0 = m[0] + m[1] + m[2];
								T num = T(n1pow(m0) * dfactorial(2 * n0 - 2 * m0 - 1) * vfactorial(n));
								T den = T((1 << m0) * vfactorial(m) * vfactorial(n - (m) * 2));
								const T fnm = num / den;
								if ((n0 == 2 && (n[0] == 2 || n[1] == 2 || n[2] == 2)) && m0 == 1) {
									continue;
								}
								for (k[0] = 0; k[0] <= m0; k[0]++) {
									for (k[1] = 0; k[1] <= m0 - k[0]; k[1]++) {
										k[2] = m0 - k[0] - k[1];
										const auto p = n - (m) * 2 + (k) * 2;
										num = factorial(m0);
										den = vfactorial(k);
										const T number = fnm * num / den;
										A(n) += number * B(p);
									}
								}
							}
						}
					}
					A(n) /= T(dfactorial(2 * n0 - 1));
				}
			}
		}
		return A;
	}

};

template<class T, int P>

tensor_sym<T, P> vector_to_sym_tensor(const array<T, NDIM>& vec) {
	tensor_sym<T, P> X;
	array<int, NDIM> n;
	T x = T(1);
	for (n[0] = 0; n[0] < P; n[0]++) {
		T y = T(1);
		for (n[1] = 0; n[1] < P - n[0]; n[1]++) {
			T z = T(1);
			for (n[2] = 0; n[2] < P - n[0] - n[1]; n[2]++) {
				X(n) = x * y * z;
				z *= vec[2];
			}
			y *= vec[1];
		}
		x *= vec[0];
	}
	return X;
}

