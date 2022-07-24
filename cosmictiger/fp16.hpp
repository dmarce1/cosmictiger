/*
 * fp16.hpp
 *
 *  Created on: Jul 20, 2022
 *      Author: dmarce1
 */

#ifndef FP16_HPP_
#define FP16_HPP_

#include <math.h>

struct float16 {
	unsigned short m :10;
	unsigned short e :5;
	unsigned short s :1;
};

inline float16 double2float16(double r) {
	float16 f16;
	if (r == 0.0) {
		f16.m = 0;
		f16.e = 0;
		f16.s = 0;
	} else {
		double absr = fabs(r);
		double e = floor(log2(absr));
		double m = (absr * pow(2.0, -e) - 1.0) * double(1 << 10);
		int m0 = round(m);
		int e0 = e;
		if (m0 == 1024) {
			m0 = 0;
			e0++;
		}
		e0 += 15;
		if (e0 > 31) {
			PRINT("fp16 overflow with %e\n", r);
			abort();
		} else if (e0 < 0) {
			f16 = double2float16(0.0);
		} else {
			f16.e = e0;
			f16.m = m0;
			f16.s = r < 0.0;
		}
	}
	return f16;
}

inline double float162double(float16 f) {
	double r;
	if (f.s == 0 && f.e == 0 && f.m == 0) {
		r = 0.0;
	} else {
		r = (1.0 + f.m / double(1 << 10)) * double(1ULL << f.e) / double(1 << 15);
		if (f.s) {
			r = -r;
		}
	}
	return r;
}

namespace std {
template<>
struct numeric_limits<float16> {
	static double max() {
		float16 f;
		f.e = 0x1F;
		f.m = 0x3FF;
		f.s = 0x0;
		return float162double(f);
	}
	static double min() {
		float16 f;
		f.e = 0x0;
		f.m = 0x1;
		f.s = 0x0;
		return float162double(f);
	}
};

}
#endif /* FP16_HPP_ */
