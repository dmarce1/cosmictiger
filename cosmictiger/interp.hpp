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


#ifndef GPUTIGER_INTERP_HPP_
#define GPUTIGER_INTERP_HPP_
#include <cuda.h>

#include <vector>

#include <cmath>
#include <stdio.h>

template<class T>
struct interp_functor {
	std::vector<T> values;
	T amin;
	T amax;
	T minloga;
	T maxloga;
	int N;
	T dloga;
	T operator()(T a) const {
		T loga = log(a);
		if (loga < minloga || loga > maxloga) {
			printf("Error in interpolation_function out of range %e %e %e\n", a, amin, amax);
		}
		int i1 = int((loga - minloga) / (dloga));
		if (i1 > N - 3) {
			i1 = N - 3;
		} else if (i1 < 1) {
			i1 = 1;
		}
		int i0 = i1 - 1;
		int i2 = i1 + 1;
		int i3 = i2 + 1;
		const T c0 = values[i1];
		const T c1 = -values[i0] / (T) 3.0 - (T) 0.5 * values[i1] + values[i2] - values[i3] / (T) 6.0;
		const T c2 = (T) 0.5 * values[i0] - values[i1] + (T) 0.5 * values[i2];
		const T c3 = -values[i0] / (T) 6.0 + (T) 0.5 * values[i1] - (T) 0.5 * values[i2] + values[i3] / (T) 6.0;
		T x = (loga - i1 * dloga - minloga) / dloga;
		return c0 + c1 * x + c2 * x * x + c3 * x * x * x;
	}
};

template<class T>
inline void build_interpolation_function(interp_functor<T>* f, const std::vector<T>& values, T amin, T amax) {
	T minloga = log(amin);
	T maxloga = log(amax);
	int N = values.size();
	T dloga = (maxloga - minloga) / (N-1);
	interp_functor<T> functor;
	functor.values = std::move(values);
	functor.maxloga = maxloga;
	functor.minloga = minloga;
	functor.dloga = dloga;
	functor.amin = amin;
	functor.amax = amax;
	functor.N = N;
	*f = functor;
}

#endif /* GPUTIGER_INTERP_HPP_ */
