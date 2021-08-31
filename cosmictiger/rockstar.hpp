/*
 * rockstar.hpp
 *
 *  Created on: Aug 18, 2021
 *      Author: dmarce1
 */

#ifndef ROCKSTAR_HPP_
#define ROCKSTAR_HPP_

#include <cosmictiger/groups.hpp>

using phase_t = array<float,2*NDIM>;

struct seed_halo {
	array<double, NDIM> x;
	array<float, NDIM> v;
	float sigma_x;
	float sigma_v;
	vector<phase_t> parts;
	seed_halo() = default;
	void normalize(float nx, float nv) {
		sigma_x *= nx;
		sigma_v *= nv;
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] *= nx;
			v[dim] *= nv;
			for (int i = 0; i < parts.size(); i++) {
				parts[i][dim] *= nx;
				parts[i][NDIM + dim] *= nv;
			}
		}
	}
	seed_halo(vector<phase_t> && _parts) {
		parts = std::move(_parts);
		ASSERT(parts.size() > 1);
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = 0.0;
			v[dim] = 0.0;
		}
		for (int i = 0; i < parts.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				x[dim] += parts[i][dim];
				v[dim] += parts[i][NDIM + dim];
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] /= parts.size();
			v[dim] /= parts.size();
		}
		sigma_x = 0.0;
		sigma_v = 0.0;
		for (int i = 0; i < parts.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				sigma_x += sqr(parts[i][dim] - x[dim]);
				sigma_v += sqr(parts[i][NDIM + dim] - v[dim]);
			}
		}
		sigma_x /= parts.size();
		sigma_v /= parts.size();
	}
	bool indistinguishable_from(const seed_halo& other) {
		int n;
		float nv, nx;
		if (other.parts.size() > parts.size()) {
			nx = sigma_x;
			nv = sigma_v;
			n = parts.size();
		} else {
			nx = other.sigma_x;
			nv = other.sigma_v;
			n = other.parts.size();
		}
		float d2 = 0.0f;
		float norm_x = n / (nx * nx);
		float norm_v = n / (nv * nv);
		for (int dim = 0; dim < NDIM; dim++) {
			d2 += sqr(x[dim] - other.x[dim]) * norm_x;
			d2 += sqr(v[dim] - other.v[dim]) * norm_v;
		}
		return (d2 < 200);
	}
	seed_halo& operator+=(const seed_halo& other) {
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] *= parts.size();
			x[dim] += other.x[dim] * other.parts.size();
			x[dim] /= other.parts.size() + parts.size();
			v[dim] *= parts.size();
			v[dim] += other.v[dim] * other.parts.size();
			v[dim] /= other.parts.size() + parts.size();
		}
		parts.insert(parts.end(), other.parts.begin(), other.parts.end());
		float sigma_x = 0.0;
		float sigma_v = 0.0;
		for (int i = 0; i < parts.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				sigma_x += sqr(parts[i][dim] - x[dim]);
				sigma_v += sqr(parts[i][NDIM + dim] - v[dim]);
			}
		}
		sigma_x /= parts.size();
		sigma_v /= parts.size();
		return *this;
	}
};

vector<seed_halo> rockstar_seed_halos(const vector<particle_data>& in_parts);

#endif /* ROCKSTAR_HPP_ */
