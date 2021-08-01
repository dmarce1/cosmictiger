#include <tigerfmm/containers.hpp>
#include <tigerfmm/map.hpp>
#include <tigerfmm/math.hpp>
#include <tigerfmm/options.hpp>
#include <tigerfmm/simd.hpp>

static float tau_max;
static int Nside;
static int Nmaps;

#include <unordered_map>

struct map_data {
	std::unordered_map<int,float> map;

};

void map_init(float tmax) {
	tau_max = tmax;
	Nside = get_options().map_size;
	Nmaps = get_options().map_count;
}

int map_add_particle(float x0, float y0, float z0, float x1, float y1, float z1, float t, float dt) {
	static simd_float images[NDIM] = { simd_float(0, -1, 0, -1, 0, -1, 0, -1), simd_float(0, 0, -1, -1, 0, 0, -1, -1), simd_float(0, 0, 0, 0, -1, -1, -1, -1) };
	static const float map_int = tau_max / Nmaps;
	static const float map_int_inv = 1.0 / map_int;
	const simd_float simd_c0 = simd_float(map_int_inv);
	array<simd_float, NDIM> X0;
	array<simd_float, NDIM> X1;
	const simd_float simd_tau0 = simd_float(t);
	const simd_float simd_tau1 = simd_float(t + dt);
	simd_float dist0;
	simd_float dist1;
	int rc = 0;
	double x20, x21, R20, R21;
	X0[XDIM] = simd_float(x0) + images[XDIM];
	X0[YDIM] = simd_float(y0) + images[YDIM];
	X0[ZDIM] = simd_float(z0) + images[ZDIM];
	X1[XDIM] = simd_float(x1) + images[XDIM];
	X1[YDIM] = simd_float(y1) + images[YDIM];
	X1[ZDIM] = simd_float(z1) + images[ZDIM];
	dist0 = sqrt(sqr(X0[0], X0[1], X0[2]));
	dist1 = sqrt(sqr(X1[0], X1[1], X1[2]));
	simd_float tau0 = simd_tau0 + dist0;
	simd_float tau1 = simd_tau1 + dist1;
	simd_int I0 = tau0 * simd_c0;
	simd_int I1 = tau1 * simd_c0;
	for (int ci = 0; ci < 8; ci++) {
		if (dist1[ci] <= 1.0 || dist0[ci] <= 1.0) {
			const int i0 = I0[ci];
			const int i1 = I1[ci];
		//	PRINT( "%e %e\n", simd_c0[0], simd_c0[0]);
			if (i0 != i1) {
				for (int j = i0; j < i1; j++) {
					rc++;
					/*					static const long Nside = global().opts.map_size;
					 double r = dist1[ci];
					 long ipring;
					 auto& this_ws = (*ws.data)[j + 1];
					 this_ws.x.push_back(x0[0][ci]);
					 this_ws.y.push_back(x0[1][ci]);
					 this_ws.z.push_back(x0[2][ci]);
					 this_ws.vx.push_back((x1[0][ci] - x0[0][ci]) * dtau_inv);
					 this_ws.vy.push_back((x1[1][ci] - x0[1][ci]) * dtau_inv);
					 this_ws.vz.push_back((x1[2][ci] - x0[2][ci]) * dtau_inv);*/
				}
			}
		}
	}
	return rc;
}
