#include <cosmictiger/containers.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>

#include <memory>
#include <unordered_set>
#include <unordered_map>

#include <healpix_cxx/healpix_base.h>

#define PIX_PER_RANK 16

using healpix_type = T_Healpix_Base< int >;

using lc_real = double;
using lc_group = long long;

struct lc_particle {
	array<lc_real, NDIM> pos;
	array<float, NDIM> vel;
	lc_group group;
};

static std::shared_ptr<healpix_type> healpix;
static pair<int> my_pix_range;
static std::unordered_set<int> bnd_pix;
static std::unordered_map<int, vector<lc_particle>> part_map;
static double tau_max;

void lc_init(double);

static int vec2pix(double x, double y, double z) {
	vec3 vec;
	vec.x = x;
	vec.y = y;
	vec.z = z;
	return healpix->vec2pix(vec);
}

static vector<int> pix_neighbors(int pix) {
	vector<int> neighbors;
	neighbors.reserve(8);
	fix_arr<int, 8> result;
	healpix->neighbors(pix, result);
	for (int i = 0; i < 8; i++) {
		if (result[i] != -1) {
			neighbors.push_back(result[i]);
		}
	}
	return neighbors;
}

HPX_PLAIN_ACTION (lc_init);

static int Nside;
static int Npix;

void lc_init(double tau_max_) {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < lc_init_action > (c, tau_max_));
	}

	Nside = std::ceil(std::sqrt(PIX_PER_RANK * hpx_size() / 12));
	Npix = 12 * sqr(Nside);
	healpix = std::make_shared < healpix_type > (Nside, NEST, SET_NSIDE);
	tau_max = tau_max_;
	my_pix_range.first = (size_t) hpx_rank() * Npix / hpx_size();
	my_pix_range.second = (size_t)(hpx_rank() + 1) * Npix / hpx_size();
	for (int pix = my_pix_range.first; pix < my_pix_range.second; pix++) {
		part_map[pix].resize(0);
		const auto neighbors = pix_neighbors(pix);
		for (const auto& n : neighbors) {
			if (n < my_pix_range.first || n >= my_pix_range.second) {
				bnd_pix.insert(n);
				part_map[n].resize(0);
			}
		}
	}

	hpx::wait_all(futs.begin(), futs.end());
}

int lc_add_particle(lc_real x0, lc_real y0, lc_real z0, lc_real x1, lc_real y1, lc_real z1, float vx, float vy, float vz, float t, float dt) {
	static simd_float8 images[NDIM] =
			{ simd_float8(0, -1, 0, -1, 0, -1, 0, -1), simd_float8(0, 0, -1, -1, 0, 0, -1, -1), simd_float8(0, 0, 0, 0, -1, -1, -1, -1) };
	const float tau_max_inv = 1.0 / tau_max;
	const simd_float8 simd_c0 = simd_float8(tau_max_inv);
	array<simd_float8, NDIM> X0;
	array<simd_float8, NDIM> X1;
	const simd_float8 simd_tau0 = simd_float8(t);
	const simd_float8 simd_tau1 = simd_float8(t + dt);
	simd_float8 dist0;
	simd_float8 dist1;
	int rc = 0;
	const int Npix = sqr(Nside) * 12;
	X0[XDIM] = simd_float8(x0) + images[XDIM];
	X0[YDIM] = simd_float8(y0) + images[YDIM];
	X0[ZDIM] = simd_float8(z0) + images[ZDIM];
	X1[XDIM] = simd_float8(x1) + images[XDIM];
	X1[YDIM] = simd_float8(y1) + images[YDIM];
	X1[ZDIM] = simd_float8(z1) + images[ZDIM];
	dist0 = sqrt(sqr(X0[0], X0[1], X0[2]));
	dist1 = sqrt(sqr(X1[0], X1[1], X1[2]));
	simd_float8 tau0 = simd_tau0 + dist0;
	simd_float8 tau1 = simd_tau1 + dist1;
	simd_int8 I0 = tau0 * simd_c0;
	simd_int8 I1 = tau1 * simd_c0;

	for (int ci = 0; ci < SIMD_FLOAT8_SIZE; ci++) {
		if (dist1[ci] <= 1.0 || dist0[ci] <= 1.0) {
			const int i0 = I0[ci];
			const int i1 = I1[ci];
			if (i0 != i1) {
				x0 = X0[XDIM][ci];
				y0 = X0[YDIM][ci];
				z0 = X0[ZDIM][ci];
				const double ti = (i0 + 1) * tau_max;
				const double sqrtauimtau0 = sqr(ti - t);
				const double tau0mtaui = t - ti;
				const double u2 = sqr(vx, vy, vz);                                    // 5
				const double x2 = sqr(x0, y0, z0);                                       // 5
				const double udotx = vx * x0 + vy * y0 + vz * z0;               // 5
				const double A = 1.f - u2;                                                     // 1
				const double B = 2.0 * (tau0mtaui - udotx);                                    // 2
				const double C = sqrtauimtau0 - x2;                                            // 1
				const double t = -(B + sqrt(B * B - 4.f * A * C)) / (2.f * A);                // 15
				const double x1 = x0 + vx * t;                                            // 2
				const double y1 = y0 + vy * t;                                            // 2
				const double z1 = z0 + vz * t;                                            // 2
				long int ipix;
				if (sqr(x1, y1, z1) <= 1.f) {                                                 // 6
					const auto pix = vec2pix(x1, y1, z1);
					lc_particle part;
					part.pos[XDIM] = x1;
					part.pos[XDIM] = y1;
					part.pos[XDIM] = z1;
					part.vel[XDIM] = vx;
					part.vel[YDIM] = vy;
					part.vel[ZDIM] = vz;
					part_map[pix].push_back(part);
				}
			}
		}
	}
	return rc;
}
