#include <cosmictiger/containers.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/map.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/simd.hpp>

#include <chealpix.h>

static double tau_max;
static int Nside;
static int Nmaps;
static healpix_map main_map;

HPX_PLAIN_ACTION(map_init);
HPX_PLAIN_ACTION(map_flush);

void map_init(float tmax) {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<map_init_action>(c, tmax));
	}
	tau_max = tmax;
	Nside = get_options().map_size;
	Nmaps = get_options().map_count;
	hpx::wait_all(futs.begin(), futs.end());
}

void map_flush(float t) {
	main_map.map_flush(t);
}

void healpix_map::map_flush(float t) {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<map_flush_action>(c, t));
	}
	static const float map_int = tau_max / Nmaps;
	for (auto i = map.begin(); i != map.end(); i++) {
		const float this_t = i->first * map_int;
		if (t > this_t) {
			PRINT("Writing map %i\n", i->first);
			char* str;
			ASPRINTF(&str, "mkdir -p healpix_%i\n", i->first);
			if (system(str) != 0) {
				THROW_ERROR("Unable to execute \"%s\"", str);
			}
			free(str);
			ASPRINTF(&str, "healpix_%i/healpix.%i", i->first, hpx_rank());
			FILE* fp = fopen(str, "wb");
			free(str);
			if (fp == NULL) {
				THROW_ERROR("Unable to open %s for writing\n", str);
			}
			auto& this_map = i->second;
			int size = this_map.size();
			fwrite(&Nside, sizeof(int), 1, fp);
			fwrite(&size, sizeof(int), 1, fp);
			for (auto j = this_map.begin(); j != this_map.end(); j++) {
				fwrite(&j->first, sizeof(int), 1, fp);
				fwrite(&j->second.i, sizeof(float), 1, fp);
			}
			fclose(fp);
			map.erase(i);
			break;
		}
	}

}

void healpix_map::map_load(FILE* fp) {
	int hpxsize;
	FREAD(&hpxsize, sizeof(int), 1, fp);
	for (int i = 0; i < hpxsize; i++) {
		int rank;
		int size;
		FREAD(&rank, sizeof(int), 1, fp);
		FREAD(&size, sizeof(int), 1, fp);
		auto& this_map = map[rank];
		for (int j = 0; j < size; j++) {
			int pix;
			float value;
			FREAD(&pix, sizeof(int), 1, fp);
			FREAD(&value, sizeof(int), 1, fp);
			this_map[pix].i = value;
		}
	}

}

void healpix_map::map_save(FILE* fp) {
	int size = map.size();
	fwrite(&size, sizeof(int), 1, fp);
	for (auto i = map.begin(); i != map.end(); i++) {
		int rank = i->first;
		int size = i->second.size();
		fwrite(&rank, sizeof(int), 1, fp);
		fwrite(&size, sizeof(int), 1, fp);
		for (auto j = i->second.begin(); j != i->second.end(); j++) {
			fwrite(&j->first, sizeof(int), 1, fp);
			fwrite(&j->second.i, sizeof(int), 1, fp);
		}
	}
}

void map_load(FILE* fp) {
	main_map.map_load(fp);
}

void map_save(FILE* fp) {
	main_map.map_save(fp);
}

void map_add_map(healpix_map&& other) {
	main_map.add_map(std::move(other));
}

void healpix_map::add_map(healpix_map&& other) {
	for (auto i = other.map.begin(); i != other.map.end(); i++) {
		for (auto j = i->second.begin(); j != i->second.end(); j++) {
			map[i->first][j->first].i += j->second.i;
		}
	}
}

int healpix_map::map_add_particle(float x0, float y0, float z0, float x1, float y1, float z1, float vx, float vy, float vz, float t, float dt) {
	static simd_float8 images[NDIM] = { simd_float8(0, -1, 0, -1, 0, -1, 0, -1), simd_float8(0, 0, -1, -1, 0, 0, -1, -1), simd_float8(0, 0, 0, 0, -1, -1, -1, -1) };
	static const float map_int = tau_max / Nmaps;
	static const float map_int_inv = 1.0 / map_int;
	const simd_float8 simd_c0 = simd_float8(map_int_inv);
	array<simd_float8, NDIM> X0;
	array<simd_float8, NDIM> X1;
	const simd_float8 simd_tau0 = simd_float8(t);
	const simd_float8 simd_tau1 = simd_float8(t + dt);
	simd_float8 dist0;
	simd_float8 dist1;
	int rc = 0;
	double x20, x21, R20, R21;
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
				for (int j = i0; j < i1; j++) {
					rc++;
					x0 = X0[XDIM][ci];
					y0 = X0[YDIM][ci];
					z0 = X0[ZDIM][ci];
					const double ti = (j + 1) * map_int;
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
					double vec[NDIM];
					long int ipix;
					if (sqr(x1, y1, z1) <= 1.f) {                                                 // 6
						double mag = 1.f / sqr(x1, y1, z1);                                               // 9
						vec[0] = x1;
						vec[1] = y1;
						vec[2] = z1;
						vec2pix_ring(Nside, vec, &ipix);
						map[j + 1][ipix].i += mag;
					}
				}
			}
		}
	}
	return rc;
}