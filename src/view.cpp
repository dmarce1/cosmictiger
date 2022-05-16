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

#include <cosmictiger/options.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/view.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/cosmology.hpp>

#include <silo.h>

#include <atomic>

#include <chealpix.h>

HPX_PLAIN_ACTION (output_view);

static vector<range<double>> view_boxes;

struct dm_part_info {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float vx;
	float vy;
	float vz;
	int rung;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & x;
		arc & y;
		arc & z;
		arc & vx;
		arc & vy;
		arc & vz;
		arc & rung;
	}

};

struct sph_part_info: public dm_part_info {
	float eint;
	float rho;
	float h;
	float H;
	float Hp;
	float H2;
	float Hn;
	float Hep;
	float Hepp;
	float He;
	float Z;
	float alpha;
	float cold_frac;
	template<class A>
	void serialize(A&& arc, unsigned ver) {
		dm_part_info::serialize(arc, ver);
		arc & eint;
		arc & h;
		arc & H;
		arc & Hp;
		arc & H2;
		arc & Hn;
		arc & Hep;
		arc & Hepp;
		arc & cold_frac;
		arc & He;
		arc & Z;
		arc & alpha;
	}

};

struct star_part_info: public dm_part_info {
	float Y;
	float Z;
	float M;
	float zform;
	template<class A>
	void serialize(A&& arc, unsigned ver) {
		dm_part_info::serialize(arc, ver);
		arc & Y;
		arc & Z;
		arc & M;
		zform;
	}

};

struct view_return {
	vector<vector<sph_part_info>> hydro;
	vector<vector<dm_part_info>> dm;
	vector<vector<star_part_info>> star;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & hydro;
		arc & dm;
		arc & star;
	}
};

view_return view_get_particles(vector<range<double>> boxes);

HPX_PLAIN_ACTION (view_get_particles);

view_return view_get_particles(vector<range<double>> boxes = vector<range<double>>()) {
	if (hpx_rank() == 0) {
		boxes = view_boxes;
	}
	view_return rc;
	rc.hydro.resize(boxes.size());
	rc.dm.resize(boxes.size());
	rc.star.resize(boxes.size());
	vector<hpx::future<view_return>> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<view_get_particles_action>(c, boxes));
	}
	const int nthreads = hpx_hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc, nthreads, boxes]() {
			view_return rc;
			rc.hydro.resize(boxes.size());
			rc.dm.resize(boxes.size());
			rc.star.resize(boxes.size());
			const part_int b = (size_t) proc * particles_size() / nthreads;
			const part_int e = (size_t) (proc+1) * particles_size() / nthreads;
			for( part_int i = b; i < e; i++) {
				array<double,NDIM> x;
				x[XDIM] = particles_pos(XDIM,i).to_double();
				x[YDIM] = particles_pos(YDIM,i).to_double();
				x[ZDIM] = particles_pos(ZDIM,i).to_double();
				for( int j = 0; j < boxes.size(); j++) {
					if( boxes[j].contains(x)) {
						dm_part_info info;
						info.x = particles_pos(XDIM,i);
						info.y = particles_pos(YDIM,i);
						info.z = particles_pos(ZDIM,i);
						info.vx = particles_vel(XDIM,i);
						info.vy = particles_vel(YDIM,i);
						info.vz = particles_vel(ZDIM,i);
						info.rung = particles_rung(i);
						rc.dm[j].push_back(info);
						break;
					}
				}
			}
			return rc;
		}));
	}

	for (auto& f : futs) {
		auto tmp = f.get();
		for (int i = 0; i < boxes.size(); i++) {
			rc.hydro[i].insert(rc.hydro[i].begin(), tmp.hydro[i].begin(), tmp.hydro[i].end());
			rc.dm[i].insert(rc.dm[i].begin(), tmp.dm[i].begin(), tmp.dm[i].end());
			rc.star[i].insert(rc.star[i].begin(), tmp.star[i].begin(), tmp.star[i].end());
		}
	}
	return rc;

}

void view_output_views(int cycle, double a) {
	profiler_enter("view_output_views");
	if (!view_boxes.size()) {
		return;
	}
	const double code_to_cm = get_options().code_to_cm;
	const double code_to_s = get_options().code_to_s;
	const double code_to_g = get_options().code_to_g;
//	const double code_to_velocity = code_to_cm / code_to_s / a;
//	const double code_to_energy = sqr(code_to_cm / code_to_s / a);
//	const double code_to_density = code_to_g / (code_to_cm * sqr(code_to_cm));
	const double code_to_velocity = 1.;
	const double code_to_energy = 1.;
	const double code_to_density = 1.;
	for (int bi = 0; bi < view_boxes.size(); bi++) {
		PRINT("Outputing view for box (%e %e) (%e %e) (%e %e)\n", view_boxes[bi].begin[XDIM], view_boxes[bi].end[XDIM], view_boxes[bi].begin[YDIM],
				view_boxes[bi].end[YDIM], view_boxes[bi].begin[ZDIM], view_boxes[bi].end[ZDIM]);
		std::string filename = "view." + std::to_string(bi) + "." + std::to_string(cycle) + ".silo";
		DBfile *db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Meshless", DB_PDB);
		view_return parts;
		parts = view_get_particles();
		vector<float> x, y, z;
		auto opts = DBMakeOptlist(1);
		float z0 = 1.0 / a - 1.0;
		float tm = cosmos_time(1e-6, a) * get_options().code_to_s * constants::seconds_to_years / 1e9;
		DBAddOption(opts, DBOPT_TIME, &tm);
		if (parts.dm[bi].size()) {
			x.resize(0);
			y.resize(0);
			z.resize(0);
			for (int i = 0; i < parts.dm[bi].size(); i++) {
				x.push_back(parts.dm[bi][i].x.to_float());
				y.push_back(parts.dm[bi][i].y.to_float());
				z.push_back(parts.dm[bi][i].z.to_float());
			}
			float *coords1[NDIM] = { x.data(), y.data(), z.data() };
			DBPutPointmesh(db, "dark_matter", NDIM, coords1, x.size(), DB_FLOAT, opts);
			x.resize(0);
			y.resize(0);
			z.resize(0);
			for (int i = 0; i < parts.dm[bi].size(); i++) {
				x.push_back(parts.dm[bi][i].vx * code_to_velocity);
				y.push_back(parts.dm[bi][i].vy * code_to_velocity);
				z.push_back(parts.dm[bi][i].vz * code_to_velocity);
			}
			DBPutPointvar1(db, "dm_vx", "dark_matter", x.data(), x.size(), DB_FLOAT, opts);
			DBPutPointvar1(db, "dm_vy", "dark_matter", y.data(), x.size(), DB_FLOAT, opts);
			DBPutPointvar1(db, "dm_vz", "dark_matter", z.data(), x.size(), DB_FLOAT, opts);
			x.resize(0);
			for (int i = 0; i < parts.dm[bi].size(); i++) {
				x.push_back(parts.dm[bi][i].rung);
			}
			DBPutPointvar1(db, "dm_rung", "dark_matter", x.data(), x.size(), DB_FLOAT, opts);
		}
		DBFreeOptlist(opts);
		DBClose(db);
	}
	profiler_exit();
}

void view_read_view_file() {
	FILE* fp = fopen("view.txt", "rt");
	if (fp == nullptr) {
		PRINT("No view file found.\n");
		return;
	}
	range<float> view_box;
	while (fscanf(fp, "%f %f %f %f %f %f\n", &view_box.begin[XDIM], &view_box.end[XDIM], &view_box.begin[YDIM], &view_box.end[YDIM], &view_box.begin[ZDIM],
			&view_box.end[ZDIM]) == 2 * NDIM) {
		PRINT("Reading box for view at x range (%f,%f) yrange (%f,%f) zrange (%f,%f)\n", view_box.begin[XDIM], view_box.end[XDIM], view_box.begin[YDIM],
				view_box.end[YDIM], view_box.begin[ZDIM], view_box.end[ZDIM]);
		range<double> view_box_d;
		for (int dim = 0; dim < NDIM; dim++) {
			view_box_d.begin[dim] = view_box.begin[dim];
			view_box_d.end[dim] = view_box.end[dim];
		}
		view_boxes.push_back(view_box_d);
	}
	fclose(fp);

}

vector<float> output_view(int number, double time) {
	profiler_enter("output_view");
	vector<hpx::future<void>> futs;
	vector<hpx::future<vector<float>>>val_futs;
	for (const auto& c : hpx_children()) {
		val_futs.push_back(hpx::async<output_view_action>(HPX_PRIORITY_HI, c, number, time));
	}
	const int nthreads = hpx::thread::hardware_concurrency();
	const int Nside = get_options().view_size;
	const int Npix = 12 * sqr(Nside);
	vector<std::atomic<float>> values(Npix);
	for (auto& v : values) {
		v = 0.0f;
	}
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([nthreads,proc,Nside,&values]() {
			const part_int begin = (size_t) proc * particles_size() / nthreads;
			const part_int end = (size_t) (proc + 1) * particles_size() / nthreads;
			for( int i = begin; i < end; i++) {
				for( int xi = -1; xi <= 0; xi++) {
					for( int yi = -1; yi <= 0; yi++) {
						for( int zi = -1; zi <= 0; zi++) {
							double vec[NDIM];
							long int ipix;
							vec[XDIM] = particles_pos(XDIM,i).to_float() + xi;
							vec[YDIM] = particles_pos(YDIM,i).to_float() + yi;
							vec[ZDIM] = particles_pos(ZDIM,i).to_float() + zi;
							const float r2 = sqr(vec[XDIM],vec[YDIM],vec[ZDIM]);
							if( r2 < 1.0 && r2 > 0.0) {
								vec2pix_ring(Nside, vec, &ipix);
								atomic_add(values[ipix], 1.0 / r2);
							}
						}
					}
				}
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	vector<float> results(values.size());
	for (int i = 0; i < results.size(); i++) {
		results[i] = values[i];
	}
	for (auto& val_fut : val_futs) {
		const auto vals = val_fut.get();
		for (int i = 0; i < vals.size(); i++) {
			results[i] += vals[i];
		}
	}
	if (hpx_rank() == 0) {
		std::string filename = "view." + std::to_string(number) + ".dat";
		FILE* fp = fopen(filename.c_str(), "wb");
		if (fp == NULL) {
			THROW_ERROR("unable to open %s for writing\n", filename.c_str());
		}
		fwrite(&Nside, sizeof(int), 1, fp);
		fwrite(&Npix, sizeof(int), 1, fp);
		fwrite(results.data(), sizeof(float), Npix, fp);
		fwrite(&number, sizeof(int), 1, fp);
		fwrite(&time, sizeof(double), 1, fp);
		fclose(fp);
	}
	profiler_exit();
	return results;
}
