#include <cosmictiger/output.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/math.hpp>

#include <silo.h>

vector<float> output_get_slice();

HPX_PLAIN_ACTION (output_get_slice);

void output_particles(const std::string filename, const vector<output_particle>& parts) {
	DBfile *db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Meshless", DB_PDB);
	vector<double> x(parts.size());
	vector<double> y(parts.size());
	vector<double> z(parts.size());
	for (int i = 0; i < parts.size(); i++) {
		x[i] = parts[i].x[XDIM].to_double();
		y[i] = parts[i].x[YDIM].to_double();
		z[i] = parts[i].x[ZDIM].to_double();
	}
	const int nparts = parts.size();
	double *coords[NDIM] = { x.data(), y.data(), z.data() };
	DBPutPointmesh(db, "points", NDIM, coords, nparts, DB_DOUBLE, NULL);
	x = decltype(x)();
	y = decltype(y)();
	z = decltype(z)();
	for (int dim = 0; dim < NDIM; dim++) {
		vector<float> v(parts.size());
		for (int i = 0; i < parts.size(); i++) {
			v[i] = parts[i].v[dim];
		}
		std::string nm = std::string() + "v_" + char('x' + char(dim));
		DBPutPointvar1(db, nm.c_str(), "points", v.data(), nparts, DB_FLOAT, NULL);
	}
	vector<int> r(parts.size());
	for (int i = 0; i < parts.size(); i++) {
		r[i] = parts[i].r;
	}
	DBPutPointvar1(db, "rung", "points", r.data(), nparts, DB_INT, NULL);
	DBClose(db);
}

void output_tracers(int number) {
	std::string filename = "tracers." + std::to_string(number) + ".silo";
	output_particles(filename, particles_get_tracers());
}

void output_slice(int number) {
	constexpr int ndim = 2;
	auto pixels = output_get_slice();
	const int res = get_options().slice_res;
	const std::string filename = std::string("slice.") + std::to_string(number) + std::string(".silo");
	DBfile *db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Data", DB_PDB);
	vector<float> x(res + 1);
	vector<float> y(res + 1);
	for (int i = 0; i < res + 1; i++) {
		y[i] = x[i] = (float) i / res;
	}
	const char* coordnames[] = { "x", "y" };
	const float* coords[] = { x.data(), y.data() };
	const int dims1[] = { res + 1, res + 1 };
	const int dims2[] = { res, res };
	DBPutQuadmesh(db, "mesh", coordnames, coords, dims1, ndim, DB_FLOAT, DB_COLLINEAR, NULL);
	DBPutQuadvar1(db, "intensity", "mesh", pixels.data(), dims2, ndim, NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
	DBClose(db);
}

vector<float> output_get_slice() {
	vector<hpx::future<vector<float>>>futs;
	const int res = get_options().slice_res;
	vector<float> pixels(sqr(res), 0.0);
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async < output_get_slice_action > (c));
	}
	for (int i = 0; i < particles_size(); i++) {
		const double z = particles_pos(ZDIM, i).to_double();
		if (z < get_options().slice_size) {
			const double x = particles_pos(XDIM, i).to_double();
			const double y = particles_pos(YDIM, i).to_double();
			const int xi = x * res;
			const int yi = y * res;
			const int j = xi * res + yi;
			pixels[j] += 1.0;
		}
	}
	for (auto& f : futs) {
		auto v = f.get();
		for (int i = 0; i < sqr(res); i++) {
			pixels[i] += v[i];
		}
	}
	return pixels;
}

