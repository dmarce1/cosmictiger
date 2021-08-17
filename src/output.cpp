#include <cosmictiger/output.hpp>
#include <cosmictiger/particles.hpp>

#include <silo.h>

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

void output_sample(int number) {
	std::string filename = "sample." + std::to_string(number) + ".silo";
	range<double> box;
	for (int dim = 0; dim < NDIM; dim++) {
		box.begin[dim] = 0.0;
		box.end[dim] = get_options().sample_dim;
	}
	output_particles(filename, particles_get_sample(box));
}

void output_tracers(int number) {
	std::string filename = "tracers." + std::to_string(number) + ".silo";
	output_particles(filename, particles_get_tracers());
}

