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

#include <cosmictiger/group_entry.hpp>

#include <silo.h>

#include <boost/program_options.hpp>

class group_files {
	FILE* fp;
	int innum;
	int current_num;
	bool eof;
	void open_next() {
		const std::string filename = "./groups." + std::to_string(innum) + "/groups." + std::to_string(innum) + "." + std::to_string(current_num) + ".dat";
		if (current_num > 0) {
			fclose(fp);
		}
		fp = fopen(filename.c_str(), "rb");
		if (fp == NULL) {
			eof = true;
			if (current_num == 0) {
				printf("Unable to open %s\n", filename.c_str());
				abort();
			}
		}
	}
public:
	group_files(int innum_) {
		innum = innum_;
		current_num = 0;
		eof = false;
		open_next();
	}
	bool get_next_entry(group_entry& entry) {
		while (!entry.read(fp)) {
			current_num++;
			open_next();
			if (eof) {
				return false;
			}
		}
		return true;
	}
	~group_files() {
		if (!eof) {
			fclose(fp);
		}
	}
};

struct halo_t {
	double mass;
	double radius;
	double x;
	double y;
	double z;
	double vx;
	double vy;
	double vz;
};

int main(int argc, char* argv[]) {
	std::string outfile;
	int innum;

	namespace po = boost::program_options;
	po::options_description command_opts("options");

	command_opts.add_options()                                                                       //
	("help", "produce help message")                                                                 //
	("in", po::value<int>(&innum)->default_value(-1), "input number") //
	("out", po::value < std::string > (&outfile)->default_value(""), "output file") //
			;
	boost::program_options::variables_map vm;
	po::store(po::parse_command_line(argc, argv, command_opts), vm);
	po::notify(vm);

	if (innum == -1) {
		printf("Input number not specified. Use --in=\n");
		return -1;
	}

	if (outfile == "") {
		printf("Output file not specified. Use --out=\n");
		return -1;
	}

	group_files file(innum);
	group_entry entry;
	vector<halo_t> halos;
	while (file.get_next_entry(entry)) {
		halo_t halo;
		halo.mass = entry.mass;
		halo.radius = std::pow(2.0, 1.0 / 3.0) * entry.r50;
		halo.x = entry.com[XDIM];
		halo.y = entry.com[YDIM];
		halo.z = entry.com[ZDIM];
		halo.vx = entry.vel[XDIM];
		halo.vy = entry.vel[YDIM];
		halo.vz = entry.vel[ZDIM];
		halos.push_back(halo);
	}
	auto db = DBCreate(outfile.c_str(), DB_CLOBBER, DB_LOCAL, NULL, DB_PDB);
	vector<double> x, y, z;
	for (const auto& halo : halos) {
		x.push_back(halo.x);
		y.push_back(halo.y);
		z.push_back(halo.z);
	}
	const double* coords[NDIM] = { x.data(), y.data(), z.data() };
	DBPutPointmesh(db, "groups", NDIM, coords, z.size(), DB_DOUBLE, NULL);
	x = decltype(x)();
	y = decltype(y)();
	z = decltype(z)();
	vector<float> v;
	for (const auto& halo : halos) {
		v.push_back(halo.radius);
	}
	DBPutPointvar1(db, "radius", "groups", v.data(), v.size(), DB_FLOAT, NULL);
	DBClose(db);
	return 0;
}
