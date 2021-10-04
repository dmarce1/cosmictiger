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
	int nbins;
	float box_size;

	namespace po = boost::program_options;
	po::options_description command_opts("options");

	command_opts.add_options()                                                                       //
	("help", "produce help message")                                                                 //
	("in", po::value<int>(&innum)->default_value(-1), "input number") //
	("box_size", po::value<float>(&box_size)->default_value(684.0), "box size in Mpc") //
	("nbins", po::value<int>(&nbins)->default_value(1000), "number of log bins") //
			;
	boost::program_options::variables_map vm;
	po::store(po::parse_command_line(argc, argv, command_opts), vm);
	po::notify(vm);

	if (innum == -1) {
		printf("Input number not specified. Use --in=\n");
		return -1;
	}

	group_files file(innum);
	group_entry entry;
	vector<float> masses;
	float mass_min = std::numeric_limits<float>::max();
	float mass_max = 0.0f;
	while (file.get_next_entry(entry)) {
		halo_t halo;
		masses.push_back(entry.mass);
		mass_min = std::min(mass_min, entry.mass);
		mass_max = std::max(mass_max, entry.mass);
	}
	const float logmax = log10(mass_max);
	const float logmin = log10(mass_min);
	const float dlog = (logmax - logmin) / nbins;
	std::vector<int> bins(nbins, 0);
	for (int i = 0; i < masses.size(); i++) {
		int bin = std::min((int) ((log10(masses[i]) - logmin) / dlog), nbins - 1);
		bins[bin]++;
	}
	for (int i = 0; i < nbins; i++) {
		float logmass = (i + 0.5) * dlog + logmin;
		printf("%e %e\n", logmass, bins[i] / dlog / box_size / box_size / box_size);
	}

	return 0;
}
