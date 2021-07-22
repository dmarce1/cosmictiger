
#pragma once

#include <string>

struct options {

	bool save_force;

	int parts_dim;
	int tree_cache_line_size;
	int part_cache_line_size;

	double hsoft;
	double GM;
	double eta;

	std::string config_file;
	std::string test;

	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & eta;
		arc & save_force;
		arc & hsoft;
		arc & parts_dim;
		arc & config_file;
		arc & GM;
		arc & test;
		arc & tree_cache_line_size;
		arc & part_cache_line_size;
	}
};

bool process_options(int argc, char *argv[]);
const options& get_options();
void set_options(const options& opts);
