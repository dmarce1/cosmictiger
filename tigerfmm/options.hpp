
#pragma once

#include <string>

struct options {
	bool cuda;
	bool save_force;

	int parts_dim;
	int tree_cache_line_size;
	int part_cache_line_size;
	int check_num;
	int check_freq;
	int max_iter;

	double hsoft;
	double GM;
	double eta;
	double code_to_s;
	double code_to_cm;
	double code_to_g;
	double omega_m;
	double omega_r;
	double z0;
	double hubble;
	double sigma8;

	std::string config_file;
	std::string test;

	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & cuda;
		arc & code_to_s;
		arc & code_to_cm;
		arc & code_to_g;
		arc & omega_m;
		arc & omega_r;
		arc & z0;
		arc & hubble;
		arc & sigma8;
		arc & eta;
		arc & save_force;
		arc & hsoft;
		arc & parts_dim;
		arc & config_file;
		arc & GM;
		arc & test;
		arc & tree_cache_line_size;
		arc & part_cache_line_size;
		arc & check_num;
		arc & check_freq;
		arc & max_iter;
	}
};

bool process_options(int argc, char *argv[]);
const options& get_options();
void set_options(const options& opts);
