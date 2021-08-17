
#pragma once

#include <string>

struct options {
	bool cuda;
	bool save_force;
	bool do_map;
	bool do_power;
	bool do_groups;
	bool do_tracers;
	bool do_sample;

	int tracer_count;
	int parts_dim;
	int tree_cache_line_size;
	int part_cache_line_size;
	int check_num;
	int check_freq;
	int max_iter;
	int map_count;
	int map_size;
	int min_group;

	double sample_dim;
	double link_len;
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
		arc &  cuda;
		arc &  save_force;
		arc &  do_map;
		arc &  do_power;
		arc &  do_groups;
		arc &  do_tracers;
		arc &  do_sample;

		arc &  tracer_count;
		arc &  parts_dim;
		arc &  tree_cache_line_size;
		arc &  part_cache_line_size;
		arc &  check_num;
		arc &  check_freq;
		arc &  max_iter;
		arc &  map_count;
		arc &  map_size;
		arc &  min_group;

		arc &  sample_dim;
		arc &  link_len;
		arc &  hsoft;
		arc &  GM;
		arc &  eta;
		arc &  code_to_s;
		arc &  code_to_cm;
		arc &  code_to_g;
		arc &  omega_m;
		arc &  omega_r;
		arc &  z0;
		arc &  hubble;
		arc &  sigma8;

		arc & config_file;
		arc & test;
	}
};

bool process_options(int argc, char *argv[]);
const options& get_options();
void set_options(const options& opts);
