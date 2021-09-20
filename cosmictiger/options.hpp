
#pragma once

#include <string>

struct options {
	bool cuda;
	bool save_force;
	bool do_lc;
	bool do_power;
	bool do_groups;
	bool do_tracers;
	bool do_slice;
	bool do_views;
	bool twolpt;
	bool groups_funnel_output;

	int tracer_count;
	int parts_dim;
	int tree_cache_line_size;
	int part_cache_line_size;
	int check_num;
	int check_freq;
	int max_iter;
	int view_size;
	int lc_min_group;
	int lc_map_size;
	int min_group;
	int slice_res;

	double lc_b;
	double slice_size;
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
	double z1;
	double hubble;
	double sigma8;
	double theta;

	std::string config_file;
	std::string test;

	template<class A>
	void serialize(A&& arc, unsigned) {
		arc &  cuda;
		arc &  do_lc;
		arc &  save_force;
		arc &  do_power;
		arc &  do_groups;
		arc &  do_tracers;
		arc &  do_slice;
		arc &  do_views;
		arc &  twolpt;
		arc &  groups_funnel_output;

		arc &  tracer_count;
		arc &  parts_dim;
		arc &  tree_cache_line_size;
		arc &  part_cache_line_size;
		arc &  check_num;
		arc &  check_freq;
		arc &  max_iter;
		arc &  lc_min_group;
		arc &  min_group;
		arc &  view_size;
		arc &  slice_res;
		arc &  lc_map_size;

		arc &  lc_b;
		arc &  theta;
		arc &  slice_size;
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
		arc &  z1;
		arc &  hubble;
		arc &  sigma8;

		arc & config_file;
		arc & test;
	}
};

bool process_options(int argc, char *argv[]);
const options& get_options();
void set_options(const options& opts);
