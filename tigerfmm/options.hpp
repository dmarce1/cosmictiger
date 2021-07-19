
#pragma once

#include <string>

struct options {

	int parts_dim;
	int tree_cache_line_size;

	double hsoft;

	std::string config_file;
	std::string test;

	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & hsoft;
		arc & parts_dim;
		arc & config_file;
		arc & test;
		arc & tree_cache_line_size;
	}
};

bool process_options(int argc, char *argv[]);
const options& get_options();
void set_options(const options& opts);
