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

#pragma once

#include <string>

struct options {
	int bucket_size;
	int Nfour;
	bool cuda;
	bool close_pack;
	bool use_glass;
	bool save_force;
	bool do_lc;
	bool do_power;
	bool do_groups;
	bool do_tracers;
	bool do_slice;
	bool do_views;
	bool twolpt;
	bool use_power_file;
	bool plummer;
	double plummerR;
	bool create_glass;
	double toler;
	int read_check;
	int tracer_count;
	int parts_dim;
	int tree_cache_line_size;
	int tree_alloc_line_size;
	int part_cache_line_size;
	int check_freq;
	int max_iter;
	int minrung;
	int view_size;
	int lc_min_group;
	int lc_map_size;
	int min_group;
	int slice_res;
	int nsteps;
	int p3m_Nmin;
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
	double omega_b;
	double omega_lam;
	double omega_c;
	double omega_gam;
	double omega_nu;
	double Theta;
	double ns;
	double Y0;
	double Neff;
	double omega_k;
	size_t nparts;
	std::string config_file;
	std::string test;
	std::string lc_dir;
	std::string gadget4_restart;

	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & close_pack;
		arc & nparts;
		arc & use_glass;
		arc & create_glass;
		arc & gadget4_restart;
		arc & minrung;
		arc & plummer;
		arc & plummerR;
		arc & omega_k;
		arc & omega_lam;
		arc & bucket_size;
		arc & toler;
		arc & nsteps;
		arc & Nfour;
		arc & cuda;
		arc & do_lc;
		arc & save_force;
		arc & do_power;
		arc & do_groups;
		arc & do_tracers;
		arc & do_slice;
		arc & do_views;
		arc & twolpt;
		arc & use_power_file;
		arc & lc_dir;
		arc & tracer_count;
		arc & parts_dim;
		arc & tree_cache_line_size;
		arc & tree_alloc_line_size;
		arc & part_cache_line_size;
		arc & read_check;
		arc & check_freq;
		arc & max_iter;
		arc & lc_min_group;
		arc & min_group;
		arc & view_size;
		arc & slice_res;
		arc & lc_map_size;
		arc & lc_b;
		arc & theta;
		arc & slice_size;
		arc & link_len;
		arc & hsoft;
		arc & GM;
		arc & eta;
		arc & code_to_s;
		arc & code_to_cm;
		arc & code_to_g;
		arc & omega_m;
		arc & omega_r;
		arc & z0;
		arc & z1;
		arc & hubble;
		arc & sigma8;
		arc & omega_b;
		arc & omega_c;
		arc & omega_gam;
		arc & omega_nu;
		arc & Theta;
		arc & ns;
		arc & Y0;
		arc & Neff;
		arc & config_file;
		arc & test;
	}
};

bool process_options(int argc, char *argv[]);
const options& get_options();
void set_options(const options& opts);
