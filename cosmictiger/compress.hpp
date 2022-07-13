#pragma once


#include <cosmictiger/containers.hpp>
#include <cosmictiger/lightcone.hpp>


struct compressed_block {
	array<fixed<int, 31>, NDIM> xc;
	array<float, NDIM> vc;
	float xmax;
	float vmax;
	array<vector<unsigned short>, NDIM> x;
	array<vector<unsigned short>, NDIM> v;
	void write(FILE* fp);
	void read(FILE* fp);
};



struct compressed_particles {
	lc_group group_id;
	array<fixed<int, 31>, NDIM> xc;
	array<float, NDIM> vc;
	float xmax;
	float vmax;
	vector<compressed_block> blocks;
	void write(FILE* fp);
	void read(FILE* fp);
	double compression_ratio();
	int size();
};


compressed_particles compress_particles(const vector<lc_entry>& inparts, lc_group);
