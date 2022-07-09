#pragma once


#include <cosmictiger/containers.hpp>
#include <cosmictiger/fixed.hpp>


struct compressed_block {
	array<fixed<int, 31>, NDIM> xc;
	array<float, NDIM> vc;
	float xmax;
	float vmax;
	array<vector<signed char>, NDIM> x;
	array<vector<signed char>, NDIM> v;
	void write(FILE* fp);
	void read(FILE* fp);
};


struct compressed_particles {
	array<fixed<int, 31>, NDIM> xc;
	array<float, NDIM> vc;
	float xmax;
	float vmax;
	vector<compressed_block> blocks;
	void write(FILE* fp);
	void read(FILE* fp);
	double compression_ratio();
};

