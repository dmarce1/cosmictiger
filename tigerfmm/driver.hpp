#pragma once


#include <tigerfmm/time.hpp>
#include <cstdio>

struct driver_params {
	double a;
	double tau;
	double tau_max;
	double cosmicK;
	double esum0;
	int iter;
	size_t total_processed;
	double flops;
	double runtime;
	time_type itime;
};

void write_checkpoint(driver_params params);
void read_checkpoint(driver_params& params, int checknum);


void driver();
