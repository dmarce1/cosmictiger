#pragma once

#include <cosmictiger/time.hpp>
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
	double years;
	time_type itime;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & a;
		arc & tau;
		arc & tau_max;
		arc & cosmicK;
		arc & esum0;
		arc & iter;
		arc & total_processed;
		arc & flops;
		arc & runtime;
		arc & itime;
		arc & years;
	}
};

void write_checkpoint(driver_params params);
driver_params read_checkpoint(int checknum);

void driver();
