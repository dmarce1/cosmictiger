/*
 * domain.hpp
 *
 *  Created on: Jul 17, 2021
 *      Author: dmarce1
 */

#ifndef DOMAIN_HPP_
#define DOMAIN_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/range.hpp>

void domains_begin();
void domains_end();
void domains_rebound();
range<double> domains_find_my_box();
range<double> domains_range(size_t key);
void domains_save(std::ofstream& fp);
void domains_load(FILE *fp);

struct domain_t {
	range<double> box;
	double midx;
	int rank;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & box;
		arc & midx;
		arc & rank;
	}
};

struct domain_global {
	range<double> box;
	pair<int> proc_range;
	part_int total_count;
	part_int lo_count;
	double midhi;
	double midlo;
	size_t key;
};

struct domain_local {
	range<double> box;
	pair<int> proc_range;
	pair<part_int> part_range;
	int depth;
};

#endif /* DOMAIN_HPP_ */
