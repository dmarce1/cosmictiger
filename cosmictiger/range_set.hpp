/*
 * range_set.hpp
 *
 *  Created on: Jul 28, 2021
 *      Author: dmarce1
 */

#ifndef RANGE_SET_HPP_
#define RANGE_SET_HPP_



#include <set>

class range_set {
	struct less {
		size_t operator()(std::pair<int, int> a, std::pair<int, int> b) {
			return a.first < b.first;
		}
	};
	int sz;
	std::set<std::pair<int, int>, less> ranges;
public:
	range_set();
	size_t size() const;
	void insert(std::pair<int, int> a);
};


#endif /* RANGE_SET_HPP_ */
