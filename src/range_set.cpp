#include <cosmictiger/range_set.hpp>

range_set::range_set() {
	sz = 0;
}

size_t range_set::size() const {
	return sz;
}

void range_set::insert(std::pair<int, int> a) {
	if (ranges.size()) {
		auto i = ranges.upper_bound(a);
		if (i == ranges.begin()) {
			if (i->first > a.second) {
				ranges.insert(a);
				sz += a.second - a.first;
			} else {
				auto new_range = *i;
				new_range.first = std::min(a.first, i->first);
				sz += new_range.second - i->second;
				ranges.erase(i);
				ranges.insert(new_range);
			}
		} else {
			i--;
			if (i->second < a.first) {
				ranges.insert(a);
				sz += a.second - a.first;
			} else {
				auto new_range = *i;
				new_range.second = std::max(a.second, i->second);
				sz += new_range.second - i->second;
				ranges.erase(i);
				ranges.insert(new_range);
			}
		}
	} else {
		sz += a.second - a.first;
		ranges.insert(a);
	}
}

