/*
 * group_entry.hpp
 *
 *  Created on: Aug 18, 2021
 *      Author: dmarce1
 */

#ifndef GROUP_ENTRY_HPP_
#define GROUP_ENTRY_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/containers.hpp>

#include <array>
#include <unordered_map>


using group_int = long long;


struct group_entry {
	group_int id;
	array<double, NDIM> com;
	array<float, NDIM> vel;
	array<float, NDIM> lang;
	float mass;
	float ekin;
	float epot;
	float r25;
	float r50;
	float r75;
	float r90;
	float rmax;
	float ravg;
	float vxdisp;
	float vydisp;
	float vzdisp;
	float xdisp;
	float ydisp;
	float zdisp;
	float Ixx;
	float Ixy;
	float Ixz;
	float Iyy;
	float Iyz;
	float Izz;
	int parent_count;
	std::unordered_map<group_int, int> parents;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & id;
		arc & com;
		arc & vel;
		arc & lang;
		arc & mass;
		arc & ekin;
		arc & epot;
		arc & r25;
		arc & r50;
		arc & r75;
		arc & r90;
		arc & rmax;
		arc & ravg;
		arc & vxdisp;
		arc & vydisp;
		arc & vzdisp;
		arc & xdisp;
		arc & ydisp;
		arc & zdisp;
		arc & Ixx;
		arc & Ixy;
		arc & Ixz;
		arc & Iyy;
		arc & Iyz;
		arc & Izz;
		arc & parent_count;
		arc & parents;
	}
	group_entry() = default;
	group_entry(group_entry&&) = default;
	group_entry(const group_entry&) = delete;
	group_entry& operator=(group_entry&&) = default;
	group_entry& operator=(const group_entry&) = delete;
	void write(FILE* fp) const;
	bool read(FILE* fp);
};




#endif /* GROUP_ENTRY_HPP_ */
