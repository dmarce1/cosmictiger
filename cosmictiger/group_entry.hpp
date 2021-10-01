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
	group_entry() = default;
	group_entry(group_entry&&) = default;
	group_entry(const group_entry&) = delete;
	group_entry& operator=(group_entry&&) = default;
	group_entry& operator=(const group_entry&) = delete;
	void write(FILE* fp);
	bool read(FILE* fp);
};




#endif /* GROUP_ENTRY_HPP_ */
