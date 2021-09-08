/*
 * lc_group_archive.hpp
 *
 *  Created on: Sep 8, 2021
 *      Author: dmarce1
 */

#ifndef LC_GROUP_ARCHIVE_HPP_
#define LC_GROUP_ARCHIVE_HPP_

#include <array>

#include <cosmictiger/defs.hpp>

struct lc_group_archive {
	long long id;
	std::array<double, NDIM> com;
	std::array<float, NDIM> vel;
	std::array<float, NDIM> lang;
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
	float Ixx;
	float Ixy;
	float Ixz;
	float Iyy;
	float Iyz;
	float Izz;
	bool incomplete;
};


#endif /* LC_GROUP_ARCHIVE_HPP_ */
