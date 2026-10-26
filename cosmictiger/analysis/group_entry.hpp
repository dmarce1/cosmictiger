/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#ifndef GROUP_ENTRY_HPP_
#define GROUP_ENTRY_HPP_

#include <array>
#include <unordered_map>

#include <cosmictiger/analysis/group_header.hpp>
#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>

using GroupInteger = long long;

struct GroupEntry {
    GroupInteger id;
    Array<double, kDimensions> com;
    Array<float, kDimensions> vel;
    Array<float, kDimensions> lang;
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
    float ixx;
    float ixy;
    float ixz;
    float iyy;
    float iyz;
    float izz;
    bool incomplete;
    std::unordered_map<GroupInteger, int> parents;
    GroupEntry() = default;
    GroupEntry(GroupEntry&&) = default;
    GroupEntry(GroupEntry const&) = delete;
    GroupEntry& operator=(GroupEntry&&) = default;
    GroupEntry& operator=(GroupEntry const&) = delete;
    void write(FILE* fp);
    bool read(FILE* fp);
};

#endif /* GROUP_ENTRY_HPP_ */
