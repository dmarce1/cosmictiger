#pragma once

#include <type_traits>

#include <cosmictiger/fp16.hpp>

// Native catalogue ABI shared by the writer and standalone snapshot exporter.
union RockstarId {
    struct {
        unsigned long long fofId : 48;
        unsigned long long haloId : 15;
        unsigned long long sys : 1;
    };
    unsigned long long id;
};

struct RockstarRecord {
    RockstarId id;
    double x;
    double y;
    double z;
    float mvir;
    float e;
    float jx;
    float jy;
    float jz;
    float m200;
    float m500;
    float m2500;
    short pid;
    Float16 zValue;
    Float16 rmax;
    Float16 rvir;
    Float16 rvmax;
    Float16 rs;
    Float16 rkly;
    Float16 sigX;
    Float16 deltaX;
    Float16 toW;
    Float16 sigV;
    Float16 sigVb;
    Float16 vcircMax;
    Float16 deltaV;
    Float16 boa;
    Float16 coa;
    Float16 ax;
    Float16 ay;
    Float16 az;
    Float16 boa500;
    Float16 coa500;
    Float16 ax500;
    Float16 ay500;
    Float16 az500;
    Float16 lambda;
    Float16 lambdaB;
    Float16 vxc;
    Float16 vyc;
    Float16 vzc;
    Float16 vxb;
    Float16 vyb;
    Float16 vzb;
};

static_assert(std::is_trivial_v<RockstarRecord> && std::is_standard_layout_v<RockstarRecord>);
static_assert(sizeof(RockstarId) == 8);
static_assert(sizeof(RockstarRecord) == 128);
