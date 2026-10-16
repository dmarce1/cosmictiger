#pragma once

#include <cosmictiger/containers.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/fp16.hpp>
#include <cosmictiger/groups.hpp>
#include <cosmictiger/lightcone.hpp>

// #define ROCKSTAR_CPU_BUCKET_SIZE 32
#define ROCKSTAR_BUCKET_SIZE 64
#define ROCKSTAR_NO_GROUP 0x7FFFFFFF
#define ROCKSTAR_HAS_GROUP -1
#define ROCKSTAR_FF 0.7
#define ROCKSTAR_MIN_GROUP 10
#define ROCKSTAR_MIN_BOUND 0.5
#define ROCKSTAR_TARGET_BLOCKS (1024)

struct RockstarParticles {
    float* x;
    float* y;
    float* z;
    float* vx;
    float* vy;
    float* vz;
    int* subgroup;
};

struct RockstarParticle {
    union {
        Array<float, 2 * NDIM> xValue;
        struct {
            float x;
            float y;
            float z;
            float vx;
            float vy;
            float vz;
        };
    };
    int subgroup;
    PartInt index;
    float minDist2;
};

struct RockstarTree {
    int partBegin;
    int partEnd;
    Array<int, NCHILD> children;
    Range<float, 2 * NDIM> box;
    int activeCount;
    bool active;
    bool lastActive;
    DeviceVector<int> neighbors;
};

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

struct Subgroup {
    int id;
    int parent;
    Vector<int> children;
    DeviceVector<RockstarParticle> parts;
    float x;
    float y;
    float z;
    float vxc;
    float vyc;
    float vzc;
    float vxb;
    float vyb;
    float vzb;
    float t;
    float w;
    float sigmaXMin;
    float rDyn;
    float sigma2V;
    float sigma2X;
    float vcircMax;
    float rVir;
    int hostPartCnt;
    int depth;
    Subgroup() {
        depth = -1;
        sigmaXMin = std::numeric_limits<float>::max();
        parent = -1;
    }
};

Vector<Subgroup> rockstarFindSubgroups(Vector<ParticleData> const& parts, float scale);
Vector<RockstarRecord> rockstarFindSubgroups(Vector<LcEntry> const& parts, double);
void rockstarFindSubgroupsCuda(DeviceVector<RockstarTree>& nodes, DeviceVector<int> const& leaves,
    DeviceVector<RockstarParticle>& parts, float linkLen, int& nextId);
void rockstarAssignLinklenCuda(DeviceVector<RockstarTree> const& nodes,
    DeviceVector<int> const& leaves, DeviceVector<RockstarParticle>& parts, float linkLen,
    bool phase);
bool rockstarCudaFree();
