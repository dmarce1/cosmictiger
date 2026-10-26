#pragma once

#include <limits>
#include <string>

#include <cosmictiger/analysis/groups.hpp>
#include <cosmictiger/analysis/lightcone.hpp>
#include <cosmictiger/analysis/rockstar_record.hpp>
#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/math/fp16.hpp>

// Paper references in this implementation use Behroozi, Wechsler & Wu (2013),
// arXiv:1110.4372v3: https://arxiv.org/html/1110.4372v3

// #define ROCKSTAR_CPU_BUCKET_SIZE 32
inline constexpr int kRockstarBucketSize = 64;
inline constexpr int kRockstarNoGroup = 0x7FFFFFFF;
// Sec. III.2: refinement fraction and minimum seed size.
inline constexpr double kRockstarLinkingFraction = 0.7;
inline constexpr int kRockstarMinGroup = 10;
// Sec. III.5.3: minimum retained bound fraction.
inline constexpr double kRockstarMinBound = 0.5;

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
        Array<float, 2 * kDimensions> xValue;
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
    ParticleInteger index;
    float minDist2;
    int rank = 0;
    GroupInteger identity = kNoParticleGroup;
    GroupInteger previousHalo = kNoParticleGroup;
    GroupInteger previousHost = kNoParticleGroup;
    int ordinal = -1;
};

struct RockstarTree {
    int particleBegin;
    int particleEnd;
    Array<int, kChildCount> children;
    Range<float, 2 * kDimensions> box;
    int activeCount;
    bool active;
    bool lastActive;
    DeviceVector<int> neighbors;
};

struct Subgroup {
    int id = -1;
    int parent;
    int internalParent = -1;
    GroupInteger catalogueId = kNoParticleGroup;
    double massVir = 0.0;
    double mass200 = 0.0;
    double mass500 = 0.0;
    double mass2500 = 0.0;
    double radius500 = 0.0;
    Vector<int> children;
    DeviceVector<RockstarParticle> parts;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float vxc = 0.0f;
    float vyc = 0.0f;
    float vzc = 0.0f;
    float vxb = 0.0f;
    float vyb = 0.0f;
    float vzb = 0.0f;
    float t = 0.0f;
    float w = 0.0f;
    float sigmaXMin;
    float rDyn = 0.0f;
    float sigma2V = 0.0f;
    float sigma2X = 0.0f;
    float vcircMax = 0.0f;
    double rVir = 0.0;
    int hostParticleCount = 0;
    int depth;
    Subgroup() {
        depth = -1;
        sigmaXMin = std::numeric_limits<float>::max();
        parent = -1;
    }
};

Vector<Subgroup> rockstarFindSubgroups(Vector<ParticleData> const& parts, float scale);
Vector<RockstarRecord> rockstarFindSubgroups(Vector<LightconeEntry> const& parts, double);
void rockstarFindSubgroupsCuda(DeviceVector<RockstarTree>& nodes, DeviceVector<int> const& leaves,
    DeviceVector<RockstarParticle>& parts, float linkingLength, int& nextId);
void rockstarAssignLinklenCuda(DeviceVector<RockstarTree> const& nodes,
    DeviceVector<int> const& leaves, DeviceVector<RockstarParticle>& parts, float linkingLength,
    bool phase);
bool rockstarCudaFree();

// Shared by the recursive finder, snapshot writer, and lightcone writer.
void rockstarUpdateStatistics(
    Subgroup& group, float positionFactor, float velocityFactor, float scale);
void rockstarFinalizeSubgroups(Vector<Subgroup>& groups, float scale);
// Final catalogue thresholding after unbinding, preserving enclosing-host mass.
void rockstarPruneCatalogueGroups(Vector<Subgroup>& groups, float scale);
DeviceVector<RockstarParticle> rockstarGatherParticles(Vector<Subgroup> const& groups, int index);
RockstarRecord rockstarMakeRecord(Vector<Subgroup> const& groups, int index, float scale,
    Array<double, kDimensions> const& origin, bool inclusive = true);
// Checked IEEE binary16 storage for finite physical catalogue properties.
Float16 rockstarEncodeHalf(double value, char const* fieldName);
// Publish native records and an encoding sidecar after a successful close.
void rockstarWriteCatalogue(
    std::string const& filename, RockstarRecord const* records, size_t count);
GroupEntry rockstarMakeGroupEntry(Vector<Subgroup> const& groups, int index, float scale,
    Array<double, kDimensions> const& origin);
// A single candidate, exposed for independent unbinding regressions.
bool rockstarHaloBound(Subgroup& halo, Vector<RockstarParticle>& unboundParticles, float scale);
