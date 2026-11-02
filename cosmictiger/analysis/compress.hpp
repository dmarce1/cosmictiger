#pragma once

#include <cosmictiger/analysis/lightcone.hpp>
#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/math/vector3.hpp>

struct CompressedBlock {
    Vector3<Fixed<int, 31>> xc;
    Vector3<float> vc;
    float xmax;
    float vmax;
    Array<Vector<unsigned short>, kDimensions> x;
    Array<Vector<unsigned short>, kDimensions> v;
    void write(FILE* fp);
    void read(FILE* fp);
};

struct CompressedParticles {
    LightconeGroup groupId;
    Vector3<Fixed<int, 31>> xc;
    Vector3<float> vc;
    float xmax;
    float vmax;
    Vector<CompressedBlock> blocks;
    void write(FILE* fp);
    void read(FILE* fp);
    double compressionRatio();
    int size();
};

CompressedParticles compressParticles(Vector<LightconeEntry> const& inputParticles, LightconeGroup);
