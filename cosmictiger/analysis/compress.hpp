#pragma once

#include <cosmictiger/analysis/lightcone.hpp>
#include <cosmictiger/core/containers.hpp>

struct CompressedBlock {
    Array<Fixed<int, 31>, NDIM> xc;
    Array<float, NDIM> vc;
    float xmax;
    float vmax;
    Array<Vector<unsigned short>, NDIM> x;
    Array<Vector<unsigned short>, NDIM> v;
    void write(FILE* fp);
    void read(FILE* fp);
};

struct CompressedParticles {
    LightconeGroup groupId;
    Array<Fixed<int, 31>, NDIM> xc;
    Array<float, NDIM> vc;
    float xmax;
    float vmax;
    Vector<CompressedBlock> blocks;
    void write(FILE* fp);
    void read(FILE* fp);
    double compressionRatio();
    int size();
};

CompressedParticles compressParticles(Vector<LightconeEntry> const& inputParticles, LightconeGroup);
