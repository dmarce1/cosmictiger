#pragma once

#include <cosmictiger/containers.hpp>
#include <cosmictiger/lightcone.hpp>

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
    LcGroup groupId;
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

CompressedParticles compressParticles(Vector<LcEntry> const& inparts, LcGroup);
