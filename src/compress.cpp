#include <cassert>

#include <cosmictiger/compress.hpp>

double CompressedParticles::compressionRatio() {
    size_t numberParticles = 0;
    size_t numberBytes = 36;
    for (int i = 0; i < blocks.size(); i++) {
        int const n = blocks[i].x[XDIM].size();
        numberBytes += 33 + 2 * NDIM * n;
        numberParticles += n;
    }
    return (double) (24 * numberParticles) / numberBytes;
}

int CompressedParticles::size() {
    int count = 0;
    for (int i = 0; i < blocks.size(); i++) {
        count += blocks[i].x[XDIM].size();
    }
    return count;
}

void CompressedBlock::write(FILE* fp) {
    unsigned char size = x[XDIM].size();
    fwrite(&size, sizeof(unsigned char), 1, fp);
    fwrite(&xc[0], NDIM * sizeof(LightconeReal), 1, fp);
    fwrite(&vc[0], NDIM * sizeof(float), 1, fp);
    fwrite(&xmax, sizeof(float), 1, fp);
    fwrite(&vmax, sizeof(float), 1, fp);
    for (int dim = 0; dim < NDIM; dim++) {
        fwrite(x[dim].data(), sizeof(unsigned short), size, fp);
    }
    for (int dim = 0; dim < NDIM; dim++) {
        fwrite(v[dim].data(), sizeof(unsigned short), size, fp);
    }
}

void CompressedBlock::read(FILE* fp) {
    unsigned char size;
    FREAD(&size, sizeof(unsigned char), 1, fp);
    for (int dim = 0; dim < NDIM; dim++) {
        x[dim].resize(size);
        v[dim].resize(size);
    }
    FREAD(&xc[0], NDIM * sizeof(LightconeReal), 1, fp);
    FREAD(&vc[0], NDIM * sizeof(float), 1, fp);
    FREAD(&xmax, sizeof(float), 1, fp);
    FREAD(&vmax, sizeof(float), 1, fp);
    for (int dim = 0; dim < NDIM; dim++) {
        FREAD(x[dim].data(), sizeof(unsigned short), size, fp);
    }
    for (int dim = 0; dim < NDIM; dim++) {
        FREAD(v[dim].data(), sizeof(unsigned short), size, fp);
    }
}

void CompressedParticles::write(FILE* fp) {
    int size = blocks.size();
    fwrite(&groupId, sizeof(LightconeGroup), 1, fp);
    fwrite(&size, sizeof(int), 1, fp);
    fwrite(&xc[0], NDIM * sizeof(LightconeReal), 1, fp);
    fwrite(&vc[0], NDIM * sizeof(float), 1, fp);
    fwrite(&xmax, sizeof(float), 1, fp);
    fwrite(&vmax, sizeof(float), 1, fp);
    for (int i = 0; i < blocks.size(); i++) {
        blocks[i].write(fp);
    }
}

void CompressedParticles::read(FILE* fp) {
    int size;
    FREAD(&groupId, sizeof(LightconeGroup), 1, fp);
    FREAD(&size, sizeof(int), 1, fp);
    blocks.resize(size);
    FREAD(&xc[0], NDIM * sizeof(LightconeReal), 1, fp);
    FREAD(&vc[0], NDIM * sizeof(float), 1, fp);
    FREAD(&xmax, sizeof(float), 1, fp);
    FREAD(&vmax, sizeof(float), 1, fp);
    for (int i = 0; i < blocks.size(); i++) {
        blocks[i].read(fp);
    }
}

Vector<CompressedBlock> compressedLeavesToBlocks(
    Vector<Pair<int>> const& leaves, Vector<Array<double, 2 * NDIM>>& parts) {
    Vector<CompressedBlock> blocks;
    Range<double, NDIM> xbox;
    Range<double, NDIM> vbox;
    for (auto const& leaf : leaves) {
        if (leaf.second - leaf.first) {
            Array<double, 2 * NDIM> xc;
            for (int dim = 0; dim < NDIM; dim++) {
                xbox.begin[dim] = std::numeric_limits<double>::max();
                xbox.end[dim] = -std::numeric_limits<double>::max();
                vbox.begin[dim] = std::numeric_limits<double>::max();
                vbox.end[dim] = -std::numeric_limits<double>::max();
            }
            for (int i = leaf.first; i < leaf.second; i++) {
                for (int dim = 0; dim < NDIM; dim++) {
                    double const x = parts[i][dim];
                    double const v = parts[i][NDIM + dim];
                    xbox.begin[dim] = std::min(x, xbox.begin[dim]);
                    xbox.end[dim] = std::max(x, xbox.end[dim]);
                    vbox.begin[dim] = std::min(v, vbox.begin[dim]);
                    vbox.end[dim] = std::max(v, vbox.end[dim]);
                }
            }
            for (int dim = 0; dim < NDIM; dim++) {
                xc[dim] = 0.5 * (xbox.begin[dim] + xbox.end[dim]);
                xc[NDIM + dim] = 0.5 * (vbox.begin[dim] + vbox.end[dim]);
            }
            for (int i = leaf.first; i < leaf.second; i++) {
                for (int dim = 0; dim < 2 * NDIM; dim++) {
                    parts[i][dim] -= xc[dim];
                }
            }
            double xmax = 0.0, vmax = 0.0;
            for (int dim = 0; dim < NDIM; dim++) {
                xmax = std::max(xmax, 0.5 * (xbox.end[dim] - xbox.begin[dim]));
                vmax = std::max(vmax, 0.5 * (vbox.end[dim] - vbox.begin[dim]));
            }
            CompressedBlock entry;
            for (int dim = 0; dim < NDIM; dim++) {
                entry.xc[dim] = xc[dim];
                entry.vc[dim] = xc[NDIM + dim];
            }
            xmax *= 1.0 + 1.0 / (1 << 16);
            vmax *= 1.0 + 1.0 / (1 << 16);
            entry.xmax = xmax;
            entry.vmax = vmax;
            for (int i = leaf.first; i < leaf.second; i++) {
                for (int dim = 0; dim < NDIM; dim++) {
                    auto& x = parts[i][dim];
                    auto& v = parts[i][NDIM + dim];
                    x /= xmax;
                    v /= vmax;
                    entry.x[dim].push_back(floor((1 << 15) * (1.0 + x)));
                    entry.v[dim].push_back(floor((1 << 15) * (1.0 + v)));
                }
            }
            blocks.push_back(entry);
        }
    }
    return blocks;
}

void compressedTreeCreate(Vector<Pair<int>>& leaves, Vector<Array<double, 2 * NDIM>>& parts,
    Range<double, 2 * NDIM> const& box, Pair<int> const& partRange) {
    int const numberParticles = partRange.second - partRange.first;
    if (numberParticles <= 64) {
        leaves.push_back(partRange);
    } else {
        int xdim = box.longestDim();
        int lo = partRange.first;
        int hi = partRange.second;
        double xmid = 0.5 * (box.begin[xdim] + box.end[xdim]);
        while (lo < hi) {
            if (parts[lo][xdim] >= xmid) {
                while (lo != hi) {
                    hi--;
                    if (parts[hi][xdim] < xmid) {
                        std::swap(parts[hi], parts[lo]);
                        break;
                    }
                }
            }
            lo++;
        }
        int const mid = hi;
        auto leftRange = partRange;
        auto rightRange = partRange;
        auto leftBox = box;
        auto rightBox = box;
        leftRange.second = rightRange.first = mid;
        leftBox.end[xdim] = rightBox.begin[xdim] = xmid;
        compressedTreeCreate(leaves, parts, leftBox, leftRange);
        compressedTreeCreate(leaves, parts, rightBox, rightRange);
    }
}

#ifndef LC2GADGET2

CompressedParticles compressParticles(Vector<LightconeEntry> const& inparts, LightconeGroup id) {
    Vector<Array<double, 2 * NDIM>> particles;
    particles.reserve(inparts.size());
    Array<double, NDIM> xc;
    Array<double, NDIM> vc;
    for (int dim = 0; dim < NDIM; dim++) {
        xc[dim] = vc[dim] = 0.0;
    }
    for (int i = 0; i < inparts.size(); i++) {
        Array<double, 2 * NDIM> x;
        x[XDIM] = inparts[i].x.toDouble();
        x[YDIM] = inparts[i].y.toDouble();
        x[ZDIM] = inparts[i].z.toDouble();
        x[NDIM + XDIM] = inparts[i].vx;
        x[NDIM + YDIM] = inparts[i].vy;
        x[NDIM + ZDIM] = inparts[i].vz;
        for (int dim = 0; dim < NDIM; dim++) {
            xc[dim] += x[dim];
            vc[dim] += x[NDIM + dim];
        }
        particles.push_back(x);
    }
    for (int dim = 0; dim < NDIM; dim++) {
        xc[dim] /= particles.size();
        vc[dim] /= particles.size();
    }
    double xmax = 0.0;
    double vmax = 0.0;
    for (int i = 0; i < particles.size(); i++) {
        for (int dim = 0; dim < NDIM; dim++) {
            auto& x = particles[i][dim];
            auto& v = particles[i][NDIM + dim];
            x -= xc[dim];
            v -= vc[dim];
            xmax = std::max(xmax, fabs(x));
            vmax = std::max(vmax, fabs(v));
        }
    }
    Range<double, 2 * NDIM> box;
    for (int dim = 0; dim < 2 * NDIM; dim++) {
        box.begin[dim] = std::numeric_limits<double>::max();
        box.end[dim] = -std::numeric_limits<double>::max();
    }
    for (int i = 0; i < particles.size(); i++) {
        for (int dim = 0; dim < NDIM; dim++) {
            particles[i][dim] /= xmax;
            particles[i][NDIM + dim] /= vmax;
        }
        for (int dim = 0; dim < 2 * NDIM; dim++) {
            auto const& x = particles[i][dim];
            box.begin[dim] = std::min(box.begin[dim], x);
            box.end[dim] = std::max(box.end[dim], x);
        }
    }
    Vector<Pair<int>> leaves;
    Pair<int> partRange;
    partRange.first = 0;
    partRange.second = particles.size();
    compressedTreeCreate(leaves, particles, box, partRange);
    CompressedParticles arc;
    arc.blocks = compressedLeavesToBlocks(leaves, particles);
    for (int dim = 0; dim < NDIM; dim++) {
        arc.xc[dim] = xc[dim];
        arc.vc[dim] = vc[dim];
    }
    arc.xmax = xmax;
    arc.vmax = vmax;
    arc.groupId = id;
    return arc;
}

#endif
