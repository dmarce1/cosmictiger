#include <algorithm>
#include <cassert>

#include <cosmictiger/compress.hpp>

double CompressedParticles::compressionRatio() {
    size_t numberOfParticles = 0;
    size_t numberOfBytes = 36;
    for (int i = 0; i < blocks.size(); i++) {
        int const n = blocks[i].x[XDIM].size();
        numberOfBytes += 33 + 2 * NDIM * n;
        numberOfParticles += n;
    }
    return (double) (24 * numberOfParticles) / numberOfBytes;
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
                // The normalized +1 endpoint is outside the signed fixed-point range.
                entry.xc[dim] = std::min(xc[dim], LightconeReal::max().toDouble());
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
                    x = xmax > 0.0 ? x / xmax : 0.0;
                    v = vmax > 0.0 ? v / vmax : 0.0;
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
    Range<double, 2 * NDIM> const& box, Pair<int> const& particleRange) {
    int const numberOfParticles = particleRange.second - particleRange.first;
    if (numberOfParticles <= 64) {
        leaves.push_back(particleRange);
    } else {
        int xdim = box.longestDim();
        int lo = particleRange.first;
        int hi = particleRange.second;
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
        int mid = hi;
        if (mid == particleRange.first || mid == particleRange.second) {
            mid = particleRange.first + numberOfParticles / 2;
            std::nth_element(parts.begin() + particleRange.first, parts.begin() + mid,
                parts.begin() + particleRange.second,
                [xdim](auto const& left, auto const& right) { return left[xdim] < right[xdim]; });
            xmid = parts[mid][xdim];
        }
        auto leftRange = particleRange;
        auto rightRange = particleRange;
        auto leftBox = box;
        auto rightBox = box;
        leftRange.second = rightRange.first = mid;
        leftBox.end[xdim] = rightBox.begin[xdim] = xmid;
        compressedTreeCreate(leaves, parts, leftBox, leftRange);
        compressedTreeCreate(leaves, parts, rightBox, rightRange);
    }
}

CompressedParticles compressParticles(
    Vector<LightconeEntry> const& inputParticles, LightconeGroup id) {
    if (inputParticles.empty()) {
        CompressedParticles arc{};
        arc.groupId = id;
        return arc;
    }
    Vector<Array<double, 2 * NDIM>> particles;
    particles.reserve(inputParticles.size());
    Array<double, NDIM> xc;
    Array<double, NDIM> vc;
    for (int dim = 0; dim < NDIM; dim++) {
        xc[dim] = vc[dim] = 0.0;
    }
    for (int i = 0; i < inputParticles.size(); i++) {
        Array<double, 2 * NDIM> x;
        x[XDIM] = inputParticles[i].x.toDouble();
        x[YDIM] = inputParticles[i].y.toDouble();
        x[ZDIM] = inputParticles[i].z.toDouble();
        x[NDIM + XDIM] = inputParticles[i].vx;
        x[NDIM + YDIM] = inputParticles[i].vy;
        x[NDIM + ZDIM] = inputParticles[i].vz;
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
            particles[i][dim] = xmax > 0.0 ? particles[i][dim] / xmax : 0.0;
            particles[i][NDIM + dim] = vmax > 0.0 ? particles[i][NDIM + dim] / vmax : 0.0;
        }
        for (int dim = 0; dim < 2 * NDIM; dim++) {
            auto const& x = particles[i][dim];
            box.begin[dim] = std::min(box.begin[dim], x);
            box.end[dim] = std::max(box.end[dim], x);
        }
    }
    Vector<Pair<int>> leaves;
    Pair<int> particleRange;
    particleRange.first = 0;
    particleRange.second = particles.size();
    compressedTreeCreate(leaves, particles, box, particleRange);
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
