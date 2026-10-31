#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <map>
#include <stdexcept>

// An epoch spans many drift batches. Retain double sums until the shell is
// flushed; the catalogue's on-disk map values remain single precision.
class LightconeMap {
    std::map<int, double> pixels_;

    template <class T>
    static void write(FILE* file, T const* values, size_t count) {
        if (std::fwrite(values, sizeof(T), count, file) != count) {
            throw std::runtime_error("Unable to write lightcone map");
        }
    }

    template <class T>
    static void read(FILE* file, T* values, size_t count) {
        if (std::fread(values, sizeof(T), count, file) != count) {
            throw std::runtime_error("Unable to read lightcone map checkpoint");
        }
    }

public:
    void clear() {
        pixels_.clear();
    }

    void add(int pixel, double weight, int nside) {
        if (pixel < 0 || size_t(pixel) >= 12 * size_t(nside) * size_t(nside)) {
            throw std::out_of_range("Lightcone map pixel is outside its HEALPix grid");
        }
        pixels_[pixel] += weight;
    }

    void writeMap(FILE* file, int nside) const {
        int const minimum = pixels_.empty() ? 0 : pixels_.begin()->first;
        int const maximum = pixels_.empty() ? -1 : pixels_.rbegin()->first;
        write(file, &nside, 1);
        write(file, &minimum, 1);
        write(file, &maximum, 1);
        std::array<float, 4096> values{};
        auto pixel = pixels_.begin();
        for (int begin = minimum; begin <= maximum;) {
            size_t const count = std::min(size_t(maximum - begin) + 1, values.size());
            values.fill(0.0f);
            while (pixel != pixels_.end() && size_t(pixel->first - begin) < count) {
                values[pixel->first - begin] = float(pixel->second);
                ++pixel;
            }
            write(file, values.data(), count);
            begin += int(count);
        }
    }

    void save(FILE* file, int nside) const {
        write(file, &nside, 1);
        size_t const count = pixels_.size();
        write(file, &count, 1);
        for (auto const& pixel : pixels_) {
            write(file, &pixel.first, 1);
            write(file, &pixel.second, 1);
        }
    }

    void load(FILE* file, int nside) {
        int savedNside;
        size_t count;
        read(file, &savedNside, 1);
        read(file, &count, 1);
        if (savedNside != nside || count > 12 * size_t(nside) * size_t(nside)) {
            throw std::runtime_error("Lightcone checkpoint map resolution mismatch");
        }
        LightconeMap restored;
        for (size_t i = 0; i < count; ++i) {
            int pixel;
            double weight;
            read(file, &pixel, 1);
            read(file, &weight, 1);
            restored.add(pixel, weight, nside);
        }
        pixels_.swap(restored.pixels_);
    }
};
