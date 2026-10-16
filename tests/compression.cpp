/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/compress.hpp>

void cosmictigerAssert(char const* expression, bool condition, char const* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "Assertion failed: %s at %s:%d\n", expression, file, line);
        std::abort();
    }
}

namespace detail {

using Particle = std::array<double, 2 * NDIM>;

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

LightconeEntry makeParticle(double x, double y, double z, float vx, float vy, float vz) {
    LightconeEntry particle{};
    particle.x = x;
    particle.y = y;
    particle.z = z;
    particle.vx = vx;
    particle.vy = vy;
    particle.vz = vz;
    return particle;
}

std::vector<Particle> decode(CompressedParticles const& archive) {
    std::vector<Particle> particles;
    require(std::isfinite(archive.xmax) && std::isfinite(archive.vmax),
        "Archive scales must be finite");
    for (auto const& block : archive.blocks) {
        size_t const count = block.x[0].size();
        require(count > 0 && count <= 64, "Compressed blocks must contain 1 to 64 particles");
        require(
            std::isfinite(block.xmax) && std::isfinite(block.vmax), "Block scales must be finite");
        for (int dim = 0; dim < NDIM; dim++) {
            require(block.x[dim].size() == count && block.v[dim].size() == count,
                "Compressed block dimensions must have equal lengths");
            require(std::isfinite(archive.vc[dim]) && std::isfinite(block.vc[dim]),
                "Compressed velocity centers must be finite");
        }
        for (size_t i = 0; i < count; i++) {
            Particle particle;
            for (int dim = 0; dim < NDIM; dim++) {
                double const x = (block.x[dim][i] + 0.5) / (1 << 15) - 1.0;
                double const v = (block.v[dim][i] + 0.5) / (1 << 15) - 1.0;
                particle[dim] = (x * block.xmax + block.xc[dim].toDouble()) * archive.xmax +
                    archive.xc[dim].toDouble();
                particle[NDIM + dim] =
                    (v * block.vmax + block.vc[dim]) * archive.vmax + archive.vc[dim];
                require(std::isfinite(particle[dim]) && std::isfinite(particle[NDIM + dim]),
                    "Decoded coordinates must be finite");
            }
            particles.push_back(particle);
        }
    }
    std::sort(particles.begin(), particles.end());
    return particles;
}

void testCompression(Vector<LightconeEntry> const& input) {
    constexpr LightconeGroup kGroupId = 12345;
    auto archive = compressParticles(input, kGroupId);
    require(archive.groupId == kGroupId && archive.size() == input.size(),
        "Compression must retain group identity and particle count");
    if (input.empty()) {
        require(archive.blocks.empty() && archive.xmax == 0.0f && archive.vmax == 0.0f,
            "An empty archive must have no blocks and zero scales");
        for (int dim = 0; dim < NDIM; dim++) {
            require(archive.xc[dim].raw() == 0 && archive.vc[dim] == 0.0f,
                "An empty archive must have zero centers");
        }
    }

    auto const closeFile = [](FILE* file) { std::fclose(file); };
    std::unique_ptr<FILE, decltype(closeFile)> file(std::tmpfile(), closeFile);
    require(file != nullptr, "Could not create compression round-trip file");
    archive.write(file.get());
    require(std::ferror(file.get()) == 0, "Could not write compression round-trip file");
    std::rewind(file.get());
    CompressedParticles restored;
    restored.read(file.get());
    require(restored.groupId == kGroupId && restored.size() == input.size(),
        "Serialization must retain group identity and particle count");
    auto const decoded = decode(restored);
    require(decoded == decode(archive), "Serialization must preserve compressed data");
    std::vector<Particle> expected;
    for (auto const& particle : input) {
        expected.push_back({particle.x.toDouble(), particle.y.toDouble(), particle.z.toDouble(),
            particle.vx, particle.vy, particle.vz});
    }
    std::sort(expected.begin(), expected.end());
    require(decoded.size() == expected.size(), "Decoding must retain every particle");
    double const positionTolerance = 4.0e-5 * std::max(1.0, double(archive.xmax));
    double const velocityTolerance = 4.0e-5 * std::max(1.0, double(archive.vmax));
    for (size_t i = 0; i < expected.size(); i++) {
        for (int dim = 0; dim < NDIM; dim++) {
            require(std::abs(decoded[i][dim] - expected[i][dim]) <= positionTolerance,
                "Position round trip exceeded quantization tolerance");
            require(std::abs(decoded[i][NDIM + dim] - expected[i][NDIM + dim]) <= velocityTolerance,
                "Velocity round trip exceeded quantization tolerance");
        }
    }
}

void testCase(std::string const& name) {
    auto const particle = makeParticle(0.125, -0.25, 0.375, 2.0f, -4.0f, 8.0f);
    if (name == "empty") {
        testCompression({});
    } else if (name == "single") {
        testCompression({particle});
    } else if (name == "repeated") {
        testCompression(Vector<LightconeEntry>(129, particle));
    } else if (name == "repeated_clusters") {
        Vector<LightconeEntry> input(65, makeParticle(-0.25, 0.0, 0.0, 0.0f, 0.0f, 0.0f));
        input.resize(130, makeParticle(0.25, 0.0, 0.0, 0.0f, 0.0f, 0.0f));
        testCompression(input);
    } else if (name == "zero_position_spread" || name == "zero_velocity_spread" ||
        name == "normal") {
        Vector<LightconeEntry> input;
        for (int i = 0; i < 193; i++) {
            auto entry = particle;
            if (name != "zero_position_spread") {
                entry.x = -0.375 + i / 512.0;
                entry.y = -0.125 + i / 1024.0;
                entry.z = 0.25 - i / 1024.0;
            }
            if (name != "zero_velocity_spread") {
                entry.vx = -3.0f + i / 32.0f;
                entry.vy = -1.0f + i / 128.0f;
                entry.vz = 2.0f - i / 64.0f;
            }
            input.push_back(entry);
        }
        testCompression(input);
    } else {
        throw std::runtime_error("Unknown compression test case");
    }
}

}    // namespace detail

int main(int argc, char** argv) {
    try {
        if (argc == 2) {
            detail::testCase(argv[1]);
        } else {
            for (char const* name : {"empty", "single", "repeated", "repeated_clusters",
                     "zero_position_spread", "zero_velocity_spread", "normal"}) {
                detail::testCase(name);
            }
        }
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }
    std::puts("Compression regressions passed");
    return EXIT_SUCCESS;
}
