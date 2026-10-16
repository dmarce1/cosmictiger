/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the GNU General Public License, version 2 or later.
 See LICENSE for details.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <utility>

#include <cosmictiger/options.hpp>

namespace legacy_options {

// The legacy lightcone format wrote native Options bytes, including string object
// representations. It is ABI-specific; only its scalar values may be restored.
static_assert(std::is_standard_layout_v<Options>);
inline constexpr size_t kLightconeOptionsBytes = offsetof(Options, outputUnits);

inline bool readLightconeOptions(FILE* stream, Options& result) {
    std::array<unsigned char, kLightconeOptionsBytes> bytes;
    if (std::fread(bytes.data(), 1, bytes.size(), stream) != bytes.size()) {
        return false;
    }
    Options options{};
    auto const readDouble = [&bytes](size_t offset, double& value) {
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
    };
    readDouble(offsetof(Options, codeToCm), options.codeToCm);
    readDouble(offsetof(Options, codeToG), options.codeToG);
    readDouble(offsetof(Options, codeToS), options.codeToS);
    readDouble(offsetof(Options, hubble), options.hubble);
    readDouble(offsetof(Options, omegaM), options.omegaM);
    readDouble(offsetof(Options, hsoft), options.hsoft);
    readDouble(offsetof(Options, lcB), options.lcB);
    result = std::move(options);
    return true;
}

}    // namespace legacy_options
