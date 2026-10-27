/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 See LICENSE for details.
 */

#pragma once

#include <string>
#include <variant>
#include <vector>

struct Options;

using OptionsMetadataValue = std::variant<bool, int, double, std::string>;

struct OptionsMetadataEntry {
    std::string name;
    OptionsMetadataValue value;
};

// Effective public application options at the output epoch, expressed in their
// CLI units, together with fixed build descriptors. Includes configuration;
// excludes help and other internal/derived fields.
std::vector<OptionsMetadataEntry> getOptionsMetadata(Options const& options);

// Call once at process startup, before HPX parses or removes its own arguments.
// The recorded vector includes argv[0] and retains argument boundaries verbatim.
void recordCommandLineArguments(int argc, char* argv[]);
std::vector<std::string> const& getCommandLineArguments();
