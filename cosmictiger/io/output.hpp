/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#ifndef OUTPUT_HPP_
#define OUTPUT_HPP_

#include <string>

// Resolve simulation output beneath output.directory; developer test modes retain
// their existing working-directory files. Absolute paths are already resolved.
std::string outputPath(std::string const& filename);

void outputTracers(int number);
void outputSlice(int number, double time);

#endif /* OUTPUT_HPP_ */
