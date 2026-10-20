/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

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

#ifndef OUTPUT_HPP_
#define OUTPUT_HPP_

#include <string>

// Resolve simulation output beneath output.directory; developer test modes retain
// their existing working-directory files. Absolute paths are already resolved.
std::string outputPath(std::string const& filename);

void outputTracers(int number);
void outputSlice(int number, double time);

#endif /* OUTPUT_HPP_ */
