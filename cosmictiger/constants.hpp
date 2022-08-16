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

#pragma once

namespace constants {
constexpr double c = 2.99792458e10;
constexpr double G = 6.67259e-8;
constexpr double h = 6.6260755e-27;
constexpr double me = 9.1093897e-28;
constexpr double mh = 1.6733e-24;
constexpr double sigma = 5.67051e-5;
constexpr double H0 = 1e7 / 3.086e24;
constexpr double evtoerg = 1.6021772e-12;
constexpr double kb = 1.380658e-16;
constexpr double hbar = 1.05457266e-27;
constexpr double Tcmb = 2.73;
constexpr double sigma_T = 6.6524587158e-25;
constexpr double evtoK = 11604.5250061657;
constexpr double seconds_to_years = 1.0 / (365.24 * 24 * 60 * 60);
constexpr double mpc_to_cm = 3.085678e24;
constexpr double spyr = 365.24 * 24.0 * 60.0 * 60.0;
constexpr double M0 = 1.989e33;
constexpr double avo = 6.0221409e+23;
constexpr double e = 4.8032068e-10;
}

namespace cosmic_constants {
constexpr double H0 = 1e7 / constants::c;
}
