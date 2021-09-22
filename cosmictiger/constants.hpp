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
constexpr double seconds_to_years = 1.0 / (365.24*24*60*60);
constexpr double mpc_to_cm = 3.086e24;
constexpr double spyr = 365.24 * 24.0 * 60.0 * 60.0;
constexpr double M0 = 1.989e33;
}

namespace cosmic_constants {
constexpr double H0 = 1e7 / constants::c;
}
