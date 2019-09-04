/*
 * field_defs.hpp
 *
 *  Created on: Jul 29, 2019
 *      Author: dmarce1
 */

#ifndef OCTOTIGER_HYDRO_DEFS_HPP_
#define OCTOTIGER_HYDRO_DEFS_HPP_

#include <cmath>

static constexpr int NDIM = 3;

static constexpr int rho_i = 0;
static constexpr int egas_i = 1;
static constexpr int tau_i = 2;
static constexpr int pot_i = 3;
static constexpr int sx_i = 4;
static constexpr int sy_i = 5;
static constexpr int sz_i = 6;
static constexpr int zx_i = 7;
static constexpr int zy_i = 8;
static constexpr int zz_i = 9;
static constexpr int spc_i = 10;

static constexpr int INX = OCTOTIGER_GRIDDIM;
static constexpr int H_BW = 3;
static constexpr int H_NX = 2 * H_BW + INX;
static constexpr int H_DNX = H_NX * H_NX;
static constexpr int H_DNY = H_NX;
static constexpr int H_DNZ = 1;
static constexpr int H_DN[NDIM] = { H_NX * H_NX, H_NX, 1 };
static constexpr int H_N3 = H_NX * H_NX * H_NX;

#endif /* OCTOTIGER_HYDRO_DEFS_HPP_ */