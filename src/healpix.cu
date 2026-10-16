/* -----------------------------------------------------------------------------
 *
 *  Copyright (C) 1997-2010 Krzysztof M. Gorski, Eric Hivon,
 *                          Benjamin D. Wandelt, Anthony J. Banday,
 *                          Matthias Bartelmann,
 *                          Reza Ansari & Kenneth M. Ganga
 *
 *
 *  This file is part of HEALPix.
 *
 *  HEALPix is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  HEALPix is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with HEALPix; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  For more information about HEALPix see http://healpix.jpl.nasa.gov
 *
 *----------------------------------------------------------------------------- */
/* ang2pix_ring.c
 *
 */

/* Standard Includes */
#include <cosmictiger/assert.hpp>
#include <cosmictiger/math.hpp>

__managed__ int x2pix[128], y2pix[128];

void healpixInit() {
    /* =======================================================================
     * subroutine mk_xy2pix
     * =======================================================================
     * sets the array giving the number of the pixel lying in (x,y)
     * x and y are in {1,128}
     * the pixel number is in {0,128**2-1}
     *
     * if  i-1 = sum_p=0  b_p * 2^p
     * then ix = sum_p=0  b_p * 4^p
     * iy = 2*ix
     * ix + iy in {0, 128**2 -1}
     * =======================================================================
     */
    int i, k, ip, iValue, j, id;

    for (i = 0; i < 127; i++) {
        x2pix[i] = 0;
    }
    for (iValue = 1; iValue <= 128; iValue++) {
        j = iValue - 1;    //            !pixel numbers
        k = 0;             //
        ip = 1;            //
    truc:
        if (j == 0) {
            x2pix[iValue - 1] = k;
            y2pix[iValue - 1] = 2 * k;
        } else {
            id = j % 2;
            j = j / 2;
            k = ip * id + k;
            ip = ip * 4;
            goto truc;
        }
    }
}

__device__ void vec2pixNest(long const nside, double* vec, long* ipix) {
    /* =======================================================================
     * subroutine vec2pix_nest(nside, vec, ipix)
     * =======================================================================
     * gives the pixel number ipix (NESTED) corresponding to vector vec
     *
     * the computation is made to the highest resolution available (nside=8192)
     * and then degraded to that required (by integer division)
     * this doesn't cost more, and it makes sure that the treatement of round-off
     * will be consistent for every resolution
     * =======================================================================
     */

    double z, za, z0, tt, tp, tmp, phi;
    int faceNum, jp, jm;
    long ifp, ifm;
    int ix, iy, ixLow, ixHi, iyLow, iyHi, ipf, ntt;
    double const piover2 = 0.5 * M_PI, twopi = 2.0 * M_PI;
    constexpr int kNsMax = 8192;

    if (nside < 1 || nside > kNsMax) {
        ALWAYS_ASSERT(false);
    }
    z = vec[2] / sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
    phi = 0.0;
    if (vec[0] != 0.0 || vec[1] != 0.0) {
        phi = atan2(vec[1], vec[0]); /* in ]-pi, pi] */
        if (phi < 0.0) {
            phi += twopi; /* in  [0, 2pi[ */
        }
    }

    za = fabs(z);
    z0 = 2. / 3.;
    tt = phi / piover2; /* in [0,4[ */

    if (za <= z0) { /* equatorial region */

        /* (the index of edge lines increase when the longitude=phi goes up) */
        jp = (int) floor(kNsMax * (0.5 + tt - z * 0.75)); /* ascending edge line index */
        jm = (int) floor(kNsMax * (0.5 + tt + z * 0.75)); /* descending edge line index */

        /* finds the face */
        ifp = jp / kNsMax; /* in {0,4} */
        ifm = jm / kNsMax;

        if (ifp == ifm) {
            faceNum = (int) (ifp % 4) + 4; /* faces 4 to 7 */
        } else if (ifp < ifm) {
            faceNum = (int) (ifp % 4); /* (half-)faces 0 to 3 */
        } else {
            faceNum = (int) (ifm % 4) + 8; /* (half-)faces 8 to 11 */
        }

        ix = (int) (jm % kNsMax);
        iy = kNsMax - (int) (jp % kNsMax) - 1;
    } else { /* polar region, za > 2/3 */

        ntt = (int) floor(tt);
        if (ntt >= 4) {
            ntt = 3;
        }
        tp = tt - ntt;
        tmp = sqrt(3. * (1. - za)); /* in ]0,1] */

        /* (the index of edge lines increase when distance from the closest pole
         * goes up)
         */
        /* line going toward the pole as phi increases */
        jp = (int) floor(kNsMax * tp * tmp);

        /* that one goes away of the closest pole */
        jm = (int) floor(kNsMax * (1. - tp) * tmp);
        jp = (int) (jp < kNsMax - 1 ? jp : kNsMax - 1);
        jm = (int) (jm < kNsMax - 1 ? jm : kNsMax - 1);

        /* finds the face and pixel's (x,y) */
        if (z >= 0) {
            faceNum = ntt; /* in {0,3} */
            ix = kNsMax - jm - 1;
            iy = kNsMax - jp - 1;
        } else {
            faceNum = ntt + 8; /* in {8,11} */
            ix = jp;
            iy = jm;
        }
    }

    ixLow = (int) (ix % 128);
    ixHi = ix / 128;
    iyLow = (int) (iy % 128);
    iyHi = iy / 128;

    ipf = (x2pix[ixHi] + y2pix[iyHi]) * (128 * 128) + (x2pix[ixLow] + y2pix[iyLow]);
    ipf = (long) (ipf / sqr(kNsMax / nside));    /* in {0, nside**2 - 1} */
    *ipix = (long) (ipf + faceNum * sqr(nside)); /* in {0, 12*nside**2 - 1} */
}
