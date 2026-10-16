//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Original Fortran source: http://cococubed.asu.edu/research_pages/sedov.shtml

/* sedov3.f -- translated by f2c (version 20160102).
 You must link the resulting object file with libf2c:
 on Microsoft Windows system, link with libf2c.lib;
 on Linux or Unix systems, link with .../path/to/libf2c.a -lm
 or, if you install libf2c.a in a standard place, with -lf2c -lm
 -- in that order, at the end of the command line, as ina
 cc *.o -lf2c -lm
 Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

 http://www.netlib.org/f2c/libf2c.zip
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <hpx/synchronization/spinlock.hpp>
#include <quadmath.h>

using SedReal = __float128;

SedReal sedovExp(SedReal a) {
    return expq(a);
}

SedReal powDd(SedReal* a, SedReal* b) {
    return powq(*a, *b);
}

SedReal dSign(SedReal* a, SedReal* b) {
    return copysignq(*a, *b);
}

/* Subroutine */ int sed1d(SedReal* time, int* nstep, SedReal* xpos, SedReal* eblast,
    SedReal* omegaInArg, SedReal* xgeomInArg, SedReal* rho0, SedReal* vel0, SedReal* ener0,
    SedReal* pres0, SedReal* cs0, SedReal* gam0, SedReal* den, SedReal* ener, SedReal* pres,
    SedReal* vel, SedReal* cs);
/* Common Block Declarations */

struct {
    SedReal gamma, gamm1, gamp1, gpogm, xgeom, xg2, rwant, r2, a0, a1, a2, a3, a4, a5, aVal, bVal,
        cVal, dVal, eVal, omega, vv, xlamWant, vwant, rvv;
    bool lsingular, lstandard, lvacuum, lomega2, lomega3;
} slap_;

#define slap_1 slap_

struct {
    SedReal gamInt;
} cmidp_;

#define cmidp_1 cmidp_

/* Table of constant values */

using DFp = SedReal (*)(SedReal*);
using SFp = int (*)(DFp, SedReal*, SedReal*, SedReal*, int*);    // Subroutine
using UFp = int (*)();                                           // Unknown procedure type

static SedReal cB52 = 2.;
static SedReal cB53 = 1e-10;
static SedReal cB79 = 0.f;
static SedReal cB80 = 1e-30;
static int c3 = 3;
static int c5 = 5;
static SedReal efun01(SedReal* v);
static SedReal efun02(SedReal* v);
static SedReal sedVFind(SedReal* v);
static SedReal sedRFind(SedReal* rArg);
/* Subroutine */ int sedovFuncs(SedReal* v, SedReal* lFunArg, SedReal* dlamdv, SedReal* fFunArg,
    SedReal* gFunArg, SedReal* hFunArg);
/* Subroutine */ static int midpnt(DFp func, SedReal* a, SedReal* b, SedReal* s, int* n);
/* Subroutine */ static int midpowl(DFp funk, SedReal* aa, SedReal* bb, SedReal* s, int* n);
/* Subroutine */ static int midpowl2(DFp funk, SedReal* aa, SedReal* bb, SedReal* s, int* n);
/* Subroutine */ static int qromo(
    DFp func, SedReal* a, SedReal* b, SedReal* eps, SedReal* ss, SFp choose);
/* Subroutine */ static int polint(
    SedReal* xa, SedReal* ya, int* n, SedReal* x, SedReal* y, SedReal* dy);
static SedReal zeroin(SedReal* ax, SedReal* bx, DFp f, SedReal* tol);

int powIi(int* a, int* b) {
    return std::pow(*a, *b);
}

/* Subroutine */ int sed1d(SedReal* time, int* nstep, SedReal* xpos, SedReal* eblast,
    SedReal* omegaInArg, SedReal* xgeomInArg, SedReal* rho0, SedReal* vel0, SedReal* ener0,
    SedReal* pres0, SedReal* cs0, SedReal* gam0, SedReal* den, SedReal* ener, SedReal* pres,
    SedReal* vel, SedReal* cs) {
    /* System generated locals */
    int i1;
    SedReal d1, d2, d3;

    /* Builtin functions */

    /* Local variables */
    static int iArg;
    static SedReal p2, v0, u2, v2, us;
    static SedReal vat, rho1, rho2;
    static SedReal vmin, eval1, eval2, alpha, fFunArg;
    static SedReal gFunArg, hFunArg, lFunArg;
    static SedReal vstar, denom2, denom3, dlamdv;
    SedReal zeroin(SedReal * ax, SedReal * bx, DFp f, SedReal * tol);

    /* ..this routine produces 1d solutions for a sedov blast wave propagating */
    /* ..through a density gradient rho = rho**(-omega) */
    /* ..in planar, cylindrical or spherical geometry */
    /* ..for the standard, singular and vaccum cases. */
    /* ..standard case: a nonzero solution extends from the shock to the origin, */
    /* ..               where the pressure is finite. */
    /* ..singular case: a nonzero solution extends from the shock to the origin, */
    /* ..               where the pressure vanishes. */
    /* ..vacuum case  : a nonzero solution extends from the shock to a boundary point, */
    /* ..               where the density vanishes making the pressure meaningless. */
    /* ..input: */
    /* ..time     = temporal point where solution is desired seconds */
    /* ..xpos(i)  = spatial points where solution is desired cm */
    /* ..eblast   = energy of blast erg */
    /* ..rho0     = ambient density g/cm**3    rho = rho0 * r**(-omega_in) */
    /* ..omegain  = density power law _exponent rho = rho0 * r**(-omega_in) */
    /* ..vel0     = ambient material speed cm/s */
    /* ..pres0    = ambient pressure erg/cm**3 */
    /* ..cs0      = ambient sound speed cm/s */
    /* ..gam0   = gamma law equation of state */
    /* ..xgeom_in = geometry factor, 3=spherical, 2=cylindircal, 1=planar */
    /* ..for efficiency reasons (doing the energy integrals only once), */
    /* ..this routine returns the solution for an array of spatial points */
    /* ..at the desired time point. */
    /* ..output: */
    /* ..den(i)  = density  g/cm**3 */
    /* ..ener(i) = specific internal energy erg/g */
    /* ..pres(i) = presssure erg/cm**3 */
    /* ..vel(i)  = velocity cm/s */
    /* ..cs(i)   = sound speed cm/s */
    /* ..this routine is based upon two papers: */
    /* .."evaluation of the sedov-von neumann-taylor blast wave solution" */
    /* ..jim kamm, la-ur-00-6055 */
    /* .."the sedov self-similiar point blast solutions in nonuniform media" */
    /* ..david book, shock waves, 4, 1, 1994 */
    /* ..although the ordinary differential equations are analytic, */
    /* ..the sedov _expressions appear to become singular for various */
    /* ..combinations of parameters and at the lower limits of the integration */
    /* ..range. all these singularies are removable and done so by this routine. */
    /* ..these routines are written in sed_real*8 precision because the */
    /* ..sed_real*8 implementations simply run out of precision "near" the origin */
    /* ..in the standard case or the transition region in the vacuum case. */
    /* ..declare the pass */
    /* ..local variables */
    /* ..eps controls the integration accuracy, don't get too greedy or the number */
    /* ..of function evaluations required kills. */
    /* ..eps2 controls the root find accuracy */
    /* ..osmall controls the size of transition regions */
    /* ..common block communication */
    /* ..common block communication with the integration stepper */
    /* ..popular formats */
    /* ..initialize the solution */
    /* Parameter adjustments */
    --cs;
    --vel;
    --pres;
    --ener;
    --den;
    --xpos;

    /* Function Body */
    /* L87: */
    /* L88: */
    i1 = *nstep;
    for (iArg = 1; iArg <= i1; ++iArg) {
        den[iArg] = 0.f;
        vel[iArg] = 0.f;
        pres[iArg] = 0.f;
        ener[iArg] = 0.f;
        cs[iArg] = 0.f;
    }
    /* ..return on unphysical cases */
    /* ..infinite mass */
    if (*omegaInArg >= *xgeomInArg) {
        return 0;
    }
    /* ..transfer the pass to common block and create some frequent combinations */
    slap_1.gamma = *gam0;
    slap_1.gamm1 = slap_1.gamma - 1.f;
    slap_1.gamp1 = slap_1.gamma + 1.f;
    slap_1.gpogm = slap_1.gamp1 / slap_1.gamm1;
    slap_1.xgeom = *xgeomInArg;
    slap_1.omega = *omegaInArg;
    slap_1.xg2 = slap_1.xgeom + 2.f - slap_1.omega;
    denom2 = slap_1.gamm1 * 2.f + slap_1.xgeom - slap_1.gamma * slap_1.omega;
    denom3 = slap_1.xgeom * (2.f - slap_1.gamma) - slap_1.omega;
    /* ..post shock location v2 and location of singular point vstar */
    /* ..kamm equation 18 and 19 */
    v2 = 4.f / (slap_1.xg2 * slap_1.gamp1);
    vstar = 2.f / (slap_1.gamm1 * slap_1.xgeom + 2.f);
    /* ..set two bools that determines the type of solution */
    slap_1.lstandard = false;
    slap_1.lsingular = false;
    slap_1.lvacuum = false;
    if ((d1 = v2 - vstar, fabs(static_cast<double>(d1))) <= 1e-4) {
        slap_1.lsingular = true;
    } else if (v2 < vstar - 1e-4) {
        slap_1.lstandard = true;
    } else if (v2 > vstar + 1e-4) {
        slap_1.lvacuum = true;
    }
    /* ..two apparent singularies, book's notation for omega2 and omega3 */
    slap_1.lomega2 = false;
    slap_1.lomega3 = false;
    if (fabs(static_cast<double>(denom2)) <= 1e-4) {
        slap_1.lomega2 = true;
        denom2 = 1e-8f;
    } else if (fabs(static_cast<double>(denom3)) <= 1e-4) {
        slap_1.lomega3 = true;
        denom3 = 1e-8f;
    }
    /* ..various _exponents, kamm equations 42-47 */
    /* ..in terms of book's notation: */
    /* ..a0=beta6 a1=beta1  a2=-beta2 a3=beta3 a4=beta4 a5=-beta5 */
    slap_1.a0 = 2.f / slap_1.xg2;
    slap_1.a2 = -slap_1.gamm1 / denom2;
    slap_1.a1 = slap_1.xg2 * slap_1.gamma / (slap_1.xgeom * slap_1.gamm1 + 2.f) *
        ((slap_1.xgeom * (2.f - slap_1.gamma) - slap_1.omega) * 2.f /
                (slap_1.gamma * slap_1.xg2 * slap_1.xg2) -
            slap_1.a2);
    slap_1.a3 = (slap_1.xgeom - slap_1.omega) / denom2;
    slap_1.a4 = slap_1.xg2 * (slap_1.xgeom - slap_1.omega) * slap_1.a1 / denom3;
    slap_1.a5 = (slap_1.omega * slap_1.gamp1 - slap_1.xgeom * 2.f) / denom3;
    /* ..frequent combinations, kamm equations 33-37 */
    slap_1.aVal = slap_1.xg2 * .25f * slap_1.gamp1;
    slap_1.bVal = slap_1.gpogm;
    slap_1.cVal = slap_1.xg2 * .5f * slap_1.gamma;
    slap_1.dVal = slap_1.xg2 * slap_1.gamp1 /
        (slap_1.xg2 * slap_1.gamp1 - (slap_1.xgeom * slap_1.gamm1 + 2.f) * 2.f);
    slap_1.eVal = (slap_1.xgeom * slap_1.gamm1 + 2.f) * .5f;
    /* ..evaluate the energy integrals */
    /* ..the singular case can be done by hand; save some cpu cycles */
    /* ..kamm equations 80, 81, and 85 */
    if (slap_1.lsingular) {
        /* Computing 2nd power */
        d1 = slap_1.gamm1 * slap_1.xgeom + 2.f;
        eval2 = slap_1.gamp1 / (slap_1.xgeom * (d1 * d1));
        eval1 = 2.f / slap_1.gamm1 * eval2;
        /* Computing 2nd power */
        d1 = slap_1.gamm1 * slap_1.xgeom + 2.f;
        alpha = slap_1.gpogm * powDd(&cB52, &slap_1.xgeom) / (slap_1.xgeom * (d1 * d1));
        if (static_cast<int>(slap_1.xgeom) != 1) {
            alpha *= 3.1415926535897932384626433832795029;
        }
        /* ..for the standard or vacuum cases */
        /* ..v0 = post-shock origin v0 and vv = vacuum boundary vv */
        /* ..set the radius corespondin to vv to zero for now */
        /* ..kamm equations 18, and 28. */
    } else {
        v0 = 2.f / (slap_1.xg2 * slap_1.gamma);
        slap_1.vv = 2.f / slap_1.xg2;
        slap_1.rvv = 0.;
        if (slap_1.lstandard) {
            vmin = v0;
        }
        if (slap_1.lvacuum) {
            vmin = slap_1.vv;
        }
        /* ..the first energy integral */
        /* ..in the standard case the term (c_val*v - 1) might be singular at v=vmin */
        if (slap_1.lstandard) {
            cmidp_1.gamInt = slap_1.a3 - slap_1.a2 * slap_1.xg2 - 1.f;
            if (cmidp_1.gamInt >= 0.) {
                qromo(
                    static_cast<DFp>(efun01), &vmin, &v2, &cB53, &eval1, static_cast<SFp>(midpnt));
            } else {
                cmidp_1.gamInt = fabs(static_cast<double>(cmidp_1.gamInt));
                qromo(
                    static_cast<DFp>(efun01), &vmin, &v2, &cB53, &eval1, static_cast<SFp>(midpowl));
            }
            /* ..in the vacuum case the term (1 - c_val/gamma*v) might be singular at v=vmin */
        } else if (slap_1.lvacuum) {
            cmidp_1.gamInt = slap_1.a5;
            if (cmidp_1.gamInt >= 0.) {
                qromo(
                    static_cast<DFp>(efun01), &vmin, &v2, &cB53, &eval1, static_cast<SFp>(midpnt));
            } else {
                cmidp_1.gamInt = fabs(static_cast<double>(cmidp_1.gamInt));
                qromo(static_cast<DFp>(efun01), &vmin, &v2, &cB53, &eval1,
                    static_cast<SFp>(midpowl2));
            }
        }
        /* ..the second energy integral */
        /* ..in the standard case the term (c_val*v - 1) might be singular at v=vmin */
        if (slap_1.lstandard) {
            cmidp_1.gamInt = slap_1.a3 - slap_1.a2 * slap_1.xg2 - 2.f;
            if (cmidp_1.gamInt >= 0.) {
                qromo(
                    static_cast<DFp>(efun02), &vmin, &v2, &cB53, &eval2, static_cast<SFp>(midpnt));
            } else {
                cmidp_1.gamInt = fabs(static_cast<double>(cmidp_1.gamInt));
                qromo(
                    static_cast<DFp>(efun02), &vmin, &v2, &cB53, &eval2, static_cast<SFp>(midpowl));
            }
            /* ..in the vacuum case the term (1 - c_val/gamma*v) might be singular at v=vmin */
        } else if (slap_1.lvacuum) {
            cmidp_1.gamInt = slap_1.a5;
            if (cmidp_1.gamInt >= 0.) {
                qromo(
                    static_cast<DFp>(efun02), &vmin, &v2, &cB53, &eval2, static_cast<SFp>(midpnt));
            } else {
                cmidp_1.gamInt = fabs(static_cast<double>(cmidp_1.gamInt));
                qromo(static_cast<DFp>(efun02), &vmin, &v2, &cB53, &eval2,
                    static_cast<SFp>(midpowl2));
            }
        }
        /* ..kamm equations 57 and 58 for alpha, in a slightly different form. */
        if (static_cast<int>(slap_1.xgeom) == 1) {
            alpha = eval1 * .5f + eval2 / slap_1.gamm1;
        } else {
            alpha = (slap_1.xgeom - 1.f) * 3.1415926535897932384626433832795029 *
                (eval1 + eval2 * 2.f / slap_1.gamm1);
        }
    }
    /* ..write what we have for the energy integrals */
    if (true) {
        //		s_wsfe(&io___42);
        //		do_fio(&c__1, "xgeom =", (ftnlen) 7);
        //		do_fio(&c__1, (char *) &slap_1.xgeom, (ftnlen) sizeof(sed_real));
        //		do_fio(&c__1, "eblast=", (ftnlen) 7);
        //		do_fio(&c__1, (char *) &(*eblast), (ftnlen) sizeof(sed_real));
        //		do_fio(&c__1, "omega =", (ftnlen) 7);
        //		do_fio(&c__1, (char *) &slap_1.omega, (ftnlen) sizeof(sed_real));
        //		do_fio(&c__1, "alpha =", (ftnlen) 7);
        //		do_fio(&c__1, (char *) &alpha, (ftnlen) sizeof(sed_real));
        //		do_fio(&c__1, "j1    =", (ftnlen) 7);
        //		do_fio(&c__1, (char *) &eval1, (ftnlen) sizeof(sed_real));
        //		do_fio(&c__1, "j2    =", (ftnlen) 7);
        //		do_fio(&c__1, (char *) &eval2, (ftnlen) sizeof(sed_real));
        //		e_wsfe();
    }
    /*      write(6,87) omega,alpha */
    /* ..immediate post-shock values */
    /* ..kamm page 14 or equations 14, 16, 5, 13 */
    /* ..r2 = shock position, u2 = shock speed, rho1 = pre-shock density, */
    /* ..u2 = post-shock material speed, rho2 = post-shock density, */
    /* ..p2 = post-shock pressure, e2 = post-shoock specific internal energy, */
    /* ..and cs2 = post-shock sound speed */
    d1 = *eblast / (alpha * *rho0);
    d2 = 1.f / slap_1.xg2;
    d3 = 2.f / slap_1.xg2;
    slap_1.r2 = powDd(&d1, &d2) * powDd(time, &d3);
    us = 2.f / slap_1.xg2 * slap_1.r2 / *time;
    d1 = -slap_1.omega;
    rho1 = *rho0 * powDd(&slap_1.r2, &d1);
    u2 = us * 2.f / slap_1.gamp1;
    rho2 = slap_1.gpogm * rho1;
    /* Computing 2nd power */
    d1 = us;
    p2 = rho1 * 2.f * (d1 * d1) / slap_1.gamp1;
    //	e2 = p2 / (slap_1.gamm1 * rho2);
    //	cs2 = sqrt(slap_1.gamma * p2 / rho2);
    /* ..find the radius corresponding to vv */
    if (slap_1.lvacuum) {
        slap_1.vwant = slap_1.vv;
        slap_1.rvv = zeroin(&cB79, &slap_1.r2, static_cast<DFp>(sedRFind), &cB80);
    }
    //	if (slap_1.lstandard) {
    //		s_wsfe(&io___50);
    //		do_fio(&c__1, "r2    =", (ftnlen) 7);
    //		do_fio(&c__1, (char *) &slap_1.r2, (ftnlen) sizeof(sed_real));
    //		do_fio(&c__1, "rho2  =", (ftnlen) 7);
    //		do_fio(&c__1, (char *) &rho2, (ftnlen) sizeof(sed_real));
    //		do_fio(&c__1, "u2    =", (ftnlen) 7);
    //		do_fio(&c__1, (char *) &u2, (ftnlen) sizeof(sed_real));
    //		do_fio(&c__1, "e2    =", (ftnlen) 7);
    //		do_fio(&c__1, (char *) &e2, (ftnlen) sizeof(sed_real));
    //		do_fio(&c__1, "p2    =", (ftnlen) 7);
    //		do_fio(&c__1, (char *) &p2, (ftnlen) sizeof(sed_real));
    //		do_fio(&c__1, "cs2   =", (ftnlen) 7);
    //		do_fio(&c__1, (char *) &cs2, (ftnlen) sizeof(sed_real));
    //		e_wsfe();
    //	}
    //	if (slap_1.lvacuum) {
    //		s_wsfe(&io___51);
    //		do_fio(&c__1, "rv    =", (ftnlen) 7);
    //		do_fio(&c__1, (char *) &slap_1.rvv, (ftnlen) sizeof(sed_real));
    //		do_fio(&c__1, "r2    =", (ftnlen) 7);
    //		do_fio(&c__1, (char *) &slap_1.r2, (ftnlen) sizeof(sed_real));
    //		do_fio(&c__1, "rho2  =", (ftnlen) 7);
    //		do_fio(&c__1, (char *) &rho2, (ftnlen) sizeof(sed_real));
    //		do_fio(&c__1, "u2    =", (ftnlen) 7);
    //		do_fio(&c__1, (char *) &u2, (ftnlen) sizeof(sed_real));
    //		do_fio(&c__1, "e2    =", (ftnlen) 7);
    //		do_fio(&c__1, (char *) &e2, (ftnlen) sizeof(sed_real));
    //		do_fio(&c__1, "p2    =", (ftnlen) 7);
    //		do_fio(&c__1, (char *) &p2, (ftnlen) sizeof(sed_real));
    //		do_fio(&c__1, "cs2   =", (ftnlen) 7);
    //		do_fio(&c__1, (char *) &cs2, (ftnlen) sizeof(sed_real));
    //		e_wsfe();
    //	}
    /* ..now start the loop over spatial positions */
    i1 = *nstep;
    for (iArg = 1; iArg <= i1; ++iArg) {
        slap_1.rwant = xpos[iArg];
        /* ..if we are upstream from the shock front */
        if (slap_1.rwant > slap_1.r2) {
            d1 = -slap_1.omega;
            den[iArg] = *rho0 * powDd(&slap_1.rwant, &d1);
            vel[iArg] = *vel0;
            pres[iArg] = *pres0;
            ener[iArg] = *ener0;
            cs[iArg] = *cs0;
            /* ..if we are between the origin and the shock front */
            /* ..find the correct similarity value for this radius in the standard or vacuum cases
             */
        } else {
            if (slap_1.lstandard) {
                d1 = v0 * .9f;
                vat = zeroin(&d1, &v2, static_cast<DFp>(sedVFind), &cB80);
            } else if (slap_1.lvacuum) {
                d1 = slap_1.vv * 1.2f;
                vat = zeroin(&v2, &d1, static_cast<DFp>(sedVFind), &cB80);
            }
            /* ..the physical solution */
            sedovFuncs(&vat, &lFunArg, &dlamdv, &fFunArg, &gFunArg, &hFunArg);
            den[iArg] = rho2 * gFunArg;
            vel[iArg] = u2 * fFunArg;
            pres[iArg] = p2 * hFunArg;
            ener[iArg] = 0.f;
            cs[iArg] = 0.f;
            if (den[iArg] != 0.f) {
                ener[iArg] = pres[iArg] / (slap_1.gamm1 * den[iArg]);
                cs[iArg] = sqrt(static_cast<double>(slap_1.gamma * pres[iArg] / den[iArg]));
            }
        }
        /* ..end of loop over positions */
    }
    return 0;
} /* sed_1d__ */

SedReal efun01(SedReal* v) {
    /* System generated locals */
    SedReal retVal, d1, d2;
    retVal = 0;

    /* Builtin functions */
    SedReal powDd(SedReal*, SedReal*);

    /* Local variables */
    static SedReal fFunArg, gFunArg, hFunArg, lFunArg, dlamdv;

    /* ..evaluates the first energy integrand, kamm equations 67 and 10. */
    /* ..the (c_val*v - 1) term might be singular at v=vmin in the standard case. */
    /* ..the (1 - c_val/gamma * v) term might be singular at v=vmin in the vacuum case. */
    /* ..due care should be taken for these removable singularities by the integrator. */
    /* ..declare the pass */
    /* ..common block communication */
    /* ..local variables */
    /* ..go */
    sedovFuncs(v, &lFunArg, &dlamdv, &fFunArg, &gFunArg, &hFunArg);
    d1 = slap_1.xgeom + 1.f;
    /* Computing 2nd power */
    d2 = *v;
    retVal = dlamdv * powDd(&lFunArg, &d1) * slap_1.gpogm * gFunArg * (d2 * d2);
    return retVal;
} /* efun01_ */

SedReal efun02(SedReal* v) {
    /* System generated locals */
    SedReal retVal, d1;
    retVal = 0;

    /* Builtin functions */
    SedReal powDd(SedReal*, SedReal*);

    /* Local variables */
    static SedReal zArg;
    static SedReal fFunArg, gFunArg, hFunArg, lFunArg, dlamdv;

    /* ..evaluates the second energy integrand, kamm equations 68 and 11. */
    /* ..the (c_val*v - 1) term might be singular at v=vmin in the standard case. */
    /* ..the (1 - c_val/gamma * v) term might be singular at v=vmin in the vacuum case. */
    /* ..due care should be taken for these removable singularities by the integrator. */
    /* ..declare the pass */
    /* ..common block communication */
    /* ..local variables */
    /* ..go */
    sedovFuncs(v, &lFunArg, &dlamdv, &fFunArg, &gFunArg, &hFunArg);
    /* Computing 2nd power */
    d1 = slap_1.xgeom + 2.f - slap_1.omega;
    zArg = 8.f / (d1 * d1 * slap_1.gamp1);
    d1 = slap_1.xgeom - 1.f;
    retVal = dlamdv * powDd(&lFunArg, &d1) * hFunArg * zArg;
    return retVal;
} /* efun02_ */

SedReal sedVFind(SedReal* v) {
    /* System generated locals */
    SedReal retVal;
    retVal = 0;

    /* Local variables */
    static SedReal fFunArg, gFunArg, hFunArg, lFunArg, dlamdv;

    /* ..given corresponding physical distances, find the similarity variable v */
    /* ..kamm equation 38 as a root find */
    /* ..declare the pass */
    /* ..common block communication */
    /* ..local variables */
    sedovFuncs(v, &lFunArg, &dlamdv, &fFunArg, &gFunArg, &hFunArg);
    retVal = slap_1.r2 * lFunArg - slap_1.rwant;
    return retVal;
} /* sed_v_find__ */

SedReal sedRFind(SedReal* rArg) {
    /* System generated locals */
    SedReal retVal;
    retVal = 0;

    /* Local variables */
    static SedReal fFunArg, gFunArg, hFunArg, lFunArg, dlamdv;

    /* ..given the similarity variable v, find the corresponding physical distance */
    /* ..kamm equation 38 as a root find */
    /* ..declare the pass */
    /* ..common block communication */
    /* ..local variables */
    sedovFuncs(&slap_1.vwant, &lFunArg, &dlamdv, &fFunArg, &gFunArg, &hFunArg);
    retVal = slap_1.r2 * lFunArg - *rArg;
    return retVal;
} /* sed_r_find__ */

/* Subroutine */ int sedovFuncs(SedReal* v, SedReal* lFunArg, SedReal* dlamdv, SedReal* fFunArg,
    SedReal* gFunArg, SedReal* hFunArg) {
    /* System generated locals */
    SedReal d1, d2, d3;

    /* Builtin functions */
    //	sed_real pow_dd(sed_real *, sed_real *), _exp(sed_real);

    /* Local variables */
    static SedReal y, zArg, c2, c6, x1, x2, x3, x4, pp1, pp2, pp3, pp4, cbag, ebag, beta0, dx1dv,
        dx2dv, dx3dv, dx4dv, dpp2dv;

    /* ..given the similarity variable v, returns functions */
    /* ..lambda, f, g, and h and the derivative of lambda with v dlamdv */
    /* ..although the ordinary differential equations are analytic, */
    /* ..the sedov _expressions appear to become singular for various */
    /* ..combinations of parameters and at the lower limits of the integration */
    /* ..range. all these singularies are removable and done so by this routine. */
    /* ..declare the pass */
    /* ..common block communication */
    /* ..local variables */
    /* ..frequent combinations and their derivative with v */
    /* ..kamm equation 29-32, x4 a bit different to save a divide */
    /* ..x1 is book's F */
    x1 = slap_1.aVal * *v;
    dx1dv = slap_1.aVal;
    /* Computing MAX */
    d1 = 1e-30, d2 = slap_1.cVal * *v - 1.f;
    cbag = fmax(static_cast<double>(d1), static_cast<double>(d2));
    x2 = slap_1.bVal * cbag;
    dx2dv = slap_1.bVal * slap_1.cVal;
    ebag = 1.f - slap_1.eVal * *v;
    x3 = slap_1.dVal * ebag;
    dx3dv = -slap_1.dVal * slap_1.eVal;
    x4 = slap_1.bVal * (1.f - slap_1.xg2 * .5f * *v);
    dx4dv = -slap_1.bVal * .5f * slap_1.xg2;
    /* ..transition region between standard and vacuum cases */
    /* ..kamm page 15 or equations 88-92 */
    /* ..lambda = l_fun is book's zeta */
    /* ..f_fun is books V, g_fun is book's D, h_fun is book's P */
    if (slap_1.lsingular) {
        *lFunArg = slap_1.rwant / slap_1.r2;
        *dlamdv = 0.f;
        *fFunArg = *lFunArg;
        d1 = slap_1.xgeom - 2.f;
        *gFunArg = powDd(lFunArg, &d1);
        *hFunArg = powDd(lFunArg, &slap_1.xgeom);
        /* ..for the vacuum case in the hole */
    } else if (slap_1.lvacuum && slap_1.rwant < slap_1.rvv) {
        *lFunArg = 0.f;
        *dlamdv = 0.f;
        *fFunArg = 0.f;
        *gFunArg = 0.f;
        *hFunArg = 0.f;
        /* ..omega = omega2 = (2*(gamma -1) + xgeom)/gamma case, denom2 = 0 */
        /* ..book _expressions 20-22 */
    } else if (slap_1.lomega2) {
        beta0 = 1.f / (slap_1.eVal * 2.f);
        pp1 = slap_1.gamm1 * beta0;
        c6 = slap_1.gamp1 * .5f;
        c2 = c6 / slap_1.gamma;
        y = 1.f / (x1 - c2);
        zArg = (1.f - x1) * y;
        pp2 = slap_1.gamp1 * beta0 * zArg;
        dpp2dv = -slap_1.gamp1 * beta0 * dx1dv * y * (zArg + 1.f);
        pp3 = (4.f - slap_1.xgeom - slap_1.gamma * 2.f) * beta0;
        pp4 = -slap_1.xgeom * slap_1.gamma * beta0;
        d1 = -slap_1.a0;
        *lFunArg = powDd(&x1, &d1) * powDd(&x2, &pp1) * sedovExp(pp2);
        *dlamdv = (-slap_1.a0 * dx1dv / x1 + pp1 * dx2dv / x2 + dpp2dv) * *lFunArg;
        *fFunArg = x1 * *lFunArg;
        d1 = slap_1.a0 * slap_1.omega;
        *gFunArg =
            powDd(&x1, &d1) * powDd(&x2, &pp3) * powDd(&x4, &slap_1.a5) * sedovExp(pp2 * -2.f);
        d1 = slap_1.a0 * slap_1.xgeom;
        d2 = slap_1.a5 + 1.f;
        *hFunArg = powDd(&x1, &d1) * powDd(&x2, &pp4) * powDd(&x4, &d2);
        /* ..omega = omega3 = xgeom*(2 - gamma) case, denom3 = 0 */
        /* ..book _expressions 23-25 */
    } else if (slap_1.lomega3) {
        beta0 = 1.f / (slap_1.eVal * 2.f);
        pp1 = slap_1.a3 + slap_1.omega * slap_1.a2;
        pp2 = 1.f - beta0 * 4.f;
        c6 = slap_1.gamp1 * .5f;
        pp3 = -slap_1.xgeom * slap_1.gamma * slap_1.gamp1 * beta0 * (1.f - x1) / (c6 - x1);
        pp4 = (slap_1.xgeom * slap_1.gamm1 - slap_1.gamma) * 2.f * beta0;
        d1 = -slap_1.a0;
        d2 = -slap_1.a2;
        d3 = -slap_1.a1;
        *lFunArg = powDd(&x1, &d1) * powDd(&x2, &d2) * powDd(&x4, &d3);
        *dlamdv =
            -(slap_1.a0 * dx1dv / x1 + slap_1.a2 * dx2dv / x2 + slap_1.a1 * dx4dv / x4) * *lFunArg;
        *fFunArg = x1 * *lFunArg;
        d1 = slap_1.a0 * slap_1.omega;
        *gFunArg = powDd(&x1, &d1) * powDd(&x2, &pp1) * powDd(&x4, &pp2) * sedovExp(pp3);
        d1 = slap_1.a0 * slap_1.xgeom;
        *hFunArg = powDd(&x1, &d1) * powDd(&x4, &pp4) * sedovExp(pp3);
        /* ..for the standard or vacuum case not in the hole */
        /* ..kamm equations 38-41 */
    } else {
        d1 = -slap_1.a0;
        d2 = -slap_1.a2;
        d3 = -slap_1.a1;
        *lFunArg = powDd(&x1, &d1) * powDd(&x2, &d2) * powDd(&x3, &d3);
        *dlamdv =
            -(slap_1.a0 * dx1dv / x1 + slap_1.a2 * dx2dv / x2 + slap_1.a1 * dx3dv / x3) * *lFunArg;
        *fFunArg = x1 * *lFunArg;
        d1 = slap_1.a0 * slap_1.omega;
        d2 = slap_1.a3 + slap_1.a2 * slap_1.omega;
        d3 = slap_1.a4 + slap_1.a1 * slap_1.omega;
        *gFunArg = powDd(&x1, &d1) * powDd(&x2, &d2) * powDd(&x3, &d3) * powDd(&x4, &slap_1.a5);
        d1 = slap_1.a0 * slap_1.xgeom;
        d2 = slap_1.a4 + slap_1.a1 * (slap_1.omega - 2.f);
        d3 = slap_1.a5 + 1.f;
        *hFunArg = powDd(&x1, &d1) * powDd(&x3, &d2) * powDd(&x4, &d3);
    }
    return 0;
} /* sedov_funcs__ */

/* Subroutine */ int midpnt(DFp func, SedReal* a, SedReal* b, SedReal* s, int* n) {
    /* System generated locals */
    int i1;
    SedReal d1;

    /* Builtin functions */
    int powIi(int*, int*);

    /* Local variables */
    static int j;
    static SedReal x;
    static int it;
    static SedReal del, tnm, sum, ddel;

    /* ..this routine computes the n'th stage of refinement of an extended midpoint */
    /* ..rule. func is input as the name of the function to be integrated between */
    /* ..limits a and b. when called with n=1, the routine returns as s the crudest */
    /* ..estimate of the integralof func from a to b. subsequent calls with n=2,3... */
    /* ..improve the accuracy of s by adding 2/3*3**(n-1) addtional interior points. */
    /* ..declare */
    if (*n == 1) {
        d1 = (*a + *b) * .5f;
        *s = (*b - *a) * (*func)(&d1);
    } else {
        i1 = *n - 2;
        it = powIi(&c3, &i1);
        tnm = static_cast<SedReal>(it);
        del = (*b - *a) / (tnm * 3.f);
        ddel = del + del;
        x = *a + del * .5f;
        sum = 0.f;
        i1 = it;
        for (j = 1; j <= i1; ++j) {
            sum += (*func)(&x);
            x += ddel;
            sum += (*func)(&x);
            x += del;
        }
        *s = (*s + (*b - *a) * sum / tnm) / 3.f;
    }
    return 0;
} /* midpnt_ */

/* Subroutine */ int midpowl(DFp funk, SedReal* aa, SedReal* bb, SedReal* s, int* n) {
    /* System generated locals */
    int i1;
    SedReal d1, d2, d3, d4;

    /* Builtin functions */
    SedReal powDd(SedReal*, SedReal*);
    int powIi(int*, int*);

    /* Local variables */
    static SedReal a, b;
    static int j;
    static SedReal x;
    static int it;
    static SedReal del, tnm, sum, ddel;

    /* ..this routine is an exact replacement for midpnt, except that it allows for */
    /* ..an integrable power-law singularity of the form (x - a)**(-gam_int) */
    /* ..at the lower limit aa for 0 < gam_int < 1. */
    /* ..declare */
    /* ..common block communication */
    /* ..a little conversion, recipe equation 4.4.3 */
    d1 = *bb - *aa;
    d2 = 1.f - cmidp_1.gamInt;
    b = powDd(&d1, &d2);
    a = 0.f;
    /* ..now exactly as midpnt */
    if (*n == 1) {
        d1 = (a + b) * .5f;
        d2 = cmidp_1.gamInt / (1.f - cmidp_1.gamInt);
        d4 = 1.f / (1.f - cmidp_1.gamInt);
        d3 = powDd(&d1, &d4) + *aa;
        *s = (b - a) * (1.f / (1.f - cmidp_1.gamInt) * powDd(&d1, &d2) * (*funk)(&d3));
    } else {
        i1 = *n - 2;
        it = powIi(&c3, &i1);
        tnm = static_cast<SedReal>(it);
        del = (b - a) / (tnm * 3.f);
        ddel = del + del;
        x = a + del * .5f;
        sum = 0.f;
        i1 = it;
        for (j = 1; j <= i1; ++j) {
            d1 = cmidp_1.gamInt / (1.f - cmidp_1.gamInt);
            d3 = 1.f / (1.f - cmidp_1.gamInt);
            d2 = powDd(&x, &d3) + *aa;
            sum += 1.f / (1.f - cmidp_1.gamInt) * powDd(&x, &d1) * (*funk)(&d2);
            x += ddel;
            d1 = cmidp_1.gamInt / (1.f - cmidp_1.gamInt);
            d3 = 1.f / (1.f - cmidp_1.gamInt);
            d2 = powDd(&x, &d3) + *aa;
            sum += 1.f / (1.f - cmidp_1.gamInt) * powDd(&x, &d1) * (*funk)(&d2);
            x += del;
        }
        *s = (*s + (b - a) * sum / tnm) / 3.f;
    }
    return 0;
} /* midpowl_ */

/* Subroutine */ int midpowl2(DFp funk, SedReal* aa, SedReal* bb, SedReal* s, int* n) {
    /* System generated locals */
    int i1;
    SedReal d1, d2, d3, d4;

    /* Builtin functions */

    /* Local variables */
    static SedReal a, b;
    static int j;
    static SedReal x;
    static int it;
    static SedReal del, tnm, sum, ddel;

    /* ..this routine is an exact replacement for midpnt, except that it allows for */
    /* ..an integrable power-law singularity of the form (a - x)**(-gam_int) */
    /* ..at the lower limit aa for 0 < gam_int < 1. */
    /* ..declare */
    /* ..common block communication */
    /* ..a little conversion, modulo recipe equation 4.4.3 */
    d1 = *aa - *bb;
    d2 = 1.f - cmidp_1.gamInt;
    b = powDd(&d1, &d2);
    a = 0.f;
    /* ..now exactly as midpnt */
    if (*n == 1) {
        d1 = (a + b) * .5f;
        d2 = cmidp_1.gamInt / (1.f - cmidp_1.gamInt);
        d4 = 1.f / (1.f - cmidp_1.gamInt);
        d3 = *aa - powDd(&d1, &d4);
        *s = (b - a) * (1.f / (cmidp_1.gamInt - 1.f) * powDd(&d1, &d2) * (*funk)(&d3));
    } else {
        i1 = *n - 2;
        it = powIi(&c3, &i1);
        tnm = static_cast<SedReal>(it);
        del = (b - a) / (tnm * 3.f);
        ddel = del + del;
        x = a + del * .5f;
        sum = 0.f;
        i1 = it;
        for (j = 1; j <= i1; ++j) {
            d1 = cmidp_1.gamInt / (1.f - cmidp_1.gamInt);
            d3 = 1.f / (1.f - cmidp_1.gamInt);
            d2 = *aa - powDd(&x, &d3);
            sum += 1.f / (cmidp_1.gamInt - 1.f) * powDd(&x, &d1) * (*funk)(&d2);
            x += ddel;
            d1 = cmidp_1.gamInt / (1.f - cmidp_1.gamInt);
            d3 = 1.f / (1.f - cmidp_1.gamInt);
            d2 = *aa - powDd(&x, &d3);
            sum += 1.f / (cmidp_1.gamInt - 1.f) * powDd(&x, &d1) * (*funk)(&d2);
            x += del;
        }
        *s = (*s + (b - a) * sum / tnm) / 3.f;
    }
    return 0;
} /* midpowl2_ */

/* Subroutine */ int qromo(
    DFp func, SedReal* a, SedReal* b, SedReal* eps, SedReal* ss, SFp choose) {
    /* Builtin functions */

    /* Local variables */
    static SedReal hArg[15];
    static int j;
    static SedReal s[15], dss;

    /* Fortran I/O blocks */

    /* ..this routine returns as s the integral of the function func from a to b */
    /* ..with fractional accuracy eps. */ /* ..jmax limits the number of steps; numberSteps =
                                           * 3**(jmax-1)
                                           */
    /* ..integration is done via romberg algorithm. */
    /* ..it is assumed the call to choose triples the number of steps on each call */
    /* ..and that its error series contains only even powers of the number of steps. */
    /* ..the external choose may be any of the above drivers, i.e midpnt,midinf... */
    /* ..declare */
    hArg[0] = 1.f;
    for (j = 1; j <= 14; ++j) {
        (*choose)(static_cast<DFp>(func), a, b, &s[j - 1], &j);
        if (j >= 5) {
            polint(&hArg[j - 5], &s[j - 5], &c5, &cB79, ss, &dss);
            if (fabs(static_cast<double>(dss)) <= *eps * fabs(static_cast<double>(*ss))) {
                return 0;
            }
        }
        s[j] = s[j - 1];
        hArg[j] = hArg[j - 1] / 9.f;
    }
    return 0;
} /* qromo_ */

/* Subroutine */ int polint(SedReal* xa, SedReal* ya, int* n, SedReal* x, SedReal* y, SedReal* dy) {
    /* System generated locals */
    int i1, i2;
    SedReal d1;

    /* Local variables */
    static SedReal cArg[20], dArg[20];
    static int iArg, m;
    static SedReal w, ho, hp;
    static int ns;
    static SedReal dif, den, dift;

    /* ..given arrays xa and ya of length n and a value x, this routine returns a */
    /* ..value y and an error estimate dy. if p(x) is the polynomial of degree n-1 */
    /* ..such that ya = p(xa) ya then the returned value is y = p(x) */
    /* ..declare */
    /* ..find the index ns of the closest table entry; initialize the c and d tables */
    /* Parameter adjustments */
    --ya;
    --xa;

    /* Function Body */
    ns = 1;
    dif = (d1 = *x - xa[1], fabs(static_cast<double>(d1)));
    i1 = *n;
    for (iArg = 1; iArg <= i1; ++iArg) {
        dift = (d1 = *x - xa[iArg], fabs(static_cast<double>(d1)));
        if (dift < dif) {
            ns = iArg;
            dif = dift;
        }
        cArg[iArg - 1] = ya[iArg];
        dArg[iArg - 1] = ya[iArg];
    }
    /* ..first guess for y */
    *y = ya[ns];
    /* ..for each column of the table, loop over the c's and d's and update them */
    --ns;
    i1 = *n - 1;
    for (m = 1; m <= i1; ++m) {
        i2 = *n - m;
        for (iArg = 1; iArg <= i2; ++iArg) {
            ho = xa[iArg] - *x;
            hp = xa[iArg + m] - *x;
            w = cArg[iArg] - dArg[iArg - 1];
            den = ho - hp;
            if (den == 0.f) {
                //			s_stop(" 2 xa entries are the same in polint", (ftnlen) 36);
            }
            den = w / den;
            dArg[iArg - 1] = hp * den;
            cArg[iArg - 1] = ho * den;
        }
        /* ..after each column is completed, decide which correction c or d, to add */
        /* ..to the accumulating value of y, that is, which path to take in the table */
        /* ..by forking up or down. ns is updated as we go to keep track of where we */
        /* ..are. the last dy added is the error indicator. */
        if (ns << 1 < *n - m) {
            *dy = cArg[ns];
        } else {
            *dy = dArg[ns - 1];
            --ns;
        }
        *y += *dy;
    }
    return 0;
} /* polint_ */

SedReal zeroin(SedReal* ax, SedReal* bx, DFp f, SedReal* tol) {
    /* System generated locals */
    SedReal retVal, d1;
    retVal = 0;

    /* Local variables */
    static SedReal a, b, cArg, dArg, e, p, q, rArg, s, fa, fb, fc, xm, eps, tol1;

    /* ----------------------------------------------------------------------- */

    /* This subroutine solves for a zero of the function  f(x)  in the */
    /* interval ax,bx. */

    /*  input.. */

    /*  ax     left endpoint of initial interval */
    /*  bx     right endpoint of initial interval */
    /*  f      function subprogram which evaluates f(x) for any x in */
    /*         the interval  ax,bx */
    /*  tol    desired length of the interval of uncertainty of the */
    /*         final result ( .ge. 0.0) */

    /*  output.. */

    /*  zeroin abcissa approximating a zero of  f  in the interval ax,bx */

    /*      it is assumed  that   f(ax)   and   f(bx)   have  opposite  signs */
    /*  without  a  check.  zeroin  returns a zero  x  in the given interval */
    /*  ax,bx  to within a tolerance  4*macheps*fabs(x) + tol, where macheps */
    /*  is the relative machine precision. */
    /*      this function subprogram is a slightly  modified  translation  of */
    /*  the algol 60 procedure  zero  given in  richard brent, algorithms for */
    /*  minimization without derivatives, prentice - hall, inc. (1973). */

    /* ----------------------------------------------------------------------- */
    /* .... call list variables */

    /* ---------------------------------------------------------------------- */

    /*  compute eps, the relative machine precision */

    eps = 1.f;
L10:
    eps /= 2.f;
    tol1 = eps + 1.f;
    if (tol1 > 1.f) {
        goto L10;
    }

    /* initialization */

    a = *ax;
    b = *bx;
    fa = (*f)(&a);
    fb = (*f)(&b);

    /* begin step */

L20:
    cArg = a;
    fc = fa;
    dArg = b - a;
    e = dArg;
L30:
    if (fabs(static_cast<double>(fc)) >= fabs(static_cast<double>(fb))) {
        goto L40;
    }
    a = b;
    b = cArg;
    cArg = a;
    fa = fb;
    fb = fc;
    fc = fa;

    /* convergence test */

L40:
    tol1 = eps * 2.f * fabs(static_cast<double>(b)) + *tol * .5f;
    xm = (cArg - b) * .5f;
    if (fabs(static_cast<double>(xm)) <= tol1) {
        goto L90;
    }
    if (fb == 0.f) {
        goto L90;
    }

    /* is bisection necessary? */

    if (fabs(static_cast<double>(e)) < tol1) {
        goto L70;
    }
    if (fabs(static_cast<double>(fa)) <= fabs(static_cast<double>(fb))) {
        goto L70;
    }

    /* is quadratic interpolation possible? */

    if (a != cArg) {
        goto L50;
    }

    /* linear interpolation */

    s = fb / fa;
    p = xm * 2.f * s;
    q = 1.f - s;
    goto L60;

    /* inverse quadratic interpolation */

L50:
    q = fa / fc;
    rArg = fb / fc;
    s = fb / fa;
    p = s * (xm * 2.f * q * (q - rArg) - (b - a) * (rArg - 1.f));
    q = (q - 1.f) * (rArg - 1.f) * (s - 1.f);

    /* adjust signs */

L60:
    if (p > 0.f) {
        q = -q;
    }
    p = fabs(static_cast<double>(p));

    /* is interpolation acceptable? */

    if (p * 2.f >= xm * 3.f * q - (d1 = tol1 * q, fabs(static_cast<double>(d1)))) {
        goto L70;
    }
    if (p >= (d1 = e * .5f * q, fabs(static_cast<double>(d1)))) {
        goto L70;
    }
    e = dArg;
    dArg = p / q;
    goto L80;

    /* bisection */

L70:
    dArg = xm;
    e = dArg;

    /* complete step */

L80:
    a = b;
    fa = fb;
    if (fabs(static_cast<double>(dArg)) > tol1) {
        b += dArg;
    }
    if (fabs(static_cast<double>(dArg)) <= tol1) {
        b += dSign(&tol1, &xm);
    }
    fb = (*f)(&b);
    if (fb * (fc / fabs(static_cast<double>(fc))) > 0.f) {
        goto L20;
    }
    goto L30;

    /* done */

L90:
    retVal = b;
    return retVal;
} /* zeroin_ */

using MutexType = hpx::spinlock;

namespace sedov {

void solution(double time, double r, double rmax, double& d, double& v, double& p, int ndim) {
    int nstep = 10000;
    constexpr int kBw = 2;
    using FunctionType = std::function<void(double, double&, double&, double&)>;
    using MapType = std::unordered_map<double, std::shared_ptr<FunctionType>>;

    static MapType map;
    static MutexType mutex;

    SedReal rho0 = 1.0;
    SedReal vel0 = 0.0;
    SedReal ener0 = 0.0;
    SedReal pres0 = 0.0;
    SedReal cs0 = 0.0;
    SedReal gamma = 5.0 / 3.0;
    SedReal omega = 0.0;
    SedReal eblast = 1.0;
    SedReal xgeom = SedReal(ndim);

    std::vector<SedReal> xpos(nstep + 2 * kBw);
    std::vector<SedReal> den(nstep + 2 * kBw);
    std::vector<SedReal> ener(nstep + 2 * kBw);
    std::vector<SedReal> pres(nstep + 2 * kBw);
    std::vector<SedReal> vel(nstep + 2 * kBw);
    std::vector<SedReal> cs(nstep + 2 * kBw);

    std::vector<double> den1(nstep + 2 * kBw);
    std::vector<double> pres1(nstep + 2 * kBw);
    std::vector<double> vel1(nstep + 2 * kBw);

    std::shared_ptr<FunctionType> ptr;

    for (int i = 0; i < nstep + 2 * kBw; i++) {
        xpos[i] = (i - kBw + 0.5) * rmax / (nstep);
    }
    nstep += kBw;

    std::unique_lock<MutexType> lock(mutex);
    auto iter = map.find(time);
    if (iter == map.end()) {
        SedReal sedTime = time;
        printf("Computing sedov solution\n");
        sed1d(&sedTime, &nstep, xpos.data() + kBw, &eblast, &omega, &xgeom, &rho0, &vel0, &ener0,
            &pres0, &cs0, &gamma, den.data() + kBw, ener.data() + kBw, pres.data() + kBw,
            vel.data() + kBw, cs.data() + kBw);

        xpos[0] = -xpos[3];
        den[0] = den[3];
        ener[0] = ener[3];
        pres[0] = pres[3];
        vel[0] = -vel[3];
        cs[0] = cs[3];

        xpos[1] = -xpos[2];
        den[1] = den[2];
        ener[1] = ener[2];
        pres[1] = pres[2];
        vel[1] = -vel[2];
        cs[1] = cs[2];

#if defined(OCTOTIGER_HAVE_BOOST_MULTIPRECISION)
        std::transform(
            den.begin(), den.end(), den1.begin(), [](SedReal v) { return v.convert_to<double>(); });
        std::transform(
            vel.begin(), vel.end(), vel1.begin(), [](SedReal v) { return v.convert_to<double>(); });
        std::transform(pres.begin(), pres.end(), pres1.begin(),
            [](SedReal v) { return v.convert_to<double>(); });
#else
        std::copy(den.begin(), den.end(), den1.begin());
        std::copy(vel.begin(), vel.end(), vel1.begin());
        std::copy(pres.begin(), pres.end(), pres1.begin());
#endif

        FunctionType func = [nstep, rmax, den1, pres1, vel1, kBw](
                                double r, double& d, double& v, double& p) {
            double dr = rmax / (nstep);
            std::array<int, 4> i;
            i[1] = (r + (kBw - 0.5) * dr) / dr;
            i[0] = i[1] - 1;
            i[2] = i[1] + 1;
            i[3] = i[1] + 2;
            double r0 = (r - (i[1] - kBw + 0.5) * dr) / dr;
            //		printf( "%i %e\n", i[0], r, dr );
            assert(i[0] >= 0);
            assert(i[3] < int(vel1.size()));
            auto const interp = [r0, i](std::vector<double> const& data) {
                double sum = 0.0;
                sum +=
                    (-0.5 * data[i[0]] + 1.5 * data[i[1]] - 1.5 * data[i[2]] + 0.5 * data[i[3]]) *
                    r0 * r0 * r0;
                sum +=
                    (+1.0 * data[i[0]] - 2.5 * data[i[1]] + 2.0 * data[i[2]] - 0.5 * data[i[3]]) *
                    r0 * r0;
                sum += (-0.5 * data[i[0]] + 0.5 * data[i[2]]) * r0;
                sum += data[i[1]];
                return sum;
            };

            d = interp(den1);
            v = interp(vel1);
            p = interp(pres1);
        };

        ptr = std::make_shared<FunctionType>(std::move(func));
        map[time] = ptr;
        lock.unlock();
    } else {
        lock.unlock();
        ptr = iter->second;
    }

    auto const& func = *(ptr);

    func(r, d, v, p);
}

}    // namespace sedov
