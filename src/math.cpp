
#include <cosmictiger/containers.hpp>
#include <cosmictiger/math.hpp>

#define Power(a, b) pow(a, b)
#define List(...) {__VA_ARGS__}

Pair<double, Array<double, NDIM>> findEigenpair(
    Array<Array<double, NDIM>, NDIM> const& aValue, double mu) {
    Array<double, NDIM> b;
    b[ZDIM] = 1.0;
    b[XDIM] = b[YDIM] = 0.0;
    double err;
    double a11 = aValue[0][0];
    double a12 = aValue[0][1];
    double a13 = aValue[0][2];
    double a22 = aValue[1][1];
    double a23 = aValue[1][2];
    double a33 = aValue[2][2];
    double lambda, lastLambda;
    lambda = 1e30;
    do {
        double det = -(Power(a13, 2) * a22) + 2 * a12 * a13 * a23 - a11 * Power(a23, 2) -
            Power(a12, 2) * a33 + a11 * a22 * a33 + Power(a12, 2) * mu + Power(a13, 2) * mu -
            a11 * a22 * mu + Power(a23, 2) * mu - a11 * a33 * mu - a22 * a33 * mu +
            a11 * Power(mu, 2) + a22 * Power(mu, 2) + a33 * Power(mu, 2) - Power(mu, 3);

        double cValue[NDIM][NDIM] =
            List(List(-Power(a23, 2) + a22 * a33 - a22 * mu - a33 * mu + Power(mu, 2),
                     a13 * a23 - a12 * a33 + a12 * mu, -(a13 * a22) + a12 * a23 + a13 * mu),
                List(a13 * a23 - a12 * a33 + a12 * mu,
                    -Power(a13, 2) + a11 * a33 - a11 * mu - a33 * mu + Power(mu, 2),
                    a12 * a13 - a11 * a23 + a23 * mu),
                List(-(a13 * a22) + a12 * a23 + a13 * mu, a12 * a13 - a11 * a23 + a23 * mu,
                    -Power(a12, 2) + a11 * a22 - a11 * mu - a22 * mu + Power(mu, 2)));

        double norm = 0.0;
        Array<double, NDIM> bn1;
        for (int dim1 = 0; dim1 < NDIM; dim1++) {
            bn1[dim1] = 0.0;
            for (int dim2 = 0; dim2 < NDIM; dim2++) {
                cValue[dim1][dim2] /= det;
                bn1[dim1] += cValue[dim1][dim2] * b[dim2];
            }
            norm += sqr(bn1[dim1]);
        }
        norm = sqrt(norm);
        for (int dim = 0; dim < NDIM; dim++) {
            bn1[dim] /= norm;
        }
        b = bn1;
        Array<double, NDIM> c;
        for (int dim1 = 0; dim1 < NDIM; dim1++) {
            c[dim1] = 0.0;
            for (int dim2 = 0; dim2 < NDIM; dim2++) {
                c[dim1] += aValue[dim1][dim2] * b[dim2];
            }
        }
        double b2 = sqr(b[XDIM], b[YDIM], b[ZDIM]);
        double cb = (c[XDIM] * b[XDIM] + c[YDIM] * b[YDIM] + c[ZDIM] * b[ZDIM]);
        lastLambda = lambda;
        lambda = cb / b2;
        err = fabs(lastLambda / lambda - 1.0);
    } while (err > 1.0e-10);
    Pair<double, Array<double, NDIM>> rc;
    rc.first = lambda;
    rc.second = b;
    return rc;
}
