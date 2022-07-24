
#include <cosmictiger/containers.hpp>
#include <cosmictiger/math.hpp>

#define Power(a,b) pow(a,b)
#define List(...) { __VA_ARGS__ }

pair<double, array<double, NDIM>> find_eigenpair(const array<array<double, NDIM>, NDIM>& A, double mu) {
	array<double, NDIM> b;
	b[ZDIM] = 1.0;
	b[XDIM] = b[YDIM] = 0.0;
	double err;
	double a11 = A[0][0];
	double a12 = A[0][1];
	double a13 = A[0][2];
	double a22 = A[1][1];
	double a23 = A[1][2];
	double a33 = A[2][2];
	double lambda, last_lambda;
	lambda = 1e30;
	do {
		double det = -(Power(a13,2) * a22) + 2 * a12 * a13 * a23 - a11 * Power(a23, 2) - Power(a12,2) * a33 + a11 * a22 * a33 + Power(a12,2) * mu
				+ Power(a13,2) * mu - a11 * a22 * mu + Power(a23,2) * mu - a11 * a33 * mu - a22 * a33 * mu + a11 * Power(mu, 2) + a22 * Power(mu, 2)
				+ a33 * Power(mu, 2) - Power(mu, 3);

		double C[NDIM][NDIM] = List(List(-Power(a23,2) + a22*a33 - a22*mu - a33*mu + Power(mu,2),a13*a23 - a12*a33 + a12*mu,-(a13*a22) + a12*a23 + a13*mu),
		   List(a13*a23 - a12*a33 + a12*mu,-Power(a13,2) + a11*a33 - a11*mu - a33*mu + Power(mu,2),a12*a13 - a11*a23 + a23*mu),List(-(a13*a22) + a12*a23 + a13*mu,a12*a13 - a11*a23 + a23*mu,-Power(a12,2) + a11*a22 - a11*mu - a22*mu + Power(mu,2)));

		double norm = 0.0;
		array<double, NDIM> bn1;
		for( int dim1 = 0; dim1 < NDIM; dim1++) {
			bn1[dim1] = 0.0;
			for( int dim2 = 0; dim2 < NDIM; dim2++) {
				C[dim1][dim2] /= det;
				bn1[dim1] += C[dim1][dim2] * b[dim2];
			}
			norm += sqr(bn1[dim1]);
		}
		norm = sqrt(norm);
		for( int dim = 0; dim < NDIM; dim++) {
			bn1[dim] /= norm;
		}
		b = bn1;
		array<double, NDIM> c;
		for( int dim1 = 0; dim1 < NDIM; dim1++) {
			c[dim1] = 0.0;
			for( int dim2 = 0; dim2 < NDIM; dim2++) {
				c[dim1] += A[dim1][dim2] * b[dim2];
			}
		}
		double b2 = sqr(b[XDIM], b[YDIM], b[ZDIM]);
		double cb = (c[XDIM] * b[XDIM] + c[YDIM] * b[YDIM] + c[ZDIM] * b[ZDIM]);
		last_lambda = lambda;
		lambda = cb / b2;
		err = fabs(last_lambda / lambda - 1.0);
	} while (err > 1.0e-10);
	pair<double, array<double, NDIM>> rc;
	rc.first = lambda;
	rc.second = b;
	return rc;
}

