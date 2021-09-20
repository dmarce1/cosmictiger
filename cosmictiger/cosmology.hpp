/*
 * cosmology.hpp
 *
 *  Created on: Aug 6, 2021
 *      Author: dmarce1
 */

#ifndef COSMOLOGY_HPP_
#define COSMOLOGY_HPP_




double cosmos_growth_factor(double omega_m, float a);
double cosmos_dadtau(double a);
double cosmos_dadt(double a);
double cosmos_time(double a0, double a1);
double cosmos_ainv(double a0, double a1);
double cosmos_conformal_time(double a0, double a1);


#endif /* COSMOLOGY_HPP_ */
