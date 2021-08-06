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
double cosmos_age(double a0);

#endif /* COSMOLOGY_HPP_ */
