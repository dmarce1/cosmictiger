/*
 * domain.hpp
 *
 *  Created on: Jul 17, 2021
 *      Author: dmarce1
 */

#ifndef DOMAIN_HPP_
#define DOMAIN_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/range.hpp>

void domains_begin();
void domains_end();
range<double> domains_find_my_box();

#endif /* DOMAIN_HPP_ */
