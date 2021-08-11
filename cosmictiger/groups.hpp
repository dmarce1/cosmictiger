/*
 * groups.hpp
 *
 *  Created on: Aug 10, 2021
 *      Author: dmarce1
 */

#ifndef GROUPS_HPP_
#define GROUPS_HPP_


#include <cosmictiger/defs.hpp>


void groups_add_particles(int wave);
void groups_reduce();
void groups_save(int number);
void groups_cull();

#endif /* GROUPS_HPP_ */
