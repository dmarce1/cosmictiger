/*
 * kick_workspace.hpp
 *
 *  Created on: Jul 24, 2021
 *      Author: dmarce1
 */


#ifndef KICK_WORKSPACE_HPP_
#define KICK_WORKSPACE_HPP_

#include <tigerfmm/cuda.hpp>
#include <tigerfmm/defs.hpp>
#include <tigerfmm/kick.hpp>


struct kick_workspace {
	vector<tree_node, pinned_allocator<tree_node>> tree_space;
	fixed32* dev_x;
	fixed32* dev_y;
	fixed32* dev_z;

};
#endif /* KICK_WORKSPACE_HPP_ */
