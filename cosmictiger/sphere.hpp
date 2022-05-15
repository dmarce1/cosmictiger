/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/



#include <cosmictiger/tree.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/containers.hpp>

void sphere_to_gpu(const vector<tree_node*>& tree_nodes);
#ifndef __CUDACC__
hpx::future<void> sphere_find_bounding(tree_node* self);
#endif
void sphere_start_daemon();
void sphere_stop_daemon();
