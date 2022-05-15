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

#include <cosmictiger/sphere.hpp>
#include <cosmictiger/hpx.hpp>
#include <queue>

static mutex_type mtx;
static std::queue<pair<tree_node*, std::shared_ptr<hpx::promise<void>>> > workq;
static bool running = false;

void sphere_daemon() {
	while (running) {
		if (workq.size()) {
			vector<tree_node*> nodes;
			vector<std::shared_ptr<hpx::promise<void>>>promises;
			{
				std::lock_guard<mutex_type> lock(mtx);
				while (workq.size()) {
					nodes.push_back(workq.front().first);
					promises.push_back(workq.front().second);
					workq.pop();
				}
			}
			sphere_to_gpu(nodes);
			for (int i = 0; i < promises.size(); i++) {
				promises[i]->set_value();
			}
		}
		hpx_yield();
	}
}

hpx::future<void> sphere_find_bounding(tree_node* self) {
	auto prms = std::make_shared<hpx::promise<void>>();
	auto f = prms->get_future();
	pair<tree_node*, std::shared_ptr<hpx::promise<void>>> entry;
	entry.first = self;
	entry.second = prms;
	std::lock_guard<mutex_type> lock(mtx);
	workq.push(entry);
	return f;
}

void sphere_start_daemon() {
	running = true;
	hpx::async(sphere_daemon);
}

void sphere_stop_daemon() {
	running = false;

}
