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

#include <unordered_map>
#include <string>
#include <stack>

#include <cosmictiger/containers.hpp>
#include <cosmictiger/timer.hpp>
#include <algorithm>

static std::unordered_map<std::string, timer> timers;
static std::stack<std::string> stack;

void profiler_enter(const char* name) {
	if (!stack.empty()) {
		timers[stack.top()].stop();
	}
	timers[name].start();
	stack.push(name);
}

void profiler_exit() {
	timers[stack.top()].stop();
	stack.pop();
	if (!stack.empty()) {
		timers[stack.top()].start();
	}
}

void profiler_output() {
	if (!stack.empty()) {
		timers[stack.top()].stop();
	}
	vector<std::pair<std::string, timer>> results(timers.begin(), timers.end());
	std::sort(results.begin(), results.end(), [](std::pair<std::string,timer>& a, std::pair<std::string,timer>& b) {
		return a.second.read() > b.second.read();
	});

	FILE* fp = fopen("profile.txt", "wt");
	double total = 0.0;
	for (int i = 0; i < results.size(); i++) {
		total += results[i].second.read();
	}
	double totalinv = 1.0 / total;
	for (int i = 0; i < results.size(); i++) {
		fprintf(fp, "%s %e %f%%\n", results[i].first.c_str(), results[i].second.read(), results[i].second.read() * totalinv * 100.0);
	}

	fclose(fp);
	if (!stack.empty()) {
		timers[stack.top()].start();
	}
}

