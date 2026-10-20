/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <stack>
#include <string>
#include <unordered_map>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/io/output.hpp>

static std::unordered_map<std::string, Timer> timers;
static std::stack<std::string> stack;

void profilerEnter(char const* name) {
    //	PRINT( "Entering %s\n", name);
    if (!stack.empty()) {
        timers[stack.top()].stop();
    }
    timers[name].start();
    stack.push(name);
}

void profilerExit() {
    //	PRINT( "Leaving %s\n", stack.top().c_str());
    timers[stack.top()].stop();
    stack.pop();
    if (!stack.empty()) {
        timers[stack.top()].start();
    }
}

void profilerOutput() {
    if (!stack.empty()) {
        timers[stack.top()].stop();
    }
    Vector<std::pair<std::string, Timer>> results(timers.begin(), timers.end());
    std::sort(results.begin(), results.end(),
        [](std::pair<std::string, Timer>& a, std::pair<std::string, Timer>& b) {
            return a.second.read() > b.second.read();
        });

    FILE* fp = fopen(outputPath("profile.txt").c_str(), "wt");
    double total = 0.0;
    for (int i = 0; i < results.size(); i++) {
        total += results[i].second.read();
    }
    double totalinv = 1.0 / total;
    for (int i = 0; i < results.size(); i++) {
        fprintf(fp, "%s %e %f%%\n", results[i].first.c_str(), results[i].second.read(),
            results[i].second.read() * totalinv * 100.0);
    }

    fclose(fp);
    if (!stack.empty()) {
        timers[stack.top()].start();
    }
}
