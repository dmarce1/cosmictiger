#include <cosmictiger/hpx.hpp>
#include <cosmictiger/timer.hpp>

#include "stdlib.h"
#include "stdio.h"
#include "string.h"

size_t max_cpu_mem_use();
HPX_PLAIN_ACTION (max_cpu_mem_use);

// some of this code adapted from https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process

static int parseLine(char* line) {
	// This assumes that a digit will be found and the line ends in " Kb".
	int i = strlen(line);
	const char* p = line;
	while (*p < '0' || *p > '9')
		p++;
	line[i - 3] = '\0';
	i = atoi(p);
	return i;
}

static size_t cpu_mem_use() { //Note: this value is in KB!
	FILE* file = fopen("/proc/self/status", "r");
	int result = -1;
	char line[128];

	while (fgets(line, 128, file) != NULL) {
		if (strncmp(line, "VmRSS:", 6) == 0) {
			result = parseLine(line);
			break;
		}
	}
	fclose(file);
	return size_t(result) * 1024;
}

static bool stop_daemon = false;
static bool daemon_stopped = false;

size_t max_cpu_mem_use() {
	vector < hpx::future < size_t >> futs;
	for (auto& c : hpx_children()) {
		futs.push_back(hpx::async<max_cpu_mem_use_action>( c));
	}
	size_t max_use = cpu_mem_use();
	for (auto& f : futs) {
		max_use = std::max(max_use, f.get());
	}

	return max_use;
}

static void memuse_daemon() {
	static size_t max_memused = 0;
	timer tm;
	tm.start();
	while (!stop_daemon) {
		tm.stop();
		if (tm.read() >= 1.0) {
			const auto thismemused = max_cpu_mem_use();
			if (thismemused > max_memused) {
				max_memused = thismemused;
				PRINT("MAX MEMUSED = %f GB\n", thismemused / 1024. / 1024. / 1024.);
				FILE* fp = fopen("maxmem.txt", "wt");
				if (fp == NULL) {
					THROW_ERROR("Unable to open maxmem.txt\n");
				}
				fprintf(fp, "%f GB\n", thismemused / 1024. / 1024. / 1024.);
				fclose(fp);
			}
			tm.reset();
		}
		tm.start();
		hpx::this_thread::yield();
	}
	daemon_stopped = true;
}

void start_memuse_daemon() {
	hpx::apply(memuse_daemon);
}

void stop_memuse_daemon() {
	stop_daemon = true;
	while (!daemon_stopped) {
		hpx::this_thread::yield();
	}
}

