#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/io/output.hpp>

size_t maxCpuMemUse();
HPX_PLAIN_ACTION(maxCpuMemUse, MaxCpuMemUseAction);

// some of this code adapted from
// https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process

static int parseLine(char* line) {
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    char const* p = line;
    while (*p < '0' || *p > '9') {
        p++;
    }
    line[i - 3] = '\0';
    i = atoi(p);
    return i;
}

static size_t cpuMemUse() {    // Note: this value is in KB!
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

static bool stopDaemon = false;
static bool daemonStopped = false;

size_t maxCpuMemUse() {
    Vector<hpx::future<size_t>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<MaxCpuMemUseAction>(c));
    }
    size_t maxUse = cpuMemUse();
    for (auto& f : futs) {
        maxUse = std::max(maxUse, f.get());
    }

    return maxUse;
}

static void memuseDaemon(std::string filename) {
    static size_t maxMemused = 0;
    Timer tm;
    tm.start();
    while (!stopDaemon) {
        tm.stop();
        if (tm.read() >= 1.0) {
            auto const thismemused = maxCpuMemUse();
            if (thismemused > maxMemused) {
                maxMemused = thismemused;
                PRINT("MAX MEMUSED = %f GB\n", thismemused / 1024. / 1024. / 1024.);
                FILE* fp = fopen(filename.c_str(), "wt");
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
    daemonStopped = true;
}

void startMemuseDaemon() {
    hpx::apply(memuseDaemon, outputPath("maxmem.txt"));
}

void stopMemuseDaemon() {
    stopDaemon = true;
    while (!daemonStopped) {
        hpx::this_thread::yield();
    }
}
