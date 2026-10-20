#pragma once

#include <string>

// Standalone utilities keep the working-directory default. The simulation
// configures this path after resolving its scenario output directory.
void setStackTraceOutputPath(std::string const& filename);
std::string stackTraceOutputPath();
