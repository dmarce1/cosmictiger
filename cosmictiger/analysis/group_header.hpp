#pragma once

#include <type_traits>

// Native groups.dat header, independent of the simulation and HPX headers.
struct GroupHeader {
    int number;
    double time;
    double boxLength;
    double scale;
};

static_assert(std::is_trivial_v<GroupHeader> && std::is_standard_layout_v<GroupHeader>);
