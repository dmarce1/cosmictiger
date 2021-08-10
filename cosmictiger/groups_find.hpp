#pragma once


#include <cosmictiger/defs.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/hpx.hpp>



hpx::future<size_t> groups_find(tree_id self, vector<tree_id> checklist, double link_len);
