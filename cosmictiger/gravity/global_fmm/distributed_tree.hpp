/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

// Distributed FMM tree construction, source selection and geometric-tree lifetime APIs.

#pragma once

#include <cosmictiger/gravity/global_fmm/source_forest.hpp>
#include <cosmictiger/gravity/tree.hpp>

TreeCreateReturn treeCreate(TreeCreateParameters params, size_t key = 1,
    Pair<int, int> processRange = Pair<int>(0, hpxSize()),
    Pair<ParticleInteger> particleRange = Pair<ParticleInteger>(-1, -1),
    Range<double> box = unitBox<double>(), int depth = 0, bool localRoot = (hpxSize() == 1));

// Build independent source trees, exchange destination-specific source views, and
// retain their descriptors for a complete-source reference traversal. Builds retain
// the resulting particle order; AllOwned requires a full particlesCurrentRange(),
// while CurrentRungRange preserves the nested hierarchical suffix boundaries.
enum class GeometricSourceSelection {
    AllOwned,           // Use every owned particle; requires the current range to contain them all.
    CurrentRungRange    // Use only the current hierarchical rung's nested particle suffix.
};
TreeCreateReturn treeCreateGeometric(TreeCreateParameters params,
    GeometricSourceSelection selection = GeometricSourceSelection::AllOwned, bool exchange = true);
// exchange=false is for a cold complete-source measurement before explicit exchange.
Vector<SourceRootDescriptor> const& treeGeometricRoots();
void treeRetireGeometric(bool freeTree = false);
