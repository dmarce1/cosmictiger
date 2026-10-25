/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/particles/particle_mesh_sort.hpp>

struct TreeCellRoot {
    uint64_t key;
    TreeId id;
    bool ghost = false;
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & key & id & ghost;
    }
};

// Collective, called once on locality zero after all previous tree readers have
// finished. Local forest builds and distributed joins share this arena epoch.
void treePrepareCellForest();
// Local preparation before any cell build or reader starts. The supplied counts
// include owned source particles and reserved ghost slices. No collective or
// particle scan is performed, and arena addresses remain fixed for the epoch.
void treePrepareCellForestLocal(size_t sourceParticleCount, size_t sourceRootCount);
// Each caller owns a disjoint range in the selected store until this completes.
// The optional source box is the owner's domain; ghost roots otherwise cover
// the full cell. The returned future/task completion publishes a read-only tree.
TreeCellRoot treeCreateCell(ParticleMeshCell const& cell, int meshSize, FmmParticleStore store,
    Range<double> const* sourceBox = nullptr);
// Compatibility wrapper for a complete forest. Cells are ordered by (ghost,
// key), and ranges are disjoint within each store. Sink roots must be owned.
Vector<TreeCellRoot> treeCreateCellForest(Vector<ParticleMeshCell> const& cells, int meshSize);
// Called once per occupied cell, after local builds finish. The contributors
// contain one local root per participating locality; the result is one logical
// root with the original domain kd hierarchy and empty branches removed.
TreeId treeCombineCellRoots(uint64_t cellKey, int meshSize, Vector<TreeId> contributors);
