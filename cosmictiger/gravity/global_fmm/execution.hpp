/* CosmicTiger - gravity execution; BSL-1.0. */
#pragma once

#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/math/vector3.hpp>

#ifndef __CUDACC__
KickReturn kick(KickParameters, ExpansionBuffer, Vector3<Position>, TreeId, Vector<TreeId>,
    Vector<TreeId>, std::shared_ptr<KickWorkspace>);
#endif
void kickSetRc(KickReturn result);
void kickClearBuffers();
