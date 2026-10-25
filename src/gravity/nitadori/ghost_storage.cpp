/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <limits>
#include <memory>
#include <stdexcept>

#include <cosmictiger/cuda/cuda_memory.hpp>
#include <cosmictiger/gravity/nitadori/ghost_storage.hpp>

namespace detail {
static char* ghostArena = nullptr;
static ParticleInteger ghostCapacity = 0;
static FmmSourceView ghostSources;
}    // namespace detail

using detail::ghostArena;
using detail::ghostCapacity;
using detail::ghostSources;

FmmSourceView fmmGhostView() {
    return ghostSources;
}

void fmmGhostReserve(ParticleInteger count) {
    static bool const registered = [] {
        particlesRegisterReleaseHook(fmmGhostFree);
        return true;
    }();
    (void) registered;
    if (count < 0) {
        throw std::invalid_argument("Ghost source count must be nonnegative");
    }
    if (count > ghostCapacity) {
        constexpr size_t kAlignment = 64;
        size_t const entries = static_cast<size_t>(count);
        if (entries >
            (std::numeric_limits<size_t>::max() - kAlignment) / (NDIM * sizeof(Position) + 2)) {
            throw std::length_error("Ghost source arena size overflows size_t");
        }
        size_t const stride =
            (entries * sizeof(Position) + kAlignment - 1) / kAlignment * kAlignment;
        if (stride > (std::numeric_limits<size_t>::max() - 2 * entries - kAlignment) / NDIM) {
            throw std::length_error("Ghost source arena size overflows size_t");
        }
        size_t const bytes = NDIM * stride + 2 * entries;
        size_t space = bytes + kAlignment - 1;
        char* replacement = static_cast<char*>(allocateManagedMemory(space));
        if (replacement == nullptr) {
            throw std::bad_alloc();
        }
        void* aligned = replacement;
        std::align(kAlignment, bytes, aligned, space);
        char* data = static_cast<char*>(aligned);
        FmmSourceView next;
        for (int dim = 0; dim < NDIM; ++dim) {
            next.positions[dim] = reinterpret_cast<Position*>(data + dim * stride);
        }
        next.types = data + NDIM * stride;
        next.rungs = next.types + entries;
        if (ghostArena != nullptr) {
            freeManagedMemory(ghostArena);
        }
        ghostArena = replacement;
        ghostCapacity = count;
        ghostSources = next;
    }
    ghostSources.count = count;
    fmmRegisterSourceView(FmmParticleStore::Ghost, ghostSources);
}

void fmmGhostClear() {
    ghostSources.count = 0;
    fmmRegisterSourceView(FmmParticleStore::Ghost, ghostSources);
}

void fmmGhostFree() {
    freeManagedMemory(ghostArena);
    ghostArena = nullptr;
    ghostCapacity = 0;
    ghostSources = {};
    fmmRegisterSourceView(FmmParticleStore::Ghost, ghostSources);
}
