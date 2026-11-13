/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <cosmictiger/gravity/sources.hpp>

/// Return a non-owning view of the local particle arrays.
FmmSourceView fmmSourceView() {
    FmmSourceView view;
    view.positions = particlesX;
    view.types = getOptions().hydro.enabled ? particlesTy : nullptr;
    view.count = particlesSize();
    return view;
}
