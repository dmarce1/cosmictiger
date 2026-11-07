/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#ifndef VIEW_HPP_
#define VIEW_HPP_

#if COSMICTIGER_WITH_GRAVITY
Vector<float> outputView(int number, double time);
#endif
// Pass the actual elapsed cosmological age when a may revisit an earlier value.
// A negative physicalYears retains the expanding-branch diagnostic fallback
// when gravity is compiled, or writes zero when cosmology is unavailable.
// Native noncosmological snapshots store elapsed code time; cosmological and
// converted snapshots store age/elapsed physical time in Gyr.
// baseTimestep is the interval of the density pass that refreshes time-derivative output.
void viewOutputViews(int cycle, double a, bool force = false, double physicalYears = -1.0,
    double baseTimestep = 0.0, double conformalTime = 0.0, double scaleDerivative = 0.0);

void viewConfigure();

#endif /* VIEW_HPP_ */
