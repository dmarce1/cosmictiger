/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#ifndef VIEW_HPP_
#define VIEW_HPP_

Vector<float> outputView(int number, double time);
// Pass the actual elapsed cosmological age when a may revisit an earlier value.
// A negative physicalYears retains the expanding-branch diagnostic fallback.
void viewOutputViews(int cycle, double a, bool force = false, double physicalYears = -1.0);

void viewConfigure();

#endif /* VIEW_HPP_ */
