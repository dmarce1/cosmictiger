/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#ifndef ANALYTIC_HPP_
#define ANALYTIC_HPP_

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/math/fixed.hpp>

std::pair<Vector<double>, Array<Vector<double>, NDIM>> gravityAnalyticCallKernel(
    Vector<Fixed32> const& sinkx, Vector<Fixed32> const& sinky, Vector<Fixed32> const& sinkz);
Pair<double> analyticCompare(int numberOfSamples);

#endif /* ANALYTIC_HPP_ */
