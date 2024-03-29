/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/


#ifndef ANALYTIC_HPP_
#define ANALYTIC_HPP_


#include <cosmictiger/containers.hpp>
#include <cosmictiger/fixed.hpp>

std::pair<vector<double>, array<vector<double>, NDIM>> gravity_analytic_call_kernel(const vector<fixed32>& sinkx,
		const vector<fixed32>& sinky, const vector<fixed32>& sinkz);
pair<double> analytic_compare(int Nsamples);



#endif /* ANALYTIC_HPP_ */
