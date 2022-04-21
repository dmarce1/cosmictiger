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


#ifndef DRIFT_HPP_
#define DRIFT_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/particles.hpp>


struct drift_return {
	double kin;
	double therm;
	double vol;
	double momx;
	double momy;
	double momz;
	double cold_mass;
	double sph_mass;
	double flops;
	part_int nmapped;
	template<class Arc>
	void serialize(Arc&& a, unsigned) {
		a & vol;
		a & therm;
		a & kin;
		a & momx;
		a & momy;
		a & momz;
		a & nmapped;
		a & flops;
	}
};

drift_return drift(double scale, double dt, double tau0, double tau1, double tau_max);


#endif /* DRIFT_HPP_ */
