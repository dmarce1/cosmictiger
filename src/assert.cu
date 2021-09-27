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

#include <cosmictiger/assert.hpp>
#include <cosmictiger/safe_io.hpp>

#ifdef __CUDA_ARCH__
__device__ void cosmictiger_cuda_assert( const char* cond_str, bool cond, const char* filename, int line) {
		if (!cond) {
			PRINT( "Condition \"%s\" failed in %s on line %i in CUDA_ARCH\n", cond_str, filename, line);
			__trap();
		}

}
#endif
