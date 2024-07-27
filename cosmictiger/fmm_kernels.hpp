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
#pragma once

#include <cosmictiger/defs.hpp>
#include <sfmm.hpp>

template<class T>
using multipole = sfmm::multipole<T, ORDER>;

template<class T>
using expansion = sfmm::expansion<T, ORDER>;

template<class T>
using expansion2 = sfmm::expansion<T, 1>;

template<class T>
using force_type = sfmm::force_type<T>;

template<class T>
using vec3 = sfmm::vec3<T>;

#define EXPANSION_SIZE (expansion<float>::size())
#define MULTIPOLE_SIZE (multipole<float>::size())
