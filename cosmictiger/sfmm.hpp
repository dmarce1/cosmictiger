#pragma once

#include <cosmictiger/defs.hpp>

#include <sfmm.hpp>

template<class T>
using multipole = sfmm::multipole<T, ORDER>;

template<class T>
using pm_multipole = sfmm::multipole<T, ORDER + 2>;

template<class T>
using expansion = sfmm::expansion<T, ORDER>;

template<class T>
using force_type = sfmm::force_type<T>;

using simd_fixed = sfmm::simd_fixed32;

static constexpr int EXPANSION_SIZE = expansion<float>::size();

static constexpr int MULTIPOLE_SIZE = multipole<float>::size();

namespace hpx {
namespace serialization {

template<class T, class A>
void serialize(A & arc, multipole<T>& m, unsigned int) {
	for (int i = 0; i < multipole<T>::size(); i++) {
		arc & m[i];
	}
	T scale = m.scale();
	arc & scale;
	m.rescale(scale);
}

template<class T, class A>
void serialize(A & arc, expansion<T>& m, unsigned int) {
	for (int i = 0; i < expansion<T>::size(); i++) {
		arc & m[i];
	}
	T scale = m.scale();
	arc & scale;
	m.rescale(scale);
}

}
}
