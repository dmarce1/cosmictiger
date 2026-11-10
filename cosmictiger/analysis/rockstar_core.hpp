#pragma once

#include <cmath>
#include <limits>

#include <cosmictiger/cuda/cuda_export.hpp>

namespace rockstar::detail {

// Bounding boxes enclose particles, so touching faces and zero-width
// dimensions must remain eligible for a distance calculation.
template <class Box>
CUDA_EXPORT bool boxesOverlap(Box const& a, Box const& b, int dimensions) {
    for (int dim = 0; dim < dimensions; dim++) {
        if (a.end[dim] < b.begin[dim] || b.end[dim] < a.begin[dim]) {
            return false;
        }
    }
    return true;
}

// Closed overlap in the unit periodic box, including point and planar leaves.
template <class Box>
CUDA_EXPORT bool periodicBoxesOverlap(Box const& a, Box const& b, int dimensions) {
    for (int dim = 0; dim < dimensions; ++dim) {
        bool overlap = false;
        for (int shift = -1; shift <= 1; ++shift) {
            overlap |= a.end[dim] + shift >= b.begin[dim] && b.end[dim] >= a.begin[dim] + shift;
        }
        if (!overlap) {
            return false;
        }
    }
    return true;
}

/// Test inclusive coordinate bounds in the requested number of dimensions.
template <class Box, class Point>
CUDA_EXPORT bool boxContains(Box const& box, Point const& point, int dimensions) {
    for (int dim = 0; dim < dimensions; dim++) {
        if (point[dim] < box.begin[dim] || point[dim] > box.end[dim]) {
            return false;
        }
    }
    return true;
}

/// Estimate particle spacing from the box volume over its nondegenerate dimensions.
template <class Box>
float initialLinkingLength(Box const& box, int count, int dimensions) {
    double volume = 1.0;
    int nonzeroDimensions = 0;
    for (int dim = 0; dim < dimensions; dim++) {
        double const width = double(box.end[dim]) - box.begin[dim];
        if (width > 0.0) {
            volume *= width;
            nonzeroDimensions++;
        }
    }
    if (nonzeroDimensions == 0) {
        return 0.0f;
    }
    float const radius = std::pow(volume / count, 1.0 / nonzeroDimensions);
    return radius > 0.0f ? radius : std::numeric_limits<float>::min();
}

/// Convert a positive variance to a dispersion; use unity for a cold or coincident
/// component.
inline float normalizationScale(float variance) {
    // A cold or coincident group contributes zero in this metric component.
    return variance > 0.0f ? std::sqrt(variance) : 1.0f;
}

/// Find the smallest subgroup label within the six-dimensional linking radius, excluding
/// the particle itself.
template <class Particle, class ReadLabel>
CUDA_EXPORT int minimumLinkedLabel(Particle const* particles, int self, int begin, int end,
    float linkingLengthSquared, int minimum, ReadLabel readLabel) {
    for (int other = begin; other < end; other++) {
        if (other == self) {
            continue;
        }
        float distanceSquared = 0.0f;
        for (int dim = 0; dim < 6; dim++) {
            float const delta = particles[self].xValue[dim] - particles[other].xValue[dim];
            distanceSquared += delta * delta;
        }
        if (distanceSquared <= linkingLengthSquared) {
            int const label = readLabel(&particles[other].subgroup);
            if (label < minimum) {
                minimum = label;
            }
        }
    }
    return minimum;
}
}    // namespace rockstar::detail
