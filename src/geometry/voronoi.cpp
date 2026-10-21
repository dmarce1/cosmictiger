/*
 * Copyright (C) 2026 Dominic C. Marcello
 * Distributed under the Boost Software License, Version 1.0.
 * See LICENSE_1_0.txt.
 */

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <Delaunay_psm.h>
#include <boost/multiprecision/cpp_int.hpp>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

#include <cosmictiger/geometry/voronoi.hpp>

namespace voronoi {
namespace detail {

    using WidePoint = std::array<long double, 3>;
    using Edge = std::array<size_t, 2>;
    using GeometricEdge = std::array<Point, 2>;

    struct CollapsedFaces {
        std::map<GeometricEdge, std::vector<Point>> splits;
        std::vector<Point> vertices;
    };

    // Exact expansion predicates need nearest rounding and gradual underflow,
    // including when a calling simulation worker uses a different FP mode.
    class StrictFloatEnvironment {
        std::fenv_t savedEnvironment_{};
        bool captured_ = false;
        bool valid_ = false;
#if defined(__SSE__)
        unsigned savedControl_ = 0;
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
        uint64_t savedControl_ = 0;
#endif

    public:
        StrictFloatEnvironment() noexcept {
#if defined(__SSE__)
            savedControl_ = _mm_getcsr();
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
            asm volatile("mrs %0, fpcr" : "=r"(savedControl_));
#endif
            captured_ = std::feholdexcept(&savedEnvironment_) == 0;
            valid_ = captured_ && std::fesetround(FE_TONEAREST) == 0;
            if (!valid_) {
                return;
            }
#if defined(__SSE__)
            _mm_setcsr(_mm_getcsr() & ~((1U << 15) | (1U << 6)));
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
            uint64_t control;
            asm volatile("mrs %0, fpcr" : "=r"(control));
            control &= ~((uint64_t(1) << 24) | (uint64_t(1) << 19));
            asm volatile("msr fpcr, %0" : : "r"(control));
#endif
            double volatile tiny = std::numeric_limits<double>::denorm_min();
            double volatile twice = tiny + tiny;
            valid_ = std::bit_cast<uint64_t>(static_cast<double>(twice)) == 2;
        }
        ~StrictFloatEnvironment() {
            if (captured_) {
                std::fesetenv(&savedEnvironment_);
            }
#if defined(__SSE__)
            _mm_setcsr(savedControl_);
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
            asm volatile("msr fpcr, %0" : : "r"(savedControl_));
#endif
        }
        StrictFloatEnvironment(StrictFloatEnvironment const&) = delete;
        StrictFloatEnvironment& operator=(StrictFloatEnvironment const&) = delete;
        bool valid() const noexcept {
            return valid_;
        }
    };

    template <typename Value>
    std::array<Value, 3> cross(std::array<Value, 3> const& a, std::array<Value, 3> const& b) {
        return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    }

    template <typename Value>
    Value dot(std::array<Value, 3> const& a, std::array<Value, 3> const& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    template <typename Value = long double>
    std::array<Value, 3> difference(Point const& a, Point const& b) {
        return {Value(a[0]) - Value(b[0]), Value(a[1]) - Value(b[1]), Value(a[2]) - Value(b[2])};
    }

    long double distanceSquared(Point const& a, Point const& b) {
        auto const d = difference(a, b);
        return dot(d, d);
    }

    auto siteKey(Site const& site) {
        return std::tuple(site.id, site.image);
    }

    template <typename Value>
    Point circumcenterWithPrecision(
        Point const& p0, Point const& p1, Point const& p2, Point const& p3) {
        auto const a = difference<Value>(p1, p0);
        auto const b = difference<Value>(p2, p0);
        auto const c = difference<Value>(p3, p0);
        auto const bc = cross(b, c);
        auto const ca = cross(c, a);
        auto const ab = cross(a, b);
        Value const determinant = dot(a, bc);
        if (determinant == 0) {
            throw std::runtime_error("Zero-volume Delaunay tetrahedron in Voronoi geometry");
        }
        Value const aa = dot(a, a);
        Value const bb = dot(b, b);
        Value const cc = dot(c, c);
        Point result;
        for (size_t dim = 0; dim < 3; dim++) {
            Value const offset =
                (aa * bc[dim] + bb * ca[dim] + cc * ab[dim]) / (Value(2) * determinant);
            result[dim] = static_cast<double>(Value(p0[dim]) + offset);
            if (!std::isfinite(result[dim])) {
                throw std::runtime_error("Nonfinite Delaunay circumcenter in Voronoi geometry");
            }
        }
        return result;
    }

    Point circumcenter(Point const& p0, Point const& p1, Point const& p2, Point const& p3) {
        auto const a = difference(p1, p0);
        auto const b = difference(p2, p0);
        auto const c = difference(p3, p0);
        long double const determinant = dot(a, cross(b, c));
        long double const scale = std::sqrt(dot(a, a) * dot(b, b) * dot(c, c));
        // Exact rational construction is reserved for ill-conditioned tetrahedra;
        // the Delaunay topology itself always uses Geogram's filtered exact tests.
        if (std::abs(determinant) <= 128 * std::numeric_limits<long double>::epsilon() * scale) {
            return circumcenterWithPrecision<boost::multiprecision::cpp_rational>(p0, p1, p2, p3);
        }
        return circumcenterWithPrecision<long double>(p0, p1, p2, p3);
    }

    bool spansVolume(std::vector<Point> const& positions) {
        if (positions.size() < 4) {
            return false;
        }
        size_t third = positions.size();
        for (size_t i = 2; i < positions.size() && third == positions.size(); i++) {
            for (size_t dim = 0; dim < 3; dim++) {
                size_t const next = (dim + 1) % 3;
                double const a[2] = {positions[0][dim], positions[0][next]};
                double const b[2] = {positions[1][dim], positions[1][next]};
                double const c[2] = {positions[i][dim], positions[i][next]};
                if (GEO::PCK::orient_2d(a, b, c) != GEO::ZERO) {
                    third = i;
                    break;
                }
            }
        }
        if (third == positions.size()) {
            return false;
        }
        for (size_t i = third + 1; i < positions.size(); i++) {
            if (GEO::PCK::orient_3d(positions[0].data(), positions[1].data(),
                    positions[third].data(), positions[i].data()) != GEO::ZERO) {
                return true;
            }
        }
        return false;
    }

    bool onSegment(Point const& point, Point const& first, Point const& second) {
        for (size_t dim = 0; dim < 3; dim++) {
            if (point[dim] < std::min(first[dim], second[dim]) ||
                point[dim] > std::max(first[dim], second[dim])) {
                return false;
            }
        }
        for (size_t dim = 0; dim < 3; dim++) {
            size_t const next = (dim + 1) % 3;
            double const a[2] = {first[dim], first[next]};
            double const b[2] = {second[dim], second[next]};
            double const c[2] = {point[dim], point[next]};
            if (GEO::PCK::orient_2d(a, b, c) != GEO::ZERO) {
                return false;
            }
        }
        return true;
    }

    void recordCollapsedFace(std::vector<Point> const& vertices, CollapsedFaces& collapsed) {
        collapsed.vertices.insert(collapsed.vertices.end(), vertices.begin(), vertices.end());
        for (size_t i = 0; i < vertices.size(); i++) {
            GeometricEdge edge{vertices[i], vertices[(i + 1) % vertices.size()]};
            if (edge[1] < edge[0]) {
                std::swap(edge[0], edge[1]);
            }
            collapsed.splits.try_emplace(edge);
        }
    }

    void removeSpurs(std::vector<Point>& vertices) {
        bool changed = true;
        while (changed && vertices.size() >= 3) {
            changed = false;
            for (size_t i = 0; i < vertices.size(); i++) {
                size_t const next = (i + 1) % vertices.size();
                size_t const previous = (i + vertices.size() - 1) % vertices.size();
                if (vertices[i] == vertices[next] || vertices[previous] == vertices[next]) {
                    vertices.erase(vertices.begin() + i);
                    changed = true;
                    break;
                }
            }
        }
    }

    void splitCollapsedEdges(std::vector<Cell>& cells, CollapsedFaces& collapsed) {
        if (collapsed.splits.empty()) {
            return;
        }
        auto& vertices = collapsed.vertices;
        std::sort(vertices.begin(), vertices.end());
        vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());
        for (auto& [edge, splits] : collapsed.splits) {
            auto const end = std::lower_bound(vertices.begin(), vertices.end(), edge[1]);
            for (auto point = std::upper_bound(vertices.begin(), vertices.end(), edge[0]);
                point != end; ++point) {
                if (onSegment(*point, edge[0], edge[1])) {
                    splits.push_back(*point);
                }
            }
        }
        // A dual polygon may collapse to a line when its circumcenters are
        // rounded to doubles. Its remaining incident faces must subdivide that
        // line identically. Use exact collinearity, never a distance tolerance,
        // so genuinely small faces and edges remain part of the mesh.
        for (auto& cell : cells) {
            for (auto& face : cell.faces) {
                std::vector<Point> split;
                for (size_t i = 0; i < face.vertices.size(); i++) {
                    Point const& first = face.vertices[i];
                    Point const& second = face.vertices[(i + 1) % face.vertices.size()];
                    split.push_back(first);
                    bool const reversed = second < first;
                    GeometricEdge const edge =
                        reversed ? GeometricEdge{second, first} : GeometricEdge{first, second};
                    auto const entry = collapsed.splits.find(edge);
                    if (entry != collapsed.splits.end()) {
                        auto const& points = entry->second;
                        if (reversed) {
                            split.insert(split.end(), points.rbegin(), points.rend());
                        } else {
                            split.insert(split.end(), points.begin(), points.end());
                        }
                    }
                }
                removeSpurs(split);
                face.vertices = std::move(split);
            }
            std::erase_if(cell.faces, [](Face const& face) { return face.vertices.size() < 3; });
        }
    }

    Face makeFace(Site const& inside, Site const& outside, std::vector<Point> vertices,
        bool reverse, CollapsedFaces& collapsed) {
        Face face;
        face.neighbor = outside;
        if (vertices.size() < 3) {
            return face;
        }
        auto normal = difference(outside.position, inside.position);
        long double const normalLength = std::sqrt(dot(normal, normal));
        for (auto& component : normal) {
            component /= normalLength;
        }
        // The Delaunay edge ring supplies the polygon order. Removing only
        // identical vertices preserves tiny genuine faces and keeps neighboring
        // polygon edges consistent at cospherical degeneracies.
        std::vector<Point> distinct;
        for (auto const& vertex : vertices) {
            if (std::find(distinct.begin(), distinct.end(), vertex) == distinct.end()) {
                distinct.push_back(vertex);
            }
        }
        if (distinct.size() < 3) {
            return face;
        }
        auto const [first, last] = std::minmax_element(distinct.begin(), distinct.end());
        if (std::all_of(distinct.begin(), distinct.end(),
                [&](Point const& point) { return onSegment(point, *first, *last); })) {
            recordCollapsedFace(distinct, collapsed);
            return face;
        }
        WidePoint center = {};
        for (auto const& vertex : distinct) {
            auto const relative = difference(vertex, inside.position);
            for (size_t dim = 0; dim < 3; dim++) {
                center[dim] += relative[dim] / distinct.size();
            }
        }
        WidePoint areaVector = {};
        WidePoint moment = {};
        long double area = 0;
        for (size_t i = 0; i < distinct.size(); i++) {
            auto a = difference(distinct[i], inside.position);
            auto b = difference(distinct[(i + 1) % distinct.size()], inside.position);
            for (size_t dim = 0; dim < 3; dim++) {
                a[dim] -= center[dim];
                b[dim] -= center[dim];
            }
            auto const triangle = cross(a, b);
            long double const triangleArea = dot(triangle, normal) / 2;
            area += triangleArea;
            for (size_t dim = 0; dim < 3; dim++) {
                areaVector[dim] += triangle[dim] / 2;
                moment[dim] += triangleArea * (center[dim] + (a[dim] + b[dim]) / 3);
            }
        }
        for (size_t dim = 0; dim < 3; dim++) {
            face.areaVector[dim] = double(areaVector[dim]);
            // Rounded, nearly collapsed polygons can have cancelling signed
            // areas without being collinear. Keep their exact ring connectivity;
            // deleting them would open the shell. Their vertex mean remains a
            // finite reference point when the area-weighted centroid is undefined.
            face.centroid[dim] = double(static_cast<long double>(inside.position[dim]) +
                (area == 0 ? center[dim] : moment[dim] / area));
        }
        if (reverse) {
            std::reverse(distinct.begin(), distinct.end());
            for (double& component : face.areaVector) {
                component = -component;
            }
        }
        // Give each polygon a deterministic starting vertex as well as orientation.
        std::rotate(
            distinct.begin(), std::min_element(distinct.begin(), distinct.end()), distinct.end());
        face.vertices = std::move(distinct);
        return face;
    }

    void finishCell(Cell& cell) {
        long double volume = 0;
        WidePoint moment = {};
        WidePoint closure = {};
        long double totalArea = 0;
        for (auto const& face : cell.faces) {
            auto const a = difference(face.centroid, cell.site.position);
            long double areaSquared = 0;
            for (size_t dim = 0; dim < 3; dim++) {
                closure[dim] += face.areaVector[dim];
                areaSquared +=
                    static_cast<long double>(face.areaVector[dim]) * face.areaVector[dim];
            }
            totalArea += std::sqrt(areaSquared);
            for (size_t i = 0; i < face.vertices.size(); i++) {
                auto const b = difference(face.vertices[i], cell.site.position);
                auto const c =
                    difference(face.vertices[(i + 1) % face.vertices.size()], cell.site.position);
                long double const tetrahedronVolume = dot(a, cross(b, c)) / 6;
                volume += tetrahedronVolume;
                for (size_t dim = 0; dim < 3; dim++) {
                    moment[dim] += tetrahedronVolume * (a[dim] + b[dim] + c[dim]) / 4;
                }
            }
        }
        long double const error = std::sqrt(dot(closure, closure));
        if (!(volume > 0) || !std::isfinite(volume) ||
            error > 2048 * std::numeric_limits<double>::epsilon() * totalArea) {
            throw std::runtime_error("Nonpositive or nonclosing Voronoi cell geometry");
        }
        cell.volume = double(volume);
        for (size_t dim = 0; dim < 3; dim++) {
            cell.centroid[dim] =
                double(static_cast<long double>(cell.site.position[dim]) + moment[dim] / volume);
        }
    }

}    // namespace detail

std::vector<Cell> buildCells(std::span<Site const> sites, std::span<size_t const> ownedIndices) {
    if (sites.size() > std::numeric_limits<GEO::index_t>::max()) {
        throw std::runtime_error("Voronoi site count exceeds backend index range");
    }
    std::vector<Cell> cells;
    cells.reserve(ownedIndices.size());
    std::vector<size_t> selected(sites.size(), ownedIndices.size());
    for (size_t const input : ownedIndices) {
        if (input >= sites.size() || selected[input] != ownedIndices.size()) {
            throw std::runtime_error("Invalid or repeated owned Voronoi site index");
        }
        selected[input] = cells.size();
        Cell cell;
        cell.site = sites[input];
        cell.inputIndex = input;
        cell.searchRadius = std::numeric_limits<double>::infinity();
        cells.push_back(std::move(cell));
    }
    if (cells.empty()) {
        return cells;
    }
    std::vector<size_t> order(sites.size());
    std::iota(order.begin(), order.end(), size_t(0));
    for (auto const& site : sites) {
        for (double const coordinate : site.position) {
            if (!std::isfinite(coordinate)) {
                throw std::runtime_error("Nonfinite Voronoi generator coordinate");
            }
        }
    }
    std::sort(order.begin(), order.end(),
        [&](size_t const a, size_t const b) { return sites[a].position < sites[b].position; });
    for (size_t i = 1; i < order.size(); i++) {
        if (sites[order[i - 1]].position == sites[order[i]].position) {
            throw std::runtime_error("Coincident Voronoi generators must be resolved explicitly");
        }
    }
    std::sort(order.begin(), order.end(), [&](size_t const a, size_t const b) {
        return detail::siteKey(sites[a]) < detail::siteKey(sites[b]);
    });
    for (size_t i = 1; i < order.size(); i++) {
        if (detail::siteKey(sites[order[i - 1]]) == detail::siteKey(sites[order[i]])) {
            throw std::runtime_error("Duplicate Voronoi generator ID and periodic image");
        }
    }
    std::vector<Point> positions;
    positions.reserve(order.size());
    std::vector<size_t> chosen;
    chosen.reserve(order.size());
    for (size_t const input : order) {
        positions.push_back(sites[input].position);
        chosen.push_back(selected[input]);
    }

    // Geogram has global initialization/PRNG state. Serialize calls within a
    // process and cap its workers explicitly; MPI/HPX localities run independently.
    static std::mutex backendMutex;
    std::lock_guard lock(backendMutex);
    detail::StrictFloatEnvironment const environment;
    if (!environment.valid()) {
        throw std::runtime_error(
            "Cannot establish strict floating-point mode for Voronoi geometry");
    }
    GEO::initialize(GEO::GEOGRAM_INSTALL_NONE);
    GEO::Process::set_max_threads(1);
    GEO::Logger::instance()->set_quiet(true);
    GEO::Numeric::random_reset(0);
    if (!detail::spansVolume(positions)) {
        return cells;
    }
    GEO::Delaunay_var triangulation = GEO::Delaunay::create(3, "BDEL");
    if (!triangulation) {
        throw std::runtime_error("Cannot create Geogram Delaunay backend");
    }
    triangulation->set_keeps_infinite(true);
    triangulation->set_vertices(GEO::index_t(positions.size()), positions[0].data());
    std::vector<bool> seen(sites.size(), false);
    std::vector<bool> hull(sites.size(), false);
    using Tetrahedron = std::array<GEO::index_t, 4>;
    std::vector<Tetrahedron> tetrahedra(triangulation->nb_cells());
    std::vector<Point> centers(tetrahedra.size());
    std::vector<bool> neededTetrahedron(tetrahedra.size(), false);
    std::vector<bool> needsExactCenter(tetrahedra.size(), false);
    std::vector<size_t> groups(tetrahedra.size());
    std::iota(groups.begin(), groups.end(), size_t(0));
    std::map<detail::Edge, std::vector<GEO::index_t>> edgeTetrahedra;
    for (auto& cell : cells) {
        cell.searchRadius = 0;
    }
    for (GEO::index_t t = 0; t < triangulation->nb_cells(); t++) {
        std::array<GEO::index_t, 4> vertices;
        bool infinite = false;
        for (GEO::index_t j = 0; j < 4; j++) {
            vertices[j] = triangulation->cell_vertex(t, j);
            infinite = infinite || vertices[j] == GEO::NO_INDEX;
            if (vertices[j] != GEO::NO_INDEX) {
                if (vertices[j] >= sites.size()) {
                    throw std::runtime_error("Invalid Delaunay vertex index");
                }
                seen[vertices[j]] = true;
            }
        }
        if (infinite) {
            for (auto const vertex : vertices) {
                if (vertex != GEO::NO_INDEX) {
                    hull[vertex] = true;
                }
            }
            continue;
        }
        bool needed = false;
        for (auto const vertex : vertices) {
            needed = needed || chosen[vertex] < cells.size();
        }
        if (!needed) {
            continue;
        }
        std::sort(vertices.begin(), vertices.end());
        tetrahedra[t] = vertices;
        neededTetrahedron[t] = true;
        centers[t] = detail::circumcenter(positions[vertices[0]], positions[vertices[1]],
            positions[vertices[2]], positions[vertices[3]]);
    }
    auto group = [&](size_t t) {
        while (groups[t] != t) {
            groups[t] = groups[groups[t]];
            t = groups[t];
        }
        return t;
    };
    // Symbolic perturbation triangulates exactly cospherical configurations.
    // Unite those tetrahedra using an unperturbed exact predicate, then construct
    // their shared dual vertex once. Face-local tolerance merging would crack
    // the shell, and would erase real small faces in nearly degenerate inputs.
    for (GEO::index_t t = 0; t < tetrahedra.size(); t++) {
        if (!neededTetrahedron[t]) {
            continue;
        }
        auto const& vertices = tetrahedra[t];
        for (GEO::index_t face = 0; face < 4; face++) {
            auto const adjacent = triangulation->cell_adjacent(t, face);
            if (adjacent == GEO::NO_INDEX || triangulation->cell_is_infinite(adjacent) ||
                (neededTetrahedron[adjacent] && adjacent < t)) {
                continue;
            }
            for (GEO::index_t local = 0; local < 4; local++) {
                auto const vertex = triangulation->cell_vertex(adjacent, local);
                if (std::find(vertices.begin(), vertices.end(), vertex) == vertices.end()) {
                    if (GEO::PCK::side4_3d(positions[vertices[0]].data(),
                            positions[vertices[1]].data(), positions[vertices[2]].data(),
                            positions[vertices[3]].data(), positions[vertex].data()) == GEO::ZERO) {
                        needsExactCenter[t] = true;
                        if (neededTetrahedron[adjacent]) {
                            needsExactCenter[adjacent] = true;
                            auto const a = group(t);
                            auto const b = group(adjacent);
                            groups[std::max(a, b)] = std::min(a, b);
                        }
                    }
                    break;
                }
            }
        }
    }
    for (size_t t = 0; t < groups.size(); t++) {
        if (group(t) == t && needsExactCenter[t]) {
            auto const& vertices = tetrahedra[t];
            centers[t] = detail::circumcenterWithPrecision<boost::multiprecision::cpp_rational>(
                positions[vertices[0]], positions[vertices[1]], positions[vertices[2]],
                positions[vertices[3]]);
        }
    }
    for (GEO::index_t t = 0; t < tetrahedra.size(); t++) {
        if (!neededTetrahedron[t]) {
            continue;
        }
        auto const& vertices = tetrahedra[t];
        centers[t] = centers[group(t)];
        auto const& center = centers[t];
        long double circumradius = 0;
        for (auto const vertex : vertices) {
            circumradius = std::max(
                circumradius, std::sqrt(detail::distanceSquared(center, positions[vertex])));
        }
        for (size_t a = 0; a < 4; a++) {
            size_t const index = vertices[a];
            if (chosen[index] < cells.size()) {
                auto& cell = cells[chosen[index]];
                cell.maxRadius = std::max(cell.maxRadius,
                    double(std::sqrt(detail::distanceSquared(center, positions[index]))));
                cell.searchRadius = std::max(cell.searchRadius,
                    double(2 * circumradius * (1 + 64 * std::numeric_limits<double>::epsilon())));
            }
            for (size_t b = a + 1; b < 4; b++) {
                size_t const other = vertices[b];
                if (chosen[index] < cells.size() || chosen[other] < cells.size()) {
                    detail::Edge const edge = {std::min(index, other), std::max(index, other)};
                    edgeTetrahedra[edge].push_back(t);
                }
            }
        }
    }
    for (size_t i = 0; i < positions.size(); i++) {
        if (!seen[i]) {
            throw std::runtime_error("Delaunay backend omitted a unique generator");
        }
        if (chosen[i] < cells.size()) {
            auto& cell = cells[chosen[i]];
            cell.complete = !hull[i];
            if (!cell.complete) {
                cell.searchRadius = std::numeric_limits<double>::infinity();
            }
        }
    }
    detail::CollapsedFaces collapsed;
    for (auto const& [edge, incident] : edgeTetrahedra) {
        size_t const a = chosen[edge[0]];
        size_t const b = chosen[edge[1]];
        bool const needA = a < cells.size() && cells[a].complete;
        bool const needB = b < cells.size() && cells[b].complete;
        if (!needA && !needB) {
            continue;
        }
        std::vector<Point> vertices;
        bool reverse = false;
        auto current = incident[0];
        auto previous = GEO::NO_INDEX;
        do {
            vertices.push_back(centers[current]);
            auto next = GEO::NO_INDEX;
            for (GEO::index_t local = 0; local < 4; local++) {
                auto const vertex = triangulation->cell_vertex(current, local);
                if (vertex != edge[0] && vertex != edge[1]) {
                    auto const adjacent = triangulation->cell_adjacent(current, local);
                    if (adjacent != previous) {
                        if (previous == GEO::NO_INDEX) {
                            for (auto const other : tetrahedra[current]) {
                                if (other != edge[0] && other != edge[1] && other != vertex) {
                                    // Circumcenters can round a very small dual face
                                    // through zero area. Its winding comes from the
                                    // exact primal tetrahedron, not that rounded area.
                                    auto const orientation = GEO::PCK::orient_3d(
                                        positions[edge[0]].data(), positions[edge[1]].data(),
                                        positions[vertex].data(), positions[other].data());
                                    if (orientation == GEO::ZERO) {
                                        throw std::runtime_error(
                                            "Zero-volume Delaunay tetrahedron in Voronoi ring");
                                    }
                                    reverse = orientation == GEO::NEGATIVE;
                                    break;
                                }
                            }
                        }
                        next = adjacent;
                        break;
                    }
                }
            }
            if (next == GEO::NO_INDEX || !neededTetrahedron[next] ||
                vertices.size() > incident.size()) {
                throw std::runtime_error("Invalid Delaunay edge ring in Voronoi geometry");
            }
            previous = current;
            current = next;
        } while (current != incident[0]);
        if (vertices.size() != incident.size()) {
            throw std::runtime_error("Disconnected Delaunay edge ring in Voronoi geometry");
        }
        Face face = detail::makeFace(
            sites[order[edge[0]]], sites[order[edge[1]]], std::move(vertices), reverse, collapsed);
        if (face.vertices.empty()) {
            continue;
        }
        if (needA) {
            cells[a].faces.push_back(face);
        }
        if (needB) {
            face.neighbor = sites[order[edge[0]]];
            std::reverse(face.vertices.begin() + 1, face.vertices.end());
            for (double& component : face.areaVector) {
                component = -component;
            }
            cells[b].faces.push_back(std::move(face));
        }
    }
    detail::splitCollapsedEdges(cells, collapsed);
    for (auto& cell : cells) {
        if (cell.complete) {
            detail::finishCell(cell);
            cell.searchRadius =
                std::nextafter(cell.searchRadius, std::numeric_limits<double>::infinity());
        }
    }
    return cells;
}

}    // namespace voronoi
