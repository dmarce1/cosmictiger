/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace hopkins {

// The Appendix G rung-contrast sweep as a resumable state machine. Cells are
// visited in increasing identifier order, pass after pass, until a pass raises
// nobody. A visit of cell i with rung r raises every stencil member j with
// rung_j + 2 < r to max(minimumRung, r) and activates it. A raise earlier in a
// pass changes what a later cell raises, so the order is part of the result.
// Two exact shortcuts skip no-op visits: a cell below lowestRung + 3 can raise
// nobody (the lowest rung is fixed when the sweep starts), and a cell revisited
// with an unchanged rung raises nobody new, because rungs only increase.
//
// The machine needs each visited cell's stencil. step() runs until it reaches a
// visit whose stencil it lacks, then returns the identifiers it needs (that
// cell and the next due cells of the pass, a bounded prefetch window) and
// suspends; provide() hands stencils back and step() resumes at the same cell.
// A provider may return the full stencil (valid at any rung of the cell), or
// only the members that cross the contrast at the cell's current rung, which
// is then valid only while the cell keeps that rung. Members carry their rung
// and activity so the machine learns cells it was not seeded with.
class WakeSweep {
public:
    struct Member {
        uint64_t identifier = 0;
        int rung = 0;
        bool active = false;
    };

    struct Raise {
        uint64_t identifier = 0;
        int rung = 0;
        bool active = false;
    };

    static constexpr int kAnyRung = std::numeric_limits<int>::max();

    // Seed the cells that may visit (any cell may also be learnt from a stencil).
    void seed(uint64_t identifier, int rung, bool active) {
        if (started_) {
            throw std::logic_error("WakeSweep seeded after it started");
        }
        insert(identifier, rung, active);
    }

    void start(int lowestRung, int minimumRung, size_t prefetch) {
        if (started_) {
            throw std::logic_error("WakeSweep started twice");
        }
        started_ = true;
        lowestRung_ = lowestRung;
        minimumRung_ = minimumRung;
        prefetch_ = std::max<size_t>(1, prefetch);
        order_.resize(cells_.size());
        for (size_t index = 0; index < cells_.size(); ++index) {
            order_[index] = index;
        }
        sortOrder();
        position_ = 0;
        changed_ = false;
    }

    // Run until done (returns an empty list) or until stencils are needed.
    std::vector<uint64_t> step() {
        if (!started_) {
            throw std::logic_error("WakeSweep stepped before start");
        }
        while (true) {
            for (; position_ < order_.size(); ++position_) {
                size_t const index = order_[position_];
                if (!due(index)) {
                    continue;
                }
                if (!known(index)) {
                    return requests();
                }
                visit(index);
            }
            if (!changed_) {
                return {};
            }
            changed_ = false;
            position_ = 0;
        }
    }

    // Hand back a stencil for a requested cell. validAtRung is kAnyRung for a
    // full stencil, or the rung at which a contrast-filtered stencil was built.
    void provide(uint64_t identifier, std::vector<Member> const& members, int validAtRung) {
        auto const found = indices_.find(identifier);
        if (found == indices_.end()) {
            throw std::logic_error("WakeSweep received an unrequested stencil");
        }
        auto& cell = cells_[found->second];
        cell.stencil.clear();
        cell.stencil.reserve(members.size());
        for (auto const& member : members) {
            auto const existing = indices_.find(member.identifier);
            size_t index;
            if (existing == indices_.end()) {
                index = insert(member.identifier, member.rung, member.active);
            } else {
                index = existing->second;
            }
            cells_[found->second].stencil.push_back(index);
        }
        cells_[found->second].validAtRung = validAtRung;
    }

    size_t woken() const {
        return woken_;
    }

    // Final state of every cell whose rung or activity the sweep changed, in
    // increasing identifier order.
    std::vector<Raise> raises() const {
        std::vector<Raise> result;
        for (size_t index : sortedIndices()) {
            auto const& cell = cells_[index];
            if (cell.raised) {
                result.push_back({cell.identifier, cell.rung, cell.active});
            }
        }
        return result;
    }

    int rung(uint64_t identifier) const {
        return cells_.at(indices_.at(identifier)).rung;
    }

    bool active(uint64_t identifier) const {
        return cells_.at(indices_.at(identifier)).active;
    }

private:
    struct Cell {
        uint64_t identifier = 0;
        int rung = 0;
        bool active = false;
        bool raised = false;
        int visitedRung = -1;
        int validAtRung = -1;
        std::vector<size_t> stencil;
    };

    std::vector<Cell> cells_;
    std::unordered_map<uint64_t, size_t> indices_;
    std::vector<size_t> order_;
    size_t position_ = 0;
    size_t prefetch_ = 1;
    size_t woken_ = 0;
    int lowestRung_ = 0;
    int minimumRung_ = 0;
    bool changed_ = false;
    bool started_ = false;

    size_t insert(uint64_t identifier, int rung, bool active) {
        auto const [found, inserted] = indices_.emplace(identifier, cells_.size());
        if (!inserted) {
            throw std::logic_error("WakeSweep cell seeded twice");
        }
        Cell cell;
        cell.identifier = identifier;
        cell.rung = rung;
        cell.active = active;
        cells_.push_back(std::move(cell));
        size_t const index = found->second;
        if (started_) {
            // A cell learnt while suspended takes its identifier position in the
            // visit order of the current pass, as if it had been seeded.
            auto const at = std::lower_bound(order_.begin(), order_.end(), identifier,
                [&](size_t other, uint64_t value) { return cells_[other].identifier < value; });
            size_t const offset = size_t(at - order_.begin());
            order_.insert(at, index);
            if (offset <= position_) {
                ++position_;
            }
        }
        return index;
    }

    void sortOrder() {
        std::sort(order_.begin(), order_.end(), [&](size_t left, size_t right) {
            return cells_[left].identifier < cells_[right].identifier;
        });
    }

    std::vector<size_t> sortedIndices() const {
        std::vector<size_t> result(cells_.size());
        for (size_t index = 0; index < cells_.size(); ++index) {
            result[index] = index;
        }
        std::sort(result.begin(), result.end(), [&](size_t left, size_t right) {
            return cells_[left].identifier < cells_[right].identifier;
        });
        return result;
    }

    bool due(size_t index) const {
        auto const& cell = cells_[index];
        return cell.rung >= lowestRung_ + 3 && cell.rung != cell.visitedRung;
    }

    bool known(size_t index) const {
        auto const& cell = cells_[index];
        return cell.validAtRung == kAnyRung || cell.validAtRung == cell.rung;
    }

    // The current cell and the next due cells of this pass whose stencils are
    // unknown, at most prefetch_ of them.
    std::vector<uint64_t> requests() const {
        std::vector<uint64_t> result;
        for (size_t at = position_; at < order_.size() && result.size() < prefetch_; ++at) {
            size_t const index = order_[at];
            if (due(index) && !known(index)) {
                result.push_back(cells_[index].identifier);
            }
        }
        return result;
    }

    void visit(size_t index) {
        int const rung = cells_[index].rung;
        cells_[index].visitedRung = rung;
        for (size_t member : cells_[index].stencil) {
            auto& neighbor = cells_[member];
            if (neighbor.rung + 2 < rung) {
                if (!neighbor.active) {
                    ++woken_;
                }
                neighbor.active = true;
                neighbor.rung = std::max(minimumRung_, rung);
                neighbor.raised = true;
                changed_ = true;
            }
        }
    }
};

}    // namespace hopkins
