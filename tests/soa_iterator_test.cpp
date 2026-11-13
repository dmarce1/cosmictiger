/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cosmictiger/soa_iterator.hpp>

namespace detail {

using std::size_t;

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

using NumericIterator = soa::Iterator<int, double, int>;
using MoveOnlyIterator = soa::Iterator<int, std::unique_ptr<int>>;
using ConstIterator = soa::Iterator<int const, double const>;
static_assert(std::random_access_iterator<NumericIterator>);
static_assert(std::permutable<NumericIterator>);
static_assert(std::random_access_iterator<MoveOnlyIterator>);
static_assert(std::permutable<MoveOnlyIterator>);
static_assert(std::random_access_iterator<ConstIterator>);
static_assert(!std::permutable<ConstIterator>);

bool belowSplit(auto const& row) {
    using std::get;
    return get<0>(row) < 1;
}

void checkIterator() {
    size_t count = 0;
    for (int n = 0; n <= 11; ++n) {
        for (unsigned mask = 0; mask < (1u << n); ++mask) {
            for (int offset : {0, 3}) {
                std::vector<int> position(n + offset + 2, 5);
                std::vector<double> mass(position.size());
                std::vector<int> identity(position.size());
                std::iota(identity.begin(), identity.end(), 0);
                int expected = offset;
                for (size_t i = 0; i < position.size(); ++i) {
                    mass[i] = 10.5 + i;
                }
                for (int i = 0; i < n; ++i) {
                    position[offset + i] = int((mask >> i) & 1u);
                    expected += position[offset + i] == 0;
                }
                auto const original = position;
                auto const first = soa::Iterator(position.data(), mass.data(), identity.data());
                auto const tail = std::ranges::partition(first + offset, first + offset + n,
                    [](auto const& row) { return belowSplit(row); });
                require(tail.begin() - first == expected, "Incorrect partition boundary");
                require(tail.end() == first + offset + n, "Incorrect partition end");
                auto sortedIds = identity;
                std::sort(sortedIds.begin(), sortedIds.end());
                for (size_t i = 0; i < position.size(); ++i) {
                    require(sortedIds[i] == int(i), "Partition lost or duplicated a row");
                    require(position[i] == original[identity[i]],
                        "Partition detached position from row");
                    require(mass[i] == 10.5 + identity[i], "Partition detached mass from row");
                    if (int(i) < offset || int(i) >= offset + n) {
                        require(identity[i] == int(i), "Partition modified outside the subrange");
                    } else {
                        require(
                            (position[i] < 1) == (int(i) < expected), "Partition misplaced a row");
                    }
                }
                ++count;
            }
        }
    }
    {
        int position[] = {1, 0, 1, 0};
        std::unique_ptr<int> data[] = {std::make_unique<int>(10), std::make_unique<int>(0),
            std::make_unique<int>(10), std::make_unique<int>(0)};
        auto const first = soa::Iterator(position, data);
        auto const tail = std::ranges::partition(
            first, first + 4, [](auto const& row) { return belowSplit(row); });
        require(tail.begin() - first == 2, "Incorrect move-only partition boundary");
        for (int i = 0; i < 4; ++i) {
            require(data[i] && *data[i] == position[i] * 10, "Partition detached move-only field");
        }
        auto snapshot = std::ranges::iter_move(first);
        require(!data[0] && std::get<1>(snapshot), "iter_move did not transfer ownership");
        *first = std::move(snapshot);
        require(data[0] && !std::get<1>(snapshot), "Proxy assignment did not restore ownership");
        soa::Reference<int const, std::unique_ptr<int> const> common = *first;
        using std::get;
        require(get<1>(common).get() == data[0].get(), "Common reference lost row alias");
    }
    {
        int position[] = {3, 1, 2};
        std::string data[] = {"three", "one", "two"};
        auto first = soa::Iterator(position, data);
        std::ranges::sort(first, first + 3, [](auto const& left, auto const& right) {
            using std::get;
            return get<0>(left) < get<0>(right);
        });
        require(data[0] == "one" && data[1] == "two" && data[2] == "three",
            "Sort detached string field");
        *first = *(first + 1);
        require(position[0] == 2 && data[0] == "two", "Proxy copy assignment failed");
        *first = std::tuple{8, std::string("eight")};
        require(position[0] == 8 && data[0] == "eight", "Owning row assignment failed");
        auto cursor = first;
        require(cursor++ == first && cursor == first + 1, "Postincrement failed");
        require(cursor-- == first + 1 && cursor == first, "Postdecrement failed");
        require(2 + first == first + 2, "Iterator addition failed");
        require(first + 1 > first && first <= first + 1, "Iterator comparison failed");
        using std::get;
        require(get<0>(first[1]) == 2, "Iterator indexing failed");
    }
    {
        NumericIterator first;
        auto tail =
            std::ranges::partition(first, first, [](auto const& row) { return belowSplit(row); });
        require(tail.begin() == first, "Default iterator empty partition failed");
    }
    std::cout << "PASS " << count
              << " exhaustive partition cases; iterator concepts, move-only fields, moves, "
                 "assignments, arithmetic and numeric/string sort.\n";
}

}    // namespace detail

int main() {
    try {
        detail::checkIterator();
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
