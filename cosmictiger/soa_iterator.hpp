/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cassert>
#include <compare>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

namespace soa {

using std::ptrdiff_t;
using std::size_t;

template <class... Fields>
class Reference {
    template <class...>
    friend class Reference;

    using Value = std::tuple<std::remove_cv_t<Fields>...>;
    std::tuple<Fields&...> fields_;

    template <class Tuple>
    void assign(Tuple&& values) const {
        [&]<size_t... indices>(std::index_sequence<indices...>) {
            ((std::get<indices>(fields_) = std::get<indices>(std::forward<Tuple>(values))), ...);
        }(std::index_sequence_for<Fields...>{});
    }

public:
    explicit Reference(Fields&... fields)
      : fields_(fields...) {}
    Reference(Reference const&) = default;

    template <class... Others>
        requires(sizeof...(Others) == sizeof...(Fields)) &&
        (std::is_convertible_v<Others&, Fields&> && ...)
    Reference(Reference<Others...> const& other)
      : fields_(other.fields_) {}

    Reference(Value const& values)
        requires(std::is_const_v<Fields> && ...)
      : fields_(values) {}

    operator Value() const
        requires(std::is_constructible_v<std::remove_cv_t<Fields>, Fields&> && ...)
    {
        return Value(fields_);
    }

    Reference const& operator=(Reference const& other) const
        requires(std::is_assignable_v<Fields&, Fields&> && ...)
    {
        assign(other.fields_);
        return *this;
    }

    Reference const& operator=(Value const& values) const
        requires(std::is_assignable_v<Fields&, std::remove_cv_t<Fields> const&> && ...)
    {
        assign(values);
        return *this;
    }

    Reference const& operator=(Value&& values) const
        requires(std::is_assignable_v<Fields&, std::remove_cv_t<Fields> &&> && ...)
    {
        assign(std::move(values));
        return *this;
    }

    template <size_t index>
    friend decltype(auto) get(Reference const& row) noexcept {
        return std::get<index>(row.fields_);
    }

    friend void swap(Reference left, Reference right) noexcept(
        (std::is_nothrow_swappable_v<Fields> && ...))
        requires(std::swappable<Fields> && ...)
    {
        [&]<size_t... indices>(std::index_sequence<indices...>) {
            (std::ranges::swap(std::get<indices>(left.fields_), std::get<indices>(right.fields_)),
                ...);
        }(std::index_sequence_for<Fields...>{});
    }
};

}    // namespace soa

// A proxy and an owning row share a read-only reference view, including
// when a field is move-only. These are permitted standard customization points.
namespace std {

template <class... Fields, template <class> class LeftQual, template <class> class RightQual>
    requires is_convertible_v<LeftQual<soa::Reference<Fields...>>,
                 soa::Reference<add_const_t<Fields>...>> &&
    is_convertible_v<RightQual<tuple<remove_cv_t<Fields>...>>,
        soa::Reference<add_const_t<Fields>...>>
struct basic_common_reference<soa::Reference<Fields...>, tuple<remove_cv_t<Fields>...>, LeftQual,
    RightQual> {
    using type = soa::Reference<add_const_t<Fields>...>;
};

template <class... Fields, template <class> class LeftQual, template <class> class RightQual>
    requires is_convertible_v<LeftQual<tuple<remove_cv_t<Fields>...>>,
                 soa::Reference<add_const_t<Fields>...>> &&
    is_convertible_v<RightQual<soa::Reference<Fields...>>, soa::Reference<add_const_t<Fields>...>>
struct basic_common_reference<tuple<remove_cv_t<Fields>...>, soa::Reference<Fields...>, LeftQual,
    RightQual> {
    using type = soa::Reference<add_const_t<Fields>...>;
};

}    // namespace std

namespace soa {

// A C++20 random-access iterator over independently swappable SoA fields.
// Every field must span the same row range, and field ranges must not overlap
// (including passing the same field twice). The arrays outlive the iterators;
// reallocation invalidates them. Bounds are the caller's responsibility.
// Use ranges algorithms: the legacy category intentionally remains input.
template <class... Fields>
class Iterator {
    static_assert(sizeof...(Fields) > 0);

    std::tuple<Fields*...> arrays_{};
    ptrdiff_t index_ = 0;

public:
    using value_type = std::tuple<std::remove_cv_t<Fields>...>;
    using difference_type = ptrdiff_t;
    using reference = Reference<Fields...>;
    using iterator_concept = std::random_access_iterator_tag;
    // Legacy forward iterators require actual references, not proxy values.
    using iterator_category = std::input_iterator_tag;

    Iterator() = default;
    explicit Iterator(Fields*... arrays)
      : arrays_(arrays...) {}

    reference operator*() const {
        return std::apply(
            [this](auto*... arrays) { return reference(arrays[index_]...); }, arrays_);
    }

    reference operator[](difference_type offset) const {
        return *(*this + offset);
    }

    Iterator& operator++() {
        ++index_;
        return *this;
    }
    Iterator& operator--() {
        --index_;
        return *this;
    }
    Iterator operator++(int) {
        auto previous = *this;
        ++*this;
        return previous;
    }
    Iterator operator--(int) {
        auto previous = *this;
        --*this;
        return previous;
    }
    Iterator& operator+=(difference_type offset) {
        index_ += offset;
        return *this;
    }
    Iterator& operator-=(difference_type offset) {
        index_ -= offset;
        return *this;
    }

    friend Iterator operator+(Iterator iterator, difference_type offset) {
        return iterator += offset;
    }
    friend Iterator operator+(difference_type offset, Iterator iterator) {
        return iterator += offset;
    }
    friend Iterator operator-(Iterator iterator, difference_type offset) {
        return iterator -= offset;
    }
    friend difference_type operator-(Iterator const& left, Iterator const& right) {
        assert(left.arrays_ == right.arrays_);
        return left.index_ - right.index_;
    }

    friend bool operator==(Iterator const&, Iterator const&) = default;
    friend auto operator<=>(Iterator const& left, Iterator const& right) {
        assert(left.arrays_ == right.arrays_);
        return left.index_ <=> right.index_;
    }

    friend void iter_swap(Iterator const& left, Iterator const& right) noexcept(
        (std::is_nothrow_swappable_v<Fields> && ...))
        requires(std::swappable<Fields> && ...)
    {
        swap(*left, *right);
    }

    friend value_type iter_move(Iterator const& iterator)
        requires(std::is_constructible_v<std::remove_cv_t<Fields>, Fields &&> && ...)
    {
        return std::apply(
            [&](auto*... arrays) { return value_type(std::move(arrays[iterator.index_])...); },
            iterator.arrays_);
    }
};

}    // namespace soa
