/*
 * serialization_impl.hpp
 *
 *  Created on: Dec 11, 2015
 *      Author: dmarce1
 */

#ifndef SERIALIZATION_IMPL_HPP_
#define SERIALIZATION_IMPL_HPP_

#include <cassert>

namespace hpx {

namespace serialization {

template<class Type>
typename std::enable_if<std::is_fundamental<typename std::remove_reference<Type>::type>::value>::type serialize(
		oarchive& arc, Type& data_out, const unsigned) {
	assert(arc.start + sizeof(Type) <= arc.size());
	std::memcpy(&data_out, arc.data() + arc.start, sizeof(Type));
	arc.start += sizeof(Type);
}

template<class Type>
typename std::enable_if<std::is_fundamental<typename std::remove_reference<Type>::type>::value>::type serialize(
		iarchive& arc, Type& data_in, const unsigned) {
	auto new_size = arc.size() + sizeof(Type);
	if (arc.capacity() < new_size) {
		arc.reserve(std::max(2 * arc.capacity(), arc.initial_size));
	}
	void* pointer = arc.data() + arc.size();
	arc.resize(new_size);
	std::memcpy(pointer, &data_in, sizeof(Type));
}

template<class Arc, class Type>
void serialize(Arc& arc, std::vector<Type>& vec, const unsigned) {
	auto size = vec.size();
	auto capacity = vec.capacity();
	arc & size;
	arc & capacity;
	if (capacity != vec.capacity()) {
		vec.reserve(capacity);
	}
	if (size != vec.size()) {
		vec.resize(size);
	}
	for (auto i = vec.begin(); i != vec.end(); ++i) {
		arc & (*i);
	}
}

template<class Arc, class Type, std::size_t Size>
void serialize(Arc& arc, std::array<Type, Size>& a, const unsigned) {
	for (auto i = a.begin(); i != a.end(); ++i) {
		arc & (*i);
	}
}

template<class Type>
void serialize(hpx::detail::ibuffer_type& arc, std::set<Type>& s, const unsigned v) {
	const std::size_t sz = s.size();
	arc << sz;
	for (auto i = s.begin(); i != s.end(); ++i) {
		arc << (*i);
	}
}

template<class Type>
void serialize(hpx::detail::obuffer_type& arc, std::set<Type>& s, const unsigned v) {
	std::size_t sz;
	arc >> sz;
	s.clear();
	Type element;
	for (std::size_t i = 0; i != sz; ++i) {
		arc >> element;
		s.insert(element);
	}
}

template<class Type>
void serialize(hpx::detail::ibuffer_type& arc, std::list<Type>& list, const unsigned v) {
	const std::size_t sz = list.size();
	arc << sz;
	for (auto& ele : list) {
		arc << ele;
	}
}

template<class Type>
void serialize(hpx::detail::obuffer_type& arc, std::list<Type>& list, const unsigned v) {
	std::size_t sz;
	arc >> sz;
	list.clear();
	Type ele;
	for (std::size_t i = 0; i != sz; ++i) {
		arc >> ele;
		list.push_back(ele);
	}
}

template<class Type>
void serialize(hpx::detail::ibuffer_type& arc, std::valarray<Type>& v, const unsigned) {
	const std::size_t sz = v.size();
	arc << sz;
	for (auto& ele : v) {
		arc << ele;
	}
}

template<class Type>
void serialize(hpx::detail::obuffer_type& arc, std::valarray<Type>& v, const unsigned) {
	std::size_t sz;
	arc >> sz;
	v.resize(sz);
	Type ele;
	for (auto& ele : v) {
		arc >> ele;
	}
}

template<class Arc, class Type1, class Type2>
void serialize(Arc& arc, std::pair<Type1, Type2>& a, const unsigned) {
	arc & a.first;
	arc & a.second;
}

template<class Arc>
void serialize(Arc& arc, std::string& vec, const unsigned) {
	auto size = vec.size();
	auto capacity = vec.capacity();
	arc & size;
	arc & capacity;
	if (capacity != vec.capacity()) {
		vec.reserve(capacity);
	}
	if (size != vec.size()) {
		vec.resize(size);
	}
	for (auto i = vec.begin(); i != vec.end(); ++i) {
		arc & (*i);
	}
}

template<class Archive, class Type>
auto serialize_imp(Archive& arc, Type& data, const long v) -> decltype(serialize(arc, data, v), void()) {
	serialize(arc, data, v);
}

template<class Archive, class Type>
auto serialize_imp(Archive& arc, Type& data, const unsigned v) -> decltype(data.serialize(arc, v), void()) {
	data.serialize(arc, v);
}

template<class Arc, class T>
Arc& operator<<(Arc& arc, const T& data) {
	serialize_imp(arc, const_cast<T&>(data), unsigned(0));
	return arc;
}

template<class Arc, class T>
const Arc& operator>>(const Arc& arc, T& data) {
	serialize_imp(const_cast<Arc&>(arc), data, unsigned(0));
	return arc;
}

template<class T>
iarchive& iarchive::operator&(const T& data) {
	*this << data;
	return *this;
}

template<class T>
const oarchive& oarchive::operator&(T& data) const {
	*this >> data;
	return *this;
}

}
}

#endif /* SERIALIZATION_IMPL_HPP_ */
