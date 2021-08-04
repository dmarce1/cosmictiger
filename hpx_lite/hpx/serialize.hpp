/*
 * serialize.hpp
 *
 *  Created on: Dec 11, 2015
 *      Author: dmarce1
 */

#ifndef SERIALIZE_HPP_
#define SERIALIZE_HPP_

#include <array>
#include <cstdint>
#include <cstring>
#include <list>
#include <set>
#include <valarray>
#include <vector>

#define HPX_SERIALIZATION_SPLIT_MEMBER() \
		void serialize(hpx::detail::ibuffer_type& arc, const unsigned v) {                           \
			save(arc, v);                                                                      \
		}                                                                                      \
		void serialize(hpx::detail::obuffer_type& arc, const unsigned v) {                           \
			typedef typename std::remove_const<decltype(this)>::type this_type;                \
			const_cast<this_type>(this)->load(const_cast<hpx::detail::obuffer_type&>(arc), v); \
		}

namespace hpx {
namespace serialization {

class archive;
class oarchive;
class iarchive;
}

namespace detail {
typedef hpx::serialization::oarchive obuffer_type;
typedef hpx::serialization::iarchive ibuffer_type;

}

namespace serialization {

class archive: public std::vector<std::uint8_t> {
protected:
	typedef std::vector<std::uint8_t> buffer_type;
	static const std::size_t initial_size;
	mutable std::size_t start;
public:
	archive(std::size_t = 0);
	~archive() = default;
	archive( const archive& ) = default;
	archive( archive&& ) = default;
	archive& operator=( const archive& ) = default;
	archive& operator=( archive&& ) = default;
	std::size_t current_position() const;
	void seek_to(std::size_t);
};

class iarchive: public archive {
public:
	iarchive(std::size_t = 0);
	~iarchive() = default;
	iarchive( const iarchive& ) = default;
	iarchive( iarchive&& ) = default;
	iarchive& operator=( const iarchive& ) = default;
	iarchive& operator=( iarchive&& ) = default;
	iarchive( const oarchive& );
	iarchive( oarchive&& );
	iarchive& operator=( const oarchive& );
	iarchive& operator=( oarchive&& );

	template<class Type>
	friend typename std::enable_if<std::is_fundamental<typename std::remove_reference<Type>::type>::value>::type
	serialize(iarchive& arc,Type& data_in, const unsigned);

	template<class T>
	iarchive& operator&(const T& data);
};

class oarchive: public archive {
public:
	oarchive(std::size_t = 0);
	oarchive(const oarchive&) = default;
	oarchive( oarchive&& ) = default;
	oarchive( const iarchive& );
	oarchive( iarchive&& );
	~oarchive() = default;
	oarchive& operator=( const oarchive& ) = default;
	oarchive& operator=( oarchive&& ) = default;
	oarchive& operator=( const iarchive& );
	oarchive& operator=( iarchive&& );

	template<class T>
	const oarchive& operator&(T& data) const;

	template<class Type>
	friend typename std::enable_if<std::is_fundamental<typename std::remove_reference<Type>::type>::value >::type
	serialize(oarchive&, Type& data_out, const unsigned );

};


template<class Type>
typename std::enable_if<std::is_fundamental<typename std::remove_reference<Type>::type>::value>::type serialize(
		oarchive& arc, Type& data_out, const unsigned);

template<class Type>
typename std::enable_if<std::is_fundamental<typename std::remove_reference<Type>::type>::value>::type serialize(
		iarchive& arc, Type& data_in, const unsigned);

template<class Arc, class Type>
void serialize(Arc& arc, std::vector<Type>& vec, const unsigned);

template<class Arc, class Type, std::size_t Size>
void serialize(Arc& arc, std::array<Type, Size>& a, const unsigned);

template<class Type>
void serialize(hpx::detail::ibuffer_type& arc, std::set<Type>& s, const unsigned v);

template<class Type>
void serialize(hpx::detail::obuffer_type& arc, std::set<Type>& s, const unsigned v);

template<class Type>
void serialize(hpx::detail::ibuffer_type& arc, std::list<Type>& list, const unsigned v);

template<class Type>
void serialize(hpx::detail::obuffer_type& arc, std::list<Type>& list, const unsigned v);

template<class Type>
void serialize(hpx::detail::ibuffer_type& arc, std::valarray<Type>& list, const unsigned v);

template<class Type>
void serialize(hpx::detail::obuffer_type& arc, std::valarray<Type>& list, const unsigned v);

template<class Arc, class Type1, class Type2>
void serialize(Arc& arc, std::pair<Type1, Type2>& a, const unsigned);

template<class Arc>
void serialize(Arc& arc, std::string& vec, const unsigned);

template<class Archive, class Type>
auto serialize_imp(Archive& arc, Type& data, const long v) -> decltype(serialize(arc, data, v), void());

template<class Archive, class Type>
auto serialize_imp(Archive& arc, Type& data, const unsigned v) -> decltype(data.serialize(arc, v), void());

template<class Arc, class T>
Arc& operator<<(Arc& arc, const T& data);

template<class Arc, class T>
const Arc& operator>>(const Arc& arc, T& data);


}

}

#include "hpx/detail/serialize_impl.hpp"

#endif /* SERIALIZE_HPP_ */
