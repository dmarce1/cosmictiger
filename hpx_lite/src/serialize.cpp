/*
 * serialize.cpp
 *
 *  Created on: Dec 11, 2015
 *      Author: dmarce1
 */




#include "hpx/serialize.hpp"


namespace hpx {
namespace serialization {

const std::size_t archive::initial_size = 1024;

oarchive::oarchive(const iarchive& other) :
		archive(other) {
}

oarchive::oarchive(iarchive&& other ) : archive( std::move(other)) {

}

oarchive& oarchive::operator=(const iarchive& other) {
	archive::operator=(*this);
	return *this;
}

oarchive& oarchive::operator=(iarchive&& other ) {
	archive::operator=(std::move(other));
	return *this;
}

iarchive::iarchive(const oarchive& other) :
		archive(other) {
}

iarchive::iarchive(oarchive&& other ) : archive( std::move(other)) {

}

iarchive& iarchive::operator=(const oarchive& other) {
	archive::operator=(other);
	return *this;
}

iarchive& iarchive::operator=(oarchive&& other ) {
	archive::operator=(std::move(other));
	return *this;
}

archive::archive(std::size_t size) :
		buffer_type(size), start(0) {

}

iarchive::iarchive(std::size_t size) :
		archive(size) {

}

oarchive::oarchive(std::size_t size) :
		archive(size) {

}

std::size_t archive::current_position() const {
	return start;
}

void archive::seek_to(std::size_t i) {
	start = i;
}

}
}
