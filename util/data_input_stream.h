/*
 * data_input_stream.h
 *
 *  Created on: Dec 13, 2010
 *      Author: hans
 */

#ifndef DATA_INPUT_STREAM_H_
#define DATA_INPUT_STREAM_H_

#include <iostream>

class DataInputStream{
public:
	DataInputStream(std::istream &in) : in(in){};

	template <typename T>
	DataInputStream& operator>>(T& rhs);

	template <typename T>
	void readVector(T& vec);

private:
	std::istream &in;
};

template <typename T>
DataInputStream& DataInputStream::operator>>(T& rhs){
	in.read(reinterpret_cast<char*>(&rhs), sizeof(T));
	return *this;
}

template <typename T>
void DataInputStream::readVector(T& vec){
	uint32_t size;
	*this >> size;
	if(size>0){
		vec.resize(size);
		in.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(vec[0]));
	}
}

#endif /* DATA_INPUT_STREAM_H_ */
