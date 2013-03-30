/*
 * data_output_stream.h
 *
 *  Created on: Dec 13, 2010
 *      Author: hans
 */

#ifndef DATA_OUTPUT_STREAM_H_
#define DATA_OUTPUT_STREAM_H_

class DataOutputStream{
public:
	DataOutputStream(std::ostream &out) : out(out){};

	template <typename T>
	DataOutputStream& operator<<(const T& rhs);

	template <typename T>
	void writeVector(const T& vec);

private:
	std::ostream &out;
};

template <typename T>
DataOutputStream& DataOutputStream::operator<<(const T& rhs){
	out.write(reinterpret_cast<const char*>(&rhs), sizeof(T));
	return *this;
}

template <typename T>
void DataOutputStream::writeVector(const T& vec){
	uint32_t size = vec.size();
	*this << size;
	if(size>0){
		out.write(reinterpret_cast<const char*>(&vec[0]), size*sizeof(vec[0]));
	}
}

#endif /* DATA_OUTPUT_STREAM_H_ */
