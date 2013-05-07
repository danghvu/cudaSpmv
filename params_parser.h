/*
 * Params.h
 *
 *  Created on: Jun 22, 2011
 *      Author: hans
 */

#ifndef PARAMS_H_
#define PARAMS_H_

#include <stdint.h>
#include <string>
#include <map>
#include <sstream>
#include <iostream>
#include <stdexcept>

class Params{
public:
	Params(uint32_t argc, const char* const* argv)  throw (std::invalid_argument);
	bool haveParam(const std::string &param) const;
	template <class T>
	void getParam(const std::string &param, T* val) const throw(std::invalid_argument);

    template <class T>
	void getParamOptional(const std::string &param, T* val);

	void getMxN(const std::string &param, uint32_t *m, uint32_t *n) const throw(std::invalid_argument);
private:
	std::map<std::string, std::string> paramsMap;
};

template <class T>
void Params::getParamOptional(const std::string &param, T* val) {
	using std::string;

	std::map<string, string>::const_iterator it = paramsMap.find(param);
	if(it != paramsMap.end()){
		std::istringstream sstr(it->second);
		T res;
		sstr >> res;
		if(sstr.fail() || sstr.bad() || !sstr.eof()){
		    throw std::invalid_argument(string("Failed to extract the value of parameter with name: ").append(param));
		}else{
			*val = res;
		}
	} 
}


template <class T>
void Params::getParam(const std::string &param, T* val) const throw(std::invalid_argument){
	using std::string;

	std::map<string, string>::const_iterator it = paramsMap.find(param);
	if(it != paramsMap.end()){
		std::istringstream sstr(it->second);
		T res;
		sstr >> res;
		if(sstr.fail() || sstr.bad() || !sstr.eof()){
		    throw std::invalid_argument(string("Failed to extract the value of parameter with name: ").append(param));
		}else{
			*val = res;
		}
	} else{
        throw std::invalid_argument(string("Unable to find parameter with name: ").append(param));
	}
}

#endif /* PARAMS_H_ */
