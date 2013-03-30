/*
 * Params.cpp
 *
 *  Created on: Jun 22, 2011
 *      Author: hans
 */

#include "params_parser.h"

Params::Params(uint32_t argc, const char* const* argv) throw (std::invalid_argument){
	using std::string;
	for(uint32_t i=1; i<argc; i++){
		string str(argv[i]);
		size_t found = str.find_first_of("=");
		if(found != string::npos){
			string key = str.substr(0, found);
			string val = str.substr(found+1);
			paramsMap.insert(std::pair<string, string>(key, val));
		}else{
			throw std::invalid_argument(string("Invalid parameter: ").append(str));
		}
	}
}

bool Params::haveParam(const std::string &param) const {
	return paramsMap.count(param)>0;
}

void Params::getMxN(const std::string &param, uint32_t *m, uint32_t *n) const throw(std::invalid_argument){
	using std::string;

	std::map<string, string>::const_iterator it = paramsMap.find(param);
	if(it != paramsMap.end()){
		string mxn(it->second);
		size_t pos = mxn.find('x');
		if(pos!=string::npos && pos+1 < mxn.length()){
			string str[2] = {mxn.substr(0, pos),
								  mxn.substr(pos+1, mxn.length() - pos - 1)};
			uint32_t res[2];
			for(int i=0; i<2; i++){
				std::istringstream sstr(str[i]);
				sstr >> res[i];
				if(sstr.bad() || sstr.fail() || !sstr.eof()){
					throw std::invalid_argument(string("Failed to extract the MxN value of parameter with name: ").append(param));
				}
			}
			*m = res[0];
			*n = res[1];
		}else{
			throw std::invalid_argument(string("Failed to extract the MxN value of parameter with name: ").append(param));
		}
	} else{
		throw std::invalid_argument(string("Unable to find parameter with name: ").append(param));
	}
}

template <>
void Params::getParam(const std::string &param, std::string* val, bool optional) const throw(std::invalid_argument){
	using std::string;

	std::map<string, string>::const_iterator it = paramsMap.find(param);
	if(it != paramsMap.end()){
		*val = it->second;
	} else{
	    if (!optional) 
            throw std::invalid_argument(string("Unable to find parameter with name: ").append(param));
	}
}
