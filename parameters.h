/*
 * Parameters.h
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <string>
#include <stdint.h>

#include "params_parser.h"

struct Parameter{
	std::string vectorFile;
	std::string matrixFile;
	std::string wdir;
    std::string datatype;
    int gpu;

	Parameter();

	void init(Params &params);
	std::string toString();
};

#endif /* PARAMETERS_H_ */
