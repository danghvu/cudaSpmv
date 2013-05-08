/*
 * factory.h
 *
 *  Created on: Dec 17, 2010
 *      Author: hans
 */

#ifndef FACTORY_H_
#define FACTORY_H_

#include <string>
#include <exception>
#include "matrix.h"
#include "sliced_coo.h"

namespace matrix_factory {

    class UnknownMatrixFormatException: public std::exception
    {
        public:
            UnknownMatrixFormatException(const std::string &msg) : std::exception(), msg(msg){};
            ~UnknownMatrixFormatException() throw(){};
            virtual const char* what() const throw(){
                return msg.c_str();
            }
        private:
            std::string msg;
    };


    const std::string SLICED_COO_256("256scoo");
    const std::string SLICED_COO_128("128scoo");
    const std::string SLICED_COO_64("64scoo");
    const std::string SLICED_COO_32("32scoo");
    const std::string SLICED_COO_16("16scoo");
    const std::string SLICED_COO_8("8scoo");
    const std::string SLICED_COO_4("4scoo");
    const std::string SLICED_COO_2("2scoo");
    const std::string SLICED_COO_1("1scoo");

    template <typename T> 
    Matrix<T>* getMatrixObject(const std::string& format) {
        if (format ==SLICED_COO_64) {
            return new SlicedCoo<T, 64>();
        } else if (format == SLICED_COO_128) {
            return new SlicedCoo<T, 128>();
        } else if (format == SLICED_COO_256) {
            return new SlicedCoo<T, 256>();
        } else if (format == SLICED_COO_32) {
            return new SlicedCoo<T,32>();
        } else if(format == SLICED_COO_16) {
            return new SlicedCoo<T, 16>();
        } else if(format == SLICED_COO_8) {
            return new SlicedCoo<T,8>();
        } else if (format == SLICED_COO_4) {
            return new SlicedCoo<T,4>();
        } else if (format == SLICED_COO_2) {
            return new SlicedCoo<T,2>();
        } else if (format == SLICED_COO_1) {
            return new SlicedCoo<T,1>();
        } else{
            std::stringstream ss;
            ss << "Format " << format << " is not recognized.";
            throw UnknownMatrixFormatException(ss.str());
        }
    }

}

template class Matrix<float>;
template class Matrix<double>;

#endif /* FACTORY_H_ */
