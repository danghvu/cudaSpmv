/*
 * bind_textures.h
 *
 *  Created on: Dec 10, 2010
 *      Author: hans
 */

#ifndef TEXTURES_H_
#define TEXTURES_H_

#include <texture_fetch_functions.h>
#include <stdint.h>
#include "../datatype/int128.h"
#include "../datatype/int256.h"


	texture<uint2> tex_ll;
	texture<uint4> tex_l2;
	texture<uint4> tex_l4;
	texture<uint32_t> tex_l;
    texture<float> tex_f;
    texture<int2, 1> tex_d;
	texture<uint32_t> tex_Ap;

	__inline__ __device__ uint64_t fetch_x(const uint32_t &i, const uint64_t * x)
	{
	   uint2 v = tex1Dfetch(tex_ll, i);
	   return ((uint64_t)v.y << 32) + v.x;
	}

    
    __inline__ __device__ double fetch_x(const uint32_t &i, const double *x)
    {
        int2 v = tex1Dfetch(tex_d,i);
        return __hiloint2double(v.y, v.x);
    }

	__inline__ __device__ Int128 fetch_x(const uint32_t &i, const Int128 * x)
	{
	   uint4 v = tex1Dfetch(tex_l2, i);
	   return Int128(v);
	}

	__inline__ __device__ Int256 fetch_x(const uint32_t &i, const Int256 * x)
	{
	   uint4 v1 = tex1Dfetch(tex_l4, i*2);
	   uint4 v2 = tex1Dfetch(tex_l4, i*2+1);
	   return Int256(v1, v2);
	}

	__inline__ __device__ uint32_t fetch_x(const uint32_t &i, const uint32_t * x)
	{
	   return tex1Dfetch(tex_l, i);
	}

	__inline__ __device__ float fetch_x(const uint32_t &i, const float * x)
	{
	   return tex1Dfetch(tex_f, i);
	}

	__inline__ __device__ unsigned int fetch_ap(const uint32_t &i, const uint32_t * x)
	{
	   return tex1Dfetch(tex_Ap, i);
	}

	inline void bind_x(const Int128 * x)
	{
		size_t offset = size_t(-1);
		cudaError_t res = cudaBindTexture(&offset, tex_l2 , x);
		if (res != cudaSuccess){
			std::cerr << "Failed to bind: " << cudaGetErrorString(res) << std::endl;
		}else if (offset !=0 ){
				std::cerr << "memory not align" << std::endl;
		}

	}

	inline void bind_x(const Int256 * x)
	{
		size_t offset = size_t(-1);
		cudaError_t res = cudaBindTexture(&offset, tex_l4 , x);
		if (res != cudaSuccess){
			std::cerr << "Failed to bind: " << cudaGetErrorString(res) << std::endl;
		}else if (offset !=0 ){
				std::cerr << "memory not align" << std::endl;
		}
	}

	inline void bind_x(const uint64_t * x)
	{
		size_t offset = size_t(-1);
		cudaError_t res = cudaBindTexture(&offset, tex_ll , x);
		if (res != cudaSuccess){
			std::cerr << "Failed to bind: " << cudaGetErrorString(res) << std::endl;
		}else if (offset !=0 ){
				std::cerr << "memory not align" << std::endl;
		}
	}

    inline void bind_x(const double *x) {
        size_t offset = size_t(-1);
		cudaError_t res = cudaBindTexture(&offset, tex_d , x);
		if (res != cudaSuccess){
			std::cerr << "Failed to bind: " << cudaGetErrorString(res) << std::endl;
		}else if (offset !=0 ){
				std::cerr << "memory not align" << std::endl;
		}
    }    

	inline void bind_x(const float * x)
	{
		size_t offset = size_t(-1);
		cudaError_t res = cudaBindTexture(&offset, tex_f , x);
		if (res != cudaSuccess){
			std::cerr << "Failed to bind x: " << cudaGetErrorString(res) << std::endl;
		}else if (offset !=0 ){
				std::cerr << "memory not align" << std::endl;
		}
	}

	inline void bind_x(const uint32_t * x)
	{
		size_t offset = size_t(-1);
		cudaError_t res = cudaBindTexture(&offset, tex_l , x);
		if (res != cudaSuccess){
			std::cerr << "Failed to bind: " << cudaGetErrorString(res) << std::endl;
		}else if (offset !=0 ){
				std::cerr << "memory not align" << std::endl;
		}
	}

	inline void bind_Ap(const uint32_t * x)
	{
		size_t offset = size_t(-1);
		cudaError_t res = cudaBindTexture(&offset, tex_Ap , x);
		if (res != cudaSuccess){
			std::cerr << "Failed to bind Ap: " << cudaGetErrorString(res) << std::endl;
		}else if (offset !=0 ){
				std::cerr << "memory not align" << std::endl;
		}
	}

    inline void unbind_x(const double *x)
    {
        cudaUnbindTexture(tex_d);
    }

	inline void unbind_x(const float * x)
	{
		cudaUnbindTexture(tex_f);
	}

	inline void unbind_Ap(const uint32_t * x)
	{
		cudaUnbindTexture(tex_Ap);
	}

	inline void unbind_x(const uint64_t * x)
	{
		cudaUnbindTexture(tex_ll);
	}

	inline void unbind_x(const uint32_t * x)
	{
		cudaUnbindTexture(tex_l);
	}

	inline void unbind_x(const Int128 * x)
	{
		cudaUnbindTexture(tex_l2);
	}

	inline void unbind_x(const Int256 * x)
	{
		cudaUnbindTexture(tex_l4);
	}
#endif /* TEXTURES_H_ */
