#ifndef INT256
#define INT256

#include <fstream>
#include <iostream>

using namespace std;

class Int256
{
    public:
        ulong4 m;
        __host__ __device__ Int256()
        {
        }

        __host__ __device__ Int256( const int val )
        {
            *this = val;
        }

        __host__ __device__ Int256( const ulong4& val )
        {
            m = val;
        }

        __host__ __device__ Int256( const uint4 val1, const uint4 val2)
        {
            m.x = ((long)val1.y << 32) | val1.x;
            m.y = ((long)val1.w << 32) | val1.z;
            m.z = ((long)val2.y << 32) | val2.x;
            m.w = ((long)val2.w << 32) | val2.z;
        }

        __host__ __device__ Int256( const Int256& val )
        {
            *this   = val;
        }

        __host__ __device__ Int256& operator =(const Int256& input )
        {
            m = input.m;
            return *this;
        }

        __host__ __device__ Int256& operator =( const int& input )
        {
            m = make_ulong4(0, 0, 0, input);
            return *this;
        }

        __host__ __device__ bool operator ==(const unsigned int input)
        {
            return (m.x == 0) && (m.y == 0) && (m.z == 0) && (m.w == input);
        }

        __host__ __device__ bool operator !=(const unsigned int input )
        {
            return (m.x != 0) || (m.y != 0) || (m.z != 0) || (m.w != input);
        }

        __host__ __device__  Int256& operator ^=( const Int256& input )
        {
            m.x^=input.m.x;
            m.y^=input.m.y;
            m.z^=input.m.z;
            m.w^=input.m.w;
            return *this;
        }

        __host__ __device__  Int256 operator ^( const Int256& input )
        {
            ulong4 n;
            n.x = m.x ^ input.m.x;
            n.y = m.y ^ input.m.y;
            n.z = m.z ^ input.m.z;
            n.w = m.w ^ input.m.w;
            return Int256(n);
        }

        friend ifstream & operator >>(ifstream &ins, Int256 &input) {
            ins >> input.m.x >> input.m.y >> input.m.z >> input.m.w;
            return ins;
        }

        friend ostream & operator <<(ostream &out, Int256 &input) {
            out << input.m.x <<" "<<input.m.y<<" "<<input.m.z<<" "<<input.m.w;
            return out;
        }

        __host__ __device__  static void volatile_xor(volatile Int256& input, const Int256& rhs)
        {
            input.m.x ^= rhs.m.x;
            input.m.y ^= rhs.m.y;
            input.m.z ^= rhs.m.z;
            input.m.w ^= rhs.m.w;
        }

        __host__ __device__  static void volatile_xor(volatile Int256& input, const volatile Int256& rhs)
        {
            input.m.x ^= rhs.m.x;
            input.m.y ^= rhs.m.y;
            input.m.z ^= rhs.m.z;
            input.m.w ^= rhs.m.w;
        }

        __host__ __device__  static void volatile_set(volatile Int256& input, const Int256& rhs)
        {
            input.m.x = rhs.m.x;
            input.m.y = rhs.m.y;
            input.m.z = rhs.m.z;
            input.m.w = rhs.m.w;
        }

        __host__ __device__  static void volatile_set(Int256& input, const volatile Int256& rhs)
        {
            input.m.x = rhs.m.x;
            input.m.y = rhs.m.y;
            input.m.z = rhs.m.z;
            input.m.w = rhs.m.w;
        }

};
#endif

