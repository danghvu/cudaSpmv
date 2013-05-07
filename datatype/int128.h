#ifndef INT128
#define INT128

#include <fstream>
#include <iostream>

using namespace std;

class Int128
{
    public:
        ulong2 m;
        __host__ __device__ Int128()
        {
        }

        __host__ __device__ Int128( const int val )
        {
            *this   = make_ulong2(0,val);
        }

        __host__ __device__ Int128( const uint4 val )
        {
            *this   = make_ulong2( (long) val.y << 32 | val.x, (long) val.w << 32 | val.z);
        }
        __host__ __device__ Int128( unsigned long x, unsigned long y)
        {
            *this = make_ulong2( x, y );
        }

        __host__ __device__ Int128( const Int128& val )
        {
            *this   = val;
        }

        __host__ __device__ Int128& operator =(const Int128& input )
        {
            m = input.m;
            return *this;
        }

        __host__ __device__ bool operator ==(const unsigned int input )
        {
            return (m.x == 0) && (m.y == input);
        }

        __host__ __device__ bool operator !=(const unsigned int input )
        {
            return (m.x != 0) || (m.y != input);
        }

        __host__ __device__ Int128& operator =( const ulong2& input )
        {
            m = input;
            return *this;
        }


        __host__ __device__ Int128& operator =( const int& input )
        {
            m = make_ulong2(0,input);
            return *this;
        }

        __host__ __device__  Int128& operator ^=( const Int128& input )
        {
            m.x^=input.m.x;
            m.y^=input.m.y;
            return *this;
        }

        __host__ __device__  Int128 operator ^( const Int128& input )
        {
            return Int128( m.x^input.m.x, m.y^input.m.y );
        }

        friend ifstream & operator >>(ifstream &ins, Int128 &input) {
            ins >> input.m.x >> input.m.y;
            return ins;
        }

        friend ostream & operator <<(ostream &out, Int128 &input) {
            out << input.m.x <<" "<<input.m.y;
            return out;
        }

        __host__ __device__  static void volatile_xor(volatile Int128& input, const Int128& rhs)
        {
            input.m.x ^= rhs.m.x;
            input.m.y ^= rhs.m.y;
        }

        __host__ __device__  static void volatile_xor(volatile Int128& input, const volatile Int128& rhs)
        {
            input.m.x ^= rhs.m.x;
            input.m.y ^= rhs.m.y;
        }

        __host__ __device__  static void volatile_set(volatile Int128& input, const Int128& rhs)
        {
            input.m.x = rhs.m.x;
            input.m.y = rhs.m.y;
        }

        __host__ __device__  static void volatile_set(Int128& input, volatile const Int128& rhs)
        {
            input.m.x = rhs.m.x;
            input.m.y = rhs.m.y;
        }

};

#endif
