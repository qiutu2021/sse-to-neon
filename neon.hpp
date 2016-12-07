/*
Copyright (c) 2012, Piotr Dollar
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
*/

#ifndef _NEON_HPP_
#define _NEON_HPP_
#include <arm_neon.h>

typedef int32x4_t __m128i;
typedef float32x4_t __m128;

#define RETf inline __m128
#define RETi inline __m128i

RETf SET(const float &x)
{
    return vmovq_n_f32(x);
}

RETf SET(float x, float y, float z, float w)
{
    float32x4_t a = vdupq_n_f32(w);
    a = vsetq_lane_f32(z, a, 1);
    a = vsetq_lane_f32(y, a, 2);
    a = vsetq_lane_f32(x, a, 3);
    return a;
}

RETi SET(const int &x)
{
    return vdupq_n_s32(x);
}

RETf LDu(const float &x)
{
    return vld1q_f32(&x);
}

RETf STRu(float &x, const __m128 y)
{
    vst1q_f32(&x, y);
    return y;
}

RETi ADD(const __m128i x, const __m128i y)
{
    return vaddq_s32(x, y);
}

RETf ADD(const __m128 x, const __m128 y)
{
    return vaddq_f32(x, y);
}

RETf ADD(const __m128 x, const __m128 y, const __m128 z)
{
    return ADD(ADD(x, y), z);
}

RETf ADD(const __m128 a, const __m128 b, const __m128 c, const __m128 &d)
{
    return ADD(ADD(ADD(a, b), c), d);
}

RETf SUB(const __m128 x, const __m128 y)
{
    return vsubq_f32(x, y);
}

RETf MUL(const __m128 x, const __m128 y)
{
    return vmulq_f32(x, y);
}

RETf MUL(const __m128 x, const float y)
{
    return MUL(x, SET(y));
}

RETf MUL(const float x, const __m128 y)
{
    return MUL(SET(x), y);
}

RETf INC(__m128 &x, const __m128 y)
{
    return x = ADD(x, y);
}

RETf DEC(__m128 &x, const __m128 y)
{
    return x = SUB(x, y);
}

RETf RCP(const __m128 x)
{
    return vrecpeq_f32(x);
}

RETf SQRT(const __m128 x)
{
    return vrecpeq_f32(vrsqrteq_f32(x));
}

RETf MAX_SSE(const __m128 x, const __m128 y)
{
    return vmaxq_f32(x, y);
}

RETf DIV(const __m128 x, const __m128 y)
{
    return vmulq_f32(x, vrecpeq_f32(y));
}

RETf DIV(const __m128 x, const float y)
{
    return DIV(x, SET(y));
}

RETf DIV(const float x, const __m128 y)
{
    return DIV(SET(x), y);
}

RETf AND(const __m128 x, const __m128 y)
{
    return reinterpret_cast<__m128>(vandq_s32(reinterpret_cast<int32x4_t>(x), reinterpret_cast<int32x4_t>(y)));
}

RETi AND(const __m128i x, const __m128i y)
{
    return vandq_s32(x, y);
}

RETf ANDNOT(const __m128 x, const __m128 y)
{
    return reinterpret_cast<__m128>(vbicq_s32(reinterpret_cast<int32x4_t>(x), reinterpret_cast<int32x4_t>(y)));
}

RETf OR(const __m128 x, const __m128 y)
{
    return reinterpret_cast<__m128>(vorrq_s32(reinterpret_cast<int32x4_t>(x), reinterpret_cast<int32x4_t>(y)));
}

RETf XOR(const __m128 x, const __m128 y)
{
    return reinterpret_cast<__m128>(veorq_s32(reinterpret_cast<int32x4_t>(x), reinterpret_cast<int32x4_t>(y)));
}

RETf CMPGT(const __m128 x, const __m128 y)
{
    return reinterpret_cast<__m128>(vcgtq_f32(x, y));
}

RETf CMPLT(const __m128 x, const __m128 y)
{
    return reinterpret_cast<__m128>(vcltq_f32(x, y));
}

RETi CMPGT(const __m128i x, const __m128i y)
{
    return reinterpret_cast<__m128i>(vcgtq_s32(x, y));
}

RETi CMPLT(const __m128i x, const __m128i y)
{
    return reinterpret_cast<__m128i>(vcltq_s32(x, y));
}

RETf CVT(const __m128i x)
{
    return vcvtq_f32_s32(x);
}

RETi CVT(const __m128 x)
{
    return vcvtq_s32_f32(x);
}

#undef RETf
#undef RETi
#endif
