#ifndef CMAKE_MULTIPLICATION_H
#define CMAKE_MULTIPLICATION_H

#include <immintrin.h>

using namespace std;

void mul_float32_SIMD(float a[4], float b[4], float result[4]);
void mul_float32_avx2(float a[8], float b[8], float result[8]);
void mul_float32(float a[8], float b[8], float result[8]);
#endif //CMAKE_MULTIPLICATION_H
