#ifndef CMAKE_ADDITION_H
#define CMAKE_ADDITION_H

#include <immintrin.h>

using namespace std;

void add_float32_simd(float a[8], float b[8], float result[8]);
void add_float32(float a[8], float b[8], float result[8]);
#endif //CMAKE_ADDITION_H
