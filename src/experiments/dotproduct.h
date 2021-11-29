#ifndef CMAKE_DOTPRODUCT_H
#define CMAKE_DOTPRODUCT_H

#include <immintrin.h>

using namespace std;

float dot_product_avx2(float *a, float *b, size_t size);
float dot_product_avx2_two_accumulators(float *a, float *b, size_t size);
float dot_product(float *a, float *b, size_t size);
#endif