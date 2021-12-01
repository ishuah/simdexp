#ifndef CMAKE_DOTPRODUCT_H
#define CMAKE_DOTPRODUCT_H

#include <immintrin.h>

using namespace std;

void fill_vectors(float *a, float *b, size_t size);
float lemires_dot_product_avx2(float *x1, float *x2, size_t len);
float dot_product_sse(float *a, float *b, size_t size);
float dot_product_avx2(float *a, float *b, size_t size);
float dot_product_avx2_multiple_accumulators(float *a, float *b, size_t size, int accumulators);
float dot_product_sse_multiple_accumulators(float *a, float *b, size_t size, int accumulators);
float dot_product(float *a, float *b, size_t size);
#endif