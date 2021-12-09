# include "dotproduct.h"
#include <assert.h>

void fill_vectors(float *a, float *b, size_t size) {
    for (size_t i = 0; i <= size; i++) {
        a[i] = (i+1) * 0.1;
        b[i] = (size - i) * 0.1;
    }
}

float dot_product_avx2(float *a, float *b, size_t size) {
    __m256 c = _mm256_setzero_ps();
     size_t limit = size - 7;

     for (size_t i = 0; i < limit; i += 8) {
      __m256 buf1 = _mm256_loadu_ps(a + i);
      __m256 buf2 = _mm256_loadu_ps(b + i);
      c = _mm256_fmadd_ps(buf1, buf2, c);
    }

    float buffer[8];
    _mm256_storeu_ps(buffer, c);
    return buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
         buffer[6] + buffer[7];
}

__attribute__((noinline)) float dot_product_sse(float *a, float *b, size_t size) {
    __m128 c = _mm_setzero_ps();
     size_t limit = size - 3;

     for (size_t i = 0; i < limit; i += 4) {
      __m128 buf1 = _mm_loadu_ps(a + i);
      __m128 buf2 = _mm_loadu_ps(b + i);
      c = _mm_fmadd_ps(buf1, buf2, c);
    }

    float buffer[4];
    _mm_storeu_ps(buffer, c);
    return buffer[0] + buffer[1] + buffer[2] + buffer[3];
}

float dot_product_avx2_two_accumulators(float *a, float *b, size_t size) {
  int accumulators = 2;
  int step = accumulators * 8;
  const float* limit = a + size;

  __m256 c0 = _mm256_setzero_ps();
  __m256 c1 = _mm256_setzero_ps();

  {

    __m256 buf1 = _mm256_loadu_ps(a);
    __m256 buf2 = _mm256_loadu_ps(b);

    c0 = _mm256_mul_ps(buf1, buf2);

    buf1 = _mm256_loadu_ps(a + 8);
    buf2 = _mm256_loadu_ps(b + 8);

    c1 = _mm256_mul_ps(buf1, buf2);

    a += step;
    b += step;
  }

  for( ; a < limit; a += step, b += step ) {
    __m256 buf1 = _mm256_loadu_ps(a);
    __m256 buf2 = _mm256_loadu_ps(b);
    c0 = _mm256_fmadd_ps(buf1, buf2, c0);

    buf1 = _mm256_loadu_ps(a + 8);
    buf2 = _mm256_loadu_ps(b + 8);
    c1 = _mm256_fmadd_ps(buf1, buf2, c1);
  }

  c0 = _mm256_add_ps(c0, c1);
  
  float buffer[8];
  _mm256_storeu_ps(buffer, c0);
  
  return buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
         buffer[6] + buffer[7];
}

float dot_product_avx2_four_accumulators(float *a, float *b, size_t size) {
  int accumulators = 4;
  int step = accumulators * 8;
  const float* limit = a + size;

  __m256 c0 = _mm256_setzero_ps();
  __m256 c1 = _mm256_setzero_ps();
  __m256 c2 = _mm256_setzero_ps();
  __m256 c3 = _mm256_setzero_ps();

  {

    __m256 buf1 = _mm256_loadu_ps(a);
    __m256 buf2 = _mm256_loadu_ps(b);
    c0 = _mm256_mul_ps(buf1, buf2);

    buf1 = _mm256_loadu_ps(a + 8);
    buf2 = _mm256_loadu_ps(b + 8);
    c1 = _mm256_mul_ps(buf1, buf2);

    buf1 = _mm256_loadu_ps(a + 16);
    buf2 = _mm256_loadu_ps(b + 16);
    c2 = _mm256_mul_ps(buf1, buf2);
    
    buf1 = _mm256_loadu_ps(a + 24);
    buf2 = _mm256_loadu_ps(b + 24);
    c3 = _mm256_mul_ps(buf1, buf2);

    a += step;
    b += step;
  }

  for( ; a < limit; a += step, b += step ) {
    __m256 buf1 = _mm256_loadu_ps(a);
    __m256 buf2 = _mm256_loadu_ps(b);
    c0 = _mm256_fmadd_ps(buf1, buf2, c0);

    buf1 = _mm256_loadu_ps(a + 8);
    buf2 = _mm256_loadu_ps(b + 8);
    c1 = _mm256_fmadd_ps(buf1, buf2, c1);

    buf1 = _mm256_loadu_ps(a + 16);
    buf2 = _mm256_loadu_ps(b + 16);
    c2 = _mm256_fmadd_ps(buf1, buf2, c2);

    buf1 = _mm256_loadu_ps(a + 24);
    buf2 = _mm256_loadu_ps(b + 24);
    c3 = _mm256_mul_ps(buf1, buf2);
  }

  c0 = _mm256_add_ps(c0, c1);
  c2 = _mm256_add_ps(c2, c3);
  c0 = _mm256_add_ps(c0, c2);
  
  float buffer[8];
  _mm256_storeu_ps(buffer, c0);
  
  return buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
         buffer[6] + buffer[7];
}

float dot_product_avx2_multiple_accumulators(float *a, float *b, size_t size, int accumulators) {
  int step = accumulators * 8;
  const float* limit = a + size;
  assert( 0 == size % accumulators );

  __m256 c0,c1,c2,c3;
  {

    __m256 buf1 = _mm256_loadu_ps(a);
    __m256 buf2 = _mm256_loadu_ps(b);
    c0 = _mm256_mul_ps(buf1, buf2);

    if (accumulators > 1) {
      buf1 = _mm256_loadu_ps(a + 8);
      buf2 = _mm256_loadu_ps(b + 8);
      c1 = _mm256_mul_ps(buf1, buf2);
    }

    if (accumulators > 2) {
      buf1 = _mm256_loadu_ps(a + 16);
      buf2 = _mm256_loadu_ps(b + 16);
      c2 = _mm256_mul_ps(buf1, buf2);
    }

    if (accumulators > 3) {
      buf1 = _mm256_loadu_ps(a + 24);
      buf2 = _mm256_loadu_ps(b + 24);
      c3 = _mm256_mul_ps(buf1, buf2);
    }

    a += step;
    b += step;
  }

  for( ; a < limit; a += step, b += step ) {
    __m256 buf1 = _mm256_loadu_ps(a);
    __m256 buf2 = _mm256_loadu_ps(b);
    c0 = _mm256_fmadd_ps(buf1, buf2, c0);

    if (accumulators > 1) {
      buf1 = _mm256_loadu_ps(a + 8);
      buf2 = _mm256_loadu_ps(b + 8);
      c1 = _mm256_fmadd_ps(buf1, buf2, c1);
    }

    if (accumulators > 2) {
      buf1 = _mm256_loadu_ps(a + 16);
      buf2 = _mm256_loadu_ps(b + 16);
      c2 = _mm256_fmadd_ps(buf1, buf2, c2);
    }

    if (accumulators > 3) {
      buf1 = _mm256_loadu_ps(a + 24);
      buf2 = _mm256_loadu_ps(b + 24);
      c3 = _mm256_fmadd_ps(buf1, buf2, c3);
    }
  }

  if (accumulators > 1)
    c0 = _mm256_add_ps(c0, c1);
  if (accumulators > 3)
    c2 = _mm256_add_ps(c2, c3);
  if (accumulators > 2)
    c0 = _mm256_add_ps(c0, c2);
  
  float buffer[8];
  _mm256_storeu_ps(buffer, c0);
  
  return buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
         buffer[6] + buffer[7];
}

float dot_product(float *a, float *b, size_t size) {
  float c = 0;
  const float* limit = a + size;
  for ( ; a < limit; a++, b++) {
    c += a[0] * b[0];
  }
  return c;
}